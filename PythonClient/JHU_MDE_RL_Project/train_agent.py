# train_drone.py
import os
import numpy as np
import time
import glob
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from airsim_gym_multirotor_env import DroneObstacleEnv
import torch

class TrainingCallback(BaseCallback):
    """Simple callback for tracking training progress with debug info"""
    def __init__(self, check_freq: int = 1000, debug_freq: int = 50, 
                 save_episode_freq: int = 100, save_path: str = "./checkpoints/", verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.debug_freq = debug_freq
        self.save_episode_freq = save_episode_freq
        self.save_path = save_path
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
        
        # Ensure save directory exists
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Track episode rewards
        current_reward = self.locals['rewards'][0]
        self.current_episode_reward += current_reward
        
        # Debug logging every debug_freq steps
        if self.n_calls % self.debug_freq == 0:
            self._log_debug_info(current_reward)
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            
            # Log episode summary
            print(f"\n=== EPISODE {self.episode_count} COMPLETED ===")
            print(f"Total Reward: {self.current_episode_reward:.2f}")
            if len(self.model.env.get_attr('current_step')) > 0:
                print(f"Steps: {self.model.env.get_attr('current_step')[0]}")
            print(f"Success: {self.current_episode_reward > 0}")
            
            # Save model every N episodes
            if self.episode_count % self.save_episode_freq == 0:
                episode_model_path = os.path.join(self.save_path, f"episode_{self.episode_count}_model")
                self.model.save(episode_model_path)
                print(f"\nModel saved at episode {self.episode_count}: {episode_model_path}")
                
                # Also save replay buffer using the correct stable_baselines3 method
                replay_buffer_path = os.path.join(self.save_path, f"episode_{self.episode_count}_replay_buffer")
                try:
                    self.model.save_replay_buffer(replay_buffer_path)
                    buffer_size = self.model.replay_buffer.size()
                    print(f" Replay buffer saved: {replay_buffer_path}.pkl ({buffer_size} experiences)")
                except Exception as e:
                    print(f" ERROR: Could not save replay buffer: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Also save as "latest" for easy resuming
                latest_model_path = os.path.join(self.save_path, "latest_model")
                latest_replay_buffer_path = os.path.join(self.save_path, "latest_replay_buffer")
                self.model.save(latest_model_path)
                try:
                    self.model.save_replay_buffer(latest_replay_buffer_path)
                    buffer_size = self.model.replay_buffer.size()
                    print(f" Latest replay buffer saved: {latest_replay_buffer_path}.pkl ({buffer_size} experiences)")
                except Exception as e:
                    print(f" ERROR: Could not save latest replay buffer: {e}")
                    import traceback
                    traceback.print_exc()
            
            if self.episode_count % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                print(f"\nEpisode {self.episode_count}, Mean Reward (last 10): {mean_reward:.2f}")
            
            self.current_episode_reward = 0
        
        # Log every check_freq steps
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                print(f"\nStep {self.n_calls}, Mean Reward (last 100): {mean_reward:.2f}")
        
        return True
    
    def _log_debug_info(self, current_reward):
        """Log detailed debug information"""
        try:
            # Get environment attributes
            env = self.model.env.envs[0].env  # Unwrap to get the actual environment
            
            # Get current observation
            obs = self.locals['observations'][0]
            
            # Extract components from observation
            vae_latent = obs[:32]
            normalized_relative_distance = obs[32:35]
            normalized_velocity = obs[35:38]
            
            # Calculate actual distance magnitude
            distance_magnitude = np.linalg.norm(normalized_relative_distance)
            
            # Get current position
            position = env._get_current_position()
            
            # Get depth map info
            depth_map = env._get_depth_image()
            depth_mean = np.mean(depth_map)
            depth_min = np.min(depth_map)
            depth_max = np.max(depth_map)
            
            print(f"\n--- DEBUG STEP {self.n_calls} ---")
            print(f"Current Reward: {current_reward:.3f}")
            print(f"Episode Reward: {self.current_episode_reward:.2f}")
            print(f"Step in Episode: {env.current_step}")
            
            print(f"\nPosition: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
            print(f"Target: [{env.target_position[0]:.2f}, {env.target_position[1]:.2f}, {env.target_position[2]:.2f}]")
            print(f"Normalized Relative Distance: [{normalized_relative_distance[0]:.3f}, {normalized_relative_distance[1]:.3f}, {normalized_relative_distance[2]:.3f}]")
            print(f"Distance Magnitude: {distance_magnitude:.3f}")
            
            print(f"Normalized Velocity: [{normalized_velocity[0]:.3f}, {normalized_velocity[1]:.3f}, {normalized_velocity[2]:.3f}]")
            
            print(f"Depth Map - Mean: {depth_mean:.3f}, Min: {depth_min:.3f}, Max: {depth_max:.3f}")
            
            # Check collision and bounds
            collision = env.client.simGetCollisionInfo().has_collided
            out_of_bounds = env._is_out_of_bounds(position)
            print(f"Collision: {collision}, Out of Bounds: {out_of_bounds}")
            
            # VAE latent stats
            vae_mean = np.mean(vae_latent)
            vae_std = np.std(vae_latent)
            print(f"VAE Latent - Mean: {vae_mean:.3f}, Std: {vae_std:.3f}")
            
        except Exception as e:
            print(f"Debug logging error: {e}")

def find_latest_checkpoint(checkpoint_dir: str = "./checkpoints/"):
    """Find the latest checkpoint model to resume training from"""
    # First check for "latest_model" - this is the most recent checkpoint
    latest_model_path = os.path.join(checkpoint_dir, "latest_model.zip")
    if os.path.exists(latest_model_path):
        # Find the episode number from the most recent episode model
        # This tells us what episode we're resuming from
        episode_models = glob.glob(os.path.join(checkpoint_dir, "episode_*_model.zip"))
        if episode_models:
            # Extract episode numbers and find the max
            episode_numbers = []
            for model_path in episode_models:
                try:
                    # Extract episode number from filename like "episode_20_model.zip"
                    filename = os.path.basename(model_path)
                    episode_num = int(filename.split("_")[1])
                    episode_numbers.append(episode_num)
                except (ValueError, IndexError):
                    continue
            
            if episode_numbers:
                # Use latest_model.zip (most recent) but get episode number from episode models
                latest_episode = max(episode_numbers)
                return latest_model_path, latest_episode
        else:
            # If latest_model.zip exists but no episode models, assume episode 0
            return latest_model_path, 0
    
    # Fallback: if no latest_model.zip, try to find the most recent episode model
    episode_models = glob.glob(os.path.join(checkpoint_dir, "episode_*_model.zip"))
    if episode_models:
        episode_numbers = []
        for model_path in episode_models:
            try:
                filename = os.path.basename(model_path)
                episode_num = int(filename.split("_")[1])
                episode_numbers.append((episode_num, model_path))
            except (ValueError, IndexError):
                continue
        
        if episode_numbers:
            latest_episode, latest_path = max(episode_numbers, key=lambda x: x[0])
            return latest_path, latest_episode
    
    return None, 0

def setup_training(resume_from_checkpoint: bool = True):
    """Setup and train the SAC agent"""
    
    # Environment parameters
    target_position = np.array([40.0, 0.0, -3.0])
    world_bounds = [-15, 70, -15, 15, -15, 0]
    vae_model_path = "./vae_data/vae_final.pth"
    
    # Create environment with speed parameter
    env = DroneObstacleEnv(
        vae_model_path=vae_model_path,
        target_position=target_position,
        world_bounds=world_bounds,
        max_steps=500
    )
    
    # Wrap for vectorized environment
    env = DummyVecEnv([lambda: Monitor(env)])
    
    # Create directories
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)
    
    # Try to load latest checkpoint
    model = None
    starting_episode = 0
    
    if resume_from_checkpoint:
        latest_checkpoint, episode_num = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            print(f"\nFound checkpoint: {latest_checkpoint}")
            print(f"Resuming from episode {episode_num}")
            try:
                model = SAC.load(latest_checkpoint, env=env)
                starting_episode = episode_num
                print(f"Successfully loaded model from episode {episode_num}")
                print(f"  Model components loaded: policy, Q-networks, optimizer states")
                
                # CRITICAL: Load replay buffer for proper training resumption
                # The replay buffer contains the experiences the model learned from
                # Without it, the agent starts with empty buffer and needs to refill before learning
                latest_replay_buffer_path = os.path.join(checkpoint_dir, "latest_replay_buffer")
                # Check both .pkl extension and without (stable_baselines3 adds .pkl automatically)
                if os.path.exists(latest_replay_buffer_path + ".pkl") or os.path.exists(latest_replay_buffer_path):
                    try:
                        model.load_replay_buffer(latest_replay_buffer_path)
                        # Get buffer size using the correct attribute (not len())
                        buffer_size = model.replay_buffer.size()
                        print(f"  Replay buffer loaded: {latest_replay_buffer_path}.pkl")
                        print(f"  Buffer size: {buffer_size} experiences")
                        
                        # Check if buffer has enough experiences for learning
                        if buffer_size < model.learning_starts:
                            print(f"  WARNING: Buffer has {buffer_size} experiences, but learning_starts={model.learning_starts}")
                            print(f"  Agent will collect {model.learning_starts - buffer_size} more steps before learning resumes")
                        else:
                            print(f"  Buffer has enough experiences - learning will resume immediately")
                    except Exception as e:
                        print(f" ERROR: Could not load replay buffer: {e}")
                        print(f" This will cause the agent to start from scratch!")
                        print(f" Training will collect {model.learning_starts} new experiences before learning begins")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"  WARNING: No replay buffer found at {latest_replay_buffer_path}.pkl")
                    print(f"  This means the agent will start with an EMPTY replay buffer!")
                    print(f"  It will need to collect {model.learning_starts} new experiences before learning begins")
                    print(f"  The model weights are loaded, but training will appear to start over")
                    print(f"  This is why training seems to start from scratch - no replay buffer was saved!")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting fresh training...")
                model = None
    
    # If no checkpoint found or loading failed, create new model
    if model is None:
        print("\nCreating new model...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=1e-4,  # Reduced from 3e-4 for more stable learning
            buffer_size=100000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),  # Train every step (more explicit)
            gradient_steps=1,
            ent_coef=0.2,  # Fixed entropy coefficient instead of 'auto' for stability
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            seed=42,
            target_update_interval=1,  # Update target network every step
            target_entropy='auto',  # Auto-adjust target entropy but keep ent_coef fixed
        )
    
    # Callbacks
    # Training callback with episode-based saving (includes replay buffer)
    training_callback = TrainingCallback(
        check_freq=1000, 
        debug_freq=50,
        save_episode_freq=100,
        save_path=checkpoint_dir
    )
    
    # Set starting episode count in callback if resuming
    if starting_episode > 0:
        training_callback.episode_count = starting_episode
    
    callbacks = [training_callback]
    
    return model, env, callbacks

def test_model(model_path: str, num_episodes: int = 3, deterministic: bool = True):
    """
    Test a trained model with visual feedback
    
    Args:
        model_path: Path to the model file (e.g., "checkpoints/episode_20_model.zip")
        num_episodes: Number of test episodes to run
        deterministic: Whether to use deterministic actions (True) or sample from policy (False)
    """
    print(f"\n{'='*60}")
    print(f"TESTING MODEL: {model_path}")
    print(f"Episodes: {num_episodes}, Deterministic: {deterministic}")
    print(f"{'='*60}\n")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("\nAvailable models in checkpoints/:")
        checkpoint_dir = "./checkpoints"
        if os.path.exists(checkpoint_dir):
            episode_models = glob.glob(os.path.join(checkpoint_dir, "episode_*_model.zip"))
            for model in sorted(episode_models):
                print(f"  - {model}")
        return
    
    # Environment parameters (same as training)
    target_position = np.array([40.0, 0.0, -3.0])
    world_bounds = [-15, 70, -15, 15, -15, 0]
    vae_model_path = "./vae_data/vae_final.pth"  # Match training setup
    
    # Create environment
    env = DroneObstacleEnv(
        vae_model_path=vae_model_path,
        target_position=target_position,
        world_bounds=world_bounds,
        max_steps=500
    )
    
    # Load the model
    try:
        model = SAC.load(model_path, env=env)
        print(f"Model loaded successfully")
        
        # Check if replay buffer exists and show info
        checkpoint_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "./checkpoints"
        model_name = os.path.basename(model_path).replace(".zip", "")
        
        # Try to find corresponding replay buffer (using correct stable_baselines3 method)
        if "latest_model" in model_name:
            replay_buffer_path = os.path.join(checkpoint_dir, "latest_replay_buffer")
        elif "episode_" in model_name:
            # Extract episode number
            try:
                parts = model_name.split("_")
                episode_num_str = parts[1]
                replay_buffer_path = os.path.join(checkpoint_dir, f"episode_{episode_num_str}_replay_buffer")
            except:
                replay_buffer_path = None
        else:
            replay_buffer_path = None
        
        # Check if file exists (stable_baselines3 adds .pkl automatically)
        if replay_buffer_path and (os.path.exists(replay_buffer_path + ".pkl") or os.path.exists(replay_buffer_path)):
            try:
                model.load_replay_buffer(replay_buffer_path)
                buffer_size = model.replay_buffer.size()
                print(f"Replay buffer loaded: {buffer_size} experiences")
            except Exception as e:
                print(f"Note: Replay buffer found but not loaded (not needed for testing): {e}")
        else:
            print("Note: No replay buffer found (not needed for testing)")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        env.close()
        return
    
    # Extract episode number from path if available (for display)
    episode_num = "unknown"
    if "episode_" in model_path:
        try:
            parts = os.path.basename(model_path).split("_")
            episode_num = parts[1] if len(parts) > 1 else "unknown"
        except:
            pass
    elif "latest_model" in model_path:
        # Try to find episode number from latest episode model
        checkpoint_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "./checkpoints"
        episode_models = glob.glob(os.path.join(checkpoint_dir, "episode_*_model.zip"))
        if episode_models:
            episode_numbers = []
            for ep_model_path in episode_models:
                try:
                    filename = os.path.basename(ep_model_path)
                    ep_num = int(filename.split("_")[1])
                    episode_numbers.append(ep_num)
                except (ValueError, IndexError):
                    continue
            if episode_numbers:
                episode_num = str(max(episode_numbers))
    
    # Run test episodes
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        truncated = False
        steps = 0
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} (Model: Episode {episode_num}) ---")
        episode_start = time.time()
        
        while not done and not truncated and steps < 500:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Progress indicator every 100 steps
            if steps % 100 == 0:
                print(f"  Step {steps}, Reward: {episode_reward:.2f}")
        
        episode_duration = time.time() - episode_start
        
        # Determine result
        if done:
            result = "SUCCESS"
            success_count += 1
        elif truncated:
            result = "TIME LIMIT"
        else:
            result = "MAX STEPS"
        
        print(f"Results: {result} | Steps: {steps} | Reward: {episode_reward:.2f} | Duration: {episode_duration:.1f}s")
        
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY (Model: Episode {episode_num})")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} � {np.std(episode_rewards):.2f}")
    print(f"Mean Steps: {np.mean(episode_steps):.1f} � {np.std(episode_steps):.1f}")
    print(f"Best Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Reward: {np.min(episode_rewards):.2f}")
    print(f"{'='*60}\n")
    
    env.close()

def main():
    """Main function - supports both training and testing modes"""
    parser = argparse.ArgumentParser(
        description="Train or test a SAC drone obstacle avoidance agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
        Examples:
        # Train the model (default behavior)
        python train_agent.py
        
        # Test a specific model (e.g., episode 100)
        python train_agent.py --test checkpoints/episode_100_model.zip
        
        # Test the latest checkpoint (recommended to verify checkpoint loading)
        python train_agent.py --test-latest
        
        # Test latest with multiple episodes
        python train_agent.py --test-latest --episodes 5
        
        # Test with multiple episodes
        python train_agent.py --test checkpoints/episode_400_model.zip --episodes 5
        
        # Test with stochastic policy (non-deterministic)
        python train_agent.py --test checkpoints/episode_400_model.zip --stochastic
        
        # List available models
        python train_agent.py --list-models
        """
    )
    
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to model file to test (e.g., 'checkpoints/episode_100_model.zip'). If specified, runs in test mode only."
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of test episodes to run (default: 3)"
    )
    
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (sample actions) instead of deterministic during testing"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available model checkpoints and exit"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch (don't resume from checkpoint)"
    )
    
    parser.add_argument(
        "--test-latest",
        action="store_true",
        help="Test the latest checkpoint model (equivalent to --test checkpoints/latest_model.zip)"
    )
    
    args = parser.parse_args()
    
    # List models mode
    if args.list_models:
        checkpoint_dir = "./checkpoints"
        print(f"\nAvailable models in {checkpoint_dir}/:\n")
        if os.path.exists(checkpoint_dir):
            episode_models = glob.glob(os.path.join(checkpoint_dir, "episode_*_model.zip"))
            if episode_models:
                for model in sorted(episode_models):
                    # Extract episode number
                    try:
                        filename = os.path.basename(model)
                        episode_num = filename.split("_")[1]
                        print(f"  Episode {episode_num:>4}: {model}")
                    except:
                        print(f"  {model}")
                print()
            else:
                print("  No episode models found.\n")
        else:
            print(f"  Checkpoint directory does not exist: {checkpoint_dir}\n")
        return
    
    # Test mode
    if args.test_latest:
        # Test the latest checkpoint
        checkpoint_dir = "./checkpoints"
        latest_checkpoint, episode_num = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            print(f"\nTesting latest checkpoint: {latest_checkpoint}")
            print(f"Resuming from episode {episode_num}\n")
            test_model(
                model_path=latest_checkpoint,
                num_episodes=args.episodes,
                deterministic=not args.stochastic
            )
        else:
            print("ERROR: No latest checkpoint found. Train a model first or use --test with a specific model path.")
        return
    
    if args.test:
        test_model(
            model_path=args.test,
            num_episodes=args.episodes,
            deterministic=not args.stochastic
        )
        return
    
    # Training mode (default)
    print("Starting Drone Training...")
    
    # Added: Just set your desired speed here, C:\Users\UserName\Documents\AirSim\settings.json
    # also set json file "ClockSpeed": 2, name to match the simulation speed
    '''
    {
        "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "ClockSpeed": 2
    }
    '''

    env = None  # Initialize to None to avoid UnboundLocalError
    
    try:
        # Setup training (will auto-resume from latest checkpoint if available)
        resume = not args.no_resume
        model, env, callbacks = setup_training(resume_from_checkpoint=resume)
        
        print(f"\nTraining Configuration:")
        print(f"  Target: [50.0, 0.0, -3.0]")
        print(f"  Episode-based saving: Every 100 episodes (with replay buffer)")
        print(f"  Resume from checkpoint: {'Enabled' if resume else 'Disabled'}")
        
        # Train the model
        print("\nTraining started...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=100000,
            callback=callbacks,
            log_interval=10,
            tb_log_name="SAC_drone"
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the model
        model.save("sac_drone_obstacle_avoidance_final")
        print("Model saved!")
        
        # Test the final model
        print("\nTesting final model...")
        test_model("sac_drone_obstacle_avoidance_final", num_episodes=3)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()