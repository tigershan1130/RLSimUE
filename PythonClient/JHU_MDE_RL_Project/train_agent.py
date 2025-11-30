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
                
                # Also save replay buffer
                replay_buffer_path = os.path.join(self.save_path, f"episode_{self.episode_count}_replay_buffer.pkl")
                try:
                    self.model.replay_buffer.save(replay_buffer_path)
                    print(f"Replay buffer saved: {replay_buffer_path} ({len(self.model.replay_buffer)} experiences)")
                except Exception as e:
                    print(f"Warning: Could not save replay buffer: {e}")
                
                # Also save as "latest" for easy resuming
                latest_model_path = os.path.join(self.save_path, "latest_model")
                latest_replay_buffer_path = os.path.join(self.save_path, "latest_replay_buffer.pkl")
                self.model.save(latest_model_path)
                try:
                    self.model.replay_buffer.save(latest_replay_buffer_path)
                    print(f"Latest model and replay buffer updated: {latest_model_path}")
                except Exception as e:
                    print(f"Warning: Could not save latest replay buffer: {e}")
            
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
    # First check for "latest_model"
    latest_model_path = os.path.join(checkpoint_dir, "latest_model.zip")
    if os.path.exists(latest_model_path):
        # Find the episode number from the most recent episode model
        episode_models = glob.glob(os.path.join(checkpoint_dir, "episode_*_model.zip"))
        if episode_models:
            # Extract episode numbers and find the max
            episode_numbers = []
            for model_path in episode_models:
                try:
                    # Extract episode number from filename like "episode_20_model.zip"
                    filename = os.path.basename(model_path)
                    episode_num = int(filename.split("_")[1])
                    episode_numbers.append((episode_num, model_path))
                except (ValueError, IndexError):
                    continue
            
            if episode_numbers:
                # Return the most recent episode model
                latest_episode, latest_path = max(episode_numbers, key=lambda x: x[0])
                return latest_path, latest_episode
    
    return None, 0

def setup_training(resume_from_checkpoint: bool = True):
    """Setup and train the SAC agent"""
    
    # Environment parameters
    target_position = np.array([15.0, 0.0, -3.0])
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
                
                # Optional: Load replay buffer (not strictly necessary - model weights already encode learned knowledge)
                # Loading buffer allows immediate learning without waiting for buffer to fill
                latest_replay_buffer_path = os.path.join(checkpoint_dir, "latest_replay_buffer.pkl")
                if os.path.exists(latest_replay_buffer_path):
                    try:
                        model.replay_buffer.load(latest_replay_buffer_path)
                        print(f"Replay buffer loaded: {latest_replay_buffer_path} ({len(model.replay_buffer)} experiences)")
                        print("  Note: Model weights already encode learned knowledge. Buffer provides immediate learning.")
                    except Exception as e:
                        print(f"Warning: Could not load replay buffer: {e}")
                        print("Continuing with empty replay buffer (will collect new experiences)...")
                else:
                    print("No replay buffer found - will collect new experiences (model weights are already loaded)")
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
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            seed=42
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
    target_position = np.array([15.0, 0.0, -3.0])
    world_bounds = [-15, 70, -15, 15, -15, 0]
    vae_model_path = "vae_data/vae_best.pth"
    
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
        print(f"Model loaded successfully\n")
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