# train_drone.py
import os
import numpy as np
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from airsim_gym_multirotor_env import DroneObstacleEnv
import torch

class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress with detailed debug info"""
    def __init__(self, check_freq: int = 1000, debug_freq: int = 100, verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.debug_freq = debug_freq
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.step_count = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Track episode rewards
        if len(self.model.env.get_attr('current_step')) > 0:
            current_reward = self.locals['rewards'][0]
            self.current_episode_reward += current_reward
            
            # Debug logging every debug_freq steps
            if self.step_count % self.debug_freq == 0:
                self._log_debug_info(current_reward)
            
            # Check if episode ended
            if self.locals['dones'][0]:
                self.episode_count += 1
                self.episode_rewards.append(self.current_episode_reward)
                
                # Log episode summary
                print(f"\n=== EPISODE {self.episode_count} COMPLETED ===")
                print(f"Total Reward: {self.current_episode_reward:.2f}")
                print(f"Steps: {self.model.env.get_attr('current_step')[0]}")
                print(f"Success: {self.current_episode_reward > 0}")
                
                self.current_episode_reward = 0
                
                # Log every N episodes
                if self.episode_count % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards[-10:])
                    print(f"\nEpisode {self.episode_count}, Mean Reward (last 10): {mean_reward:.2f}")
        
        # Log every check_freq steps
        if self.step_count % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                print(f"\nStep {self.step_count}, Mean Reward (last 100): {mean_reward:.2f}")
        
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
            
            print(f"\n--- DEBUG STEP {self.step_count} ---")
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

def setup_training():
    """Setup and train the SAC agent"""
    
    # Environment parameters
    target_position = np.array([50.0, 0.0, -3.0])  # 50m forward, center, 3m height
    world_bounds = [0, 60, -10, 10, -10, 0]  # [x_min, x_max, y_min, y_max, z_min, z_max]
    # AirSim: z is negative for up, positive for down
    # So z_min = -10 (10m up), z_max = 0 (ground level)
    
    vae_model_path = "vae_data/vae_best.pth"
    
    # Create environment
    env = DroneObstacleEnv(
        vae_model_path=vae_model_path,
        target_position=target_position,
        world_bounds=world_bounds,
        max_steps=500  # Reduced for faster training
    )
    
    # Wrap for vectorized environment (required by SB3)
    env = DummyVecEnv([lambda: Monitor(env)])
    
    # Create directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)
    os.makedirs("./eval_logs", exist_ok=True)
    os.makedirs("./best_model", exist_ok=True)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="drone_sac"
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    training_callback = TrainingCallback(check_freq=1000, debug_freq=50)  # Debug every 50 steps
    
    callbacks = [checkpoint_callback, eval_callback, training_callback]
    
    # SAC model configuration
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=10000,  # Start learning after collecting some samples
        batch_size=256,
        tau=0.005,  # Soft update coefficient
        gamma=0.99,  # Discount factor
        train_freq=1,  # Update every step
        gradient_steps=1,  # Number of gradient steps per update
        ent_coef='auto',  # Automatic entropy tuning
        target_update_interval=1,
        policy_kwargs=dict(
            net_arch=[256, 256],  # Network architecture
            log_std_init=-2,  # Initial log std
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        seed=42  # For reproducibility
    )
    
    return model, env, callbacks

def test_model(model_path: str = "best_model/best_model"):
    """Test the trained model"""
    print("Testing trained model...")
    
    # Environment parameters (same as training)
    target_position = np.array([50.0, 0.0, -3.0])
    world_bounds = [0, 60, -10, 10, -10, 0]
    vae_model_path = "vae_data/vae_best.pth"
    
    # Create environment
    env = DroneObstacleEnv(
        vae_model_path=vae_model_path,
        target_position=target_position,
        world_bounds=world_bounds,
        max_steps=500
    )
    
    # Load the trained model
    model = SAC.load(model_path, env=env)
    
    # Test for 5 episodes
    for episode in range(5):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\n=== Episode {episode + 1} ===")
        
        while not done and steps < 500:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Print progress occasionally
            if steps % 50 == 0:
                print(f"Step {steps}, Reward: {episode_reward:.2f}")
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Success: {steps < 500}")  # If didn't timeout, likely reached target
    
    env.close()

def main():
    """Main training function"""
    print("Starting Drone SAC Training...")
    print("Make sure AirSim is running and the environment is ready!")
    
    try:
        # Setup
        model, env, callbacks = setup_training()
        
        print("Model Configuration:")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Target position: [50.0, 0.0, -3.0]")
        print(f"  World bounds: [0, 60, -10, 10, -10, 0]")
        
        # Train the model
        print("\nTraining started...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=200000,  # Reduced for testing, increase for full training
            callback=callbacks,
            log_interval=10,
            tb_log_name="SAC_drone"
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the final model
        model.save("sac_drone_obstacle_avoidance_final")
        print("Final model saved!")
        
        # Test the trained model
        test_model("sac_drone_obstacle_avoidance_final")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("Training session ended.")

if __name__ == "__main__":
    main()