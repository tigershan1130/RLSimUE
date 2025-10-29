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
    """Simple callback for tracking training progress with debug info"""
    def __init__(self, check_freq: int = 1000, debug_freq: int = 50, verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.debug_freq = debug_freq
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0

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

def setup_training():
    """Setup and train the SAC agent"""
    
    # Environment parameters
    target_position = np.array([50.0, 0.0, -3.0])
    world_bounds = [0, 60, -10, 10, -10, 0]
    vae_model_path = "vae_data/vae_best.pth"
    
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
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="drone_sac"
    )
    
    training_callback = TrainingCallback(check_freq=1000, debug_freq=50)  # Debug every 50 steps
    
    callbacks = [checkpoint_callback, training_callback]
    
    # SAC model configuration
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
    
    return model, env, callbacks

def test_model(model_path: str = "sac_drone_obstacle_avoidance_final", simulation_speed: float = 1.0):
    """Test the trained model"""
    print(f"Testing at {simulation_speed}x speed...")
    
    target_position = np.array([50.0, 0.0, -3.0])
    world_bounds = [0, 60, -10, 10, -10, 0]
    vae_model_path = "vae_data/vae_best.pth"
    
    # Create environment with test speed
    env = DroneObstacleEnv(
        vae_model_path=vae_model_path,
        target_position=target_position,
        world_bounds=world_bounds,
        max_steps=500,
        simulation_speed=simulation_speed
    )
    
    model = SAC.load(model_path, env=env)
    
    for episode in range(3):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\n=== Episode {episode + 1} (Speed: {simulation_speed}x) ===")
        
        while not done and steps < 500:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        print(f"Episode finished: Steps: {steps}, Reward: {episode_reward:.2f}")
    
    env.close()

def main():
    """Main training function"""
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
        model, env, callbacks = setup_training()
        
        print(f"Training Configuration:")
        print(f"  Target: [50.0, 0.0, -3.0]")
        
        # Train the model
        print("\nTraining started...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=10000,
            callback=callbacks,
            log_interval=10,
            tb_log_name="SAC_drone"
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the model
        model.save("sac_drone_obstacle_avoidance_final")
        print("Model saved!")
        
        # Test the model
        test_model("sac_drone_obstacle_avoidance_final")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()