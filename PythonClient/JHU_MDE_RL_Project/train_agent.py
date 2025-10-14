from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from enhancedFeatureExtractor import EnhancedFeatureExtractor
from airsim_gym_env import AirSimMultirotorEnv
import torch

class EnhancedTrainingCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super(EnhancedTrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Log additional metrics including velocity information
            infos = self.locals.get('infos', [{}])
            for info in infos:
                if 'drone_velocity' in info:
                    velocity = info['drone_velocity']
                    speed = np.linalg.norm(velocity)
                    self.logger.record('custom/speed', speed)
                    self.logger.record('custom/velocity_x', velocity[0])
                    self.logger.record('custom/velocity_y', velocity[1])
                    self.logger.record('custom/velocity_z', velocity[2])
                
                if 'target_reached' in info:
                    self.logger.record('custom/success_rate', float(info['target_reached']))
                    
        return True

# Create enhanced environment
print("Initialize AirSimMultirotorEnv...")
env = AirSimMultirotorEnv(depth_image_size=(84, 84))


print("Initialize TD3 model...")


# Initialize model with enhanced feature extractor
model = TD3(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=150000,  # Increased for more complex state
    learning_starts=10000,  # More exploration before training
    batch_size=128,
    gamma=0.99,
    train_freq=(1, "episode"),
    gradient_steps=-1,
    verbose=1,
    tensorboard_log="./airsim_enhanced_tensorboard/",
    policy_kwargs=dict(
        features_extractor_class=EnhancedFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[400, 300],  # Larger network for complex state
        activation_fn=torch.nn.ReLU
    )
)

print("Starting training with enhanced observations (position + velocity + depth)...")
model.learn(
    total_timesteps=150000,  # More timesteps for complex learning
    callback=EnhancedTrainingCallback(),
    log_interval=5
)

# Save the trained model
model.save("airsim_enhanced_td3")
print("Training completed!")