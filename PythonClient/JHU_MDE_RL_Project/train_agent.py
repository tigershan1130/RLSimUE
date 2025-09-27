import gymnasium as gym
from stable_baselines3 import PPO
from airsim_gym_env import AirSimCarGymEnv  # Your custom environment

# Create your environment
env = AirSimCarGymEnv(image_shape=(84, 84, 1))

# Instantiate the RL algorithm
model = PPO("CnnPolicy", env, verbose=1)

# Start training - the interface works the same way!
model.learn(total_timesteps=100000)

# Save the trained model
model.save("airsim_car_ppo")

# Close the environment
env.close()