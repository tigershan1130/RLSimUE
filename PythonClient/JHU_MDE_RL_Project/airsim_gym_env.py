import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import time

class AirSimCarGymEnv(gym.Env):
    """
    A custom Gymnasium environment for AirSim Car.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, image_shape=(84, 84, 1)):
        super(AirSimCarGymEnv, self).__init__()

        # Define action and observation spaces (same as before)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=image_shape, dtype=np.uint8)

        # Connect to the AirSim simulator
        self.car_client = airsim.CarClient()
        self.car_client.confirmConnection()
        self.car_client.enableApiControl(True)
        self.car_controls = airsim.CarControls()

        # Environment specific parameters
        self.image_shape = image_shape
        self.image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        self.start_ts = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Returns:
            observation (array): Initial observation
            info (dict): Additional information
        """
        # Gymnasium requires handling the seed
        super().reset(seed=seed)
        
        # Reset the car
        self.car_client.reset()
        self.car_client.enableApiControl(True)
        time.sleep(0.01)

        # Get initial observation
        observation = self._get_obs()
        self.start_ts = time.time()
        
        # Gymnasium returns (observation, info) instead of just observation
        info = {}
        return observation, info

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Returns:
            observation (array): New observation after action
            reward (float): Reward for this step
            terminated (bool): Whether episode ends permanently
            truncated (bool): Whether episode ends due to time limit
            info (dict): Additional information
        """
        # Apply the action
        self._do_action(action)
        time.sleep(0.1)

        # Get new observation
        observation = self._get_obs()

        # Calculate reward
        reward = self._compute_reward()

        # Check termination conditions
        terminated = self._is_done()
        
        # Check truncation (e.g., time limit)
        truncated = False
        if time.time() - self.start_ts > 60:  # 60-second episode
            truncated = True

        info = {}  # Additional info can go here
        
        # Gymnasium returns 5 values instead of 4
        return observation, reward, terminated, truncated, info

    # The rest of your methods remain the same:
    def _do_action(self, action):
        """Map discrete action to AirSim car controls."""
        self.car_controls.brake = 0
        self.car_controls.throttle = 1.0
        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 0.5
        elif action == 3:
            self.car_controls.steering = -0.5
        elif action == 4:
            self.car_controls.steering = 0.25
        else:
            self.car_controls.steering = -0.25

        self.car_client.setCarControls(self.car_controls)

    def _get_obs(self):
        """Get observation from the environment."""
        responses = self.car_client.simGetImages([self.image_request])
        if responses:
            img1d = np.array(responses[0].image_data_float, dtype=np.float64)
            img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
            
            from PIL import Image
            image = Image.fromarray(img2d)
            im_final = np.array(image.resize(self.image_shape[0:2]).convert("L"))
            return im_final.reshape(self.image_shape)
        else:
            return np.zeros(self.image_shape, dtype=np.uint8)

    def _compute_reward(self):
        """Define your reward function."""
        reward = 0
        car_state = self.car_client.getCarState()
        reward += car_state.speed

        collision_info = self.car_client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward -= 100

        return reward

    def _is_done(self):
        """Check if episode should end permanently."""
        collision_info = self.car_client.simGetCollisionInfo()
        return collision_info.has_collided

    def render(self):
        """Optional rendering."""
        pass

    def close(self):
        """Clean up."""
        self.car_client.enableApiControl(False)