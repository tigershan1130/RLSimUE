import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import math
import random
import cv2
from PIL import Image
import io

class AirSimMultirotorEnv(gym.Env):
    def __init__(self, target_radius=500.0, depth_image_size=(84, 84)):
        super().__init__()
        
        print("Connect to AirSim...")
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        print("initialize bounds...")
        # Unreal Engine coordinate system in centimeters
        self.bounds = {
            'x_min': -20000, 'x_max': 20000,
            'y_min': -20000, 'y_max': 20000, 
            'z_min': 200, 'z_max': 1000
        }
        self.player_start = (0, 0, 200)
        
        # Depth image configuration
        self.depth_image_size = depth_image_size
        self.camera_name = "0"  # Front-center camera
        
        # Setup camera for depth
        self._setup_depth_camera()

        self.target_radius = target_radius
        self.current_target = None
        self.max_speed = 1500.0  # cm/s
        self.collision_penalty = -1000
        self.out_of_bounds_penalty = -800  # Slightly less than collision but still severe
        self.success_reward = 1000
        self.step_penalty = -0.1
        self.previous_distance = None
        
        # Action space: [vx, vy, vz, yaw_rate] - normalized values
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Enhanced observation space: vector (position + velocity + target) + depth image
        self.observation_space = spaces.Dict({
            'vector': spaces.Box(
                low=np.array([
                    self.bounds['x_min'], self.bounds['y_min'], self.bounds['z_min'],  # position
                    -self.max_speed, -self.max_speed, -self.max_speed,  # linear velocity (cm/s)
                    -180.0, -180.0, -180.0,  # angular velocity (deg/s) - pitch, roll, yaw rates
                    self.bounds['x_min'], self.bounds['y_min'], self.bounds['z_min'],  # target position
                    0  # distance to target
                ]),
                high=np.array([
                    self.bounds['x_max'], self.bounds['y_max'], self.bounds['z_max'],  # position
                    self.max_speed, self.max_speed, self.max_speed,  # linear velocity (cm/s)
                    180.0, 180.0, 180.0,  # angular velocity (deg/s)
                    self.bounds['x_max'], self.bounds['y_max'], self.bounds['z_max'],  # target position
                    40000  # max distance
                ]),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(depth_image_size[0], depth_image_size[1], 1),
                dtype=np.float32
            )
        })
        
        
    def _setup_depth_camera(self):
        """Configure the depth camera in AirSim"""
        print("config depth camera...")
        #self.client.simSetCameraOrientation(0, airsim.to_quaternion(0, 0, 0))
        
    def _get_drone_state(self):
        """Get comprehensive drone state including position and velocity"""
        drone_state = self.client.getMultirotorState()
        kinematics = drone_state.kinematics_estimated
        position = kinematics.position
        linear_velocity = kinematics.linear_velocity
        angular_velocity = kinematics.angular_velocity
        
        # Convert to Unreal coordinates
        unreal_x = position.x_val
        unreal_y = position.y_val  
        unreal_z = -position.z_val
        
        # Convert velocities to Unreal coordinates
        unreal_vx = linear_velocity.x_val
        unreal_vy = linear_velocity.y_val
        unreal_vz = -linear_velocity.z_val  # Convert NED to Unreal Z
        
        # Angular velocities (typically in radians/s in AirSim, convert to degrees/s)
        unreal_pitch_rate = math.degrees(angular_velocity.x_val)
        unreal_roll_rate = math.degrees(angular_velocity.y_val)
        unreal_yaw_rate = math.degrees(angular_velocity.z_val)
        
        position_3d = (unreal_x, unreal_y, unreal_z)
        linear_vel_3d = (unreal_vx, unreal_vy, unreal_vz)
        angular_vel_3d = (unreal_pitch_rate, unreal_roll_rate, unreal_yaw_rate)
        
        return position_3d, linear_vel_3d, angular_vel_3d
    
    def _get_drone_position(self):
        """Get only drone position (for backward compatibility)"""
        position, _, _ = self._get_drone_state()
        return position
    
    def _is_out_of_bounds(self, position):
        """Check if drone is outside the defined boundaries"""
        x, y, z = position
        
        # Check if position exceeds any boundary
        if (x < self.bounds['x_min'] or x > self.bounds['x_max'] or
            y < self.bounds['y_min'] or y > self.bounds['y_max'] or
            z < self.bounds['z_min'] or z > self.bounds['z_max']):
            return True
        return False
    
    def _get_boundary_distance_penalty(self, position):
        """Calculate penalty based on how far outside bounds the drone is"""
        x, y, z = position
        
        # Calculate how far outside each boundary
        x_violation = max(self.bounds['x_min'] - x, 0, x - self.bounds['x_max'])
        y_violation = max(self.bounds['y_min'] - y, 0, y - self.bounds['y_max'])
        z_violation = max(self.bounds['z_min'] - z, 0, z - self.bounds['z_max'])
        
        # Total violation distance
        total_violation = x_violation + y_violation + z_violation
        
        # Progressive penalty - more penalty the further out of bounds
        if total_violation > 0:
            return -total_violation / 1000.0  # Scale penalty appropriately
        return 0.0
    
    def _get_proximity_to_boundary_reward(self, position):
        """Reward for staying away from boundaries (safety margin)"""
        x, y, z = position
        
        # Calculate distance to each boundary
        dist_to_x_min = x - self.bounds['x_min']
        dist_to_x_max = self.bounds['x_max'] - x
        dist_to_y_min = y - self.bounds['y_min']
        dist_to_y_max = self.bounds['y_max'] - y
        dist_to_z_min = z - self.bounds['z_min']
        dist_to_z_max = self.bounds['z_max'] - z
        
        # Find the minimum distance to any boundary
        min_distance = min(dist_to_x_min, dist_to_x_max, dist_to_y_min, 
                          dist_to_y_max, dist_to_z_min, dist_to_z_max)
        
        # Small positive reward for staying away from boundaries
        # More reward when further from boundaries
        if min_distance > 2000:  # 20m safety margin
            return 0.1
        elif min_distance > 1000:  # 10m safety margin
            return 0.05
        elif min_distance > 500:   # 5m safety margin
            return 0.02
        elif min_distance < 100:   # Too close to boundary
            return -0.1
        else:
            return 0.0
    
    def _get_depth_image(self):
        """Capture and process depth image from AirSim"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    self.camera_name, 
                    airsim.ImageType.DepthPerspective,
                    pixels_as_float=True, 
                    compress=False
                )
            ])
            
            if responses and responses[0]:
                depth_array = self._parse_depth_response(responses[0])
                depth_resized = cv2.resize(
                    depth_array, 
                    self.depth_image_size, 
                    interpolation=cv2.INTER_AREA
                )
                depth_normalized = np.clip(depth_array / 100.0, 0.0, 1.0)
                depth_normalized = np.expand_dims(depth_normalized, axis=-1)
                return depth_normalized
            else:
                return np.zeros((self.depth_image_size[0], self.depth_image_size[1], 1), dtype=np.float32)
                
        except Exception as e:
            print(f"Error getting depth image: {e}")
            return np.zeros((self.depth_image_size[0], self.depth_image_size[1], 1), dtype=np.float32)
    
    def _parse_depth_response(self, response):
        """Parse AirSim depth image response"""
        depth_img = np.frombuffer(response.image_data_float, dtype=np.float32)
        depth_img = depth_img.reshape(response.height, response.width)
        return depth_img
    
    def _has_collided(self):
        """Check if drone has collided"""
        collision_info = self.client.simGetCollisionInfo()
        return collision_info.has_collided
    
    def _distance_to_target(self, position=None):
        """Calculate Euclidean distance to current target in cm"""
        if position is None:
            position, _, _ = self._get_drone_state()
        x, y, z = position
            
        tx, ty, tz = self.current_target
        return math.sqrt((x-tx)**2 + (y-ty)**2 + (z-tz)**2)
    
    def _generate_reachable_target(self):
        """Generate a random target within bounds"""
        margin = 2000
        target_x = random.uniform(
            self.bounds['x_min'] + margin, 
            self.bounds['x_max'] - margin
        )
        target_y = random.uniform(
            self.bounds['y_min'] + margin, 
            self.bounds['y_max'] - margin
        )
        target_z = random.uniform(
            self.bounds['z_min'] + 10,
            self.bounds['z_max'] - 100
        )
        
        return (target_x, target_y, target_z)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset drone
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Takeoff and move to starting position
        start_x, start_y, start_z = self.player_start
        
        self.client.takeoffAsync().join()
        
        # Move to exact starting position
        self.client.moveToPositionAsync(
            float(start_x),
            float(start_y),  
            float(-start_z),
            5.0,
            timeout_sec=30
        ).join()
        
        # Generate initial target
        self.current_target = self._generate_reachable_target()
        self.previous_distance = self._distance_to_target()
        
        # Get initial observation with all states
        observation = self._get_observation()
        info = {
            "target_position": self.current_target,
            "start_position": self.player_start
        }
        
        return observation, info
    
    def _get_observation(self):
        """Construct comprehensive observation with position, velocity, and depth"""
        # Get complete drone state
        position, linear_vel, angular_vel = self._get_drone_state()
        drone_x, drone_y, drone_z = position
        vx, vy, vz = linear_vel
        pitch_rate, roll_rate, yaw_rate = angular_vel
        
        target_x, target_y, target_z = self.current_target
        distance = self._distance_to_target(position)
        
        # Build vector observation with all state information
        vector_obs = np.array([
            # Position (3)
            drone_x, drone_y, drone_z,
            # Linear velocity (3)
            vx, vy, vz,
            # Angular velocity (3)
            pitch_rate, roll_rate, yaw_rate,
            # Target position (3)
            target_x, target_y, target_z,
            # Distance to target (1)
            distance
        ], dtype=np.float32)
        
        # Get depth image
        depth_obs = self._get_depth_image()
        
        return {
            'vector': vector_obs,
            'depth': depth_obs
        }
    
    def step(self, action):
        # Execute action
        vx = action[0] * self.max_speed
        vy = action[1] * self.max_speed
        vz = -action[2] * self.max_speed  # Convert to NED
        yaw_rate = action[3] * 90.0
        
        self.client.moveByVelocityBodyFrameAsync(
            vx, vy, vz, duration=0.1,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )
        
        # Get new state
        position, linear_vel, angular_vel = self._get_drone_state()
        observation = self._get_observation()
        distance = self._distance_to_target(position)
        collided = self._has_collided()
        out_of_bounds = self._is_out_of_bounds(position)
        
        # Calculate reward with boundary checking
        reward = self._compute_reward(distance, collided, out_of_bounds, linear_vel, angular_vel, position)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if collided:
            terminated = True
            reward += self.collision_penalty
            print("Episode terminated: Collision detected!")
        elif out_of_bounds:
            terminated = True
            reward += self.out_of_bounds_penalty
            print(f"Episode terminated: Out of bounds! Position: {position}")
        elif distance < self.target_radius:
            terminated = True
            reward += self.success_reward
            print("Episode terminated: Target reached!")
        
        self.previous_distance = distance
        
        info = {
            "distance_to_target": distance,
            "collision": collided,
            "out_of_bounds": out_of_bounds,
            "target_reached": distance < self.target_radius,
            "drone_position": position,
            "drone_velocity": linear_vel,
            "drone_angular_velocity": angular_vel,
            "target_position": self.current_target,
            "is_out_of_bounds": out_of_bounds  # Duplicate for compatibility
        }
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(self, distance, collided, out_of_bounds, linear_vel, angular_vel, position):
        """Enhanced reward function with boundary checking and velocity-based penalties"""
        if collided:
            return self.collision_penalty
        
        if out_of_bounds:
            return self.out_of_bounds_penalty
        
        if distance < self.target_radius:
            return self.success_reward
        
        reward = 0.0
        
        # Distance improvement reward
        if self.previous_distance is not None:
            distance_improvement = (self.previous_distance - distance) / 100.0
            reward += distance_improvement * 10.0
        
        # Velocity-based rewards/penalties
        speed = math.sqrt(linear_vel[0]**2 + linear_vel[1]**2 + linear_vel[2]**2)
        
        # Penalize excessive speed (encourage controlled flight)
        if speed > self.max_speed * 0.8:  # If exceeding 80% of max speed
            reward -= 0.5
        
        # Penalize high angular velocities (encourage smooth flight)
        angular_speed = math.sqrt(angular_vel[0]**2 + angular_vel[1]**2 + angular_vel[2]**2)
        if angular_speed > 45.0:  # If angular speed > 45 deg/s
            reward -= 0.3
        
        # Boundary proximity rewards/penalties
        boundary_reward = self._get_proximity_to_boundary_reward(position)
        reward += boundary_reward
        
        # Progressive out-of-bounds penalty (even if not fully out of bounds yet)
        boundary_penalty = self._get_boundary_distance_penalty(position)
        reward += boundary_penalty
        
        # Reward efficient movement toward target
        if distance < self.previous_distance:
            # Additional reward for moving in the right direction efficiently
            direction_reward = (self.previous_distance - distance) / self.previous_distance
            reward += direction_reward * 2.0
        
        # Time penalty
        reward += self.step_penalty
        
        return reward

    def render(self):
        # Optional: Add visualization if needed
        pass

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)