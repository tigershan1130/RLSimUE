# airsim_gym_multirotor_env.py
import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import torch
import math
from typing import Tuple, Optional
import cv2
import time
import threading
from concurrent.futures import ThreadPoolExecutor

class DroneObstacleEnv(gym.Env):
    """
    Drone Obstacle Avoidance Environment with VAE + SAC
    SIMPLIFIED: Continuous action space: [speed_factor, lateral, vertical]
    Observation space: [VAE_latent(32) + relative_distance(3) + velocity(3)] = 38D
    """
    
    def __init__(self, vae_model_path: str, target_position: np.ndarray, 
                 world_bounds: list, max_steps: int = 1000):
        super(DroneObstacleEnv, self).__init__()
        
        # AirSim connection
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Load pre-trained VAE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = self._load_vae(vae_model_path)
        
        # Environment parameters
        self.target_position = target_position
        self.world_bounds = world_bounds  # [x_min, x_max, y_min, y_max, z_min, z_max]
        self.max_steps = max_steps
        self.current_step = 0
        
        # SIMPLIFIED: Action space: 3D continuous [speed_factor, lateral, vertical]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: 38D continuous
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(38,),  # 32 (VAE) + 3 (relative_distance) + 3 (velocity)
            dtype=np.float32
        )
        
        # Control parameters
        self.base_forward_speed = 3.0    # m/s (base speed when speed_factor=1)
        self.max_lateral_speed = 2.0     # m/s
        self.max_vertical_speed = 2.0    # m/s
        
        # Reward function parameters
        self.progress_scale = 10.0
        self.obstacle_k = 2.0
        self.previous_normalized_distance = None
        
        # Timer reward parameters
        self.speed_reward_scale = 0.5    # Reward for maintaining speed
        self.time_penalty_scale = 0.01   # Small penalty per step to encourage efficiency

        # Normalization parameters
        self.max_expected_distance = np.linalg.norm(
            np.array([world_bounds[1] - world_bounds[0], 
                     world_bounds[3] - world_bounds[2],
                     world_bounds[5] - world_bounds[4]])
        )
        self.max_expected_velocity = np.array([
            self.base_forward_speed * 1.5,
            self.max_lateral_speed * 1.5, 
            self.max_vertical_speed * 1.5
        ])

        # Enhanced timing diagnostics
        self.step_times = []
        self.episode_start_time = None
        self.episode_step_times = []
        self.component_times = {
            'action_execution': [],
            'observation': [],
            'reward_calculation': [],
            'reward_computation': [],  # Pure reward computation (excluding client calls)
            'reward_client_calls': [],  # Time spent in client calls during reward calculation
            'vae_encoding': [],
            'image_processing': [],
            'simulation_sleep': []
        }
        
        # For parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Thread-safe shared state for observation collection
        self.latest_observation = None
        self.observation_lock = threading.Lock()
        self.client_lock = threading.Lock()  # Lock for AirSim client (not thread-safe)
        
        # Control flags for background observation thread
        self.observation_thread_running = False
        self.observation_thread = None
        
        # Observation collection interval 0.2, but simulation runs at x2 clock speed
        self.observation_interval = 0.2  

    # Helper method to load pre-trained VAE model
    def _load_vae(self, model_path: str):
        """Load pre-trained VAE model"""
        from vae_model import VAE
        
        vae = VAE(input_shape=(72, 128), latent_dim=32).to(self.device)
        vae.load_state_dict(torch.load(model_path, map_location=self.device))
        vae.eval()
        return vae

    # Gymnasium reset function
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Stop any existing background threads FIRST (before any client calls)
        self._stop_background_threads()
        
        # Make sure simulation is unpaused for reset (thread-safe)
        with self.client_lock:
            self.client.simPause(False)
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        
        # Takeoff and set to initial position (thread-safe)
        with self.client_lock:
            self.client.takeoffAsync().join()
            start_position = [0.0, 0.0, -3.0]  # x, y, z (z is up, negative in AirSim)
            self.client.moveToPositionAsync(*start_position, 5).join()
        
        self.current_step = 0
        self.previous_normalized_distance = None
        self.episode_start_time = time.time()
        self.episode_step_times = []
        self.component_times = {k: [] for k in self.component_times.keys()}
        
        # Draw target visualization (thread-safe)
        self._draw_target_visualization()
        
        # Initialize observation
        observation = self._get_observation()
        with self.observation_lock:
            self.latest_observation = observation
        
        # Start background observation thread
        self._start_background_threads()
        
        # Keep simulation unpaused - observation thread and actions will run continuously
        with self.client_lock:
            self.client.simPause(False)
        
        info = {}
        
        return observation, info

    # Gymnasium step function
    def step(self, action: np.ndarray):
        """Execute one environment step - action executed synchronously, observation collected in background"""
        step_start_time = time.time()
        self.current_step += 1
        
        # Execute action synchronously (thread-safe)
        action_start = time.time()
        speed_factor, lateral, vertical = action
        
        # Calculate velocities from simplified actions
        forward_velocity = speed_factor * self.base_forward_speed
        lateral_velocity = lateral * self.max_lateral_speed
        vertical_velocity = vertical * self.max_vertical_speed
        
        # Send command to AirSim (thread-safe, non-blocking - simulation runs continuously)
        with self.client_lock:
            self.client.moveByVelocityAsync(
                float(forward_velocity),
                float(lateral_velocity), 
                float(vertical_velocity),
                duration=0.4  # Command duration 0.4, but simulation runs at x2 clock speed
            )
        
        action_time = time.time() - action_start
        self.component_times['action_execution'].append(action_time)
        
        # Get latest observation (collected by background thread)
        observation_start = time.time()
        with self.observation_lock:
            observation = self.latest_observation.copy() if self.latest_observation is not None else self._get_observation()
        observation_time = time.time() - observation_start
        self.component_times['observation'].append(observation_time)
        
        # Calculate reward
        reward_start = time.time()
        reward, terminated = self._calculate_reward(observation, action)
        reward_time = time.time() - reward_start
        self.component_times['reward_calculation'].append(reward_time)
        
        # Check episode truncation (max steps)
        truncated = self.current_step >= self.max_steps
        
        # Calculate total step time
        step_duration = time.time() - step_start_time
        self.episode_step_times.append(step_duration)
        self.step_times.append(step_duration)
        
        # Enhanced timing diagnostics
        if self.current_step % 50 == 0 or terminated or truncated:
            self._log_timing_diagnostics()
        
        info = {
            'step_time': step_duration,
            'current_step': self.current_step,
            'component_times': self.component_times.copy(),
            'terminated': terminated,
            'truncated': truncated
        }
        
        # Log episode summary if episode ended
        if terminated or truncated:
            self._log_episode_summary(terminated, truncated)
        
        return observation, reward, terminated, truncated, info

    def _start_background_threads(self):
        """Start background thread for observation collection"""
        self.observation_thread_running = True
        
        self.observation_thread = threading.Thread(target=self._observation_collection_loop, daemon=True)
        self.observation_thread.start()
        
    def _stop_background_threads(self):
        """Stop background thread"""
        self.observation_thread_running = False
        
        if self.observation_thread is not None:
            self.observation_thread.join(timeout=1.0)
    
    def _observation_collection_loop(self):
        """Background thread: Continuously collect observations every 0.1s in parallel"""
        while self.observation_thread_running:
            loop_start = time.time()
            
            # Collect observation in parallel
            observation_start = time.time()
            
            # Step 1: Get depth map (already thread-safe inside method)
            image_start = time.time()
            depth_map = self._get_depth_image()
            image_time = time.time() - image_start
            self.component_times['image_processing'].append(image_time)
            
            # Step 2: Start VAE encoding in parallel (using thread pool)
            vae_start = time.time()
            vae_future = self.thread_pool.submit(self._encode_depth_map, depth_map)
            vae_time = time.time() - vae_start
            self.component_times['vae_encoding'].append(vae_time)
            
            # Step 3: Get drone state while VAE processes (already thread-safe in _get_current_position)
            # We'll use a simpler method here
            with self.client_lock:
                drone_state = self.client.getMultirotorState()
            kinematics = drone_state.kinematics_estimated
            
            position = np.array([
                kinematics.position.x_val,
                kinematics.position.y_val, 
                kinematics.position.z_val
            ])
            
            velocity = np.array([
                kinematics.linear_velocity.x_val,
                kinematics.linear_velocity.y_val,
                kinematics.linear_velocity.z_val
            ])
            
            # Step 4: Get VAE result (should be ready by now)
            vae_latent = vae_future.result()
            
            # Step 5: Calculate relative distance and normalize
            relative_distance = self.target_position - position
            
            normalized_relative_distance = relative_distance / self.max_expected_distance
            normalized_relative_distance = np.clip(normalized_relative_distance, -1.0, 1.0)
            
            normalized_velocity = velocity / self.max_expected_velocity
            normalized_velocity = np.clip(normalized_velocity, -1.0, 1.0)
            
            # Update visualization (already thread-safe inside method)
            self._update_drone_to_target_line(position)
            
            # Combine observations
            observation = np.concatenate([
                vae_latent,                           # 32D (normalized)
                normalized_relative_distance,         # 3D  (normalized relative distance)
                normalized_velocity,                  # 3D  (normalized velocity)
            ]).astype(np.float32)
            
            # Update latest observation (thread-safe)
            with self.observation_lock:
                self.latest_observation = observation
            
            observation_time = time.time() - observation_start
            self.component_times['observation'].append(observation_time)
            
            # Sleep to maintain 0.1s interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.observation_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _log_timing_diagnostics(self):
        """Log detailed timing diagnostics"""
        if len(self.episode_step_times) == 0:
            return
            
        avg_step_time = np.mean(self.episode_step_times)
        max_step_time = np.max(self.episode_step_times)
        min_step_time = np.min(self.episode_step_times)
        
        print(f"\n=== TIMING DIAGNOSTICS (Step {self.current_step}) ===")
        print(f"Step Time: avg={avg_step_time:.3f}s, min={min_step_time:.3f}s, max={max_step_time:.3f}s")
        print(f"Steps per second: {1/avg_step_time:.1f}")
        
        # Component breakdown
        print("Component Times:")
        for component, times in self.component_times.items():
            if times:
                avg_time = np.mean(times)
                total_time = np.sum(times)
                percentage = (total_time / np.sum(self.episode_step_times)) * 100
                print(f"  {component}: {avg_time:.3f}s avg ({percentage:.1f}%)")
        
        # Calculate parallel efficiency
        sequential_estimate = (np.mean(self.component_times['simulation_sleep']) + 
                             np.mean(self.component_times['vae_encoding']) +
                             np.mean(self.component_times['image_processing']))
        parallel_actual = avg_step_time
        if sequential_estimate > 0:
            efficiency_gain = (sequential_estimate - parallel_actual) / sequential_estimate * 100
            print(f"Parallel efficiency gain: {efficiency_gain:.1f}%")
        
        # Memory usage (if available)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"Memory Usage: {memory_mb:.1f} MB")
        except ImportError:
            pass
            
        print("===")

    def _log_episode_summary(self, terminated: bool, truncated: bool):
        """Log comprehensive episode summary"""
        episode_duration = time.time() - self.episode_start_time
        
        print(f"\nEPISODE SUMMARY ")
        print(f"Result: {'SUCCESS' if terminated else 'TIME LIMIT' if truncated else 'UNKNOWN'}")
        print(f"Total Steps: {self.current_step}")
        print(f"Episode Duration: {episode_duration:.2f}s")
        print(f"Average Step Time: {np.mean(self.episode_step_times):.3f}s")
        print(f"Steps per Second: {len(self.episode_step_times)/episode_duration:.1f}")
        
        # Performance metrics
        if len(self.step_times) >= 100:
            recent_avg = np.mean(self.step_times[-100:])
            overall_avg = np.mean(self.step_times)
            print(f"Recent 100-step avg: {recent_avg:.3f}s")
            print(f"Overall avg step time: {overall_avg:.3f}s")
            
            # Estimate training speed
            steps_per_hour = 3600 / recent_avg
            print(f"Estimated training speed: {steps_per_hour:.0f} steps/hour")

    def _get_observation(self) -> np.ndarray:
        """Get current observation without executing action (used in reset, thread-safe)"""
        # Get drone state (thread-safe)
        with self.client_lock:
            drone_state = self.client.getMultirotorState()
        kinematics = drone_state.kinematics_estimated
        
        # Current position
        position = np.array([
            kinematics.position.x_val,
            kinematics.position.y_val, 
            kinematics.position.z_val
        ])
        
        # Current velocity
        velocity = np.array([
            kinematics.linear_velocity.x_val,
            kinematics.linear_velocity.y_val,
            kinematics.linear_velocity.z_val
        ])
        
        # Calculate relative distance vector (target - drone)
        relative_distance = self.target_position - position
        
        # Get depth image and encode with VAE
        depth_map = self._get_depth_image()
        vae_latent = self._encode_depth_map(depth_map)
        
        # Normalize observations
        normalized_relative_distance = relative_distance / self.max_expected_distance
        normalized_relative_distance = np.clip(normalized_relative_distance, -1.0, 1.0)
        
        normalized_velocity = velocity / self.max_expected_velocity
        normalized_velocity = np.clip(normalized_velocity, -1.0, 1.0)
        
        # Update visualization
        self._update_drone_to_target_line(position)
        
        # Combine observations
        observation = np.concatenate([
            vae_latent,                           # 32D (normalized)
            normalized_relative_distance,         # 3D  (normalized relative distance)
            normalized_velocity,                  # 3D  (normalized velocity)
        ])
        
        return observation.astype(np.float32)

    def _get_depth_image(self) -> np.ndarray:
        """Get depth image from AirSim camera (thread-safe)"""
        with self.client_lock:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
            ])
        
        response = responses[0]
        depth_array = np.array(response.image_data_float, dtype=np.float32)
        depth_array = depth_array.reshape(response.height, response.width)
        
        # Normalize and resize to match VAE input
        depth_array = np.clip(depth_array, 0, 100) / 100.0
        depth_array = cv2.resize(depth_array, (128, 72))
        
        return depth_array

    def _encode_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Encode depth map using VAE to get latent vector"""
        with torch.no_grad():
            depth_tensor = torch.FloatTensor(depth_map).unsqueeze(0).unsqueeze(0).to(self.device)
            mu, logvar = self.vae.encode(depth_tensor)
            return mu.cpu().numpy().flatten()

    def _calculate_reward(self, observation: np.ndarray, action: np.ndarray) -> Tuple[float, bool]:
        """Calculate reward and done flag using relative distance"""
        # Extract components from observation
        normalized_relative_distance = observation[32:35]    # normalized relative distance at indices 32-34
        velocity = observation[35:38]                        # velocity at 35-37
        
        # Track client call time separately
        client_call_start = time.time()
        
        # Get current position for bounds checking
        position = self._get_current_position()
        
        total_reward = 0.0
        terminated = False
        reward_breakdown = {}
        
        # 1. Check collision (thread-safe)
        with self.client_lock:
            collision = self.client.simGetCollisionInfo().has_collided
        
        client_call_time = time.time() - client_call_start
        self.component_times['reward_client_calls'].append(client_call_time)
        
        # Now time pure computation (excluding client calls)
        computation_start = time.time()
        if collision:
            reward_breakdown['collision'] = -100.0
            self._log_reward_breakdown(reward_breakdown, total_reward + reward_breakdown['collision'])
            return -100.0, True
        
        # 2. Check out of bounds
        if self._is_out_of_bounds(position):
            reward_breakdown['out_of_bounds'] = -100.0
            self._log_reward_breakdown(reward_breakdown, total_reward + reward_breakdown['out_of_bounds'])
            return -100.0, True
        
        # 3. Check target reached (using actual distance)
        distance_magnitude = np.linalg.norm(normalized_relative_distance)
        if distance_magnitude < 0.02: # 0.02 meters is the threshold for target reached
            # Bonus for fast completion
            time_bonus = max(0, (self.max_steps - self.current_step) / self.max_steps) * 50.0
            reward_breakdown['target_reached'] = 100.0
            reward_breakdown['time_bonus'] = time_bonus
            self._log_reward_breakdown(reward_breakdown, 100.0 + time_bonus)
            return 100.0 + time_bonus, True
        
        # 4. Progress reward using normalized relative distance
        progress_reward = self._calculate_progress_reward(normalized_relative_distance)
        total_reward += progress_reward
        reward_breakdown['progress'] = progress_reward
        
        # 5. Obstacle reward (tiled approach) - this requires a client call
        client_call_start = time.time()
        depth_map = self._get_depth_image()
        client_call_time = time.time() - client_call_start
        self.component_times['reward_client_calls'][-1] += client_call_time
        
        # Pure computation resumes
        obstacle_reward = self._calculate_tiled_obstacle_reward(depth_map, velocity)
        total_reward += obstacle_reward
        reward_breakdown['obstacle'] = obstacle_reward
        
        # 6. Speed reward - encourage maintaining forward speed
        speed_factor = action[0]  # First action component is speed factor
        speed_reward = speed_factor * self.speed_reward_scale
        total_reward += speed_reward
        reward_breakdown['speed'] = speed_reward
        
        # 7. Time penalty - small penalty per step to encourage efficiency
        time_penalty = self.time_penalty_scale
        total_reward -= time_penalty
        reward_breakdown['time_penalty'] = -time_penalty
        
        computation_time = time.time() - computation_start
        self.component_times['reward_computation'].append(computation_time)
        
        # Log reward breakdown every 50 steps
        if self.current_step % 50 == 0:
            self._log_reward_breakdown(reward_breakdown, total_reward)
        
        return total_reward, terminated

    def _log_reward_breakdown(self, reward_breakdown: dict, total_reward: float):
        """Log detailed reward breakdown"""
        print(f"\n--- REWARD BREAKDOWN (Step {self.current_step}) ---")
        print(f"Total Reward: {total_reward:.3f}")
        for component, value in reward_breakdown.items():
            print(f"  {component}: {value:.3f}")
        print("---")

    def _calculate_progress_reward(self, normalized_relative_distance: np.ndarray) -> float:
        """Calculate progress toward target using normalized relative distance"""
        current_normalized_distance = np.linalg.norm(normalized_relative_distance)
        
        if self.previous_normalized_distance is None:
            reward = 0.0
        else:
            # As normalized distance gets smaller, we get positive reward
            reward = (self.previous_normalized_distance - current_normalized_distance) * self.progress_scale
        
        self.previous_normalized_distance = current_normalized_distance
        return reward

    def _calculate_tiled_obstacle_reward(self, depth_map: np.ndarray, velocity: np.ndarray) -> float:
        """Tiled obstacle reward with directional guidance"""
        height, width = depth_map.shape
        
        # Create 2x3 grid of tiles
        tile_height = height // 2
        tile_width = width // 3
        
        tile_means = []
        
        for i in range(2):  # rows
            for j in range(3):  # columns
                row_start = i * tile_height
                row_end = (i + 1) * tile_height
                col_start = j * tile_width
                col_end = (j + 1) * tile_width
                
                tile = depth_map[row_start:row_end, col_start:col_end]
                tile_mean = np.mean(tile)
                tile_means.append(tile_mean)
        
        # Get velocity direction (simplified - no yaw)
        vel_y, vel_z = velocity[1], velocity[2]
        vel_magnitude = np.sqrt(vel_y**2 + vel_z**2)
        
        if vel_magnitude > 0:
            vel_y_norm = vel_y / vel_magnitude
            vel_z_norm = vel_z / vel_magnitude
        else:
            vel_y_norm, vel_z_norm = 0, 0
        
        # Directional weights
        directional_weights = np.array([
            max(0, -vel_y_norm) * max(0, vel_z_norm),    # top-left
            max(0, vel_z_norm),                          # top-middle
            max(0, vel_y_norm) * max(0, vel_z_norm),     # top-right
            max(0, -vel_y_norm) * max(0, -vel_z_norm),   # bottom-left  
            max(0, -vel_z_norm),                         # bottom-middle
            max(0, vel_y_norm) * max(0, -vel_z_norm)     # bottom-right
        ])
        
        # Normalize weights
        if np.sum(directional_weights) > 0:
            directional_weights = directional_weights / np.sum(directional_weights)
        
        # Calculate weighted reward
        weighted_tile_mean = np.sum(tile_means * directional_weights)
        overall_mean = np.mean(depth_map)
        obstacle_reward = self.obstacle_k * (weighted_tile_mean - overall_mean)
        
        return obstacle_reward

    def _is_out_of_bounds(self, position: np.ndarray) -> bool:
        """Check if drone is out of world bounds"""
        x, y, z = position
        x_min, x_max, y_min, y_max, z_min, z_max = self.world_bounds
        return (x < x_min or x > x_max or 
                y < y_min or y > y_max or 
                z < z_min or z > z_max)

    def _get_current_position(self) -> np.ndarray:
        """Helper method to get current position for bounds checking (thread-safe)"""
        with self.client_lock:
            drone_state = self.client.getMultirotorState()
        kinematics = drone_state.kinematics_estimated
        return np.array([
            kinematics.position.x_val,
            kinematics.position.y_val, 
            kinematics.position.z_val
        ])

    # Visualization methods
    def _draw_target_visualization(self):
        """Draw the target point in 3D space (thread-safe)"""
        # Create target point as a red sphere
        target_point = airsim.Vector3r(
            self.target_position[0],
            self.target_position[1], 
            self.target_position[2]
        )
        
        # All visualization calls (thread-safe)
        with self.client_lock:
            # Clear any previous persistent markers
            self.client.simFlushPersistentMarkers()
            
            # Draw the target as a persistent red sphere
            self.client.simPlotPoints(
                points=[target_point],
                color_rgba=[1.0, 0.0, 0.0, 1.0],  # Red, fully opaque
                size=25.0,  # Larger size for visibility
                duration=-1,  # Persistent (-1 means forever)
                is_persistent=True
            )
            
            # Also draw a text label at the target
            self.client.simPlotStrings(
                strings=["TARGET"],
                positions=[target_point],
                scale=2.0,
                color_rgba=[1.0, 0.0, 0.0, 1.0],
                duration=-1
            )
        
        print(f"Target visualization drawn at: {self.target_position}")

    def _update_drone_to_target_line(self, drone_position: np.ndarray):
        """Draw a line between current drone position and target point (thread-safe)"""
        # Convert positions to AirSim Vector3r
        drone_point = airsim.Vector3r(
            drone_position[0],
            drone_position[1],
            drone_position[2]
        )
        
        target_point = airsim.Vector3r(
            self.target_position[0],
            self.target_position[1],
            self.target_position[2]
        )
        
        # Draw line from drone to target (green, semi-transparent, thread-safe)
        with self.client_lock:
            self.client.simPlotLineList(
                points=[drone_point, target_point],
                color_rgba=[0.0, 1.0, 0.0, 0.7],  # Green, semi-transparent
                thickness=3.0,
                duration=0.2,  # Temporary - will be updated next step
                is_persistent=False
            )

    def close(self):
        """Clean up environment"""
        # Stop background threads FIRST
        self._stop_background_threads()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # All cleanup operations (thread-safe)
        with self.client_lock:
            # Unpause before closing
            self.client.simPause(False)
            
            # Clear all visualizations
            self.client.simFlushPersistentMarkers()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)