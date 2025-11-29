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
import os
from datetime import datetime

STEP_LOGGING = False
DEPTH_MAP_LOGGING = False
FORWARD_ONLY = False

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

        self.current_forward_velocity = 0.
        self.current_lateral_velocity = 0.
        self.current_vertical_velocity = 0.
        
        # Load pre-trained VAE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = self._load_vae(vae_model_path)
        
        # Environment parameters
        self.target_position = target_position
        self.world_bounds = world_bounds  # [x_min, x_max, y_min, y_max, z_min, z_max]
        self.max_steps = max_steps
        self.current_step = 0
        self._last_dt = 0.03
        
        # SIMPLIFIED: Action space: 3D continuous [speed_factor, lateral, vertical]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
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
        self.base_forward_speed = 10.0    # m/s (base speed when speed_factor=1)
        self.max_lateral_speed = 8.0     # m/s
        self.max_vertical_speed = 8.0    # m/s
        
        # Normalize/validate world bounds ordering (ensure mins <= maxs)
        x_min, x_max, y_min, y_max, z_min, z_max = self.world_bounds
        self.world_bounds = [
            min(x_min, x_max), max(x_min, x_max),
            min(y_min, y_max), max(y_min, y_max),
            min(z_min, z_max), max(z_min, z_max)
        ]

        # Reward function parameters
        self.progress_scale = 50.0
        self.progress_exponent = 1.2  # Exponent for progress reward (higher = more reward near target)
        self.obstacle_k = 1.0
        self.previous_normalized_distance = None
        self.previous_velocity = None  # Track previous velocity for smoothness penalty
        self.velocity_change_penalty_scale = 2.0  # Penalty for large velocity changes
        self.velocity_magnitude_penalty = 1.5    # Penalty for larger velocities
        self.velocity_magnitude_collision_scalar_penalty = 0.5  # Penalty for large velocity collisions
        self.overshot_penalty = -200.0  # Penalty for overshooting the target
        
        # Timer reward parameters
        self.speed_reward_scale = 0.1    # Reward for maintaining speed
        self.time_penalty_scale = 0.01   # Small penalty per step to encourage efficiency
        # Boundary warning distance (meters). Within this distance, apply scaled penalty to -100 at boundary
        self.boundary_warning_distance = 3.0
        self.boundary_penalty_scale = 3.0
        # Directional weighting strength for 3x3 tiles (0 = uniform, 1 = fully directional)
        self.direction_weight_k = 0.3
        # Small tolerance for boundary checks to avoid false positives at edges
        self.boundary_epsilon = 0.05

        self.start_position = [0.0, 0.0, -3.0]

        # Normalization parameters
        self.max_expected_distance = np.linalg.norm(
            np.array([self.target_position[0] - self.start_position[0],
                     self.target_position[1] - self.start_position[1],
                     self.target_position[2] - self.start_position[2]])
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
        self.episode_reward_total = 0.0
        self.episode_rewards_history = []  # store per-episode totals for rolling stats
        self.last_termination_reason = None
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
        self.latest_depth_map = None  # Cache depth map to avoid re-fetching in reward calculation
        self.observation_lock = threading.Lock()
        self.client_lock = threading.Lock()  # Lock for AirSim client (not thread-safe)
        
        # Control flags for background observation thread
        self.observation_thread_running = False
        self.observation_thread = None
        
        # Control flags for background depth map thread
        self.depth_map_thread_running = False
        self.depth_map_thread = None
        self.depth_map_interval = 0.2  # Update depth map every 0.2s
        self._last_step_time = None
        
        # Observation collection interval 0.2, but simulation runs at x2 clock speed
        self.observation_interval = 0.2
        
        # Setup logging to file
        self.log_file = None
        self._setup_logging()  

    def _setup_logging(self):
        """Setup logging to file"""
        # Create eval_logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), "eval_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.log(f"=== Training Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    def log(self, message: str):
        """Log message to both console and file"""
        print(message, end='')
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()  # Ensure it's written immediately

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

        self.current_forward_velocity = 0.
        self.current_lateral_velocity = 0.
        self.current_vertical_velocity = 0.
        
        # Make sure simulation is unpaused for reset (thread-safe)
        with self.client_lock:
            self.client.simPause(False)
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        
        # Takeoff and set to initial position (thread-safe)
        with self.client_lock:
            self.client.takeoffAsync().join() # x, y, z (z is up, negative in AirSim)
            self.client.moveToPositionAsync(*self.start_position, 5).join()
        
        self.current_step = 0
        self.previous_normalized_distance = None
        self.previous_velocity = None  # Reset velocity tracking
        self.episode_start_time = time.time()
        self.episode_step_times = []
        self.episode_reward_total = 0.0
        self.component_times = {k: [] for k in self.component_times.keys()}
        self.last_termination_reason = None
        
        # Draw target visualization (thread-safe)
        self._draw_target_visualization()
        # Draw world bounds wireframe for debugging
        self._draw_world_bounds()
        
        # Initialize observation
        observation = self._get_observation()
        # Also cache the depth map for reward calculation
        depth_map = self._get_depth_image()
        with self.observation_lock:
            self.latest_observation = observation
            self.latest_depth_map = depth_map.copy()
        
        # Start background observation thread
        self._start_background_threads()
        
        # Keep simulation unpaused - observation thread and actions will run continuously
        with self.client_lock:
            self.client.simPause(False)
        
        info = {}
        self._last_step_time = time.time()
        
        return observation, info

    # Gymnasium step function
    def step(self, action: np.ndarray):
        """Execute one environment step - loop action while waiting for observation"""
        step_start_time = time.time()
        self.current_step += 1
        
        # Execute action synchronously (thread-safe)
        action_start = time.time()

        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Action execution START at {action_start:.6f}\n")
        
        speed_factor, lateral, vertical = action
        
        # Calculate velocities from simplified actions

        self.current_forward_velocity += speed_factor * self.base_forward_speed
        self.current_lateral_velocity += lateral * self.max_lateral_speed
        self.current_vertical_velocity += vertical * self.max_vertical_speed

        self.current_forward_velocity = np.clip(self.current_forward_velocity, min=-self.base_forward_speed,max=self.base_forward_speed)
        self.current_lateral_velocity = np.clip(self.current_lateral_velocity, min=-self.max_lateral_speed,max=self.max_lateral_speed)
        self.current_vertical_velocity = np.clip(self.current_vertical_velocity, min=-self.max_vertical_speed,max=self.max_vertical_speed)

        if FORWARD_ONLY:
            self.current_forward_velocity = max(self.current_forward_velocity, 0.0)

        # Send command to AirSim (thread-safe, non-blocking - simulation runs continuously)
        action_send_start = time.time()
        with self.client_lock:
            self.client.moveByVelocityAsync(
                float(self.current_forward_velocity),
                float(self.current_lateral_velocity),
                float(self.current_vertical_velocity),
                duration=0.07 # Command duration 0.4, but simulation runs at x2 clock speed
            )
        action_send_time = time.time() - action_send_start

        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Action sent to AirSim in {action_send_time:.6f}s\n")
        
        action_time = time.time() - action_start
        self.component_times['action_execution'].append(action_time)
        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Action execution END, Total duration: {action_time:.6f}s\n")
        
        # Get latest observation (collected by background thread)
        # Log observation timing details
        observation_start = time.time()
        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Observation collection START at {observation_start:.6f}\n")
        
        lock_acquire_start = time.time()
        with self.observation_lock:
            lock_acquire_time = time.time() - lock_acquire_start
            if(STEP_LOGGING):
                self.log(f"[Step {self.current_step}] Lock acquired in {lock_acquire_time:.6f}s\n")
            
            observation_get_start = time.time()
            observation = self.latest_observation.copy() if self.latest_observation is not None else self._get_observation()
            observation_get_time = time.time() - observation_get_start
            if(STEP_LOGGING):
                self.log(f"[Step {self.current_step}] Observation retrieved in {observation_get_time:.6f}s\n")
        
        observation_end = time.time()
        observation_time = observation_end - observation_start
        self.component_times['observation'].append(observation_time)
        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Observation collection END at {observation_end:.6f}, Total duration: {observation_time:.6f}s\n")
        
        # Calculate reward
        reward_start = time.time()
        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Reward calculation START at {reward_start:.6f}\n")
        reward, terminated = self._calculate_reward(observation, action)
        reward_time = time.time() - reward_start
        self.component_times['reward_calculation'].append(reward_time)
        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Reward calculation END, Total duration: {reward_time:.6f}s\n")
        
        # Accumulate episode reward
        self.episode_reward_total += float(reward)
        
        # Check episode truncation (max steps)
        truncated = self.current_step >= self.max_steps
        if truncated:
            self.last_termination_reason = 'time_limit'
        
        # Calculate total step time
        step_duration = time.time() - step_start_time
        self.episode_step_times.append(step_duration)
        self.step_times.append(step_duration)
        
        # Log step timing summary
        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] STEP SUMMARY - Total: {step_duration:.6f}s | "
                    f"Action: {action_time:.6f}s | Observation: {observation_time:.6f}s | "
                    f"Reward: {reward_time:.6f}s\n")
        
        # Enhanced timing diagnostics
        if self.current_step % 1000 == 0:
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
            # Keep history for rolling stats
            self.episode_rewards_history.append(self.episode_reward_total)
            self._log_episode_summary(terminated, truncated)

        
        return observation, reward, terminated, truncated, info

    def _start_background_threads(self):
        """Start background threads for observation collection and depth map updates"""
        # Start observation collection thread
        self.observation_thread_running = True
        self.observation_thread = threading.Thread(target=self._observation_collection_loop, daemon=True)
        self.observation_thread.start()
        
        # Start depth map collection thread (separate, independent)
        self.depth_map_thread_running = True
        self.depth_map_thread = threading.Thread(target=self._depth_map_collection_loop, daemon=True)
        self.depth_map_thread.start()
        
    def _stop_background_threads(self):
        """Stop background threads"""
        # Stop observation thread
        self.observation_thread_running = False
        if self.observation_thread is not None:
            self.observation_thread.join(timeout=1.0)
        
        # Stop depth map thread
        self.depth_map_thread_running = False
        if self.depth_map_thread is not None:
            self.depth_map_thread.join(timeout=1.0)
    
    def _observation_collection_loop(self):
        """Background thread: Continuously collect observations every 0.1s in parallel"""
        while self.observation_thread_running:
            loop_start = time.time()
            
            # Collect observation in parallel
            observation_start = time.time()
            
            # Step 1: Get depth map from cache (updated by depth_map_collection_loop thread)
            # This avoids duplicate fetching - depth map thread already keeps it fresh
            image_start = time.time()
            with self.observation_lock:
                if self.latest_depth_map is not None:
                    depth_map = self.latest_depth_map.copy()  # Use cached depth map
                else:
                    # Fallback: fetch if cache not available (shouldn't happen after thread starts)
                    depth_map = self._get_depth_image()
                    print("Warning: Depth map cache miss, fetched from AirSim")
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
            # Note: Depth map is updated by separate depth_map_collection_loop thread
            with self.observation_lock:
                self.latest_observation = observation
            
            observation_time = time.time() - observation_start
            self.component_times['observation'].append(observation_time)
            
            # Sleep to maintain observation interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.observation_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    

    # Tiger Depth Map Collection Loop: So this background thread can collect depth maps independently of the observation collection thread
    # TODO: Fetch Unreal Engine Color Map
    # TODO: Convert Color Map to Depth Map using Depth Anything 2.0
    # this whole thread is recorded for timing diagnostics, don't remove any of that code in here.
    def _depth_map_collection_loop(self):
        depth_map_count = 0
        depth_map_times = []
        while self.depth_map_thread_running:
            loop_start = time.time()
            depth_map_count += 1
            
            # get depth map (thread-safe)
            depth_map_start = time.time()
            depth_map = self._get_depth_image()
            depth_map_time = time.time() - depth_map_start
            depth_map_times.append(depth_map_time)
            
            # update cached depth map (thread-safe)
            #The lock protects self.latest_depth_map, which is accessed by multiple threads:
            #Depth map thread (line 441): writes self.latest_depth_map = depth_map.copy()
            #Observation thread (line 359): reads self.latest_depth_map for VAE encoding
            #Reward calculation (line 692): reads self.latest_depth_map for obstacle reward
            with self.observation_lock:
                self.latest_depth_map = depth_map.copy()
            # Print depth map generation speed

            if(DEPTH_MAP_LOGGING):
                step_info = f" (Step {self.current_step})" if hasattr(self, 'current_step') else ""
                print(f"[DepthMap Thread{step_info}] Depth map #{depth_map_count} generated in {depth_map_time:.6f}s")
                
                # Print statistics every 10 depth maps
                if depth_map_count % 10 == 0:
                    recent_times = depth_map_times[-10:]
                    avg_time = np.mean(recent_times)
                    min_time = np.min(recent_times)
                    max_time = np.max(recent_times)
                    depth_maps_per_sec = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"[DepthMap Thread{step_info}] Stats (last 10): Avg={avg_time:.6f}s, Min={min_time:.6f}s, Max={max_time:.6f}s, Speed={depth_maps_per_sec:.2f} maps/sec")
                
            # Sleep to maintain depth map update interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.depth_map_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _log_timing_diagnostics(self):
        """Log detailed timing diagnostics"""
        if len(self.episode_step_times) == 0:
            return
            
        avg_step_time = np.mean(self.episode_step_times)
        max_step_time = np.max(self.episode_step_times)
        min_step_time = np.min(self.episode_step_times)
        
        self.log(f"\n=== TIMING DIAGNOSTICS (Step {self.current_step}) ===\n")
        self.log(f"Step Time: avg={avg_step_time:.3f}s, min={min_step_time:.3f}s, max={max_step_time:.3f}s\n")
        self.log(f"Steps per second: {1/avg_step_time:.1f}\n")
        
        # Component breakdown
        self.log("Component Times:\n")
        for component, times in self.component_times.items():
            if times:
                avg_time = np.mean(times)
                total_time = np.sum(times)
                percentage = (total_time / np.sum(self.episode_step_times)) * 100
                self.log(f"  {component}: {avg_time:.3f}s avg ({percentage:.1f}%)\n")
        
        # Calculate parallel efficiency
        if self.component_times.get('simulation_sleep') and self.component_times.get('vae_encoding') and self.component_times.get('image_processing'):
            sequential_estimate = (np.mean(self.component_times['simulation_sleep']) + 
                                 np.mean(self.component_times['vae_encoding']) +
                                 np.mean(self.component_times['image_processing']))
            parallel_actual = avg_step_time
            if sequential_estimate > 0:
                efficiency_gain = (sequential_estimate - parallel_actual) / sequential_estimate * 100
                self.log(f"Parallel efficiency gain: {efficiency_gain:.1f}%\n")
        
        # Memory usage (if available)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.log(f"Memory Usage: {memory_mb:.1f} MB\n")
        except ImportError:
            pass
            
        self.log("===\n")

    def _log_episode_summary(self, terminated: bool, truncated: bool):
        """Log comprehensive episode summary"""
        episode_duration = time.time() - self.episode_start_time
        
        self.log(f"\nEpisode \n")
        # Reason detail for termination
        if terminated:
            reason = self.last_termination_reason or 'success'
        elif truncated:
            reason = 'time_limit'
        else:
            reason = 'unknown'
        self.log(f"Result: {reason.upper()}\n")
        self.log(f"Total Steps: {self.current_step}\n")
        self.log(f"Episode Reward: {self.episode_reward_total:.2f}\n")
        self.log(f"Episode Duration: {episode_duration:.2f}s\n")
        self.log(f"Average Step Time: {np.mean(self.episode_step_times):.3f}s\n")
        self.log(f"Steps per Second: {len(self.episode_step_times)/episode_duration:.1f}\n")
        
        # Rolling mean of last 10 episodes
        if len(self.episode_rewards_history) > 0:
            last_n = self.episode_rewards_history[-10:]
            mean_last_10 = float(np.mean(last_n))
            self.log(f"Mean Reward (last {len(last_n)}): {mean_last_10:.2f}\n")
        
        # Performance metrics
        if len(self.step_times) >= 100:
            recent_avg = np.mean(self.step_times[-100:])
            overall_avg = np.mean(self.step_times)
            self.log(f"Recent 100-step avg: {recent_avg:.3f}s\n")
            self.log(f"Overall avg step time: {overall_avg:.3f}s\n")
            
            # Estimate training speed
            steps_per_hour = 3600 / recent_avg
            self.log(f"Estimated training speed: {steps_per_hour:.0f} steps/hour\n")

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
        reward_func_start = time.time()

        
        # Extract components from observation
        normalized_relative_distance = observation[32:35]    # normalized relative distance at indices 32-34
        velocity = observation[35:38]                        # velocity at 35-37
        velocity_magnitude = np.linalg.norm(velocity)
        
        # Track client call time separately
        client_call_start = time.time()
        
        # Get current position for bounds checking
        position_start = time.time()
        position = self._get_current_position()
        position_time = time.time() - position_start

        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Reward: Get position took {position_time:.6f}s\n")

        total_reward = 0.0
        terminated = False
        reward_breakdown = {}
        
        # 1. Check collision (thread-safe)
        collision_check_start = time.time()
        with self.client_lock:
            collision = self.client.simGetCollisionInfo().has_collided
        collision_check_time = time.time() - collision_check_start

        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Reward: Collision check took {collision_check_time:.6f}s\n")
        
        client_call_time = time.time() - client_call_start
        self.component_times['reward_client_calls'].append(client_call_time)
        
        # Now time pure computation (excluding client calls)
        computation_start = time.time()
        if collision:
            reward_breakdown['collision'] = -200.0
            self._log_reward_breakdown(reward_breakdown, total_reward + reward_breakdown['collision'])
            self.last_termination_reason = 'collision'
            return total_reward + reward_breakdown['collision'], True
        
        # 2. Boundary penalty (unified):
        # - If out-of-bounds (min distance <= 0): terminate with -100
        # - If within warning distance: apply scaled penalty up to -100 at boundary
        x, y, z = position
        x_min, x_max, y_min, y_max, z_min, z_max = self.world_bounds
        distances_to_boundaries = [
            x - x_min,
            x_max - x,
            y - y_min,
            y_max - y,
            z - z_min,
            z_max - z
        ]
        min_distance_to_boundary = float(min(distances_to_boundaries))
        if min_distance_to_boundary <= -self.boundary_epsilon:
            reward_breakdown['boundary'] = -200.0
            # Add context for debugging unexpected OOB
            self.log(f"OOB: pos={position}, bounds={self.world_bounds}\n")
            self._log_reward_breakdown(reward_breakdown, total_reward + reward_breakdown['boundary'])
            self.last_termination_reason = 'out_of_bounds'
            return total_reward + reward_breakdown['boundary'], True
        boundary_penalty = 0.0
        if min_distance_to_boundary < self.boundary_warning_distance:
            safe_ratio = max(0.0, min_distance_to_boundary / self.boundary_warning_distance)
            boundary_penalty = -self.boundary_penalty_scale * (1.0 - safe_ratio)
            total_reward += boundary_penalty
        reward_breakdown['boundary'] = boundary_penalty
        
        # 3. Check target reached (using actual distance)
        distance_magnitude = np.linalg.norm(normalized_relative_distance)
        if distance_magnitude < 0.066: # 0.02 meters is the threshold for target reached
            # Bonus for fast completion
            time_bonus = max(0, (self.max_steps - self.current_step) / self.max_steps) * 50.0
            reward_breakdown['target_reached'] = 100.0
            reward_breakdown['time_bonus'] = time_bonus
            # Penalty for completion at high speed
            reward_breakdown['target_reached'] *= 1. if velocity_magnitude <= 0.5 else max(0., 1 - ((velocity_magnitude - 0.5) / 20.0))
            self._log_reward_breakdown(reward_breakdown, reward_breakdown['target_reached'] + time_bonus)
            self.last_termination_reason = 'target_reached'
            return 100.0 + time_bonus, True
        
        # 3b. Check if drone has overshot target on X axis
        # Target is at x=50.0, drone starts at x=0.0, so if drone x > 50.0, it has overshot
        drone_x = position[0]
        target_x = self.target_position[0]
        
        # Check if we've passed the target (overshot forward)
        if drone_x > target_x + 10.:
            # Drone has passed the target on X axis - terminate with negative reward
            reward_breakdown['missed_target'] = self.overshot_penalty
            final_reward = total_reward + self.overshot_penalty
            self._log_reward_breakdown(reward_breakdown, final_reward)
            self.last_termination_reason = 'missed_target_overshot_x'
            return final_reward, True  # Return True to terminate episode, Gymnasium will call reset()
        
        # 4. Progress reward using normalized relative distance
        progress_reward = self._calculate_progress_reward(normalized_relative_distance)
        total_reward += progress_reward
        reward_breakdown['progress'] = progress_reward
        
        # 5. Obstacle reward (tiled approach) - reuse cached depth map from observation thread
        depth_image_start = time.time()

        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Reward: Getting depth image START at {depth_image_start:.6f}\n")
        
        # Use cached depth map from observation thread (no client call needed!)
        with self.observation_lock:
            if self.latest_depth_map is not None:
                depth_map = self.latest_depth_map.copy()
                if(STEP_LOGGING):
                    self.log(f"[Step {self.current_step}] Reward: Using cached depth map (no client call!)\n")
            else:
                # Fallback: fetch if cache is not available (shouldn't happen normally)
                if(STEP_LOGGING):
                    self.log(f"[Step {self.current_step}] Reward: WARNING - Cache miss, fetching depth image\n")
                client_call_start = time.time()
                depth_map = self._get_depth_image()
                client_call_time = time.time() - client_call_start
                self.component_times['reward_client_calls'][-1] += client_call_time
        
        depth_image_time = time.time() - depth_image_start
        if(STEP_LOGGING):
            self.log(f"[Step {self.current_step}] Reward: Getting depth image took {depth_image_time:.6f}s\n")
        
        # Pure computation resumes
        obstacle_calc_start = time.time()
        obstacle_reward = self._calculate_tiled_obstacle_reward(depth_map, velocity)
        obstacle_calc_time = time.time() - obstacle_calc_start

        # If not only moving forward, the vehicle may have a tendency to optimize reward by moving in the opposite
        # of any obstacle
        if FORWARD_ONLY:
            if(STEP_LOGGING):
                self.log(f"[Step {self.current_step}] Reward: Obstacle reward calculation took {obstacle_calc_time:.6f}s\n")
            total_reward += obstacle_reward
            reward_breakdown['obstacle'] = obstacle_reward

        # 6. High velocity penalty - Discourage higher velocities. Reward strategy taken from
        #    https://arxiv.org/pdf/2509.13943
        total_reward -= velocity_magnitude**2 * self.velocity_magnitude_penalty

        # Update previous velocity for next step
        self.previous_velocity = velocity.copy()
        
        # 7. Time penalty - small penalty per step to encourage efficiency
        time_penalty = self.time_penalty_scale
        total_reward -= time_penalty
        reward_breakdown['time_penalty'] = -time_penalty
        
        computation_time = time.time() - computation_start
        self.component_times['reward_computation'].append(computation_time)

        if self.current_step >= self.max_steps:
            return -200.0, terminated
        
        # Log reward breakdown every 1000 steps ( NO Point of now, too spamming)
        #if self.current_step % 10 == 0:
        #    self._log_reward_breakdown(reward_breakdown, total_reward)

        self._last_dt = time.time() - self._last_step_time
        self._last_step_time = time.time()
        
        return total_reward * self._last_dt, terminated

    def _log_reward_breakdown(self, reward_breakdown: dict, total_reward: float):
        """Log detailed reward breakdown"""
        self.log(f"\n--- REWARD BREAKDOWN (Step {self.current_step}) ---\n")
        self.log(f"Total Reward: {total_reward:.3f}\n")
        for component, value in reward_breakdown.items():
            self.log(f"  {component}: {value:.3f}\n")
        self.log("---\n")

    def _calculate_progress_reward(self, normalized_relative_distance: np.ndarray) -> float:
        """Calculate progress reward based on current distance to target (every step)
        
        Reward is given every step based on how close we are to the target.
        Uses exponential scaling: closer = exponentially more reward.
        This provides continuous feedback and stronger incentive to get very close.
        """
        current_normalized_distance = min(float(np.linalg.norm(normalized_relative_distance)), 1.0)
        
        # Exponential reward based on current distance: closer = exponentially more reward
        # Distance is normalized (0-1), so reward = (1 - distance)^exponent * scale
        # When distance = 0 (at target): reward = 1^exponent * scale = max reward
        # When distance = 1 (far): reward = 0^exponent * scale = 0 reward
        # With exponent > 1, reward increases faster as we get closer
        closeness = 1.0 - current_normalized_distance
        reward = (closeness ** self.progress_exponent) * self.progress_scale
        
        # Also track for logging/debugging
        self.previous_normalized_distance = current_normalized_distance
        
        return reward

    def _calculate_tiled_obstacle_reward(self, depth_map: np.ndarray, velocity: np.ndarray) -> float:
        """Tiled obstacle reward with directional guidance (3x3, first-person).
        - 3x3 tiles: top-left..bottom-right
        - Center tile uses neutral weighting (no velocity-based emphasis)
        - Other tiles weighted by lateral (y) and vertical (z) velocity direction
        """
        height, width = depth_map.shape

        # Create 3x3 grid of tiles
        tile_height = height // 3
        tile_width = width // 3

        tile_means = []  # order: [tl, tc, tr, ml, mc, mr, bl, bc, br]
        for i in range(3):  # rows: 0=top,1=middle,2=bottom
            for j in range(3):  # cols: 0=left,1=center,2=right
                row_start = i * tile_height
                row_end = (i + 1) * tile_height if i < 2 else height
                col_start = j * tile_width
                col_end = (j + 1) * tile_width if j < 2 else width

                tile = depth_map[row_start:row_end, col_start:col_end]
                tile_means.append(float(np.mean(tile)))

        # Velocity-based directional emphasis
        vel_y, vel_z = float(velocity[1]), float(velocity[2])
        vel_mag = np.sqrt(vel_y**2 + vel_z**2)
        if vel_mag > 1e-6:
            vel_y_norm = vel_y / vel_mag
            vel_z_norm = vel_z / vel_mag
        else:
            vel_y_norm, vel_z_norm = 0.0, 0.0

        # Column weights from lateral velocity (left/right); center column neutral
        col_left = max(0.0, -vel_y_norm)
        col_center = 1.0  # neutral for center column
        col_right = max(0.0, vel_y_norm)

        # Row weights from vertical velocity (up/down); center row neutral
        # Note: In AirSim, z up is negative; moving up => vel_z < 0 -> emphasize top row
        row_top = max(0.0, -vel_z_norm)
        row_mid = 1.0  # neutral for center row
        row_bot = max(0.0, vel_z_norm)

        # Build 3x3 weights grid (center tile neutral, others directional)
        weights_grid = np.array([
            [row_top * col_left,  row_top * col_center,  row_top * col_right],
            [row_mid * col_left,  1.0,                   row_mid * col_right],  # center tile fixed 1.0
            [row_bot * col_left,  row_bot * col_center,  row_bot * col_right],
        ], dtype=np.float32)

        weights = weights_grid.flatten()

        # Blend between uniform and directional weights: blended = (1-k)*uniform + k*directional
        num_tiles = weights.shape[0]
        uniform = np.ones_like(weights, dtype=np.float32) / float(num_tiles)
        w_sum = float(np.sum(weights))
        if w_sum > 1e-6:
            weights = weights / w_sum
        else:
            weights = uniform.copy()
        k = float(self.direction_weight_k)
        blended_weights = (1.0 - k) * uniform + k * weights
        blended_weights = blended_weights / float(np.sum(blended_weights))

        # Calculate weighted openness: prefer tiles with larger depth (more free space)
        tile_means_arr = np.array(tile_means, dtype=np.float32)
        weighted_tile_mean = float(np.sum(tile_means_arr * blended_weights))
        overall_mean = float(np.mean(depth_map))

        # Reward moving towards open space vs overall scene openness
        obstacle_reward = self.obstacle_k * (weighted_tile_mean - overall_mean)
        return obstacle_reward

    def _is_out_of_bounds(self, position: np.ndarray) -> bool:
        """Check if drone is out of world bounds"""
        x, y, z = position
        x_min, x_max, y_min, y_max, z_min, z_max = self.world_bounds
        # Use epsilon tolerance near edges to avoid false positives
        eps = self.boundary_epsilon if hasattr(self, 'boundary_epsilon') else 0.0
        return (x < x_min - eps or x > x_max + eps or 
                y < y_min - eps or y > y_max + eps or 
                z < z_min - eps or z > z_max + eps)

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
        
        self.log(f"Target visualization drawn at: {self.target_position}\n")

    def _draw_world_bounds(self):
        """Draw a wireframe box representing world bounds (thread-safe)"""
        x_min, x_max, y_min, y_max, z_min, z_max = self.world_bounds
        # 8 corners of the box
        corners = [
            airsim.Vector3r(x_min, y_min, z_min),
            airsim.Vector3r(x_max, y_min, z_min),
            airsim.Vector3r(x_max, y_max, z_min),
            airsim.Vector3r(x_min, y_max, z_min),
            airsim.Vector3r(x_min, y_min, z_max),
            airsim.Vector3r(x_max, y_min, z_max),
            airsim.Vector3r(x_max, y_max, z_max),
            airsim.Vector3r(x_min, y_max, z_max),
        ]

        # Edges (pairs of indices)
        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom rectangle
            (4,5),(5,6),(6,7),(7,4),  # top rectangle
            (0,4),(1,5),(2,6),(3,7)   # verticals
        ]

        # Build line list
        line_points = []
        for a, b in edges:
            line_points.append(corners[a])
            line_points.append(corners[b])

        # Draw persistent wireframe box
        with self.client_lock:
            self.client.simPlotLineList(
                points=line_points,
                color_rgba=[0.0, 0.5, 1.0, 0.8],  # cyan-ish
                thickness=3.0,
                duration=-1.0,
                is_persistent=True
            )
        self.log("World bounds wireframe drawn.\n")

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
        
        # Log closing
        if self.log_file:
            self.log(f"\n=== Training Log Ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            self.log_file.close()
            self.log_file = None
        
        # All cleanup operations (thread-safe)
        with self.client_lock:
            # Unpause before closing
            self.client.simPause(False)
            
            # Clear all visualizations
            self.client.simFlushPersistentMarkers()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)