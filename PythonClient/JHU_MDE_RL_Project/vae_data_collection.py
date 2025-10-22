import airsim
import numpy as np
import cv2
import os
import time
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir="./vae_data"):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "depth_images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Flight boundaries in NED coordinates (meters) - only XY now
        self.bounds = {
            'x_min': -100, 'x_max': 100,
            'y_min': -100, 'y_max': 100
        }
        
        # Ideal height range (NED coordinates: negative = above ground)
        self.ideal_height_min = -3.5  # 3.5 meters above ground
        self.ideal_height_max = -3.0  # 3.0 meters above ground
        
        # Data collection parameters
        self.max_flight_time = 300
        self.image_interval = 0.5
        self.total_images_target = 10000
        
    def get_depth_image(self, camera_name="0"):
        """Capture depth image from AirSim"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    camera_name, 
                    airsim.ImageType.DepthPlanar,
                    pixels_as_float=True, 
                    compress=False
                )
            ])
            
            if responses and responses[0]:
                depth_array = self._parse_depth_response(responses[0])
                return depth_array
            return None
        except Exception as e:
            print(f"Error getting depth image: {e}")
            return None
    
    def _parse_depth_response(self, response):
        """Parse AirSim depth image response"""
        try:
            if hasattr(response, 'image_data_float') and response.image_data_float:
                if isinstance(response.image_data_float, list):
                    depth_img = np.array(response.image_data_float, dtype=np.float32)
                else:
                    depth_img = np.frombuffer(response.image_data_float, dtype=np.float32)
                
                depth_img = depth_img.reshape(response.height, response.width)
                return depth_img
            else:
                print("No valid image data found in response")
                return np.zeros((response.height, response.width), dtype=np.float32)
                
        except Exception as e:
            print(f"Error parsing depth response: {e}")
            if hasattr(response, 'height') and hasattr(response, 'width'):
                return np.zeros((response.height, response.width), dtype=np.float32)
            else:
                return np.zeros((72, 128), dtype=np.float32)
    
    def depth_to_visual(self, depth_image):
        """Convert depth image to visualizable PNG format"""
        depth_visual = np.clip(depth_image / 100.0 * 255, 0, 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
        return depth_colored
    
    def save_depth_image(self, depth_image, filename):
        """Save depth image in both PNG (visual) and numpy format"""
        png_path = os.path.join(self.images_dir, f"{filename}.png")
        depth_visual = self.depth_to_visual(depth_image)
        cv2.imwrite(png_path, depth_visual)
        
        npy_path = os.path.join(self.images_dir, f"{filename}.npy")
        depth_normalized = np.clip(depth_image / 100.0, 0.0, 1.0)
        np.save(npy_path, depth_normalized)
    
    def get_height_adjustment(self, current_height):
        """Calculate vertical velocity to maintain ideal height range"""
        # If within ideal range, no vertical movement
        if self.ideal_height_min <= current_height <= self.ideal_height_max:
            return 0.0
        
        # If below ideal range (too low), move up (negative velocity in NED)
        elif current_height > self.ideal_height_max:
            height_error = current_height - self.ideal_height_max
            vz = -min(abs(height_error) * 0.8, 1.0)  # Proportional control, max 1 m/s
            print(f"Too low ({current_height:.2f}m), moving up: {vz:.2f} m/s")
            return vz
        
        # If above ideal range (too high), move down (positive velocity in NED)
        else:  # current_height < self.ideal_height_min
            height_error = current_height - self.ideal_height_min
            vz = min(abs(height_error) * 0.8, 1.0)  # Proportional control, max 1 m/s
            print(f"Too high ({current_height:.2f}m), moving down: {vz:.2f} m/s")
            return vz
    
    def random_move(self):
        """Generate random movement command in m/s with height maintenance"""
        # Get current height to determine vertical adjustment
        position = self.get_drone_position()
        current_height = position[2]
        
        # Horizontal movement
        vx = np.random.uniform(-3, 3)
        vy = np.random.uniform(-3, 3)
        
        # Vertical movement based on height maintenance
        vz = self.get_height_adjustment(current_height)
        
        yaw_rate = np.random.uniform(-45, 45)
        
        return vx, vy, vz, yaw_rate
    
    def is_safe_position(self, position):
        """Check if position is within safe XY bounds (Z bounds removed)"""
        x, y, z = position

        x_ok = self.bounds['x_min'] <= x <= self.bounds['x_max']
        y_ok = self.bounds['y_min'] <= y <= self.bounds['y_max']
        
        if not (x_ok and y_ok):
            print(f"OUT OF BOUNDS - X:{x:.2f}, Y:{y:.2f}")
        
        return x_ok and y_ok
    
    def get_drone_position(self):
        """Get current drone position in NED coordinates (meters)"""
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        return position.x_val, position.y_val, position.z_val
    
    def collect_data(self):
        """Main data collection routine"""
        print("Starting data collection...")
        print(f"Ideal height range: {abs(self.ideal_height_min):.1f} to {abs(self.ideal_height_max):.1f} meters above ground")
        
        # Get initial position
        position = self.get_drone_position()
        print(f"Starting Position (NED): X:{position[0]:.2f}, Y:{position[1]:.2f}, Z:{position[2]:.2f}")
        
        # Takeoff
        print("Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(3)
                
        # Check position after takeoff
        position = self.get_drone_position()
        print(f"After takeoff Position (NED): X:{position[0]:.2f}, Y:{position[1]:.2f}, Z:{position[2]:.2f}")
        
        # Move to middle of ideal height range
        target_height = (self.ideal_height_min + self.ideal_height_max) / 2
        print(f"Moving to ideal height: {abs(target_height):.1f}m above ground")
        self.client.moveToZAsync(target_height, 3).join()
        time.sleep(2)

        # Check position after moving to height
        position = self.get_drone_position()
        print(f"At ideal height Position (NED): X:{position[0]:.2f}, Y:{position[1]:.2f}, Z:{position[2]:.2f}")

        collected_images = 0
        start_time = time.time()
        last_image_time = 0
        consecutive_out_of_bounds = 0
        
        try:
            while collected_images < self.total_images_target:
                current_time = time.time()
                
                # Generate movement with height maintenance
                vx, vy, vz, yaw_rate = self.random_move()
                
                # Execute movement
                self.client.moveByVelocityBodyFrameAsync(
                    vx, vy, vz, duration=1.0,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
                )
                
                # Capture image at intervals
                if current_time - last_image_time >= self.image_interval:
                    depth_image = self.get_depth_image()
                    
                    if depth_image is not None and depth_image.size > 0:
                        # Resize to standard size (72x128 as in paper)
                        if depth_image.shape != (72, 128):
                            depth_image = cv2.resize(depth_image, (128, 72))
                        
                        # Save image in both formats
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        self.save_depth_image(depth_image, f"depth_{timestamp}")
                        
                        collected_images += 1
                        last_image_time = current_time
                        
                        if collected_images % 100 == 0:
                            position = self.get_drone_position()
                            height_above_ground = abs(position[2])
                            print(f"Collected {collected_images} images")
                            print(f"Current position: X:{position[0]:.2f}, Y:{position[1]:.2f}, Height:{height_above_ground:.2f}m")
                
                # Check XY position safety only
                position = self.get_drone_position()
                if not self.is_safe_position(position):
                    consecutive_out_of_bounds += 1
                    if consecutive_out_of_bounds >= 3:
                        print("Drone out of XY bounds, returning to center...")
                        current_height = position[2]
                        # Return to center at current height
                        self.client.moveToPositionAsync(0, 0, current_height, 5.0).join()
                        print("Returned to center")
                        consecutive_out_of_bounds = 0
                else:
                    consecutive_out_of_bounds = 0
                
                # Check flight time
                if time.time() - start_time > self.max_flight_time:
                    print("Flight time exceeded, landing...")
                    break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Data collection interrupted by user")
        except Exception as e:
            print(f"Error during data collection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Land and cleanup
            print("Landing...")
            self.client.landAsync().join()
            time.sleep(2)
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print(f"Data collection completed. Total images: {collected_images}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()