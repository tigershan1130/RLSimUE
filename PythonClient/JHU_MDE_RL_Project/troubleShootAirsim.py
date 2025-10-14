import airsim
import time

# Step 1: Connect
client = airsim.MultirotorClient()
client.confirmConnection() # Look for "Connected!" message

# Step 2: Check API control is enabled
client.enableApiControl(True) 
# Optional: Check control is actually enabled
if client.isApiControlEnabled():
    print("API Control is enabled.")
else:
    print("Failed to get API control.")

# Step 3: Add a delay before arming
time.sleep(5) # Wait for 5 seconds to let the simulation stabilize

# Step 4: Attempt to Arm
try:
    armed_success = client.armDisarm(True)
    print(f"Arming result: {armed_success}")
    if not armed_success:
        print("Arming command returned False. Check Unreal logs for details.")
except Exception as e:
    print(f"An exception occurred during arming: {e}")

# ... rest of your code ...