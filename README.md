# AirSim Monocular Depth Estimation With Multirotor Reinforcement Learning
![Screenshot](https://private-user-images.githubusercontent.com/39791762/500980081-d10c54f6-a5cc-41cb-85e2-378521e5b7d2.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjUyNDQzMTEsIm5iZiI6MTc2NTI0NDAxMSwicGF0aCI6Ii8zOTc5MTc2Mi81MDA5ODAwODEtZDEwYzU0ZjYtYTVjYy00MWNiLTg1ZTItMzc4NTIxZTViN2QyLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjA5VDAxMzMzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPThjYmYwNDgyNGU3NjlkZjBlNzAzZTIxMjEzYzFmZjAxYzEyZDA2NGRjNDc2ZWU2N2Y0ZjY1Y2Y1ZjdkOWU4MDcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Fzyf0lBPpvlHJW2xYAT1csV67PjB1IhkpK0nNi6XKEo)

## Introduction
A key challenge in UAV autonomy is robust obstacle avoidance in complex environments. Recent research addresses this by leveraging Monocular Depth Estimation (MDE), which enables drones to perceive 3D structure from a single camera, facilitating effective navigation and collision avoidance. A promising approach for UAV obstacle avoidance is to combine the perceptual capabilities of Monocular Depth Estimation (MDE) with the decision-making power of Reinforcement Learning (RL). In this framework, MDE processes a single camera image to generate a real-time depth map, providing a compact representation of the 3D environment. This depth information then serves as the state input for an RL agent, which is trained to learn an optimal navigation policy. Through repeated simulation, the agent learns to map these depth states to the correct flight actions—such as turning, ascending, or slowing down—to efficiently navigate through cluttered spaces while proactively avoiding collisions. This synergy allows the UAV to not just perceive its surroundings but also to learn complex, robust avoidance behaviours directly from experience. In this project, we take a closer look at how to use Microsoft Airsim with RL Agent to allow UAV to perceive its surroundings and learning robust avoidance in complex simulation enviornment such as the Unreal Engine.

## Project Overview
This project implements a **Deep Reinforcement Learning (DRL)** system for autonomous drone navigation in Microsoft AirSim using Stable-Baselines3. The drone learns to navigate to randomly generated target points while avoiding collisions using the **Soft Actor Crtic (SAC)** algorithm with VAE for multi-modal observations.


## 🚀 AirSim Multirotor Quick Setup Instructions

### 1. Download and Setup Unreal Engine Executable
- Download the latest release of our current project(executable on the right side of release page)
- Ensure the executable is properly installed and configured

### 2. Configure AirSim Settings
Create or modify the AirSim configuration file at `Users\Documents\AirSim\settings.json`:

```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor"
}
```
### 3. Python Environment Setup

#### Create Virtual Environment and install required packages.
```bash
# Create virtual environment
python -m venv airsim_env

# Activate virtual environment
# On Windows:
airsim_env\Scripts\activate
# On macOS/Linux:
source airsim_env/bin/activate

# Activate virtual environment
# On Windows:
airsim_env\Scripts\activate
# On macOS/Linux:
source airsim_env/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 4. Launch Project

#### Start Our excutable.
- Launch our executable MultiRotor demo with the AirSim environment

#### Run Python Training
Navigate to the project directory and start the training script:

```bash
cd .\PythonClient\JHU_MDE_RL_Project\
python train_agent.py
#or test airsim with python troubleShootAirsim.py
```

### 5. Trouble Shoot: 
We have running into problem of getting python enviornment running with airsim:
https://github.com/microsoft/AirSim/issues/4920
You can look modify the setup.py following this link and rerun the setup.py and everything should be up and running.

## :biohazard: Methodology

### Reinforcement Learning Framework
Base on [1] paper, The full pipeline of the RL setup. The blue box is the simulation environment. The yellow box represents the MSA in which the policy neural network (πθ) is who chooses the agent’s actions. The API provided by AirSim ables the communication between both components. We refer to the architecture of airsim-RL framework, we formulate drone navigation as a **Partially Observable Markov Decision Process (POMDP)** where the agent learns optimal policies through trial and error:

![Screenshot1]([https://private-user-images.githubusercontent.com/39791762/501044347-f3ffabd0-ff63-41cd-94f1-ce5f14a4612b.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjA0NTg2ODgsIm5iZiI6MTc2MDQ1ODM4OCwicGF0aCI6Ii8zOTc5MTc2Mi81MDEwNDQzNDctZjNmZmFiZDAtZmY2My00MWNkLTk0ZjEtY2U1ZjE0YTQ2MTJiLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDE0VDE2MTMwOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTljYjBkNDJkZjBkY2EyOWFjNWJiNTg3MDU0NDMzOTUwMzcwOThjOTc0MzAzMzAxNmY2ZTA4NTMwYTc1OWVmNWEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.2lRq6mDOUoobsx-gzLGxenZWfcbmmWK8l8n6YAUAI5o](https://private-user-images.githubusercontent.com/39791762/501044347-f3ffabd0-ff63-41cd-94f1-ce5f14a4612b.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjUyNDM5MjgsIm5iZiI6MTc2NTI0MzYyOCwicGF0aCI6Ii8zOTc5MTc2Mi81MDEwNDQzNDctZjNmZmFiZDAtZmY2My00MWNkLTk0ZjEtY2U1ZjE0YTQ2MTJiLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjA5VDAxMjcwOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTQ1MGQwMDA1NmYyNjg5NTExMjI5MTJiOTc2MTVmNzkzMDdmMmYzYTA1NTJhNjkxZDRmMDk2OTliZGNjMzcyYTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.tNVwOpjc43rfrpW6BiAw5NUMKhkFr8YR1meK99blxlg))


* **Agent**: Multirotor drone
* **Environment**: AirSim simulation with Unreal Engine  
* **State**: Multi-modal observations (relative distance+dir, velocity, depth images)
* **Actions**: Velocity commands in 3D space + yaw rate
* **Rewards**: Sparse + shaped rewards for efficient navigation and obstacle avoidance

### Training/Testing Pipeline 
![Screenshot](https://private-user-images.githubusercontent.com/39791762/501044347-f3ffabd0-ff63-41cd-94f1-ce5f14a4612b.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjUyNDM5MjgsIm5iZiI6MTc2NTI0MzYyOCwicGF0aCI6Ii8zOTc5MTc2Mi81MDEwNDQzNDctZjNmZmFiZDAtZmY2My00MWNkLTk0ZjEtY2U1ZjE0YTQ2MTJiLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjA5VDAxMjcwOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTQ1MGQwMDA1NmYyNjg5NTExMjI5MTJiOTc2MTVmNzkzMDdmMmYzYTA1NTJhNjkxZDRmMDk2OTliZGNjMzcyYTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.tNVwOpjc43rfrpW6BiAw5NUMKhkFr8YR1meK99blxlg)

This is referenced from paper[3]. SAC + VAE training pipeline.

## Why SAC Algorithm?

### Advantages for Drone Navigation
* PPO (Proximal Policy Optimization):
PPO is an on-policy algorithm, meaning it discards data after each update. This leads to lower sample efficiency compared to off-policy methods like SAC.
In robotics tasks, where data collection can be time-consuming and expensive, sample efficiency is crucial.
PPO typically requires more interactions with the environment to converge, which can be slow in a high-fidelity simulation like AirSim.

* DQN (Deep Q-Network):
DQN is designed for discrete action spaces. Our drone control task requires continuous control (continuous action space), so DQN is not directly applicable without discretization, which can lead to the curse of dimensionality and loss of fine control.

* DDPG (Deep Deterministic Policy Gradient):
DDPG is an actor-critic method for continuous control. However, it is known to be prone to overestimating Q-values, which can lead to unstable training and suboptimal policies.
The overestimation bias in DDPG arises because the same network is used to select and evaluate the next action. This can be particularly problematic in environments with high-dimensional observation spaces and complex dynamics, such as drone navigation.

* TD3 (Twin Delayed DDPG):
TD3 addresses the overestimation bias of DDPG by introducing two critic networks (twin) and taking the minimum of their Q-values for the target. This reduces overestimation and leads to more stable training.
Additionally, TD3 uses delayed policy updates, which means the policy is updated less frequently than the critics. It can also use replay buffer for simlar temporal storage from training. The momentum and dynamics handling would be better.

* SAC (Soft Actor-Critic):
SAC combines the best of both worlds: the sample efficiency of off-policy methods with the exploration benefits of stochastic policies. The entropy regularization encourages exploration without the need for explicit noise injection.

### Comparison between TD3 and SAC
We are comparing TD3 and SAC for our drone navigation task. We have a multi-modal observation space (depth image and vector states) and a continuous action space.
We are thinking about TD3+CNN or SAC+VAE.

1. Algorithmic Differences:
  - TD3 (Twin Delayed Deep Deterministic Policy Gradient): Deterministic policy (outputs exact action).
  -  SAC (Soft Actor-Critic): Stochastic policy (outputs a distribution over actions) with entropy regularization. Maximizes both expected return and entropy, which encourages exploration.

2. Exploration:
  - TD3: Explores by adding noise to the actions (e.g., Gaussian noise). The policy itself is deterministic.
  - SAC: The stochastic policy naturally explores, and the entropy term encourages trying new actions.

3. Handling Uncertainty: Our environment is partially observable.
  - SAC's stochastic policy may be better at handling uncertainty because it can learn to take a variety of actions in the same state (non-deterministic).
  - TD3's deterministic policy might be more prone to getting stuck in a single behavior, which might not be robust to uncertainties. (Get Stuck circling around)

#### Conclusion:
  SAC + VAE(Decided): Allows for a two-stage training process. First, train the VAE on depth images (offline). Then, train SAC using the compressed latent vectors. This separation simplifies debugging and can lead to faster initial results, The paper from [2][3] gives pretty good results using SAC+VAE. We were thinking about using (TD3 + CNN), but it requires end-to-end training, where the CNN and policy must learn simultaneously, we have doubts it will be trained faster as input vector would be messy going into this CNN+TD3 at once, even though TD3 itself have fast intial learning process we have doubts it can be trained faster and have better results at the end.

### Results:

![Screenshot](https://private-user-images.githubusercontent.com/39791762/524033886-a6efae6e-75f2-4470-aafa-e9ae53ba3609.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjUyNDQxNDksIm5iZiI6MTc2NTI0Mzg0OSwicGF0aCI6Ii8zOTc5MTc2Mi81MjQwMzM4ODYtYTZlZmFlNmUtNzVmMi00NDcwLWFhZmEtZTlhZTUzYmEzNjA5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjA5VDAxMzA0OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJhYzRkZTUxMDZlMDNiZmNhZjE2YjYzMmMzZjE3NTJhZDhiOTRhY2VkNmVhN2Q3MjJlZjJkNTZjNDU1Mzk0ZGYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.-QX1kDQmC6KgN4Tp31o1NtMUmOcYt6aEpjcSQPQOmME)

The agent’s improvement during training is shown in the Table. In the sparse environment, clear learning is observed after approximately 750 episodes, with substantial reward improvement from initial to final episodes. The dense environment presented a more challenging learning problem, with negative rewards persisting throughout the training. When considering these results, it is important to note that these two trainings were conducted on different hardware; for the dense environment we were only able to train 55,773 steps in 8.31 hours while for the sparse environment we were able to train 99,929 steps in 7.38 hours, close to double the speed. With more training, we would expect to see convergence in our reinforcement learning model training in the dense environment. Consideration should also be made for the impact of progress reward toward the training, as training on hardware with a higher rate of steps-per-second could mean more positive reward being accumulated.


 ## 🚀 Future Work / TODO

### 1. **Scalable & Parallel Training Infrastructure**
- Migrate from single-machine RL training to a distributed framework using **Soft Actor-Critic (SAC)**.
- Leverage Linux-based HPC clusters for:
  - Parallel experience collection
  - Faster convergence and improved sample efficiency
  - Support for larger-scale policies and curriculum learning

### 2. **Enhanced Environment Complexity & Reward Shaping**
- Introduce **dynamic obstacles**, variable terrain, and adverse weather (wind, fog).
- Refine reward function using **3D pathfinding priors** (e.g., 3D Dijkstra’s / A*) for reward shaping:
  - Reduce target overshoot
  - Improve success rate & path smoothness
  - Enable progressive guidance in sparse-reward settings

### 3. **Zero-Shot Generalization Evaluation**
- Test trained policies in **unseen, out-of-distribution environments** (no fine-tuning).
- Quantify robustness and transferability—critical for real-world deployment readiness.

### 4. **Realistic Control Stack for Sim-to-Real Transfer**
> ☑️ *Goal: Close the sim-to-real gap via a production-grade control architecture.*

#### Proposed Architecture:
- **Hierarchical Control Pipeline**:
  - **High-level (RL Agent)**: Outputs waypoints or trajectory segments (not raw velocities)
  - **Mid-level (Trajectory Planner)**: Generates dynamically feasible, smooth trajectories
  - **Low-level (Flight Controller)**: PID or MPC for actuator-level commands (e.g., motor PWM)

#### Key Upgrades:
- ✅ **ROS integration** as middleware (standardizes sim ↔ real interface)
- ✅ **PX4 SITL (Software-in-the-Loop)** for realistic flight dynamics & sensor emulation
- ✅ **AirSim + ROS Bridge**: Use AirSim as physics engine, but bypass built-in controllers  
  → Expose raw IMU/GPS/vision data; accept low-level actuator commands
- ✅ Enforce physical constraints: max tilt, acceleration, actuator saturation, and latency

#### Expected Benefits:
- Realistic actuator dynamics & sensor fusion
- Consistent control stack across simulation and hardware
- Built-in safety via low-level constraint enforcement
- Easier integration with commercial autopilots (PX4/ArduPilot)
- 
## Reference:
1. Alberto Musa, etc.,"A Method for Accelerated Simulations of Reinforcement Learning Tasks of UAVs in AirSim", 2022/05/17, https://personales.upv.es/thinkmind/dl/conferences/simul/simul_2022/simul_2022_1_90_50041.pdf, DOI  - 10.1145/3528416.3530865
2. Jeremy Roghair, Kyungtae Ko, Amir Ehsan Niaraki Asli, Ali Jannesari. "A Vision Based Deep Reinforcement Learning Algorithm for UAV Obstacle Avoidance." arXiv preprint arXiv:2103.06403 (2021), https://arxiv.org/pdf/2103.06403
3. Zhihan Xue, Tad Gonsalves., "Vision Based Drone Obstacle Avoidance by Deep Reinforcement Learning", DOI:10.3390/ai2030023, August 2021, https://www.researchgate.net/publication/354017223_Vision_Based_Drone_Obstacle_Avoidance_by_Deep_Reinforcement_Learning
4. Davide Falanga, Kevin Kleber, Davide Scaramuzza, “Dynamic Obstacle Avoidance for Quadrotors with Event Cameras”, Science Robotics, March 18, 2020. DOI: 10.1126/scirobotics.aaz9712
