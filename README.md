# AirSim TD3 Multirotor Reinforcement Learning
![Screenshot](https://private-user-images.githubusercontent.com/39791762/500980081-d10c54f6-a5cc-41cb-85e2-378521e5b7d2.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjA0NTA0MjAsIm5iZiI6MTc2MDQ1MDEyMCwicGF0aCI6Ii8zOTc5MTc2Mi81MDA5ODAwODEtZDEwYzU0ZjYtYTVjYy00MWNiLTg1ZTItMzc4NTIxZTViN2QyLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDE0VDEzNTUyMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwNGZlOThkOTM5ZGQ1ZjdjMzIzNDRhZDRlMDBiZmI1MWRlNzBlNDNjZDRhZDllMmNjNTFlYzRhNGIzODYzZDgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.YNgyuvdDubxAM9XtqmzSjY4Qqzqe5DEmg1Aau6NEkKQ)


## Project Overview
This project implements a **Deep Reinforcement Learning (DRL)** system for autonomous drone navigation in Microsoft AirSim using Stable-Baselines3. The drone learns to navigate to randomly generated target points while avoiding collisions using the **Twin Delayed DDPG (TD3)** algorithm with multi-modal observations.


## 🚀 AirSim TD3 Multirotor Quick Setup Instructions

### 1. Download and Setup Unreal Engine Executable
- Download the latest release of Unreal Engine executable
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

#### Start Unreal Engine
- Launch your Unreal Engine executable with the AirSim environment

#### Run Python Training
Navigate to the project directory and start the training script:

```bash
cd .\PythonClient\JHU_MDE_RL_Project\
python train_agent.py
#or test airsim with python troubleShootAirsim.py
```


## Methodology

### Reinforcement Learning Framework

![Screenshot2](https://private-user-images.githubusercontent.com/39791762/501265618-1fd3b12c-01b5-43de-af49-ecd66e5961c2.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjA1MDA2MjgsIm5iZiI6MTc2MDUwMDMyOCwicGF0aCI6Ii8zOTc5MTc2Mi81MDEyNjU2MTgtMWZkM2IxMmMtMDFiNS00M2RlLWFmNDktZWNkNjZlNTk2MWMyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMTUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDE1VDAzNTIwOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUzMmYyNDI4MjhlNzMxNTI0YmQ2MjczMTYxODU3ODMyNTliZGY1NDUyMDkxYTMwOGNhOTlhMDZhNTA2YTk3ZjgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.AeJhAeGaer8lPmd1uNxfRK6JWUrcMIWQuVrYcOTVJeQ)

Base on [1] paper, The full pipeline of the RL setup. The blue box is the simulation environment. The yellow box represents the MSA in which the policy neural network (πθ) is who chooses the agent’s actions. The API provided by AirSim ables the communication between both components. We refer to the architecture of airsim-RL framework, we formulate drone navigation as a **Partially Observable Markov Decision Process (POMDP)** where the agent learns optimal policies through trial and error:

![Screenshot1](https://private-user-images.githubusercontent.com/39791762/501044347-f3ffabd0-ff63-41cd-94f1-ce5f14a4612b.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjA0NTg2ODgsIm5iZiI6MTc2MDQ1ODM4OCwicGF0aCI6Ii8zOTc5MTc2Mi81MDEwNDQzNDctZjNmZmFiZDAtZmY2My00MWNkLTk0ZjEtY2U1ZjE0YTQ2MTJiLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDE0VDE2MTMwOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTljYjBkNDJkZjBkY2EyOWFjNWJiNTg3MDU0NDMzOTUwMzcwOThjOTc0MzAzMzAxNmY2ZTA4NTMwYTc1OWVmNWEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.2lRq6mDOUoobsx-gzLGxenZWfcbmmWK8l8n6YAUAI5o)


* **Agent**: Multirotor drone
* **Environment**: AirSim simulation with Unreal Engine  
* **State**: Multi-modal observations (position, velocity, depth images)
* **Actions**: Velocity commands in 3D space + yaw rate
* **Rewards**: Sparse + shaped rewards for efficient navigation and obstacle avoidance

### Training Pipeline 



## Why TD3 Algorithm?

### Advantages for Drone Navigation
* PPO (Proximal Policy Optimization):
PPO is an on-policy algorithm, meaning it discards data after each update. This leads to lower sample efficiency compared to off-policy methods like TD3.
In robotics tasks, where data collection can be time-consuming and expensive, sample efficiency is crucial.
PPO typically requires more interactions with the environment to converge, which can be slow in a high-fidelity simulation like AirSim.

* DQN (Deep Q-Network):
DQN is designed for discrete action spaces. Our drone control task requires continuous control (continuous action space), so DQN is not directly applicable without discretization, which can lead to the curse of dimensionality and loss of fine control.

* DDPG (Deep Deterministic Policy Gradient):
DDPG is an actor-critic method for continuous control. However, it is known to be prone to overestimating Q-values, which can lead to unstable training and suboptimal policies.
The overestimation bias in DDPG arises because the same network is used to select and evaluate the next action. This can be particularly problematic in environments with high-dimensional observation spaces and complex dynamics, such as drone navigation.

* TD3 (Twin Delayed DDPG)(We decided to use):
TD3 addresses the overestimation bias of DDPG by introducing two critic networks (twin) and taking the minimum of their Q-values for the target. This reduces overestimation and leads to more stable training.
Additionally, TD3 uses delayed policy updates, which means the policy is updated less frequently than the critics. It can also use replay buffer for simlar temporal storage from training. The momentum and dynamics handling would be better.

### Expected Performance
* Navigation Accuracy: ??
* Collision Avoidance: ??
* Training Stability: Consistent learning curve?


## Reference:
1. A Method for Accelerated Simulations of Reinforcement Learning Tasks of UAVs in AirSim, Alberto Musa, etc., 2022/05/17, https://personales.upv.es/thinkmind/dl/conferences/simul/simul_2022/simul_2022_1_90_50041.pdf, DOI  - 10.1145/3528416.3530865
