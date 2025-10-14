# AirSim TD3 Multirotor Reinforcement Learning

![Screenshot](https://github.com/tigershan1130/RLSimUE/blob/main/Images/Environment.png)


## Project Overview
This project implements a **Deep Reinforcement Learning (DRL)** system for autonomous drone navigation in Microsoft AirSim using Stable-Baselines3. The drone learns to navigate to randomly generated target points while avoiding collisions using the **Twin Delayed DDPG (TD3)** algorithm with multi-modal observations.

## Methodology

### Reinforcement Learning Framework
We formulate drone navigation as a **Partially Observable Markov Decision Process (POMDP)** where the agent learns optimal policies through trial and error:

* **Agent**: Multirotor drone
* **Environment**: AirSim simulation with Unreal Engine  
* **State**: Multi-modal observations (position, velocity, depth images)
* **Actions**: Velocity commands in 3D space + yaw rate
* **Rewards**: Sparse + shaped rewards for efficient navigation

### Training Pipeline 



## Why TD3 Algorithm?

### Advantages for Drone Navigation
* **Continuous Action Space**: TD3 excels in continuous control tasks like velocity control
* **Sample Efficiency**: Off-policy nature allows replay buffer utilization  
* **Stability**: Twin critics and delayed updates prevent overestimation
* **Deterministic Policies**: Provides consistent navigation behavior

### Comparison with Other Algorithms
| Algorithm | Why Not Chosen |
|-----------|----------------|
| **PPO** | Less sample efficient for robotics, slower convergence |
| **DDPG** | Tends to overestimate Q-values, unstable |
| **SAC** | More complex, harder to tune for precise navigation |
| **A2C** | Less sample efficient for complex continuous control |


## Why EnhancedFeatureExtractor?

### The Multi-Modal Challenge
Drone navigation requires processing **heterogeneous input types**:
* **Low-dimensional vectors**: Position, velocity, target information
* **High-dimensional images**: Depth maps for obstacle perception

### Architecture Benefits


### Expected Performance
* Navigation Accuracy: >90% target reach rate
* Collision Avoidance: <5% collision rate
* Path Efficiency: Within 120% of optimal path length
* Training Stability: Consistent learning curves