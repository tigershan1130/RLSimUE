# AirSim TD3 Multirotor Reinforcement Learning

![Screenshot](https://private-user-images.githubusercontent.com/39791762/500980081-d10c54f6-a5cc-41cb-85e2-378521e5b7d2.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjA0NTA0MjAsIm5iZiI6MTc2MDQ1MDEyMCwicGF0aCI6Ii8zOTc5MTc2Mi81MDA5ODAwODEtZDEwYzU0ZjYtYTVjYy00MWNiLTg1ZTItMzc4NTIxZTViN2QyLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDE0VDEzNTUyMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwNGZlOThkOTM5ZGQ1ZjdjMzIzNDRhZDRlMDBiZmI1MWRlNzBlNDNjZDRhZDllMmNjNTFlYzRhNGIzODYzZDgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.YNgyuvdDubxAM9XtqmzSjY4Qqzqe5DEmg1Aau6NEkKQ)


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
| **DDPG/DQN** | Tends to overestimate Q-values, unstable |

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
