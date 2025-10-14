import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class EnhancedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # CNN for depth images (same as before)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output size
        with torch.no_grad():
            sample_depth = torch.randn(1, 1, 84, 84)
            n_flatten = self.cnn(sample_depth).shape[1]
        
        # Vector features now have 13 elements (position(3) + velocity(6) + target(3) + distance(1))
        vector_dim = 13
        
        # Combined feature processing
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + vector_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations):
        if isinstance(observations, dict):
            vector_obs = observations['vector']
            depth_obs = observations['depth']
        else:
            vector_obs = observations
            depth_obs = torch.zeros(observations.shape[0], 1, 84, 84)
        
        # Process depth image
        depth_obs = depth_obs.permute(0, 3, 1, 2)
        depth_features = self.cnn(depth_obs)
        
        # Concatenate all features
        combined_features = torch.cat([depth_features, vector_obs], dim=1)
        
        return self.linear(combined_features)