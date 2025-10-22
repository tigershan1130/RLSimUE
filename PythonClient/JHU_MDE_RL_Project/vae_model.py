import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_shape=(72, 128), latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        
        # Encoder - Fixed to produce consistent shapes
        self.encoder = nn.Sequential(
            # Input: 1x72x128
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 32x36x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 64x18x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128x9x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 256x4x8 (NOT 5x8)
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the flattened size properly
        with torch.no_grad():
            sample = torch.zeros(1, 1, *input_shape)
            encoder_output = self.encoder(sample)
            self.flattened_size = encoder_output.shape[1]
            print(f"Encoder output size: {self.flattened_size}")
        
        # The correct decoder input shape should be (256, 4, 8)
        self.decoder_input_shape = (256, 4, 8)
        self.decoder_input_size = 256 * 4 * 8
        
        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder - Fixed to match encoder
        self.decoder_input = nn.Linear(latent_dim, self.decoder_input_size)
        
        self.decoder = nn.Sequential(
            # Input: 256x4x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 128x8x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x16x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32x32x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 1x64x128
            nn.Sigmoid()
        )
        
        # Use adaptive pooling to get exact input size
        self.final_resize = nn.AdaptiveAvgPool2d(input_shape)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        # Reshape to match the encoder's final output shape
        h = h.view(-1, *self.decoder_input_shape)
        # Apply decoder
        decoded = self.decoder(h)
        # Use adaptive pooling to get exact input size
        return self.final_resize(decoded)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def improved_vae_loss(recon_x, x, mu, logvar, beta=0.1):
    """Improved VAE loss with better balancing"""
    # Use MSE instead of BCE for depth images
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Combine losses - normalize by batch size
    batch_size = x.size(0)
    total_loss = (recon_loss + beta * kld_loss) / batch_size
    
    return total_loss

def save_vae(model, path="vae_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"VAE saved to {path}")

def load_vae(input_shape=(72, 128), latent_dim=32, path="vae_model.pth"):
    model = VAE(input_shape, latent_dim)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    print(f"VAE loaded from {path}")
    return model