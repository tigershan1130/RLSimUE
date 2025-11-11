import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from vae_model import VAE, improved_vae_loss, save_vae
import time
from pathlib import Path
from datetime import datetime
import json

class DepthImageDataset(Dataset):
    def __init__(self, data_dir="./vae_data", img_size=(72, 128)):
        self.data_dir = data_dir
        self.img_size = img_size
        images_dir = os.path.join(data_dir, "depth_images")
        self.image_files = glob.glob(os.path.join(images_dir, "*.npy"))
        
        print(f"Looking for images in: {images_dir}")
        print(f"Found {len(self.image_files)} total images")
        
        # Analyze data statistics
        self._analyze_data()
    
    def _analyze_data(self):
        """Analyze the data distribution"""
        if len(self.image_files) == 0:
            return
            
        sample_data = []
        for i in range(min(100, len(self.image_files))):
            try:
                data = np.load(self.image_files[i])
                sample_data.append(data.flatten())
            except:
                continue
        
        if sample_data:
            all_data = np.concatenate(sample_data)
            print("=== DATA STATISTICS ===")
            print(f"Range: [{all_data.min():.4f}, {all_data.max():.4f}]")
            print(f"Mean: {all_data.mean():.4f} ± {all_data.std():.4f}")
            print(f"NaN values: {np.isnan(all_data).sum()}")
            print(f"Zero values: {(all_data == 0).sum() / len(all_data) * 100:.1f}%")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        npy_path = self.image_files[idx]
        
        try:
            depth_normalized = np.load(npy_path)
            
            # Data validation and cleaning
            depth_normalized = np.nan_to_num(depth_normalized, 0.0)
            depth_normalized = np.clip(depth_normalized, 0.0, 1.0)
            
            # Ensure the image is the correct size
            if depth_normalized.shape != self.img_size:
                depth_normalized = cv2.resize(depth_normalized, (self.img_size[1], self.img_size[0]))
            
            # Convert to tensor and add channel dimension
            depth_tensor = torch.FloatTensor(depth_normalized).unsqueeze(0)
            
            return depth_tensor
            
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            # Return a zero tensor as fallback
            return torch.zeros((1, *self.img_size), dtype=torch.float32)

def test_model_shapes():
    """Test the model with actual data to verify shapes"""
    print("Testing model shapes...")
    
    # Create a small test dataset
    dataset = DepthImageDataset("./vae_data", img_size=(72, 128))
    if len(dataset) == 0:
        print("No data found for testing")
        return False
    
    # Get a sample batch
    sample_batch = torch.stack([dataset[i] for i in range(min(4, len(dataset)))])
    print(f"Sample batch shape: {sample_batch.shape}")
    
    # Test the model
    model = VAE(input_shape=(72, 128), latent_dim=32)
    
    try:
        with torch.no_grad():
            recon, mu, logvar = model(sample_batch)
            print(f"Input shape: {sample_batch.shape}")
            print(f"Reconstruction shape: {recon.shape}")
            print(f"Latent mu shape: {mu.shape}")
            print(f"Model shape test passed!")
            return True
    except Exception as e:
        print(f"Model shape test failed: {e}")
        return False

def validate_model(model, val_loader, device, beta=0.1):
    """Validate the model on validation set"""
    model.eval()
    total_val_loss = 0
    total_val_recon = 0
    total_val_kld = 0
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            
            # Calculate loss components
            recon_loss = torch.nn.functional.mse_loss(recon_batch, data, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + beta * kld_loss) / data.size(0)
            
            total_val_loss += loss.item()
            total_val_recon += recon_loss.item() / data.size(0)
            total_val_kld += kld_loss.item() / data.size(0)
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_recon = total_val_recon / len(val_loader)
    avg_val_kld = total_val_kld / len(val_loader)
    
    model.train()
    return avg_val_loss, avg_val_recon, avg_val_kld

def debug_reconstruction(model, dataset, device, epoch, split="train", output_dir:str='.'):
    """Save sample reconstructions for debugging"""
    model.eval()
    with torch.no_grad():
        # Get a sample batch
        sample_indices = [0, min(1, len(dataset)-1), min(2, len(dataset)-1)]
        samples = torch.stack([dataset[i] for i in sample_indices]).to(device)
        recon, mu, logvar = model(samples)
        
        # Convert to numpy for visualization
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        
        for i in range(len(samples)):
            original = samples[i, 0].cpu().numpy()
            reconstructed = recon[i, 0].cpu().numpy()
            
            # Original
            im1 = axes[i, 0].imshow(original, cmap='viridis')
            axes[i, 0].set_title(f'Sample {i+1} - Original')
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
            
            # Reconstructed
            im2 = axes[i, 1].imshow(reconstructed, cmap='viridis')
            axes[i, 1].set_title(f'Sample {i+1} - Reconstructed')
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
        
        plt.tight_layout()
        os.makedirs("vae_data/"+output_dir+"/debug", exist_ok=True)
        plt.savefig(f'vae_data/{output_dir}/debug/reconstruction_{split}_epoch_{epoch}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics for these samples
        mse_per_pixel = torch.nn.functional.mse_loss(recon, samples, reduction='mean').item()
        print(f"Debug {split} samples at epoch {epoch}: MSE/pixel = {mse_per_pixel:.4f}")
    
    model.train()

def plot_training_history(train_losses, val_losses, val_recon_losses, val_kld_losses, output_dir:str='.'):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    # Plot total loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation components
    plt.subplot(1, 3, 2)
    plt.plot(val_recon_losses, label='Val Recon Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(val_kld_losses, label='Val KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation KL Divergence Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('vae_data/'+output_dir+'/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

def train_vae(output_dir:str='.'):
    # Extended training parameters
    batch_size = 32
    learning_rate = 1e-4
    epochs = 200  # Extended training
    latent_dim = 32
    img_size = (72, 128)
    beta = 5.  # KL divergence weight
    debug_epochs = 25

    training_stats = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "latent_dimension": latent_dim,
        "Beta": beta
    }
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test model first
    if not test_model_shapes():
        print("Model shape test failed. Cannot proceed with training.")
        return
    
    # Load and split dataset
    full_dataset = DepthImageDataset("./vae_data", img_size=img_size)
    
    if len(full_dataset) == 0:
        print("No training images found!")
        return


    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = VAE(input_shape=img_size, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"Starting training with {len(train_dataset)} training images")
    print(f"Batch size: {batch_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total epochs: {epochs}")
    
    # Create output directory

    os.makedirs("vae_data", exist_ok=True)
    os.makedirs("vae_data/"+output_dir+"/debug", exist_ok=True)


    # Save off information for the current run
    with open("vae_data/"+output_dir+"/training_stats.json", 'w') as fp:
        json.dump(training_stats, fp)
    
    # Training history
    train_losses = []
    val_losses = []
    val_recon_losses = []
    val_kld_losses = []
    
    # Training loop with validation
    model.train()
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = improved_vae_loss(recon_batch, data, mu, logvar, beta=beta)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 20 == 0:
                current_loss = loss.item()
                print(f'Epoch: {epoch}, Loss: {current_loss:.2f}')
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss, val_recon, val_kld = validate_model(model, val_loader, device, beta=beta)
        val_losses.append(val_loss)
        val_recon_losses.append(val_recon)
        val_kld_losses.append(val_kld)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        if epoch % debug_epochs == 0:
            print(f'====> Epoch: {epoch:3d} | '
                f'Train Loss: {avg_train_loss:7.2f} | '
                f'Val Loss: {val_loss:7.2f} | '
                f'Val Recon: {val_recon:7.2f} | '
                f'Val KLD: {val_kld:6.2f} | '
                f'LR: {current_lr:.2e} | '
                f'Time: {epoch_time:.1f}s')
            
        # Save debug reconstructions every 50 epochs
        if epoch % debug_epochs == 0:
            debug_reconstruction(model, train_dataset, device, epoch, "train", output_dir=output_dir)
            debug_reconstruction(model, val_dataset, device, epoch, "val", output_dir=output_dir)
        
        # Save model every 50 epochs and when validation loss improves
        if epoch % debug_epochs == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.2f}")
                save_vae(model, "vae_data/"+output_dir+"/vae_best.pth")
            
            save_vae(model, f"vae_data/"+output_dir+f"/vae_epoch_{epoch}.pth")
        
        # Plot training history every 50 epochs
        if epoch % debug_epochs == 0:
            plot_training_history(train_losses, val_losses, val_recon_losses, val_kld_losses, output_dir=output_dir)
    
    # Final save and plots
    save_vae(model, "vae_data/"+output_dir+"/vae_final.pth")
    plot_training_history(train_losses, val_losses, val_recon_losses, val_kld_losses, output_dir=output_dir)
    
    total_time = time.time() - start_time
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Best validation loss: {best_val_loss:.2f}")
    print(f"Final training loss: {train_losses[-1]:.2f}")
    print(f"Final validation loss: {val_losses[-1]:.2f}")
    
    # Calculate final MSE per pixel
    final_mse_per_pixel = val_recon_losses[-1] / (img_size[0] * img_size[1])
    print(f"Final validation MSE per pixel: {final_mse_per_pixel:.6f}")

def test_trained_model(output_dir:str='.'):
    """Test the trained model on validation set"""
    print("\n=== TESTING TRAINED MODEL ===")

    beta = 3.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load best model
    model = VAE(input_shape=(72, 128), latent_dim=32).to(device)
    try:
        model.load_state_dict(torch.load("vae_data/"+output_dir+"/vae_best.pth", map_location=device))
        print("Loaded best model for testing")
    except:
        model.load_state_dict(torch.load("vae_data/"+output_dir+"/vae_final.pth", map_location=device))
        print("Loaded final model for testing")
    
    model.eval()
    
    # Load validation dataset
    full_dataset = DepthImageDataset("./vae_data", img_size=(72, 128))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Test on validation set
    test_loss, test_recon, test_kld = validate_model(model, val_loader, device, beta=beta)
    
    print(f"Test Results:")
    print(f"Total Loss: {test_loss:.2f}")
    print(f"Reconstruction Loss: {test_recon:.2f}")
    print(f"KL Divergence: {test_kld:.2f}")
    print(f"MSE per pixel: {test_recon / (72*128):.6f}")
    
    # Generate sample reconstructions
    debug_reconstruction(model, val_dataset, device, "final_test", "test", output_dir=output_dir)

if __name__ == "__main__":

    # Get location for current run directory
    output_path = 'vae_' + str(datetime.now().strftime("%m%d%Y_%H%M"))

    # Train the model
    train_vae(output_dir=output_path)
    
    # Test the trained model
    test_trained_model(output_dir=output_path)