"""
DC GAN Training Script for MNIST Dataset
Implements training loop with logging, visualization, and model checkpointing
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np

from models.dcgan import Generator, Discriminator


# Configuration
class Config:
    # Model parameters
    latent_dim = 100
    img_channels = 1
    img_size = 28
    
    # Training parameters
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.0002
    beta1 = 0.5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    checkpoint_dir = 'checkpoints'
    samples_dir = 'samples'
    plots_dir = 'plots'
    
    # Logging
    log_interval = 100
    sample_interval = 5


def setup_directories():
    """Create necessary directories for saving outputs"""
    directories = [Config.checkpoint_dir, Config.samples_dir, Config.plots_dir]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)


def get_data_loaders():
    """Create and return MNIST data loaders"""
    # Transform for MNIST images
    transform = transforms.Compose([
        transforms.Resize(Config.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if Config.device.type == 'cuda' else False
    )
    
    return train_loader


def train_discriminator(real_images, device, discriminator, generator, optimizer_D, criterion):
    """
    Train discriminator on real and fake images
    Returns: discriminator loss
    """
    batch_size = real_images.size(0)
    real_labels = torch.ones(batch_size, device=device)
    fake_labels = torch.zeros(batch_size, device=device)
    
    # Train on real images
    optimizer_D.zero_grad()
    output_real = discriminator(real_images)
    loss_real = criterion(output_real, real_labels)
    
    # Train on fake images
    z = torch.randn(batch_size, Config.latent_dim, device=device)  # 2D latent vector
    fake_images = generator(z)
    output_fake = discriminator(fake_images.detach())
    loss_fake = criterion(output_fake, fake_labels)
    
    # Total discriminator loss
    loss_D = (loss_real + loss_fake) / 2
    loss_D.backward()
    optimizer_D.step()
    
    return loss_D.item()


def train_generator(discriminator, generator, optimizer_G, criterion):
    """
    Train generator to fool discriminator
    Returns: generator loss
    """
    batch_size = Config.batch_size
    real_labels = torch.ones(batch_size, device=Config.device)
    
    optimizer_G.zero_grad()
    
    # Generate fake images
    z = torch.randn(batch_size, Config.latent_dim, device=Config.device)  # 2D latent vector
    fake_images = generator(z)
    
    # Try to fool discriminator
    output = discriminator(fake_images)
    loss_G = criterion(output, real_labels)
    
    loss_G.backward()
    optimizer_G.step()
    
    return loss_G.item()


def plot_losses(d_losses, g_losses, epoch, save_path):
    """Plot and save training losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'DC GAN Training Losses - Epoch {epoch}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_generated_images(generator, epoch, device, save_path, num_images=64):
    """Generate and save sample images from generator"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, Config.latent_dim, 1, 1, device=device)
        fake_images = generator(z)
        
        # Denormalize from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        
        # Create grid
        grid = make_grid(fake_images, nrow=8, padding=2, pad_value=1.0)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f'Generated MNIST Images - Epoch {epoch}')
        plt.savefig(save_path)
        plt.close()
    
    generator.train()


def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, loss_G, loss_D, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss_G': loss_G,
        'loss_D': loss_D
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'dcgan_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def train():
    """Main training function"""
    print("=" * 60)
    print("DC GAN Training for MNIST Dataset")
    print("=" * 60)
    print(f"Device: {Config.device}")
    print(f"Batch Size: {Config.batch_size}")
    print(f"Learning Rate: {Config.learning_rate}")
    print(f"Epochs: {Config.num_epochs}")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Get data loaders
    print("\nLoading MNIST dataset...")
    train_loader = get_data_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Initialize models
    print("\nInitializing DC GAN models...")
    generator = Generator(latent_dim=Config.latent_dim).to(Config.device)
    discriminator = Discriminator().to(Config.device)
    
    # Print model architectures
    print(f"\nGenerator Architecture:")
    print(generator)
    print(f"\nDiscriminator Architecture:")
    print(discriminator)
    
    # Count parameters
    total_params_G = sum(p.numel() for p in generator.parameters())
    total_params_D = sum(p.numel() for p in discriminator.parameters())
    print(f"\nGenerator Parameters: {total_params_G:,}")
    print(f"Discriminator Parameters: {total_params_D:,}")
    
    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=Config.learning_rate, betas=(Config.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=Config.learning_rate, betas=(Config.beta1, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training history
    history = {
        'd_losses': [],
        'g_losses': [],
        'epochs': []
    }
    
    # Fixed latent vector for visualization
    fixed_z = torch.randn(64, Config.latent_dim, device=Config.device)  # 2D latent vector
    
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    for epoch in range(1, Config.num_epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0
        
        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(Config.device)
            
            # Train discriminator
            loss_D = train_discriminator(real_images, Config.device, discriminator, generator, optimizer_D, criterion)
            
            # Train generator
            loss_G = train_generator(discriminator, generator, optimizer_G, criterion)
            
            # Accumulate losses
            epoch_d_loss += loss_D
            epoch_g_loss += loss_G
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % Config.log_interval == 0:
                print(f"Epoch [{epoch}/{Config.num_epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | "
                      f"D_Loss: {loss_D:.4f} | G_Loss: {loss_G:.4f}")
        
        # Calculate average losses for epoch
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        
        history['d_losses'].append(avg_d_loss)
        history['g_losses'].append(avg_g_loss)
        history['epochs'].append(epoch)
        
        print(f"\nEpoch [{epoch}/{Config.num_epochs}] Summary:")
        print(f"  Average D_Loss: {avg_d_loss:.4f}")
        print(f"  Average G_Loss: {avg_g_loss:.4f}")
        
        # Save generated images at specified intervals
        if epoch % Config.sample_interval == 0 or epoch == 1:
            # Save sample images
            sample_path = os.path.join(Config.samples_dir, f'samples_epoch_{epoch}.png')
            with torch.no_grad():
                generator.eval()
                fake_images = generator(fixed_z)
                fake_images = (fake_images + 1) / 2  # Denormalize
                grid = make_grid(fake_images, nrow=8, padding=2, pad_value=1.0)
                save_image(grid, sample_path)
                generator.train()
            print(f"  Sample images saved: {sample_path}")
        
        # Save checkpoint at specified intervals
        if epoch % 10 == 0:
            save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, 
                          epoch, avg_g_loss, avg_d_loss, Config.checkpoint_dir)
        
        # Save loss plot
        if epoch % 10 == 0:
            plot_path = os.path.join(Config.plots_dir, f'losses_epoch_{epoch}.png')
            plot_losses(history['d_losses'], history['g_losses'], epoch, plot_path)
            print(f"  Loss plot saved: {plot_path}")
        
        print("-" * 60)
    
    # Save final checkpoint
    save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, 
                   Config.num_epochs, avg_g_loss, avg_d_loss, Config.checkpoint_dir)
    
    # Save final loss plot
    final_plot_path = os.path.join(Config.plots_dir, 'final_losses.png')
    plot_losses(history['d_losses'], history['g_losses'], Config.num_epochs, final_plot_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Generator Loss: {avg_g_loss:.4f}")
    print(f"Final Discriminator Loss: {avg_d_loss:.4f}")
    print(f"Checkpoints saved in: {Config.checkpoint_dir}")
    print(f"Sample images saved in: {Config.samples_dir}")
    print(f"Loss plots saved in: {Config.plots_dir}")
    
    return generator, discriminator


if __name__ == "__main__":
    train()
