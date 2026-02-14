"""
DC GAN Model Architecture for MNIST Dataset
Implements Generator and Discriminator networks for Deep Convolutional GAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator network for DC GAN
    Takes latent vector and generates 28x28 grayscale images
    """
    def __init__(self, latent_dim=100, img_channels=1):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Project and reshape: latent_dim -> 4x4x512
        self.project = nn.Linear(latent_dim, 512 * 4 * 4, bias=False)
        
        # First transposed conv: 4x4x512 -> 8x8x256
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        
        # Second transposed conv: 8x8x256 -> 16x16x128
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Third transposed conv: 16x16x128 -> 32x32x1 (will crop to 28x28)
        self.deconv3 = nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using normal distribution with mean=0, std=0.02"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        """
        Forward pass of Generator
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        Returns:
            Generated image of shape (batch_size, 1, 28, 28)
        """
        # Project and reshape: (batch_size, latent_dim) -> (batch_size, 512, 4, 4)
        x = self.project(z)
        x = x.view(x.size(0), 512, 4, 4)
        
        # Upsample: 4x4 -> 8x8
        x = F.relu(self.bn1(self.deconv1(x)))
        
        # Upsample: 8x8 -> 16x16
        x = F.relu(self.bn2(self.deconv2(x)))
        
        # Upsample: 16x16 -> 32x32
        x = torch.tanh(self.deconv3(x))
        
        # Crop to 28x28 (center crop)
        x = F.pad(x, (-2, -2, -2, -2))  # Crop 2 pixels from each side
        
        return x


class Discriminator(nn.Module):
    """
    Discriminator network for DC GAN
    Classifies images as real or fake
    """
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        
        # First conv: 28x28 -> 14x14
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False)
        
        # Second conv: 14x14 -> 7x7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Third conv: 7x7 -> 4x4
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Final conv: 4x4 -> 1x1
        self.conv4 = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using normal distribution with mean=0, std=0.02"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        """
        Forward pass of Discriminator
        Args:
            x: Input image of shape (batch_size, 1, 28, 28)
        Returns:
            Logit score of shape (batch_size)
        """
        x = F.leaky_relu(self.conv1(x), 0.2)   # 28x28 -> 14x14
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)  # 14x14 -> 7x7
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)  # 7x7 -> 4x4
        x = torch.sigmoid(self.conv4(x))       # 4x4 -> 1x1
        
        return x.view(-1)


def create_generator(latent_dim=100):
    """Create and return Generator model"""
    return Generator(latent_dim=latent_dim)


def create_discriminator():
    """Create and return Discriminator model"""
    return Discriminator()


if __name__ == "__main__":
    # Test the models
    device = torch.device('cpu')
    
    # Test Generator
    generator = Generator(latent_dim=100).to(device)
    z = torch.randn(8, 100, device=device)
    fake_images = generator(z)
    print(f"Generator output shape: {fake_images.shape}")
    
    # Test Discriminator
    discriminator = Discriminator().to(device)
    real_images = torch.randn(8, 1, 28, 28, device=device)
    output = discriminator(real_images)
    print(f"Discriminator output shape: {output.shape}")
    
    # Count parameters
    total_params_G = sum(p.numel() for p in generator.parameters())
    total_params_D = sum(p.numel() for p in discriminator.parameters())
    print(f"\nGenerator Parameters: {total_params_G:,}")
    print(f"Discriminator Parameters: {total_params_D:,}")
