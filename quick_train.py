"""
Quick Training Script for DC GAN
Runs training with default parameters and generates sample outputs
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
    if result.returncode != 0:
        print("Error installing requirements. Please install manually.")
        return False
    print("Requirements installed successfully!")
    return True

def run_training():
    """Run the training script"""
    print("\n" + "=" * 60)
    print("Starting DC GAN Training")
    print("=" * 60)
    
    result = subprocess.run([sys.executable, "train.py"])
    
    if result.returncode != 0:
        print("Error during training!")
        return False
    
    print("\nTraining completed successfully!")
    return True

def main():
    """Main entry point"""
    print("DC GAN MNIST Training Pipeline")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Run training
    if not run_training():
        return
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start Flask app: python app.py")
    print("2. Open http://localhost:5000 in your browser")
    print("=" * 60)

if __name__ == "__main__":
    main()
