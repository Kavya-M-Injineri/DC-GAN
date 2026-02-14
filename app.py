"""
Flask Web Application for DC GAN MNIST Generator
Provides API endpoints for generating and displaying synthetic MNIST-like images
"""

import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, jsonify, request
from PIL import Image
from models.dcgan import Generator

# Initialize Flask app
app = Flask(__name__)

# Configuration
LATENT_DIM = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'checkpoints/latest.pth'


def load_generator(model_path):
    """Load trained generator model"""
    generator = Generator(latent_dim=LATENT_DIM)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.to(DEVICE)
        generator.eval()
        print(f"Model loaded from {model_path}")
    else:
        # Try to find latest checkpoint
        checkpoints_dir = 'checkpoints'
        if os.path.exists(checkpoints_dir):
            checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints)[-1]
                model_path = os.path.join(checkpoints_dir, latest_checkpoint)
                checkpoint = torch.load(model_path, map_location=DEVICE)
                generator.load_state_dict(checkpoint['generator_state_dict'])
                generator.to(DEVICE)
                generator.eval()
                print(f"Model loaded from {model_path}")
            else:
                print("No checkpoint found. Please train the model first.")
                return None
        else:
            print("Checkpoints directory not found. Please train the model first.")
            return None
    
    return generator


def tensor_to_base64(tensor):
    """Convert PyTorch tensor to base64 encoded image"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL Image
    array = tensor.permute(1, 2, 0).cpu().numpy()
    array = (array * 255).astype('uint8')
    
    if array.shape[-1] == 1:
        array = array[:, :, 0]
    
    image = Image.fromarray(array, mode='L' if len(array.shape) == 2 else 'RGB')
    
    # Save to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return img_str


# Load generator on startup
generator = load_generator(MODEL_PATH)


@app.route('/')
def home():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate synthetic MNIST images"""
    if generator is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'status': 'error'
        }), 500
    
    try:
        # Get number of images from request
        num_images = request.json.get('num_images', 1)
        num_images = min(max(num_images, 1), 16)  # Limit between 1 and 16
        
        # Generate random latent vectors
        z = torch.randn(num_images, LATENT_DIM, device=DEVICE)  # 2D latent vector
        
        # Generate images
        with torch.no_grad():
            fake_images = generator(z)
        
        # Convert to base64
        images_base64 = []
        for i in range(num_images):
            img_str = tensor_to_base64(fake_images[i])
            images_base64.append(f"data:image/png;base64,{img_str}")
        
        return jsonify({
            'status': 'success',
            'images': images_base64,
            'num_generated': num_images
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/generate-single', methods=['POST'])
def generate_single():
    """Generate a single synthetic MNIST image"""
    if generator is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'status': 'error'
        }), 500
    
    try:
        # Generate random latent vector
        z = torch.randn(1, LATENT_DIM, device=DEVICE)  # 2D latent vector
        
        # Generate image
        with torch.no_grad():
            fake_image = generator(z)
        
        # Convert to base64
        img_str = tensor_to_base64(fake_image[0])
        
        return jsonify({
            'status': 'success',
            'image': f"data:image/png;base64,{img_str}"
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/status')
def status():
    """Check model status"""
    model_loaded = generator is not None
    
    return jsonify({
        'status': 'ready' if model_loaded else 'not_loaded',
        'model_path': MODEL_PATH if model_loaded else None,
        'device': str(DEVICE),
        'latent_dim': LATENT_DIM
    })


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': generator is not None
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("DC GAN MNIST Generator - Flask App")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {'Loaded' if generator else 'Not Loaded'}")
    print("\nEndpoints:")
    print("  GET  /           - Home page")
    print("  GET  /api/status - Check model status")
    print("  GET  /api/health - Health check")
    print("  POST /api/generate        - Generate multiple images")
    print("  POST /api/generate-single  - Generate single image")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
