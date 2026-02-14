# DC GAN MNIST Generator

A complete Deep Convolutional Generative Adversarial Network (DC GAN) implementation for generating synthetic handwritten digit images using the MNIST dataset.

## Project Structure

```
DC GAN/
├── requirements.txt       # Python dependencies
├── train.py              # DC GAN training script
├── app.py                # Flask web application
├── quick_train.py        # Quick training pipeline script
├── models/
│   └── dcgan.py          # DC GAN model architecture
├── templates/
│   └── index.html        # Web interface
├── checkpoints/          # Saved model checkpoints
├── samples/              # Generated sample images
├── plots/               # Training loss plots
└── data/                # MNIST dataset
```

## Features

- **Generator Network**: Creates 28x28 grayscale images from 100-dimensional latent vectors
- **Discriminator Network**: Classifies images as real or fake
- **Training Pipeline**: Complete training loop with logging, checkpointing, and visualization
- **Flask Web App**: Interactive web interface for generating images via API
- **GPU Support**: Automatically uses CUDA if available

## Installation

All dependencies are already installed. If needed:

```bash
pip install -r requirements.txt
```

## Dependencies

- torch >= 1.9.0
- torchvision >= 0.10.0
- flask >= 2.0.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- pillow >= 8.0.0
- gunicorn >= 20.0.0

## Usage

### 1. Train the Model

Run the training script:

```bash
python train.py
```

Training will:
- Download MNIST dataset automatically
- Train for 100 epochs by default
- Save checkpoints to `checkpoints/`
- Generate sample images to `samples/`
- Plot training losses to `plots/`

Configuration in `train.py`:
- Batch size: 64
- Learning rate: 0.0002
- Latent dimension: 100
- Epochs: 100

### 2. Start the Web Application

After training, start the Flask app:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### 3. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/generate` | POST | Generate multiple images (body: `{"num_images": N}`) |
| `/api/generate-single` | POST | Generate single image |
| `/api/status` | GET | Check model status |
| `/api/health` | GET | Health check |

Example API call:
```bash
curl -X POST http://localhost:5000/api/generate -H "Content-Type: application/json" -d '{"num_images": 8}'
```

## Model Architecture

### Generator
- Input: 100-dimensional latent vector
- Architecture: ConvTranspose2d -> BatchNorm -> ReLU
- Output: 28x28 grayscale image

### Discriminator
- Input: 28x28 grayscale image
- Architecture: Conv2d -> BatchNorm -> LeakyReLU -> Sigmoid
- Output: Real/Fake probability

## Training Tips

- Training typically takes 1-3 hours on CPU
- GPU training is significantly faster
- Monitor generator/discriminator loss balance
- Higher epoch counts (100+) yield better results

## Troubleshooting

1. **Import errors**: Ensure all dependencies are installed
2. **Out of memory**: Reduce batch size in Config class
3. **No checkpoint found**: Train the model first before running app.py
4. **Port already in use**: Change port in app.run()

## License

MIT License
