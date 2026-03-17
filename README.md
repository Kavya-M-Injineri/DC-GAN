# DC-GAN MNIST Generator

A complete **Deep Convolutional Generative Adversarial Network** implementation in PyTorch that generates synthetic handwritten digit images, served via a Flask REST API with a web interface for real-time image generation.

---

## Highlights

- Implemented a **DCGAN architecture from scratch** in PyTorch — Generator using transposed convolutions (ConvTranspose2d + BatchNorm + ReLU) and Discriminator using strided convolutions (Conv2d + BatchNorm + LeakyReLU + Sigmoid)
- Built a **complete adversarial training pipeline** with alternating Generator/Discriminator updates, loss logging, checkpoint saving every N epochs, and sample grid visualization to track generation quality over time
- Deployed the trained model as a **Flask REST API** with endpoints for single/batch image generation, model status checks, and a browser-based generation interface
- Configured **automatic CUDA/CPU detection** — training and inference adapt to available hardware with no code changes
- Designed for **reproducibility** — fixed latent dimension (z=100), documented hyperparameters (lr=0.0002, batch=64), and checkpoint resume support

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch, TorchVision |
| Model | DCGAN (Conv + ConvTranspose) |
| Backend | Flask, Gunicorn |
| Visualization | Matplotlib, Pillow |
| Dataset | MNIST (auto-downloaded via TorchVision) |
| Hardware | CUDA (auto-detected) / CPU |

---

## Model Architecture

**Generator** — latent vector → image
```
z (100-dim) → ConvTranspose2d → BatchNorm → ReLU
            → ConvTranspose2d → BatchNorm → ReLU
            → ConvTranspose2d → Tanh → 28×28 grayscale image
```

**Discriminator** — image → real/fake probability
```
28×28 image → Conv2d → LeakyReLU
            → Conv2d → BatchNorm → LeakyReLU
            → Conv2d → Sigmoid → P(real)
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** `torch >= 1.9`, `torchvision >= 0.10`, `flask >= 2.0`, `numpy`, `matplotlib`, `pillow`, `gunicorn`

---

## Usage

### Train

```bash
python train.py
```

| Config | Value |
|---|---|
| Epochs | 100 |
| Batch size | 64 |
| Learning rate | 0.0002 |
| Latent dim | 100 |

Outputs: `checkpoints/` (model weights) · `samples/` (generated grids) · `plots/` (loss curves)

### Serve

```bash
python app.py
# http://localhost:5000
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Browser-based generation UI |
| `POST` | `/api/generate` | Generate N images — body: `{"num_images": N}` |
| `POST` | `/api/generate-single` | Generate one image |
| `GET` | `/api/status` | Model load status |
| `GET` | `/api/health` | Health check |

```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"num_images": 8}'
```

---

## Project Structure

```
├── models/
│   └── dcgan.py          # Generator + Discriminator architecture
├── train.py              # Adversarial training loop
├── quick_train.py        # Faster training pipeline
├── app.py                # Flask API + web interface
├── templates/
│   └── index.html        # Generation UI
├── checkpoints/          # Saved model weights
├── samples/              # Generated image grids per epoch
└── plots/                # Generator / Discriminator loss curves
```
