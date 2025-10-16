# VAE/AE Implementation on MNIST

## Installation

```bash
# Create conda environment
conda create -n vaenv python=3.10 pip

# Activate environment
conda activate vaenv

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install pythae and dependencies
pip install pythae matplotlib scikit-learn
```

## Scripts

### Training & Visualization

- **`ae_demo.py`** - Train/load Autoencoder and generate visualizations (normal/GMM sampling, reconstructions, interpolations)
- **`vae_demo.py`** - Train/load Variational Autoencoder and generate visualizations (normal/GMM sampling, reconstructions, interpolations)

### Analysis

- **`analyze_ae_model.py`** - Analyze AE: model stats, compression ratio, latent space distribution (PCA, t-SNE)
- **`analyze_vae_model.py`** - Analyze VAE: model stats, compression ratio, latent space distribution (PCA, t-SNE)
- **`check_vae_loss.py`** - Breakdown VAE loss into reconstruction and KL divergence components

### Comparison

- **`compare_ae_vae_latent_distributions.py`** - Compare latent space distributions between AE and VAE
- **`compare_vae_models_checkpoints.py`** - Compare two VAE training checkpoints to track improvements

## Usage

```bash
# Set TRAIN_NEW_MODEL = True in the script to train, or False to load existing
python ae_demo.py          # Run AE pipeline
python vae_demo.py         # Run VAE pipeline

# Analyze models
python analyze_ae_model.py
python analyze_vae_model.py

# Compare models
python compare_ae_vae_latent_distributions.py
```

## Configuration

Both demo scripts have configurable parameters at the top:
- `NUM_EPOCHS` - Training epochs
- `LEARNING_RATE` - Optimizer learning rate
- `BATCH_SIZE` - Batch size (optimized for H100)
- `WEIGHT_DECAY` - L2 regularization
- `USE_AMP` - Mixed precision training

## Output Files

All visualizations are saved with prefixes:
- `ae_*.png` - Autoencoder outputs
- `vae_*.png` - VAE outputs

