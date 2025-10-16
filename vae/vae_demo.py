import torch
import torchvision.datasets as datasets
import os
from pythae.models import AutoModel
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
TRAIN_NEW_MODEL = False  # Set to True to train a new model, False to load existing
NUM_EPOCHS = 200        # Longer training for better convergence
LEARNING_RATE = 5e-4    # Good balance for AdamW with weight decay
LATENT_DIM = 16         # Size of latent space
BATCH_SIZE = 128        # Good balance: fast + enough updates per epoch
WEIGHT_DECAY = 0.01     # L2 regularization for AdamW
USE_AMP = True          # Enable mixed precision for faster training on H100
# ===================================

device = "cuda" if torch.cuda.is_available() else "cpu"

mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.

from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST

if TRAIN_NEW_MODEL:
    print("=" * 70)
    print("🚀 TRAINING NEW VAE MODEL ON H100")
    print(f"   Epochs:          {NUM_EPOCHS}")
    print(f"   Learning Rate:   {LEARNING_RATE}")
    print(f"   Batch Size:      {BATCH_SIZE}")
    print(f"   Weight Decay:    {WEIGHT_DECAY}")
    print(f"   Optimizer:       AdamW")
    print(f"   Mixed Precision: {USE_AMP}")
    print(f"   Latent Dims:     {LATENT_DIM}")
    print("=" * 70)
    
    config = BaseTrainerConfig(
        output_dir='my_model',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        optimizer_cls="AdamW",  # AdamW with weight decay for better regularization
        optimizer_params={
            "betas": (0.9, 0.999),
            "weight_decay": WEIGHT_DECAY
        },
        amp=USE_AMP  # Enable automatic mixed precision for H100
    )

    model_config = VAEConfig(
        input_dim=(1, 28, 28),
        latent_dim=LATENT_DIM
        # Use default reconstruction loss settings from pythae
    )

    model = VAE(
        model_config=model_config,
        encoder=Encoder_ResNet_VAE_MNIST(model_config), 
        decoder=Decoder_ResNet_AE_MNIST(model_config)  # This decoder works for both AE and VAE
    )

    pipeline = TrainingPipeline(
        training_config=config,
        model=model
    )

    pipeline(
        train_data=train_dataset,
        eval_data=eval_dataset
    )
    
    print("\n✓ Training complete! Model saved to my_model/")
    print("=" * 70)

# Load the VAE model specifically (most recent VAE training)
print("\n" + "=" * 70)
print("📊 LOADING MODEL FOR VISUALIZATION")
print("=" * 70)

trainings = sorted(os.listdir('my_model'))
vae_trainings = [t for t in trainings if t.startswith('VAE_')]
if vae_trainings:
    last_training = vae_trainings[-1]
else:
    # Fallback to the last training
    last_training = trainings[-1]

print(f"Loading VAE model from: my_model/{last_training}")
trained_model = AutoModel.load_from_folder(os.path.join('my_model', last_training, 'final_model'))
print("✓ Model loaded successfully!\n")

from pythae.samplers import NormalSampler

# create normal sampler
normal_samper = NormalSampler(
    model=trained_model
)

# sample
gen_data = normal_samper.sample(
    num_samples=25
)

# show results with normal sampler
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(gen_data[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('vae_generated_normal_sampler.png', dpi=150, bbox_inches='tight')
plt.close()

from pythae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig

# set up GMM sampler config
gmm_sampler_config = GaussianMixtureSamplerConfig(
    n_components=10
)

# create gmm sampler
gmm_sampler = GaussianMixtureSampler(
    sampler_config=gmm_sampler_config,
    model=trained_model
)

# fit the sampler
gmm_sampler.fit(train_dataset)

# sample
gen_data = gmm_sampler.sample(
    num_samples=25
)

# show results with gmm sampler
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(gen_data[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('vae_generated_gmm_sampler.png', dpi=150, bbox_inches='tight')
plt.close()

reconstructions = trained_model.reconstruct(eval_dataset[:25].to(device)).detach().cpu()

# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i*5 + j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('vae_reconstructions.png', dpi=150, bbox_inches='tight')
plt.close()

# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(eval_dataset[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('vae_original_images.png', dpi=150, bbox_inches='tight')
plt.close()

interpolations = trained_model.interpolate(eval_dataset[:5].to(device), eval_dataset[5:10].to(device), granularity=10).detach().cpu()

# show interpolations
fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(10, 5))

for i in range(5):
    for j in range(10):
        axes[i][j].imshow(interpolations[i, j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('vae_interpolations.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print("✓ All VAE visualizations saved successfully!")
print("=" * 70)
print("Generated files:")
print("  - vae_generated_normal_sampler.png (random sampling)")
print("  - vae_generated_gmm_sampler.png (GMM sampling)")
print("  - vae_original_images.png (real MNIST images)")
print("  - vae_reconstructions.png (VAE reconstructions)")
print("  - vae_interpolations.png (latent space interpolations)")
print("=" * 70)