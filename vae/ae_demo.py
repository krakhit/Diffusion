# code adapted from pythae https://github.com/clementchadebec/benchmark_VAE?tab=readme-ov-file
import torch
import torchvision.datasets as datasets
import os
from pythae.models import AutoModel
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
TRAIN_NEW_MODEL = False  # Set to True to train a new model, False to load existing
NUM_EPOCHS = 30          # Good balance for MNIST without overfitting
LEARNING_RATE = 1e-3     # Higher LR works well with larger batch size
BATCH_SIZE = 256         # Optimized for H100 (4x larger than before)
WEIGHT_DECAY = 0.005     # Mild regularization to prevent overfitting
USE_AMP = True           # Enable mixed precision for 2-3x speedup on H100
# ===================================

device = "cuda" if torch.cuda.is_available() else "cpu"

mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.

from pythae.models import AE, AEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_AE_MNIST, Decoder_ResNet_AE_MNIST

config = BaseTrainerConfig(
    output_dir='my_model',
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    optimizer_cls="AdamW",
    optimizer_params={
        "betas": (0.9, 0.999),
        "weight_decay": WEIGHT_DECAY
    },
    amp=USE_AMP
)


model_config = AEConfig(
    input_dim=(1, 28, 28),
    latent_dim=16
)

model = AE(
    model_config=model_config,
    encoder=Encoder_ResNet_AE_MNIST(model_config), 
    decoder=Decoder_ResNet_AE_MNIST(model_config) 
)

if TRAIN_NEW_MODEL:
    print("=" * 70)
    print("🚀 TRAINING NEW AE MODEL ON H100")
    print(f"   Epochs:          {NUM_EPOCHS}")
    print(f"   Learning Rate:   {LEARNING_RATE}")
    print(f"   Batch Size:      {BATCH_SIZE}")
    print(f"   Weight Decay:    {WEIGHT_DECAY}")
    print(f"   Optimizer:       AdamW")
    print(f"   Mixed Precision: {USE_AMP}")
    print(f"   Latent Dims:     16")
    print("=" * 70)
    
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

# Load the most recent AE model
trainings = sorted(os.listdir('my_model'))
ae_trainings = [t for t in trainings if t.startswith('AE_')]
if ae_trainings:
    last_training = ae_trainings[-1]
else:
    last_training = trainings[-1]

print(f"\nLoading AE model from: my_model/{last_training}")
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
plt.savefig('ae_generated_normal_sampler.png', dpi=150, bbox_inches='tight')
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
plt.savefig('ae_generated_gmm_sampler.png', dpi=150, bbox_inches='tight')
plt.close()

reconstructions = trained_model.reconstruct(eval_dataset[:25].to(device)).detach().cpu()

# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i*5 + j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('ae_reconstructions.png', dpi=150, bbox_inches='tight')
plt.close()

# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(eval_dataset[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('ae_original_images.png', dpi=150, bbox_inches='tight')
plt.close()

interpolations = trained_model.interpolate(eval_dataset[:5].to(device), eval_dataset[5:10].to(device), granularity=10).detach().cpu()

# show interpolations
fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(10, 5))

for i in range(5):
    for j in range(10):
        axes[i][j].imshow(interpolations[i, j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('ae_interpolations.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ All AE visualizations saved successfully!")
print("Generated files:")
print("  - ae_generated_normal_sampler.png (random sampling)")
print("  - ae_generated_gmm_sampler.png (GMM sampling)")
print("  - ae_original_images.png (real MNIST images)")
print("  - ae_reconstructions.png (autoencoder reconstructions)")
print("  - ae_interpolations.png (latent space interpolations)")