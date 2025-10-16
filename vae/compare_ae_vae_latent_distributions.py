import torch
import torchvision.datasets as datasets
import os
import numpy as np
import matplotlib.pyplot as plt
from pythae.models import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load both models
trainings = sorted(os.listdir('my_model'))
ae_trainings = [t for t in trainings if t.startswith('AE_')]
vae_trainings = [t for t in trainings if t.startswith('VAE_')]

print("=" * 80)
print("LATENT SPACE DISTRIBUTION COMPARISON: AE vs VAE")
print("=" * 80)

ae_model = AutoModel.load_from_folder(os.path.join('my_model', ae_trainings[-1], 'final_model')).to(device)
vae_model = AutoModel.load_from_folder(os.path.join('my_model', vae_trainings[-1], 'final_model')).to(device)

# Load MNIST dataset
mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.

# Encode a subset of training data
sample_size = 1000
sample_data = train_dataset[:sample_size].to(device)

with torch.no_grad():
    # AE encoding
    ae_encoded = ae_model.encoder(sample_data)
    if hasattr(ae_encoded, 'embedding'):
        ae_latent = ae_encoded.embedding.cpu().numpy()
    else:
        ae_latent = ae_encoded.cpu().numpy()
    
    # VAE encoding
    vae_encoded = vae_model.encoder(sample_data)
    if hasattr(vae_encoded, 'embedding'):
        vae_latent = vae_encoded.embedding.cpu().numpy()
    else:
        vae_latent = vae_encoded.cpu().numpy()

# Compute statistics
ae_mean = ae_latent.mean()
ae_std = ae_latent.std()
ae_min = ae_latent.min()
ae_max = ae_latent.max()

vae_mean = vae_latent.mean()
vae_std = vae_latent.std()
vae_min = vae_latent.min()
vae_max = vae_latent.max()

print("\n📊 OVERALL STATISTICS (1000 samples, 16 dimensions):")
print("=" * 80)
print(f"{'Metric':<20} {'AE':<20} {'VAE':<20} {'Difference':<20}")
print("-" * 80)
print(f"{'Mean':<20} {ae_mean:<20.4f} {vae_mean:<20.4f} {vae_mean - ae_mean:<20.4f}")
print(f"{'Std Dev':<20} {ae_std:<20.4f} {vae_std:<20.4f} {vae_std - ae_std:<20.4f}")
print(f"{'Min Value':<20} {ae_min:<20.4f} {vae_min:<20.4f} {vae_min - ae_min:<20.4f}")
print(f"{'Max Value':<20} {ae_max:<20.4f} {vae_max:<20.4f} {vae_max - ae_max:<20.4f}")
print(f"{'Range':<20} {ae_max - ae_min:<20.4f} {vae_max - vae_min:<20.4f} {(vae_max - vae_min) - (ae_max - ae_min):<20.4f}")

# Per-dimension statistics
ae_dim_means = ae_latent.mean(axis=0)
ae_dim_stds = ae_latent.std(axis=0)
vae_dim_means = vae_latent.mean(axis=0)
vae_dim_stds = vae_latent.std(axis=0)

print("\n📈 PER-DIMENSION STATISTICS:")
print("=" * 80)
print(f"AE  - Mean per dim: avg={ae_dim_means.mean():.4f}, std={ae_dim_means.std():.4f}, range=[{ae_dim_means.min():.4f}, {ae_dim_means.max():.4f}]")
print(f"      Std per dim:  avg={ae_dim_stds.mean():.4f}, std={ae_dim_stds.std():.4f}, range=[{ae_dim_stds.min():.4f}, {ae_dim_stds.max():.4f}]")
print()
print(f"VAE - Mean per dim: avg={vae_dim_means.mean():.4f}, std={vae_dim_means.std():.4f}, range=[{vae_dim_means.min():.4f}, {vae_dim_means.max():.4f}]")
print(f"      Std per dim:  avg={vae_dim_stds.mean():.4f}, std={vae_dim_stds.std():.4f}, range=[{vae_dim_stds.min():.4f}, {vae_dim_stds.max():.4f}]")

# Distance from N(0,1)
ae_mean_dist = abs(ae_mean)
ae_std_dist = abs(ae_std - 1.0)
vae_mean_dist = abs(vae_mean)
vae_std_dist = abs(vae_std - 1.0)

print("\n🎯 DISTANCE FROM STANDARD NORMAL N(0,1):")
print("=" * 80)
print(f"Target: Mean=0.0, Std=1.0\n")
print(f"AE  - |mean - 0|: {ae_mean_dist:.4f},  |std - 1|: {ae_std_dist:.4f},  total: {ae_mean_dist + ae_std_dist:.4f}")
print(f"VAE - |mean - 0|: {vae_mean_dist:.4f},  |std - 1|: {vae_std_dist:.4f},  total: {vae_mean_dist + vae_std_dist:.4f}")

if (vae_mean_dist + vae_std_dist) < (ae_mean_dist + ae_std_dist):
    print(f"\n✓ VAE is {((ae_mean_dist + ae_std_dist) - (vae_mean_dist + vae_std_dist)):.4f} closer to N(0,1)")
else:
    print(f"\n✗ AE is {((vae_mean_dist + vae_std_dist) - (ae_mean_dist + ae_std_dist)):.4f} closer to N(0,1)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Overall distribution histograms
axes[0, 0].hist(ae_latent.flatten(), bins=50, alpha=0.6, label='AE', edgecolor='black')
axes[0, 0].hist(vae_latent.flatten(), bins=50, alpha=0.6, label='VAE', edgecolor='black')
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Target Mean=0')
axes[0, 0].set_title('Overall Latent Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Latent Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Mean per dimension
x = np.arange(16)
axes[0, 1].bar(x - 0.2, ae_dim_means, 0.4, label='AE', alpha=0.7)
axes[0, 1].bar(x + 0.2, vae_dim_means, 0.4, label='VAE', alpha=0.7)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2, label='Target=0')
axes[0, 1].set_title('Mean per Latent Dimension', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Dimension')
axes[0, 1].set_ylabel('Mean')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Std per dimension
axes[0, 2].bar(x - 0.2, ae_dim_stds, 0.4, label='AE', alpha=0.7)
axes[0, 2].bar(x + 0.2, vae_dim_stds, 0.4, label='VAE', alpha=0.7)
axes[0, 2].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Target=1')
axes[0, 2].set_title('Std Dev per Latent Dimension', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Dimension')
axes[0, 2].set_ylabel('Std Dev')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Box plot comparison
data_to_plot = [ae_latent.flatten(), vae_latent.flatten()]
axes[1, 0].boxplot(data_to_plot, labels=['AE', 'VAE'])
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 0].set_title('Distribution Spread (Box Plot)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Latent Value')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Scatter plot of first 2 dimensions
axes[1, 1].scatter(ae_latent[:, 0], ae_latent[:, 1], alpha=0.5, s=10, label='AE')
axes[1, 1].scatter(vae_latent[:, 0], vae_latent[:, 1], alpha=0.5, s=10, label='VAE')
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_title('Latent Space (First 2 Dims)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Dimension 0')
axes[1, 1].set_ylabel('Dimension 1')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Comparison of spreads
metrics = ['Mean', 'Std', 'Range']
ae_metrics = [abs(ae_mean), ae_std, ae_max - ae_min]
vae_metrics = [abs(vae_mean), vae_std, vae_max - vae_min]
x_pos = np.arange(len(metrics))
axes[1, 2].bar(x_pos - 0.2, ae_metrics, 0.4, label='AE', alpha=0.7)
axes[1, 2].bar(x_pos + 0.2, vae_metrics, 0.4, label='VAE', alpha=0.7)
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(metrics)
axes[1, 2].set_title('Distribution Metrics Comparison', fontsize=14, fontweight='bold')
axes[1, 2].set_ylabel('Value')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latent_distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Saved visualization to 'latent_distribution_comparison.png'")

print("\n" + "=" * 80)
print("💡 INTERPRETATION")
print("=" * 80)

if vae_std < ae_std:
    print("\n✓ VAE latent space is MORE COMPACT (tighter) than AE")
    print(f"  VAE std ({vae_std:.4f}) < AE std ({ae_std:.4f})")
    print("\n  Why? VAE's KL divergence term regularizes the latent space:")
    print("    - Encourages latent codes to have unit variance (std ≈ 1)")
    print("    - Prevents the latent space from spreading out too much")
    print("    - Results in more structured, compact representations")
else:
    print("\n✗ AE latent space is MORE COMPACT than VAE")
    print(f"  AE std ({ae_std:.4f}) < VAE std ({vae_std:.4f})")
    print("  This is unusual - VAE should typically be more compact")

if abs(vae_mean) < abs(ae_mean):
    print(f"\n✓ VAE is more centered around 0")
    print(f"  VAE mean ({vae_mean:.4f}) is closer to 0 than AE mean ({ae_mean:.4f})")
    print("  This is expected due to KL divergence regularization")

print("\n📚 Key Takeaway:")
print("-" * 80)
print("VAE explicitly learns a regularized latent space through KL divergence:")
print("  Loss = Reconstruction + β × KL(q(z|x) || N(0,1))")
print("\nThis forces the latent distribution towards N(0,1):")
print("  • Mean ≈ 0")
print("  • Std ≈ 1")
print("  • More compact, structured space")
print("\nAE has no such constraint:")
print("  • Latent space can be arbitrary")
print("  • May spread out more freely")
print("  • Only constrained by reconstruction accuracy")

print("\n" + "=" * 80)


