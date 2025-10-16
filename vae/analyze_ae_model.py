import torch
import torchvision.datasets as datasets
import os
import numpy as np
import matplotlib.pyplot as plt
from pythae.models import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the trained AE model specifically (most recent AE training)
trainings = sorted(os.listdir('my_model'))
ae_trainings = [t for t in trainings if t.startswith('AE_')]
if ae_trainings:
    last_training = ae_trainings[-1]
else:
    # Fallback to the last training
    last_training = trainings[-1]
print(f"Loading AE model from: my_model/{last_training}")
trained_model = AutoModel.load_from_folder(os.path.join('my_model', last_training, 'final_model'))
trained_model = trained_model.to(device)

# Load MNIST dataset
mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.

print("=" * 70)
print("AE MODEL ANALYSIS")
print("=" * 70)

# 1. MODEL SIZE AND PARAMETERS
print("\n📊 MODEL ARCHITECTURE:")
print("-" * 70)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

encoder_params = count_parameters(trained_model.encoder)
decoder_params = count_parameters(trained_model.decoder)
total_params = encoder_params + decoder_params

print(f"Encoder parameters:        {encoder_params:,}")
print(f"Decoder parameters:        {decoder_params:,}")
print(f"Total trainable parameters: {total_params:,}")

# Model size in MB
model_size_mb = (total_params * 4) / (1024 * 1024)  # assuming float32
print(f"Estimated model size:       {model_size_mb:.2f} MB")

# 2. LATENT SPACE SIZE
print("\n🔢 LATENT SPACE:")
print("-" * 70)

latent_dim = trained_model.model_config.latent_dim
input_shape = trained_model.model_config.input_dim
input_size = np.prod(input_shape)

print(f"Input dimensions:          {input_shape} = {input_size} values")
print(f"Latent dimensions:         {latent_dim}")

# 3. COMPRESSION RATIO
print("\n📉 COMPRESSION:")
print("-" * 70)

compression_ratio = input_size / latent_dim
print(f"Compression ratio:         {compression_ratio:.2f}x")
print(f"Compression percentage:    {(1 - latent_dim/input_size) * 100:.2f}%")
print(f"Information retention:     {(latent_dim/input_size) * 100:.2f}%")

# 4. ENCODE DATASET TO GET LATENT REPRESENTATIONS
print("\n🔍 ANALYZING LATENT SPACE...")
print("-" * 70)

# Sample a subset for analysis (use 1000 images from training set)
sample_size = 1000
sample_data = train_dataset[:sample_size].to(device)
sample_labels = mnist_trainset.targets[:sample_size].numpy()

# Encode to latent space
with torch.no_grad():
    encoded = trained_model.encoder(sample_data)
    if hasattr(encoded, 'embedding'):
        latent_codes = encoded.embedding.cpu().numpy()
    else:
        latent_codes = encoded.cpu().numpy()

print(f"Encoded {sample_size} images")
print(f"Latent codes shape:        {latent_codes.shape}")

# Statistics of latent space
print(f"Latent mean:               {latent_codes.mean():.4f}")
print(f"Latent std:                {latent_codes.std():.4f}")
print(f"Latent min:                {latent_codes.min():.4f}")
print(f"Latent max:                {latent_codes.max():.4f}")

# 5. VISUALIZE LATENT SPACE DISTRIBUTION
print("\n📈 Creating visualizations...")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# Plot 1: Overall latent space distribution (histogram)
ax1 = plt.subplot(2, 3, 1)
ax1.hist(latent_codes.flatten(), bins=50, alpha=0.7, edgecolor='black')
ax1.set_title('Latent Space Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Latent Value')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution per dimension (first 8 dimensions)
ax2 = plt.subplot(2, 3, 2)
num_dims_to_show = min(8, latent_dim)
for i in range(num_dims_to_show):
    ax2.hist(latent_codes[:, i], bins=30, alpha=0.5, label=f'Dim {i}')
ax2.set_title(f'First {num_dims_to_show} Latent Dimensions', fontsize=14, fontweight='bold')
ax2.set_xlabel('Latent Value')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Mean activation per dimension
ax3 = plt.subplot(2, 3, 3)
mean_activations = latent_codes.mean(axis=0)
ax3.bar(range(latent_dim), mean_activations)
ax3.set_title('Mean Activation per Latent Dimension', fontsize=14, fontweight='bold')
ax3.set_xlabel('Latent Dimension')
ax3.set_ylabel('Mean Value')
ax3.grid(True, alpha=0.3)

# Plot 4: Standard deviation per dimension
ax4 = plt.subplot(2, 3, 4)
std_activations = latent_codes.std(axis=0)
ax4.bar(range(latent_dim), std_activations, color='orange')
ax4.set_title('Std Deviation per Latent Dimension', fontsize=14, fontweight='bold')
ax4.set_xlabel('Latent Dimension')
ax4.set_ylabel('Std Dev')
ax4.grid(True, alpha=0.3)

# Plot 5: 2D visualization using first 2 principal components or dimensions
ax5 = plt.subplot(2, 3, 5)
scatter = ax5.scatter(latent_codes[:, 0], latent_codes[:, 1], 
                     c=sample_labels, cmap='tab10', alpha=0.6, s=10)
ax5.set_title('Latent Space (First 2 Dimensions)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Latent Dim 0')
ax5.set_ylabel('Latent Dim 1')
plt.colorbar(scatter, ax=ax5, label='Digit Class')
ax5.grid(True, alpha=0.3)

# Plot 6: Correlation matrix of latent dimensions (sample first 16 dims if more)
ax6 = plt.subplot(2, 3, 6)
dims_for_corr = min(16, latent_dim)
corr_matrix = np.corrcoef(latent_codes[:, :dims_for_corr].T)
im = ax6.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax6.set_title(f'Correlation Matrix (First {dims_for_corr} Dims)', fontsize=14, fontweight='bold')
ax6.set_xlabel('Latent Dimension')
ax6.set_ylabel('Latent Dimension')
plt.colorbar(im, ax=ax6, label='Correlation')

plt.tight_layout()
plt.savefig('ae_latent_space_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# 6. PCA VISUALIZATION (if latent_dim > 2)
if latent_dim > 2:
    from sklearn.decomposition import PCA
    
    print("Computing PCA for 2D visualization...")
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_codes)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    scatter = ax.scatter(latent_pca[:, 0], latent_pca[:, 1], 
                        c=sample_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('Latent Space PCA Projection (2D)', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.colorbar(scatter, ax=ax, label='Digit Class')
    ax.grid(True, alpha=0.3)
    
    # Add legend for digit classes
    for digit in range(10):
        mask = sample_labels == digit
        ax.scatter([], [], c=[plt.cm.tab10(digit/10)], label=f'Digit {digit}', s=50)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('ae_latent_space_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"PCA explained variance:    PC1={pca.explained_variance_ratio_[0]*100:.2f}%, PC2={pca.explained_variance_ratio_[1]*100:.2f}%")

# 7. t-SNE VISUALIZATION (if latent_dim > 2)
if latent_dim > 2:
    from sklearn.manifold import TSNE
    
    print("Computing t-SNE for 2D visualization (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_tsne = tsne.fit_transform(latent_codes)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                        c=sample_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('Latent Space t-SNE Projection (2D)', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter, ax=ax, label='Digit Class')
    ax.grid(True, alpha=0.3)
    
    # Add legend for digit classes
    for digit in range(10):
        mask = sample_labels == digit
        ax.scatter([], [], c=[plt.cm.tab10(digit/10)], label=f'Digit {digit}', s=50)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('ae_latent_space_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()

print("\n✓ AE Analysis complete!")
print("\nGenerated files:")
print("  - ae_latent_space_analysis.png (comprehensive analysis)")
if latent_dim > 2:
    print("  - ae_latent_space_pca.png (PCA projection)")
    print("  - ae_latent_space_tsne.png (t-SNE projection)")

print("\n" + "=" * 70)

