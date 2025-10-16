import torch
import torchvision.datasets as datasets
import os
import numpy as np
import matplotlib.pyplot as plt
from pythae.models import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Find all VAE models
trainings = sorted(os.listdir('my_model'))
vae_trainings = [t for t in trainings if t.startswith('VAE_')]

if len(vae_trainings) < 2:
    print("ERROR: Need at least 2 VAE models to compare!")
    print(f"Found only {len(vae_trainings)} VAE model(s):")
    for vae in vae_trainings:
        print(f"  - {vae}")
    print("\nTrain more models to compare them.")
    exit(1)

# Load the two most recent VAE models
previous_vae_name = vae_trainings[-2]
current_vae_name = vae_trainings[-1]

print("=" * 80)
print("VAE MODEL COMPARISON: PREVIOUS vs CURRENT")
print("=" * 80)
print(f"\n📁 Previous VAE: {previous_vae_name}")
print(f"📁 Current VAE:  {current_vae_name}\n")

previous_vae = AutoModel.load_from_folder(os.path.join('my_model', previous_vae_name, 'final_model')).to(device)
current_vae = AutoModel.load_from_folder(os.path.join('my_model', current_vae_name, 'final_model')).to(device)

# Load MNIST dataset
mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.

# Test on eval set
test_size = 100
test_data = eval_dataset[:test_size].to(device)

print("=" * 80)
print("📊 LOSS BREAKDOWN (on 100 test images)")
print("=" * 80)

with torch.no_grad():
    # Previous VAE
    prev_input = {"data": test_data}
    prev_output = previous_vae(prev_input)
    prev_total_loss = prev_output.loss.item()
    prev_recon_loss = prev_output.recon_loss.item()
    prev_kl_loss = prev_output.reg_loss.item()
    
    # Current VAE
    curr_input = {"data": test_data}
    curr_output = current_vae(curr_input)
    curr_total_loss = curr_output.loss.item()
    curr_recon_loss = curr_output.recon_loss.item()
    curr_kl_loss = curr_output.reg_loss.item()

print("\n🔵 PREVIOUS VAE:")
print(f"   Total Loss:           {prev_total_loss:.4f}")
print(f"   Reconstruction Loss:  {prev_recon_loss:.4f} ({prev_recon_loss/prev_total_loss*100:.1f}%)")
print(f"   KL Divergence:        {prev_kl_loss:.4f} ({prev_kl_loss/prev_total_loss*100:.1f}%)")

print("\n🟢 CURRENT VAE:")
print(f"   Total Loss:           {curr_total_loss:.4f}")
print(f"   Reconstruction Loss:  {curr_recon_loss:.4f} ({curr_recon_loss/curr_total_loss*100:.1f}%)")
print(f"   KL Divergence:        {curr_kl_loss:.4f} ({curr_kl_loss/curr_total_loss*100:.1f}%)")

print("\n📈 IMPROVEMENT:")
total_improvement = ((prev_total_loss - curr_total_loss) / prev_total_loss) * 100
recon_improvement = ((prev_recon_loss - curr_recon_loss) / prev_recon_loss) * 100
kl_improvement = ((prev_kl_loss - curr_kl_loss) / prev_kl_loss) * 100

print(f"   Total Loss:           {total_improvement:+.2f}% {'✓' if total_improvement > 0 else '✗'}")
print(f"   Reconstruction Loss:  {recon_improvement:+.2f}% {'✓' if recon_improvement > 0 else '✗'}")
print(f"   KL Divergence:        {kl_improvement:+.2f}% {'✓' if kl_improvement > 0 else '✗'}")

# Compute pixel-level MSE for visualization
with torch.no_grad():
    prev_recon = previous_vae.reconstruct(test_data)
    curr_recon = current_vae.reconstruct(test_data)
    
    prev_mse = torch.nn.functional.mse_loss(prev_recon, test_data, reduction='none')
    prev_mse = prev_mse.view(test_size, -1).mean(dim=1)
    
    curr_mse = torch.nn.functional.mse_loss(curr_recon, test_data, reduction='none')
    curr_mse = curr_mse.view(test_size, -1).mean(dim=1)

print("\n📉 PIXEL-LEVEL MSE (Lower is Better):")
print(f"   Previous VAE: {prev_mse.mean():.6f} ± {prev_mse.std():.6f}")
print(f"   Current VAE:  {curr_mse.mean():.6f} ± {curr_mse.std():.6f}")
mse_improvement = ((prev_mse.mean() - curr_mse.mean()) / prev_mse.mean()) * 100
print(f"   Improvement:  {mse_improvement:+.2f}% {'✓' if mse_improvement > 0 else '✗'}")

# Check latent space distribution
print("\n🎨 LATENT SPACE DISTRIBUTION:")
print("-" * 80)

with torch.no_grad():
    prev_encoded = previous_vae.encoder(test_data[:50])
    curr_encoded = current_vae.encoder(test_data[:50])
    
    if hasattr(prev_encoded, 'embedding'):
        prev_latent = prev_encoded.embedding
    else:
        prev_latent = prev_encoded
    
    if hasattr(curr_encoded, 'embedding'):
        curr_latent = curr_encoded.embedding
    else:
        curr_latent = curr_encoded
    
    prev_mean = prev_latent.mean().item()
    prev_std = prev_latent.std().item()
    curr_mean = curr_latent.mean().item()
    curr_std = curr_latent.std().item()
    
    print(f"Previous VAE - Mean: {prev_mean:.4f}, Std: {prev_std:.4f}")
    print(f"Current VAE  - Mean: {curr_mean:.4f}, Std: {curr_std:.4f}")
    print(f"\nTarget: N(0,1) → Mean=0.0, Std=1.0")
    
    prev_dist_from_target = abs(prev_mean) + abs(prev_std - 1.0)
    curr_dist_from_target = abs(curr_mean) + abs(curr_std - 1.0)
    
    print(f"Distance from target:")
    print(f"  Previous: {prev_dist_from_target:.4f}")
    print(f"  Current:  {curr_dist_from_target:.4f}")
    if curr_dist_from_target < prev_dist_from_target:
        print("  ✓ Current VAE has better latent space distribution!")
    else:
        print("  ✗ Previous VAE had better latent space distribution")

# Create visual comparison
fig, axes = plt.subplots(3, 10, figsize=(20, 6))

for i in range(10):
    # Original
    axes[0, i].imshow(test_data[i].cpu().squeeze(0), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontsize=10, fontweight='bold')
    
    # Previous VAE reconstruction
    axes[1, i].imshow(prev_recon[i].cpu().squeeze(0), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title(f'Previous VAE\nMSE: {prev_mse[i]:.4f}', fontsize=10, fontweight='bold')
    else:
        axes[1, i].set_title(f'{prev_mse[i]:.4f}', fontsize=8)
    
    # Current VAE reconstruction
    axes[2, i].imshow(curr_recon[i].cpu().squeeze(0), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title(f'Current VAE\nMSE: {curr_mse[i]:.4f}', fontsize=10, fontweight='bold')
    else:
        axes[2, i].set_title(f'{curr_mse[i]:.4f}', fontsize=8)

plt.suptitle('VAE Comparison: Original vs Previous VAE vs Current VAE', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('vae_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Saved comparison visualization to 'vae_model_comparison.png'")

# Overall recommendation
print("\n" + "=" * 80)
print("🏆 OVERALL VERDICT:")
print("=" * 80)

# Count improvements
improvements = 0
if total_improvement > 0: improvements += 1
if recon_improvement > 0: improvements += 1
if mse_improvement > 0: improvements += 1
if curr_dist_from_target < prev_dist_from_target: improvements += 1

if improvements >= 3:
    print("🟢 CURRENT VAE is BETTER!")
    print(f"   Improvements in {improvements}/4 metrics")
    print("   ✓ Keep using the current model")
    print("   ✓ Training improvements were successful")
elif improvements >= 2:
    print("🟡 CURRENT VAE is SLIGHTLY BETTER")
    print(f"   Improvements in {improvements}/4 metrics")
    print("   ≈ Marginal improvements - both models are similar")
    print("   → Consider training longer or adjusting hyperparameters")
elif improvements >= 1:
    print("🟡 MIXED RESULTS")
    print(f"   Improvements in {improvements}/4 metrics")
    print("   → Some aspects improved, others regressed")
    print("   → Review training configuration")
else:
    print("🔴 PREVIOUS VAE was BETTER")
    print("   Current training made things worse")
    print("   → Revert to previous model")
    print("   → Review training hyperparameters (learning rate, batch size, epochs)")

print("\n💡 Training Info:")
print(f"   Previous: {previous_vae_name}")
print(f"   Current:  {current_vae_name}")
print("\n" + "=" * 80)


