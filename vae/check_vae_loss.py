import torch
import torchvision.datasets as datasets
import os
from pythae.models import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the latest VAE model
trainings = sorted(os.listdir('my_model'))
vae_trainings = [t for t in trainings if t.startswith('VAE_')]
last_training = vae_trainings[-1]

print("=" * 80)
print("VAE LOSS BREAKDOWN ANALYSIS")
print("=" * 80)
print(f"\nLoading: {last_training}")

vae_model = AutoModel.load_from_folder(os.path.join('my_model', last_training, 'final_model')).to(device)

# Load MNIST
mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.

# Test on a batch
test_batch = eval_dataset[:100].to(device)

print("\n" + "=" * 80)
print("COMPUTING LOSS ON 100 TEST IMAGES")
print("=" * 80)

with torch.no_grad():
    # Forward pass through the VAE (pythae expects dict input)
    model_input = {"data": test_batch}
    output = vae_model(model_input)
    
    # Check what's in the output
    print("\nVAE Output Keys:")
    if hasattr(output, '__dict__'):
        for key in output.__dict__.keys():
            print(f"  - {key}: {getattr(output, key) if not torch.is_tensor(getattr(output, key)) else f'tensor shape {getattr(output, key).shape}'}")
    
    # Try to get individual loss components
    if hasattr(output, 'loss'):
        total_loss = output.loss
        print(f"\n✓ Total Loss: {total_loss:.4f}")
    
    if hasattr(output, 'recon_loss'):
        recon_loss = output.recon_loss
        print(f"  - Reconstruction Loss: {recon_loss:.4f}")
    
    if hasattr(output, 'kld') or hasattr(output, 'kld_loss'):
        kld = output.kld if hasattr(output, 'kld') else output.kld_loss
        print(f"  - KL Divergence: {kld:.4f}")
    
    # Manual reconstruction loss calculation
    recon = vae_model.reconstruct(test_batch)
    mse_loss = torch.nn.functional.mse_loss(recon, test_batch, reduction='mean')
    print(f"\n✓ Manual MSE Loss: {mse_loss:.6f}")
    
    # Scaled by number of pixels
    num_pixels = 28 * 28
    scaled_mse = mse_loss * num_pixels
    print(f"✓ MSE × pixels (28×28): {scaled_mse:.4f}")

print("\n" + "=" * 80)


