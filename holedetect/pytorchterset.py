import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Print the result
print(f"CUDA available: {cuda_available}")

# If CUDA is available, print the CUDA version
if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")

# Print the PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check the number of GPUs available
print(f"Number of GPUs: {torch.cuda.device_count()}")

# If GPUs are available, print the name of the first GPU
if torch.cuda.device_count() > 0:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
