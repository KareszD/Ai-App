# Dockerfile
FROM python:3.10-slim

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements.txt if you have additional dependencies
# COPY requirements.txt .

# Install PyTorch and other Python dependencies
# It's recommended to install PyTorch via pip with CUDA support if needed
# For CPU-only:
#RUN pip install --no-cache-dir torch torchvision torchaudio
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the DDP script
COPY ddpbasics.py .

# Expose the master port (e.g., 29500)
EXPOSE 29500

# Set the entrypoint to run the script
# The actual launch command will be handled via torchrun
CMD ["python", "ddpbasics.py"]
