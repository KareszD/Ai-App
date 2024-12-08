# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

ENV AIP_STORAGE_URI=/gcs
ENV AIP_MODEL_DIR=/gcs/model
ENV AIP_DATA_FORMAT=csv

# Run the application to create the settings file
RUN python -c "from ultralytics import YOLO; YOLO('yolov9c.pt')"

# Modify the settings.json file to set datasets_dir to an empty string
RUN sed -i 's|"datasets_dir": ".*"|"datasets_dir": ""|' /root/.config/Ultralytics/settings.json

EXPOSE 6006

# Run the application
CMD ["python", "Main.py"]