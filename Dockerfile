# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy the application files and models
COPY app.py .
COPY models/diagnosis_model.h5 models/
COPY models/benign_malignant_model.h5 models/

# Set environment variables for TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_ENABLE_AUTO_MIXED_PRECISION=1

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"] 