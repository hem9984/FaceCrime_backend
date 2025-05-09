# Use NVIDIA CUDA runtime (with cuDNN) for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Avoid .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system-level build dependencies & Python headers
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3-pip python3-dev build-essential \
      libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Copy and install only essential Python requirements
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8443

# Run with multiple workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8443", "--workers", "4"]

