FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Use /workspace as the project working directory inside the container.
WORKDIR /workspace

# Install Linux packages needed for scientific Python and image processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip before installing project dependencies.
RUN python -m pip install --upgrade pip

# Copy Docker runtime dependencies first for better Docker layer caching.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the full project into the image.
COPY . .

# Make project modules importable from anywhere in the container.
ENV PYTHONPATH=/workspace

# Show the PyTorch and CUDA runtime status by default.
CMD ["python", "-c", "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"]
