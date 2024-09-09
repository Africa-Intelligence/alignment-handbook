# Start from NVIDIA's CUDA 12.1 image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set up environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /root/

# Setup Lambda repository
ADD lambda.gpg .
RUN apt-get update && \
    apt-get install --yes gnupg && \
    gpg --dearmor -o /etc/apt/trusted.gpg.d/lambda.gpg < lambda.gpg && \
    rm lambda.gpg && \
    echo "deb http://archive.lambdalabs.com/ubuntu jammy main" > /etc/apt/sources.list.d/lambda.list && \
    echo "Package: *" > /etc/apt/preferences.d/lambda && \
    echo "Pin: origin archive.lambdalabs.com" >> /etc/apt/preferences.d/lambda && \
    echo "Pin-Priority: 1001" >> /etc/apt/preferences.d/lambda && \
    echo "cudnn cudnn/license_preseed select ACCEPT" | debconf-set-selections

# Update and install packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    --allow-change-held-packages \
    --option "Acquire::http::No-Cache=true" \
    --option "Acquire::http::Pipeline-Depth=0" \
    lambda-server \
    git \
    git-lfs \
    libnccl2 \
    libnccl-dev && \
    rm -rf /var/lib/apt/lists/*

# Setup for nvidia-docker
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.1"

# Set up the application
WORKDIR /app

# Copy everything from the current directory to the working directory in the image
COPY . .

# Install Python 3.10 and create a virtual environment
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv && \
    python3.10 -m venv /opt/venv && \
    rm -rf /var/lib/apt/lists/*

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
RUN pip3 install torch torchvision torchaudio

# Install the project and its dependencies
RUN pip install .

# Install Flash Attention 2
RUN pip install flash-attn --no-build-isolation

# Install wandb
RUN pip install wandb

# Make the entrypoint.sh script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint to run your main script
ENTRYPOINT ["/app/entrypoint.sh"]