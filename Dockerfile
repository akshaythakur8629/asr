FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install system dependencies needed for building extensions and running audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    rustc \
    cargo \
    libsndfile1-dev \
    libicu-dev \
    libfst-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first to maximize build caching
COPY requirements.txt .

# Install cython/packaging first as they are required compile-time setup dependencies for NeMo/DeepFilterNet
RUN pip install --no-cache-dir Cython packaging

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and static assets
COPY . .

# Expose the API and frontend port
EXPOSE 8000

# Ensure logs flush immediately and temp directory matches pipeline configurations
ENV PYTHONUNBUFFERED=1
ENV TEMP_DIR=/tmp/nemotron-test

# Default ASR batching configurations optimized for production (L40S)
ENV ASR_MAX_BATCH_SIZE=16
ENV ASR_BATCH_TIMEOUT_SEC=0.05

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
