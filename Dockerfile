FROM nvcr.io/nvidia/pytorch:23.06-py3
LABEL authors="mesuttoruk"


# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sox \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install NeMo
RUN pip install nemo_toolkit[all]
RUN pip install git+https://github.com/NVIDIA/NeMo.git

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy application code
COPY . .

# Expose WebSocket server port
EXPOSE 8766

# Run the ASR WebSocket server
CMD ["python", "server.py"]
