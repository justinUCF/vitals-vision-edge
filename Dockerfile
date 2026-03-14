# -----------------------------------------------------------------------
# VITALS Vision Edge — Jetson Orin Nano (JetPack 6.x / L4T r36)
#
# Base image ships PyTorch 2.1 + CUDA built for aarch64/Jetson.
# Run with:  docker run --runtime nvidia --device /dev/video0 ...
# -----------------------------------------------------------------------
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

WORKDIR /app

# System libraries required by OpenCV headless on L4T
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# src/ must be on PYTHONPATH so sub-packages import correctly
ENV PYTHONPATH=/app/src

# Default config — override at runtime via -e or docker-compose environment:
ENV DEVICE=cuda
ENV CAMERA_INDEX=0
ENV YOLO_MODEL=/app/yolo_models/rf3v1.pt
ENV OLLAMA_HOST=http://localhost:11434
ENV AGENT_B_HOST=192.168.1.100
ENV AGENT_B_PORT=9000
ENV UAV_ID=UAV_1

CMD ["python", "src/main.py"]

