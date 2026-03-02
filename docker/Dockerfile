FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git curl ca-certificates \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch (newer CUDA wheels - must support sm_120)
RUN pip install --no-cache-dir --upgrade \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Common deps
RUN pip install --no-cache-dir \
    numpy scipy scikit-learn pyyaml tqdm \
    matplotlib pandas \
    opencv-python \
    imageio imageio-ffmpeg

# Build-time sanity check (prints arch list)
RUN python3 - << 'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("arch list:", torch.cuda.get_arch_list())
PY
