# =============================================================================
# Qwen Image API — Docker Image
#
# Target hardware : NVIDIA RTX Pro 6000 (Blackwell, sm_120)
# CUDA toolkit    : 12.8
# cuDNN           : 9
# Python          : 3.10
# PyTorch         : 2.6.0 + cu126  (first release with native sm_120 support)
#
# Endpoints:
#   POST /text2img      — Text → Image  (Qwen-Image-2512)
#   POST /edit          — Image + Prompt → Image  (Qwen-Image-Edit-2511)
#   POST /edit-multi    — Multiple Images + Prompt → Image
#   POST /angle         — Image + Angle → Image  (with LoRA)
#   GET  /health
#
# Build:
#   docker build -t qwen-image-api:latest .
#
# Run (quick test):
#   docker run --gpus all -p 8190:8190 qwen-image-api:latest
#
# NOTE: First request triggers lazy model download from Hugging Face.
#       Mount a volume at /notebooks/model_team/huggingface_cache to persist
#       models across container restarts (see docker-compose.yml).
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# ── Build-time arguments ──────────────────────────────────────────────────────
# CUDA arch list — sm_120 = RTX Pro 6000 (Blackwell).
# Trim to just "12.0" for faster single-target builds.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"

# Parallel compile jobs for any CUDA extensions compiled at build time.
ARG MAX_JOBS=4

# ── Proxy (build-time + runtime) ──────────────────────────────────────────────
ARG http_proxy="http://proxy.intra:80"
ARG https_proxy="http://proxy.intra:80"
ARG no_proxy="localhost,127.0.0.1"

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    no_proxy=${no_proxy} \
    NO_PROXY=${no_proxy}

# ── Environment variables ─────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # CUDA
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    # Hugging Face model cache.
    # Host path /pegaai is mounted as /notebooks inside the container,
    # so the effective host location is /pegaai/model_team/huggingface_cache.
    HF_HOME=/notebooks/model_team/huggingface_cache \
    TRANSFORMERS_CACHE=/notebooks/model_team/huggingface_cache \
    HUGGINGFACE_HUB_CACHE=/notebooks/model_team/huggingface_cache

# ── apt proxy config (only takes effect if http_proxy ARG is set) ─────────────
RUN printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' \
      "${http_proxy}" "${https_proxy}" \
      > /etc/apt/apt.conf.d/99proxy

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python runtime + headers
    python3.10 \
    python3.10-dev \
    python3-pip \
    # Build tools (needed for any pip packages with C extensions)
    build-essential \
    ninja-build \
    cmake \
    git \
    wget \
    curl \
    # OpenCV / PIL runtime libs
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # OpenMP
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.10 as default interpreter ───────────────────────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade --no-cache-dir pip setuptools wheel

WORKDIR /app

# =============================================================================
# STEP 1 — PyTorch 2.6 with CUDA 12.6 wheels
#
# cu126 wheels are forward-compatible with the CUDA 12.8 runtime.
# PyTorch 2.6 is the first stable release that includes sm_120 (Blackwell)
# kernels compiled into the distributed binaries.
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu126

# =============================================================================
# STEP 2 — Application dependencies
#
# Installed after torch so the pinned diffusers git commit and other packages
# resolve against the already-installed torch version.
# torch / torchvision are intentionally excluded from requirements-api.txt
# to prevent pip from overwriting the cu126 wheels with CPU-only wheels.
# =============================================================================
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# =============================================================================
# Application source
# =============================================================================
COPY qwen_image_api.py /app/qwen_image_api.py

# Create mount point for the host HF cache and local output directory.
# /notebooks is the container-side mount of the host's /pegaai directory.
RUN mkdir -p /notebooks /app/outputs

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8190

# ── Health check ──────────────────────────────────────────────────────────────
# start-period is long because the first request triggers a lazy model download
# (~minutes depending on network speed and VRAM).
HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=300s \
    --retries=5 \
    CMD curl -f http://localhost:8190/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Single worker: the GPU pipeline is protected by asyncio.Lock() internally,
# so multiple uvicorn workers would fight for the GPU without benefit.
CMD ["python", "-m", "uvicorn", "qwen_image_api:app", \
     "--host", "0.0.0.0", \
     "--port", "8190", \
     "--workers", "1", \
     "--log-level", "info"]
