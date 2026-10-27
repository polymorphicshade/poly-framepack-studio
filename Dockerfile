# poly-framepack-studio: the Gradio web UI and HTTP API on an NVIDIA GPU.
#
# The PyTorch CUDA wheels bundle the CUDA runtime, so a slim Python base is enough. The host
# only needs the NVIDIA driver and the NVIDIA Container Toolkit. Model weights are not baked in:
# they download on first use into /data/cache (mount a volume there).
#
#   docker build -t poly-framepack-studio .
#   docker run --gpus all -p 7860:7860 -v fps-cache:/data/cache -v fps-outputs:/data/outputs poly-framepack-studio

ARG PYTHON_VERSION=3.12
# CUDA 12.8 wheels run on driver >= 570 and support every GPU from Turing to Blackwell.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
# The newest torch the pinned diffusers/transformers versions in requirements.txt are known to work with.
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
# Attention backend: "sage" (fastest; compiled from source, as there are no Linux wheels), "flash"
# (a prebuilt flash-attn wheel, no compiling) or "none" (PyTorch's own attention). Both need an
# Ampere (RTX 30-series) or newer GPU; older GPUs fall back to PyTorch's attention.
ARG ATTENTION=sage
# GPU architectures SageAttention is compiled for: Ampere (8.0, 8.6), Ada (8.9), Hopper (9.0) and
# Blackwell (12.0). Listing only your own GPUs (e.g. "8.6;8.9") builds faster.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ARG SAGEATTENTION_REF=v2.2.0
# The SageAttention build runs MAX_JOBS compiler jobs with NVCC_THREADS threads each, and a single
# thread can take several GB of RAM. Upstream's defaults (4 x 8) can exhaust the memory of a desktop
# or a Docker Desktop VM and hang the machine; raise these only with plenty of RAM to spare.
ARG MAX_JOBS=1
ARG NVCC_THREADS=2
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# ---------------------------------------------------------------------------- build
FROM python:${PYTHON_VERSION}-slim-bookworm AS build
ARG TORCH_INDEX
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG ATTENTION
ARG TORCH_CUDA_ARCH_LIST
ARG SAGEATTENTION_REF
ARG MAX_JOBS
ARG NVCC_THREADS
ARG FLASH_ATTN_WHEEL

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Torch first, from the CUDA index, in its own layer: it is by far the largest and changes least.
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX}

# basicsr (for the ESRGAN upscaler) is only an sdist and lists torch in setup_requires. Built in
# pip's usual isolated environment it would fetch a second, CPU-only torch, so build it against
# the one above instead.
RUN pip install "setuptools<81" wheel cython numpy==1.26.2 \
    && pip install --no-build-isolation basicsr

# The attention backend (see ATTENTION above). SageAttention builds against the torch above, so it
# needs nvcc and the CUDA headers matching the cu128 wheels. The toolkit stays in this stage; only
# the compiled package in /opt/venv reaches the runtime image.
RUN set -e; \
    if [ "$ATTENTION" = "sage" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends build-essential git wget ca-certificates; \
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb; \
        dpkg -i cuda-keyring_1.1-1_all.deb; rm cuda-keyring_1.1-1_all.deb; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8 cuda-nvtx-12-8 cuda-profiler-api-12-8 \
            libcublas-dev-12-8 libcusparse-dev-12-8 libcusolver-dev-12-8 libcurand-dev-12-8; \
        rm -rf /var/lib/apt/lists/*; \
        git clone --depth 1 --branch "$SAGEATTENTION_REF" https://github.com/thu-ml/SageAttention.git /tmp/sageattention; \
        sed -i "s/--threads=8/--threads=${NVCC_THREADS}/" /tmp/sageattention/setup.py; \
        pip install packaging ninja; \
        CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="$MAX_JOBS" \
            pip install --no-build-isolation /tmp/sageattention; \
        rm -rf /tmp/sageattention; \
    elif [ "$ATTENTION" = "flash" ]; then \
        pip install "$FLASH_ATTN_WHEEL"; \
    elif [ "$ATTENTION" != "none" ]; then \
        echo "ATTENTION must be sage, flash or none, not '$ATTENTION'" >&2; exit 1; \
    fi

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm

# ffmpeg: video encoding and the post-processing toolbox (so it is never downloaded at runtime);
# libgl1 and libglib2.0-0: OpenCV; gcc and libc6-dev: Triton (used by SageAttention) compiles a
# small launcher with the system C compiler the first time it runs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/cache /data/outputs /data/loras \
    && chown -R app:app /data

COPY --from=build /opt/venv /opt/venv
COPY --chown=root:root . /app

# FRAMEPACK_DATA_DIR holds settings, presets, the queue, LoRAs, outputs and temp files;
# FRAMEPACK_CACHE_DIR holds the downloaded weights (see modules/paths.py).
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FRAMEPACK_DATA_DIR=/data \
    FRAMEPACK_CACHE_DIR=/data/cache \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache \
    TRITON_CACHE_DIR=/data/cache/triton \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    FRAMEPACK_SERVE_ARGS=""

USER app
WORKDIR /data
VOLUME ["/data", "/data/cache", "/data/outputs"]
EXPOSE 7860

# The first boot downloads the HunyuanVideo text encoders, VAE and image encoder (tens of GB)
# before the server starts listening, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4)"

# Extra studio.py flags (e.g. "--offline" or "--lora my_lora.safetensors") go in FRAMEPACK_SERVE_ARGS.
CMD ["sh", "-c", "exec python /app/studio.py --server 0.0.0.0 --port 7860 $FRAMEPACK_SERVE_ARGS"]
