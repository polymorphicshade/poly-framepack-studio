# poly-framepack-studio: the Gradio web UI on an NVIDIA GPU.
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

# ---------------------------------------------------------------------------- build
FROM python:${PYTHON_VERSION}-slim-bookworm AS build
ARG TORCH_INDEX
ARG TORCH_VERSION
ARG TORCHVISION_VERSION

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Torch first, from the CUDA index, in its own layer: it is by far the largest and changes least.
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX}

# basicsr (for the ESRGAN upscaler) is only an sdist and lists torch in setup_requires. Built in
# pip's usual isolated environment it would fetch a second, CPU-only torch, so build it against
# the one above instead.
RUN pip install setuptools wheel cython numpy==1.26.2 \
    && pip install --no-build-isolation basicsr

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm

# ffmpeg: video encoding and the post-processing toolbox (so it is never downloaded at runtime);
# libgl1 and libglib2.0-0: OpenCV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
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
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    FRAMEPACK_SERVE_ARGS=""

USER app
WORKDIR /data
VOLUME ["/data", "/data/cache", "/data/outputs"]
EXPOSE 7860

# The first boot downloads the HunyuanVideo text encoders, VAE and image encoder (tens of GB)
# before the server starts listening, hence the long start period. /gradio_api/queue/status is a small JSON response; "/" renders the whole UI config.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/gradio_api/queue/status', timeout=4)"

# Extra studio.py flags (e.g. "--offline" or "--lora my_lora.safetensors") go in FRAMEPACK_SERVE_ARGS.
CMD ["sh", "-c", "exec python /app/studio.py --server 0.0.0.0 --port 7860 $FRAMEPACK_SERVE_ARGS"]
