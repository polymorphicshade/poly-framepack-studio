"""
Where poly-framepack-studio keeps its files.

Both roots default to the project folder, so a local install keeps everything next to the code.
The Docker image moves them onto volumes:

    FRAMEPACK_DATA_DIR    settings, presets, the queue, LoRAs, outputs and temp files
    FRAMEPACK_CACHE_DIR   downloaded weights: Hugging Face models, ESRGAN, GFPGAN and RIFE
"""

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FRAMEPACK_DATA_DIR") or APP_DIR).resolve()
CACHE_DIR = Path(os.environ.get("FRAMEPACK_CACHE_DIR") or DATA_DIR).resolve()

# State
SETTINGS_DIR = DATA_DIR / ".framepack"
TOOLBOX_DATA_DIR = SETTINGS_DIR / "toolbox"
QUEUE_FILE = DATA_DIR / "queue.json"
QUEUE_IMAGES_DIR = DATA_DIR / "queue_images"
OUTPUTS_DIR = DATA_DIR / "outputs"
LORAS_DIR = DATA_DIR / "loras"
TEMP_DIR = DATA_DIR / "temp"
INPUT_FILES_DIR = DATA_DIR / "input_files"

# Weights
HF_HOME = CACHE_DIR / "hf_download"
ESRGAN_DIR = CACHE_DIR / "model_esrgan"
GFPGAN_DIR = CACHE_DIR / "model_gfpgan"
RIFE_DIR = CACHE_DIR / "model_rife"
FFMPEG_BIN_DIR = CACHE_DIR / "ffmpeg"
