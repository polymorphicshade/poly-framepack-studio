# poly-framepack-studio

A video generation studio for [FramePack](https://github.com/lllyasviel/FramePack) (HunyuanVideo), with a Gradio web UI styled like [poly-wan](https://github.com/polymorphicshade/poly-wan). It is based on [FramePack Studio](https://github.com/FP-Studio/framepack-studio) and adds poly-wan's look and Docker deployment, including HTTPS.

| Generation type | What it does |
|---|---|
| **Original** | Image-to-video (or text-to-video) with the original FramePack model. |
| **Original with Endframe** | Same, with an end frame whose influence you set. |
| **F1** | The FramePack F1 model, which generates forwards in time. |
| **Video**, **Video with Endframe**, **Video F1** | Extends an input video, optionally towards an end frame, and can join the result onto the source. |

Every type runs in the same job queue. Other features:

- **Timestamped prompts**: give different prompts for different times, e.g. `[1s: waves hello] [3s: jumps]`, and choose how many sections to blend between them.
- **LoRAs**: works with most HunyuanVideo LoRAs, each with its own weight.
- **Queue**: jobs run one after another while you keep working. Queues can be exported to a zip and imported again.
- **Metadata**: the prompt and seed are stored in the output PNG; all other settings go into a JSON file that can be loaded back later.
- **Presets**: named groups of settings per generation type, plus a startup model and preset.
- **Latent image options**: with no input image, start from noise, black, white or a green screen.
- **XY plot**: compare two parameters across a grid of generations.
- **Prompt enhancer and captioner**: rewrite a prompt with an LLM, or caption the start image.
- **Post-processing**: ESRGAN upscaling with optional GFPGAN face restoration, RIFE frame interpolation, filters, frame extraction and reassembly, and export to MP4, WebM or GIF.
- **Caches**: MagCache or TeaCache speed up sampling.

## Requirements

- An NVIDIA GPU with at least 8 GB VRAM (16 GB or more recommended).
- 16 GB of system RAM (32 GB or more strongly recommended).
- 80 GB of free disk space or more: about 25 GB for each model family (Original and F1), plus the shared text encoders.

## Install

On Windows, `install.bat` creates `.venv`, picks the right PyTorch build for your GPU, optionally installs Sage Attention and Flash Attention, and installs the requirements. Start the app with `START.bat`. `update.bat` pulls the latest code and updates the packages.

By hand, on any platform:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate elsewhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python studio.py                  # or START.bat / ./START.sh
```

Open http://127.0.0.1:7860. `studio.py` takes `--port`, `--server` (default `0.0.0.0`), `--share`, `--inbrowser`, `--offline` and `--lora`.

## Docker (Linux server)

There are two compose files. Both run the app on an NVIDIA GPU. The host needs Docker, the NVIDIA driver (570 or newer) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). On Ubuntu or Debian, `sudo ./NVIDIA.sh` installs all three and checks that a container can see the GPU.

**Plain HTTP** (`docker-compose.yml`):

```bash
cp .env.example .env          # optional: port, LoRA folder and serve flags
mkdir -p loras
docker compose up -d --build
docker compose logs -f app
```

The UI is then at `http://<host>:7860`.

**HTTPS** (`docker-compose.https.yml`) runs the same app plus an nginx sidecar that terminates TLS:

```bash
cp .env.https.example .env    # list the host's LAN address in TLS_SAN
mkdir -p loras
docker compose -f docker-compose.https.yml up -d --build
```

The UI is then at `https://<host>:8443`.

- **Certificate:** a self-signed certificate is generated into `./certs` on first boot, so the browser warns once until you accept it. Every address you type in the browser must be in `TLS_SAN`, e.g. `IP:192.168.1.99`. After changing it, delete `./certs` and restart to mint a new one. To use a CA-issued certificate, put `fullchain.pem` and `privkey.pem` in `./certs`.
- **Ports:** in the HTTPS stack, 7860 is the app's plain HTTP listener and is published on `127.0.0.1` only. `https://<host>:7860` fails with `SSL_ERROR_RX_RECORD_TOO_LONG` because nothing there speaks TLS. Change either port with `HTTPS_PORT` and `PORT` in `.env`.
- **First start:** the text encoders, VAE and image encoder download before the UI comes up, which takes a while. Each model family downloads the first time you use it. Follow progress with `docker compose logs -f app`.
- **Storage:** weights download into the `models` volume. Generated videos and metadata go to the `outputs` volume. Settings, presets, the queue and uploaded inputs live in the `state` volume. Both stacks share these volumes.
- **LoRAs:** put `.safetensors` or `.pt` files in `./loras` (or the folder `LORAS_DIR` names) and restart the app to list them. Create the folder before the first start, as shown above, so it is owned by you rather than root.
- **Settings:** `FRAMEPACK_SERVE_ARGS` passes extra `studio.py` flags, for example `--offline` once everything is downloaded. `HF_TOKEN` speeds up downloads. FFmpeg is installed in the image, so it is never downloaded at runtime.

The image runs on its own too: `docker build -t poly-framepack-studio .`, then `docker run --gpus all -p 7860:7860 -v fps-cache:/data/cache -v fps-outputs:/data/outputs poly-framepack-studio`.

## Where files go

Everything the app writes lives under two folders, which both default to the project folder:

| Variable | Holds |
|---|---|
| `FRAMEPACK_DATA_DIR` | `.framepack/` (settings and presets), `outputs/`, `loras/`, `queue.json`, `queue_images/`, `input_files/`, `temp/` |
| `FRAMEPACK_CACHE_DIR` | `hf_download/` (Hugging Face models), `model_esrgan/`, `model_gfpgan/`, `model_rife/`, `ffmpeg/` |

The Docker image sets them to `/data` and `/data/cache`. The output, metadata, LoRA and temp folders can also be changed in the **Settings** tab.

## Look and feel

The default **poly** theme uses poly-wan's palette, type and shapes: a top bar with the queue and RAM/VRAM/GPU stats, folder tabs, segmented mode buttons, flat cards and an accent progress bar. It follows the browser's light or dark mode. The stock Gradio themes are still available under **Settings → Theme** (takes effect after a restart).

## Credits

Based on [FramePack Studio](https://github.com/FP-Studio/framepack-studio) and its contributors, under the Apache 2.0 license (see `LICENSE`).

Many thanks to [Lvmin Zhang](https://github.com/lllyasviel) for the original [FramePack](https://github.com/lllyasviel/FramePack) code, to [Rickard Edén](https://github.com/neph1) for the LoRA code, and to [Zehong Ma](https://github.com/Zehong-Ma) for [MagCache](https://github.com/Zehong-Ma/MagCache): Fast Video Generation with Magnitude-Aware Cache.

    @article{zhang2025framepack,
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        journal={Arxiv},
        year={2025}
    }

    @misc{zhang2025packinginputframecontext,
        title={Packing Input Frame Context in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        year={2025},
        eprint={2504.12626},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2504.12626}
    }
