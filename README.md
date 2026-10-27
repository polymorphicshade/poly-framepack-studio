# poly-framepack-studio

A video generation studio for [FramePack](https://github.com/lllyasviel/FramePack) (HunyuanVideo), with a Gradio web UI styled like [poly-wan](https://github.com/polymorphicshade/poly-wan). It is based on [FramePack Studio](https://github.com/FP-Studio/framepack-studio) and adds poly-wan's look, an HTTP API for every feature, and poly-wan's Docker deployment, including HTTPS.

| Generation type | What it does |
|---|---|
| **Original** | Image-to-video (or text-to-video) with the original FramePack model. |
| **Original with Endframe** | Same, with an end frame whose influence you set. |
| **F1** | The FramePack F1 model, which generates forwards in time. |
| **Video**, **Video with Endframe**, **Video F1** | Extends an input video, optionally towards an end frame, and can join the result onto the source. |
| **Sequence** | One video through 2 to 10 keyframes: each gap between neighbouring keyframes is its own Original with Endframe segment with its own prompt and length, and the segments are joined into one video. With **Seamless transitions** on, each segment starts from the frame the previous one actually ended on, so the joins have no jump. |

Every type runs in the same job queue, from the web UI or the [HTTP API](#http-api). Other features:

- **Timestamped prompts**: give different prompts for different times, e.g. `[1s: waves hello] [3s: jumps]`, and choose how many sections to blend between them.
- **LoRAs**: works with most HunyuanVideo LoRAs, each with its own weight.
- **Workspace tabs**: the Generate page has browser-style tabs (＋ to add, ✕ Close, type in the box to rename), each with its own generation type, prompts, settings and input images, so you can keep several ideas going and switch between them. Tabs and their settings are kept in your browser across reloads (images are not). All tabs share one job queue and one result panel.
- **Queue**: jobs run one after another while you keep working. Queues can be exported to a zip and imported again.
- **Metadata**: the prompt and seed are stored in the output PNG; all other settings go into a JSON file that can be loaded back later.
- **Presets**: named groups of settings per generation type, plus a startup model and preset.
- **Latent image options**: with no input image, start from noise, black, white or a green screen.
- **Prompt enhancer and captioner**: rewrite a prompt with an LLM, or caption the start image.
- **Post-processing**: ESRGAN upscaling with optional GFPGAN face restoration, RIFE frame interpolation, filters, frame extraction and reassembly, and export to MP4, WebM or GIF.
- **Caches**: MagCache or TeaCache speed up sampling.

## Requirements

- An NVIDIA GPU with at least 8 GB VRAM (16 GB or more recommended).
- 16 GB of system RAM (32 GB or more strongly recommended).
- 80 GB of free disk space or more: about 25 GB for each model family (Original and F1), plus the shared text encoders.
- Python 3.10, 3.11 or 3.12. Several pinned packages have no wheels for 3.13 or newer. `install.bat` finds a supported version itself, and if there is none it offers to install 3.12 with winget alongside your other versions.

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

## HTTP API

The API is served by the same server as the UI, at `/api`. Interactive docs are at `/api/docs` (`/docs` redirects there). Generation jobs go into the same queue as the UI's **Add to Queue**, so they show up in the Queue tab and share the GPU with it. Post-processing jobs run one at a time on a worker of their own.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/generate` | `prompt`, `model` (`original`/`f1`), optional `image` (start frame), `end_frame` (original only), `end_frame_strength`, `latent_type` (`noise`/`white`/`black`/`green`, used without an image) |
| POST | `/api/extend` | Video extension: `video` or `source_job`, `prompt`, `model` (`original` = Video, `f1` = Video F1), optional `end_frame`, `combine_with_source`, `context_frames` |
| POST | `/api/batch` | `images[]`, one job each with the same settings; seeds count up from `seed` unless `random_seeds` |
| POST | `/api/sequence` | `images[]` (2–10 keyframes, in order), `prompts` JSON list (one per segment; blank entries use `prompt`), `segment_seconds` JSON list (default: `seconds` each), `chain` (seamless joins, default true), `end_frame_strength` |
| POST | `/api/enhance` · `/api/caption` | `prompt` → a more detailed prompt · `image` → a prompt describing it |
| POST | `/api/postprocess` | `video` or `source_job`, `operations` JSON: a chain of `upscale`, `interpolate`, `filters`, `loop` and `export` steps |
| POST | `/api/upscale` | Real-ESRGAN: `model`, `scale`, `tile`, `enhance_face` (GFPGAN), `denoise`, `streaming` |
| POST | `/api/interpolate` | RIFE: `interpolation` (`none`/`2x`/`4x`), `speed`, `streaming` |
| POST | `/api/filters` | `values` JSON: `brightness`, `contrast`, `saturation`, `temperature`, `sharpen`, `blur`, `denoise`, `vignette`, `s_curve_contrast`, `film_grain_strength` |
| POST | `/api/loop` · `/api/export` | `type` (`loop`/`ping-pong`), `loops` · `format` (`mp4`/`webm`/`gif`), `quality`, `max_width`, `name` |
| POST | `/api/join` | `videos[]` and/or `source_jobs` JSON list, in order |
| POST | `/api/analyze` | `video`: format, size, frame rate, length and audio |
| GET | `/api/jobs` · DELETE `/api/jobs` | list jobs, newest first · cancel every queued job |
| GET / DELETE | `/api/jobs/{id}` | poll or cancel a job |
| GET | `/api/jobs/{id}/videos/0` | the result video (while a generation runs, the video so far); `?download=true` |
| GET | `/api/jobs/{id}/files/metadata` · `/files/image` | the metadata JSON and start frame of a generation |
| GET | `/api/jobs/{id}/preview` | latest latent preview (PNG) while a generation runs |
| GET | `/api/health` · `/api/options` · `/api/loras` · `/api/presets` · `/api/models` | status, field choices and ranges, LoRAs, saved presets, checkpoints and their download state |

The generation endpoints share these options: `seed` (omit or `-1` for random), `seconds`, `width`, `height`, `steps`, `cfg`, `distilled_cfg`, `cfg_rescale`, `latent_window_size`, `blend_sections`, `negative_prompt`, `cache` (`magcache`/`teacache`/`none`) with `magcache_threshold`, `magcache_max_skips`, `magcache_retention`, `teacache_steps` and `teacache_threshold`, `loras` (JSON `{"name": weight}`), `save_metadata` and `preset`. A `preset` is one saved in the UI for that generation type; the fields you send override it. Prompts can use timestamps, as in the UI.

Every operation returns a job (`202`) that you poll; send `wait=true` to block until it finishes instead. `source_job` takes the video of a finished job, so results can be chained without downloading them. A job's `status` is `queued`, `running`, `done`, `error` or `cancelled`.

```bash
curl -F prompt="[1s: the fox looks up] [3s: it runs into the snow]" -F image=@fox.png -F seconds=5 -F wait=true http://127.0.0.1:7860/api/generate
curl -F video=@clip.mp4 -F prompt="the camera keeps panning" -F model=f1 -F wait=true http://127.0.0.1:7860/api/extend
curl -F source_job=<id> -F operations='[{"op":"upscale","model":"RealESRGAN_x2plus"},{"op":"interpolate","interpolation":"2x"},{"op":"export","format":"gif","max_width":512}]' \
     -F wait=true http://127.0.0.1:7860/api/postprocess
curl -o fox.mp4 "http://127.0.0.1:7860/api/jobs/<id>/videos/0?download=true"
```

## Docker (Linux server)

There are two compose files. Both run the app on an NVIDIA GPU. The host needs Docker, the NVIDIA driver (570 or newer) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). On Ubuntu or Debian, `sudo ./NVIDIA.sh` installs all three and checks that a container can see the GPU.

**Plain HTTP** (`docker-compose.yml`):

```bash
cp .env.example .env          # optional: port, LoRA folder and serve flags
mkdir -p loras
docker compose up -d --build
docker compose logs -f app
```

The UI is then at `http://<host>:7860`, and the API at `http://<host>:7860/api`.

**HTTPS** (`docker-compose.https.yml`) runs the same app plus an nginx sidecar that terminates TLS:

```bash
cp .env.https.example .env    # list the host's LAN address in TLS_SAN
mkdir -p loras
docker compose -f docker-compose.https.yml up -d --build
```

The UI is then at `https://<host>:8443`, and the API at `https://<host>:8443/api`.

- **Certificate:** a self-signed certificate is generated into `./certs` on first boot, so the browser warns once until you accept it. Every address you type in the browser must be in `TLS_SAN`, e.g. `IP:192.168.1.99`. After changing it, delete `./certs` and restart to mint a new one. To use a CA-issued certificate, put `fullchain.pem` and `privkey.pem` in `./certs`.
- **Ports:** in the HTTPS stack, 7860 is the app's plain HTTP listener and is published on `127.0.0.1` only. `https://<host>:7860` fails with `SSL_ERROR_RX_RECORD_TOO_LONG` because nothing there speaks TLS. Change either port with `HTTPS_PORT` and `PORT` in `.env`.
- **Attention:** by default the image compiles [SageAttention 2](https://github.com/thu-ml/SageAttention), the fastest attention backend, so the first build takes a while. It runs on Ampere, Ada, Hopper and Blackwell GPUs (RTX 30-series and newer); older GPUs fall back to PyTorch's attention. The build is limited to one compiler job (`MAX_JOBS`) with two threads (`NVCC_THREADS`) because each thread can take several GB of RAM, and running out can hang the machine. To build faster, list only your own GPUs in `TORCH_CUDA_ARCH_LIST`, e.g. `8.6;8.9` for an RTX 3090 and an RTX 4070. To skip compiling altogether, set `ATTENTION=flash` for a prebuilt Flash Attention, or `ATTENTION=none`.
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

## Development

```bash
pip install -r requirements-dev.txt
pytest           # API tests against the real job queue with a fake worker: no GPU, torch or weights needed
```

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
