"""HTTP API for poly-framepack-studio, served at /api next to the Gradio UI.

Every operation is a job: ``POST`` returns the job immediately (``202``) and clients poll
``GET /api/jobs/{id}``; pass ``wait=true`` to block until it finishes instead. Images and videos
are sent as multipart file fields, so the API is easy to drive with ``curl -F``.

Generation jobs go into the same queue as the UI's "Add to Queue", so they show up in the Queue
tab and share the GPU with it. Post-processing jobs run one at a time on their own worker.
Interactive docs are at /api/docs.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import random
import re
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from modules import DUMMY_LORA_NAME, paths
from modules.api_jobs import (MEDIA_TYPES, TERMINAL, ToolResult, ToolWorker, describe_queue_job, encode_preview,
                              queue_files, queue_status)
from modules.version import APP_NAME, APP_VERSION

# ---------------------------------------------------------------------------- choices

GENERATE_MODELS = {"original": "Original", "f1": "F1"}
EXTEND_MODELS = {"original": "Video", "f1": "Video F1"}
LATENT_TYPES = {"noise": "Noise", "white": "White", "black": "Black", "green": "Green Screen"}
CACHES = ("magcache", "teacache", "none")

# The FramePack checkpoints (one per model family) and the parts every family shares.
MODEL_REPOS = {
    "original": ("lllyasviel/FramePackI2V_HY", "Original, Original with Endframe, Video"),
    "f1": ("lllyasviel/FramePack_F1_I2V_HY_20250503", "F1, Video F1"),
    "hunyuan": ("hunyuanvideo-community/HunyuanVideo", "Text encoders, tokenizers and VAE (all types)"),
    "flux_redux": ("lllyasviel/flux_redux_bfl", "SigLIP image encoder (all types)"),
}

# name: (default, minimum, maximum, description)
OPTIONS = {
    "seconds": (6.0, 1.0, 120.0, "Video length in seconds"),
    "width": (640, 128, 768, "Width; the nearest supported bucket is used (with an image, its aspect ratio wins)"),
    "height": (640, 128, 768, "Height; the nearest supported bucket is used"),
    "steps": (25, 1, 100, "Sampling steps"),
    "cfg": (1.0, 1.0, 3.0, "Real CFG scale; above 1 roughly doubles the time"),
    "distilled_cfg": (10.0, 1.0, 32.0, "Distilled CFG scale"),
    "cfg_rescale": (0.0, 0.0, 1.0, "CFG re-scale"),
    "latent_window_size": (9, 1, 33, "Latent window size (experimental)"),
    "blend_sections": (4, 0, 10, "Sections to blend between timestamped prompts"),
    "magcache_threshold": (0.1, 0.01, 1.0, "MagCache error tolerance (lower = more estimated steps)"),
    "magcache_max_skips": (2, 1, 5, "MagCache: estimated steps allowed in a row"),
    "magcache_retention": (0.25, 0.0, 1.0, "MagCache: share of early steps never estimated"),
    "teacache_steps": (25, 1, 50, "TeaCache steps"),
    "teacache_threshold": (0.15, 0.01, 1.0, "TeaCache relative L1 threshold"),
    "end_frame_strength": (1.0, 0.05, 1.0, "How strongly the end frame guides the video"),
    "context_frames": (5, 2, 10, "extend: frames of the input video used as context"),
}
INT_OPTIONS = {"width", "height", "steps", "latent_window_size", "blend_sections", "magcache_max_skips",
               "teacache_steps", "context_frames"}

# Saved UI preset keys -> API option names.
PRESET_KEYS = {
    "prompt": "prompt", "n_prompt": "negative_prompt", "blend_sections": "blend_sections", "steps": "steps",
    "total_second_length": "seconds", "resolutionW": "width", "resolutionH": "height", "seed": "seed",
    "gs": "distilled_cfg", "cfg": "cfg", "rs": "cfg_rescale", "latent_window_size": "latent_window_size",
    "cache_type": "cache", "teacache_num_steps": "teacache_steps", "teacache_rel_l1_thresh": "teacache_threshold",
    "magcache_threshold": "magcache_threshold", "magcache_max_consecutive_skips": "magcache_max_skips",
    "magcache_retention_ratio": "magcache_retention", "latent_type": "latent_type",
    "end_frame_strength_original": "end_frame_strength", "combine_with_source": "combine_with_source",
    "num_cleaned_frames": "context_frames", "lora_values": "loras",
}

INTERPOLATION = {"none": "No Interpolation", "2x": "2x Frames", "4x": "4x Frames"}
LOOP_TYPES = ("loop", "ping-pong")
EXPORT_FORMATS = {"mp4": "MP4", "webm": "WebM", "gif": "GIF"}
TILE_SIZES = (0, 256, 512)
FILTERS = {  # name: (default, minimum, maximum)
    "brightness": (0, -100, 100), "contrast": (1, 0, 3), "saturation": (1, 0, 3), "temperature": (0, -100, 100),
    "sharpen": (0, 0, 5), "blur": (0, 0, 5), "denoise": (0, 0, 10), "vignette": (0, 0, 100),
    "s_curve_contrast": (0, 0, 100), "film_grain_strength": (0, 0, 50),
}
POSTPROCESS_OPS = ("upscale", "interpolate", "filters", "loop", "export")


def _bad(message: str):
    raise HTTPException(400, message)


def _read_image(upload: UploadFile, name: str) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(upload.file.read()))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        _bad(f"'{name}' is not a readable image: {e}")
    return np.array(ImageOps.exif_transpose(img).convert("RGB"))


def _image(upload: UploadFile | None, name: str) -> np.ndarray | None:
    return _read_image(upload, name) if upload is not None and upload.filename else None


def _json(text: str | None, name: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError as e:
        _bad(f"'{name}' is not valid JSON: {e}")


def _number(name: str, value, lo, hi, integer: bool):
    try:
        value = int(value) if integer else float(value)
    except (TypeError, ValueError):
        _bad(f"{name} must be a number")
    if not lo <= value <= hi:
        _bad(f"{name} must be between {lo} and {hi}")
    return value


def generation_options(
    prompt: str | None = Form(None, description="Prompt; timestamps give sections their own prompt, e.g. '[1s: waves] [3s: jumps]'"),
    negative_prompt: str | None = Form(None),
    seed: int | None = Form(None, description="Random seed; omit or -1 for a random one"),
    seconds: float | None = Form(None, description=OPTIONS["seconds"][3]),
    width: int | None = Form(None, description=OPTIONS["width"][3]),
    height: int | None = Form(None, description=OPTIONS["height"][3]),
    steps: int | None = Form(None),
    cfg: float | None = Form(None, description=OPTIONS["cfg"][3]),
    distilled_cfg: float | None = Form(None),
    cfg_rescale: float | None = Form(None),
    latent_window_size: int | None = Form(None),
    blend_sections: int | None = Form(None, description=OPTIONS["blend_sections"][3]),
    cache: str | None = Form(None, description="magcache, teacache or none"),
    magcache_threshold: float | None = Form(None),
    magcache_max_skips: int | None = Form(None),
    magcache_retention: float | None = Form(None),
    teacache_steps: int | None = Form(None),
    teacache_threshold: float | None = Form(None),
    loras: str | None = Form(None, description='JSON {"name": weight, ...} (or a list of names, weight 1.0); see /api/loras'),
    preset: str | None = Form(None, description="A preset saved in the UI for this generation type; explicit fields override it"),
    save_metadata: bool = Form(True, description="Write the metadata JSON and start image next to the video"),
) -> dict:
    """The options shared by every generation endpoint. Only the fields a client sent are kept, so a
    preset can fill in the rest before the defaults do."""
    given = dict(locals())
    return {k: v for k, v in given.items() if v is not None}


def create_api(
    job_queue,
    submit_generation: Callable[..., str],
    settings,
    lora_names: list[str],
    toolbox=None,
    enhance_prompt: Callable[[str], str] | None = None,
    caption_image: Callable[[np.ndarray], str] | None = None,
    loaded_model: Callable[[], str | None] = lambda: None,
) -> FastAPI:
    """Build the API app; studio.py mounts it at /api on the Gradio server.

    ``submit_generation`` is studio.add_generation_job, ``job_queue`` the studio's VideoJobQueue and
    ``toolbox`` its post-processing VideoProcessor (post-processing endpoints are off without one).
    """
    app = FastAPI(title=APP_NAME, version=APP_VERSION, docs_url="/docs", redoc_url=None,
                  description="FramePack video generation (Original, F1, endframes and video extension), the queue, "
                              "and post-processing: upscaling, frame interpolation, filters, loops, joins and export.")
    tools = ToolWorker(toolbox) if toolbox is not None else None
    uploads = Path(settings.get("gradio_temp_dir") or paths.TEMP_DIR) / "api"
    presets_file = paths.SETTINGS_DIR / "generation_presets.json"
    app.state.tools = tools

    # ------------------------------------------------------------------ helpers

    def available_loras() -> list[str]:
        return [n for n in lora_names if n != DUMMY_LORA_NAME]

    def load_presets() -> dict:
        try:
            with open(presets_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    async def save_upload(upload: UploadFile | None, name: str, required: bool = True) -> str | None:
        """Store an uploaded video where the UI keeps its uploads (cleaned with the temp files)."""
        if upload is None or not upload.filename:
            if required:
                _bad(f"'{name}' (a video file) is required")
            return None
        uploads.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(upload.filename).stem)[:60] or "video"
        path = uploads / f"{stem}_{uuid.uuid4().hex[:8]}{Path(upload.filename).suffix.lower() or '.mp4'}"
        data = await upload.read()
        if not data:
            _bad(f"'{name}' is empty")
        path.write_bytes(data)
        return str(path)

    def source_video(job_id: str) -> str:
        """The video of a finished generation or post-processing job, to process it further."""
        job = job_queue.get_job(job_id)
        if job is not None:
            if queue_status(job) != "done" or not job.result or not os.path.isfile(job.result):
                _bad(f"Job {job_id} has no finished video")
            return job.result
        tool_job = tools.get(job_id) if tools else None
        if tool_job is None:
            raise HTTPException(404, f"No such job: {job_id}")
        if tool_job.status != "done" or not tool_job.videos:
            _bad(f"Job {job_id} has no finished video")
        return tool_job.videos[0]

    def resolve(model_type: str, given: dict) -> dict:
        """Merge explicit fields over the preset over the defaults, and validate."""
        values = {k: v[0] for k, v in OPTIONS.items()} | {
            "negative_prompt": "", "cache": "magcache", "latent_type": "noise", "combine_with_source": True, "loras": {},
        }
        preset_name = given.pop("preset", None)
        randomize = False
        if preset_name:
            preset = load_presets().get(model_type, {}).get(preset_name)
            if preset is None:
                _bad(f"No '{model_type}' preset named '{preset_name}' (see /api/presets)")
            for key, value in preset.items():
                if key in PRESET_KEYS and value is not None:
                    values[PRESET_KEYS[key]] = value
            randomize = bool(preset.get("randomize_seed"))
            if isinstance(values.get("cache"), str):
                values["cache"] = values["cache"].lower()
            if isinstance(values.get("latent_type"), str):
                ui_to_api = {v.lower(): k for k, v in LATENT_TYPES.items()}
                values["latent_type"] = ui_to_api.get(values["latent_type"].lower(), values["latent_type"])
        values |= given
        if randomize and "seed" not in given:
            values["seed"] = None

        if not (values.get("prompt") or "").strip():
            _bad("'prompt' is required (or a preset with one)")
        for name, (_, lo, hi, _) in OPTIONS.items():
            values[name] = _number(name, values[name], lo, hi, name in INT_OPTIONS)
        if values["cache"] not in CACHES:
            _bad(f"cache must be one of {', '.join(CACHES)}")
        if values["latent_type"] not in LATENT_TYPES:
            _bad(f"latent_type must be one of {', '.join(LATENT_TYPES)}")
        seed = values.get("seed")
        values["seed"] = random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)

        loras = values["loras"]
        if isinstance(loras, str):
            loras = _json(loras, "loras")
        if isinstance(loras, list):
            loras = {name: 1.0 for name in loras}
        if not isinstance(loras, dict):
            _bad("'loras' must be a JSON object of name: weight")
        unknown = [n for n in loras if n not in available_loras()]
        if unknown:
            _bad(f"Unknown LoRA(s): {', '.join(unknown)}. Available: {', '.join(available_loras()) or 'none'}")
        values["loras"] = {n: _number(f"LoRA weight for {n}", w, 0.0, 2.0, False) for n, w in loras.items()}
        return values

    def submit(model_type: str, values: dict, input_image=None, end_frame=None, video_path: str | None = None) -> str:
        weights = values["loras"]
        return submit_generation(
            model_type=model_type,
            input_image=video_path if video_path else input_image,
            end_frame_image=end_frame,
            end_frame_strength=values["end_frame_strength"],
            prompt_text=values["prompt"],
            n_prompt=values["negative_prompt"],
            seed=values["seed"],
            total_second_length=values["seconds"],
            latent_window_size=values["latent_window_size"],
            steps=values["steps"],
            cfg=values["cfg"],
            gs=values["distilled_cfg"],
            rs=values["cfg_rescale"],
            use_teacache=values["cache"] == "teacache",
            teacache_num_steps=values["teacache_steps"],
            teacache_rel_l1_thresh=values["teacache_threshold"],
            use_magcache=values["cache"] == "magcache",
            magcache_threshold=values["magcache_threshold"],
            magcache_max_consecutive_skips=values["magcache_max_skips"],
            magcache_retention_ratio=values["magcache_retention"],
            blend_sections=values["blend_sections"],
            latent_type=LATENT_TYPES[values["latent_type"]],
            clean_up_videos=settings.get("clean_up_videos", True),
            selected_loras=list(weights),
            resolutionW=values["width"],
            resolutionH=values["height"],
            input_image_path=video_path,
            combine_with_source=bool(values["combine_with_source"]),
            num_cleaned_frames=values["context_frames"],
            # The worker matches weights to names by position, over every LoRA the UI knows.
            lora_loaded_names=list(lora_names),
            lora_values=[weights.get(n, 1.0) for n in lora_names],
            save_metadata_checked=values.get("save_metadata", True),
        )

    def describe(job_id: str) -> dict:
        job = job_queue.get_job(job_id)
        if job is not None:
            return describe_queue_job(job, job_queue, settings)
        tool_job = tools.get(job_id) if tools else None
        if tool_job is None:
            raise HTTPException(404, "No such job")
        return tool_job.to_dict(tools.queue_position(tool_job))

    async def respond(job_ids: list[str], wait: bool, many: bool = False) -> JSONResponse:
        if wait:
            while not all(describe(j)["status"] in TERMINAL for j in job_ids):
                await asyncio.sleep(0.5)
        bodies = [describe(j) for j in job_ids]
        done = all(b["status"] in TERMINAL for b in bodies)
        return JSONResponse({"jobs": bodies} if many else bodies[0], status_code=200 if done else 202)

    def need_tools():
        if tools is None:
            raise HTTPException(503, "Post-processing is not available in this server")

    # ------------------------------------------------------------------ meta

    @app.get("/health")
    def health():
        jobs = job_queue.get_all_jobs()
        counts = {s: sum(queue_status(j) == s for j in jobs) for s in ("queued", "running")}
        return {"ok": True, "version": APP_VERSION, "loaded": loaded_model(), "queue": counts}

    @app.get("/options", summary="Choices, defaults and ranges for the operation fields")
    def options():
        upscale_models = {}
        if toolbox is not None:
            for key, m in toolbox.esrgan_upscaler.supported_models.items():
                upscale_models[key] = {"scale": m.get("scale"), "description": m.get("description", "")}
        return {
            "generate_models": list(GENERATE_MODELS), "extend_models": list(EXTEND_MODELS),
            "latent_types": list(LATENT_TYPES), "caches": list(CACHES),
            "options": {k: {"default": v[0], "min": v[1], "max": v[2], "description": v[3]} for k, v in OPTIONS.items()},
            "postprocess": {
                "operations": list(POSTPROCESS_OPS), "upscale_models": upscale_models, "tile_sizes": list(TILE_SIZES),
                "interpolation": list(INTERPOLATION), "loop_types": list(LOOP_TYPES),
                "export_formats": list(EXPORT_FORMATS),
                "filters": {k: {"default": v[0], "min": v[1], "max": v[2]} for k, v in FILTERS.items()},
            },
        }

    @app.get("/loras", summary="LoRAs found in the LoRA folder at startup")
    def loras():
        return {"folder": settings.get("lora_dir"), "loras": available_loras()}

    @app.get("/presets", summary="Generation presets saved in the UI, by generation type")
    def presets():
        return {model_type: sorted(entries) for model_type, entries in load_presets().items() if isinstance(entries, dict)}

    @app.get("/models", summary="The model checkpoints and whether they are downloaded")
    async def models():
        def downloaded(repo: str) -> bool:
            snapshots = paths.HF_HOME / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
            return snapshots.is_dir() and any(snapshots.iterdir())

        rows = []
        for name, (repo, used_by) in MODEL_REPOS.items():
            rows.append({"name": name, "repo": repo, "used_by": used_by,
                         "downloaded": await asyncio.to_thread(downloaded, repo)})
        return {"loaded": loaded_model(), "cache": str(paths.HF_HOME), "models": rows}

    # ------------------------------------------------------------------ generation

    @app.post("/generate", status_code=202, summary="Image-to-video (or text-to-video) with Original or F1")
    async def generate(
        model: str = Form("original", description="original or f1"),
        image: UploadFile | None = File(None, description="Start frame; without one the video starts from latent_type"),
        end_frame: UploadFile | None = File(None, description="Optional end frame (original only: 'Original with Endframe')"),
        end_frame_strength: float | None = Form(None, description=OPTIONS["end_frame_strength"][3]),
        latent_type: str | None = Form(None, description="Start without an image from: noise, white, black or green"),
        wait: bool = Form(False), opts: dict = Depends(generation_options),
    ):
        model = model.lower()
        if model not in GENERATE_MODELS:
            _bad(f"model must be one of {', '.join(GENERATE_MODELS)}")
        img, end = _image(image, "image"), _image(end_frame, "end_frame")
        if end is not None and model != "original":
            _bad("end_frame needs model=original")
        model_type = "Original with Endframe" if end is not None else GENERATE_MODELS[model]
        given = opts | {k: v for k, v in (("end_frame_strength", end_frame_strength), ("latent_type", latent_type)) if v is not None}
        values = resolve(model_type, given)
        job_id = await asyncio.to_thread(submit, model_type, values, img, end)
        return await respond([job_id], wait)

    @app.post("/extend", status_code=202, summary="Extend a video (Video / Video F1), optionally towards an end frame")
    async def extend(
        video: UploadFile | None = File(None, description="The video to continue"),
        source_job: str | None = Form(None, description="Or: continue the video of a finished job"),
        model: str = Form("original", description="original (Video) or f1 (Video F1)"),
        end_frame: UploadFile | None = File(None, description="Optional end frame (original only: 'Video with Endframe')"),
        end_frame_strength: float | None = Form(None),
        combine_with_source: bool | None = Form(None, description="Join the result onto the source video (default true)"),
        context_frames: int | None = Form(None, description=OPTIONS["context_frames"][3]),
        wait: bool = Form(False), opts: dict = Depends(generation_options),
    ):
        model = model.lower()
        if model not in EXTEND_MODELS:
            _bad(f"model must be one of {', '.join(EXTEND_MODELS)}")
        end = _image(end_frame, "end_frame")
        if end is not None and model != "original":
            _bad("end_frame needs model=original")
        model_type = EXTEND_MODELS[model]
        given = opts | {k: v for k, v in (("end_frame_strength", end_frame_strength),
                                          ("combine_with_source", combine_with_source),
                                          ("context_frames", context_frames)) if v is not None}
        # Presets are saved per UI type, and the UI calls the endframe variant its own type.
        values = resolve("Video with Endframe" if end is not None else model_type, given)
        path = source_video(source_job) if source_job and not (video and video.filename) else await save_upload(video, "video")
        job_id = await asyncio.to_thread(submit, model_type, values, None, end, path)
        return await respond([job_id], wait)

    @app.post("/batch", status_code=202, summary="One image-to-video job per image, with the same settings")
    async def batch(
        images: list[UploadFile] = File(..., description="Start frames, one job each"),
        model: str = Form("original", description="original or f1"),
        random_seeds: bool = Form(False, description="A new random seed per image (default: seed, seed+1, ...)"),
        wait: bool = Form(False), opts: dict = Depends(generation_options),
    ):
        model = model.lower()
        if model not in GENERATE_MODELS:
            _bad(f"model must be one of {', '.join(GENERATE_MODELS)}")
        frames = [_read_image(u, "images") for u in images if u.filename]
        if not frames:
            _bad("'images' needs at least one image")
        model_type = GENERATE_MODELS[model]
        values = resolve(model_type, opts)
        job_ids = []
        for i, img in enumerate(frames):
            seed = random.randint(0, 2**31 - 1) if random_seeds and i else values["seed"] + (0 if random_seeds else i)
            job_ids.append(await asyncio.to_thread(submit, model_type, values | {"seed": seed}, img))
        return await respond(job_ids, wait, many=True)

    # ------------------------------------------------------------------ prompt helpers

    @app.post("/enhance", summary="Rewrite a prompt with more visual detail (a small local LLM)")
    async def enhance(prompt: str = Form(...)):
        if enhance_prompt is None:
            raise HTTPException(503, "The prompt enhancer is not available in this server")
        return {"prompt": await asyncio.to_thread(enhance_prompt, prompt)}

    @app.post("/caption", summary="Describe an image as a prompt (a small local vision model)")
    async def caption(image: UploadFile = File(...)):
        if caption_image is None:
            raise HTTPException(503, "The captioner is not available in this server")
        return {"prompt": await asyncio.to_thread(caption_image, _read_image(image, "image"))}

    # ------------------------------------------------------------------ post-processing

    def op_upscale(p: dict) -> dict:
        models_ = toolbox.esrgan_upscaler.supported_models
        key = p.get("model") or next(iter(models_))
        if key not in models_:
            _bad(f"upscale model must be one of {', '.join(models_)}")
        tile = _number("tile", p.get("tile", 0), 0, 512, True)
        if tile not in TILE_SIZES:
            _bad(f"tile must be one of {', '.join(map(str, TILE_SIZES))}")
        native = float(models_[key].get("scale") or 4)
        return {"model_key": key,
                "output_scale_factor_ui": _number("scale", p.get("scale", native), 1.0, native, False),
                "tile_size": tile, "enhance_face": bool(p.get("enhance_face", False)),
                "denoise_strength_ui": _number("denoise", p.get("denoise", 0.5), 0.0, 1.0, False),
                "use_streaming": bool(p.get("streaming", False))}

    def op_interpolate(p: dict) -> dict:
        mode = str(p.get("interpolation", "none")).lower()
        if mode not in INTERPOLATION:
            _bad(f"interpolation must be one of {', '.join(INTERPOLATION)}")
        return {"target_fps_mode": INTERPOLATION[mode], "speed_factor": _number("speed", p.get("speed", 1.0), 0.25, 4.0, False),
                "use_streaming": bool(p.get("streaming", False))}

    def op_filters(p: dict) -> dict:
        unknown = [k for k in p if k not in FILTERS]
        if unknown:
            _bad(f"Unknown filter(s): {', '.join(unknown)}")
        return {k: _number(k, p.get(k, d), lo, hi, False) for k, (d, lo, hi) in FILTERS.items()}

    def op_loop(p: dict) -> dict:
        kind = p.get("type", "loop")
        if kind not in LOOP_TYPES:
            _bad(f"loop type must be one of {', '.join(LOOP_TYPES)}")
        return {"loop_type": kind, "num_loops": _number("loops", p.get("loops", 1), 1, 10, True)}

    def op_export(p: dict) -> dict:
        fmt = str(p.get("format", "mp4")).lower()
        if fmt not in EXPORT_FORMATS:
            _bad(f"format must be one of {', '.join(EXPORT_FORMATS)}")
        name = p.get("name") or None
        return {"export_format": EXPORT_FORMATS[fmt], "quality_slider": _number("quality", p.get("quality", 85), 0, 100, True),
                "max_width": _number("max_width", p.get("max_width", 1024), 256, 2048, True),
                "output_base_name_override": re.sub(r"[^A-Za-z0-9_-]+", "_", name) if name else None}

    # API op name -> (toolbox pipeline op name, parameter builder)
    OPS = {"upscale": ("upscale", op_upscale), "interpolate": ("frame_adjust", op_interpolate),
           "filters": ("filters", op_filters), "loop": ("loop", op_loop), "export": ("export", op_export)}

    def pipeline(operations) -> list[dict]:
        if not isinstance(operations, list) or not operations:
            _bad("'operations' must be a non-empty JSON list")
        steps = []
        for i, op in enumerate(operations):
            if not isinstance(op, dict) or op.get("op") not in OPS:
                _bad(f"operations[{i}] needs \"op\": one of {', '.join(OPS)}")
            name, build = OPS[op["op"]]
            steps.append({"name": name, "params": build({k: v for k, v in op.items() if k != "op"})})
        return steps

    async def run_pipeline(kind: str, steps: list[dict], video: UploadFile | None, source_job: str | None, wait: bool, params: dict):
        need_tools()
        path = source_video(source_job) if source_job and not (video and video.filename) else await save_upload(video, "video")

        def run(progress):
            out = toolbox.tb_process_video_batch([path], {"operations": steps}, progress=progress)
            return ToolResult([str(out)] if out and os.path.isfile(out) else [])

        job = tools.submit(kind, params | {"source": os.path.basename(path)}, run)
        return await respond([job.id], wait)

    @app.post("/postprocess", status_code=202, summary="Run a chain of post-processing operations on a video")
    async def postprocess(
        operations: str = Form(..., description='JSON list, e.g. [{"op": "upscale", "model": "RealESRGAN_x2plus"}, {"op": "export", "format": "gif"}]'),
        video: UploadFile | None = File(None), source_job: str | None = Form(None, description="Or: the video of a finished job"),
        wait: bool = Form(False),
    ):
        need_tools()
        ops = _json(operations, "operations")
        return await run_pipeline("postprocess", pipeline(ops), video, source_job, wait, {"operations": ops})

    @app.post("/upscale", status_code=202, summary="Upscale a video with Real-ESRGAN, optionally restoring faces (GFPGAN)")
    async def upscale(
        video: UploadFile | None = File(None), source_job: str | None = Form(None),
        model: str | None = Form(None, description="See /api/options; default the first model"),
        scale: float | None = Form(None, description="Output scale, up to the model's native scale"),
        tile: int = Form(0, description="0 (no tiling), 256 or 512; tiles use less VRAM"),
        enhance_face: bool = Form(False), denoise: float = Form(0.5, description="RealESR-general-x4v3 only, 0-1"),
        streaming: bool = Form(False, description="Low-memory mode for long or large videos"),
        wait: bool = Form(False),
    ):
        need_tools()
        p = {k: v for k, v in dict(model=model, scale=scale, tile=tile, enhance_face=enhance_face, denoise=denoise,
                                   streaming=streaming).items() if v is not None}
        return await run_pipeline("upscale", [{"name": "upscale", "params": op_upscale(p)}], video, source_job, wait, p)

    @app.post("/interpolate", status_code=202, summary="RIFE frame interpolation and speed change")
    async def interpolate(
        video: UploadFile | None = File(None), source_job: str | None = Form(None),
        interpolation: str = Form("2x", description="none, 2x or 4x frames"),
        speed: float = Form(1.0, description="0.25-4; below 1 slows the video down (ignored when streaming)"),
        streaming: bool = Form(False), wait: bool = Form(False),
    ):
        need_tools()
        p = {"interpolation": interpolation, "speed": speed, "streaming": streaming}
        return await run_pipeline("interpolate", [{"name": "frame_adjust", "params": op_interpolate(p)}], video, source_job, wait, p)

    @app.post("/filters", status_code=202, summary="Colour and detail filters (FFmpeg)")
    async def filters(
        video: UploadFile | None = File(None), source_job: str | None = Form(None),
        values: str = Form(..., description='JSON, e.g. {"brightness": 10, "saturation": 1.2, "vignette": 30}; see /api/options'),
        wait: bool = Form(False),
    ):
        need_tools()
        p = _json(values, "values")
        if not isinstance(p, dict):
            _bad("'values' must be a JSON object")
        return await run_pipeline("filters", [{"name": "filters", "params": op_filters(p)}], video, source_job, wait, p)

    @app.post("/loop", status_code=202, summary="Repeat a video, as a loop or ping-pong")
    async def loop(
        video: UploadFile | None = File(None), source_job: str | None = Form(None),
        type: str = Form("loop", description="loop or ping-pong"), loops: int = Form(1, description="Extra repeats, 1-10"),
        wait: bool = Form(False),
    ):
        need_tools()
        p = {"type": type, "loops": loops}
        return await run_pipeline("loop", [{"name": "loop", "params": op_loop(p)}], video, source_job, wait, p)

    @app.post("/export", status_code=202, summary="Convert to MP4, WebM or GIF, with a quality and a maximum width")
    async def export(
        video: UploadFile | None = File(None), source_job: str | None = Form(None),
        format: str = Form("mp4", description="mp4, webm or gif"), quality: int = Form(85, description="0-100"),
        max_width: int = Form(1024, description="256-2048; keeps the aspect ratio"),
        name: str | None = Form(None, description="Output file name (optional)"),
        wait: bool = Form(False),
    ):
        need_tools()
        p = {k: v for k, v in dict(format=format, quality=quality, max_width=max_width, name=name).items() if v is not None}
        return await run_pipeline("export", [{"name": "export", "params": op_export(p)}], video, source_job, wait, p)

    @app.post("/join", status_code=202, summary="Join two or more videos end to end")
    async def join(
        videos: list[UploadFile] = File(default=[], description="Videos in order"),
        source_jobs: str | None = Form(None, description='Or/also: JSON list of job IDs whose videos to append'),
        name: str | None = Form(None), wait: bool = Form(False),
    ):
        need_tools()
        files = [await save_upload(v, "videos") for v in videos if v.filename]
        job_list = _json(source_jobs, "source_jobs") or []
        if not isinstance(job_list, list):
            _bad("'source_jobs' must be a JSON list")
        files += [source_video(j) for j in job_list]
        if len(files) < 2:
            _bad("Joining needs at least two videos")
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", name) if name else None

        def run(progress):
            out = toolbox.tb_join_videos(files, base, progress=progress)
            return ToolResult([str(out)] if out and os.path.isfile(out) else [])

        job = tools.submit("join", {"videos": [os.path.basename(f) for f in files], "name": base}, run)
        return await respond([job.id], wait)

    @app.post("/analyze", summary="Report a video's format, size, frame rate, length and audio")
    async def analyze(video: UploadFile = File(...)):
        need_tools()
        path = await save_upload(video, "video")
        report = await asyncio.to_thread(toolbox.tb_analyze_video_input, path)
        return {"report": report}

    # ------------------------------------------------------------------ jobs

    @app.get("/jobs", summary="Generation and post-processing jobs, newest first")
    def list_jobs(limit: int = 50):
        rows = [describe_queue_job(j, job_queue, settings) for j in job_queue.get_all_jobs()]
        if tools:
            rows += [j.to_dict(tools.queue_position(j)) for j in tools.list()]
        return sorted(rows, key=lambda r: r["created"] or 0, reverse=True)[:limit]

    @app.delete("/jobs", summary="Cancel every queued job (running jobs keep going)")
    def cancel_queued():
        cancelled = job_queue.clear_queue()
        if tools:
            cancelled += tools.cancel_queued()
        return {"cancelled": cancelled}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str, wait: bool = False):
        describe(job_id)  # 404 before waiting
        return await respond([job_id], wait)

    @app.delete("/jobs/{job_id}", summary="Cancel a queued or running job")
    def cancel_job(job_id: str):
        if job_queue.get_job(job_id) is not None:
            job_queue.cancel_job(job_id)
        elif tools and tools.get(job_id):
            tools.cancel(job_id)
        else:
            raise HTTPException(404, "No such job")
        return describe(job_id)

    @app.get("/jobs/{job_id}/preview", summary="Latest latent preview of a running generation (PNG)")
    def job_preview(job_id: str):
        job = job_queue.get_job(job_id)
        if job is None:
            describe(job_id)
            raise HTTPException(404, "Only generation jobs have previews")
        preview = (job.progress_data or {}).get("preview")
        if preview is None or queue_status(job) != "running":
            raise HTTPException(404, "No preview yet")
        return Response(encode_preview(preview), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/jobs/{job_id}/videos/{index}", summary="A result video (while a generation runs: the video so far)")
    def job_video(job_id: str, index: int, download: bool = False):
        job = job_queue.get_job(job_id)
        if job is not None:
            videos = [job.result] if job.result and os.path.isfile(job.result) else []
            kind = describe_queue_job(job, job_queue, settings)["kind"]
        else:
            tool_job = tools.get(job_id) if tools else None
            if tool_job is None:
                raise HTTPException(404, "No such job")
            videos, kind = tool_job.videos, tool_job.kind
        if not 0 <= index < len(videos):
            raise HTTPException(404, "No such video")
        path = videos[index]
        ext = os.path.splitext(path)[1].lower()
        return FileResponse(path, media_type=MEDIA_TYPES.get(ext, "application/octet-stream"),
                            filename=f"{kind}-{job_id}-{index}{ext}" if download else None)

    @app.get("/jobs/{job_id}/files/{name}", summary="A side output: 'metadata' (JSON) or 'image' (the start frame)")
    def job_file(job_id: str, name: str, download: bool = False):
        job = job_queue.get_job(job_id)
        if job is None:
            describe(job_id)
            raise HTTPException(404, "No such file")
        files = queue_files(job, settings)
        if name not in files:
            raise HTTPException(404, "No such file")
        path = files[name]
        ext = os.path.splitext(path)[1].lower()
        return FileResponse(path, media_type=MEDIA_TYPES.get(ext, "application/octet-stream"),
                            filename=os.path.basename(path) if download else None)

    return app
