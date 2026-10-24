"""HTTP API tests against the real job queue with a fake worker and a fake toolbox: no GPU,
torch or model weights needed.

    pip install -r requirements-dev.txt
    pytest
"""

import atexit
import io
import json
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

# Keep settings, the queue file and outputs out of the project folder. Must run before any
# modules.* import, since modules.paths reads it once.
DATA = Path(tempfile.mkdtemp(prefix="fps-api-test-"))
os.environ["FRAMEPACK_DATA_DIR"] = str(DATA)
atexit.register(shutil.rmtree, DATA, ignore_errors=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# video_queue imports metadata_utils from modules.pipelines, whose package imports the torch
# pipelines; stand in for just the one function the queue uses.
_pipelines = types.ModuleType("modules.pipelines")
_pipelines.__path__ = []
_metadata = types.ModuleType("modules.pipelines.metadata_utils")
_metadata.create_metadata = lambda params, job_id, settings, save_placeholder=False: {}
sys.modules.setdefault("modules.pipelines", _pipelines)
sys.modules.setdefault("modules.pipelines.metadata_utils", _metadata)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from modules import DUMMY_LORA_NAME, paths  # noqa: E402
from modules.api import create_api  # noqa: E402
from modules.settings import Settings  # noqa: E402
from modules.toolbox.message_manager import MessageManager  # noqa: E402
from modules.video_queue import VideoJobQueue  # noqa: E402

OUTPUT_ID = "250101_120000_123_4567"


def png(size=(64, 48), color=(40, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def fake_worker(job_stream=None, prompt_text="", seed=0, **params):
    """Behaves like modules.pipelines.worker.worker towards the queue: progress, file, end."""
    out_dir = Path(params["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    job_stream.output_queue.push(("job_id", OUTPUT_ID))
    if "fail" in prompt_text:
        job_stream.output_queue.push(("error", "Error during generation: Traceback (most recent call last):\n  ...\nValueError: bad latent"))
        job_stream.output_queue.push(("end", None))
        return
    preview = np.full((16, 16, 3), 200, dtype=np.uint8)
    for step in (1, 2):
        html = f"<span>Sampling {step}/2</span><span>Total {step}/2</span>"
        job_stream.output_queue.push(("progress", (preview, f"Section 1, step {step}", html)))
        time.sleep(0.05)
    video = out_dir / f"{OUTPUT_ID}_33.mp4"
    video.write_bytes(b"fake mp4")
    (out_dir / f"{OUTPUT_ID}.png").write_bytes(png())
    (out_dir / f"{OUTPUT_ID}.json").write_text(json.dumps({"prompt": prompt_text, "seed": seed}))
    job_stream.output_queue.push(("file", str(video)))
    job_stream.output_queue.push(("end", None))


class FakeUpscaler:
    supported_models = {
        "RealESRGAN_x2plus": {"scale": 2, "description": "2x"},
        "RealESRGAN_x4plus": {"scale": 4, "description": "4x"},
    }


class FakeToolbox:
    """The parts of VideoProcessor the API uses."""

    def __init__(self, out_dir: Path):
        self.message_manager = MessageManager()
        self.esrgan_upscaler = FakeUpscaler()
        self.out_dir = out_dir
        self.calls = []

    def tb_process_video_batch(self, video_paths, pipeline_config, progress):
        ops = pipeline_config["operations"]
        self.calls.append(ops)
        for _ in progress.tqdm(range(3), desc="Processing"):
            pass
        if any(op["name"] == "filters" and op["params"]["blur"] == 5 for op in ops):
            self.message_manager.add_error("FFmpeg failed during filters.")
            return None
        ext = ".gif" if ops[-1]["name"] == "export" and ops[-1]["params"]["export_format"] == "GIF" else ".mp4"
        out = self.out_dir / f"{Path(video_paths[0]).stem}_{ops[-1]['name']}{ext}"
        shutil.copy(video_paths[0], out)
        return str(out)

    def tb_join_videos(self, video_paths, output_base_name_override=None, progress=None):
        out = self.out_dir / f"{output_base_name_override or 'joined'}.mp4"
        out.write_bytes(b"".join(Path(p).read_bytes() for p in video_paths))
        return str(out)

    def tb_analyze_video_input(self, video_path):
        return f"File: {Path(video_path).name}"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the queue writes queue.json to the working directory
    settings = Settings()
    settings.settings.update(output_dir=str(tmp_path / "outputs"), metadata_dir=str(tmp_path / "outputs"),
                             gradio_temp_dir=str(tmp_path / "temp"))
    job_queue = VideoJobQueue()
    job_queue.set_worker_function(fake_worker)
    submitted = []

    def submit_generation(**kw):
        submitted.append(kw)
        return job_queue.add_job({
            "model_type": kw["model_type"], "input_image": kw["input_image"], "prompt_text": kw["prompt_text"],
            "seed": kw["seed"], "output_dir": settings.get("output_dir"),
        })

    (tmp_path / "pp").mkdir()
    toolbox = FakeToolbox(tmp_path / "pp")
    api = create_api(job_queue, submit_generation, settings, ["style_a", "style_b"], toolbox=toolbox,
                     enhance_prompt=lambda p: p + ", cinematic lighting", caption_image=lambda img: f"a {img.shape[1]}x{img.shape[0]} image")
    app = FastAPI()
    app.mount("/api", api)
    return types.SimpleNamespace(client=TestClient(app), submitted=submitted, toolbox=toolbox, queue=job_queue)


def finished(env, body):
    assert body["status"] == "done", body
    return body


def test_health_options_loras_and_docs(env):
    c = env.client
    assert c.get("/api/health").json()["ok"] is True
    o = c.get("/api/options").json()
    assert o["generate_models"] == ["original", "f1"] and "RealESRGAN_x4plus" in o["postprocess"]["upscale_models"]
    assert c.get("/api/loras").json()["loras"] == ["style_a", "style_b"]
    assert c.get("/api/docs").status_code == 200
    assert "/generate" in c.get("/api/openapi.json").json()["paths"]
    names = {m["name"]: m["downloaded"] for m in c.get("/api/models").json()["models"]}
    assert names == {"original": False, "f1": False, "hunyuan": False, "flux_redux": False}


def test_generate_waits_and_serves_outputs(env):
    c = env.client
    r = c.post("/api/generate", data={"prompt": "[1s: a fox runs] [3s: it jumps]", "model": "f1", "seed": 7,
                                      "seconds": 3, "loras": '{"style_b": 0.6}', "wait": "true"},
               files={"image": ("start.png", png(), "image/png")})
    assert r.status_code == 200
    body = finished(env, r.json())
    assert body["kind"] == "generate" and body["model_type"] == "F1" and body["seed"] == 7
    assert body["progress"] == {"step": 2, "total": 2}
    kw = env.submitted[-1]
    assert kw["input_image"].shape == (48, 64, 3) and kw["total_second_length"] == 3.0
    assert kw["selected_loras"] == ["style_b"] and kw["lora_values"] == [1.0, 0.6]
    assert kw["use_magcache"] and not kw["use_teacache"] and kw["latent_type"] == "Noise"
    assert c.get(body["videos"][0]["url"]).content == b"fake mp4"
    assert c.get(body["files"]["metadata"]).json()["seed"] == 7
    assert c.get(body["files"]["image"]).headers["content-type"] == "image/png"


def test_generate_poll_and_preview(env):
    c = env.client
    r = c.post("/api/generate", data={"prompt": "a fox"})
    assert r.status_code == 202 and r.json()["status"] in ("queued", "running")
    job_id, seen_preview = r.json()["id"], False
    for _ in range(100):
        body = c.get(f"/api/jobs/{job_id}").json()
        if body["preview"]:
            resp = c.get(body["preview"])
            seen_preview = seen_preview or (resp.status_code == 200 and resp.headers["content-type"] == "image/png")
        if body["status"] == "done":
            break
        time.sleep(0.02)
    assert body["status"] == "done" and seen_preview
    assert any(j["id"] == job_id for j in c.get("/api/jobs").json())


def test_worker_errors_fail_the_job(env):
    body = env.client.post("/api/generate", data={"prompt": "please fail", "wait": "true"}).json()
    assert body["status"] == "error" and body["error"] == "ValueError: bad latent" and body["videos"] == []
    # The UI's queue sees the same status.
    assert env.queue.get_job(body["id"]).status.value == "failed"


def test_endframe_extend_and_batch(env):
    c = env.client
    body = c.post("/api/generate", data={"prompt": "a fox", "end_frame_strength": 0.5, "wait": "true"},
                  files={"image": ("a.png", png(), "image/png"), "end_frame": ("b.png", png(color=(9, 9, 9)), "image/png")}).json()
    assert body["model_type"] == "Original with Endframe" and env.submitted[-1]["end_frame_strength"] == 0.5
    assert c.post("/api/generate", data={"prompt": "a fox", "model": "f1"},
                  files={"end_frame": ("b.png", png(), "image/png")}).status_code == 400

    ext = c.post("/api/extend", data={"prompt": "keep running", "source_job": body["id"], "model": "f1",
                                      "context_frames": 3, "wait": "true"}).json()
    assert ext["kind"] == "extend" and ext["model_type"] == "Video F1"
    kw = env.submitted[-1]
    assert kw["input_image"] == kw["input_image_path"] and kw["num_cleaned_frames"] == 3
    up = c.post("/api/extend", data={"prompt": "more"}, files={"video": ("clip.mp4", b"video bytes", "video/mp4")}).json()
    assert up["model_type"] == "Video" and Path(env.submitted[-1]["input_image"]).read_bytes() == b"video bytes"

    r = c.post("/api/batch", data={"prompt": "a fox", "seed": 10, "wait": "true"},
               files=[("images", ("1.png", png(), "image/png")), ("images", ("2.png", png(), "image/png"))])
    jobs = r.json()["jobs"]
    assert r.status_code == 200 and [j["seed"] for j in jobs] == [10, 11] and all(j["status"] == "done" for j in jobs)


def test_presets_fill_in_unsent_fields(env):
    paths.SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.SETTINGS_DIR / "generation_presets.json").write_text(json.dumps({
        "F1": {"fast": {"prompt": "a preset prompt", "steps": 12, "cache_type": "TeaCache", "latent_type": "Green Screen",
                        "randomize_seed": False, "seed": 99, "lora_values": {"style_a": 0.3}}},
    }))
    c = env.client
    assert c.get("/api/presets").json() == {"F1": ["fast"]}
    c.post("/api/generate", data={"model": "f1", "preset": "fast", "steps": 20, "wait": "true"})
    kw = env.submitted[-1]
    assert kw["prompt_text"] == "a preset prompt" and kw["steps"] == 20 and kw["seed"] == 99
    assert kw["use_teacache"] and kw["latent_type"] == "Green Screen" and kw["selected_loras"] == ["style_a"]
    assert c.post("/api/generate", data={"model": "f1", "preset": "nope"}).status_code == 400


def test_rejects_bad_options(env):
    c = env.client
    bad = [{"prompt": ""}, {"prompt": "x", "steps": 0}, {"prompt": "x", "width": 2000}, {"prompt": "x", "cache": "fast"},
           {"prompt": "x", "loras": '{"missing": 1}'}, {"prompt": "x", "loras": "{oops"}, {"prompt": "x", "model": "v2"},
           {"prompt": "x", "latent_type": "purple"}]
    for data in bad:
        assert c.post("/api/generate", data=data).status_code == 400, data
    assert c.post("/api/generate", data={"prompt": "x"}, files={"image": ("a.png", b"not an image", "image/png")}).status_code == 400
    assert c.post("/api/extend", data={"prompt": "x"}).status_code == 400
    assert c.get("/api/jobs/nope").status_code == 404


def test_cancel_queued_generation(env):
    c = env.client
    first = c.post("/api/generate", data={"prompt": "a"}).json()
    second = c.post("/api/generate", data={"prompt": "b"}).json()
    body = c.delete(f"/api/jobs/{second['id']}").json()
    assert body["status"] == "cancelled"
    c.get(f"/api/jobs/{first['id']}", params={"wait": "true"})
    assert c.get(f"/api/jobs/{second['id']}").json()["status"] == "cancelled"


def test_postprocess_ops_and_chaining(env):
    c = env.client
    gen = c.post("/api/generate", data={"prompt": "a fox", "wait": "true"}).json()

    up = finished(env, c.post("/api/upscale", data={"source_job": gen["id"], "model": "RealESRGAN_x2plus", "scale": 1.5,
                                                    "wait": "true"}).json())
    step = env.toolbox.calls[-1][0]
    assert step["name"] == "upscale" and step["params"]["output_scale_factor_ui"] == 1.5 and step["params"]["model_key"] == "RealESRGAN_x2plus"
    assert c.get(up["videos"][0]["url"]).content == b"fake mp4"
    assert up["progress"] == {"step": 3, "total": 3}

    gif = finished(env, c.post("/api/postprocess", data={
        "source_job": up["id"], "wait": "true",
        "operations": json.dumps([{"op": "interpolate", "interpolation": "4x"}, {"op": "loop", "type": "ping-pong", "loops": 2},
                                  {"op": "export", "format": "gif", "max_width": 512}])}).json())
    names = [s["name"] for s in env.toolbox.calls[-1]]
    assert names == ["frame_adjust", "loop", "export"] and env.toolbox.calls[-1][0]["params"]["target_fps_mode"] == "4x Frames"
    assert gif["videos"][0]["format"] == "gif"
    assert c.get(gif["videos"][0]["url"]).headers["content-type"] == "image/gif"

    joined = finished(env, c.post("/api/join", data={"source_jobs": json.dumps([gen["id"], up["id"]]), "name": "both",
                                                     "wait": "true"}).json())
    assert c.get(joined["videos"][0]["url"]).content == b"fake mp4fake mp4"
    assert c.post("/api/analyze", files={"video": ("clip.mp4", b"x", "video/mp4")}).json()["report"].startswith("File: clip")


def test_postprocess_failures_and_validation(env):
    c = env.client
    video = {"video": ("clip.mp4", b"video bytes", "video/mp4")}
    body = c.post("/api/filters", data={"values": '{"blur": 5}', "wait": "true"}, files=video).json()
    assert body["status"] == "error" and body["error"] == "FFmpeg failed during filters."
    assert any("FFmpeg failed" in line for line in body["log"])
    for path, data in [("/api/upscale", {"model": "nope"}), ("/api/upscale", {"model": "RealESRGAN_x2plus", "scale": 3}),
                       ("/api/interpolate", {"interpolation": "8x"}), ("/api/filters", {"values": '{"glow": 1}'}),
                       ("/api/loop", {"loops": 50}), ("/api/export", {"format": "avi"}),
                       ("/api/postprocess", {"operations": '[{"op": "sharpen"}]'}), ("/api/postprocess", {"operations": "[]"})]:
        assert c.post(path, data=data, files=video).status_code == 400, (path, data)
    assert c.post("/api/upscale", data={}).status_code == 400  # no video
    assert c.post("/api/join", files=[("videos", ("a.mp4", b"a", "video/mp4"))]).status_code == 400


def test_prompt_helpers(env):
    c = env.client
    assert c.post("/api/enhance", data={"prompt": "a fox"}).json() == {"prompt": "a fox, cinematic lighting"}
    assert c.post("/api/caption", files={"image": ("a.png", png(), "image/png")}).json() == {"prompt": "a 64x48 image"}


def test_dummy_lora_is_hidden(env, tmp_path):
    job_queue = VideoJobQueue()
    api = create_api(job_queue, lambda **kw: "x", Settings(), ["only", DUMMY_LORA_NAME])
    app = FastAPI()
    app.mount("/api", api)
    c = TestClient(app)
    assert c.get("/api/loras").json()["loras"] == ["only"]
    assert c.post("/api/upscale", files={"video": ("a.mp4", b"a", "video/mp4")}).status_code == 503


def test_last_frame_of_a_result_video(env):
    cv2 = pytest.importorskip("cv2")
    body = env.client.post("/api/generate", data={"prompt": "a fox", "wait": "true"}).json()
    video = env.queue.get_job(body["id"]).result
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 48))
    for shade in (0, 100, 200):
        writer.write(np.full((48, 64, 3), shade, dtype=np.uint8))
    writer.release()
    r = env.client.get(f"/api/jobs/{body['id']}/videos/0/last-frame", params={"download": "true"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "attachment" in r.headers["content-disposition"]
    frame = np.array(Image.open(io.BytesIO(r.content)))
    assert frame.shape == (48, 64, 3) and abs(int(frame.mean()) - 200) < 12
    assert env.client.get(f"/api/jobs/{body['id']}/videos/5/last-frame").status_code == 404


def test_generate_from_the_last_frame_of_a_job(env):
    cv2 = pytest.importorskip("cv2")
    first = env.client.post("/api/generate", data={"prompt": "a fox", "wait": "true"}).json()
    video = env.queue.get_job(first["id"]).result
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 48))
    for _ in range(3):
        writer.write(np.full((48, 64, 3), 200, dtype=np.uint8))
    writer.release()
    r = env.client.post("/api/generate", data={"prompt": "a fox", "image_from_job": first["id"], "wait": "true"})
    assert r.status_code == 200 and r.json()["status"] == "done"
    assert env.submitted[-1]["input_image"].shape == (48, 64, 3)
    assert env.client.post("/api/generate", data={"prompt": "x", "image_from_job": "nope"}).status_code == 404
