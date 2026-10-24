"""Sequence generation (modules/sequence.py) with a fake segment worker that writes real videos:
no GPU, torch or model weights needed."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

av = pytest.importorskip("av")

from diffusers_helper.gradio.progress_bar import make_progress_bar_html  # noqa: E402
from diffusers_helper.thread_utils import AsyncStream  # noqa: E402
from modules.sequence import (SegmentStream, join_segments, natural_key, plan_segments, run_sequence,  # noqa: E402
                              save_keyframes, segment_sections)

W, H = 64, 48  # not a FramePack bucket, so the keyframes get resized to one


def write_video(path, colors, size=(W, H)):
    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("libx264", rate=30)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}
        for color in colors:
            frame = np.empty((size[1], size[0], 3), dtype=np.uint8)
            frame[:] = color
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def read_frames(path):
    with av.open(str(path)) as source:
        return [f.to_ndarray(format="rgb24") for f in source.decode(video=0)]


def solid(color, size=(W, H)):
    image = np.empty((size[1], size[0], 3), dtype=np.uint8)
    image[:] = color
    return image


def close(a, b, tol=6):
    return np.abs(np.asarray(a, float).reshape(-1, 3).mean(0) - np.asarray(b, float)).max() < tol


def test_plan_segments_pads_and_trims():
    assert plan_segments(3, ["a", None], [2, 4, 5, 9], 6) == (["a", "", ""], [2.0, 4.0, 5.0])
    assert plan_segments(2) == (["", ""], [6.0, 6.0])
    assert segment_sections(3, 9) == 2 and segment_sections(0.1, 9) == 1


def test_natural_order():
    names = ["shot10.png", "shot2.png", "Shot1.png"]
    assert sorted(names, key=natural_key) == ["Shot1.png", "shot2.png", "shot10.png"]


def test_join_drops_the_repeated_first_frame(tmp_path):
    write_video(tmp_path / "a.mp4", [(255, 0, 0)] * 3 + [(0, 255, 0)])
    write_video(tmp_path / "b.mp4", [(0, 255, 0)] + [(0, 0, 255)] * 3)
    frames = join_segments([tmp_path / "a.mp4", tmp_path / "b.mp4"], str(tmp_path / "joined.mp4"), crf=0)
    out = read_frames(tmp_path / "joined.mp4")
    assert frames == len(out) == 7
    assert close(out[3], (0, 255, 0)) and close(out[4], (0, 0, 255))
    write_video(tmp_path / "c.mp4", [(0, 0, 0)] * 2, size=(32, 32))
    with pytest.raises(ValueError):
        join_segments([tmp_path / "a.mp4", tmp_path / "c.mp4"], str(tmp_path / "bad.mp4"))


def test_segment_progress_counts_over_the_whole_sequence():
    stream = AsyncStream()
    proxy = SegmentStream(stream, index=1, count=3, steps_before=50, total_steps=150)
    html = make_progress_bar_html(40, "Sampling 10/25  ETA 0:00:15") + make_progress_bar_html(20, "Total 10/50  ETA 0:00:40")
    proxy.output_queue.push(("progress", (None, "Generating", html)))
    proxy.output_queue.push(("job_id", "seg"))
    proxy.output_queue.push(("error", "boom"))
    proxy.output_queue.push(("end", None))
    flag, (_, desc, new_html) = stream.output_queue.next()
    assert flag == "progress" and desc == "Segment 2/3 · Generating"
    assert "Segment 2/3 · Sampling 10/25" in new_html and "Total 60/150" in new_html
    assert "ETA 0:01:30" in new_html  # 40 s for this segment's last 40 steps: 1 s per step, 90 steps left
    assert stream.output_queue.top() is None  # job_id, error and end stay with the sequence
    assert (proxy.output_id, proxy.error) == ("seg", "boom")


class FakeSegments:
    """Stands in for the worker: each segment is a video fading from its start to its end frame."""

    def __init__(self, out_dir, fail_at=None, cancel_at=None):
        self.out_dir, self.fail_at, self.cancel_at, self.calls = Path(out_dir), fail_at, cancel_at, []

    def __call__(self, job_stream=None, **params):
        i = len(self.calls)
        self.calls.append(params)
        out = job_stream.output_queue
        out.push(("job_id", f"seg{i}"))
        if i == self.cancel_at:
            job_stream.input_queue.push("end")
            out.push(("end", None))
            return
        if i == self.fail_at:
            out.push(("error", "Error during generation: ValueError: bad segment"))
            out.push(("end", None))
            return
        out.push(("progress", (None, "", make_progress_bar_html(0, "Total 1/2  ETA 0:00:01"))))
        start = params["input_image"].reshape(-1, 3).mean(0)
        end = params["end_frame_image"].reshape(-1, 3).mean(0)
        h, w = params["input_image"].shape[:2]
        video = self.out_dir / f"seg{i}_9.mp4"
        write_video(video, [tuple(int(c) for c in start + (end - start) * t / 4) for t in range(5)], size=(w, h))
        (self.out_dir / f"seg{i}_end_frame_processed.png").write_bytes(b"png")
        out.push(("file", str(video)))
        out.push(("end", None))


RED, GREEN, BLUE = (220, 30, 30), (30, 220, 30), (30, 30, 220)


def sequence_params(tmp_path, **overrides):
    paths = save_keyframes([solid(RED), solid(GREEN), solid(BLUE)], str(tmp_path / "inputs"), "seq")
    return {
        "model_type": "Sequence", "prompt_text": "main prompt", "seed": 100, "steps": 5, "latent_window_size": 9,
        "resolutionW": 256, "total_second_length": 6, "output_dir": str(tmp_path), "metadata_dir": str(tmp_path),
        "end_frame_strength": 0.8, "save_metadata_checked": True, "job_stream": None,
        "sequence_image_paths": paths, "sequence_prompts": ["walks to the door", ""], "sequence_seconds": [2, 3],
        "chain_segments": True,
    } | overrides


def drain(stream):
    items = []
    while stream.output_queue.top() is not None:
        items.append(stream.output_queue.next())
    return items


def test_run_sequence_chains_segments_and_joins_them(tmp_path):
    fake, stream, saved = FakeSegments(tmp_path), AsyncStream(), []
    settings = {"mp4_crf": 0, "save_metadata": True, "clean_up_videos": True}
    run_sequence(sequence_params(tmp_path), fake, "seqid", settings, stream,
                 save_start_image=lambda params, job_id, s: saved.append((params, job_id)))
    items = drain(stream)
    assert items[0] == ("job_id", "seqid") and items[-1] == ("end", None)
    assert not [d for f, d in items if f == "error"]

    first, second = fake.calls
    assert first["model_type"] == second["model_type"] == "Original with Endframe"
    assert (first["prompt_text"], second["prompt_text"]) == ("walks to the door", "main prompt")
    assert (first["total_second_length"], second["total_second_length"]) == (2.0, 3.0)
    assert (first["seed"], second["seed"]) == (100, 101) and first["end_frame_strength"] == 0.8
    assert not first["reuse_loaded_model"] and second["reuse_loaded_model"] and not first["save_metadata_checked"]
    assert "sequence_image_paths" not in first
    # All keyframes take the first one's bucket, and segment 2 starts where segment 1 ended.
    assert first["input_image"].shape == second["end_frame_image"].shape == (224, 288, 3)
    assert close(first["input_image"], RED) and close(first["end_frame_image"], GREEN)
    assert close(second["input_image"], GREEN) and close(second["end_frame_image"], BLUE)

    joined = [d for f, d in items if f == "file"][-1]
    assert Path(joined).name == "seqid_9.mp4"
    frames = read_frames(joined)
    assert len(frames) == 9 and close(frames[0], RED) and close(frames[4], GREEN) and close(frames[-1], BLUE)
    # The segments were intermediate files; the joined video's metadata covers the whole sequence.
    assert not list(tmp_path.glob("seg*"))
    params, job_id = saved[0]
    assert job_id == "seqid" and params["total_second_length"] == 5.0
    assert params["sequence_prompts"] == ["walks to the door", "main prompt"]
    progress = [d for f, d in items if f == "progress"]
    assert any("Segment 2/2" in desc for _, desc, _ in progress)


def test_run_sequence_without_chaining_starts_on_each_keyframe(tmp_path):
    fake, stream = FakeSegments(tmp_path), AsyncStream()
    run_sequence(sequence_params(tmp_path, chain_segments=False), fake, "seqid",
                 {"mp4_crf": 0, "clean_up_videos": False}, stream)
    drain(stream)
    assert close(fake.calls[1]["input_image"], GREEN)
    assert len(list(tmp_path.glob("seg*_9.mp4"))) == 2  # kept when clean-up is off


def test_a_failed_segment_fails_the_sequence(tmp_path):
    fake, stream = FakeSegments(tmp_path, fail_at=1), AsyncStream()
    run_sequence(sequence_params(tmp_path), fake, "seqid", {"clean_up_videos": True}, stream)
    items = drain(stream)
    assert ("error", "Error during generation: ValueError: bad segment") in items and items[-1] == ("end", None)
    assert not [d for f, d in items if f == "file" and "seqid" in d]
    assert (tmp_path / "seg0_9.mp4").exists()  # the finished segment stays for the user


def test_cancelling_stops_after_the_running_segment(tmp_path):
    fake, stream = FakeSegments(tmp_path, cancel_at=0), AsyncStream()
    run_sequence(sequence_params(tmp_path), fake, "seqid", {}, stream)
    items = drain(stream)
    assert len(fake.calls) == 1 and items[-1] == ("end", None)
    assert not [d for f, d in items if f in ("error", "file")]


def test_a_sequence_needs_two_keyframes(tmp_path):
    stream = AsyncStream()
    params = sequence_params(tmp_path)
    run_sequence(params | {"sequence_image_paths": params["sequence_image_paths"][:1]}, FakeSegments(tmp_path),
                 "seqid", {}, stream)
    items = drain(stream)
    assert any(f == "error" and "at least two keyframes" in d for f, d in items) and items[-1] == ("end", None)
