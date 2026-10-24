"""Sequence generation: one video through two or more keyframes.

Each pair of neighbouring keyframes gets its own segment, an "Original with Endframe" generation
from one keyframe towards the next with its own prompt and length, and the segments are joined
end to end. With chaining on (the default), a segment starts from the frame the previous segment
actually ended on rather than from its keyframe, so the joins have no jump; either way the first
frame of every later segment repeats the last frame of the one before and is dropped.

Kept free of torch so the tests can run the whole sequence with a fake worker: worker.py passes
in the pieces that need it.
"""

import datetime
import os
import re
import traceback

import numpy as np
from PIL import Image

from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from modules.last_frame import extract_last_frame

MAX_SEQUENCE_IMAGES = 10
FPS = 30

# The job parameters only a Sequence job has; its segments are ordinary endframe jobs without them.
SEQUENCE_PARAMS = ("sequence_image_paths", "sequence_prompts", "sequence_seconds", "chain_segments")

_SPAN = re.compile(r"<span>(.*?)</span>", re.S)
_VALUE = re.compile(r'value="(\d+)"')
_TOTAL = re.compile(r"Total (\d+)/(\d+)")
_ETA = re.compile(r"ETA (?:(\d+) days?, )?(\d+):(\d\d):(\d\d)")


def plan_segments(count, prompts=None, seconds=None, default_seconds=6.0):
    """(prompts, seconds) for ``count`` segments. A blank prompt means "use the main prompt"; a
    missing length is ``default_seconds``."""
    prompts = [str(p or "").strip() for p in (prompts or [])][:count]
    prompts += [""] * (count - len(prompts))
    lengths = [float(s) for s in (seconds or []) if s is not None][:count]
    lengths += [float(default_seconds)] * (count - len(lengths))
    return prompts, lengths


def segment_sections(seconds, latent_window_size):
    """How many sections the worker samples for a video of this length (as worker.py counts them)."""
    return int(max(round((seconds * FPS) / (latent_window_size * 4)), 1))


def natural_key(path):
    """Sort key putting 'frame2.png' before 'frame10.png'."""
    name = os.path.basename(str(path)).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def load_image(path):
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def save_keyframes(images, directory, stem):
    """Write the keyframes as PNGs (so a queued job survives a restart) and return their paths."""
    os.makedirs(directory, exist_ok=True)
    paths = []
    for i, image in enumerate(images, 1):
        path = os.path.join(directory, f"{stem}_keyframe_{i}.png")
        Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB").save(path)
        paths.append(path)
    return paths


def fit(image, width, height):
    """Resize to cover width x height and centre-crop, like diffusers_helper.utils.resize_and_center_crop
    (which imports torch)."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    pil = Image.fromarray(image)
    scale = max(width / pil.width, height / pil.height)
    resized = pil.resize((int(round(pil.width * scale)), int(round(pil.height * scale))), Image.LANCZOS)
    left, top = (resized.width - width) / 2, (resized.height - height) / 2
    return np.array(resized.crop((left, top, left + width, top + height)))


def join_segments(video_paths, output_path, fps=FPS, crf=16):
    """Join the segment videos into one, dropping the first frame of every segment after the first
    (it repeats the frame the previous segment ended on). Streams frame by frame, so long sequences
    don't have to fit in memory. Returns the number of frames written."""
    import av

    frames = 0
    out = av.open(output_path, mode="w")
    try:
        stream = None
        for index, path in enumerate(video_paths):
            with av.open(path) as source:
                for n, frame in enumerate(source.decode(video=0)):
                    if n == 0 and index > 0:
                        continue
                    image = frame.to_ndarray(format="rgb24")
                    if stream is None:
                        stream = out.add_stream("libx264", rate=fps)
                        stream.height, stream.width = image.shape[:2]
                        stream.pix_fmt = "yuv420p"
                        stream.options = {"crf": str(int(crf))}
                    elif image.shape[:2] != (stream.height, stream.width):
                        raise ValueError(f"{os.path.basename(path)} is {image.shape[1]}x{image.shape[0]}, "
                                         f"the sequence is {stream.width}x{stream.height}")
                    for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                        out.mux(packet)
                    frames += 1
        if stream is None:
            raise ValueError("The segments have no frames")
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()
    return frames


def _format_eta(seconds):
    try:
        return str(datetime.timedelta(seconds=int(seconds)))
    except (OverflowError, ValueError):
        return "--:--"


class SegmentStream:
    """Stands in for the job's stream while one segment runs. Progress is relabelled with the
    segment and counted over the whole sequence, the segment's videos are passed on (the UI shows
    the video so far), and its 'job_id', 'error' and 'end' are kept back for the sequence to act on.
    Cancelling the job still reaches the segment through the shared input queue."""

    def __init__(self, stream, index, count, steps_before, total_steps):
        self.input_queue = stream.input_queue
        self.output_queue = self
        self._out = stream.output_queue
        self.label = f"Segment {index + 1}/{count}"
        self.steps_before = steps_before
        self.total_steps = max(total_steps, 1)
        self.video = None
        self.error = None
        self.output_id = None

    def push(self, item):
        flag, data = item
        if flag == "progress":
            preview, desc, html = data
            self._out.push(("progress", (preview, *self.relabel(desc, html))))
        elif flag == "file":
            self.video = data
            self._out.push(item)
        elif flag == "error":
            self.error = data
        elif flag == "job_id":
            self.output_id = data

    def relabel(self, desc, html):
        texts = [t.strip() for t in _SPAN.findall(html or "")]
        total = _TOTAL.search(html or "")
        if total:
            done_here, total_here = int(total.group(1)), int(total.group(2))
            done = self.steps_before + done_here
            values = _VALUE.findall(html)
            segment_percent = int(values[0]) if values else int(100 * done_here / max(total_here, 1))
            overall = f"Total {done}/{self.total_steps}"
            eta = _ETA.search(texts[-1] if texts else "")
            if eta and done_here < total_here:
                days, h, m, s = (int(g or 0) for g in eta.groups())
                per_step = (((days * 24 + h) * 60 + m) * 60 + s) / (total_here - done_here)
                overall += f"  ETA {_format_eta(per_step * (self.total_steps - done))}"
            html = (make_progress_bar_html(segment_percent, f"{self.label} · {texts[0] if texts else ''}")
                    + make_progress_bar_html(int(100 * done / self.total_steps), overall))
        else:
            text = texts[0] if texts else ""
            html = make_progress_bar_html(int(100 * self.steps_before / self.total_steps),
                                          f"{self.label} · {text}" if text else self.label)
        return (f"{self.label} · {desc}" if desc else self.label), html


def run_sequence(params, segment_worker, job_id, settings, stream, save_start_image=None, resize=fit):
    """Generate a Sequence job.

    ``params`` are the worker's arguments, ``segment_worker`` the worker itself (called once per
    segment as an "Original with Endframe" job), ``job_id`` names the joined video and its metadata,
    and ``stream`` is the job's stream. Talks to the queue like the worker: progress, the videos so
    far, then the joined video, and always 'end'.
    """
    out = stream.output_queue
    out.push(("job_id", job_id))
    out.push(("progress", (None, "", make_progress_bar_html(0, "Starting sequence..."))))
    segments_done = []  # (video, output_id) of each finished segment
    joined = None
    try:
        keyframes = [load_image(p) for p in params.get("sequence_image_paths") or []]
        if len(keyframes) < 2:
            raise ValueError("A sequence needs at least two keyframes")
        prompts, seconds = plan_segments(len(keyframes) - 1, params.get("sequence_prompts"),
                                         params.get("sequence_seconds"), params.get("total_second_length") or 6)
        prompts = [p or params.get("prompt_text") or "" for p in prompts]

        # Every segment must come out the same size, so all keyframes take the first one's bucket.
        height, width = find_nearest_bucket(*keyframes[0].shape[:2], resolution=params.get("resolutionW", 640))
        keyframes = [resize(k, width, height) for k in keyframes]
        steps = [segment_sections(s, params["latent_window_size"]) * params["steps"] for s in seconds]
        chain = params.get("chain_segments", True)

        start = keyframes[0]
        for i, (prompt, length) in enumerate(zip(prompts, seconds)):
            if stream.input_queue.top() == "end":
                return
            proxy = SegmentStream(stream, i, len(prompts), sum(steps[:i]), sum(steps))
            segment = {k: v for k, v in params.items() if k not in SEQUENCE_PARAMS}
            segment.update(
                model_type="Original with Endframe", input_image=start, has_input_image=True,
                input_image_path=None, end_frame_image=keyframes[i + 1], end_frame_image_path=None,
                prompt_text=prompt, total_second_length=length, seed=int(params["seed"]) + i,
                # The sequence saves one start image and metadata JSON for the whole video.
                save_metadata_checked=False, job_stream=proxy, reuse_loaded_model=i > 0,
            )
            segment_worker(**segment)
            if stream.input_queue.top() == "end":
                return
            if proxy.error:
                out.push(("error", proxy.error))
                return
            if not proxy.video or not os.path.isfile(proxy.video):
                raise RuntimeError(f"Segment {i + 1} produced no video")
            segments_done.append((proxy.video, proxy.output_id))
            start = resize(extract_last_frame(proxy.video), width, height) if chain else keyframes[i + 1]

        out.push(("progress", (None, "", make_progress_bar_html(100, "Joining segments..."))))
        output_dir = params["output_dir"]
        temp_path = os.path.join(output_dir, f"{job_id}_joining.mp4")
        frames = join_segments([v for v, _ in segments_done], temp_path, crf=settings.get("mp4_crf", 16))
        joined = os.path.join(output_dir, f"{job_id}_{frames}.mp4")
        os.replace(temp_path, joined)

        if save_start_image and settings.get("save_metadata") and params.get("save_metadata_checked", True):
            try:
                save_start_image(params | {
                    "input_image": keyframes[0], "height": height, "width": width,
                    "total_second_length": sum(seconds), "sequence_prompts": prompts, "sequence_seconds": seconds,
                }, job_id, settings)
            except Exception:
                traceback.print_exc()
        out.push(("file", joined))
    except Exception:
        traceback.print_exc()
        out.push(("error", f"Error during generation: {traceback.format_exc()}"))
    finally:
        # Once the joined video exists the segments are intermediate files, like a job's partial videos.
        if joined and settings.get("clean_up_videos", True):
            for video, output_id in segments_done:
                leftovers = [video]
                if output_id:
                    leftovers.append(os.path.join(params.get("metadata_dir") or "", f"{output_id}_end_frame_processed.png"))
                for path in leftovers:
                    try:
                        if os.path.isfile(path):
                            os.remove(path)
                    except OSError as e:
                        print(f"Could not delete {path}: {e}")
        out.push(("end", None))
