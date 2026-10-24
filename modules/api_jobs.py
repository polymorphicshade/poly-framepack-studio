"""Jobs behind the HTTP API.

Generation jobs live in the studio's VideoJobQueue, so the API and the web UI share one queue and
one GPU; this module only describes them. Post-processing jobs (the toolbox) run one at a time on
a worker thread of their own, as they do when started from the UI.

Both kinds are reported in the same shape, with statuses queued, running, done, error or cancelled.
"""

from __future__ import annotations

import io
import logging
import os
import queue
import re
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

TERMINAL = ("done", "error", "cancelled")

# VideoJobQueue's JobStatus values -> API statuses.
QUEUE_STATUS = {"pending": "queued", "running": "running", "completed": "done", "failed": "error", "cancelled": "cancelled"}

MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".gif": "image/gif", ".png": "image/png", ".json": "application/json"}

# The worker names its outputs <output_id>_<frames>.mp4, <output_id>.png and <output_id>.json.
_OUTPUT_ID = re.compile(r"^(\d{6}_\d{6}_\d{3}_\d{1,4})")
_TOTAL = re.compile(r"Total (\d+)/(\d+)")
_SAMPLING = re.compile(r"Sampling (\d+)/(\d+)")
_SPAN = re.compile(r"<span>(.*?)</span>", re.S)


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------------------- generation jobs

def queue_status(job) -> str:
    return QUEUE_STATUS.get(getattr(job.status, "value", str(job.status)), "error")


def queue_kind(job) -> str:
    if getattr(job.job_type, "value", None) == "grid":
        return "grid"
    model_type = job.params.get("model_type")
    if model_type == "Sequence":
        return "sequence"
    return "extend" if model_type in ("Video", "Video F1") else "generate"


def output_id(job) -> str | None:
    """The timestamp ID the worker named this job's files with."""
    if getattr(job, "output_id", None):
        return job.output_id
    match = _OUTPUT_ID.match(os.path.basename(job.result or ""))
    return match.group(1) if match else None


def queue_progress(job) -> tuple[int, int, str]:
    """(step, total, message) from the worker's progress update: steps over the whole video."""
    data = job.progress_data or {}
    html, desc = data.get("html") or "", (data.get("desc") or "").strip()
    match = _TOTAL.search(html) or _SAMPLING.search(html)
    step, total = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    texts = [t.strip() for t in _SPAN.findall(html) if t.strip()]
    return step, total, desc or " · ".join(texts)


def queue_files(job, settings) -> dict[str, str]:
    """Side outputs on disk: the metadata JSON and the start image."""
    oid = output_id(job)
    if not oid:
        return {}
    candidates = {
        "metadata": os.path.join(settings.get("metadata_dir"), f"{oid}.json"),
        "image": os.path.join(settings.get("output_dir"), f"{oid}.png"),
    }
    return {name: path for name, path in candidates.items() if os.path.isfile(path)}


def describe_queue_job(job, job_queue, settings, base: str = "/api") -> dict:
    root = f"{base}/jobs/{job.id}"
    status = queue_status(job)
    step, total, message = queue_progress(job)
    if status == "done":
        message = "Done"
    elif status == "cancelled":
        message = "Cancelled"
    elif status == "error":
        message = "Failed"
    elif status == "queued":
        message = "Waiting in the queue"
    preview = (job.progress_data or {}).get("preview")
    has_video = bool(job.result and os.path.isfile(job.result))
    p = job.params
    return {
        "id": job.id,
        "kind": queue_kind(job),
        "model_type": p.get("model_type"),
        "status": status,
        "progress": {"step": step, "total": total},
        "message": message,
        "error": job.error if status == "error" else None,
        "created": job.created_at,
        "started": job.started_at,
        "finished": job.completed_at,
        "seed": p.get("seed"),
        "prompt": p.get("prompt_text"),
        "queue_position": (job_queue.get_queue_position(job.id) or 0) if status == "queued" else 0,
        "params": {k: v for k, v in p.items() if isinstance(v, (str, int, float, bool, type(None)))},
        # While a job runs this is the video so far (FramePack writes a longer one after each section).
        "videos": [{"url": f"{root}/videos/0", "final": status == "done"}] if has_video else [],
        "files": {name: f"{root}/files/{name}" for name in queue_files(job, settings)},
        "children": [f"{base}/jobs/{c}" for c in job.child_job_ids],
        "preview": f"{root}/preview?t={id(preview)}" if status == "running" and preview is not None else None,
    }


def encode_preview(array) -> bytes:
    """A latent preview (an HxWx3 uint8 array) as PNG."""
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(array).astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------- post-processing jobs

class Progress:
    """Stands in for gr.Progress inside the toolbox: ``progress(fraction, desc=...)`` and
    ``progress.tqdm(iterable, desc=...)``. Raises Cancelled at the next update after a cancel."""

    def __init__(self, job: "ToolJob"):
        self.job = job

    def _check(self):
        if self.job.cancel_requested:
            raise Cancelled()

    def __call__(self, value=None, desc=None, total=None, unit=None, **_):
        self._check()
        if isinstance(value, (int, float)):
            self.job.step, self.job.total = int(round(float(value) * 100)), 100
        if desc:
            self.job.message = str(desc)

    def tqdm(self, iterable: Iterable, desc=None, total=None, unit=None, **_):
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = 0
        if desc:
            self.job.message = str(desc)
        self.job.step, self.job.total = 0, total
        for i, item in enumerate(iterable, 1):
            self._check()
            yield item
            self.job.step = i


@dataclass
class ToolResult:
    videos: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


@dataclass
class ToolJob:
    id: str
    kind: str
    params: dict
    run: Callable[[Progress], ToolResult] = field(repr=False)
    status: str = "queued"
    step: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    videos: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    cancel_requested: bool = False
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self, queue_position: int = 0, base: str = "/api") -> dict:
        root = f"{base}/jobs/{self.id}"
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": {"step": self.step, "total": self.total},
            "message": self.message,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "queue_position": queue_position,
            "params": self.params,
            "info": self.info,
            "log": self.log[-50:],
            "videos": [{"url": f"{root}/videos/{i}", "format": os.path.splitext(v)[1].lstrip(".")}
                       for i, v in enumerate(self.videos)],
        }


class ToolWorker:
    """A single-worker queue for post-processing, so jobs never share the toolbox's state."""

    def __init__(self, toolbox, max_history: int = 200):
        self.toolbox = toolbox
        self.max_history = max_history
        self.jobs: dict[str, ToolJob] = {}
        self._queue: queue.Queue[ToolJob] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="api-postprocess-worker", daemon=True)
        self._thread.start()

    def submit(self, kind: str, params: dict, run: Callable[[Progress], ToolResult]) -> ToolJob:
        job = ToolJob(id=f"pp-{uuid.uuid4().hex[:12]}", kind=kind, params=params, run=run)
        with self._lock:
            self.jobs[job.id] = job
            self._prune()
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> ToolJob | None:
        return self.jobs.get(job_id)

    def list(self) -> list[ToolJob]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> ToolJob | None:
        job = self.jobs.get(job_id)
        if job and job.status not in TERMINAL:
            job.cancel_requested = True
            if job.status == "queued":
                job.message = "Cancelled"
                self._finish(job, "cancelled")
        return job

    def cancel_queued(self) -> int:
        queued = [j for j in self.jobs.values() if j.status == "queued"]
        for job in queued:
            self.cancel(job.id)
        return len(queued)

    def queue_position(self, job: ToolJob) -> int:
        queued = sorted((j for j in self.jobs.values() if j.status == "queued"), key=lambda j: j.created)
        return queued.index(job) + 1 if job in queued else 0

    def _prune(self):
        done = sorted((j for j in self.jobs.values() if j.status in TERMINAL), key=lambda j: j.created)
        for j in done[: max(0, len(self.jobs) - self.max_history)]:
            self.jobs.pop(j.id, None)

    def _finish(self, job: ToolJob, status: str, error: str | None = None):
        job.status, job.error, job.finished = status, error, time.time()
        job.done_event.set()

    def _loop(self):
        while True:
            job = self._queue.get()
            if job.status != "queued":
                continue
            job.status, job.started = "running", time.time()
            errors: list[str] = []

            # The toolbox reports problems through its message log rather than by raising.
            def listen(kind, message, job=job):
                job.log.append(message)
                if kind == "ERROR":
                    errors.append(message.removeprefix("❌").strip())

            manager = getattr(self.toolbox, "message_manager", None)
            if manager is not None:
                manager.add_listener(listen)
            try:
                result = job.run(Progress(job))
                if job.cancel_requested:
                    raise Cancelled()
                if not result or not result.videos:
                    raise RuntimeError(errors[0] if errors else "Post-processing produced no output (see the job log)")
                job.videos, job.info = list(result.videos), dict(result.info)
                job.message = "Done"
                self._finish(job, "done")
            except Cancelled:
                job.message = "Cancelled"
                self._finish(job, "cancelled")
            except Exception as e:  # surfaced to the client
                log.error("Post-processing job %s failed:\n%s", job.id, traceback.format_exc())
                job.message = "Failed"
                self._finish(job, "error", str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}")
            finally:
                if manager is not None:
                    manager.remove_listener(listen)
