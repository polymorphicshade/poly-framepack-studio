"""Extract the last frame of a video (the UI's "download / send to input" buttons and the API)."""

import os
import uuid

import cv2
import numpy as np
from PIL import Image

from modules import paths


def extract_last_frame(video_path: str) -> np.ndarray:
    """The last frame of a video as an RGB array. Raises ValueError if the video has no readable frame."""
    if not video_path or not os.path.isfile(video_path):
        raise ValueError("Video not found")
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError("Could not open the video")
        frame = None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Seeking is quick, but the reported count can be off by a few frames, so step back until one reads.
        for index in range(count - 1, max(count - 6, -1), -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, candidate = cap.read()
            if ok:
                frame = candidate
                break
        if frame is None:  # no usable frame count: read through to the end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, candidate = cap.read()
                if not ok:
                    break
                frame = candidate
        if frame is None:
            raise ValueError("The video has no frames")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def save_last_frame(video_path: str, directory: str | None = None) -> str:
    """Write the last frame next to the other temporary files as a PNG and return its path."""
    frame = extract_last_frame(video_path)
    directory = directory or str(paths.TEMP_DIR)
    os.makedirs(directory, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out = os.path.join(directory, f"{stem}_last_frame_{uuid.uuid4().hex[:6]}.png")
    Image.fromarray(frame).save(out)
    return out
