"""save_bcthw_as_mp4 (diffusers_helper/utils.py): the frames it encodes, converted a chunk at a time."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
einops = pytest.importorskip("einops")
pytest.importorskip("av")

from diffusers_helper.utils import save_bcthw_as_mp4  # noqa: E402


@pytest.mark.parametrize("b,t", [(1, 37), (4, 20)])  # frame counts that don't fill the last chunk
def test_frames_match_whole_video_conversion(tmp_path, b, t):
    x = torch.empty(b, 3, t, 32, 48, dtype=torch.float16).uniform_(-1.2, 1.2)
    out = tmp_path / "out.mp4"

    frames = save_bcthw_as_mp4(x, str(out), fps=30, crf=0)

    per_row = next((p for p in [6, 5, 4, 3, 2] if b % p == 0), b)
    expected = (torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5).to(torch.uint8)
    expected = einops.rearrange(expected, '(m n) c t h w -> t (m h) (n w) c', n=per_row)
    assert torch.equal(frames, expected)
    assert out.stat().st_size > 0
