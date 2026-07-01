"""
Integration tests for run_cycle(). No camera, torch, or trained weights
required — the SR model is a lightweight fake object exposing .scale and
.upscale(), matching the SRModel interface.
"""
import numpy as np
from src.pipeline import run_cycle


class _FakeSRModel:
    def __init__(self, scale):
        self.scale = scale

    def upscale(self, img):
        h, w = img.shape[:2]
        return np.zeros((h * self.scale, w * self.scale, 3), dtype=np.uint8)


_FRAME = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)


def test_run_cycle_bilinear_fallback_full_frame():
    result = run_cycle(_FRAME, roi=None, scale=4, model=None)
    assert result.shape == (800, 1600, 3)
    assert result.dtype == np.uint8


def test_run_cycle_bilinear_fallback_with_roi():
    result = run_cycle(_FRAME, roi=(0, 0, 100, 50), scale=4, model=None)
    assert result.shape == (200, 400, 3)


def test_run_cycle_with_sr_model_matching_scale():
    model = _FakeSRModel(scale=4)
    result = run_cycle(_FRAME, roi=(0, 0, 100, 50), scale=4, model=model)
    assert result.shape == (200, 400, 3)


def test_run_cycle_with_sr_model_scale_mismatch_resizes_to_requested_scale():
    model = _FakeSRModel(scale=2)
    result = run_cycle(_FRAME, roi=(0, 0, 100, 50), scale=4, model=model)
    # model upsamples by 2x internally, run_cycle must resize up to the requested 4x
    assert result.shape == (200, 400, 3)
