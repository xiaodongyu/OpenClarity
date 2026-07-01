"""
Tests for sr_model.SRModel.

These require torch (skipped otherwise, since it's a heavy optional
dependency not needed for the rest of the pipeline). Weights are randomly
initialised — tests verify shape/dtype contracts, not reconstruction quality
(that's covered by src/benchmark.py against a trained checkpoint).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.sr_model import SRModel


@pytest.mark.parametrize("arch", ["espcn", "fsrcnn"])
def test_upscale_output_shape(arch):
    model = SRModel(scale=4, arch=arch)
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    result = model.upscale(img)
    assert result.shape == (128, 128, 3)


@pytest.mark.parametrize("arch", ["espcn", "fsrcnn"])
def test_upscale_output_dtype(arch):
    model = SRModel(scale=2, arch=arch)
    img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    result = model.upscale(img)
    assert result.dtype == np.uint8


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        SRModel(scale=4, arch="not-a-real-arch")


def test_scale_attribute_matches_requested_scale():
    model = SRModel(scale=3, arch="espcn")
    assert model.scale == 3
