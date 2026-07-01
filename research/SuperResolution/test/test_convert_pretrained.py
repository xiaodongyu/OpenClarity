"""
Tests for convert_pretrained.py's key-remapping logic.

No network access required: synthetic "source" state dicts are built using
the exact key/shape convention of yjn870's FSRCNN/ESPCN repos (first_part /
mid_part / last_part), then remapped and loaded into our own model structure.
This verifies the remap is shape/key-correct without depending on a
third-party download.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.convert_pretrained import remap_espcn_state_dict, remap_fsrcnn_state_dict
from src.sr_model import _build_espcn, _build_fsrcnn


def _make_yjn870_espcn_state_dict(scale=3):
    first_part = nn.Sequential(
        nn.Conv2d(1, 64, 5, padding=2), nn.Tanh(), nn.Conv2d(64, 32, 3, padding=1), nn.Tanh()
    )
    last_part = nn.Sequential(nn.Conv2d(32, scale**2, 3, padding=1), nn.PixelShuffle(scale))
    return {
        "first_part.0.weight": first_part[0].weight,
        "first_part.0.bias": first_part[0].bias,
        "first_part.2.weight": first_part[2].weight,
        "first_part.2.bias": first_part[2].bias,
        "last_part.0.weight": last_part[0].weight,
        "last_part.0.bias": last_part[0].bias,
    }


def _make_yjn870_fsrcnn_state_dict(scale=4, d=56, s=12, m=4):
    sd = {}
    first = nn.Conv2d(1, d, 5, padding=2)
    sd["first_part.0.weight"] = first.weight
    sd["first_part.0.bias"] = first.bias
    sd["first_part.1.weight"] = nn.PReLU(d).weight

    mid_shrink = nn.Conv2d(d, s, 1)
    sd["mid_part.0.weight"] = mid_shrink.weight
    sd["mid_part.0.bias"] = mid_shrink.bias
    sd["mid_part.1.weight"] = nn.PReLU(s).weight

    for i in range(m):
        idx = 2 + 2 * i
        conv = nn.Conv2d(s, s, 3, padding=1)
        sd[f"mid_part.{idx}.weight"] = conv.weight
        sd[f"mid_part.{idx}.bias"] = conv.bias
        sd[f"mid_part.{idx + 1}.weight"] = nn.PReLU(s).weight

    mid_expand = nn.Conv2d(s, d, 1)
    sd[f"mid_part.{2 + 2 * m}.weight"] = mid_expand.weight
    sd[f"mid_part.{2 + 2 * m}.bias"] = mid_expand.bias
    sd[f"mid_part.{2 + 2 * m + 1}.weight"] = nn.PReLU(d).weight

    last = nn.ConvTranspose2d(d, 1, 9, stride=scale, padding=4, output_padding=scale - 1)
    sd["last_part.weight"] = last.weight
    sd["last_part.bias"] = last.bias
    return sd


def test_remap_espcn_loads_strictly_and_preserves_values():
    src_sd = _make_yjn870_espcn_state_dict(scale=3)
    remapped = remap_espcn_state_dict(src_sd)

    model = _build_espcn(scale=3)
    model.load_state_dict(remapped, strict=True)

    assert torch.equal(model.features[0].weight, src_sd["first_part.0.weight"])
    assert torch.equal(model.to_subpixel.weight, src_sd["last_part.0.weight"])


def test_remap_fsrcnn_loads_strictly_and_preserves_values():
    src_sd = _make_yjn870_fsrcnn_state_dict(scale=4)
    remapped = remap_fsrcnn_state_dict(src_sd, m=4)

    model = _build_fsrcnn(scale=4)
    model.load_state_dict(remapped, strict=True)

    assert torch.equal(model.feature_extraction[0].weight, src_sd["first_part.0.weight"])
    assert torch.equal(model.shrink[0].weight, src_sd["mid_part.0.weight"])
    assert torch.equal(model.expand[0].weight, src_sd["mid_part.10.weight"])
    assert torch.equal(model.deconv.weight, src_sd["last_part.weight"])


def test_remap_fsrcnn_mapping_layers_in_order():
    src_sd = _make_yjn870_fsrcnn_state_dict(scale=4)
    remapped = remap_fsrcnn_state_dict(src_sd, m=4)

    model = _build_fsrcnn(scale=4)
    model.load_state_dict(remapped, strict=True)

    for i in range(4):
        src_idx = 2 + 2 * i
        assert torch.equal(model.mapping[2 * i].weight, src_sd[f"mid_part.{src_idx}.weight"])


def test_remap_espcn_missing_keys_fails_strict_load():
    model = _build_espcn(scale=3)
    with pytest.raises(RuntimeError):
        model.load_state_dict(remap_espcn_state_dict({}), strict=True)


def test_remap_fsrcnn_missing_keys_raises_keyerror():
    with pytest.raises(KeyError):
        remap_fsrcnn_state_dict({}, m=4)
