"""
Tests for src/benchmark.py's evaluation logic. Uses a fake SR model (matching
the SRModel interface: `.upscale(lr) -> np.ndarray`) so these run without
torch or real weights -- mirrors the approach in test_pipeline.py.
"""
import cv2
import numpy as np
import pytest
from src.benchmark import ModelSpec, evaluate, summarize, write_report


class _FakeSRModel:
    def __init__(self, scale):
        self.scale = scale

    def upscale(self, img):
        h, w = img.shape[:2]
        return cv2.resize(img, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)


@pytest.fixture
def hr_dir(tmp_path):
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "sample.png"), img)
    return str(tmp_path)


def test_model_spec_parse():
    spec = ModelSpec.parse("fsrcnn_x4:fsrcnn:4:src/weights/fsrcnn_x4_pretrained.pt")
    assert spec == ModelSpec("fsrcnn_x4", "fsrcnn", 4, "src/weights/fsrcnn_x4_pretrained.pt")


def test_model_spec_parse_invalid():
    with pytest.raises(ValueError):
        ModelSpec.parse("not-enough-parts")


def test_evaluate_no_scales_raises(hr_dir):
    with pytest.raises(ValueError):
        evaluate(hr_dir)


def test_evaluate_baseline_only(hr_dir):
    results = evaluate(hr_dir, baseline_scales=[4])
    methods = {r["method"] for r in results}
    assert methods == {"bilinear_x4", "adaptive_sharpen_x4"}
    assert all(r["scale"] == 4 for r in results)


def test_evaluate_with_model_at_its_own_scale(hr_dir):
    models_by_scale = {3: {"fake": _FakeSRModel(scale=3)}}
    results = evaluate(hr_dir, models_by_scale=models_by_scale)
    methods = {r["method"] for r in results}
    assert methods == {"bilinear_x3", "adaptive_sharpen_x3", "fake", "fake_hybrid"}
    assert all(r["scale"] == 3 for r in results)


def test_evaluate_decouples_multiple_scales(hr_dir):
    models_by_scale = {
        2: {"fake_x2": _FakeSRModel(scale=2)},
        4: {"fake_x4": _FakeSRModel(scale=4)},
    }
    results = evaluate(hr_dir, models_by_scale=models_by_scale)
    scales_seen = {r["scale"] for r in results}
    assert scales_seen == {2, 4}
    methods_at_2 = {r["method"] for r in results if r["scale"] == 2}
    methods_at_4 = {r["method"] for r in results if r["scale"] == 4}
    assert "fake_x2" in methods_at_2 and "fake_x2" not in methods_at_4
    assert "fake_x4" in methods_at_4 and "fake_x4" not in methods_at_2


def test_evaluate_keep_images(hr_dir):
    results = evaluate(hr_dir, baseline_scales=[4], keep_images=True)
    assert all("output" in r and "hr" in r and "lr" in r for r in results)
    assert results[0]["output"].dtype == np.uint8


def test_summarize_groups_by_scale_and_method(hr_dir):
    results = evaluate(hr_dir, baseline_scales=[2, 4])
    summary = summarize(results)
    assert (2, "bilinear_x2") in summary
    assert (4, "bilinear_x4") in summary
    assert set(summary[(2, "bilinear_x2")]) == {"psnr", "ssim", "latency_ms"}


def test_write_report_groups_by_scale(hr_dir, tmp_path):
    results = evaluate(hr_dir, baseline_scales=[2, 4])
    summary = summarize(results)
    report_path = str(tmp_path / "report.md")
    write_report(summary, report_path)

    with open(report_path) as f:
        content = f.read()
    assert "## Scale x2" in content
    assert "## Scale x4" in content
    assert "bilinear_x2" in content
