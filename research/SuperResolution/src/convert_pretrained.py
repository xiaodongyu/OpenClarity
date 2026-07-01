"""
Port yjn870's public pretrained FSRCNN/ESPCN PyTorch checkpoints
(https://github.com/yjn870/FSRCNN-pytorch, https://github.com/yjn870/ESPCN-pytorch)
onto our `_build_fsrcnn`/`_build_espcn` module structure.

The two architectures are numerically identical to the ones in `src/sr_model.py`
(same layer shapes, default d=56/s=12/m=4 for FSRCNN and channels=64 for ESPCN)
but organised into differently-named submodules (`first_part`/`mid_part`/
`last_part` vs. our `feature_extraction`/`shrink`/`mapping`/`expand`/`deconv` and
`features`/`to_subpixel`), so a checkpoint can't be loaded directly — this module
remaps the state-dict keys.

Usage:
    python -m src.convert_pretrained --arch fsrcnn --scale 4 \\
        --output src/weights/fsrcnn_x4_pretrained.pt

    python -m src.convert_pretrained --arch espcn --scale 3 \\
        --source /path/to/espcn_x3.pth --output src/weights/espcn_x3_pretrained.pt
"""
import argparse
import os
import sys
import tempfile
import urllib.request

# yjn870's Dropbox-hosted checkpoints (see repo READMEs). ESPCN is only published
# at x3; FSRCNN is published at x2/x3/x4. Both use the default hyperparameters
# our _BUILDERS also default to, so shapes match without any resizing.
PRETRAINED_URLS = {
    ("fsrcnn", 2): "https://www.dropbox.com/s/1k3dker6g7hz76s/fsrcnn_x2.pth?dl=1",
    ("fsrcnn", 3): "https://www.dropbox.com/s/pm1ed2nyboulz5z/fsrcnn_x3.pth?dl=1",
    ("fsrcnn", 4): "https://www.dropbox.com/s/vsvumpopupdpmmu/fsrcnn_x4.pth?dl=1",
    ("espcn", 3): "https://www.dropbox.com/s/2fl5jz5nw9oiw1f/espcn_x3.pth?dl=1",
}


def download(url: str, dest_path: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with open(dest_path, "wb") as f:
        f.write(data)
    return dest_path


def remap_espcn_state_dict(src_sd: dict) -> dict:
    """yjn870 ESPCN: first_part.{0,2}, last_part.0 -> ours: features.{0,2}, to_subpixel."""
    mapping = {
        "first_part.0.weight": "features.0.weight",
        "first_part.0.bias": "features.0.bias",
        "first_part.2.weight": "features.2.weight",
        "first_part.2.bias": "features.2.bias",
        "last_part.0.weight": "to_subpixel.weight",
        "last_part.0.bias": "to_subpixel.bias",
    }
    return {mapping[k]: v for k, v in src_sd.items() if k in mapping}


def remap_fsrcnn_state_dict(src_sd: dict, m: int = 4) -> dict:
    """yjn870 FSRCNN: first_part/mid_part/last_part -> ours: feature_extraction/shrink/mapping/expand/deconv."""
    remapped = {
        "feature_extraction.0.weight": src_sd["first_part.0.weight"],
        "feature_extraction.0.bias": src_sd["first_part.0.bias"],
        "feature_extraction.1.weight": src_sd["first_part.1.weight"],
        "shrink.0.weight": src_sd["mid_part.0.weight"],
        "shrink.0.bias": src_sd["mid_part.0.bias"],
        "shrink.1.weight": src_sd["mid_part.1.weight"],
        "expand.0.weight": src_sd[f"mid_part.{2 + 2 * m}.weight"],
        "expand.0.bias": src_sd[f"mid_part.{2 + 2 * m}.bias"],
        "expand.1.weight": src_sd[f"mid_part.{2 + 2 * m + 1}.weight"],
        "deconv.weight": src_sd["last_part.weight"],
        "deconv.bias": src_sd["last_part.bias"],
    }
    for i in range(m):
        src_conv_idx = 2 + 2 * i
        remapped[f"mapping.{2 * i}.weight"] = src_sd[f"mid_part.{src_conv_idx}.weight"]
        remapped[f"mapping.{2 * i}.bias"] = src_sd[f"mid_part.{src_conv_idx}.bias"]
        remapped[f"mapping.{2 * i + 1}.weight"] = src_sd[f"mid_part.{src_conv_idx + 1}.weight"]
    return remapped


_REMAPPERS = {"espcn": remap_espcn_state_dict, "fsrcnn": remap_fsrcnn_state_dict}


def convert(arch: str, scale: int, output_path: str, source: str = None) -> str:
    """Download (or load) a pretrained checkpoint, remap it, and validate it loads
    cleanly into our model before saving. Returns `output_path`."""
    import torch

    from src.sr_model import _BUILDERS

    if arch not in _REMAPPERS:
        raise ValueError(f"Unknown SR architecture: {arch!r}")

    tmp_path = None
    if source is None:
        key = (arch, scale)
        if key not in PRETRAINED_URLS:
            raise ValueError(
                f"No known public checkpoint for {arch!r} at scale {scale}; pass --source explicitly"
            )
        tmp_path = tempfile.mktemp(suffix=".pth")
        source = download(PRETRAINED_URLS[key], tmp_path)

    try:
        src_sd = torch.load(source, map_location="cpu")
        remapped = _REMAPPERS[arch](src_sd)

        model = _BUILDERS[arch](scale)
        model.load_state_dict(remapped, strict=True)  # raises on any shape/key mismatch

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), output_path)
        return output_path
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def verify_against_bilinear(weights_path: str, arch: str, scale: int, image_path: str) -> tuple[float, float]:
    """Sanity-check a converted checkpoint: PSNR of the SR model vs. bilinear on a
    real image, both reconstructed from the same synthetic LR input.

    IMPORTANT: the LR input is generated with PIL's BICUBIC filter, not
    cv2.INTER_CUBIC. FSRCNN/ESPCN (like most classical, non-blind SR nets) are
    sensitive to the *exact* downsampling kernel used to create their training
    LR data — yjn870's checkpoints were trained on PIL-bicubic-degraded LR.
    Benchmarking with cv2-bicubic LR instead (as `src/benchmark.py` and
    `src/train.py` do internally, which is fine since they're self-consistent
    end-to-end) makes an externally pretrained checkpoint look *worse than
    bilinear* here, even though the checkpoint is correct — a kernel-mismatch
    artifact, not a model defect. Any future fine-tuning (see
    docs/finetuning_plan.md) must keep the LR-generation kernel consistent
    between whatever data it's trained on and whatever it's evaluated on.
    """
    import cv2
    import numpy as np
    import PIL.Image as pil_image

    from src.enhance import bilinear_upscale
    from src.sr_model import SRModel
    from src.train import psnr

    img_bgr = cv2.imread(image_path)
    h, w = img_bgr.shape[:2]
    hh, ww = h - h % scale, w - w % scale
    hr = img_bgr[:hh, :ww]

    hr_rgb = pil_image.fromarray(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
    lr_rgb = hr_rgb.resize((ww // scale, hh // scale), resample=pil_image.BICUBIC)
    lr = cv2.cvtColor(np.array(lr_rgb), cv2.COLOR_RGB2BGR)

    model = SRModel(scale=scale, arch=arch, weights_path=weights_path)
    sr_psnr = psnr(model.upscale(lr), hr)
    bilinear_psnr = psnr(bilinear_upscale(lr, scale), hr)
    return sr_psnr, bilinear_psnr


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Remap a public pretrained FSRCNN/ESPCN checkpoint onto our model structure"
    )
    parser.add_argument("--arch", required=True, choices=["espcn", "fsrcnn"])
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source", default=None, help="Local path or URL to the source .pth (default: known public checkpoint)"
    )
    parser.add_argument(
        "--verify-image", default=None, help="If set, sanity-check the converted checkpoint against bilinear on this image"
    )
    args = parser.parse_args(argv)

    output_path = convert(args.arch, args.scale, args.output, source=args.source)
    print(f"[convert_pretrained] wrote {output_path}", file=sys.stderr)

    if args.verify_image:
        sr_psnr, bilinear_psnr = verify_against_bilinear(output_path, args.arch, args.scale, args.verify_image)
        print(
            f"[convert_pretrained] verify: SR PSNR={sr_psnr:.2f} dB vs bilinear PSNR={bilinear_psnr:.2f} dB",
            file=sys.stderr,
        )
        if sr_psnr <= bilinear_psnr:
            print("[convert_pretrained] WARNING: converted model did not beat bilinear", file=sys.stderr)


if __name__ == "__main__":
    main()
