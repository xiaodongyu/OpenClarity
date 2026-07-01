"""
Lightweight super-resolution model wrapper (FSRCNN / ESPCN).

torch is imported lazily inside the functions below so that importing this
module (and mocking it in tests) never requires torch to be installed.
"""
import sys
import time
from typing import Optional

import cv2
import numpy as np


def _build_fsrcnn(scale: int, d: int = 56, s: int = 12, m: int = 4):
    """FSRCNN: feature extraction -> shrink -> mapping -> expand -> deconv upsample."""
    import torch.nn as nn

    class FSRCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extraction = nn.Sequential(nn.Conv2d(1, d, 5, padding=2), nn.PReLU(d))
            self.shrink = nn.Sequential(nn.Conv2d(d, s, 1), nn.PReLU(s))
            layers = []
            for _ in range(m):
                layers += [nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)]
            self.mapping = nn.Sequential(*layers)
            self.expand = nn.Sequential(nn.Conv2d(s, d, 1), nn.PReLU(d))
            self.deconv = nn.ConvTranspose2d(
                d, 1, 9, stride=scale, padding=4, output_padding=scale - 1
            )

        def forward(self, x):
            x = self.feature_extraction(x)
            x = self.shrink(x)
            x = self.mapping(x)
            x = self.expand(x)
            return self.deconv(x)

    return FSRCNN()


def _build_espcn(scale: int, channels: int = 64):
    """ESPCN: conv feature extraction in LR space + sub-pixel (pixel-shuffle) upsample."""
    import torch.nn as nn

    class ESPCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, channels, 5, padding=2),
                nn.Tanh(),
                nn.Conv2d(channels, channels // 2, 3, padding=1),
                nn.Tanh(),
            )
            self.to_subpixel = nn.Conv2d(channels // 2, scale**2, 3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(scale)

        def forward(self, x):
            x = self.features(x)
            x = self.to_subpixel(x)
            return self.pixel_shuffle(x)

    return ESPCN()


_BUILDERS = {"fsrcnn": _build_fsrcnn, "espcn": _build_espcn}


class SRModel:
    """Inference wrapper around a lightweight SR network.

    Operates on the luma (Y) channel only — standard practice for these small
    SR networks — and upsamples chroma (Cr/Cb) via bicubic interpolation,
    since the eye is far less sensitive to chroma detail than luma detail.
    """

    def __init__(
        self,
        scale: int = 4,
        arch: str = "espcn",
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ):
        if arch not in _BUILDERS:
            raise ValueError(f"Unknown SR architecture: {arch!r} (expected one of {list(_BUILDERS)})")
        import torch

        self.scale = scale
        self.arch = arch
        self.device = device
        self.model = _BUILDERS[arch](scale)
        if weights_path:
            state = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state)
        self.model.to(device).eval()

    def upscale(self, img: np.ndarray) -> np.ndarray:
        import torch

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        tensor = (
            torch.from_numpy(y.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        start = time.perf_counter()
        with torch.no_grad():
            out = self.model(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[sr_model] {self.arch} inference: {elapsed_ms:.1f} ms", file=sys.stderr)

        y_hr = (out.squeeze(0).squeeze(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        h, w = y_hr.shape
        cr_hr = cv2.resize(cr, (w, h), interpolation=cv2.INTER_CUBIC)
        cb_hr = cv2.resize(cb, (w, h), interpolation=cv2.INTER_CUBIC)
        merged = cv2.merge([y_hr, cr_hr, cb_hr])
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
