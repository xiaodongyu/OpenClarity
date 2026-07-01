"""
Training / fine-tuning loop for the SR model (FSRCNN or ESPCN).

Trains on HR image crops (DIV2K by default) plus an optional directory of
text/label crops so the model is biased toward sharp glyph edges, which
matter most for reading zoomed-in text.

Usage:
    python -m src.train --data-dir data/hr_crops --arch espcn --scale 4 \\
        --epochs 50 --output src/weights/espcn_x4.pt
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training: pip install torch") from exc
    return torch


class HRCropDataset:
    """Loads HR image crops and synthesises LR/HR training pairs on the fly.

    LR is produced by bicubic downsampling the HR patch by `scale`, mirroring
    the standard FSRCNN/ESPCN training recipe.
    """

    def __init__(self, image_dir: str, scale: int = 4, patch_size: int = 128):
        self.paths = sorted(
            p
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            for p in glob.glob(os.path.join(image_dir, ext))
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.scale = scale
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.paths)

    def _random_patch(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            img = cv2.resize(img, (max(w, ps), max(h, ps)))
            h, w = img.shape[:2]
        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)
        return img[top : top + ps, left : left + ps]

    def __getitem__(self, idx: int):
        torch = _require_torch()
        img = cv2.imread(self.paths[idx])
        patch = self._random_patch(img)
        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        hr_y = ycrcb[:, :, 0]

        ps = self.patch_size
        lr_y = cv2.resize(
            hr_y, (ps // self.scale, ps // self.scale), interpolation=cv2.INTER_CUBIC
        )

        hr_tensor = torch.from_numpy(hr_y.astype(np.float32) / 255.0).unsqueeze(0)
        lr_tensor = torch.from_numpy(lr_y.astype(np.float32) / 255.0).unsqueeze(0)
        return lr_tensor, hr_tensor


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def train(
    data_dir: str,
    arch: str = "espcn",
    scale: int = 4,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.1,
    output_path: str = "src/weights/model.pt",
    device: str = "cpu",
) -> str:
    """Fine-tune an SR model; saves the best-validation-PSNR checkpoint to `output_path`."""
    torch = _require_torch()
    from torch.utils.data import DataLoader, random_split

    from src.sr_model import _BUILDERS

    dataset = HRCropDataset(data_dir, scale=scale)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = _BUILDERS[arch](scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()  # more robust to outliers than L2; preserves sharp edges

    best_psnr = -float("inf")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            optimizer.zero_grad()
            pred = model(lr_batch)
            loss = criterion(pred, hr_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        psnrs = []
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                pred = model(lr_batch).clamp(0, 1).cpu().numpy() * 255
                target = hr_batch.cpu().numpy() * 255
                for p, t in zip(pred, target):
                    psnrs.append(psnr(p, t))
        mean_psnr = float(np.mean(psnrs))
        print(f"[train] epoch {epoch + 1}/{epochs} val PSNR: {mean_psnr:.2f} dB", file=sys.stderr)

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save(model.state_dict(), output_path)

    print(f"[train] best val PSNR: {best_psnr:.2f} dB -> saved {output_path}", file=sys.stderr)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train/fine-tune the SR model")
    parser.add_argument("--data-dir", required=True, help="Directory of HR image crops")
    parser.add_argument("--arch", default="espcn", choices=["espcn", "fsrcnn"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="src/weights/model.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    train(
        data_dir=args.data_dir,
        arch=args.arch,
        scale=args.scale,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
