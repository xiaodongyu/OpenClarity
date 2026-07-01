"""
Export a trained SR model to ONNX and apply INT8 post-training quantization
for edge/mobile deployment.

Usage:
    python -m src.export --weights src/weights/espcn_x4.pt --arch espcn --scale 4 \\
        --output src/weights/espcn_x4.onnx
"""
import argparse
import os


def export_onnx(
    weights_path: str,
    arch: str,
    scale: int,
    output_path: str,
    input_size: tuple[int, int] = (64, 64),
) -> str:
    import torch

    from src.sr_model import _BUILDERS

    model = _BUILDERS[arch](scale)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 1, *input_size)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["lr_y"],
        output_names=["hr_y"],
        dynamic_axes={"lr_y": {2: "height", 3: "width"}, "hr_y": {2: "height", 3: "width"}},
        opset_version=13,
    )
    return output_path


def quantize_int8(onnx_path: str, output_path: str) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)
    return output_path


def max_psnr_delta(
    fp32_path: str, int8_path: str, val_dir: str, scale: int
) -> float:
    """Return the largest PSNR drop of the quantized model vs. FP32 across a
    validation crop directory. Used to gate acceptable quantization loss."""
    import glob

    import cv2
    import numpy as np
    import onnxruntime as ort

    from src.train import psnr

    fp32_sess = ort.InferenceSession(fp32_path)
    int8_sess = ort.InferenceSession(int8_path)
    input_name = fp32_sess.get_inputs()[0].name

    deltas = []
    for path in sorted(glob.glob(os.path.join(val_dir, "*"))):
        img = cv2.imread(path)
        if img is None:
            continue
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        hr_y = ycrcb[:, :, 0]
        h, w = hr_y.shape
        lr_y = cv2.resize(hr_y, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        tensor = (lr_y.astype(np.float32) / 255.0)[None, None, :, :]

        fp32_out = fp32_sess.run(None, {input_name: tensor})[0]
        int8_out = int8_sess.run(None, {input_name: tensor})[0]

        fp32_y = np.clip(fp32_out[0, 0] * 255, 0, 255)
        int8_y = np.clip(int8_out[0, 0] * 255, 0, 255)
        deltas.append(psnr(fp32_y, hr_y.astype(np.float64)) - psnr(int8_y, hr_y.astype(np.float64)))

    return max(deltas) if deltas else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export SR model to ONNX (+ INT8 quantized)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--arch", default="espcn", choices=["espcn", "fsrcnn"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--output", required=True, help="Path for the FP32 ONNX model")
    parser.add_argument("--quantize", action="store_true", help="Also emit an INT8 quantized model")
    args = parser.parse_args(argv)

    onnx_path = export_onnx(args.weights, args.arch, args.scale, args.output)
    print(f"[export] wrote {onnx_path}")

    if args.quantize:
        int8_path = onnx_path.replace(".onnx", "_int8.onnx")
        quantize_int8(onnx_path, int8_path)
        print(f"[export] wrote {int8_path}")


if __name__ == "__main__":
    main()
