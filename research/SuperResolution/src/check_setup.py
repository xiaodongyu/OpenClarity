"""
Setup verification script.

Run before the demo to confirm camera, PyTorch, and the SR model all work.

Usage:
    python -m src.check_setup
"""
import sys


def check_camera():
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Camera device 0 not accessible"
    ret, _ = cap.read()
    cap.release()
    if not ret:
        return False, "Camera opened but failed to read a frame"
    return True, "Camera OK"


def check_torch():
    try:
        import torch

        return True, f"PyTorch OK ({torch.__version__})"
    except ImportError:
        return False, "torch not installed (pip install torch)"


def check_sr_model():
    try:
        from src.sr_model import SRModel

        SRModel(scale=4, arch="espcn")
        return True, "SR model (ESPCN) builds OK"
    except ImportError:
        return False, "torch not installed (pip install torch)"
    except Exception as exc:
        return False, f"SR model error: {exc}"


def main():
    checks = [
        ("Camera", check_camera),
        ("PyTorch", check_torch),
        ("SR model", check_sr_model),
    ]
    all_ok = True
    for name, fn in checks:
        ok, msg = fn()
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        if not ok:
            all_ok = False
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
