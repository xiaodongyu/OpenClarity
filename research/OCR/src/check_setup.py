"""
Setup verification script.

Run before the demo to confirm camera, PaddleOCR model, and pyttsx3 are all working.

Usage:
    python src/check_setup.py
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


def check_paddleocr():
    try:
        from paddleocr import PaddleOCR
        PaddleOCR(use_angle_cls=True, lang="en")
        return True, "PaddleOCR model loaded OK"
    except ImportError:
        return False, "paddleocr package not installed (pip install paddleocr)"
    except Exception as exc:
        return False, f"PaddleOCR error: {exc}"


def check_pyttsx3():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.stop()
        return True, "pyttsx3 OK"
    except ImportError:
        return False, "pyttsx3 not installed (pip install pyttsx3)"
    except Exception as exc:
        return False, f"pyttsx3 error: {exc}"


def main():
    checks = [
        ("Camera", check_camera),
        ("PaddleOCR", check_paddleocr),
        ("pyttsx3", check_pyttsx3),
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
