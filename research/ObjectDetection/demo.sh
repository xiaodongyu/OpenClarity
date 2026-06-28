#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

echo "=== ObjectDetection Demo Pre-flight ==="

# Check camera
python3 -c "
import cv2, sys
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('ERROR: Cannot open webcam (device 0)', file=sys.stderr)
    sys.exit(1)
cap.release()
print('Camera: OK')
"

# Check audio output
python3 -c "
import sounddevice as sd, sys
try:
    devices = sd.query_devices()
    print('Audio: OK (', len(devices), 'devices)')
except Exception as e:
    print(f'WARNING: audio check failed: {e}', file=sys.stderr)
"

echo "=== Starting pipeline ==="
exec python3 -m src.pipeline "$@"
