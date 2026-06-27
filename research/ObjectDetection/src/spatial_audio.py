import numpy as np

SAMPLE_RATE = 44100
EARCON_DURATION = 0.15  # seconds

EARCON_FREQ: dict[str, float] = {
    "person":     440.0,  # A4
    "chair":      330.0,  # E4
    "laptop":     523.0,  # C5
    "bottle":     392.0,  # G4
    "door":       261.0,  # C4
    "cup":        349.0,  # F4
    "backpack":   466.0,  # Bb4
    "cell phone": 587.0,  # D5
    "suitcase":   311.0,  # Eb4
    "table":      294.0,  # D4
}
DEFAULT_FREQ = 300.0


def centroid_to_pan(cx: int, frame_width: int) -> float:
    """Map horizontal centroid to stereo pan: -1.0 (left) to +1.0 (right)."""
    return (cx / frame_width) * 2 - 1


def bbox_to_proximity(bbox: tuple, frame_width: int, frame_height: int) -> float:
    """Normalised bounding box area: 0.0 (far) to 1.0 (fills frame)."""
    x1, y1, x2, y2 = bbox
    return ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)


def _make_earcon(freq: float, pan: float, proximity: float) -> np.ndarray:
    n = int(SAMPLE_RATE * EARCON_DURATION)
    t = np.linspace(0, EARCON_DURATION, n, endpoint=False)
    mono = np.sin(2 * np.pi * freq * t)
    amplitude = max(0.05, min(1.0, proximity))
    left = mono * amplitude * (1 - max(0.0, pan)) / 1.0
    right = mono * amplitude * (1 - max(0.0, -pan)) / 1.0
    # normalise so pan=-1 → full left, pan=+1 → full right
    left_gain = (1.0 - pan) / 2.0
    right_gain = (1.0 + pan) / 2.0
    stereo = np.column_stack([mono * amplitude * left_gain,
                               mono * amplitude * right_gain])
    return stereo.astype(np.float32)


def emit(detections: list[dict], frame_dims: tuple[int, int]) -> None:
    if not detections:
        return
    frame_width, frame_height = frame_dims
    n = int(SAMPLE_RATE * EARCON_DURATION)
    mix = np.zeros((n, 2), dtype=np.float32)
    for det in detections:
        freq = EARCON_FREQ.get(det["label"], DEFAULT_FREQ)
        cx, _ = det["centroid"]
        pan = centroid_to_pan(cx, frame_width)
        proximity = bbox_to_proximity(det["bbox"], frame_width, frame_height)
        mix += _make_earcon(freq, pan, proximity)
    # soft clip
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix /= peak
    import sounddevice as sd
    sd.play(mix, samplerate=SAMPLE_RATE, blocking=False)
