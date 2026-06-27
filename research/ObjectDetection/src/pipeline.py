import argparse
import sys
import time

import cv2

from src.capture import capture_frame, get_frame_dims, release as release_camera
from src.detector import detect
from src.priority_filter import top_n
from src.spatial_audio import emit

TARGET_FPS = 10
FRAME_BUDGET = 1.0 / TARGET_FPS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ObjectDetection spatial audio pipeline")
    p.add_argument("--visualise", action="store_true",
                   help="Show bounding box overlay with cv2.imshow")
    p.add_argument("--no-audio", action="store_true",
                   help="Print detections to stdout instead of emitting earcons")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Detection confidence threshold (default: 0.5)")
    return p.parse_args(argv)


def run(args=None):
    if args is None:
        args = parse_args()

    frame_dims = get_frame_dims()

    try:
        while True:
            t0 = time.perf_counter()

            frame = capture_frame()
            detections = detect(frame, conf=args.conf)
            filtered = top_n(detections)

            if args.no_audio:
                for d in filtered:
                    print(d)
            else:
                emit(filtered, frame_dims)

            if args.visualise:
                for d in filtered:
                    x1, y1, x2, y2 = d["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{d['label']} {d['confidence']:.2f}",
                                (x1, max(y1 - 6, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("ObjectDetection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.perf_counter() - t0
            sleep_time = FRAME_BUDGET - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        release_camera()
        if args.visualise:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
