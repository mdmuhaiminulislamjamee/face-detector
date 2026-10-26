"""Synthetic 1080p inference benchmark; does not include webcam/display latency."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robust_face_detector import RobustFaceDetector


def benchmark(frame_count=60):
    portrait = cv2.imread(str(Path(__file__).with_name('fixtures') / 'face.jpg'))
    results = {}
    for name, side, brightness in [('near', 720, 1),
                                   ('distant_backlit', 192, .23),
                                   ('empty', 0, 1)]:
        frame = np.full((1080, 1920, 3), 215, np.uint8)
        if side:
            frame[180:180 + side, 800:800 + side] = (
                cv2.resize(portrait, (side, side)) * brightness).astype(np.uint8)
        detector = RobustFaceDetector()
        detector.detect(frame)  # Warm-up (model allocation is reported separately by real startup).
        durations, correct = [], 0
        for _ in range(frame_count):
            start = time.perf_counter()
            faces = detector.detect(frame)
            durations.append((time.perf_counter() - start) * 1000)
            correct += len(faces) == int(side > 0)
        mean = statistics.mean(durations)
        results[name] = {
            'median_ms': round(statistics.median(durations), 2),
            'p95_ms': round(float(np.percentile(durations, 95)), 2),
            'mean_ms': round(mean, 2),
            'inference_fps': round(1000 / mean, 1),
            'correct_frames': correct,
            'frames': frame_count,
        }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=60)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    print(json.dumps(benchmark(args.frames), indent=2))
