"""NCM-sized backlight/pose probe; synthetic views do not model 3-D occlusion."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robust_face_detector import RobustFaceDetector


def backlit_scene(portrait, side=144, brightness=.15, pitch=.2, x=245, y=100):
    image = cv2.resize(portrait, (side, side))
    image = np.clip(image.astype(float) * brightness, 0, 255).astype(np.uint8)
    corners = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    target = corners.copy()
    target[:2] += [[side * pitch, 0], [-side * pitch, 0]]
    image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(corners, target),
                               (side, side), borderValue=(220, 220, 220))
    scene = np.full((480, 640, 3), 220, np.uint8)
    scene[y:y + side, x:x + side] = image
    ok, jpeg = cv2.imencode('.jpg', scene, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return cv2.imdecode(jpeg, cv2.IMREAD_COLOR)


def benchmark(frames=12):
    portrait = cv2.imread(str(Path(__file__).with_name('fixtures') / 'face.jpg'))
    results = {}
    cv2.setNumThreads(2)
    for side, brightness, pitch in ((144, .15, .2), (128, .18, .2),
                                    (112, .23, .2), (128, .15, 0),
                                    (144, .1, .15), (160, .18, .3)):
        scene = backlit_scene(portrait, side, brightness, pitch)
        detector = RobustFaceDetector()
        durations, hits = [], []
        for _ in range(frames):
            started = time.perf_counter()
            found = detector.detect(scene)
            durations.append((time.perf_counter() - started) * 1000)
            hits.append(bool(found))
        results[f'{side}/{brightness}/{pitch}'] = {
            'hits': sum(hits), 'frames': frames,
            'first_hit': next((i for i, hit in enumerate(hits) if hit), None),
            'last_six_hits': sum(hits[-6:]),
            'median_ms': round(float(np.median(durations)), 2),
            'p95_ms': round(float(np.percentile(durations, 95)), 2),
        }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=12)
    print(json.dumps(benchmark(parser.parse_args().frames), indent=2))
