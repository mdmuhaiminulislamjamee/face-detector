"""Synthetic pose regression/latency probe, not a substitute for real 3-D views."""
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robust_face_detector import RobustFaceDetector


def pose_scene(portrait, angle=0, pitch=0, yaw=0, brightness=1):
    side = 384
    image = cv2.resize(portrait, (side, side))
    image = (image.astype(float) * brightness).astype(np.uint8)
    corners = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    target = corners.copy()
    if pitch > 0:
        target[:2] += [[side * pitch, 0], [-side * pitch, 0]]
    elif pitch < 0:
        target[2:] += [[side * pitch, 0], [-side * pitch, 0]]
    if yaw > 0:
        target[1:3] += [[0, side * yaw], [0, -side * yaw]]
    elif yaw < 0:
        target[[0, 3]] += [[0, -side * yaw], [0, side * yaw]]
    perspective = cv2.getPerspectiveTransform(corners, target)
    rotated = np.vstack([cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1), [0, 0, 1]])
    image = cv2.warpPerspective(image, rotated @ perspective, (side, side),
                               borderValue=(215, 215, 215))
    frame = np.full((720, 1280, 3), 215, np.uint8)
    frame[160:160 + side, 450:450 + side] = image
    return frame


def benchmark():
    portrait = cv2.imread(str(Path(__file__).with_name('fixtures') / 'face.jpg'))
    scenarios = [(f'roll_{angle}', angle, 0, 0, 1) for angle in (-90, -60, -40, -25, 0, 25, 40, 60, 90, 180)]
    scenarios += [(f'pitch_{pitch}', 0, pitch, 0, 1) for pitch in (-.25, .25)]
    scenarios += [(f'yaw_{yaw}', 0, 0, yaw, 1) for yaw in (-.25, .25)]
    scenarios += [('low_tilt_backlit', -25, .25, 0, .3)]
    results = {}
    detector = RobustFaceDetector()
    for name, angle, pitch, yaw, brightness in scenarios:
        detector.reset()
        frame = pose_scene(portrait, angle, pitch, yaw, brightness)
        durations, hits = [], []
        for index in range(12):
            started = time.perf_counter()
            found = detector.detect(frame)
            durations.append((time.perf_counter() - started) * 1000)
            hits.append(bool(found))
        results[name] = {'hits': sum(hits), 'frames': len(hits),
                         'first_hit': next((i for i, hit in enumerate(hits) if hit), None),
                         'median_ms': round(float(np.median(durations)), 2),
                         'p95_ms': round(float(np.percentile(durations, 95)), 2)}
    return results


if __name__ == '__main__':
    print(json.dumps(benchmark(), indent=2))
