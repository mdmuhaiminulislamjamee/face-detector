"""Multi-scale YuNet inference with conservative geometry and score checks."""

from pathlib import Path

import cv2
import numpy as np


class RobustFaceDetector:
    SCORE_THRESHOLD = 0.88
    MIN_FACE_PIXELS = 18
    DETAIL_SCAN_INTERVAL = 5

    def __init__(self, model_path=None):
        model_path = Path(model_path or Path(__file__).with_name(
            "face_detection_yunet_2023mar.onnx"))
        if not model_path.is_file():
            raise FileNotFoundError(f"Required YuNet model is missing: {model_path}")
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (640, 640), 0.60, 0.3, 5000,
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._gamma_lut = np.rint(
            255 * (np.arange(256) / 255.0) ** 0.72).astype(np.uint8)
        self._frame_index = 0
        self._previous_boxes = []
        self._frame_size = None
        self._missed_region_frames = 0

    def reset(self):
        self._frame_index = 0
        self._previous_boxes = []
        self._frame_size = None
        self._missed_region_frames = 0

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        if self._frame_size != (width, height):
            self.reset()
            self._frame_size = (width, height)
        # The entire image is scanned cheaply on every frame. Native pixels are
        # retained for fresh crop inference; cached boxes are never returned as hits.
        results = self._infer(frame, min(1.0, 640 / max(width, height)), True)
        detail_scan = self._frame_index % self.DETAIL_SCAN_INTERVAL == 0
        self._frame_index += 1
        if detail_scan and max(width, height) > 640:
            results.extend(self._infer(frame, min(1.0, 1280 / max(width, height)), True))
        proposals = self._deduplicate(results)
        accepted = [item for item in proposals if item[1] >= self.SCORE_THRESHOLD]
        regions = [(box, 1.0, []) for box in self._previous_boxes]
        regions.extend(item for item in proposals if item[1] < self.SCORE_THRESHOLD)
        for box, _, _ in self._deduplicate(regions)[:4]:
            if any(self._overlap(box, item[0]) > .3 for item in accepted):
                continue
            accepted.extend(self._refine_region(frame, box))
        # Expensive backlight search is only needed while searching, at the
        # detailed-scan cadence. Once found, the face gets a new crop check each frame.
        if detail_scan and not accepted:
            scale = min(1.0, 960 / max(width, height))
            small = cv2.resize(frame, (max(32, round(width * scale)),
                                       max(32, round(height * scale))))
            enhanced = self.enhance_shadows(small)
            if enhanced is not None:
                candidates = self._infer(enhanced, 1.0, True)
                sx, sy = width / small.shape[1], height / small.shape[0]
                for box, _, _ in candidates[:4]:
                    mapped = [round(box[0] * sx), round(box[1] * sy),
                              round(box[2] * sx), round(box[3] * sy)]
                    accepted.extend(self._refine_region(frame, mapped))
        accepted = self._deduplicate(accepted)
        # Keep the crop context steady; resizing a slightly different rectangle
        # every frame can make otherwise stationary small-face scores fluctuate.
        if accepted:
            self._previous_boxes = [self._stable_region(item[0], regions)
                                    for item in accepted]
            self._missed_region_frames = 0
        else:
            # A missed region remains a SEARCH hint for a few fresh crop checks.
            # It never contributes a detection or a confirmation hit by itself.
            self._missed_region_frames += 1
            if self._missed_region_frames > 3:
                self._previous_boxes = []
        return accepted

    def _stable_region(self, box, regions):
        x, y, width, height = box
        for old, _, _ in regions:
            ox, oy, ow, oh = old
            if (self._overlap(box, old) > .3 and .7 <= width / ow <= 1.4
                    and .7 <= height / oh <= 1.4
                    and abs(x + width / 2 - ox - ow / 2) < ow * .25
                    and abs(y + height / 2 - oy - oh / 2) < oh * .25):
                return old
        return box

    def _refine_region(self, frame, box):
        height, width = frame.shape[:2]
        x, y, bw, bh = box
        margin = max(bw, bh) * .65
        left, top = max(0, int(x - margin)), max(0, int(y - margin))
        right, bottom = min(width, int(x + bw + margin)), min(height, int(y + bh + margin))
        crop = frame[top:bottom, left:right]
        scale = min(4.0, 192 / max(crop.shape[:2]))
        enhanced = None
        results = self._infer(crop, scale)
        if not results:
            enhanced = self.enhance_shadows(crop)
            if enhanced is not None:
                results = self._infer(enhanced, scale)
        if not results:
            results = self._infer(crop, min(4.0, 320 / max(crop.shape[:2])))
        if not results and enhanced is not None:
            results = self._infer(enhanced, min(4.0, 320 / max(crop.shape[:2])))
        return [([local[0] + left, local[1] + top, local[2], local[3]], score,
                 [(px + left, py + top) for px, py in landmarks])
                for local, score, landmarks in results]

    def enhance_shadows(self, frame):
        """Lift recoverable shadows while limiting amplification of sensor noise."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        light = lab[:, :, 0]
        sampled = light[::8, ::8]
        # Includes a dark subject against a bright window, even if the mean is high.
        if np.mean((sampled >= 8) & (sampled < 85)) < 0.01:
            return None
        local = self.clahe.apply(light)
        lifted = cv2.LUT(light, self._gamma_lut)
        lab[:, :, 0] = cv2.addWeighted(lifted, 0.65, local, 0.35, 0)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _infer(self, frame, scale, include_proposals=False):
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (max(32, round(width * scale)),
                                     max(32, round(height * scale))))
        sx, sy = resized.shape[1] / width, resized.shape[0] / height
        self.detector.setInputSize((resized.shape[1], resized.shape[0]))
        _, detections = self.detector.detect(resized)
        results = []
        if detections is None:
            return results
        for detection in detections:
            values = detection.astype(float).copy()
            values[0:14:2] /= sx
            values[1:14:2] /= sy
            box, score = values[:4], float(values[14])
            landmarks = values[4:14].reshape(5, 2)
            validation_score = max(score, self.SCORE_THRESHOLD) if include_proposals else score
            if not self.valid_candidate(box, validation_score, landmarks):
                continue
            x, y, bw, bh = box
            left, top = max(0, round(x)), max(0, round(y))
            right, bottom = min(width, round(x + bw)), min(height, round(y + bh))
            if min(right - left, bottom - top) < self.MIN_FACE_PIXELS:
                continue
            results.append(([left, top, right - left, bottom - top], score,
                            [tuple(map(int, np.rint(point))) for point in landmarks]))
        return results

    @classmethod
    def valid_candidate(cls, box, score, landmarks):
        box, landmarks = np.asarray(box), np.asarray(landmarks)
        if (box.shape != (4,) or landmarks.shape != (5, 2)
                or not np.isfinite(box).all() or not np.isfinite(landmarks).all()
                or not np.isfinite(score) or score < cls.SCORE_THRESHOLD):
            return False
        x, y, width, height = box
        if min(width, height) < cls.MIN_FACE_PIXELS or not 0.65 <= height / width <= 1.9:
            return False
        if np.any(landmarks < (x - width * .2, y - height * .2)) or np.any(
                landmarks > (x + width * 1.2, y + height * 1.2)):
            return False
        eye_a, eye_b, nose, mouth_a, mouth_b = landmarks
        eye_distance = np.linalg.norm(eye_b - eye_a)
        if not .15 * width <= eye_distance <= .85 * width:
            return False
        eye_mid = (eye_a + eye_b) / 2
        mouth_mid = (mouth_a + mouth_b) / 2
        down = mouth_mid - eye_mid
        face_height = np.linalg.norm(down)
        if not .12 * height <= face_height <= .85 * height:
            return False
        # Relative geometry works with head roll; no skin-colour or open-eye test.
        if abs(np.dot((eye_b - eye_a) / eye_distance, down / face_height)) > .65:
            return False
        nose_depth = np.dot(nose - eye_mid, down) / (face_height ** 2)
        return bool(-.15 <= nose_depth <= 1.25)

    @staticmethod
    def _overlap(box, other):
        x, y, w, h = box
        ox, oy, ow, oh = other
        area = max(0, min(x + w, ox + ow) - max(x, ox)) * max(
            0, min(y + h, oy + oh) - max(y, oy))
        return area / max(1, w * h + ow * oh - area)

    @classmethod
    def _deduplicate(cls, results):
        kept = []
        for candidate in sorted(results, key=lambda result: result[1], reverse=True):
            duplicate = False
            for other, _, _ in kept:
                if cls._overlap(candidate[0], other) > .3:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept
