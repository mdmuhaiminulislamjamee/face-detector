"""Multi-scale YuNet inference with conservative geometry and score checks."""

from pathlib import Path
import time

import cv2
import numpy as np


class RobustFaceDetector:
    SCORE_THRESHOLD = 0.88
    MIN_FACE_PIXELS = 18
    DETAIL_SCAN_INTERVAL = 5
    POSE_SCORE_THRESHOLD = .94
    RECOVERY_SCORE_THRESHOLD = SCORE_THRESHOLD
    # Rotate into a more upright view and compensate moderate foreshortening.
    # Only two alternatives are searched per frame, never this whole list.
    POSE_TRANSFORMS = ((0, 1, 1.35), (25, 1, 1), (-25, 1, 1),
                       (25, 1, 1.35), (-25, 1, 1.35), (0, 1.35, 1),
                       (50, 1, 1), (-50, 1, 1), (90, 1, 1),
                       (-90, 1, 1), (180, 1, 1), (0, 1, .75))

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
        self._pose_index = 0
        self._pose_hints = []

    def reset(self):
        self._frame_index = 0
        self._previous_boxes = []
        self._frame_size = None
        self._missed_region_frames = 0
        self._pose_index = 0
        self._pose_hints = []

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        started = time.perf_counter()
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
        accepted = [item for item in proposals if self.valid_candidate(*item)]
        for candidate in proposals[:2]:
            if any(self._overlap(candidate[0], item[0]) > .3 for item in accepted):
                continue
            if time.perf_counter() - started >= .070:
                break
            box, _, points = candidate
            descriptor = self._landmark_descriptor(box, points)
            rectified = self._infer_landmark_pose(frame, box, descriptor)
            if rectified:
                self._remember_pose(rectified[0][0], self._landmark_descriptor(
                    rectified[0][0], rectified[0][2]))
                accepted.extend(rectified)
        regions = [(box, 1.0, []) for box in self._previous_boxes]
        regions.extend(item for item in proposals if not self.valid_candidate(*item))
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
        # Search missing orientations using a small, scheduled inference budget.
        # Once acquired, the pose is freshly checked on its native crop each frame.
        if (not accepted or detail_scan) and time.perf_counter() - started < .070:
            accepted.extend(self._search_poses(frame))
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
                self._pose_hints = []
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

    def _refine_region(self, frame, box, allow_pose=True):
        height, width = frame.shape[:2]
        x, y, bw, bh = box
        margin = max(bw, bh) * .65
        left, top = max(0, int(x - margin)), max(0, int(y - margin))
        right, bottom = min(width, int(x + bw + margin)), min(height, int(y + bh + margin))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return []
        for index, (hint_box, transform) in enumerate(self._pose_hints if allow_pose else []):
            if self._overlap(box, hint_box) <= .3:
                continue
            # This is inference on the current image, not reuse of an old hit.
            if transform[0] == 'landmarks':
                pose_results = self._infer_landmark_pose(crop,
                    [hint_box[0] - left, hint_box[1] - top,
                     hint_box[2], hint_box[3]], transform)
            else:
                pose_results = self._infer_pose(crop, transform, 320)
            if pose_results:
                mapped_results = [([local[0] + left, local[1] + top, local[2], local[3]], score,
                         [(px + left, py + top) for px, py in landmarks])
                        for local, score, landmarks in pose_results]
                if transform[0] == 'landmarks':
                    transform = self._landmark_descriptor(mapped_results[0][0], mapped_results[0][2])
                self._pose_hints[index] = (mapped_results[0][0], transform)
                return mapped_results
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
            validation_score = max(score, self.POSE_SCORE_THRESHOLD) if include_proposals else score
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
        if min(width, height) < cls.MIN_FACE_PIXELS or not .45 <= height / width <= 2.4:
            return False
        if np.any(landmarks < (x - width * .2, y - height * .2)) or np.any(
                landmarks > (x + width * 1.2, y + height * 1.2)):
            return False
        eye_a, eye_b, nose, mouth_a, mouth_b = landmarks
        eye_distance = np.linalg.norm(eye_b - eye_a)
        if not .08 * width <= eye_distance <= .85 * width:
            return False
        eye_mid = (eye_a + eye_b) / 2
        mouth_mid = (mouth_a + mouth_b) / 2
        down = mouth_mid - eye_mid
        face_height = np.linalg.norm(down)
        if not .08 * height <= face_height <= .90 * height:
            return False
        # Relative geometry works with head roll; no skin-colour or open-eye test.
        skew = abs(np.dot((eye_b - eye_a) / eye_distance, down / face_height))
        nose_depth = np.dot(nose - eye_mid, down) / (face_height ** 2)
        if (.65 <= height / width <= 1.9 and eye_distance >= .15 * width
                and .12 * height <= face_height <= .85 * height
                and skew <= .65 and -.15 <= nose_depth <= 1.25):
            return True
        # Difficult pitch/yaw gets a separate, HIGHER confidence gate, plus mouth
        # topology checks. Collapsed eyes and arbitrary landmark clouds still fail.
        mouth_line = mouth_b - mouth_a
        mouth_distance = np.linalg.norm(mouth_line)
        if (score < cls.POSE_SCORE_THRESHOLD or skew > .9
                or not -.4 <= nose_depth <= 1.35
                or not .05 * width <= mouth_distance <= .85 * width):
            return False
        aligned_mouth = np.dot(eye_b - eye_a, mouth_line) / (eye_distance * mouth_distance)
        handedness = ((eye_b[0] - eye_a[0]) * down[1] -
                      (eye_b[1] - eye_a[1]) * down[0]) / (eye_distance * face_height)
        return bool(aligned_mouth >= .2 and handedness >= .2)

    @staticmethod
    def _pose_matrix(width, height, transform):
        angle, sx, sy = transform
        rotation = cv2.getRotationMatrix2D((0, 0), angle, 1)
        rotation[0] *= sx
        rotation[1] *= sy
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float64)
        projected = corners @ rotation[:, :2].T
        minimum, maximum = projected.min(axis=0), projected.max(axis=0)
        rotation[:, 2] = -minimum
        size = tuple(np.maximum(32, np.ceil(maximum - minimum - 1e-9)).astype(int))
        return rotation, size

    @staticmethod
    def _landmark_descriptor(box, points):
        x, y, width, height = box
        eye_a, eye_b, _, mouth_a, mouth_b = np.asarray(points, dtype=float)
        anchor = np.array([eye_a, eye_b, (mouth_a + mouth_b) / 2])
        normalized = (anchor - (x, y)) / (width, height)
        return ('landmarks', normalized)

    def _infer_landmark_pose(self, frame, box, descriptor):
        x, y, width, height = box
        anchors = descriptor[1] * (width, height) + (x, y)
        eye_line, down = anchors[1] - anchors[0], anchors[2] - (anchors[0] + anchors[1]) / 2
        determinant = eye_line[0] * down[1] - eye_line[1] * down[0]
        if determinant < max(4, width * height * .008):
            return []
        target = np.float32([[64, 72], [128, 72], [96, 136]])
        matrix = cv2.getAffineTransform(anchors.astype(np.float32), target)
        warped = cv2.warpAffine(frame, matrix, (192, 224), borderValue=(127, 127, 127))
        candidates = self._infer(warped, 1.0)
        if not candidates:
            enhanced = self.enhance_shadows(warped)
            if enhanced is not None:
                candidates = self._infer(enhanced, 1.0)
        inverse = cv2.invertAffineTransform(matrix)
        mapped = self._map_pose_results(frame, candidates, inverse)
        return [item for item in mapped if self._overlap(box, item[0]) > .25]

    def _remember_pose(self, box, transform):
        self._pose_hints = [(old, pose) for old, pose in self._pose_hints
                            if self._overlap(old, box) <= .3]
        self._pose_hints.append((box, transform))
        self._pose_hints = self._pose_hints[-4:]

    def _infer_pose(self, frame, transform, max_side=640):
        height, width = frame.shape[:2]
        scale = min(1.0, max_side / max(width, height))
        small = cv2.resize(frame, (max(32, round(width * scale)),
                                   max(32, round(height * scale))))
        matrix, size = self._pose_matrix(small.shape[1], small.shape[0], transform)
        warped = cv2.warpAffine(small, matrix, size, borderValue=(127, 127, 127))
        proposals = self._infer(warped, min(1.0, max_side / max(warped.shape[:2])), True)
        if not proposals:
            enhanced = self.enhance_shadows(warped)
            if enhanced is not None:
                warped = enhanced
                proposals = self._infer(warped, min(1.0, max_side / max(warped.shape[:2])), True)
        candidates = []
        for candidate in proposals[:2]:
            if candidate[1] >= self.RECOVERY_SCORE_THRESHOLD and self.valid_candidate(*candidate):
                candidates.append(candidate)
            else:
                # A weak rotated view is a search proposal only. Re-run the
                # native crop in its upright coordinates before accepting it.
                candidates.extend(self._refine_region(warped, candidate[0], allow_pose=False))
                if not candidates:
                    descriptor = self._landmark_descriptor(candidate[0], candidate[2])
                    candidates.extend(self._infer_landmark_pose(warped, candidate[0], descriptor))
        inverse = cv2.invertAffineTransform(matrix)
        inverse[0] *= width / small.shape[1]
        inverse[1] *= height / small.shape[0]
        return self._map_pose_results(frame, candidates, inverse)

    def _map_pose_results(self, frame, candidates, inverse):
        height, width = frame.shape[:2]
        results = []
        for box, score, landmarks in candidates:
            if score < self.RECOVERY_SCORE_THRESHOLD:
                continue
            x, y, bw, bh = box
            corners = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
            mapped = corners @ inverse[:, :2].T + inverse[:, 2]
            lower, upper = mapped.min(axis=0), mapped.max(axis=0)
            left, top = max(0, round(lower[0])), max(0, round(lower[1]))
            right, bottom = min(width, round(upper[0])), min(height, round(upper[1]))
            if min(right - left, bottom - top) < self.MIN_FACE_PIXELS:
                continue
            mapped_points = np.asarray(landmarks) @ inverse[:, :2].T + inverse[:, 2]
            # Geometry was verified in the upright view. Re-validating against
            # its enlarged axis-aligned rotated box would reject valid poses again.
            results.append(([left, top, right - left, bottom - top], score,
                            [tuple(map(int, np.rint(point))) for point in mapped_points]))
        return results

    def _search_poses(self, frame):
        results = []
        deadline = time.perf_counter() + .025
        for attempt in range(2):
            if attempt and time.perf_counter() >= deadline:
                break
            transform = self.POSE_TRANSFORMS[self._pose_index % len(self.POSE_TRANSFORMS)]
            self._pose_index += 1
            found = self._infer_pose(frame, transform)
            for box, _, points in found:
                self._remember_pose(box, self._landmark_descriptor(box, points))
            results.extend(found)
            if found:
                break
        return results

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
