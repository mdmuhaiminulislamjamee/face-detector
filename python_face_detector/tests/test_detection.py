"""Run: python -m unittest discover -s tests -v (from python_face_detector)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_face_detector import FaceTrackingStabilizer
from robust_face_detector import RobustFaceDetector


class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = RobustFaceDetector()
        fixtures = Path(__file__).with_name('fixtures')
        cls.face = cv2.imread(str(fixtures / 'face.jpg'))
        cls.non_face = cv2.imread(str(fixtures / 'non_face.jpg'))

    def scene(self, side, brightness=1):
        scene = np.full((720, 1280, 3), 215, dtype=np.uint8)
        portrait = cv2.resize(self.face, (side, side))
        scene[180:180 + side, 500:500 + side] = (
            portrait.astype(float) * brightness).astype(np.uint8)
        return scene

    def test_near_and_distant_faces_with_bright_background(self):
        for side in (512, 256, 128):
            for brightness in (1, .23):
                with self.subTest(side=side, brightness=brightness):
                    self.detector.reset()
                    detections = self.detector.detect(self.scene(side, brightness))
                    self.assertEqual(len(detections), 1)
                    box, score, landmarks = detections[0]
                    self.assertGreaterEqual(score, self.detector.SCORE_THRESHOLD)
                    # Confirm it found the face inside the portrait, not the background.
                    self.assertTrue(500 <= box[0] < 500 + side)
                    self.assertTrue(180 <= box[1] < 180 + side)
                    self.assertEqual(len(landmarks), 5)

    def test_non_face_inputs(self):
        rng = np.random.default_rng(7)
        for name, frame in (
            ('black', np.zeros((360, 640, 3), np.uint8)),
            ('white', np.full((360, 640, 3), 255, np.uint8)),
            ('noise', rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)),
            ('animal', self.non_face),
        ):
            with self.subTest(name=name):
                self.assertEqual(self.detector.detect(frame), [])

    def test_multiple_faces_are_not_merged(self):
        self.detector.reset()
        scene = self.scene(256)
        scene[400:528, 900:1028] = cv2.resize(self.face, (128, 128))
        detections = self.detector.detect(scene)
        self.assertEqual(len(detections), 2)

    def test_invalid_geometry_and_weak_scores_are_rejected(self):
        points = np.array([[125, 130], [175, 130], [150, 155], [130, 180], [170, 180]])
        box = [100, 100, 100, 120]
        self.assertTrue(self.detector.valid_candidate(box, .95, points))
        self.assertFalse(self.detector.valid_candidate(box, .7, points))
        self.assertFalse(self.detector.valid_candidate([0, 0, -1, 120], .95, points))
        self.assertFalse(self.detector.valid_candidate(box, float('nan'), points))
        points[1] = points[0]
        self.assertFalse(self.detector.valid_candidate(box, .95, points))

    def test_distant_backlit_face_gets_fresh_detections_between_detail_scans(self):
        self.detector.reset()
        frame = self.scene(128, .23)
        for _ in range(8):
            self.assertEqual(len(self.detector.detect(frame)), 1)
        # Removing the face must not turn a cached search region into a hit.
        blank = np.full_like(frame, 215)
        self.assertEqual(self.detector.detect(blank), [])

    def test_new_distant_face_is_acquired_by_next_detail_scan(self):
        self.detector.reset()
        self.detector.detect(np.full((720, 1280, 3), 215, np.uint8))
        frame = self.scene(128, .23)
        found = [self.detector.detect(frame)
                 for _ in range(self.detector.DETAIL_SCAN_INTERVAL)]
        self.assertTrue(any(len(faces) == 1 for faces in found))

    def test_mjpeg_motion_preserves_near_distant_and_backlit_detection(self):
        for side in (256, 128):
            for brightness in (1, .23):
                with self.subTest(side=side, brightness=brightness):
                    self.detector.reset()
                    hits = []
                    for index in range(12):
                        frame = np.full((720, 1280, 3), 215, np.uint8)
                        x = 450 + index * 6
                        frame[180:180 + side, x:x + side] = (
                            cv2.resize(self.face, (side, side)) * brightness).astype(np.uint8)
                        ok, jpeg = cv2.imencode('.jpg', frame,
                                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                        self.assertTrue(ok)
                        decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
                        detections = self.detector.detect(decoded)
                        self.assertLessEqual(len(detections), 1)
                        self.assertTrue(all(score >= self.detector.SCORE_THRESHOLD
                                            for _, score, _ in detections))
                        hits.append(bool(detections))
                    # A brief miss is allowed; moving faces must be reacquired
                    # promptly rather than only on periodic full-frame scans.
                    self.assertGreaterEqual(sum(hits), 11)
                    blank = np.full_like(frame, 215)
                    for _ in range(4):
                        self.assertEqual(self.detector.detect(blank), [])


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch('live_face_detector.time.monotonic', return_value=100.0)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.tracker = FaceTrackingStabilizer()
        self.hit = ([100, 100, 35, 50], .95, [])

    def step(self, seconds, hits):
        self.now.return_value = 100.0 + seconds
        return self.tracker.update(hits)

    def test_requires_consecutive_hits_and_rejects_transient_noise(self):
        self.assertEqual(self.step(0, [self.hit]), [])
        self.assertEqual(self.step(.05, []), [])
        self.assertEqual(self.step(.1, [self.hit]), [])
        self.assertEqual(self.step(.15, [self.hit]), [])
        self.assertEqual(len(self.step(.2, [self.hit])), 1)

    def test_lost_face_is_held_briefly_and_expires(self):
        for seconds in (0, .05, .1):
            self.step(seconds, [self.hit])
        held = self.step(.2, [])
        self.assertEqual(len(held), 1)
        self.assertTrue(held[0].is_holding)
        self.assertEqual(self.step(.4, []), [])

    def test_small_unrelated_object_cannot_inherit_confirmation(self):
        for seconds in (0, .05, .1):
            self.step(seconds, [self.hit])
        unrelated = ([185, 100, 35, 50], .95, [])
        result = self.step(.15, [unrelated])
        self.assertFalse(any(not face.is_holding for face in result))

    def test_confirmation_also_works_with_slow_inference(self):
        self.assertEqual(self.step(0, [self.hit]), [])
        self.assertEqual(self.step(.4, [self.hit]), [])
        self.assertEqual(len(self.step(.8, [self.hit])), 1)

    def test_confirmation_cannot_reuse_a_stale_record(self):
        self.step(0, [self.hit])
        self.step(.05, [self.hit])
        self.assertEqual(self.step(2, [self.hit]), [])

    def test_deliberate_motion_is_followed_without_smoothing_lag(self):
        for seconds in (0, .05, .1):
            self.step(seconds, [self.hit])
        moved = ([115, 100, 35, 50], .95, [])
        result = self.step(.15, [moved])
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].is_holding)
        self.assertGreaterEqual(result[0].smoothed_bbox[0], 113)
        self.assertLessEqual(result[0].smoothed_bbox[0], 115)
        self.assertEqual(result[0].tracking_id, 1)


if __name__ == '__main__':
    unittest.main()
