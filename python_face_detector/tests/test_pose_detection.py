"""Pose recovery, coordinate alignment, and non-face rejection regressions."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robust_face_detector import RobustFaceDetector
from benchmark_pose import pose_scene
from live_face_detector import LiveFaceDetectorApp
from ncm_camera import JLIP_JPEG, build_jlip_packet
from test_ncm_camera import BoardServer


class PoseDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = Path(__file__).with_name('fixtures')
        cls.portrait = cv2.imread(str(fixtures / 'face.jpg'))
        cls.animal = cv2.imread(str(fixtures / 'non_face.jpg'))

    def test_low_and_high_perspective_views_get_fresh_hits_and_expire(self):
        detector = RobustFaceDetector()
        for pitch in (-.25, .25):
            with self.subTest(pitch=pitch):
                detector.reset()
                frame = pose_scene(self.portrait, pitch=pitch)
                hits = [detector.detect(frame) for _ in range(12)]
                self.assertTrue(all(len(found) == 1 for found in hits[-6:]))
                self.assertTrue(all(score >= detector.SCORE_THRESHOLD for found in hits for _, score, _ in found))
                for _ in range(5):
                    self.assertEqual(detector.detect(np.full_like(frame, 215)), [])

    def test_roll_and_inverted_views_reacquire_without_stale_cached_hits(self):
        detector = RobustFaceDetector()
        for angle in (-90, -60, -40, 40, 60, 90, 180):
            with self.subTest(angle=angle):
                detector.reset()
                frame = pose_scene(self.portrait, angle=angle)
                hits = [detector.detect(frame) for _ in range(24)]
                self.assertTrue(any(hits), 'Rotated face was never acquired')
                self.assertGreaterEqual(sum(bool(found) for found in hits[-6:]), 5)
                for found in hits:
                    self.assertLessEqual(len(found), 1)
                    for box, score, landmarks in found:
                        self.assertGreaterEqual(score, detector.SCORE_THRESHOLD)
                        self.assertTrue(450 <= box[0] + box[2] / 2 <= 834)
                        self.assertTrue(160 <= box[1] + box[3] / 2 <= 544)
                        self.assertEqual(len(landmarks), 5)
                self.assertEqual(detector.detect(np.full_like(frame, 215)), [])

    def test_tilt_with_backlight_and_left_right_perspective(self):
        for options in ({'angle': -25, 'pitch': .25, 'brightness': .3},
                        {'yaw': -.25}, {'yaw': .25}):
            with self.subTest(options=options):
                detector = RobustFaceDetector()
                frame = pose_scene(self.portrait, **options)
                hits = [detector.detect(frame) for _ in range(18)]
                self.assertGreaterEqual(sum(bool(found) for found in hits[-6:]), 5)

    def test_recovery_cycle_does_not_invent_faces_in_negative_scenes(self):
        rng = np.random.default_rng(7)
        for image in (np.zeros((360, 640, 3), np.uint8),
                      np.full((360, 640, 3), 255, np.uint8),
                      rng.integers(0, 256, (360, 640, 3), dtype=np.uint8), self.animal):
            detector = RobustFaceDetector()
            for _ in range(18):
                self.assertEqual(detector.detect(image), [])

    def test_foreshortened_geometry_requires_stronger_score_and_mouth_topology(self):
        box = [100, 100, 100, 120]
        points = np.array([[145, 135], [155, 135], [170, 150], [145, 180], [163, 180]])
        self.assertTrue(RobustFaceDetector.valid_candidate(box, .96, points))
        self.assertFalse(RobustFaceDetector.valid_candidate(box, .90, points))
        collapsed = points.copy()
        collapsed[4] = collapsed[3]
        self.assertFalse(RobustFaceDetector.valid_candidate(box, .99, collapsed))
        reversed_mouth = points.copy()
        reversed_mouth[[3, 4]] = reversed_mouth[[4, 3]]
        self.assertFalse(RobustFaceDetector.valid_candidate(box, .99, reversed_mouth))
        low_view = np.array([[125, 150], [175, 150], [150, 143], [130, 170], [170, 170]])
        self.assertTrue(RobustFaceDetector.valid_candidate(box, .96, low_view))

    def test_degenerate_landmarks_cannot_create_an_unbounded_rectification(self):
        detector = RobustFaceDetector()
        points = [[125, 130], [175, 130], [150, 130], [130, 130], [170, 130]]
        descriptor = detector._landmark_descriptor([100, 100, 100, 120], points)
        self.assertEqual(detector._infer_landmark_pose(
            np.zeros((360, 640, 3), np.uint8), [100, 100, 100, 120], descriptor), [])

    def test_low_angle_jpegs_work_through_ncm_capture_and_confirmation(self):
        frame = pose_scene(self.portrait, pitch=.25, angle=-25, brightness=.3)
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.assertTrue(ok)

        def stream(board, connection, index):
            sequence = 0
            while not board.stop.is_set():
                connection.sendall(build_jlip_packet(JLIP_JPEG, sequence, bytes(jpeg)))
                sequence += 1
                board.stop.wait(.025)

        with BoardServer(stream) as board, redirect_stdout(io.StringIO()):
            app = LiveFaceDetectorApp(ncm_config=board.config)
            report = app.run(max_frames=18, show_window=False, frame_timeout=5)
        self.assertEqual(report['source'], 'ncm')
        self.assertLessEqual(report['live_fps'], 10)
        self.assertTrue(any(face.confirmed for face in app.stabilizer.records.values()))
        self.assertEqual(report['ncm']['crc_errors'], 0)


if __name__ == '__main__':
    unittest.main()
