"""Small JPEG faces with combined backlight/pitch, motion, and fresh-hit checks."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robust_face_detector import RobustFaceDetector
from live_face_detector import FaceTrackingStabilizer, LiveFaceDetectorApp
from benchmark_backlight import backlit_scene
from ncm_camera import JLIP_JPEG, build_jlip_packet
from test_ncm_camera import BoardServer


class BacklightDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(2)
        cls.portrait = cv2.imread(str(Path(__file__).with_name('fixtures') / 'face.jpg'))

    def test_small_backlit_pitched_faces_produce_consecutive_fresh_hits(self):
        for side, brightness, pitch in ((112, .23, .2), (160, .18, .3), (128, .15, 0)):
            with self.subTest(side=side, brightness=brightness, pitch=pitch):
                detector, tracker = RobustFaceDetector(), FaceTrackingStabilizer()
                frame = backlit_scene(self.portrait, side, brightness, pitch)
                hits = []
                for _ in range(12):
                    found = detector.detect(frame)
                    self.assertLessEqual(len(found), 1)
                    for box, score, points in found:
                        self.assertGreaterEqual(score, detector.SCORE_THRESHOLD)
                        self.assertTrue(245 <= box[0] < 245 + side)
                        self.assertTrue(100 <= box[1] < 100 + side)
                        self.assertEqual(len(points), 5)
                    hits.append(bool(found))
                    tracker.update(found)
                self.assertTrue(all(hits[-6:]))
                self.assertTrue(any(face.confirmed for face in tracker.records.values()))
                # Search hints must not turn into detections when the face leaves.
                for _ in range(5):
                    self.assertEqual(detector.detect(np.full_like(frame, 220)), [])

    def test_recovered_face_can_move_without_becoming_a_stale_search_hint(self):
        detector = RobustFaceDetector()
        tracker = FaceTrackingStabilizer()
        confirmed = False
        for index in range(18):
            frame = backlit_scene(self.portrait, 112, .23, .2, x=245 + 2 * index)
            found = detector.detect(frame)
            self.assertLessEqual(len(found), 1)
            confirmed = bool(tracker.update(found)) or confirmed
            for box, _, _ in found:
                self.assertTrue(245 + 2 * index <= box[0] < 357 + 2 * index)
        # The former implementation only had isolated hits in this sequence,
        # which could never produce a confirmed face for the user.
        self.assertTrue(confirmed)

    def test_new_tile_and_exposure_cycle_rejects_window_patterns_and_non_face(self):
        window = np.full((480, 640, 3), 235, np.uint8)
        for left, top, right, bottom in ((90, 60, 180, 290), (330, 170, 410, 460),
                                         (450, 85, 530, 145)):
            cv2.rectangle(window, (left, top), (right, bottom), (20, 20, 20), -1)
        animal = cv2.imread(str(Path(__file__).with_name('fixtures') / 'non_face.jpg'))
        for scene in (window, cv2.resize(animal, (640, 480))):
            detector = RobustFaceDetector()
            for _ in range(40):
                self.assertEqual(detector.detect(scene), [])

    def test_ncm_jpeg_pipeline_confirms_recovered_small_face_below_ten_fps(self):
        frame = backlit_scene(self.portrait, 112, .23, .2)
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.assertTrue(ok)

        def stream(board, connection, _):
            sequence = 0
            while not board.stop.is_set():
                connection.sendall(build_jlip_packet(JLIP_JPEG, sequence, bytes(jpeg)))
                sequence += 1
                board.stop.wait(.025)

        with BoardServer(stream) as board, redirect_stdout(io.StringIO()):
            app = LiveFaceDetectorApp(ncm_config=board.config)
            report = app.run(max_frames=18, show_window=False, frame_timeout=5)
        self.assertTrue(any(face.confirmed for face in app.stabilizer.records.values()))
        self.assertEqual((report['width'], report['height']), (640, 480))
        self.assertLessEqual(report['live_fps'], 10)
        self.assertEqual(report['ncm']['crc_errors'], 0)


if __name__ == '__main__':
    unittest.main()
