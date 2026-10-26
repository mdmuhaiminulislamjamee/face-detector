"""One visible target, stable locking, and automatic replacement regressions."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_face_detector import FaceTrackingStabilizer, LiveFaceDetectorApp
from single_face_selector import SingleFaceSelector
from ncm_camera import JLIP_JPEG, build_jlip_packet
from test_ncm_camera import BoardServer


class SingleFaceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch('live_face_detector.time.monotonic', return_value=100.)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.tracker = FaceTrackingStabilizer()
        self.selector = SingleFaceSelector()
        self.a = ([100, 100, 60, 80], .95, [])
        self.b = ([400, 100, 100, 120], .96, [])

    def step(self, offset, hits):
        self.now.return_value = 100 + offset
        result = self.selector.select(self.tracker.update(hits))
        self.assertLessEqual(len(result), 1)
        return result

    def test_only_largest_confirmed_face_is_selected_and_order_does_not_switch_lock(self):
        self.assertEqual(self.step(0, [self.a, self.b]), [])
        self.assertEqual(self.step(.1, [self.b, self.a]), [])
        selected = self.step(.2, [self.a, self.b])[0]
        self.assertEqual(selected.smoothed_bbox, self.b[0])
        self.assertEqual(len(self.tracker.records), 2)
        target_id = selected.tracking_id
        for offset in (.3, .4, .5):
            self.assertEqual(self.step(offset, [self.b, self.a])[0].tracking_id, target_id)

    def test_target_does_not_switch_to_a_new_larger_face_while_still_present(self):
        for offset in (0, .1, .2):
            self.step(offset, [self.a])
        target_id = self.selector.locked_tracking_id
        for offset in (.3, .4, .5, .6):
            self.assertEqual(self.step(offset, [self.b, self.a])[0].tracking_id, target_id)

    def test_missing_target_holds_briefly_then_replaces_it_with_fresh_confirmed_face(self):
        for offset in (0, .1, .2):
            self.step(offset, [self.a, self.b])
        target_id = self.selector.locked_tracking_id
        held = self.step(.3, [self.a])[0]
        self.assertEqual(held.tracking_id, target_id)
        self.assertTrue(held.is_holding)
        replacement = self.step(.45, [self.a])[0]
        self.assertNotEqual(replacement.tracking_id, target_id)
        self.assertFalse(replacement.is_holding)

    def test_unconfirmed_replacement_cannot_inherit_the_old_targets_confirmation(self):
        for offset in (0, .1, .2):
            self.step(offset, [self.a])
        self.assertEqual(self.step(.45, [self.b]), [])
        self.assertEqual(self.step(.55, [self.b]), [])
        self.assertFalse(self.step(.65, [self.b])[0].is_holding)

    def test_other_held_faces_cannot_acquire_a_new_lock_and_reset_clears_lock(self):
        for offset in (0, .1, .2):
            self.step(offset, [self.a, self.b])
        self.selector.clear()
        self.assertEqual(self.step(.3, []), [])
        self.assertIsNone(self.selector.locked_tracking_id)
        self.assertEqual(self.step(.5, []), [])

    def test_app_clears_lock_when_camera_or_resolution_tracking_resets(self):
        with patch.object(LiveFaceDetectorApp, '_setup_detector'):
            app = LiveFaceDetectorApp()
        app.face_selector.locked_tracking_id = 42
        app.detector = unittest.mock.Mock()
        app._reset_tracking()
        self.assertIsNone(app.face_selector.locked_tracking_id)
        app.detector.reset.assert_called_once()

    def test_real_yunet_ncm_profile_displays_one_target_and_replaces_it(self):
        portrait = cv2.imread(str(Path(__file__).with_name('fixtures') / 'face.jpg'))
        both = np.full((480, 640, 3), 215, np.uint8)
        both[40:296, 20:276] = cv2.resize(portrait, (256, 256))
        both[160:288, 400:528] = cv2.resize(portrait, (128, 128))
        other = both.copy()
        other[40:296, 20:276] = 215
        jpegs = [bytes(cv2.imencode('.jpg', image)[1]) for image in (both, other)]

        def stream(board, connection, _):
            sequence = 0
            while not board.stop.is_set():
                connection.sendall(build_jlip_packet(JLIP_JPEG, sequence,
                    jpegs[int(sequence >= 28)]))
                sequence += 1
                board.stop.wait(.025)

        # Use the real monotonic clock for this network/pacing integration run.
        self.clock.stop()
        with BoardServer(stream) as board, redirect_stdout(io.StringIO()):
            app = LiveFaceDetectorApp(ncm_config=board.config)
            with patch.object(app, '_draw_face_overlay', wraps=app._draw_face_overlay) as overlay:
                report = app.run(max_frames=20, show_window=False, frame_timeout=5)
        self.assertEqual(report['max_selected_faces'], 1)
        self.assertLessEqual(report['live_fps'], 10)
        target_ids = [call.args[1].tracking_id for call in overlay.call_args_list]
        self.assertGreaterEqual(len(set(target_ids)), 2)
        self.assertNotEqual(target_ids[0], target_ids[-1])


if __name__ == '__main__':
    unittest.main()
