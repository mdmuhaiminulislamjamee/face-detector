import sys
import time
import unittest
from pathlib import Path
from threading import Event, Lock
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_capture import FrameRateLimiter, LatestFrameCapture, open_camera


class FrameRateLimiterTests(unittest.TestCase):
    def test_limits_new_frames_to_one_per_100_ms(self):
        limiter = FrameRateLimiter(10)
        self.assertEqual(limiter.remaining(1.0), 0)
        limiter.started(1.0)
        self.assertAlmostEqual(limiter.remaining(1.05), .05)
        self.assertAlmostEqual(limiter.remaining(1.10), 0)

    def test_slow_processing_never_causes_catch_up_bursts(self):
        limiter = FrameRateLimiter(10)
        limiter.started(1.0)
        self.assertEqual(limiter.remaining(1.4), 0)
        limiter.started(1.4)
        self.assertAlmostEqual(limiter.remaining(1.4), .1)
        limiter.reset()
        self.assertEqual(limiter.remaining(1.4), 0)


class FakeCamera:
    def __init__(self, failing=False):
        self.count = 0
        self.released = False
        self.failing = failing
        self.lock = Lock()
        self.reached = Event()

    def read(self):
        time.sleep(.002)
        with self.lock:
            self.count += 1
            if self.count >= 12:
                self.reached.set()
            count = self.count
        if self.failing:
            return False, None
        return True, np.full((2, 2, 3), count % 256, np.uint8)

    def release(self):
        self.released = True


class CaptureTests(unittest.TestCase):
    def test_slow_consumer_receives_latest_frame_without_backlog(self):
        camera = FakeCamera()
        reader = LatestFrameCapture(camera)
        try:
            first = reader.read_after(timeout=1)
            self.assertIsNotNone(first)
            self.assertTrue(camera.reached.wait(1))
            current = reader.read_after(first.sequence, timeout=1)
            self.assertGreaterEqual(current.sequence, 12)
            self.assertGreater(current.sequence, first.sequence + 1)
            self.assertEqual(int(current.image[0, 0, 0]), current.sequence % 256)
            self.assertLess(time.perf_counter() - current.captured_at, .1)
            self.assertGreater(current.capture_fps, 0)
        finally:
            reader.close()
        self.assertTrue(camera.released)

    def test_published_frames_are_not_overwritten_by_future_reads(self):
        reader = LatestFrameCapture(FakeCamera())
        try:
            first = reader.read_after(timeout=1)
            before = first.image.copy()
            reader.read_after(first.sequence + 5, timeout=1)
            np.testing.assert_array_equal(first.image, before)
        finally:
            reader.close()

    def test_read_failures_propagate_and_release_camera(self):
        camera = FakeCamera(failing=True)
        reader = LatestFrameCapture(camera)
        try:
            with self.assertRaisesRegex(RuntimeError, 'stopped delivering'):
                reader.read_after(timeout=2)
        finally:
            reader.close()
        self.assertTrue(camera.released)

    def test_mjpeg_is_requested_after_resolution_and_fps_negotiation(self):
        class NegotiatingCamera:
            def __init__(self):
                self.codec = cv2.VideoWriter_fourcc(*'YUY2')
                self.values = {}

            def isOpened(self):
                return True

            def set(self, prop, value):
                self.values[prop] = value
                if prop == cv2.CAP_PROP_FOURCC:
                    self.codec = value
                else:
                    # DirectShow can reset a previously selected format.
                    self.codec = cv2.VideoWriter_fourcc(*'YUY2')
                return True

        camera = NegotiatingCamera()
        with patch('camera_capture.cv2.VideoCapture', return_value=camera):
            result = open_camera()
        self.assertIs(result, camera)
        self.assertEqual(camera.codec, cv2.VideoWriter_fourcc(*'MJPG'))
        self.assertEqual(camera.values[cv2.CAP_PROP_FRAME_WIDTH], 1920)
        self.assertEqual(camera.values[cv2.CAP_PROP_FRAME_HEIGHT], 1080)


if __name__ == '__main__':
    unittest.main()
