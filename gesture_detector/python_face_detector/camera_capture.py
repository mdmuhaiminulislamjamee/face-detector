"""Camera negotiation and a single-slot capture reader for a live preview."""
from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Thread
import time

from native_runtime import load_native_libraries

cv2, _ = load_native_libraries()


class FrameRateLimiter:
    """Limit work starts without queuing frames or catching up after a slow pass."""
    def __init__(self, fps=10):
        if fps <= 0:
            raise ValueError('FPS must be positive')
        self.interval = 1.0 / fps
        self._next_start = 0.0

    def remaining(self, now):
        return max(0.0, self._next_start - now)

    def started(self, now):
        self._next_start = now + self.interval

    def reset(self):
        self._next_start = 0.0


def open_camera(camera_id=0):
    capture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_id)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera {camera_id}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # DirectShow renegotiates the format when dimensions/FPS change. Request
    # MJPEG LAST, otherwise a successful request can silently revert to YUY2.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return capture


def camera_description(capture):
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join(chr((code >> (8 * i)) & 255) for i in range(4)) if code > 0 else 'unknown'
    return {
        'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'codec': codec,
    }


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    image: object
    captured_at: float
    read_duration_sec: float
    capture_fps: float
    stream_epoch: int = 0


class LatestFrameCapture:
    """Owns the camera and keeps ONLY its newest frame, never a frame queue.

    The capture thread never modifies a published image. The consumer owns its
    displayed copy and runs inference on that same image for aligned overlays.
    """
    def __init__(self, capture):
        self._capture = capture
        self._condition = Condition()
        self._stop = Event()
        self._latest = None
        self._error = None
        self._closed = False
        self._thread = Thread(target=self._run, name='latest-camera-frame', daemon=True)
        self._thread.start()

    def _run(self):
        timestamps = deque(maxlen=30)
        sequence = 0
        failures = 0
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                ok, frame = self._capture.read()
                now = time.perf_counter()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError('The camera stopped delivering frames.')
                    self._stop.wait(.01)
                    continue
                failures = 0
                sequence += 1
                timestamps.append(now)
                elapsed = timestamps[-1] - timestamps[0]
                fps = (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0
                packet = CapturedFrame(sequence, frame, now, now - started, fps)
                with self._condition:
                    self._latest = packet
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
        finally:
            self._capture.release()
            with self._condition:
                self._closed = True
                self._condition.notify_all()

    def read_after(self, sequence=0, timeout=.05):
        with self._condition:
            self._condition.wait_for(
                lambda: self._error is not None or self._closed or
                    (self._latest is not None and self._latest.sequence > sequence),
                timeout=timeout,
            )
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            if self._latest is not None and self._latest.sequence > sequence:
                return self._latest
            if self._closed and not self._stop.is_set():
                raise RuntimeError('The camera stream has closed.')
            return None

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError('The camera driver did not finish its pending read.')
