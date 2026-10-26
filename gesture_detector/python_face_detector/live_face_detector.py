"""
Live Face Detection & Tracking System
------------------------------------
Real-time webcam face detector featuring:
  * Multi-scale YuNet detection with shadow recovery for distant/backlit faces.
  * Temporal Tracking Stabilizer (exponential smoothing + dropout grace period).
  * Persistent face IDs that do not flicker or lose track of moving targets.
  * Approximate distance estimation and landmark visualization.
"""

import sys
import time
import math
import argparse
from collections import deque
import json
from native_runtime import load_native_libraries, startup_error_message

try:
    cv2, np = load_native_libraries()
except (ImportError, OSError, MemoryError) as error:
    print(startup_error_message(error), file=sys.stderr)
    raise SystemExit(1) from None

from robust_face_detector import RobustFaceDetector
from camera_capture import FrameRateLimiter, LatestFrameCapture, camera_description, open_camera
from ncm_camera import NcmCameraConfig, NcmLatestFrameCapture, discover_camera
from single_face_selector import SingleFaceSelector

TARGET_FPS = 10


class TrackedFace:
    """Maintains continuous tracking state and smoothed coordinates for one face."""
    def __init__(self, tracking_id, bbox, confidence=0.0):
        self.tracking_id = tracking_id
        self.smoothed_bbox = list(bbox)  # [x, y, w, h]
        self.velocity = [0.0, 0.0]
        self.confidence = confidence
        self.last_seen_time = time.monotonic()
        self.missed_frames = 0
        self.is_holding = False
        self.landmarks = []
        self.hit_count = 1
        self.confirmed = False


class FaceTrackingStabilizer:
    """
    Temporal tracker that eliminates bounding-box jitter and bridges momentary
    detector dropouts when a target turns, blinks, or moves far away.
    Uses hit-confirmation to reject transient 1-frame false positives (e.g. on clothes).
    """
    def __init__(self, smoothing_alpha=0.65, hold_duration_sec=0.20, iou_threshold=0.20, min_hits=3):
        self.smoothing_alpha = smoothing_alpha
        self.hold_duration_sec = hold_duration_sec
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.records = {}
        self.next_id = 1
        self.last_time = time.monotonic()

    def update(self, detected_faces):
        """
        detected_faces: list of tuples (bbox, confidence, landmarks)
        where bbox = [x, y, w, h] in pixel coordinates.
        """
        now = time.monotonic()
        dt = max(0.016, now - self.last_time)
        self.last_time = now

        # Expire before matching: an old box must not confirm an unrelated object.
        self.records = {tid: rec for tid, rec in self.records.items()
                        if now - rec.last_seen_time <= max(1.0, self.hold_duration_sec)}
        matched_ids = set()
        active_faces = []

        # 1. Match newly detected faces to existing tracked records using IOU + Centroid
        unmatched_detections = []
        for bbox, conf, lms in detected_faces:
            best_id = None
            best_score = -1.0

            for tid, record in self.records.items():
                if tid in matched_ids:
                    continue
                iou = self._iou(bbox, record.smoothed_bbox)
                c_dist = self._centroid_dist(bbox, record.smoothed_bbox)

                size_ratio = (bbox[2] * bbox[3]) / max(
                    1, record.smoothed_bbox[2] * record.smoothed_bbox[3])
                distance_limit = min(120.0, 0.6 * max(
                    math.hypot(bbox[2], bbox[3]),
                    math.hypot(*record.smoothed_bbox[2:])))
                if 0.35 <= size_ratio <= 2.85 and (
                        iou >= self.iou_threshold or c_dist <= distance_limit):
                    score = (iou * 2.0) + (1.0 / (1.0 + c_dist / 50.0))
                    if score > best_score:
                        best_score = score
                        best_id = tid

            if best_id is not None:
                matched_ids.add(best_id)
                rec = self.records[best_id]
                self._update_record(rec, bbox, conf, lms, dt)
                if rec.confirmed:
                    active_faces.append(rec)
            else:
                unmatched_detections.append((bbox, conf, lms))

        # 2. Register brand new candidate faces
        for bbox, conf, lms in unmatched_detections:
            tid = self.next_id
            self.next_id += 1
            new_rec = TrackedFace(tid, bbox, conf)
            new_rec.landmarks = lms
            if self.min_hits <= 1:
                new_rec.confirmed = True
                active_faces.append(new_rec)
            self.records[tid] = new_rec
            matched_ids.add(tid)

        # 3. Handle dropouts (bridge temporary misses with grace period only for confirmed faces)
        expired_ids = []
        for tid, rec in self.records.items():
            if tid in matched_ids:
                continue

            # Unconfirmed transient noise is immediately discarded
            if not rec.confirmed:
                expired_ids.append(tid)
                continue

            elapsed = now - rec.last_seen_time
            if elapsed <= self.hold_duration_sec:
                rec.missed_frames += 1
                rec.is_holding = True
                # Dead reckoning velocity shift (dampened)
                rec.smoothed_bbox[0] += int(rec.velocity[0] * dt * 0.4)
                rec.smoothed_bbox[1] += int(rec.velocity[1] * dt * 0.4)
                active_faces.append(rec)
            else:
                expired_ids.append(tid)

        for tid in expired_ids:
            del self.records[tid]

        return active_faces

    def _update_record(self, record, new_bbox, conf, lms, dt):
        record.hit_count += 1
        if record.hit_count >= self.min_hits:
            record.confirmed = True

        alpha = self.smoothing_alpha
        old_center = (
            record.smoothed_bbox[0] + record.smoothed_bbox[2] / 2,
            record.smoothed_bbox[1] + record.smoothed_bbox[3] / 2,
        )
        new_center = (
            new_bbox[0] + new_bbox[2] / 2,
            new_bbox[1] + new_bbox[3] / 2,
        )
        # Follow deliberate movement immediately; retain smoothing for tiny jitter.
        relative_motion = math.hypot(new_center[0] - old_center[0],
                                     new_center[1] - old_center[1]) / max(
                                         1.0, math.hypot(new_bbox[2], new_bbox[3]))
        if relative_motion >= .10:
            alpha = max(alpha, .92)

        if dt > 0:
            inst_vx = (new_center[0] - old_center[0]) / dt
            inst_vy = (new_center[1] - old_center[1]) / dt
            record.velocity[0] = 0.4 * record.velocity[0] + 0.6 * inst_vx
            record.velocity[1] = 0.4 * record.velocity[1] + 0.6 * inst_vy

        # Smooth bounding box
        s_x = int(alpha * new_bbox[0] + (1 - alpha) * record.smoothed_bbox[0])
        s_y = int(alpha * new_bbox[1] + (1 - alpha) * record.smoothed_bbox[1])
        s_w = int(alpha * new_bbox[2] + (1 - alpha) * record.smoothed_bbox[2])
        s_h = int(alpha * new_bbox[3] + (1 - alpha) * record.smoothed_bbox[3])

        record.smoothed_bbox = [s_x, s_y, s_w, s_h]
        record.confidence = conf
        record.landmarks = lms
        record.last_seen_time = time.monotonic()
        record.missed_frames = 0
        record.is_holding = False

    @staticmethod
    def _iou(box_a, box_b):
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
        yB = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        area_a = box_a[2] * box_a[3]
        area_b = box_b[2] * box_b[3]
        union = area_a + area_b - inter
        return (inter / union) if union > 0 else 0.0

    @staticmethod
    def _centroid_dist(box_a, box_b):
        cA = (box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2)
        cB = (box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2)
        return math.hypot(cA[0] - cB[0], cA[1] - cB[1])


class LiveFaceDetectorApp:
    def __init__(self, camera_id=0, ncm_config=None, mirror=None):
        self.camera_id = camera_id
        self.ncm_config = ncm_config
        self.mirror = (ncm_config is None) if mirror is None else mirror
        self.cap = None
        self.capture_reader = None
        self.detector = None
        self.stabilizer = FaceTrackingStabilizer()
        self.face_selector = SingleFaceSelector()
        self.show_landmarks = True
        self.show_distance = True
        self.use_stabilizer = True

        self._setup_detector()

    def _setup_detector(self):
        self.detector = RobustFaceDetector()
        print("[INFO] YuNet loaded (multi-scale, shadow + pose recovery, strict validation).")

    def _detect_faces(self, frame_bgr):
        return self.detector.detect(frame_bgr)

    def _open_camera(self):
        return open_camera(self.camera_id)

    def _start_source(self):
        if self.ncm_config is not None:
            config = self.ncm_config
            print(f"[INFO] USB-NCM: {config.host_ip or 'auto route'} -> "
                  f"{config.device_ip}:{config.tcp_port} (TCP/JLIP JPEG)")
            print('[INFO] The USB-NCM adapter must have a working route to the board; '
                  'the default host address is 192.168.50.1/30.')
            self.capture_reader = NcmLatestFrameCapture(config)
            return {'source': 'ncm', 'codec': 'JPEG/JLIP', 'mirror': self.mirror}
        print(f"[INFO] Opening camera index {self.camera_id}...")
        self.cap = self._open_camera()
        description = camera_description(self.cap)
        print(f"[INFO] Camera: {description['width']}x{description['height']} "
              f"{description['codec']} (requested 30 FPS; live rates are measured).")
        self.capture_reader = LatestFrameCapture(self.cap)
        return dict(description, source='webcam', mirror=self.mirror)

    def _reset_tracking(self):
        self.stabilizer = FaceTrackingStabilizer()
        self.face_selector.clear()
        self.detector.reset()

    def _draw_ncm_waiting(self, status):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self._draw_hud(frame, 0, 0)
        lines = [f"USB-NCM: {status['state']} - waiting for a fresh JPEG",
                 f"Host {status['host_ip']} -> {status['device_ip']}:{status['tcp_port']}",
                 str(status['error'] or 'Check board power, USB-NCM IP/route and camera firmware.'),
                 f"Received: {status['frames_received']}  CRC errors: {status['crc_errors']}  "
                 f"Invalid JPEGs: {status['invalid_jpegs']}  Reconnects: {status['reconnects']}"]
        for index, line in enumerate(lines):
            cv2.putText(frame, line[:140], (32, 120 + index * 42),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (200, 200, 200), 1, cv2.LINE_AA)
        return frame

    def _prepare_preview(self, frame):
        h, w = frame.shape[:2]
        if self.ncm_config is not None:
            # Board frames may be only 320/640px wide. Keep the HUD readable
            # without changing inference resolution or stretching the picture.
            scale = min(1280 / w, 720 / h)
            width, height = max(1, round(w * scale)), max(1, round(h * scale))
            resized = cv2.resize(frame, (width, height),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            preview = np.zeros((720, 1280, 3), dtype=np.uint8)
            left, top = (1280 - width) // 2, (720 - height) // 2
            preview[top:top + height, left:left + width] = resized
            return preview
        if w > 1280:
            return cv2.resize(frame, (1280, round(h * 1280 / w)),
                              interpolation=cv2.INTER_AREA)
        return frame

    def run(self, max_frames=None, show_window=True, frame_timeout=10.0):
        description = self._start_source()
        window_name = "Live Face Detector & Tracker - Press 'Q' to Exit"
        display_times = deque(maxlen=30)
        limiter = FrameRateLimiter(TARGET_FPS)
        sequence = 0
        fps = 0.0
        samples = []
        last_preview = None
        last_frame_at = time.perf_counter()
        last_waiting_at = 0.0
        source_state = None
        frame_size = None
        try:
            if show_window:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
            while True:
                now = time.perf_counter()
                if self.ncm_config is not None:
                    transport = self.capture_reader.status()
                    signature = (transport['state'], transport['stream_epoch'])
                    if signature != source_state:
                        source_state = signature
                        self._reset_tracking()
                        last_preview = None
                        display_times.clear()
                        fps = 0.0
                        print(f"[NCM] {transport['state']}: "
                              f"{transport['error'] or 'USB camera stream'}")
                    if (now - last_frame_at > frame_timeout and
                            (not samples or not show_window)):
                        raise RuntimeError(f"No usable NCM frame for {frame_timeout:g}s. "
                            f"{transport['error'] or transport['state']}. "
                            'Use --ncm-discover and check the USB-NCM host IP/route.')
                    if last_preview is not None and now - last_frame_at > .75:
                        self._reset_tracking()
                        last_preview = None
                    if show_window and last_preview is None and now - last_waiting_at >= .25:
                        cv2.imshow(window_name, self._draw_ncm_waiting(transport))
                        last_waiting_at = now
                remaining = limiter.remaining(time.perf_counter())
                packet = None if remaining > 0 else self.capture_reader.read_after(sequence)
                if packet is not None:
                    sequence = packet.sequence
                    # Capture continues on its own thread, overwriting old frames.
                    # Inference and overlays use this SAME frame, so boxes align.
                    started = time.perf_counter()
                    limiter.started(started)
                    display_times.append(started)
                    if len(display_times) >= 2:
                        fps = (len(display_times) - 1) / max(.001, display_times[-1] - display_times[0])
                    frame = cv2.flip(packet.image, 1) if self.mirror else packet.image.copy()
                    h, w = frame.shape[:2]
                    if frame_size != (w, h):
                        self._reset_tracking()
                        frame_size = (w, h)
                    description.update(width=w, height=h)
                    inference_started = time.perf_counter()
                    raw_detections = self._detect_faces(frame)
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    if self.ncm_config is not None and not self.capture_reader.is_current(packet):
                        self._reset_tracking()
                        last_preview = None
                        continue
                    last_frame_at = packet.captured_at
                    self.stabilizer.smoothing_alpha = 0.65 if self.use_stabilizer else 1.0
                    self.stabilizer.hold_duration_sec = 0.20 if self.use_stabilizer else 0.0
                    tracked_faces = self.face_selector.select(
                        self.stabilizer.update(raw_detections))
                    for face in tracked_faces:
                        self._draw_face_overlay(frame, face, h)
                    frame = self._prepare_preview(frame)
                    age_ms = (time.perf_counter() - packet.captured_at) * 1000
                    self._draw_hud(frame, sum(not face.is_holding for face in tracked_faces), fps)
                    status = (f"Limit: {TARGET_FPS} FPS  |  "
                              f"Detection: {inference_ms:.0f} ms  |  Frame age: {age_ms:.0f} ms")
                    if self.ncm_config is not None:
                        status += f"  |  NCM camera: {packet.capture_fps:.1f} FPS"
                    cv2.rectangle(frame, (0, 43), (frame.shape[1], 70), (18, 18, 18), cv2.FILLED)
                    cv2.putText(frame, status, (16, 62), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (200, 200, 200), 1, cv2.LINE_AA)
                    if show_window:
                        cv2.imshow(window_name, frame)
                    last_preview = frame
                    completed = time.perf_counter()
                    samples.append({
                        'started_at': started,
                        'completed_at': completed,
                        'capture_read_ms': packet.read_duration_sec * 1000,
                        'capture_fps': packet.capture_fps,
                        'detection_ms': inference_ms,
                        'frame_age_ms': (completed - packet.captured_at) * 1000,
                        'processing_ms': (completed - started) * 1000,
                        'selected_faces': len(tracked_faces),
                    })
                    if max_frames is not None and len(samples) >= max_frames:
                        break
                    # Bound diagnostics storage during an unlimited live session.
                    if max_frames is None and len(samples) > 120:
                        samples = samples[-60:]
                key = (cv2.waitKey(1) & 0xFF) if show_window else -1
                if key in (27, ord('q'), ord('Q')):
                    break
                elif key in (ord('s'), ord('S')) and last_preview is not None:
                    filename = f"snapshot_{int(time.time())}.png"
                    cv2.imwrite(filename, last_preview)
                    print(f"[INFO] Saved screenshot to {filename}")
                elif key in (ord('m'), ord('M')):
                    self.show_landmarks = not self.show_landmarks
                elif key in (ord('d'), ord('D')):
                    self.show_distance = not self.show_distance
                elif key in (ord('t'), ord('T')):
                    self.use_stabilizer = not self.use_stabilizer
                    print(f"[INFO] Tracking Stabilizer: {'ON' if self.use_stabilizer else 'OFF'}")
                elif key in (ord('c'), ord('C')):
                    if self.ncm_config is not None:
                        self.capture_reader.reconnect()
                        self._reset_tracking()
                        last_preview = None
                        continue
                    self.capture_reader.close()
                    self.camera_id = (self.camera_id + 1) % 3
                    self.cap = self._open_camera()
                    description = camera_description(self.cap)
                    self.capture_reader = LatestFrameCapture(self.cap)
                    self.stabilizer = FaceTrackingStabilizer()
                    self.face_selector.clear()
                    self.detector.reset()
                    sequence = 0
                    display_times.clear()
                    limiter.reset()
                    fps = 0.0
                    print(f"[INFO] Switched to camera index {self.camera_id}: {description}")
                # Wait before selecting the next fresh frame. A high-resolution
                # clock avoids Windows clock ticks turning 10 FPS into ~9.1 FPS.
                remaining = limiter.remaining(time.perf_counter())
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self.capture_reader.close()
            if show_window:
                cv2.destroyAllWindows()
        report = None
        if max_frames is not None and len(samples) > 1:
            usable = samples[5:] if len(samples) > 10 else samples
            elapsed = usable[-1]['started_at'] - usable[0]['started_at']
            report = dict(description, preview=show_window, frames=len(usable),
                          target_fps=TARGET_FPS,
                          live_fps=round((len(usable) - 1) / max(.001, elapsed), 2))
            report['max_selected_faces'] = max(sample['selected_faces'] for sample in samples)
            if self.ncm_config is not None:
                report['ncm'] = self.capture_reader.status()
            for metric in ('capture_read_ms', 'capture_fps', 'detection_ms',
                           'frame_age_ms', 'processing_ms'):
                values = [sample[metric] for sample in usable]
                report[metric + '_median'] = round(float(np.median(values)), 2)
                report[metric + '_p95'] = round(float(np.percentile(values, 95)), 2)
            print(json.dumps(report, indent=2))
        print("[INFO] Live Face Detector stopped cleanly.")
        return report

    def _draw_face_overlay(self, frame, face, frame_h):
        x, y, w, h = face.smoothed_bbox
        is_holding = face.is_holding

        # Color: Amber when holding lock during temporary dropout, Cyan when actively locked
        color = (0, 180, 255) if is_holding else (255, 220, 0)  # BGR order: Cyan / Amber

        # Draw glowing bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        # Draw corner accents
        corner_len = max(10, min(30, int(min(w, h) * 0.2)))
        thick = 3
        white = (255, 255, 255)

        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), white, thick)
        cv2.line(frame, (x, y), (x, y + corner_len), white, thick)
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), white, thick)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), white, thick)
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), white, thick)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), white, thick)
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), white, thick)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), white, thick)

        # Draw ID & Status Pill
        status_text = f"Face #{face.tracking_id}"
        if is_holding:
            status_text += " [Holding Lock]"
        elif face.confidence > 0:
            status_text += f" ({int(face.confidence * 100)}%)"

        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pill_y = max(24, y - 8)
        cv2.rectangle(
            frame,
            (x, pill_y - th - 6),
            (x + tw + 12, pill_y + 4),
            (20, 20, 20),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            status_text,
            (x + 6, pill_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        # Distance estimation based on face height proportion
        if self.show_distance and h > 0:
            ratio = h / frame_h
            # Approx focal distance model: ratio ~ 0.5 at 0.5m, ~0.15 at 1.8m, ~0.06 at 3.5m
            dist_est = 0.28 / max(0.04, ratio)
            if dist_est > 2.5:
                dist_str = f"Far ({dist_est:.1f}m)"
                dist_color = (0, 220, 255)
            elif dist_est > 1.2:
                dist_str = f"Normal ({dist_est:.1f}m)"
                dist_color = (0, 255, 150)
            else:
                dist_str = f"Close ({dist_est:.1f}m)"
                dist_color = (255, 200, 100)

            (dw, dh), _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(
                frame,
                (x, y + h + 6),
                (x + dw + 10, y + h + dh + 14),
                (20, 20, 20),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                dist_str,
                (x + 5, y + h + dh + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                dist_color,
                1,
                cv2.LINE_AA,
            )

        # Facial landmarks
        if self.show_landmarks and face.landmarks and not is_holding:
            for lx, ly in face.landmarks:
                point = (round(lx), round(ly))
                cv2.circle(frame, point, 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, point, 2, (0, 255, 255), -1, cv2.LINE_AA)

    def _draw_hud(self, frame, face_count, fps):
        # Top banner background
        cv2.rectangle(frame, (0, 0), (1280, 42), (18, 18, 18), cv2.FILLED)
        cv2.line(frame, (0, 42), (1280, 42), (40, 40, 40), 1)

        # Live dot + Title
        cv2.circle(frame, (20, 21), 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "FACE DETECTOR & TRACKER",
            (36, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Mode Badge
        engine_str = "Engine: YuNet / Multi-scale + Shadow + Pose"
        cv2.putText(
            frame,
            engine_str,
            (330, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # Face count & FPS
        count_str = f"Faces: {face_count}"
        fps_str = f"FPS: {fps:.1f}"
        cv2.putText(frame, count_str, (840, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, fps_str, (960, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Bottom shortcut bar
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h - 30), (w, h), (18, 18, 18), cv2.FILLED)
        source_control = 'Reconnect NCM' if self.ncm_config is not None else 'Camera'
        shortcuts = ("[Q] Quit  |  [S] Save Snapshot  |  [T] Toggle Stabilizer  |  "
                     f"[M] Landmarks  |  [D] Distance  |  [C] {source_control}")
        cv2.putText(
            frame,
            shortcuts,
            (16, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )


def build_argument_parser():
    parser = argparse.ArgumentParser(description='Live webcam or USB-NCM face detector and tracker.')
    parser.add_argument('camera_index', type=int, nargs='?', default=0)
    parser.add_argument('--source', choices=('webcam', 'ncm'), default='webcam')
    parser.add_argument('--ncm', action='store_const', const='ncm', dest='source',
                        help='use the USB-NCM board JPEG stream instead of a webcam')
    parser.add_argument('--ncm-host-ip', default='192.168.50.1',
                        help='USB-NCM adapter IP, or auto to let the OS select a route')
    parser.add_argument('--ncm-device-ip', default='192.168.50.2')
    parser.add_argument('--ncm-port', type=int, default=5000)
    parser.add_argument('--ncm-discovery-port', type=int, default=5001)
    parser.add_argument('--ncm-discover', action='store_true',
                        help='check the board using UDP discovery, print JSON and exit')
    parser.add_argument('--ncm-rotate-180', action='store_true',
                        help='rotate NCM frames for an upside-down camera mount')
    parser.add_argument('--ncm-timeout', type=float, default=10.0,
                        help='seconds to wait for initial frames or a stalled headless profile')
    parser.add_argument('--mirror', action=argparse.BooleanOptionalAction, default=None,
                        help='horizontal mirror (default: on for webcam, off for NCM)')
    parser.add_argument('--check', '--test', '--verify', action='store_true',
                        help='verify the model without opening the camera')
    parser.add_argument('--profile-camera', action='store_true',
                        help='measure actual camera, detection, rendering, and frame age')
    parser.add_argument('--preview', action='store_true',
                        help='include the real preview window while profiling')
    parser.add_argument('--frames', type=int, default=60)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.frames < 2:
        parser.error('--frames must be at least 2')
    if not math.isfinite(args.ncm_timeout) or args.ncm_timeout <= 0:
        parser.error('--ncm-timeout must be finite and positive')
    try:
        config = NcmCameraConfig(
            host_ip='' if args.ncm_host_ip.lower() == 'auto' else args.ncm_host_ip,
            device_ip=args.ncm_device_ip, tcp_port=args.ncm_port,
            discovery_port=args.ncm_discovery_port, rotate_180=args.ncm_rotate_180)
    except ValueError as error:
        parser.error(str(error))
    if args.ncm_discover:
        result = discover_camera(config)
        print(json.dumps(result, indent=2))
        return 0 if result['ok'] else 1
    try:
        app = LiveFaceDetectorApp(camera_id=args.camera_index,
            ncm_config=config if args.source == 'ncm' else None, mirror=args.mirror)
        if args.check:
            assert not app._detect_faces(np.zeros((360, 640, 3), dtype=np.uint8))
            print("[SUCCESS] Models and inference verified successfully.")
        else:
            app.run(max_frames=args.frames if args.profile_camera else None,
                    show_window=not args.profile_camera or args.preview,
                    frame_timeout=args.ncm_timeout)
        return 0
    except KeyboardInterrupt:
        print('[INFO] Stopped by user.')
        return 0
    except (RuntimeError, OSError, cv2.error) as error:
        print(f'[ERROR] {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
