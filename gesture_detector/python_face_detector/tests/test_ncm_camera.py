"""Protocol, loopback transport, and real YuNet integration for USB-NCM."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import socket
import struct
import sys
from threading import Event, Thread
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_face_detector import LiveFaceDetectorApp, build_argument_parser, main
from ncm_camera import (JLIP_HEARTBEAT, JLIP_JPEG, MAX_PAYLOAD_BYTES,
                        JlipStreamParser, NcmCameraConfig, NcmLatestFrameCapture,
                        build_jlip_packet, discover_camera, validated_jpeg)


FACE_JPEG = (Path(__file__).with_name('fixtures') / 'face.jpg').read_bytes()


def wait_until(condition, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if condition():
            return
        time.sleep(.005)
    raise AssertionError('Timed out waiting for the local test board')


class BoardServer:
    """An explicitly loopback-only JLIP test board, with no external services."""
    def __init__(self, handler):
        self.stop = Event()
        self.handler = handler
        self.errors = []
        self.clients = []
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('127.0.0.1', 0))
        self.server.listen()
        self.server.settimeout(.1)
        self.config = NcmCameraConfig(host_ip='127.0.0.1', device_ip='127.0.0.1',
            tcp_port=self.server.getsockname()[1], reconnect_delay=.02)
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        index = 0
        while not self.stop.is_set():
            try:
                connection, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.clients.append(connection)
            try:
                self.handler(self, connection, index)
            except OSError:
                pass  # A consumer may finish profiling and close the socket.
            except Exception as error:
                self.errors.append(error)
            finally:
                connection.close()
            index += 1

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, trace):
        self.stop.set()
        self.server.close()
        for connection in self.clients:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(timeout=2)
        if error_type is None and self.errors:
            raise AssertionError(self.errors)


class ProtocolTests(unittest.TestCase):
    def test_header_matches_board_network_byte_order_and_known_crc(self):
        packet = build_jlip_packet(JLIP_JPEG, 0x01020304, b'123456789', 0x11223344)
        self.assertEqual(packet[:16], bytes.fromhex('4a4c4950011000180102030400000009'))
        self.assertEqual(packet[16:20], bytes.fromhex('cbf43926'))
        self.assertEqual(packet[20:24], bytes.fromhex('11223344'))

    def test_byte_fragmentation_and_coalesced_packets(self):
        parser = JlipStreamParser()
        packets = []
        for byte in build_jlip_packet(JLIP_JPEG, 9, FACE_JPEG):
            packets.extend(parser.feed(bytes([byte])))
        self.assertEqual(packets[0].payload, FACE_JPEG)
        combined = b'garbageJLI' + build_jlip_packet(JLIP_HEARTBEAT, 10) + build_jlip_packet(JLIP_JPEG, 11, FACE_JPEG)
        self.assertEqual([p.sequence for p in parser.feed(combined)], [10, 11])
        self.assertEqual(parser.buffer, bytearray())

    def test_corrupt_crc_and_invalid_headers_recover(self):
        bad_crc = bytearray(build_jlip_packet(JLIP_JPEG, 1, FACE_JPEG))
        bad_crc[40] ^= 1
        for field in (4, 6, 12):
            invalid = bytearray(build_jlip_packet(JLIP_JPEG, 2))
            invalid[field] = 255
            parser = JlipStreamParser()
            packets = parser.feed(bytes(bad_crc) + invalid + build_jlip_packet(JLIP_HEARTBEAT, 3))
            self.assertEqual([p.sequence for p in packets], [3])
            self.assertEqual(parser.crc_errors, 1)
            self.assertEqual(parser.header_errors, 1)

    def test_buffer_is_bounded_for_junk_and_maximum_payload(self):
        parser = JlipStreamParser()
        parser.feed(b'X' * (MAX_PAYLOAD_BYTES * 8))
        self.assertLessEqual(len(parser.buffer), 3)
        packet = build_jlip_packet(JLIP_JPEG, 3, b'X' * MAX_PAYLOAD_BYTES)
        self.assertEqual(len(parser.feed(packet + packet)), 2)
        self.assertLessEqual(len(parser.buffer), 3)
        with self.assertRaises(ValueError):
            build_jlip_packet(JLIP_JPEG, 0, b'X' * (MAX_PAYLOAD_BYTES + 1))

    def test_jpeg_padding_dimensions_and_invalid_payloads(self):
        self.assertEqual(validated_jpeg(FACE_JPEG + b'\0\0'), FACE_JPEG)
        for invalid in (b'', b'\xff\xd8\xff\xd9', FACE_JPEG[:-2],
                bytes.fromhex('ffd8ffc00008082000200001ffd9')):
            self.assertIsNone(validated_jpeg(invalid))


class TransportTests(unittest.TestCase):
    def test_fragmented_jpeg_heartbeat_reply_and_sequence_wrap(self):
        replied = Event()

        def handler(board, connection, index):
            packet = build_jlip_packet(JLIP_HEARTBEAT, 0xffffffff)
            packet += build_jlip_packet(JLIP_JPEG, 0, FACE_JPEG + b'\0')
            connection.sendall(packet[:13])
            connection.sendall(packet[13:])
            connection.settimeout(2)
            parser = JlipStreamParser()
            while not board.stop.is_set():
                data = connection.recv(4096)
                if not data:
                    return
                for reply in parser.feed(data):
                    if reply.packet_type == JLIP_HEARTBEAT and not reply.payload:
                        replied.set()

        with BoardServer(handler) as board:
            reader = NcmLatestFrameCapture(board.config)
            try:
                packet = reader.read_after(timeout=2)
                self.assertIsNotNone(packet)
                self.assertTrue(replied.wait(2))
                expected = cv2.imdecode(np.frombuffer(FACE_JPEG, np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(packet.image, expected)
                self.assertEqual(reader.status()['sequence_gaps'], 0)
                self.assertFalse(reader.is_current(replace(packet, captured_at=time.perf_counter() - 1)))
            finally:
                reader.close()
            self.assertFalse(reader._thread.is_alive())

    def test_slow_consumer_decodes_only_latest_jpeg(self):
        def handler(board, connection, index):
            for sequence in range(1, 41):
                connection.sendall(build_jlip_packet(JLIP_JPEG, sequence, FACE_JPEG))
            board.stop.wait(3)

        with BoardServer(handler) as board:
            reader = NcmLatestFrameCapture(board.config)
            try:
                wait_until(lambda: reader.status()['frames_received'] == 40)
                self.assertEqual(reader.status()['frames_decoded'], 0)
                packet = reader.read_after(timeout=1)
                self.assertEqual(packet.sequence, 40)
                self.assertEqual(reader.status()['frames_decoded'], 1)
                self.assertIsNone(reader.read_after(packet.sequence, timeout=.01))
            finally:
                reader.close()

    def test_corrupt_and_invalid_frames_never_reach_decoder(self):
        def handler(board, connection, index):
            corrupted = bytearray(build_jlip_packet(JLIP_JPEG, 1, FACE_JPEG))
            corrupted[45] ^= 1
            connection.sendall(corrupted + build_jlip_packet(JLIP_JPEG, 2, b'bad JPEG') +
                               build_jlip_packet(JLIP_JPEG, 3, FACE_JPEG))
            board.stop.wait(3)

        with BoardServer(handler) as board:
            reader = NcmLatestFrameCapture(board.config)
            try:
                self.assertIsNotNone(reader.read_after(timeout=2))
                status = reader.status()
                self.assertEqual(status['crc_errors'], 1)
                self.assertEqual(status['invalid_jpegs'], 1)
                self.assertEqual(status['frames_received'], 1)
            finally:
                reader.close()

    def test_reconnect_invalidates_old_frame_and_preserves_local_sequence(self):
        consumed = Event()

        def handler(board, connection, index):
            connection.sendall(build_jlip_packet(JLIP_JPEG, 1, FACE_JPEG))
            (consumed if index == 0 else board.stop).wait(3)

        with BoardServer(handler) as board:
            reader = NcmLatestFrameCapture(board.config)
            try:
                first = reader.read_after(timeout=2)
                self.assertIsNotNone(first)
                consumed.set()
                wait_until(lambda: reader.status()['stream_epoch'] >= 2)
                second = reader.read_after(first.sequence, timeout=2)
                self.assertGreater(second.sequence, first.sequence)
                self.assertGreater(second.stream_epoch, first.stream_epoch)
                self.assertFalse(reader.is_current(first))
                self.assertTrue(reader.is_current(second))
            finally:
                reader.close()

    def test_stale_frame_is_not_reused_and_rotation_is_explicit(self):
        def handler(board, connection, index):
            connection.sendall(build_jlip_packet(JLIP_JPEG, 1, FACE_JPEG))
            board.stop.wait(3)

        with BoardServer(handler) as board:
            reader = NcmLatestFrameCapture(replace(board.config, rotate_180=True))
            try:
                packet = reader.read_after(timeout=2)
                expected = cv2.imdecode(np.frombuffer(FACE_JPEG, np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(packet.image, cv2.rotate(expected, cv2.ROTATE_180))
                time.sleep(.55)
                self.assertIsNone(reader.read_after(0, timeout=.01))
            finally:
                reader.close()

    def test_discovery_uses_exact_board_payload(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.bind(('127.0.0.1', 0))
            server.settimeout(2)
            received = []

            def respond():
                payload, peer = server.recvfrom(4096)
                received.append(payload)
                server.sendto(b'{"camera":"test board"}', peer)

            thread = Thread(target=respond, daemon=True)
            thread.start()
            config = NcmCameraConfig(host_ip='127.0.0.1', device_ip='127.0.0.1',
                                     discovery_port=server.getsockname()[1])
            result = discover_camera(config)
            thread.join(2)
            self.assertTrue(result['ok'])
            self.assertEqual(received, [b'JL-CAMERA-DISCOVER'])

    def test_close_interrupts_retry_wait(self):
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))  # Not listening, so connect fails.
            reader = NcmLatestFrameCapture(NcmCameraConfig(host_ip='127.0.0.1',
                device_ip='127.0.0.1', tcp_port=reservation.getsockname()[1], reconnect_delay=10))
            wait_until(lambda: reader.status()['state'] == 'reconnecting')
            started = time.perf_counter()
            reader.close()
            self.assertLess(time.perf_counter() - started, 1)
            self.assertFalse(reader._thread.is_alive())


class AppIntegrationTests(unittest.TestCase):
    @staticmethod
    def stream(board, connection, index):
        sequence = 0
        while not board.stop.is_set():
            connection.sendall(build_jlip_packet(JLIP_JPEG, sequence, FACE_JPEG))
            sequence += 1
            board.stop.wait(.025)

    def test_ncm_headless_profile_uses_real_yunet_at_the_ten_fps_cap(self):
        with BoardServer(self.stream) as board:
            output = io.StringIO()
            with redirect_stdout(output):
                app = LiveFaceDetectorApp(ncm_config=board.config)
                self.assertFalse(app.mirror)
                with patch.object(app, '_draw_hud', wraps=app._draw_hud) as hud:
                    report = app.run(max_frames=6, show_window=False, frame_timeout=3)
                    self.assertEqual(hud.call_args[0][0].shape[:2], (720, 1280))
            self.assertEqual(report['source'], 'ncm')
            self.assertLessEqual(report['live_fps'], 10)
            self.assertGreater(report['live_fps'], 0)
            self.assertEqual(report['width'], 512)
            self.assertTrue(any(face.confirmed for face in app.stabilizer.records.values()))
            self.assertEqual(report['ncm']['crc_errors'], 0)
            self.assertGreater(report['ncm']['frames_received'], report['frames'])
            self.assertIn('JPEG/JLIP', output.getvalue())

    def test_no_frames_times_out_and_closes_receiver(self):
        def handler(board, connection, index):
            board.stop.wait(2)

        with BoardServer(handler) as board:
            with redirect_stdout(io.StringIO()):
                app = LiveFaceDetectorApp(ncm_config=board.config)
                with self.assertRaisesRegex(RuntimeError, 'No usable NCM frame'):
                    app.run(max_frames=2, show_window=False, frame_timeout=.15)
            self.assertFalse(app.capture_reader._thread.is_alive())

    def test_cli_keeps_webcam_default_and_accepts_ncm_alias(self):
        parser = build_argument_parser()
        self.assertEqual(parser.parse_args([]).source, 'webcam')
        args = parser.parse_args(['--ncm', '--ncm-host-ip', 'auto', '--no-mirror'])
        self.assertEqual(args.source, 'ncm')
        self.assertFalse(args.mirror)
        with patch('live_face_detector.discover_camera', return_value={'ok': True}), \
                patch('live_face_detector.LiveFaceDetectorApp') as detector, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--ncm-discover']), 0)
            detector.assert_not_called()


if __name__ == '__main__':
    unittest.main()
