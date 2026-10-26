"""USB-NCM JLIP/TCP camera source for the existing PC YuNet pipeline.

The transport preserves only the newest validated JPEG. Decoding happens when
the 10-FPS consumer requests a frame, rather than decoding a backlog of images.
"""
from collections import deque
from dataclasses import dataclass
import ipaddress
import json
import math
import socket
import struct
from threading import Condition, Event, Thread
import time
import zlib

from camera_capture import CapturedFrame


JLIP_MAGIC = b'JLIP'
JLIP_HEADER = struct.Struct('>4sBBHIIII')
JLIP_HEADER_SIZE = JLIP_HEADER.size
JLIP_HELLO_ACK = 0x02
JLIP_HEARTBEAT = 0x03
JLIP_JPEG = 0x10
MAX_PAYLOAD_BYTES = 200 * 1024


@dataclass(frozen=True)
class NcmCameraConfig:
    host_ip: str = '192.168.50.1'
    device_ip: str = '192.168.50.2'
    tcp_port: int = 5000
    discovery_port: int = 5001
    connect_timeout: float = 3.0
    receive_timeout: float = .25
    reconnect_delay: float = 1.0
    silent_timeout: float = 5.0
    rotate_180: bool = False

    def __post_init__(self):
        if self.host_ip:
            ipaddress.IPv4Address(self.host_ip)
        ipaddress.IPv4Address(self.device_ip)
        for value in (self.tcp_port, self.discovery_port):
            if not 1 <= value <= 65535:
                raise ValueError('NCM ports must be in the range 1-65535')
        for value in (self.connect_timeout, self.receive_timeout,
                      self.reconnect_delay, self.silent_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('NCM timeouts must be finite and positive')


@dataclass(frozen=True)
class JlipPacket:
    packet_type: int
    sequence: int
    timestamp_ms: int
    payload: bytes


def build_jlip_packet(packet_type, sequence, payload=b'', timestamp_ms=0):
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError('JLIP payload is too large')
    return JLIP_HEADER.pack(JLIP_MAGIC, 1, packet_type, JLIP_HEADER_SIZE,
                            sequence & 0xffffffff, len(payload),
                            zlib.crc32(payload) & 0xffffffff,
                            timestamp_ms & 0xffffffff) + payload


class JlipStreamParser:
    """Reassembles fragmented/coalesced reads with bounded buffering and CRC32."""
    def __init__(self, max_payload_bytes=MAX_PAYLOAD_BYTES):
        self.max_payload_bytes = max_payload_bytes
        self.buffer = bytearray()
        self.crc_errors = 0
        self.header_errors = 0
        self.discarded_bytes = 0

    def feed(self, data):
        packets = []
        # A caller may pass many packets at once; retained partial data is still
        # bounded to one maximum-size packet rather than the whole input.
        capacity = self.max_payload_bytes + JLIP_HEADER_SIZE
        for offset in range(0, len(data), capacity):
            self.buffer.extend(data[offset:offset + capacity])
            consumed = 0
            while True:
                start = self.buffer.find(JLIP_MAGIC, consumed)
                if start < 0:
                    keep = min(3, len(self.buffer) - consumed)
                    discarded = len(self.buffer) - consumed - keep
                    self.discarded_bytes += discarded
                    consumed += discarded
                    break
                self.discarded_bytes += start - consumed
                consumed = start
                if len(self.buffer) - start < JLIP_HEADER_SIZE:
                    break
                (_, version, packet_type, header_size, sequence, length,
                 checksum, timestamp) = JLIP_HEADER.unpack_from(self.buffer, start)
                if version != 1 or header_size != JLIP_HEADER_SIZE or length > self.max_payload_bytes:
                    consumed += 1
                    self.header_errors += 1
                    self.discarded_bytes += 1
                    continue
                end = start + header_size + length
                if len(self.buffer) < end:
                    break
                payload = bytes(self.buffer[start + header_size:end])
                consumed = end
                if zlib.crc32(payload) & 0xffffffff != checksum:
                    self.crc_errors += 1
                    continue
                packets.append(JlipPacket(packet_type, sequence, timestamp, payload))
            if consumed:
                del self.buffer[:consumed]
        return packets


def validated_jpeg(payload):
    """Trim board padding and reject invalid/oversized dimensions before decode."""
    if not payload.startswith(b'\xff\xd8'):
        return None
    end = payload.rfind(b'\xff\xd9')
    if end < 2:
        return None
    offset = 2
    has_dimensions = False
    while offset < end:
        if payload[offset] != 0xff:
            return None
        while offset < end and payload[offset] == 0xff:
            offset += 1
        if offset >= end:
            return None
        marker = payload[offset]
        offset += 1
        if marker == 0xda:
            break
        if marker == 1 or 0xd0 <= marker <= 0xd7:
            continue
        if offset + 2 > end:
            return None
        length = int.from_bytes(payload[offset:offset + 2], 'big')
        if length < 2 or offset + length > end:
            return None
        if marker in (0xc0, 0xc1, 0xc2):
            if length < 8:
                return None
            height, width = struct.unpack_from('>HH', payload, offset + 3)
            if (min(width, height) < 32 or max(width, height) > 8192
                    or width * height > 12000000):
                return None
            has_dimensions = True
        offset += length
    return payload[:end + 2] if has_dimensions else None


def discover_camera(config):
    """Unicast discovery; does not modify the host's network configuration."""
    started = time.perf_counter()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as discovery:
            discovery.settimeout(2.0)
            discovery.bind((config.host_ip or '0.0.0.0', 0))
            discovery.sendto(b'JL-CAMERA-DISCOVER',
                             (config.device_ip, config.discovery_port))
            deadline = time.perf_counter() + 2.0
            while True:
                payload, peer = discovery.recvfrom(4096)
                if peer == (config.device_ip, config.discovery_port):
                    return {'ok': True, 'peer_ip': peer[0], 'peer_port': peer[1],
                            'reply': payload.decode('utf-8', errors='replace'),
                            'round_trip_ms': round((time.perf_counter() - started) * 1000, 2)}
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise socket.timeout('No reply from the configured board')
                discovery.settimeout(remaining)
    except OSError as error:
        return {'ok': False, 'device_ip': config.device_ip,
                'discovery_port': config.discovery_port, 'error': str(error),
                'hint': 'Check the USB-NCM adapter, host IP/prefix, board power and firmware.'}


@dataclass(frozen=True)
class _EncodedFrame:
    sequence: int
    jpeg: bytes
    received_at: float
    camera_fps: float
    stream_epoch: int


class NcmLatestFrameCapture:
    """LatestFrameCapture-compatible source with heartbeat/reconnect support."""
    def __init__(self, config=None):
        self.config = config or NcmCameraConfig()
        self._condition = Condition()
        self._stop = Event()
        self._socket = None
        self._latest = None
        self._state = 'connecting'
        self._error = None
        self._sequence = 0
        self._epoch = 0
        self._client_sequence = 0
        self._times = deque(maxlen=300)
        self._counts = dict(frames_received=0, frames_decoded=0, crc_errors=0,
                            header_errors=0, invalid_jpegs=0, decode_errors=0,
                            sequence_gaps=0, reconnects=0)
        self._metadata = None
        self._thread = Thread(target=self._run, name='ncm-jlip-receiver', daemon=True)
        self._thread.start()

    def _set_state(self, state, error=None):
        with self._condition:
            self._state, self._error = state, error
            if state != 'connected':
                self._latest = None
                self._times.clear()
            self._condition.notify_all()

    def _close_socket(self, connection=None):
        with self._condition:
            connection = connection or self._socket
            if self._socket is connection:
                self._socket = None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def _run(self):
        try:
            while not self._stop.is_set():
                self._set_state('connecting')
                connection = None
                try:
                    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    with self._condition:
                        if self._stop.is_set():
                            connection.close()
                            break
                        self._socket = connection
                    connection.settimeout(self.config.connect_timeout)
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    connection.bind((self.config.host_ip or '0.0.0.0', 0))
                    connection.connect((self.config.device_ip, self.config.tcp_port))
                    connection.settimeout(self.config.receive_timeout)
                    with self._condition:
                        self._epoch += 1
                    self._set_state('connected')
                    self._receive(connection)
                    if not self._stop.is_set():
                        raise ConnectionError('Board closed the JLIP TCP stream')
                except OSError as error:
                    if not self._stop.is_set():
                        with self._condition:
                            self._counts['reconnects'] += 1
                        self._set_state('reconnecting', str(error))
                finally:
                    self._close_socket(connection)
                self._stop.wait(self.config.reconnect_delay)
        finally:
            self._set_state('stopped')

    def _receive(self, connection):
        parser = JlipStreamParser()
        last_sequence = None
        last_bytes_at = time.perf_counter()
        while not self._stop.is_set():
            try:
                data = connection.recv(65536)
            except socket.timeout:
                if time.perf_counter() - last_bytes_at >= self.config.silent_timeout:
                    raise ConnectionError('Board stream timed out; reconnecting')
                continue
            if not data:
                return
            last_bytes_at = time.perf_counter()
            old_crc, old_headers = parser.crc_errors, parser.header_errors
            packets = parser.feed(data)
            newest = None
            with self._condition:
                self._counts['crc_errors'] += parser.crc_errors - old_crc
                self._counts['header_errors'] += parser.header_errors - old_headers
            for packet in packets:
                if last_sequence is not None:
                    gap = (packet.sequence - last_sequence - 1) & 0xffffffff
                    if gap < 0x80000000:
                        with self._condition:
                            self._counts['sequence_gaps'] += gap
                last_sequence = packet.sequence
                if packet.packet_type == JLIP_HEARTBEAT:
                    # The supplied board tester responds with 0x03 (not 0x04).
                    connection.sendall(build_jlip_packet(JLIP_HEARTBEAT,
                        self._client_sequence, timestamp_ms=int(time.time() * 1000)))
                    self._client_sequence = (self._client_sequence + 1) & 0xffffffff
                elif packet.packet_type == JLIP_HELLO_ACK:
                    try:
                        metadata = json.loads(packet.payload.decode('utf-8'))
                    except (ValueError, UnicodeDecodeError):
                        metadata = packet.payload.decode('utf-8', errors='replace')
                    with self._condition:
                        self._metadata = metadata
                elif packet.packet_type == JLIP_JPEG:
                    jpeg = validated_jpeg(packet.payload)
                    if jpeg is None:
                        with self._condition:
                            self._counts['invalid_jpegs'] += 1
                        continue
                    now = time.perf_counter()
                    with self._condition:
                        self._sequence += 1
                        self._counts['frames_received'] += 1
                        self._times.append(now)
                        self._prune_times(now)
                        newest = _EncodedFrame(self._sequence, jpeg, now,
                                               self._camera_fps(), self._epoch)
            if newest is not None:
                with self._condition:
                    if self._state == 'connected' and not self._stop.is_set():
                        self._latest = newest
                        self._condition.notify_all()

    def _prune_times(self, now):
        while self._times and now - self._times[0] > 3.0:
            self._times.popleft()

    def _camera_fps(self):
        span = self._times[-1] - self._times[0] if len(self._times) > 1 else 0
        return (len(self._times) - 1) / span if span > 0 else 0.0

    def status(self):
        with self._condition:
            self._prune_times(time.perf_counter())
            return dict(self._counts, state=self._state, error=self._error,
                        stream_epoch=self._epoch, camera_fps=round(self._camera_fps(), 2),
                        device_ip=self.config.device_ip, host_ip=self.config.host_ip or 'auto',
                        tcp_port=self.config.tcp_port, device_metadata=self._metadata)

    def is_current(self, packet, max_age=.75):
        with self._condition:
            return (not self._stop.is_set() and self._state == 'connected'
                    and packet.stream_epoch == self._epoch
                    and time.perf_counter() - packet.captured_at <= max_age)

    def read_after(self, sequence=0, timeout=.05):
        with self._condition:
            self._condition.wait_for(lambda: self._stop.is_set() or
                (self._latest is not None and self._latest.sequence > sequence), timeout)
            latest = self._latest
            if (self._stop.is_set() or latest is None or latest.sequence <= sequence
                    or time.perf_counter() - latest.received_at > .5):
                if latest is not None and time.perf_counter() - latest.received_at > .5:
                    self._latest = None
                return None
        # Import only here: protocol/discovery do not need a second image decoder.
        import cv2
        import numpy as np
        started = time.perf_counter()
        try:
            image = cv2.imdecode(np.frombuffer(latest.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError('JPEG decoding failed')
            if self.config.rotate_180:
                image = cv2.rotate(image, cv2.ROTATE_180)
        except (cv2.error, ValueError):
            with self._condition:
                self._counts['decode_errors'] += 1
                if self._latest is latest:
                    self._latest = None
            return None
        packet = CapturedFrame(latest.sequence, image, latest.received_at,
                               time.perf_counter() - started, latest.camera_fps,
                               latest.stream_epoch)
        if not self.is_current(packet):
            return None
        with self._condition:
            self._counts['frames_decoded'] += 1
        return packet

    def reconnect(self):
        self._set_state('reconnecting', 'Reconnect requested')
        self._close_socket()

    def close(self):
        self._stop.set()
        self._set_state('stopped')
        self._close_socket()
        self._thread.join(timeout=4.0)
        if self._thread.is_alive():
            raise RuntimeError('NCM receiver did not stop')
