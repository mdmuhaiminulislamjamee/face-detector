import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'jlip_protocol.dart';

class NcmCameraConfig {
  const NcmCameraConfig({
    this.hostAddress = '192.168.50.1',
    this.deviceAddress = '192.168.50.2',
    this.tcpPort = 5000,
    this.discoveryPort = 5001,
    this.rotate180 = false,
  });

  /// Bind to the USB interface so Wi-Fi/cellular cannot capture the route.
  /// Empty means let the operating system choose the local interface.
  final String hostAddress;
  final String deviceAddress;
  final int tcpPort;
  final int discoveryPort;
  final bool rotate180;
}

class NcmFrame {
  const NcmFrame(this.jpeg, this.sequence, this.receivedAtMicros);
  final Uint8List jpeg;
  final int sequence;

  /// Monotonic receive time in the client's clock; board clocks are independent.
  final int receivedAtMicros;
}

class NcmStatus {
  const NcmStatus({
    this.state = 'stopped',
    this.error,
    this.frames = 0,
    this.cameraFps = 0,
    this.crcErrors = 0,
    this.headerErrors = 0,
    this.invalidJpegs = 0,
    this.sequenceGaps = 0,
    this.reconnects = 0,
  });
  final String state;
  final String? error;
  final int frames,
      crcErrors,
      headerErrors,
      invalidJpegs,
      sequenceGaps,
      reconnects;
  final double cameraFps;
  bool get connected => state == 'connected';
}

/// Direct, outbound USB-NCM TCP connection. No PC/backend is involved.
class NcmCameraClient {
  final _clock = Stopwatch()..start();
  final _frames = StreamController<NcmFrame>.broadcast(sync: true);
  final _statuses = StreamController<NcmStatus>.broadcast(sync: true);
  final _frameTimes = <int>[];
  Socket? _socket;
  Completer<void>? _wakeReconnect;
  bool _disposed = false;
  int _generation = 0;
  int _clientSequence = 0;
  int _frameCount = 0, _crcErrors = 0, _headerErrors = 0;
  int _invalidJpegs = 0, _sequenceGaps = 0, _reconnects = 0;
  NcmStatus _status = const NcmStatus();

  Stream<NcmFrame> get frames => _frames.stream;
  Stream<NcmStatus> get statuses => _statuses.stream;
  NcmStatus get status => _status;
  int get elapsedMicros => _clock.elapsedMicroseconds;

  void start(NcmCameraConfig config) {
    if (_disposed) throw StateError('NCM client disposed');
    stop();
    _frameCount = _crcErrors = _headerErrors = _invalidJpegs = 0;
    _sequenceGaps = _reconnects = _clientSequence = 0;
    unawaited(_run(config, ++_generation));
  }

  void stop() {
    _generation++;
    _socket?.destroy();
    _socket = null;
    final wake = _wakeReconnect;
    _wakeReconnect = null;
    if (wake != null && !wake.isCompleted) wake.complete();
    _frameTimes.clear();
    if (!_disposed) _emit('stopped');
  }

  Future<void> _run(NcmCameraConfig config, int generation) async {
    bool active() => !_disposed && generation == _generation;
    while (active()) {
      _emit('connecting');
      Socket? connection;
      try {
        connection = await Socket.connect(
          config.deviceAddress,
          config.tcpPort,
          sourceAddress: config.hostAddress.isEmpty ? null : config.hostAddress,
          timeout: const Duration(seconds: 3),
        );
        if (!active()) {
          connection.destroy();
          return;
        }
        _socket = connection;
        connection.setOption(SocketOption.tcpNoDelay, true);
        _frameTimes.clear();
        _emit('connected');
        await _receive(connection, generation);
        if (active()) throw const SocketException('Camera stream closed');
      } catch (error) {
        if (!active()) return;
        _reconnects++;
        _frameTimes.clear();
        _emit('reconnecting', error.toString());
      } finally {
        connection?.destroy();
        if (identical(_socket, connection)) _socket = null;
      }
      if (!active()) return;
      final wake = Completer<void>();
      _wakeReconnect = wake;
      await Future.any([
        wake.future,
        Future<void>.delayed(const Duration(seconds: 1)),
      ]);
      if (identical(_wakeReconnect, wake)) _wakeReconnect = null;
    }
  }

  Future<void> _receive(Socket connection, int generation) async {
    final parser = JlipStreamParser();
    int? lastSequence;
    var lastBytesAt = elapsedMicros;
    var oldCrc = 0, oldHeaders = 0;
    // A silent but open TCP socket must reconnect too.
    final watchdog = Timer.periodic(const Duration(seconds: 1), (_) {
      if (elapsedMicros - lastBytesAt > 5000000) connection.destroy();
    });
    try {
      await for (final data in connection) {
        if (generation != _generation || _disposed) return;
        lastBytesAt = elapsedMicros;
        NcmFrame? newest;
        for (final packet in parser.feed(data)) {
          if (lastSequence != null) {
            final gap = (packet.sequence - lastSequence - 1) & 0xffffffff;
            // Ignore repeats/out-of-order values; don't count them as billions of losses.
            if (gap < 0x80000000) _sequenceGaps += gap;
          }
          lastSequence = packet.sequence;
          if (packet.type == jlipHeartbeat) {
            // The board's host tester replies with 0x03, rather than 0x04.
            connection.add(
              buildJlipPacket(
                type: jlipHeartbeat,
                sequence: _clientSequence++,
                timestampMs: elapsedMicros ~/ 1000,
              ),
            );
          } else if (packet.type == jlipJpeg) {
            final jpeg = validatedJpeg(packet.payload);
            if (jpeg == null) {
              _invalidJpegs++;
              continue;
            }
            final now = elapsedMicros;
            _frameCount++;
            _frameTimes.removeWhere((time) => now - time > 3000000);
            _frameTimes.add(now);
            newest = NcmFrame(jpeg, packet.sequence, now);
          }
        }
        _crcErrors += parser.crcErrors - oldCrc;
        _headerErrors += parser.headerErrors - oldHeaders;
        oldCrc = parser.crcErrors;
        oldHeaders = parser.headerErrors;
        // Coalesced reads publish only their newest image, while still replying
        // to every heartbeat and checking every packet.
        if (newest != null) _frames.add(newest);
        _emit('connected');
      }
    } finally {
      watchdog.cancel();
    }
  }

  void _emit(String state, [String? error]) {
    final span = _frameTimes.length < 2
        ? 0
        : _frameTimes.last - _frameTimes.first;
    _status = NcmStatus(
      state: state,
      error: error,
      frames: _frameCount,
      cameraFps: span <= 0 ? 0 : (_frameTimes.length - 1) * 1000000 / span,
      crcErrors: _crcErrors,
      headerErrors: _headerErrors,
      invalidJpegs: _invalidJpegs,
      sequenceGaps: _sequenceGaps,
      reconnects: _reconnects,
    );
    if (!_disposed) _statuses.add(_status);
  }

  Future<void> dispose() async {
    if (_disposed) return;
    stop();
    _disposed = true;
    await _frames.close();
    await _statuses.close();
  }

  static Future<List<String>> localAddresses() async {
    final interfaces = await NetworkInterface.list(
      type: InternetAddressType.IPv4,
    );
    return [
      for (final network in interfaces)
        for (final address in network.addresses)
          '${network.name}: ${address.address}',
    ];
  }

  static Future<String> discover(NcmCameraConfig config) async {
    final socket = await RawDatagramSocket.bind(
      config.hostAddress.isEmpty ? InternetAddress.anyIPv4 : config.hostAddress,
      0,
    );
    final reply = Completer<String>();
    final subscription = socket.listen(
      (event) {
        if (event != RawSocketEvent.read) return;
        final datagram = socket.receive();
        if (datagram != null &&
            datagram.address.address == config.deviceAddress &&
            datagram.port == config.discoveryPort &&
            !reply.isCompleted) {
          reply.complete(utf8.decode(datagram.data, allowMalformed: true));
        }
      },
      onError: (Object error) {
        if (!reply.isCompleted) reply.completeError(error);
      },
    );
    try {
      socket.send(
        ascii.encode('JL-CAMERA-DISCOVER'),
        InternetAddress(config.deviceAddress),
        config.discoveryPort,
      );
      return await reply.future.timeout(const Duration(seconds: 2));
    } finally {
      await subscription.cancel();
      socket.close();
    }
  }
}
