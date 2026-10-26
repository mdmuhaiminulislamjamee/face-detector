library;

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';

import 'src/models/detected_face.dart';
import 'src/ncm/ncm_camera_client.dart';
import 'src/services/ncm_face_detection_service.dart';
import 'src/services/yunet_worker.dart';

export 'src/models/detected_face.dart';
export 'src/ncm/ncm_camera_client.dart' show NcmCameraConfig, NcmStatus;

/// A processed JPEG and its exclusive selected face. Never has more than one face.
class NcmFaceFrame {
  NcmFaceFrame._(NcmDetectionFrame frame)
    : jpeg = frame.source.jpeg,
      frameId = frame.source.receivedAtMicros,
      sourceSequence = frame.source.sequence,
      width = frame.result.width,
      height = frame.result.height,
      processingDuration = frame.result.duration,
      rotate180 = frame.rotate180,
      faces = List.unmodifiable(frame.faces);

  /// Treat these camera-owned bytes as read-only. Display this JPEG with this result.
  final Uint8List jpeg;
  final int frameId, sourceSequence, width, height;
  final Duration processingDuration;
  final bool rotate180;
  final List<DetectedFace> faces;

  /// A held search lock is never returned here as a fresh detection.
  DetectedFace? get freshFace =>
      faces.isNotEmpty && !faces.single.isTemporarilyLost ? faces.single : null;
}

/// One controller, one persistent worker, one TCP source, and one selected target.
class NcmFaceDetector extends ChangeNotifier with WidgetsBindingObserver {
  NcmFaceDetector({bool manageAppLifecycle = true})
    : _manageAppLifecycle = manageAppLifecycle,
      _service = NcmFaceDetectionService(
        processor: YuNetWorker(
          modelAssetPath: 'packages/ncm_face_sdk/assets/models/face_detection_yunet_2023mar.onnx',
        ),
      ) {
    _service.addListener(_changed);
    if (_manageAppLifecycle) WidgetsBinding.instance.addObserver(this);
  }

  final NcmFaceDetectionService _service;
  final bool _manageAppLifecycle;
  bool _disposed = false, _requested = false;
  NcmCameraConfig? _config;
  NcmDetectionFrame? _sourceFrame;
  NcmFaceFrame? _frame;

  NcmFaceFrame? get frame => _frame;
  NcmStatus get status => _service.status;
  String? get error => _service.error;
  bool get initializing => _service.initializing;
  double get detectionFps => _service.detectionFps;
  Duration get frameAge => _service.frameAge;
  int get droppedFrames => _service.droppedFrames;

  Future<void> start([NcmCameraConfig config = const NcmCameraConfig()]) {
    if (_disposed) throw StateError('NCM detector is disposed');
    for (final address in [
      config.deviceAddress,
      if (config.hostAddress.isNotEmpty) config.hostAddress,
    ]) {
      final parsed = InternetAddress.tryParse(address);
      if (parsed == null || parsed.type != InternetAddressType.IPv4) {
        throw ArgumentError('Expected an IPv4 address: $address');
      }
    }
    if (config.tcpPort < 1 ||
        config.tcpPort > 65535 ||
        config.discoveryPort < 1 ||
        config.discoveryPort > 65535) {
      throw ArgumentError('NCM ports must be between 1 and 65535');
    }
    _config = config;
    _requested = true;
    return _service.start(config);
  }

  void stop() {
    _requested = false;
    _service.stop();
  }

  static Future<List<String>> localAddresses() =>
      NcmCameraClient.localAddresses();
  static Future<String> discover([
    NcmCameraConfig config = const NcmCameraConfig(),
  ]) => NcmCameraClient.discover(config);

  void _changed() {
    if (_disposed) return;
    final source = _service.frame;
    if (!identical(source, _sourceFrame)) {
      _sourceFrame = source;
      _frame = source == null ? null : NcmFaceFrame._(source);
    }
    notifyListeners();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      if (_requested && _config != null && !_disposed) _service.start(_config!);
    } else {
      // Keep user intent so the same camera can reconnect on resume.
      _service.stop();
    }
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _requested = false;
    if (_manageAppLifecycle) WidgetsBinding.instance.removeObserver(this);
    _service.removeListener(_changed);
    _service.dispose();
    _sourceFrame = null;
    _frame = null;
    super.dispose();
  }
}
