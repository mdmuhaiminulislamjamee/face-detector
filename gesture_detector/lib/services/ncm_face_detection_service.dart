import 'dart:async';
import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../models/detected_face.dart';
import '../ncm/latest_frame_runner.dart';
import '../ncm/ncm_camera_client.dart';
import 'face_tracking_stabilizer.dart';
import 'yunet_worker.dart';
import 'single_face_selector.dart';

class NcmDetectionFrame {
  const NcmDetectionFrame(this.source, this.result, this.faces, this.rotate180);
  final NcmFrame source;
  final YuNetFrameResult result;
  final List<DetectedFace> faces;
  final bool rotate180;
}

class NcmFaceDetectionService extends ChangeNotifier {
  NcmFaceDetectionService({
    NcmCameraClient? client,
    JpegFaceProcessor? processor,
  }) : client = client ?? NcmCameraClient(),
       _processor = processor ?? YuNetWorker() {
    _runner = LatestFrameRunner(
      process: _process,
      onResult: _onResult,
      onError: _onError,
      isFresh: (frame) =>
          this.client.elapsedMicros - frame.receivedAtMicros <= 500000,
    );
    _frameSubscription = this.client.frames.listen(
      (frame) => _runner.submit(frame),
    );
    _statusSubscription = this.client.statuses.listen((status) {
      _status = status;
      if (!status.connected) _clearFrames();
      _notify();
    });
    _watchdog = Timer.periodic(const Duration(milliseconds: 250), (_) {
      final current = _frame;
      if (current != null && frameAge > const Duration(milliseconds: 750)) {
        _clearFrames();
        _notify();
      }
    });
  }

  final NcmCameraClient client;
  final JpegFaceProcessor _processor;
  final _faceSelector = SingleFaceSelector();
  final _tracker = FaceTrackingStabilizer(
    responsiveMotion: true,
    minIouAssociationThreshold: .20,
  );
  late final LatestFrameRunner<NcmFrame, YuNetFrameResult> _runner;
  late final StreamSubscription<NcmFrame> _frameSubscription;
  late final StreamSubscription<NcmStatus> _statusSubscription;
  late final Timer _watchdog;
  final _processingTimes = <int>[];
  NcmCameraConfig _config = const NcmCameraConfig();
  NcmStatus _status = const NcmStatus();
  NcmDetectionFrame? _frame;
  String? _error;
  bool _resetWorker = true, _disposed = false, _initializing = false;
  int _generation = 0;

  NcmStatus get status => _status;
  NcmDetectionFrame? get frame => _frame;
  String? get error => _error ?? _status.error;
  bool get initializing => _initializing;
  int get droppedFrames => _runner.droppedFrames;
  Duration get frameAge => _frame == null
      ? Duration.zero
      : Duration(
          microseconds: client.elapsedMicros - _frame!.source.receivedAtMicros,
        );
  double get detectionFps {
    final now = client.elapsedMicros;
    _processingTimes.removeWhere((time) => now - time > 3000000);
    if (_processingTimes.length < 2) return 0;
    return (_processingTimes.length - 1) *
        1000000 /
        (_processingTimes.last - _processingTimes.first);
  }

  Future<void> start(NcmCameraConfig config) async {
    stop();
    final generation = ++_generation;
    _config = config;
    _initializing = true;
    _notify();
    try {
      await _processor.initialize();
      if (_disposed || generation != _generation) return;
      client.start(config);
    } catch (error) {
      if (!_disposed && generation == _generation) _error = error.toString();
    } finally {
      if (!_disposed && generation == _generation) {
        _initializing = false;
        _notify();
      }
    }
  }

  void stop() {
    _generation++;
    _initializing = false;
    client.stop();
    _clearFrames();
    _error = null;
    _notify();
  }

  void _clearFrames() {
    _runner.clear();
    _tracker.clear();
    _faceSelector.clear();
    _processingTimes.clear();
    _frame = null;
    _resetWorker = true;
  }

  Future<YuNetFrameResult> _process(NcmFrame frame) {
    final reset = _resetWorker;
    _resetWorker = false;
    _processingTimes.add(client.elapsedMicros);
    return _processor.process(
      frame.jpeg,
      rotate180: _config.rotate180,
      reset: reset,
    );
  }

  void _onResult(NcmFrame source, YuNetFrameResult result) {
    // Never display a late answer after a stall. A new frame will be processed next.
    if (client.elapsedMicros - source.receivedAtMicros > 750000) {
      _tracker.clear();
      _faceSelector.clear();
      _resetWorker = true;
      return;
    }
    if (_frame != null &&
        (_frame!.result.width != result.width ||
            _frame!.result.height != result.height)) {
      _tracker.clear();
      _faceSelector.clear();
    }
    const landmarkTypes = [
      FaceLandmarkType.rightEye,
      FaceLandmarkType.leftEye,
      FaceLandmarkType.noseBase,
      FaceLandmarkType.rightMouth,
      FaceLandmarkType.leftMouth,
    ];
    final rawFaces = [
      for (final candidate in result.faces)
        DetectedFace(
          boundingBox: Rect.fromLTWH(
            candidate.box.x,
            candidate.box.y,
            candidate.box.width,
            candidate.box.height,
          ),
          displayBox: Rect.fromLTWH(
            candidate.box.x / result.width,
            candidate.box.y / result.height,
            candidate.box.width / result.width,
            candidate.box.height / result.height,
          ),
          detectionConfidence: candidate.score,
          headEulerAngleZ:
              math.atan2(
                candidate.landmarks[1].$2 - candidate.landmarks[0].$2,
                candidate.landmarks[1].$1 - candidate.landmarks[0].$1,
              ) *
              180 /
              math.pi,
          displayLandmarks: {
            for (var i = 0; i < 5; i++)
              landmarkTypes[i]: Offset(
                candidate.landmarks[i].$1 / result.width,
                candidate.landmarks[i].$2 / result.height,
              ),
          },
        ),
    ];
    final faces = _faceSelector.select(
      _tracker.process(rawFaces: rawFaces, now: DateTime.now()),
    );
    _frame = NcmDetectionFrame(source, result, faces, _config.rotate180);
    _error = null;
    _notify();
  }

  void _onError(Object error, StackTrace trace) {
    _frame = null;
    _tracker.clear();
    _faceSelector.clear();
    _resetWorker = true;
    _error = 'Face processing failed: $error';
    _notify();
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _generation++;
    _watchdog.cancel();
    _runner.dispose();
    unawaited(_frameSubscription.cancel());
    unawaited(_statusSubscription.cancel());
    unawaited(client.dispose());
    unawaited(_processor.dispose());
    super.dispose();
  }
}
