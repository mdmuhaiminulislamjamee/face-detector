import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart'
    as ml_face;

import '../models/detected_face.dart';
import '../models/face_detector_options_model.dart';
import '../utils/camera_image_converter.dart';
import '../utils/face_candidate_validator.dart';
import 'face_tracking_stabilizer.dart';

/// Service managing the ML Kit Face Detector lifecycle, configuration, image processing,
/// and temporal tracking stabilization.
class FaceDetectorService {
  FaceDetectorService({FaceDetectorConfig config = const FaceDetectorConfig()})
    : _config = config,
      _stabilizer = FaceTrackingStabilizer(
        smoothingAlpha: config.enableStabilizer ? config.smoothingFactor : 1.0,
        maxDropoutHoldDuration: Duration(
          milliseconds: config.enableStabilizer
              ? config.maxDropoutHoldDurationMs
              : 0,
        ),
      ) {
    _initializeDetector();
  }

  FaceDetectorConfig _config;
  final FaceTrackingStabilizer _stabilizer;
  final CameraFrameProcessor _frameProcessor = CameraFrameProcessor();
  ml_face.FaceDetector? _detector;
  ml_face.FaceDetector? _contourDetector;
  bool _isProcessing = false;
  bool _needsReinit = false;
  bool _isDisposed = false;
  int _generation = 0;
  Duration _lastProcessingDuration = Duration.zero;
  List<ml_face.Face> _cachedContourFaces = const [];
  DateTime? _lastContourUpdateAt;
  static const _contourInterval = Duration(milliseconds: 300);
  static const _contourLifetime = Duration(milliseconds: 400);

  FaceDetectorConfig get config => _config;
  bool get isProcessing => _isProcessing;
  Duration get lastProcessingDuration => _lastProcessingDuration;

  void _initializeDetector() {
    _detector?.close();
    _contourDetector?.close();
    _detector = ml_face.FaceDetector(options: _config.toMlKitOptions());
    _contourDetector = _config.enableContours && _config.showMesh
        ? ml_face.FaceDetector(
            options: ml_face.FaceDetectorOptions(
              enableContours: true,
              performanceMode: ml_face.FaceDetectorMode.accurate,
            ),
          )
        : null;
    _needsReinit = false;
  }

  /// Clears active tracking histories (e.g. when camera flips or restarts).
  void clearTracking() {
    _generation++;
    _stabilizer.clear();
    _cachedContourFaces = const [];
    _lastContourUpdateAt = null;
  }

  /// Updates detector options and reinitializes the underlying ML Kit instance if needed.
  void updateConfig(FaceDetectorConfig newConfig) {
    final needsReinit =
        _config.performanceMode != newConfig.performanceMode ||
        _config.enableTracking != newConfig.enableTracking ||
        _config.enableLandmarks != newConfig.enableLandmarks ||
        _config.enableContours != newConfig.enableContours ||
        _config.showMesh != newConfig.showMesh ||
        _config.showMetrics != newConfig.showMetrics ||
        _config.enableClassification != newConfig.enableClassification ||
        _config.minFaceSize != newConfig.minFaceSize;

    _config = newConfig;
    _stabilizer.smoothingAlpha = newConfig.enableStabilizer
        ? newConfig.smoothingFactor
        : 1.0;
    _stabilizer.maxDropoutHoldDuration = Duration(
      milliseconds: newConfig.enableStabilizer
          ? newConfig.maxDropoutHoldDurationMs
          : 0,
    );

    if (needsReinit) {
      clearTracking();
      _needsReinit = true;
      if (!_isProcessing && !_isDisposed) _initializeDetector();
    }
  }

  /// Processes one [CameraImage] frame and returns detected faces with normalized coordinates.
  Future<List<DetectedFace>> processFrame({
    required CameraImage image,
    required CameraDescription camera,
    required DeviceOrientation deviceOrientation,
  }) async {
    final detector = _detector;
    if (detector == null || _isProcessing || _isDisposed) {
      return const [];
    }

    _isProcessing = true;
    final generation = _generation;
    final stopwatch = Stopwatch()..start();

    try {
      final isFrontCamera = camera.lensDirection == CameraLensDirection.front;
      final rotation = computeMlKitRotation(
        sensorOrientation: camera.sensorOrientation,
        deviceOrientation: deviceOrientation,
        isFrontCamera: isFrontCamera,
      );

      final inputImage = await convertCameraImageToMlKitInput(
        image: image,
        rotation: rotation,
        processor: _frameProcessor,
      );
      if (inputImage == null) return const [];

      final faces = (await detector.processImage(
        inputImage,
      )).where(isPlausibleFace).toList();
      if (_isDisposed || generation != _generation) return const [];
      // Detailed contours need substantially more pixels. Keep their single-face
      // mode separate so it cannot suppress other or distant faces.
      final now = DateTime.now();
      final lastContourAt = _lastContourUpdateAt;
      var contourFaces =
          lastContourAt != null &&
              now.difference(lastContourAt) <= _contourLifetime
          ? _cachedContourFaces
          : <ml_face.Face>[];
      final contourDetector = _contourDetector;
      if (contourDetector != null &&
          stopwatch.elapsed < const Duration(milliseconds: 50) &&
          (lastContourAt == null ||
              now.difference(lastContourAt) >= _contourInterval) &&
          faces.any(
            (face) =>
                face.boundingBox.width >= 200 && face.boundingBox.height >= 200,
          )) {
        try {
          contourFaces = await contourDetector.processImage(inputImage);
          if (generation == _generation && !_isDisposed) {
            _cachedContourFaces = contourFaces;
            _lastContourUpdateAt = DateTime.now();
          }
        } catch (error) {
          debugPrint('Optional face contours unavailable: $error');
        }
      }
      if (_isDisposed || generation != _generation) return const [];
      final imageSize = Size(image.width.toDouble(), image.height.toDouble());
      final isIOS = Platform.isIOS;
      final mirrorHorizontally = isFrontCamera && !isIOS;

      final detectedFaces = <DetectedFace>[];
      for (final face in faces) {
        final displayBox = projectMlKitRectToDisplay(
          face.boundingBox,
          imageSize: imageSize,
          rotation: rotation,
          isIOS: isIOS,
          mirrorHorizontally: mirrorHorizontally,
        );

        final displayLandmarks = <ml_face.FaceLandmarkType, Offset>{};
        if (_config.enableLandmarks) {
          for (final entry in face.landmarks.entries) {
            final landmark = entry.value;
            if (landmark != null) {
              displayLandmarks[entry.key] = projectMlKitPointToDisplay(
                Offset(
                  landmark.position.x.toDouble(),
                  landmark.position.y.toDouble(),
                ),
                imageSize: imageSize,
                rotation: rotation,
                isIOS: isIOS,
                mirrorHorizontally: mirrorHorizontally,
              );
            }
          }
        }

        final displayContours = <ml_face.FaceContourType, List<Offset>>{};
        if (_config.enableContours) {
          ml_face.Face? contourFace;
          for (final candidate in contourFaces) {
            final intersection = candidate.boundingBox.intersect(
              face.boundingBox,
            );
            if (intersection.isEmpty) continue;
            final area = intersection.width * intersection.height;
            final union =
                candidate.boundingBox.width * candidate.boundingBox.height +
                face.boundingBox.width * face.boundingBox.height -
                area;
            if (area / union >= 0.4) {
              contourFace = candidate;
              break;
            }
          }
          for (final entry in (contourFace ?? face).contours.entries) {
            final contour = entry.value;
            if (contour != null && contour.points.isNotEmpty) {
              displayContours[entry.key] = [
                for (final point in contour.points)
                  projectMlKitPointToDisplay(
                    contourFace == null
                        ? Offset(point.x.toDouble(), point.y.toDouble())
                        : Offset(
                            face.boundingBox.left +
                                (point.x - contourFace.boundingBox.left) *
                                    face.boundingBox.width /
                                    contourFace.boundingBox.width,
                            face.boundingBox.top +
                                (point.y - contourFace.boundingBox.top) *
                                    face.boundingBox.height /
                                    contourFace.boundingBox.height,
                          ),
                    imageSize: imageSize,
                    rotation: rotation,
                    isIOS: isIOS,
                    mirrorHorizontally: mirrorHorizontally,
                  ),
              ];
            }
          }
        }

        detectedFaces.add(
          DetectedFace(
            trackingId: face.trackingId,
            boundingBox: face.boundingBox,
            displayBox: displayBox,
            headEulerAngleX: face.headEulerAngleX,
            headEulerAngleY: face.headEulerAngleY,
            headEulerAngleZ: face.headEulerAngleZ,
            smilingProbability: face.smilingProbability,
            leftEyeOpenProbability: face.leftEyeOpenProbability,
            rightEyeOpenProbability: face.rightEyeOpenProbability,
            displayLandmarks: displayLandmarks,
            displayContours: displayContours,
          ),
        );
      }

      // Apply tracking stabilizer (smoothing + dropout tolerance)
      // Confirmation remains enabled when visual smoothing is turned off.
      return _stabilizer.process(rawFaces: detectedFaces, now: DateTime.now());
    } catch (error, stackTrace) {
      debugPrint('Face detection error: $error\n$stackTrace');
      _stabilizer.clear();
      return const [];
    } finally {
      stopwatch.stop();
      _lastProcessingDuration = stopwatch.elapsed;
      _isProcessing = false;
      if (_isDisposed) {
        await detector.close();
        await _contourDetector?.close();
        _contourDetector = null;
        await _frameProcessor.dispose();
      } else if (_needsReinit) {
        _initializeDetector();
      }
    }
  }

  /// Closes the underlying ML Kit face detector.
  Future<void> dispose() async {
    _isDisposed = true;
    clearTracking();
    final detector = _detector;
    _detector = null;
    if (!_isProcessing && detector != null) {
      await detector.close();
      await _contourDetector?.close();
      _contourDetector = null;
    }
    if (!_isProcessing) await _frameProcessor.dispose();
  }
}
