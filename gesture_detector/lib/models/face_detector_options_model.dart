import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

/// Runtime configuration options for the face detector engine and overlay painter.
class FaceDetectorConfig {
  const FaceDetectorConfig({
    this.performanceMode = FaceDetectorMode.accurate,
    this.enableTracking = true,
    this.enableLandmarks = true,
    this.enableContours = true,
    this.enableClassification = true,
    this.minFaceSize = 0.03,
    this.showBoundingBoxes = true,
    this.showMesh = true,
    this.showLandmarks = true,
    this.showMetrics = true,
    this.enableStabilizer = true,
    this.maxDropoutHoldDurationMs = 200,
    this.smoothingFactor = 0.65,
  });

  /// ML Kit detection performance mode (accurate by default for far distance & angled faces).
  final FaceDetectorMode performanceMode;

  /// Whether ML Kit should assign persistent tracking IDs across frames.
  final bool enableTracking;

  /// Whether to detect facial landmarks (eyes, ears, nose, mouth points).
  final bool enableLandmarks;

  /// Whether to detect detailed facial contours (face oval, lips, eyebrows).
  final bool enableContours;

  /// Whether to detect smile and eye open probabilities.
  final bool enableClassification;

  /// Desired minimum face width relative to the image (pixel detail still matters).
  final double minFaceSize;

  /// UI painter toggle: display bounding boxes.
  final bool showBoundingBoxes;

  /// UI painter toggle: display contour mesh lines.
  final bool showMesh;

  /// UI painter toggle: display landmark points.
  final bool showLandmarks;

  /// UI painter toggle: display smile and eye metrics badges.
  final bool showMetrics;

  /// Whether to apply temporal tracking stabilization (smoothing and dropout tolerance).
  final bool enableStabilizer;

  /// How long in milliseconds to hold a temporarily lost target face before dropping (grace period).
  final int maxDropoutHoldDurationMs;

  /// Exponential smoothing factor for bounding box jitter reduction (0.0 to 1.0).
  final double smoothingFactor;

  /// Converts this config into options accepted by [FaceDetector].
  FaceDetectorOptions toMlKitOptions() {
    return FaceDetectorOptions(
      performanceMode: performanceMode,
      enableTracking: enableTracking,
      // Landmarks also validate detections when their overlay is hidden.
      enableLandmarks: true,
      // Contours run separately: enabling them here restricts detection to one face.
      enableContours: false,
      enableClassification: enableClassification && showMetrics,
      minFaceSize: minFaceSize,
    );
  }

  FaceDetectorConfig copyWith({
    FaceDetectorMode? performanceMode,
    bool? enableTracking,
    bool? enableLandmarks,
    bool? enableContours,
    bool? enableClassification,
    double? minFaceSize,
    bool? showBoundingBoxes,
    bool? showMesh,
    bool? showLandmarks,
    bool? showMetrics,
    bool? enableStabilizer,
    int? maxDropoutHoldDurationMs,
    double? smoothingFactor,
  }) {
    return FaceDetectorConfig(
      performanceMode: performanceMode ?? this.performanceMode,
      enableTracking: enableTracking ?? this.enableTracking,
      enableLandmarks: enableLandmarks ?? this.enableLandmarks,
      enableContours: enableContours ?? this.enableContours,
      enableClassification: enableClassification ?? this.enableClassification,
      minFaceSize: minFaceSize ?? this.minFaceSize,
      showBoundingBoxes: showBoundingBoxes ?? this.showBoundingBoxes,
      showMesh: showMesh ?? this.showMesh,
      showLandmarks: showLandmarks ?? this.showLandmarks,
      showMetrics: showMetrics ?? this.showMetrics,
      enableStabilizer: enableStabilizer ?? this.enableStabilizer,
      maxDropoutHoldDurationMs:
          maxDropoutHoldDurationMs ?? this.maxDropoutHoldDurationMs,
      smoothingFactor: smoothingFactor ?? this.smoothingFactor,
    );
  }
}
