import 'dart:ui';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

/// Represents a detected face projected into normalized [0..1] preview coordinates.
class DetectedFace {
  const DetectedFace({
    this.trackingId,
    this.detectionConfidence,
    required this.boundingBox,
    required this.displayBox,
    this.headEulerAngleX,
    this.headEulerAngleY,
    this.headEulerAngleZ,
    this.smilingProbability,
    this.leftEyeOpenProbability,
    this.rightEyeOpenProbability,
    this.displayLandmarks = const {},
    this.displayContours = const {},
    this.isTemporarilyLost = false,
    this.missedFrameCount = 0,
  });

  /// Unique tracking identifier assigned across consecutive video frames.
  final int? trackingId;

  /// YuNet model confidence. ML Kit does not supply this value.
  final double? detectionConfidence;

  /// Raw detector bounding box in camera sensor pixel coordinates.
  final Rect boundingBox;

  /// Bounding box normalized to [0..1] preview coordinates after rotation and mirroring.
  final Rect displayBox;

  /// True when the face is temporarily occluded or missed by detector but kept active by the stabilizer.
  final bool isTemporarilyLost;

  /// Consecutive missed frame count since last detector hit.
  final int missedFrameCount;

  /// Head pitch in degrees (looking up / down).
  final double? headEulerAngleX;

  /// Head yaw in degrees (looking left / right).
  final double? headEulerAngleY;

  /// Head roll in degrees (tilting head left / right).
  final double? headEulerAngleZ;

  /// Probability that the face is smiling (0.0 to 1.0).
  final double? smilingProbability;

  /// Probability that the left eye is open (0.0 to 1.0).
  final double? leftEyeOpenProbability;

  /// Probability that the right eye is open (0.0 to 1.0).
  final double? rightEyeOpenProbability;

  /// Normalized positions of facial landmarks (eyes, ears, nose, mouth).
  final Map<FaceLandmarkType, Offset> displayLandmarks;

  /// Normalized point sequences tracing facial contours (eyes, lips, brows, face oval).
  final Map<FaceContourType, List<Offset>> displayContours;

  /// True when the smiling probability exceeds 65%.
  bool get isSmiling => (smilingProbability ?? 0) >= 0.65;

  /// True when the left eye open probability exceeds 50%.
  bool get isLeftEyeOpen => (leftEyeOpenProbability ?? 0) >= 0.50;

  /// True when the right eye open probability exceeds 50%.
  bool get isRightEyeOpen => (rightEyeOpenProbability ?? 0) >= 0.50;

  /// True when both eyes are detected as open.
  bool get areBothEyesOpen => isLeftEyeOpen && isRightEyeOpen;

  /// Formatted smiling percentage string.
  String? get smilingPercentageText => smilingProbability != null
      ? '${(smilingProbability! * 100).toStringAsFixed(0)}%'
      : null;

  /// Formatted left eye open percentage.
  String? get leftEyeOpenPercentageText => leftEyeOpenProbability != null
      ? '${(leftEyeOpenProbability! * 100).toStringAsFixed(0)}%'
      : null;

  /// Formatted right eye open percentage.
  String? get rightEyeOpenPercentageText => rightEyeOpenProbability != null
      ? '${(rightEyeOpenProbability! * 100).toStringAsFixed(0)}%'
      : null;

  DetectedFace copyWith({
    int? trackingId,
    double? detectionConfidence,
    Rect? boundingBox,
    Rect? displayBox,
    double? headEulerAngleX,
    double? headEulerAngleY,
    double? headEulerAngleZ,
    double? smilingProbability,
    double? leftEyeOpenProbability,
    double? rightEyeOpenProbability,
    Map<FaceLandmarkType, Offset>? displayLandmarks,
    Map<FaceContourType, List<Offset>>? displayContours,
    bool? isTemporarilyLost,
    int? missedFrameCount,
  }) {
    return DetectedFace(
      trackingId: trackingId ?? this.trackingId,
      detectionConfidence: detectionConfidence ?? this.detectionConfidence,
      boundingBox: boundingBox ?? this.boundingBox,
      displayBox: displayBox ?? this.displayBox,
      headEulerAngleX: headEulerAngleX ?? this.headEulerAngleX,
      headEulerAngleY: headEulerAngleY ?? this.headEulerAngleY,
      headEulerAngleZ: headEulerAngleZ ?? this.headEulerAngleZ,
      smilingProbability: smilingProbability ?? this.smilingProbability,
      leftEyeOpenProbability:
          leftEyeOpenProbability ?? this.leftEyeOpenProbability,
      rightEyeOpenProbability:
          rightEyeOpenProbability ?? this.rightEyeOpenProbability,
      displayLandmarks: displayLandmarks ?? this.displayLandmarks,
      displayContours: displayContours ?? this.displayContours,
      isTemporarilyLost: isTemporarilyLost ?? this.isTemporarilyLost,
      missedFrameCount: missedFrameCount ?? this.missedFrameCount,
    );
  }
}
