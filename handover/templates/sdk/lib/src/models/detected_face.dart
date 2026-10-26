import 'dart:ui';

/// Real YuNet landmarks. This SDK has no dependency on Google ML Kit.
enum FaceLandmarkType { rightEye, leftEye, noseBase, rightMouth, leftMouth }

/// One selected track. Coordinates refer to this processed JPEG, after rotation.
class DetectedFace {
  const DetectedFace({
    this.trackingId,
    this.detectionConfidence,
    required this.boundingBox,
    required this.displayBox,
    this.headEulerAngleZ,
    this.displayLandmarks = const {},
    this.isTemporarilyLost = false,
    this.missedFrameCount = 0,
  });

  final int? trackingId;
  final double? detectionConfidence;
  final Rect boundingBox;
  final Rect displayBox;
  final double? headEulerAngleZ;
  final Map<FaceLandmarkType, Offset> displayLandmarks;
  final bool isTemporarilyLost;
  final int missedFrameCount;

  DetectedFace copyWith({
    int? trackingId,
    double? detectionConfidence,
    Rect? boundingBox,
    Rect? displayBox,
    double? headEulerAngleZ,
    Map<FaceLandmarkType, Offset>? displayLandmarks,
    bool? isTemporarilyLost,
    int? missedFrameCount,
  }) => DetectedFace(
    trackingId: trackingId ?? this.trackingId,
    detectionConfidence: detectionConfidence ?? this.detectionConfidence,
    boundingBox: boundingBox ?? this.boundingBox,
    displayBox: displayBox ?? this.displayBox,
    headEulerAngleZ: headEulerAngleZ ?? this.headEulerAngleZ,
    displayLandmarks: displayLandmarks ?? this.displayLandmarks,
    isTemporarilyLost: isTemporarilyLost ?? this.isTemporarilyLost,
    missedFrameCount: missedFrameCount ?? this.missedFrameCount,
  );
}
