import 'dart:ui';

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

/// ML Kit has no detection-confidence score. Validate measurable face geometry
/// and let the temporal gate require repeated observations before display.
bool isPlausibleFace(Face face) {
  final box = face.boundingBox;
  if (!box.left.isFinite ||
      !box.top.isFinite ||
      !box.width.isFinite ||
      !box.height.isFinite ||
      box.width < 28 ||
      box.height < 28 ||
      box.height / box.width < 0.65 ||
      box.height / box.width > 2.0) {
    return false;
  }
  Offset? point(FaceLandmarkType type) {
    final landmark = face.landmarks[type];
    if (landmark == null) return null;
    return Offset(
      landmark.position.x.toDouble(),
      landmark.position.y.toDouble(),
    );
  }

  final left = point(FaceLandmarkType.leftEye);
  final right = point(FaceLandmarkType.rightEye);
  final nose = point(FaceLandmarkType.noseBase);
  if (left == null || right == null || nose == null) return false;
  final bounds = box.inflate(box.width * 0.2);
  if (![left, right, nose].every(bounds.contains)) return false;
  final eyeLine = right - left;
  final separation = eyeLine.distance;
  if (separation < box.width * 0.12 || separation > box.width * 0.9) {
    return false;
  }
  final eyeMid = (left + right) / 2;
  final noseVector = nose - eyeMid;
  final depth =
      (eyeLine.dx * noseVector.dy - eyeLine.dy * noseVector.dx).abs() /
      separation;
  if (depth < box.height * 0.04 || depth > box.height * 0.65) return false;
  final mouth = point(FaceLandmarkType.bottomMouth);
  if (mouth != null) {
    if (!bounds.contains(mouth)) return false;
    final down = mouth - eyeMid;
    final length = down.distance;
    if (length < box.height * 0.12 || length > box.height * 0.9) return false;
    final noseDepth =
        (noseVector.dx * down.dx + noseVector.dy * down.dy) / (length * length);
    if (noseDepth < -0.15 || noseDepth > 1.25) return false;
  }
  return true;
}
