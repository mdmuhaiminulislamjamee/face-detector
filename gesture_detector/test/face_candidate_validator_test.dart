import 'dart:math';
import 'dart:ui';

import 'package:face_detector/utils/face_candidate_validator.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

Face candidate({
  Rect box = const Rect.fromLTWH(100, 100, 100, 120),
  Point<int> leftEye = const Point(125, 135),
  Point<int> rightEye = const Point(175, 135),
  Point<int> nose = const Point(150, 160),
  Point<int> mouth = const Point(150, 185),
  bool includeLandmarks = true,
}) => Face(
  boundingBox: box,
  landmarks: includeLandmarks
      ? {
          FaceLandmarkType.leftEye: FaceLandmark(
            type: FaceLandmarkType.leftEye,
            position: leftEye,
          ),
          FaceLandmarkType.rightEye: FaceLandmark(
            type: FaceLandmarkType.rightEye,
            position: rightEye,
          ),
          FaceLandmarkType.noseBase: FaceLandmark(
            type: FaceLandmarkType.noseBase,
            position: nose,
          ),
          FaceLandmarkType.bottomMouth: FaceLandmark(
            type: FaceLandmarkType.bottomMouth,
            position: mouth,
          ),
        }
      : {},
  contours: {},
);

void main() {
  test('accepts measurable frontal and rolled face geometry', () {
    expect(isPlausibleFace(candidate()), isTrue);
    expect(
      isPlausibleFace(
        candidate(
          leftEye: const Point(165, 125),
          rightEye: const Point(165, 175),
          nose: const Point(140, 150),
          mouth: const Point(115, 150),
        ),
      ),
      isTrue,
    );
  });

  test('rejects missing landmarks, collapsed eyes, and misplaced nose', () {
    expect(isPlausibleFace(candidate(includeLandmarks: false)), isFalse);
    expect(
      isPlausibleFace(candidate(rightEye: const Point(125, 135))),
      isFalse,
    );
    expect(isPlausibleFace(candidate(nose: const Point(150, 300))), isFalse);
    expect(isPlausibleFace(candidate(nose: const Point(150, 135))), isFalse);
  });

  test('rejects undersized and impossible boxes', () {
    expect(
      isPlausibleFace(candidate(box: const Rect.fromLTWH(0, 0, 20, 20))),
      isFalse,
    );
    expect(
      isPlausibleFace(candidate(box: const Rect.fromLTWH(0, 0, 200, 40))),
      isFalse,
    );
  });
}
