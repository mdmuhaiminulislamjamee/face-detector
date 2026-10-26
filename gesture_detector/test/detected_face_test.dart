import 'dart:ui';
import 'package:face_detector/models/detected_face.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

void main() {
  group('DetectedFace', () {
    test('computes smiling status correctly', () {
      const smilingFace = DetectedFace(
        boundingBox: Rect.fromLTWH(10, 10, 100, 100),
        displayBox: Rect.fromLTWH(0.1, 0.1, 0.5, 0.5),
        smilingProbability: 0.85,
      );
      expect(smilingFace.isSmiling, isTrue);
      expect(smilingFace.smilingPercentageText, '85%');

      const neutralFace = DetectedFace(
        boundingBox: Rect.fromLTWH(10, 10, 100, 100),
        displayBox: Rect.fromLTWH(0.1, 0.1, 0.5, 0.5),
        smilingProbability: 0.20,
      );
      expect(neutralFace.isSmiling, isFalse);
      expect(neutralFace.smilingPercentageText, '20%');

      const unknownFace = DetectedFace(
        boundingBox: Rect.fromLTWH(10, 10, 100, 100),
        displayBox: Rect.fromLTWH(0.1, 0.1, 0.5, 0.5),
      );
      expect(unknownFace.isSmiling, isFalse);
      expect(unknownFace.smilingPercentageText, isNull);
    });

    test('computes eye open status correctly', () {
      const openEyesFace = DetectedFace(
        boundingBox: Rect.fromLTWH(10, 10, 100, 100),
        displayBox: Rect.fromLTWH(0.1, 0.1, 0.5, 0.5),
        leftEyeOpenProbability: 0.95,
        rightEyeOpenProbability: 0.90,
      );
      expect(openEyesFace.isLeftEyeOpen, isTrue);
      expect(openEyesFace.isRightEyeOpen, isTrue);
      expect(openEyesFace.areBothEyesOpen, isTrue);
      expect(openEyesFace.leftEyeOpenPercentageText, '95%');
      expect(openEyesFace.rightEyeOpenPercentageText, '90%');

      const winkingFace = DetectedFace(
        boundingBox: Rect.fromLTWH(10, 10, 100, 100),
        displayBox: Rect.fromLTWH(0.1, 0.1, 0.5, 0.5),
        leftEyeOpenProbability: 0.10,
        rightEyeOpenProbability: 0.92,
      );
      expect(winkingFace.isLeftEyeOpen, isFalse);
      expect(winkingFace.isRightEyeOpen, isTrue);
      expect(winkingFace.areBothEyesOpen, isFalse);
    });

    test('stores landmarks and contours correctly', () {
      const face = DetectedFace(
        trackingId: 42,
        boundingBox: Rect.fromLTWH(0, 0, 200, 200),
        displayBox: Rect.fromLTWH(0, 0, 1, 1),
        displayLandmarks: {
          FaceLandmarkType.noseBase: Offset(0.5, 0.5),
        },
        displayContours: {
          FaceContourType.face: [Offset(0.2, 0.2), Offset(0.8, 0.8)],
        },
      );

      expect(face.trackingId, 42);
      expect(face.displayLandmarks[FaceLandmarkType.noseBase], const Offset(0.5, 0.5));
      expect(face.displayContours[FaceContourType.face]?.length, 2);
    });
  });
}
