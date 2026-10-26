import 'package:face_detector/models/detected_face.dart';
import 'package:face_detector/models/face_detector_options_model.dart';
import 'package:face_detector/presentation/painters/face_detection_painter.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

void main() {
  group('FaceDetectionPainter', () {
    test('repaints when faces or config change', () {
      const config1 = FaceDetectorConfig(showMesh: true);
      const config2 = FaceDetectorConfig(showMesh: false);

      final painter1 = FaceDetectionPainter(faces: const [], config: config1);
      final painter2 = FaceDetectionPainter(faces: const [], config: config2);

      expect(painter1.shouldRepaint(painter2), isTrue);
    });

    testWidgets('paints detected face without throwing', (tester) async {
      const face = DetectedFace(
        trackingId: 1,
        boundingBox: Rect.fromLTWH(50, 50, 100, 100),
        displayBox: Rect.fromLTWH(0.2, 0.2, 0.6, 0.6),
        smilingProbability: 0.9,
        leftEyeOpenProbability: 0.8,
        rightEyeOpenProbability: 0.85,
        headEulerAngleZ: 4.5,
        displayLandmarks: {
          FaceLandmarkType.noseBase: Offset(0.5, 0.5),
        },
        displayContours: {
          FaceContourType.face: [
            Offset(0.2, 0.2),
            Offset(0.8, 0.2),
            Offset(0.8, 0.8),
            Offset(0.2, 0.8),
          ],
        },
      );

      final painter = FaceDetectionPainter(
        faces: const [face],
        config: const FaceDetectorConfig(),
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: CustomPaint(
              size: const Size(400, 800),
              painter: painter,
            ),
          ),
        ),
      );

      expect(find.byType(CustomPaint), findsWidgets);
    });
  });
}
