import 'package:face_detector/models/face_detector_options_model.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

void main() {
  group('FaceDetectorConfig', () {
    test('defaults to accurate mode and enables standard features', () {
      const config = FaceDetectorConfig();
      expect(config.performanceMode, FaceDetectorMode.accurate);
      expect(config.minFaceSize, 0.03);
      expect(config.maxDropoutHoldDurationMs, 200);
      expect(config.enableTracking, isTrue);
      expect(config.enableStabilizer, isTrue);
      expect(config.enableLandmarks, isTrue);
      expect(config.enableContours, isTrue);
      expect(config.enableClassification, isTrue);
      expect(config.showBoundingBoxes, isTrue);
      expect(config.showMesh, isTrue);
      expect(config.showLandmarks, isTrue);
      expect(config.showMetrics, isTrue);
    });

    test('copyWith updates properties properly', () {
      const config = FaceDetectorConfig();
      final updated = config.copyWith(
        performanceMode: FaceDetectorMode.accurate,
        showMesh: false,
        minFaceSize: 0.25,
      );

      expect(updated.performanceMode, FaceDetectorMode.accurate);
      expect(updated.showMesh, isFalse);
      expect(updated.minFaceSize, 0.25);
      expect(updated.showLandmarks, isTrue);
    });

    test('converts to valid ML Kit options', () {
      const config = FaceDetectorConfig(
        performanceMode: FaceDetectorMode.accurate,
        enableTracking: false,
      );
      final mlKitOptions = config.toMlKitOptions();

      expect(mlKitOptions.performanceMode, FaceDetectorMode.accurate);
      expect(mlKitOptions.enableTracking, isFalse);
      expect(mlKitOptions.enableContours, isFalse);
      expect(mlKitOptions.enableLandmarks, isTrue);
    });

    test('hiding landmarks does not disable geometry validation', () {
      const config = FaceDetectorConfig(enableLandmarks: false);
      expect(config.toMlKitOptions().enableLandmarks, isTrue);
    });
  });
}
