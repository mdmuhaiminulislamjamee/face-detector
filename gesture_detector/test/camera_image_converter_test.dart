import 'package:face_detector/utils/camera_image_converter.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

void main() {
  group('computeMlKitRotation', () {
    test('computes rotation for back camera', () {
      final rotation = computeMlKitRotation(
        sensorOrientation: 90,
        deviceOrientation: DeviceOrientation.portraitUp,
        isFrontCamera: false,
      );
      expect(rotation, InputImageRotation.rotation90deg);
    });

    test('computes rotation for front camera', () {
      final rotation = computeMlKitRotation(
        sensorOrientation: 270,
        deviceOrientation: DeviceOrientation.portraitUp,
        isFrontCamera: true,
      );
      expect(rotation, InputImageRotation.rotation270deg);
    });

    test('compensates for landscape device orientation', () {
      final rotation = computeMlKitRotation(
        sensorOrientation: 90,
        deviceOrientation: DeviceOrientation.landscapeLeft,
        isFrontCamera: false,
      );
      expect(rotation, InputImageRotation.rotation0deg);
    });
  });

  group('projectMlKitPointToDisplay', () {
    const imageSize = Size(640, 480);

    test('normalizes point with rotation 0 without mirroring', () {
      final point = projectMlKitPointToDisplay(
        const Offset(160, 120),
        imageSize: imageSize,
        rotation: InputImageRotation.rotation0deg,
        isIOS: false,
        mirrorHorizontally: false,
      );
      expect(point.dx, closeTo(0.25, 0.001));
      expect(point.dy, closeTo(0.25, 0.001));
    });

    test('mirrors horizontally for front camera', () {
      final point = projectMlKitPointToDisplay(
        const Offset(160, 120),
        imageSize: imageSize,
        rotation: InputImageRotation.rotation0deg,
        isIOS: false,
        mirrorHorizontally: true,
      );
      expect(point.dx, closeTo(0.75, 0.001));
      expect(point.dy, closeTo(0.25, 0.001));
    });

    test('handles 90 degree Android rotation swap', () {
      final point = projectMlKitPointToDisplay(
        const Offset(120, 160),
        imageSize: imageSize,
        rotation: InputImageRotation.rotation90deg,
        isIOS: false,
        mirrorHorizontally: false,
      );
      expect(point.dx, closeTo(120 / 480, 0.001));
      expect(point.dy, closeTo(160 / 640, 0.001));
    });
  });

  group('projectMlKitRectToDisplay', () {
    const imageSize = Size(200, 100);
    const rawRect = Rect.fromLTRB(20, 10, 80, 50);

    test('normalizes an unrotated bounding box', () {
      final rect = projectMlKitRectToDisplay(
        rawRect,
        imageSize: imageSize,
        rotation: InputImageRotation.rotation0deg,
        isIOS: false,
        mirrorHorizontally: false,
      );
      expect(rect.left, closeTo(0.1, 0.001));
      expect(rect.top, closeTo(0.1, 0.001));
      expect(rect.right, closeTo(0.4, 0.001));
      expect(rect.bottom, closeTo(0.5, 0.001));
    });

    test('mirrors bounding box horizontally', () {
      final rect = projectMlKitRectToDisplay(
        rawRect,
        imageSize: imageSize,
        rotation: InputImageRotation.rotation0deg,
        isIOS: false,
        mirrorHorizontally: true,
      );
      expect(rect.left, closeTo(0.6, 0.001));
      expect(rect.top, closeTo(0.1, 0.001));
      expect(rect.right, closeTo(0.9, 0.001));
      expect(rect.bottom, closeTo(0.5, 0.001));
    });
  });

  group('encodeNv21', () {
    test('returns null for insufficient planes', () {
      final frame = CameraPixelFrameData(
        width: 100,
        height: 100,
        planes: [
          CameraPixelPlaneData(bytes: Uint8List(10000), bytesPerRow: 100, bytesPerPixel: 1),
        ],
      );
      expect(encodeNv21(frame), isNull);
    });

    test('encodes valid yuv420 frame to nv21 byte array', () {
      const width = 4;
      const height = 4;
      final yBytes = Uint8List(width * height)..fillRange(0, 16, 200);
      final uBytes = Uint8List((width ~/ 2) * (height ~/ 2))..fillRange(0, 4, 100);
      final vBytes = Uint8List((width ~/ 2) * (height ~/ 2))..fillRange(0, 4, 50);

      final frame = CameraPixelFrameData(
        width: width,
        height: height,
        planes: [
          CameraPixelPlaneData(bytes: yBytes, bytesPerRow: width, bytesPerPixel: 1),
          CameraPixelPlaneData(bytes: uBytes, bytesPerRow: width ~/ 2, bytesPerPixel: 1),
          CameraPixelPlaneData(bytes: vBytes, bytesPerRow: width ~/ 2, bytesPerPixel: 1),
        ],
      );

      final nv21 = encodeNv21(frame);
      expect(nv21, isNotNull);
      expect(nv21!.length, 16 + 8);
      expect(nv21[0], 200);
      expect(nv21[16], 50);
      expect(nv21[17], 100);
    });
  });
}
