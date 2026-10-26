import 'dart:ui';
import 'package:face_detector/utils/face_geometry_utils.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('orientedPreviewSize', () {
    test('aligns preview to portrait aspect ratio', () {
      final size = orientedPreviewSize(
        rawPreviewSize: const Size(1920, 1080),
        isLandscape: false,
      );
      expect(size.width, 1080);
      expect(size.height, 1920);
    });

    test('aligns preview to landscape aspect ratio', () {
      final size = orientedPreviewSize(
        rawPreviewSize: const Size(1920, 1080),
        isLandscape: true,
      );
      expect(size.width, 1920);
      expect(size.height, 1080);
    });

    test('falls back safely for null or zero size', () {
      final sizePortrait = orientedPreviewSize(
        rawPreviewSize: null,
        isLandscape: false,
      );
      expect(sizePortrait, const Size(9, 16));

      final sizeLandscape = orientedPreviewSize(
        rawPreviewSize: Size.zero,
        isLandscape: true,
      );
      expect(sizeLandscape, const Size(16, 9));
    });
  });

  group('displayRectToCanvasRect', () {
    test('maps normalized box to canvas pixel coordinates', () {
      const displayRect = Rect.fromLTRB(0.1, 0.2, 0.6, 0.8);
      const canvasSize = Size(1000, 2000);
      final canvasRect = displayRectToCanvasRect(displayRect, canvasSize);

      expect(canvasRect.left, 100);
      expect(canvasRect.top, 400);
      expect(canvasRect.right, 600);
      expect(canvasRect.bottom, 1600);
    });

    test('clamps bounds to canvas boundaries', () {
      const displayRect = Rect.fromLTRB(-0.2, -0.1, 1.5, 1.2);
      const canvasSize = Size(500, 1000);
      final canvasRect = displayRectToCanvasRect(displayRect, canvasSize);

      expect(canvasRect.left, 0);
      expect(canvasRect.top, 0);
      expect(canvasRect.right, 500);
      expect(canvasRect.bottom, 1000);
    });

    test('returns empty rect for empty input or zero canvas', () {
      expect(
        displayRectToCanvasRect(Rect.zero, const Size(100, 100)),
        Rect.zero,
      );
      expect(
        displayRectToCanvasRect(const Rect.fromLTWH(0, 0, 1, 1), Size.zero),
        Rect.zero,
      );
    });
  });

  group('displayPointToCanvasPoint', () {
    test('maps normalized point to canvas offset', () {
      const point = Offset(0.25, 0.75);
      const canvasSize = Size(800, 1200);
      final canvasPoint = displayPointToCanvasPoint(point, canvasSize);

      expect(canvasPoint.dx, 200);
      expect(canvasPoint.dy, 900);
    });
  });
}
