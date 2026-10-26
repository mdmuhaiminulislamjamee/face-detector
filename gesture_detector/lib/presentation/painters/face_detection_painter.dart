import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../../models/detected_face.dart';
import '../../models/face_detector_options_model.dart';
import '../../utils/face_geometry_utils.dart';

/// CustomPainter rendering face bounding boxes, tracking badges, landmarks, and contour meshes.
class FaceDetectionPainter extends CustomPainter {
  FaceDetectionPainter({required this.faces, required this.config});

  final List<DetectedFace> faces;
  final FaceDetectorConfig config;

  @override
  void paint(Canvas canvas, Size size) {
    if (faces.isEmpty || size.isEmpty) return;

    for (final face in faces) {
      final rect = displayRectToCanvasRect(face.displayBox, size);
      if (rect.isEmpty) continue;

      // 1. Draw facial contours (mesh)
      if (config.showMesh && !face.isTemporarilyLost) {
        _drawContours(canvas, face, size);
      }

      // 2. Draw facial landmarks
      if (config.showLandmarks && !face.isTemporarilyLost) {
        _drawLandmarks(canvas, face, size);
      }

      // 3. Draw bounding box with corner accents and glow
      if (config.showBoundingBoxes) {
        _drawBoundingBox(
          canvas,
          rect,
          face.trackingId,
          isHoldingLock: face.isTemporarilyLost,
        );
      }

      // 4. Draw metrics badge (smile, eyes, head rotation)
      if (config.showMetrics && !face.isTemporarilyLost) {
        _drawMetrics(canvas, rect, face);
      }
    }
  }

  void _drawBoundingBox(
    Canvas canvas,
    Rect rect,
    int? trackingId, {
    bool isHoldingLock = false,
  }) {
    final borderColor = isHoldingLock
        ? const Color(0xFFFFB300)
        : const Color(0xFF00E5FF);
    final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(12));

    // Outer glow
    final glowPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 6.0
      ..strokeCap = StrokeCap.round
      ..color = borderColor.withValues(alpha: isHoldingLock ? 0.15 : 0.25);
    canvas.drawRRect(rrect, glowPaint);

    // Primary border
    final borderPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round
      ..color = isHoldingLock
          ? borderColor.withValues(alpha: 0.8)
          : borderColor;
    canvas.drawRRect(rrect, borderPaint);

    // Corner guides
    _drawCornerAccents(canvas, rect);

    // Tracking ID badge
    final idText = trackingId != null
        ? (isHoldingLock ? 'Face #$trackingId (Holding)' : 'Face #$trackingId')
        : 'Face';
    _drawBadge(
      canvas: canvas,
      position: Offset(rect.left, rect.top - 24),
      text: idText,
      backgroundColor: isHoldingLock
          ? const Color(0xCC332200)
          : const Color(0xCC002B36),
      textColor: borderColor,
    );
  }

  void _drawCornerAccents(Canvas canvas, Rect rect) {
    final cornerPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.5
      ..strokeCap = StrokeCap.round
      ..color = Colors.white;

    final length = (rect.shortestSide * 0.18).clamp(10.0, 26.0);

    // Top-left
    canvas.drawLine(
      rect.topLeft,
      rect.topLeft + Offset(length, 0),
      cornerPaint,
    );
    canvas.drawLine(
      rect.topLeft,
      rect.topLeft + Offset(0, length),
      cornerPaint,
    );

    // Top-right
    canvas.drawLine(
      rect.topRight,
      rect.topRight + Offset(-length, 0),
      cornerPaint,
    );
    canvas.drawLine(
      rect.topRight,
      rect.topRight + Offset(0, length),
      cornerPaint,
    );

    // Bottom-left
    canvas.drawLine(
      rect.bottomLeft,
      rect.bottomLeft + Offset(length, 0),
      cornerPaint,
    );
    canvas.drawLine(
      rect.bottomLeft,
      rect.bottomLeft + Offset(0, -length),
      cornerPaint,
    );

    // Bottom-right
    canvas.drawLine(
      rect.bottomRight,
      rect.bottomRight + Offset(-length, 0),
      cornerPaint,
    );
    canvas.drawLine(
      rect.bottomRight,
      rect.bottomRight + Offset(0, -length),
      cornerPaint,
    );
  }

  void _drawContours(Canvas canvas, DetectedFace face, Size canvasSize) {
    final contourLinePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.4
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..color = const Color(0xFF00E5FF).withValues(alpha: 0.65);

    final contourDotPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFF76FF03).withValues(alpha: 0.85);

    for (final entry in face.displayContours.entries) {
      final points = entry.value;
      if (points.isEmpty) continue;

      final path = Path();
      for (var i = 0; i < points.length; i++) {
        final canvasPoint = displayPointToCanvasPoint(points[i], canvasSize);
        if (i == 0) {
          path.moveTo(canvasPoint.dx, canvasPoint.dy);
        } else {
          path.lineTo(canvasPoint.dx, canvasPoint.dy);
        }
        // Draw small dot on contour vertices
        canvas.drawCircle(canvasPoint, 1.2, contourDotPaint);
      }

      // Close loop for eyes, lips, and face oval
      if (entry.key == FaceContourType.face ||
          entry.key == FaceContourType.leftEye ||
          entry.key == FaceContourType.rightEye ||
          entry.key == FaceContourType.upperLipTop ||
          entry.key == FaceContourType.lowerLipBottom) {
        path.close();
      }

      canvas.drawPath(path, contourLinePaint);
    }
  }

  void _drawLandmarks(Canvas canvas, DetectedFace face, Size canvasSize) {
    final landmarkGlowPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFFFF5252).withValues(alpha: 0.4);

    final landmarkDotPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFFFFD700); // Gold

    for (final point in face.displayLandmarks.values) {
      final canvasPoint = displayPointToCanvasPoint(point, canvasSize);
      canvas.drawCircle(canvasPoint, 5.0, landmarkGlowPaint);
      canvas.drawCircle(canvasPoint, 2.5, landmarkDotPaint);
    }
  }

  void _drawMetrics(Canvas canvas, Rect rect, DetectedFace face) {
    final items = <String>[];

    if (face.detectionConfidence != null) {
      items.add(
        'Confidence: ${(face.detectionConfidence! * 100).toStringAsFixed(1)}%',
      );
    }

    if (face.smilingPercentageText != null) {
      items.add(
        face.isSmiling
            ? '😊 ${face.smilingPercentageText}'
            : '😐 ${face.smilingPercentageText}',
      );
    }

    if (face.leftEyeOpenPercentageText != null ||
        face.rightEyeOpenPercentageText != null) {
      final left = face.leftEyeOpenPercentageText ?? '-';
      final right = face.rightEyeOpenPercentageText ?? '-';
      items.add('👁️ L:$left R:$right');
    }

    if (face.headEulerAngleZ != null) {
      final roll = face.headEulerAngleZ!.toStringAsFixed(1);
      items.add('Tilt: $roll°');
    }

    if (items.isEmpty) return;

    final metricsText = items.join('  •  ');
    _drawBadge(
      canvas: canvas,
      position: Offset(rect.left, rect.bottom + 6),
      text: metricsText,
      backgroundColor: const Color(0xD9000000),
      textColor: Colors.white,
      fontSize: 11,
    );
  }

  void _drawBadge({
    required Canvas canvas,
    required Offset position,
    required String text,
    required Color backgroundColor,
    required Color textColor,
    double fontSize = 12,
  }) {
    final textSpan = TextSpan(
      text: text,
      style: TextStyle(
        color: textColor,
        fontSize: fontSize,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.3,
      ),
    );

    final textPainter = TextPainter(
      text: textSpan,
      textDirection: ui.TextDirection.ltr,
    )..layout();

    final paddingH = 7.0;
    final paddingV = 3.5;
    final badgeRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(
        position.dx,
        position.dy,
        textPainter.width + (paddingH * 2),
        textPainter.height + (paddingV * 2),
      ),
      const Radius.circular(6),
    );

    canvas.drawRRect(badgeRect, Paint()..color = backgroundColor);
    textPainter.paint(
      canvas,
      Offset(position.dx + paddingH, position.dy + paddingV),
    );
  }

  @override
  bool shouldRepaint(covariant FaceDetectionPainter oldDelegate) {
    return oldDelegate.faces != faces || oldDelegate.config != config;
  }
}
