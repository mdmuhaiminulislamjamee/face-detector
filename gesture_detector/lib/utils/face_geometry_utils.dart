import 'dart:ui';

/// Returns camera preview size aligned to the current screen orientation.
Size orientedPreviewSize({
  required Size? rawPreviewSize,
  required bool isLandscape,
}) {
  final validSize =
      rawPreviewSize != null &&
      rawPreviewSize.width.isFinite &&
      rawPreviewSize.height.isFinite &&
      rawPreviewSize.width > 0 &&
      rawPreviewSize.height > 0;
  final source = validSize ? rawPreviewSize : const Size(16, 9);
  final shortSide = source.shortestSide;
  final longSide = source.longestSide;
  return isLandscape ? Size(longSide, shortSide) : Size(shortSide, longSide);
}

/// Converts a normalized [0..1] preview rect into exact canvas pixel coordinates.
Rect displayRectToCanvasRect(Rect displayRect, Size canvasSize) {
  if (displayRect.isEmpty || canvasSize.width <= 0 || canvasSize.height <= 0) {
    return Rect.zero;
  }
  return Rect.fromLTRB(
    (displayRect.left * canvasSize.width).clamp(0.0, canvasSize.width),
    (displayRect.top * canvasSize.height).clamp(0.0, canvasSize.height),
    (displayRect.right * canvasSize.width).clamp(0.0, canvasSize.width),
    (displayRect.bottom * canvasSize.height).clamp(0.0, canvasSize.height),
  );
}

/// Converts a normalized [0..1] preview offset into exact canvas pixel coordinates.
Offset displayPointToCanvasPoint(Offset displayPoint, Size canvasSize) {
  return Offset(
    (displayPoint.dx * canvasSize.width).clamp(0.0, canvasSize.width),
    (displayPoint.dy * canvasSize.height).clamp(0.0, canvasSize.height),
  );
}

