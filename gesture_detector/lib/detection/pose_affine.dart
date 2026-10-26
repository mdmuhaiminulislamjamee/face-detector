import 'dart:math' as math;

import 'yunet_candidate.dart';

/// Original-to-upright pixel transform, with its exact inverse for overlays.
class PoseAffine {
  const PoseAffine(
    this.a,
    this.b,
    this.tx,
    this.c,
    this.d,
    this.ty,
    this.width,
    this.height,
  );

  final double a, b, tx, c, d, ty;
  final int width, height;
  List<double> get values => [a, b, tx, c, d, ty];

  FacePoint map(FacePoint p) =>
      (a * p.$1 + b * p.$2 + tx, c * p.$1 + d * p.$2 + ty);

  factory PoseAffine.rotated(
    int width,
    int height,
    (double, double, double) pose,
  ) {
    final radians = pose.$1 * math.pi / 180;
    final a = math.cos(radians) * pose.$2;
    final b = math.sin(radians) * pose.$2;
    final c = -math.sin(radians) * pose.$3;
    final d = math.cos(radians) * pose.$3;
    final points = [
      for (final p in <FacePoint>[
        (0, 0),
        (width.toDouble(), 0),
        (width.toDouble(), height.toDouble()),
        (0, height.toDouble()),
      ])
        (a * p.$1 + b * p.$2, c * p.$1 + d * p.$2),
    ];
    final left = points.map((p) => p.$1).reduce(math.min);
    final top = points.map((p) => p.$2).reduce(math.min);
    final right = points.map((p) => p.$1).reduce(math.max);
    final bottom = points.map((p) => p.$2).reduce(math.max);
    return PoseAffine(
      a,
      b,
      -left,
      c,
      d,
      -top,
      math.max(32, (right - left - 1e-9).ceil()),
      math.max(32, (bottom - top - 1e-9).ceil()),
    );
  }

  static List<FacePoint> landmarkAnchors(YuNetCandidate candidate) {
    final box = candidate.box, points = candidate.landmarks;
    final mouth = (
      (points[3].$1 + points[4].$1) / 2,
      (points[3].$2 + points[4].$2) / 2,
    );
    return [
      for (final p in [points[0], points[1], mouth])
        ((p.$1 - box.x) / box.width, (p.$2 - box.y) / box.height),
    ];
  }

  static PoseAffine? fromLandmarks(FaceBox box, List<FacePoint> normalized) {
    final points = [
      for (final p in normalized)
        (box.x + p.$1 * box.width, box.y + p.$2 * box.height),
    ];
    final dx1 = points[1].$1 - points[0].$1;
    final dy1 = points[1].$2 - points[0].$2;
    final dx2 = points[2].$1 - points[0].$1;
    final dy2 = points[2].$2 - points[0].$2;
    final determinant = dx1 * dy2 - dy1 * dx2;
    if (determinant < math.max(4, box.width * box.height * .008)) {
      return null;
    }
    final a = (64 * dy2 - 32 * dy1) / determinant;
    final b = (-64 * dx2 + 32 * dx1) / determinant;
    final c = -64 * dy1 / determinant;
    final d = 64 * dx1 / determinant;
    return PoseAffine(
      a,
      b,
      64 - a * points[0].$1 - b * points[0].$2,
      c,
      d,
      72 - c * points[0].$1 - d * points[0].$2,
      192,
      224,
    );
  }

  PoseAffine inverse({double scaleX = 1, double scaleY = 1}) {
    final determinant = a * d - b * c;
    return PoseAffine(
      d / determinant * scaleX,
      -b / determinant * scaleX,
      (b * ty - d * tx) / determinant * scaleX,
      -c / determinant * scaleY,
      a / determinant * scaleY,
      (c * tx - a * ty) / determinant * scaleY,
      width,
      height,
    );
  }
}
