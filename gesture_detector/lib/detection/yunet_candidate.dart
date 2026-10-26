import 'dart:math' as math;

typedef FacePoint = (double, double);

class FaceBox {
  const FaceBox(this.x, this.y, this.width, this.height);
  final double x, y, width, height;

  double overlap(FaceBox other) {
    final area =
        math.max(
          0,
          math.min(x + width, other.x + other.width) - math.max(x, other.x),
        ) *
        math.max(
          0,
          math.min(y + height, other.y + other.height) - math.max(y, other.y),
        );
    return area /
        math.max(1, width * height + other.width * other.height - area);
  }
}

class YuNetCandidate {
  const YuNetCandidate(this.box, this.score, this.landmarks);
  final FaceBox box;
  final double score;

  /// YuNet order: right eye, left eye, nose, right mouth, left mouth.
  final List<FacePoint> landmarks;

  List<double> toRow() => [
    box.x,
    box.y,
    box.width,
    box.height,
    for (final point in landmarks) ...[point.$1, point.$2],
    score,
  ];

  factory YuNetCandidate.fromRow(List<double> row) => YuNetCandidate(
    FaceBox(row[0], row[1], row[2], row[3]),
    row[14],
    [for (var i = 4; i < 14; i += 2) (row[i], row[i + 1])],
  );

  YuNetCandidate shifted(double dx, double dy) => YuNetCandidate(
    FaceBox(box.x + dx, box.y + dy, box.width, box.height),
    score,
    [for (final point in landmarks) (point.$1 + dx, point.$2 + dy)],
  );
}

/// The same score and roll-tolerant landmark geometry checks as the PC detector.
bool validYuNetCandidate(YuNetCandidate face, {bool proposal = false}) {
  const threshold = 0.88;
  final box = face.box;
  if (face.landmarks.length != 5 ||
      !face.score.isFinite ||
      (!proposal && face.score < threshold) ||
      ![box.x, box.y, box.width, box.height].every((v) => v.isFinite) ||
      math.min(box.width, box.height) < 18 ||
      box.height / box.width < 0.65 ||
      box.height / box.width > 1.9) {
    return false;
  }
  for (final (x, y) in face.landmarks) {
    if (!x.isFinite ||
        !y.isFinite ||
        x < box.x - box.width * .2 ||
        y < box.y - box.height * .2 ||
        x > box.x + box.width * 1.2 ||
        y > box.y + box.height * 1.2) {
      return false;
    }
  }
  final [eyeA, eyeB, nose, mouthA, mouthB] = face.landmarks;
  final ex = eyeB.$1 - eyeA.$1, ey = eyeB.$2 - eyeA.$2;
  final eyeDistance = math.sqrt(ex * ex + ey * ey);
  if (eyeDistance < .15 * box.width || eyeDistance > .85 * box.width) {
    return false;
  }
  final mx = (eyeA.$1 + eyeB.$1) / 2, my = (eyeA.$2 + eyeB.$2) / 2;
  final dx = (mouthA.$1 + mouthB.$1) / 2 - mx;
  final dy = (mouthA.$2 + mouthB.$2) / 2 - my;
  final faceHeight = math.sqrt(dx * dx + dy * dy);
  if (faceHeight < .12 * box.height ||
      faceHeight > .85 * box.height ||
      ((ex * dx + ey * dy) / (eyeDistance * faceHeight)).abs() > .65) {
    return false;
  }
  final noseDepth =
      ((nose.$1 - mx) * dx + (nose.$2 - my) * dy) / (faceHeight * faceHeight);
  return noseDepth >= -.15 && noseDepth <= 1.25;
}

List<YuNetCandidate> deduplicateYuNet(List<YuNetCandidate> candidates) {
  final sorted = [...candidates]..sort((a, b) => b.score.compareTo(a.score));
  final kept = <YuNetCandidate>[];
  for (final candidate in sorted) {
    if (!kept.any((other) => candidate.box.overlap(other.box) > .3)) {
      kept.add(candidate);
    }
  }
  return kept;
}
