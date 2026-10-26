import 'package:face_detector/detection/pose_affine.dart';
import 'package:face_detector/detection/yunet_candidate.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
    'rotated/stretch recovery maps landmarks back to their original pixels',
    () {
      for (final angle in [-90.0, -50.0, -25.0, 25.0, 50.0, 90.0, 180.0]) {
        final matrix = PoseAffine.rotated(640, 360, (angle, 1.0, 1.35));
        final inverse = matrix.inverse();
        for (final point in <FacePoint>[(0, 0), (640, 360), (250, 130)]) {
          final restored = inverse.map(matrix.map(point));
          expect(restored.$1, closeTo(point.$1, 1e-8));
          expect(restored.$2, closeTo(point.$2, 1e-8));
        }
      }
    },
  );

  test('eye/mouth alignment maps to the same canonical template as Python', () {
    final candidate = YuNetCandidate(
      const FaceBox(100, 100, 100, 120),
      .96,
      const [(145, 135), (155, 135), (170, 150), (145, 180), (163, 180)],
    );
    final normalized = PoseAffine.landmarkAnchors(candidate);
    final matrix = PoseAffine.fromLandmarks(candidate.box, normalized)!;
    final expected = <FacePoint>[(64, 72), (128, 72), (96, 136)];
    final source = <FacePoint>[(145, 135), (155, 135), (154, 180)];
    for (var i = 0; i < source.length; i++) {
      final mapped = matrix.map(source[i]);
      expect(mapped.$1, closeTo(expected[i].$1, 1e-8));
      expect(mapped.$2, closeTo(expected[i].$2, 1e-8));
      final restored = matrix.inverse().map(mapped);
      expect(restored.$1, closeTo(source[i].$1, 1e-8));
      expect(restored.$2, closeTo(source[i].$2, 1e-8));
    }
  });

  test('collinear landmarks reject affine rectification', () {
    expect(
      PoseAffine.fromLandmarks(const FaceBox(0, 0, 100, 100), const [
        (.2, .3),
        (.8, .3),
        (.5, .3),
      ]),
      isNull,
    );
    expect(
      PoseAffine.fromLandmarks(const FaceBox(0, 0, 100, 100), const [
        (.2, .7),
        (.8, .7),
        (.5, .3),
      ]),
      isNull,
    );
  });
}
