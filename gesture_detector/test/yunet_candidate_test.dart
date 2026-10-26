import 'dart:math' as math;

import 'package:face_detector/detection/yunet_candidate.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
    'foreshortened pose uses a stronger score and coherent mouth landmarks',
    () {
      final profile = YuNetCandidate(
        const FaceBox(100, 100, 100, 120),
        .96,
        const [(145, 135), (155, 135), (170, 150), (145, 180), (163, 180)],
      );
      expect(validYuNetCandidate(profile), isTrue);
      expect(
        validYuNetCandidate(
          YuNetCandidate(profile.box, .90, profile.landmarks),
        ),
        isFalse,
      );
      expect(
        validYuNetCandidate(
          YuNetCandidate(profile.box, .90, profile.landmarks),
          proposal: true,
        ),
        isTrue,
      );
      expect(
        validYuNetCandidate(
          YuNetCandidate(profile.box, .99, [
            ...profile.landmarks.take(4),
            profile.landmarks[3],
          ]),
        ),
        isFalse,
      );
      expect(
        validYuNetCandidate(
          YuNetCandidate(profile.box, .99, [
            ...profile.landmarks.take(3),
            profile.landmarks[4],
            profile.landmarks[3],
          ]),
        ),
        isFalse,
      );
    },
  );
  final valid = YuNetCandidate(const FaceBox(100, 100, 100, 120), .95, const [
    (125, 130),
    (175, 130),
    (150, 155),
    (130, 180),
    (170, 180),
  ]);

  test('strict geometry and confidence match the PC acceptance gates', () {
    expect(validYuNetCandidate(valid), isTrue);
    expect(
      validYuNetCandidate(YuNetCandidate(valid.box, .7, valid.landmarks)),
      isFalse,
    );
    expect(
      validYuNetCandidate(
        YuNetCandidate(valid.box, double.nan, valid.landmarks),
      ),
      isFalse,
    );
    expect(
      validYuNetCandidate(
        YuNetCandidate(const FaceBox(0, 0, 10, 10), .99, valid.landmarks),
      ),
      isFalse,
    );
    expect(
      validYuNetCandidate(
        YuNetCandidate(valid.box, .99, [
          valid.landmarks[0],
          valid.landmarks[0],
          ...valid.landmarks.skip(2),
        ]),
      ),
      isFalse,
    );
  });

  test('head roll passes without relying on skin colour or open eyes', () {
    final rolled = [
      for (final point in valid.landmarks)
        (
          (point.$1 - 150) * math.cos(.4) -
              (point.$2 - 160) * math.sin(.4) +
              150,
          (point.$1 - 150) * math.sin(.4) +
              (point.$2 - 160) * math.cos(.4) +
              160,
        ),
    ];
    expect(validYuNetCandidate(YuNetCandidate(valid.box, .95, rolled)), isTrue);
  });

  test('weak proposals require fresh high-confidence refinement', () {
    final weak = YuNetCandidate(valid.box, .65, valid.landmarks);
    expect(validYuNetCandidate(weak, proposal: true), isTrue);
    expect(validYuNetCandidate(weak), isFalse);
  });

  test(
    'duplicate suppression preserves the strongest face and separate targets',
    () {
      final other = valid.shifted(300, 0);
      final weakDuplicate = YuNetCandidate(valid.box, .89, valid.landmarks);
      expect(deduplicateYuNet([weakDuplicate, valid, other]), [valid, other]);
      expect(YuNetCandidate.fromRow(valid.toRow()).toRow(), valid.toRow());
    },
  );
}
