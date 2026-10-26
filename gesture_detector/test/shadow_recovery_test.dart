import 'dart:typed_data';

import 'package:face_detector/utils/shadow_recovery.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('small shadow region against a bright background triggers recovery', () {
    final lut = shadowRecoveryLut([40, ...List.filled(99, 230)]);
    expect(lut, isNotNull);
    expect(lut![40], greaterThan(40));
    expect(lut[230], 230);
    expect(lut[0], 0);
    for (var i = 1; i < 256; i++) {
      expect(lut[i], greaterThanOrEqualTo(lut[i - 1]));
    }
  });

  test('bright frames and entirely black frames are preserved', () {
    expect(shadowRecoveryLut(List.filled(100, 200)), isNull);
    expect(shadowRecoveryLut(List.filled(100, 0)), isNull);
  });

  test('NV21 shadow lift preserves chroma', () {
    final bytes = Uint8List.fromList([40, 40, 40, 40, 110, 140]);
    recoverNv21Shadows(bytes, 2, 2);
    expect(bytes[0], greaterThan(40));
    expect(bytes.sublist(4), [110, 140]);
  });

  test('BGRA recovery preserves alpha, row padding, and camera buffer', () {
    final source = Uint8List.fromList([
      40,
      40,
      40,
      255,
      231,
      232,
      233,
      234,
      40,
      40,
      40,
      250,
      241,
      242,
      243,
      244,
    ]);
    final result = recoverBgraShadows(source, 1, 2, 8);
    expect(result[0], greaterThan(40));
    expect(result[8], greaterThan(40));
    expect(result[3], 255);
    expect(result[11], 250);
    expect(result.sublist(4, 8), [231, 232, 233, 234]);
    expect(result.sublist(12), [241, 242, 243, 244]);
    expect(source[0], 40);
  });
}
