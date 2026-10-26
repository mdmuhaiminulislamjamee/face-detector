import 'dart:math' as math;
import 'dart:typed_data';

final _shadowLiftLut = Uint8List.fromList(
  List.generate(256, (value) {
    final weight = (1 - value / 180).clamp(0.0, 1.0);
    final lifted = 255 * math.pow(value / 255, 0.72);
    return (value + (lifted - value) * weight).round().clamp(0, 255);
  }),
);

/// Bounded shadow lift. Highlights stay unchanged and black stays black.
/// Returns null for well-lit frames to avoid unnecessary processing.
Uint8List? shadowRecoveryLut(List<int> samples) {
  if (samples.isEmpty) return null;
  final dark = samples.where((value) => value >= 12 && value < 90).length;
  if (dark / samples.length < 0.01) return null;
  return _shadowLiftLut;
}

/// Edits only the packed Y plane; the NV21 chroma planes are preserved.
void recoverNv21Shadows(Uint8List bytes, int width, int height) {
  final size = width * height;
  final lut = shadowRecoveryLut([
    for (var y = 0; y < height; y += 8)
      for (var x = 0; x < width; x += 8) bytes[y * width + x],
  ]);
  if (lut == null) return;
  for (var i = 0; i < size; i++) {
    bytes[i] = lut[bytes[i]];
  }
}

/// Handles padded BGRA rows without modifying alpha or the source camera buffer.
Uint8List recoverBgraShadows(
  Uint8List source,
  int width,
  int height,
  int bytesPerRow,
) {
  final samples = <int>[];
  for (var y = 0; y < height; y += 8) {
    for (var x = 0; x < width; x += 8) {
      final i = y * bytesPerRow + x * 4;
      samples.add(
        (29 * source[i] + 150 * source[i + 1] + 77 * source[i + 2]) >> 8,
      );
    }
  }
  final lut = shadowRecoveryLut(samples);
  if (lut == null) return source;
  final output = Uint8List.fromList(source);
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final i = y * bytesPerRow + x * 4;
      final luma =
          (29 * source[i] + 150 * source[i + 1] + 77 * source[i + 2]) >> 8;
      final gain = luma == 0 ? 1.0 : lut[luma] / luma;
      for (var channel = 0; channel < 3; channel++) {
        output[i + channel] = (source[i + channel] * gain).round().clamp(
          0,
          255,
        );
      }
    }
  }
  return output;
}
