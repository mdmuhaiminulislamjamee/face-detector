import 'package:face_detector/utils/camera_image_converter.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

CameraPixelFrameData yuvFrame() => CameraPixelFrameData(
  width: 4,
  height: 4,
  planes: [
    CameraPixelPlaneData(
      bytes: Uint8List(16)..fillRange(0, 16, 200),
      bytesPerRow: 4,
      bytesPerPixel: 1,
    ),
    CameraPixelPlaneData(
      bytes: Uint8List(4)..fillRange(0, 4, 100),
      bytesPerRow: 2,
      bytesPerPixel: 1,
    ),
    CameraPixelPlaneData(
      bytes: Uint8List(4)..fillRange(0, 4, 50),
      bytesPerRow: 2,
      bytesPerPixel: 1,
    ),
  ],
);

void main() {
  test('reuses a worker across frames and preserves camera buffers', () async {
    final worker = CameraFrameProcessor();
    addTearDown(worker.dispose);
    final frame = yuvFrame();
    for (var i = 0; i < 3; i++) {
      final result = await worker.process(frame);
      expect(result, encodeNv21(frame, recoverShadows: true));
      expect(frame.planes.first.bytes[0], 200);
    }
  });

  test('worker handles BGRA alpha and padded rows', () async {
    final worker = CameraFrameProcessor();
    addTearDown(worker.dispose);
    final source = Uint8List.fromList([40, 40, 40, 255, 9, 8, 7, 6]);
    final result = await worker.process(
      CameraPixelFrameData(
        width: 1,
        height: 1,
        planes: [
          CameraPixelPlaneData(bytes: source, bytesPerRow: 8, bytesPerPixel: 4),
        ],
      ),
      bgra: true,
    );
    expect(result![0], greaterThan(40));
    expect(result.sublist(3), [255, 9, 8, 7, 6]);
    expect(source[0], 40);
  });

  test(
    'worker reports malformed input and can process the next frame',
    () async {
      final worker = CameraFrameProcessor();
      addTearDown(worker.dispose);
      await expectLater(
        worker.process(
          const CameraPixelFrameData(width: 2, height: 2, planes: []),
          bgra: true,
        ),
        throwsStateError,
      );
      expect(await worker.process(yuvFrame()), isNotNull);
    },
  );

  test('dispose drains an in-flight frame and rejects new work', () async {
    final worker = CameraFrameProcessor();
    final result = worker.process(yuvFrame());
    final disposing = worker.dispose();
    expect(await result, isNotNull);
    await disposing;
    await expectLater(worker.process(yuvFrame()), throwsStateError);
    await worker.dispose();
  });

  test('packing handles padded rows and interleaved chroma', () {
    final frame = CameraPixelFrameData(
      width: 4,
      height: 2,
      planes: [
        CameraPixelPlaneData(
          bytes: Uint8List.fromList([10, 11, 12, 13, 99, 99, 20, 21, 22, 23]),
          bytesPerRow: 6,
          bytesPerPixel: 1,
        ),
        CameraPixelPlaneData(
          bytes: Uint8List.fromList([30, 99, 31]),
          bytesPerRow: 4,
          bytesPerPixel: 2,
        ),
        CameraPixelPlaneData(
          bytes: Uint8List.fromList([40, 99, 41]),
          bytesPerRow: 4,
          bytesPerPixel: 2,
        ),
      ],
    );
    expect(encodeNv21(frame), [10, 11, 12, 13, 20, 21, 22, 23, 40, 30, 41, 31]);
  });

  test('packing handles a non-contiguous luma pixel stride', () {
    final frame = CameraPixelFrameData(
      width: 2,
      height: 2,
      planes: [
        CameraPixelPlaneData(
          bytes: Uint8List.fromList([10, 99, 11, 99, 20, 99, 21]),
          bytesPerRow: 4,
          bytesPerPixel: 2,
        ),
        CameraPixelPlaneData(
          bytes: Uint8List.fromList([30]),
          bytesPerRow: 1,
          bytesPerPixel: 1,
        ),
        CameraPixelPlaneData(
          bytes: Uint8List.fromList([40]),
          bytesPerRow: 1,
          bytesPerPixel: 1,
        ),
      ],
    );
    expect(encodeNv21(frame), [10, 11, 20, 21, 40, 30]);
  });
}
