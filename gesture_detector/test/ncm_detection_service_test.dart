import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:face_detector/detection/yunet_candidate.dart';
import 'package:face_detector/ncm/jlip_protocol.dart';
import 'package:face_detector/ncm/ncm_camera_client.dart';
import 'package:face_detector/services/ncm_face_detection_service.dart';
import 'package:face_detector/services/yunet_worker.dart';
import 'package:flutter_test/flutter_test.dart';

class _Processor implements JpegFaceProcessor {
  final resets = <bool>[];
  bool disposed = false;
  @override
  Future<void> initialize() async {}
  @override
  Future<YuNetFrameResult> process(
    Uint8List jpeg, {
    bool rotate180 = false,
    bool reset = false,
  }) async {
    resets.add(reset);
    return YuNetFrameResult(640, 480, [
      YuNetCandidate(const FaceBox(100, 100, 100, 120), .95, const [
        (125, 130),
        (175, 130),
        (150, 155),
        (130, 180),
        (170, 180),
      ]),
    ], const Duration(milliseconds: 10));
  }

  @override
  Future<void> dispose() async {
    disposed = true;
  }
}

void main() {
  test(
    'NCM pipeline confirms fresh faces and clears results on disconnect',
    () async {
      final server = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
      final jpeg = File(
        'python_face_detector/tests/fixtures/face.jpg',
      ).readAsBytesSync();
      Socket? peer;
      Timer? producer;
      final subscription = server.listen((socket) {
        peer = socket;
        socket.listen((_) {}, onError: (Object _) {});
        producer = Timer.periodic(const Duration(milliseconds: 30), (timer) {
          socket.add(
            buildJlipPacket(
              type: jlipJpeg,
              sequence: timer.tick,
              payload: jpeg,
            ),
          );
        });
      });
      final processor = _Processor();
      final service = NcmFaceDetectionService(processor: processor);
      final confirmed = Completer<void>();
      service.addListener(() {
        if (service.frame?.faces.any((face) => !face.isTemporarilyLost) ==
                true &&
            !confirmed.isCompleted) {
          confirmed.complete();
        }
      });
      try {
        await service.start(
          NcmCameraConfig(
            hostAddress: '127.0.0.1',
            deviceAddress: '127.0.0.1',
            tcpPort: server.port,
          ),
        );
        await confirmed.future.timeout(const Duration(seconds: 3));
        final face = service.frame!.faces.single;
        expect(face.detectionConfidence, .95);
        expect(face.displayLandmarks, hasLength(5));
        expect(face.displayBox.left, closeTo(100 / 640, .001));
        expect(processor.resets.first, isTrue);
        expect(processor.resets.skip(1), everyElement(isFalse));
        expect(service.detectionFps, lessThanOrEqualTo(10.1));
        service.stop();
        expect(service.frame, isNull);
        expect(service.status.state, 'stopped');
      } finally {
        producer?.cancel();
        service.dispose();
        peer?.destroy();
        await subscription.cancel();
        await server.close();
      }
      expect(processor.disposed, isTrue);
    },
  );
}
