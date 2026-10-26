import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:face_detector/ncm/jlip_protocol.dart';
import 'package:face_detector/ncm/ncm_camera_client.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
    'TCP client reassembles JPEG, responds to heartbeat, and stops cleanly',
    () async {
      final server = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
      final client = NcmCameraClient();
      Socket? peer;
      final heartbeat = Completer<JlipPacket>();
      final serverSubscription = server.listen((socket) {
        peer = socket;
        final parser = JlipStreamParser();
        socket.listen((bytes) {
          for (final packet in parser.feed(bytes)) {
            if (!heartbeat.isCompleted) heartbeat.complete(packet);
          }
        });
        final jpeg = File(
          'python_face_detector/tests/fixtures/face.jpg',
        ).readAsBytesSync();
        final bytes = buildJlipPacket(
          type: jlipJpeg,
          sequence: 0xffffffff,
          payload: jpeg,
        );
        socket.add(bytes.sublist(0, 13));
        socket.add(bytes.sublist(13));
        socket.add(buildJlipPacket(type: jlipHeartbeat, sequence: 0));
      });
      final frame = client.frames.first.timeout(const Duration(seconds: 3));
      try {
        client.start(
          NcmCameraConfig(
            hostAddress: '127.0.0.1',
            deviceAddress: '127.0.0.1',
            tcpPort: server.port,
          ),
        );
        expect((await frame).sequence, 0xffffffff);
        final reply = await heartbeat.future.timeout(
          const Duration(seconds: 3),
        );
        expect(reply.type, jlipHeartbeat);
        expect(reply.payload, isEmpty);
        expect(client.status.sequenceGaps, 0);
        expect(client.status.frames, 1);
        client.stop();
        expect(client.status.state, 'stopped');
      } finally {
        await client.dispose();
        peer?.destroy();
        await serverSubscription.cancel();
        await server.close();
      }
    },
  );

  test(
    'discovery sends the exact firmware message to UDP 5001 equivalent',
    () async {
      final server = await RawDatagramSocket.bind(
        InternetAddress.loopbackIPv4,
        0,
      );
      final subscription = server.listen((event) {
        if (event == RawSocketEvent.read) {
          final packet = server.receive()!;
          expect(ascii.decode(packet.data), 'JL-CAMERA-DISCOVER');
          server.send(
            Uint8List.fromList(utf8.encode('board camera')),
            packet.address,
            packet.port,
          );
        }
      });
      try {
        expect(
          await NcmCameraClient.discover(
            NcmCameraConfig(
              hostAddress: '127.0.0.1',
              deviceAddress: '127.0.0.1',
              discoveryPort: server.port,
            ),
          ),
          'board camera',
        );
      } finally {
        await subscription.cancel();
        server.close();
      }
    },
  );
}
