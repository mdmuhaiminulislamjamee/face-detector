import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:face_detector/ncm/jlip_protocol.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final jpeg = File(
    'python_face_detector/tests/fixtures/face.jpg',
  ).readAsBytesSync();

  test('matches the reference big-endian header and IEEE CRC32', () {
    expect(
      jlipCrc32(Uint8List.fromList(ascii.encode('123456789'))),
      0xcbf43926,
    );
    final bytes = buildJlipPacket(
      type: jlipJpeg,
      sequence: 0x01020304,
      timestampMs: 0x11223344,
      payload: Uint8List.fromList([1, 2, 3]),
    );
    expect(bytes.take(16), [
      74,
      76,
      73,
      80,
      1,
      16,
      0,
      24,
      1,
      2,
      3,
      4,
      0,
      0,
      0,
      3,
    ]);
    expect(bytes.sublist(20, 24), [0x11, 0x22, 0x33, 0x44]);
  });

  test('reassembles a JPEG fragmented at every byte boundary', () {
    final parser = JlipStreamParser();
    final bytes = buildJlipPacket(type: jlipJpeg, sequence: 7, payload: jpeg);
    final packets = <JlipPacket>[];
    for (final byte in bytes) {
      packets.addAll(parser.feed(Uint8List.fromList([byte])));
    }
    expect(packets, hasLength(1));
    expect(packets.single.sequence, 7);
    expect(packets.single.payload, jpeg);
    expect(parser.bufferedBytes, 0);
  });

  test(
    'handles multiple packets in one TCP read and resynchronizes after junk',
    () {
      final parser = JlipStreamParser();
      final first = buildJlipPacket(type: jlipHeartbeat, sequence: 0xffffffff);
      final second = buildJlipPacket(
        type: jlipJpeg,
        sequence: 0,
        payload: jpeg,
      );
      final packets = parser.feed(
        Uint8List.fromList([5, 74, 76, ...first, ...second]),
      );
      expect(packets.map((p) => p.type), [jlipHeartbeat, jlipJpeg]);
      expect(parser.discardedBytes, 3);
    },
  );

  test('rejects corrupt CRC and recovers the following good packet', () {
    final bad = buildJlipPacket(type: jlipJpeg, sequence: 1, payload: jpeg);
    bad[30] ^= 1;
    final parser = JlipStreamParser();
    final packets = parser.feed(
      Uint8List.fromList([
        ...bad,
        ...buildJlipPacket(type: jlipHeartbeat, sequence: 2),
      ]),
    );
    expect(parser.crcErrors, 1);
    expect(packets.single.sequence, 2);
  });

  test('rejects wrong version, header length and oversized declarations', () {
    for (final change in [4, 6, 12]) {
      final bad = buildJlipPacket(type: jlipJpeg, sequence: 1);
      bad[change] = 255;
      final parser = JlipStreamParser();
      expect(
        parser
            .feed(
              Uint8List.fromList([
                ...bad,
                ...buildJlipPacket(type: jlipHeartbeat, sequence: 2),
              ]),
            )
            .single
            .sequence,
        2,
      );
      expect(parser.headerErrors, 1);
    }
  });

  test('buffer stays bounded with huge malformed and coalesced reads', () {
    final parser = JlipStreamParser();
    parser.feed(Uint8List(jlipMaxPayload * 8));
    expect(parser.bufferedBytes, lessThan(4));
    final bytes = buildJlipPacket(
      type: jlipJpeg,
      sequence: 9,
      payload: Uint8List(jlipMaxPayload),
    );
    expect(parser.feed(Uint8List.fromList([...bytes, ...bytes])), hasLength(2));
    expect(parser.bufferedBytes, 0);
  });

  test('validates dimensions and trims JPEG padding', () {
    expect(validatedJpeg(jpeg), jpeg);
    expect(validatedJpeg(Uint8List.fromList([...jpeg, 0, 0, 0])), jpeg);
    expect(validatedJpeg(Uint8List.fromList([255, 216, 255, 217])), isNull);
    expect(validatedJpeg(Uint8List.fromList([1, 2, 3, 4])), isNull);
    expect(validatedJpeg(jpeg.sublist(0, jpeg.length - 2)), isNull);
    // 8192 x 8192 decompression bomb, encoded in an otherwise plausible header.
    expect(
      validatedJpeg(
        Uint8List.fromList([
          255,
          216,
          255,
          192,
          0,
          8,
          8,
          32,
          0,
          32,
          0,
          1,
          255,
          217,
        ]),
      ),
      isNull,
    );
  });
}
