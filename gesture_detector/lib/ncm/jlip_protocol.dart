import 'dart:math' as math;
import 'dart:typed_data';

const jlipHeaderSize = 24;
const jlipHelloAck = 0x02;
const jlipHeartbeat = 0x03;
const jlipJpeg = 0x10;
const jlipMaxPayload = 200 * 1024;
const _magic = [0x4a, 0x4c, 0x49, 0x50];

final _crcTable = Uint32List.fromList(
  List.generate(256, (value) {
    var crc = value;
    for (var bit = 0; bit < 8; bit++) {
      crc = (crc & 1) != 0 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
    }
    return crc;
  }),
);

int jlipCrc32(Uint8List bytes) {
  var crc = 0xffffffff;
  for (final byte in bytes) {
    crc = _crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) & 0xffffffff;
}

class JlipPacket {
  const JlipPacket(this.type, this.sequence, this.timestampMs, this.payload);
  final int type;
  final int sequence;
  final int timestampMs;
  final Uint8List payload;
}

Uint8List buildJlipPacket({
  required int type,
  required int sequence,
  Uint8List? payload,
  int timestampMs = 0,
}) {
  final data = payload ?? Uint8List(0);
  if (data.length > jlipMaxPayload) {
    throw ArgumentError('JLIP payload too large');
  }
  final result = Uint8List(jlipHeaderSize + data.length);
  result.setRange(0, 4, _magic);
  final header = ByteData.sublistView(result);
  header.setUint8(4, 1);
  header.setUint8(5, type);
  header.setUint16(6, jlipHeaderSize);
  header.setUint32(8, sequence & 0xffffffff);
  header.setUint32(12, data.length);
  header.setUint32(16, jlipCrc32(data));
  header.setUint32(20, timestampMs & 0xffffffff);
  result.setRange(jlipHeaderSize, result.length, data);
  return result;
}

/// Fixed-capacity, incremental TCP parser. TCP reads are not packet boundaries.
class JlipStreamParser {
  JlipStreamParser({this.maxPayloadBytes = jlipMaxPayload})
    : _buffer = Uint8List(maxPayloadBytes + jlipHeaderSize);

  final int maxPayloadBytes;
  final Uint8List _buffer;
  int _start = 0;
  int _end = 0;
  int crcErrors = 0;
  int headerErrors = 0;
  int discardedBytes = 0;
  int get bufferedBytes => _end - _start;

  List<JlipPacket> feed(Uint8List bytes) {
    final packets = <JlipPacket>[];
    var offset = 0;
    while (offset < bytes.length) {
      if (_start > 0) {
        _buffer.setRange(0, bufferedBytes, _buffer, _start);
        _end -= _start;
        _start = 0;
      }
      final count = math.min(bytes.length - offset, _buffer.length - _end);
      _buffer.setRange(_end, _end + count, bytes, offset);
      _end += count;
      offset += count;
      _parse(packets);
    }
    return packets;
  }

  void _parse(List<JlipPacket> packets) {
    while (bufferedBytes >= 4) {
      if (_buffer[_start] != _magic[0] ||
          _buffer[_start + 1] != _magic[1] ||
          _buffer[_start + 2] != _magic[2] ||
          _buffer[_start + 3] != _magic[3]) {
        _start++;
        discardedBytes++;
        continue;
      }
      if (bufferedBytes < jlipHeaderSize) return;
      final header = ByteData.sublistView(
        _buffer,
        _start,
        _start + jlipHeaderSize,
      );
      final length = header.getUint32(12);
      if (header.getUint8(4) != 1 ||
          header.getUint16(6) != jlipHeaderSize ||
          length > maxPayloadBytes) {
        headerErrors++;
        discardedBytes++;
        _start++;
        continue;
      }
      if (bufferedBytes < jlipHeaderSize + length) return;
      final payload = Uint8List.fromList(
        Uint8List.sublistView(
          _buffer,
          _start + jlipHeaderSize,
          _start + jlipHeaderSize + length,
        ),
      );
      _start += jlipHeaderSize + length;
      if (jlipCrc32(payload) != header.getUint32(16)) {
        crcErrors++;
        continue;
      }
      packets.add(
        JlipPacket(
          header.getUint8(5),
          header.getUint32(8),
          header.getUint32(20),
          payload,
        ),
      );
    }
  }
}

/// Trims firmware padding and checks JPEG dimensions before native decoding.
/// The cap prevents a tiny compressed payload allocating an unbounded image.
Uint8List? validatedJpeg(Uint8List payload) {
  if (payload.length < 4 || payload[0] != 0xff || payload[1] != 0xd8) {
    return null;
  }
  var end = payload.length - 2;
  while (end > 1 && !(payload[end] == 0xff && payload[end + 1] == 0xd9)) {
    end--;
  }
  if (end <= 1) return null;
  var offset = 2;
  var hasDimensions = false;
  while (offset < end) {
    if (payload[offset++] != 0xff) return null;
    while (offset < end && payload[offset] == 0xff) {
      offset++;
    }
    if (offset >= end) return null;
    final marker = payload[offset++];
    if (marker == 0xda) break; // Start of compressed scan.
    if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    if (offset + 2 > end) return null;
    final length = (payload[offset] << 8) | payload[offset + 1];
    if (length < 2 || offset + length > end) return null;
    if (const [0xc0, 0xc1, 0xc2].contains(marker)) {
      if (length < 8) return null;
      final height = (payload[offset + 3] << 8) | payload[offset + 4];
      final width = (payload[offset + 5] << 8) | payload[offset + 6];
      if (width < 32 ||
          height < 32 ||
          width > 8192 ||
          height > 8192 ||
          width * height > 12000000) {
        return null;
      }
      hasDimensions = true;
    }
    offset += length;
  }
  return hasDimensions ? Uint8List.sublistView(payload, 0, end + 2) : null;
}
