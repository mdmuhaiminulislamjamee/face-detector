import 'dart:async';
import 'dart:isolate';

import 'package:dartcv4/dartcv.dart' as cv;
import 'package:flutter/services.dart';

import '../detection/robust_yunet.dart';
import '../detection/yunet_candidate.dart';

class YuNetFrameResult {
  const YuNetFrameResult(this.width, this.height, this.faces, this.duration);
  final int width, height;
  final List<YuNetCandidate> faces;
  final Duration duration;
}

abstract interface class JpegFaceProcessor {
  Future<void> initialize();
  Future<YuNetFrameResult> process(
    Uint8List jpeg, {
    bool rotate180 = false,
    bool reset = false,
  });
  Future<void> dispose();
}

/// One model, one isolate, one request at a time. No image work on Flutter's UI isolate.
class YuNetWorker implements JpegFaceProcessor {
  Isolate? _isolate;
  SendPort? _commands;
  ReceivePort? _messages;
  StreamSubscription<dynamic>? _subscription;
  Completer<void>? _ready;
  Completer<YuNetFrameResult>? _pending;
  Future<void>? _initializing;
  bool _disposed = false;

  @override
  Future<void> initialize() async {
    if (_disposed) throw StateError('YuNet worker disposed');
    try {
      await (_initializing ??= _initialize());
    } catch (_) {
      _initializing = null;
      rethrow;
    }
  }

  Future<void> _initialize() async {
    if (_disposed) throw StateError('YuNet worker disposed');
    final model = await rootBundle.load(
      'assets/models/face_detection_yunet_2023mar.onnx',
    );
    if (_disposed) throw StateError('YuNet worker disposed');
    final messages = _messages = ReceivePort();
    final ready = _ready = Completer<void>();
    // A spawn failure or disposal can race the first await of ready.
    unawaited(ready.future.catchError((Object _) {}));
    _subscription = messages.listen((dynamic message) {
      if (message is SendPort) {
        _commands = message;
        if (!ready.isCompleted) ready.complete();
      } else if (message is Map && message['error'] != null) {
        _fail(StateError(message['error'] as String));
      } else if (message is Map) {
        final pending = _pending;
        _pending = null;
        pending?.complete(
          YuNetFrameResult(message['width'] as int, message['height'] as int, [
            for (final row in message['faces'] as List)
              YuNetCandidate.fromRow((row as List).cast<double>()),
          ], Duration(microseconds: message['micros'] as int)),
        );
      } else {
        _fail(StateError('YuNet isolate exited unexpectedly: $message'));
        unawaited(_shutdown());
      }
    });
    try {
      final isolate = await Isolate.spawn(
        _entry,
        (
          messages.sendPort,
          TransferableTypedData.fromList([
            model.buffer.asUint8List(model.offsetInBytes, model.lengthInBytes),
          ]),
        ),
        onError: messages.sendPort,
        onExit: messages.sendPort,
      );
      if (_disposed) {
        isolate.kill(priority: Isolate.immediate);
        throw StateError('YuNet worker disposed');
      }
      _isolate = isolate;
      await ready.future.timeout(const Duration(seconds: 15));
      if (_disposed) throw StateError('YuNet worker disposed');
    } catch (_) {
      await _shutdown();
      rethrow;
    }
  }

  void _fail(Object error) {
    if (_ready != null && !_ready!.isCompleted) _ready!.completeError(error);
    final pending = _pending;
    _pending = null;
    if (pending != null && !pending.isCompleted) pending.completeError(error);
  }

  @override
  Future<YuNetFrameResult> process(
    Uint8List jpeg, {
    bool rotate180 = false,
    bool reset = false,
  }) async {
    await initialize();
    if (_disposed) throw StateError('YuNet worker disposed');
    if (_pending != null) {
      throw StateError('Only one YuNet request may be in flight');
    }
    final pending = _pending = Completer<YuNetFrameResult>();
    _commands!.send((TransferableTypedData.fromList([jpeg]), rotate180, reset));
    try {
      return await pending.future.timeout(const Duration(seconds: 10));
    } on TimeoutException {
      await _shutdown();
      rethrow;
    }
  }

  static void _entry((SendPort, TransferableTypedData) args) {
    RobustYuNet? detector;
    final commands = ReceivePort();
    try {
      // Avoid oversubscribing phone CPUs while keeping the same detector policy.
      cv.setNumThreads(2);
      detector = RobustYuNet(args.$2.materialize().asUint8List());
      args.$1.send(commands.sendPort);
      commands.listen((dynamic request) {
        if (request == null) {
          detector!.dispose();
          commands.close();
          return;
        }
        cv.Mat? decoded, rotated;
        try {
          final (data, rotate, reset) =
              request as (TransferableTypedData, bool, bool);
          final watch = Stopwatch()..start();
          if (reset) detector!.reset();
          decoded = cv.imdecode(
            data.materialize().asUint8List(),
            cv.IMREAD_COLOR,
          );
          if (decoded.isEmpty) {
            throw const FormatException('Invalid JPEG image');
          }
          if (rotate) rotated = cv.rotate(decoded, cv.ROTATE_180);
          final image = rotated ?? decoded;
          final faces = detector!.detect(image);
          args.$1.send({
            'width': image.cols,
            'height': image.rows,
            'faces': [for (final face in faces) face.toRow()],
            'micros': watch.elapsedMicroseconds,
          });
        } catch (error, trace) {
          args.$1.send({'error': '$error\n$trace'});
        } finally {
          rotated?.dispose();
          decoded?.dispose();
        }
      });
    } catch (error, trace) {
      detector?.dispose();
      commands.close();
      args.$1.send({'error': '$error\n$trace'});
    }
  }

  @override
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    await _shutdown();
  }

  Future<void> _shutdown() async {
    _commands?.send(null);
    _commands = null;
    _initializing = null;
    _fail(StateError('YuNet worker stopped'));
    // Normal shutdown lets the worker release native allocations first.
    final isolate = _isolate;
    _isolate = null;
    if (isolate != null) {
      Timer(
        const Duration(seconds: 1),
        () => isolate.kill(priority: Isolate.immediate),
      );
    }
    final subscription = _subscription;
    final messages = _messages;
    _subscription = null;
    _messages = null;
    await subscription?.cancel();
    messages?.close();
  }
}
