import 'dart:async';

/// A single in-flight job plus one replaceable pending frame. The cap limits
/// actual processing starts, including failed attempts, using a monotonic clock.
class LatestFrameRunner<T, R> {
  LatestFrameRunner({
    required this.process,
    required this.onResult,
    required this.onError,
    this.isFresh,
    this.interval = const Duration(milliseconds: 100),
  });

  final Future<R> Function(T) process;
  final void Function(T, R) onResult;
  final void Function(Object, StackTrace) onError;
  final bool Function(T)? isFresh;
  final Duration interval;
  final _clock = Stopwatch()..start();
  Timer? _timer;
  T? _pending;
  bool _busy = false, _disposed = false;
  int _generation = 0;
  int? _lastStart;
  int droppedFrames = 0;

  void submit(T frame) {
    if (_disposed) return;
    if (_pending != null) droppedFrames++;
    _pending = frame;
    _schedule();
  }

  void _schedule() {
    if (_disposed || _busy || _timer != null || _pending == null) return;
    final remaining = _lastStart == null
        ? 0
        : interval.inMicroseconds - (_clock.elapsedMicroseconds - _lastStart!);
    if (remaining > 0) {
      _timer = Timer(Duration(microseconds: remaining), () {
        _timer = null;
        _schedule();
      });
    } else {
      unawaited(_run());
    }
  }

  Future<void> _run() async {
    final frame = _pending as T;
    _pending = null;
    if (!(isFresh?.call(frame) ?? true)) {
      droppedFrames++;
      return;
    }
    _busy = true;
    _lastStart = _clock.elapsedMicroseconds;
    final generation = _generation;
    try {
      final result = await process(frame);
      if (!_disposed && generation == _generation) onResult(frame, result);
    } catch (error, trace) {
      if (!_disposed && generation == _generation) onError(error, trace);
    } finally {
      _busy = false;
      _schedule();
    }
  }

  void clear() {
    _generation++;
    _pending = null;
    _timer?.cancel();
    _timer = null;
  }

  void dispose() {
    clear();
    _disposed = true;
  }
}
