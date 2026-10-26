import 'dart:async';

import 'package:face_detector/ncm/latest_frame_runner.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
    'slow inference processes only newest pending frame and never overlaps',
    () async {
      final gates = <Completer<int>>[];
      final started = <int>[];
      final results = <int>[];
      final runner = LatestFrameRunner<int, int>(
        process: (value) {
          started.add(value);
          final gate = Completer<int>();
          gates.add(gate);
          return gate.future;
        },
        onResult: (frame, result) => results.add(result),
        onError: (e, s) => fail('$e'),
      );
      try {
        runner.submit(1);
        for (var i = 2; i <= 50; i++) {
          runner.submit(i);
        }
        await Future<void>.delayed(const Duration(milliseconds: 130));
        expect(started, [1]);
        gates[0].complete(1);
        await Future<void>.delayed(const Duration(milliseconds: 10));
        expect(started, [1, 50]);
        gates[1].complete(50);
        await Future<void>.delayed(Duration.zero);
        expect(results, [1, 50]);
        expect(runner.droppedFrames, 48);
      } finally {
        runner.dispose();
      }
    },
  );

  test('100 ms start cap applies to fast jobs and failures', () async {
    final clock = Stopwatch()..start();
    final starts = <int>[];
    final runner = LatestFrameRunner<int, int>(
      process: (value) async {
        starts.add(clock.elapsedMicroseconds);
        if (value == 0) throw StateError('bad frame');
        return value;
      },
      onResult: (f, r) {},
      onError: (e, s) {},
    );
    final producer = Timer.periodic(const Duration(milliseconds: 3), (timer) {
      runner.submit(timer.tick);
    });
    try {
      runner.submit(0);
      await Future<void>.delayed(const Duration(milliseconds: 360));
      expect(starts.length, inInclusiveRange(2, 4));
      for (var i = 1; i < starts.length; i++) {
        expect(starts[i] - starts[i - 1], greaterThanOrEqualTo(100000));
      }
    } finally {
      producer.cancel();
      runner.dispose();
    }
  });

  test('clear suppresses results from a previous camera session', () async {
    final gate = Completer<int>();
    final results = <int>[];
    final runner = LatestFrameRunner<int, int>(
      process: (f) => gate.future,
      onResult: (f, r) => results.add(r),
      onError: (e, s) => fail('$e'),
    );
    runner.submit(1);
    runner.clear();
    gate.complete(1);
    await Future<void>.delayed(Duration.zero);
    expect(results, isEmpty);
    runner.dispose();
  });

  test('expired frames are discarded before inference', () async {
    var calls = 0;
    final runner = LatestFrameRunner<int, int>(
      process: (f) async {
        calls++;
        return f;
      },
      isFresh: (_) => false,
      onResult: (f, r) {},
      onError: (e, s) => fail('$e'),
    );
    runner.submit(1);
    await Future<void>.delayed(Duration.zero);
    expect(calls, 0);
    expect(runner.droppedFrames, 1);
    runner.dispose();
  });
}
