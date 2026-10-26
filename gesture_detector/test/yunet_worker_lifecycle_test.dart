import 'dart:async';

import 'package:face_detector/services/yunet_worker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final binding = TestWidgetsFlutterBinding.ensureInitialized();

  tearDown(() {
    binding.defaultBinaryMessenger.setMockMessageHandler(
      'flutter/assets',
      null,
    );
  });

  test(
    'model loading failure can be retried instead of retaining a failed future',
    () async {
      var attempts = 0;
      binding.defaultBinaryMessenger.setMockMessageHandler('flutter/assets', (
        _,
      ) async {
        attempts++;
        return null;
      });
      final worker = YuNetWorker();
      try {
        await expectLater(worker.initialize(), throwsA(isA<FlutterError>()));
        await expectLater(worker.initialize(), throwsA(isA<FlutterError>()));
        expect(attempts, 2);
      } finally {
        await worker.dispose();
      }
    },
  );

  test(
    'disposing during asset loading prevents native worker startup',
    () async {
      final requested = Completer<void>();
      final bytes = Completer<ByteData?>();
      binding.defaultBinaryMessenger.setMockMessageHandler('flutter/assets', (
        _,
      ) {
        requested.complete();
        return bytes.future;
      });
      final worker = YuNetWorker();
      final loading = expectLater(worker.initialize(), throwsStateError);
      await requested.future;
      await worker.dispose();
      bytes.complete(ByteData(1));
      await loading;
      await expectLater(worker.initialize(), throwsStateError);
    },
  );
}
