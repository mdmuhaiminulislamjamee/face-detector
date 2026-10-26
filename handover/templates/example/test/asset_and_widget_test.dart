import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ncm_face_example/main.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('the package model resolves at the exact worker asset path', () async {
    final bundled = await rootBundle.load(
      'packages/ncm_face_sdk/assets/models/face_detection_yunet_2023mar.onnx',
    );
    final source = File(
      '../sdk/assets/models/face_detection_yunet_2023mar.onnx',
    ).readAsBytesSync();
    expect(
      bundled.buffer.asUint8List(bundled.offsetInBytes, bundled.lengthInBytes),
      orderedEquals(source),
    );
  });

  testWidgets(
    'the NCM example starts disconnected without phone camera access',
    (tester) async {
      await tester.pumpWidget(const MaterialApp(home: NcmAcceptanceScreen()));
      expect(find.text('NCM single-face acceptance'), findsOneWidget);
      expect(find.text('Connect'), findsOneWidget);
      await tester.pumpWidget(const SizedBox.shrink());
      await tester.pump();
    },
  );
}
