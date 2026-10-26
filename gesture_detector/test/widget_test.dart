import 'package:face_detector/app/face_detector_app.dart';
import 'package:face_detector/presentation/screens/camera_source_screen.dart';
import 'package:face_detector/presentation/screens/face_detector_screen.dart';
import 'package:face_detector/presentation/widgets/camera_loading_view.dart';
import 'package:face_detector/presentation/widgets/face_count_badge.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('NCM route opens without requesting the phone camera', (
    tester,
  ) async {
    await tester.pumpWidget(const FaceDetectorApp());
    await tester.tap(find.text('USB-NCM camera'));
    await tester.pumpAndSettle();
    expect(find.text('Connect the camera through USB'), findsOneWidget);
    expect(find.text('Connect camera'), findsOneWidget);
    expect(find.text('Phone USB IP'), findsOneWidget);
    expect(find.byType(FaceDetectorScreen), findsNothing);
    await tester.pumpWidget(const SizedBox.shrink());
  });
  testWidgets(
    'FaceDetectorApp offers NCM and phone cameras before requesting access',
    (tester) async {
      await tester.pumpWidget(const FaceDetectorApp());

      expect(find.byType(MaterialApp), findsOneWidget);
      expect(find.byType(CameraSourceScreen), findsOneWidget);
      expect(find.text('USB-NCM camera'), findsOneWidget);
      expect(find.text('Phone camera'), findsOneWidget);
    },
  );

  testWidgets(
    'FaceDetectorScreen with autoStartCamera=false shows ready screen',
    (tester) async {
      await tester.pumpWidget(
        const MaterialApp(home: FaceDetectorScreen(autoStartCamera: false)),
      );

      expect(find.byType(CameraLoadingView), findsOneWidget);
      expect(find.text('Face Detector'), findsOneWidget);
      expect(find.text('Start Camera'), findsOneWidget);
    },
  );

  testWidgets('FaceCountBadge shows searching text when 0 faces', (
    tester,
  ) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: FaceCountBadge(faceCount: 0, latency: Duration.zero),
        ),
      ),
    );

    expect(find.text('Searching for faces...'), findsOneWidget);
  });

  testWidgets('FaceCountBadge shows count when faces detected', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: FaceCountBadge(
            faceCount: 2,
            latency: Duration(milliseconds: 25),
          ),
        ),
      ),
    );

    expect(find.text('2 Faces Detected'), findsOneWidget);
    expect(find.text('25ms'), findsOneWidget);
  });
}
