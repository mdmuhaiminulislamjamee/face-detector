import 'dart:ui';

import 'package:face_detector/models/detected_face.dart';
import 'package:face_detector/services/face_tracking_stabilizer.dart';
import 'package:flutter_test/flutter_test.dart';

const face = DetectedFace(
  trackingId: 7,
  boundingBox: Rect.fromLTWH(100, 100, 40, 50),
  displayBox: Rect.fromLTWH(0.2, 0.2, 0.04, 0.05),
);

void main() {
  final start = DateTime(2026);

  test('requires three consecutive observations even with smoothing off', () {
    final tracker = FaceTrackingStabilizer(
      smoothingAlpha: 1,
      maxDropoutHoldDuration: Duration.zero,
    );
    expect(tracker.process(rawFaces: [face], now: start), isEmpty);
    expect(
      tracker.process(
        rawFaces: [face],
        now: start.add(const Duration(milliseconds: 50)),
      ),
      isEmpty,
    );
    final confirmed = tracker.process(
      rawFaces: [face],
      now: start.add(const Duration(milliseconds: 100)),
    );
    expect(confirmed, hasLength(1));
    expect(confirmed.single.isTemporarilyLost, isFalse);
    expect(
      tracker.process(
        rawFaces: [],
        now: start.add(const Duration(milliseconds: 150)),
      ),
      isEmpty,
    );
  });

  test('a missed candidate cannot accumulate intermittent hits', () {
    final tracker = FaceTrackingStabilizer();
    for (var i = 0; i < 6; i++) {
      expect(
        tracker.process(
          rawFaces: i.isEven ? [face] : [],
          now: start.add(Duration(milliseconds: i * 50)),
        ),
        isEmpty,
      );
    }
  });

  test('nearby small objects and recycled IDs cannot inherit confirmation', () {
    final tracker = FaceTrackingStabilizer();
    for (var i = 0; i < 3; i++) {
      tracker.process(
        rawFaces: [face],
        now: start.add(Duration(milliseconds: i * 50)),
      );
    }
    final unrelated = face.copyWith(
      displayBox: const Rect.fromLTWH(0.3, 0.2, 0.04, 0.05),
    );
    final result = tracker.process(
      rawFaces: [unrelated],
      now: start.add(const Duration(milliseconds: 150)),
    );
    expect(result.where((f) => !f.isTemporarilyLost), isEmpty);
    final late = tracker.process(
      rawFaces: [face],
      now: start.add(const Duration(seconds: 2)),
    );
    expect(late, isEmpty);
  });

  test('two detections with one native ID never confirm each other', () {
    final tracker = FaceTrackingStabilizer();
    final other = face.copyWith(
      displayBox: const Rect.fromLTWH(0.7, 0.2, 0.04, 0.05),
    );
    expect(tracker.process(rawFaces: [face, other], now: start), isEmpty);
    expect(
      tracker.process(
        rawFaces: [face, other],
        now: start.add(const Duration(milliseconds: 50)),
      ),
      isEmpty,
    );
    final confirmed = tracker.process(
      rawFaces: [face, other],
      now: start.add(const Duration(milliseconds: 100)),
    );
    expect(confirmed, hasLength(2));
    expect(confirmed.map((f) => f.trackingId).toSet(), hasLength(2));
  });
}
