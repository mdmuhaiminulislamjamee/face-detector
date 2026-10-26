import 'dart:ui';

import 'package:face_detector/models/detected_face.dart';
import 'package:face_detector/services/face_tracking_stabilizer.dart';
import 'package:face_detector/services/single_face_selector.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const a = DetectedFace(
    trackingId: 1,
    detectionConfidence: .95,
    boundingBox: Rect.fromLTWH(100, 100, 60, 80),
    displayBox: Rect.fromLTWH(.1, .2, .1, .15),
  );
  const b = DetectedFace(
    trackingId: 2,
    detectionConfidence: .96,
    boundingBox: Rect.fromLTWH(400, 100, 100, 120),
    displayBox: Rect.fromLTWH(.6, .2, .2, .25),
  );

  test(
    'only one target is returned and detector order cannot switch a live lock',
    () {
      final selector = SingleFaceSelector();
      expect(selector.select([a, b]).single.trackingId, 2);
      expect(selector.select([b, a]).single.trackingId, 2);
      expect(
        selector
            .select([
              b,
              a.copyWith(displayBox: const Rect.fromLTWH(0, 0, .8, .8)),
            ])
            .single
            .trackingId,
        2,
      );
    },
  );

  test(
    'a held target keeps its brief lock then a fresh replacement is selected',
    () {
      final selector = SingleFaceSelector();
      selector.select([a, b]);
      final held = b.copyWith(isTemporarilyLost: true);
      expect(selector.select([a, held]).single.trackingId, 2);
      expect(selector.select([a]).single.trackingId, 1);
      expect(selector.select([a]).single.isTemporarilyLost, isFalse);
    },
  );

  test(
    'held alternatives cannot acquire a lock and clear resets selection',
    () {
      final selector = SingleFaceSelector();
      expect(selector.select([a.copyWith(isTemporarilyLost: true)]), isEmpty);
      selector.select([a]);
      selector.clear();
      expect(selector.lockedTrackingId, isNull);
      expect(selector.select([b, a]).single.trackingId, 2);
      expect(selector.select([]), isEmpty);
      expect(selector.lockedTrackingId, isNull);
    },
  );

  test(
    'replacement waits for its own confirmation and output never exceeds one',
    () {
      final tracker = FaceTrackingStabilizer();
      final selector = SingleFaceSelector();
      final start = DateTime(2026);
      List<DetectedFace> step(int ms, List<DetectedFace> faces) {
        final result = selector.select(
          tracker.process(
            rawFaces: faces,
            now: start.add(Duration(milliseconds: ms)),
          ),
        );
        expect(result.length, lessThanOrEqualTo(1));
        return result;
      }

      expect(step(0, [a]), isEmpty);
      expect(step(100, [a]), isEmpty);
      expect(step(200, [a]).single.trackingId, 1);
      expect(step(450, [b]), isEmpty);
      expect(step(550, [b]), isEmpty);
      expect(step(650, [b]).single.trackingId, 2);
    },
  );

  test('a ready alternative takes over after the 200ms dropout hold', () {
    final tracker = FaceTrackingStabilizer();
    final selector = SingleFaceSelector();
    final start = DateTime(2026);
    for (final ms in [0, 100, 200]) {
      selector.select(
        tracker.process(
          rawFaces: [a, b],
          now: start.add(Duration(milliseconds: ms)),
        ),
      );
    }
    final held = selector.select(
      tracker.process(
        rawFaces: [a],
        now: start.add(const Duration(milliseconds: 300)),
      ),
    );
    expect(held.single.trackingId, 2);
    expect(held.single.isTemporarilyLost, isTrue);
    final replacement = selector.select(
      tracker.process(
        rawFaces: [a],
        now: start.add(const Duration(milliseconds: 450)),
      ),
    );
    expect(replacement.single.trackingId, 1);
    expect(replacement.single.isTemporarilyLost, isFalse);
  });
}
