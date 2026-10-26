import 'dart:ui';
import 'package:face_detector/models/detected_face.dart';
import 'package:face_detector/services/face_tracking_stabilizer.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
    'NCM motion follows deliberate movement promptly without changing phone smoothing',
    () {
      final tracker = FaceTrackingStabilizer(
        responsiveMotion: true,
        minConsecutiveHits: 1,
      );
      final now = DateTime(2026);
      const face = DetectedFace(
        boundingBox: Rect.fromLTWH(100, 100, 100, 100),
        displayBox: Rect.fromLTWH(.2, .2, .2, .2),
      );
      tracker.process(rawFaces: [face], now: now);
      final moved = tracker.process(
        rawFaces: [
          face.copyWith(displayBox: const Rect.fromLTWH(.25, .2, .2, .2)),
        ],
        now: now.add(const Duration(milliseconds: 100)),
      );
      expect(moved.single.displayBox.left, greaterThanOrEqualTo(.246));
    },
  );
  group('FaceTrackingStabilizer', () {
    test('smooths bounding box jitter across consecutive hits', () {
      final stabilizer = FaceTrackingStabilizer(
        smoothingAlpha: 0.5,
        minConsecutiveHits: 1,
      );
      final t0 = DateTime(2026, 1, 1, 12, 0, 0);
      final t1 = t0.add(const Duration(milliseconds: 33));

      const rawFace1 = DetectedFace(
        trackingId: 1,
        boundingBox: Rect.fromLTWH(100, 100, 100, 100),
        displayBox: Rect.fromLTWH(0.2, 0.2, 0.2, 0.2),
      );

      final stabilized1 = stabilizer.process(rawFaces: [rawFace1], now: t0);
      expect(stabilized1.length, 1);
      expect(
        stabilized1.first.displayBox,
        const Rect.fromLTWH(0.2, 0.2, 0.2, 0.2),
      );

      // Second frame with jump
      const rawFace2 = DetectedFace(
        trackingId: 1,
        boundingBox: Rect.fromLTWH(120, 100, 100, 100),
        displayBox: Rect.fromLTWH(0.3, 0.2, 0.2, 0.2),
      );

      final stabilized2 = stabilizer.process(rawFaces: [rawFace2], now: t1);
      expect(stabilized2.length, 1);
      // Smoothed: 0.5 * 0.3 + 0.5 * 0.2 = 0.25 left
      expect(stabilized2.first.displayBox.left, closeTo(0.25, 0.001));
      expect(stabilized2.first.isTemporarilyLost, isFalse);
    });

    test(
      'holds target lock during temporary frame dropouts within grace period',
      () {
        final stabilizer = FaceTrackingStabilizer(
          minConsecutiveHits: 1,
          maxDropoutHoldDuration: const Duration(milliseconds: 500),
        );
        final t0 = DateTime(2026, 1, 1, 12, 0, 0);

        const rawFace = DetectedFace(
          trackingId: 1,
          boundingBox: Rect.fromLTWH(100, 100, 100, 100),
          displayBox: Rect.fromLTWH(0.3, 0.3, 0.2, 0.2),
        );

        stabilizer.process(rawFaces: [rawFace], now: t0);

        // Frame at +100ms with 0 detected faces (temporary miss)
        final t1 = t0.add(const Duration(milliseconds: 100));
        final held1 = stabilizer.process(rawFaces: const [], now: t1);
        expect(held1.length, 1);
        expect(held1.first.trackingId, 1);
        expect(held1.first.isTemporarilyLost, isTrue);
        expect(held1.first.missedFrameCount, 1);

        // Frame at +300ms still 0 faces
        final t2 = t0.add(const Duration(milliseconds: 300));
        final held2 = stabilizer.process(rawFaces: const [], now: t2);
        expect(held2.length, 1);
        expect(held2.first.isTemporarilyLost, isTrue);
        expect(held2.first.missedFrameCount, 2);

        // Re-acquired at +400ms
        final t3 = t0.add(const Duration(milliseconds: 400));
        final recovered = stabilizer.process(rawFaces: [rawFace], now: t3);
        expect(recovered.length, 1);
        expect(recovered.first.isTemporarilyLost, isFalse);
        expect(recovered.first.missedFrameCount, 0);
      },
    );

    test('drops target when grace period expires', () {
      final stabilizer = FaceTrackingStabilizer(
        minConsecutiveHits: 1,
        maxDropoutHoldDuration: const Duration(milliseconds: 300),
      );
      final t0 = DateTime(2026, 1, 1, 12, 0, 0);

      const rawFace = DetectedFace(
        trackingId: 1,
        boundingBox: Rect.fromLTWH(100, 100, 100, 100),
        displayBox: Rect.fromLTWH(0.3, 0.3, 0.2, 0.2),
      );

      stabilizer.process(rawFaces: [rawFace], now: t0);

      // Frame at +500ms (exceeds 300ms grace period)
      final tExpired = t0.add(const Duration(milliseconds: 500));
      final result = stabilizer.process(rawFaces: const [], now: tExpired);
      expect(result, isEmpty);
    });

    test('associates by IOU and proximity when tracking ID is absent', () {
      final stabilizer = FaceTrackingStabilizer(minConsecutiveHits: 1);
      final t0 = DateTime(2026, 1, 1, 12, 0, 0);

      const initialHit = DetectedFace(
        trackingId: null,
        boundingBox: Rect.fromLTWH(100, 100, 100, 100),
        displayBox: Rect.fromLTWH(0.4, 0.4, 0.2, 0.2),
      );

      final out0 = stabilizer.process(rawFaces: [initialHit], now: t0);
      final assignedId = out0.first.trackingId;
      expect(assignedId, isNotNull);

      // Next frame without tracking ID but with overlapping box
      const nextHit = DetectedFace(
        trackingId: null,
        boundingBox: Rect.fromLTWH(105, 105, 100, 100),
        displayBox: Rect.fromLTWH(0.41, 0.41, 0.2, 0.2),
      );

      final t1 = t0.add(const Duration(milliseconds: 33));
      final out1 = stabilizer.process(rawFaces: [nextHit], now: t1);
      expect(out1.first.trackingId, assignedId); // Maintains same tracking ID
    });
  });
}
