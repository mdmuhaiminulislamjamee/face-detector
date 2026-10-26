import 'dart:math' as math;
import 'dart:ui';

import '../models/detected_face.dart';

/// Internal tracking state maintained across consecutive video frames for one target face.
class _TrackedFaceRecord {
  _TrackedFaceRecord({
    required this.trackingId,
    required this.lastFace,
    required this.smoothedBox,
    required this.lastSeenAt,
  }) : velocity = Offset.zero,
       missedFrames = 0;

  final int trackingId;
  DetectedFace lastFace;
  Rect smoothedBox;
  Offset velocity;
  DateTime lastSeenAt;
  int missedFrames;
  int consecutiveHits = 1;
}

/// Advanced face tracking stabilizer that eliminates jitter and prevents target loss
/// when faces move far, turn, or experience temporary frame-by-frame dropouts.
class FaceTrackingStabilizer {
  FaceTrackingStabilizer({
    this.smoothingAlpha = 0.65,
    this.maxDropoutHoldDuration = const Duration(milliseconds: 200),
    this.minIouAssociationThreshold = 0.18,
    this.maxCenterDistanceThreshold = 0.12,
    this.minConsecutiveHits = 3,
    this.responsiveMotion = false,
  });

  /// Exponential moving average weight (0.0 to 1.0). Higher values adapt faster to motion;
  /// lower values smooth out micro-jitter.
  double smoothingAlpha;

  /// Consecutive verified observations required before showing a new face.
  final int minConsecutiveHits;

  /// Follow deliberate motion with the PC YuNet tracker's fast response.
  final bool responsiveMotion;

  /// Grace period to keep tracking a target face when ML Kit temporarily misses it.
  Duration maxDropoutHoldDuration;

  /// Minimum bounding-box Intersection-over-Union to re-associate same target face.
  double minIouAssociationThreshold;

  /// Maximum center-distance threshold to match moving target when IOU is low.
  double maxCenterDistanceThreshold;

  final Map<int, _TrackedFaceRecord> _records = {};
  int _nextGeneratedTrackingId = 1000;
  DateTime? _lastUpdateTimestamp;

  /// Clears all tracking records (e.g. on camera switch).
  void clear() {
    _records.clear();
    _lastUpdateTimestamp = null;
  }

  /// Stabilizes a batch of raw detected faces from the latest camera frame.
  List<DetectedFace> process({
    required List<DetectedFace> rawFaces,
    required DateTime now,
  }) {
    final dtSeconds = _lastUpdateTimestamp != null
        ? math.max(
            0.016,
            now.difference(_lastUpdateTimestamp!).inMicroseconds / 1000000.0,
          )
        : 0.033;
    _lastUpdateTimestamp = now;

    // A stale record cannot be revived by an unrelated detection after a pause.
    _records.removeWhere(
      (_, record) =>
          now.difference(record.lastSeenAt) >
          (maxDropoutHoldDuration > const Duration(milliseconds: 500)
              ? maxDropoutHoldDuration
              : const Duration(milliseconds: 500)),
    );
    final matchedRecordIds = <int>{};
    final stabilizedFaces = <DetectedFace>[];

    for (final raw in rawFaces) {
      final box = raw.displayBox;
      if (!box.left.isFinite ||
          !box.top.isFinite ||
          !box.right.isFinite ||
          !box.bottom.isFinite ||
          box.isEmpty) {
        continue;
      }
      _TrackedFaceRecord? bestRecord;
      var bestScore = -1.0;
      for (final record in _records.values) {
        if (matchedRecordIds.contains(record.trackingId)) continue;
        final iou = _computeIou(box, record.smoothedBox);
        final distance = (box.center - record.smoothedBox.center).distance;
        final areaRatio =
            (box.width * box.height) /
            (record.smoothedBox.width * record.smoothedBox.height);
        final size = math.max(
          box.size.longestSide,
          record.smoothedBox.size.longestSide,
        );
        final maxDistance = math.min(maxCenterDistanceThreshold, size * 0.75);
        if (areaRatio < 0.35 ||
            areaRatio > 2.85 ||
            (iou < minIouAssociationThreshold && distance > maxDistance)) {
          continue;
        }
        final sameId =
            raw.trackingId != null &&
            raw.trackingId == record.lastFace.trackingId;
        final score = iou * 2 + (1 - distance) + (sameId ? 1 : 0);
        if (score > bestScore) {
          bestScore = score;
          bestRecord = record;
        }
      }
      if (bestRecord != null) {
        matchedRecordIds.add(bestRecord.trackingId);
        _updateRecordWithHit(bestRecord, raw, now, dtSeconds);
        if (bestRecord.consecutiveHits >= minConsecutiveHits) {
          stabilizedFaces.add(_buildStabilizedFace(bestRecord, raw));
        }
      } else {
        var assignedId = raw.trackingId;
        if (assignedId == null || _records.containsKey(assignedId)) {
          while (_records.containsKey(_nextGeneratedTrackingId)) {
            _nextGeneratedTrackingId++;
          }
          assignedId = _nextGeneratedTrackingId++;
        }
        final record = _TrackedFaceRecord(
          trackingId: assignedId,
          lastFace: raw,
          smoothedBox: box,
          lastSeenAt: now,
        );
        _records[assignedId] = record;
        matchedRecordIds.add(assignedId);
        if (minConsecutiveHits <= 1) {
          stabilizedFaces.add(_buildStabilizedFace(record, raw));
        }
      }
    }

    // Phase 4: Handle dropouts for existing faces that were missed in this frame
    final expiredIds = <int>[];
    for (final entry in _records.entries) {
      final record = entry.value;
      if (matchedRecordIds.contains(record.trackingId)) continue;

      final elapsedSinceHit = now.difference(record.lastSeenAt);
      if (record.consecutiveHits < minConsecutiveHits) {
        expiredIds.add(record.trackingId);
        continue;
      }
      if (elapsedSinceHit <= maxDropoutHoldDuration) {
        // Face is temporarily dropped (blinking, head turned, distant motion).
        // Apply dead-reckoning velocity damping and hold target lock!
        record.missedFrames++;
        final dampedDt = dtSeconds * 0.35;
        final predictedOffset = record.velocity * dampedDt;
        record.smoothedBox = _shiftAndClampRect(
          record.smoothedBox,
          predictedOffset,
        );

        stabilizedFaces.add(
          record.lastFace.copyWith(
            trackingId: record.trackingId,
            displayBox: record.smoothedBox,
            isTemporarilyLost: true,
            missedFrameCount: record.missedFrames,
          ),
        );
      } else {
        expiredIds.add(record.trackingId);
      }
    }

    for (final expiredId in expiredIds) {
      _records.remove(expiredId);
    }

    return stabilizedFaces;
  }

  void _updateRecordWithHit(
    _TrackedFaceRecord record,
    DetectedFace hit,
    DateTime now,
    double dtSeconds,
  ) {
    record.consecutiveHits++;
    final oldCenter = record.smoothedBox.center;
    final newCenter = hit.displayBox.center;

    // Velocity update: pixels/second in normalized preview space
    if (dtSeconds > 0) {
      final instantVelocity = (newCenter - oldCenter) / dtSeconds;
      record.velocity = (record.velocity * 0.4) + (instantVelocity * 0.6);
    }

    // Exponential smoothing for bounding box coordinates
    final relativeMotion =
        (newCenter - oldCenter).distance /
        math.max(0.000001, hit.displayBox.size.longestSide);
    final alpha = responsiveMotion && relativeMotion >= .10
        ? math.max(smoothingAlpha, .92).clamp(0.0, 1.0)
        : smoothingAlpha.clamp(0.0, 1.0);
    final sLeft =
        (alpha * hit.displayBox.left) + ((1 - alpha) * record.smoothedBox.left);
    final sTop =
        (alpha * hit.displayBox.top) + ((1 - alpha) * record.smoothedBox.top);
    final sRight =
        (alpha * hit.displayBox.right) +
        ((1 - alpha) * record.smoothedBox.right);
    final sBottom =
        (alpha * hit.displayBox.bottom) +
        ((1 - alpha) * record.smoothedBox.bottom);

    record.smoothedBox = Rect.fromLTRB(sLeft, sTop, sRight, sBottom);
    record.lastFace = hit;
    record.lastSeenAt = now;
    record.missedFrames = 0;
  }

  DetectedFace _buildStabilizedFace(
    _TrackedFaceRecord record,
    DetectedFace raw,
  ) {
    return raw.copyWith(
      trackingId: record.trackingId,
      displayBox: record.smoothedBox,
      isTemporarilyLost: false,
      missedFrameCount: 0,
    );
  }

  double _computeIou(Rect a, Rect b) {
    final intersection = a.intersect(b);
    if (intersection.width <= 0 || intersection.height <= 0) return 0.0;

    final intersectionArea = intersection.width * intersection.height;
    final unionArea =
        (a.width * a.height) + (b.width * b.height) - intersectionArea;
    if (unionArea <= 0) return 0.0;

    return (intersectionArea / unionArea).clamp(0.0, 1.0);
  }

  Rect _shiftAndClampRect(Rect rect, Offset offset) {
    final shifted = rect.shift(offset);
    final width = shifted.width;
    final height = shifted.height;
    final left = shifted.left.clamp(0.0, 1.0 - width);
    final top = shifted.top.clamp(0.0, 1.0 - height);
    return Rect.fromLTWH(left, top, width, height);
  }
}
