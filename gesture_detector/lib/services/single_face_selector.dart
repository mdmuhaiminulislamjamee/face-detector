import '../models/detected_face.dart';

/// Selects one confirmed face; never supplies a detector or confirmation hit.
class SingleFaceSelector {
  int? _lockedTrackingId;
  int? get lockedTrackingId => _lockedTrackingId;

  void clear() => _lockedTrackingId = null;

  List<DetectedFace> select(List<DetectedFace> trackedFaces) {
    if (_lockedTrackingId != null) {
      for (final face in trackedFaces) {
        if (face.trackingId == _lockedTrackingId) return [face];
      }
    }
    final fresh = trackedFaces
        .where((face) => !face.isTemporarilyLost && face.trackingId != null)
        .toList();
    if (fresh.isEmpty) {
      clear();
      return const [];
    }
    fresh.sort((a, b) {
      final area = (b.displayBox.width * b.displayBox.height).compareTo(
        a.displayBox.width * a.displayBox.height,
      );
      if (area != 0) return area;
      final score = (b.detectionConfidence ?? 0).compareTo(
        a.detectionConfidence ?? 0,
      );
      return score != 0 ? score : a.trackingId!.compareTo(b.trackingId!);
    });
    _lockedTrackingId = fresh.first.trackingId;
    return [fresh.first];
  }
}
