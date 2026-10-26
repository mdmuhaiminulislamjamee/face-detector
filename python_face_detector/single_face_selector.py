"""Expose one confirmed target while keeping automatic replacement available."""


class SingleFaceSelector:
    def __init__(self):
        self.locked_tracking_id = None

    def clear(self):
        self.locked_tracking_id = None

    def select(self, tracked_faces):
        # The tracker supplies the existing 200ms dropout grace period. A held
        # target keeps its lock briefly, but a held alternative cannot acquire it.
        if self.locked_tracking_id is not None:
            for face in tracked_faces:
                if face.tracking_id == self.locked_tracking_id:
                    return [face]
        fresh = [face for face in tracked_faces if not face.is_holding]
        if not fresh:
            self.clear()
            return []
        selected = min(fresh, key=lambda face: (
            -(face.smoothed_bbox[2] * face.smoothed_bbox[3]),
            -face.confidence, face.tracking_id))
        self.locked_tracking_id = selected.tracking_id
        return [selected]
