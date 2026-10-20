"""Class-aware greedy centroid tracker; optimized for simple demos, not occlusions."""
from math import hypot
from src.components.models import Detection, Track


class VehicleTracker:
    def __init__(self, max_distance: float = 90, max_missing: int = 10):
        if max_distance <= 0 or max_missing < 0:
            raise ValueError("Invalid tracker parameters")
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> dict[int, Track]:
        """Return *visible* tracks only, retaining missing tracks internally.

        Match existing IDs to current detections by smallest Euclidean distance.
        """
        candidates = []
        for track_id, track in self.tracks.items():
            for detection_idx, det in enumerate(detections):
                if track.class_id != det.class_id:
                    continue
                distance = hypot(track.x - det.center[0], track.y - det.center[1])
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, detection_idx))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, detection_idx in sorted(candidates):
            if track_id in used_tracks or detection_idx in used_detections:
                continue
            det = detections[detection_idx]
            track = self.tracks[track_id]
            track.previous = track.center
            track.x, track.y = det.center
            track.box = det.box
            track.confidence = det.confidence
            track.missing = 0
            used_tracks.add(track_id)
            used_detections.add(detection_idx)
        for track_id in list(self.tracks):
            if track_id not in used_tracks:
                self.tracks[track_id].missing += 1
                if self.tracks[track_id].missing > self.max_missing:
                    del self.tracks[track_id]
        for index, det in enumerate(detections):
            if index in used_detections:
                continue
            x, y = det.center
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(track_id, x, y, det.class_id, det.box, det.confidence)
            used_tracks.add(track_id)
        return {track_id: self.tracks[track_id] for track_id in sorted(used_tracks)}
