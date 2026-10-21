"""Lightweight tracker with duplicate suppression and short-gap motion prediction.

The same track ID is required by the crossing counter; prevent unnecessary ID
changes due to close overlapping detections, intermittent detection and a
single unstable class prediction. This is not an occlusion-proof MOT tracker.
"""
from collections import Counter, deque
from math import hypot

from src.components.models import Detection, Track


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of (x, y, width, height) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    overlap = overlap_w * overlap_h
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def deduplicate_detections(detections: list[Detection], overlap: float = 0.65) -> list[Detection]:
    """Suppress near-identical boxes, including predictions with different classes.

    A high overlap threshold helps retain nearby genuine road users.
    """
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(box_iou(det.box, other.box) < overlap for other in kept):
            kept.append(det)
    return kept


class VehicleTracker:
    def __init__(self, max_distance: float = 90, max_missing: int = 10):
        if max_distance <= 0 or max_missing < 0:
            raise ValueError("Invalid tracker parameters")
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks: dict[int, Track] = {}
        self._velocity: dict[int, tuple[float, float]] = {}
        self._classes: dict[int, deque[int]] = {}

    def update(self, detections: list[Detection]) -> dict[int, Track]:
        """Associate visible detections and return only visible tracks.

        Match to a predicted position after a short gap. A short-lived class
        change can match only where the actual boxes strongly overlap.
        """
        detections = deduplicate_detections(detections)
        candidates = []
        for track_id, track in self.tracks.items():
            gap = track.missing + 1
            vx, vy = self._velocity.get(track_id, (0.0, 0.0))
            # Extrapolate briefly; long outages should not jump onto another car.
            predicted = (track.x + vx * min(gap, 3), track.y + vy * min(gap, 3))
            # A little more latitude for smaller, fast-moving motorcycles.
            gate = self.max_distance * (1.35 if track.class_id == 3 else 1.0)
            gate *= 1 + 0.2 * min(track.missing, 3)
            for detection_idx, det in enumerate(detections):
                d = hypot(predicted[0] - det.center[0], predicted[1] - det.center[1])
                if d > gate:
                    continue
                overlap = box_iou(track.box, det.box)
                if track.class_id != det.class_id and overlap < 0.35:
                    continue
                # Strong overlap helps avoid ID swaps between neighboring vehicles.
                cost = d - min(gate * 0.25, overlap * 20)
                if track.class_id != det.class_id:
                    cost += gate * 0.25
                candidates.append((cost, track_id, detection_idx))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, detection_idx in sorted(candidates):
            if track_id in used_tracks or detection_idx in used_detections:
                continue
            det = detections[detection_idx]
            track = self.tracks[track_id]
            old_center = track.center
            gap = track.missing + 1
            new_center = det.center
            self._velocity[track_id] = ((new_center[0] - old_center[0]) / gap,
                                        (new_center[1] - old_center[1]) / gap)
            track.previous = old_center
            track.x, track.y = new_center
            track.box = det.box
            track.confidence = det.confidence
            track.missing = 0
            votes = self._classes[track_id]
            votes.append(det.class_id)
            counts = Counter(votes)
            # Preserve the established label on a tie.
            track.class_id = max(counts, key=lambda cls: (counts[cls], cls == track.class_id))
            used_tracks.add(track_id)
            used_detections.add(detection_idx)

        for track_id in list(self.tracks):
            if track_id not in used_tracks:
                self.tracks[track_id].missing += 1
                if self.tracks[track_id].missing > self.max_missing:
                    del self.tracks[track_id]
                    self._velocity.pop(track_id, None)
                    self._classes.pop(track_id, None)

        for index, det in enumerate(detections):
            if index in used_detections:
                continue
            x, y = det.center
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(track_id, x, y, det.class_id, det.box, det.confidence)
            self._velocity[track_id] = (0.0, 0.0)
            self._classes[track_id] = deque([det.class_id], maxlen=5)
            used_tracks.add(track_id)
        return {track_id: self.tracks[track_id] for track_id in sorted(used_tracks)}
