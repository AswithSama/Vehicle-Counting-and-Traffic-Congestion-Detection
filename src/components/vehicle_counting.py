"""Independent unique line-crossing flow and sustained ROI occupancy metrics.

Flow = total distinct tracked IDs that cross a horizontal line (per class).
Occupancy = visible tracked IDs whose centers are inside the configured ROI.
Persistent occupancy triggers a *potential congestion* alert; it does not
measure speed or prove traffic is stopped.
"""
from collections import Counter
from dataclasses import dataclass
from src.components.models import Track, VEHICLE_LABELS


@dataclass(frozen=True)
class TrafficConfig:
    occupancy_threshold: int = 25
    persistence_seconds: float = 180.0
    recovery_seconds: float = 10.0
    line_y_fraction: float = 0.65
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    line_margin_px: int = 4

    def __post_init__(self):
        if self.occupancy_threshold < 1 or self.persistence_seconds < 0 or self.recovery_seconds < 0:
            raise ValueError("Threshold must be positive and durations nonnegative")
        if not 0.0 < self.line_y_fraction < 1.0:
            raise ValueError("Counting line fraction must be between 0 and 1")
        x1, y1, x2, y2 = self.roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("ROI should be fractional coordinates: 0 <= x1 < x2 <= 1; same for y")
        if self.line_margin_px < 0:
            raise ValueError("Line margin must be nonnegative")


class TrafficMonitor:
    def __init__(self, config: TrafficConfig):
        self.config = config
        self.seen_side: dict[int, int] = {}
        self.counted_ids: set[int] = set()
        self.flow_by_class: Counter = Counter()
        self.above_since: float | None = None
        self.below_since: float | None = None
        self.congested = False
        self.last_time: float | None = None

    def geometry(self, frame_shape) -> tuple[int, tuple[int, int, int, int]]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = self.config.roi
        roi = (int(x1 * w), int(y1 * h), min(w - 1, int(x2 * w)), min(h - 1, int(y2 * h)))
        return int(h * self.config.line_y_fraction), roi

    def update(self, timestamp_seconds: float, tracks: dict[int, Track], frame_shape) -> dict:
        if timestamp_seconds < 0 or (self.last_time is not None and timestamp_seconds < self.last_time):
            raise ValueError("Video timestamps must be nonnegative and monotonically increasing")
        self.last_time = timestamp_seconds
        line_y, (x1, y1, x2, y2) = self.geometry(frame_shape)
        current_by_class = Counter()
        for track_id, track in tracks.items():
            if x1 <= track.x <= x2 and y1 <= track.y <= y2:
                current_by_class[track.class_id] += 1
            side = 0
            if track.y < line_y - self.config.line_margin_px:
                side = -1
            elif track.y > line_y + self.config.line_margin_px:
                side = 1
            if side:
                previous_side = self.seen_side.get(track_id)
                if previous_side is not None and previous_side != side and track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.flow_by_class[track.class_id] += 1
                self.seen_side[track_id] = side

        occupancy = sum(current_by_class.values())
        event = None
        if occupancy >= self.config.occupancy_threshold:
            self.below_since = None
            if self.above_since is None:
                self.above_since = timestamp_seconds
            if not self.congested and timestamp_seconds - self.above_since >= self.config.persistence_seconds:
                self.congested = True
                event = "CONGESTION_STARTED"
        else:
            self.above_since = None
            if self.congested:
                if self.below_since is None:
                    self.below_since = timestamp_seconds
                if timestamp_seconds - self.below_since >= self.config.recovery_seconds:
                    self.congested = False
                    self.below_since = None
                    event = "CONGESTION_CLEARED"

        return {
            "timestamp_seconds": round(timestamp_seconds, 3),
            "occupancy": occupancy,
            "occupancy_by_class": {VEHICLE_LABELS[i]: current_by_class[i] for i in VEHICLE_LABELS},
            "total_line_crossings": len(self.counted_ids),
            "flow_by_class": {VEHICLE_LABELS[i]: self.flow_by_class[i] for i in VEHICLE_LABELS},
            "threshold": self.config.occupancy_threshold,
            "congested": self.congested,
            "above_threshold_seconds": round(timestamp_seconds - self.above_since, 3) if self.above_since is not None else 0.0,
            "event": event,
        }
