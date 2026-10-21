"""Shared representations of YOLO detections and associated vehicle tracks."""
from dataclasses import dataclass

VEHICLE_LABELS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]  # x, y, width, height
    class_id: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)


@dataclass
class Track:
    track_id: int
    x: int
    y: int
    class_id: int
    box: tuple[int, int, int, int]
    confidence: float
    missing: int = 0
    # Last two *observed* centers, even after a temporarily missing frame.
    previous: tuple[int, int] | None = None

    @property
    def center(self) -> tuple[int, int]:
        return (self.x, self.y)
