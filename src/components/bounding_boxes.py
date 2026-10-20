"""Overlay vehicle boxes, ROI, crossing line, and live traffic counters."""
import cv2
from src.components.models import VEHICLE_LABELS

UNCROSSED_COLOR = (255, 160, 60)  # BGR blue
CROSSED_COLOR = (0, 255, 255)  # BGR yellow


def draw_overlay(frame, tracks, result, line_y: int, roi: tuple[int, int, int, int], counted_ids=None):
    counted_ids = counted_ids if counted_ids is not None else set()
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 196, 248), 2)
    cv2.line(frame, (0, line_y), (frame.shape[1] - 1, line_y), (245, 200, 100), 2)
    cv2.putText(frame, "ROI", (x1 + 4, min(frame.shape[0] - 5, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, .55, (90, 196, 248), 2)
    for track_id, track in tracks.items():
        bx, by, bw, bh = track.box
        color = CROSSED_COLOR if track_id in counted_ids else UNCROSSED_COLOR
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
        title = f"#{track_id} {VEHICLE_LABELS[track.class_id]} {track.confidence:.2f}"
        cv2.putText(frame, title, (bx, max(18, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2)
        cv2.circle(frame, track.center, 3, color, -1)
    h, w = frame.shape[:2]
    layer = frame.copy()
    cv2.rectangle(layer, (0, 0), (min(w, 445), min(h, 126)), (20, 25, 37), -1)
    cv2.addWeighted(layer, .72, frame, .28, 0, frame)
    lines = [
        f"In ROI: {result['occupancy']}  |  Line crossings: {result['total_line_crossings']}",
        f"Traffic: {'CONGESTED' if result['congested'] else 'monitoring'}",
        f"Time: {result['timestamp_seconds']:.1f}s | Threshold: {result['threshold']}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, 32 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, .54, (255, 255, 255), 2)
    return frame
