"""Main end-to-end recorded-video processing pipeline; callable from CLI and web."""
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from src.alerts import AlertDispatcher
from src.components.bounding_boxes import draw_overlay
from src.components.vehicle_counting import TrafficConfig, TrafficMonitor
from src.components.vehicle_detection import YoloVehicleDetector
from src.components.vehicle_tracking import VehicleTracker


@dataclass(frozen=True)
class ProcessingOptions:
    road_name: str = "Road Segment A"
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    confidence: float = 0.5
    max_track_distance: float = 90
    max_missing_frames: int = 10
    live_sms: bool = False

    def __post_init__(self):
        if not self.road_name.strip() or len(self.road_name) > 80:
            raise ValueError("Road label must have 1-80 characters")
        if not 0 < self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.max_track_distance <= 0 or self.max_missing_frames < 0:
            raise ValueError("Tracker limits must be nonnegative")


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_browser_video(raw: Path, output: Path) -> str:
    """Prefer H.264 for browser playback, retain mp4v if ffmpeg unavailable."""
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
                check=True, timeout=1800, capture_output=True,
            )
            raw.unlink(missing_ok=True)
            return "annotated.mp4"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            output.unlink(missing_ok=True)
    raw.rename(output)
    return "annotated.mp4"  # May require downloading if browser lacks mp4v decoder.


def process_video(input_path: Path, output_dir: Path, options: ProcessingOptions,
                  *, detector=None, dispatcher: AlertDispatcher | None = None,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    """Process each decoded frame, preserving video-time semantics across slow inference.

    `detector` and `dispatcher` can be injected for deterministic offline tests.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if dispatcher is None:
        dispatcher = AlertDispatcher(live=options.live_sms)
    if detector is None:
        detector = YoloVehicleDetector(confidence=options.confidence)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open the selected video. Try MP4 (H.264) or AVI.")
    writer = None
    raw_video = output_dir / "annotated_raw.mp4"
    events = []
    samples = []
    last_result = None
    frame_index = 0
    last_saved_second = -1
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps < 0.1 or fps > 240:
        capture.release()
        raise ValueError("Could not determine reliable video FPS (needs 0.1–240 FPS)")
    monitor = TrafficMonitor(options.traffic)
    tracker = VehicleTracker(options.max_track_distance, options.max_missing_frames)
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Could not initialize OpenCV video writer")
            video_seconds = frame_index / fps
            tracks = tracker.update(detector.detect(frame))
            result = monitor.update(video_seconds, tracks, frame.shape)
            line_y, roi = monitor.geometry(frame.shape)
            draw_overlay(frame, tracks, result, line_y, roi)
            if result["event"] is not None:
                event = {"type": result["event"], "timestamp_seconds": result["timestamp_seconds"],
                         "occupancy": result["occupancy"], "total_line_crossings": result["total_line_crossings"]}
                event["alert"] = dispatcher.on_event(options.road_name, result, options.traffic.persistence_seconds)
                screenshot = f"event_{len(events)+1:03d}.jpg"
                if not cv2.imwrite(str(output_dir / screenshot), frame):
                    raise RuntimeError("Failed to write congestion event screenshot")
                event["screenshot"] = screenshot
                events.append(event)
            current_second = int(video_seconds)
            if current_second > last_saved_second or result["event"]:
                samples.append({k: result[k] for k in (
                    "timestamp_seconds", "occupancy", "total_line_crossings", "congested", "above_threshold_seconds")})
                last_saved_second = current_second
            writer.write(frame)
            if frame_index == 0 or frame_index % 20 == 0:
                cv2.imwrite(str(output_dir / "preview.jpg"), frame)
            frame_index += 1
            last_result = result
            if progress is not None and (frame_index == 1 or frame_index % 20 == 0):
                progress(frame_index, frame_count)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frame_index == 0:
        raw_video.unlink(missing_ok=True)
        raise ValueError("Selected file contains no readable video frames")
    if progress:
        progress(frame_index, frame_count)
    # Save CSV, events and preview alerts independently of video-transcoding support.
    with (output_dir / "counts.csv").open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=["timestamp_seconds", "occupancy", "total_line_crossings", "congested", "above_threshold_seconds"])
        csv_writer.writeheader()
        csv_writer.writerows(samples)
    _write_json(output_dir / "events.json", events)
    _write_json(output_dir / "alerts.json", dispatcher.alerts)
    video_name = _make_browser_video(raw_video, output_dir / "annotated.mp4")
    summary = {
        "road_name": options.road_name,
        "duration_seconds": round(frame_index / fps, 3),
        "fps": fps,
        "processed_frames": frame_index,
        "final_occupancy": last_result["occupancy"],
        "cumulative_crossings": last_result["total_line_crossings"],
        "flow_by_class": last_result["flow_by_class"],
        "last_occupancy_by_class": last_result["occupancy_by_class"],
        "congestion_events": sum(event["type"] == "CONGESTION_STARTED" for event in events),
        "sms_mode": "LIVE" if options.live_sms else "DRY_RUN",
        "alerts": dispatcher.alerts,
        "events": events,
        "video": video_name,
        "congestion_config": {
            "occupancy_threshold": options.traffic.occupancy_threshold,
            "persistence_seconds": options.traffic.persistence_seconds,
            "recovery_seconds": options.traffic.recovery_seconds,
            "line_y_fraction": options.traffic.line_y_fraction,
            "roi": options.traffic.roi,
        },
        "notes": "Congestion is inferred from sustained ROI occupancy, not measured vehicle speed.",
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
