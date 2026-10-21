"""CLI alternative to the web interface: python -m src.components.main --help"""
import argparse
import json
from pathlib import Path

from src.components.vehicle_counting import TrafficConfig
from src.pipeline.predict_pipeline import ProcessingOptions, process_video


def main():
    parser = argparse.ArgumentParser(description="YOLOv3 traffic counting and congestion alerts")
    parser.add_argument("--input", type=Path, required=True, help="Local video file")
    parser.add_argument("--output", type=Path, default=Path("outputs/cli"))
    parser.add_argument("--road", default="Road Segment A")
    parser.add_argument("--threshold", type=int, default=25)
    parser.add_argument("--duration", type=float, default=180, help="Continuous high occupancy seconds")
    parser.add_argument("--recovery", type=float, default=10)
    parser.add_argument("--line", type=float, default=0.65, help="Counting line height, 0–1")
    parser.add_argument("--roi", type=float, nargs=4, default=(0, 0, 1, 1), metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--live-sms", action="store_true", help="CAUTION: SEND REAL TWILIO SMS")
    args = parser.parse_args()
    traffic = TrafficConfig(occupancy_threshold=args.threshold, persistence_seconds=args.duration,
                            recovery_seconds=args.recovery, line_y_fraction=args.line, roi=tuple(args.roi))
    opts = ProcessingOptions(road_name=args.road, traffic=traffic, confidence=args.confidence, live_sms=args.live_sms)
    result = process_video(args.input, args.output, opts,
                           progress=lambda n, total: print(f"Processed {n}/{total or '?'} frames", flush=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
