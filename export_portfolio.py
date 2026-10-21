import json
import shutil
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
JOB = ROOT / "outputs" / "ac675d16ccef"
OUT = ROOT / "portfolio"
OUT.mkdir(exist_ok=True)

def load_json(name, default):
    path = JOB / name
    return json.loads(path.read_text()) if path.exists() else default

def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()

summary = load_json("summary.json", {})
events = load_json("events.json", [])
alerts = load_json("alerts.json", [])

congestion = next(
    (e for e in events if e.get("type") == "CONGESTION_STARTED"),
    None,
)

if not congestion:
    print("Warning: No CONGESTION_STARTED event in this job.")

# 1. Preserve actual evidence, without modifying originals.
for name in ("summary.json", "events.json", "alerts.json", "counts.csv"):
    source = JOB / name
    if source.exists():
        shutil.copy2(source, OUT / name)

if congestion:
    screenshot = congestion.get("screenshot")
    if screenshot and (JOB / screenshot).is_file():
        shutil.copy2(JOB / screenshot, OUT / "congestion_event.jpg")

# 2. Read the existing annotated video.
video_name = summary.get("video", "annotated.mp4")
video_path = JOB / video_name

if not video_path.is_file():
    raise SystemExit(f"Annotated video not found: {video_path}")

capture = cv2.VideoCapture(str(video_path))
if not capture.isOpened():
    raise SystemExit("Could not open annotated video.")

fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

if fps <= 0:
    raise SystemExit("Invalid video FPS.")

duration = frame_count / fps

# Start near the congestion event, leaving room to show
# vehicles approaching and crossing the counting line.
event_time = float(congestion.get("timestamp_seconds", 3)) if congestion else 3
start = max(0.0, min(event_time - 2.0, max(0.0, duration - 10)))
end = min(duration, start + 10)

# 3. Create a 10-second annotated GIF at 5 FPS.
gif_frames = []
gif_fps = 5
sample_count = max(1, int((end - start) * gif_fps))

for i in range(sample_count):
    timestamp = start + i / gif_fps
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = capture.read()
    if not ok:
        continue

    h, w = frame.shape[:2]
    new_w = min(w, 640)
    new_h = round(h * new_w / w)
    frame = cv2.resize(frame, (new_w, new_h))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gif_frames.append(Image.fromarray(rgb))

if not gif_frames:
    capture.release()
    raise SystemExit("No readable frames found for GIF.")

gif_frames[0].save(
    OUT / "roadwatch_demo.gif",
    save_all=True,
    append_images=gif_frames[1:],
    duration=200,
    loop=0,
    optimize=True,
)

# 4. Save a full-resolution annotated frame.
frame_time = min(event_time, max(0, duration - 1 / fps))
capture.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
ok, frame = capture.read()

if ok:
    cv2.imwrite(str(OUT / "annotated_frame.jpg"), frame)

capture.release()

# 5. Create a clearly labeled alert preview.
occupancy = congestion.get("occupancy", "N/A") if congestion else "N/A"
timestamp = congestion.get("timestamp_seconds", "N/A") if congestion else "N/A"
threshold = summary.get("congestion_config", {}).get(
    "occupancy_threshold", "N/A"
)
persistence = summary.get("congestion_config", {}).get(
    "persistence_seconds", "N/A"
)
road = summary.get("road_name", "Road Segment A")

lines = [
    "TRAFFIC CONGESTION ALERT",
    "",
    f"Location: {road}",
    f"Vehicles currently in ROI: {occupancy}",
    f"Threshold: {threshold} for {persistence}s",
    f"Video timestamp: {timestamp}s",
    "",
    "Please review the road camera",
    "and investigate.",
]

canvas = Image.new("RGB", (900, 670), "#f4f6fa")
draw = ImageDraw.Draw(canvas)

draw.rounded_rectangle(
    (55, 55, 845, 610),
    radius=25,
    fill="white",
    outline="#dce2eb",
    width=2,
)

draw.text(
    (90, 90),
    "RoadWatch",
    font=font(38, True),
    fill="#152238",
)
draw.text(
    (90, 145),
    "SMS ALERT PREVIEW - NOT DELIVERED",
    font=font(19, True),
    fill="#b45309",
)

y = 215
for index, line in enumerate(lines):
    draw.text(
        (90, y),
        line,
        font=font(25, index == 0),
        fill="#172033",
    )
    y += 43

canvas.save(OUT / "twilio_alert_preview.png")

# 6. Create a compact README-ready Markdown snippet.
readme = """## RoadWatch Demo

### Annotated vehicle tracking and counting

![RoadWatch annotated demo](roadwatch_demo.gif)

### Vehicle detection and tracking

![Annotated video frame](annotated_frame.jpg)

### Congestion detection evidence

![Congestion event](congestion_event.jpg)

### Twilio SMS alert preview

![Twilio alert preview](twilio_alert_preview.png)

The application successfully submitted a congestion alert
to Twilio. Carrier delivery was blocked with error 30034
(unregistered US A2P 10DLC sender). The image above is
an illustrative alert preview, not a delivered SMS screenshot.
"""
(OUT / "README_SNIPPET.md").write_text(readme)

print("\nPORTFOLIO EXPORT COMPLETE")
print(f"Source job: {JOB}")
print(f"Output folder: {OUT}")
print(f"GIF duration: {end - start:.1f} seconds")
print("\nGenerated files:")
for path in sorted(OUT.iterdir()):
    print(f"  {path.name} ({path.stat().st_size / 1024:.1f} KB)")
