from pathlib import Path
import shutil
import os
import cv2

ROOT = Path(__file__).resolve().parent
PORTFOLIO = ROOT / "portfolio"
ASSETS = ROOT / "docs" / "assets"
README = ROOT / "README.md"

ASSETS.mkdir(parents=True, exist_ok=True)

files = [
    "roadwatch_demo.gif",
    "annotated_frame.jpg",
    "congestion_event.jpg",
    "twilio_alert_preview.png",
]

for filename in files:
    source = PORTFOLIO / filename

    if not source.is_file():
        raise SystemExit(f"Missing portfolio asset: {source}")

    shutil.copy2(source, ASSETS / filename)
    print(f"Copied: {filename}")

# Find the original input video, if still present locally.
video_name = "14828714_1080_1920_30fps.mp4"
input_video = None

for folder, dirs, filenames in os.walk(ROOT):
    dirs[:] = [
        d for d in dirs
        if d not in {".git", ".venv", "node_modules", "portfolio"}
    ]
    if video_name in filenames:
        input_video = Path(folder) / video_name
        break

if input_video:
    cap = cv2.VideoCapture(str(input_video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    if ok:
        cv2.imwrite(str(ASSETS / "input_frame.jpg"), frame)
        print("Created: input_frame.jpg")
    cap.release()

input_preview = (
    "![Original traffic footage](docs/assets/input_frame.jpg)"
    if (ASSETS / "input_frame.jpg").exists()
    else f"`{video_name}` — original traffic footage"
)

section = f"""
<!-- ROADWATCH_SHOWCASE_START -->

## Demo — Input to Output

RoadWatch processes recorded traffic footage using YOLOv3 and OpenCV.
It detects, classifies, and tracks vehicles, counts unique line crossings,
and monitors sustained road occupancy to identify potential congestion.

### Input and Annotated Output

| Original traffic footage | Processed vehicle detection |
| --- | --- |
| {input_preview} | ![Annotated traffic footage](docs/assets/annotated_frame.jpg) |

### Live Vehicle Detection and Counting

The following **10-second GIF** demonstrates annotated video output,
class-aware bounding boxes, vehicle tracking, and traffic counters.

![RoadWatch 10-second demonstration](docs/assets/roadwatch_demo.gif)

**Vehicle colors:** Blue = Car, Purple = Motorcycle,
Orange = Bus, Green = Truck, Yellow = Counted after line crossing.

### Congestion Detection

RoadWatch monitors the number of vehicles simultaneously present
inside a configured region of interest (ROI). If the occupancy
threshold remains exceeded for the configured duration,
a potential congestion event is recorded.

![Congestion event evidence](docs/assets/congestion_event.jpg)

### Automated Twilio SMS Alert

When a congestion event is detected, RoadWatch generates an
SMS notification containing the road name, vehicle occupancy,
threshold, and video timestamp.

![Twilio SMS alert preview](docs/assets/twilio_alert_preview.png)

**Integration status:** The application successfully submitted an
SMS request to Twilio. Actual carrier delivery was blocked by
US A2P 10DLC registration requirements (Twilio error 30034).
The image above is an illustrative alert preview, not proof
of delivery to a mobile device.

### Technical Details

- **Detection:** YOLOv3 with OpenCV DNN
- **Tracking:** Vehicle identity association and class-aware tracking
- **Counting:** Unique tracked vehicles crossing a virtual line
- **Congestion:** Sustained occupancy within a configurable ROI
- **Backend:** FastAPI
- **Notifications:** Twilio SMS integration
- **Evidence:** Annotated video, event screenshots, CSV, and JSON reports

<!-- ROADWATCH_SHOWCASE_END -->
"""

start_marker = "<!-- ROADWATCH_SHOWCASE_START -->"
end_marker = "<!-- ROADWATCH_SHOWCASE_END -->"

existing = README.read_text(encoding="utf-8") if README.exists() else "# RoadWatch\n"

if start_marker in existing and end_marker in existing:
    start = existing.index(start_marker)
    end = existing.index(end_marker) + len(end_marker)
    updated = existing[:start] + section.strip() + existing[end:]
else:
    lines = existing.splitlines(keepends=True)
    if lines and lines[0].startswith("# "):
        updated = lines[0] + "\n" + section.strip() + "\n\n" + "".join(lines[1:])
    else:
        updated = section.strip() + "\n\n" + existing

if (ROOT / "DEEPDIVE.md").is_file() and "](DEEPDIVE.md)" not in updated:
    updated += "\n\nFor the detailed system architecture and implementation, see [DEEPDIVE.md](DEEPDIVE.md).\n"

README.write_text(updated, encoding="utf-8")

print("\nREADME showcase updated successfully.")
print(f"Assets: {ASSETS}")
print("\nAsset sizes:")
for path in sorted(ASSETS.iterdir()):
    if path.is_file():
        print(f"  {path.name}: {path.stat().st_size / (1024 * 1024):.2f} MB")
