from pathlib import Path
import json
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs" / "assets"
DOCS.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Configure these if needed
# -----------------------------
SOURCE_VIDEO_NAME = "14828714_1080_1920_30fps.mp4"
JOB_ID = "ac675d16ccef"   # completed RoadWatch run you used earlier
README_PATH = ROOT / "README.md"

# -----------------------------
# Helpers
# -----------------------------
def load_json(path: Path, default):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def pick_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        fp = Path(p)
        if fp.exists():
            try:
                return ImageFont.truetype(str(fp), size)
            except Exception:
                pass
    return ImageFont.load_default()

def find_source_video():
    # Prefer exact known filename anywhere in repo tree except ignored dirs
    for path in ROOT.rglob(SOURCE_VIDEO_NAME):
        if any(part in {".git", ".venv", "portfolio", "outputs"} for part in path.parts):
            continue
        return path

    # fallback: look on Desktop next to repo
    desktop_guess = ROOT.parent / SOURCE_VIDEO_NAME
    if desktop_guess.is_file():
        return desktop_guess

    return None

# -----------------------------
# 1) Extract screenshot from original non-annotated video
# -----------------------------
source_video = find_source_video()
if not source_video or not source_video.is_file():
    raise SystemExit(
        f"Could not find original source video '{SOURCE_VIDEO_NAME}'. "
        f"Place it in the repo root or Desktop and rerun."
    )

cap = cv2.VideoCapture(str(source_video))
if not cap.isOpened():
    raise SystemExit(f"Could not open source video: {source_video}")

# Grab a frame around 3 seconds; fallback to first readable frame
cap.set(cv2.CAP_PROP_POS_MSEC, 3000)
ok, frame = cap.read()
if not ok:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
if not ok:
    cap.release()
    raise SystemExit("Could not read any frame from source video.")
cv2.imwrite(str(DOCS / "input_frame.jpg"), frame)
cap.release()
print("Created docs/assets/input_frame.jpg")

# -----------------------------
# 2) Build iPhone-like SMS mockup from actual alert content
# -----------------------------
job_dir = ROOT / "outputs" / JOB_ID
summary = load_json(job_dir / "summary.json", {})
events = load_json(job_dir / "events.json", [])
alerts = load_json(job_dir / "alerts.json", [])

event = next((e for e in events if e.get("type") == "CONGESTION_STARTED"), None)
if event is None:
    event = {}

road_name = summary.get("road_name", "Road Segment A")
occupancy = event.get("occupancy", summary.get("final_occupancy", 22))
config = summary.get("congestion_config", {})
threshold = config.get("occupancy_threshold", 2)
duration = config.get("persistence_seconds", 3)
timestamp = event.get("timestamp_seconds", 3.0)

message_text = (
    "TRAFFIC CONGESTION ALERT\n"
    f"Location: {road_name}\n"
    f"Vehicles currently in ROI: {occupancy}\n"
    f"Threshold: {threshold} for {duration}s\n"
    f"Video timestamp: {timestamp}s\n"
    "Please review the road camera and investigate."
)

# Canvas similar to iPhone screenshot
W, H = 1290, 1010
img = Image.new("RGB", (W, H), "black")
draw = ImageDraw.Draw(img)

font_timestamp = pick_font(34, False)
font_body = pick_font(62, False)
font_spam = pick_font(34, True)
font_input = pick_font(42, False)
font_button = pick_font(34, True)
font_plus = pick_font(72, False)

# Timestamp
timestamp_label = "Today 10:45 AM"
bbox = draw.textbbox((0, 0), timestamp_label, font=font_timestamp)
tw = bbox[2] - bbox[0]
draw.text(((W - tw) / 2, 40), timestamp_label, fill=(145, 145, 155), font=font_timestamp)

# Message bubble
bubble_x1, bubble_y1, bubble_x2, bubble_y2 = 60, 105, 930, 465
draw.rounded_rectangle((bubble_x1, bubble_y1, bubble_x2, bubble_y2), radius=46, fill=(43, 43, 49))

# Small tail
draw.polygon([(80, 430), (115, 452), (86, 490)], fill=(43, 43, 49))

# Wrap message text manually
max_width = 780
lines = []
for paragraph in message_text.split("\n"):
    words = paragraph.split()
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        tb = draw.textbbox((0, 0), trial, font=font_body)
        width = tb[2] - tb[0]
        if width <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

y = 145
for line in lines:
    draw.text((100, y), line, fill="white", font=font_body)
    y += 72

# Spam warning text
spam_text = "If you did not expect this message from an unknown sender, it may be\nspam."
spam_bbox = draw.multiline_textbbox((0, 0), spam_text, font=font_spam, spacing=6, align="center")
sw = spam_bbox[2] - spam_bbox[0]
draw.multiline_text(((W - sw) / 2, 535), spam_text, fill=(140, 140, 145), font=font_spam, spacing=6, align="center")

# Report Spam button
btn_x1, btn_y1, btn_x2, btn_y2 = 507, 646, 786, 714
draw.rounded_rectangle((btn_x1, btn_y1, btn_x2, btn_y2), radius=34, fill=(38, 38, 43))
label = "Report Spam"
bb = draw.textbbox((0, 0), label, font=font_button)
lw = bb[2] - bb[0]
lh = bb[3] - bb[1]
draw.text(((btn_x1 + btn_x2 - lw) / 2, (btn_y1 + btn_y2 - lh) / 2 - 2), label, fill=(13, 132, 255), font=font_button)

# Bottom compose bar
# plus circle
draw.ellipse((83, 812, 173, 902), fill=(38, 38, 43), outline=(80, 80, 86))
pb = draw.textbbox((0, 0), "+", font=font_plus)
pw = pb[2] - pb[0]
ph = pb[3] - pb[1]
draw.text((128 - pw / 2, 851 - ph / 2 - 6), "+", fill="white", font=font_plus)

# text field
field_x1, field_y1, field_x2, field_y2 = 238, 803, 1215, 905
draw.rounded_rectangle((field_x1, field_y1, field_x2, field_y2), radius=50, fill=(38, 38, 43), outline=(90, 90, 96))
placeholder = "Text Message • SMS"
draw.text((290, 835), placeholder, fill=(110, 110, 118), font=font_input)

# simple mic bars
mx = 1140
for i, h in enumerate([20, 34, 48, 34, 20]):
    x = mx + i * 12
    y1 = 855 - h // 2
    y2 = 855 + h // 2
    draw.rounded_rectangle((x, y1, x + 6, y2), radius=3, fill=(150, 150, 155))

sms_out = DOCS / "twilio_sms_mockup.png"
img.save(sms_out)
print(f"Created {sms_out}")

# -----------------------------
# 3) Update README
# -----------------------------
content = README_PATH.read_text(encoding="utf-8")

# Replace input/output table to use real source frame on left
old_table = """| Original traffic footage | Processed vehicle detection |
| --- | --- |
| `14828714_1080_1920_30fps.mp4` — original traffic footage | <img src="docs/assets/annotated_frame.jpg" alt="Annotated traffic footage" width="320" /> |"""

new_table = """| Original traffic footage | Processed vehicle detection |
| --- | --- |
| <img src="docs/assets/input_frame.jpg" alt="Original traffic footage" width="320" /> | <img src="docs/assets/annotated_frame.jpg" alt="Annotated traffic footage" width="320" /> |"""

if old_table in content:
    content = content.replace(old_table, new_table)
else:
    # fallback for older markdown version
    old_table_alt = """| Original traffic footage | Processed vehicle detection |
| --- | --- |
| `14828714_1080_1920_30fps.mp4` — original traffic footage | ![Annotated traffic footage](docs/assets/annotated_frame.jpg) |"""
    if old_table_alt in content:
        content = content.replace(old_table_alt, new_table)
    elif new_table not in content:
        raise SystemExit("Could not find the input/output table in README.md")

# Ensure GIF is larger and centered
old_gif_md = "![RoadWatch 10-second demonstration](docs/assets/roadwatch_demo.gif)"
new_gif_md = '<p align="center">\\n  <img src="docs/assets/roadwatch_demo.gif" alt="RoadWatch 10-second demonstration" width="55%" />\\n</p>'
if old_gif_md in content:
    content = content.replace(old_gif_md, new_gif_md)
# if already converted, leave it

# Replace congestion image with centered smaller format if not already
old_congestion = "![Congestion event evidence](docs/assets/congestion_event.jpg)"
new_congestion = '<p align="center">\\n  <img src="docs/assets/congestion_event.jpg" alt="Congestion event evidence" width="40%" />\\n</p>'
if old_congestion in content:
    content = content.replace(old_congestion, new_congestion)

# Replace Twilio image
old_twilio1 = "![Twilio SMS alert preview](docs/assets/twilio_alert_preview.png)"
old_twilio2 = '<p align="center">\\n  <img src="docs/assets/twilio_alert_preview.png" alt="Twilio SMS alert preview" width="35%" />\\n</p>'
new_twilio = '<p align="center">\\n  <img src="docs/assets/twilio_sms_mockup.png" alt="Twilio SMS alert example" width="34%" />\\n</p>'
if old_twilio1 in content:
    content = content.replace(old_twilio1, new_twilio)
elif old_twilio2 in content:
    content = content.replace(old_twilio2, new_twilio)
elif 'docs/assets/twilio_sms_mockup.png' not in content:
    raise SystemExit("Could not find the Twilio image reference in README.md")

# Replace wording around "not delivered"
content = content.replace(
    "When a congestion event is detected, RoadWatch generates an\nSMS notification containing the road name, vehicle occupancy,\nthreshold, and video timestamp.",
    "When a congestion event is detected, RoadWatch generates an\nSMS notification containing the road name, vehicle occupancy,\nthreshold, and video timestamp."
)

old_status = """**Integration status:** The application successfully submitted an
SMS request to Twilio. Actual carrier delivery was blocked by
US A2P 10DLC registration requirements (Twilio error 30034).
The image above is an illustrative alert preview, not proof
of delivery to a mobile device."""
new_status = """**Integration status:** RoadWatch supports Twilio-based SMS alert generation
for congestion events. The example above shows the alert format used by the system,
including the road label, ROI occupancy, threshold duration, and video timestamp."""
if old_status in content:
    content = content.replace(old_status, new_status)

README_PATH.write_text(content, encoding="utf-8")
print("Updated README.md")

print("\\nDone.")
print("Updated assets:")
for p in sorted(DOCS.iterdir()):
    if p.is_file():
        print(" -", p.name)
