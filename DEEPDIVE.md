# RoadWatch — Vehicle Counting & Sustained Traffic Congestion Detection

*From recorded traffic footage to vehicle tracks, two independent traffic metrics, time-based congestion events, and an auditable SMS alert workflow.*

**How to explore:** Follow a single traffic video through the system, or jump to the engineering question you care about. The [README](README.md) is the quick-start and visual showcase; this document explains the implementation, data flow, tradeoffs, failure cases, and evidence behind it.

## Explore the implementation

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Technical Deep Dive](#3-technical-deep-dive)
   - [3.1 Input, Configuration, and Job Creation](#31-input-configuration-and-job-creation)
   - [3.2 YOLOv3 Inference and Detection Filtering](#32-yolov3-inference-and-detection-filtering)
   - [3.3 Vehicle Tracking and Identity Continuity](#33-vehicle-tracking-and-identity-continuity)
   - [3.4 Unique Virtual-Line Crossings](#34-unique-virtual-line-crossings)
   - [3.5 ROI Occupancy and Congestion State](#35-roi-occupancy-and-congestion-state)
   - [3.6 Video-Time Semantics](#36-video-time-semantics)
   - [3.7 Twilio Alert Lifecycle](#37-twilio-alert-lifecycle)
   - [3.8 Processing Pipeline and Evidence](#38-processing-pipeline-and-evidence)
   - [3.9 Browser Dashboard and Live Preview](#39-browser-dashboard-and-live-preview)
   - [3.10 Validation, Failure Handling, and Security Boundaries](#310-validation-failure-handling-and-security-boundaries)
   - [3.11 Testing and Verification](#311-testing-and-verification)
4. [Engineering Decisions and Tradeoffs](#4-engineering-decisions-and-tradeoffs)
5. [Limitations and Future Improvements](#5-limitations-and-future-improvements)
6. [Codebase Map and Reproduction](#6-codebase-map-and-reproduction)
7. [Closing Perspective](#7-closing-perspective)

---

## 1. Introduction

A traffic camera can produce thousands of detections over a short video, but detecting objects is only the beginning of useful traffic analytics. A car visible in thirty consecutive frames is still one car; a motorcycle passing through a counting line should contribute to throughput only once; and a road with a high cumulative vehicle count is not necessarily congested. A system that treats all three observations as the same quantity will produce misleading results, even if its object detector is accurate.

I built RoadWatch around that distinction. The project extends a basic YOLOv3/OpenCV vehicle-counting prototype into a local application that accepts a recorded road video, detects selected vehicle classes, associates detections across frames, measures traffic movement and occupancy independently, evaluates a duration-based congestion condition, and records the resulting evidence. The workflow is accessible through a FastAPI website or a command-line interface. Twilio SMS is available as an explicitly enabled notification path, with a preview-only mode as the default.

The engineering goal was not to train a new object detector or claim city-scale traffic-management accuracy. It was to connect computer-vision output to **interpretable, stateful application behavior**: a count with a defined meaning, an alert with a defined trigger, and artifacts that make the result inspectable after the run.

### The two measurements that define the system

| Measurement | Question answered | State maintained | Reset behavior |
| --- | --- | --- | --- |
| **Cumulative line crossings (flow)** | How many distinct tracked vehicles crossed the configured line? | A set of already-counted track IDs and counts by class | Accumulates across the video; a surviving track ID is counted once |
| **Concurrent ROI occupancy** | How many currently visible tracked vehicles are inside the monitored region? | Per-frame visible track counts, including a class breakdown | Recomputed for every decoded frame |

A sustained occupancy threshold is then used as a **potential congestion indicator**. It does not measure speed, queue length, travel time, or whether a road is physically stopped. This interpretation is deliberate: a system can observe a crowded scene and ask for human review without claiming more than the evidence supports.

### Scope of the implemented application

RoadWatch supports the four COCO vehicle categories used by this project: **car, motorcycle, bus, and truck**. The pretrained YOLOv3 model does **not** have a dedicated auto-rickshaw category, and bicycles are not in the application’s selected class list. These are model/data boundaries, not omissions in the UI. The code uses pretrained Darknet YOLOv3 for inference; no YOLO training or fine-tuning occurs in this repository.

The application is designed for **one prerecorded video at a time on a local machine**. Its strengths are an inspectable end-to-end path, configurable geometry and timing, deterministic reporting, and testable behavior around alerts. Its limits are the ones expected of a lightweight single-camera prototype: detector misses, identity switches during occlusion, perspective sensitivity, and uncertain congestion inference without motion or speed measurements.

---

## 2. System Architecture

The most useful way to read the architecture is to follow a single uploaded video. The browser validates the request and creates a job; the processing pipeline opens the video and coordinates the detector, tracker, traffic monitor, drawing layer, alert dispatcher, and artifact writers. The dashboard reads progress and preview images during processing and displays the completed evidence afterward. The CLI enters the **same** processing function without the web layer.

### End-to-end execution map

```text
                               USER / LOCAL OPERATOR
                                        │
                       ┌────────────────┴────────────────┐
                       │                                 │
                 FASTAPI WEB UI                       CLI COMMAND
                 POST /api/jobs                    python -m ...main
                       │                                 │
                       └──────────────┬──────────────────┘
                                      ▼
                            PROCESSING OPTIONS
                   ROI / counting line / threshold /
                   persistence / confidence / SMS mode
                                      │
                                      ▼
                           process_video(...)
                                      │
                             OpenCV video decoder
                             frame index / source FPS
                                      │
                                      ▼
                          YOLOv3 VEHICLE DETECTOR
                       blob → network → filter → NMS
                                      │
                                      ▼
                             VEHICLE TRACKER
                      deduplicate → associate → IDs
                                      │
                         ┌────────────┴─────────────┐
                         ▼                          ▼
                    LINE CROSSINGS               ROI OCCUPANCY
                    unique counted IDs           visible track count
                    + per-class flow             + per-class occupancy
                         │                          │
                         │                          ▼
                         │                  CONGESTION MONITOR
                         │                  threshold + duration
                         │                          │
                         │                   STARTED / CLEARED
                         │                          │
                         │                   ALERT DISPATCHER
                         │                   DRY_RUN or LIVE
                         │                   Twilio API attempt
                         │                          │
                         └────────────┬─────────────┘
                                      ▼
                            OVERLAY + ARTIFACTS
                   annotated video / preview JPEG /
                   event JPEGs / CSV / JSON summaries
                                      │
                                      ▼
                           FASTAPI RESULT ROUTES
                      polling / downloads / audit view
```

The two analytics branches are intentionally independent. An ID crossing the line affects throughput; a visible ID inside the ROI affects occupancy. Only the occupancy branch can start a congestion event. A line crossing alone does not send an SMS.

### Module responsibilities

| Responsibility | Implementation | Why it is separate |
| --- | --- | --- |
| Network loading | [`src/components/video_network_setup.py`](src/components/video_network_setup.py) | Resolves local Darknet assets and initializes OpenCV DNN |
| Detection | [`src/components/vehicle_detection.py`](src/components/vehicle_detection.py) | Turns a video frame into candidate vehicle detections |
| Shared representations | [`src/components/models.py`](src/components/models.py) | Defines boxes, classes, detection scores, and tracks |
| Association | [`src/components/vehicle_tracking.py`](src/components/vehicle_tracking.py) | Maintains identity across frames and reduces duplicate detections |
| Traffic state | [`src/components/vehicle_counting.py`](src/components/vehicle_counting.py) | Owns the counted IDs, occupancy, and congestion state machine |
| Drawing | [`src/components/bounding_boxes.py`](src/components/bounding_boxes.py) | Renders boxes, class colors, line, ROI, and live metrics |
| Alerting | [`src/alerts.py`](src/alerts.py) | Creates the SMS text and controls dry-run/live attempts |
| Orchestration | [`src/pipeline/predict_pipeline.py`](src/pipeline/predict_pipeline.py) | Coordinates frames, components, files, and summary results |
| Web application | [`src/web/app.py`](src/web/app.py) and [`src/web/static/app.js`](src/web/static/app.js) | Uploads jobs, reports status, serves previews and outputs |
| CLI | [`src/components/main.py`](src/components/main.py) | Runs the same pipeline without a browser |

A noteworthy boundary is the one between **computer vision** and **business interpretation**. YOLOv3 reports detections; the tracker adds identities; the traffic monitor assigns meaning to identity movement and visible density; the alert dispatcher decides whether a congestion transition should cause an outbound action. This separation is what makes the counting and notification behavior independently testable without a 237 MB model file or a live Twilio account.

---

## 3. Technical Deep Dive

To make the mechanisms concrete, imagine uploading a road video and configuring a **2-vehicle occupancy threshold sustained for 3 seconds**, a horizontal counting line at **65% of frame height**, and a full-frame ROI. These values are intentionally suitable for a quick demo, not a recommended real-world congestion policy. The following sections trace what the application does with that request, how it responds as vehicles move, and how the outcome becomes a recorded artifact.

### 3.1 Input, Configuration, and Job Creation

The workflow begins before the first frame is decoded. A user selects a video and configures the road label, occupancy threshold, persistence and recovery durations, counting-line position, ROI coordinates, confidence cutoff, and whether to attempt a real SMS. The FastAPI `POST /api/jobs` route translates those values into a `TrafficConfig` and a `ProcessingOptions` object.

The configuration is represented explicitly rather than distributed as unrelated global variables:

```python
@dataclass(frozen=True)
class TrafficConfig:
    occupancy_threshold: int = 25
    persistence_seconds: float = 180.0
    recovery_seconds: float = 10.0
    line_y_fraction: float = 0.65
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    line_margin_px: int = 4
```

`TrafficConfig.__post_init__` rejects nonpositive occupancy thresholds, negative time durations, invalid ROI rectangles, and counting-line fractions outside `(0, 1)`. `ProcessingOptions` separately validates the road label, confidence, and tracker parameters. This means malformed requests are rejected before the expensive model/video-processing path begins.

The ROI is expressed in **normalized coordinates**, not pixels. For a frame of width `W` and height `H`, `(0.1, 0.2, 0.9, 0.95)` is converted to a rectangle covering the middle 80% of the width and the lower 75% of the height. Normalized geometry lets the same UI configuration be expressed independently of the input video's resolution. Whether that geometry is appropriate for another camera angle is still an operator decision.

#### Upload acceptance and background execution

The web API accepts MP4, AVI, MOV, and MKV filename extensions. The default upload limit is **500 MB**, configurable via `TRAFFIC_MAX_UPLOAD_MB`. Instead of trusting the original filename as an output path, it saves the upload inside a job-specific directory as `input.<extension>`. A UUID-derived 12-character job identifier separates one run’s results from another.

Only one web job can be active in the local process. The API tracks `active_id` behind a `threading.Lock`; a concurrent upload or processing request receives **409 Conflict**. Successful request acceptance returns **202 Accepted** and a job ID. The processing work is then performed through a FastAPI background task; the browser polls a separate status endpoint rather than waiting for a single long HTTP response.

```text
POST /api/jobs
    │
    ├── invalid settings / unsupported extension ──► 422
    ├── oversized upload ───────────────────────────► 413
    ├── another active job ─────────────────────────► 409
    └── accepted ───────────────────────────────────► 202 + job_id
                                                       │
                                                       ▼
                                             uploading → queued
                                                   → processing
                                                   → completed / failed
```

The code checks the filename extension and size at upload time, but does **not** guarantee the bytes form a decodable video before returning 202. A fake `.mp4` may be accepted and then fail when OpenCV opens it. That distinction matters for both API semantics and test expectations: **request accepted** is not **processing succeeded**.

### 3.2 YOLOv3 Inference and Detection Filtering

After the job starts, the processing pipeline initializes `YoloVehicleDetector`. This component loads local `yolov3.cfg`, `yolov3.weights`, and `coco.names` through OpenCV's Darknet loader. The model files are acquired separately by `python -m src.download_models`; the large weights are deliberately excluded from Git.

Each decoded BGR frame is turned into a 416 × 416 network input:

```python
blob = cv2.dnn.blobFromImage(
    frame,
    scalefactor=1 / 255,
    size=(416, 416),
    swapRB=True,
    crop=False,
)
net.setInput(blob)
outputs = net.forward(output_layers)
```

YOLOv3 produces multiple detection candidates, but those raw candidates are not yet application-level vehicle objects. The detector extracts the highest-probability COCO class, keeps only class IDs `2`, `3`, `5`, and `7`, computes confidence, clips the bounding box to the frame, and applies nonmaximum suppression (NMS).

| COCO ID | Accepted label | Initial annotation color |
| ---: | --- | --- |
| `2` | Car | Blue |
| `3` | Motorcycle | Purple |
| `5` | Bus | Orange |
| `7` | Truck | Green |

**Confidence semantics.** The code uses the product of proposal objectness and the selected class probability:

```python
class_id = int(np.argmax(candidate[5:]))
score = float(candidate[4] * candidate[5:][class_id])
if score < self.confidence:
    continue
```

At a confidence threshold of `0.50`, an objectness value of `0.80` and class probability of `0.55` produce `0.44`, so the candidate is filtered out. Lowering the threshold may recover small or partially obscured motorcycles, but can also introduce false positives and extra boxes. It is a detection tradeoff, not a reliable solution to identity tracking problems.

**Nonmaximum suppression.** The detector applies NMS separately for each accepted class. That helps preserve nearby detections of different vehicle classes, but two highly overlapping boxes with **different class labels** can survive this stage. A later tracker-side deduplication pass addresses near-identical boxes across classes; this two-stage behavior explains why overlapping-box cleanup belongs in tracking as well as detection.

The detector returns a small typed representation:

```python
@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]  # x, y, width, height
    class_id: int
    confidence: float
```

Once inference ends, the rest of the pipeline no longer needs YOLO-specific output tensors. It operates on detections, tracks, and traffic state.

### 3.3 Vehicle Tracking and Identity Continuity

Object detection answers “what is visible in this frame?” Vehicle counting requires the harder question: **“is this the same vehicle I saw a moment ago?”** Without identity association, a single car appearing in dozens of frames could inflate the count dozens of times.

`VehicleTracker` maintains monotonically assigned IDs and associates current detections with existing tracks. It is a lightweight, greedy tracker rather than a re-identification model such as DeepSORT or a full ByteTrack implementation. The later iteration of the tracker introduces three practical improvements to address issues observed in road footage: duplicate suppression, short-gap motion prediction, and tolerance for brief class instability.

#### A. Cross-class duplicate suppression

Before association, detections are sorted by confidence. A candidate is removed if its bounding-box intersection-over-union (IoU) with an already-kept box reaches the configured overlap threshold, currently `0.65`.

```python
def deduplicate_detections(detections, overlap=0.65):
    kept = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(box_iou(det.box, other.box) < overlap for other in kept):
            kept.append(det)
    return kept
```

Unlike the detector's per-class NMS, this pass can suppress two near-identical detections even if one was called a car and the other a truck. A high overlap threshold is used to reduce the risk of deleting distinct nearby vehicles. It is still a heuristic: heavy occlusion and truly overlapping road users can make duplicate removal ambiguous.

#### B. Motion-assisted association

The tracker stores a simple velocity estimate for each ID. When a vehicle reappears after a short detection gap, it compares the new detection to a **predicted** location rather than only the last observed center. Extrapolation is capped at three frames, limiting how far a stale track can project forward.

```python
gap = track.missing + 1
vx, vy = self._velocity.get(track_id, (0.0, 0.0))
predicted = (
    track.x + vx * min(gap, 3),
    track.y + vy * min(gap, 3),
)
```

The default matching gate is based on `max_track_distance = 90` pixels; motorcycle tracks receive a `1.35` multiplier, with a modest additional allowance after a few missing frames. Candidate matches are scored using distance, box overlap, and a penalty for class mismatch. Matches are assigned greedily, with each track and each detection used at most once.

This is intentionally bounded. Increasing the maximum distance indiscriminately could make fast motorcycles easier to associate but also cause neighboring vehicles to exchange IDs in dense traffic.

#### C. Stable class decisions

One temporarily incorrect classification should not automatically destroy an existing vehicle identity. A class-mismatched detection is allowed to match only when its box strongly overlaps the existing track (`IoU >= 0.35`). The tracker then maintains a short rolling history of up to five class predictions and uses a majority vote, preserving the established class on ties.

The goal is **stability, not a new classifier**. A sustained incorrect YOLO prediction can still become the track label. In particular, auto-rickshaws may be labeled as trucks because the pretrained COCO taxonomy does not distinguish them.

#### Track lifecycle

```text
New visible detection
        │
        ▼
   Assign new ID
        │
        ▼
Match on next frame? ── YES ──► update box / velocity / class votes
        │                             │
        NO                            └──────────► visible track
        │
        ▼
Increment missing counter
        │
        ├── missing <= max_missing ──► keep internally for re-association
        └── missing > max_missing ───► delete track and motion history
```

An important implementation detail is that `VehicleTracker.update()` returns **visible tracks only**, even though missing tracks may be retained internally. This prevents an old position from being counted as an occupied vehicle while the detector cannot actually see it.

**What an ID switch means:** if a motorcycle approaches the line as `#12`, disappears briefly, and returns as `#27`, the traffic monitor no longer has a continuous crossing history for that physical motorcycle. Tracking reliability therefore directly affects count reliability. Motion prediction and class stabilization reduce this failure mode, but cannot eliminate it in crowded scenes.

### 3.4 Unique Virtual-Line Crossings

Once detections have stable IDs, RoadWatch can measure how many distinct tracked vehicles pass a configured horizontal line. `TrafficMonitor` stores `seen_side` for the last confirmed side of each ID, `counted_ids` for tracks already counted, and `flow_by_class` for class-wise totals.

The line is defined as:

\[
 y_{\text{line}} = H \cdot f_{\text{line}}
\]

Here `H` is frame height and `f_line` is the normalized fraction, defaulting to `0.65`. The code includes a `±4` pixel deadband around the line. Points inside that narrow margin do not establish a new side; this avoids rapid count oscillation when a tracked coordinate jitters near the line.

#### Why bottom-center is used for crossings

An early version compared the object's **bounding-box center** to the line. In traffic footage, the front or wheels of a motorcycle could visibly reach the line while the center of its bounding box was still above it. If the detector then lost the motorcycle, no crossing would be registered.

The updated crossing logic uses the **bottom-center road-contact approximation**, implemented as the vertical bottom edge of the bounding box:

```python
foot_y = track.box[1] + track.box[3]

side = 0
if foot_y < line_y - self.config.line_margin_px:
    side = -1
elif foot_y > line_y + self.config.line_margin_px:
    side = 1
```

This is a more useful crossing anchor for many elevated road-camera views, although it remains an approximation: a detection box does not guarantee an exact physical wheel-contact coordinate.

#### First side change wins

When an ID appears on a nonzero side for the first time, the monitor records its side but **does not count it yet**. When the same ID is later observed on the opposite side, the monitor increments the cumulative total and adds the ID to `counted_ids`.

```python
previous_side = self.seen_side.get(track_id)
if (
    previous_side is not None
    and previous_side != side
    and track_id not in self.counted_ids
):
    self.counted_ids.add(track_id)
    self.flow_by_class[track.class_id] += 1
```

```text
Frame 1      Motorcycle #12 above line      side = -1    count = 0
Frame 2      Motorcycle #12 near deadband   side =  0    count = 0
Frame 3      Motorcycle #12 below line      side = +1    count = 1
Frame 4      Motorcycle #12 crosses back    side = -1    count = 1
```

The rule is **bidirectional**: either sign change can count. The application currently does not report inbound and outbound movement separately. And the ID matters more than the apparent visual path: a newly assigned ID cannot retroactively claim the old ID's crossing history.

#### Visual feedback: class color → yellow

A vehicle's box starts in its class color (blue car, purple motorcycle, orange bus, green truck). After `TrafficMonitor` adds that track ID to `counted_ids`, the drawing layer changes its box and label color to **yellow**. It remains yellow for the lifetime of that tracked ID, providing a direct visual check of whether the counting state machine registered a crossing.

![Annotated RoadWatch vehicle tracks and counting line](docs/assets/annotated_frame.jpg)

*Actual annotated-output frame: class-colored boxes identify detections; yellow indicates that an ID has already been counted. The bounding-box label retains its predicted vehicle class.*

### 3.5 ROI Occupancy and Congestion State

The other branch of the monitor does not care how many vehicles have ever crossed the line. It asks how many are **visible now** inside a configured region of interest (ROI).

For each visible track, the monitor checks whether the **bounding-box center** lies inside the ROI rectangle:

```python
for track_id, track in tracks.items():
    if x1 <= track.x <= x2 and y1 <= track.y <= y2:
        current_by_class[track.class_id] += 1

occupancy = sum(current_by_class.values())
```

Notice the intentional asymmetry: **line crossing uses the bottom edge**, while **occupancy uses the box center**. They measure different relationships to the scene geometry. Changing one should not silently change the other.

#### Congestion is a state transition, not an instantaneous label

A temporary occupancy spike should not automatically generate a persistent congestion alert. `TrafficMonitor` maintains `above_since`, `below_since`, and `congested`. It evaluates the condition on **every processed frame**, even though the exported CSV is sampled less frequently.

The start condition is:

\[
\operatorname{occupancy}(t) \geq N
\quad\text{continuously for}\quad t - t_{\text{above}} \geq P
\]

where `N` is `occupancy_threshold` and `P` is `persistence_seconds`.

For a quick demo with `N = 2` and `P = 3`, two or more detected vehicles must remain in the ROI at every evaluated frame across a three-second interval of recorded video. The **same two IDs do not need to remain**—only the observed concurrent count must stay above the threshold. A dip below the threshold resets the persistence timer.

A separate recovery duration avoids repeatedly starting and clearing an event when the count fluctuates around the threshold. While congested, the system requires occupancy to remain **below** the threshold for `recovery_seconds` before producing `CONGESTION_CLEARED`.

#### State machine

```text
                 occupancy < threshold
                       ┌─────────┐
                       │         │
                       ▼         │
                NOT_CONGESTED ◄──┘
                       │
            occupancy >= threshold
                 start above_since
                       │
            continuous for P seconds?
                       │ YES
                       ▼
               CONGESTION_STARTED ─────► AlertDispatcher
                       │
                       ▼
                   CONGESTED
                       │
             occupancy < threshold
                start below_since
                       │
            continuous for R seconds?
                       │ YES
                       ▼
               CONGESTION_CLEARED
                       │
                       └──────────────► NOT_CONGESTED
```

If the observed count rises again during recovery, `below_since` resets and the existing congestion episode continues. On clear, the monitor does not send an SMS. A later interval of sustained high occupancy can start a new episode.

![Actual RoadWatch congestion-event evidence](docs/assets/congestion_event.jpg)

*Event screenshot written when the monitor transitions into or out of congestion. Its on-frame metrics capture the detection state at that video timestamp.*

**Interpretation boundary:** occupancy is the number of tracked visible vehicles inside a rectangle, not the percentage of road surface occupied and not a measured traffic speed. A busy road with fast-moving traffic may still meet the threshold. This is why RoadWatch records **potential congestion** rather than asserting a verified road blockage.

### 3.6 Video-Time Semantics

The timing choice is subtle but fundamental. YOLOv3 inference can be slow on a laptop, and processing a short video may take much longer than the source recording's duration. If the congestion monitor used wall-clock processing time, the same footage could yield a different alert depending on the computer, model speed, or temporary CPU load.

Instead, the pipeline uses the decoded frame index and the source video's frames per second:

\[
 t_i = \frac{i}{\mathrm{FPS}}
\]

For a 30 FPS video, frame 90 corresponds to **3.0 seconds of video time**, whether decoding and inference reached that frame in 3 seconds or in 90 seconds of real time.

```python
fps = float(capture.get(cv2.CAP_PROP_FPS))
if fps < 0.1 or fps > 240:
    raise ValueError("Could not determine reliable video FPS")

video_seconds = frame_index / fps
result = monitor.update(video_seconds, tracks, frame.shape)
```

This ensures that persistence, recovery, event timestamps, and the SMS cooldown are anchored to the video timeline. `TrafficMonitor` also rejects negative or decreasing timestamps, because a state machine cannot evaluate “continuous for three seconds” reliably if time moves backward.

#### Example timeline

| Video time | ROI occupancy | State |
| ---: | ---: | --- |
| `0.0s` | 4 | Start tracking an above-threshold interval |
| `1.0s` | 5 | Still above threshold; no event yet |
| `2.0s` | 3 | Still above threshold; no event yet |
| `3.0s` | 22 | `CONGESTION_STARTED`, assuming no intervening dip |
| `4.0s` | 1 | Recovery interval begins |
| `6.0s` | 4 | Recovery interval resets if threshold is reached again |

*Illustrative sequence, not a frame-by-frame transcription of a particular video. It assumes `threshold = 2`, `persistence = 3s`, and recovery longer than the temporary dip.*

There is an explicit limitation: the implementation uses container FPS metadata and a frame counter, which best fits **constant-frame-rate recordings**. Variable-frame-rate source videos are not timestamped from each frame's presentation timestamp. Re-encoding such material to constant FPS is preferable when accurate durations matter.

### 3.7 Twilio Alert Lifecycle

The alert subsystem is separate from congestion classification. `TrafficMonitor` emits transitions; `AlertDispatcher` decides which transitions result in message creation and whether a real API call is permitted.

The dispatcher ignores `CONGESTION_CLEARED` for outbound SMS. For `CONGESTION_STARTED`, it formats a message using the road label, current occupancy, configured threshold and persistence duration, and video timestamp:

```text
TRAFFIC CONGESTION ALERT
Location: Road Segment A
Vehicles currently in ROI: 22
Threshold: 2 for 3s
Video timestamp: 3.0s
Please review the road camera and investigate.
```

This message is evidence of the **application's proposed notification content**. It does not, by itself, prove that a phone received it.

#### Preview-first behavior

The default is `live_sms = False`. In that mode, `AlertDispatcher` writes a `preview_only` record and **does not contact Twilio**. A real attempt is enabled only with the explicit web checkbox or CLI `--live-sms` flag and a locally configured `.env` file.

```text
CONGESTION_STARTED
        │
        ▼
    Build message
        │
        ├── DRY_RUN ─────────────► record preview_only
        │
        └── LIVE
             │
             ├── missing credentials ──► reject live configuration
             ├── cooldown/attempt cap ─► suppressed_cooldown_or_limit
             └── permitted attempt ───► Twilio messages.create(...)
                                              │
                                  ┌───────────┴───────────┐
                                  ▼                       ▼
                           accepted_by_api           send_failed
                           + Twilio SID              + generic error
```

Live mode requires a valid-looking account SID, authentication token, and sender/recipient phone numbers in E.164-style format. The dispatcher uses `twilio.rest.Client(...).messages.create(...)` and records the resulting message SID if the API call returns successfully.

#### Why outbound controls are necessary

A congested road can fluctuate around its threshold. Without limits, repeated congestion episodes could result in many SMS messages, unnecessary cost, and noisy notifications. The current dispatcher enforces:

| Control | Implemented behavior |
| --- | --- |
| Default mode | `DRY_RUN`; no Twilio request |
| Trigger | `CONGESTION_STARTED` only |
| Cooldown | 600 **video seconds** between permitted live attempts |
| Maximum attempts | 3 live attempts per video job |
| Error artifact | Generic failure status; no raw exception, token, or provider headers |
| API success label | `accepted_by_api`, **not** `delivered` |

The cooldown and cap are **per processing run**, not a persistent, account-wide rate limiter. Because the clock uses video time, reprocessing the same footage in a new job creates an independent notification budget. A production messaging service would need account-level budgets, delivery callbacks, idempotency keys, and persistent suppression rules.

#### Twilio submission versus handset delivery

A real project test demonstrated this boundary. The RoadWatch application generated the SMS body and Twilio's messaging log showed the message being created, queued, and processed, but carrier delivery failed with **Twilio error `30034`: US A2P 10DLC — Message from an Unregistered Number**. That result confirms that a request reached Twilio for that attempt; it does **not** establish successful receipt on the user's phone. Registration or an appropriately verified sender may resolve the specific restriction, but end-to-end delivery still requires another real delivery test.

The README includes an **illustrative SMS-interface mockup**, not a screenshot of a delivered message. It is useful for demonstrating the message format, but must not be treated as delivery evidence.

### 3.8 Processing Pipeline and Evidence

`process_video()` is the central coordinator. Both the CLI and web API call it with a video path, output directory, and validated `ProcessingOptions`. Its arguments allow the detector and dispatcher to be injected, which supports deterministic tests without invoking the real network or Twilio.

The core frame loop is deliberately straightforward:

```python
while True:
    success, frame = capture.read()
    if not success:
        break

    video_seconds = frame_index / fps
    tracks = tracker.update(detector.detect(frame))
    result = monitor.update(video_seconds, tracks, frame.shape)
    line_y, roi = monitor.geometry(frame.shape)
    draw_overlay(frame, tracks, result, line_y, roi, monitor.counted_ids)

    # On congestion transitions: record an event and its screenshot.
    # On every frame: write annotated video and update progress/preview.
```

A significant consequence of this design is that **all decoded frames participate in detection, tracking, counting, and congestion decisions**. The output CSV does not drive the event detector; it is just a downsampled reporting artifact. That avoids an accidental one-second-only monitor whose behavior would miss shorter changes in occupancy.

#### What the overlay communicates

The annotation layer draws four kinds of information on the processed frame: the ROI rectangle, virtual counting line, per-vehicle boxes and labels, and a top-left metrics panel. Box color represents classification before counting and count status afterward. Labels include the internal track ID, class, and model score. The metrics panel reports current ROI occupancy, cumulative line crossings, congestion/monitoring state, video time, and threshold.

The **video** is therefore both a presentation artifact and a debugging aid. For example, if a motorcycle remains purple after visibly crossing the line, it suggests that the traffic monitor did not register a crossing for its current track ID. The camera view and tracked boxes can then be checked for identity changes or geometry mismatch instead of treating the numeric total as unquestionable.

#### Recorded artifact contract

```text
outputs/<job-id>/
├── input.mp4              # original local upload, suffix follows source
├── annotated.mp4          # processed and optionally H.264-transcoded video
├── preview.jpg            # most recently published preview frame
├── counts.csv             # sampled occupancy and line-count timeseries
├── summary.json           # final metrics, config, event and alert references
├── events.json            # congestion transitions with event screenshots
├── alerts.json            # dry-run/live SMS audit records
├── event_001.jpg          # screenshot at first event transition
└── event_002.jpg          # screenshot at next transition, if any
```

The intermediate `annotated_raw.mp4` is written using OpenCV's `mp4v` codec. If `ffmpeg` is installed, the pipeline transcodes it to H.264 with `yuv420p` and `+faststart` for browser-friendly playback. If transcoding is unavailable or fails, the original `mp4v` content is retained under `annotated.mp4`. That file can still be downloaded, although some browsers will not play it inline.

#### Evidence written at event time

When `result["event"]` is not `None`, the pipeline constructs an event with its type, timestamp, occupancy, and cumulative crossings, then attaches the alert-dispatcher result and a JPEG screenshot. The event screenshot is generated **after** the overlay is drawn, so it captures the same metrics and colored boxes the operator would have seen at that point in the video.

A simplified event record has this shape:

```json
{
  "type": "CONGESTION_STARTED",
  "timestamp_seconds": 3.0,
  "occupancy": 22,
  "total_line_crossings": 7,
  "alert": {
    "at_seconds": 3.0,
    "mode": "DRY_RUN",
    "message": "TRAFFIC CONGESTION ALERT ...",
    "status": "preview_only"
  },
  "screenshot": "event_001.jpg"
}
```

*Illustrative record structure. The values and alert status depend on the job. A live attempt may instead record `accepted_by_api`, `send_failed`, or suppression.*

`counts.csv` records a sample when a new integer video second is reached, or an event occurs. The sample fields are `timestamp_seconds`, `occupancy`, `total_line_crossings`, `congested`, and `above_threshold_seconds`. This is suitable for inspecting a rough occupancy timeline, but not a full per-frame detection log.

At the end, `summary.json` includes the video duration and FPS, processed-frame count, final occupancy, cumulative crossings, class-wise flow, final class occupancy, congestion-event count, SMS mode, alert records, event references, and the congestion configuration. Together, the files let a completed job be inspected without rerunning YOLOv3.

### 3.9 Browser Dashboard and Live Preview

The browser experience is built with FastAPI, Jinja2 templates, static JavaScript, and CSS. It deliberately stays local and simple: upload a video, change monitoring parameters, start a run, watch progress and preview frames, then inspect the completed video, metrics, events, and downloads.

#### Why the preview is frame polling, not a live video stream

The application performs model inference frame by frame, possibly slower than real-time playback. Serving the partially written MP4 during this process would be unreliable because the video container may not yet be finalized. Rather than adding MJPEG streaming or WebSockets, the processor periodically saves an **independent JPEG preview**.

Every five processed frames, the pipeline writes `preview_pending.jpg` and atomically replaces `preview.jpg`. The browser polls the `/api/jobs/{job_id}/preview` endpoint approximately every **500 milliseconds** while the job is active. Each request includes a cache-busting query value and the server replies with `Cache-Control: no-store`.

```text
Video processing thread                Browser
         │                                │
     process frame                        │
         │                                │
     every 5 frames                       │
         │                                │
   write preview_pending.jpg              │
         │                                │
   atomically replace preview.jpg         │
         │                                │
         │             GET /preview ◄─────┤ every ~500 ms
         │                                │
         └──────── latest JPEG ──────────►│ update <img>
                                          │
                                   continue polling
```

This is a **live view of processing progress**, not real-time camera ingestion or 30 FPS live streaming. The actual update rate depends on the detector's speed and how often the browser polls. The frontend also avoids overlapping preview fetches and releases previous blob URLs to limit browser memory use.

#### Status and result flow

The UI polls `/api/jobs/{job_id}` roughly every 1.2 seconds. It uses the tracked processed-frame count and source total-frame count for a progress estimate, then switches to the result view when the job becomes `completed`. The result view includes cumulative crossings, congestion episodes, video duration, class-wise crossing totals, a completed annotated video player, the event history, the SMS message content, and download links.

The API does not expose arbitrary filesystem paths. `/api/jobs/{job_id}/files/{filename}` checks the completed job status and allows only a fixed set of known filenames plus numbered event JPEGs. The frontend escapes dynamic text before inserting it into generated HTML. These are useful local safety controls, though they do not turn the website into a hardened public service.

The website's job status lives **in memory**. Restarting the process clears job status and therefore normal UI access to old jobs, but the output files stay under `outputs/` until removed manually. This is an intentional simplification for a localhost prototype; persistent job management would require a datastore and a recovery policy.

### 3.10 Validation, Failure Handling, and Security Boundaries

This project has fewer independent services than a production traffic platform, but the boundaries around configuration, uploaded files, output exposure, and SMS attempts still matter. A pipeline that works only on ideal inputs is harder to demonstrate and harder to troubleshoot.

#### Failures have different meanings

| Situation | Behavior | Engineering reason |
| --- | --- | --- |
| Invalid threshold, duration, confidence, ROI, or road label | Reject configuration (`422` via web API) | Do not run inference with nonsensical monitoring rules |
| Unsupported upload extension | `422` | Bound supported container types |
| Upload exceeds configured size | `413` | Prevent unbounded local uploads |
| Another active web job | `409` | Keep one in-process processing workflow at a time |
| Invalid MP4 bytes under a supported extension | Job may initially return `202`, then become `failed` | File decode is validated when processing opens it |
| Model assets missing | Informative `FileNotFoundError` | Avoid silent detector fallback |
| Unreliable FPS metadata | Processing fails | Preserve meaningful video-time semantics |
| Video has no readable frames | Processing fails | Do not report a fabricated successful summary |
| Twilio credentials missing with live mode requested | Reject live configuration | Require explicit setup for outbound side effects |
| Twilio API request raises an exception | Record `send_failed` with generic description | Avoid exposing credentials or raw provider details |
| Twilio accepted request but carrier rejects delivery | Inspect Twilio delivery logs separately | API acceptance is not handset receipt |

Processing releases the OpenCV capture and video writer in a `finally` block, even when inference or artifact generation fails. The background task records a failed job status and an error safe enough for the local UI, while its terminal traceback supports local debugging.

#### Local safety model

- **Local interface:** the recommended launch binds to `127.0.0.1`; the application has no authentication and is not designed to be internet-facing.
- **Credential handling:** `.env` is loaded locally and ignored by Git. SMS settings are not supposed to appear in JSON artifacts or the repository.
- **Explicit outbound action:** live SMS begins unchecked; preview-only behavior does not contact Twilio.
- **Outbound limits:** cooldown and attempt cap reduce accidental repeat sends within a job.
- **Upload guardrails:** filename extension checking, bounded upload size, normalized destination filename, and single active job.
- **Artifact allowlist:** only named output files and numbered event screenshots can be served after a job completes.
- **UI output escaping:** generated frontend HTML escapes dynamic event text.

These controls are **necessary but not sufficient** for public deployment. There is no account system, authorization model, persistent job queue, malware scanning, robust file-type inspection, per-user quotas, or isolation of untrusted media decoding. The design should remain a localhost application until those capabilities are introduced.

### 3.11 Testing and Verification

The project uses `pytest` to validate logic at several levels, with a particularly important distinction between **deterministic software tests** and **real-video/telecom observations**. Unit and integration tests can establish what happens with known synthetic detections; they cannot establish detector accuracy on arbitrary real roads or guarantee carrier SMS delivery.

#### A. Tracker and traffic-monitor tests

`tests/test_traffic_monitor.py` verifies that IDs persist across short missing-detection gaps, expired tracks receive new IDs, a vehicle counts only once across the line, ROI occupancy is independent of cumulative flow, persistence resets when occupancy drops, recovery clears a congestion episode, and invalid configurations are rejected.

The tracker and counting logic were subsequently adjusted to improve fast-motorcycle continuity and use bottom-center line crossings. These changes should be checked with visual playback and targeted regression tests when modifying the tracker further; a passing simple-scene suite is not a dense-traffic benchmark.

#### B. Alert tests

`tests/test_alerts.py` injects a fake send function. It checks that dry-run mode never sends a message, permitted live events invoke the fake sender, the cooldown and maximum-attempt cap work, invalid credentials are rejected, and raised provider errors produce sanitized failure records. No real phone number or Twilio call is necessary for these tests.

#### C. Video pipeline tests

`tests/test_pipeline.py` generates a short synthetic MP4 and injects a deterministic fake detector. This exercises the real decoder, tracker, traffic monitor, overlay, video writer, preview image, event screenshots, CSV output, and JSON results, while avoiding a network model download. It verifies the expected frame count, class flow, congestion transitions, SMS preview status, artifact presence, and that OpenCV can reopen the annotated output.

#### D. Web application tests

`tests/test_web.py` uses FastAPI's `TestClient` to verify dashboard availability, upload creation, completed job responses, preview-image access and cache settings, artifact serving, missing-job handling, unsupported filenames, and invalid parameters.

A practical test-isolation issue appeared after configuring actual Twilio credentials in `.env`: a test expected a live-SMS request to be rejected because Twilio was *not* configured, but with credentials present, the route correctly returned `202`. Running tests with `.env` temporarily unavailable resulted in **13 passing tests**. The more durable fix is to patch the credential-configuration function in that test so the simulated missing-credential condition does not depend on the developer's machine.

The recorded local pytest result was **13 passed, 1 dependency deprecation warning** after excluding credentials. This is the observed test-suite result, not a claim of 100% coverage or validated detection accuracy. The warning related to the Starlette/httpx test-client compatibility layer, not traffic logic.

#### E. End-to-end observations with recorded footage

The user ran RoadWatch on actual prerecorded traffic footage and generated annotated output, line-crossing counts, congestion events, and supporting JSON/JPEG evidence. One demonstrated configuration produced a congestion event at approximately **3 seconds of video time** with a low demonstration threshold, as shown in the saved event outputs. The actual Twilio Console showed a submitted message entering its delivery workflow but failing carrier delivery under error **30034** for an unregistered US long-code sender.

These observations establish that the integrated workflow ran on the chosen video and reached the messaging provider in that test. They do **not** establish precision/recall, multi-camera robustness, consistent motorcycle-counting accuracy, or successful SMS delivery to a handset.

#### Verification summary

| Layer | Verification available | Not established |
| --- | --- | --- |
| Frame and artifact pipeline | Generated-video integration test, output files, decoded annotated MP4 | Performance across all video formats/codecs |
| Tracking and counting | Synthetic track and crossing tests; annotated real-video inspection | MOT accuracy, ID-switch rate, crowded-scene accuracy |
| Congestion logic | Threshold, persistence, recovery, and event-transition tests | Validated correlation with actual traffic jams |
| Web workflow | Upload, polling/preview, and artifact endpoint tests | Internet-facing security and high-concurrency behavior |
| Twilio adapter | Fake-sender tests and provider-side submission evidence | Successful handset delivery for the unregistered sender |
| Model accuracy | Observed vehicle detections in real footage | Ground-truth mAP/precision/recall on this footage |

---

## 4. Engineering Decisions and Tradeoffs

A useful deep dive should explain not only **what** was implemented but also **why the boundaries are where they are**. Several decisions materially affect the project's behavior.

### 4.1 Separate flow from occupancy

A running sum of crossings is useful for throughput, but cannot establish the number of vehicles currently on the road. A high total after ten minutes might reflect healthy traffic moving efficiently. Conversely, a stationary line of vehicles can produce high occupancy and almost no new line crossings. Keeping both counters independent prevents the congestion alert from being driven by a metric that answers the wrong question.

### 4.2 Prefer interpretable congestion rules over an unsupported AI claim

The project does not train a congestion classifier. Instead, it defines explicit threshold, persistence, and recovery parameters that an operator can inspect. This makes the trigger understandable and testable. The tradeoff is that occupancy alone is only a proxy for congestion and must be calibrated for each camera and traffic pattern.

### 4.3 Improve centroid tracking before introducing a heavier MOT stack

Greedy association, prediction, duplicate suppression, and short class histories preserve the lightweight OpenCV-based architecture. They address visible failure modes without adding an appearance-embedding model or a new detection framework. The tradeoff is continued sensitivity to long occlusions, crossing trajectories, and similar vehicles close together. ByteTrack, DeepSORT, or another established tracker is a reasonable future step **if measured failures justify the added complexity**.

### 4.4 Use separate geometry anchors for separate measurements

Bottom-center is a reasonable proxy for line crossing on an elevated road view. Box center is a straightforward choice for whether a visible vehicle lies within the ROI. Sharing one coordinate only for implementation convenience would change one metric's meaning. Both anchors still depend on box quality and camera perspective.

### 4.5 Measure durations from the recording, not processor speed

Video-time timestamps allow the same clip to produce the same three-second persistence decision on different machines. The cost is reliance on FPS metadata, which is imperfect for variable-frame-rate sources. Accurate variable-frame-rate support would need frame presentation timestamps rather than `frame_index / FPS`.

### 4.6 Poll a completed JPEG rather than stream a partially written MP4

A periodically replaced `preview.jpg` is simple, robust enough for a single local user, and avoids complex video-streaming infrastructure. It also remains usable when inference is significantly slower than the video's native FPS. The cost is a lower, irregular preview refresh rate rather than smooth live playback.

### 4.7 Treat SMS as a separate side effect with an audit record

The core congestion result must remain useful without a Twilio account. Preview-first behavior lets users inspect messages and test events safely; opting in triggers a bounded provider request. Recording API status separately from actual delivery avoids falsely equating a returned message SID with a delivered SMS. Provider registration and carrier behavior remain external dependencies.

---

## 5. Limitations and Future Improvements

RoadWatch is an inspectable recorded-video traffic-monitoring prototype, not a production traffic-control product. The following limitations are important precisely because the demo can look convincing even when individual detections or counts are wrong.

| Current limitation | Consequence | Practical next step |
| --- | --- | --- |
| Pretrained YOLOv3 with four selected COCO classes | Auto-rickshaws may be labeled as trucks; bicycles excluded | Evaluate a suitable traffic dataset and fine-tune a detector with explicit classes |
| Greedy association without appearance re-identification | ID switches and missed/double counts in occlusions | Benchmark ByteTrack/DeepSORT against labeled trajectories |
| Basic geometric line crossing | Incorrect crossing for poor boxes or camera perspective | Camera-specific calibration, directional counting, trajectory validation |
| ROI occupancy is based on visible centers | Misses and intermittent detections can reset duration | Temporal smoothing and occupancy confidence, then revalidate alert semantics |
| No vehicle speed estimation | Dense moving traffic can trigger a congestion proxy | Calibrated motion/speed estimates and stopped-vehicle duration |
| Constant-FPS timestamp assumption | Inaccurate timing for variable-frame-rate footage | Use actual frame timestamps or normalize source FPS |
| Single job, in-memory job registry | No persistent queues or restart recovery | Job database and durable background worker |
| Periodic JPEG polling | Preview is not continuous video streaming | MJPEG/WebSocket only if smooth streaming becomes necessary |
| SMS cooldown is per run | Reprocessing may repeat an alert | Persistent notification idempotency and provider delivery callbacks |
| Local unauthenticated web app | Unsafe to expose publicly as-is | Authentication, authorization, content validation, quotas, isolated workers |
| No labeled test corpus | No defensible accuracy or performance claim | Annotate sample videos; report per-class precision, tracking quality, false alerts |

Two future enhancements would be especially valuable for demonstrating applied computer vision rather than just adding features. First, **evaluation against labeled road footage** would turn visual anecdotes into measured false-count, missed-crossing, and event-detection rates. Second, **speed-aware congestion inference** would help distinguish sustained high occupancy from smoothly flowing heavy traffic.

---

## 6. Codebase Map and Reproduction

### Project layout

```text
Vehicle-Counting-and-Traffic-Congestion-Detection/
├── README.md                         # setup and public-facing demo
├── DEEPDIVE.md                       # this implementation walkthrough
├── .env.example                      # Twilio environment template (no secrets)
├── .gitignore                        # ignore credentials, models, videos, outputs
├── requirements.txt
├── requirements-dev.txt
├── docs/assets/                      # README showcase assets
│   ├── input_frame.jpg               # unannotated source-frame example
│   ├── annotated_frame.jpg           # annotated output still
│   ├── congestion_event.jpg          # event evidence still
│   ├── roadwatch_demo.gif            # short annotated demo
│   └── twilio_sms_mockup.png          # illustrative message-interface image
├── src/
│   ├── download_models.py            # external YOLOv3 asset download
│   ├── alerts.py                     # SMS formatting and dispatch
│   ├── components/
│   │   ├── models.py                 # Detection, Track, vehicle labels
│   │   ├── video_network_setup.py    # Darknet model initialization
│   │   ├── vehicle_detection.py      # frame inference and class-wise NMS
│   │   ├── vehicle_tracking.py       # association and duplicate removal
│   │   ├── vehicle_counting.py       # throughput, occupancy, event monitor
│   │   ├── bounding_boxes.py        # visualization overlays
│   │   └── main.py                  # CLI
│   ├── pipeline/
│   │   └── predict_pipeline.py       # frame orchestration and output writer
│   └── web/
│       ├── app.py                    # FastAPI routes and in-memory jobs
│       ├── templates/index.html      # dashboard page
│       └── static/                   # browser JS and styles
├── tests/                             # traffic, pipeline, alerts, HTTP
└── outputs/                           # generated locally; not committed
```

### Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.download_models
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, upload a recording, configure the monitoring parameters, and process it. The YOLOv3 weights are downloaded once and stored locally under `src/Data/`. Installing `ffmpeg` is recommended for H.264 browser video playback.

### Command-line equivalent

```bash
python -m src.components.main \
  --input path/to/traffic.mp4 \
  --output outputs/cli-demo \
  --road "Road Segment A" \
  --threshold 2 \
  --duration 3 \
  --recovery 10 \
  --line 0.65 \
  --roi 0 0 1 1 \
  --confidence 0.5
```

By default, this runs with **SMS preview only**. The `--live-sms` flag is an explicit, potentially billable action and should only be used after configuring a permitted sender and recipient in a local `.env` file.

### Test command

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use injected detectors and SMS callbacks rather than downloading the YOLOv3 weights or sending actual messages. A credential-dependent negative test must explicitly mock an unconfigured Twilio environment to remain stable on machines where `.env` contains real credentials.

---

## 7. Closing Perspective

The interesting part of RoadWatch is not simply that a YOLOv3 model draws boxes around vehicles. The application turns noisy frame-level predictions into persistent identities, defines what counts as a unique crossing, maintains a separate and time-aware occupancy signal, recognizes congestion transitions, and produces evidence that can be reviewed independently of the original processing session.

The implementation also makes its boundaries visible. A crossing count is only as trustworthy as the track ID behind it. Sustained occupancy is evidence of a crowded monitored region, not proof of stopped traffic. A Twilio API request is not the same thing as a delivered text message. Treating these distinctions as part of the design—rather than hiding them behind an attractive dashboard—is what makes the project a useful end-to-end computer-vision engineering exercise.

For a quick view of the annotated video GIF, input/output comparison, and SMS example, return to the [RoadWatch README](README.md).
