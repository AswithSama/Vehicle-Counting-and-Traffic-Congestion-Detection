# Deep Dive — RoadWatch implementation

## Intent and correctness boundaries

The camera observes a limited field of view. `TrafficMonitor` computes two independent metrics each time the video decoder yields a frame:

1. **Flow count** = number of distinct internal track IDs that have transitioned across the virtual counting line. One track increments the count once, regardless of repeated later crossings.
2. **Instantaneous ROI occupancy** = count of *currently visible* tracked detections whose box centers fall inside the configured rectangle. It resets every frame; it is not a unique all-time count.

The congestion decision is driven **only by occupancy**, not by cumulative flow. Sustained occupancy can also occur on a road with many moving vehicles, so the output is *potential congestion*, requiring human verification.

## Detection

`YoloVehicleDetector` loads `yolov3.cfg` + `yolov3.weights` using `cv2.dnn.readNetFromDarknet`. Inputs are resized to `416 × 416` using `blobFromImage(..., scalefactor=1/255, swapRB=True)`. For each YOLO proposal, confidence is `objectness × class probability`; only COCO class IDs `2,3,5,7` are retained. Bounding boxes are clipped to image dimensions and filtered by per-class nonmaximum suppression. Each returned `Detection` contains `class_id`, `(x,y,w,h)`, and a score.

Models are downloaded separately. This is inference using pretrained weights, not model training. Source videos stay local.

## Tracking

`VehicleTracker` maintains `Track` objects with persistent IDs and class labels. Pair candidates use Euclidean center distance and matching class ID. Candidates are greedily chosen shortest first with each previous track and new detection used at most once. Unmatched detections get new increasing IDs. Missing tracks are retained up to `max_missing_frames`, but **only visible matches count as occupancy**. Heavy congestion can cause ID switches due to occlusions; this is not multi-object tracking state of the art.

## Line-crossing throughput

The counting line is defined as `line_y = height * line_y_fraction`. An ID initially observed on either side of a ±4px deadband saves that side. The first subsequent observation on the opposite side is a crossing: its class counter increments and its ID enters `counted_ids`. Further crossings by that same ID are ignored. A new ID after tracking loss may be counted again; the tracker is the weak point, not the crossing state machine.

## ROI occupancy and clock

The UI asks for fractional ROI boundaries `[x1,y1,x2,y2]` between 0 and 1. A point is counted if its center is within that rectangle. The monitor checks the threshold every frame, not just once every second. Export CSV is sampled to about one row per *video second*, while actual event evaluation happens per frame.

For decoded video frame index `i` and metadata FPS `f`, the congestion timestamp is `t = i/f` seconds. This is intentionally independent of actual inference execution time. A 180-second rule needs 180 seconds of recorded high occupancy, even when processing takes longer or shorter than real time. Input with unreliable frame-rate metadata is rejected instead of silently assigning a made-up rate. Variable-frame-rate videos may be best normalized to constant FPS before processing.

### Event state machine

```text
NOT_CONGESTED
  | occupancy >= threshold: store above_since
  | below threshold: clear above_since
  | above threshold continuously for persistence_seconds
  v
CONGESTION_STARTED   -> SMS preview or live send attempt (once on transition)
CONGESTED
  | occupancy < threshold: store below_since
  | returns above threshold: clear below_since
  | remains below threshold for recovery_seconds
  v
CONGESTION_CLEARED   -> no SMS; new event can form on future sustained occupancy
```

No particular vehicle is required to remain in the ROI for the entire interval. The state machine considers total observed occupancy. A one-frame false negative resets the high-occupancy timer; production tuning could add a short tolerance or temporal smoothing, but that changes the semantics and must be independently validated.

## SMS policy and audit trail

`AlertDispatcher` accepts `CONGESTION_STARTED` only. Default is **DRY_RUN**, recording the full message text to `alerts.json` but never contacting Twilio. On explicit live enable, it reads `.env` locally and calls `Client(account_sid, auth_token).messages.create(body=..., from_=..., to=...)`. The dispatcher records returned `sid` as `accepted_by_api`, not `delivered`; errors are intentionally generic and don't serialize the provider exception or credentials. A video-time 600-second cooldown and a limit of 3 live SMS attempts per job prevent runaway duplicate sends. Because requests are tied to video time rather than wall time, short video playback or accelerated processing won't change the threshold logic.

## Evidence, reproducibility and UX

A successful run writes `annotated.mp4`, `preview.jpg`, JSON metrics/events/alerts, `counts.csv`, and screenshots for each congestion event. The local UI starts a background task for one run, polls `/api/jobs/<id>`, and exposes only named output artifacts. The input filename is replaced with `input.<extension>` to avoid path traversal. Upload size is capped (default 500 MB). The web app is a localhost prototype, not a security-hardened public service.

If ffmpeg is installed, the OpenCV intermediate `mp4v` output is converted to browser-friendly H.264. Otherwise it's still downloadable, but some browsers may not support direct playback. Job status does not survive restarting the server; files remain.

## Offline testing strategy

Tests replace YOLO inference with a deterministic fake detector and process a generated MP4 with moving bounding boxes. They exercise decoder → track association → line and occupancy counts → timestamp event → preview SMS → video encoder → JSON/CSV/JPEG artifacts. Separate HTTP tests exercise form validation, upload, background job results and download routing. Live SMS sending is mocked using a callback, so the user's phone and real account are never contacted during tests.

**Not a claim of model quality:** no annotated ground-truth benchmark has yet been supplied. Tracking performance, alert false positive rate, SMS handset delivery, and live camera reliability remain to be evaluated with real recordings.

## Future extensions

Area-normalized occupancy and per-lane polygons; speed and stop-duration estimation via calibrated homography; temporal filtering of intermittent detections; ByteTrack or DeepSORT for occluded tracks; adaptive per-camera congestion thresholds; robust timestamp handling for variable-FPS or real live streams; SMS delivery callbacks; persistent jobs, security hardening and asynchronous workers for production multi-camera use.
