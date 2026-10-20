"""FastAPI localhost UI with bounded upload and single-video processing queue."""
import os
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from src.alerts import configured
from src.components.vehicle_counting import TrafficConfig
from src.pipeline.predict_pipeline import ProcessingOptions, process_video

load_dotenv(override=False)
ROOT = Path(__file__).resolve().parent
FILE_TYPES = {".mp4", ".avi", ".mov", ".mkv"}


def create_app(*, output_root: Path | None = None, processor=process_video) -> FastAPI:
    app = FastAPI(title="RoadWatch · Vehicle Traffic Monitor")
    root = Path(output_root or os.getenv("TRAFFIC_OUTPUT_DIR", "outputs")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    max_upload_bytes = max(1, int(os.getenv("TRAFFIC_MAX_UPLOAD_MB", "500"))) * 1024 * 1024
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    templates = Jinja2Templates(directory=ROOT / "templates")
    lock = threading.Lock()
    jobs: dict[str, dict] = {}
    active_id: str | None = None

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return templates.TemplateResponse(request=request, name="index.html", context={
            "twilio_ready": configured(),
            "max_upload_mb": max_upload_bytes // (1024 * 1024),
        })

    def background_process(job_id: str, input_path: Path, opts: ProcessingOptions):
        nonlocal active_id
        def progress(frames, total):
            with lock:
                jobs[job_id]["frames"] = frames
                jobs[job_id]["total_frames"] = total
        with lock:
            jobs[job_id]["status"] = "processing"
        try:
            summary = processor(input_path, root / job_id, opts, progress=progress)
            with lock:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["summary"] = summary
        except Exception as exc:
            # User-facing failure only; never serialize credentials from exceptions.
            with lock:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = (
                    str(exc)[:300] if isinstance(exc, (ValueError, FileNotFoundError))
                    else f"Processing failed ({type(exc).__name__}). See local terminal logs."
                )
            # Local operator can inspect the traceback in terminal, without credentials.
            import traceback
            traceback.print_exc()
        finally:
            with lock:
                if active_id == job_id:
                    active_id = None

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        tasks: BackgroundTasks,
        video: UploadFile = File(...),
        road_name: str = Form("Road Segment A"),
        threshold: int = Form(25),
        duration: float = Form(180),
        recovery: float = Form(10),
        line_fraction: float = Form(0.65),
        confidence: float = Form(0.5),
        roi_x1: float = Form(0.0),
        roi_y1: float = Form(0.0),
        roi_x2: float = Form(1.0),
        roi_y2: float = Form(1.0),
        live_sms: bool = Form(False),
    ):
        nonlocal active_id
        try:
            traffic = TrafficConfig(
                occupancy_threshold=threshold, persistence_seconds=duration,
                recovery_seconds=recovery, line_y_fraction=line_fraction,
                roi=(roi_x1, roi_y1, roi_x2, roi_y2),
            )
            opts = ProcessingOptions(road_name=road_name.strip(), traffic=traffic,
                                     confidence=confidence, live_sms=live_sms)
            if live_sms and not configured():
                raise ValueError("Live SMS requested but Twilio credentials are not configured. Set .env and restart the server.")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in FILE_TYPES:
            raise HTTPException(status_code=422, detail="Supported file extensions: MP4, AVI, MOV, MKV")
        with lock:
            if active_id is not None:
                raise HTTPException(status_code=409, detail="A video is already processing or uploading; retry after it finishes")
            job_id = uuid.uuid4().hex[:12]
            active_id = job_id
            jobs[job_id] = {"job_id": job_id, "status": "uploading", "frames": 0, "total_frames": 0}
        destination = root / job_id
        destination.mkdir(parents=True, exist_ok=True)
        input_path = destination / f"input{suffix}"
        uploaded_bytes = 0
        try:
            with input_path.open("wb") as out:
                while chunk := await video.read(1024 * 1024):
                    uploaded_bytes += len(chunk)
                    if uploaded_bytes > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Video exceeds the configured upload size limit")
                    out.write(chunk)
            if not uploaded_bytes:
                raise HTTPException(status_code=422, detail="The uploaded video is empty")
            tasks.add_task(background_process, job_id, input_path, opts)
            with lock:
                jobs[job_id]["status"] = "queued"
            return {"job_id": job_id, "status": "queued"}
        except Exception:
            input_path.unlink(missing_ok=True)
            with lock:
                active_id = None
                jobs.pop(job_id, None)
            raise
        finally:
            await video.close()

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        with lock:
            result = jobs.get(job_id)
            if result is None:
                raise HTTPException(status_code=404, detail="Job not found; jobs are stored in memory until server restarts")
            return dict(result)

    @app.get("/api/jobs/{job_id}/files/{filename}")
    def artifact(job_id: str, filename: str):
        with lock:
            state = jobs.get(job_id)
            if state is None or state["status"] != "completed":
                raise HTTPException(status_code=404, detail="Artifact not available")
        allowed = {"annotated.mp4", "preview.jpg", "summary.json", "events.json", "alerts.json", "counts.csv"}
        if filename not in allowed and not (filename.startswith("event_") and filename.endswith(".jpg") and filename[6:-4].isdigit()):
            raise HTTPException(status_code=404, detail="Unknown artifact")
        path = root / job_id / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact file not found")
        return FileResponse(path, filename=filename if filename != "annotated.mp4" else None,
                            media_type="video/mp4" if filename == "annotated.mp4" else None)

    return app


app = create_app()
