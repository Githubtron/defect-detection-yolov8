import asyncio
import base64
import io
import os
import tempfile
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from cv_detector import CvDefectDetector
from ultralytics import YOLO

app = FastAPI(title="Aircraft Defect Detection API")
BASE_DIR = Path(__file__).resolve().parent


class RuntimeMetrics:
    """In-memory latency/throughput stats for quick operational visibility."""

    def __init__(self, max_samples: int = 500):
        self._http_ms: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max_samples))
        self._ws_latency_ms: deque[float] = deque(maxlen=max_samples)
        self._ws_fps: deque[float] = deque(maxlen=max_samples)
        self._ws_sessions = 0
        self._ws_frames = 0
        self._lock = threading.Lock()

    def record_http(self, key: str, latency_ms: float) -> None:
        with self._lock:
            self._http_ms[key].append(float(latency_ms))

    def record_ws_frame(self, latency_ms: float, fps: float) -> None:
        with self._lock:
            self._ws_latency_ms.append(float(latency_ms))
            if fps > 0:
                self._ws_fps.append(float(fps))
            self._ws_frames += 1

    def record_ws_session(self) -> None:
        with self._lock:
            self._ws_sessions += 1

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "avg_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
        arr = np.array(values, dtype=np.float32)
        return {
            "count": int(arr.size),
            "avg_ms": round(float(arr.mean()), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "min_ms": round(float(arr.min()), 2),
            "max_ms": round(float(arr.max()), 2),
        }

    def snapshot(self) -> dict:
        with self._lock:
            http = {k: self._summary(list(v)) for k, v in self._http_ms.items()}
            ws_latency = self._summary(list(self._ws_latency_ms))
            ws_fps_vals = list(self._ws_fps)
            ws_fps = None if not ws_fps_vals else round(float(np.mean(np.array(ws_fps_vals, dtype=np.float32))), 2)
            return {
                "http_latency": http,
                "streaming": {
                    "websocket_sessions": self._ws_sessions,
                    "websocket_frames_processed": self._ws_frames,
                    "avg_stream_fps": ws_fps,
                    "frame_latency": ws_latency,
                },
            }


RUNTIME_METRICS = RuntimeMetrics()

# Enable CORS so your React frontend can talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# "defect" is the deterministic simulation engine (classical CV, no weights)
# that detects the six aircraft-surface defect classes on procedurally
# generated panels -- the knocked-down replacement for the under-trained
# checkpoint when no real dataset exists. "demo" is the stock COCO-pretrained
# model (person, chair, book, bottle, laptop, ...) for everyday-object testing.
# Both expose the same minimal YOLO-like interface so every endpoint works
# unchanged (predict / batch / video / websocket / report).
MODEL_REGISTRY = {
    "defect": {"path": str(BASE_DIR / "best.pt"), "label": "Aircraft Defect Model"},
    "demo": {"path": str(BASE_DIR / "yolov8n.pt"), "label": "General Object Demo"},
}

# Load eagerly at startup so a bad model key fails fast, and the first request
# for a given model isn't slowed down by a lazy load (or download, for demo).
for _key, _entry in MODEL_REGISTRY.items():
    if _key == "defect":
        try:
            _entry["model"] = CvDefectDetector()
        except Exception:
            _entry["model"] = YOLO(_entry["path"])
    else:
        _entry["model"] = YOLO(_entry["path"])

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi"}


@app.middleware("http")
async def http_timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    RUNTIME_METRICS.record_http(f"{request.method} {request.url.path}", elapsed_ms)
    return response


def get_model(model_key: str) -> Any:
    entry = MODEL_REGISTRY.get(model_key)
    if entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'. Valid options: {list(MODEL_REGISTRY)}")
    return entry["model"]


def encode_frame_to_data_url(bgr_array) -> str:
    """Encode a BGR numpy array (as produced by cv2/ultralytics) into a base64 data URL."""
    ok, buffer = cv2.imencode(".jpg", bgr_array)
    if not ok:
        raise RuntimeError("Failed to encode annotated frame")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def run_inference(image_source, model: Any, include_annotated: bool = True) -> tuple[list[dict], str]:
    """Run YOLO inference on a PIL Image or numpy array and return (detections, annotated_data_url)."""
    results = model(image_source)
    detections = []
    annotated_data_url = ""
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detections.append({"class": class_name, "confidence": round(confidence, 2)})
        if include_annotated:
            annotated_data_url = encode_frame_to_data_url(r.plot())
    return detections, annotated_data_url


@app.get("/")
def home():
    return {"status": "Backend is up and running!"}


@app.get("/models")
def get_models():
    return {
        "models": [
            {"key": key, "label": entry["label"], "classes": list(entry["model"].names.values())}
            for key, entry in MODEL_REGISTRY.items()
        ]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = "defect", include_annotated: bool = True):
    active_model = get_model(model)
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    detections, annotated_image = run_inference(image, active_model, include_annotated=include_annotated)

    return {
        "filename": file.filename,
        "detections": detections,
        "annotated_image": annotated_image,
    }


@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...), model: str = "defect", include_annotated: bool = True):
    active_model = get_model(model)
    results = []
    for f in files:
        image_bytes = await f.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            results.append({"filename": f.filename, "error": "Could not read image", "detections": []})
            continue

        detections, annotated_image = run_inference(image, active_model, include_annotated=include_annotated)
        results.append({
            "filename": f.filename,
            "detections": detections,
            "annotated_image": annotated_image,
        })

    return {"results": results}


@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    frame_stride: int = 5,
    model: str = "defect",
    include_annotated: bool = True,
):
    """Process an uploaded video, sampling every `frame_stride` frames to control cost."""
    active_model = get_model(model)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .mp4 and .avi files are supported")

    # OpenCV's VideoCapture needs a real file path, so persist the upload to a temp file.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        frame_index = 0
        analyzed_frames = 0
        class_counts: dict[str, int] = {}
        detections_by_frame = []
        best_frame_data_url = None
        best_frame_confidence = -1.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % frame_stride == 0:
                    analyzed_frames += 1
                    frame_detections, annotated_data_url = run_inference(
                        frame,
                        active_model,
                        include_annotated=include_annotated,
                    )

                    if frame_detections:
                        detections_by_frame.append({"frame": frame_index, "detections": frame_detections})
                        for d in frame_detections:
                            class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

                        top_conf = max(d["confidence"] for d in frame_detections)
                        if top_conf > best_frame_confidence:
                            best_frame_confidence = top_conf
                            best_frame_data_url = annotated_data_url

                frame_index += 1
        finally:
            cap.release()

        return {
            "filename": file.filename,
            "frames_total": frame_index,
            "frames_analyzed": analyzed_frames,
            "class_counts": class_counts,
            "detections_by_frame": detections_by_frame,
            "annotated_image": best_frame_data_url,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.websocket("/ws/live")
async def live_feed(websocket: WebSocket, model: str = "defect", include_annotated: bool = True):
    """Live inspection stream: client sends binary JPEG frames, server replies with
    detections + annotated frame per message over one persistent connection."""
    entry = MODEL_REGISTRY.get(model)
    if entry is None:
        # Accept then close with a policy code so the client sees a clean error
        await websocket.accept()
        await websocket.close(code=4400, reason=f"Unknown model '{model}'")
        return
    active_model = entry["model"]

    await websocket.accept()
    RUNTIME_METRICS.record_ws_session()
    last_frame_at = time.perf_counter()
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            start = time.perf_counter()
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            # Inference is CPU-bound; run it off the event loop so the
            # connection stays responsive.
            detections, annotated_image = await asyncio.to_thread(
                run_inference,
                frame,
                active_model,
                include_annotated,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            now = time.perf_counter()
            frame_dt = max(1e-9, now - last_frame_at)
            frame_fps = 1.0 / frame_dt
            last_frame_at = now
            RUNTIME_METRICS.record_ws_frame(latency_ms=latency_ms, fps=frame_fps)

            await websocket.send_json({
                "detections": detections,
                "annotated_image": annotated_image,
                "latency_ms": latency_ms,
            })
    except WebSocketDisconnect:
        pass


@app.get("/metrics/system")
def get_system_metrics():
    return RUNTIME_METRICS.snapshot()


class ReportRequest(BaseModel):
    filename: str = "inspection"
    detections: List[dict] = []
    annotated_image: Optional[str] = None


@app.post("/generate-report")
async def generate_report(payload: ReportRequest):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    story = []

    total_defects = len(payload.detections)
    class_counts: dict[str, int] = {}
    highest_conf_detection = None
    for d in payload.detections:
        cls = d.get("class", "unknown")
        conf = float(d.get("confidence", 0))
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if highest_conf_detection is None or conf > highest_conf_detection["confidence"]:
            highest_conf_detection = {"class": cls, "confidence": conf}
    status = "FAIL" if total_defects > 0 else "PASS"
    status_color = colors.HexColor("#dc2626") if status == "FAIL" else colors.HexColor("#16a34a")

    story.append(Paragraph("Aircraft Surface Defect Inspection Report", styles["Title"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"Source file: {payload.filename}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 8 * mm))

    summary_style = styles["Heading2"]
    story.append(Paragraph(f"Total Defects Found: {total_defects}", styles["Normal"]))
    status_paragraph_style = styles["Heading2"].clone("StatusStyle")
    status_paragraph_style.textColor = status_color
    story.append(Paragraph(f"Surface Integrity Status: {status}", status_paragraph_style))
    if highest_conf_detection:
        story.append(Paragraph(
            f"Highest Confidence Detection: {highest_conf_detection['class']} "
            f"({highest_conf_detection['confidence'] * 100:.0f}%)",
            styles["Normal"],
        ))
    story.append(Spacer(1, 6 * mm))

    if payload.annotated_image and "," in payload.annotated_image:
        try:
            _, b64data = payload.annotated_image.split(",", 1)
            img_bytes = base64.b64decode(b64data)
            pil_img = Image.open(io.BytesIO(img_bytes))
            img_w, img_h = pil_img.size
            max_w = 150 * mm
            scale = max_w / img_w
            img_buffer = io.BytesIO(img_bytes)
            story.append(RLImage(img_buffer, width=max_w, height=img_h * scale))
            story.append(Spacer(1, 6 * mm))
        except Exception:
            pass

    if class_counts:
        story.append(Paragraph("Defect Class Breakdown", summary_style))
        table_data = [["Defect Class", "Count"]] + [[cls, str(count)] for cls, count in class_counts.items()]
        table = Table(table_data, colWidths=[90 * mm, 40 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(table)

    doc.build(story)
    buffer.seek(0)

    safe_name = os.path.splitext(payload.filename)[0] or "inspection"
    pdf_filename = f"{safe_name}_report_{int(time.time())}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{pdf_filename}"'},
    )
BASE_DIR = Path(__file__).resolve().parent
