# Aircraft Surface Defect Detection System

An end-to-end computer-vision inspection system **intended for aircraft parts**.
It detects surface defects — **cracks, dents, corrosion, scratches, missing
rivet/fastener heads, and paint peel-off** — from images, batches of images,
video, or a live camera feed, and produces a downloadable PDF inspection
report.

It is served through a **FastAPI** backend and a **React** dashboard frontend,
branded as an "HAL Inspection System" demo/prototype for maintenance
inspection workflows.

> **Where it stands today.** The system was built with aircraft parts
> inspection as the end goal, but the models currently shipped are validated
> on **metal sheets and panels** (the same class of surface as aircraft skin),
> and a second model detects **everyday objects** for pipeline demos. Real
> aircraft-part imagery has not yet been used for training or validation —
> see [Intended use vs. current scope](#intended-use-vs-current-scope) and the
> [Roadmap](#roadmap-to-aircraft-parts).

> **Status: local/LAN demo.** There is no authentication, no database, and no
> persistence — inspections are stateless and results live only in the browser
> session (plus whatever PDFs you download). Do not expose the backend to the
> public internet.

---

## Table of Contents

- [What it does](#what-it-does)
- [Intended use vs. current scope](#intended-use-vs-current-scope)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Using the dashboard](#using-the-dashboard)
- [API reference](#api-reference)
- [Models and defect classes](#models-and-defect-classes)
- [Known limitations](#known-limitations)
- [Synthetic data generator](#synthetic-data-generator)
- [Training pipeline](#training-pipeline)
- [Metrics & benchmarking](#metrics--benchmarking)
- [Roadmap to aircraft parts](#roadmap-to-aircraft-parts)
- [Technical details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Repository notes](#repository-notes)

---

## What it does

Manual visual inspection of metal surfaces — aircraft skin, sheet-metal panels,
machined parts — for damage is slow and depends on inspector attentiveness.
This project prototypes an automated assistant:

1. **Upload or capture** a photo of a panel or part, a batch of photos, a
   walkaround video (`.mp4` / `.avi`), or a live camera stream.
2. The backend runs the selected detection model and returns **per-detection
   class + confidence**, plus an **annotated image** with bounding boxes drawn.
3. The dashboard shows summary stats (total defects, top class, PASS/FAIL
   surface-integrity status) and a one-click **PDF inspection report** — the
   kind of artifact that would be attached to a maintenance log.

Four inspection modes are supported: **Single Image**, **Batch Images**,
**Video** (frame-sampled and aggregated), and **Live Feed** (camera frames
streamed over WebSocket at roughly 1 frame/second).

---

## Intended use vs. current scope

| | Intended (end goal) | Current reality |
|---|---|---|
| **Target subjects** | Aircraft parts: skin panels, fasteners, control surfaces | **Metal sheets and panels** (flat brushed-metal surfaces) |
| **Defect classes** | Six surface-defect classes (same taxonomy) | The same six classes, demonstrated on metal sheets: `crack`, `dent`, `corrosion`, `scratch`, `missing-head`, `paint-peel-off` |
| **Detector** | Custom-trained YOLOv8 on real aircraft-part imagery | Classical-CV heuristics tuned on **synthetic metal panels**; the custom YOLO checkpoint (`best.pt`) is under-trained and **not usable in production** (see [Known limitations](#known-limitations)) |
| **Everyday objects** | — | A `demo` model (stock COCO YOLOv8n) detects **everyday objects** — person, chair, bottle, laptop, … — so the full pipeline can be exercised end-to-end without any metal/aircraft imagery |
| **Validation data** | Real labeled aircraft inspection photos | 60 procedurally generated metal panels with ground truth (`backend/bench_panels/`) |

**Why two models?** The workflow, API, report, and dashboard are
subject-agnostic: whatever detects "things of interest" and exposes a
YOLO-compatible interface drops straight in. That decoupling is deliberate —
it lets the pipeline be developed and benchmarked today (on metal sheets and
even everyday objects) and lets a future aircraft-trained model be swapped in
without touching a single endpoint. The models section below shows exactly
where that swap happens.

---

## Architecture

```
┌──────────────────────┐   HTTP multipart/JSON   ┌────────────────────────┐
│  React Dashboard      │ ──────────────────────▶ │  FastAPI Backend        │
│  (Vite + React 19)    │ ◀────────────────────── │  (Python)               │
│                       │   JSON / base64 / PDF   │                         │
│  localhost:5173       │ ──── binary frames ───▶ │  WS /ws/live            │
└──────────────────────┘                          └───────────┬────────────┘
                                                              │
                                              ┌───────────────┴───────────────┐
                                              │ MODEL_REGISTRY (loaded eager)  │
                                              │  defect → CvDefectDetector     │
                                              │    (metal-sheet surface        │
                                              │     defects, classical CV;     │
                                              │     falls back to best.pt)     │
                                              │  demo   → YOLOv8n (COCO)       │
                                              │    (everyday objects)          │
                                              └────────────────────────────────┘
```

- **Frontend** (`frontend/`) — React 19 + Vite single-page dashboard. Model
  switcher, four inspection modes, summary stats, scope banners, and PDF
  report download. Also ships a static marketing/landing page
  (`frontend/public/landing.html`, three.js hero).
- **Backend** (`backend/main.py`) — FastAPI service that loads the model
  registry at startup and exposes REST + WebSocket endpoints for prediction
  and PDF report generation.
- **Training** (`training/`) — offline scripts (not part of the running app)
  used to build/expand training datasets and fine-tune the YOLO model — the
  path to true aircraft-part detection.

### How the project is built (components)

What ships in the repo and how the pieces relate:

```
┌──────────────────────────────┐         ┌───────────────────────────────────────┐
│  frontend/ — React 19 + Vite │         │  backend/ — FastAPI (single module)   │
│                              │         │                                       │
│  src/App.jsx                 │  HTTP   │  main.py                              │
│   ├ Single Image mode        │ ──────▶ │   ├ RuntimeMetrics (in-memory stats)  │
│   ├ Batch Images mode        │  JSON/  │   ├ MODEL_REGISTRY (eager-loaded)     │
│   ├ Video mode               │  base64 │   ├ REST  /predict  /predict-batch    │
│   └ Live Feed (camera)       │ ◀────── │   │       /predict-video              │
│                              │  PDF/WS │   ├ REST  /generate-report → PDF      │
│  axios · lucide-react        │         │   │       /metrics/system             │
│                              │         │   └ WS    /ws/live (live camera)      │
│  public/landing.html         │         │                                       │
│  (static marketing page)     │         │  cv_detector.py   classical-CV engine │
└──────────────────────────────┘         │  best.pt · yolov8n.pt   YOLO weights  │
                                         │  synth_defects.py   panel generator   │
┌──────────────────────────────┐         │  evaluate/benchmark   metrics tooling │
│  training/ — offline scripts │         └───────────────────────────────────────┘
│   merge_yolo_datasets.py     │
│   finetune.py                │   → writes the .pt weights MODEL_REGISTRY loads
└──────────────────────────────┘
```

### How it works (inspection request lifecycle)

The common lifecycle shared by all four modes — batch loops this per file,
video loops it per sampled frame, and Live Feed streams frames over the
`/ws/live` WebSocket instead of `POST /predict`:

```
 Inspector         Dashboard (React)            Backend (FastAPI)              Model
     │                    │                           │                          │
     │ pick model,        │                           │                          │
     │ upload/capture     │                           │                          │
     │───────────────────▶│                           │                          │
     │                    │  POST /predict (multipart)│                          │
     │                    │──────────────────────────▶│  decode image            │
     │                    │                           │───── inference ─────────▶│
     │                    │                           │◀ detections + drawn frame│
     │                    │  JSON: detections + base64│                          │
     │                    │  annotated image          │                          │
     │ view results,      │◀──────────────────────────│                          │
     │ stats, PASS/FAIL   │                           │                          │
     │◀───────────────────│                           │                          │
     │ Download report    │                           │                          │
     │───────────────────▶│  POST /generate-report    │                          │
     │                    │  (detections echoed back) │                          │
     │                    │──────────────────────────▶│  ReportLab builds PDF    │
     │ save inspection PDF│◀──────────────────────────│                          │
     │◀───────────────────│        application/pdf    │                          │
```

### How the model is built (training → serving loop)

How the detection model is produced and how it gets served:

```
  Roboflow datasets ──┐
                      ├─▶ merge_yolo_datasets.py ──▶ merged_dataset/ ──▶ finetune.py
  synthetic panels ───┘     (reconcile class names,                              │
  (synth_defects.py)         YOLO label format)                                   ▼
                                                                        retrained YOLO .pt
                                                                                  │
        ┌─────────────────────────────────────────────────────────────────────────┘
        ▼
  MODEL_REGISTRY (backend/main.py — loaded eagerly at startup)
  ├─ defect → CvDefectDetector (classical CV, metal sheets; falls back to best.pt)
  └─ demo   → yolov8n.pt       (everyday objects, COCO)

  Every registered model speaks the same YOLO-like interface
  (names + __call__ → results with .boxes and .plot()),
  so swapping in a future aircraft-trained checkpoint changes no endpoint code.
```

---

## Tech stack

| Layer | Technology | Notes |
|---|---|---|
| Backend | Python, FastAPI, Uvicorn, WebSockets | single module, `backend/main.py` |
| Vision | Ultralytics YOLOv8, OpenCV (headless), Pillow, NumPy | OpenCV also powers the classical-CV defect detector |
| Reports | ReportLab | A4 PDF with pass/fail status, class table, annotated image |
| Frontend | React 19, Vite, axios, lucide-react | plain CSS design system (`src/App.css`) |
| Landing page | Static HTML + vendored three.js | `frontend/public/landing.html` |
| Training | ultralytics, roboflow, pyyaml | offline only |

No database, message queue, cache, or auth provider is used.

---

## Project structure

```
.
├── README.md
├── backend/
│   ├── main.py                        # FastAPI app: all 8 endpoints + PDF report
│   ├── cv_detector.py                 # Classical-CV surface-defect detector (YOLO-compatible interface)
│   ├── synth_defects.py               # Procedural synthetic metal-panel generator
│   ├── evaluate_metrics.py            # mAP / precision / recall / F1 evaluation harness
│   ├── benchmark_system.py            # HTTP latency + WebSocket stream benchmark
│   ├── project_scale_metrics.py       # LOC / endpoint / component counters
│   ├── build_final_metrics_report.py  # Merges metric JSONs into one report
│   ├── requirements.txt
│   ├── best.pt                        # Custom YOLO checkpoint (under-trained, see limitations)
│   ├── yolov8n.pt                     # Stock COCO-pretrained YOLOv8n (everyday-object demo model)
│   ├── bench_panels/                  # 60 synthetic metal panels + ground-truth JSON (eval set)
│   ├── sample_panels/                 # Sample images for manual testing
│   └── metrics/                       # Generated metric JSON artifacts
├── frontend/
│   ├── src/App.jsx                    # Dashboard (all four inspection modes)
│   ├── src/App.css, index.css         # Styling + design tokens
│   ├── public/landing.html            # Static "DefectVision" landing page
│   ├── public/vendor/                 # Vendored three.js for the landing page
│   └── vite.config.js
├── aircraft-defect-system/            # ⚠ Experimental TypeScript/shadcn frontend scaffold
│   └── ...                            #    (design system only — no features yet, not wired up)
└── training/
    ├── merge_yolo_datasets.py         # Merge Roboflow YOLO datasets + remap classes
    └── finetune.py                    # Fine-tune best.pt on a merged dataset
```

---

## Getting started

### Prerequisites

- **Python 3.10+** (the benchmark run in `backend/metrics/` used Python 3.14 on Windows 11)
- **Node.js 20.19+ or 22.12+** (required by Vite 8) with npm
- A webcam, **only if** you want the Live Feed mode (browsers expose camera
  capture only in a secure context — `localhost` or HTTPS)

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- The first startup loads both models eagerly (downloading `yolov8n.pt` is not
  needed — it is committed — but the Ultralytics package will want Torch, which
  pip resolves from `requirements.txt`).
- `--host 0.0.0.0` makes the API reachable from other devices on your LAN; use
  `127.0.0.1` if you want it local-only.
- Interactive API docs: <http://localhost:8000/docs>

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed URL (typically <http://localhost:5173>). The dashboard
**automatically targets the backend at `http://<current-hostname>:8000`**, so:

- On `localhost` it hits `http://localhost:8000`.
- If you open the dev server from another device on the LAN, it hits
  `http://<that-machine's-LAN-IP>:8000` — keep the backend bound to `0.0.0.0`
  for this to work.

### 3. Landing page (optional)

The static product page is served by Vite at
<http://localhost:5173/landing.html>. Its "Launch Live Feed" buttons deep-link
into the dashboard via `/?mode=live`.

### Quick self-check

```bash
curl http://localhost:8000/
# {"status":"Backend is up and running!"}
```

### First-run suggestion

Point the dashboard at a sample metal panel to see the defect pipeline work:

```bash
# sample panels ship with the repo (backend/sample_panels/)
```

…or switch to the **General Object Demo** model and analyze any photo (a
chair, a bottle, a person) to see the same pipeline detect everyday objects.

---

## Using the dashboard

The top bar shows the active model; switch models with the **Active Model**
tabs. A **scope banner** under the tabs always states what the selected model
is (and is not) trained to detect — e.g. the surface-defect model is scoped to
metal-sheet defects and will (correctly) report no defects on photos outside
its scope, and the demo model detects everyday objects, not defects.

| Mode | How to use | What you get |
|---|---|---|
| **Single Image** | Click the dropzone, pick one image, press *Analyze Image* | Annotated image with bounding boxes, detection list with confidence %, summary stats, PDF report button |
| **Batch Images** | Pick multiple images, press *Analyze N Images* | Per-image results in one request; unreadable files are reported individually without failing the batch |
| **Video** | Pick an `.mp4`/`.avi`, press *Analyze Video* | Every N-th frame is analyzed (default N=5); returns total vs. analyzed frame counts, aggregated class counts, per-frame detections, and the single best (highest-confidence) annotated frame. Large videos can take minutes — the UI warns you. |
| **Live Feed** | *Enable Camera* → *Start Inspection* | Camera preview + live annotated frames + running session class counts. Frames are captured roughly every second. |

**PASS/FAIL logic:** with the surface-defect model selected, the surface
status is `FAIL` when the total detection count is greater than zero, `PASS`
otherwise.

**PDF report:** every result view has a *Download Inspection Report* button.
It sends the current detections + annotated image back to the backend and
downloads an A4 PDF containing the source filename, generation timestamp,
total defects, colored PASS/FAIL status, highest-confidence detection, the
annotated image, and a defect-class breakdown table.

---

## API reference

Base URL: `http://localhost:8000` (or your LAN IP). All endpoints are
unauthenticated. `model` is always a query parameter: `defect` (default) or
`demo`.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check. |
| `/models` | GET | Lists available models with their class names (drives the dashboard's model tabs and scope banner). |
| `/predict` | POST | Detection on a single uploaded image. Returns detections + base64 annotated image. |
| `/predict-batch` | POST | Same, for multiple images in one request. Per-file errors are returned inline, not as HTTP errors. |
| `/predict-video` | POST | Samples an uploaded `.mp4`/`.avi` every `frame_stride` frames (default 5), aggregates class counts, returns the highest-confidence annotated frame. |
| `/ws/live` | WS | Live stream: client sends binary JPEG frames, server replies with `{detections, annotated_image, latency_ms}` per frame. |
| `/metrics/system` | GET | Runtime snapshot: per-endpoint avg/p95 HTTP latency + WebSocket session/frame stats (in-memory, resets on restart). |
| `/generate-report` | POST | Takes `{filename, detections, annotated_image}` and returns a generated PDF inspection report. |

### Query parameters (predict endpoints)

| Param | Applies to | Default | Meaning |
|---|---|---|---|
| `model` | all | `defect` | `defect` (surface defects) or `demo` (everyday objects) |
| `include_annotated` | `/predict`, `/predict-batch`, `/predict-video`, `/ws/live` | `true` | Set `false` to skip base64 annotated frames — smaller payloads, faster benchmarking |
| `frame_stride` | `/predict-video` | `5` | Analyze every Nth frame |

### Examples

```bash
# Single image, surface-defect model, no annotated image back
curl -X POST "http://localhost:8000/predict?model=defect&include_annotated=false" \
  -F "file=@backend/sample_panels/panel_000.jpg"

# Everyday-object demo on any photo
curl -X POST "http://localhost:8000/predict?model=demo" -F "file=@desk.jpg"

# Batch
curl -X POST "http://localhost:8000/predict-batch?model=defect" \
  -F "files=@a.jpg" -F "files=@b.jpg"

# Video, sample every 10th frame
curl -X POST "http://localhost:8000/predict-video?frame_stride=10" -F "file=@walkaround.mp4"

# PDF report (detections from a previous /predict response)
curl -X POST "http://localhost:8000/generate-report" \
  -H "Content-Type: application/json" \
  -d '{"filename":"panel_000.jpg","detections":[{"class":"crack","confidence":0.72}],"annotated_image":null}'
```

Response shape for `/predict`:

```json
{
  "filename": "panel_000.jpg",
  "detections": [{"class": "crack", "confidence": 0.72}],
  "annotated_image": "data:image/jpeg;base64,..."
}
```

Error behavior: unknown `model` → `400` with the valid options; disallowed
video extension or unopenable video → `400`; a corrupt image in `/predict`
currently surfaces as a `500` (see [Known limitations](#known-limitations)).

---

## Models and defect classes

Two models are registered and loaded eagerly at startup
(`MODEL_REGISTRY` in `backend/main.py`):

| Key | Dashboard label | What it actually detects | Implementation |
|---|---|---|---|
| `defect` | "Aircraft Defect Model" | **Surface defects on metal sheets/panels** — the six classes below. Tuned for flat brushed-metal surfaces (procedurally generated panels and similar real metal sheets); best-effort on arbitrary photos. | **`CvDefectDetector`** (`backend/cv_detector.py`) — deterministic classical-CV pipeline, **no ML weights, no dataset**; if its construction ever fails it silently falls back to YOLO `best.pt` |
| `demo` | "General Object Demo" | **Everyday objects** (COCO classes: person, chair, bottle, laptop, …) — useful to exercise the full pipeline end-to-end without metal/aircraft imagery | `YOLO("yolov8n.pt")` — stock COCO-pretrained YOLOv8n |

> The dashboard label "Aircraft Defect Model" reflects the project's intent;
> the honest scope of what ships today is metal-sheet surface defects. The
> scope banner in the UI spells this out on every screen so a clean result is
> never misread as a clean bill of health for an actual aircraft part.

Both expose the same minimal YOLO-like interface (`names` + callable returning
results with `.boxes` and `.plot()`), so every endpoint — batch, video,
WebSocket, report — works unchanged with either, and a future aircraft-trained
YOLO checkpoint drops in the same way (see
[Roadmap](#roadmap-to-aircraft-parts)).

**Defect classes** (surface-defect taxonomy):
`crack`, `dent`, `corrosion`, `scratch`, `missing-head` (missing
rivet/fastener head), `paint-peel-off`.

The classical-CV engine, per class:

| Class | Detection principle |
|---|---|
| crack | morphological black-hat + elongation filter |
| scratch | morphological white-hat + elongation filter |
| dent | local-background dark blob, roundish |
| corrosion | HSV rust-hue segmentation |
| missing-head | dark small circular blob |
| paint-peel-off | brightness threshold with jagged-boundary check |

Confidences are heuristic (clipped to `[0.15, 0.98]`) and detections pass a
per-class NMS (IoU 0.35) with a 0.30 confidence threshold. You can run the
detector standalone:

```bash
cd backend
python cv_detector.py --image bench_panels/panel_000.jpg --conf 0.30
```

---

## Known limitations

Be honest with yourself about these before trusting any result:

1. **Not yet validated on real aircraft parts.** No real aircraft-part imagery
   has been used for training or validation. The defect detector works on
   metal sheets/panels; treat any result on an actual aircraft part as
   indicative only.
2. **`best.pt` is not a usable detector.** The committed checkpoint was
   trained for only 3 epochs on 5% of the dataset (`fraction: 0.05`) at 256px
   (metadata baked into the weights). Recorded validation metrics: precision
   0.2%, recall 25%, mAP50 1.5%, mAP50-95 0.4%. It does not reliably fire on
   real defect photos (confirmed by testing at confidence thresholds down to
   0.01 with zero detections on real-world corrosion/paint-peel imagery).
   That is why the `defect` key runs the classical-CV engine instead.
   `training/finetune.py` is already set up to retrain it properly once real
   labeled data is available.
3. **The classical-CV engine is tuned to synthetic metal panels.** The
   benchmark numbers in `backend/metrics/` are measured on panels generated by
   `synth_defects.py` — i.e., fit-to-synthetic, not real-world performance.
4. **No persistence or audit trail.** Inspections, detections, and runtime
   metrics are held in memory only; everything resets on restart. Reports are
   generated from data echoed by the client, so the backend cannot vouch for
   them.
5. **No authentication / open CORS.** `allow_origins=["*"]` with
   `allow_credentials=True` and no rate limiting — appropriate only for a
   trusted local/LAN environment.
6. **Concurrency.** Inference in `/predict`, `/predict-batch`, and
   `/predict-video` runs inline on the event loop, so one long video upload
   can delay other requests (the WebSocket path correctly offloads inference
   to a worker thread). This is a single-user demo.
7. **No upload size limits.** Whole uploads are read into memory.

---

## Synthetic data generator

`backend/synth_defects.py` renders brushed-aluminum metal-sheet panels with
the six defect classes. Every image ships with ground truth in both YOLO
format (`.txt`) and JSON (`.json` + `manifest.json`), so the output can drive
end-to-end demos, the evaluation harness, and a future YOLO fine-tune.

```bash
cd backend
python synth_defects.py --out out_panels --count 20 --seed 0
python synth_defects.py --out out_panels --count 100 --defects dent,crack
python synth_defects.py --out out_panels --count 50 --size 640,480
```

The committed `backend/bench_panels/` set (60 panels, 112 defect instances)
was produced this way and is used as the validation set by
`evaluate_metrics.py`.

---

## Training pipeline

Offline scripts — not called by the running app; they produce the weights the
backend can load. This is also the path from metal sheets to real aircraft
parts (see [Roadmap](#roadmap-to-aircraft-parts)).

### 1. Merge datasets (`training/merge_yolo_datasets.py`)

Merges multiple Roboflow-exported YOLOv8 datasets into one unified dataset.
Because different public datasets label the same defect differently (e.g.
`paint-off` vs `paint-peel-off`), reconciling class names is deliberately a
two-step, human-in-the-loop process:

```bash
cd training
# Step 1 — inspect each dataset's classes and write a starter class_map.json
python merge_yolo_datasets.py scan path/to/dataset1 path/to/dataset2

# Step 2 — edit class_map.json by hand, mapping every source class name onto
#          one of the target classes (or "SKIP" to drop irrelevant ones)

# Step 3 — remap every label file and write merged data.yaml + train/valid/test
python merge_yolo_datasets.py build path/to/dataset1 path/to/dataset2 \
    --class-map class_map.json --out merged_dataset
```

Images that end up with zero labels after remapping are kept as background
examples (helps reduce false positives) rather than discarded.

### 2. Fine-tune (`training/finetune.py`)

Fine-tunes starting from the existing `best.pt` checkpoint (rather than from
generic COCO weights), so the model retains what it already learned and adapts
faster to newly merged data. Wraps `ultralytics.YOLO.train()` with configurable
epochs, image size, batch size, and early-stopping patience, then reports
validation mAP50-95 and writes a `training_summary.json` (epochs, duration,
hardware) into the run folder.

```bash
cd training
python finetune.py --data merged_dataset/data.yaml --weights ../backend/best.pt --epochs 100
```

---

## Metrics & benchmarking

All previously unavailable metrics can be regenerated as concrete JSON
artifacts:

1. **Model metrics** (mAP@0.5, mAP@0.5:0.95, per-class precision/recall/F1,
   inference speed, dataset size/split/class counts, correctly flagged
   instances) — evaluated against `bench_panels/`:

   ```bash
   cd backend
   python evaluate_metrics.py --val-dir bench_panels --output metrics/model_metrics.json
   ```

   Options include `--model defect|demo`, `--weights <custom .pt>`,
   `--iou-threshold`, `--confidence-threshold`.

2. **System metrics** (FastAPI latency avg/p95, WebSocket real-time FPS,
   concurrent-feed stress test, uptime %):

   ```bash
   cd backend
   python benchmark_system.py --output metrics/system_metrics.json
   ```

3. **Project scale metrics** (LOC, API endpoint count, React component count,
   latest training epochs/hardware/time from training summaries):

   ```bash
   cd backend
   python project_scale_metrics.py --root .. --output metrics/project_scale.json
   ```

4. **Final consolidated report:**

   ```bash
   cd backend
   python build_final_metrics_report.py --output metrics/final_metrics_report.json
   ```

If you have manual/baseline numbers, copy
`backend/metrics/impact_baseline.template.json` to
`backend/metrics/impact_baseline.json` before step 4 to automatically compute
manual-vs-automated time reduction and baseline-vs-current accuracy lift.

### Last recorded benchmark (from `backend/metrics/`, CPU, Python 3.14 / Win 11)

| Metric | Value |
|---|---|
| mAP@0.5 / mAP@0.5:0.95 (60 synthetic metal panels) | 0.5199 / 0.3059 |
| Inference speed | ~86 ms/frame avg (~11.7 FPS), p95 ~97 ms |
| `POST /predict` HTTP latency | ~93 ms avg, ~116 ms p95 |
| `POST /predict-batch` (3 images) | ~264 ms avg |
| WebSocket stream | ~12 FPS per feed, ~83 ms avg frame latency |
| Concurrent load | 4 simultaneous feeds, 100% frames successful |

Corrosion is effectively solved on synthetic panels (P/R/F1 = 1.0); missing
rivet heads are the weakest class (F1 ≈ 0.11).

---

## Roadmap to aircraft parts

The pipeline is aircraft-ready; the model is not — yet. The intended path:

1. **Collect real labeled imagery** of aircraft parts/skin — the six defect
   classes, photographed under realistic lighting and angles. Even a few
   hundred labeled images per class is a meaningful start.
2. **Merge datasets** with `training/merge_yolo_datasets.py` — combine the
   real aircraft data with the existing metal-sheet/Roboflow datasets,
   reconciling class names through the human-in-the-loop `class_map.json`.
3. **Fine-tune** with `training/finetune.py` (standard 640px input, enough
   epochs to converge, early stopping) starting from `best.pt` or fresh COCO
   weights.
4. **Validate on held-out real images** with `evaluate_metrics.py --weights`
   — *not* on synthetic panels — and publish the per-class P/R/F1 honestly.
5. **Swap it in**: place the new checkpoint in `backend/` and either make it
   the `best.pt` fallback or register it as a new key in `MODEL_REGISTRY`
   (`backend/main.py`) — no other code changes are needed thanks to the
   shared YOLO-compatible interface.
6. **Keep the scope banners truthful** — update the dashboard copy so the UI
   states exactly what the new model was validated on.

Everything else (API, dashboard, four inspection modes, PDF reports,
benchmarks) already works subject-agnostically and needs no changes in this
transition.

---

## Technical details

- **CORS is fully open** (`allow_origins=["*"]`, all methods/headers, with
  credentials enabled) since this is a local/LAN prototype, not a public
  deployment.
- **Annotated frames** are JPEG-encoded and transported as base64 data URLs
  (~33% size overhead). Pass `include_annotated=false` for lean payloads.
- **Video handling** uses a temp file on disk (OpenCV's `VideoCapture` needs a
  real path); the file is deleted in a `finally` block regardless of outcome.
  Only `.mp4`/`.avi` are accepted.
- **Live feed transport**: the dashboard first opens a WebSocket to
  `ws://<host>:8000/ws/live?model=<key>`; if the socket cannot be established
  it falls back to HTTP polling (`POST /predict`) on a ~900 ms interval. Only
  one frame is ever in flight (`inFlightRef` guard).
- **Runtime metrics** (`/metrics/system`) are ring buffers (500 samples) in
  memory, guarded by a lock — they reset when the process restarts.
- **PDF reports** are built with ReportLab into an in-memory buffer and
  streamed back as an attachment named `<source>_report_<unix-time>.pdf`.
- **Model loading is eager and fail-fast**: a missing/broken `demo` model
  crashes startup; a broken classical-CV detector silently falls back to the
  (under-trained) `best.pt`.
- **No bounding boxes are returned over the API** — only class + confidence
  per detection, plus the rendered annotated image. Coordinates exist only
  inside the drawn JPEG.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Dashboard says *"Failed to connect to backend server"* | Backend isn't running on port 8000, or you opened the dashboard from another device while the backend is bound to `127.0.0.1`. Start it with `--host 0.0.0.0`. |
| Defect model reports nothing on a photo of, say, a person or a room | Expected — the surface-defect model is scoped to metal-sheet defects. Switch to the **General Object Demo** model for everyday objects. |
| Live Feed: *"Camera capture needs a secure context"* | Browsers only expose `getUserMedia` on `localhost` or HTTPS. Open the dashboard via `http://localhost:5173`, or serve it over HTTPS. |
| Live Feed shows only HTTP fallback in the meta line | The WebSocket handshake failed (proxy/firewall) — polling still works. |
| *Analyzing Video* takes minutes | Expected: inference runs per sampled frame. Increase `frame_stride` (or use shorter clips). |
| "Could not read image" inside a batch result | That one file was unreadable; the rest of the batch still processed. |
| `500` on `/predict` for a valid-looking image | Unsupported/corrupt format — `/predict` does not currently guard decode errors (batch mode does). |
| Model tabs missing / empty classes | `GET /models` failed — check backend logs and CORS. |
| GPU not used | Torch CPU build installed by default on Windows; install a CUDA build of Torch if you need GPU inference (CPU is ~86 ms/frame, fine for this demo). |

---

## Repository notes

- `backend/best.pt` (6.2 MB) and `backend/yolov8n.pt` (6.5 MB) are committed so
  the app runs out of the box; expect the repo to be ~15 MB.
- `aircraft-defect-system/` is a **second, experimental frontend scaffold**
  (TypeScript + shadcn/ui + Tailwind v4). It currently contains only the design
  system and a placeholder screen — it is not wired to the backend and is not
  part of the running product.
- There are **no automated tests, no CI/CD, and no containerization** yet.
  `backend/benchmark_system.py` (FastAPI `TestClient` smoke benchmark) is the
  closest thing to a regression check.
- `requirements.txt` is intentionally unpinned for demo convenience; pin before
  any serious use.
