# Aircraft Surface Defect Detection System

An end-to-end computer vision system that detects surface defects on aircraft
(cracks, dents, corrosion, scratches, missing rivet/fastener heads, and paint
peel-off) from images, batches of images, or video, using a custom-trained
YOLOv8 model served through a FastAPI backend and a React dashboard frontend.

The dashboard is branded as an "HAL Inspection System" (HAL = Hindustan
Aeronautics Limited) demo/prototype for aircraft maintenance inspection
workflows.

## What problem this solves

Manual visual inspection of aircraft skin for surface damage is slow and
depends on inspector attentiveness. This project prototypes an automated
assistant: upload a photo (or a walkaround video) of an aircraft panel, and
the system localizes any defects with bounding boxes, a confidence score per
detection, and a downloadable PDF inspection report — the kind of artifact
that would be attached to a maintenance log.

## Architecture

```
┌─────────────────┐      HTTP (multipart/JSON)      ┌───────────────────────┐
│  React Frontend  │ ───────────────────────────────▶ │   FastAPI Backend     │
│  (Vite + React)  │ ◀─────────────────────────────── │   (Python)            │
└─────────────────┘        JSON / PDF response        └───────────┬───────────┘
                                                                   │
                                                        ┌──────────┴──────────┐
                                                        │   YOLOv8 (Ultralytics)│
                                                        │  best.pt (custom)     │
                                                        │  yolov8n.pt (demo)    │
                                                        └───────────────────────┘
```

- **Frontend** (`frontend/`) — React 19 + Vite single-page dashboard. Lets a
  user pick a model, then upload a single image, a batch of images, or a
  video, and renders the annotated detections, summary stats, and a
  "Download Inspection Report" PDF button.
- **Backend** (`backend/main.py`) — FastAPI service that loads the YOLO
  model(s) at startup and exposes REST endpoints for prediction and PDF
  report generation.
- **Training** (`training/`) — offline scripts (not part of the running app)
  used to build/expand the training dataset and fine-tune the model.

## Backend details (`backend/`)

Built with **FastAPI**, **Ultralytics YOLOv8**, **OpenCV**, **Pillow**, and
**ReportLab**. Two models are registered and loaded eagerly on startup:

| Key      | Weights file  | Purpose                                                                 |
|----------|---------------|--------------------------------------------------------------------------|
| `defect` | `best.pt`     | Custom-trained aircraft surface defect detector (6 classes, see below). |
| `demo`   | `yolov8n.pt`  | Stock COCO-pretrained YOLOv8n, used to demo the pipeline end-to-end on everyday objects when real aircraft imagery isn't on hand. |

**Defect classes** (`best.pt`): `crack`, `dent`, `corrosion`, `scratch`,
`missing-head` (missing rivet/fastener head), `paint-peel-off`.

> **Known limitation — `best.pt` is not yet a validated detector.** The
> checkpoint currently in this repo was trained for only 3 epochs on 5% of
> the dataset (`fraction: 0.05`) at 256px, per the training metadata baked
> into the weights file. Recorded validation metrics: precision 0.2%,
> recall 25%, mAP50 1.5%, mAP50-95 0.4%. In practice it does not reliably
> fire on real aircraft defect photos (confirmed by testing at confidence
> thresholds down to 0.01 with zero detections on real-world corrosion/
> paint-peel imagery). Treat the "Aircraft Defect Model" as a pipeline
> demonstration, not a working detector, until it's retrained with the full
> dataset, standard image size (640px), and enough epochs to converge —
> `training/finetune.py` is already set up to do this once real labeled
> data is available.

### API endpoints

| Endpoint            | Method | Description                                                                 |
|----------------------|--------|-------------------------------------------------------------------------------|
| `/`                  | GET    | Health check.                                                                  |
| `/models`            | GET    | Lists available models and each one's class names (drives the frontend's model tabs and scope banner). |
| `/predict`           | POST   | Runs detection on a single uploaded image. Returns detections + a base64 annotated image. |
| `/predict-batch`     | POST   | Same as above, for multiple images in one request.                            |
| `/predict-video`     | POST   | Samples an uploaded `.mp4`/`.avi` every N frames (`frame_stride`, default 5), aggregates class counts across frames, and returns the highest-confidence annotated frame. |
| `/generate-report`   | POST   | Takes detections + annotated image and returns a generated PDF inspection report (pass/fail status, defect class breakdown table, annotated image). |
| `/metrics/system`    | GET    | Runtime latency/throughput snapshot: per-endpoint avg/p95 HTTP latency + websocket streaming frame stats. |

`/predict`, `/predict-batch`, `/predict-video`, and `/ws/live` also support
`include_annotated=false` to skip base64 annotated frames for faster
benchmarking and lower payload size.

CORS is fully open (`allow_origins=["*"]`) since this is a local/LAN
prototype, not a public deployment.

## Frontend details (`frontend/`)

React 19 + Vite app (`src/App.jsx`) with three inspection modes:

- **Single Image** — upload one photo, view bounding-box overlay + defect list.
- **Batch Images** — upload multiple photos, get per-image results in one pass.
- **Video** — upload a walkaround video; frames are sampled and aggregated
  into a class-count summary plus the best (highest-confidence) frame.

Other UI features:
- **Model switcher** — toggle between the real "Aircraft Defect Model" and
  the "General Object Demo" model, with a scope banner clarifying what each
  model is (and isn't) trained to detect, so results aren't misread as a
  clean bill of health outside the model's actual scope.
- **Summary stats** — total defects, top detected class, and a PASS/FAIL
  surface-integrity status.
- **PDF report download** — calls `/generate-report` and downloads the
  resulting inspection report.

Uses `axios` for API calls and `lucide-react` for icons. The frontend talks
to the backend at `http://<current-hostname>:8000`, so it works whether
you're running it on `localhost` or accessing it from another device on the
same LAN.

## Training pipeline (`training/`)

Two offline scripts used to build and improve the custom `best.pt` model —
these are not called by the running app, they're the tooling used to produce
the weights the backend loads:

- **`merge_yolo_datasets.py`** — merges multiple Roboflow-exported YOLOv8
  datasets into one unified dataset. Since different public datasets label
  the same defect with different names (e.g. `paint-off` vs.
  `paint-peel-off`), this is a two-step, human-in-the-loop process:
  1. `scan` — inspects each dataset's classes and writes a starter
     `class_map.json`.
  2. A person edits that JSON to map every source class name onto one of the
     model's target classes (or `"SKIP"` to drop irrelevant classes like
     `screw` or `oil-stain`).
  3. `build` — remaps every label file accordingly and writes a merged
     `data.yaml` + combined `train/valid/test` image/label folders. Images
     that end up with zero labels after remapping are kept as background
     examples (helps reduce false positives) rather than discarded.

- **`finetune.py`** — fine-tunes the model starting from the existing
  `best.pt` checkpoint (rather than training from scratch on generic COCO
  weights), so the model retains what it already learned and adapts faster
  to newly merged data. Wraps `ultralytics.YOLO.train()` with configurable
  epochs, image size, batch size, and early-stopping patience, then reports
  validation mAP50-95 and writes `training_summary.json` (epochs completed,
  train duration, mAP, hardware metadata) into the run folder.

## Resume/report metrics pipeline (new)

All unavailable metrics are now generated as concrete JSON artifacts:

1. **Model metrics** (mAP@0.5, mAP@0.5:0.95, per-class precision/recall/F1,
   inference speed, dataset size/split/class counts, correctly flagged
   instances):
   ```bash
   cd backend
   python evaluate_metrics.py --val-dir bench_panels --output metrics/model_metrics.json
   ```

2. **System metrics** (FastAPI latency avg/p95, websocket real-time FPS, stress
   test with concurrent feeds, uptime %):
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

4. **Final consolidated report**:
   ```bash
   cd backend
   python build_final_metrics_report.py --output metrics/final_metrics_report.json
   ```

If you have manual/baseline numbers, copy
`backend/metrics/impact_baseline.template.json` to
`backend/metrics/impact_baseline.json` before step 4 to automatically compute
manual-vs-automated time reduction and baseline-vs-current accuracy lift.

## Running the project locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open the Vite dev server URL in a browser. The frontend will call the
backend automatically at port 8000 on the same host.

## Summary of work done

1. Trained a custom YOLOv8 object detection model (`best.pt`) to recognize
   six aircraft surface defect classes, using a dataset built by merging
   multiple public (Roboflow) defect datasets with a reconciled, canonical
   class taxonomy.
2. Built a FastAPI backend that serves this model (plus a stock COCO model
   for demo purposes) over REST, supporting single-image, batch-image, and
   video inference, and generates formatted PDF inspection reports from the
   results.
3. Built a React dashboard providing an inspector-facing UI for all three
   inspection modes, with clear messaging about each model's detection
   scope, summary statistics, and one-click PDF report download.
4. Wrote reusable dataset-merging and fine-tuning tooling so the model can be
   iteratively improved as more labeled data becomes available, without
   retraining from scratch each time.
