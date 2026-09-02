"""Fine-tune the existing aircraft-defect model on a merged dataset.

Usage:
  python finetune.py --data merged_dataset/data.yaml --weights ../backend/best.pt --epochs 100

Starting from the existing best.pt checkpoint (rather than a generic COCO
checkpoint) means the model keeps whatever it already learned and just adapts
to the additional images, which converges faster on a small dataset than
training from scratch.
"""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

from ultralytics import YOLO


def latest_run_dir(project_dir: Path, name_prefix: str) -> Path | None:
    candidates = [p for p in project_dir.glob(f"{name_prefix}*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_completed_epochs(results_csv: Path) -> int | None:
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return len(rows) if rows else None


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "accelerator": "cpu",
        "gpu_name": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["accelerator"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to merged dataset's data.yaml")
    parser.add_argument("--weights", default="../backend/best.pt", help="Starting checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="Early-stop after N epochs with no improvement")
    parser.add_argument("--project", default="runs/finetune")
    parser.add_argument("--name", default="aircraft_defects")
    parser.add_argument("--summary-name", default="training_summary.json")
    args = parser.parse_args()

    model = YOLO(args.weights)
    started_at = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )
    ended_at = time.time()

    metrics = model.val()
    print(f"\nValidation mAP50-95: {metrics.box.map:.4f}")
    print(f"Best weights saved under {args.project}/{args.name}/weights/best.pt")

    run_dir = latest_run_dir(Path(args.project), args.name)
    completed_epochs = None
    if run_dir:
        completed_epochs = read_completed_epochs(run_dir / "results.csv")

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training": {
            "data": args.data,
            "starting_weights": args.weights,
            "epochs_configured": args.epochs,
            "epochs_completed": completed_epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "training_time_minutes": round((ended_at - started_at) / 60.0, 3),
            "run_dir": str(run_dir) if run_dir else None,
        },
        "validation": {
            "mAP@0.5": round(float(metrics.box.map50), 4),
            "mAP@0.5:0.95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "hardware": hardware_info(),
    }

    if run_dir:
        summary_path = run_dir / args.summary_name
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Training summary written to {summary_path}")


if __name__ == "__main__":
    main()
