import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from cv_detector import CvDefectDetector

IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]
CLASS_ALIASES = {
    "missing-head": "missing_rivet",
    "paint-peel-off": "paint_peel_off",
}


def normalized_class_name(name: str) -> str:
    return CLASS_ALIASES.get(name, name)


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def parse_manifest(split_dir: Path) -> tuple[list[str], list[dict]]:
    classes_path = split_dir / "classes.txt"
    manifest_path = split_dir / "manifest.json"
    if not classes_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Expected classes.txt and manifest.json in {split_dir}")

    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = manifest.get("images", [])
    return classes, images


def build_dataset_index(split_dirs: dict[str, Path]) -> tuple[list[str], dict[str, list[dict]], dict]:
    classes: list[str] | None = None
    split_images: dict[str, list[dict]] = {}
    split_stats = {}
    total_instances_per_class = defaultdict(int)
    total_images = 0

    for split, split_dir in split_dirs.items():
        local_classes, images = parse_manifest(split_dir)
        if classes is None:
            classes = local_classes
        elif classes != local_classes:
            raise ValueError(f"Class mismatch in split '{split}': {split_dir}")

        per_class = defaultdict(int)
        for image in images:
            for defect in image.get("defects", []):
                per_class[defect["class"]] += 1
                total_instances_per_class[defect["class"]] += 1

        split_images[split] = images
        split_stats[split] = {
            "images": len(images),
            "instances_per_class": {
                normalized_class_name(k): v for k, v in sorted(per_class.items())
            },
            "total_instances": int(sum(per_class.values())),
        }
        total_images += len(images)

    assert classes is not None
    dataset_stats = {
        "total_images": total_images,
        "splits": split_stats,
        "total_instances_per_class": {
            normalized_class_name(k): v for k, v in sorted(total_instances_per_class.items())
        },
    }
    return classes, split_images, dataset_stats


def get_model(model_key: str, weights_path: str | None) -> tuple[Any, str]:
    if weights_path:
        return YOLO(weights_path), f"weights:{weights_path}"
    if model_key == "defect":
        return CvDefectDetector(), "cv_detector"
    if model_key == "demo":
        return YOLO("yolov8n.pt"), "yolov8n.pt"
    raise ValueError("Choose --model defect|demo or provide --weights")


def infer_image(model: Any, image_path: Path, class_names: list[str]) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    results = model(image)
    preds = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [float(x) for x in box.xyxy[0].tolist()]
            preds.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "confidence": conf,
                    "bbox": xyxy,
                }
            )
    return preds


def evaluate_map(
    class_names: list[str],
    gt_by_image: dict[str, list[dict]],
    preds_by_image: dict[str, list[dict]],
) -> dict:
    gt_count_by_class = defaultdict(int)
    for gt_list in gt_by_image.values():
        for gt in gt_list:
            gt_count_by_class[gt["class_name"]] += 1

    per_class_ap = {name: {} for name in class_names}
    for iou_thr in IOU_THRESHOLDS:
        for class_name in class_names:
            pred_rows = []
            for image_key, pred_list in preds_by_image.items():
                for pred in pred_list:
                    if pred["class_name"] == class_name:
                        pred_rows.append((image_key, pred["confidence"], pred["bbox"]))
            pred_rows.sort(key=lambda row: row[1], reverse=True)

            gt_boxes_per_image = defaultdict(list)
            for image_key, gt_list in gt_by_image.items():
                gt_boxes_per_image[image_key] = [gt["bbox"] for gt in gt_list if gt["class_name"] == class_name]

            matched = {image_key: np.zeros(len(boxes), dtype=bool) for image_key, boxes in gt_boxes_per_image.items()}
            tp = np.zeros(len(pred_rows), dtype=np.float32)
            fp = np.zeros(len(pred_rows), dtype=np.float32)

            for idx, (image_key, _conf, pbox) in enumerate(pred_rows):
                gt_boxes = gt_boxes_per_image.get(image_key, [])
                if not gt_boxes:
                    fp[idx] = 1.0
                    continue

                ious = np.array([compute_iou(pbox, gbox) for gbox in gt_boxes], dtype=np.float32)
                best_i = int(np.argmax(ious))
                if ious[best_i] >= iou_thr and not matched[image_key][best_i]:
                    matched[image_key][best_i] = True
                    tp[idx] = 1.0
                else:
                    fp[idx] = 1.0

            total_gt = gt_count_by_class[class_name]
            if total_gt == 0:
                per_class_ap[class_name][f"{iou_thr:.2f}"] = None
                continue

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / max(total_gt, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            per_class_ap[class_name][f"{iou_thr:.2f}"] = voc_ap(recall, precision)

    map50_values = [per_class_ap[name]["0.50"] for name in class_names if per_class_ap[name]["0.50"] is not None]
    map5095_values = []
    for name in class_names:
        vals = [v for v in per_class_ap[name].values() if v is not None]
        if vals:
            map5095_values.append(float(np.mean(np.array(vals, dtype=np.float32))))
    return {
        "mAP@0.5": round(float(np.mean(np.array(map50_values, dtype=np.float32))), 4) if map50_values else None,
        "mAP@0.5:0.95": round(float(np.mean(np.array(map5095_values, dtype=np.float32))), 4) if map5095_values else None,
        "per_class_ap": per_class_ap,
    }


def evaluate_prf1(
    class_names: list[str],
    gt_by_image: dict[str, list[dict]],
    preds_by_image: dict[str, list[dict]],
    iou_threshold: float,
    conf_threshold: float,
) -> dict:
    per_class = {}
    totals = {}
    for class_name in class_names:
        tp = 0
        fp = 0
        fn = 0
        for image_key, gt_list in gt_by_image.items():
            gt_boxes = [g["bbox"] for g in gt_list if g["class_name"] == class_name]
            pred_boxes = [
                p for p in preds_by_image.get(image_key, [])
                if p["class_name"] == class_name and p["confidence"] >= conf_threshold
            ]
            pred_boxes.sort(key=lambda p: p["confidence"], reverse=True)

            matched = np.zeros(len(gt_boxes), dtype=bool)
            for pred in pred_boxes:
                if not gt_boxes:
                    fp += 1
                    continue
                ious = np.array([compute_iou(pred["bbox"], gt) for gt in gt_boxes], dtype=np.float32)
                best_i = int(np.argmax(ious))
                if ious[best_i] >= iou_threshold and not matched[best_i]:
                    matched[best_i] = True
                    tp += 1
                else:
                    fp += 1
            fn += int((~matched).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        totals[class_name] = tp
    return {"per_class": per_class, "correctly_flagged_instances": totals, "total_correctly_flagged": int(sum(totals.values()))}


def benchmark_inference(
    model: Any,
    eval_images: list[Path],
    class_names: list[str],
    warmup: int,
    max_samples: int,
) -> dict:
    image_paths = eval_images[:max_samples]
    for path in image_paths[:warmup]:
        infer_image(model, path, class_names)

    times = []
    for path in image_paths:
        t0 = time.perf_counter()
        infer_image(model, path, class_names)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times, dtype=np.float32)
    avg_ms = float(arr.mean()) if arr.size else 0.0
    return {
        "samples": int(arr.size),
        "avg_ms_per_frame": round(avg_ms, 3),
        "p95_ms_per_frame": round(float(np.percentile(arr, 95)), 3) if arr.size else 0.0,
        "fps": round(1000.0 / avg_ms, 3) if avg_ms > 0 else 0.0,
    }


def get_hardware_info() -> dict:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate detector and generate quantifiable metrics JSON.")
    parser.add_argument("--train-dir", default=None, help="Path to split directory with classes.txt + manifest.json")
    parser.add_argument("--val-dir", default="bench_panels", help="Path to validation split directory")
    parser.add_argument("--test-dir", default=None, help="Path to test split directory")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--model", choices=["defect", "demo"], default="defect")
    parser.add_argument("--weights", default=None, help="Optional custom .pt weights path")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--speed-warmup", type=int, default=5)
    parser.add_argument("--speed-samples", type=int, default=100)
    parser.add_argument("--output", default="metrics\\model_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    split_dirs = {"val": Path(args.val_dir)}
    if args.train_dir:
        split_dirs["train"] = Path(args.train_dir)
    if args.test_dir:
        split_dirs["test"] = Path(args.test_dir)
    if args.eval_split not in split_dirs:
        raise ValueError(f"eval split '{args.eval_split}' not available from provided split dirs")

    class_names, split_images, dataset_stats = build_dataset_index(split_dirs)
    model, model_source = get_model(args.model, args.weights)

    eval_images = split_images[args.eval_split]
    gt_by_image: dict[str, list[dict]] = {}
    preds_by_image: dict[str, list[dict]] = {}
    eval_image_paths: list[Path] = []
    split_root = split_dirs[args.eval_split]

    for row in eval_images:
        image_name = row["file"]
        image_path = split_root / image_name
        eval_image_paths.append(image_path)
        gt_by_image[image_name] = [
            {"class_name": d["class"], "bbox": [float(x) for x in d["bbox"]]}
            for d in row.get("defects", [])
        ]
        preds_by_image[image_name] = infer_image(model, image_path, class_names)

    map_metrics = evaluate_map(class_names, gt_by_image, preds_by_image)
    prf1_metrics = evaluate_prf1(
        class_names,
        gt_by_image,
        preds_by_image,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.confidence_threshold,
    )
    speed_metrics = benchmark_inference(
        model,
        eval_image_paths,
        class_names=class_names,
        warmup=args.speed_warmup,
        max_samples=args.speed_samples,
    )

    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_source": model_source,
        "evaluation_split": args.eval_split,
        "model_performance": {
            "mAP@0.5": map_metrics["mAP@0.5"],
            "mAP@0.5:0.95": map_metrics["mAP@0.5:0.95"],
            "average_precision_per_class_by_iou": {
                normalized_class_name(k): v for k, v in map_metrics["per_class_ap"].items()
            },
            "precision_recall_f1_per_class": {
                normalized_class_name(k): v for k, v in prf1_metrics["per_class"].items()
            },
            "inference_speed": speed_metrics,
        },
        "dataset_size": dataset_stats,
        "impact": {
            "correctly_flagged_instances": {
                normalized_class_name(k): v for k, v in prf1_metrics["correctly_flagged_instances"].items()
            },
            "total_correctly_flagged_instances": prf1_metrics["total_correctly_flagged"],
        },
        "hardware": get_hardware_info(),
        "notes": {
            "iou_threshold_for_prf1": args.iou_threshold,
            "confidence_threshold_for_prf1": args.confidence_threshold,
            "map_iou_thresholds": IOU_THRESHOLDS,
            "class_aliases": CLASS_ALIASES,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote model metrics to {out_path}")
    print(json.dumps(output["model_performance"], indent=2))


if __name__ == "__main__":
    main()
