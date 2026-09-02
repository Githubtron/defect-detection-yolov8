import json
from pathlib import Path
import cv2
from synth_defects import CLASSES
from cv_detector import CvDefectDetector

panels = sorted(Path("bench_panels").glob("panel_*.jpg"))
det = CvDefectDetector(conf_threshold=0.30)
stats = {c: {"tp": 0, "fn": 0, "fp": 0} for c in CLASSES}
for img_path in panels:
    gt = json.loads(Path(f"bench_panels/{img_path.stem}.json").read_text())["defects"]
    bgr = cv2.imread(str(img_path))
    dets = [d for d in det.detect(bgr) if d[1] >= 0.30]
    matched = set()
    for cls_idx, conf, x1, y1, x2, y2 in dets:
        cls = CLASSES[cls_idx]
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gt):
            if gi in matched or g["class"] != cls:
                continue
            gx1, gy1, gx2, gy2 = g["bbox"]
            ix1, iy1 = max(x1, gx1), max(y1, gy1)
            ix2, iy2 = min(x2, gx2), min(y2, gy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (x2 - x1) * (y2 - y1) + (gx2 - gx1) * (gy2 - gy1) - inter
            iou = inter / union if union else 0
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= 0.25:
            matched.add(best_gi)
            stats[cls]["tp"] += 1
        else:
            stats[cls]["fp"] += 1
    for gi, g in enumerate(gt):
        if gi not in matched:
            stats[g["class"]]["fn"] += 1

print("\n=== FINAL BENCHMARK (60 panels, IoU>=0.25, conf>=0.30) ===")
print(f"{'class':<14} {'recall':>7} {'precision':>9} {'tp':>3} {'fp':>3} {'fn':>3}")
all_tp = all_fp = all_fn = 0
for cls in CLASSES:
    s = stats[cls]
    rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0
    prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0
    all_tp += s["tp"]
    all_fp += s["fp"]
    all_fn += s["fn"]
    print(f"{cls:<14} {rec:>6.2f} {prec:>8.2f} {s['tp']:>3} {s['fp']:>3} {s['fn']:>3}")
rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0
prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0
print("-" * 40)
print(f"{'OVERALL':<14} {rec:>6.2f} {prec:>8.2f} {all_tp:>3} {all_fp:>3} {all_fn:>3}")
