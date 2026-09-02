"""Classical computer-vision defect detector for the knocked-down inspection demo.

No ML weights, no dataset, no GPU: detects the six aircraft-surface defect
classes with deterministic OpenCV pipelines. Exposes a minimal YOLO-compatible
interface (names + callable returning results with .boxes and .plot()) so it
drops straight into main.py's run_inference without touching the websocket /
video / batch / report paths.

    crack         dark jagged line        -> morphological black-hat + elongation
    scratch       bright thin line        -> morphological white-hat + elongation
    dent          soft elliptical shading -> local-background dark blob, roundish
    corrosion     rust-colored speckle    -> HSV hue segmentation
    missing-head  dark circular hole      -> dark blob, small + circular
    paint-peel    bright rough patch      -> brightness threshold, jagged boundary

Usage:
    python cv_detector.py --image panel.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["crack", "dent", "corrosion", "scratch", "missing-head", "paint-peel-off"]
COLORS = {
    0: (0, 0, 255),      # crack       -> red
    1: (0, 140, 255),    # dent        -> orange
    2: (0, 255, 255),    # corrosion   -> yellow
    3: (255, 0, 0),      # scratch     -> blue
    4: (255, 0, 255),    # missing-head-> magenta
    5: (255, 255, 0),    # paint-peel  -> cyan
}


def _conf(value, lo, hi):
    """Map a raw feature strength onto a heuristic confidence in [0.15, 0.98]."""
    return float(np.clip((value - lo) / (hi - lo), 0.15, 0.98))


class _Box:
    """Mimics a single ultralytics box (cls, conf)."""

    def __init__(self, cls_idx, conf, x1, y1, x2, y2):
        self.cls = np.array([cls_idx], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)
        self.xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)


class _Boxes:
    def __init__(self, boxes):
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class _Result:
    """Mimics an ultralytics Results object (boxes + plot())."""

    def __init__(self, bgr, detections):
        # detections: list of (cls_idx, conf, x1, y1, x2, y2)
        self.boxes = _Boxes([_Box(c, conf, x1, y1, x2, y2) for (c, conf, x1, y1, x2, y2) in detections])
        self._bgr = bgr
        self._dets = detections
        self._annotated = None

    def plot(self):
        if self._annotated is None:
            out = self._bgr.copy()
            for cls_idx, conf, x1, y1, x2, y2 in self._dets:
                color = COLORS.get(cls_idx, (0, 255, 0))
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"{CLASSES[cls_idx]} {conf:.0%}"
                (tw, th), _base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(out, label, (x1 + 3, max(th + 2, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            self._annotated = out
        return self._annotated


class CvDefectDetector:
    """Deterministic classical-CV engine behind the 'defect' model key."""

    def __init__(self, conf_threshold=0.30):
        self.names = {i: name for i, name in enumerate(CLASSES)}
        self.conf_threshold = conf_threshold

    # ---- YOLO-compatible entry point --------------------------------------
    def __call__(self, source):
        bgr = self._to_bgr(source)
        detections = self.detect(bgr)
        detections = [d for d in detections if d[1] >= self.conf_threshold]
        return [_Result(bgr, detections)]

    @staticmethod
    def _to_bgr(source):
        if hasattr(source, "convert") and hasattr(source, "size"):
            return np.array(source.convert("RGB"))[:, :, ::-1].copy()
        return source  # already a BGR numpy array (video frames)

    # ---- component helpers -------------------------------------------------
    @staticmethod
    def _components(binary):
        n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        comps = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            comps.append((x, y, w, h, area))
        return comps

    @staticmethod
    def _aspect_ratio(mask):
        pts = np.column_stack(np.where(mask > 0))
        if len(pts) < 5:
            return 0.0
        rect = cv2.minAreaRect(pts)
        (rw, rh) = rect[1]
        return max(rw, rh) / max(1.0, min(rw, rh))

    @staticmethod
    def _jaggedness(mask):
        border = float(np.abs(np.diff(mask.astype(np.int16), axis=0)).sum()
                       + np.abs(np.diff(mask.astype(np.int16), axis=1)).sum())
        area = float((mask > 0).sum())
        if area < 1:
            return 0.0
        return border * border / (4.0 * np.pi * area)

    # ---- per-class detectors ----------------------------------------------
    def _cracks(self, gray, rust_mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        _, bw = cv2.threshold(blackhat, 24, 255, cv2.THRESH_BINARY)
        bw[rust_mask > 0] = 0
        bw_d = cv2.dilate(bw, np.ones((3, 3), np.uint8))
        out = []
        for (x, y, w, h, area) in self._components(bw_d):
            if area < 60 or max(w, h) < 25:
                continue
            mask = bw_d[y:y + h, x:x + w]
            if self._aspect_ratio(mask) < 2.0:
                continue
            orig = bw[y:y + h, x:x + w]
            if (orig > 0).sum() < 25:
                continue
            mean_hat = float(blackhat[y:y + h, x:x + w][orig > 0].mean())
            out.append((0, _conf(mean_hat, 24, 90), x, y, x + w, y + h))
        return out

    def _scratches(self, gray):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        whitehat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        _, bw = cv2.threshold(whitehat, 26, 255, cv2.THRESH_BINARY)
        bw_d = cv2.dilate(bw, np.ones((3, 3), np.uint8))
        out = []
        for (x, y, w, h, area) in self._components(bw_d):
            if area < 50 or max(w, h) < 40:
                continue
            mask = bw_d[y:y + h, x:x + w]
            if self._aspect_ratio(mask) < 3.2:
                continue
            orig = bw[y:y + h, x:x + w]
            if (orig > 0).sum() < 25:
                continue
            mean_hat = float(whitehat[y:y + h, x:x + w][orig > 0].mean())
            out.append((3, _conf(mean_hat, 26, 80), x, y, x + w, y + h))
        return out

    def _dark_blobs(self, gray, rust_mask, line_mask):
        """Split dark regions into dents vs missing-head holes."""
        bg = cv2.GaussianBlur(gray, (0, 0), 25)
        diff = cv2.subtract(bg, gray)
        _, bw = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        bw[line_mask > 0] = 0
        bw[rust_mask > 0] = 0
        bw_d = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        dents, holes = [], []
        for (x, y, w, h, area) in self._components(bw_d):
            if area < 150 or area > 40000:
                continue
            if max(w, h) > 0.6 * gray.shape[1]:
                continue
            mask = bw_d[y:y + h, x:x + w]
            aspect = self._aspect_ratio(mask)
            if aspect > 3.2:
                continue
            orig = bw[y:y + h, x:x + w]
            o = orig > 0
            if o.sum() < 15:
                continue
            region_gray = gray[y:y + h, x:x + w][o]
            min_g = float(region_gray.min())
            mean_g = float(region_gray.mean())
            if mean_g > 150:
                continue  # only relatively darker vs a bright neighbour (peel edges)
            circular = float(o.sum()) / max(1.0, np.pi * (max(w, h) / 2.0) ** 2)
            if min_g < 85 and max(w, h) < 55 and circular > 0.2:
                holes.append((4, _conf(110 - min_g, 25, 100), x, y, x + w, y + h))
            elif min_g >= 80:
                depth = float(bg[y:y + h, x:x + w][o].mean()) - mean_g
                dents.append((1, _conf(depth, 8, 40), x, y, x + w, y + h))
        return dents, holes

    def _corrosion(self, hsv):
        h = hsv[:, :, 0].astype(np.int16)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        rust = ((h < 35) | (h > 150)) & (s > 45) & (v > 30) & (v < 225)
        rust = rust.astype(np.uint8) * 255
        rust = cv2.morphologyEx(rust, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        rust = cv2.morphologyEx(rust, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        out = []
        for (x, y, w, hh, area) in self._components(rust):
            if area < 120:
                continue
            mask = rust[y:y + hh, x:x + w]
            sat = float(s[y:y + hh, x:x + w][mask > 0].mean())
            out.append((2, _conf(sat, 50, 110), x, y, x + w, y + hh))
        return out, rust

    def _paint_peels(self, gray):
        _, bw = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        bw_d = cv2.dilate(bw, np.ones((5, 5), np.uint8))
        out = []
        for (x, y, w, h, area) in self._components(bw_d):
            if area < 200:
                continue
            mask = bw_d[y:y + h, x:x + w]
            if self._aspect_ratio(mask) > 3.5:
                continue
            orig = bw[y:y + h, x:x + w]
            o = orig > 0
            if o.sum() < 100 or o.sum() / max(1.0, w * h) < 0.15:
                continue
            mean_g = float(gray[y:y + h, x:x + w][o].mean())
            out.append((5, _conf(mean_g - 205, 4, 20), x, y, x + w, y + h))
        return out

    # ---- master pipeline ----------------------------------------------------
    def detect(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        corrosion, rust_mask = self._corrosion(hsv)
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k7)
        _, line_bw = cv2.threshold(blackhat, 24, 255, cv2.THRESH_BINARY)
        thick = cv2.morphologyEx(line_bw, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        line_mask = cv2.dilate(line_bw & ~thick, np.ones((3, 3), np.uint8))
        dents, holes = self._dark_blobs(gray, rust_mask, line_mask)
        detections = (self._cracks(gray, rust_mask)
                      + self._scratches(gray)
                      + dents
                      + corrosion
                      + holes
                      + self._paint_peels(gray))
        return self._nms(detections)

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a[2:]
        bx1, by1, bx2, by2 = b[2:]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / max(1.0, union)

    @staticmethod
    def _nms(detections, iou_thresh=0.35):
        kept = []
        for d in detections:
            dup = None
            for k in kept:
                if k[0] == d[0] and CvDefectDetector._iou(k, d) > iou_thresh:
                    dup = k
                    break
            if dup is None:
                kept.append(d)
            elif d[1] > dup[1]:
                kept[kept.index(dup)] = d
        return kept


def main():
    ap = argparse.ArgumentParser(description="Classical CV defect detector")
    ap.add_argument("--image", required=True, help="path to image")
    ap.add_argument("--conf", type=float, default=0.30, help="confidence threshold")
    args = ap.parse_args()

    bgr = cv2.imread(str(Path(args.image)))
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    det = CvDefectDetector(conf_threshold=args.conf)
    detections = det.detect(bgr)
    detections = [d for d in detections if d[1] >= args.conf]
    if not detections:
        print("No defects detected (PASS).")
    for cls_idx, conf, x1, y1, x2, y2 in detections:
        print(f"{CLASSES[cls_idx]:<14} conf={conf:.2f}  box=({x1},{y1})-({x2},{y2})")


if __name__ == "__main__":
    main()
