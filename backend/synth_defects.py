"""Procedural synthetic aircraft-panel defect generator.

Renders realistic brushed-aluminum aircraft skin panels with the six defect
classes the system is built to find:

    crack, dent, corrosion, scratch, missing-head, paint-peel-off

Every image ships with ground-truth bounding boxes (YOLO format) so output
can drive (a) end-to-end pipeline demos and (b) a future YOLO fine-tune if a
real dataset ever becomes available.

Usage:
    python synth_defects.py --out out_panels --count 20 --seed 0
    python synth_defects.py --out out_panels --count 100 --defects dent,crack
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["crack", "dent", "corrosion", "scratch", "missing-head", "paint-peel-off"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}


def make_background(w, h, rng):
    """Brushed aluminum skin: mid-gray base, vertical streaks, soft lighting."""
    base = int(rng.integers(150, 175))
    img = np.full((h, w, 3), base, dtype=np.int16)
    # brushed streaks: per-column brightness jitter (subtle enough that the
    # line detectors respond to real cracks/scratches, not the texture)
    streak = rng.integers(-2, 3, size=(w, 1), dtype=np.int16)
    img += streak
    # soft lighting gradient (slightly darker at edges)
    grad_y = np.linspace(-9, 9, h, dtype=np.float32)[:, None]
    grad_x = np.linspace(-6, 6, w, dtype=np.float32)[None, :]
    img += (grad_y + grad_x).astype(np.int16)[..., None]
    # subtle grain
    img += rng.integers(-2, 3, size=(h, w, 1), dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_rivets(img, rng):
    """Rows of bright rivet heads so panels look like real skin."""
    h, w = img.shape[:2]
    for row in range(3):
        y = int(h * (0.22 + row * 0.28))
        xs = np.arange(40, w - 20, 70).astype(int)
        xs = xs + rng.integers(-6, 7, size=xs.size)
        for x in xs:
            if 20 <= x < w - 20:
                r = int(rng.integers(5, 8))
                head = int(rng.integers(185, 200))
                cv2.circle(img, (int(x), y), r, (head, head, head), -1, cv2.LINE_AA)
                cv2.circle(img, (int(x), y), r, (125, 125, 125), 1, cv2.LINE_AA)
                cv2.circle(img, (int(x - 1), y - 1), max(1, r - 3), (228, 228, 228), -1, cv2.LINE_AA)
    return img


def draw_crack(img, rng):
    """Dark elongated jagged line with occasional branches. Returns bbox."""
    h, w = img.shape[:2]
    cx = int(rng.integers(int(w * 0.2), int(w * 0.8)))
    cy = int(rng.integers(int(h * 0.2), int(h * 0.8)))
    angle = rng.uniform(0, np.pi)
    length = int(rng.integers(90, 170))
    steps = int(rng.integers(8, 14))
    pts = []
    for i in range(steps + 1):
        t = i / steps
        wiggle = int(rng.integers(-14, 15))
        px = int(cx + length * t * np.cos(angle) + wiggle * np.sin(angle))
        py = int(cy + length * t * np.sin(angle) - wiggle * np.cos(angle))
        pts.append((max(4, min(w - 5, px)), max(4, min(h - 5, py))))
    color = (36, 36, 40)
    for i in range(len(pts) - 1):
        wline = int(rng.integers(2, 5))
        cv2.line(img, pts[i], pts[i + 1], color, wline, cv2.LINE_AA)
        if rng.random() < 0.25:
            bx, by = pts[i]
            ex = int(bx + rng.integers(-40, 41) * np.cos(angle) + rng.integers(-18, 19) * np.sin(angle))
            ey = int(by + rng.integers(-18, 19) * np.cos(angle) - rng.integers(-40, 41) * np.sin(angle))
            ex = max(0, min(w - 1, ex))
            ey = max(0, min(h - 1, ey))
            cv2.line(img, (bx, by), (ex, ey), color, max(2, wline - 1), cv2.LINE_AA)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4)


def draw_scratch(img, rng):
    """Thin bright line (scuff on polished metal). Returns bbox."""
    h, w = img.shape[:2]
    x1 = int(rng.integers(10, max(11, int(w * 0.35))))
    y1 = int(rng.integers(10, h - 10))
    ang = rng.uniform(-0.9, 0.9)
    length = int(rng.integers(90, int(w * 0.7)))
    x2 = int(max(2, min(w - 2, x1 + length * np.cos(ang))))
    y2 = int(max(2, min(h - 2, y1 + length * np.sin(ang))))
    bright = int(rng.integers(205, 232))
    cv2.line(img, (x1, y1), (x2, y2), (bright, bright, bright + 4), int(rng.integers(1, 2)), cv2.LINE_AA)
    return (min(x1, x2) - 4, min(y1, y2) - 4, max(x1, x2) + 4, max(y1, y2) + 4)


def draw_dent(img, rng):
    """Elliptical indentation with soft radial shading. Returns bbox."""
    h, w = img.shape[:2]
    cx = int(rng.integers(int(w * 0.25), int(w * 0.75)))
    cy = int(rng.integers(int(h * 0.25), int(h * 0.75)))
    rx = int(rng.integers(20, 42))
    ry = int(rng.integers(15, 32))
    theta = rng.uniform(0, np.pi)
    depth = int(rng.integers(30, 55))
    yy, xx = np.mgrid[0:h, 0:w]
    xr = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    yr = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
    d2 = (xr / rx) ** 2 + (yr / ry) ** 2
    mask = np.clip(1 - d2, 0, 1)
    shade = (mask * depth).astype(np.int16)
    img_int = img.astype(np.int16)
    img_int -= shade[..., None]
    img[:] = np.clip(img_int, 0, 255).astype(np.uint8)
    return (cx - rx, cy - ry, cx + rx, cy + ry)


def draw_corrosion(img, rng):
    """Rust-colored speckle patch with rough edges. Returns bbox."""
    h, w = img.shape[:2]
    cx = int(rng.integers(int(w * 0.2), int(w * 0.8)))
    cy = int(rng.integers(int(h * 0.2), int(h * 0.8)))
    radius = int(rng.integers(22, 46))
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), (radius, radius), 0, 0, 360, 255, -1)
    # solid rust base with speckle variation, dense enough to read as one region
    base_b = int(rng.integers(45, 70))
    base_g = int(rng.integers(70, 95))
    base_r = int(rng.integers(135, 175))
    img_int = img.astype(np.int16)
    region = mask > 0
    img_int[region] = (base_b, base_g, base_r)
    speck = (rng.integers(-25, 26, size=(h, w)) * (mask > 0)).astype(np.int16)
    img_int += speck[..., None]
    img[:] = np.clip(img_int, 0, 255).astype(np.uint8)
    # feather only inside the rust region so earlier defects stay crisp
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    img[:] = np.where(region[..., None], blurred, img)
    return (cx - radius - 6, cy - radius - 6, cx + radius + 6, cy + radius + 6)


def draw_missing_head(img, rng):
    """Dark circular hole where a rivet head has popped off. Returns bbox."""
    h, w = img.shape[:2]
    cx = int(rng.integers(int(w * 0.2), int(w * 0.8)))
    cy = int(rng.integers(int(h * 0.2), int(h * 0.8)))
    radius = int(rng.integers(9, 16))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = np.clip(radius - d + rng.integers(-1, 2, size=(h, w)), 0, None)
    depth = (np.clip(ring / radius, 0, 1) * 75 + 22).astype(np.int16)
    img_int = img.astype(np.int16)
    img_int -= depth[..., None]
    img[:] = np.clip(img_int, 0, 255).astype(np.uint8)
    return (cx - radius - 4, cy - radius - 4, cx + radius + 4, cy + radius + 4)


def draw_paint_peel(img, rng):
    """Patch of exposed bare metal (bright) with jagged boundary. Returns bbox or None."""
    h, w = img.shape[:2]
    cx = int(rng.integers(int(w * 0.2), int(w * 0.8)))
    cy = int(rng.integers(int(h * 0.2), int(h * 0.8)))
    radius = int(rng.integers(18, 40))
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), (radius, radius), rng.uniform(0, 180), 0, 360, 255, -1)
    noise = cv2.GaussianBlur(rng.integers(0, 255, size=(h, w), dtype=np.uint8), (15, 15), 0)
    jag = (((noise > int(rng.integers(95, 145))) & (mask > 0)).astype(np.uint8)) * 255
    jag = cv2.morphologyEx(jag, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ys, xs = np.where(jag > 0)
    if len(xs) < 30:
        return None
    bright = int(rng.integers(218, 235))
    img[ys, xs] = (bright, bright, bright)
    jag_mask = (jag > 0)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    img[:] = np.where(jag_mask[..., None], blurred, img)
    return (int(xs.min()) - 4, int(ys.min()) - 4, int(xs.max()) + 4, int(ys.max()) + 4)


DRAWERS = {
    "crack": draw_crack,
    "dent": draw_dent,
    "corrosion": draw_corrosion,
    "scratch": draw_scratch,
    "missing-head": draw_missing_head,
    "paint-peel-off": draw_paint_peel,
}


def generate_panel(w, h, rng, defect_pool):
    img = make_background(w, h, rng)
    img = draw_rivets(img, rng)
    specs = []
    n_defects = int(rng.integers(1, 4))
    for _ in range(n_defects):
        cls = defect_pool[int(rng.integers(0, len(defect_pool)))]
        box = DRAWERS[cls](img, rng)
        if box is not None:
            specs.append({"class": cls, "bbox": [int(v) for v in box]})
    return img, specs


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic aircraft defect panels")
    ap.add_argument("--out", default="out_panels", help="output directory")
    ap.add_argument("--count", type=int, default=20, help="number of panels")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--defects", default=",".join(CLASSES), help="comma-separated defect classes to place")
    ap.add_argument("--size", default="640,480", help="width,height")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in args.size.split(","))
    pool = [c.strip() for c in args.defects.split(",") if c.strip() in CLASSES]
    if not pool:
        pool = CLASSES

    rng = np.random.default_rng(args.seed)
    manifest = {"size": [w, h], "classes": CLASSES, "images": []}

    for i in range(args.count):
        img, specs = generate_panel(w, h, rng, pool)
        stem = f"panel_{i:03d}"
        cv2.imwrite(str(out / f"{stem}.jpg"), img)
        lines = []
        for s in specs:
            x1, y1, x2, y2 = s["bbox"]
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{CLASS_ID[s['class']]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        (out / f"{stem}.json").write_text(json.dumps({"file": f"{stem}.jpg", "defects": specs}, indent=1))
        manifest["images"].append({"file": f"{stem}.jpg", "defects": specs})

    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(len(im["defects"]) for im in manifest["images"])
    print(f"Wrote {args.count} panels to {out}/ with {total} total defect instances.")


if __name__ == "__main__":
    main()
