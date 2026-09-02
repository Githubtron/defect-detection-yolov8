"""Merge multiple Roboflow-exported YOLOv8 datasets into one, remapping each
dataset's own class names onto a canonical target class list (by default, the
classes already baked into backend/best.pt).

Two-step workflow, because every public dataset names its classes slightly
differently (e.g. "paint-off" vs "paint-peel-off") and that must be reconciled
by a human, not guessed:

  1. Scan the downloaded datasets and generate a starter class_map.json:

     python merge_yolo_datasets.py scan path/to/dataset1 path/to/dataset2 ...

     Edit the generated class_map.json so every source class name maps to one
     of the target classes printed by the scan (or "SKIP" to drop it, e.g. for
     classes irrelevant to this project like "screw" or "oil-stain").

  2. Build the merged dataset:

     python merge_yolo_datasets.py build path/to/dataset1 path/to/dataset2 ... \
         --class-map class_map.json --out merged_dataset

Each input dataset must already be in Roboflow's standard YOLOv8 export layout:
  <dataset>/data.yaml
  <dataset>/train/images/*  <dataset>/train/labels/*.txt
  <dataset>/valid/images/*  <dataset>/valid/labels/*.txt
  <dataset>/test/images/*   <dataset>/test/labels/*.txt   (optional)
"""

import argparse
import json
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")


def load_dataset_classes(dataset_dir: Path) -> list[str]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No data.yaml found in {dataset_dir}")
    spec = yaml.safe_load(data_yaml.read_text())
    names = spec["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=int)]
    return list(names)


def get_target_classes(weights: str | None, target_classes: list[str] | None) -> list[str]:
    if target_classes:
        return target_classes
    if weights:
        from ultralytics import YOLO

        return list(YOLO(weights).names.values())
    raise ValueError("Must supply either --weights or --target-classes")


def cmd_scan(args):
    target_classes = get_target_classes(args.weights, None)

    class_map = {}
    for ds in args.datasets:
        ds_path = Path(ds)
        names = load_dataset_classes(ds_path)
        print(f"\n{ds_path} ({len(names)} classes):")
        for name in names:
            print(f"  - {name}")
            class_map.setdefault(name, name if name in target_classes else "SKIP")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(class_map, indent=2, sort_keys=True))
    print(f"\nTarget classes (from {args.weights}): {target_classes}")
    print(f"\nWrote starter mapping to {out_path}")
    print('Edit it so every source class name maps to one of the target classes above, or "SKIP" to drop it.')


def cmd_build(args):
    class_map = json.loads(Path(args.class_map).read_text())
    target_classes = get_target_classes(args.weights, args.target_classes)
    target_index = {name: i for i, name in enumerate(target_classes)}

    out_root = Path(args.out)
    for split in SPLITS:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {split: 0 for split in SPLITS}
    for ds in args.datasets:
        ds_path = Path(ds)
        local_names = load_dataset_classes(ds_path)

        remap = {}
        for local_id, local_name in enumerate(local_names):
            if local_name not in class_map:
                raise KeyError(f"'{local_name}' from {ds_path} has no entry in {args.class_map}")
            mapped = class_map[local_name]
            if mapped != "SKIP" and mapped not in target_index:
                raise KeyError(
                    f"'{local_name}' -> '{mapped}' in {args.class_map} is not one of the target classes: {target_classes}"
                )
            remap[local_id] = None if mapped == "SKIP" else target_index[mapped]

        for split in SPLITS:
            img_dir = ds_path / split / "images"
            lbl_dir = ds_path / split / "labels"
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.iterdir()):
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                new_lines = []
                if lbl_path.exists():
                    for line in lbl_path.read_text().splitlines():
                        if not line.strip():
                            continue
                        parts = line.split()
                        target_id = remap.get(int(parts[0]))
                        if target_id is None:
                            continue  # this object's class was mapped to SKIP
                        new_lines.append(" ".join([str(target_id), *parts[1:]]))
                # Images with zero remaining labels are kept as background
                # examples (helps reduce false positives) rather than dropped.

                prefixed_stem = f"{ds_path.name}_{img_path.stem}"
                (out_root / split / "images" / f"{prefixed_stem}{img_path.suffix}").write_bytes(img_path.read_bytes())
                (out_root / split / "labels" / f"{prefixed_stem}.txt").write_text("\n".join(new_lines))
                counts[split] += 1

    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(target_classes),
        "names": target_classes,
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"Merged dataset written to {out_root}")
    print(f"Image counts: {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    scan_p = sub.add_parser("scan", help="List classes found in each dataset and generate a starter class_map.json")
    scan_p.add_argument("datasets", nargs="+", help="Paths to downloaded Roboflow YOLOv8 dataset folders")
    scan_p.add_argument("--weights", default="../backend/best.pt", help="Model to source the target class list from")
    scan_p.add_argument("--out", default="class_map.json")
    scan_p.set_defaults(func=cmd_scan)

    build_p = sub.add_parser("build", help="Merge datasets into one, remapping classes per class_map.json")
    build_p.add_argument("datasets", nargs="+", help="Paths to downloaded Roboflow YOLOv8 dataset folders")
    build_p.add_argument("--class-map", required=True, help="Path to the (edited) class_map.json from `scan`")
    build_p.add_argument("--weights", default="../backend/best.pt", help="Model to source the target class list from")
    build_p.add_argument("--target-classes", nargs="+", default=None, help="Override target classes explicitly")
    build_p.add_argument("--out", default="merged_dataset")
    build_p.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
