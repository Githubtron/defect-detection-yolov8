import argparse
import json
import re
import time
from pathlib import Path


def count_loc(path: Path, exts: set[str]) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in exts:
            continue
        if "node_modules" in file_path.parts or "__pycache__" in file_path.parts or "dist" in file_path.parts:
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        total += sum(1 for line in text.splitlines() if line.strip())
    return total


def count_fastapi_endpoints(main_py: Path) -> int:
    text = main_py.read_text(encoding="utf-8")
    return len(re.findall(r"@app\.(get|post|put|delete|patch|websocket)\(", text))


def count_react_components(frontend_src: Path) -> int:
    total = 0
    component_re = re.compile(r"\bfunction\s+([A-Z][A-Za-z0-9_]*)\s*\(|\bconst\s+([A-Z][A-Za-z0-9_]*)\s*=\s*\(")
    for file_path in frontend_src.rglob("*"):
        if file_path.suffix.lower() not in {".jsx", ".tsx"}:
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        names = set()
        for match in component_re.finditer(text):
            name = match.group(1) or match.group(2)
            if name and name not in {"React"}:
                names.add(name)
        total += len(names)
    return total


def find_latest_training_summary(training_root: Path) -> Path | None:
    candidates = sorted(training_root.rglob("training_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_args():
    parser = argparse.ArgumentParser(description="Generate project scale metrics JSON.")
    parser.add_argument("--root", default="..")
    parser.add_argument("--output", default="metrics\\project_scale.json")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    backend_root = root / "backend"
    training_root = root / "training"
    frontend_src = root / "frontend" / "src"

    latest_summary_path = find_latest_training_summary(training_root)
    latest_summary = None
    if latest_summary_path:
        latest_summary = json.loads(latest_summary_path.read_text(encoding="utf-8"))

    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lines_of_code": {
            "backend_python_loc": count_loc(backend_root, {".py"}),
            "training_python_loc": count_loc(training_root, {".py"}),
            "frontend_source_loc": count_loc(frontend_src, {".js", ".jsx", ".ts", ".tsx", ".css"}),
        },
        "api": {
            "fastapi_endpoint_count": count_fastapi_endpoints(backend_root / "main.py"),
        },
        "frontend": {
            "react_component_count": count_react_components(frontend_src),
        },
        "training": {
            "latest_training_summary_path": str(latest_summary_path) if latest_summary_path else None,
            "epochs_configured": (latest_summary or {}).get("training", {}).get("epochs_configured"),
            "epochs_completed": (latest_summary or {}).get("training", {}).get("epochs_completed"),
            "training_time_minutes": (latest_summary or {}).get("training", {}).get("training_time_minutes"),
            "hardware": (latest_summary or {}).get("hardware"),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote project scale metrics to {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
