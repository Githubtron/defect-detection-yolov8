import argparse
import json
import time
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Merge model/system/scale metrics into one final report file.")
    parser.add_argument("--model-metrics", default="metrics\\model_metrics.json")
    parser.add_argument("--system-metrics", default="metrics\\system_metrics.json")
    parser.add_argument("--project-scale", default="metrics\\project_scale.json")
    parser.add_argument("--impact-baseline", default="metrics\\impact_baseline.json")
    parser.add_argument("--output", default="metrics\\final_metrics_report.json")
    return parser.parse_args()


def main():
    args = parse_args()
    model_metrics = load_json(Path(args.model_metrics))
    system_metrics = load_json(Path(args.system_metrics))
    project_scale = load_json(Path(args.project_scale))
    impact_baseline = load_json(Path(args.impact_baseline))

    impact = {
        "manual_vs_automated": None,
        "accuracy_vs_baseline": None,
        "correctly_flagged_instances": None,
    }

    if model_metrics:
        impact["correctly_flagged_instances"] = model_metrics.get("impact", {}).get("total_correctly_flagged_instances")

    if impact_baseline and model_metrics:
        automated_ms = (
            model_metrics.get("model_performance", {})
            .get("inference_speed", {})
            .get("avg_ms_per_frame")
        )
        manual_seconds = impact_baseline.get("manual_inspection_seconds_per_image")
        baseline_map50 = impact_baseline.get("baseline_map50")
        map50 = model_metrics.get("model_performance", {}).get("mAP@0.5")
        if manual_seconds and automated_ms:
            automated_seconds = float(automated_ms) / 1000.0
            reduction = (1.0 - (automated_seconds / float(manual_seconds))) * 100.0
            impact["manual_vs_automated"] = {
                "manual_seconds_per_image": float(manual_seconds),
                "automated_seconds_per_image": round(automated_seconds, 4),
                "time_reduction_percent": round(reduction, 2),
            }
        if baseline_map50 is not None and map50 is not None:
            lift = ((float(map50) - float(baseline_map50)) / max(1e-9, float(baseline_map50))) * 100.0
            impact["accuracy_vs_baseline"] = {
                "baseline_map50": float(baseline_map50),
                "current_map50": float(map50),
                "lift_percent": round(lift, 2),
            }

    final_report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_performance": (model_metrics or {}).get("model_performance"),
        "dataset_size": (model_metrics or {}).get("dataset_size"),
        "system_performance": {
            "api_latency_ms": (system_metrics or {}).get("api_latency_ms"),
            "realtime_stream": (system_metrics or {}).get("realtime_stream"),
            "load_stress": (system_metrics or {}).get("load_stress"),
        },
        "impact_efficiency": impact,
        "project_scale": project_scale,
        "source_files": {
            "model_metrics": args.model_metrics if model_metrics else None,
            "system_metrics": args.system_metrics if system_metrics else None,
            "project_scale": args.project_scale if project_scale else None,
            "impact_baseline": args.impact_baseline if impact_baseline else None,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"Wrote final consolidated report to {out_path}")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
