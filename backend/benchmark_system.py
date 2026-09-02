import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from main import app


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.array(values, dtype=np.float32), q)), 3)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "avg_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    arr = np.array(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "avg_ms": round(float(arr.mean()), 3),
        "p95_ms": percentile(values, 95),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }


def encode_jpeg(path: Path) -> bytes:
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError(f"Could not encode JPEG: {path}")
    return buffer.tobytes()


def benchmark_http(client: TestClient, image_path: Path, repeats: int) -> dict:
    metrics = {}

    def timed_call(name: str, fn):
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            response = fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if response.status_code >= 400:
                raise RuntimeError(f"{name} failed with {response.status_code}: {response.text}")
            samples.append(elapsed_ms)
        metrics[name] = summarize(samples)

    timed_call("GET /", lambda: client.get("/"))
    timed_call("GET /models", lambda: client.get("/models"))
    timed_call(
        "POST /predict",
        lambda: client.post(
            "/predict?model=defect&include_annotated=false",
            files={"file": (image_path.name, image_path.read_bytes(), "image/jpeg")},
        ),
    )
    timed_call(
        "POST /predict-batch",
        lambda: client.post(
            "/predict-batch?model=defect&include_annotated=false",
            files=[
                ("files", ("a.jpg", image_path.read_bytes(), "image/jpeg")),
                ("files", ("b.jpg", image_path.read_bytes(), "image/jpeg")),
                ("files", ("c.jpg", image_path.read_bytes(), "image/jpeg")),
            ],
        ),
    )
    return metrics


def benchmark_websocket_stream(client: TestClient, frame_bytes: bytes, frames: int) -> dict:
    latencies = []
    t0 = time.perf_counter()
    with client.websocket_connect("/ws/live?model=defect&include_annotated=false") as ws:
        for _ in range(frames):
            ws.send_bytes(frame_bytes)
            response = ws.receive_json()
            latencies.append(float(response.get("latency_ms", 0.0)))
    elapsed = time.perf_counter() - t0
    fps = frames / elapsed if elapsed > 0 else 0.0
    return {
        "frames": frames,
        "elapsed_seconds": round(elapsed, 3),
        "stream_fps": round(fps, 3),
        "frame_latency": summarize(latencies),
    }


def benchmark_concurrent_streams(frame_bytes: bytes, concurrent_feeds: int, frames_per_feed: int) -> dict:
    lock = threading.Lock()
    feed_results: list[dict] = []

    def worker(feed_id: int):
        success = 0
        start = time.perf_counter()
        ok = True
        try:
            with TestClient(app) as local_client:
                with local_client.websocket_connect("/ws/live?model=defect&include_annotated=false") as ws:
                    for _ in range(frames_per_feed):
                        ws.send_bytes(frame_bytes)
                        response = ws.receive_json()
                        if "detections" in response:
                            success += 1
                        else:
                            ok = False
        except Exception:
            ok = False
        elapsed = time.perf_counter() - start
        with lock:
            feed_results.append(
                {
                    "feed_id": feed_id,
                    "ok": ok,
                    "frames_successful": success,
                    "elapsed_seconds": elapsed,
                }
            )

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(concurrent_feeds)]
    wall_start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall_elapsed = time.perf_counter() - wall_start

    total_expected = concurrent_feeds * frames_per_feed
    total_success = sum(row["frames_successful"] for row in feed_results)
    fully_successful = sum(1 for row in feed_results if row["ok"] and row["frames_successful"] == frames_per_feed)

    return {
        "requested_concurrent_feeds": concurrent_feeds,
        "frames_per_feed": frames_per_feed,
        "fully_successful_feeds": fully_successful,
        "max_concurrent_feeds_handled": fully_successful,
        "uptime_percent": round((total_success / total_expected) * 100.0, 2) if total_expected else 0.0,
        "aggregate_stream_fps": round(total_success / wall_elapsed, 3) if wall_elapsed > 0 else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="System benchmark for FastAPI latency and stream throughput.")
    parser.add_argument("--image-path", default="bench_panels\\panel_000.jpg")
    parser.add_argument("--http-repeats", type=int, default=20)
    parser.add_argument("--stream-frames", type=int, default=120)
    parser.add_argument("--concurrent-feeds", type=int, default=4)
    parser.add_argument("--frames-per-feed", type=int, default=60)
    parser.add_argument("--output", default="metrics\\system_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    frame_bytes = encode_jpeg(image_path)

    with TestClient(app) as client:
        api_latency = benchmark_http(client, image_path, repeats=args.http_repeats)
        stream = benchmark_websocket_stream(client, frame_bytes, frames=args.stream_frames)
        stress = benchmark_concurrent_streams(
            frame_bytes=frame_bytes,
            concurrent_feeds=args.concurrent_feeds,
            frames_per_feed=args.frames_per_feed,
        )
        runtime_snapshot = client.get("/metrics/system").json()

    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_latency_ms": api_latency,
        "realtime_stream": stream,
        "load_stress": stress,
        "runtime_metrics_snapshot": runtime_snapshot,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote system metrics to {out_path}")
    print(json.dumps({"api_latency_ms": api_latency, "realtime_stream": stream, "load_stress": stress}, indent=2))


if __name__ == "__main__":
    main()
