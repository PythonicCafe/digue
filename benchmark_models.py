#!/usr/bin/env python3
"""Benchmarks digue with different models and backends.

Downloads a known audio sample and measures transcription time in-process
(no interpreter startup overhead in the measured runs).

Usage:
    python benchmark_models.py                          # auto-detected backend
    python benchmark_models.py --backends intel cpu      # compare backends
    python benchmark_models.py --models small medium     # specific models only
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import digue

SAMPLE_URL = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"
SAMPLE_PATH = Path("/tmp/digue-bench-jfk.wav")
ALL_MODELS = ("small", "medium", "large-v3-turbo")
# Approximate GGML model sizes (MB), to warn before benchmarking triggers downloads
MODEL_SIZES_MB = {"tiny": 75, "base": 142, "small": 466, "medium": 1500, "large-v3-turbo": 1620, "large-v3": 3100}
RUNS = 3


def download_sample() -> None:
    if SAMPLE_PATH.exists():
        print(f"Sample: {SAMPLE_PATH}", file=sys.stderr)
        return
    print("Downloading sample audio...", file=sys.stderr, flush=True)
    digue._download_file(SAMPLE_URL, SAMPLE_PATH, SAMPLE_PATH.name)
    print(f"Saved: {SAMPLE_PATH} ({SAMPLE_PATH.stat().st_size / 1024:.0f} KB)", file=sys.stderr)


def benchmark_case(config: dict[str, dict[str, Any]], backend: str, model: str) -> dict[str, Any] | None:
    """Benchmarks a single backend+model combination. Returns dict or None."""
    label = f"{backend} / {model}"
    print(f"\n=== {label} ===", file=sys.stderr)

    models_dir = Path(config["server"]["data_dir"]) / "models"
    model_path = models_dir / f"ggml-{model}.bin"
    if not model_path.exists():
        print(
            f"  Model {model} not found locally - it will be downloaded first (~{MODEL_SIZES_MB.get(model, '?')} MB).",
            file=sys.stderr,
        )

    bench_config = {**config, "models": {**config["models"], backend: model}}

    print("  Starting server...", file=sys.stderr, flush=True)
    try:
        try:
            digue.create_container(bench_config, backend)
        except RuntimeError as exc:
            print(f"  Skipped: {exc}", file=sys.stderr)
            return None

        if not digue._wait_for_server(config, verbose=True):
            print("  Server failed to start (see: docker logs digue), skipping", file=sys.stderr)
            return None

        url = digue.server_url(config)

        try:
            digue.transcribe(url, SAMPLE_PATH, "en", timeout=digue.BENCHMARK_TRANSCRIPTION_TIMEOUT)
        except Exception as exc:
            print(f"  Skipped: warm-up transcription failed: {exc}", file=sys.stderr)
            return None

        results = []
        text = ""
        for run_idx in range(1, RUNS + 1):
            start = time.perf_counter()
            text = digue.transcribe(url, SAMPLE_PATH, "en", timeout=digue.BENCHMARK_TRANSCRIPTION_TIMEOUT)
            elapsed = time.perf_counter() - start
            results.append(elapsed)
            print(f"  run {run_idx}: {elapsed:.2f}s", file=sys.stderr)

        avg = sum(results) / len(results)
        print(f"  avg: {avg:.2f}s", file=sys.stderr)
        print(f"  text: {text}", file=sys.stderr)

        return {
            "backend": backend,
            "model": model,
            "avg_s": round(avg, 2),
            "runs": [round(result, 2) for result in results],
            "text": text,
        }
    finally:
        if digue.container_exists():
            digue.remove_container()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark digue with different models and backends")
    parser.add_argument(
        "-b",
        "--backends",
        nargs="+",
        default=None,
        choices=list(digue.DOCKER_IMAGES.keys()),
        help=f"Backends to test (default: auto-detected + cpu). Options: {', '.join(digue.DOCKER_IMAGES.keys())}",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=list(ALL_MODELS),
        help=f"Models to test (default: {', '.join(ALL_MODELS)})",
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=RUNS,
        help=f"Runs per case (default: {RUNS})",
    )
    return parser


def main() -> None:
    global RUNS

    parser = create_parser()
    args = parser.parse_args()
    RUNS = args.runs

    download_sample()
    config = digue.load_config()

    if args.backends is None:
        detected = digue.detect_backend()
        backends = [detected]
        if detected != "cpu":
            backends.append("cpu")
    else:
        backends = args.backends

    print(f"Backends: {', '.join(backends)}", file=sys.stderr)
    print(f"Models: {', '.join(args.models)}", file=sys.stderr)
    print(f"Runs per case: {RUNS}", file=sys.stderr)

    # Warn up front about models that will be downloaded (can take minutes)
    models_dir = Path(config["server"]["data_dir"]) / "models"
    missing = [model for model in args.models if not (models_dir / f"ggml-{model}.bin").exists()]
    if missing:
        total_mb = sum(MODEL_SIZES_MB.get(model, 0) for model in missing)
        listing = ", ".join(f"{model} (~{MODEL_SIZES_MB.get(model, '?')} MB)" for model in missing)
        print(f"Missing models (will download, ~{total_mb} MB total): {listing}", file=sys.stderr)

    all_results = []
    try:
        with digue.preserve_container_for_benchmark():
            for backend in backends:
                for model in args.models:
                    result = benchmark_case(config, backend, model)
                    if result:
                        all_results.append(result)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("Summary", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  {'Backend + Model':<35} {'Avg':>8}", file=sys.stderr)
    print(f"  {'-' * 35} {'-' * 8}", file=sys.stderr)
    for result in all_results:
        label = f"{result['backend']} / {result['model']}"
        print(f"  {label:<35} {result['avg_s']:>7.2f}s", file=sys.stderr)

    print(json.dumps(all_results, default=str, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
