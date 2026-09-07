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
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import digue

SAMPLE_URL = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"
SAMPLE_PATH = Path("/tmp/digue-bench-jfk.wav")
ALL_MODELS = ("small", "medium", "large-v3-turbo")
RUNS = 3


def download_sample():
    if SAMPLE_PATH.exists():
        print(f"Sample: {SAMPLE_PATH}", file=sys.stderr)
        return
    print("Downloading sample audio...", file=sys.stderr, flush=True)
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    print(f"Saved: {SAMPLE_PATH} ({SAMPLE_PATH.stat().st_size / 1024:.0f} KB)", file=sys.stderr)


def benchmark_case(config, backend, model):
    """Benchmarks a single backend+model combination. Returns dict or None."""
    label = f"{backend} / {model}"
    print(f"\n=== {label} ===", file=sys.stderr)

    bench_config = {**config, "models": {**config["models"], backend: model}}

    digue.remove_container()
    time.sleep(2)

    print("  Starting server...", file=sys.stderr, flush=True)
    try:
        digue.create_container(bench_config, backend)
    except RuntimeError as exc:
        print(f"  Skipped: {exc}", file=sys.stderr)
        return None

    if not digue._wait_for_server(config, verbose=True):
        print("  Server failed to start (see: docker logs digue), skipping", file=sys.stderr)
        digue.remove_container()
        return None

    url = digue.server_url(config)

    # Warm-up (not measured)
    try:
        digue.transcribe(url, SAMPLE_PATH, "en", timeout=digue.BENCHMARK_TRANSCRIPTION_TIMEOUT)
    except Exception as exc:
        print(f"  Skipped: warm-up transcription failed: {exc}", file=sys.stderr)
        digue.remove_container()
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

    digue.remove_container()

    return {
        "backend": backend,
        "model": model,
        "avg_s": round(avg, 2),
        "runs": [round(r, 2) for r in results],
        "text": text,
    }


def create_parser():
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


def main():
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

    all_results = []
    try:
        for backend in backends:
            for model in args.models:
                result = benchmark_case(config, backend, model)
                if result:
                    all_results.append(result)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)

    # Restore default container
    print("\nRestoring default container...", file=sys.stderr, flush=True)
    try:
        digue.create_container(config)
        digue._wait_for_server(config, verbose=True)
    except RuntimeError as exc:
        print(f"  Could not restore default container: {exc}", file=sys.stderr)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("Summary", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  {'Backend + Model':<35} {'Avg':>8}", file=sys.stderr)
    print(f"  {'-' * 35} {'-' * 8}", file=sys.stderr)
    for result in all_results:
        label = f"{result['backend']} / {result['model']}"
        print(f"  {label:<35} {result['avg_s']:>7.2f}s", file=sys.stderr)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
