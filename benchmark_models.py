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
from digue import benchmark as benchmark_mod
from digue import container as container_mod
from digue import recording as recording_mod
from digue import transcribe as transcribe_mod
from digue.config import load_config

SAMPLE_URL = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"
ALL_MODELS = ("small", "medium", "large-v3-turbo")
# Approximate GGML model sizes (MB), to warn before benchmarking triggers downloads
MODEL_SIZES_MB = {"tiny": 75, "base": 142, "small": 466, "medium": 1500, "large-v3-turbo": 1620, "large-v3": 3100}
RUNS = 3


def sample_path() -> Path:
    """The runtime dir is private to the user; a fixed name in /tmp could be a
    symlink planted by another local user."""
    return recording_mod._runtime_dir() / "digue-bench-jfk.wav"


def download_sample() -> None:
    sample = sample_path()
    if sample.exists():
        print(f"Sample: {sample}", file=sys.stderr)
        return
    print("Downloading sample audio...", file=sys.stderr, flush=True)
    container_mod._download_file(SAMPLE_URL, sample, sample.name)
    print(f"Saved: {sample} ({sample.stat().st_size / 1024:.0f} KB)", file=sys.stderr)


def benchmark_case(config: dict[str, dict[str, Any]], backend: str, model: str) -> dict[str, Any] | None:
    """Benchmarks a single backend+model combination. Returns dict or None."""
    label = f"{backend} / {model}"
    print(f"\n=== {label} ===", file=sys.stderr)

    resolved = container_mod.resolve_backend(config)
    models_dir = Path(config["server"]["data_dir"]) / "models"
    model_path = models_dir / f"ggml-{model}.bin"
    if not model_path.exists():
        print(
            f"  Model {model} not found locally - it will be downloaded first (~{MODEL_SIZES_MB.get(model, '?')} MB).",
            file=sys.stderr,
        )

    # Same rule as run_benchmark: the image override only applies to the
    # backend the config resolved to; other cases fall back to DOCKER_IMAGES.
    bench_server = {**config["server"]}
    if backend != resolved:
        bench_server["image"] = ""
    bench_config = {**config, "server": bench_server, "models": {**config["models"], backend: model}}
    print(f"  Image: {container_mod.resolve_image(backend, bench_config)}", file=sys.stderr)

    print("  Starting server...", file=sys.stderr, flush=True)
    try:
        try:
            container_mod.create_container(bench_config, backend)
        except RuntimeError as exc:
            print(f"  Skipped: {exc}", file=sys.stderr)
            return None

        if not container_mod._wait_for_server(config, verbose=True):
            print("  Server failed to start (see: docker logs digue), skipping", file=sys.stderr)
            return None

        url = container_mod.server_url(config)

        try:
            transcribe_mod.transcribe(url, sample_path(), "en", timeout=benchmark_mod.BENCHMARK_TRANSCRIPTION_TIMEOUT)
        except Exception as exc:
            print(f"  Skipped: warm-up transcription failed: {exc}", file=sys.stderr)
            return None

        results = []
        text = ""
        for run_idx in range(1, RUNS + 1):
            start = time.perf_counter()
            text = transcribe_mod.transcribe(
                url, sample_path(), "en", timeout=benchmark_mod.BENCHMARK_TRANSCRIPTION_TIMEOUT
            )
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
        if container_mod.container_exists():
            container_mod.remove_container()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark digue with different models and backends")
    parser.add_argument(
        "-b",
        "--backends",
        nargs="+",
        default=None,
        choices=list(container_mod.DOCKER_IMAGES.keys()),
        help=f"Backends to test (default: auto-detected + cpu). Options: {', '.join(container_mod.DOCKER_IMAGES.keys())}",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=list(ALL_MODELS),
        choices=digue.AVAILABLE_MODELS,
        help=f"Models to test (default: {', '.join(ALL_MODELS)})",
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=positive_int,
        default=RUNS,
        help=f"Runs per case (default: {RUNS})",
    )
    return parser


def main() -> int:
    global RUNS

    parser = create_parser()
    args = parser.parse_args()
    RUNS = args.runs

    config = load_config()
    if container_mod._is_remote(config):
        print("Benchmarking requires a local container; backend 'remote' is not supported.", file=sys.stderr)
        return 1

    download_sample()

    if args.backends is None:
        resolved = container_mod.resolve_backend(config)
        backends = [resolved]
        if resolved != "cpu":
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
        with container_mod.preserve_container_for_benchmark():
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
