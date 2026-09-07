#!/usr/bin/env python3
"""Benchmarks digue with different models and backends.

Downloads a known audio sample and measures transcription time.
Usage:
    python benchmark_models.py                          # auto-detected backend
    python benchmark_models.py --backends intel cpu      # compare backends
    python benchmark_models.py --models small medium     # specific models only
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SAMPLE_URL = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"
SAMPLE_PATH = Path("/tmp/digue-bench-jfk.wav")
ALL_MODELS = ("small", "medium", "large-v3-turbo")
RUNS = 3
SCRIPT = Path(__file__).resolve().parent / "digue.py"
STARTUP_TIMEOUT = 180
DOCKER_IMAGES = {
    "nvidia": "ghcr.io/ggml-org/whisper.cpp:main-cuda",
    "amd": "ghcr.io/ggml-org/whisper.cpp:main-vulkan",
    "intel": "ghcr.io/ggml-org/whisper.cpp:main-vulkan",
    # CPU: uses main-vulkan without GPU devices (main image crashes on AMX)
    "cpu": "ghcr.io/ggml-org/whisper.cpp:main-vulkan",
}


def download_sample():
    if SAMPLE_PATH.exists():
        print(f"Sample: {SAMPLE_PATH}", file=sys.stderr)
        return
    print("Downloading sample audio...", file=sys.stderr, flush=True)
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    print(f"Saved: {SAMPLE_PATH} ({SAMPLE_PATH.stat().st_size / 1024:.0f} KB)", file=sys.stderr)


def run_whisper(args, timeout=600):
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def get_config():
    result = run_whisper(["config"])
    return json.loads(result.stdout)


def wait_for_server(config):
    for attempt in range(STARTUP_TIMEOUT):
        result = run_whisper(["status"])
        if "responding" in result.stderr and "not responding" not in result.stderr:
            return True
        time.sleep(1)
        if attempt > 0 and attempt % 15 == 0:
            print(f"    loading model... {attempt}s", file=sys.stderr, flush=True)
    return False


def start_container(config, backend, model):
    """Creates a digue container with a specific backend and model."""
    port = config["server"]["port"]
    models_dir = f"{config['server']['data_dir']}/models"
    image = DOCKER_IMAGES[backend]

    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        "digue",
        "-p",
        f"127.0.0.1:{port}:8080",
    ]
    if backend == "nvidia":
        docker_cmd += ["--gpus", "all"]
    elif backend == "amd":
        docker_cmd += ["--device", "/dev/kfd", "--device", "/dev/dri"]
    elif backend == "intel":
        docker_cmd += ["--device", "/dev/dri"]

    docker_cmd += [
        "-v",
        f"{models_dir}:/models:ro",
        "--entrypoint",
        "digue",
        image,
        "--model",
        f"/models/ggml-{model}.bin",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--vad",
        "--vad-model",
        "/models/ggml-silero-v6.2.0.bin",
    ]
    subprocess.run(docker_cmd, capture_output=True, timeout=60)


def benchmark_case(config, backend, model):
    """Benchmarks a single backend+model combination. Returns dict or None."""
    label = f"{backend} / {model}"
    print(f"\n=== {label} ===", file=sys.stderr)

    run_whisper(["download", model], timeout=600)
    run_whisper(["destroy"])
    time.sleep(2)

    print("  Starting server...", file=sys.stderr, flush=True)
    start_container(config, backend, model)

    if not wait_for_server(config):
        print("  Server failed to start, skipping", file=sys.stderr)
        run_whisper(["destroy"])
        return None

    # Warm-up
    run_whisper(["transcribe", str(SAMPLE_PATH), "-l", "en"])

    results = []
    text = ""
    for run_idx in range(1, RUNS + 1):
        start = time.perf_counter()
        result = run_whisper(["transcribe", str(SAMPLE_PATH), "-l", "en"])
        elapsed = time.perf_counter() - start
        text = result.stdout.strip()
        results.append(elapsed)
        print(f"  run {run_idx}: {elapsed:.2f}s", file=sys.stderr)

    avg = sum(results) / len(results)
    print(f"  avg: {avg:.2f}s", file=sys.stderr)
    print(f"  text: {text}", file=sys.stderr)

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
        choices=list(DOCKER_IMAGES.keys()),
        help=f"Backends to test (default: auto-detected). Options: {', '.join(DOCKER_IMAGES.keys())}",
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
    config = get_config()

    if args.backends is None:
        result = run_whisper(["detect"])
        detected = result.stdout.strip()
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
    run_whisper(["destroy"])
    run_whisper(["start"])

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
