"""Backend and model benchmark runner."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

BENCHMARK_TRANSCRIPTION_TIMEOUT = 300

BENCHMARK_RUNS = 3

# The quick comparison: the CPU fallback model and the GPU default.
BENCHMARK_MODELS = ("small", "large-v3-turbo")

SAMPLE_URL = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"

# Approximate GGML model sizes (MB), to warn before a benchmark triggers downloads.
MODEL_SIZES_MB = {"tiny": 75, "base": 142, "small": 466, "medium": 1500, "large-v3-turbo": 1620, "large-v3": 3100}

# -- Benchmark ----------------------------------------------------------------


def sample_path() -> Path:
    """The whisper.cpp JFK sample, kept in the runtime dir: it is private to
    the user (a fixed name in /tmp could be a symlink planted by another
    local user)."""
    from digue.recording import _runtime_dir

    return _runtime_dir() / "digue-bench-jfk.wav"


def download_sample() -> Path:
    """Downloads the JFK sample once and returns its path."""
    from digue.container import _download_file

    sample = sample_path()
    if sample.exists():
        print(f"Sample: {sample}", file=sys.stderr)
        return sample
    print("Downloading sample audio...", file=sys.stderr, flush=True)
    _download_file(SAMPLE_URL, sample, sample.name)
    print(f"Saved: {sample} ({sample.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
    return sample


def _benchmark_run(url: str, audio_path: str | Path, language: str, runs: int) -> list[tuple[int, str]]:
    """Runs one warm-up plus N timed transcriptions; returns (elapsed_ms, text) per run."""
    import time

    from digue.transcribe import transcribe

    with contextlib.suppress(Exception):
        transcribe(url, audio_path, language, timeout=BENCHMARK_TRANSCRIPTION_TIMEOUT)

    results = []
    for _run in range(runs):
        start = time.perf_counter()
        text = transcribe(url, audio_path, language, timeout=BENCHMARK_TRANSCRIPTION_TIMEOUT)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        results.append((elapsed_ms, text))
    return results


def _case_config(config: dict[str, dict[str, Any]], backend: str, model: str) -> dict[str, dict[str, Any]]:
    """The config for one backend+model case.

    server.image is a single global override that only makes sense for the
    backend the config resolved to (e.g. image "main" pinned for a Kaby Lake
    CPU); other cases fall back to DOCKER_IMAGES through the empty image.
    """
    from digue.container import resolve_backend

    bench_server = {**config["server"]}
    if backend != resolve_backend(config):
        bench_server["image"] = ""
    return {**config, "server": bench_server, "models": {**config["models"], backend: model}}


def benchmark_case(
    config: dict[str, dict[str, Any]],
    backend: str,
    model: str,
    audio_path: str | Path,
    runs: int = BENCHMARK_RUNS,
) -> dict[str, Any] | None:
    """Benchmarks one backend+model combination on audio_path.

    Creates the container for the case (downloading the model if missing),
    waits for the server, runs `_benchmark_run` and removes the container
    again -- also on Ctrl+c or an error. Returns the case result, or None
    when the case was skipped (image incompatible with this CPU, server did
    not come up, transcription failed); the skip reason goes to stderr.
    """
    from digue.container import (
        _wait_for_server,
        container_exists,
        create_container,
        remove_container,
        resolve_image,
        server_url,
    )

    label = f"{backend} / {model}"
    print(f"=== {label} ===", file=sys.stderr)
    case_config = _case_config(config, backend, model)
    print(f"  Image: {resolve_image(backend, case_config)}", file=sys.stderr)
    language = config["transcribe"]["language"]
    try:
        try:
            create_container(case_config, backend)
        except RuntimeError as exc:
            print(f"  Skipped: {exc}", file=sys.stderr)
            return None

        print("  Waiting for server...", file=sys.stderr, flush=True)
        if not _wait_for_server(case_config, verbose=True):
            print("  Server failed to start (see: docker logs digue), skipping", file=sys.stderr)
            return None

        try:
            results = _benchmark_run(server_url(case_config), audio_path, language, runs)
        except Exception as exc:
            print(f"  Skipped: transcription failed: {exc}", file=sys.stderr)
            return None
        if not results:
            print("  Skipped: no runs", file=sys.stderr)
            return None
        for idx, (elapsed_ms, _text) in enumerate(results, 1):
            print(f"  run {idx}: {elapsed_ms}ms", file=sys.stderr)
        avg_ms = sum(elapsed for elapsed, _ in results) // len(results)
        print(f"  avg: {avg_ms}ms", file=sys.stderr)
        print(f"  text: {results[-1][1]}", file=sys.stderr)
        print(file=sys.stderr)
        return {
            "backend": backend,
            "model": model,
            "avg_ms": avg_ms,
            "runs_ms": [elapsed for elapsed, _ in results],
            "text": results[-1][1],
        }
    finally:
        if container_exists():
            remove_container()


def default_backends(config: dict[str, dict[str, Any]]) -> list[str]:
    """The resolved backend plus cpu: a forced backend in the config wins
    over hardware detection (a cpu pinned with an image override must not be
    bypassed by the GPU it cannot run on)."""
    from digue.container import resolve_backend

    resolved = resolve_backend(config)
    return [resolved] if resolved == "cpu" else [resolved, "cpu"]


def warn_missing_models(models: list[str], config: dict[str, dict[str, Any]]) -> None:
    """Says up front which models the benchmark will download (minutes each)."""
    models_dir = Path(config["server"]["data_dir"]) / "models"
    missing = [model for model in models if not (models_dir / f"ggml-{model}.bin").exists()]
    if not missing:
        return
    total_mb = sum(MODEL_SIZES_MB.get(model, 0) for model in missing)
    listing = ", ".join(f"{model} (~{MODEL_SIZES_MB.get(model, '?')} MB)" for model in missing)
    print(f"Missing models (will download, ~{total_mb} MB total): {listing}", file=sys.stderr)


def print_summary(results: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 50}", file=sys.stderr)
    print("Summary", file=sys.stderr)
    print(f"{'=' * 50}", file=sys.stderr)
    for result in results:
        label = f"{result['backend']} / {result['model']}"
        print(f"  {label:<35} {result['avg_ms']}ms", file=sys.stderr)


def run_benchmark(
    audio_path: str | Path,
    config: dict[str, dict[str, Any]],
    backends: list[str] | None = None,
    models: list[str] | None = None,
    runs: int = BENCHMARK_RUNS,
) -> list[dict[str, Any]]:
    """Benchmarks every backend x model case on the same audio and returns
    the results (also printed as a summary). Defaults: the resolved backend
    plus cpu, and BENCHMARK_MODELS. The user's own container is preserved
    around the run; Ctrl+c stops the cases, prints the partial summary and
    propagates (the caller decides how to exit)."""
    from digue.container import preserve_container_for_benchmark

    backends = backends or default_backends(config)
    models = list(models or BENCHMARK_MODELS)

    print("digue benchmark", file=sys.stderr)
    print(f"Audio: {audio_path}", file=sys.stderr)
    print(f"Backends: {', '.join(backends)}", file=sys.stderr)
    print(f"Models: {', '.join(models)}", file=sys.stderr)
    print(f"Runs per case: {runs}", file=sys.stderr)
    warn_missing_models(models, config)
    print(file=sys.stderr)

    results: list[dict[str, Any]] = []
    try:
        with preserve_container_for_benchmark():
            for backend in backends:
                for model in models:
                    result = benchmark_case(config, backend, model, audio_path, runs)
                    if result is not None:
                        results.append(result)
    except KeyboardInterrupt:
        print_summary(results)
        raise
    print_summary(results)
    return results


def record_benchmark_audio(
    output_path: str | Path, duration_seconds: int = 10, config: dict[str, dict[str, Any]] | None = None
) -> None:
    """Records audio from microphone for benchmark, with the configured recorder."""
    import time

    from digue.recording import _popen_recorder, recording_command

    recorder = config["dictate"]["recorder"] if config else "auto"
    device = str(config["dictate"].get("device") or "") if config else ""
    print(f"Recording {duration_seconds}s from microphone...", file=sys.stderr)
    print("(speak something so there is content to transcribe)", file=sys.stderr)
    proc = _popen_recorder(
        recording_command(output_path, recorder=recorder, device=device),
        start_new_session=False,
    )
    time.sleep(duration_seconds)
    proc.terminate()
    time.sleep(0.5)
    print(f"Recorded: {output_path}", file=sys.stderr)


def cmd_benchmark(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    """`digue benchmark`: audio from a file, the JFK sample or the microphone;
    backends, models ("all" = every model) and runs from the options; the
    summary on stderr and, with --json, the results on stdout (also the
    partial ones on Ctrl+c, exit 130)."""
    import json

    from digue import AVAILABLE_MODELS
    from digue.container import _is_remote
    from digue.recording import _runtime_dir

    if _is_remote(config):
        print("Error: benchmark creates local containers; it is not available with backend 'remote'", file=sys.stderr)
        return 1

    if args.audio:
        audio_path = args.audio
        if not audio_path.exists():
            print(f"Error: file not found: {audio_path}", file=sys.stderr)
            return 1
    elif args.sample:
        audio_path = download_sample()
    else:
        # the runtime dir is private to the user; a fixed name in /tmp could be
        # a symlink planted by another local user
        audio_path = _runtime_dir() / "digue-bench.wav"
        record_benchmark_audio(audio_path, config=config)
        print(file=sys.stderr)

    models = args.models
    if models and "all" in models:
        models = list(AVAILABLE_MODELS)
    results: list[dict[str, Any]] = []
    exit_code = 0
    try:
        results = run_benchmark(audio_path, config, backends=args.backends, models=models, runs=args.runs)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        exit_code = 130
    if args.json:
        print(json.dumps(results, default=str, ensure_ascii=False, indent=2))
    return exit_code
