#!/usr/bin/env python3
"""Benchmarks digue with different models and backends.

Thin wrapper kept for checkouts: the runner lives in `digue.benchmark` and
`digue benchmark` exposes the same options to installed users.

Usage:
    python benchmark_models.py                          # auto-detected backend
    python benchmark_models.py --backends intel cpu      # compare backends
    python benchmark_models.py --models small medium     # specific models only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import digue
from digue import benchmark as benchmark_mod
from digue import container as container_mod
from digue.config import load_config

ALL_MODELS = ("small", "medium", "large-v3-turbo")


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
        default=benchmark_mod.BENCHMARK_RUNS,
        help=f"Runs per case (default: {benchmark_mod.BENCHMARK_RUNS})",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    config = load_config()
    if container_mod._is_remote(config):
        print("Benchmarking requires a local container; backend 'remote' is not supported.", file=sys.stderr)
        return 1
    sample = benchmark_mod.download_sample()
    try:
        results = benchmark_mod.run_benchmark(
            sample, config, backends=args.backends, models=args.models, runs=args.runs
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    print(json.dumps(results, default=str, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
