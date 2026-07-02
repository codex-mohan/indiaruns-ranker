"""Benchmark JSONL parser throughput on the candidate file.

Usage:
    uv run python scripts/benchmark_jsonl_parsers.py ../data/India_runs_data_and_ai_challenge/candidates.jsonl
"""
from __future__ import annotations

from pathlib import Path
import argparse
import gc
import time

import msgspec
import orjson
import simdjson


def _candidate_id(row: object) -> str:
    if isinstance(row, dict):
        value = row.get("candidate_id", "")
    else:
        value = row["candidate_id"]
    return str(value)


def bench_orjson(path: Path) -> tuple[int, int]:
    count = 0
    checksum = 0
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            count += 1
            checksum ^= hash(_candidate_id(row))
    return count, checksum


def bench_msgspec(path: Path) -> tuple[int, int]:
    count = 0
    checksum = 0
    decoder = msgspec.json.Decoder()
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = decoder.decode(line)
            count += 1
            checksum ^= hash(_candidate_id(row))
    return count, checksum


def bench_simdjson(path: Path) -> tuple[int, int]:
    count = 0
    checksum = 0
    parser = simdjson.Parser()
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = parser.parse(line, recursive=True)
            count += 1
            checksum ^= hash(_candidate_id(row))
    return count, checksum


def run_one(name: str, fn, path: Path) -> tuple[str, float, int, int]:
    gc.collect()
    t0 = time.perf_counter()
    count, checksum = fn(path)
    elapsed = time.perf_counter() - t0
    return name, elapsed, count, checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    path = args.path
    size_mb = path.stat().st_size / (1024 * 1024)
    benches = [
        ("orjson", bench_orjson),
        ("msgspec", bench_msgspec),
        ("pysimdjson", bench_simdjson),
    ]

    print(f"File: {path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Rounds: {args.rounds}")
    print()

    results: list[tuple[str, float, int, int]] = []
    for round_no in range(1, args.rounds + 1):
        print(f"Round {round_no}")
        for name, fn in benches:
            result = run_one(name, fn, path)
            results.append(result)
            _, elapsed, count, checksum = result
            print(f"  {name:10s} {elapsed:8.3f}s  {size_mb / elapsed:8.1f} MB/s  rows={count} checksum={checksum}")
        print()

    print("Best by parser")
    for name, _ in benches:
        best = min((r for r in results if r[0] == name), key=lambda r: r[1])
        _, elapsed, count, _ = best
        print(f"  {name:10s} {elapsed:8.3f}s  {size_mb / elapsed:8.1f} MB/s  rows={count}")


if __name__ == "__main__":
    main()
