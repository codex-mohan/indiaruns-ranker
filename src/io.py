"""Streaming JSONL reader for the candidates dataset."""
import json
from typing import Iterator


def stream_candidates(path: str) -> Iterator[dict]:
    """Yield one candidate dict per line from a JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_candidates(path: str) -> list[dict]:
    """Load all candidates into memory (for precompute phase)."""
    return list(stream_candidates(path))
