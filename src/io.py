"""Streaming JSONL reader for the candidates dataset."""
import json
from typing import Iterator


def stream_candidates(path: str) -> Iterator[dict]:
    """Yield one candidate dict per line from a JSONL file.

    Invalid JSON is fatal: silently skipping rows can attach scores to the
    wrong candidates or produce a partial submission that still looks valid.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc


def load_candidates(path: str) -> list[dict]:
    """Load all candidates into memory (for precompute phase)."""
    return list(stream_candidates(path))
