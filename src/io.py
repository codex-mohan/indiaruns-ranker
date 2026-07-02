"""Streaming JSONL reader for the candidates dataset."""
import orjson
from typing import Iterator


def stream_candidates(path: str) -> Iterator[dict]:
    """Yield one candidate dict per line from a JSONL file.

    Invalid JSON is fatal: silently skipping rows can attach scores to the
    wrong candidates or produce a partial submission that still looks valid.
    """
    with open(path, "rb") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc


def load_candidates(path: str) -> list[dict]:
    """Load all candidates into memory (for precompute phase)."""
    return list(stream_candidates(path))
