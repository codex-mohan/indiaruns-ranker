"""Download and stage local model weights used by the ranker.

This script is allowed to use network. Run it before the constrained ranking
step, or as part of precompute setup, so rank.py can run offline.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C


def _model_dir(artifacts_dir: Path, model_name: str) -> Path:
    return artifacts_dir / "models" / model_name.replace("/", "_")


def run(artifacts_dir: str) -> None:
    artifacts = Path(artifacts_dir)
    models_dir = artifacts / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Staging models under {models_dir}")

    from sentence_transformers import CrossEncoder, SentenceTransformer

    embed_dir = _model_dir(artifacts, C.EMBED_MODEL)
    print(f"Downloading embedding model: {C.EMBED_MODEL}")
    embed_model = SentenceTransformer(C.EMBED_MODEL)
    embed_model.save(str(embed_dir))
    print(f"  saved to {embed_dir}")

    cross_dir = _model_dir(artifacts, C.CROSS_ENCODER_MODEL)
    print(f"Downloading cross-encoder model: {C.CROSS_ENCODER_MODEL}")
    cross_model = CrossEncoder(C.CROSS_ENCODER_MODEL)
    cross_model.save(str(cross_dir))
    print(f"  saved to {cross_dir}")

    print("Done. The ranking step can load these models offline after precompute artifacts exist.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local model weights for offline ranking")
    parser.add_argument("--artifacts", default=C.ARTIFACTS_DIR, help="Artifacts directory")
    args = parser.parse_args()
    run(args.artifacts)


if __name__ == "__main__":
    main()
