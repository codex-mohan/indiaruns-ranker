"""Phase A — Precompute artifacts.

Builds embeddings, TF-IDF index, feature vectors, and JD embedding.
Run this once (may exceed 5 min, may use network to fetch model).
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

from . import config as C
from .io import stream_candidates
from .features import extract
from .semantic import load_model, encode_texts
from .sparse import build_tfidf


def run(candidates_path: str, artifacts_dir: str):
    t0 = time.time()
    os.makedirs(artifacts_dir, exist_ok=True)

    # ── stream candidates ──────────────────────────────────────────────
    print("Loading candidates...")
    candidates = []
    ids = []
    for cand in stream_candidates(candidates_path):
        candidates.append(cand)
        ids.append(cand.get("candidate_id", ""))
    n = len(candidates)
    print(f"  Loaded {n} candidates in {time.time()-t0:.1f}s")

    # ── extract features ───────────────────────────────────────────────
    t1 = time.time()
    print("Extracting features...")
    feats = [extract(c) for c in candidates]
    print(f"  Features extracted in {time.time()-t1:.1f}s")

    # ── build text blobs ───────────────────────────────────────────────
    text_blobs = [f["text_blob"] for f in feats]

    # ── JD embedding ───────────────────────────────────────────────────
    jd_text_path = os.path.join(artifacts_dir, "jd_text.txt")
    if not os.path.exists(jd_text_path):
        # Load the JD text from the docx we already extracted
        jd_path = os.path.join(C.PROJECT_ROOT, "data", "job_description.md")
        if not os.path.exists(jd_path):
            jd_path = os.path.join(
                os.path.dirname(C.PROJECT_ROOT),
                "data", "India_runs_data_and_ai_challenge",
                "job_description.md",
            )
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
    else:
        with open(jd_text_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

    # ── embeddings ─────────────────────────────────────────────────────
    t2 = time.time()
    print("Loading embedding model...")
    model = load_model(C.EMBED_MODEL)
    print(f"  Model loaded in {time.time()-t2:.1f}s")

    t3 = time.time()
    print("Encoding JD...")
    jd_emb = encode_texts(model, [jd_text], batch_size=1)
    print(f"  JD encoded in {time.time()-t3:.1f}s")

    t4 = time.time()
    print(f"Encoding {n} candidates...")
    cand_embs = encode_texts(model, text_blobs, batch_size=512)
    print(f"  Candidates encoded in {time.time()-t4:.1f}s")

    # ── save embeddings ────────────────────────────────────────────────
    np.save(os.path.join(artifacts_dir, "jd_emb.npy"), jd_emb)
    np.save(os.path.join(artifacts_dir, "cand_embs.npy"), cand_embs)

    # ── save model weights for offline Stage 3 ────────────────────────
    model_dir = os.path.join(artifacts_dir, "models", C.EMBED_MODEL.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    print(f"  Model saved to {model_dir}")

    # ── TF-IDF index ───────────────────────────────────────────────────
    t5 = time.time()
    print("Building TF-IDF index...")
    vec, matrix = build_tfidf(text_blobs)
    with open(os.path.join(artifacts_dir, "tfidf.pkl"), "wb") as f:
        pickle.dump((vec, matrix), f)
    print(f"  TF-IDF built in {time.time()-t5:.1f}s")

    # ── save feature data as JSONL ─────────────────────────────────────
    feat_path = os.path.join(artifacts_dir, "features.jsonl")
    with open(feat_path, "w", encoding="utf-8") as f:
        for feat in feats:
            # Convert any non-serializable types
            row = {k: v for k, v in feat.items()}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── save IDs ───────────────────────────────────────────────────────
    np.save(os.path.join(artifacts_dir, "ids.npy"), np.array(ids))

    # ── save JD text ───────────────────────────────────────────────────
    with open(jd_text_path, "w", encoding="utf-8") as f:
        f.write(jd_text)

    total = time.time() - t0
    print(f"\nDone. Artifacts in {artifacts_dir}")
    print(f"  Total time: {total:.1f}s")
    print(f"  Embeddings: {cand_embs.shape} ({cand_embs.nbytes / 1e6:.1f} MB)")
    print(f"  TF-IDF: {matrix.shape}")


def main():
    parser = argparse.ArgumentParser(description="Precompute artifacts")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--artifacts", default=C.ARTIFACTS_DIR, help="Artifacts output dir")
    args = parser.parse_args()
    run(args.candidates, args.artifacts)


if __name__ == "__main__":
    main()
