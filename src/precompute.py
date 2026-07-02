"""Phase A — Precompute artifacts.

Builds embeddings, TF-IDF index, feature vectors, and JD embedding.
Run this once (may exceed 5 min, may use network to fetch model).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import pickle
import time

import numpy as np
import orjson

from . import config as C
from .io import stream_candidates
from .features import extract
from .semantic import load_model, encode_texts
from .sparse import build_tfidf


def _model_dir(artifacts_dir: str, model_name: str) -> str:
    return os.path.join(artifacts_dir, "models", model_name.replace("/", "_"))


def _model_source(artifacts_dir: str, model_name: str) -> str:
    local_dir = _model_dir(artifacts_dir, model_name)
    return local_dir if os.path.exists(local_dir) else model_name


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_workers() -> int:
    return max(1, os.cpu_count() or 1)


def _extract_features(candidates: list[dict], workers: int | None) -> list[dict]:
    worker_count = _default_workers() if workers is None else workers
    if workers is not None and workers < 1:
        raise ValueError("--feature-workers must be >= 1")
    if worker_count == 1 or len(candidates) < 1000:
        return [extract(c) for c in candidates]

    chunksize = max(1, len(candidates) // (worker_count * 16))
    print(f"  Using {worker_count} feature workers (chunksize={chunksize})")
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        return list(pool.map(extract, candidates, chunksize=chunksize))


def run(
    candidates_path: str,
    artifacts_dir: str,
    embed_batch_size: int = 512,
    feature_workers: int | None = None,
    embed_backend: str = "torch",
    onnx_quantization: str = "avx2",
):
    if embed_batch_size < 1:
        raise ValueError("--embed-batch-size must be >= 1")
    t0 = time.time()
    os.makedirs(artifacts_dir, exist_ok=True)
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
    if os.path.getsize(candidates_path) == 0:
        raise ValueError(f"Candidates file is empty: {candidates_path}")

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
    feats = _extract_features(candidates, feature_workers)
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
    model_dir = _model_dir(artifacts_dir, C.EMBED_MODEL)
    model_source = _model_source(artifacts_dir, C.EMBED_MODEL) if embed_backend == "torch" else C.EMBED_MODEL
    model = load_model(
        model_source,
        backend=embed_backend,
        model_dir=model_dir,
        quantization=onnx_quantization,
    )
    print(f"  Model loaded from {model_source} with {embed_backend} in {time.time()-t2:.1f}s")

    t3 = time.time()
    print("Encoding JD...")
    jd_emb = encode_texts(model, [jd_text], batch_size=1)
    print(f"  JD encoded in {time.time()-t3:.1f}s")

    t4 = time.time()
    print(f"Encoding {n} candidates...")
    cand_embs = encode_texts(model, text_blobs, batch_size=embed_batch_size)
    print(f"  Candidates encoded in {time.time()-t4:.1f}s")

    # ── save embeddings ────────────────────────────────────────────────
    np.save(os.path.join(artifacts_dir, "jd_emb.npy"), jd_emb)
    np.save(os.path.join(artifacts_dir, "cand_embs.npy"), cand_embs)

    # ── save model weights for offline Stage 3 ────────────────────────
    if embed_backend == "torch":
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
    with open(feat_path, "wb") as f:
        for feat in feats:
            f.write(orjson.dumps(feat))
            f.write(b"\n")

    # ── save IDs ───────────────────────────────────────────────────────
    np.save(os.path.join(artifacts_dir, "ids.npy"), np.array(ids))

    manifest = {
        "candidate_count": n,
        "candidate_file_sha256": _file_sha256(candidates_path),
        "candidate_ids_sha256": hashlib.sha256(
            "\n".join(ids).encode("utf-8")
        ).hexdigest(),
        "embedding_model": C.EMBED_MODEL,
        "cross_encoder_model": C.CROSS_ENCODER_MODEL,
        "candidate_embedding_shape": list(cand_embs.shape),
        "tfidf_shape": list(matrix.shape),
    }
    with open(os.path.join(artifacts_dir, "manifest.json"), "wb") as f:
        f.write(orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))

    # ── save JD text ───────────────────────────────────────────────────
    with open(jd_text_path, "w", encoding="utf-8") as f:
        f.write(jd_text)

    # ── download and cache cross-encoder model ───────────────────────────
    t6 = time.time()
    print("Loading cross-encoder model for re-ranking...")
    from sentence_transformers import CrossEncoder

    ce_model_dir = _model_dir(artifacts_dir, C.CROSS_ENCODER_MODEL)
    os.makedirs(ce_model_dir, exist_ok=True)
    ce_source = ce_model_dir if os.path.exists(os.path.join(ce_model_dir, "config.json")) else C.CROSS_ENCODER_MODEL
    ce_model = CrossEncoder(ce_source)
    ce_model.save(ce_model_dir)
    print(f"  Cross-encoder saved to {ce_model_dir} in {time.time()-t6:.1f}s")

    total = time.time() - t0
    print(f"\nDone. Artifacts in {artifacts_dir}")
    print(f"  Total time: {total:.1f}s")
    print(f"  Embeddings: {cand_embs.shape} ({cand_embs.nbytes / 1e6:.1f} MB)")
    print(f"  TF-IDF: {matrix.shape}")


def main():
    parser = argparse.ArgumentParser(description="Precompute artifacts")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--artifacts", default=C.ARTIFACTS_DIR, help="Artifacts output dir")
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=512,
        help="SentenceTransformer encode batch size for candidate embeddings",
    )
    parser.add_argument(
        "--embed-backend",
        choices=["torch", "onnx-int8", "openvino", "openvino-int8"],
        default="torch",
        help="Embedding inference backend. Use onnx-int8/openvino-int8 for optimized CPU precompute.",
    )
    parser.add_argument(
        "--onnx-quantization",
        choices=["arm64", "avx2", "avx512", "avx512_vnni"],
        default="avx2",
        help="ONNX dynamic quantization target for --embed-backend onnx-int8",
    )
    parser.add_argument(
        "--feature-workers",
        type=int,
        default=None,
        help="Parallel feature extraction workers; default uses all logical CPUs",
    )
    args = parser.parse_args()
    run(
        args.candidates,
        args.artifacts,
        embed_batch_size=args.embed_batch_size,
        feature_workers=args.feature_workers,
        embed_backend=args.embed_backend,
        onnx_quantization=args.onnx_quantization,
    )


if __name__ == "__main__":
    main()
