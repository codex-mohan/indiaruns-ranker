"""Phase B — Ranking entry point (the reproduce command).

Loads pre-computed artifacts, scores all candidates, outputs top-100 CSV.
Must run < 5 min on CPU, no network.
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import time

import numpy as np

from . import config as C
from .io import stream_candidates
from .features import extract
from .semantic import cosine_scores
from .sparse import lexical_scores
from .honeypot import gate, hard_disqualifier
from .scoring import (
    skill_evidence_score,
    career_fit_score,
    location_score,
    behavioral_multiplier,
    compute_score,
)
from .reasoning import generate_reasoning
from .rerank import rerank as ce_rerank


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _validate_artifacts(
    artifacts_dir: str,
    ids: np.ndarray,
    cand_embs: np.ndarray,
    matrix,
    feats: list[dict],
) -> None:
    """Fail fast when precomputed artifacts do not match the candidate stream."""
    artifact_ids = [str(x) for x in ids.tolist()]
    candidate_ids = [f["candidate_id"] for f in feats]

    if len(candidate_ids) != len(artifact_ids):
        raise ValueError(
            f"Artifact/candidate count mismatch: ids.npy has {len(artifact_ids)}, "
            f"candidate file has {len(candidate_ids)}"
        )
    if candidate_ids != artifact_ids:
        for i, (cand_id, artifact_id) in enumerate(zip(candidate_ids, artifact_ids), start=1):
            if cand_id != artifact_id:
                raise ValueError(
                    f"Artifact/candidate order mismatch at row {i}: "
                    f"candidate file has {cand_id}, ids.npy has {artifact_id}"
                )
        raise ValueError("Artifact/candidate order mismatch")
    if cand_embs.shape[0] != len(candidate_ids):
        raise ValueError(
            f"cand_embs.npy row count {cand_embs.shape[0]} does not match "
            f"{len(candidate_ids)} candidates"
        )
    if matrix.shape[0] != len(candidate_ids):
        raise ValueError(
            f"TF-IDF matrix row count {matrix.shape[0]} does not match "
            f"{len(candidate_ids)} candidates"
        )

    manifest_path = os.path.join(artifacts_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected_count = manifest.get("candidate_count")
        if expected_count != len(candidate_ids):
            raise ValueError(
                f"manifest candidate_count {expected_count} does not match "
                f"{len(candidate_ids)} candidates"
            )
        expected_hash = manifest.get("candidate_ids_sha256")
        actual_hash = _ids_sha256(candidate_ids)
        if expected_hash != actual_hash:
            raise ValueError("manifest candidate_ids_sha256 does not match candidate file")


def _validate_inputs(candidates_path: str, artifacts_dir: str, out_path: str) -> None:
    """Fail fast on bad inputs before doing any expensive work."""
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
    if os.path.getsize(candidates_path) == 0:
        raise ValueError(f"Candidates file is empty: {candidates_path}")
    out_dir = os.path.dirname(out_path) or "."
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Output directory does not exist: {out_dir}")


def run(candidates_path: str, artifacts_dir: str, out_path: str):
    t0 = time.time()
    _validate_inputs(candidates_path, artifacts_dir, out_path)

    # ── check artifacts exist ───────────────────────────────────────────
    required = ["jd_emb.npy", "cand_embs.npy", "ids.npy", "tfidf.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(artifacts_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {artifacts_dir}: {', '.join(missing)}. "
            "Run precompute.py first: python -m src.precompute --candidates <path> --artifacts ./artifacts"
        )

    # ── load artifacts ──────────────────────────────────────────────────
    jd_emb = np.load(os.path.join(artifacts_dir, "jd_emb.npy"))
    cand_embs = np.load(os.path.join(artifacts_dir, "cand_embs.npy"))
    ids = np.load(os.path.join(artifacts_dir, "ids.npy"), allow_pickle=False)

    with open(os.path.join(artifacts_dir, "tfidf.pkl"), "rb") as f:
        vec, matrix = pickle.load(f)

    # ── stream candidates + extract features ───────────────────────────
    print("Loading candidates and extracting features...")
    feats = []
    for cand in stream_candidates(candidates_path):
        feats.append(extract(cand))
    print(f"  {len(feats)} candidates loaded in {time.time()-t0:.1f}s")
    if not feats:
        raise ValueError(f"No candidates found in {candidates_path}")
    _validate_artifacts(artifacts_dir, ids, cand_embs, matrix, feats)

    # ── compute component scores ───────────────────────────────────────
    t1 = time.time()
    print("Scoring...")

    # Semantic scores (vectorized)
    sem_scores = cosine_scores(jd_emb, cand_embs)

    # Lexical scores (vectorized)
    jd_text = feats[0].get("text_blob", "") if feats else ""
    # Load JD text from artifacts
    jd_text_path = os.path.join(artifacts_dir, "jd_text.txt")
    if os.path.exists(jd_text_path):
        with open(jd_text_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
    lex_scores = lexical_scores(vec, matrix, jd_text)

    # Per-candidate scores
    results = []
    for i, feat in enumerate(feats):
        g = gate(feat)
        sem = float(sem_scores[i])
        lex = float(lex_scores[i])
        sk_ev = skill_evidence_score(feat)
        car = career_fit_score(feat)
        yoe_b = feat.get("yoe_band", 0.0)
        loc = location_score(feat)
        behav = behavioral_multiplier(feat)
        score = compute_score(g, sem, lex, sk_ev, car, yoe_b, loc, behav)

        results.append({
            "candidate_id": feat["candidate_id"],
            "score": score,
            "gate": g,
            "feat": feat,
            "sem": sem,
            "lex": lex,
            "sk_ev": sk_ev,
            "car": car,
            "yoe_b": yoe_b,
            "loc": loc,
            "behav": behav,
        })

    print(f"  Scoring done in {time.time()-t1:.1f}s")

    # ── sort by bi-encoder score desc, then candidate_id asc ───────────
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # ── take top N for cross-encoder re-ranking ─────────────────────────
    top_n = [r for r in results[:C.RERANK_TOP_N] if r["gate"] == 1]
    print(f"  Bi-encoder top {len(top_n)} (gate-passed) selected for re-ranking")

    # ── cross-encoder re-ranking ───────────────────────────────────────
    jd_text_path = os.path.join(artifacts_dir, "jd_text.txt")
    jd_text = ""
    if os.path.exists(jd_text_path):
        with open(jd_text_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

    ce_model_path = os.path.join(
        artifacts_dir, "models",
        C.CROSS_ENCODER_MODEL.replace("/", "_"),
    )

    if len(top_n) > 100 and os.path.exists(ce_model_path):
        top_n = ce_rerank(jd_text, top_n, model_path=ce_model_path)
    elif len(top_n) > 100:
        raise FileNotFoundError(
            f"Cross-encoder model not found at {ce_model_path}. "
            "Run python scripts/download_models.py --artifacts ./artifacts before precompute/ranking."
        )
    else:
        for r in top_n:
            r["final_score"] = r["score"]

    # ── take top 100 from re-ranked list ───────────────────────────────
    top100 = top_n[:100]
    if len(top100) != 100:
        raise ValueError(f"Expected at least 100 gate-passed candidates, found {len(top100)}")

    # ── re-sort top 100 by rounded score for correct tie-breaking ─────
    for r in top100:
        r["rounded_score"] = round(r["final_score"], 4)
    top100.sort(key=lambda x: (-x["rounded_score"], x["candidate_id"]))

    # ── generate reasoning ─────────────────────────────────────────────
    for rank, r in enumerate(top100, start=1):
        r["rank"] = rank
        r["reasoning"] = generate_reasoning(r["feat"], rank)

    # ── write CSV ──────────────────────────────────────────────────────
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in top100:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['rounded_score']:.4f}",
                r["reasoning"],
            ])

    total = time.time() - t0
    print(f"\nDone. Top-100 written to {out_path}")
    print(f"  Total time: {total:.1f}s")

    # ── stats ──────────────────────────────────────────────────────────
    gate_pass = sum(1 for r in results if r["gate"] == 1)
    gate_fail = sum(1 for r in results if r["gate"] == 0)
    honeypots_top100 = sum(1 for r in top100 if r["gate"] == 0)
    print(f"  Gate pass: {gate_pass}, fail: {gate_fail}")
    print(f"  Honeypots in top-100: {honeypots_top100}/100")
    if honeypots_top100 > 10:
        print("  WARNING: honeypot rate > 10% in top-100 — submission would be DQ!")


def main():
    parser = argparse.ArgumentParser(description="Rank candidates")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--artifacts", default=C.ARTIFACTS_DIR, help="Artifacts dir")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    run(args.candidates, args.artifacts, args.out)


if __name__ == "__main__":
    main()
