"""Phase B — Ranking entry point (the reproduce command).

Loads pre-computed artifacts, scores all candidates, outputs top-100 CSV.
Must run < 5 min on CPU, no network.
"""
import argparse
import csv
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


def run(candidates_path: str, artifacts_dir: str, out_path: str):
    t0 = time.time()

    # ── check artifacts exist ───────────────────────────────────────────
    required = ["jd_emb.npy", "cand_embs.npy", "ids.npy", "tfidf.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(artifacts_dir, f))]
    if missing:
        print(f"ERROR: Missing artifacts in {artifacts_dir}: {', '.join(missing)}")
        print("Run precompute.py first: python -m src.precompute --candidates <path> --artifacts ./artifacts")
        return

    # ── load artifacts ──────────────────────────────────────────────────
    jd_emb = np.load(os.path.join(artifacts_dir, "jd_emb.npy"))
    cand_embs = np.load(os.path.join(artifacts_dir, "cand_embs.npy"))
    ids = np.load(os.path.join(artifacts_dir, "ids.npy"), allow_pickle=True)

    with open(os.path.join(artifacts_dir, "tfidf.pkl"), "rb") as f:
        vec, matrix = pickle.load(f)

    # ── stream candidates + extract features ───────────────────────────
    print("Loading candidates and extracting features...")
    feats = []
    for cand in stream_candidates(candidates_path):
        feats.append(extract(cand))
    print(f"  {len(feats)} candidates loaded in {time.time()-t0:.1f}s")

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
        print("  Cross-encoder model not found — using bi-encoder scores only")
        for r in top_n:
            r["final_score"] = r["score"]
    else:
        for r in top_n:
            r["final_score"] = r["score"]

    # ── take top 100 from re-ranked list ───────────────────────────────
    top100 = top_n[:100]

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
