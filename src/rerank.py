"""Cross-encoder re-ranking.

Takes the top N candidates from bi-encoder scoring and re-ranks them
using a cross-encoder that reads (JD, candidate) pairs jointly.

The cross-encoder provides a corrective signal — it catches cases where
bi-encoder cosine similarity is wrong (e.g. scoring CV work as retrieval).
But the bi-encoder is still valuable for broad semantic matching. The final
score blends both to prevent overcorrection in either direction.
"""
import os
import time

import numpy as np

from . import config as C
from .scoring import compute_score


def load_cross_encoder(model_path: str | None = None):
    from sentence_transformers import CrossEncoder

    if model_path and os.path.exists(model_path):
        return CrossEncoder(model_path)
    return CrossEncoder(C.CROSS_ENCODER_MODEL)


def build_ce_input(feat: dict) -> str:
    """Build text for the cross-encoder.

    Uses the full career description text (what they actually built)
    plus the text_blob (headline, summary, skill names) for tooling context.
    Keeps under 512 tokens for ms-marco-MiniLM compatibility.
    """
    career_text = feat.get("career_text", "")
    text_blob = feat.get("text_blob", "")

    ce_text = career_text[:400] + " " + text_blob[:512 - len(career_text[:400]) - 1]
    return ce_text[:500]


def rerank(
    jd_text: str,
    candidates: list[dict],
    model_path: str | None = None,
) -> list[dict]:
    if not candidates:
        return candidates
    required_keys = {"feat", "sem", "lex", "sk_ev", "car", "yoe_b", "loc", "behav", "gate", "candidate_id"}
    missing = required_keys - set(candidates[0].keys())
    if missing:
        raise ValueError(f"Candidate dict missing required keys: {missing}")
    t0 = time.time()

    print("Loading cross-encoder...")
    model = load_cross_encoder(model_path)
    print(f"  Cross-encoder loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    n = len(candidates)
    print(f"Re-ranking {n} candidates with cross-encoder...")

    pairs = []
    for c in candidates:
        cand_text = build_ce_input(c["feat"])
        pairs.append((jd_text, cand_text))

    ce_raw = model.predict(pairs, batch_size=32, show_progress_bar=False)
    ce_raw = np.asarray(ce_raw, dtype=np.float32)

    ce_min, ce_max = ce_raw.min(), ce_raw.max()
    if ce_max > ce_min:
        ce_normalized = (ce_raw - ce_min) / (ce_max - ce_min)
    else:
        ce_normalized = np.ones_like(ce_raw) * 0.5

    for i, c in enumerate(candidates):
        ce = float(ce_normalized[i])
        bi_sem = c["sem"]
        blended_sem = 0.45 * ce + 0.55 * bi_sem

        final = compute_score(
            c["gate"],
            blended_sem,
            c["lex"],
            c["sk_ev"],
            c["car"],
            c["yoe_b"],
            c["loc"],
            c["behav"],
        )
        c["ce_score"] = ce
        c["final_score"] = final

    candidates.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    print(f"  Re-ranking done in {time.time()-t1:.1f}s")
    print(f"  CE raw range: {ce_min:.2f} -> {ce_max:.2f}")
    print(f"  Final score range: {candidates[0]['final_score']:.4f} "
          f"-> {candidates[-1]['final_score']:.4f}")

    return candidates
