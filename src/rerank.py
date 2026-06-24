"""Cross-encoder re-ranking.

Takes the top N candidates from bi-encoder scoring and re-ranks them
using a cross-encoder that reads (JD, candidate) pairs jointly.

Unlike bi-encoder cosine similarity (where JD and candidate are encoded
separately and never "see" each other), a cross-encoder processes both
texts together — so it can reason about *why* a candidate fits (or doesn't),
not just how similar the vocabulary is.

Example of what this catches that bi-encoders miss:
  JD wants: "embeddings, retrieval, ranking, vector search"
  Candidate: "Built computer vision models for image moderation,
              fine-tuned ResNet on 200K images"
  Bi-encoder: sees "model", "fine-tuned", "production" -> moderate similarity
  Cross-encoder: reads both texts together -> "this is CV, not IR" -> low score
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


def extract_career_descriptions(feat: dict) -> str:
    """Build text for cross-encoder: career descriptions + evidenced skills.

    Career descriptions show what they built (substance).
    Skill names show tooling expertise (overlap with JD requirements).
    Both together give the cross-encoder a complete picture.
    """
    career_text = feat.get("career_text", "")
    se = feat.get("skills_evidenced", {})
    evidenced_skills = [k for k, v in se.items() if v > 0]
    labels = {
        "retrieval_must": "retrieval/embeddings/vector-search",
        "retrieval_nice": "search/ranking/recsys",
        "llm_finetune": "LLM-fine-tuning",
        "ml_support": "ML-frameworks",
    }
    skill_parts = []
    for cat in ["retrieval_must", "retrieval_nice", "llm_finetune", "ml_support"]:
        if cat in evidenced_skills:
            skill_parts.append(labels.get(cat, cat))
    skill_str = ", ".join(skill_parts) if skill_parts else ""

    cand_text = f"Work: {career_text[:400]}. Skills: {skill_str}."
    return cand_text[:512]


def rerank(
    jd_text: str,
    candidates: list[dict],
    model_path: str | None = None,
) -> list[dict]:
    t0 = time.time()

    print("Loading cross-encoder...")
    model = load_cross_encoder(model_path)
    print(f"  Cross-encoder loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    n = len(candidates)
    print(f"Re-ranking {n} candidates with cross-encoder...")

    pairs = []
    for c in candidates:
        cand_text = extract_career_descriptions(c["feat"])
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
        final = compute_score(
            c["gate"],
            ce,
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