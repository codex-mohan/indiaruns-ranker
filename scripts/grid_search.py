"""Grid search over scoring weights using proxy metrics.

Precomputes all component scores once, then sweeps weight combinations
by recomputing the linear combination only. No cross-encoder reranking
per combination — uses the existing blended semantic from artifacts.
"""
import argparse
import csv
import itertools
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C
from src.io import stream_candidates
from src.features import extract
from src.semantic import cosine_scores
from src.sparse import lexical_scores
from src.honeypot import gate, honeypot_score
from src.scoring import (
    skill_evidence_score,
    career_fit_score,
    location_score,
    behavioral_multiplier,
)


# ── known trap titles that should NOT appear in top 100 ─────────────────
_TRAP_TITLES = {
    "marketing manager", "accountant", "hr manager", "operations manager",
    "project manager", "business analyst", "customer support",
    "graphic designer", "civil engineer", "mechanical engineer",
}


def _precompute_components(candidates_path, artifacts_dir):
    """Load artifacts and compute all component scores once."""
    t0 = time.time()

    jd_emb = np.load(os.path.join(artifacts_dir, "jd_emb.npy"))
    cand_embs = np.load(os.path.join(artifacts_dir, "cand_embs.npy"))
    ids = np.load(os.path.join(artifacts_dir, "ids.npy"), allow_pickle=False)

    with open(os.path.join(artifacts_dir, "tfidf.pkl"), "rb") as f:
        vec, matrix = pickle.load(f)

    jd_text_path = os.path.join(artifacts_dir, "jd_text.txt")
    with open(jd_text_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    # vectorized scores
    sem_scores = cosine_scores(jd_emb, cand_embs)
    lex_scores = lexical_scores(vec, matrix, jd_text)

    # per-candidate features and component scores
    gates = []
    sems = []
    lexs = []
    sk_evs = []
    cars = []
    yoe_bs = []
    locs = []
    behavs = []
    titles = []
    candidate_ids = []

    print("Extracting features and component scores...")
    for i, cand in enumerate(stream_candidates(candidates_path)):
        feat = extract(cand)
        gates.append(gate(feat))
        sems.append(float(sem_scores[i]))
        lexs.append(float(lex_scores[i]))
        sk_evs.append(skill_evidence_score(feat))
        cars.append(career_fit_score(feat))
        yoe_bs.append(feat.get("yoe_band", 0.0))
        locs.append(location_score(feat))
        behavs.append(behavioral_multiplier(feat))
        titles.append(feat.get("title_raw", "").lower())
        candidate_ids.append(feat["candidate_id"])

    elapsed = time.time() - t0
    print(f"  {len(candidate_ids)} candidates processed in {elapsed:.1f}s")

    return {
        "gates": np.array(gates),
        "sems": np.array(sems),
        "lexs": np.array(lexs),
        "sk_evs": np.array(sk_evs),
        "cars": np.array(cars),
        "yoe_bs": np.array(yoe_bs),
        "locs": np.array(locs),
        "behavs": np.array(behavs),
        "titles": titles,
        "candidate_ids": candidate_ids,
    }


def _evaluate_top100(comps, weights):
    """Score a weight combination and return proxy metrics."""
    w_sem, w_lex, w_sk, w_car, w_yoe, w_loc = weights

    raw = (
        w_sem * comps["sems"]
        + w_lex * comps["lexs"]
        + w_sk * comps["sk_evs"]
        + w_car * comps["cars"]
        + w_yoe * comps["yoe_bs"]
        + w_loc * comps["locs"]
    )
    final = comps["gates"] * raw * comps["behavs"]

    # rank by final score desc, then candidate_id asc for determinism
    cid_arr = np.array(comps["candidate_ids"])
    order = np.lexsort((cid_arr, -final))
    top100 = order[:100]

    # honeypot rate
    honeypots = int(np.sum(comps["gates"][top100] == 0))

    # trap titles in top 100
    traps = sum(1 for i in top100 if comps["titles"][i] in _TRAP_TITLES)

    # score spread
    top_scores = final[top100]
    score_spread = float(top_scores.max() - top_scores.min()) if len(top_scores) > 0 else 0.0

    # top-10 averages
    top10 = top100[:10]
    avg_sk = float(np.mean(comps["sk_evs"][top10]))
    avg_car = float(np.mean(comps["cars"][top10]))
    avg_yoe = float(np.mean(comps["yoe_bs"][top10]))
    avg_sem = float(np.mean(comps["sems"][top10]))
    avg_behav = float(np.mean(comps["behavs"][top10]))

    # composite proxy: maximize skill + career + semantic, penalize honeypots and traps
    proxy = (
        avg_sk * 0.25
        + avg_car * 0.20
        + avg_sem * 0.20
        + avg_yoe * 0.10
        + score_spread * 0.10
        + avg_behav * 0.05
        - honeypots * 0.50
        - traps * 0.30
    )

    return {
        "proxy": proxy,
        "honeypots": honeypots,
        "traps": traps,
        "score_spread": score_spread,
        "avg_sk": avg_sk,
        "avg_car": avg_car,
        "avg_yoe": avg_yoe,
        "avg_sem": avg_sem,
        "avg_behav": avg_behav,
    }


def _generate_weight_combos(step=0.05):
    """Generate weight tuples that sum to 1.0."""
    values = [round(x * step, 2) for x in range(1, int(1.0 / step))]
    combos = []
    for w_sem in values:
        for w_lex in values:
            if w_sem + w_lex >= 1.0:
                continue
            for w_sk in values:
                if w_sem + w_lex + w_sk >= 1.0:
                    continue
                for w_car in values:
                    remaining = round(1.0 - w_sem - w_lex - w_sk - w_car, 2)
                    if remaining < 0.10:  # need at least 0.05 each for yoe + loc
                        continue
                    # split remaining between yoe and loc
                    for w_yoe in [round(x * step, 2) for x in range(1, int(remaining / step))]:
                        w_loc = round(remaining - w_yoe, 2)
                        if w_loc >= step:
                            combos.append((w_sem, w_lex, w_sk, w_car, w_yoe, w_loc))
    return combos


def run(candidates_path, artifacts_dir, out_dir, step=0.05, top_n=20):
    os.makedirs(out_dir, exist_ok=True)

    comps = _precompute_components(candidates_path, artifacts_dir)

    combos = _generate_weight_combos(step)
    print(f"Evaluating {len(combos)} weight combinations (step={step})...")

    results = []
    t0 = time.time()
    for i, weights in enumerate(combos):
        metrics = _evaluate_top100(comps, weights)
        results.append({"weights": weights, **metrics})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(combos)} evaluated in {time.time()-t0:.1f}s")

    # sort by proxy score descending
    results.sort(key=lambda x: -x["proxy"])

    # save full results
    results_path = os.path.join(out_dir, "grid_search_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results[:top_n], f, indent=2)

    # save CSV summary
    summary_path = os.path.join(out_dir, "grid_search_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "w_sem", "w_lex", "w_sk", "w_car", "w_yoe", "w_loc",
            "proxy", "honeypots", "traps", "score_spread",
            "avg_sk", "avg_car", "avg_yoe", "avg_sem", "avg_behav",
        ])
        for rank, r in enumerate(results[:top_n], start=1):
            w = r["weights"]
            writer.writerow([
                rank, w[0], w[1], w[2], w[3], w[4], w[5],
                f"{r['proxy']:.4f}", r["honeypots"], r["traps"],
                f"{r['score_spread']:.4f}",
                f"{r['avg_sk']:.4f}", f"{r['avg_car']:.4f}",
                f"{r['avg_yoe']:.4f}", f"{r['avg_sem']:.4f}",
                f"{r['avg_behav']:.4f}",
            ])

    # report
    best = results[0]
    original = (C.W_SEMANTIC, C.W_LEXICAL, C.W_SKILL_EVIDENCE, C.W_CAREER_FIT, C.W_YOE, C.W_LOCATION)
    orig_metrics = _evaluate_top100(comps, original)

    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE ({len(combos)} combos, {time.time()-t0:.1f}s)")
    print(f"{'='*80}")
    print(f"\nOriginal weights: sem={original[0]}, lex={original[1]}, sk={original[2]}, "
          f"car={original[3]}, yoe={original[4]}, loc={original[5]}")
    print(f"  proxy={orig_metrics['proxy']:.4f}  honeypots={orig_metrics['honeypots']}  "
          f"traps={orig_metrics['traps']}  spread={orig_metrics['score_spread']:.4f}")
    print(f"  top10: sk={orig_metrics['avg_sk']:.3f} car={orig_metrics['avg_car']:.3f} "
          f"yoe={orig_metrics['avg_yoe']:.3f} sem={orig_metrics['avg_sem']:.3f}")

    print(f"\nBest weights:     sem={best['weights'][0]}, lex={best['weights'][1]}, "
          f"sk={best['weights'][2]}, car={best['weights'][3]}, "
          f"yoe={best['weights'][4]}, loc={best['weights'][5]}")
    print(f"  proxy={best['proxy']:.4f}  honeypots={best['honeypots']}  "
          f"traps={best['traps']}  spread={best['score_spread']:.4f}")
    print(f"  top10: sk={best['avg_sk']:.3f} car={best['avg_car']:.3f} "
          f"yoe={best['avg_yoe']:.3f} sem={best['avg_sem']:.3f}")

    print(f"\nTop {top_n} results saved to:")
    print(f"  {results_path}")
    print(f"  {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid search over scoring weights")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--artifacts", default=C.ARTIFACTS_DIR, help="Artifacts dir")
    parser.add_argument("--out", default=os.path.join(C.PROJECT_ROOT, "artifacts", "grid_search"),
                        help="Output directory for results")
    parser.add_argument("--step", type=float, default=0.05, help="Weight increment step")
    parser.add_argument("--top", type=int, default=20, help="Number of top results to save")
    args = parser.parse_args()
    run(args.candidates, args.artifacts, args.out, step=args.step, top_n=args.top)


if __name__ == "__main__":
    main()
