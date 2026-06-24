"""Validate enhancement proposals against real candidate data."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C
from src.io import stream_candidates
from src.features import extract

cands = {}
for c in stream_candidates(r"../data/India_runs_data_and_ai_challenge/candidates.jsonl"):
    cands[c["candidate_id"]] = c

feats = {}
for cid, c in cands.items():
    feats[cid] = extract(c)

top100_ids = []
with open("codexmohan_6487.csv") as f:
    next(f)
    for line in f:
        top100_ids.append(line.split(",")[0])

DIVIDER = "=" * 80

# ── 1. EDUCATION TIER ANALYSIS ─────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("1. EDUCATION TIER ANALYSIS — Top 100 vs All Candidates")
print(DIVIDER)

tier_counts_top100 = {}
tier_counts_all = {}
for cid, c in cands.items():
    edu = c.get("education", [])
    tier = "none"
    for e in edu:
        t = e.get("tier", "unknown")
        if t in ("tier_1", "tier_2"):
            tier = t
            break
        elif t != "unknown" and tier == "none":
            tier = t
    tier_counts_all[tier] = tier_counts_all.get(tier, 0) + 1
    if cid in top100_ids:
        tier_counts_top100[tier] = tier_counts_top100.get(tier, 0) + 1

print(f"\n  {'Tier':<15} {'Top 100':>8} {'All 100K':>10} {'Top100 %':>10} {'All %':>8}")
print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
total_all = sum(tier_counts_all.values())
total_top = sum(tier_counts_top100.values())
for tier in ["tier_1", "tier_2", "tier_3", "tier_4", "unknown", "none"]:
    t_count = tier_counts_top100.get(tier, 0)
    a_count = tier_counts_all.get(tier, 0)
    print(f"  {tier:<15} {t_count:>8} {a_count:>10} {t_count/total_top*100:>9.1f}% {a_count/total_all*100:>7.1f}%")

# Show top-10 candidates' education
print(f"\n  Top 10 candidates' education:")
for cid in top100_ids[:10]:
    c = cands[cid]
    edu = c.get("education", [])
    tier = edu[0].get("tier", "?") if edu else "none"
    inst = edu[0].get("institution", "?") if edu else "none"
    field = edu[0].get("field_of_study", "?") if edu else "none"
    print(f"    {cid} | {tier:8s} | {inst[:30]:30s} | {field}")

# ── 2. CERTIFICATION ANALYSIS ───────────────────────────────────────────────
print(f"\n\n{DIVIDER}")
print("2. CERTIFICATION ANALYSIS — Top 100")
print(DIVIDER)

cert_count_top = 0
cert_relevant_top = 0
cert_relevant_keywords = ["ml", "ai", "aws", "gcp", "azure", "tensorflow", "pytorch",
                          "deep learning", "machine learning", "data science", "nlp"]
for cid in top100_ids:
    c = cands[cid]
    certs = c.get("certifications", [])
    if certs:
        cert_count_top += 1
        for cert in certs:
            name = cert.get("name", "").lower()
            if any(k in name for k in cert_relevant_keywords):
                cert_relevant_top += 1
                break

print(f"\n  Candidates with certifications: {cert_count_top}/100")
print(f"  With AI/ML/cloud certs:          {cert_relevant_top}/100")
if cert_count_top > 0:
    print(f"\n  Sample certifications from top 100:")
    shown = 0
    for cid in top100_ids:
        c = cands[cid]
        certs = c.get("certifications", [])
        for cert in certs:
            if shown >= 10: break
            print(f"    {cid}: {cert.get('name', '?')} ({cert.get('issuer', '?')}, {cert.get('year', '?')})")
            shown += 1
        if shown >= 10: break

# ── 3. SKILL ASSESSMENT ANALYSIS ───────────────────────────────────────────
print(f"\n\n{DIVIDER}")
print("3. SKILL ASSESSMENT ANALYSIS — Top 100")
print(DIVIDER)

relevant_assessments = 0
irrelevant_assessments = 0
relevant_keywords = {"nlp", "embeddings", "vector search", "faiss", "machine learning",
                     "deep learning", "rag", "pytorch", "tensorflow", "python",
                     "information retrieval", "ranking", "recommendation"}
for cid in top100_ids:
    c = cands[cid]
    signals = c.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    for skill, score in assessments.items():
        if skill.lower() in relevant_keywords or any(k in skill.lower() for k in relevant_keywords):
            relevant_assessments += 1
        else:
            irrelevant_assessments += 1

print(f"\n  Relevant assessment entries:     {relevant_assessments}")
print(f"  Irrelevant assessment entries:   {irrelevant_assessments}")
print(f"\n  Sample assessments from top 10:")
for cid in top100_ids[:10]:
    c = cands[cid]
    signals = c.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        items = sorted(assessments.items(), key=lambda x: -x[1])[:3]
        items_str = ", ".join(f"{k}={v:.0f}" for k, v in items)
        print(f"    {cid}: {items_str}")
    else:
        print(f"    {cid}: (no assessments)")

# ── 4. CAREER DESCRIPTION ANALYSIS ─────────────────────────────────────────
print(f"\n\n{DIVIDER}")
print("4. CAREER DESCRIPTION ANALYSIS — Top 10 vs Bottom 10 of Top 100")
print(DIVIDER)

print(f"\n  TOP 10 career descriptions (first 200 chars):")
for cid in top100_ids[:10]:
    c = cands[cid]
    career = c.get("career_history", [])
    if career:
        desc = career[0].get("description", "")[:200].encode("ascii", "replace").decode()
        print(f"    {cid}: {desc}...")

print(f"\n  BOTTOM 10 of top 100 (ranks 91-100) career descriptions:")
for cid in top100_ids[90:]:
    c = cands[cid]
    career = c.get("career_history", [])
    if career:
        desc = career[0].get("description", "")[:200].encode("ascii", "replace").decode()
        print(f"    {cid}: {desc}...")

# ── 5. WHERE WOULD CROSS-ENCODER HELP? ─────────────────────────────────────
print(f"\n\n{DIVIDER}")
print("5. CROSS-ENCODER OPPORTUNITY ANALYSIS")
print(DIVIDER)

# Check score gaps in top 100 — where are candidates clustered?
scores = []
with open("codexmohan_6487.csv") as f:
    next(f)
    for line in f:
        parts = line.strip().split(",", 3)
        rank = int(parts[1])
        score = float(parts[2])
        scores.append((rank, score, parts[0]))

print(f"\n  Score distribution in top 100:")
print(f"    Rank 1:   {scores[0][1]:.4f}")
print(f"    Rank 10:  {scores[9][1]:.4f}")
print(f"    Rank 25:  {scores[24][1]:.4f}")
print(f"    Rank 50:  {scores[49][1]:.4f}")
print(f"    Rank 75:  {scores[74][1]:.4f}")
print(f"    Rank 100: {scores[99][1]:.4f}")
print(f"    Score gap (1→10):  {scores[0][1] - scores[9][1]:.4f}")
print(f"    Score gap (10→25): {scores[9][1] - scores[24][1]:.4f}")
print(f"    Score gap (25→50): {scores[24][1] - scores[49][1]:.4f}")
print(f"    Score gap (50→100):{scores[49][1] - scores[99][1]:.4f}")

# How many candidates have scores in the range 0.6-0.8 (where re-ranking matters)?
from src.honeypot import gate
from src.scoring import skill_evidence_score, career_fit_score, location_score, behavioral_multiplier, compute_score

score_buckets = {"<0.3": 0, "0.3-0.4": 0, "0.4-0.5": 0, "0.5-0.6": 0,
                 "0.6-0.7": 0, "0.7-0.8": 0, ">0.8": 0}
passing_scores = []
for cid, feat in feats.items():
    g = gate(feat)
    if g == 0: continue
    # Approximate score without sem/lex (just components we can compute)
    sk = skill_evidence_score(feat)
    car = career_fit_score(feat)
    loc = location_score(feat)
    behav = behavioral_multiplier(feat)
    # Approximate (sem and lex unknown, assume ~0.5 each)
    approx = compute_score(g, 0.5, 0.3, sk, car, feat["yoe_band"], loc, behav)
    passing_scores.append(approx)
    if approx < 0.3: score_buckets["<0.3"] += 1
    elif approx < 0.4: score_buckets["0.3-0.4"] += 1
    elif approx < 0.5: score_buckets["0.4-0.5"] += 1
    elif approx < 0.6: score_buckets["0.5-0.6"] += 1
    elif approx < 0.7: score_buckets["0.6-0.7"] += 1
    elif approx < 0.8: score_buckets["0.7-0.8"] += 1
    else: score_buckets[">0.8"] += 1

print(f"\n  Gate-passing candidates by approx score:")
for bucket, count in score_buckets.items():
    print(f"    {bucket:10s}: {count:6d}")
print(f"    Total passing: {sum(score_buckets.values())}")
print(f"\n  Candidates in 0.5-0.7 range (where re-ranking matters most): "
      f"{score_buckets['0.5-0.6'] + score_buckets['0.6-0.7']}")