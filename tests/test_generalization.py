"""Test generalization across different candidate titles."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract
from src.scoring import skill_evidence_score, career_fit_score, location_score, behavioral_multiplier
from src.io import stream_candidates

cands_by_title = {}
for c in stream_candidates(r"../data/India_runs_data_and_ai_challenge/candidates.jsonl"):
    title = c["profile"]["current_title"]
    if title not in cands_by_title:
        cands_by_title[title] = c
    if len(cands_by_title) >= 30000:
        break

test_titles = [
    "Marketing Manager", "Accountant", "HR Manager",
    "AI Engineer", "ML Engineer", "Software Engineer",
    "DevOps Engineer", "Data Scientist", "Backend Engineer",
    "Senior AI Engineer", "Staff Machine Learning Engineer",
]

print("HOW THE SYSTEM TREATS DIFFERENT TITLES (current AI-Engineer config):")
print()
header = f"{'Title':35s} {'Archtype':15s} {'YOE':>7s} {'SkillEv':>7s} {'Career':>7s} {'Loc':>7s} {'Behav':>7s}"
print(header)
print("-" * 95)

for title in test_titles:
    if title not in cands_by_title:
        continue
    cand = cands_by_title[title]
    feat = extract(cand)

    t = feat["title"]
    yoe = feat["yoe_band"]
    se = skill_evidence_score(feat)
    car = career_fit_score(feat)
    loc = location_score(feat)
    behav = behavioral_multiplier(feat)

    row = f"{title:35s} {t:15s} {yoe:7.3f} {se:7.3f} {car:7.3f} {loc:7.3f} {behav:7.3f}"
    print(row)

print()
print("KEY INSIGHT:")
print("  Non-AI titles -> career_fit ~0.15, skill_evidence ~0.0 -> correctly demoted")
print("  But this ONLY works because config.py was handwritten for THIS JD.")
print()
print("  For a Marketing Manager JD, you would need to:")
print("  1. Rewrite skill taxonomy: replace FAISS/RAG with SEO/content/marketing skills")
print("  2. Flip title archetypes: marketing manager -> high score, ai_engineer -> low")
print("  3. Change disqualifiers: add 'no marketing experience' instead of 'no retrieval'")
print("  4. Update career_signals in reasoning.py: 'ran campaigns' instead of 'built FAISS'")
print()
print("  The core architecture (bi-encoder + gate + scoring + reasoning) is REUSABLE.")
print("  Only config.py and keyword lists need changes.")
