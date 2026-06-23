"""Diagnostic: trace the full pipeline for specific candidates."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C
from src.io import stream_candidates
from src.features import extract
from src.honeypot import gate, hard_disqualifier
from src.scoring import skill_evidence_score, career_fit_score, location_score, behavioral_multiplier
from src.reasoning import generate_reasoning
from src.features import _SKILL_MAP

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


def trace(cid, rank, label=""):
    c = cands[cid]
    feat = feats[cid]
    p = c["profile"]
    print(f"\n{DIVIDER}")
    print(f"RANK {rank} | {cid} | {label}")
    print(DIVIDER)
    print(f"  Title:        {p['current_title']}")
    print(f"  Company:      {p['current_company']}")
    print(f"  YOE:          {p['years_of_experience']}")
    print(f"  Location:     {p['location']}, {p['country']}")
    print(f"  Industry:     {p['current_industry']}")
    print(f"  Skills:       {[s['name'] for s in c['skills'][:8]]}...")
    print(f"  Career path:  ", end="")
    for ch in c["career_history"]:
        print(f"  {ch['title']}@{ch['company']}({ch['duration_months']}mo)", end="")
    print()

    print(f"\n  [FEATURE EXTRACTION]")
    print(f"    Title archetype:       {feat['title']}")
    print(f"    YOE band (Gaussian):   {feat['yoe_band']:.3f}")
    print(f"    Consulting only:       {feat['is_consulting_only']}")
    print(f"    Recent code gap:       {feat['has_recent_code_gap']}")
    print(f"    Title chaser:          {feat['is_title_chaser']}")
    print(f"    Research only:         {feat['is_research_only']}")
    print(f"    Closed source only:    {feat['closed_source_only']}")
    print(f"    Eval experience:       {feat['has_eval_experience']}")
    print(f"    Scale experience:      {feat['has_scale_experience']}")
    print(f"    Location tier1 India:  {feat['location_tier1_india']}")

    print(f"\n  [SKILL EVIDENCE]")
    se = feat["skills_evidenced"]
    for cat, score in sorted(se.items(), key=lambda x: -x[1]):
        print(f"    {cat:20s} = {score:.3f}")
    print(f"    retrieval_must count:  {feat['retrieval_must_count']}")
    print(f"    retrieval_nice count:  {feat['retrieval_nice_count']}")
    print(f"    llm_finetune count:    {feat['llm_finetune_count']}")
    print(f"    ml_support count:      {feat['ml_support_count']}")

    print(f"\n  [BEHAVIORAL SIGNALS]")
    print(f"    Open to work:          {feat['open_to_work']}")
    print(f"    Last active:           {feat['last_active_days_ago']}d ago")
    print(f"    Recruiter response:    {feat['recruiter_response_rate']}")
    print(f"    Interview completion:  {feat['interview_completion_rate']}")
    print(f"    Offer acceptance:      {feat['offer_acceptance_rate']}")
    print(f"    GitHub activity:       {feat['github_activity_score']}")
    print(f"    Notice period:         {feat['notice_period_days']}d")
    print(f"    Salary range:          {feat['salary_min']}-{feat['salary_max']} LPA")

    print(f"\n  [GATE]")
    g = gate(feat)
    print(f"    Gate: {g} ({'PASS' if g == 1 else 'FAIL'})")
    if g == 0:
        disq, reason = hard_disqualifier(feat)
        print(f"    Reason: {reason}")

    print(f"\n  [SCORING]")
    sk_ev = skill_evidence_score(feat)
    car = career_fit_score(feat)
    loc = location_score(feat)
    behav = behavioral_multiplier(feat)
    print(f"    Skill evidence:  {sk_ev:.3f}")
    print(f"    Career fit:      {car:.3f}")
    print(f"    YOE band:        {feat['yoe_band']:.3f}")
    print(f"    Location:        {loc:.3f}")
    print(f"    Behavioral mult: {behav:.3f}")
    print(f"    Reasoning:       {generate_reasoning(feat, rank)}")
    return g, sk_ev, car, loc, behav


# ── TOP CANDIDATE ──────────────────────────────────────────────────────────
trace(top100_ids[0], 1, "TOP CANDIDATE")

# ── KEYWORD STUFFER (HR Manager with many AI skills) ──────────────────────
print(f"\n\n{'#' * 80}")
print("KEYWORD STUFFER DETECTION")
print("#" * 80)
hr_with_skills = []
for cid, c in cands.items():
    feat = feats[cid]
    title = p["current_title"] if (p := c["profile"]) else ""
    if "hr" in title.lower() or "marketing" in title.lower() or "accountant" in title.lower():
        r_skills = [s["name"] for s in c["skills"]
                    if any(s["name"].lower() in group for group in C.RETRIEVAL_MUST)]
        if len(r_skills) >= 3:
            hr_with_skills.append((cid, title, len(r_skills), c["profile"]["current_company"]))

print(f"\n  Found {len(hr_with_skills)} non-AI titles with 3+ retrieval skills (keyword stuffers)")
for cid, title, count, company in hr_with_skills[:5]:
    trace(cid, 999, f"KEYWORD STUFFER: {title} at {company} ({count} retrieval skills)")

# ── CONSULTING-ONLY CANDIDATE ─────────────────────────────────────────────
print(f"\n\n{'#' * 80}")
print("CONSULTING-ONLY DETECTION")
print("#" * 80)
consulting_only = []
for cid, feat in feats.items():
    if feat["is_consulting_only"]:
        c = cands[cid]
        consulting_only.append((cid, c["profile"]["current_title"], c["profile"]["current_company"]))

print(f"\n  Found {len(consulting_only)} consulting-only candidates")
if consulting_only:
    cid, title, company = consulting_only[0]
    trace(cid, 999, f"CONSULTING-ONLY: {title} at {company}")

# ── HONEYPOT CANDIDATE ───────────────────────────────────────────────────
print(f"\n\n{'#' * 80}")
print("HONEYPOT DETECTION")
print("#" * 80)
honeypots = []
for cid, feat in feats.items():
    g = gate(feat)
    if g == 0:
        disq, reason = hard_disqualifier(feat)
        if reason:
            c = cands[cid]
            honeypots.append((cid, c["profile"]["current_title"], c["profile"]["current_company"], reason))

print(f"\n  Found {len(honeypots)} candidates gated out")
for cid, title, company, reason in honeypots[:5]:
    trace(cid, 999, f"GATED: {title} at {company} (reason: {reason})")

# ── GHOST CANDIDATE (perfect on paper, not available) ─────────────────────
print(f"\n\n{'#' * 80}")
print("GHOST CANDIDATE DETECTION")
print("#" * 80)
ghosts = []
for cid, feat in feats.items():
    if (feat["last_active_days_ago"] > 150 and
        feat["recruiter_response_rate"] < 0.15 and
        feat["retrieval_must_count"] >= 2):
        c = cands[cid]
        ghosts.append((cid, c["profile"]["current_title"], c["profile"]["current_company"],
                       feat["last_active_days_ago"], feat["recruiter_response_rate"]))

print(f"\n  Found {len(ghosts)} ghost candidates (strong skills but inactive/unresponsive)")
for cid, title, company, days, rrr in ghosts[:3]:
    trace(cid, 999, f"GHOST: {title} at {company} (inactive {days}d, response {rrr})")
