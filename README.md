# INDIA RUNS — Intelligent Candidate Discovery & Ranking

> **Team: The Monolith** | Mohana Krishna | codexmohan@gmail.com
> Redrob AI Hiring Challenge — INDIA RUNS

---

## What this does

Ranks 100,000 candidate profiles against one hand-written job description and produces the top 100 — the way a great recruiter would, not the way a keyword filter would.

**The JD is a trap.** It's written so that candidates who *list* the most AI keywords (HR Managers with "RAG" and "Pinecone" on their profile) rank at the top of naive systems. The sample submission proves it — HR Managers and Civil Engineers at ranks 1–10. Our system sees through this.

## How it works

Two-phase hybrid ranker:

### Phase 1: Precompute (run once, ~6 min on GPU)
```bash
python -m src.precompute \
  --candidates ../data/India_runs_data_and_ai_challenge/candidates.jsonl \
  --artifacts ./artifacts
```
- Encodes all 100K candidate profiles using `all-MiniLM-L6-v2` (384-d embeddings)
- Builds TF-IDF index over skills + career descriptions
- Extracts per-candidate feature vectors
- Downloads model from HuggingFace on first run (cached locally after)

### Phase 2: Rank (< 30 sec on CPU, no network)
```bash
python -m src.rank \
  --candidates ../data/India_runs_data_and_ai_challenge/candidates.jsonl \
  --artifacts ./artifacts \
  --out ./codexmohan_6487.csv
```

### Validate
```bash
python ../data/India_runs_data_and_ai_challenge/validate_submission.py ./codexmohan_6487.csv
```

## The scoring formula

```
gate         = 0  if honeypot OR hard disqualifier
raw_score    = 0.30×semantic + 0.20×lexical + 0.25×skill_evidence
             + 0.15×career_fit + 0.05×yoe_band + 0.05×location
behavioral   ∈ [0.5, 1.15]   (multiplicative envelope)
final_score  = gate × raw_score × behavioral
```

### What each component does

| Component | How | Why |
|---|---|---|
| **Semantic** | `all-MiniLM-L6-v2` embedding cosine(JD, candidate) | Catches "built a recommendation system" matching "ranking/retrieval" even without shared keywords |
| **Lexical** | TF-IDF cosine over skills + career text | Catches exact tech names (Pinecone, Qdrant, NDCG) |
| **Skill-evidence** | Match against curated JD taxonomy × trust factor | A skill with 0 endorsements + 0 months = ~0 contribution. **This kills keyword-stuffers.** |
| **Career-fit** | Title archetype + product vs services + eval/scale experience | "AI Engineer at Razorpay" scores higher than "HR Manager at TCS" |
| **YOE band** | Gaussian peaking at 6.5 yr (JD's "5–9") | Rewards the sweet spot, doesn't punish 4 or 12 |
| **Location** | Pune/Noida/Tier-1 India = 1.0; other India = 0.8; relocate = 0.6; outside = 0.3 | Matches JD's preference |
| **Behavioral** | open_to_work, response rate, recency, interview completion, offer acceptance | A perfect-on-paper candidate who hasn't logged in for 6 months is not actually available |

### The honeypot gate

Before scoring, a gate checks for:
- **Impossible profiles**: "expert" in 10+ skills with 0 months usage each
- **Consulting-only careers**: TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini
- **Title-chasers**: Senior → Staff → Principal every ~1.5 yr
- **Research-only**: Academic background without production deployment
- **No recent code**: Tech-lead/architecture-only for 18+ months

Candidates failing the gate get score 0 and never appear in the top 100.

## Results

- **100,000 candidates** → gate passes ~35,000 → top 100 selected
- **0 honeypots** in top-100 (well under 10% DQ threshold)
- **Top candidates**: AI/ML engineers at product companies (Apple, Paytm, Meta, Razorpay, Microsoft, Amazon)
- **Ranking time**: ~27 seconds (well under 5-min limit)

## Reasoning

Every candidate in the top 100 gets a factual 1–2 sentence reasoning built from real profile fields only. Three tone tiers:
- **Ranks 1–25**: assertive ("AI Engineer with 6.4 yrs at product cos; qualified on retrieval/embeddings...")
- **Ranks 26–75**: trade-off-aware ("...strong on X but 120-day notice")
- **Ranks 76–100**: hesitant ("Adjacent — Operations Manager; surfaced for X but below bar on Y")

No LLM at rank time. No hallucination. Deterministic template variants for variation.

## Project structure

```
indiaruns-ranker/
  src/
    config.py          — skill taxonomy, weights, thresholds (single source of truth)
    io.py              — JSONL streaming reader
    features.py        — per-candidate feature extraction
    semantic.py        — embedding build + cosine similarity
    sparse.py          — TF-IDF index
    honeypot.py        — gate detector (honeypot + hard disqualifiers)
    scoring.py         — final score formula
    reasoning.py       — templated, factual reasonings
    precompute.py      — Phase A: build artifacts (may exceed 5 min)
    rank.py            — Phase B: rank candidates (< 5 min, CPU, no network)
  data/
    sample/            — 100-candidate sample for sandbox testing
  sandbox/
    app.py             — Gradio HF Space (Section 10.5)
    Dockerfile         — Option 2: self-contained CLI
  docs/
    deck.md            — presentation outline
  codexmohan_6487.csv  — submission file
  submission_metadata.yaml
  AGENTS.md            — project contract for AI sessions
```

## Sandbox

**Option 1 — Gradio HF Space**: Upload a candidates JSONL, click "Rank", download CSV.
**Option 2 — Docker**:
```bash
docker build -t indiaruns-ranker .
docker run --rm indiaruns-ranker --candidates data/sample/sample_candidates.jsonl --out /output/ranked.csv
```

## Team

| | |
|---|---|
| **Team name** | The Monolith |
| **Member** | Mohana Krishna |
| **Location** | Vellore, Tamil Nadu, India |
| **Email** | codexmohan@gmail.com |
| **Phone** | +91-6381131277 |
