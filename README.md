# INDIA RUNS - Intelligent Candidate Discovery & Ranking

> **Team:** The Monolith  
> **Member:** Mohana Krishna  
> **Challenge:** Redrob AI Hiring Challenge - INDIA RUNS

## What This Does

This project ranks 100,000 candidate profiles against the released Senior AI Engineer job description and outputs the best 100 candidates in the required CSV format.

The JD is intentionally adversarial. A naive keyword matcher can over-rank profiles that list AI buzzwords without real production evidence. This ranker is designed to reward actual retrieval, ranking, ML systems, product-engineering, and availability signals instead of raw keyword count.

## Current Status

- Submission file: `codexmohan_6487.csv`
- Official validator: passing
- Latest full ranking run: `225.5s` on CPU, cold start, no network during ranking
- Latest gate result: `35,039` pass, `64,961` fail
- Gated honeypots in top 100: `0/100`
- Runtime constraint: under the 5-minute Stage 3 ranking limit

## How It Works

The system is a two-phase hybrid ranker. Model download/precompute are setup steps; only `src.rank` is the constrained no-network ranking step.

### Optional Setup: Download Model Weights

Run this on a machine with network access before the no-network ranking step:

```bash
python scripts/download_models.py --artifacts ./artifacts
```

This stages the local Hugging Face models under `artifacts/models/`:

- `sentence-transformers/all-MiniLM-L6-v2` for candidate/JD embeddings.
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for top-1,000 re-ranking.

### Phase 1: Precompute

Precompute may use network if models are not already staged. It may exceed the 5-minute ranking budget because it prepares reusable local artifacts, not the final constrained ranking run.

```bash
python -m src.precompute \
  --candidates ../data/India_runs_data_and_ai_challenge/candidates.jsonl \
  --artifacts ./artifacts
```

This step:

- Extracts per-candidate features from profile, career, skills, education, and Redrob signals.
- Encodes candidate text with `sentence-transformers/all-MiniLM-L6-v2`.
- Builds a TF-IDF index over candidate text.
- Saves the JD embedding, candidate embeddings, ordered candidate IDs, TF-IDF artifacts, feature JSONL, JD text, and `manifest.json`.
- Saves both model directories under `artifacts/models/` so ranking can run offline.

`manifest.json` records the candidate count and ordered candidate-ID hash. `rank.py` refuses to score if the current candidate file does not match the precomputed artifacts.

Artifacts are treated as trusted, self-generated files. Do not run `src.rank` against artifact bundles from an unknown source; regenerate them with `src.precompute` when in doubt.

### Phase 2: Rank

Ranking is the constrained reproduction step: CPU only, no hosted APIs, no network required when artifacts are present. It requires the cross-encoder model directory produced by `scripts/download_models.py` or `src.precompute`; missing model/artifact files are fatal because fallback ranking would not reproduce the submitted CSV.

```bash
python -m src.rank \
  --candidates ../data/India_runs_data_and_ai_challenge/candidates.jsonl \
  --artifacts ./artifacts \
  --out ./codexmohan_6487.csv
```

This step:

- Loads precomputed embeddings, TF-IDF, ordered IDs, JD text, manifest, and cached models.
- Validates artifact/candidate alignment before scoring.
- Extracts current candidate features from `candidates.jsonl`.
- Applies honeypot and JD hard-disqualifier gates.
- Computes semantic, lexical, skill-evidence, career-fit, experience, location, and behavioral scores.
- Re-ranks the top 1,000 gate-passed candidates with the cached cross-encoder.
- Writes the top 100 candidates with deterministic, field-grounded reasoning.

### Validate

```bash
python ../data/India_runs_data_and_ai_challenge/validate_submission.py ./codexmohan_6487.csv
```

## Scoring Formula

```text
gate        = 0 if honeypot OR hard disqualifier, else 1

raw_score   = 0.30 * semantic
            + 0.20 * lexical
            + 0.25 * skill_evidence
            + 0.15 * career_fit
            + 0.05 * yoe_band
            + 0.05 * location

final_score = gate * raw_score * behavioral_multiplier
```

The cross-encoder is used as a corrective semantic signal for the top 1,000 candidates:

```text
semantic = 0.45 * cross_encoder_normalized + 0.55 * bi_encoder_cosine
```

## Components

| Component | How it works | Why it matters |
|---|---|---|
| Semantic | MiniLM embedding cosine between JD and candidate text | Finds related experience even when exact wording differs |
| Lexical | TF-IDF cosine over career, skills, title, summary, and profile text | Preserves exact-match signals like FAISS, BM25, NDCG, Pinecone |
| Skill evidence | JD taxonomy matched through proficiency, duration, and endorsements | Suppresses keyword-stuffed skills with no usage evidence |
| Career fit | Title archetype plus eval, scale, education, certification, and company-context signals | Rewards applied ML/search/retrieval profiles over unrelated roles |
| Experience band | Gaussian around the JD's 5-9 year preference. YOE computed from career history dates, not self-reported field. Overreported YOE penalized proportional to sqrt of deviation. | Favors the senior IC sweet spot. Catches typos and fraud. |
| Location | Pune/Noida/Tier-1 India preference, relocation fallback | Matches the JD logistics |
| Behavioral | Recency, response rate, interview completion, offer acceptance, active applications, recruiter interest, GitHub, notice period | Down-weights candidates who are strong on paper but unlikely to engage. No self-reported flags — only observed behavior |

## Honeypot And Gate Logic

Before scoring, the gate removes candidates that should not compete for top-100 slots.

Honeypot signatures currently checked:

- Impossible or unreasonable years of experience.
- Three or more `expert` skills with `0` months of usage.
- Eight or more skills with both `0` endorsements and `0` months of usage.
- Three or more advanced/expert claims with no endorsement and no duration evidence.

JD hard disqualifiers currently checked:

- Consulting-only career history (all companies are consulting firms).
- Current consulting role penalty: candidates currently at a consulting firm (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, Genpact, etc.) receive a moderate career-fit penalty even if prior roles were at product companies. Per the JD: "currently at one of these companies but have prior product-company experience, that's fine" — these candidates are not disqualified, but ranked lower than equivalent product-company candidates.
- Research-heavy background without production-code evidence.
- Title-chaser pattern with very short average tenure.
- Tech-lead/architecture-only profile with no recent production-code signal.
- Closed-source-only services/consulting background without external validation.
- No relevant retrieval or ML-support skills.

## Reasoning

Each selected candidate gets a deterministic 1-2 sentence explanation built from actual candidate fields only. No LLM is called at rank time.

Reasoning tiers:

- Ranks 1-25: assertive, strength-led reasoning.
- Ranks 26-75: trade-off-aware reasoning with an honest concern when available.
- Ranks 76-100: cautious tail-end reasoning.

The generator references real fields such as title, years of experience, company, evidenced skill categories, career text signals, response rate, recency, location, and notice period.

## Project Structure

```text
indiaruns-ranker/
  src/
    config.py          - taxonomy, weights, thresholds
    io.py              - JSONL streaming reader
    features.py        - per-candidate feature extraction
    semantic.py        - MiniLM embedding helpers
    sparse.py          - TF-IDF helpers
    honeypot.py        - honeypot and hard-disqualifier gate
    scoring.py         - final score formula
    reasoning.py       - deterministic factual reasonings
    precompute.py      - offline artifact builder
    rank.py            - constrained ranking entry point
  data/
    sample/            - small sample for sandbox use
    job_description.md - local JD copy when available
  sandbox/
    app.py             - Gradio sandbox app
  tests/
    *.py               - validation and analysis helpers
  artifacts/           - generated/cached local ranking artifacts
  codexmohan_6487.csv  - current validated submission CSV
  submission_metadata.yaml
```

## Sandbox

Run the Gradio sandbox locally:

```bash
python sandbox/app.py
```

Docker option:

```bash
docker build -t indiaruns-ranker .
docker run --rm indiaruns-ranker --candidates data/sample/sample_candidates.jsonl --out /output/ranked.csv
```

## Notes For Reviewers

- The hidden ground-truth metrics cannot be verified locally because the leaderboard labels are private.
- Local validation confirms format compliance, runtime readiness, deterministic reproduction, and no gated honeypots in the produced top 100.
- The latest measured runtime on this machine was `225.5s` (cold start), well under the 5-minute Stage 3 limit.

## Architecture Decisions

### YOE Computation: Career Dates Over Self-Reported Field

The `years_of_experience` field in candidate profiles is unreliable. We found candidates where the field says 2.9 years but career history shows 6.2 years, and the candidate's own summary says "6.3 years." Rather than trusting a single number field, we compute YOE from career history dates (earliest start → today).

**Logic:**
- If reported ≤ calculated: use calculated (handles typos, underreporting)
- If reported > calculated: penalize with `calculated * max(0.3, sqrt(calculated/reported))` (handles fraud)
- Fallback to reported if no career dates are parseable

This means a candidate who underreports (typo like 2.9 instead of 6.9) gets the correct YOE, while a candidate who overreports (claiming 15yr on a 6yr career) gets a proportional penalty.

### Behavioral Signals: Observed Behavior Over Self-Reported Flags

The JD explicitly says: "a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available."

We removed the `open_to_work_flag` from scoring. It's a self-reported button click. Instead, we use observed signals:
- `recruiter_response_rate`: do they reply to recruiters?
- `last_active_date`: when did they last log in?
- `interview_completion_rate`: do they show up to interviews?
- `offer_acceptance_rate`: do they accept offers?
- `applications_submitted_30d`: are they actively applying?
- `saved_by_recruiters_30d`: are recruiters interested in them?
- `notice_period_days`: how soon can they start?

### Consulting Detection: Current Role Matters

The JD says: "People who have only worked at consulting firms... we've had bad fit experiences. If you're currently at one of these companies but have prior product-company experience, that's fine."

We implement this as a two-tier system:
- **Consulting-only** (all companies are consulting): career_fit × 0.1 (near-disqualification)
- **Current consulting** (current company is consulting, prior is product): career_fit × 0.55 (moderate penalty)

This catches candidates who spent 4 years at Genpact after LinkedIn — they're not disqualified per the JD, but ranked lower than equivalent product-company candidates.

### Location: Hard Penalty for Outside India

The JD says: "Outside India: case-by-case, but we don't sponsor work visas."

We implement this as:
- Tier-1 India (Pune, Noida, Delhi, Mumbai, etc.): 1.0
- Other India: 0.7
- Outside India, willing to relocate: 0.15
- Outside India, no relocation: 0.05

A candidate in New York with strong skills will score near-zero on location, which significantly reduces their final ranking.

### Honeypot Detection

The dataset contains ~80 impossible profiles. We detect:
- Impossible YOE (< 0 or > 45)
- 3+ expert skills with 0 months of usage
- 8+ skills with both 0 endorsements and 0 months
- 3+ advanced/expert claims with no endorsement and no duration

Candidates failing the gate receive score 0 and are excluded from top 100.

### Cross-Encoder Re-Ranking

The top 1,000 gate-passed candidates are re-ranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`, which reads (JD, candidate career text) jointly. This catches false positives the bi-encoder misses — e.g., computer vision work scored as relevant because both share "model", "fine-tuned", "production" keywords.

Final semantic score: `0.45 × cross_encoder_normalized + 0.55 × bi_encoder_cosine`

### Grid Search (Preserved, Not Applied)

We ran a grid search over 10,517 weight combinations (step=0.05). The best proxy score was only 1.5% better than hand-tuned weights, and the top 10 changed significantly (6/10 different candidates). Without ground truth labels, we cannot verify the grid search improves NDCG@10. The weights are preserved on branch `feat/grid-search-weights` for future use.

## Team

| Field | Value |
|---|---|
| Team name | The Monolith |
| Member | Mohana Krishna |
| Location | Vellore, Tamil Nadu, India |
| Email | codexmohan@gmail.com |
| Phone | +91-6381131277 |
