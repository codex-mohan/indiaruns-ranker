# indiaruns-ranker

**The Monolith** — Redrob AI INDIA RUNS hackathon entry.

AI candidate ranking system that understands *who fits the role*, not just who matches keywords.

## Quick start

```bash
pip install -r requirements.txt
```

### Reproduce the submission CSV

```bash
python -m src.rank \
  --candidates ../data/India_runs_data_and_ai_challenge/candidates.jsonl \
  --artifacts ./artifacts \
  --out ./codexmohan_6487.csv
```

### Validate

```bash
python ../data/India_runs_data_and_ai_challenge/validate_submission.py codexmohan_6487.csv
```

### Precompute artifacts (if artifacts/ is empty)

```bash
python -m src.precompute \
  --candidates ../data/India_runs_data_and_ai_challenge/candidates.jsonl \
  --artifacts ./artifacts
```

## Architecture

Two-phase hybrid ranker:

1. **Precompute** (`src/precompute.py`): Embed all 100K candidates via `all-MiniLM-L6-v2`, build TF-IDF index, extract feature vectors.
2. **Rank** (`src.rank.py`): Score candidates via gate (honeypot/disqualifier) × (0.30·semantic + 0.20·lexical + 0.25·skill_evidence + 0.15·career_fit + 0.05·yoe + 0.05·loc) × behavioral_mult. Output top-100 CSV with factual reasoning.

### Key design decisions
- **Config-driven**: all weights, thresholds, and taxonomy live in `src/config.py` — single source of truth.
- **Trust-weighted skills**: a skill with 0 endorsements + 0 months contributes ~0 — defeats keyword-stuffers.
- **Honeypot gate**: impossible-profile detection before scoring — keeps honeypot rate well under the 10% DQ threshold.
- **Reasoning from real fields only**: no hallucination, three tone tiers, deterministic template variants.

## Sandbox (Section 10.5)

**Option 1 — Gradio HF Space**: `sandbox/app.py` wraps `rank.py` in a thin upload/preview/download UI.

**Option 2 — Docker CLI**:
```bash
docker build -t indiaruns-ranker .
docker run --rm -v "$(pwd)/output:/app/output" indiaruns-ranker \
  --candidates data/sample/sample_candidates.jsonl \
  --artifacts artifacts \
  --out output/ranked.csv
```

## Submission

- CSV filename: `codexmohan_6487.csv`
- Team: The Monolith
- Contact: Mohana Krishna (codexmohan@gmail.com)
