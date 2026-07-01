# AGENTS.md — TalentLens

> Read this before touching anything. It's the contract that keeps the project
> coherent across sessions.

## 1. What this project is

**The Monolith** — sole-member entry for the **INDIA RUNS** hiring challenge
(run by Redrob AI). Goal: rank 100,000 candidate profiles against one
hand-written job description and produce the top-100 in a submission CSV.

- **Team**: The Monolith
- **Owner**: Mohana Krishna (Vellore, Tamil Nadu, India)
- **Contact**: 6381131277 · codexmohan@gmail.com
- **GitHub**: @codex-mohan  (`codex-mohan/TalentLens`)
- **Participant ID for CSV filename**: `codexmohan_6487`  → CSV = `codexmohan_6487.csv`

## 2. The dataset (do NOT modify, do NOT commit raw)

Raw bundle extracted to: `../data/India_runs_data_and_ai_challenge/`
(one level up from this repo, sibling of `indiaruns-ranker/`).

Files of record (contents already read into context in prior session):
- `candidates.jsonl`  — 487 MB, 100,000 lines. **The pool.**
- `candidate_schema.json` — JSON schema for one candidate record.
- `job_description.docx` — the JD (read in full). A *trap* by design.
- `redrob_signals_doc.docx` — 23 behavioral signals in each candidate's
  `redrob_signals` object.
- `submission_spec.docx` — the rules (compute limits, metrics, stages).
- `validate_submission.py` — official format validator. Run before submit.
- `sample_submission.csv` — **intentionally bad** example (HR Managers at
  rank 1 because they have "8 AI skills"). Do not treat as ground truth.

Raw `candidates.jsonl` is **gitignored** (too big). Repo ships a small
`data/sample/sample_candidates.jsonl` (~100 candidates) for sandbox/tests.

## 3. The problem in one paragraph

The JD is written as a behavioral/anti-keyword trap. Keyword-counting fails
because the sample submission's top ranker (HR Manager with 8 AI skills) is
explicitly *not* a fit. Real signals per the JD: production
embeddings-retrieval experience, vector DBs / hybrid search, strong Python,
ranking-eval fluency (NDCG/MRR/MAP/A-B), product-company tenure (not pure
services/research), LLM fine-tuning (nice-to-have), and behavioral
availability (recent active, responsive, not ghosting). Honeypots (~80
candidates with impossible profiles) must be filtered or the submission is
disqualified (honeypot rate >10% in top 100).

## 4. Hard rules from the spec (do not violate)

- **Ranking step** (`rank.py`) must run: ≤ 5 min wall, ≤ 16 GB RAM, **CPU
  only, no network** (no OpenAI/Anthropic/etc.). This is enforced at Stage 3
  inside a sandboxed Docker container; if it can't be reproduced, DQ.
- **Pre-computation** (`precompute.py`) is allowed to exceed the 5-min window
  and may use network/disk, but must be reproducible from the repo.
- **CSV format** (validated by `validate_submission.py`): header
  `candidate_id,rank,score,reasoning`, exactly 100 data rows, ranks 1..100
  each once, scores non-increasing, ties broken by ascending `candidate_id`.
- **3 submissions max**; final submission counts.
- **AI tools allowed** but must be declared in `submission_metadata.yaml`.

## 5. The architecture (do not redesign without a strong reason)

**Two-phase hybrid ranker**, config-driven:

```
precompute.py  (offline, may exceed 5 min)         rank.py  (timed, <5min CPU no-net)
─────────────────────────────────────────         ─────────────────────────────────
candidates.jsonl                                  candidates.jsonl (for IDs/order)
   │                                                 │
   ├─ io.py: stream JSONL                            ├─ load embeddings.npy
   ├─ features.py: extract per-cand features        ├─ load tfidf.npz + jd_vec
   ├─ semantic.py: all-MiniLM-L6-v2 emb             ├─ load features.parquet
   ├─ sparse.py: TfidfVectorizer                    ├─ honeypot.py: gate ∈{0,1}
   └─ write artifacts/                               ├─ scoring.py: weighted sum × behavioral
                                                     ├─ take top 100
                                                     ├─ reasoning.py: templated, factual
                                                     └─ write ../codexmohan_6487.csv
```

### Scoring formula (single source of truth: `src/config.py`)
```
gate        = 0  if honeypot signature OR any hard disqualifier (see §6)
raw         = 0.30*semantic + 0.20*lexical + 0.25*skill_evidence
             + 0.15*career_fit + 0.05*yoe_band + 0.05*loc_score
behavioral  ∈ [0.5, 1.15]   (multiplicative envelope from redrob_signals)
score       = gate * raw * behavioral
sort:       score desc, then candidate_id asc
top 100      → CSV
```

### Skill-evidence key idea
Skill names are matched against a curated JD taxonomy, but each match is
weighted by a **trust factor** = f(endorsements, duration_months,
skill_assessment_scores, github_activity_score). A skill with 0
endorsements + 0 months + no assessment contributes ~0 — this is what
defeats keyword-stuffers (HR Manager with "RAG" listed but never used).

### Honeypot & hard-disqualifier gate
`honeypot.py` returns `gate=0` for (full list in that file):
- Impossible profile: "expert" in ≥N skills with 0 months used each; YOE >
  employment window; company tenure longer than company age; offered-rate
  or assessment-score contradictions the schema forbids.
- JD hard disqualifiers (verbatim intent from JD):
  - Pure research w/o production deployment.
  - "AI experience" = only <12 mo of LangChain+OpenAI, no pre-LLM ML prod.
  - No production code in last 18 mo (tech-lead/architecture-only).
  - Career only at consulting firms: TCS, Infosys, Wipro, Accenture,
    Cognizant, Capgemini. (If currently at one *but* has prior product
    company experience — OK, don't gate.)
  - Primary expertise is CV/speech/robotics with no NLP/IR exposure.
  - 5+ yrs on closed-source with zero external validation (no papers,
    talks, OSS).
  - Title-chaser: Senior→Staff→Principal each ~1.5 yr jumps.

## 6. Reasoning design (survives Stage 4 manual review)

**No LLM at rank time** (banned + too slow). `reasoning.py` builds each
string from fields actually present in the record — only.
- Three tone tiers by rank: 1–25 assertive, 26–75 trade-off-aware
  (notes one honest gap), 76–100 hesitant ("Adjacent — …").
- Template variant chosen deterministically by `hash(candidate_id)` so the
  10 sampled rows vary without being random.
- Every interpolated value must come from a real field. Never invent a
  skill, employer, or year.

## 7. Repo layout (target — some files not yet written)

```
indiaruns-ranker/
  AGENTS.md                  # this file
  README.md                  # one-command reproduction for organizers
  requirements.txt
  pyproject.toml
  submission_metadata.yaml
  .gitignore
  src/
    config.py                # skill taxonomy, weights, thresholds, disqualifiers
    io.py                    # JSONL streaming reader
    features.py              # per-candidate feature extraction
    semantic.py              # embedding build + cosine (all-MiniLM-L6-v2)
    sparse.py                # TfidfVectorizer index
    honeypot.py              # gate detector
    scoring.py               # final score formula
    reasoning.py             # templated, no-hallucination reasonings
    precompute.py            # Phase A entry point
    rank.py                  # Phase B entry point (the reproduce command)
  artifacts/                 # committed: embeddings.npy, tfidf.npz,
                             #   features.parquet, jd_emb.npy, jd_text.txt
  data/
    sample/sample_candidates.jsonl   # ~100 cand sample (committed, for sandbox)
    job_description.md
  sandbox/
    app.py                   # Gradio HF Space (Section 10.5 sandbox)
    requirements.txt
    Dockerfile               # Option 2: docker pull + run (spec §10.5 alt)
  docs/
    deck.md                  # → exported to deck.pdf (Mohana builds PPT)
  tests/
    test_validator.py
    test_scoring.py
```

## 8. Build / run / test commands

(Bit of state may not exist yet — verify before running.)

```powershell
# One-time: install deps
pip install -r requirements.txt

# Phase A — precompute artifacts (may exceed 5 min, may use network first
# run to fetch all-MiniLM-L6-v2 weights; weights then committed under artifacts/models/)
python src/precompute.py --candidates ../data/India_runs_data_and_ai_challenge/candidates.jsonl --artifacts ./artifacts

# Phase B — the reproduce command (spec §10.3). <5 min, CPU, no net.
python src/rank.py --candidates ../data/India_runs_data_and_ai_challenge/candidates.jsonl --artifacts ./artifacts --out ./codexmohan_6487.csv

# Validate the CSV
python ../data/India_runs_data_and_ai_challenge/validate_submission.py ./codexmohan_6487.csv

# Tests
python -m pytest tests/ -q

# Sandbox (local Gradio)
python sandbox/app.py
```

## 9. Current state (updated as work progresses)

- [x] Repo scaffold created under `indiaruns-ranker/`.
- [x] Raw dataset extracted and read.
- [x] AGENTS.md (this file).
- [ ] requirements.txt, pyproject.toml
- [ ] src/config.py (skill taxonomy + weights + thresholds)
- [ ] src/io.py (streaming JSONL reader)
- [ ] src/features.py (feature extraction)
- [ ] src/semantic.py (embeddings)
- [ ] src/sparse.py (TF-IDF)
- [ ] src/honeypot.py (gate)
- [ ] src/scoring.py (formula)
- [ ] src/reasoning.py (templated reasonings)
- [ ] src/precompute.py (Phase A entry)
- [ ] src/rank.py (Phase B entry)
- [ ] Run precompute → artifacts
- [ ] Run rank → codexmohan_6487.csv
- [ ] Pass validate_submission.py
- [ ] sandbox/app.py + Dockerfile
- [ ] README.md + submission_metadata.yaml
- [ ] docs/deck.md
- [ ] Commit + push to github.com/codex-mohan/TalentLens

## 10. Conventions for any future session

- **Never** call hosted LLM APIs from `rank.py` or anything it imports.
  `precompute.py` may use network *only* to fetch the local
  sentence-transformer model weights the first time.
- Commit the MiniLM weights under `artifacts/models/` so Stage 3
  reproduction works with no network.
- Keep `config.py` as the single source of truth for all weights and
  thresholds — don't sprinkle magic numbers across modules.
- No comments unless asked (per opencode defaults). Code must be readable
  via tests + the docstrings in `config.py`.
- Keep the closure of `rank.py` imports minimal and pure-Python-CPU.
- If the dataset path changes, update §2 and §8 of this file, not scattered
  scripts.
- After any code change to ranking logic, re-run rank.py and
  validate_submission.py and tick off §9 checkboxes.
