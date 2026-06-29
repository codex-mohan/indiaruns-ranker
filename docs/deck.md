# INDIA RUNS — Candidate Ranking System
## Team: The Monolith | Mohana Krishna

---

## Slide 1: Title
**INDIA RUNS — Intelligent Candidate Discovery & Ranking**
Redrob AI Hiring Challenge

Team: The Monolith | Mohana Krishna | codexmohan@gmail.com

---

## Slide 2: The Problem
Recruiters miss good candidates because keyword filters can't see what matters.
The challenge: rank 100,000 candidates against one hand-written JD — and get the top 100 right.

**The JD is a trap by design.** The sample submission ranks HR Managers at #1 because the dataset seeds them with "8 AI skills" like Pinecone and RAG. But the JD explicitly says: *"The right answer is not find candidates whose skills section contains the most AI keywords. A Marketing Manager with 8 AI skills is not a fit, no matter how perfect their skill list looks."*

The dataset also contains ~80 honeypots (impossible profiles), keyword stuffers (HR Managers listing NLP skills never used), and behavioral twins (identical profiles with opposite availability signals).

---

## Slide 3: Architecture — Three Layers
**Layer 1 — Gate** (filters noise → 35,039 of 100K pass)
- Honeypot detection: impossible profiles → score = 0
- Hard disqualifiers: consulting-only careers (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/Genpact etc.), research-only, title-chasers, no production code in 18+ months, closed-source careers without external validation
- Current consulting role penalty: candidates at consulting firms with prior product-company experience are penalized but not disqualified (per JD)

**Layer 2 — Bi-encoder ranking** (7-component hybrid score)
- Semantic: `all-MiniLM-L6-v2` cosine similarity (catches "built a recsys" ≈ "retrieval")
- Lexical: TF-IDF cosine (catches exact tech: Pinecone, Qdrant, NDCG)
- Skill evidence: trust-weighted — 0 endorsements + 0 months = ~0 contribution
- Career fit: title archetype + product-company boost + education tier + certifications
- YOE band: Gaussian peaked at 6.5yr, softened wings for out-of-band candidates
- Location: Pune/Noida/Tier-1 India preferred
- Behavioral multiplier: observed signals only — recency, reply rate, interview completion, active applications, recruiter interest, notice period ∈ [0.5, 1.15]. No self-reported open-to-work flag.

**Layer 3 — Cross-encoder re-ranking** (corrective layer for top 1000)
- `ms-marco-MiniLM-L-6-v2` reads (JD, candidate career text) **jointly**
- Catches false positives the bi-encoder misses: computer vision work scored as relevant because both share "model", "fine-tuned", "production" keywords
- Blended 45% CE + 55% bi-encoder → prevents overcorrection

---

## Slide 4: Scoring Formula
```
gate = 0 if honeypot OR disqualifier, else 1

raw = 0.30×semantic + 0.20×lexical + 0.25×skill_ev
    + 0.15×career_fit + 0.05×yoe_band + 0.05×location

semantic = 0.45×CE_normalized + 0.55×bi_encoder_cosine

final = gate × raw × behavioral_multiplier    # behavioral ∈ [0.5, 1.15]
```

Skill trust factor per skill:
```
trust = proficiency_mult × min(1, months/12) × min(1, endorsements/10)
if endorsements == 0 and months == 0: trust ×= 0.05
```

Career fit composition:
```
base = title_archetype_weight
     + eval_experience_bonus (+0.15)
     + scale_experience_bonus (+0.05)
     + education_tier_bonus (+0.03 tier_1, +0.02 tier_2)
     + certification_bonus (+0.02 for 1+, +0.03 for 3+)
     × disqualifier_penalties
```

---

## Slide 5: How We Handle the Traps

| Trap | Problem | Our Solution |
|---|---|---|
| **Keyword stuffers** | HR Manager lists "RAG", "Pinecone" as skills | Trust factor kills them: 0 endorsements + 0 months × 0.05 = near-zero skill evidence |
| **Honeypots** | Impossible profiles (8yr at company founded 3yr ago; 10 expert skills with 0 months each) | Gate: YOE > 45, expert skills without duration → score = 0 |
| **CV/speech/robotics** | Candidate builds image moderation but shares ML keywords | Cross-encoder reads "computer vision + ResNet" against "retrieval + embeddings" and scores low |
| **Consulting firms** | Career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/Genpact | Gate: consulting-only → score = 0. Current consulting role with prior product experience: moderate career-fit penalty (0.55x), not disqualified per JD |
| **Behavioral ghosts** | Perfect profile, inactive 200 days, 6% response rate | Behavioral multiplier: -0.12 for stale, -0.05 for low response → effective score drops 15-20% |
| **Title-chasers** | Senior → Staff → Principal every 1.5yr, avg tenure < 18mo | Gate: title_chaser flag → score = 0 |

---

## Slide 6: Results

| Metric | Value |
|---|---|
| Candidates | 100,000 |
| Gate pass | 35,039 (35%) |
| Gate fail | 64,961 (65%) |
| Honeypots in top 100 | 0 (DQ threshold: >10%) |
| Cross-encoder re-ranked | Top 1,000 |
| Ranking time | 120.9 seconds (< 5 min limit) |
| Memory | < 16 GB |
| Network during ranking | None |

**Top 10 candidates:**
1. Senior AI Engineer @ Apple — 5.9yr, retrieval/embeddings, 80% reply, Trivandrum
2. Senior ML Engineer @ PhonePe — 6.2yr, RAG/PEFT/QLoRA, 75% reply, Coimbatore
3. Senior ML Engineer @ Zomato — 7.2yr, retrieval/embeddings, 61% reply, Noida
4. Staff ML Engineer @ Paytm — 7.0yr, BM25/IR expert, 95% reply, Kochi
5. Staff ML Engineer @ Yellow.ai — 8.6yr, RAG/Pinecone, 83% reply, Jaipur
6. Lead AI Engineer @ Razorpay — 6.7yr, IR/LTR expert, 73% reply, Jaipur
7. Senior AI Engineer @ Netflix — 7.8yr, BM25/LTR, 76% reply, Vizag
8. Senior AI Engineer @ Meta — 7.9yr, BM25/Search, 79% reply, Noida
9. Lead AI Engineer @ Sarvam AI — 6.4yr, pgvector/Qdrant, 86% reply, Delhi
10. Senior ML Engineer @ Genpact AI — 6.1yr, IR/LLMs, 88% reply, Pune (consulting penalty)
---

## Slide 7: Reasoning Design
No LLM at rank time (banned + too slow for 100K candidates).

**Built from real fields only** — no hallucination possible. Every name, number, and fact comes from the candidate's actual profile.

**Three tone tiers:**
- Ranks 1–25: Assertive, cites specific career signals ("uses FAISS", "built ranking", "uses Pinecone")
- Ranks 26–75: Trade-off-aware, surfaces one honest gap alongside strengths
- Ranks 76–100: Hesitant, leads with the concern

**Variation:** 5-6 template variants per tier, selected deterministically by candidate_id hash. A reviewer sampling 10 random rows sees distinct sentence structures, not cloned templates.

**Career-specific facts:** The system extracts notable signals from career descriptions (FAISS mentions, BM25 usage, RAG implementation, A/B testing experience, NDCG/MRR evaluation) and includes them in the reasoning.

---

## Slide 8: Reproducibility

**Single reproduce command:**
```bash
python -m src.rank --candidates candidates.jsonl --artifacts ./artifacts --out submission.csv
```

**Download models (run once, requires network):**
```bash
python scripts/download_models.py --artifacts ./artifacts
```

**Pre-computation (run once):**
```bash
python -m src.precompute --candidates candidates.jsonl --artifacts ./artifacts
```
Encodes 100K candidates, builds TF-IDF index, saves model caches for offline ranking. May exceed 5 min (allowed for pre-compute).

**Reproducibility guarantees:**
- All weights and thresholds in a single `config.py` — no magic numbers
- Artifacts directory is gitignored; all generated by `precompute.py` from source
- Models sourced from HuggingFace (cached after first download)
- Ranking step (`rank.py`) runs with no network, no GPU, < 16 GB RAM
- Validated with official `validate_submission.py`

**Sandbox:** Gradio HF Space (upload → rank → download CSV) + Docker option

---

## Slide 9: What We'd Improve
- **L12 cross-encoder**: larger model for better reading comprehension of career descriptions
- **Learning-to-rank**: replace fixed weights with an XGBoost ranker trained on behavioral signal feedback loops
- **Richer career parsing**: extract specific project signals (scale: "50M+ queries", eval rigor: "NDCG", domain: "matching/recommendations")
- **Approximate nearest neighbors**: FAISS IVF for sub-second candidate retrieval when scaling beyond 200K

---

## Slide 10: Submission

| Artifact | Status |
|---|---|
| GitHub repo | codex-mohan/indiaruns-ranker (private) |
| Ranked CSV | codexmohan_6487.csv (100 candidates, validated) |
| Runtime | 120.9 seconds (spec: < 5 min) |
| Honeypots in top-100 | 0 (spec: < 10%) |
| Deck | This document (→ PDF) |
| Sandbox | Gradio HF Space + Docker |
| Team | The Monolith — Mohana Krishna, Vellore, Tamil Nadu |
