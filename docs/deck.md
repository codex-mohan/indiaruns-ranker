# INDIA RUNS — Candidate Ranking System
## Team: The Monolith | Mohana Krishna

---

## Slide 1: Title
**INDIA RUNS — Intelligent Candidate Discovery & Ranking**
Team: The Monolith | Mohana Krishna | codexmohan@gmail.com

---

## Slide 2: The Problem
Recruiters miss good candidates because keyword filters can't see what matters.
The challenge: rank 100,000 candidates against one hand-written JD — and get the top 100 right.

**The JD is a trap on purpose.** The sample submission ranks HR Managers at #1 because they have "8 AI skills." That's exactly what the JD warns against.

---

## Slide 3: What We Built
A **rule-grounded hybrid ranker** that mirrors how a great recruiter thinks:
- **Gate**: catches honeypots + hard disqualifiers (consulting-only, research-only, title-chasers)
- **Semantic**: sentence-transformer embeddings (all-MiniLM-L6-v2) capture "built a recommendation system" matching "ranking/retrieval" even without shared words
- **Lexical**: TF-IDF catches exact tech (Pinecone, Qdrant, NDCG)
- **Skill-evidence**: trust-weighted — a skill with 0 endorsements + 0 months contributes ~0 (defeats keyword-stuffers)
- **Career-fit**: title archetype + product vs services company + eval/scale experience
- **Behavioral multiplier**: open-to-work, response rate, recency, interview completion

---

## Slide 4: Architecture
Two-phase design:
1. **Precompute** (offline): encode 100K candidates → embeddings.npy + TF-IDF + features
2. **Rank** (< 5 min, CPU, no network): load artifacts, score, output top-100 CSV

Single reproduce command:
```
python -m src.rank --candidates candidates.jsonl --artifacts ./artifacts --out submission.csv
```

---

## Slide 5: The Trap — How We Handle It
The dataset contains:
- **Keyword stuffers**: HR Managers listing "RAG" and "Pinecone" but never using them → trust-weighted skill evidence kills them
- **Honeypots**: ~80 candidates with impossible profiles → honeypot gate filters them (0 in top-100)
- **Consulting-only careers**: TCS/Infosys/Wipro → hard disqualifier
- **Title-chasers**: Senior→Staff→Principal every 1.5 yr → flagged and gated

---

## Slide 6: Results
- **100,000 candidates** → gate passes 35,052 → top 100 selected
- **0 honeypots** in top-100 (well under 10% DQ threshold)
- **Top candidates**: AI/ML engineers at product companies (Apple, Paytm, Meta, Razorpay, Microsoft, Amazon)
- **Ranking time**: 26 seconds (well under 5-min limit)
- **Precompute time**: ~6.5 min (model download + encoding 100K on GPU)

---

## Slide 7: Reasoning Design
No LLM at rank time (banned + too slow). Three tone tiers:
- **Top 1–25**: assertive, specific facts (title, YOE, skills, company, signals)
- **Mid 26–75**: trade-off-aware, notes one honest gap
- **Tail 76–100**: hesitant, explicit concerns

Every value comes from real candidate fields. No hallucination. Deterministic template variants for variation.

---

## Slide 8: Reproducibility
- **GitHub repo**: clean, complete, working code
- **Single command**: `python -m src.rank --candidates ... --out ...`
- **Artifacts committed**: model weights, embeddings, TF-IDF, features
- **Sandbox**: Gradio HF Space + Docker option
- **Compute**: 26s ranking, < 16 GB RAM, CPU-only, no network

---

## Slide 9: What We'd Improve
- Fine-tune a cross-encoder (ms-marco-MiniLM) for re-ranking the top 1K
- Add online A/B testing infrastructure
- Build recruiter-feedback loops for continuous improvement
- Scale to 200K+ candidates with approximate nearest neighbors (FAISS IVF)

---

## Slide 10: Submission
- **CSV**: `codexmohan_6487.csv` (100 candidates, validated)
- **Repo**: github.com/codex-mohan/indiaruns-ranker
- **Sandbox**: HuggingFace Spaces
- **Deck**: this document (→ PDF)
