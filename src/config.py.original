"""JD-derived config — single source of truth for all taxonomy, weights, thresholds.

Everything downstream reads from this file. If the JD changes, only this file changes.
"""

import os

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CANDIDATES_JSONL = os.path.join(DATA_DIR, "candidates.jsonl")

# ── embedding model ────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# ── cross-encoder model (for re-ranking top N candidates) ──────────────────
# Reads (JD, candidate) pairs jointly — understands relevance, not just similarity.
# 80MB, runs on CPU. Downloaded during precompute, loaded locally during ranking.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 1000       # how many candidates to re-rank with cross-encoder

# ── scoring weights (raw component) ────────────────────────────────────────
# Must sum to 1.0 — these are the linear weights for the soft score.
W_SEMANTIC = 0.30
W_LEXICAL = 0.20
W_SKILL_EVIDENCE = 0.25
W_CAREER_FIT = 0.15
W_YOE = 0.05
W_LOCATION = 0.05
# Behavioral multiplier range
BEHAVIORAL_MIN = 0.50
BEHAVIORAL_MAX = 1.15

# ── experience band (peak of Gaussian) ─────────────────────────────────────
YOE_PEAK = 6.5
YOE_SIGMA = 3.0

# ── salary band (INR LPA) — reasonable for senior AI eng in India ─────────
SALARY_MIN_REASONABLE = 15.0   # LPA
SALARY_MAX_REASONABLE = 80.0   # LPA

# ── notice period bands ────────────────────────────────────────────────────
NOTICE_GOOD_DAYS = 30          # ≤ this → best
NOTICE_MEDIUM_DAYS = 60        # ≤ this → neutral

# ── recency bands (days since last_active_date) ────────────────────────────
RECENCY_FRESH_DAYS = 30
RECENCY_RECENT_DAYS = 90
RECENCY_STALE_DAYS = 180

# ── skill taxonomy ─────────────────────────────────────────────────────────
# RETRIEVAL_MUST: must-have retrieval/embedding/ranking skills from the JD.
# Each entry is a set of aliases (all lowercased) that map to one logical skill.
RETRIEVAL_MUST = [
    {"embeddings", "vector search", "vector representations", "semantic search",
     "sentence transformers", "text encoders"},
    {"faiss", "milvus", "pinecone", "qdrant", "weaviate", "pgvector",
     "opensearch", "elasticsearch", "bm25"},
    {"information retrieval", "information retrieval systems",
     "ranking systems", "search & discovery", "search backend", "content matching"},
    {"rag"},
]

# RETRIEVAL_NICE: nice-to-have per the JD.
RETRIEVAL_NICE = [
    {"learning to rank", "recommendation systems"},
    {"haystack", "llamaindex"},
]

# LLM_FINETUNE: nice-to-have LLM fine-tuning skills.
LLM_FINETUNE = [
    {"fine-tuning llms", "lora", "qlora", "peft",
     "hugging face transformers", "llms"},
    {"langchain", "prompt engineering"},
    {"model adaptation"},
]

# ML_SUPPORT: supporting ML skills (boost but not primary signal).
ML_SUPPORT = [
    {"python", "scikit-learn", "pytorch", "tensorflow"},
    {"machine learning", "deep learning", "nlp", "natural language processing",
     "statistical modeling", "feature engineering"},
    {"mlops", "mlflow", "weights & biases", "kubeflow", "data science"},
]

# EVAL_KEYWORDS: terms to scan career descriptions for ranking-eval experience.
EVAL_KEYWORDS = [
    "ndcg", "mrr", "map", "mean average precision", "a/b test", "ab test",
    "evaluation framework", "ranking evaluation", "offline", "online",
    "correlation", "precision", "recall", "retrieval quality",
    "ranking quality", "eval framework",
]

# SCALE_KEYWORDS: distributed systems / large-scale inference experience.
SCALE_KEYWORDS = [
    "distributed", "large-scale", "scale", "inference optimization",
    "serving", "latency", "throughput", "pipeline",
]

# ── hard disqualifiers ────────────────────────────────────────────────────
# Consulting firms per the JD: "TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini"
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hcl", "mindtree", "larsen toubro", "ltimindtree",
    "mpHASIS", "mphasis", "hexaware", "niit technologies", "bsnl",
}

# Titles that indicate pure research / academic background
RESEARCH_TITLES = [
    "researcher", "research scientist", "research intern",
    "phd", "postdoc", "postdoctoral", "professor", "assistant professor",
    "research assistant", "research associate", "research engineer",
]

# Titles that indicate pure tech lead / architecture (no recent code)
NON_CODING_TITLES = [
    "tech lead", "techlead", "tech lead architect", "architect",
    "principal architect", "chief architect", "vp engineering",
    "director of engineering", "head of engineering",
    "cto", "chief technology officer",
]

# ── location tiers ─────────────────────────────────────────────────────────
LOCATION_PREFERRED = {
    "pune", "noida", "delhi ncr", "delhi", "gurgaon", "gurugram", "noida",
    "mumbai", "bangalore", "bengaluru", "hyderabad", "chennai",
    "kolkata", "ahmedabad", "jaipur", "lucknow", "chandigarh",
    "coimbatore", "indore", "bhopal", "nagpur",
}
LOCATION_INDIAN_CITIES = LOCATION_PREFERRED | {
    "thane", "navi mumbai", "faridabad", "ghaziabad", "noida sector",
    "greater noida", "visakhapatnam", "patna", "raipur", "ranchi",
    "surat", "vadodara", "rajkot", "mysore", "mangalore",
    "thiruvananthapuram", "kochi", "cochin", "guwahati", "imphal",
    "shillong", "agartala", "shimla", "dehradun",
}

# ── honeypot heuristics ────────────────────────────────────────────────────
# A candidate is flagged honeypot if ANY of these are true.
HONEYPOT_MAX_EXPERT_SKILLS_WITH_ZERO_DURATION = 5
HONEYPOT_MIN_EXPERT_SKILLS_FOR_FLAG = 3
HONEYPOT_MAX_SKILLS_WITH_ZERO_ENDORSEMENTS_AND_ZERO_DURATION = 8
HONEYPOT_YOE_MINIMUM_REASONABLE = 0
HONEYPOT_YOE_MAXIMUM = 45
