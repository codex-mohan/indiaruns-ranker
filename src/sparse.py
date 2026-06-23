"""TF-IDF sparse index for lexical matching.

Precompute phase: fit TF-IDF over all candidate text blobs, store matrix.
Ranking phase: transform JD text → sparse vector, compute cosine scores.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf(texts: list[str], max_features: int = 8000):
    """Fit TF-IDF on candidate texts, return (vectorizer, sparse_matrix)."""
    vec = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    matrix = vec.fit_transform(texts)
    return vec, matrix


def lexical_scores(vec: TfidfVectorizer, matrix, jd_text: str) -> np.ndarray:
    """Compute TF-IDF cosine between JD text and all candidate vectors.

    Returns (N,) array of scores in [0, 1].
    """
    jd_vec = vec.transform([jd_text])
    scores = cosine_similarity(jd_vec, matrix).squeeze()
    return np.asarray(scores, dtype=np.float32)
