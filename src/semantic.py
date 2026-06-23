"""Embedding build + cosine similarity using sentence-transformers.

Precompute phase: encode all candidate text blobs + JD → embeddings.npy
Ranking phase: cosine(JD_emb, cand_embs) → semantic scores.
"""
import numpy as np


def load_model(model_name: str):
    """Load sentence-transformer model (downloads on first call)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def encode_texts(model, texts: list[str], batch_size: int = 256,
                 show_progress: bool = True) -> np.ndarray:
    """Encode a list of texts into a (N, dim) float32 numpy array."""
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def cosine_scores(jd_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a single JD vector and all candidates.

    Both inputs should be L2-normalized (normalize_embeddings=True above).
    Returns (N,) array of scores in [-1, 1].
    """
    # If jd_emb is (D,), treat as (1, D)
    if jd_emb.ndim == 1:
        jd_emb = jd_emb[np.newaxis, :]
    # cosine = dot product when vectors are normalized
    scores = (cand_embs @ jd_emb.T).squeeze(-1)
    return scores
