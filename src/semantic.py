"""Embedding build + cosine similarity using sentence-transformers.

Precompute phase: encode all candidate text blobs + JD → embeddings.npy
Ranking phase: cosine(JD_emb, cand_embs) → semantic scores.
"""
import os
from pathlib import Path
import numpy as np


def _cpu_threads() -> int:
    configured = os.getenv("TALENTLENS_CPU_THREADS")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError as exc:
            raise ValueError("TALENTLENS_CPU_THREADS must be an integer") from exc
    return max(1, os.cpu_count() or 1)


def _configure_torch_cpu() -> None:
    import torch

    threads = _cpu_threads()
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("high")
    try:
        torch.set_num_interop_threads(max(1, min(4, threads)))
    except RuntimeError:
        # PyTorch allows setting inter-op threads only before parallel work starts.
        pass


def _onnx_file_name(quantization: str) -> str:
    return f"model_qint8_{quantization}.onnx"


def _openvino_file_name() -> str:
    return "openvino_model_qint8_quantized.xml"


def load_model(
    model_name: str,
    backend: str = "torch",
    model_dir: str | None = None,
    quantization: str = "avx2",
):
    """Load SentenceTransformer with the requested inference backend."""
    if backend == "torch":
        _configure_torch_cpu()
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    if backend not in {"onnx-int8", "openvino", "openvino-int8"}:
        raise ValueError("--embed-backend must be 'torch', 'onnx-int8', 'openvino', or 'openvino-int8'")
    from sentence_transformers import SentenceTransformer
    if model_dir is None:
        raise ValueError("model_dir is required for optimized embedding backends")

    target_dir = Path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if backend == "onnx-int8":
        if quantization not in {"arm64", "avx2", "avx512", "avx512_vnni"}:
            raise ValueError("--onnx-quantization must be arm64, avx2, avx512, or avx512_vnni")
        from sentence_transformers import export_dynamic_quantized_onnx_model

        quantized_name = _onnx_file_name(quantization)
        quantized_path = target_dir / "onnx" / quantized_name
        model_kwargs = {
            "provider": "CPUExecutionProvider",
            "file_name": quantized_name,
        }
        if quantized_path.exists():
            return SentenceTransformer(str(target_dir), backend="onnx", model_kwargs=model_kwargs)

        export_model = SentenceTransformer(
            model_name,
            backend="onnx",
            model_kwargs={"provider": "CPUExecutionProvider", "export": True},
        )
        export_model.save(str(target_dir))
        export_dynamic_quantized_onnx_model(
            export_model,
            quantization,
            str(target_dir),
            file_suffix=f"qint8_{quantization}",
        )
        return SentenceTransformer(str(target_dir), backend="onnx", model_kwargs=model_kwargs)

    if backend == "openvino":
        openvino_path = target_dir / "openvino" / "openvino_model.xml"
        if openvino_path.exists():
            return SentenceTransformer(str(target_dir), backend="openvino")
        export_model = SentenceTransformer(
            model_name,
            backend="openvino",
            model_kwargs={"export": True},
        )
        export_model.save(str(target_dir))
        return SentenceTransformer(str(target_dir), backend="openvino")

    from sentence_transformers import export_static_quantized_openvino_model

    quantized_name = _openvino_file_name()
    quantized_path = target_dir / "openvino" / quantized_name
    model_kwargs = {"file_name": quantized_name}
    if quantized_path.exists():
        return SentenceTransformer(str(target_dir), backend="openvino", model_kwargs=model_kwargs)

    export_model = SentenceTransformer(
        model_name,
        backend="openvino",
        model_kwargs={"export": True},
    )
    export_model.save(str(target_dir))
    export_static_quantized_openvino_model(
        export_model,
        quantization_config=None,
        model_name_or_path=str(target_dir),
    )
    return SentenceTransformer(str(target_dir), backend="openvino", model_kwargs=model_kwargs)


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
