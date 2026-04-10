# ============================================================
# FILE: person_b/embedder.py
# ============================================================
from __future__ import annotations
from functools import lru_cache
from config import EMBED_MODEL


def _best_device() -> str:
    """Pick the fastest available device automatically."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[embedder] using GPU: {name}")
            return "cuda"
        if torch.backends.mps.is_available():
            print("[embedder] using Apple MPS (Metal)")
            return "mps"
    except ImportError:
        pass
    print("[embedder] using CPU (no GPU found)")
    return "cpu"


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    device = _best_device()
    print(f"[embedder] loading {EMBED_MODEL} on {device} …")
    return SentenceTransformer(EMBED_MODEL, device=device)


def embed(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """
    Encode texts → float vectors.
    Batch size is tuned automatically per device if not specified.
    """
    model  = _load_model()
    device = model.device.type

    # Larger batches = faster on GPU, but OOM on CPU with big models
    if batch_size is None:
        if device == "cuda":
            batch_size = 256
        elif device == "mps":
            batch_size = 128
        else:
            batch_size = 32     # conservative for CPU

    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.tolist()