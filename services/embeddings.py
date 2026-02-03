from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
from typing import Union

from config import get_settings

settings = get_settings()


@lru_cache()
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model (cached for reuse)."""
    return SentenceTransformer(settings.embedding_model)


def get_embedding(text: Union[str, list[str]]) -> np.ndarray:
    """Generate embeddings for text or list of texts."""
    model = get_embedding_model()

    if isinstance(text, str):
        text = [text]

    embeddings = model.encode(text, normalize_embeddings=True)
    return embeddings


def get_embedding_dimension() -> int:
    """Return the dimension of embeddings."""
    return settings.embedding_dim
