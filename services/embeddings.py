from openai import OpenAI
from functools import lru_cache
import numpy as np
from typing import Union

from config import get_settings

settings = get_settings()

_client: OpenAI = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def get_embedding(text: Union[str, list[str]]) -> np.ndarray:
    """Generate embeddings using OpenAI API."""
    client = get_openai_client()

    if isinstance(text, str):
        text = [text]

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text
    )

    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


def get_embedding_dimension() -> int:
    """Return the dimension of embeddings."""
    return settings.embedding_dim
