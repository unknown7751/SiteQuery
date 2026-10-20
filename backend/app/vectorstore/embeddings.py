from pathlib import Path

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings


def build_embeddings(model_name: str, cache_dir: Path) -> Embeddings:
    """Local ONNX embeddings: no API key, model is cached on the mounted volume."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return FastEmbedEmbeddings(model_name=model_name, cache_dir=str(cache_dir))
