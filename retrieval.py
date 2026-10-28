"""Bi-encoder dense retrieval: embed the query, then dot-product search.

The question and every document are embedded *independently* into vectors.
Because embeddings are L2-normalized, the dot product equals cosine similarity,
so search is a single matrix-vector product over the precomputed corpus.
"""

import os

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# BGE v1.5 recommends this prefix for short queries that retrieve passages.
# Documents are embedded without any prefix.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEVICE = os.environ.get("LAB_DEVICE", "cpu")


def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)


def embed_documents(model, texts, batch_size=32, show_progress_bar=True):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    ).astype(np.float32)


def embed_query(model, question):
    return model.encode(
        QUERY_INSTRUCTION + question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def search(query_embedding, doc_embeddings, k=20):
    """Return (indices, scores) of the k most similar documents, best first."""
    scores = doc_embeddings @ query_embedding
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]


def retrieve(model, question, doc_embeddings, k=20):
    """Pipeline A, step 1: embed the question and return the top-k candidates."""
    return search(embed_query(model, question), doc_embeddings, k)
