"""Cross-encoder reranking of a fixed candidate list.

A cross-encoder reads the question and one document *together* in a single
transformer pass and outputs one relevance score. That joint attention is more
accurate than comparing two independent embeddings, but it must run once per
(question, document) pair, so it is only applied to the bi-encoder's top-k.
"""

import os

import numpy as np
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"
MAX_LENGTH = 512  # tokens for question + document; longer abstracts are truncated
DEVICE = os.environ.get("LAB_DEVICE", "cpu")


def load_reranker():
    return CrossEncoder(RERANKER_MODEL, max_length=MAX_LENGTH, device=DEVICE)


def score_pairs(model, question, doc_texts, batch_size=20):
    """One relevance score per document, higher = more relevant.

    sentence-transformers applies a sigmoid to this single-label model, so
    scores fall in (0, 1). The sigmoid is monotonic: ranking is unaffected.
    """
    pairs = [(question, text) for text in doc_texts]
    return np.asarray(
        model.predict(pairs, batch_size=batch_size, show_progress_bar=False),
        dtype=np.float32,
    )


def rerank(model, question, candidate_indices, doc_texts):
    """Pipeline B: reorder the *same* candidates by cross-encoder score.

    `doc_texts[i]` is the text of `candidate_indices[i]`. Returns
    (reordered_indices, scores, original_ranks), where original_ranks are
    0-based positions in the bi-encoder list.
    """
    scores = score_pairs(model, question, doc_texts)
    order = np.argsort(-scores, kind="stable")
    return np.asarray(candidate_indices)[order], scores[order], order
