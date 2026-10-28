"""Loading the prepared questions, corpus, embeddings and relevance judgments.

All files are produced by `prepare_data.py` and live in `data/`.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RAW_BIOASQ_PATH = DATA_DIR / "raw" / "training11b.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "doc_embeddings.npy"
METADATA_PATH = DATA_DIR / "metadata.json"
BENCHMARK_PATH = DATA_DIR / "benchmark_results.json"


def load_questions():
    """Evaluation questions: id, body, type, relevant_pmids, snippets."""
    return json.loads(QUESTIONS_PATH.read_text())


def load_corpus():
    """Corpus documents in embedding-row order: pmid, title, abstract."""
    with CORPUS_PATH.open() as f:
        return [json.loads(line) for line in f]


def load_embeddings():
    """Normalized document embeddings, shape (num_docs, dim). Row i = corpus[i]."""
    return np.load(EMBEDDINGS_PATH)


def load_metadata():
    return json.loads(METADATA_PATH.read_text())


def relevance_judgments(questions):
    """Map question id -> set of relevant PubMed IDs (official BioASQ labels)."""
    return {q["id"]: set(q["relevant_pmids"]) for q in questions}


def document_text(doc):
    """Text representation used by both models: title followed by abstract."""
    return f"{doc['title']}\n{doc['abstract']}".strip()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
