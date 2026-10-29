"""Bi-encoder × reranker grid: does a better first stage change the picture?

Each bi-encoder retrieves its own top-20 candidates, so this compares two things:
1. each bi-encoder alone (its top 5, and Recall@20 = how good a pool it hands on);
2. every bi-encoder followed by every reranker from analysis.py.

    LAB_DEVICE=mps python bi_encoder_grid.py

Writes data/bi_encoder_grid.json. Reference pipeline for paired statistics:
bge-small-en-v1.5 alone, i.e. the app's baseline.
"""

import statistics
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from analysis import RERANKERS, oracle_ranking, paired_stats
from data import DATA_DIR, document_text, load_corpus, load_questions, save_json
from evaluation import CANDIDATES, TOP_K, all_metrics, hardware_info, ndcg_at_k, recall_at_k
from reranker import MAX_LENGTH, score_pairs
from retrieval import DEVICE, QUERY_INSTRUCTION, search

BGE_PREFIX = QUERY_INSTRUCTION
BI_ENCODERS = {
    "bge-small (app)": {"model": "BAAI/bge-small-en-v1.5", "query_prefix": BGE_PREFIX},
    "bge-base": {"model": "BAAI/bge-base-en-v1.5", "query_prefix": BGE_PREFIX},
    "MedCPT": {"model": "ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder"},
    "MiniLM-L6 (MS MARCO)": {"model": "sentence-transformers/msmarco-MiniLM-L6-cos-v5"},
    "mxbai-embed-xsmall": {"model": "mixedbread-ai/mxbai-embed-xsmall-v1"},
    "MedEmbed-small": {"model": "abhinand/MedEmbed-small-v0.1"},
    "PubMedBERT (NeuML)": {"model": "NeuML/pubmedbert-base-embeddings"},
}
REFERENCE = ("bge-small (app)", "none")
RESULTS_PATH = DATA_DIR / "bi_encoder_grid.json"
SEED = 0


# ---------------------------------------------------------- bi-encoders ---
# Each loader returns (embed_query, embed_docs, info). Embeddings are compared
# with a plain dot product via retrieval.search.


def load_sentence_transformer(model_name, query_prefix=""):
    """Normalized embeddings (cosine), documents capped at 512 tokens like the rerankers."""
    model = SentenceTransformer(model_name, device=DEVICE)
    model.max_seq_length = min(model.max_seq_length, MAX_LENGTH)

    def embed_query(text):
        return model.encode(query_prefix + text, normalize_embeddings=True)

    def embed_docs(docs):
        return model.encode([document_text(d) for d in docs], batch_size=32,
                            normalize_embeddings=True, show_progress_bar=False)

    info = {"params": sum(p.numel() for p in model.parameters()),
            "dim": model.get_embedding_dimension(),
            "max_seq_length": model.max_seq_length, "query_prefix": query_prefix,
            "similarity": "cosine"}
    return embed_query, embed_docs, info


def load_medcpt():
    """MedCPT: separate query/article encoders, [CLS] vectors, unnormalized dot product."""
    q_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    q_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(DEVICE).eval()
    a_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    a_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(DEVICE).eval()

    @torch.no_grad()
    def embed_query(text):
        enc = q_tok([text], truncation=True, padding=True, max_length=64, return_tensors="pt")
        return q_model(**enc.to(DEVICE)).last_hidden_state[0, 0].float().cpu().numpy()

    @torch.no_grad()
    def embed_docs(docs, batch_size=32):
        out = []
        for i in range(0, len(docs), batch_size):
            pairs = [[d["title"], d["abstract"]] for d in docs[i : i + batch_size]]
            enc = a_tok(pairs, truncation=True, padding=True, max_length=MAX_LENGTH,
                        return_tensors="pt")
            out.append(a_model(**enc.to(DEVICE)).last_hidden_state[:, 0].float().cpu().numpy())
        return np.concatenate(out)

    info = {"params": sum(p.numel() for p in [*q_model.parameters(), *a_model.parameters()]),
            "dim": q_model.config.hidden_size, "max_seq_length": "64 query / 512 article",
            "query_prefix": "", "similarity": "dot product (unnormalized)"}
    return embed_query, embed_docs, info


def load_bi_encoder(label):
    spec = BI_ENCODERS[label]
    if label == "MedCPT":
        return load_medcpt()
    return load_sentence_transformer(spec["model"], spec.get("query_prefix", ""))


def retrieve_with(label, questions, corpus):
    """Top-20 pmids per question for one bi-encoder, with timing."""
    embed_query, embed_docs, info = load_bi_encoder(label)
    t0 = time.perf_counter()
    doc_embeddings = np.asarray(embed_docs(corpus), dtype=np.float32)
    corpus_seconds = time.perf_counter() - t0

    search(np.asarray(embed_query("warm-up query"), dtype=np.float32), doc_embeddings, CANDIDATES)
    candidates, latencies = {}, []
    for q in questions:
        t0 = time.perf_counter()
        idx, _ = search(np.asarray(embed_query(q["body"]), dtype=np.float32),
                        doc_embeddings, CANDIDATES)
        latencies.append((time.perf_counter() - t0) * 1000)
        candidates[q["id"]] = [corpus[i]["pmid"] for i in idx]

    info |= {"model": BI_ENCODERS[label]["model"],
             "corpus_embed_seconds": corpus_seconds,
             "query_latency_ms": {"median": statistics.median(latencies),
                                  "mean": statistics.fmean(latencies)}}
    return candidates, info


# ------------------------------------------------------------ rerankers ---


def score_candidate_union(model_name, questions, doc_by_pmid, all_candidates):
    """Cross-encoder score for every (question, pmid) any bi-encoder retrieved.

    A pair's score does not depend on which bi-encoder found the document, so
    each pair is scored once and shared across the grid.
    """
    model = CrossEncoder(model_name, max_length=MAX_LENGTH, device=DEVICE)
    scores = {}
    for q in questions:
        pmids = sorted({p for cands in all_candidates.values() for p in cands[q["id"]]})
        values = score_pairs(model, q["body"], [document_text(doc_by_pmid[p]) for p in pmids])
        scores[q["id"]] = dict(zip(pmids, values.tolist()))
    return scores


def rerank_by(scores, candidates):
    """Reorder each bi-encoder's own candidates by the shared reranker scores."""
    return {qid: sorted(c, key=lambda p: -scores[qid][p]) for qid, c in candidates.items()}


# ----------------------------------------------------------------- main ---


def main():
    questions = load_questions()
    corpus = load_corpus()
    doc_by_pmid = {d["pmid"]: d for d in corpus}
    qrels = {q["id"]: set(q["relevant_pmids"]) for q in questions}
    qids = list(qrels)

    all_candidates, bi_info = {}, {}
    for label in BI_ENCODERS:
        print(f"retrieving with {label} ...", flush=True)
        all_candidates[label], bi_info[label] = retrieve_with(label, questions, corpus)
        if DEVICE == "mps":
            torch.mps.empty_cache()

    rankings = {(bi, "none"): c for bi, c in all_candidates.items()}
    for rr_label, rr_model in RERANKERS.items():
        print(f"scoring with {rr_model} ...", flush=True)
        scores = score_candidate_union(rr_model, questions, doc_by_pmid, all_candidates)
        for bi, cands in all_candidates.items():
            rankings[(bi, rr_label)] = rerank_by(scores, cands)

    ndcg = {key: np.array([ndcg_at_k(r[q], qrels[q], TOP_K) for q in qids])
            for key, r in rankings.items()}
    for bi, cands in all_candidates.items():
        bi_info[bi]["recall@20"] = float(np.mean([recall_at_k(cands[q], qrels[q], CANDIDATES)
                                                  for q in qids]))
        bi_info[bi]["oracle_ndcg@5"] = float(np.mean(
            [ndcg_at_k(oracle_ranking(cands[q], qrels[q]), qrels[q], TOP_K) for q in qids]))

    rng = np.random.default_rng(SEED)
    grid = []
    for (bi, rr), ranking in rankings.items():
        per_q = [all_metrics(ranking[q], qrels[q]) for q in qids]
        grid.append({
            "bi_encoder": bi,
            "reranker": rr,
            "metrics": {m: float(np.mean([r[m] for r in per_q])) for m in per_q[0]},
            "vs_reference": None if (bi, rr) == REFERENCE
            else paired_stats(ndcg[(bi, rr)] - ndcg[REFERENCE], rng),
            "vs_same_bi_encoder": None if rr == "none"
            else paired_stats(ndcg[(bi, rr)] - ndcg[(bi, "none")], rng),
        })

    results = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware_info(),
        "num_questions": len(qids),
        "candidates": CANDIDATES,
        "top_k": TOP_K,
        "reference": {"bi_encoder": REFERENCE[0], "reranker": REFERENCE[1]},
        "bi_encoders": bi_info,
        "rerankers": RERANKERS,
        "grid": grid,
        "question_ids": qids,
        "per_question_ndcg@5": {f"{bi} | {rr}": v.tolist() for (bi, rr), v in ndcg.items()},
    }
    save_json(RESULTS_PATH, results)
    print_summary(results)
    print(f"Saved {RESULTS_PATH}")


def print_summary(results):
    print(f"\n{'bi-encoder':<22}{'params':>8}{'R@20':>7}{'oracle':>8}{'q ms':>7}")
    for label, info in results["bi_encoders"].items():
        print(f"{label:<22}{info['params'] / 1e6:>7.0f}M{info['recall@20']:>7.3f}"
              f"{info['oracle_ndcg@5']:>8.3f}{info['query_latency_ms']['median']:>7.1f}")
    rerankers = ["none", *results["rerankers"]]
    print(f"\nnDCG@5{'':<16}" + "".join(f"{r[:12]:>14}" for r in rerankers))
    cell = {(g["bi_encoder"], g["reranker"]): g["metrics"]["ndcg@5"] for g in results["grid"]}
    for bi in results["bi_encoders"]:
        print(f"{bi:<22}" + "".join(f"{cell[(bi, r)]:>14.3f}" for r in rerankers))


if __name__ == "__main__":
    main()
