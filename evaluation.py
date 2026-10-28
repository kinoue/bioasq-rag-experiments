"""Metrics (Recall@k, MRR@k, nDCG@k) and the benchmark that compares pipelines.

Run from the command line to benchmark all questions and save the results:

    python evaluation.py
"""

import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from data import (
    BENCHMARK_PATH,
    document_text,
    load_corpus,
    load_embeddings,
    load_questions,
    save_json,
)
from reranker import RERANKER_MODEL, load_reranker, rerank
from retrieval import DEVICE, EMBEDDING_MODEL, load_embedding_model, retrieve

CANDIDATES = 20
TOP_K = 5


# ---------------------------------------------------------------- metrics ---
# All metrics take a ranked list of document IDs and a set of relevant IDs.
# Relevance is binary: a document is relevant iff BioASQ lists it.


def recall_at_k(ranked_ids, relevant, k):
    """Fraction of the known relevant documents that appear in the top k."""
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(ranked_ids, relevant, k):
    """1 / rank of the first relevant document in the top k (0 if none)."""
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevant, k):
    """DCG of the ranking divided by the DCG of a perfect ranking.

    DCG = sum over relevant hits of 1 / log2(rank + 1). The ideal ranking puts
    min(|relevant|, k) relevant documents at the top.
    """
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def all_metrics(ranked_ids, relevant, k=TOP_K):
    return {
        f"recall@{k}": recall_at_k(ranked_ids, relevant, k),
        f"mrr@{k}": reciprocal_rank_at_k(ranked_ids, relevant, k),
        f"ndcg@{k}": ndcg_at_k(ranked_ids, relevant, k),
    }


# ------------------------------------------------------------- pipelines ---


def compare_pipelines(question, embedder, reranker, corpus, doc_embeddings):
    """Run both pipelines on one question, timing each stage.

    Both pipelines share the same 20 candidates: Pipeline A keeps the
    bi-encoder order, Pipeline B reorders them by cross-encoder score.
    """
    t0 = time.perf_counter()
    cand_idx, cand_scores = retrieve(embedder, question, doc_embeddings, CANDIDATES)
    t1 = time.perf_counter()
    texts = [document_text(corpus[i]) for i in cand_idx]
    reranked_idx, rerank_scores, original_ranks = rerank(
        reranker, question, cand_idx, texts
    )
    t2 = time.perf_counter()

    pmid = lambda i: corpus[i]["pmid"]  # noqa: E731
    baseline = [
        {"pmid": pmid(i), "score": float(s), "rank": r}
        for r, (i, s) in enumerate(zip(cand_idx, cand_scores), start=1)
    ]
    reranked = [
        {"pmid": pmid(i), "score": float(s), "rank": r, "original_rank": int(o) + 1}
        for r, (i, s, o) in enumerate(
            zip(reranked_idx, rerank_scores, original_ranks), start=1
        )
    ]
    return {
        "baseline": baseline,  # all 20, bi-encoder order
        "reranked": reranked,  # same 20, cross-encoder order
        "latency_ms": {
            "retrieval": (t1 - t0) * 1000,
            "reranking": (t2 - t1) * 1000,
            "total": (t2 - t0) * 1000,
        },
    }


# ------------------------------------------------------------- benchmark ---


def cpu_name():
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if out.stdout.strip():
            return out.stdout.strip()
    return platform.processor() or platform.machine()


def hardware_info():
    import torch

    return {
        "device": DEVICE,
        "cpu": cpu_name(),
        "cpu_cores": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": float(np.percentile(values, 90)),
    }


def run_benchmark(embedder, reranker, corpus, doc_embeddings, questions, progress=None):
    """Evaluate both pipelines on every question.

    Timing: one warm-up query is run first and discarded; model loading and
    corpus embedding are never timed. Each question is then processed one at a
    time (query batch size 1, 20 reranker pairs in one batch).
    """
    compare_pipelines("warm-up query", embedder, reranker, corpus, doc_embeddings)

    rows = []
    for n, q in enumerate(questions):
        relevant = set(q["relevant_pmids"])
        result = compare_pipelines(q["body"], embedder, reranker, corpus, doc_embeddings)
        base_ids = [d["pmid"] for d in result["baseline"]]
        rr_ids = [d["pmid"] for d in result["reranked"]]
        rows.append(
            {
                "id": q["id"],
                "question": q["body"],
                "type": q["type"],
                "num_relevant": len(relevant),
                "baseline": all_metrics(base_ids, relevant),
                "reranked": all_metrics(rr_ids, relevant),
                # Sanity check: reranking must not change the candidate pool.
                "recall@20_baseline": recall_at_k(base_ids, relevant, CANDIDATES),
                "recall@20_reranked": recall_at_k(rr_ids, relevant, CANDIDATES),
                "same_candidates": set(base_ids) == set(rr_ids),
                "latency_ms": result["latency_ms"],
            }
        )
        if progress:
            progress(n + 1, len(questions))

    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "candidates": CANDIDATES,
        "top_k": TOP_K,
        "num_questions": len(rows),
        "hardware": hardware_info(),
        "aggregate": aggregate(rows),
        "per_question": rows,
    }


def aggregate(rows):
    metrics = list(rows[0]["baseline"].keys())
    agg = {
        pipeline: {m: statistics.fmean(r[pipeline][m] for r in rows) for m in metrics}
        for pipeline in ("baseline", "reranked")
    }
    for pipeline in ("baseline", "reranked"):
        agg[pipeline]["recall@20"] = statistics.fmean(
            r[f"recall@20_{pipeline}"] for r in rows
        )
    agg["all_candidates_identical"] = all(r["same_candidates"] for r in rows)
    agg["latency_ms"] = {
        stage: summarize([r["latency_ms"][stage] for r in rows])
        for stage in ("retrieval", "reranking", "total")
    }
    deltas = [r["reranked"]["ndcg@5"] - r["baseline"]["ndcg@5"] for r in rows]
    agg["ndcg@5_outcomes"] = {
        "improved": sum(d > 1e-9 for d in deltas),
        "unchanged": sum(abs(d) <= 1e-9 for d in deltas),
        "degraded": sum(d < -1e-9 for d in deltas),
    }
    return agg


def main():
    questions = load_questions()
    corpus = load_corpus()
    doc_embeddings = load_embeddings()
    embedder, reranker = load_embedding_model(), load_reranker()

    def progress(done, total):
        print(f"\r{done}/{total} questions", end="", flush=True)

    results = run_benchmark(embedder, reranker, corpus, doc_embeddings, questions, progress)
    print()
    save_json(BENCHMARK_PATH, results)

    agg = results["aggregate"]
    print(f"{'metric':<10}{'baseline':>10}{'reranked':>10}")
    for m in ("recall@5", "mrr@5", "ndcg@5", "recall@20"):
        print(f"{m:<10}{agg['baseline'][m]:>10.3f}{agg['reranked'][m]:>10.3f}")
    print("candidates identical:", agg["all_candidates_identical"])
    print("nDCG@5 outcomes:", agg["ndcg@5_outcomes"])
    for stage, s in agg["latency_ms"].items():
        print(f"{stage:<10} median {s['median']:.1f} ms  mean {s['mean']:.1f} ms")
    print(f"Saved {BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
