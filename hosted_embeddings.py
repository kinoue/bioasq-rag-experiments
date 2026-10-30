"""Experiment 4: a hosted embedding API as the first stage.

Six pipelines: OpenAI's text-embedding-3-small and -large, each alone and
followed by the MedCPT and MiniLM rerankers, which run locally.

    LAB_DEVICE=mps python hosted_embeddings.py

Needs OPENAI_API_KEY, as an environment variable or in KEY_FILE. API embeddings
are cached in data/embedding_cache/, so a rerun isn't billed again.
Writes data/hosted_embeddings.json.

Voyage AI (Anthropic has no embedding model and points to Voyage) is switched
off: its limits without a payment method are too low for BATCH_SIZE. To add it,
pip install voyageai, set VOYAGE_API_KEY and uncomment its PROVIDERS entry.
"""

import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analysis import RERANKERS as ALL_RERANKERS
from analysis import oracle_ranking, paired_stats
from bi_encoder_grid import RESULTS_PATH as GRID_PATH
from bi_encoder_grid import rerank_by, score_candidate_union
from data import DATA_DIR, document_text, load_corpus, load_questions, save_json
from evaluation import CANDIDATES, TOP_K, all_metrics, hardware_info, ndcg_at_k, recall_at_k
from retrieval import search

# Outside the project, because this folder syncs to Dropbox.
KEY_FILE = Path.home() / ".config" / "bioasq-rag-experiments" / ".env"
PROVIDERS = {
    # "Voyage (voyage-4)": {"model": "voyage-4", "key": "VOYAGE_API_KEY"},
    "OpenAI (text-embedding-3-small)": {"model": "text-embedding-3-small", "key": "OPENAI_API_KEY"},
    "OpenAI (text-embedding-3-large)": {"model": "text-embedding-3-large", "key": "OPENAI_API_KEY"},
}
RERANKERS = {k: ALL_RERANKERS[k] for k in ("MedCPT", "MiniLM-L6 (MS MARCO)")}
RESULTS_PATH = DATA_DIR / "hosted_embeddings.json"
CACHE_DIR = DATA_DIR / "embedding_cache"
BATCH_SIZE = 100
MAX_RETRIES = 8
APP_BASELINE = "bge-small (app) | none"
SEED = 0


# ------------------------------------------------------------ API clients ---
# Each loader returns embed(texts, input_type) -> (vectors, tokens_billed).


def load_voyage(model):
    import voyageai

    client = voyageai.Client(max_retries=MAX_RETRIES)

    def embed(texts, input_type):
        result = client.embed(texts, model=model, input_type=input_type)
        return result.embeddings, result.total_tokens

    return embed


def load_openai(model):
    from openai import OpenAI

    client = OpenAI(max_retries=MAX_RETRIES)

    def embed(texts, input_type):  # OpenAI doesn't distinguish queries from documents
        response = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in response.data], response.usage.total_tokens

    return embed


def load_embedder(label):
    loader = load_voyage if label.startswith("Voyage") else load_openai
    return loader(PROVIDERS[label]["model"])


def require_api_keys():
    from dotenv import load_dotenv

    load_dotenv(KEY_FILE)  # real environment variables take priority
    missing = [p["key"] for p in PROVIDERS.values() if not os.environ.get(p["key"])]
    if missing:
        sys.exit(f"Missing {', '.join(missing)}. Set them as environment variables or in {KEY_FILE}")


# ------------------------------------------------------- cached embeddings ---


def corpus_fingerprint(corpus, questions):
    """Cached vectors are only valid for this exact corpus and question set."""
    text = "\n".join([*(document_text(d) for d in corpus), *(q["body"] for q in questions)])
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def embed_corpus(embed, corpus, stem):
    """Embed in batches, saving after each one so an interrupted run can resume."""
    partial, meta_path = CACHE_DIR / f"{stem}_docs.npy", CACHE_DIR / f"{stem}_docs.json"
    done = np.load(partial) if partial.exists() else None
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"seconds": 0.0, "tokens": 0}
    start = 0 if done is None else len(done)
    for i in range(start, len(corpus), BATCH_SIZE):
        t0 = time.perf_counter()
        vectors, tokens = embed([document_text(d) for d in corpus[i : i + BATCH_SIZE]], "document")
        meta["seconds"] += time.perf_counter() - t0
        meta["tokens"] += tokens
        batch = np.asarray(vectors, dtype=np.float32)
        done = batch if done is None else np.concatenate([done, batch])
        np.save(partial, done)
        meta_path.write_text(json.dumps(meta))
        print(f"  documents {len(done)}/{len(corpus)}", flush=True)
    return done, meta


def embed_questions(embed, questions, stem):
    """One API call per question, as a live search would make, timing each call."""
    vec_path, meta_path = CACHE_DIR / f"{stem}_queries.npy", CACHE_DIR / f"{stem}_queries.json"
    if vec_path.exists() and meta_path.exists():
        return np.load(vec_path), json.loads(meta_path.read_text())
    embed(["warm-up query"], "query")
    vectors, latencies, tokens = [], [], 0
    for q in questions:
        t0 = time.perf_counter()
        result, used = embed([q["body"]], "query")
        latencies.append((time.perf_counter() - t0) * 1000)
        vectors.append(result[0])
        tokens += used
    meta = {"latencies_ms": latencies, "tokens": tokens}
    np.save(vec_path, np.asarray(vectors, dtype=np.float32))
    meta_path.write_text(json.dumps(meta))
    return np.load(vec_path), meta


def retrieve_with(label, questions, corpus, fingerprint):
    """Top-20 pmids per question for one hosted bi-encoder."""
    model = PROVIDERS[label]["model"]
    stem = f"{model}_{fingerprint}"
    embed = load_embedder(label)
    print(f"embedding with {model} ...", flush=True)
    docs, doc_meta = embed_corpus(embed, corpus, stem)
    queries, query_meta = embed_questions(embed, questions, stem)

    # Both providers document unit-length vectors; dot product then equals cosine.
    for name, vectors in (("documents", docs), ("queries", queries)):
        deviation = np.abs(np.linalg.norm(vectors, axis=1) - 1).max()
        if deviation > 1e-3:
            raise ValueError(f"{model} {name} are not normalized (off by {deviation:.4f})")

    candidates = {q["id"]: [corpus[i]["pmid"] for i in search(queries[n], docs, CANDIDATES)[0]]
                  for n, q in enumerate(questions)}
    latencies = query_meta["latencies_ms"]
    info = {"model": model, "dim": int(docs.shape[1]),
            "corpus_embed_seconds": doc_meta["seconds"],
            "tokens_billed": doc_meta["tokens"] + query_meta["tokens"],
            "query_api_latency_ms": {"median": statistics.median(latencies),
                                     "mean": statistics.fmean(latencies),
                                     "p90": float(np.percentile(latencies, 90))}}
    return candidates, info


# ------------------------------------------------------------------- main ---


def main():
    questions, corpus = load_questions(), load_corpus()
    require_api_keys()
    CACHE_DIR.mkdir(exist_ok=True)
    fingerprint = corpus_fingerprint(corpus, questions)
    doc_by_pmid = {d["pmid"]: d for d in corpus}
    qrels = {q["id"]: set(q["relevant_pmids"]) for q in questions}
    qids = list(qrels)

    all_candidates, bi_info = {}, {}
    for label in PROVIDERS:
        all_candidates[label], bi_info[label] = retrieve_with(label, questions, corpus, fingerprint)

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

    # Compare against Experiment 3's runs, question by question.
    grid = json.loads(GRID_PATH.read_text())
    if grid["question_ids"] != qids:
        raise ValueError("bi_encoder_grid.json was built on different questions")
    earlier = {k: np.array(v) for k, v in grid["per_question_ndcg@5"].items()}
    all_means = [v.mean() for v in earlier.values()] + [v.mean() for v in ndcg.values()]

    rng = np.random.default_rng(SEED)
    pipelines = []
    for (bi, rr), ranking in rankings.items():
        per_q = [all_metrics(ranking[q], qrels[q]) for q in qids]
        pipelines.append({
            "bi_encoder": bi,
            "reranker": rr,
            "metrics": {m: float(np.mean([r[m] for r in per_q])) for m in per_q[0]},
            "rank_of_all": 1 + sum(bool(m > ndcg[(bi, rr)].mean()) for m in all_means),
            "vs_app_baseline": paired_stats(ndcg[(bi, rr)] - earlier[APP_BASELINE], rng),
            "vs_bge_small_bi_encoder": paired_stats(ndcg[(bi, rr)] - earlier[f"bge-small (app) | {rr}"], rng),
            "vs_medcpt_bi_encoder": paired_stats(ndcg[(bi, rr)] - earlier[f"MedCPT | {rr}"], rng),
            "gain_from_reranking": None if rr == "none"
            else paired_stats(ndcg[(bi, rr)] - ndcg[(bi, "none")], rng),
        })

    results = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware_info(),
        "num_questions": len(qids),
        "candidates": CANDIDATES,
        "top_k": TOP_K,
        "total_pipelines_ranked": len(all_means),
        "bi_encoders": bi_info,
        "rerankers": RERANKERS,
        "pipelines": pipelines,
        "question_ids": qids,
        "per_question_ndcg@5": {f"{bi} | {rr}": v.tolist() for (bi, rr), v in ndcg.items()},
    }
    save_json(RESULTS_PATH, results)
    print_summary(results)
    print(f"Saved {RESULTS_PATH}")


def print_summary(results):
    print(f"\n{'bi-encoder':<34}{'reranker':<24}{'nDCG@5':>8}{'rank':>6}{'vs bge-small':>14}{'p':>8}")
    for p in results["pipelines"]:
        vs = p["vs_app_baseline"]
        print(f"{p['bi_encoder']:<34}{p['reranker']:<24}{p['metrics']['ndcg@5']:>8.3f}"
              f"{p['rank_of_all']:>6}{vs['delta_ndcg@5']:>+14.3f}{vs['wilcoxon_p']:>8.3f}")
    for label, info in results["bi_encoders"].items():
        print(f"{label}: Recall@20 {info['recall@20']:.3f}, query API "
              f"{info['query_api_latency_ms']['median']:.0f} ms median, "
              f"{info['tokens_billed']:,} tokens")


if __name__ == "__main__":
    main()
