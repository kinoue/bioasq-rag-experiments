"""Follow-up analysis: why did bge-reranker-base barely beat the bi-encoder?

Scores the *same* bi-encoder top-20 candidates with several cross-encoders and
asks whether each one's nDCG@5 change is distinguishable from noise:

    python analysis.py

Writes data/reranker_comparison.json.
"""

import math

import numpy as np
from scipy.stats import wilcoxon
from sentence_transformers import CrossEncoder

from data import DATA_DIR, document_text, load_corpus, load_embeddings, load_questions, save_json
from evaluation import CANDIDATES, TOP_K, all_metrics, ndcg_at_k
from reranker import MAX_LENGTH, score_pairs
from retrieval import DEVICE, load_embedding_model, retrieve

RERANKERS = {
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "MedCPT": "ncbi/MedCPT-Cross-Encoder",
    "MiniLM-L6 (MS MARCO)": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "mxbai-rerank-base": "mixedbread-ai/mxbai-rerank-base-v1",
}
RESULTS_PATH = DATA_DIR / "reranker_comparison.json"
BOOTSTRAP_SAMPLES = 10_000
SEED = 0


def bi_encoder_candidates(questions, doc_embeddings):
    """Bi-encoder top-20 per question, computed once and shared by all rerankers."""
    embedder = load_embedding_model()
    return {
        q["id"]: [int(i) for i in retrieve(embedder, q["body"], doc_embeddings, CANDIDATES)[0]]
        for q in questions
    }


def rerank_all(model_name, questions, corpus, candidates):
    """Ranked pmid lists per question after reordering the shared candidates."""
    model = CrossEncoder(model_name, max_length=MAX_LENGTH, device=DEVICE)
    rankings = {}
    for q in questions:
        cand = candidates[q["id"]]
        scores = score_pairs(model, q["body"], [document_text(corpus[i]) for i in cand])
        order = np.argsort(-scores, kind="stable")
        rankings[q["id"]] = [corpus[cand[j]]["pmid"] for j in order]
    return rankings


def oracle_ranking(ranked_ids, relevant):
    """Best possible reorder of the candidates: relevant ones first."""
    return [d for d in ranked_ids if d in relevant] + [d for d in ranked_ids if d not in relevant]


def paired_stats(deltas, rng):
    """Mean difference, bootstrap 95% CI, Wilcoxon p and win/loss counts."""
    deltas = np.asarray(deltas)
    n = len(deltas)
    boot = deltas[rng.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))].mean(axis=1)
    nonzero = deltas[np.abs(deltas) > 1e-9]
    sd = float(deltas.std(ddof=1))
    return {
        "delta_ndcg@5": float(deltas.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
        "improved": int((deltas > 1e-9).sum()),
        "degraded": int((deltas < -1e-9).sum()),
        "sd": sd,
        # Smallest true mean change detectable at 80% power, two-sided alpha 0.05.
        "mde_80pct_power": 2.8 * sd / math.sqrt(n),
    }


def mean_metrics(rankings, qrels):
    per_q = [all_metrics(rankings[qid], qrels[qid]) for qid in qrels]
    return {m: float(np.mean([r[m] for r in per_q])) for m in per_q[0]}


def main():
    questions = load_questions()
    corpus = load_corpus()
    qrels = {q["id"]: set(q["relevant_pmids"]) for q in questions}
    candidates = bi_encoder_candidates(questions, load_embeddings())

    rankings = {"bi-encoder": {qid: [corpus[i]["pmid"] for i in c] for qid, c in candidates.items()}}
    for label, name in RERANKERS.items():
        print(f"scoring with {name} ...", flush=True)
        rankings[label] = rerank_all(name, questions, corpus, candidates)

    ndcg = {
        label: np.array([ndcg_at_k(r[qid], qrels[qid], TOP_K) for qid in qrels])
        for label, r in rankings.items()
    }
    oracle = np.array(
        [ndcg_at_k(oracle_ranking(rankings["bi-encoder"][qid], qrels[qid]), qrels[qid], TOP_K) for qid in qrels]
    )

    rng = np.random.default_rng(SEED)
    results = {
        "num_questions": len(questions),
        "candidates": CANDIDATES,
        "top_k": TOP_K,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "models": {"bi-encoder": "BAAI/bge-small-en-v1.5", **RERANKERS},
        "oracle_ndcg@5": float(oracle.mean()),
        "questions_already_at_oracle": int((np.abs(ndcg["bi-encoder"] - oracle) < 1e-9).sum()),
        "pipelines": {},
    }
    for label in rankings:
        entry = {"metrics": mean_metrics(rankings[label], qrels)}
        if label != "bi-encoder":
            entry["vs_bi_encoder"] = paired_stats(ndcg[label] - ndcg["bi-encoder"], rng)
        results["pipelines"][label] = entry
    save_json(RESULTS_PATH, results)

    print(f"\n{'pipeline':<22}{'nDCG@5':>8}{'MRR@5':>8}{'R@5':>8}{'ΔnDCG@5':>9}  {'95% CI':<17}{'p':>7}{'+/−':>8}{'MDE':>7}")
    for label, entry in results["pipelines"].items():
        m = entry["metrics"]
        row = f"{label:<22}{m['ndcg@5']:>8.3f}{m['mrr@5']:>8.3f}{m['recall@5']:>8.3f}"
        if "vs_bi_encoder" in entry:
            s = entry["vs_bi_encoder"]
            ci = f"[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]"
            p = "<0.001" if s["wilcoxon_p"] < 0.001 else f"{s['wilcoxon_p']:.3f}"
            row += f"{s['delta_ndcg@5']:>+9.3f}  {ci:<17}{p:>7}"
            row += f"{s['improved']:>4}/{s['degraded']:<3}{s['mde_80pct_power']:>7.3f}"
        print(row)
    print(f"\noracle nDCG@5 (perfect reorder of the {CANDIDATES} candidates): {results['oracle_ndcg@5']:.3f}")
    print(f"questions where the bi-encoder is already at the oracle: "
          f"{results['questions_already_at_oracle']}/{len(questions)}")
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
