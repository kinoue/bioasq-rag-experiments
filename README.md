# BioASQ RAG Experiments

How much does a reranker improve retrieval on biomedical questions? Three experiments,
each prompted by the last, found that **which models you pick matters more than whether
you rerank**. A Streamlit app lets you inspect the results and individual questions.

This is a learning project, not a production RAG system. It does not generate answers.

**The two stages being compared:**

- A **bi-encoder** embeds the question and each document separately, then compares the
  vectors. Document vectors are precomputed, so searching a corpus takes milliseconds, but
  the model never reads the question and document together.
- A **cross-encoder** (reranker) reads the question and one document together and scores
  the pair. This is slower and can't be precomputed, so it only reorders the bi-encoder's
  top candidates.

**Shared setup:** 100 BioASQ questions and 2,500 PubMed abstracts. Each bi-encoder
retrieves 20 candidates and each pipeline returns a top 5, scored mainly by nDCG@5.
Changes are compared question by question, using a 95% bootstrap CI and a Wilcoxon
signed-rank p-value. See [Data and evaluation](#data-and-evaluation).

## Experiment 1: Does a reranker help?

**Setup:** `BAAI/bge-small-en-v1.5` bi-encoder alone vs. the same bi-encoder followed by
`BAAI/bge-reranker-base`. Both rank the same 20 candidates. Run on CPU (Apple M3 Max).

**Primary observation: no significant difference.** nDCG@5 changed by +0.005
(95% CI [−0.037, +0.047], p = 0.82), while median latency rose about 130×.

| Metric | Bi-encoder | + bge-reranker-base |
|---|---:|---:|
| nDCG@5 | 0.871 | 0.875 |
| MRR@5 | 0.943 | 0.949 |
| Recall@5 | 0.728 | 0.732 |
| Recall@20 (same candidates) | 0.921 | 0.921 |
| Median latency per query | 13 ms | 1,674 ms |

**Analysis:**

- **The bi-encoder was already doing well.** It ranked a relevant document first on 92 of
  100 questions. On 71 questions its top 5 was already the best possible ordering of its
  candidates, so a reranker could only make those worse. Even a perfect reordering would
  only reach nDCG@5 0.961.
- **The cross-encoder hurt almost as much as it helped.** nDCG@5 improved on 24 questions,
  got worse on 18, and didn't change on 58. Large swings nearly cancelled out:
  - *"List the components of the COMPASS complex"* improved from 0.00 to 0.83. The
    bi-encoder matched generic "complex" and "component" papers; the reranker promoted real
    COMPASS papers from ranks 9, 10 and 14.
  - *"Is there any evidence of dysregulated long non coding RNAs and Alzheimer's disease?"*
    fell from 1.00 to 0.32. The bi-encoder's top 5 were all relevant; the reranker moved
    two papers on lncRNAs in cancer to the top.
- **Next question:** is this true of reranking in general, or of this reranker?

## Experiment 2: What about other cross-encoders?

**Setup:** the same bge-small candidates, reordered by three more cross-encoders
(`analysis.py`).

**Primary observation: some cross-encoders improved results significantly.** MedCPT and
MiniLM each gained about +0.06 nDCG@5 on the candidates where bge-reranker-base gained
almost nothing.

| Reranker | nDCG@5 | Change | 95% CI | p | Better / worse |
|---|---:|---:|---|---:|---:|
| None (bi-encoder only) | 0.871 | | | | |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.936 | +0.065 | [+0.032, +0.102] | <0.001 | 24 / 5 |
| `ncbi/MedCPT-Cross-Encoder` | 0.931 | +0.061 | [+0.033, +0.092] | <0.001 | 27 / 4 |
| `mixedbread-ai/mxbai-rerank-base-v1` | 0.913 | +0.043 | [+0.006, +0.082] | 0.06 | 24 / 12 |
| `BAAI/bge-reranker-base` | 0.875 | +0.005 | [−0.037, +0.047] | 0.82 | 24 / 18 |

**Analysis:**

- **Experiment 1's result came from the model, not from reranking itself.** The better
  rerankers rarely made a question worse. Size doesn't explain the gap either: the 22M-parameter
  MiniLM beat the 278M-parameter bge-reranker-base. We also confirmed bge-reranker-base's
  scores match a direct `transformers` run, so this isn't a loading bug.
- **100 questions was enough to see this.** The sample can reliably detect a change of about
  0.06 nDCG@5. A one-off check on 1,000 questions and a ~40,000-document corpus showed the
  same pattern: bge-reranker-base +0.012 (95% CI [−0.003, +0.025], p = 0.07), MedCPT +0.069
  ([+0.058, +0.080]) and MiniLM +0.067. The scripts for this run aren't in the repo.
- **Next question:** we've only varied the reranker. What if we change the bi-encoder?

## Experiment 3: What about other bi-encoders?

**Setup:** seven bi-encoders, each alone and followed by each of the four rerankers, for 35
pipelines (`bi_encoder_grid.py`). Each bi-encoder retrieves its own 20 candidates, so a
pipeline can win through better candidates, better ordering, or both.

**Primary observation: MedCPT bi-encoder + MedCPT reranker performed best**, with nDCG@5
0.961. That is +0.090 over bge-small alone (95% CI [+0.054, +0.131], p < 0.001).

| Bi-encoder | Recall@20 | No reranker | MedCPT | MiniLM | mxbai | bge-reranker |
|---|---:|---:|---:|---:|---:|---:|
| MedCPT | 0.990 | 0.908 | **0.961** | **0.959** | 0.932 | 0.900 |
| bge-base | 0.940 | 0.900 | **0.940** | **0.942** | 0.911 | 0.868 |
| bge-small | 0.921 | 0.871 | 0.931 | 0.936 | 0.913 | 0.875 |
| MedEmbed-small | 0.914 | 0.861 | 0.930 | 0.934 | 0.911 | 0.877 |
| mxbai-embed-xsmall | 0.901 | 0.836 | 0.923 | 0.934 | 0.906 | 0.863 |
| PubMedBERT | 0.846 | 0.773 | 0.892 | 0.890 | 0.873 | 0.840 |
| MiniLM | 0.755 | 0.737 | 0.839 | 0.844 | 0.835 | 0.810 |

Table values other than Recall@20 are nDCG@5. Bold marks the top four pipelines. Models:
`ncbi/MedCPT-Query-Encoder` + `ncbi/MedCPT-Article-Encoder`, `BAAI/bge-base-en-v1.5`,
`BAAI/bge-small-en-v1.5`, `abhinand/MedEmbed-small-v0.1`,
`mixedbread-ai/mxbai-embed-xsmall-v1`, `NeuML/pubmedbert-base-embeddings`,
`sentence-transformers/msmarco-MiniLM-L6-cos-v5`.

**Analysis:**

- **The top four form a clean block: {MedCPT, bge-base} bi-encoder × {MedCPT, MiniLM}
  reranker.** Inside the block, the bi-encoder matters and the reranker doesn't. MedCPT
  beats bge-base by about +0.02 (p = 0.01 and 0.04), while the two rerankers are
  indistinguishable (p = 0.89 and 0.45). Only the two MedCPT bi-encoder pipelines pull away
  from ranks 5–9, by about +0.02 to +0.03, and not every one of those comparisons is
  significant. The two bge-base pipelines can't be separated from ranks 5–9.
- **The MedCPT bi-encoder alone holds up well.** At 0.908 it ranks 15th of 35 and beats
  every bge-reranker-base pipeline, including the original two-stage setup from Experiment 1
  (0.875). Only three pipelines are clearly ahead of it: the two MedCPT bi-encoder pipelines
  above and bge-base + MedCPT (p = 0.016). It also avoids the cross-encoder step.
- **A reranker can't recover what the bi-encoder missed.** The MiniLM bi-encoder gains up to
  +0.107 from reranking but tops out at 0.844, because its candidates contain only 75% of the
  relevant documents.
- **bge-reranker-base is the weakest reranker for every bi-encoder**, and it lowers the score
  of the two strongest (bge-base and MedCPT, neither significantly).
- **Domain tuning isn't automatically better.** MedEmbed-small, bge-small fine-tuned on
  clinical text, didn't beat bge-small (−0.010, not significant).

## Limitations

- **Incomplete labels.** BioASQ annotates a limited set of documents, so a top-ranked
  "unlabeled" document may actually be relevant.
- **Small, curated corpus.** Every relevant document is guaranteed to be among 2,500 abstracts,
  many distractors are off-topic, and absolute scores would be lower across all of PubMed.
- **Many comparisons, uncorrected.** Experiment 3 makes 34 comparisons against one baseline,
  so one or two could look significant by chance. Trust patterns over single cells.
- **MedCPT may have a home advantage.** It was trained on PubMed search clicks, and BioASQ
  experts also found their documents through PubMed search.
- **Latency is only partly measured.** The CPU figures cover Experiment 1's pipeline.
  Experiment 2 didn't measure latency, and Experiment 3 measured only bi-encoder query time,
  on the Apple GPU (5–9 ms for every model).
- **Truncation.** Models read at most 512 tokens (MedCPT queries 64, MiniLM bi-encoder 384), so
  long abstracts are cut off.

## Run it

Requires Python 3.10+. No API keys or paid services are needed. Prepared data and results are
in `data/`, so the app starts right away after downloading its two models (about 1.5 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app has four tabs:

- **Query Explorer:** compare the Experiment 1 rankings side by side for one question.
- **Benchmark Results:** Experiments 1 and 2.
- **Bi-encoder × Reranker:** Experiment 3.
- **Experiment Details:** setup, methodology and limitations.

To reproduce the results (the last two scripts download more models):

```bash
python prepare_data.py                    # build the corpus and embeddings (~2 min)
python evaluation.py                      # Experiment 1 (~3 min on CPU)
python analysis.py                        # Experiment 2
LAB_DEVICE=mps python bi_encoder_grid.py  # Experiment 3 (roughly 25 min on Apple GPU)
pytest                                    # unit tests
```

Everything runs on CPU unless you set `LAB_DEVICE` to `mps` or `cuda`.

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `data.py` | File paths and data loaders |
| `retrieval.py` | Default bi-encoder and dot-product search |
| `reranker.py` | Default cross-encoder and reranking |
| `evaluation.py` | Metrics, timing and the Experiment 1 benchmark |
| `analysis.py` | Experiment 2 |
| `bi_encoder_grid.py` | Experiment 3 |
| `prepare_data.py` | Dataset download, corpus construction and embeddings |
| `tests/test_evaluation.py` | Metric and search tests |

## Data and evaluation

- **Dataset:** the official BioASQ Task 11b training set (4,719 questions, CC BY 2.5), from an
  unchanged copy in
  [`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq),
  pinned to commit `224a87f` and verified by SHA-256.
- **Questions:** 100 sampled with seed 13 from those with 2–10 relevant documents (20 factoid,
  19 list, 29 yes/no, 32 summary).
- **Labels:** BioASQ's own document annotations, unchanged. A document counts as relevant only
  if BioASQ lists it for that question.
- **Corpus:** 522 relevant documents plus 1,978 distractors, which are relevant documents of
  other, randomly sampled BioASQ questions. Titles and abstracts come from PubMed, and 14
  records without abstracts were dropped. `data/metadata.json` records the provenance.
- **Metrics (binary relevance, averaged over questions):**
  - **Recall@k:** share of a question's relevant documents in the top k. It can't reach 1 at
    k = 5 when a question has more than 5 relevant documents.
  - **MRR@5:** 1 / rank of the first relevant document, or 0 if none is in the top 5.
  - **nDCG@5:** ranking quality relative to a perfect ordering.
- **Latency:** `time.perf_counter()`, one question at a time, after a discarded warm-up query.
  Model loading and corpus embedding are excluded.
