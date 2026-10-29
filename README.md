# BioASQ RAG Experiments

A small, educational experiment comparing two ways to find PubMed abstracts that
answer a biomedical question:

- **A. Bi-encoder retrieval:** embed the question, return the 5 most similar abstracts.
- **B. Bi-encoder + cross-encoder reranking:** take the *same* top 20 candidates from A,
  rescore each (question, abstract) pair with a cross-encoder, and return the new top 5.

Both pipelines rank an identical candidate pool, so any difference comes from reranking
alone. A Streamlit app shows the two rankings side by side, how every candidate moved,
and aggregate benchmark results on 100 BioASQ questions.

This is a learning project, not a production RAG system: there is no answer generation,
and the corpus is a small curated set (see [Limitations](#limitations)).

## Results

100 BioASQ questions, a 2,500-abstract corpus, CPU (Apple M3 Max, 10 PyTorch threads):

| Metric | Bi-encoder | + Reranker | Difference |
|---|---:|---:|---:|
| Recall@5 | 0.728 | 0.732 | +0.004 |
| MRR@5 | 0.943 | 0.949 | +0.006 |
| nDCG@5 | 0.871 | 0.875 | +0.005 |
| Recall@20 (shared candidate pool) | 0.921 | 0.921 | 0 |
| Median latency per query | 13 ms | 1,674 ms | ~130× slower |

Per question, reranking **improved** nDCG@5 on 24 questions, left it **unchanged** on 58,
and **degraded** it on 18. On the Apple GPU (`LAB_DEVICE=mps`), reranking took about
300 ms per query (measured on the first 30 questions).

### What I learned

1. **This reranker barely helped here, but reranking in general does.** The +0.005
   nDCG@5 gain from `bge-reranker-base` is within noise. Two other cross-encoders, scoring
   the *same* 20 candidates, gained about +0.06 with clear statistical significance. The
   choice of reranker mattered more than the decision to rerank. See the
   [follow-up analysis](#why-reranking-barely-helped-follow-up-analysis).
2. **The average hides large swings in both directions.** Individual questions moved by
   between −0.68 and +0.83 nDCG@5. Reranking is not a uniform improvement.
3. **The cross-encoder fixes "keyword-similar but off-topic" errors.** It reads the
   question and abstract together, so it can tell that "COMPASS complex" is a histone
   methyltransferase complex and not just any protein complex.
4. **It also makes confident mistakes.** It sometimes pushes documents that match a
   prominent phrase in the question (e.g. "long non-coding RNA") above documents that
   address the actual question (lncRNAs *in Alzheimer's disease*).
5. **Cost is the real trade-off.** Scoring 20 pairs with a 278M-parameter cross-encoder
   costs ~1.7 s on CPU, versus ~13 ms for the whole bi-encoder search.

### Examples

Where reranking **helped** (nDCG@5, bi-encoder → reranked):

- **"List the components of the COMPASS complex"** (0.00 → 0.83). The bi-encoder's top 5
  were generic "complex"/"component" papers (complexins, MRI independent components).
  The reranker promoted three real COMPASS papers from ranks 9, 10 and 14.
- **"Are there negative enhancers?"** (0.13 → 0.87). The bi-encoder matched "enhancer"
  broadly (enhancer maps, insulators). The reranker promoted papers on
  *negative enhancer-like elements* and silencers from ranks 6–19.
- **"What is filipin staining used for?"** (0.15 → 0.72). The bi-encoder retrieved
  Western-blot staining papers; the reranker found the cholesterol/Niemann-Pick type C
  papers where filipin is actually used.

Where reranking **hurt**:

- **"Is there any evidence of dysregulated long non coding RNAs and Alzheimer's
  disease?"** (1.00 → 0.32). The bi-encoder's top 5 were all relevant. The reranker
  promoted "Long non-coding RNA in cancer" and a HOTAIR/metastasis paper to the top two
  spots.
- **"Which biological process takes place in nuclear speckles?"** (1.00 → 0.35). The
  reranker put a DNA-damage-response paper first and scored a relevant mRNA-trafficking
  paper at only 0.012.
- **"Which olfactory gene senses androsterone?"** (0.87 → 0.32). The reranker preferred
  steroid-synthesis papers (CYP17 inhibitors) over the odorant-perception papers.

Select any question in the app's **Benchmark Results** table to see these rankings.

## Why reranking barely helped (follow-up analysis)

`python analysis.py` takes the bi-encoder's top 20 for each question once, then reorders
those same candidates with four different cross-encoders. Every model reads at most 512
tokens of `title + "\n" + abstract`. For each reranker it reports the paired change in
nDCG@5 against the bi-encoder:

- a 95% bootstrap confidence interval (10,000 resamples of the 100 questions);
- a Wilcoxon signed-rank p-value (on questions whose score changed);
- how many questions improved or got worse.

Results are saved to `data/reranker_comparison.json`.

Same 2,500-abstract corpus and 100 questions as above:

| Pipeline | nDCG@5 | MRR@5 | Recall@5 | ΔnDCG@5 | 95% CI | Wilcoxon p | Improved / degraded |
|---|---:|---:|---:|---:|---|---:|---:|
| Bi-encoder only | 0.871 | 0.943 | 0.728 | | | | |
| + `BAAI/bge-reranker-base` (default) | 0.875 | 0.949 | 0.732 | +0.005 | [−0.037, +0.047] | 0.82 | 24 / 18 |
| + `ncbi/MedCPT-Cross-Encoder` | 0.931 | 0.985 | 0.778 | +0.061 | [+0.033, +0.092] | <0.001 | 27 / 4 |
| + `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.936 | 0.993 | 0.780 | +0.065 | [+0.032, +0.102] | <0.001 | 24 / 5 |
| + `mixedbread-ai/mxbai-rerank-base-v1` | 0.913 | 0.980 | 0.758 | +0.043 | [+0.006, +0.082] | 0.06 | 24 / 12 |

What this shows:

- **The sample was big enough to see a real gain.** Per-question changes vary a lot
  (standard deviation ≈ 0.22 for `bge-reranker-base`). With 100 questions, the smallest
  mean change we could reliably detect (80% power, two-sided α = 0.05, `2.8·sd/√n`) is
  about **0.06** nDCG@5. MedCPT and MiniLM clear that bar. `bge-reranker-base`'s +0.005
  does not. mxbai is borderline: its bootstrap interval just excludes zero, but the
  Wilcoxon test does not reach significance.
- **The headroom is small but not zero.** Perfectly reordering the 20 candidates would
  reach nDCG@5 = **0.961**, and the bi-encoder is already at that ceiling on **71 of 100**
  questions. MedCPT and MiniLM recover about two-thirds of the 0.09 gap.
  `bge-reranker-base` recovers almost none. It helps and hurts about equally often
  (24 vs 18 questions), while the better rerankers rarely make things worse.
- **It is not an implementation bug.** `bge-reranker-base` scores loaded through
  sentence-transformers match a raw `transformers` forward pass of the same model, so the
  weak result comes from the model on this data, not from how it is called.
- **Model size is not the explanation.** The 22M-parameter MiniLM, trained on web
  search (MS MARCO), beat the 278M-parameter `bge-reranker-base`. MedCPT was trained on
  PubMed search logs.

**Larger corpus (one-off run, not reproduced by `analysis.py`).** We repeated the
comparison with the bi-encoder searching a corpus of ~40,000 BioASQ documents. This is
much harder, and the bi-encoder scores drop accordingly.

| Questions | Bi-encoder | + bge-reranker-base | + MedCPT | + MiniLM |
|---|---:|---:|---:|---:|
| 100 | 0.689 | 0.679 (−0.010, n.s.) | 0.742 (+0.053) | 0.741 (+0.053) |
| 1,000 | 0.658 | 0.669 (+0.012, 95% CI [−0.003, +0.025], p = 0.07) | 0.727 (+0.069, 95% CI [+0.058, +0.080]) | 0.724 (+0.067) |

Even with ten times as many questions, `bge-reranker-base`'s gain is small and not
significant. MedCPT and MiniLM gain about +0.07. The pattern is the same as on the small
corpus.

The app still uses `bge-reranker-base`. To try another model, change `RERANKER_MODEL` in
`reranker.py` (one line), then re-run `python evaluation.py` to refresh the benchmark
results.

## How bi-encoders and cross-encoders differ

A **bi-encoder** (`BAAI/bge-small-en-v1.5`, 33M parameters) maps the question and each
document to vectors *independently*. Document vectors are computed once, ahead of time.
At query time we embed only the question and take a dot product with every document
vector (embeddings are L2-normalized, so this is cosine similarity). That is fast enough
to search a whole corpus, but the model never sees the question and document together.

A **cross-encoder** (`BAAI/bge-reranker-base`, 278M parameters) takes the question and one
document as a single input and outputs a relevance score. Every question token can
attend to every document token, so it can model interactions such as "this abstract
mentions COMPASS *and* lists its subunits". Nothing can be precomputed, so it runs once
per pair. That is why it is used only to rerank a short candidate list.

## Install and run

Requires Python 3.10+ and about 1.5 GB of disk for the two models (downloaded from
Hugging Face on first run). No API keys or paid services are needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The prepared corpus, embeddings and benchmark results are already in `data/`, so the app
starts immediately. To rebuild or re-run everything:

```bash
python prepare_data.py   # download BioASQ, fetch abstracts from PubMed, embed (~2 min)
python evaluation.py     # benchmark both pipelines on all questions (~3 min on CPU)
python analysis.py       # compare four rerankers on the same candidates (downloads ~1.3 GB more)
pytest                   # metric unit tests
```

Everything runs on CPU by default. Set `LAB_DEVICE=mps` (Apple silicon) or
`LAB_DEVICE=cuda` to use a GPU.

## Dataset and corpus

- **Source:** the official **BioASQ Task 11b training set** (`training11b.json`, 4,719
  questions), licensed CC BY 2.5. BioASQ distributes it after free registration; this
  project downloads an unchanged copy from the
  [`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq)
  Hugging Face repository, pinned to commit `224a87f` and verified by SHA-256.
- **Questions:** 100 questions sampled with seed 13 from those with 2–10 relevant
  documents (20 factoid, 19 list, 29 yes/no, 32 summary). The range keeps Recall@5
  meaningful and leaves room for the ranking to change.
- **Relevance labels:** the official BioASQ document lists, unchanged. On average there
  are 5.22 relevant documents per question.
- **Corpus:** 2,500 PubMed abstracts: all 522 relevant documents of the evaluation
  questions, plus 1,978 distractors. Distractors are the relevant documents of other,
  randomly sampled BioASQ 11b questions. They are real biomedical abstracts, but they
  were not judged for the evaluation questions.
- **Text:** titles and abstracts fetched from NCBI E-utilities (`efetch`). The 14 records
  with no abstract were dropped from the corpus and from the labels. Each document is
  represented as `title + "\n" + abstract`.
- `data/metadata.json` records the source URL, checksum, seed, fetch date and counts.
  Because PubMed records can be revised, re-running `prepare_data.py` later may give
  slightly different text.

## How the evaluation works

For each question, both pipelines produce a ranked top 5. We then compute three metrics,
treating a document as relevant only if BioASQ lists it:

- **Recall@5:** the fraction of the question's relevant documents found in the top 5.
  When a question has more than 5 relevant documents, the maximum possible score is
  below 1.
- **MRR@5:** 1 / rank of the first relevant document (0 if none is in the top 5).
- **nDCG@5:** the sum of `1 / log2(rank + 1)` over relevant hits, divided by the same sum
  for a perfect ranking of `min(#relevant, 5)` relevant documents.

Scores are averaged over the 100 questions. As a sanity check, the benchmark confirms
that both pipelines rank identical candidate sets and therefore have the same Recall@20.

**Latency** is measured with `time.perf_counter()`, one question at a time, after one
discarded warm-up query:

- *Retrieval* covers query embedding plus the dot-product search.
- *Reranking* covers 20 cross-encoder pairs in one batch plus the sort.
- Model loading and corpus embedding are excluded.

We report median, mean and p90.

## Limitations

- **Incomplete judgments.** BioASQ experts annotated a limited set of documents per
  question. An unlabeled document in the top 5 may actually be relevant, which can
  penalize either pipeline unfairly.
- **Small, curated corpus.** Every relevant document is guaranteed to be among only
  2,500 abstracts. Searching all 36M+ PubMed records is far harder, and absolute scores
  here are much higher than they would be in real search.
- **Random distractors.** Many distractors are on unrelated topics, which makes the
  bi-encoder's job easier. Some are near-misses because BioASQ revisits topics.
- **Small sample.** With 100 questions, nDCG@5 differences smaller than about 0.06 cannot
  be reliably detected. The main benchmark reports no significance tests. `analysis.py`
  does (bootstrap CI and Wilcoxon), but it compares five pipelines without correcting
  for multiple comparisons.
- **Truncation.** Both models read at most 512 tokens; long abstracts are cut off.
- **Latency is machine-specific** and measured on one laptop, one query at a time.

## Code layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: Query Explorer, Benchmark Results, Experiment Details |
| `data.py` | File paths and loaders for questions, corpus, embeddings |
| `retrieval.py` | Bi-encoder model, query embedding, dot-product search |
| `reranker.py` | Cross-encoder scoring and reranking of a fixed candidate list |
| `evaluation.py` | Metrics, timed pipeline comparison, benchmark CLI |
| `analysis.py` | Follow-up: four rerankers on the same candidates, with significance tests |
| `prepare_data.py` | Dataset download, corpus construction, PubMed fetch, embeddings |
| `tests/test_evaluation.py` | Unit tests for metrics and search |
