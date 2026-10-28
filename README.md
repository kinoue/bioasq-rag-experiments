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

1. **On average, reranking barely helped here.** The bi-encoder already puts a known
   relevant document first for most questions (MRR@5 = 0.943), so there is little room to
   improve. A +0.005 nDCG@5 gain on 100 questions is within noise.
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
- **Small sample, no significance testing.** With 100 questions, differences of a few
  nDCG points are not meaningful.
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
| `prepare_data.py` | Dataset download, corpus construction, PubMed fetch, embeddings |
| `tests/test_evaluation.py` | Unit tests for metrics and search |
