"""Build the evaluation set and corpus, then precompute document embeddings.

    python prepare_data.py            # download (if needed), build, embed
    python prepare_data.py --skip-embed

Steps
1. Download the official BioASQ Task 11b training file (if missing).
2. Sample 100 evaluation questions with 2-10 relevant documents (fixed seed).
3. Add distractors: the relevant documents of *other*, randomly chosen BioASQ
   questions, until the corpus reaches ~CORPUS_TARGET documents.
4. Fetch title + abstract for every PubMed ID from NCBI E-utilities.
   Documents without an abstract are dropped (from the corpus and the labels).
5. Embed every document with the bi-encoder and save everything to data/.
"""

import argparse
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from data import (
    CORPUS_PATH,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    QUESTIONS_PATH,
    RAW_BIOASQ_PATH,
    document_text,
    save_json,
)

# Official BioASQ 11b training set, redistributed unchanged (CC BY 2.5) in the
# rag-datasets/rag-mini-bioasq Hugging Face repository. Pinned to a commit.
HF_REVISION = "224a87f64a5c3a720b5bc627cf760f543b2f1e79"
RAW_URL = (
    "https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq/resolve/"
    f"{HF_REVISION}/raw_data/training11b.json"
)
RAW_SHA256 = "6df656862ca860efc355c7805d07ddca700d64ecc3785c519a49afccaaeeac98"

SEED = 13
NUM_QUESTIONS = 100
MIN_RELEVANT, MAX_RELEVANT = 2, 10
CORPUS_TARGET = 2500

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EFETCH_BATCH = 200
NCBI_DELAY_S = 0.4  # NCBI allows 3 requests/second without an API key


# ------------------------------------------------------------- download ---


def download_raw_bioasq():
    if not RAW_BIOASQ_PATH.exists():
        print(f"Downloading {RAW_URL}")
        RAW_BIOASQ_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, RAW_BIOASQ_PATH)
    digest = hashlib.sha256(RAW_BIOASQ_PATH.read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise ValueError(f"Unexpected checksum for {RAW_BIOASQ_PATH}: {digest}")
    return json.loads(RAW_BIOASQ_PATH.read_text())["questions"]


def pmid_from_url(url):
    return url.rstrip("/").split("/")[-1]


def relevant_pmids(question):
    """Deduplicated PubMed IDs, in the order BioASQ lists them."""
    return list(dict.fromkeys(pmid_from_url(u) for u in question["documents"]))


# ---------------------------------------------------------------- PubMed ---


def fetch_pubmed(pmids):
    """Return {pmid: {"title", "abstract"}} for the given IDs via efetch."""
    docs = {}
    pmids = sorted(pmids, key=int)
    for start in range(0, len(pmids), EFETCH_BATCH):
        batch = pmids[start : start + EFETCH_BATCH]
        body = urllib.parse.urlencode(
            {"db": "pubmed", "retmode": "xml", "id": ",".join(batch),
             "tool": "biomedical-reranking-lab"}
        ).encode()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(EFETCH_URL, data=body, timeout=60) as r:
                    xml = r.read()
                break
            except OSError as err:
                print(f"  retry {attempt + 1} after error: {err}")
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("PubMed efetch failed repeatedly")
        docs.update(parse_pubmed_xml(xml))
        print(f"  fetched {min(start + EFETCH_BATCH, len(pmids))}/{len(pmids)}")
        time.sleep(NCBI_DELAY_S)
    return docs


def parse_pubmed_xml(xml):
    docs = {}
    for article in ET.fromstring(xml).iter("PubmedArticle"):
        citation = article.find("MedlineCitation")
        pmid = citation.findtext("PMID")
        title = "".join(citation.find(".//ArticleTitle").itertext()).strip()
        sections = []
        for part in citation.findall(".//Abstract/AbstractText"):
            text = "".join(part.itertext()).strip()
            label = part.get("Label")
            if text:
                sections.append(f"{label}: {text}" if label else text)
        docs[pmid] = {"title": title, "abstract": " ".join(sections)}
    return docs


# --------------------------------------------------------------- corpus ---


def sample_questions(all_questions, rng):
    eligible = [
        q for q in all_questions
        if MIN_RELEVANT <= len(relevant_pmids(q)) <= MAX_RELEVANT
    ]
    rng.shuffle(eligible)
    # Over-sample: some questions may lose documents that lack an abstract.
    return eligible[: int(NUM_QUESTIONS * 1.3)]


def sample_distractor_pmids(all_questions, eval_ids, needed, rng):
    """Relevant documents of other BioASQ questions, added question by question."""
    others = [q for q in all_questions if q["id"] not in eval_ids]
    rng.shuffle(others)
    pmids = []
    for q in others:
        if len(pmids) >= needed:
            break
        pmids.extend(relevant_pmids(q))
    return pmids


def build(all_questions):
    rng = random.Random(SEED)
    candidates = sample_questions(all_questions, rng)
    eval_pmids = {p for q in candidates for p in relevant_pmids(q)}
    # Fetch a margin of extra distractors to cover missing abstracts.
    distractors = sample_distractor_pmids(
        all_questions, {q["id"] for q in candidates},
        int((CORPUS_TARGET - len(eval_pmids)) * 1.15), rng,
    )
    all_pmids = list(dict.fromkeys([*eval_pmids, *distractors]))
    print(f"Fetching {len(all_pmids)} PubMed records")
    fetched = fetch_pubmed(all_pmids)
    usable = {p for p, d in fetched.items() if d["abstract"]}

    questions = []
    for q in candidates:
        rel = [p for p in relevant_pmids(q) if p in usable]
        if len(rel) < MIN_RELEVANT:
            continue
        questions.append(
            {
                "id": q["id"],
                "body": q["body"],
                "type": q["type"],
                "relevant_pmids": rel,
                "num_relevant_original": len(relevant_pmids(q)),
                "snippets": [
                    {"pmid": pmid_from_url(s["document"]), "text": s["text"]}
                    for s in q.get("snippets", [])
                    if pmid_from_url(s["document"]) in rel
                ],
            }
        )
        if len(questions) == NUM_QUESTIONS:
            break

    positive = list(dict.fromkeys(p for q in questions for p in q["relevant_pmids"]))
    distractor_pool = [p for p in dict.fromkeys(distractors)
                       if p in usable and p not in set(positive)]
    corpus_pmids = positive + distractor_pool[: max(0, CORPUS_TARGET - len(positive))]
    corpus_pmids.sort(key=int)
    corpus = [{"pmid": p, **fetched[p]} for p in corpus_pmids]

    stats = {
        "num_questions": len(questions),
        "question_types": {t: sum(q["type"] == t for q in questions)
                           for t in ("factoid", "list", "yesno", "summary")},
        "corpus_size": len(corpus),
        "num_positive_docs": len(positive),
        "num_distractor_docs": len(corpus) - len(positive),
        "relevant_per_question_mean": sum(len(q["relevant_pmids"]) for q in questions)
        / len(questions),
        "pubmed_records_requested": len(all_pmids),
        "pubmed_records_without_abstract_or_missing": len(all_pmids) - len(usable),
    }
    return questions, corpus, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    all_questions = download_raw_bioasq()
    questions, corpus, stats = build(all_questions)
    save_json(QUESTIONS_PATH, questions)
    with CORPUS_PATH.open("w") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    from retrieval import EMBEDDING_MODEL

    metadata = {
        "dataset": "BioASQ Task 11b training set (training11b.json)",
        "dataset_license": "CC BY 2.5 (BioASQ); abstracts from PubMed/NCBI",
        "dataset_source": RAW_URL,
        "dataset_sha256": RAW_SHA256,
        "bioasq_questions_total": len(all_questions),
        "seed": SEED,
        "selection": f"{MIN_RELEVANT}-{MAX_RELEVANT} relevant documents per question",
        "distractors": "relevant documents of other, randomly sampled BioASQ 11b "
        "questions (not judged for the evaluation questions)",
        "pubmed_fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": EMBEDDING_MODEL,
        **stats,
    }
    save_json(METADATA_PATH, metadata)
    print(json.dumps(stats, indent=2))

    if not args.skip_embed:
        import numpy as np

        from retrieval import embed_documents, load_embedding_model

        print(f"Embedding {len(corpus)} documents with {EMBEDDING_MODEL}")
        model = load_embedding_model()
        embeddings = embed_documents(model, [document_text(d) for d in corpus])
        np.save(EMBEDDINGS_PATH, embeddings)
        print(f"Saved {EMBEDDINGS_PATH} {embeddings.shape}")


if __name__ == "__main__":
    main()
