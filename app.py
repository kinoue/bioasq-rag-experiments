"""Streamlit UI: compare bi-encoder retrieval with cross-encoder reranking.

    streamlit run app.py
"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from analysis import RESULTS_PATH as RERANKER_COMPARISON_PATH
from bi_encoder_grid import RESULTS_PATH as GRID_PATH
from data import (
    BENCHMARK_PATH,
    load_corpus,
    load_embeddings,
    load_metadata,
    load_questions,
    save_json,
)
from evaluation import (
    CANDIDATES,
    TOP_K,
    all_metrics,
    compare_pipelines,
    hardware_info,
    run_benchmark,
)
from reranker import MAX_LENGTH, RERANKER_MODEL, load_reranker
from retrieval import EMBEDDING_MODEL, QUERY_INSTRUCTION, load_embedding_model

BASELINE_COLOR = "#2a78d6"
RERANKED_COLOR = "#eb6834"
RELEVANT_COLOR = "#1baf7a"
UNLABELED_COLOR = "#9a9993"
PIPELINE_NAMES = {"baseline": "Bi-encoder", "reranked": "Bi-encoder + reranker"}
EXAMPLE_CUSTOM_QUESTION = "Which drugs are used to treat idiopathic pulmonary fibrosis?"

st.set_page_config(page_title="Biomedical Reranking Lab", layout="wide")


# ------------------------------------------------------ cached resources ---


@st.cache_resource(show_spinner="Loading models...")
def get_models():
    embedder, reranker = load_embedding_model(), load_reranker()
    # Warm up both models so the first timed query is not inflated.
    embedder.encode("warm-up")
    reranker.predict([("warm-up", "warm-up")])
    return embedder, reranker


@st.cache_resource(show_spinner="Loading corpus...")
def get_data():
    questions = load_questions()
    corpus = load_corpus()
    return {
        "questions": questions,
        "by_id": {q["id"]: q for q in questions},
        "corpus": corpus,
        "doc_by_pmid": {d["pmid"]: d for d in corpus},
        "embeddings": load_embeddings(),
        "metadata": load_metadata(),
    }


def load_saved_benchmark():
    if BENCHMARK_PATH.exists():
        return json.loads(BENCHMARK_PATH.read_text())
    return None


data = get_data()
embedder, reranker = get_models()
state = st.session_state
state.setdefault("benchmark", load_saved_benchmark())


def open_in_explorer():
    """Callback for the benchmark table: load the selected question."""
    rows = state["bench_table"]["selection"]["rows"]
    if rows:
        state["explorer_mode"] = "BioASQ question"
        state["explorer_qid"] = state["bench_table_ids"][rows[0]]
        state["auto_run"] = True
        state["active_tab"] = "Query Explorer"


# ----------------------------------------------------------- explorer ---


def movement_label(original_rank, new_rank):
    if new_rank < original_rank:
        return f":green[▲ {original_rank - new_rank}]"
    if new_rank > original_rank:
        return f":red[▼ {new_rank - original_rank}]"
    return ":gray[= same]"


def relevance_badge(pmid, relevant):
    if relevant is None:
        return ""
    if pmid in relevant:
        return ":green-badge[:material/check: relevant]"
    return ":gray-badge[not labeled relevant]"


def render_doc(item, kind, relevant, question, rerank_rank_by_pmid):
    doc = data["doc_by_pmid"][item["pmid"]]
    if kind == "baseline":
        new_rank = rerank_rank_by_pmid[item["pmid"]]
        rank_line = (
            f"**#{item['rank']}** · cosine {item['score']:.3f} · "
            f"after reranking: #{new_rank} {movement_label(item['rank'], new_rank)}"
        )
    else:
        rank_line = (
            f"**#{item['rank']}** · reranker {item['score']:.3f} · "
            f"was #{item['original_rank']} {movement_label(item['original_rank'], item['rank'])}"
        )
    with st.container(border=True):
        st.markdown(f"{rank_line} {relevance_badge(item['pmid'], relevant)}")
        st.markdown(
            f"**{doc['title']}** "
            f"[PMID {doc['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{doc['pmid']}/)"
        )
        abstract = doc["abstract"]
        st.caption(abstract if len(abstract) <= 320 else abstract[:320] + "…")
        with st.expander("Full abstract"):
            st.write(abstract)
            snippets = [s["text"] for s in (question or {}).get("snippets", [])
                        if s["pmid"] == doc["pmid"]]
            if snippets:
                st.markdown("**BioASQ snippets annotated for this question:**")
                for s in snippets:
                    st.markdown(f"> {s}")


def rank_movement_chart(result, relevant):
    rows = []
    for item in result["reranked"]:
        doc = data["doc_by_pmid"][item["pmid"]]
        if relevant is None:
            label = "no labels (custom question)"
        else:
            label = "relevant" if item["pmid"] in relevant else "not labeled relevant"
        for stage, rank in (("Bi-encoder rank", item["original_rank"]),
                            ("Reranked rank", item["rank"])):
            rows.append({"pmid": item["pmid"], "title": doc["title"][:90],
                         "stage": stage, "rank": rank, "label": label,
                         "from": item["original_rank"], "to": item["rank"]})
    df = pd.DataFrame(rows)
    if relevant is None:
        domain, colors = ["no labels (custom question)"], [BASELINE_COLOR]
    else:
        domain, colors = ["relevant", "not labeled relevant"], [RELEVANT_COLOR, UNLABELED_COLOR]
    color = alt.Color(
        "label:N", title=None,
        scale=alt.Scale(domain=domain, range=colors),
        legend=alt.Legend(orient="top"),
    )
    y = alt.Y("rank:Q", scale=alt.Scale(domain=[CANDIDATES + 0.5, 0.5]),
              axis=alt.Axis(values=list(range(1, CANDIDATES + 1)), title="Rank"))
    x = alt.X("stage:N", sort=["Bi-encoder rank", "Reranked rank"], title=None,
              axis=alt.Axis(labelAngle=0, orient="top"))
    tooltip = [alt.Tooltip("title:N", title="Title"), alt.Tooltip("pmid:N", title="PMID"),
               alt.Tooltip("from:Q", title="Bi-encoder rank"),
               alt.Tooltip("to:Q", title="Reranked rank"), alt.Tooltip("label:N", title="Label")]
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=x, y=y, color=color, detail="pmid:N", tooltip=tooltip)
    points = alt.Chart(df).mark_circle(size=90, opacity=1).encode(
        x=x, y=y, color=color, tooltip=tooltip)
    cutoff = alt.Chart(pd.DataFrame({"rank": [TOP_K + 0.5]})).mark_rule(
        strokeDash=[4, 4], color=UNLABELED_COLOR).encode(y="rank:Q")
    return (lines + points + cutoff).properties(height=460)


def candidates_table(result, relevant):
    base_score = {d["pmid"]: d["score"] for d in result["baseline"]}
    rows = []
    for item in result["reranked"]:
        rows.append({
            "Reranked rank": item["rank"],
            "Bi-encoder rank": item["original_rank"],
            "Moved": item["original_rank"] - item["rank"],
            "Relevant": "—" if relevant is None else ("yes" if item["pmid"] in relevant else ""),
            "Cosine": base_score[item["pmid"]],
            "Reranker": item["score"],
            "Title": data["doc_by_pmid"][item["pmid"]]["title"],
            "PMID": item["pmid"],
        })
    return pd.DataFrame(rows)


def render_question_metrics(result, relevant):
    base = all_metrics([d["pmid"] for d in result["baseline"]], relevant)
    rr = all_metrics([d["pmid"] for d in result["reranked"]], relevant)
    cols = st.columns(4)
    for col, (m, name) in zip(cols, [("ndcg@5", "nDCG@5"), ("mrr@5", "MRR@5"),
                                     ("recall@5", "Recall@5")]):
        delta = rr[m] - base[m]
        col.metric(f"{name} (bi-encoder → reranked)", f"{base[m]:.3f} → {rr[m]:.3f}",
                   delta=f"{delta:+.3f}" if abs(delta) > 1e-9 else None)
    cols[3].metric("Known relevant docs in corpus", len(relevant))


def explorer_tab():
    st.radio("Question source", ["BioASQ question", "Custom question"],
             key="explorer_mode", horizontal=True)
    question = None
    if state["explorer_mode"] == "BioASQ question":
        qids = [q["id"] for q in data["questions"]]
        qid = st.selectbox(
            "BioASQ question", qids, key="explorer_qid",
            format_func=lambda i: f"[{data['by_id'][i]['type']}] {data['by_id'][i]['body']}",
        )
        question = data["by_id"][qid]
        text = question["body"]
    else:
        text = st.text_input("Your biomedical question", value=EXAMPLE_CUSTOM_QUESTION,
                             key="custom_question").strip()

    run = st.button("Run Comparison", type="primary", disabled=not text)
    if run or state.pop("auto_run", False):
        with st.spinner("Retrieving and reranking..."):
            result = compare_pipelines(text, embedder, reranker, data["corpus"],
                                       data["embeddings"])
        state["explorer_result"] = {"text": text,
                                    "qid": question["id"] if question else None, **result}

    result = state.get("explorer_result")
    if not result:
        st.info("Pick a question and press **Run Comparison**.")
        return
    if result["text"] != text:
        st.warning("Showing results for a previous question. Press **Run Comparison** to update.")
    question = data["by_id"].get(result["qid"])
    relevant = set(question["relevant_pmids"]) if question else None

    st.markdown(f"#### {result['text']}")
    lat = result["latency_ms"]
    st.caption(
        f"Latency: retrieval {lat['retrieval']:.0f} ms · reranking {lat['reranking']:.0f} ms "
        f"· total {lat['total']:.0f} ms ({hardware_info()['device'].upper()})"
    )
    if relevant is None:
        st.caption("Custom question: there are no BioASQ annotations, so no relevance "
                   "labels or accuracy metrics are shown.")
    else:
        render_question_metrics(result, relevant)

    rerank_rank = {d["pmid"]: d["rank"] for d in result["reranked"]}
    left, right = st.columns(2)
    with left:
        st.subheader(f":blue[■] Bi-encoder top {TOP_K}")
        for item in result["baseline"][:TOP_K]:
            render_doc(item, "baseline", relevant, question, rerank_rank)
    with right:
        st.subheader(f":orange[■] Reranked top {TOP_K}")
        for item in result["reranked"][:TOP_K]:
            render_doc(item, "reranked", relevant, question, rerank_rank)

    st.subheader(f"How all {CANDIDATES} candidates moved")
    st.caption(f"Both pipelines rank the same {CANDIDATES} candidates. Each line is one "
               f"document; the dashed line is the top-{TOP_K} cutoff. Hover for titles.")
    chart_col, table_col = st.columns([2, 3])
    chart_col.altair_chart(rank_movement_chart(result, relevant), use_container_width=True)
    table_col.dataframe(
        candidates_table(result, relevant), hide_index=True, height=460,
        column_config={
            "Moved": st.column_config.NumberColumn(help="Positive = moved up", format="%+d"),
            "Cosine": st.column_config.NumberColumn(format="%.3f"),
            "Reranker": st.column_config.NumberColumn(format="%.3f"),
        },
    )


# ---------------------------------------------------------- benchmark ---


def aggregate_table(agg):
    rows = []
    for m, name in [("recall@5", "Recall@5"), ("mrr@5", "MRR@5"), ("ndcg@5", "nDCG@5"),
                    ("recall@20", "Recall@20 (candidate pool)")]:
        b, r = agg["baseline"][m], agg["reranked"][m]
        rows.append({"Metric": name, "Bi-encoder": b, "Bi-encoder + reranker": r,
                     "Difference": r - b})
    return pd.DataFrame(rows)


def metrics_chart(agg):
    rows = [{"metric": name, "pipeline": PIPELINE_NAMES[p], "value": agg[p][m]}
            for m, name in [("recall@5", "Recall@5"), ("mrr@5", "MRR@5"), ("ndcg@5", "nDCG@5")]
            for p in ("baseline", "reranked")]
    return alt.Chart(pd.DataFrame(rows)).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("metric:N", title=None, sort=["Recall@5", "MRR@5", "nDCG@5"],
                axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("pipeline:N", sort=list(PIPELINE_NAMES.values())),
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
        color=alt.Color("pipeline:N", title=None, sort=list(PIPELINE_NAMES.values()),
                        scale=alt.Scale(range=[BASELINE_COLOR, RERANKED_COLOR]),
                        legend=alt.Legend(orient="top")),
        tooltip=["pipeline:N", "metric:N", alt.Tooltip("value:Q", format=".3f")],
    ).properties(height=280)


def per_question_frame(results, sort_by):
    df = pd.DataFrame([{
        "id": r["id"],
        "Question": r["question"],
        "Type": r["type"],
        "Relevant docs": r["num_relevant"],
        "Bi-encoder nDCG@5": r["baseline"]["ndcg@5"],
        "Reranked nDCG@5": r["reranked"]["ndcg@5"],
        "Difference": r["reranked"]["ndcg@5"] - r["baseline"]["ndcg@5"],
    } for r in results["per_question"]])
    if sort_by == "Largest improvement first":
        df = df.sort_values("Difference", ascending=False, kind="stable")
    elif sort_by == "Largest degradation first":
        df = df.sort_values("Difference", ascending=True, kind="stable")
    return df.reset_index(drop=True)


def benchmark_tab():
    st.caption(f"Evaluates both pipelines on all {len(data['questions'])} BioASQ questions. "
               "Takes a few minutes on CPU (mostly cross-encoder time).")
    if st.button("Run Benchmark", type="primary"):
        bar = st.progress(0.0, text="Running benchmark...")
        results = run_benchmark(
            embedder, reranker, data["corpus"], data["embeddings"], data["questions"],
            progress=lambda done, total: bar.progress(done / total, text=f"{done}/{total} questions"),
        )
        save_json(BENCHMARK_PATH, results)
        state["benchmark"] = results
        bar.empty()

    results = state["benchmark"]
    if not results:
        st.info("No saved benchmark yet. Press **Run Benchmark**.")
        return
    agg = results["aggregate"]
    hw = results["hardware"]
    st.caption(f"Results from {results['created']} · {results['num_questions']} questions · "
               f"{hw['device'].upper()} ({hw.get('cpu', '')})")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Retrieval quality")
        st.dataframe(aggregate_table(agg), hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.3f")
            for c in ("Bi-encoder", "Bi-encoder + reranker")
        } | {"Difference": st.column_config.NumberColumn(format="%+.3f")})
        if agg["all_candidates_identical"]:
            st.success(f"Sanity check passed: both pipelines ranked the identical "
                       f"{CANDIDATES} candidates for every question, so Recall@20 is equal "
                       f"({agg['baseline']['recall@20']:.3f}).", icon=":material/check:")
        else:
            st.error("Sanity check failed: candidate pools differ between pipelines.")
    with right:
        st.altair_chart(metrics_chart(agg), use_container_width=True)

    st.subheader("Latency per query")
    lat = agg["latency_ms"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieval (median)", f"{lat['retrieval']['median']:.1f} ms")
    c2.metric("Reranking (median)", f"{lat['reranking']['median']:.0f} ms")
    c3.metric("Total (median)", f"{lat['total']['median']:.0f} ms")
    c4.metric("Reranking slowdown", f"{lat['total']['median'] / lat['retrieval']['median']:.0f}×",
              help="Median total latency divided by median retrieval-only latency")
    st.caption(
        "Mean / p90 — " + " · ".join(
            f"{stage}: {s['mean']:.1f} / {s['p90']:.1f} ms" for stage, s in lat.items())
        + ". Model loading and corpus embedding are excluded; one warm-up query is discarded.")

    st.subheader("Per-question nDCG@5")
    out = agg["ndcg@5_outcomes"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Improved by reranking", out["improved"])
    c2.metric("Unchanged", out["unchanged"])
    c3.metric("Degraded by reranking", out["degraded"])

    sort_by = st.radio("Sort", ["Largest improvement first", "Largest degradation first",
                                "Original order"], horizontal=True, key="bench_sort")
    df = per_question_frame(results, sort_by)
    state["bench_table_ids"] = df["id"].tolist()
    st.caption("Select a row to open that question in the Query Explorer.")
    st.dataframe(
        df.drop(columns=["id"]), hide_index=True, key="bench_table",
        on_select=open_in_explorer, selection_mode="single-row",
        column_config={
            "Question": st.column_config.TextColumn(width="large"),
            "Bi-encoder nDCG@5": st.column_config.NumberColumn(format="%.3f"),
            "Reranked nDCG@5": st.column_config.NumberColumn(format="%.3f"),
            "Difference": st.column_config.NumberColumn(format="%+.3f"),
        },
    )


def reranker_comparison_frame(comp):
    rows = []
    for name, p in comp["pipelines"].items():
        m, vs = p["metrics"], p.get("vs_bi_encoder")
        rows.append({
            "Pipeline": "Bi-encoder only" if vs is None else f"+ {name}",
            "nDCG@5": m["ndcg@5"], "MRR@5": m["mrr@5"], "Recall@5": m["recall@5"],
            "ΔnDCG@5": vs and vs["delta_ndcg@5"],
            "95% CI": vs and f"[{vs['ci95'][0]:+.3f}, {vs['ci95'][1]:+.3f}]",
            "Wilcoxon p": vs and ("<0.001" if vs["wilcoxon_p"] < 0.001 else f"{vs['wilcoxon_p']:.2f}"),
            "Improved / degraded": vs and f"{vs['improved']} / {vs['degraded']}",
        })
    return pd.DataFrame(rows)


def delta_ci_chart(comp, mde):
    """ΔnDCG@5 per reranker with its 95% CI; the shaded band is what n can't resolve."""
    df = pd.DataFrame([
        {"reranker": name, "delta": vs["delta_ndcg@5"], "lo": vs["ci95"][0], "hi": vs["ci95"][1],
         "p": vs["wilcoxon_p"], "ours": name == "bge-reranker-base"}
        for name, p in comp["pipelines"].items() if (vs := p.get("vs_bi_encoder"))
    ])
    y = alt.Y("reranker:N", title=None, sort=alt.EncodingSortField("delta", order="descending"),
              axis=alt.Axis(labelLimit=200))
    band = alt.Chart(pd.DataFrame({"lo": [-mde], "hi": [mde]})).mark_rect(
        color=UNLABELED_COLOR, opacity=0.18).encode(x="lo:Q", x2="hi:Q")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=UNLABELED_COLOR).encode(x="x:Q")
    tooltip = ["reranker:N", alt.Tooltip("delta:Q", format="+.3f", title="ΔnDCG@5"),
               alt.Tooltip("lo:Q", format="+.3f", title="CI low"),
               alt.Tooltip("hi:Q", format="+.3f", title="CI high"),
               alt.Tooltip("p:Q", format=".3g", title="Wilcoxon p")]
    whiskers = alt.Chart(df).mark_rule(strokeWidth=2).encode(
        y=y, x=alt.X("lo:Q", title="ΔnDCG@5 vs. bi-encoder (95% CI)"), x2="hi:Q",
        color=alt.condition("datum.ours", alt.value(RERANKED_COLOR), alt.value(BASELINE_COLOR)),
        tooltip=tooltip)
    dots = alt.Chart(df).mark_circle(size=110, opacity=1).encode(
        y=y, x="delta:Q", tooltip=tooltip,
        color=alt.condition("datum.ours", alt.value(RERANKED_COLOR), alt.value(BASELINE_COLOR)))
    return (band + zero + whiskers + dots).properties(height=220)


def reranker_comparison_section():
    st.divider()
    st.subheader("Follow-up: is it the reranker or the sample size?")
    if not RERANKER_COMPARISON_PATH.exists():
        st.info("Run `python analysis.py` to compare several rerankers on the same candidates.")
        return
    comp = json.loads(RERANKER_COMPARISON_PATH.read_text())
    bge = comp["pipelines"]["bge-reranker-base"]["vs_bi_encoder"]
    bi_ndcg = comp["pipelines"]["bi-encoder"]["metrics"]["ndcg@5"]
    st.caption(
        f"Precomputed by `python analysis.py` (not re-run by the button above). Four "
        f"cross-encoders reorder the *same* bi-encoder top {comp['candidates']} for "
        f"{comp['num_questions']} questions. CIs use {comp['bootstrap_samples']:,} bootstrap "
        f"resamples of the questions."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Smallest detectable ΔnDCG@5", f"≈ {bge['mde_80pct_power']:.2f}",
              help="Mean change detectable with 80% power at α = 0.05 for "
                   "bge-reranker-base's spread of per-question changes: 2.8·sd/√n")
    c2.metric("Best possible reordering of the 20", f"{comp['oracle_ndcg@5']:.3f}",
              delta=f"{comp['oracle_ndcg@5'] - bi_ndcg:+.3f} headroom over bi-encoder",
              delta_color="off")
    c3.metric("Questions already at that ceiling",
              f"{comp['questions_already_at_oracle']} / {comp['num_questions']}",
              help="Bi-encoder top 5 is already the best ordering of its 20 candidates; "
                   "a reranker can only make these worse")

    st.dataframe(reranker_comparison_frame(comp), hide_index=True, placeholder="—", column_config={
        **{c: st.column_config.NumberColumn(format="%.3f") for c in ("nDCG@5", "MRR@5", "Recall@5")},
        "ΔnDCG@5": st.column_config.NumberColumn(format="%+.3f"),
    })

    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(delta_ci_chart(comp, bge["mde_80pct_power"]), use_container_width=True)
        st.caption(":orange[●] the app's reranker · :blue[●] alternatives · shaded band: "
                   "changes smaller than ±MDE that 100 questions cannot reliably detect.")
    with right:
        st.markdown(
            "- **100 questions is enough to see a real gain.** Two rerankers improve nDCG@5 by "
            "about +0.06 with p < 0.001 on the same candidates and labels.\n"
            "- **The app's reranker is the weak link.** `bge-reranker-base` helps and hurts about "
            f"equally often ({bge['improved']} vs. {bge['degraded']} questions); its scores match "
            "a raw `transformers` run, so this is the model, not a bug.\n"
            "- **Headroom is limited on this corpus.** The bi-encoder is often already optimal, so "
            "even a good reranker can only add a few points.\n"
            "- Switching rerankers is a one-line change to `RERANKER_MODEL` in `reranker.py`."
        )



# ------------------------------------------------- bi-encoder × reranker ---

NO_RERANKER = "No reranker"


def format_p(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def bi_encoder_frame(grid):
    alone = {g["bi_encoder"]: g for g in grid["grid"] if g["reranker"] == "none"}
    rows = []
    for label, info in grid["bi_encoders"].items():
        vs = alone[label]["vs_reference"]
        rows.append({
            "Bi-encoder": label,
            "Model": info["model"],
            "Params (M)": info["params"] / 1e6,
            "Dim": info["dim"],
            "nDCG@5": alone[label]["metrics"]["ndcg@5"],
            "Δ vs app baseline": vs and vs["delta_ndcg@5"],
            "95% CI": vs and f"[{vs['ci95'][0]:+.3f}, {vs['ci95'][1]:+.3f}]",
            "p": vs and format_p(vs["wilcoxon_p"]),
            "Recall@20": info["recall@20"],
            "Best possible nDCG@5": info["oracle_ndcg@5"],
            "Query ms": info["query_latency_ms"]["median"],
            "Index s": info["corpus_embed_seconds"],
        })
    return pd.DataFrame(rows)


def pipelines_frame(grid):
    rows = []
    for g in grid["grid"]:
        ref, gain = g["vs_reference"], g["vs_same_bi_encoder"]
        rows.append({
            "Bi-encoder": g["bi_encoder"],
            "Reranker": NO_RERANKER if g["reranker"] == "none" else g["reranker"],
            "nDCG@5": g["metrics"]["ndcg@5"],
            "MRR@5": g["metrics"]["mrr@5"],
            "Recall@5": g["metrics"]["recall@5"],
            "Δ vs app baseline": ref and ref["delta_ndcg@5"],
            "95% CI": ref and f"[{ref['ci95'][0]:+.3f}, {ref['ci95'][1]:+.3f}]",
            "p": ref and format_p(ref["wilcoxon_p"]),
            "Gain from reranking": gain and gain["delta_ndcg@5"],
            "Gain p": gain and format_p(gain["wilcoxon_p"]),
        })
    return pd.DataFrame(rows).sort_values("nDCG@5", ascending=False, kind="stable")


def grid_heatmap(grid, view):
    bi_order = list(grid["bi_encoders"])
    rr_order = [NO_RERANKER, *grid["rerankers"]]
    rows = []
    for g in grid["grid"]:
        gain = g["vs_same_bi_encoder"]
        if view == "Gain from reranking":
            if gain is None:
                continue
            value = gain["delta_ndcg@5"]
        else:
            value = g["metrics"][view.lower()]
        rows.append({
            "bi": g["bi_encoder"],
            "rr": NO_RERANKER if g["reranker"] == "none" else g["reranker"],
            "value": value,
            "reference": (g["bi_encoder"], g["reranker"]) == (
                grid["reference"]["bi_encoder"], grid["reference"]["reranker"]),
            "ndcg": g["metrics"]["ndcg@5"],
            "gain": gain["delta_ndcg@5"] if gain else None,
            "gain_p": format_p(gain["wilcoxon_p"]) if gain else "—",
        })
    df = pd.DataFrame(rows)
    if view == "Gain from reranking":
        bound = max(abs(df["value"].min()), abs(df["value"].max()))
        scale = alt.Scale(scheme="redblue", domain=[-bound, 0, bound])
        rr_order = rr_order[1:]
        fmt, title = "+.3f", "nDCG@5 gain from reranking"
    else:
        scale = alt.Scale(scheme="blues", domain=[df["value"].min(), df["value"].max()])
        fmt, title = ".3f", view
    x = alt.X("rr:N", sort=rr_order, title="Reranker", axis=alt.Axis(labelAngle=0, orient="top"))
    y = alt.Y("bi:N", sort=bi_order, title="Bi-encoder (first stage)",
              axis=alt.Axis(labelLimit=200))
    tooltip = [alt.Tooltip("bi:N", title="Bi-encoder"), alt.Tooltip("rr:N", title="Reranker"),
               alt.Tooltip("ndcg:Q", format=".3f", title="nDCG@5"),
               alt.Tooltip("gain:Q", format="+.3f", title="Gain from reranking"),
               alt.Tooltip("gain_p:N", title="Gain p (Wilcoxon)")]
    cells = alt.Chart(df).mark_rect(stroke="white", strokeWidth=2).encode(
        x=x, y=y, color=alt.Color("value:Q", scale=scale, title=title,
                                  legend=alt.Legend(orient="bottom", gradientLength=300)),
        tooltip=tooltip)
    # Dark text on light cells, white text on saturated ones.
    if view == "Gain from reranking":
        dark_text = (alt.datum.value > -0.5 * bound) & (alt.datum.value < 0.5 * bound)
    else:
        dark_text = alt.datum.value < (df["value"].min() + df["value"].max()) / 2
    labels = alt.Chart(df).mark_text(fontSize=13).encode(
        x=x, y=y, text=alt.Text("value:Q", format=fmt), tooltip=tooltip,
        color=alt.condition(dark_text, alt.value("#1a1a19"), alt.value("#ffffff")))
    outline = alt.Chart(df[df["reference"]]).mark_rect(
        fill=None, stroke=RERANKED_COLOR, strokeWidth=3).encode(x=x, y=y)
    return (cells + labels + outline).properties(height=60 * len(bi_order))


def grid_takeaways(grid):
    entries = grid["grid"]
    best = max(entries, key=lambda g: g["metrics"]["ndcg@5"])
    alone = [g for g in entries if g["reranker"] == "none"]
    best_alone = max(alone, key=lambda g: g["metrics"]["ndcg@5"])
    app_pipeline = next(g for g in entries if g["bi_encoder"] == grid["reference"]["bi_encoder"]
                        and g["reranker"] == "bge-reranker-base")
    ref_ndcg = next(g for g in alone if g["bi_encoder"] == grid["reference"]["bi_encoder"])

    def pipe(g):
        rr = "no reranker" if g["reranker"] == "none" else g["reranker"]
        return f"**{g['bi_encoder']} + {rr}**"

    def delta(g):
        vs = g["vs_reference"]
        return (f"{vs['delta_ndcg@5']:+.3f} vs. the app's baseline, 95% CI "
                f"[{vs['ci95'][0]:+.3f}, {vs['ci95'][1]:+.3f}], p {format_p(vs['wilcoxon_p'])}")

    n_tests = len(entries) - 1
    gains = [g for g in entries if g["vs_same_bi_encoder"]]
    by_reranker = {}
    for g in gains:
        by_reranker.setdefault(g["reranker"], []).append(g["vs_same_bi_encoder"]["delta_ndcg@5"])
    gain_lines = ", ".join(f"{rr} {min(v):+.3f} to {max(v):+.3f}" for rr, v in by_reranker.items())
    return "\n".join([
        f"- **Best pipeline:** {pipe(best)} at nDCG@5 {best['metrics']['ndcg@5']:.3f} "
        f"({delta(best)}).",
        f"- **Best bi-encoder alone:** {pipe(best_alone)} at "
        f"{best_alone['metrics']['ndcg@5']:.3f} ({delta(best_alone)}). For comparison, the app's "
        f"two-stage pipeline (bge-small + bge-reranker-base) scores "
        f"{app_pipeline['metrics']['ndcg@5']:.3f} and the app's baseline "
        f"{ref_ndcg['metrics']['ndcg@5']:.3f}.",
        f"- **Gain from reranking depends on the first stage** (range across bi-encoders): "
        f"{gain_lines}. Switch the view to *Gain from reranking* to see it per cell.",
        f"- **Many comparisons:** this page makes {n_tests} comparisons against one baseline. "
        f"At p < 0.05, one or two could look significant by chance; trust consistent patterns "
        f"over single cells.",
    ])


def grid_tab():
    if not GRID_PATH.exists():
        st.info("Run `LAB_DEVICE=mps python bi_encoder_grid.py` to build this comparison.")
        return
    grid = json.loads(GRID_PATH.read_text())
    hw = grid["hardware"]
    st.caption(
        f"Precomputed by `bi_encoder_grid.py` on {grid['created'][:10]} · "
        f"{grid['num_questions']} questions · {hw['device'].upper()} ({hw.get('cpu', '')}). "
        f"Unlike the Benchmark tab, each bi-encoder retrieves its **own** top "
        f"{grid['candidates']}, so a pipeline can win by finding better candidates, by "
        f"ordering them better, or both. All differences are paired against the app's "
        f"baseline: **bge-small, no reranker**."
    )

    st.subheader("1. Each bi-encoder on its own")
    st.caption(f"**Recall@{grid['candidates']}** is the share of relevant documents in the "
               f"bi-encoder's {grid['candidates']} candidates: the most any reranker can work "
               f"with. **Best possible nDCG@5** is a perfect reordering of those candidates. "
               f"Query latency includes embedding the question and searching; index time is "
               f"embedding all {load_metadata()['corpus_size']:,} abstracts.")
    st.dataframe(bi_encoder_frame(grid), hide_index=True, placeholder="—", column_config={
        "Model": st.column_config.TextColumn(width="small"),
        "Params (M)": st.column_config.NumberColumn(format="%.0f"),
        "Best possible nDCG@5": st.column_config.NumberColumn("Best possible", format="%.3f"),
        **{c: st.column_config.NumberColumn(format="%.3f") for c in ("nDCG@5", "Recall@20")},
        "Δ vs app baseline": st.column_config.NumberColumn(format="%+.3f"),
        "Query ms": st.column_config.NumberColumn(format="%.1f"),
        "Index s": st.column_config.NumberColumn(format="%.0f"),
    })

    st.subheader("2. Every bi-encoder × every reranker")
    view = st.radio("Cell value", ["nDCG@5", "MRR@5", "Recall@5", "Gain from reranking"],
                    horizontal=True, key="grid_view")
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(grid_heatmap(grid, view), use_container_width=True)
        st.caption("Rows: first-stage bi-encoder. Columns: reranker applied to that "
                   "bi-encoder's own top 20. :orange[Outlined]: the app's baseline. "
                   "Hover a cell for the reranking gain and its p-value.")
    with right:
        st.markdown(grid_takeaways(grid))
        st.markdown(
            "- **Caveats:** MedCPT was trained on PubMed search clicks, and BioASQ experts "
            "found their documents through PubMed search, which may favour it. Latency was "
            "measured on the device above, not the app's CPU setting."
        )

    st.subheader(f"All {len(grid['grid'])} pipelines")
    st.dataframe(pipelines_frame(grid), hide_index=True, placeholder="—", column_config={
        **{c: st.column_config.NumberColumn(format="%.3f")
           for c in ("nDCG@5", "MRR@5", "Recall@5")},
        "Δ vs app baseline": st.column_config.NumberColumn(format="%+.3f"),
        "Gain from reranking": st.column_config.NumberColumn(
            format="%+.3f", help="nDCG@5 change from adding this reranker to the same bi-encoder"),
    })


# ------------------------------------------------------------ details ---


def details_tab():
    meta = data["metadata"]
    bench = state["benchmark"]
    hw = bench["hardware"] if bench else hardware_info()

    st.subheader("Bi-encoder vs. cross-encoder")
    st.markdown(
        "A **bi-encoder** turns the question and each document into vectors *separately*. "
        "Document vectors are computed once in advance, so searching 2,500 abstracts is a "
        "single matrix multiplication and takes milliseconds — but the model never sees the "
        "question and document together, so it can only judge rough topical similarity.\n\n"
        "A **cross-encoder** reads the question and a document *together* in one transformer "
        "pass and outputs a relevance score. Every word of the question can attend to every "
        "word of the abstract, which usually gives sharper judgments, but nothing can be "
        "precomputed: it runs once per pair. That is why it only reranks the bi-encoder's "
        f"top {CANDIDATES} instead of searching the whole corpus."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Setup")
        st.markdown(f"""
| Setting | Value |
|---|---|
| Dataset | {meta['dataset']} |
| License | {meta['dataset_license']} |
| Evaluation questions | {meta['num_questions']} ({', '.join(f'{v} {k}' for k, v in meta['question_types'].items())}) |
| Corpus size | {meta['corpus_size']:,} PubMed abstracts ({meta['num_positive_docs']} relevant to an evaluation question, {meta['num_distractor_docs']:,} distractors) |
| Known relevant docs per question | {meta['relevant_per_question_mean']:.2f} on average |
| Embedding model | `{EMBEDDING_MODEL}` (384-d, normalized, dot product) |
| Reranker model | `{RERANKER_MODEL}` (max {MAX_LENGTH} tokens per pair) |
| Retrieval candidates | {CANDIDATES} |
| Final results | {TOP_K} |
| Random seed | {meta['seed']} |
""")
    with right:
        st.subheader("Hardware")
        st.markdown(f"""
| Setting | Value |
|---|---|
| Device | **{hw['device'].upper()}** |
| CPU | {hw.get('cpu', '')} ({hw.get('cpu_cores', '?')} cores) |
| PyTorch | {hw['torch']} ({hw['torch_threads']} threads) |
| Python | {hw['python']} |
| Platform | {hw['platform']} |
""")
        st.caption("Set the environment variable `LAB_DEVICE=mps` or `cuda` to use a GPU.")

    st.subheader("Evaluation methodology")
    st.markdown(f"""
- **Questions:** {meta['num_questions']} questions sampled with a fixed seed from the
  {meta['bioasq_questions_total']:,} BioASQ 11b training questions, restricted to questions
  with {meta['selection']}.
- **Ground truth:** the official BioASQ document annotations. A document counts as relevant
  only if BioASQ lists it for that question. No labels were added or inferred.
- **Corpus:** every relevant document of the evaluation questions plus distractors:
  {meta['distractors']}. Title and abstract fetched from PubMed; documents without an
  abstract were dropped ({meta['pubmed_records_without_abstract_or_missing']} records).
- **Text:** title + newline + abstract for both models. The bi-encoder query gets BGE's
  retrieval instruction prefix (`{QUERY_INSTRUCTION.strip()}`).
- **Pipelines:** A returns the bi-encoder's top {TOP_K}; B reorders the *same* top
  {CANDIDATES} by cross-encoder score and returns its top {TOP_K}.
- **Metrics:** Recall@{TOP_K}, MRR@{TOP_K} and nDCG@{TOP_K} with binary relevance,
  averaged over questions. Recall@{CANDIDATES} is checked to be identical for both.
- **Latency:** `time.perf_counter()` around each stage, one question at a time, after one
  discarded warm-up query. Model loading and corpus embedding are excluded.
""")

    st.subheader("Known limitations")
    st.markdown("""
- **Incomplete judgments.** BioASQ experts annotated a limited set of documents per
  question. A highly ranked "not labeled relevant" document may in fact be relevant, so
  both pipelines can be penalized unfairly — and the reranker more so, if it is better at
  surfacing unannotated relevant documents.
- **Small, curated corpus.** All relevant documents are guaranteed to be in a 2,500-document
  corpus. Real retrieval over 36M+ PubMed records is far harder; absolute scores here are
  much higher than they would be in practice.
- **Distractors are random.** Distractors are relevant documents of *other*, randomly sampled
  BioASQ questions. Many are on unrelated topics, which makes the bi-encoder's job easier
  than in real search; some are near-misses because BioASQ revisits topics (e.g. lncRNAs in
  cancer vs. lncRNAs in Alzheimer's disease).
- **Small sample.** With 100 questions, differences of a few points of nDCG are within
  noise; no significance test is applied.
- **Truncation.** The reranker sees at most 512 tokens of question + abstract; the
  bi-encoder sees at most 512 tokens of the document.
- **Latency is machine-specific** and measured on a laptop, one query at a time.
""")


# --------------------------------------------------------------- page ---

st.title("Biomedical Retrieval & Reranking Lab")
st.caption(
    f"Bi-encoder retrieval (`{EMBEDDING_MODEL}`) vs. the same candidates reranked by a "
    f"cross-encoder (`{RERANKER_MODEL}`) on BioASQ questions. Educational experiment over a "
    f"small curated corpus with incomplete relevance labels — not full PubMed search."
)
tab_explorer, tab_bench, tab_grid, tab_details = st.tabs(
    ["Query Explorer", "Benchmark Results", "Bi-encoder × Reranker", "Experiment Details"],
    key="active_tab", on_change="rerun",
)
with tab_explorer:
    explorer_tab()
with tab_bench:
    benchmark_tab()
    reranker_comparison_section()
with tab_grid:
    grid_tab()
with tab_details:
    details_tab()
