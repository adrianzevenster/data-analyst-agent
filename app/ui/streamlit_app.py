from __future__ import annotations

import os
import requests
import pandas as pd
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("Data Analyst Agent")


def render_metric_table(table: dict) -> None:
    data = table.get("data", [])
    if not data:
        return

    df = pd.DataFrame(data)

    if {"metric", "value"}.issubset(df.columns):
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True)


def render_ml_summary(resp: dict) -> None:
    tool_results = resp.get("tool_results", [])
    ml_result = next(
        (
            tr.get("result")
            for tr in tool_results
            if tr.get("name") == "evaluate_ml_predictions" and tr.get("ok")
        ),
        None,
    )

    if not isinstance(ml_result, dict):
        return

    st.subheader("Model Evaluation Summary")

    readout = ml_result.get("engineering_readout")
    if readout:
        st.success(readout)

    evaluation = ml_result.get("evaluation", {})
    score_summary = evaluation.get("score_summary", {})
    confidence_bands = evaluation.get("confidence_bands", {})

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{evaluation.get('n_rows_scored', ml_result.get('n_rows', 0)):,}")
    c2.metric("Mean score", f"{score_summary.get('mean', 0):.4f}" if score_summary else "N/A")
    c3.metric("P95 score", f"{score_summary.get('p95', 0):.4f}" if score_summary else "N/A")
    c4.metric(
        "High confidence",
        f"{confidence_bands.get('high_confidence_0_80_plus', 0):,}"
        if confidence_bands
        else "N/A",
    )

    with st.expander("Full ML evaluation payload"):
        st.json(ml_result)


with st.sidebar:
    st.header("Upload data")
    up = st.file_uploader(
        "CSV / Excel / PDF / Image",
        type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg", "webp"],
    )

    if up is not None and st.button("Upload"):
        files = {"file": (up.name, up.getvalue(), up.type)}
        r = requests.post(f"{API}/uploads", files=files, timeout=120)

        if r.status_code == 200:
            body = r.json()
            st.success("Uploaded")
            st.session_state["dataset_id"] = body["dataset_id"]
            st.session_state["upload_notes"] = body.get("notes", [])
        else:
            st.error(r.text)

    st.divider()
    st.header("Datasets")

    r = requests.get(f"{API}/datasets", timeout=30)
    if r.status_code == 200:
        ds = r.json()
        options = [d["dataset_id"] for d in ds]

        if options:
            selected_index = (
                options.index(st.session_state.get("dataset_id"))
                if st.session_state.get("dataset_id") in options
                else 0
            )

            chosen = st.selectbox("Active dataset_id", options=options, index=selected_index)
            st.session_state["dataset_id"] = chosen

            selected_meta = next((d for d in ds if d["dataset_id"] == chosen), None)
            if selected_meta:
                st.caption(selected_meta.get("filename", ""))
                st.caption(f"{selected_meta.get('n_rows', 0):,} rows × {selected_meta.get('n_cols', 0):,} cols")
        else:
            st.info("No datasets uploaded yet.")

dataset_id = st.session_state.get("dataset_id")

left, right = st.columns([0.9, 1.1])

with left:
    st.subheader("Chat")

    prompt = st.text_area(
        "Ask for analysis",
        height=140,
        placeholder=(
            "Examples:\n"
            "- evaluate model performance using churn probability\n"
            "- profile this dataset\n"
            "- find missing values\n"
            "- sql: SELECT * FROM t LIMIT 10"
        ),
    )

    if st.button("Run", type="primary"):
        payload = {"dataset_id": dataset_id, "message": prompt, "top_k": 6}
        r = requests.post(f"{API}/chat", json=payload, timeout=120)

        if r.status_code == 200:
            st.session_state["last_resp"] = r.json()
        else:
            st.error(r.text)

    resp = st.session_state.get("last_resp")

    if resp:
        st.markdown(resp["message"])

        if resp.get("tool_calls"):
            with st.expander("Planned tool calls", expanded=True):
                for call in resp["tool_calls"]:
                    st.code(call, language="json")

        if resp.get("citations"):
            with st.expander("RAG citations"):
                for c in resp["citations"]:
                    st.write(f"{c['source_id']} (score={c['score']:.3f})")
                    st.write(c["text"])

with right:
    st.subheader("Results")

    resp = st.session_state.get("last_resp")

    if resp:
        render_ml_summary(resp)

        for t in resp.get("tables", []):
            st.markdown(f"### {t['title']}")
            render_metric_table(t)

        for ch in resp.get("charts", []):
            st.markdown(f"### {ch.get('title', 'Chart')}")

            if ch.get("type") == "bar":
                chart_df = pd.DataFrame(ch["data"])
                st.bar_chart(chart_df.set_index(ch["x"])[ch["y"]])
            else:
                st.json(ch)
    else:
        st.info("Upload a dataset and run a query to see results.")