from __future__ import annotations

import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("Data Analyst Agent")

with st.sidebar:
    st.header("Upload data")
    up = st.file_uploader("CSV / Excel / PDF / Image", type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg", "webp"])
    if up is not None and st.button("Upload"):
        files = {"file": (up.name, up.getvalue(), up.type)}
        r = requests.post(f"{API}/uploads", files=files, timeout=120)
        if r.status_code == 200:
            st.success("Uploaded")
            st.session_state["dataset_id"] = r.json()["dataset_id"]
            st.session_state["upload_notes"] = r.json().get("notes", [])
        else:
            st.error(r.text)

    st.divider()
    st.header("Datasets")
    r = requests.get(f"{API}/datasets", timeout=30)
    if r.status_code == 200:
        ds = r.json()
        options = [d["dataset_id"] for d in ds]
        chosen = st.selectbox("Active dataset_id", options=options, index=options.index(st.session_state.get("dataset_id")) if st.session_state.get("dataset_id") in options else 0)
        st.session_state["dataset_id"] = chosen

dataset_id = st.session_state.get("dataset_id")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Chat")
    prompt = st.text_area("Ask for analysis", height=120, placeholder="Examples:\n- profile this dataset\n- pivot by region and month\n- sql: SELECT region, SUM(revenue) AS revenue FROM t GROUP BY 1 ORDER BY 2 DESC\n- find outliers")
    if st.button("Run"):
        payload = {"dataset_id": dataset_id, "message": prompt, "top_k": 6}
        r = requests.post(f"{API}/chat", json=payload, timeout=120)
        if r.status_code == 200:
            st.session_state["last_resp"] = r.json()
        else:
            st.error(r.text)

    resp = st.session_state.get("last_resp")
    if resp:
        st.markdown(resp["message"])
        if resp.get("citations"):
            with st.expander("RAG citations"):
                for c in resp["citations"]:
                    st.write(f"{c['source_id']} (score={c['score']:.3f})")
                    st.write(c["text"])

with col2:
    st.subheader("Results")

    resp = st.session_state.get("last_resp")
    if resp:
        for t in resp.get("tables", []):
            st.markdown(f"**{t['title']}**")
            st.dataframe(t["data"], use_container_width=True)

        for ch in resp.get("charts", []):
            st.markdown(f"**{ch.get('title','Chart')}**")
            # Minimal chart render: bar
            if ch.get("type") == "bar":
                import pandas as pd
                df = pd.DataFrame(ch["data"])
                st.bar_chart(df.set_index(ch["x"])[ch["y"]])
            else:
                st.json(ch)
    else:
        st.info("Upload a dataset and run a query to see results.")
