from __future__ import annotations

import json
import os
import requests
import pandas as pd
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8080")
CHAT_TIMEOUT_SECONDS = int(os.getenv("CHAT_TIMEOUT_SECONDS", "300"))

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


def render_chart(ch: dict) -> None:
    st.markdown(f"### {ch.get('title', 'Chart')}")

    data = ch.get("data", [])
    if not data:
        st.info("No data available for this chart.")
        return

    chart_type = ch.get("type")
    chart_df = pd.DataFrame(data)

    if chart_type == "bar":
        y_series = ch.get("y_series") or [ch.get("y")]
        st.bar_chart(chart_df.set_index(ch["x"])[y_series])

    elif chart_type == "histogram":
        st.bar_chart(chart_df.set_index("bin_label")["count"])
        st.caption(f"{ch.get('x_label', ch.get('column', ''))} — {len(data)} bins")

    elif chart_type == "line":
        st.line_chart(chart_df.set_index(ch["x"])[ch["y"]])

    elif chart_type == "scatter":
        st.scatter_chart(chart_df, x=ch["x"], y=ch["y"])
        if ch.get("correlation") is not None:
            st.caption(f"Correlation: {ch['correlation']:.3f}")

    else:
        st.json(ch)


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


def render_train_summary(resp: dict) -> None:
    tool_results = resp.get("tool_results", [])
    train_result = next(
        (
            tr.get("result")
            for tr in tool_results
            if tr.get("name") == "train_supervised_model" and tr.get("ok")
        ),
        None,
    )

    if not isinstance(train_result, dict) or "error" in train_result:
        return

    st.subheader("Model Training Summary")

    readout = train_result.get("engineering_readout")
    if readout:
        st.success(readout)

    model_id = train_result.get("model_id")
    if model_id:
        st.code(model_id, language=None)
        st.caption('Model ID — reuse it in a follow-up request, e.g. "score with model <id>".')

        trained_ids = st.session_state.setdefault("trained_model_ids", [])
        if model_id not in trained_ids:
            trained_ids.append(model_id)

    evaluation = train_result.get("evaluation", {})
    task_type = train_result.get("task_type")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Task", task_type or "N/A")
    c2.metric("Train rows", f"{train_result.get('n_rows_train', 0):,}")
    c3.metric("Test rows", f"{train_result.get('n_rows_test', 0):,}")

    if task_type == "classification":
        accuracy = evaluation.get("accuracy")
        c4.metric("Accuracy", f"{accuracy:.4f}" if accuracy is not None else "N/A")
    else:
        wmape = evaluation.get("wmape")
        c4.metric("WMAPE", f"{wmape:.4f}" if wmape is not None else "N/A")

    feature_importance = train_result.get("feature_importance")
    if feature_importance:
        st.caption("Top features")
        st.dataframe(pd.DataFrame(feature_importance), use_container_width=True, hide_index=True)

    best_params = train_result.get("best_params")
    if best_params:
        st.caption("Best hyperparameters (tuned)")
        st.json(best_params)

    cv = train_result.get("cv")
    if cv:
        mean_display = abs(cv["mean"]) if cv["mean"] < 0 else cv["mean"]
        st.caption(f"CV {cv['folds']}-fold {cv['scoring'].replace('neg_', '')}: {mean_display:.4f} ± {cv['std']:.4f}")

    imbalance_ratio = train_result.get("imbalance_ratio")
    if imbalance_ratio is not None and imbalance_ratio > 5:
        st.warning(f"Class imbalance detected (majority/minority ratio: {imbalance_ratio:.1f}×). `class_weight='balanced'` applied where supported.")

    with st.expander("Full training payload"):
        st.json(train_result)


def render_score_summary(resp: dict) -> None:
    tool_results = resp.get("tool_results", [])
    score_result = next(
        (
            tr.get("result")
            for tr in tool_results
            if tr.get("name") == "score_with_model" and tr.get("ok")
        ),
        None,
    )

    if not isinstance(score_result, dict) or "error" in score_result:
        return

    st.subheader("Model Scoring Summary")

    readout = score_result.get("engineering_readout")
    if readout:
        st.success(readout)

    c1, c2 = st.columns(2)
    c1.metric("Rows scored", f"{score_result.get('n_rows_scored', 0):,}")
    c2.metric("Task", score_result.get("task_type") or "N/A")

    with st.expander("Full scoring payload"):
        st.json(score_result)


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

            with st.expander("Preview data"):
                try:
                    prev = requests.get(f"{API}/datasets/{chosen}/sample", params={"limit": 50}, timeout=15)
                    if prev.status_code == 200:
                        rows = prev.json().get("data", [])
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        else:
                            st.caption("No rows returned.")
                    else:
                        st.caption(f"Preview unavailable ({prev.status_code})")
                except requests.RequestException as e:
                    st.caption(f"Preview unavailable: {e}")

            if len(options) > 1:
                other_names = [d.get("filename", d["dataset_id"][:8]) for d in ds if d["dataset_id"] != chosen]
                st.caption(f"SQL tables: **t** (active) + {', '.join(other_names)}")
        else:
            st.info("No datasets uploaded yet.")

    trained_ids = st.session_state.get("trained_model_ids", [])
    if trained_ids:
        st.divider()
        st.header("Trained this session")
        for mid in reversed(trained_ids):
            st.code(mid, language=None)

    st.divider()
    with st.expander("Model registry"):
        try:
            mr = requests.get(f"{API}/models", timeout=10)
        except requests.RequestException as e:
            mr = None
            st.caption(f"Unavailable: {e}")

        if mr is not None and mr.status_code == 200:
            models_list = mr.json()
            if not models_list:
                st.caption("No models trained yet.")
            else:
                rows = [
                    {
                        "id": m["model_id"][:8] + "…",
                        "type": m["model_type"],
                        "task": m["task_type"],
                        "target": m["target_col"],
                        "created": m["created_at"][:10],
                        "_full_id": m["model_id"],
                    }
                    for m in models_list
                ]
                st.dataframe(
                    pd.DataFrame(rows).drop(columns=["_full_id"]),
                    use_container_width=True,
                    hide_index=True,
                )
                selected_model = st.selectbox(
                    "Copy model ID",
                    options=[r["_full_id"] for r in rows],
                    format_func=lambda x: x[:8] + "…",
                )
                if selected_model:
                    st.code(selected_model, language=None)

    if st.session_state.get("conversation_id"):
        st.divider()
        st.caption(f"Conversation: {st.session_state['conversation_id']}")

    st.divider()
    with st.expander("LLM health"):
        try:
            stats_resp = requests.get(f"{API}/health/llm", timeout=10)
        except requests.RequestException as e:
            stats_resp = None
            st.caption(f"Unavailable: {e}")

        if stats_resp is not None and stats_resp.status_code == 200:
            stats = stats_resp.json()
            if stats["window_size"] == 0:
                st.caption("No LLM calls recorded yet.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
                c2.metric("Error rate", f"{stats['error_rate'] * 100:.1f}%")
                c3.metric("Tokens sampled", f"{stats['total_tokens_sampled']:,}")
                st.caption(f"Window: last {stats['window_size']} calls")
                for op, bucket in stats.get("by_operation", {}).items():
                    st.write(f"**{op}**: {bucket['count']} calls, {bucket['errors']} errors, avg {bucket['avg_latency_ms']:.0f} ms")

    with st.expander("LLM judge (sampled groundedness)"):
        try:
            judge_resp = requests.get(f"{API}/health/llm-judge", timeout=10)
        except requests.RequestException as e:
            judge_resp = None
            st.caption(f"Unavailable: {e}")

        if judge_resp is not None and judge_resp.status_code == 200:
            judge_stats = judge_resp.json()
            if judge_stats["sampled_count"] == 0:
                st.caption("No responses judged yet (sampled at LLM_JUDGE_SAMPLE_RATE).")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg score", f"{judge_stats['avg_groundedness_score']:.1f}/5")
                c2.metric("Low-score rate", f"{judge_stats['low_score_rate'] * 100:.1f}%")
                c3.metric("Flagged rate", f"{judge_stats['flagged_rate'] * 100:.1f}%")
                st.caption(f"Sampled {judge_stats['sampled_count']} responses")

    with st.expander("LLM repair loop"):
        try:
            repair_resp = requests.get(f"{API}/health/llm-repair", timeout=10)
        except requests.RequestException as e:
            repair_resp = None
            st.caption(f"Unavailable: {e}")

        if repair_resp is not None and repair_resp.status_code == 200:
            repair_stats = repair_resp.json()
            if repair_stats["repair_attempts"] == 0:
                st.caption("No repair attempts yet.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Attempts", repair_stats["repair_attempts"])
                c2.metric("Fix rate", f"{repair_stats['fix_rate'] * 100:.0f}%")
                c3.metric("Dropped", repair_stats["total_dropped"])
                st.caption(
                    f"{repair_stats['total_fixed']} of {repair_stats['total_problems']} invalid calls repaired"
                )

    try:
        rag_eval_resp = requests.get(f"{API}/health/rag-eval", timeout=10)
        rag_eval_available = (
            rag_eval_resp is not None
            and rag_eval_resp.status_code == 200
            and rag_eval_resp.json().get("available")
        )
    except requests.RequestException:
        rag_eval_resp = None
        rag_eval_available = False

    if rag_eval_available:
        with st.expander("RAG retrieval eval (recall@k / precision@k)"):
            rag_eval = rag_eval_resp.json()
            st.caption(f"{rag_eval['n_queries']} labeled golden queries")
            for k, stats_at_k in sorted(rag_eval["aggregate"].items(), key=lambda kv: int(kv[0])):
                c1, c2 = st.columns(2)
                c1.metric(f"Recall@{k}", f"{stats_at_k['recall_at_k'] * 100:.0f}%")
                c2.metric(f"Precision@{k}", f"{stats_at_k['precision_at_k'] * 100:.0f}%")

dataset_id = st.session_state.get("dataset_id")

left, right = st.columns([0.9, 1.1])

with left:
    st.subheader("Chat")

    conversation_id = st.session_state.get("conversation_id")
    if conversation_id:
        try:
            hr = requests.get(f"{API}/chat/{conversation_id}/history", timeout=30)
            turns = hr.json().get("turns", []) if hr.status_code == 200 else []
        except requests.RequestException:
            turns = []
        for turn in turns:
            avatar = "U" if turn["role"] == "user" else "A"
            with st.chat_message(turn["role"], avatar=avatar):
                st.markdown(turn["content"])
    else:
        st.caption("No messages yet in this conversation.")

    prompt = st.text_area(
        "Ask for analysis",
        height=140,
        placeholder=(
            "Examples:\n"
            "- evaluate model performance using churn probability\n"
            "- profile this dataset\n"
            "- find missing values\n"
            "- sql: SELECT * FROM t LIMIT 10\n"
            "- train a model to predict churn\n"
            "- score with model <model_id>"
        ),
    )

    run_col, clear_col = st.columns([1, 1])
    run_clicked = run_col.button("Run", type="primary")
    if clear_col.button("New conversation"):
        st.session_state.pop("conversation_id", None)
        st.session_state.pop("last_resp", None)
        st.rerun()

    if run_clicked and prompt.strip():
        payload = {
            "dataset_id": dataset_id,
            "message": prompt,
            "top_k": 6,
            "conversation_id": conversation_id,
        }

        status_placeholder = st.empty()
        tool_lines: list[str] = []

        try:
            with requests.post(
                f"{API}/chat/stream", json=payload, stream=True, timeout=CHAT_TIMEOUT_SECONDS
            ) as r:
                if r.status_code != 200:
                    st.error(r.text)
                else:
                    for raw_line in r.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])

                        if event["type"] == "plan":
                            names = [tc["name"] for tc in event.get("tool_calls", [])]
                            status_placeholder.caption(f"Planning: {', '.join(names) or 'no tools'}")
                            st.session_state["conversation_id"] = event["conversation_id"]

                        elif event["type"] == "tool_result":
                            icon = "✓" if event["ok"] else "✗"
                            suffix = f": {event['error']}" if event.get("error") else ""
                            tool_lines.append(f"{icon} {event['name']}{suffix}")
                            status_placeholder.caption("  \n".join(tool_lines))

                        elif event["type"] == "error":
                            status_placeholder.empty()
                            st.error(event["detail"])
                            break

                        elif event["type"] == "done":
                            status_placeholder.empty()
                            st.session_state["last_resp"] = event["response"]
                            st.rerun()
        except requests.RequestException as e:
            status_placeholder.empty()
            st.error(f"Request failed: {e}")

    resp = st.session_state.get("last_resp")

    if resp:
        if resp.get("dataset_id") and resp.get("dataset_id") != dataset_id:
            st.caption(f"Using dataset from earlier in this conversation: {resp['dataset_id']}")

        if resp.get("llm_enabled"):
            if resp.get("synthesis_source") == "llm":
                st.caption(f"Latest reply: LLM-synthesized (planning: {resp.get('planning_source', 'rules')})")
            else:
                st.caption("Latest reply: rule-based — LLM enabled but unavailable")
                if resp.get("llm_error"):
                    st.caption(f"LLM error: {resp['llm_error']}")
        else:
            st.caption("Latest reply: rule-based — LLM reasoning is disabled (set LLM_ENABLED=true)")

        if resp.get("groundedness_score") is not None:
            st.caption(f"Groundedness (sampled judge): {resp['groundedness_score']}/5")
            if resp.get("groundedness_issues"):
                with st.expander("Unsupported claims flagged by judge"):
                    for issue in resp["groundedness_issues"]:
                        st.write(issue)

        if resp.get("llm_notes"):
            with st.expander(f"LLM planning notes ({len(resp['llm_notes'])})"):
                for note in resp["llm_notes"]:
                    st.write(note)

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
        render_train_summary(resp)
        render_score_summary(resp)

        for t in resp.get("tables", []):
            st.markdown(f"### {t['title']}")
            render_metric_table(t)
            data = t.get("data", [])
            if data:
                st.download_button(
                    label="Download CSV",
                    data=pd.DataFrame(data).to_csv(index=False),
                    file_name=f"{t['title'].replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"dl_{t['title']}_{id(t)}",
                )

        for ch in resp.get("charts", []):
            render_chart(ch)
    else:
        st.info("Upload a dataset and run a query to see results.")
