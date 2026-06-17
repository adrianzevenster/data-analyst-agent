from __future__ import annotations

import pandas as pd

from app.analytics.anomalies import anomaly_scan
from app.analytics.quality import data_quality_report
from app.analytics.relationships import correlation_analysis
from app.analytics.trends import detect_datetime_col, trend_analysis

MAX_CHARTS_PER_SECTION = 2
MAX_TOTAL_CHARTS = 8
MAX_ANOMALY_NUMERIC_COLS = 8


def auto_insights(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Orchestrate quality, relationship, anomaly, and trend analyses into one
    ranked set of findings, so a user doesn't need to know which specific
    tool to ask for to get broad automated coverage of a dataset.
    """
    sections: dict[str, dict] = {}
    findings: list[str] = []
    charts: list[dict] = []

    quality = data_quality_report(df)
    sections["data_quality"] = quality
    findings.extend(quality.get("findings", []))
    charts.extend(quality.get("charts", [])[:MAX_CHARTS_PER_SECTION])

    numeric_cols = [
        str(c) for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]

    if len(numeric_cols) >= 2:
        relationships = correlation_analysis(df)
        sections["relationships"] = relationships
        findings.extend(relationships.get("findings", []))
        charts.extend(relationships.get("charts", [])[:MAX_CHARTS_PER_SECTION])

    if len(numeric_cols) >= 2:
        anomalies = anomaly_scan(df, numeric_cols=numeric_cols[:MAX_ANOMALY_NUMERIC_COLS])
        sections["anomalies"] = anomalies
        n_anomalies = anomalies.get("n_anomalies", 0)
        if n_anomalies:
            findings.append(
                f"{n_anomalies} anomalous row(s) detected ({anomalies.get('anomaly_rate_pct', 0):.1f}% of scanned rows)."
            )

    date_col = detect_datetime_col(df)
    if date_col:
        trend = trend_analysis(df, date_col=date_col)
        sections["trend"] = trend
        if "error" not in trend:
            findings.append(trend["engineering_readout"])
            charts.extend(trend.get("charts", [])[:MAX_CHARTS_PER_SECTION])

    findings = findings[:top_n]
    charts = charts[:MAX_TOTAL_CHARTS]

    if findings:
        readout = f"Automated analysis surfaced {len(findings)} key finding(s): " + " ".join(findings[:5])
        if len(findings) > 5:
            readout += f" (+{len(findings) - 5} more below.)"
    else:
        readout = "Automated analysis did not surface any notable findings."

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "insights": [{"rank": i + 1, "finding": finding} for i, finding in enumerate(findings)],
        "charts": charts,
        "analyses_run": list(sections.keys()),
        "engineering_readout": readout,
    }
