"""Streamlit dashboard.

    python app.py            starts everything
    streamlit run app.py     also works, if you prefer

Streamlit normally refuses to run under a plain `python app.py` because it needs
its own server. The shim below spots that case and relaunches itself correctly,
so either command works.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "outputs"


def _under_streamlit() -> bool:
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:
        return False


if not _under_streamlit():
    if not (OUT / "decisions.csv").exists():
        print("No outputs yet — running the pipeline first (about 95 seconds).\n")
        subprocess.run([sys.executable, str(HERE / "run_pipeline.py")], check=True)
    print("\nStarting the dashboard. Press Ctrl+C to stop.\n")
    sys.exit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(HERE / "app.py")]
        ).returncode
    )

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rolling stock maintenance decisions", layout="wide")


@st.cache_data
def load():
    need = ["decisions", "model_comparison", "spares", "fleet_demand", "telemetry", "feature_importance"]
    missing = [n for n in need if not (OUT / f"{n}.csv").exists()]
    if missing:
        st.error(f"Missing {missing}. Run `python run_pipeline.py` first.")
        st.stop()
    d = {n: pd.read_csv(OUT / f"{n}.csv") for n in need}
    d["telemetry"]["date"] = pd.to_datetime(d["telemetry"]["date"])
    return d


d = load()
dec, cmp_, sp, dem, tel, imp = (
    d["decisions"], d["model_comparison"], d["spares"], d["fleet_demand"], d["telemetry"], d["feature_importance"]
)

st.title("Rolling stock maintenance decisions")
st.caption("Door and bogie telemetry to a scheduling and preparation decision. Synthetic data.")

c = st.columns(5)
c[0].metric("Assets monitored", len(dec))
c[1].metric("Urgent", int((dec["status"] == "urgent").sum()))
c[2].metric("Low margin", int((dec["status"] == "low margin").sum()))
c[3].metric("Sensor faults excluded", int((dec["status"] == "sensor fault").sum()))
c[4].metric("Obsolete parts", sp.loc[sp["flag"] == "obsolete", "part"].nunique())

tab1, tab2, tab3, tab4 = st.tabs(["Fleet", "Asset detail", "Model benchmark", "Spares"])

with tab1:
    f = st.multiselect("Status", sorted(dec["status"].unique()), default=["urgent", "low margin"])
    view = dec[dec["status"].isin(f)] if f else dec
    st.dataframe(
        view[[
            "asset_id", "line", "system", "status", "remaining_lo", "remaining_hi",
            "agrees", "days_to_pm", "margin_days", "life_used_pct", "duty_vs_peers",
            "sensor_verdict",
        ]],
        use_container_width=True, hide_index=True, height=460,
    )
    st.caption(
        "Remaining life shows as a single figure when the condition and reliability "
        "estimates agree, and as a range when they do not. Urgency is judged on the "
        "shorter end. Sorted so anything close to its window surfaces first."
    )

with tab2:
    aid = st.selectbox("Asset", dec["asset_id"].tolist())
    r = dec[dec["asset_id"] == aid].iloc[0]
    a, b = st.columns([1, 2])
    with a:
        rng = f"{int(r.remaining_lo)} days" if r.agrees else f"{int(r.remaining_lo)}–{int(r.remaining_hi)} days"
        st.metric("Remaining life", rng, help="Range width is the confidence")
        st.metric("Status", r.status)
        st.write(f"Condition estimate **{int(r.condition_days)} d**")
        st.write(f"Reliability estimate **{int(r.reliability_days)} d**")
        st.write(f"Life used **{int(r.life_used_pct)}%**")
        st.write(f"Duty vs peers **{int(r.duty_vs_peers):+d}%**")
        st.write(f"Next PM in **{int(r.days_to_pm)} d**, margin **{int(r.margin_days):+d} d**")
        if r.sensor_verdict != "ok":
            st.warning(f"Signal validation: {r.sensor_verdict} on {r.failed_channel}. Excluded from scoring.")
    with b:
        g = tel[tel["asset_id"] == aid].set_index("date")
        ch = "door_close_time_s" if r.system == "door" else "bogie_axlebox_temp_rise_c"
        cols = [c for c in g.columns if c.startswith(r.system) and g[c].notna().any()]
        st.line_chart(g[[ch]] if ch in g else g[cols[:1]])
        st.caption(f"{ch} · alarm threshold {3.30 if r.system == 'door' else 25.0}")
    st.subheader("Spares to prepare")
    s = sp[sp["asset_id"] == aid]
    st.dataframe(s[["part", "probability", "lead_time_days", "flag"]], hide_index=True, use_container_width=True)

with tab3:
    st.dataframe(cmp_, hide_index=True, use_container_width=True)
    st.caption(
        "Identical features, identical time-based split. Alerting on the top 5% of "
        "asset-days. PR-AUC rather than ROC-AUC because faults are rare, and lift "
        "over the base rate because PR-AUC alone means nothing without it."
    )
    st.subheader("What the model is using")
    st.dataframe(imp, hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Expected fleet demand, next 90 days")
    st.dataframe(dem, hide_index=True, use_container_width=True)
    st.subheader("Risk flags")
    st.dataframe(
        sp[sp["flag"] != "ok"][["asset_id", "part", "probability", "lead_time_days", "remaining_lo", "flag"]]
        .head(200), hide_index=True, use_container_width=True,
    )
    st.caption(
        "`late` means the order lead time is longer than the asset has left. "
        "`obsolete` means the OEM no longer supports the component, which LTA's "
        "Rail Reliability Taskforce named as a live concern in February 2026."
    )
