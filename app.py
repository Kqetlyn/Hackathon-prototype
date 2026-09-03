"""Fleet maintenance dashboard.

    python app.py            starts everything
    streamlit run app.py     also works

Streamlit will not run under a plain `python app.py` because it needs its own
server, so the block below detects that case and relaunches correctly.

Presentation rule: internal column names never reach the screen, and the
recommended action comes before the numbers that produced it.
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
        print("No outputs found. Running the pipeline first, about 95 seconds.\n")
        subprocess.run([sys.executable, str(HERE / "run_pipeline.py")], check=True)
    print("\nStarting dashboard. Ctrl+C to stop.\n")
    sys.exit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(HERE / "app.py")]
        ).returncode
    )

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fleet maintenance", layout="wide")

INDICATOR = {"door": "Door closing time", "bogie": "Axle box temperature"}
ALARM_LIMIT = {"door": 3.30, "bogie": 25.0}
PRIMARY_CHANNEL = {
    "door": "door_close_time_s",
    "bogie": "bogie_axlebox_temp_rise_c",
}
STATUS_LABEL = {
    "urgent": "Urgent",
    "low margin": "Low margin",
    "not urgent": "Scheduled",
    "sensor fault": "Sensor fault",
}


@st.cache_data
def load():
    need = [
        "decisions", "model_comparison", "spares",
        "fleet_demand", "telemetry", "feature_importance",
    ]
    missing = [n for n in need if not (OUT / f"{n}.csv").exists()]
    if missing:
        st.error(f"Missing {missing}. Run `python run_pipeline.py` first.")
        st.stop()
    d = {n: pd.read_csv(OUT / f"{n}.csv") for n in need}
    d["telemetry"]["date"] = pd.to_datetime(d["telemetry"]["date"])
    d["decisions"]["next_pm"] = pd.to_datetime(d["decisions"]["next_pm"])
    return d


def remaining(r) -> str:
    if r["status"] == "sensor fault":
        return "n/a"
    if r.get("beyond_horizon", False):
        return "> 12 months"
    if r["agrees"]:
        return f"{int(r['remaining_lo'])} d"
    return f"{int(r['remaining_lo'])}-{int(r['remaining_hi'])} d"


def action(r) -> str:
    when = r["next_pm"].strftime("%d %b")
    if r["status"] == "sensor fault":
        return "Raise sensor job"
    if r["status"] == "urgent":
        return f"Advance, will not reach {when}"
    if r["status"] == "low margin":
        return f"Hold for {when}, {int(r['margin_days'])} d slack"
    return f"Include in PM on {when}"


def basis(r) -> str:
    """Why the tool reached that call, in the engineer's terms."""
    if r["status"] == "sensor fault":
        verdict = r["sensor_verdict"].replace("_", " ")
        return f"{verdict.capitalize()} signal on {r['failed_channel']}"

    duty = int(r["duty_vs_peers"])
    load = (
        f", duty {duty:+d}% vs peers" if abs(duty) >= 15 else ""
    )
    if r["condition_days"] < r["reliability_days"]:
        return f"{INDICATOR[r['system']]} trending to limit{load}"
    return (
        f"{int(r['life_used_pct'])}% of typical life used "
        f"({int(r['usage_since_replacement']):,} {r['usage_unit']}){load}"
    )


d = load()
dec = d["decisions"]
cmp_ = d["model_comparison"]
sp = d["spares"]
dem = d["fleet_demand"]
tel = d["telemetry"]
imp = d["feature_importance"]

counts = dec["status"].value_counts()
n_urgent = int(counts.get("urgent", 0))
n_low = int(counts.get("low margin", 0))
n_sensor = int(counts.get("sensor fault", 0))
n_ok = int(counts.get("not urgent", 0))

st.title("Fleet maintenance")
st.caption(
    f"{len(dec)} assets monitored, doors and bogies, five lines. "
    "Synthetic data. Recommendations require engineer sign-off."
)

m = st.columns(4)
m[0].metric("Urgent", n_urgent, help="Will not last until the next scheduled PM")
m[1].metric("Low margin", n_low, help="Will reach PM with 7 days or less to spare")
m[2].metric("Scheduled", n_ok)
m[3].metric("Sensor faults", n_sensor, help="Excluded from scoring, raised separately")
st.divider()

tab_work, tab_asset, tab_spares, tab_models, tab_about = st.tabs(
    ["Work list", "Asset detail", "Spares", "Model performance", "About"]
)

# Work list
with tab_work:
    action_only = st.toggle("Action required only", value=True)
    view = dec[dec["status"].isin(["urgent", "low margin"])] if action_only else dec

    if view.empty:
        st.info("No assets require action ahead of their scheduled maintenance.")
    else:
        table = pd.DataFrame({
            "Asset": view["asset_id"].values,
            "Type": view["system"].str.capitalize().values,
            "Status": view["status"].map(STATUS_LABEL).values,
            "Est. remaining": view.apply(remaining, axis=1).values,
            "Next PM": view["next_pm"].dt.strftime("%d %b").values,
            "Recommended action": view.apply(action, axis=1).values,
            "Basis": view.apply(basis, axis=1).values,
        })
        # Plain frame rather than a pandas Styler: Styler needs jinja2 >= 3.1.2,
        # and a version mismatch would blank the table at exactly the wrong moment.
        st.dataframe(table, hide_index=True, use_container_width=True, height=440)

    st.caption(
        "Sorted by urgency, then by remaining slack, so assets close to their "
        "window appear before comfortable ones."
    )
    with st.expander("How estimated remaining life is derived"):
        st.markdown(
            "Two independent estimates per asset. The first follows the asset's own "
            "condition and projects when it crosses its alarm limit. The second "
            "compares accumulated use since replacement against typical life for "
            "that component type, counted in door cycles or kilometres rather than "
            "calendar days, consistent with MKBF reporting.\n\n"
            "A single figure means both estimates agree. A range means they differ, "
            "and the width of the range indicates confidence. Urgency is judged on "
            "the shorter estimate."
        )

# Asset detail
with tab_asset:
    aid = st.selectbox("Asset", dec["asset_id"].tolist())
    r = dec[dec["asset_id"] == aid].iloc[0]

    left, right = st.columns([1, 1.6])
    with left:
        st.metric("Estimated remaining life", remaining(r))
        st.markdown(f"**{action(r)}**")
        st.write(basis(r))

        if r["status"] == "sensor fault":
            st.warning(
                f"The {r['failed_channel']} channel failed validation "
                f"({r['sensor_verdict'].replace('_', ' ')}), so this asset is "
                "excluded from scoring. Raise a sensor job rather than an "
                "equipment job."
            )

        detail = pd.DataFrame({
            "Field": [
                "Condition estimate", "Usage estimate", "Life used",
                "Duty vs peers", "Next PM", "Slack", "Model risk score",
            ],
            "Value": [
                f"{int(r['condition_days'])} d",
                f"{int(r['reliability_days'])} d",
                f"{int(r['life_used_pct'])}%",
                f"{int(r['duty_vs_peers']):+d}%",
                r["next_pm"].strftime("%d %b %Y"),
                f"{int(r['margin_days']):+d} d",
                f"{r['risk']:.2f}" if pd.notna(r["risk"]) else "n/a",
            ],
        })
        st.dataframe(detail, hide_index=True, use_container_width=True)

    with right:
        ch = PRIMARY_CHANNEL[r["system"]]
        g = tel[tel["asset_id"] == aid].set_index("date")
        if ch in g.columns and g[ch].notna().any():
            chart = g[[ch]].rename(columns={ch: INDICATOR[r["system"]]})
            chart["Alarm limit"] = ALARM_LIMIT[r["system"]]
            st.line_chart(chart)
            st.caption(
                f"{INDICATOR[r['system']]}, last 180 days. Alarm limit "
                f"{ALARM_LIMIT[r['system']]}. The condition estimate is where the "
                "fitted trend meets that limit."
            )

    st.subheader("Spares required")
    s = sp[sp["asset_id"] == aid]
    if s.empty:
        st.caption("No parts history for this fault type.")
    else:
        st.dataframe(
            pd.DataFrame({
                "Part": s["part"].values,
                "Probability": (s["probability"] * 100).round().astype(int).astype(str).values + "%",
                "Lead time": s["lead_time_days"].astype(int).astype(str).values + " d",
                "Note": s["flag"].map({
                    "ok": "", "late": "Lead time exceeds remaining life",
                    "obsolete": "No OEM support",
                }).values,
            }),
            hide_index=True, use_container_width=True,
        )
        st.caption("Derived from parts issued against comparable faults in work order history.")

# Spares
with tab_spares:
    st.subheader("Forecast demand, next 90 days")
    st.dataframe(
        pd.DataFrame({
            "Part": dem["part"].values,
            "Expected units": dem["expected_units"].round(1).values,
            "Lead time": dem["lead_time_days"].astype(int).astype(str).values + " d",
            "OEM support": dem["oem_supported"].map({True: "Active", False: "Withdrawn"}).values,
        }),
        hide_index=True, use_container_width=True,
    )

    risky = sp[sp["flag"] != "ok"]
    if not risky.empty:
        st.subheader("Procurement risks")
        st.dataframe(
            pd.DataFrame({
                "Asset": risky["asset_id"].values,
                "Part": risky["part"].values,
                "Lead time": risky["lead_time_days"].astype(int).astype(str).values + " d",
                "Remaining life": risky["remaining_lo"].astype(int).astype(str).values + " d",
                "Risk": risky["flag"].map({
                    "late": "Lead time exceeds remaining life",
                    "obsolete": "No OEM support",
                }).values,
            }).head(150),
            hide_index=True, use_container_width=True,
        )
    st.caption(
        "LTA's Rail Reliability Taskforce identified spares forecasting and "
        "OEM-discontinued parts as a workstream in February 2026."
    )

# Model performance
with tab_models:
    best = cmp_.iloc[0]
    st.markdown(
        f"Best performing model: **{best['model']}**. At an inspection budget of "
        f"the top 5% of assets per night it detects {best['recall_at_5pct']:.0%} of "
        f"faults with a median {best['median_lead_days']:.0f} days warning, and "
        f"{best['precision_at_5pct']:.0%} of its alerts are genuine."
    )
    st.dataframe(
        pd.DataFrame({
            "Model": cmp_["model"].values,
            "Type": cmp_["kind"].values,
            "PR-AUC": cmp_["pr_auc"].values,
            "Lift": cmp_["lift_over_base"].astype(str).values + "x",
            "Precision": (cmp_["precision_at_5pct"] * 100).round().astype(int).astype(str).values + "%",
            "Recall": (cmp_["recall_at_5pct"] * 100).round().astype(int).astype(str).values + "%",
            "False alarms / 1k asset-days": cmp_["false_alarms_per_1k_days"].values,
            "Lead time (d)": cmp_["median_lead_days"].values,
        }),
        hide_index=True, use_container_width=True,
    )
    with st.expander("Method and interpretation"):
        st.markdown(
            "All models are trained on identical features with a time-based split, "
            "so the comparison is like for like. A random split would leak later "
            "data into training and inflate every figure.\n\n"
            "Tree ensembles handle the fault types with labels. The isolation "
            "forest is unsupervised and requires none, which is how previously "
            "unseen failure modes get caught.\n\n"
            "PR-AUC is used rather than ROC-AUC because faults are rare, and lift "
            "over the base rate is shown alongside it since PR-AUC alone is not "
            "interpretable. Detection of under half of faults at this inspection "
            "budget is consistent with published results in this domain. Figures "
            "materially above that on data of this size usually indicate leakage. "
            "See docs/MODELS.md."
        )
    st.subheader("Highest weighted signals")
    st.dataframe(
        pd.DataFrame({
            "Signal": imp["feature"].values,
            "Weight": imp["importance"].values,
        }).head(10),
        hide_index=True, use_container_width=True,
    )

# About
with tab_about:
    st.subheader("Purpose")
    st.markdown(
        "Predictive maintenance normally stops at an anomaly score. The operational "
        "question is when to act on it, given the limited engineering hours "
        "available each night.\n\n"
        "This tool takes door and bogie telemetry and returns, per asset, estimated "
        "remaining life, whether the work can wait for the next scheduled "
        "maintenance, and the spares to prepare."
    )

    st.subheader("Position relative to existing systems")
    st.dataframe(
        pd.DataFrame({
            "System": [
                "SMRT-NTU door sensors", "Bogie monitoring",
                "PDSS (Bentley AssetWise)", "REAMS (Siemens, ST Engineering)",
                "This tool",
            ],
            "Scope": [
                "Air pressure, movement speed, power on pneumatic doors",
                "Wheel wear, axle load, temperature, vibration",
                "Permanent way, track infrastructure",
                "Enterprise asset management",
                "Rolling stock signals to scheduling and preparation decision",
            ],
            "Status": [
                "Trialled 2018, progressive rollout", "Being fitted",
                "NSEWL, extending to Circle Line", "Downtown Line first",
                "Prototype",
            ],
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "The sensing layer is largely in place. PDSS covers track rather than "
        "rolling stock, so the gap addressed here is the decision layer above "
        "door and bogie monitoring."
    )

    st.subheader("Pipeline")
    st.markdown(
        "1. **Validation.** Missing, frozen, drifting, out of range, spiking, or "
        "channels that have decoupled from one another. Rule based, no model, so "
        "each check is explainable in one sentence. Failing assets are raised as "
        "sensor jobs rather than equipment jobs.\n"
        "2. **Detection.** Several models on identical features with a time-based "
        "split. Tree ensembles for labelled fault types, isolation forest for "
        "unlabelled behaviour.\n"
        "3. **Remaining life.** Condition trend projected to the alarm limit, and "
        "accumulated use against typical component life. Measured in door cycles "
        "and kilometres rather than calendar days, consistent with MKBF.\n"
        "4. **Scheduling.** Compared against the next planned maintenance window "
        "and judged on the shorter estimate, so uncertainty counts against the "
        "asset.\n"
        "5. **Preparation.** Spares drawn from work order history, flagging parts "
        "whose lead time exceeds remaining life or that are no longer supported."
    )

    st.subheader("Data")
    st.markdown(
        f"Currently synthetic. {len(dec)} assets across five lines, generated by "
        "`src/generate_data.py`. Line codes and rolling stock designations are real "
        "(C151B, C651, C830, C751A, C951A) and the monitored parameters match those "
        "the deployed sensors measure. All figures, thresholds, part names and lead "
        "times are illustrative.\n\n"
        "Substituting the real dataset is a single-file change. Downstream stages "
        "read the same four tables, and channel names are mapped in "
        "`config/signals.yaml` rather than hardcoded, since monitoring systems "
        "across the network were installed at different times by different "
        "manufacturers and measure different parameters."
    )

    st.subheader("Limitations")
    st.markdown(
        "- Results demonstrate that the pipeline runs, not that it performs on the "
        "operational fleet.\n"
        "- Maintenance intervals and parts history are stand-ins. The comparison "
        "logic is the transferable part.\n"
        "- Survival analysis with right-censoring is the stronger remaining life "
        "model. It requires event histories not available at build time, so "
        "threshold-crossing extrapolation is used instead, giving the same output "
        "with fewer failure modes.\n"
        "- Batch processing on daily aggregates, not streaming.\n"
        "- No recommendation is actioned automatically. Sign-off is required and "
        "logged, which also builds the feedback record needed for retraining."
    )
    st.caption(
        "NEBULA X, problem statement 3, predictive fault detection. "
        "LTA Rail Digitalisation and Guild, 18-20 September 2026."
    )
