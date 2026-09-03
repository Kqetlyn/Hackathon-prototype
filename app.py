"""Streamlit dashboard.

    python app.py            starts everything
    streamlit run app.py     also works, if you prefer

Streamlit normally refuses to run under a plain `python app.py` because it needs
its own server. The shim below spots that case and relaunches itself correctly,
so either command works.

Design rule for this file: lead with the decision, put the maths underneath.
A maintenance engineer opening this at 6am wants to know what to do tonight,
not what the model's PR-AUC was. Every internal column name gets a human label
before it reaches the screen.
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

st.set_page_config(page_title="Fleet maintenance", page_icon="🚈", layout="wide")

INDICATOR = {"door": "door closing time", "bogie": "axle box temperature"}
STATUS_ICON = {
    "urgent": "🔴", "low margin": "🟠", "not urgent": "🟢", "sensor fault": "⚪",
}


@st.cache_data
def load():
    need = ["decisions", "model_comparison", "spares", "fleet_demand", "telemetry", "feature_importance"]
    missing = [n for n in need if not (OUT / f"{n}.csv").exists()]
    if missing:
        st.error(f"Missing {missing}. Run `python run_pipeline.py` first.")
        st.stop()
    d = {n: pd.read_csv(OUT / f"{n}.csv") for n in need}
    d["telemetry"]["date"] = pd.to_datetime(d["telemetry"]["date"])
    d["decisions"]["next_pm"] = pd.to_datetime(d["decisions"]["next_pm"])
    return d


def life_text(r) -> str:
    if r["status"] == "sensor fault":
        return "not scored"
    if r.get("beyond_horizon", False):
        return "over a year"
    if r["agrees"]:
        return f"{int(r['remaining_lo'])} days"
    return f"{int(r['remaining_lo'])}–{int(r['remaining_hi'])} days"


def action_text(r) -> str:
    when = r["next_pm"].strftime("%d %b")
    if r["status"] == "sensor fault":
        return "Check the sensor, not the equipment"
    if r["status"] == "urgent":
        return f"Bring forward — won't last to {when}"
    if r["status"] == "low margin":
        return f"Do at {when}, only {int(r['margin_days'])} days spare"
    return f"Fold into scheduled work on {when}"


def why_text(r) -> str:
    """Plain English reason, built from the numbers behind the decision."""
    if r["status"] == "sensor fault":
        return f"{r['sensor_verdict'].replace('_', ' ').capitalize()} reading on {r['failed_channel']}"

    duty = int(r["duty_vs_peers"])
    duty_note = (
        f", and it works {duty}% harder than similar assets" if duty >= 15
        else f", despite working {abs(duty)}% less than similar assets" if duty <= -15
        else ""
    )
    if r["condition_days"] < r["reliability_days"]:
        return f"Its {INDICATOR[r['system']]} is trending toward the limit{duty_note}"
    return (
        f"It has used {int(r['life_used_pct'])}% of typical component life "
        f"({int(r['usage_since_replacement']):,} {r['usage_unit']}){duty_note}"
    )


d = load()
dec, cmp_, sp, dem, tel, imp = (
    d["decisions"], d["model_comparison"], d["spares"], d["fleet_demand"],
    d["telemetry"], d["feature_importance"],
)

n_urgent = int((dec["status"] == "urgent").sum())
n_low = int((dec["status"] == "low margin").sum())
n_sensor = int((dec["status"] == "sensor fault").sum())

st.title("Fleet maintenance")
if n_urgent:
    st.markdown(
        f"### {n_urgent} assets won't last until their next scheduled maintenance."
        f"  \nAnother {n_low} will only just make it. {n_sensor} alerts were sensor "
        f"faults and have been kept off the list."
    )
else:
    st.markdown(
        f"### Nothing needs bringing forward.  \n{n_low} assets are close to their "
        f"margin. {n_sensor} alerts were sensor faults."
    )
st.caption(
    f"{len(dec)} assets monitored across doors and bogies · synthetic data · "
    f"recommendations are advisory and need engineer sign-off"
)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["What needs doing", "Asset detail", "Spares", "How the models did", "More information"]
)

# ─────────────────────────────── what needs doing ───────────────────────────────
with tab1:
    only_action = st.toggle("Show only what needs attention", value=True)
    view = dec[dec["status"].isin(["urgent", "low margin"])] if only_action else dec

    if view.empty:
        st.success("Nothing outstanding. Everything fits its scheduled window.")
    else:
        table = pd.DataFrame({
            "": view["status"].map(STATUS_ICON),
            "Asset": view["asset_id"],
            "Part": view["system"].str.capitalize(),
            "Time left": view.apply(life_text, axis=1),
            "What to do": view.apply(action_text, axis=1),
            "Why": view.apply(why_text, axis=1),
        })
        st.dataframe(table, hide_index=True, use_container_width=True, height=440)

    st.caption(
        "🔴 won't last to its next maintenance · 🟠 will, but with little to spare · "
        "🟢 fine · ⚪ sensor problem, not an equipment problem"
    )
    with st.expander("How 'time left' is worked out"):
        st.markdown(
            "Two independent estimates per asset. One follows the asset's own "
            "condition and projects when it crosses the maintenance limit. The other "
            "compares how much use it has had since replacement against how long that "
            "component type normally lasts, counted in door cycles or kilometres "
            "rather than calendar days.\n\n"
            "**A single figure means both agree. A range means they don't, and the "
            "width of the range is how confident to be.** Urgency is always judged on "
            "the shorter end, so uncertainty counts against the asset."
        )

# ───────────────────────────────── asset detail ─────────────────────────────────
with tab2:
    pick = dec.copy()
    pick["label"] = pick["status"].map(STATUS_ICON) + "  " + pick["asset_id"]
    aid = st.selectbox("Asset", pick["label"].tolist()).split("  ", 1)[1]
    r = dec[dec["asset_id"] == aid].iloc[0]

    left, right = st.columns([1, 1.6])
    with left:
        st.metric("Time left", life_text(r), help="A range means the two estimates disagree")
        st.markdown(f"**{action_text(r)}**")
        st.write(why_text(r) + ".")

        if r["status"] == "sensor fault":
            st.warning(
                f"The {r['failed_channel']} channel is unreliable, so this asset was "
                "kept out of scoring. Raise a sensor job — replacing the equipment "
                "would waste an engineering slot."
            )

        with st.expander("The numbers behind it"):
            st.write(f"Condition estimate · **{int(r['condition_days'])} days**")
            st.write(f"Usage estimate · **{int(r['reliability_days'])} days**")
            st.write(f"Life used · **{int(r['life_used_pct'])}%**")
            st.write(f"Duty vs similar assets · **{int(r['duty_vs_peers']):+d}%**")
            st.write(f"Next scheduled maintenance · **{r['next_pm'].strftime('%d %b')}** "
                     f"({int(r['days_to_pm'])} days)")
            if pd.notna(r["risk"]):
                st.write(f"Model risk score · **{r['risk']:.2f}**")

    with right:
        ch = "door_close_time_s" if r["system"] == "door" else "bogie_axlebox_temp_rise_c"
        limit = 3.30 if r["system"] == "door" else 25.0
        g = tel[tel["asset_id"] == aid].set_index("date")
        if ch in g.columns and g[ch].notna().any():
            chart = g[[ch]].rename(columns={ch: INDICATOR[r["system"]].capitalize()})
            chart["Maintenance limit"] = limit
            st.line_chart(chart)
            st.caption(
                f"Last 180 days. The flat line is the limit; the estimate is where the "
                f"trend meets it."
            )

    st.subheader("Spares to have ready")
    s = sp[sp["asset_id"] == aid]
    if s.empty:
        st.caption("No parts history for this fault type.")
    else:
        flag_label = {
            "ok": "In stock", "late": "⚠️ Arrives too late", "obsolete": "⚠️ No longer made",
        }
        st.dataframe(
            pd.DataFrame({
                "Part": s["part"],
                "Chance it's needed": (s["probability"] * 100).round().astype(int).astype(str) + "%",
                "Delivery": s["lead_time_days"].astype(int).astype(str) + " days",
                "Status": s["flag"].map(flag_label),
            }),
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "Based on which parts were actually issued against similar faults before."
        )

# ─────────────────────────────────── spares ────────────────────────────────────
with tab3:
    st.subheader("What to order, next 90 days")
    st.dataframe(
        pd.DataFrame({
            "Part": dem["part"],
            "Expected needed": dem["expected_units"].round(1),
            "Delivery": dem["lead_time_days"].astype(int).astype(str) + " days",
            "Still made?": dem["oem_supported"].map({True: "Yes", False: "⚠️ No"}),
        }),
        hide_index=True, use_container_width=True,
    )

    risky = sp[sp["flag"] != "ok"]
    if not risky.empty:
        st.subheader("Parts that won't arrive in time")
        st.dataframe(
            pd.DataFrame({
                "Asset": risky["asset_id"],
                "Part": risky["part"],
                "Delivery": risky["lead_time_days"].astype(int).astype(str) + " days",
                "Asset has": risky["remaining_lo"].astype(int).astype(str) + " days",
                "Problem": risky["flag"].map(
                    {"late": "Arrives too late", "obsolete": "No longer made by supplier"}
                ),
            }).head(150),
            hide_index=True, use_container_width=True,
        )
    st.caption(
        "LTA's Rail Reliability Taskforce named spares forecasting and parts "
        "discontinued by the manufacturer as a live workstream in February 2026."
    )

# ─────────────────────────────────── models ────────────────────────────────────
with tab4:
    best = cmp_.iloc[0]
    st.markdown(
        f"**{best['model']}** performed best, catching "
        f"{best['recall_at_5pct']:.0%} of faults with about "
        f"{best['median_lead_days']:.0f} days of warning, when engineers inspect the "
        f"top 5% of assets each night. Roughly "
        f"{best['precision_at_5pct']:.0%} of those alerts were real."
    )
    st.dataframe(
        pd.DataFrame({
            "Model": cmp_["model"],
            "Type": cmp_["kind"],
            "Quality (PR-AUC)": cmp_["pr_auc"],
            "vs guessing": cmp_["lift_over_base"].astype(str) + "×",
            "Alerts that were real": (cmp_["precision_at_5pct"] * 100).round().astype(int).astype(str) + "%",
            "Faults caught": (cmp_["recall_at_5pct"] * 100).round().astype(int).astype(str) + "%",
            "False alarms /1k days": cmp_["false_alarms_per_1k_days"],
            "Warning (days)": cmp_["median_lead_days"],
        }),
        hide_index=True, use_container_width=True,
    )
    with st.expander("Why these models, and why these numbers look modest"):
        st.markdown(
            "Every model sees identical features and the same time-based split, so "
            "the comparison is fair. Tree models handle the fault types we have "
            "labels for; the isolation forest needs no labels at all, which is how "
            "you catch failure modes nobody has seen before.\n\n"
            "**Catching under half of faults is what honest predictive maintenance "
            "looks like** at a realistic inspection budget. A model claiming 95% "
            "recall on data like this usually has a leak. See `docs/MODELS.md`."
        )
    st.subheader("What the model watches most")
    st.dataframe(
        pd.DataFrame({"Signal": imp["feature"], "Weight": imp["importance"]}).head(10),
        hide_index=True, use_container_width=True,
    )

# ──────────────────────────────── more information ─────────────────────────────
with tab5:
    st.subheader("What we set out to do")
    st.markdown(
        "Predictive maintenance usually stops at an anomaly score. The harder "
        "question is **when to act on it**, since the railway only gets a few hours "
        "a night to do any work.\n\n"
        "This tool takes door and bogie telemetry and returns, for each asset, how "
        "long it has left, whether it can wait for the next scheduled maintenance, "
        "and which spares to have ready."
    )

    st.subheader("Where it sits alongside what LTA already has")
    st.dataframe(
        pd.DataFrame({
            "System": [
                "SMRT–NTU door sensors", "Bogie monitoring",
                "PDSS (Bentley AssetWise)", "REAMS (Siemens + ST Eng)", "This tool",
            ],
            "Covers": [
                "Air pressure, movement speed, power on pneumatic doors",
                "Wheel wear, axle load, temperature, vibration",
                "Permanent way — track infrastructure",
                "Enterprise asset management",
                "Turning door and bogie signals into a scheduling decision",
            ],
            "Status": [
                "Trialled 2018, rolling out", "Being fitted",
                "NSEWL, extending to Circle Line", "Downtown Line first", "This prototype",
            ],
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "The sensing layer already exists. PDSS covers track, not rolling stock. "
        "The gap is the layer that turns rolling stock signals into a decision about "
        "which night to act and what to prepare."
    )

    st.subheader("How it works, end to end")
    st.markdown(
        "**1 · Validate** — is the reading trustworthy? Missing, frozen, drifting, "
        "out of range, spiking, or channels that have decoupled from each other. "
        "Rules only, no model, so every check is explainable in one sentence. "
        "Failing assets are raised as sensor jobs, not maintenance jobs.\n\n"
        "**2 · Detect** — several models on identical features with a strict "
        "time-based split. Tree ensembles for labelled fault types, isolation forest "
        "for behaviour nobody has labelled yet.\n\n"
        "**3 · Estimate remaining life** — two independent answers. The asset's own "
        "condition trend projected to its limit, and its accumulated use against "
        "typical component life. Measured in door cycles and kilometres, not calendar "
        "days, which is also how LTA reports reliability (MKBF).\n\n"
        "**4 · Decide** — does it run out before its next scheduled maintenance? "
        "Judged on the shorter estimate, so uncertainty counts against the asset.\n\n"
        "**5 · Prepare** — which spares were actually issued against similar faults "
        "before, flagging anything that arrives too late or is no longer manufactured."
    )

    st.subheader("Data")
    st.markdown(
        f"**Currently synthetic.** {len(dec)} assets across five lines, "
        f"{len(tel):,} readings shown, generated by `src/generate_data.py`. Line codes "
        "and rolling stock designations are real (C151B, C651, C830, C751A, C951A) and "
        "the monitored parameters match what the deployed sensors actually measure. "
        "All figures, thresholds, part names and lead times are invented.\n\n"
        "**Swapping in the real dataset** is a one-file change. Everything downstream "
        "reads the same four tables, and channel names are mapped in "
        "`config/signals.yaml` rather than hardcoded — because LTA noted in February "
        "2026 that monitoring systems across the network were installed at different "
        "times by different manufacturers and measure different parameters."
    )

    st.subheader("What we would say to a judge before they ask")
    st.markdown(
        "- The numbers here prove the pipeline runs, not that it works on your fleet.\n"
        "- The maintenance schedule and parts history are stand-ins. The comparison "
        "logic is the part that matters.\n"
        "- Survival analysis with censoring (Weibull) is the better remaining-life "
        "model. It needs event histories we would not have until day one, so we used "
        "threshold-crossing extrapolation, which gives the same output shape with far "
        "less that can go wrong.\n"
        "- Nothing is scheduled automatically. Every recommendation needs engineer "
        "sign-off, and the decision is logged — which also builds the feedback record "
        "needed to retrain."
    )
    st.caption(
        "NEBULA X · Problem statement 3, predictive fault detection · "
        "LTA Rail Digitalisation & Guild, 18–20 September 2026"
    )
