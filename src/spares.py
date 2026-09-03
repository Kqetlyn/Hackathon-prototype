"""Stage 5 — spares to prepare.

Not predicted from telemetry. Learned from work order history: for past faults
of this type, which parts were actually issued, and how often. That gives a
ranked list with a likelihood attached rather than a flat guess, which is what
you need to decide whether to pull stock.

Two flags matter operationally:
  late  order lead time exceeds the asset's remaining life
  obsolete  the component is no longer supported by the OEM

The second one is there because LTA's Rail Reliability Taskforce named spares
forecasting and OEM discontinuation as a live workstream in February 2026.

Fleet-level demand is the same history read the other way: expected failures
per period times parts per failure, which is a stocking decision rather than
an per-asset alert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def part_probabilities(work_orders: pd.DataFrame) -> pd.DataFrame:
    """P(part issued | fault type), from history."""
    if work_orders is None or len(work_orders) == 0:
        return pd.DataFrame(columns=["fault_type", "part", "probability", "lead_time_days", "oem_supported"])
    jobs = work_orders.groupby("fault_type")["date"].nunique().rename("jobs")
    hits = (
        work_orders.groupby(["fault_type", "part"])
        .agg(
            times=("date", "nunique"),
            lead_time_days=("lead_time_days", "median"),
            oem_supported=("oem_supported", "min"),
        )
        .reset_index()
        .merge(jobs, on="fault_type")
    )
    hits["probability"] = (hits["times"] / hits["jobs"]).clip(0, 1).round(2)
    return hits.sort_values(["fault_type", "probability"], ascending=[True, False]).reset_index(drop=True)


DOOR_PREFIXES = ("door_",)
BOGIE_PREFIXES = ("axlebox_", "bearing_", "suspension_")


def system_fault_mix(work_orders: pd.DataFrame, system: str) -> pd.Series:
    """P(fault type | system), from history.

    Picking one modal fault type per system, which is what this did first, is
    too blunt: every asset of a system got an identical parts list and rarer
    fault types never surfaced at all — including the ones that consume the
    obsolete components. Marginalising over the mix fixes that.
    """
    if work_orders is None or len(work_orders) == 0:
        return pd.Series(dtype=float)
    pref = DOOR_PREFIXES if system == "door" else BOGIE_PREFIXES
    m = work_orders["fault_type"].astype(str).str.startswith(pref)
    jobs = work_orders.loc[m].drop_duplicates(["asset_id", "date"])["fault_type"]
    return jobs.value_counts(normalize=True) if len(jobs) else pd.Series(dtype=float)


def recommend(decisions: pd.DataFrame, work_orders: pd.DataFrame, min_prob: float = 0.12) -> pd.DataFrame:
    probs = part_probabilities(work_orders)
    if probs.empty:
        return pd.DataFrame(columns=["asset_id", "status", "part", "probability", "lead_time_days", "remaining_lo", "flag"])

    # marginal P(part | system) = sum over fault types of P(fault) * P(part | fault)
    marginals: dict[str, pd.DataFrame] = {}
    for system in decisions["system"].unique():
        mix = system_fault_mix(work_orders, system)
        if mix.empty:
            continue
        sub = probs[probs["fault_type"].isin(mix.index)].copy()
        sub["weighted"] = sub["probability"] * sub["fault_type"].map(mix)
        agg = (
            sub.groupby("part")
            .agg(
                probability=("weighted", "sum"),
                lead_time_days=("lead_time_days", "median"),
                oem_supported=("oem_supported", "min"),
                top_fault=("probability", "idxmax"),
            )
            .reset_index()
        )
        agg["top_fault"] = probs.loc[agg["top_fault"], "fault_type"].to_numpy()
        agg["probability"] = agg["probability"].round(2)
        marginals[system] = agg[agg["probability"] >= min_prob].sort_values("probability", ascending=False)

    rows = []
    for r in decisions.itertuples():
        cand = marginals.get(r.system)
        if cand is None:
            continue
        for c in cand.itertuples():
            late = bool(c.lead_time_days > r.remaining_lo)
            rows.append(
                dict(
                    asset_id=r.asset_id,
                    status=r.status,
                    fault_type=c.top_fault,
                    part=c.part,
                    probability=c.probability,
                    lead_time_days=int(c.lead_time_days),
                    remaining_lo=int(r.remaining_lo),
                    flag=("obsolete" if not c.oem_supported else ("late" if late else "ok")),
                )
            )
    return pd.DataFrame(rows)


def fleet_demand(decisions: pd.DataFrame, work_orders: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
    """Expected parts consumption across the fleet over the horizon."""
    probs = part_probabilities(work_orders)
    if probs.empty or decisions.empty:
        return pd.DataFrame(columns=["part", "expected_units", "lead_time_days", "oem_supported"])
    due = decisions[decisions["remaining_lo"] <= horizon_days]
    counts: dict[str, float] = {}
    mixes = {s: system_fault_mix(work_orders, s) for s in decisions["system"].unique()}
    for r in due.itertuples():
        mix = mixes.get(r.system)
        if mix is None or mix.empty:
            continue
        for c in probs[probs["fault_type"].isin(mix.index)].itertuples():
            counts[c.part] = counts.get(c.part, 0.0) + float(c.probability) * float(mix[c.fault_type])
    meta = probs.groupby("part").agg(
        lead_time_days=("lead_time_days", "median"), oem_supported=("oem_supported", "min")
    )
    out = (
        pd.DataFrame({"part": list(counts), "expected_units": [round(v, 1) for v in counts.values()]})
        .merge(meta, on="part", how="left")
        .sort_values("expected_units", ascending=False)
        .reset_index(drop=True)
    )
    return out
