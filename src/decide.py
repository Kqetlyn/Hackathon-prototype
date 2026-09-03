"""Stage 4 — urgency.

One comparison: does the asset run out before its next scheduled maintenance?

  urgent      likely to fail first, needs an earlier slot
  not urgent  folds into the scheduled window

Judged on the shorter end of the remaining life range, so uncertainty counts
against the asset rather than for it. Margin in days is carried through and the
list is sorted by it, so a job sitting two days clear of its window surfaces
instead of disappearing into the safe pile.

Assets that failed signal validation never reach this stage. They are raised as
sensor jobs, which is a different queue and a much cheaper one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Preventive maintenance intervals. Real values come from the maintenance
# regime; these stand in so the demo has a calendar to compare against.
PM_INTERVAL_DAYS = {"door": 90, "bogie": 120}


def next_pm_dates(assets: pd.DataFrame, as_of: pd.Timestamp, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    offs = {
        a.asset_id: int(rng.integers(3, PM_INTERVAL_DAYS.get(a.system, 90)))
        for a in assets.itertuples()
    }
    return pd.Series({k: as_of + pd.Timedelta(days=v) for k, v in offs.items()})


def decide(life: pd.DataFrame, sensors: pd.DataFrame, pm: pd.Series, as_of: pd.Timestamp) -> pd.DataFrame:
    df = life.merge(sensors[["asset_id", "sensor_verdict", "failed_channel", "trusted"]], on="asset_id", how="left")
    df["next_pm"] = df["asset_id"].map(pm)
    df["days_to_pm"] = (pd.to_datetime(df["next_pm"]) - as_of).dt.days

    # decide on the pessimistic end
    df["margin_days"] = df["remaining_lo"] - df["days_to_pm"]
    df["urgent"] = df["margin_days"] < 0

    def status(r):
        if not r["trusted"]:
            return "sensor fault"
        if r["urgent"]:
            return "urgent"
        if r["margin_days"] <= 7:
            return "low margin"
        return "not urgent"

    df["status"] = df.apply(status, axis=1)
    df.loc[~df["trusted"].astype(bool), ["urgent"]] = False

    order = {"urgent": 0, "low margin": 1, "not urgent": 2, "sensor fault": 3}
    df["_o"] = df["status"].map(order)
    df = df.sort_values(["_o", "margin_days", "remaining_lo"]).drop(columns="_o").reset_index(drop=True)
    return df
