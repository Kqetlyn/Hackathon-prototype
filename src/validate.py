"""Stage 1 — signal validation.

Answers "can this reading be trusted", not "is this asset faulty". Assets whose
primary channel fails validation are pulled out of scoring entirely and raised
as sensor jobs instead, so engineering hours are not spent chasing a dead
transducer.

Rules only. No model. This is deliberate: the checks must be explainable to a
maintenance engineer in one sentence each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Physically plausible operating envelopes, from the maintenance manual.
LIMITS = {
    "door_close_time_s": (1.5, 8.0),
    "door_motor_current_a": (1.0, 12.0),
    "door_air_pressure_bar": (4.0, 7.5),
    "bogie_axlebox_temp_rise_c": (0.0, 80.0),
    "bogie_vibration_rms_g": (0.0, 5.0),
    "bogie_axle_load_t": (2.0, 12.0),
}

FROZEN_DAYS = 10
DRIFT_WINDOW = 30
DRIFT_SIGMA = 4.0
MISSING_FRAC = 0.15


def _check_one(s: pd.Series, name: str, usage: pd.Series) -> str | None:
    tail = s.tail(60)
    if tail.isna().mean() > MISSING_FRAC:
        return "missing"

    clean = tail.dropna()
    if len(clean) < FROZEN_DAYS:
        return "missing"

    # frozen: the channel stopped moving while the train kept running
    if clean.tail(FROZEN_DAYS).std() < 1e-9 and usage.tail(FROZEN_DAYS).mean() > 0:
        return "frozen"

    lo, hi = LIMITS.get(name, (-np.inf, np.inf))
    if (clean < lo).mean() > 0.05 or (clean > hi).mean() > 0.05:
        return "out_of_range"

    # spike: isolated values far outside the channel's own recent spread
    med, mad = clean.median(), (clean - clean.median()).abs().median()
    if mad > 0 and ((clean - med).abs() > 8 * mad).mean() > 0.02:
        return "spike"

    # drift: step change in level that usage does not account for
    if len(clean) >= 2 * DRIFT_WINDOW:
        a, b = clean.iloc[-2 * DRIFT_WINDOW : -DRIFT_WINDOW], clean.iloc[-DRIFT_WINDOW:]
        ua = usage.iloc[-2 * DRIFT_WINDOW : -DRIFT_WINDOW].mean()
        ub = usage.iloc[-DRIFT_WINDOW:].mean()
        usage_change = abs(ub - ua) / max(ua, 1e-9)
        if a.std() > 0 and abs(b.mean() - a.mean()) > DRIFT_SIGMA * a.std() and usage_change < 0.15:
            return "drift"
    return None


def validate(telemetry: pd.DataFrame, channels: dict[str, list[str]]) -> pd.DataFrame:
    """Return one row per asset with a verdict and the channel that failed."""
    out = []
    for asset_id, g in telemetry.groupby("asset_id", sort=False):
        g = g.sort_values("date")
        system = g["system"].iloc[0]
        verdict, bad_channel = "ok", ""
        for ch in channels[system]:
            if ch not in g.columns:
                continue
            r = _check_one(g[ch], ch, g["usage"])
            if r is not None:
                verdict, bad_channel = r, ch
                break

        # cross-channel agreement: on a healthy door, close time and motor
        # current move together. If they decouple, one of them is lying.
        if verdict == "ok" and system == "door":
            w = g.tail(90)[["door_close_time_s", "door_motor_current_a"]].dropna()
            if len(w) > 40 and w.std().min() > 1e-9:
                if abs(w.corr().iloc[0, 1]) < 0.15:
                    verdict, bad_channel = "cross_channel", "door_motor_current_a"

        out.append(
            dict(
                asset_id=asset_id,
                system=system,
                sensor_verdict=verdict,
                failed_channel=bad_channel,
                trusted=verdict == "ok",
            )
        )
    return pd.DataFrame(out)
