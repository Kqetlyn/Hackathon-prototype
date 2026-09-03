"""Stage 2a — windowed features shared by every model.

One feature table, several models. That is the point: a model comparison is
only meaningful if the models see identical inputs.

Peer normalisation is the piece worth explaining to judges. Raw wear tells you
a door is worn. Wear per cycle compared against the fleet tells you whether it
is worn *faster than its workload explains*, which is a different and more
useful statement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (7, 30, 90)


def rolling_slope(s: pd.Series, w: int = 30) -> pd.Series:
    """Least squares slope over a trailing window, vectorised.

    The obvious `rolling(w).apply(polyfit)` is correct but roughly two orders of
    magnitude slower — it was the whole runtime of the pipeline. For a fixed
    window the slope is a linear filter, so one convolution per asset does it:

        slope = [ S(tx) - S(t)S(x)/w ] / [ S(t^2) - S(t)^2/w ]

    with t = 0..w-1 constant within every window.
    """
    x = s.to_numpy(dtype=float)
    n = len(x)
    if n < w:
        return pd.Series(np.nan, index=s.index)
    filled = np.nan_to_num(x, nan=0.0)
    valid = (~np.isnan(x)).astype(float)

    t = np.arange(w, dtype=float)
    st, stt = t.sum(), (t * t).sum()

    ones = np.ones(w)
    sx = np.convolve(filled, ones, mode="valid")
    cnt = np.convolve(valid, ones, mode="valid")
    stx = np.convolve(filled, t[::-1], mode="valid")

    denom = stt - st * st / w
    slope = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = (stx - st * sx / w) / denom
    vals[cnt < max(3, w // 3)] = np.nan
    slope[w - 1 :] = vals
    return pd.Series(slope, index=s.index)


def build_features(telemetry: pd.DataFrame, channels: dict[str, list[str]]) -> pd.DataFrame:
    frames = []
    for system, g_sys in telemetry.groupby("system", sort=False):
        chans = [c for c in channels[system] if c in g_sys.columns]
        g = g_sys.sort_values(["asset_id", "date"]).copy()
        grp = g.groupby("asset_id", sort=False)

        feat = g[["asset_id", "date", "line", "system"]].copy()
        feat["cum_usage"] = grp["usage"].cumsum()
        feat["usage_7"] = grp["usage"].transform(lambda s: s.rolling(7, min_periods=2).mean())

        for ch in chans:
            for w in WINDOWS:
                r = grp[ch].transform(lambda s, w=w: s.rolling(w, min_periods=max(2, w // 3)).mean())
                feat[f"{ch}_mean_{w}"] = r
                feat[f"{ch}_std_{w}"] = grp[ch].transform(
                    lambda s, w=w: s.rolling(w, min_periods=max(2, w // 3)).std()
                )
            feat[f"{ch}_slope_30"] = grp[ch].transform(lambda s: rolling_slope(s, 30))
            # change relative to the asset's own healthy baseline
            base = grp[ch].transform(lambda s: s.head(60).mean())
            feat[f"{ch}_delta_base"] = feat[f"{ch}_mean_7"] - base
            # wear per unit of use, then z-scored against same-day fleet peers
            per_use = feat[f"{ch}_delta_base"] / feat["cum_usage"].replace(0, np.nan)
            feat[f"{ch}_per_use"] = per_use
            by_day = per_use.groupby(feat["date"])
            feat[f"{ch}_peer_z"] = (per_use - by_day.transform("mean")) / by_day.transform("std").replace(0, np.nan)

        frames.append(feat)

    out = pd.concat(frames, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan)


def label_horizon(features: pd.DataFrame, faults: pd.DataFrame, horizon_days: int = 21) -> pd.Series:
    """1 if a verified fault occurs on this asset within the next N days."""
    y = pd.Series(0, index=features.index, dtype=int)
    if faults is None or len(faults) == 0:
        return y
    f = faults.copy()
    f["date"] = pd.to_datetime(f["date"])
    by_asset = f.groupby("asset_id")["date"].apply(list).to_dict()
    dates = pd.to_datetime(features["date"]).values
    for asset_id, idx in features.groupby("asset_id").groups.items():
        fds = by_asset.get(asset_id)
        if not fds:
            continue
        idx = np.asarray(list(idx))
        d = dates[idx]
        hit = np.zeros(len(idx), dtype=bool)
        for fd in fds:
            delta = (np.datetime64(fd) - d) / np.timedelta64(1, "D")
            hit |= (delta >= 0) & (delta <= horizon_days)
        y.iloc[idx] = hit.astype(int)
    return y
