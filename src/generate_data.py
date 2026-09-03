"""Synthetic rolling stock telemetry for development.

Replace this module with the real NEBULA X dataset loader on hackathon day.
Everything downstream reads the three dataframes this produces, so swapping
the source is a one-file change.

Shapes produced:
  telemetry    one row per asset per service day
  faults       one row per confirmed fault event
  work_orders  one row per part issued against a fault
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260918)

LINES = ["NSL", "EWL", "CCL", "NEL", "DTL"]
STOCK = {"NSL": "C151B", "EWL": "C651", "CCL": "C830", "NEL": "C751A", "DTL": "C951A"}

# Parts consumed per fault type: (name, P(issued | this fault), lead days, OEM supported).
# Different faults consume different parts, which is what makes the work order
# history worth learning from rather than just listing every part in the bin.
PARTS_BY_FAULT = {
    "door_close_slow": [
        ("Door operator actuator", 0.72, 26, True),
        ("Guide roller set", 0.44, 9, True),
        ("Drive belt", 0.21, 12, True),
    ],
    "door_obstruction": [
        ("Obstruction sensor", 0.68, 5, True),
        ("Guide roller set", 0.33, 9, True),
    ],
    "door_motor_overload": [
        ("Door motor assembly", 0.64, 41, False),
        ("Drive belt", 0.48, 12, True),
        ("Door operator actuator", 0.19, 26, True),
    ],
    "axlebox_overheat": [
        ("Axle box bearing", 0.79, 34, True),
        ("Grease seal kit", 0.55, 18, True),
        ("Temperature probe", 0.16, 7, True),
    ],
    "bearing_wear": [
        ("Axle box bearing", 0.86, 34, True),
        ("Grease seal kit", 0.30, 18, True),
    ],
    "suspension_degradation": [
        ("Suspension spring", 0.61, 22, True),
        ("Damper unit", 0.52, 15, False),
    ],
}

# Characteristic life of the component type, in the unit that drives its wear.
# Doors wear per cycle, bogies per kilometre. Learned from work order history
# in production; fixed here so the generator has something to sample around.
NOMINAL_LIFE = {"door": 1_200_000.0, "bogie": 620_000.0}

# Physical alarm limits. In production these come from the maintenance manual.
THRESHOLDS = {
    "door_close_time_s": 3.30,
    "door_motor_current_a": 5.20,
    "bogie_axlebox_temp_rise_c": 25.0,
    "bogie_vibration_rms_g": 0.85,
}


def _assets(n_doors: int, n_bogies: int) -> pd.DataFrame:
    rows = []
    for i in range(n_doors):
        line = LINES[i % len(LINES)]
        rows.append(
            dict(
                asset_id=f"{line}/{STOCK[line]}-{i // 24 + 1:03d}/C{i % 6 + 1}/D{i % 4 + 1}",
                line=line,
                system="door",
                # duty varies a lot by door position and line loading
                duty=float(np.clip(RNG.normal(1.0, 0.22), 0.55, 1.8)),
                installed_days_ago=int(RNG.integers(120, 1400)),
            )
        )
    for i in range(n_bogies):
        line = LINES[i % len(LINES)]
        rows.append(
            dict(
                asset_id=f"{line}/{STOCK[line]}-{i // 8 + 1:03d}/B{i % 4 + 1}",
                line=line,
                system="bogie",
                duty=float(np.clip(RNG.normal(1.0, 0.15), 0.7, 1.5)),
                installed_days_ago=int(RNG.integers(200, 1600)),
            )
        )
    return pd.DataFrame(rows)


def _inject_sensor_fault(series: np.ndarray, kind: str, start: int) -> np.ndarray:
    s = series.copy()
    if kind == "frozen":
        s[start:] = s[start]
    elif kind == "missing":
        s[start : start + 40] = np.nan
    elif kind == "drift":
        s[start:] = s[start:] + np.linspace(0, 0.9 * np.nanstd(series) * 6, len(s) - start)
    elif kind == "spike":
        idx = RNG.choice(np.arange(start, len(s)), size=min(12, len(s) - start), replace=False)
        s[idx] = s[idx] * RNG.uniform(2.5, 4.0, size=len(idx))
    return s


def build(n_days: int = 400, n_doors: int = 180, n_bogies: int = 80):
    assets = _assets(n_doors, n_bogies)
    dates = pd.date_range("2025-07-01", periods=n_days, freq="D")

    tel_rows, fault_rows, wo_rows = [], [], []

    for a in assets.itertuples():
        door = a.system == "door"

        usage_rate = (
            RNG.normal(1150, 90) * a.duty if door else RNG.normal(455, 40) * a.duty
        )
        usage = np.maximum(RNG.normal(usage_rate, usage_rate * 0.06, n_days), 0)

        # Use already accumulated since the last replacement, before our window
        # opens. Without this every asset looks the same age and the
        # reliability estimate has nothing to discriminate on.
        nominal_life = NOMINAL_LIFE["door" if door else "bogie"]
        usage_offset = float(RNG.uniform(0.05, 0.70) * nominal_life)
        life_fraction = usage_offset / nominal_life

        # 30% of assets are on a degradation path, and worn assets degrade
        # faster, so condition and reliability correlate without being identical
        degrading = RNG.random() < 0.30
        sensor_fault = RNG.random() < 0.07
        degrade_start = int(RNG.integers(30, n_days - 90)) if degrading else n_days + 1
        rate = RNG.uniform(1.0, 3.2) * (0.4 + 1.4 * life_fraction) if degrading else 0.0

        t = np.arange(n_days)
        ramp = np.clip(t - degrade_start, 0, None)

        if door:
            base_close = RNG.normal(2.78, 0.05)
            close = base_close + 0.00012 * np.cumsum(usage) / 1000 * a.duty
            close = close + rate * 0.0016 * ramp + RNG.normal(0, 0.012, n_days)
            current = 4.02 + (close - base_close) * 2.6 + RNG.normal(0, 0.05, n_days)
            pressure = RNG.normal(5.35, 0.09, n_days)
            primary, secondary, tertiary = close, current, pressure
            pnames = ("door_close_time_s", "door_motor_current_a", "door_air_pressure_bar")
            thr_key = "door_close_time_s"
        else:
            base_temp = RNG.normal(11.0, 0.7)
            temp = base_temp + 0.000021 * np.cumsum(usage) * a.duty
            temp = temp + rate * 0.020 * ramp + RNG.normal(0, 0.30, n_days)
            vib = 0.40 + (temp - base_temp) * 0.017 + RNG.normal(0, 0.010, n_days)
            load = RNG.normal(6.4, 0.25, n_days)
            primary, secondary, tertiary = temp, vib, load
            pnames = ("bogie_axlebox_temp_rise_c", "bogie_vibration_rms_g", "bogie_axle_load_t")
            thr_key = "bogie_axlebox_temp_rise_c"

        sf_kind = None
        if sensor_fault:
            sf_kind = str(RNG.choice(["frozen", "missing", "drift", "spike"]))
            sf_start = int(RNG.integers(n_days // 2, n_days - 30))
            # the sensor breaks, the equipment does not
            secondary = _inject_sensor_fault(secondary, sf_kind, sf_start)

        thr = THRESHOLDS[thr_key]
        crossed = np.where(primary >= thr * 0.985)[0]
        fault_day = int(crossed[0]) if len(crossed) else None
        if fault_day is not None and fault_day < n_days - 3:
            ftype = (
                str(RNG.choice(["door_close_slow", "door_obstruction", "door_motor_overload"]))
                if door
                else str(RNG.choice(["axlebox_overheat", "bearing_wear", "suspension_degradation"]))
            )
            fault_rows.append(
                dict(
                    asset_id=a.asset_id,
                    date=dates[fault_day],
                    system=a.system,
                    fault_type=ftype,
                    verified=True,
                )
            )
            for pname, prob, lead, oem in PARTS_BY_FAULT[ftype]:
                if RNG.random() < prob:
                    wo_rows.append(
                        dict(
                            asset_id=a.asset_id,
                            date=dates[fault_day],
                            fault_type=ftype,
                            part=pname,
                            qty=int(RNG.integers(1, 3)),
                            lead_time_days=lead,
                            oem_supported=oem,
                        )
                    )
            # Repair returns the indicator to baseline and it starts ageing
            # again. Assigning a scalar here instead flat-lines the channel,
            # which signal validation then correctly reports as a frozen
            # sensor. That bug cost us an afternoon.
            tail = n_days - fault_day
            noise = 0.012 if door else 0.30
            creep = 0.00004 if door else 0.0025
            primary[fault_day:] = (
                primary[0] + np.arange(tail) * creep + RNG.normal(0, noise, tail)
            )
            # secondary stays physically coupled to primary after the repair.
            # Regenerating it as independent noise decouples the channels, and
            # the cross-channel check then correctly reports a sensor fault.
            coupling = 2.6 if door else 0.017
            secondary[fault_day:] = (
                secondary[0]
                + (primary[fault_day:] - primary[0]) * coupling
                + RNG.normal(0, noise * 0.4, tail)
            )

        # Routine corrective jobs, with a hazard proportional to use. Real work
        # order history is mostly these, not dramatic threshold crossings, and
        # without them the learned MTBF comes out implausibly long because the
        # denominator (failure count) is far too small.
        cum = usage_offset + np.cumsum(usage)
        p_daily = usage / nominal_life
        for day in np.where(RNG.random(n_days) < p_daily)[0]:
            ftype = (
                str(RNG.choice(["door_close_slow", "door_obstruction", "door_motor_overload"]))
                if door
                else str(RNG.choice(["axlebox_overheat", "bearing_wear", "suspension_degradation"]))
            )
            fault_rows.append(
                dict(
                    asset_id=a.asset_id, date=dates[day], system=a.system,
                    fault_type=ftype, verified=True, severity="corrective",
                )
            )
            for pname, prob, lead, oem in PARTS_BY_FAULT[ftype]:
                if RNG.random() < prob:
                    wo_rows.append(
                        dict(
                            asset_id=a.asset_id, date=dates[day], fault_type=ftype,
                            part=pname, qty=int(RNG.integers(1, 3)),
                            lead_time_days=lead, oem_supported=oem,
                        )
                    )

        tel_rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "asset_id": a.asset_id,
                    "line": a.line,
                    "system": a.system,
                    pnames[0]: primary,
                    pnames[1]: secondary,
                    pnames[2]: tertiary,
                    "usage": usage,
                    "usage_offset": usage_offset,
                    "_sensor_fault_truth": sf_kind or "",
                }
            )
        )

    telemetry = pd.concat(tel_rows, ignore_index=True)
    faults = pd.DataFrame(fault_rows)
    work_orders = pd.DataFrame(wo_rows)
    return assets, telemetry, faults, work_orders


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)
    assets, telemetry, faults, work_orders = build()
    assets.to_csv(out / "assets.csv", index=False)
    telemetry.to_csv(out / "telemetry.csv", index=False)
    faults.to_csv(out / "faults.csv", index=False)
    work_orders.to_csv(out / "work_orders.csv", index=False)
    print(f"assets {len(assets)}  telemetry {len(telemetry)}  faults {len(faults)}  work orders {len(work_orders)}")
