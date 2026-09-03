"""End to end run. Writes everything the dashboard reads into outputs/.

    python run_pipeline.py

On hackathon day, replace the generate_data.build() call with a loader for the
real dataset and map its columns in config/signals.yaml. Nothing else changes.
"""
from __future__ import annotations

import pathlib
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import decide, detect, features, generate_data, remaining_life, spares, validate  # noqa: E402

CHANNELS = {
    "door": ["door_close_time_s", "door_motor_current_a", "door_air_pressure_bar"],
    "bogie": ["bogie_axlebox_temp_rise_c", "bogie_vibration_rms_g", "bogie_axle_load_t"],
}

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    t0 = time.time()

    print("1/6  loading data")
    assets, telemetry, faults, work_orders = generate_data.build()
    telemetry["date"] = pd.to_datetime(telemetry["date"])
    as_of = telemetry["date"].max()
    print(f"     {len(assets)} assets, {len(telemetry):,} readings, {len(faults)} faults, {len(work_orders)} work order lines")

    print("2/6  validating signals")
    sensors = validate.validate(telemetry, CHANNELS)
    bad = int((~sensors["trusted"]).sum())
    print(f"     {bad} assets excluded as sensor faults "
          f"({sensors.loc[~sensors['trusted'], 'sensor_verdict'].value_counts().to_dict()})")

    print("3/6  building features")
    feat = features.build_features(telemetry, CHANNELS)
    y = features.label_horizon(feat, faults, horizon_days=21)
    print(f"     {feat.shape[0]:,} rows x {feat.shape[1]} columns, positive rate {y.mean():.3%}")

    print("4/6  benchmarking models")
    table, risk, imp = detect.benchmark(feat, y, faults)
    feat["risk"] = risk
    print(table.to_string(index=False))
    print(f"     best supervised model: {risk.attrs.get('model')}")

    print("5/6  estimating remaining life and urgency")
    life = remaining_life.estimate(telemetry, work_orders)
    latest_risk = (
        feat.dropna(subset=["risk"]).sort_values("date").groupby("asset_id")["risk"].last().rename("risk")
    )
    life = life.merge(latest_risk, on="asset_id", how="left")
    pm = decide.next_pm_dates(assets, as_of)
    decisions = decide.decide(life, sensors, pm, as_of)
    print(f"     {decisions['status'].value_counts().to_dict()}")

    print("6/6  recommending spares")
    parts = spares.recommend(decisions, work_orders)
    demand = spares.fleet_demand(decisions, work_orders)
    late = int((parts["flag"] == "late").sum())
    obs = parts.loc[parts["flag"] == "obsolete", "part"].nunique()
    print(f"     {len(parts)} part lines, {late} lead time risks, {obs} distinct obsolete parts")

    decisions.to_csv(OUT / "decisions.csv", index=False)
    table.to_csv(OUT / "model_comparison.csv", index=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    parts.to_csv(OUT / "spares.csv", index=False)
    demand.to_csv(OUT / "fleet_demand.csv", index=False)
    sensors.to_csv(OUT / "sensor_status.csv", index=False)
    # only the columns the dashboard charts, rounded — the full frame is large
    # and writing it dominates the runtime
    keep = ["date", "asset_id", "system"] + [
        c for c in telemetry.columns if c.startswith(("door_", "bogie_"))
    ]
    recent = telemetry[telemetry["date"] >= as_of - pd.Timedelta(days=180)]
    recent[keep].round(4).to_csv(OUT / "telemetry.csv", index=False)

    print(f"\ndone in {time.time() - t0:.1f}s. outputs written to {OUT}")
    print("next:  streamlit run app.py")


if __name__ == "__main__":
    main()
