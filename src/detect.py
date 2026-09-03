"""Stage 2b — model benchmark.

PS3 asks teams to "identify the best models to be used to detect future
anomalies". This module answers that with evidence rather than assertion:
identical features, identical time-based split, several model families,
one comparison table.

Two families, because "future anomalies" means two different things:
  supervised   fault types we have labels for
  unsupervised behaviour no one has labelled yet

Split is strictly by time. A random split leaks the future into the training
set and inflates every number on the board.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def time_split(features: pd.DataFrame, frac: float = 0.7):
    d = pd.to_datetime(features["date"])
    cut = d.quantile(frac)
    return d <= cut, d > cut


def _feature_columns(features: pd.DataFrame) -> list[str]:
    drop = {"asset_id", "date", "line", "system"}
    return [c for c in features.columns if c not in drop and pd.api.types.is_numeric_dtype(features[c])]


def top_k_mask(score: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Alert on the highest-scoring `frac` of rows.

    Deliberately rank based, not `score >= quantile`. Tree models produce heavy
    ties, and a value threshold sitting on a tied block silently alerts on far
    more rows than intended. That bug reads as a model with perfect recall and
    a 99% false alarm rate, which is how it was caught.
    """
    k = max(1, int(round(len(score) * frac)))
    idx = np.argsort(score, kind="stable")[::-1][:k]
    m = np.zeros(len(score), dtype=bool)
    m[idx] = True
    return m


def _lead_time_days(features: pd.DataFrame, mask, score, faults, alert_mask) -> float:
    """Median days between first alert and the fault it preceded."""
    if faults is None or len(faults) == 0:
        return float("nan")
    sub = features.loc[mask, ["asset_id", "date"]].copy()
    sub["alert"] = alert_mask
    f = faults.copy()
    f["date"] = pd.to_datetime(f["date"])
    leads = []
    for asset_id, g in sub.groupby("asset_id"):
        fd = f.loc[f["asset_id"] == asset_id, "date"]
        if fd.empty:
            continue
        alerts = pd.to_datetime(g.loc[g["alert"], "date"])
        if alerts.empty:
            continue
        for d in fd:
            prior = alerts[alerts <= d]
            if len(prior):
                leads.append((d - prior.min()).days)
    return float(np.median(leads)) if leads else float("nan")


def benchmark(features: pd.DataFrame, y: pd.Series, faults: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    cols = _feature_columns(features)
    tr, te = time_split(features)
    X = features[cols]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr_i, Xte_i = imp.transform(Xtr), imp.transform(Xte)

    supervised = {
        "Logistic regression (baseline)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        # max_depth and min_samples_leaf are not cosmetic here. Left unbounded
        # the forest grows to purity on 70k rows and takes ~80s to fit, which
        # is most of the pipeline runtime, for no measurable accuracy gain.
        "Random forest": RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "Hist gradient boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=RANDOM_STATE
        ),
    }
    try:  # optional, only if the team pip installs it
        from xgboost import XGBClassifier

        pos = max((ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0)
        supervised["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.9, n_jobs=-1,
            colsample_bytree=0.8, scale_pos_weight=pos, eval_metric="aucpr",
            random_state=RANDOM_STATE,
        )
    except ImportError:
        pass

    rows, scores = [], {}
    base_rate = float(yte.mean())
    for name, model in supervised.items():
        if isinstance(model, HistGradientBoostingClassifier):
            w = np.where(ytr == 1, max((ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0), 1.0)
            model.fit(Xtr_i, ytr, sample_weight=w)
        else:
            model.fit(Xtr_i, ytr)
        p = model.predict_proba(Xte_i)[:, 1]
        scores[name] = p
        # operating point: alert on the top 5% of asset-days
        alert = top_k_mask(p, 0.05)
        pred = alert.astype(int)
        fa = int(((pred == 1) & (yte.to_numpy() == 0)).sum())
        rows.append(
            dict(
                model=name,
                kind="supervised",
                pr_auc=round(average_precision_score(yte, p), 3),
                lift_over_base=round(average_precision_score(yte, p) / max(base_rate, 1e-9), 1),
                precision_at_5pct=round(precision_score(yte, pred, zero_division=0), 3),
                recall_at_5pct=round(recall_score(yte, pred, zero_division=0), 3),
                false_alarms_per_1k_days=round(1000 * fa / max(len(yte), 1), 1),
                median_lead_days=_lead_time_days(features, te, p, faults, alert),
            )
        )

    # Unsupervised: trained on the training window only, no labels used.
    iso = IsolationForest(n_estimators=150, contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(Xtr_i)
    p_iso = -iso.score_samples(Xte_i)
    scores["Isolation forest"] = p_iso
    alert = top_k_mask(p_iso, 0.05)
    pred = alert.astype(int)
    fa = int(((pred == 1) & (yte.to_numpy() == 0)).sum())
    rows.append(
        dict(
            model="Isolation forest",
            kind="unsupervised",
            pr_auc=round(average_precision_score(yte, p_iso), 3),
            lift_over_base=round(average_precision_score(yte, p_iso) / max(base_rate, 1e-9), 1),
            precision_at_5pct=round(precision_score(yte, pred, zero_division=0), 3),
            recall_at_5pct=round(recall_score(yte, pred, zero_division=0), 3),
            false_alarms_per_1k_days=round(1000 * fa / max(len(yte), 1), 1),
            median_lead_days=_lead_time_days(features, te, p_iso, faults, alert),
        )
    )

    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    table.attrs["test_base_rate"] = round(base_rate, 4)

    best = table.loc[table["kind"] == "supervised", "model"].iloc[0]
    risk = pd.Series(np.nan, index=features.index)
    risk[te] = scores[best]
    risk.attrs["model"] = best

    # importances come from the forest we already fitted, rather than training a
    # second one purely to read them off
    rf = supervised["Random forest"]
    imp_df = (
        pd.DataFrame({"feature": cols, "importance": rf.feature_importances_.round(4)})
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    return table, risk, imp_df
