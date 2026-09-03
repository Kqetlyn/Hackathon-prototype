# Models — what we use, why, and how to defend it

PS3 asks for two things: a tool with AI/ML models that detects anomalies across
train systems, and an answer to *which models are best for detecting future
anomalies*. This document is the answer to the second part.

Read this before the hackathon. The whole point is that you can explain every
choice to a rail engineer without hiding behind "the model said so".

---

## The shape of the problem

Three facts determine every model choice below.

**Faults are rare.** Roughly 0.3% of asset-days in our synthetic data are within
21 days of a fault. On the real dataset expect something similar. This kills
accuracy as a metric — a model that predicts "no fault" every time scores 99.7%
and is worthless. It also means you need class weighting almost everywhere.

**Time matters.** Yesterday predicts today. If you split the data randomly, the
model trains on next month and tests on last month, which inflates every score
on the board. **Always split by time.** This is the single most common way
hackathon projects produce fake results.

**"Future anomalies" means two different things.** Fault types you have labels
for, and behaviour nobody has labelled yet. Those need different model
families, which is precisely why a benchmark is the right answer to PS3's
question rather than picking one model.

---

## The models

### Logistic regression — the baseline

```python
make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
```

Not a serious contender. It's there so you know what "hard" looks like. If your
random forest barely beats logistic regression, your features are doing the
work and the model choice barely matters — which is useful to know and honest
to say.

**Always include a baseline.** Judges who know ML will notice if you don't, and
it costs you thirty seconds.

### Random forest — the workhorse

```python
RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                       class_weight="balanced_subsample")
```

Many decision trees, each trained on a random subset of rows and columns, voting
together. Averaging over trees is what makes it robust to the noise in sensor
data.

Why it suits this problem: handles mixed feature scales without normalisation,
tolerates missing values reasonably, resists overfitting better than a single
tree, and gives you feature importances for free. A published empirical
comparison on railway sensor anomaly detection put CatBoost first at 96%,
random forest at 91% and XGBoost at 90%, so this family is the right
neighbourhood to be in.

`min_samples_leaf=20` stops the trees memorising individual assets.
`class_weight="balanced_subsample"` makes it attend to the rare positive class.

`max_depth=14` is there for a practical reason worth knowing: left unbounded,
the forest grows to purity on 70,000 rows and takes about 80 seconds to fit,
which was most of the pipeline's runtime, for no measurable accuracy gain. On
hackathon day your iteration speed matters more than the fourth decimal place.
Cap the depth.

### Gradient boosting — histogram and XGBoost

```python
HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06)
XGBClassifier(n_estimators=400, max_depth=5, scale_pos_weight=ratio,
              eval_metric="aucpr")
```

Where the forest builds trees independently and averages, boosting builds them
in sequence, each one correcting the previous one's mistakes. Usually a little
sharper than a forest on tabular data, and usually a little more willing to
overfit.

`HistGradientBoosting` ships with scikit-learn, so it always runs.
XGBoost is optional — the benchmark adds it automatically if it's installed.

Two settings carry the imbalance. `scale_pos_weight` set to the
negative-to-positive ratio, and `eval_metric="aucpr"` so it optimises for the
precision-recall curve rather than accuracy. Without these, boosting collapses
to predicting the majority class and scores worse than the baseline. We hit
exactly that during development — `HistGradientBoosting` came back at 0.038
PR-AUC, below the base rate, until sample weights were added.

**CatBoost is worth twenty minutes if you have them.** It topped that railway
comparison, it handles categorical features like line and stock type natively,
and it's a one-line addition to the benchmark dictionary. Being able to say "we
tried the model that leads the published rail benchmark" is a good sentence to
have.

### Isolation forest — the unlabelled case

```python
IsolationForest(n_estimators=300, contamination=0.05)
```

Unsupervised. It isolates points by random splitting and measures how few splits
each point needs — outliers separate quickly, normal points don't. No labels
required.

This is your answer to "detect **future** anomalies". A supervised model can
only find fault types someone already labelled. Every genuinely new failure
mode is invisible to it by construction. The isolation forest doesn't care what
the fault is called.

Trained on the training window only, scored on the test window, same as the
supervised models, so the comparison is fair.

**Say this on stage:** supervised for known faults, unsupervised for unknown
ones, and we report both because they answer different halves of the question.

### Deliberately not used

**Sequence autoencoders and LSTMs.** The obvious "impressive" choice, and the
wrong one for a weekend. They need far more data than you'll have, they're slow
to train, they're hard to debug at 3am, and their output is a reconstruction
error that's harder to explain than a tree's feature importances. Name them as
the next step if a judge asks about deep learning; don't build one.

**Weibull survival models.** Genuinely the right tool for remaining life, and
`lifelines` makes it three lines of code. The risk isn't the code, it's the
data — survival analysis needs per-asset event histories with proper censoring
and mileage, and you won't know until Friday night whether the dataset supports
that. See `remaining_life.py` for what we do instead, and keep Weibull as the
honest "with more time" answer.

---

## Metrics, and why not the obvious ones

| Metric | Why it's here |
|---|---|
| PR-AUC (average precision) | The right summary when positives are rare. ROC-AUC looks flattering on imbalanced data and hides the false alarm problem |
| Lift over base rate | PR-AUC of 0.6 means nothing on its own. Against a 0.3% base rate it means the model is roughly 200× better than guessing. Always quote both |
| Precision at top 5% | Realistic operating point. Engineers can inspect a fixed number of assets per night, not "everything above 0.5" |
| Recall at top 5% | Of the faults that happened, how many did we catch inside that budget |
| False alarms per 1,000 asset-days | Speaks the operator's language. Each false alarm is engineering hours burned |
| Median lead days | How much warning you actually get. A model that alerts the day before failure is useless even at perfect precision |

**One implementation warning.** Select the top 5% by *rank*, not by
`score >= quantile(score, 0.95)`. Tree models produce heavy ties, and a value
threshold landing on a tied block alerts on far more rows than you intended.
Our first run showed random forest with 100% recall and 992 false alarms per
1,000 days, which is what that bug looks like. `detect.top_k_mask()` does it
correctly.

---

## Features, which matter more than the model

All models see identical features, because otherwise the comparison is
meaningless. From `features.py`:

- Rolling mean and standard deviation over 7, 30 and 90 days
- Rolling 30-day slope — the rate of change, usually more predictive than level
- Delta from the asset's own healthy baseline, so each asset is its own control
- Cumulative usage and recent usage rate
- **Wear per unit of use, z-scored against same-day fleet peers**

That last one is the one to talk about. Raw wear tells you a door is worn. Wear
per cycle compared against the fleet tells you whether it's wearing *faster than
its workload explains*. Those are different statements, and only the second is
an anomaly. A busy door on the North East Line accumulating cycles faster than a
quiet one isn't faulty, it's busy.

### Leakage — check this before you trust any number

Two ways to leak the answer into the features:

1. **Anything derived from the fault record.** Repair dates, work order flags,
   part replacements. Obvious once stated, easy to merge in by accident.
2. **Cumulative counters that reset on repair.** If mileage-since-overhaul
   resets at the fault, its low value *is* the label. We compute the baseline
   from each asset's first 60 days only, before any repair.

Sanity check: if any single feature gives you PR-AUC above about 0.9, you have
leakage. Go and find it.

---

## What we actually got

On the synthetic fleet, 260 assets over 400 days, 2.8% positive rate, alerting
on the top 5% of asset-days:

| Model | Kind | PR-AUC | Lift | Precision | Recall | False alarms /1k days | Lead days |
|---|---|---|---|---|---|---|---|
| Hist gradient boosting | supervised | 0.447 | 11.0× | 0.369 | 0.455 | 31.6 | 22 |
| Random forest | supervised | 0.337 | 8.3× | 0.304 | 0.376 | 34.8 | 25 |
| XGBoost | supervised | 0.301 | 7.4× | 0.269 | 0.333 | 36.5 | 17 |
| Isolation forest | unsupervised | 0.211 | 5.2× | 0.308 | 0.380 | 34.6 | 27 |
| Logistic regression | baseline | 0.165 | 4.1× | 0.278 | 0.343 | 36.1 | 25 |

Read this the right way. Roughly a third of the alerts are real and it catches
under half the faults, with about three weeks of warning. That is an ordinary
result, not a triumph — and it is what an honest predictive maintenance model
looks like at a realistic operating point. **Expect the real dataset to give you
numbers in this neighbourhood.** If you see 0.95 recall on day one, go looking
for leakage before you celebrate.

The isolation forest scoring below the trees is expected: it has no labels. It's
there for the fault types that aren't in the label set at all, which is exactly
the case its score can't measure.

**One thing that will happen to you.** During development, on an earlier version
of the synthetic data, logistic regression came top at 0.881 PR-AUC and beat
every tree. That wasn't a fluke, it was a message: the engineered features
(delta from baseline, 30-day slope, peer-normalised wear) had already made the
problem close to linearly separable, so the model choice barely mattered. If
your baseline is competitive with your best model, say so on stage. It means
your feature engineering is carrying the result, which is a better story than a
marginally higher number, and a judge who knows ML will respect you for
noticing.

## Running the benchmark

```bash
python run_pipeline.py          # writes outputs/model_comparison.csv
streamlit run app.py            # model benchmark tab
```

To add a model, put it in the `supervised` dict in `src/detect.py`. Everything
else — split, imputation, metrics, table — is shared.

```python
from catboost import CatBoostClassifier
supervised["CatBoost"] = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.06,
    auto_class_weights="Balanced", verbose=0,
)
```

---

## What to say when a judge asks "why this model"

> We didn't pick one. PS3 asks which models work best, so we benchmarked
> several on identical features with a time-based split. Tree ensembles win on
> the labelled fault types, which matches the published rail literature. But
> supervised models can only find faults someone already labelled, so we run an
> isolation forest alongside for behaviour that has never been labelled. The
> table shows both, and which one wins depends on the system — doors and bogies
> don't behave the same way.

Then, if they push on deep learning:

> A sequence autoencoder is the natural next step and we'd expect it to help on
> the raw high-frequency signals. In a weekend, on this much data, it would have
> been slower to train and much harder to explain, and explainability matters
> more than half a point of PR-AUC when an engineer has to act on the output.
