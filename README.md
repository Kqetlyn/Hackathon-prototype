# Rolling stock maintenance decision tool

NEBULA X hackathon · Problem statement 3, predictive fault detection · LTA Rail
Digitalisation & Guild, 18–20 September 2026.

Door and bogie telemetry goes in. For each asset, what comes out is how long it
has left, whether it can wait for the next scheduled maintenance, and which
spares to have ready.

**Predictive maintenance usually stops at an anomaly score.** The harder
question is when to act on it, since the railway only gets a few hours a night
to do any work at all.

---

## Quick start

**Windows** — double-click `run.bat`, or from a terminal:

```
run.bat            first run sets up the venv and trains, then opens the dashboard
run.bat fresh      retrain from scratch
```

**Any platform**, once dependencies are installed:

```bash
pip install -r requirements.txt
python app.py      runs the pipeline if needed, then opens the dashboard
```

`python app.py` works because the app relaunches itself under Streamlit when it
notices it wasn't started that way. `streamlit run app.py` still works too.

To retrain without opening the dashboard: `python run_pipeline.py` (~95s).

The pipeline generates its own synthetic fleet, so the whole thing runs before
you have the real dataset.

---

## Where this sits

Singapore already has the sensing layer. Worth knowing before you claim novelty
in front of an SMRT engineer:

| System | What it covers | Status |
|---|---|---|
| SMRT–NTU door sensors | Air pressure, movement speed, power on pneumatic doors | Trialled 2018, rolling out |
| Bogie monitoring | Wheel wear, axle load, temperature, vibration | Being fitted |
| PDSS (Bentley AssetWise) | **Permanent way** — track infrastructure | NSEWL, extending to CCL |
| REAMS (Siemens + ST Eng) | Enterprise asset management, S$18.8m | Downtown Line first |

Note the fourth row carefully. PDSS is track, not rolling stock. Doors and
bogies have sensors going in, but the layer that turns those signals into a
scheduling and preparation decision is thin. That gap is what this builds in.

Three things from LTA's [Rail Reliability Taskforce report (13 Feb
2026)](https://www.lta.gov.sg/content/ltagov/en/newsroom/2026/2/news-releases/lta-rail-operators-progressively-implement-rail-reliability-taskforce-to-strengthen-network-reliability.html)
shape the design directly:

1. **Spares forecasting is a named workstream**, including parts discontinued by
   the OEM. Hence `spares.py` and the `obsolete` flag.
2. **Condition monitoring is fragmented** — different vendors, different
   vintages, different parameters. Hence the config-driven channel mapping
   rather than hardcoding one schema.
3. **Engineering hours are getting more contested**, not less, because renewal
   work now competes with routine maintenance for the same windows. Hence
   framing everything as a scheduling decision.

---

## Pipeline

```
generate_data → validate → features → detect → remaining_life → decide → spares
```

**1. Validate** (`src/validate.py`) — can this reading be trusted? Missing,
frozen, drifting, out of range, spiking, or channels that have decoupled from
each other. Rules only, no model, because each check has to be explainable to
an engineer in one sentence. Assets that fail are pulled out of scoring and
raised as sensor jobs instead. A broken sensor and a failing door look alike,
and chasing the wrong one costs engineering hours nobody has spare.

**2. Detect** (`src/features.py`, `src/detect.py`) — several models on
identical features with a strict time-based split. Tree ensembles for labelled
fault types, isolation forest for behaviour nobody has labelled. See
[docs/MODELS.md](docs/MODELS.md) for the full reasoning.

**3. Remaining life** (`src/remaining_life.py`) — two independent estimates.
Condition: fit a robust trend to the health indicator and project to the alarm
threshold. Reliability: compare accumulated use since replacement against the
characteristic life of that component type, counted in **door cycles and
train-km rather than calendar days**, which is also how LTA measures
reliability (MKBF). The engineer sees one number: a point estimate when the two
agree, a range when they don't, with the width of the range as the confidence.

**4. Decide** (`src/decide.py`) — one comparison. Does it run out before its
next scheduled maintenance? Judged on the shorter end of the range, so
uncertainty counts against the asset. Margin in days is carried through and the
list sorts by it, so a job sitting two days clear of its window surfaces
instead of vanishing into the safe pile.

**5. Spares** (`src/spares.py`) — from work order history, not telemetry. For
past faults of this type, which parts were actually issued and how often. Flags
anything whose lead time exceeds the asset's remaining life, or that the OEM no
longer supports.

---

## Swapping in the real dataset

Replace the `generate_data.build()` call in `run_pipeline.py` with a loader
returning the same four frames, and map the real column names in `CHANNELS`:

| Frame | Required columns |
|---|---|
| `assets` | `asset_id`, `line`, `system` |
| `telemetry` | `date`, `asset_id`, `line`, `system`, `usage`, one column per channel |
| `faults` | `asset_id`, `date`, `system`, `fault_type` |
| `work_orders` | `asset_id`, `date`, `fault_type`, `part`, `lead_time_days`, `oem_supported` |

Nothing else changes. If the real data has no work order history, `spares.py`
degrades gracefully and everything upstream still runs.

---

## Honest limitations

Say these before a judge finds them.

- **Synthetic data.** Numbers in `outputs/` demonstrate that the pipeline works,
  not that it works on the real fleet. Every figure gets replaced on day one.
- **PM schedule is invented.** Real intervals come from the maintenance regime.
  The comparison logic is what matters, not the dates.
- **Parts data is invented.** LTA's listed datasets are door, bogie and verified
  fault data — work order history probably isn't among them. Show this layer on
  stand-in data and say so.
- **No Weibull.** Survival analysis with right-censoring is the better remaining
  life model. It needs event histories we won't have until Friday night. See
  the note in `remaining_life.py`.
- **Batch, not streaming.** Daily aggregation. Real-time adds nothing to the
  decision when the decision is about which night to act.

---

## Layout

```
run_pipeline.py     end to end, writes outputs/
app.py              Streamlit dashboard
src/
  generate_data.py  synthetic fleet — replace on hackathon day
  validate.py       stage 1, sensor triage
  features.py       stage 2a, windowed + peer-normalised features
  detect.py         stage 2b, model benchmark
  remaining_life.py stage 3, condition + reliability estimates
  decide.py         stage 4, urgency vs the PM window
  spares.py         stage 5, parts from work order history
docs/
  MODELS.md         why each model, how to run it, how to defend it
```
