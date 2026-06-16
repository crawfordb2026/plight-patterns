# flyght-patterns

> Predicting time-to-death in *Drosophila* from sleep and circadian behavior using Random Forest regression

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

This project trains Random Forest regressors to predict **time-to-death (TTD)** — the number of hours remaining until a fly's natural death — from behavioral features derived from Drosophila Activity Monitor (DAM) recordings.

Activity is measured in 1-minute bins (MT values). For each fly, the recording is divided into non-overlapping 1-hour windows anchored backward from the confirmed time of death. Each window becomes one training row labeled with its TTD. Features are computed from the window itself (sleep architecture) and from the 24-hour lookback ending at the window boundary (circadian rhythm).

Three model variants quantify how much age and genotype contribute on top of behavior:

| Model | Features |
|-------|---------|
| A | Sleep + circadian + ZT encoding |
| B | Model A + fly age in hours |
| C | Model B + one-hot genotype |

Performance is assessed via grouped 5-fold cross-validation — all windows from one fly always remain in the same fold — with fold assignment stratified by genotype.

---

## Feature Set

**Sleep features** (computed from the 1-hour prediction window)

- Total sleep time, sleep bout count, mean and max bout duration
- P(wake): transition probability from sleep to wake per minute
- P(doze): transition probability from wake to sleep per minute
- WASO: wake-after-sleep-onset

Sleep is defined as 5 or more consecutive minutes with MT = 0.

**Circadian features** (computed from the 24-hour lookback ending at window end)

- Mesor, amplitude, phase — from cosinor regression on hourly activity totals
- IV (intradaily variability) — fragmentation of the rest-activity cycle within a day
- RA (relative amplitude) — contrast between most and least active periods
- L5 — mean activity of the least active 5 consecutive hours (circular)
- M10 — mean activity of the most active 10 consecutive hours (circular)

**ZT encoding**

- ZT_sin, ZT_cos — circular encoding of the window-start Zeitgeber hour (period 24, ZT0 = 9am)
- Raw ZT is never used as a feature

---

## Death Detection

Death is defined as the last minute with MT > 0, confirmed by 48 subsequent hours of zero activity. Flies that cannot be confirmed by this criterion (insufficient post-death recording) are excluded.

---

## Project Structure

```
flyght-patterns/
├── Python/
│   ├── Monitors_raw/              raw DAM monitor files (not tracked)
│   ├── monitors_date_filtered/    date-trimmed monitor files (Step 0 output)
│   ├── metadata.txt               fly metadata (genotype per channel)
│   ├── metadata_EXAMPLE.txt       format reference
│   ├── AWS/
│   │   ├── s3_sync.py             push/pull data and results to/from S3
│   │   └── s3_config.example.json bucket configuration template
│   ├── QUICKSTART/
│   │   └── QUICKSTART.md          step-by-step usage guide
│   └── src/
│       ├── 0-filter_dates.py      trim raw monitor files to experimental window
│       ├── data_loader.py         parse monitor files and metadata
│       ├── features.py            sleep and circadian feature computation
│       ├── windowing.py           death detection and backward window generation
│       └── ttd_rf.py              main script: windowing → features → RF → outputs
├── analyses/
│   └── analysis_results/          all model outputs written here
├── R/
│   └── hmm.r                      R pipeline (separate, untouched)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt

cp Python/metadata_EXAMPLE.txt Python/metadata.txt
# edit metadata.txt for your experiment

# Step 0: trim raw monitor files (repeat per monitor)
cd Python/src
python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 30 --offset 1

# Step 1: run TTD random forest
python ttd_rf.py
```

See **[`Python/QUICKSTART/QUICKSTART.md`](Python/QUICKSTART/QUICKSTART.md)** for full instructions.

---

## Outputs

All outputs are written to `analyses/analysis_results/`:

| File | Contents |
|------|----------|
| `windowed_features.csv` | Full feature matrix with fly_id, TTD, all features |
| `model_performance.txt` | Per-fold and mean±SD MAE/RMSE/R² for all three model variants |
| `feature_importances.csv` | Importances averaged across folds, per model variant |
| `predicted_vs_actual.png` | Scatter plot with diagonal reference, colored by fold |

---

## Light Cycle

ZT0 = 9am. 12:12 LD cycle. All ZT calculations use `lights_on = 9`.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Developed by the **Bedont Lab**.
