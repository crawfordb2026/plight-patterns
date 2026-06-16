# Quickstart: TTD Random Forest Pipeline

This guide walks through every step from raw DAM monitor files to trained models.

---

## Overview

```
Step 0  Filter raw monitor files by date range   →  Python/monitors_date_filtered/
Step 1  Run TTD Random Forest (all features)     →  analyses/analysis_results/
```

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Create `Python/metadata.txt` from the provided example:

```bash
cp Python/metadata_EXAMPLE.txt Python/metadata.txt
# then edit metadata.txt to reflect your actual flies
```

---

## Step 0 — Filter Monitor Files (optional)

Raw monitor files can optionally be pre-trimmed to the desired 
experimental window. This removes activity before flies were 
loaded and any trailing recordings after all deaths are confirmed.

```bash
cd Python/src

python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 30 --offset 1
```

| Flag | Meaning |
|------|---------|
| `--input` | Monitor name (e.g. `Monitor51`). File must exist in `Python/Monitors_raw/`. |
| `--load` | Load date in MM/DD/YY format — when flies were placed in the monitor. |
| `--days` | Number of days of data to keep after the offset. |
| `--offset` | Days after load date before recording starts (e.g. 1 = skip acclimation day). |

Output is written to `Python/monitors_date_filtered/Monitor51_06_20_25.txt`.

Repeat for each monitor used in the experiment.

---

## Step 1 — Run TTD Random Forest

```bash
cd Python/src

python ttd_rf.py
```

By default the script reads from `Python/monitors_date_filtered/` and
`Python/metadata.txt`, and writes outputs to `analyses/analysis_results/`.

Override paths if needed:

```bash
python ttd_rf.py \
  --monitors-dir ../monitors_date_filtered \
  --metadata ../metadata.txt \
  --output-dir ../../analyses/analysis_results
```

### What it does

1. Loads all `Monitor*.txt` files and merges with metadata.
2. For each fly, detects time of death (last movement + 48h zero confirmation).
3. Generates non-overlapping 1-hour windows anchored backward from death. Each
   window is labeled with TTD (hours remaining until death). Windows before the
   first 24h of recording and windows with fewer than 55 min of data are discarded.
4. Computes features per window:
   - **Sleep** (from 1-hr window): total sleep, bout count/duration, P(wake), P(doze), WASO
   - **Circadian** (from 24-hr lookback ending at window end): mesor, amplitude, phase, IV, RA, L5, M10
   - **ZT encoding**: ZT_sin, ZT_cos (circular, period 24; ZT0 = 9am)
5. Trains three Random Forest regressors with grouped 5-fold CV (all windows
   from one fly stay in the same fold, folds stratified by genotype):
   - **Model A**: sleep + circadian + ZT
   - **Model B**: Model A + fly age in hours
   - **Model C**: Model B + one-hot genotype

### Outputs

| File | Contents |
|------|----------|
| `windowed_features.csv` | Full feature matrix (fly_id, TTD, all features) |
| `model_performance.txt` | MAE, RMSE, R² per fold and mean±SD for all three models |
| `feature_importances.csv` | Mean feature importances across folds, per model |
| `predicted_vs_actual.png` | Scatter plot with diagonal reference, colored by fold |

---

## AWS S3 Sync (optional)

Push data and results to S3 for collaboration:

```bash
cd Python/AWS

# One-time setup
cp s3_config.example.json s3_config.json
# edit s3_config.json: {"bucket": "your-bucket-name"}

# Upload everything
python s3_sync.py push --prefix experiment-june-2025

# Upload only analysis results
python s3_sync.py push --results --prefix experiment-june-2025

# Check sync status
python s3_sync.py status --prefix experiment-june-2025

# Download on another machine
python s3_sync.py pull --prefix experiment-june-2025
```

---

## Metadata Format

`Python/metadata.txt` (tab/space separated, header required):

```
Monitor  Channel  Genotype  Sex  Treatment
51_06_20_25  ch1  w1118  Female  VEH
51_06_20_25  ch2  w1118  Female  VEH
51_06_20_25  ch3  na     na      na
```

- Monitor format: `{number}_{MM}_{DD}_{YY}` or plain integer
- Use `na` for empty channels — they are automatically skipped
- Genotype is the only metadata column used as a feature (in Model C)
