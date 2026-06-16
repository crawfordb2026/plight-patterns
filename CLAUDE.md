# Project: flyght-patterns (TTD prediction)

## Goals
- Prioritize small, safe, readable changes.
- Fix root causes; avoid overengineering.
- Keep behavior consistent unless explicitly asked to change it.

## Keeping CLAUDE.md Updated
Update this file whenever:
- Scripts or modules are added/renamed
- Naming conventions or column names change
- New CLI flags or established patterns are introduced
- The repo structure or documentation layout changes
- A non-obvious design decision is made that future sessions should know about

The goal is that any new conversation can read CLAUDE.md and immediately understand the project's architecture and conventions without exploring the codebase from scratch.

## Project Summary
This project trains Random Forest regressors to predict **time-to-death (TTD)** in hours for Drosophila flies monitored until natural death via DAM activity monitors (1-min MT bins). TTD is a regression target only — never an input feature.

## Environment
- OS: macOS (zsh)
- Python scripts are under `Python/src/`
- Monitor data input: `Python/monitors_date_filtered/` (Monitor*.txt files)
- Metadata: `Python/metadata.txt`
- Main outputs: `analyses/analysis_results/` (repo root level)

## Repo Structure
```
flyght-patterns/
├── Python/
│   ├── Monitors_raw/              raw DAM files (not tracked)
│   ├── monitors_date_filtered/    Step 0 output (not tracked)
│   ├── metadata.txt               fly metadata (genotype per channel)
│   ├── metadata_EXAMPLE.txt       format reference
│   ├── AWS/
│   │   ├── s3_sync.py             push/pull/status for S3 sync
│   │   └── s3_config.example.json bucket config template
│   ├── QUICKSTART/
│   │   └── QUICKSTART.md          step-by-step usage guide
│   └── src/
│       ├── 0-filter_dates.py      trim raw monitor files to experimental window
│       ├── data_loader.py         parse monitor files + metadata
│       ├── features.py            sleep + circadian feature computation
│       ├── windowing.py           death detection + backward window generation
│       └── ttd_rf.py              main script: builds feature matrix, runs 3 RF variants
├── analyses/
│   └── analysis_results/          all outputs from ttd_rf.py
├── R/hmm.r                        R pipeline (separate, do not touch)
├── requirements.txt
└── README.md
```

## Documentation Structure
- `Python/QUICKSTART/QUICKSTART.md` — step-by-step usage guide (Step 0 + Step 1)
- `README.md` — scientific overview of the project; not a how-to guide

## Pipeline Architecture

### Step 0 — Date Filtering (`0-filter_dates.py`)
- Input: `Python/Monitors_raw/Monitor*.txt`
- Output: `Python/monitors_date_filtered/Monitor{N}_{MM}_{DD}_{YY}.txt`
- Trims raw monitor files to the experimental window by load date + offset + days

### Step 1 — TTD Random Forest (`ttd_rf.py`)
Imports from sibling modules in `Python/src/`:
- `data_loader.py`: parses Monitor*.txt (MT readings only) + metadata.txt → per-fly pd.Series (DatetimeIndex)
- `windowing.py`: detects death (last MT>0 + 48h zero confirmation), generates backward 1-hr windows with TTD labels
- `features.py`: computes sleep features (1-hr window) + circadian features (24-hr lookback) per window

Outputs to `analyses/analysis_results/`:
- `windowed_features.csv` — full feature matrix
- `model_performance.txt` — MAE/RMSE/R² per fold and mean±SD for models A/B/C
- `feature_importances.csv` — importances averaged across folds per model
- `predicted_vs_actual.png` — scatter with diagonal, colored by fold

## Key Design Decisions

### Death detection
Last minute with MT > 0, confirmed by 48 subsequent hours of zero MT. Flies without confirmed death are excluded entirely.

### Windowing
Non-overlapping 1-hr windows anchored backward from death_time. First 24h of each fly's recording is excluded (needed as lookback for circadian features). Windows with < 55 min of data are skipped.

### Circadian lookback
Circadian features use [window_end - 24h, window_end). This always has valid data because window_start >= recording_start + 24h.

### ZT encoding
`zt_sin = sin(2π * zt / 24)`, `zt_cos = cos(2π * zt / 24)` where `zt = (clock_hour - 9) % 24`. Raw ZT is never a feature.

### Model variants
- A: sleep + circadian + ZT
- B: A + age_hours
- C: B + one-hot genotype (drop_first=True)

### Validation
`StratifiedGroupKFold(n_splits=5)` — groups=fly_id (all windows from one fly in same fold), stratification label=genotype. Falls back to `GroupKFold` if stratification fails.

### Feature column names (all lowercase)
Sleep: `total_sleep_min`, `sleep_bout_count`, `mean_bout_min`, `max_bout_min`, `p_wake`, `p_doze`, `waso_min`
Circadian: `mesor`, `amplitude`, `phase`, `iv`, `ra`, `l5`, `m10`
ZT: `zt_sin`, `zt_cos`
Other: `age_hours`, `genotype_*` (one-hot, Model C only)

## Metadata Format
```
Monitor  Channel  Genotype  Sex  Treatment
51_06_20_25  ch1  w1118  Female  VEH
```
Monitor column can be `51_06_20_25` (number_MM_DD_YY) or plain integer.
Rows with `na` genotype are skipped automatically.
Only `genotype` is used as a feature (Model C). Sex and Treatment are parsed but not used.

## fly_id Convention
`M{monitor_num}_Ch{channel:02d}` — e.g. `M51_Ch01`, `M51_Ch12`.
Generated in `data_loader.parse_metadata()`.

## S3 Sync Categories
Defined in `Python/AWS/s3_sync.py`:
- `raw`: `Python/Monitors_raw/`
- `filtered`: `Python/monitors_date_filtered/`
- `results`: `analyses/analysis_results/`
metadata.txt is synced as a standalone file alongside filtered category.

## Code Style / Editing Rules
- Keep code simple and explicit.
- Do not introduce unnecessary abstractions.
- Do not rename files/functions unless requested.
- Preserve existing output filenames unless requested.
- Add comments only when needed for non-obvious logic.
- Do not touch unrelated files. Never touch `R/hmm.r`.

## Robustness Expectations
- Clear error messages for missing input files.
- Handle column name normalization defensively (lowercase where needed).
- Prefer deterministic behavior (`random_state=42` where applicable).

## Plotting/Runtime
- Use non-interactive matplotlib backend: `matplotlib.use('Agg')`
- Avoid GUI/Tkinter dependencies.

## Dependencies
- Keep `requirements.txt` updated when adding imports.
- Pin to reasonable minimum versions.
- No database dependencies — this project is CSV/file-only.
