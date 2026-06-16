#!/usr/bin/env python3
"""
Random Forest regressor to predict time-to-death (TTD) in hours from
Drosophila DAM monitor activity data.

Three model variants:
  A: sleep features + circadian features + ZT encoding
  B: Model A + fly age in hours
  C: Model B + one-hot genotype

Validation: grouped 5-fold CV (all windows from one fly stay together),
stratified by genotype.

Outputs (saved to analyses/analysis_results/):
  windowed_features.csv     full feature matrix
  model_performance.txt     per-fold and mean±SD metrics for all variants
  feature_importances.csv   importances averaged across folds per variant
  predicted_vs_actual.png   scatter plot colored by fold

Usage:
  python ttd_rf.py
  python ttd_rf.py --monitors-dir ../monitors_date_filtered --metadata ../metadata.txt
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from data_loader import load_fly_series
from features import (
    compute_sleep_features, compute_circadian_features,
    SLEEP_FEATURE_NAMES, CIRCADIAN_FEATURE_NAMES,
)
from windowing import detect_death, generate_windows

LIGHTS_ON = 9
N_FOLDS = 5
MIN_DATA_MINUTES = 55
RANDOM_STATE = 42

SLEEP_COLS = SLEEP_FEATURE_NAMES
CIRC_COLS = CIRCADIAN_FEATURE_NAMES
ZT_COLS = ['zt_sin', 'zt_cos']
BASE_COLS = SLEEP_COLS + CIRC_COLS + ZT_COLS

MODEL_FEATURE_SETS = {
    'A': BASE_COLS,
    'B': BASE_COLS + ['age_hours'],
}


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(monitors_dir, metadata_path):
    """Load data, detect deaths, generate windows, compute features per window."""
    print("\nLoading fly data...")
    fly_meta, fly_series = load_fly_series(monitors_dir, metadata_path)
    genotype_map = fly_meta.set_index('fly_id')['genotype'].to_dict()

    rows = []
    n_no_death = 0
    n_no_windows = 0

    for fly_id, mt_series in sorted(fly_series.items()):
        genotype = genotype_map.get(fly_id, 'unknown')

        death_time = detect_death(mt_series)
        if death_time is None:
            n_no_death += 1
            continue

        windows = list(generate_windows(mt_series, death_time, LIGHTS_ON, MIN_DATA_MINUTES))
        if not windows:
            n_no_windows += 1
            continue

        for w in windows:
            mt_window = mt_series[
                (mt_series.index >= w['window_start']) &
                (mt_series.index < w['window_end'])
            ]
            sleep_feats = compute_sleep_features(mt_window)

            lookback_start = w['window_end'] - pd.Timedelta(hours=24)
            mt_24h = mt_series[
                (mt_series.index >= lookback_start) &
                (mt_series.index < w['window_end'])
            ]
            circ_feats = compute_circadian_features(mt_24h, LIGHTS_ON)

            rows.append({
                'fly_id': fly_id,
                'genotype': genotype,
                'ttd': w['ttd'],
                'age_hours': w['age_hours'],
                'zt_sin': w['zt_sin'],
                'zt_cos': w['zt_cos'],
                **sleep_feats,
                **circ_feats,
            })

    n_flies = len(fly_series)
    print(f"  Total flies: {n_flies}")
    print(f"  Flies with confirmed death: {n_flies - n_no_death}")
    print(f"  Skipped (death not confirmed): {n_no_death}")
    print(f"  Skipped (no valid windows): {n_no_windows}")
    print(f"  Total windows: {len(rows)}")

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _make_splitter(strat_enc, groups):
    """Return StratifiedGroupKFold, fall back to GroupKFold if stratification fails."""
    try:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        # Validate that it produces N_FOLDS splits
        splits = list(sgkf.split(np.zeros(len(groups)), strat_enc, groups))
        if len(splits) == N_FOLDS:
            return sgkf, strat_enc
    except Exception:
        pass
    gkf = GroupKFold(n_splits=N_FOLDS)
    return gkf, None


def run_cv(df, feature_cols, model_name):
    """
    Grouped 5-fold CV for one model variant.

    Returns (fold_metrics list, mean_importances array, predictions list).
    """
    df = df.copy().reset_index(drop=True)
    y = df['ttd'].values.astype(float)
    groups = df['fly_id'].values

    fly_genotype = df.groupby('fly_id')['genotype'].first()
    strat_labels = df['fly_id'].map(fly_genotype).values
    le = LabelEncoder()
    strat_enc = le.fit_transform(strat_labels)

    col_means = df[feature_cols].mean()
    X = df[feature_cols].fillna(col_means).values

    splitter, strat_arg = _make_splitter(strat_enc, groups)

    fold_metrics = []
    importances_sum = np.zeros(len(feature_cols))
    predictions = []

    split_iter = (
        splitter.split(X, strat_arg, groups)
        if strat_arg is not None
        else splitter.split(X, groups=groups)
    )

    for fold, (train_idx, val_idx) in enumerate(split_iter, 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf = RandomForestRegressor(
            n_estimators=500,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2 = float(r2_score(y_val, y_pred))

        fold_metrics.append({'fold': fold, 'mae': mae, 'rmse': rmse, 'r2': r2})
        importances_sum += rf.feature_importances_

        for actual, predicted in zip(y_val, y_pred):
            predictions.append({
                'fold': fold,
                'ttd_actual': float(actual),
                'ttd_predicted': float(predicted),
            })

        print(f"  [{model_name}] Fold {fold}: MAE={mae:.2f}h  RMSE={rmse:.2f}h  R²={r2:.3f}")

    mean_importances = importances_sum / N_FOLDS
    return fold_metrics, mean_importances, predictions


def run_model_c(df):
    """Model C: Base features + age + one-hot genotype."""
    geno_dummies = pd.get_dummies(df['genotype'], prefix='genotype', drop_first=True)
    df_c = pd.concat([df, geno_dummies.astype(float)], axis=1)
    geno_cols = list(geno_dummies.columns)
    feature_cols = MODEL_FEATURE_SETS['B'] + geno_cols
    metrics, importances, preds = run_cv(df_c, feature_cols, 'C')
    return metrics, importances, preds, feature_cols


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_performance(all_results, output_dir):
    lines = []
    for model_name, (fold_metrics, _, _) in all_results.items():
        lines.append(f"Model {model_name}")
        lines.append("=" * 50)
        maes = [m['mae'] for m in fold_metrics]
        rmses = [m['rmse'] for m in fold_metrics]
        r2s = [m['r2'] for m in fold_metrics]
        for m in fold_metrics:
            lines.append(
                f"  Fold {m['fold']}:  MAE={m['mae']:.3f}h  "
                f"RMSE={m['rmse']:.3f}h  R²={m['r2']:.4f}"
            )
        lines.append(
            f"  Mean±SD:    MAE={np.mean(maes):.3f}±{np.std(maes):.3f}h  "
            f"RMSE={np.mean(rmses):.3f}±{np.std(rmses):.3f}h  "
            f"R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}"
        )
        lines.append("")

    path = Path(output_dir) / 'model_performance.txt'
    path.write_text('\n'.join(lines))
    print(f"Saved: {path}")


def save_importances(all_results, feature_cols_per_model, output_dir):
    frames = []
    for model_name, (_, importances, _) in all_results.items():
        feat_cols = feature_cols_per_model[model_name]
        frame = pd.DataFrame({
            'model': model_name,
            'feature': feat_cols,
            'importance': importances,
        }).sort_values('importance', ascending=False)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    path = Path(output_dir) / 'feature_importances.csv'
    out.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_scatter(all_results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fold_colors = plt.cm.tab10.colors

    for ax, (model_name, (_, _, preds)) in zip(axes, all_results.items()):
        pred_df = pd.DataFrame(preds)
        for fold in range(1, N_FOLDS + 1):
            sub = pred_df[pred_df['fold'] == fold]
            ax.scatter(
                sub['ttd_actual'], sub['ttd_predicted'],
                c=[fold_colors[fold - 1]], alpha=0.35, s=6, label=f'Fold {fold}',
            )
        max_val = max(pred_df['ttd_actual'].max(), pred_df['ttd_predicted'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
        ax.set_xlabel('Actual TTD (h)')
        ax.set_ylabel('Predicted TTD (h)')
        ax.set_title(f'Model {model_name}')
        ax.legend(markerscale=2, fontsize=7)

    plt.tight_layout()
    path = Path(output_dir) / 'predicted_vs_actual.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Predict time-to-death (TTD) from DAM monitor activity.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ttd_rf.py
  python ttd_rf.py --monitors-dir ../monitors_date_filtered --metadata ../metadata.txt
        """
    )
    parser.add_argument('--monitors-dir', default=None,
                        help='Directory with Monitor*.txt files '
                             '(default: ../monitors_date_filtered)')
    parser.add_argument('--metadata', default=None,
                        help='metadata.txt path (default: ../metadata.txt)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory '
                             '(default: ../../analyses/analysis_results)')
    args = parser.parse_args()

    src_dir = Path(__file__).parent
    python_dir = src_dir.parent

    monitors_dir = (
        Path(args.monitors_dir) if args.monitors_dir
        else python_dir / 'monitors_date_filtered'
    )
    metadata_path = (
        Path(args.metadata) if args.metadata
        else python_dir / 'metadata.txt'
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else python_dir.parent / 'analyses' / 'analysis_results'
    )

    if not monitors_dir.exists():
        print(f"ERROR: monitors directory not found: {monitors_dir}")
        sys.exit(1)
    if not metadata_path.exists():
        print(f"ERROR: metadata file not found: {metadata_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TTD RANDOM FOREST REGRESSOR")
    print("=" * 60)
    print(f"Monitors: {monitors_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Output:   {output_dir}")

    # Build feature matrix
    df = build_feature_matrix(monitors_dir, metadata_path)

    if len(df) == 0:
        print("ERROR: No valid windows generated. Check monitor files and metadata.")
        sys.exit(1)

    feat_path = output_dir / 'windowed_features.csv'
    df.to_csv(feat_path, index=False)
    print(f"\nSaved: {feat_path}  ({len(df)} rows, {df['fly_id'].nunique()} flies)")

    # Run models A and B
    all_results = {}
    feature_cols_per_model = {}

    for model_name in ['A', 'B']:
        feat_cols = MODEL_FEATURE_SETS[model_name]
        print(f"\nModel {model_name}: {len(feat_cols)} features")
        metrics, importances, preds = run_cv(df, feat_cols, model_name)
        all_results[model_name] = (metrics, importances, preds)
        feature_cols_per_model[model_name] = feat_cols

    # Model C
    print(f"\nModel C: Model B + one-hot genotype")
    c_metrics, c_importances, c_preds, c_cols = run_model_c(df)
    all_results['C'] = (c_metrics, c_importances, c_preds)
    feature_cols_per_model['C'] = c_cols

    # Save outputs
    print()
    save_performance(all_results, output_dir)
    save_importances(all_results, feature_cols_per_model, output_dir)
    save_scatter(all_results, output_dir)

    print("\n" + "=" * 60)
    print("DONE — outputs in:", output_dir)
    print("=" * 60)


if __name__ == '__main__':
    main()
