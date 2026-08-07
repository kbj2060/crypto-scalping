"""Three tuning levers for the ETH 1m scalp model, run after the base result
(scripts/train_eval_scalp_1m_hgb_20260716.py) showed OOS hit rate > 50% but negative PnL after
fees on every variant -- i.e. the classifier has real edge but the fee/TP-magnitude ratio kills
it. Each lever tests a different fix for that specific problem, reusing the base script's data
loading / split / feature-set helpers rather than duplicating them:

  1. WIDE   - does widening TP/SL + lengthening the hold (60min vs 20min, ~2.7x wider TP/SL)
              reduce fee drag as a fraction of the move? Uses
              build_scalp_1m_tb_labels_wide_20260716.py's labels, same A/B2 configs otherwise.
  2. CONF   - does filtering to only the classifier's highest-confidence predictions (skip low
              max-proba rows) raise realized hit rate enough to flip PnL positive? Threshold
              swept on val only, then applied as-is to OOS (no OOS-side selection).
  3. MAKER  - how much of the loss is fee structure vs. genuine lack of edge? Recomputes the
              *same* trained models' backtest PnL at an assumed maker fee (~2bps/side) instead of
              the taker fee (~4.5bps/side) used in the base run -- no retrain, isolates the
              fee-assumption effect from the model itself.

Fresh-forward compliance: same as the base script -- OOS never touches fitting or threshold
selection; CONF's threshold is chosen on val and applied unchanged to OOS.

Output: data/ensemble/reports/scalp_1m_tune_{wide,conf,maker}_20260716.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import (
    BASE_CSV, MICRO_CSV, MICROSTRUCTURE_COLS, HGB_PARAMS,
    load_experiment, feature_cols_for, split_by_date,
)

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
LABELS_WIDE_CSV = os.path.join(DATA_DIR, 'training_features_1m_scalp_labels_wide.csv')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

TAKER_FEE_PER_SIDE = 0.00045
MAKER_FEE_PER_SIDE = 0.0002
CONF_THRESHOLDS = [0.34, 0.40, 0.45, 0.50, 0.55, 0.60]

COMPLIANCE = {
    'fresh_forward_bar_by_bar': True,
    'trade_ledgers_used_as_input': False,
    'saved_parent_exit_timestamps_used': False,
    'future_rows_used_for_entry': False,
}


def load_experiment_wide(csv_path: str, ts_start: str, ts_end: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df[(df['timestamp'] >= ts_start) & (df['timestamp'] <= ts_end)].reset_index(drop=True)
    labels = pd.read_csv(LABELS_WIDE_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    return df


def backtest_replay(df: pd.DataFrame, pred_action: np.ndarray, fee_per_side: float) -> dict:
    mask = pred_action != 'CASH'
    n_trades = int(mask.sum())
    if n_trades == 0:
        return {'n_trades': 0, 'hit_rate': None, 'avg_pnl_pct': None, 'total_pnl_pct': None}
    correct = (pred_action[mask] == df.loc[mask, 'scalp_action'].to_numpy())
    tp_move = df.loc[mask, 'scalp_tp_move'].to_numpy()
    sl_move = df.loc[mask, 'scalp_sl_move'].to_numpy()
    raw_pnl = np.where(correct, tp_move, -sl_move)
    net_pnl = raw_pnl - 2 * fee_per_side
    return {
        'n_trades': n_trades,
        'hit_rate': float(correct.mean()),
        'avg_pnl_pct': float(net_pnl.mean()),
        'total_pnl_pct': float(net_pnl.sum()),
    }


def fit_model(train: pd.DataFrame, feat_cols: list[str]) -> HistGradientBoostingClassifier:
    X_train = train[feat_cols].fillna(0.0)
    y_train = train['scalp_action']
    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    clf.fit(X_train, y_train)
    return clf


def predict_with_threshold(clf, X: pd.DataFrame, threshold: float) -> np.ndarray:
    proba = clf.predict_proba(X)
    classes = clf.classes_
    max_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), max_idx]
    pred = classes[max_idx].copy()
    pred = np.where(max_proba >= threshold, pred, 'CASH')
    return pred


# ---------------------------------------------------------------------------
# Lever 1: WIDE (60min horizon, ~2.7x wider TP/SL)
# ---------------------------------------------------------------------------
def run_wide():
    print(f"\n{'=' * 60}\nLever 1: WIDE (60min horizon, wider ATR-scaled TP/SL)\n{'=' * 60}")
    results = {}

    df_a = load_experiment_wide(BASE_CSV, '2024-01-01', '2026-07-12')
    feat_cols = feature_cols_for(df_a, [])
    train, val, oos = split_by_date(df_a, '2026-04-30', '2026-05-31', '2026-07-12')
    print(f"A_wide: train={len(train):,} val={len(val):,} oos={len(oos):,}")
    clf = fit_model(train, feat_cols)
    results['A_wide'] = _eval_split(clf, feat_cols, val, oos, TAKER_FEE_PER_SIDE)

    df_b = load_experiment_wide(MICRO_CSV, '2026-05-03', '2026-07-12')
    feat_cols_b = feature_cols_for(df_b, MICROSTRUCTURE_COLS)
    train_b, val_b, oos_b = split_by_date(df_b, '2026-06-20', '2026-06-30', '2026-07-12')
    print(f"B2_wide: train={len(train_b):,} val={len(val_b):,} oos={len(oos_b):,}")
    clf_b = fit_model(train_b, feat_cols_b)
    results['B2_wide'] = _eval_split(clf_b, feat_cols_b, val_b, oos_b, TAKER_FEE_PER_SIDE)

    results['compliance'] = COMPLIANCE
    results['note'] = ('60-bar/1h horizon, TP_ATR_MULT=2.0/SL_ATR_MULT=1.3, bounds ~2.7x wider than '
                        'the base 20-bar labels. Same splits as the base A/B2 experiments.')
    _save('scalp_1m_tune_wide_20260716.json', results)
    return results


def _eval_split(clf, feat_cols, val, oos, fee_per_side):
    out = {}
    for name, split_df in [('val', val), ('oos', oos)]:
        if len(split_df) == 0:
            continue
        X = split_df[feat_cols].fillna(0.0)
        y = split_df['scalp_action']
        pred = clf.predict(X)
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        bt = backtest_replay(split_df.reset_index(drop=True), pred, fee_per_side)
        print(f"  [{name}] macro_f1={report['macro avg']['f1-score']:.3f} trades={bt['n_trades']:,} "
              f"hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")
        out[name] = {'classification_report': report, 'backtest': bt}
    return out


# ---------------------------------------------------------------------------
# Lever 2: CONF (confidence-threshold filtering) + Lever 3: MAKER (fee sensitivity)
# reuse the same A / B2 base-label models, so trained once and shared.
# ---------------------------------------------------------------------------
def run_conf_and_maker():
    print(f"\n{'=' * 60}\nLever 2+3: CONF (confidence threshold) & MAKER (fee sensitivity)\n{'=' * 60}")
    conf_results, maker_results = {}, {}

    configs = [
        ('A', load_experiment(BASE_CSV, '2024-01-01', '2026-07-12', []),
         '2026-04-30', '2026-05-31', '2026-07-12', []),
        ('B2', load_experiment(MICRO_CSV, '2026-05-03', '2026-07-12', MICROSTRUCTURE_COLS),
         '2026-06-20', '2026-06-30', '2026-07-12', MICROSTRUCTURE_COLS),
    ]

    for name, df, train_end, val_end, oos_end, extra_cols in configs:
        print(f"\n--- {name} ---")
        feat_cols = feature_cols_for(df, extra_cols)
        train, val, oos = split_by_date(df, train_end, val_end, oos_end)
        clf = fit_model(train, feat_cols)

        # Lever 2: sweep threshold on val, apply best (by val total_pnl_pct among >=20 trades) to OOS.
        X_val = val[feat_cols].fillna(0.0)
        sweep = []
        for thr in CONF_THRESHOLDS:
            pred_val = predict_with_threshold(clf, X_val, thr)
            bt_val = backtest_replay(val.reset_index(drop=True), pred_val, TAKER_FEE_PER_SIDE)
            sweep.append({'threshold': thr, **bt_val})
            print(f"  [val thr={thr}] trades={bt_val['n_trades']:,} hit_rate={bt_val['hit_rate']} "
                  f"total_pnl_pct={bt_val['total_pnl_pct']}")
        viable = [s for s in sweep if s['n_trades'] and s['n_trades'] >= 20]
        best = max(viable, key=lambda s: s['total_pnl_pct']) if viable else sweep[0]
        best_thr = best['threshold']
        print(f"  -> best val threshold: {best_thr}")

        X_oos = oos[feat_cols].fillna(0.0)
        pred_oos_thr = predict_with_threshold(clf, X_oos, best_thr)
        bt_oos_thr = backtest_replay(oos.reset_index(drop=True), pred_oos_thr, TAKER_FEE_PER_SIDE)
        print(f"  [oos @ thr={best_thr}] trades={bt_oos_thr['n_trades']:,} hit_rate={bt_oos_thr['hit_rate']} "
              f"total_pnl_pct={bt_oos_thr['total_pnl_pct']}")
        conf_results[name] = {
            'val_sweep': sweep, 'chosen_threshold': best_thr,
            'oos_at_chosen_threshold': bt_oos_thr,
        }

        # Lever 3: unfiltered (argmax) predictions from the SAME model, recompute PnL at maker fee.
        pred_val_unfiltered = clf.predict(X_val)
        pred_oos_unfiltered = clf.predict(X_oos)
        maker_results[name] = {
            'val': {
                'taker_fee': backtest_replay(val.reset_index(drop=True), pred_val_unfiltered, TAKER_FEE_PER_SIDE),
                'maker_fee': backtest_replay(val.reset_index(drop=True), pred_val_unfiltered, MAKER_FEE_PER_SIDE),
            },
            'oos': {
                'taker_fee': backtest_replay(oos.reset_index(drop=True), pred_oos_unfiltered, TAKER_FEE_PER_SIDE),
                'maker_fee': backtest_replay(oos.reset_index(drop=True), pred_oos_unfiltered, MAKER_FEE_PER_SIDE),
            },
        }
        print(f"  [oos maker-fee] total_pnl_pct={maker_results[name]['oos']['maker_fee']['total_pnl_pct']} "
              f"(taker was {maker_results[name]['oos']['taker_fee']['total_pnl_pct']})")

    conf_results['compliance'] = COMPLIANCE
    conf_results['note'] = ('Confidence threshold chosen on val by max total_pnl_pct among thresholds with '
                             '>=20 val trades; applied unchanged to OOS (no OOS-side selection).')
    _save('scalp_1m_tune_conf_20260716.json', conf_results)

    maker_results['compliance'] = COMPLIANCE
    maker_results['note'] = (f'Same trained models/predictions as the unfiltered baseline (argmax, no '
                              f'confidence filter); only the assumed round-trip fee changes '
                              f'({TAKER_FEE_PER_SIDE*2:.4%} taker vs {MAKER_FEE_PER_SIDE*2:.4%} maker). '
                              f'No retrain -- isolates fee-assumption effect from the model itself.')
    _save('scalp_1m_tune_maker_20260716.json', maker_results)

    return conf_results, maker_results


def _save(fname: str, obj: dict):
    with open(os.path.join(REPORT_DIR, fname), 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Saved {fname}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    run_wide()
    run_conf_and_maker()


if __name__ == '__main__':
    main()
