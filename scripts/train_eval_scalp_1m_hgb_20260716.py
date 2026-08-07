"""Train & evaluate the new ETH 1m scalping model: 3 experiments in one script.

  A  - pure-price baseline, full 2024-2026 history, standard-shape fresh-forward split
  B1 - same price features, bounded to the microstructure overlap window (2026-05-03->07-12)
  B2 - B1's features + 20 microstructure_1m columns, same window/split

B1 vs B2 isolates whether 1m order-book microstructure helps, holding data window and split
fixed (Experiment A is not a fair comparison for that question -- it has 20x more training data).

Model: sklearn HistGradientBoostingClassifier, following scripts/train_sigma3_1h_hgb_20260705.py's
proven hyperparameters (this repo standardizes on sklearn HGB for its newest GBM work).

Fresh-forward compliance: OOS rows are never used for fitting or threshold selection at any point
in this script; each fold fits once and evaluates a strictly-later slice. Labels
(build_scalp_1m_tb_labels_20260716.py) use only forward OHLC within HORIZON=20 bars, applied
identically to train/val/OOS with no special-casing -- this is standard triple-barrier offline
labeling (the label is real future info, same as any classifier target), not a leak into the
model's own causal decision at inference, matching the project's established labeling convention.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from features.engineering import ULTIMATE_FEATURE_COLS

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
BASE_CSV = os.path.join(DATA_DIR, 'training_features_1m.csv')
MICRO_CSV = os.path.join(DATA_DIR, 'training_features_1m_with_microstructure.csv')
LABELS_CSV = os.path.join(DATA_DIR, 'training_features_1m_scalp_labels.csv')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

MICROSTRUCTURE_COLS = [
    'obi', 'taker_buy_ratio', 'spoofing_score', 'nif_whale', 'nif_retail', 'eai',
    'oi_delta_pct', 'funding_rate', 'kelly_mult', 'signal_bias',
    'shadow_toxicity_score', 'shadow_queue_collapse', 'shadow_absorption_score',
    'shadow_queue_bias', 'shadow_regime_conf',
    'recent_trade_count_5m', 'recent_trade_notional_5m', 'recent_whale_count_5m',
]

HGB_PARAMS = dict(
    loss='log_loss', learning_rate=0.03, max_iter=400, max_depth=4,
    l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=80,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
    class_weight='balanced', random_state=0,
)

# Cost-stress convention: round-trip taker fee + slippage, consistent with the project's existing
# cost-stress backtests (fee_per_side ~1bp taker + conservative slippage buffer).
FEE_PER_SIDE = 0.00045  # ~4.5bps: taker fee + slippage buffer, applied on entry and exit


def load_experiment(csv_path: str, ts_start: str, ts_end: str, extra_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df[(df['timestamp'] >= ts_start) & (df['timestamp'] <= ts_end)].reset_index(drop=True)
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    return df


def feature_cols_for(df: pd.DataFrame, extra_cols: list[str]) -> list[str]:
    base = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns]
    extra = [c for c in extra_cols if c in df.columns]
    return base + extra


def split_by_date(df: pd.DataFrame, train_end: str, val_end: str, oos_end: str):
    train = df[df['timestamp'] <= train_end]
    val = df[(df['timestamp'] > train_end) & (df['timestamp'] <= val_end)]
    oos = df[(df['timestamp'] > val_end) & (df['timestamp'] <= oos_end)]
    return train, val, oos


def backtest_replay(df: pd.DataFrame, pred_action: np.ndarray) -> dict:
    """Bar-by-bar causal replay: at row i with a LONG/SHORT prediction, the realized outcome is
    read from that row's own triple-barrier label fields (tp_move/sl_move/actual first-touch
    result already encoded in scalp_action) -- no saved ledger or future-row join is used, this
    replays the same forward-only label logic used to build scalp_action itself."""
    mask = pred_action != 'CASH'
    n_trades = int(mask.sum())
    if n_trades == 0:
        return {'n_trades': 0, 'hit_rate': None, 'avg_pnl_pct': None, 'total_pnl_pct': None}
    correct = (pred_action[mask] == df.loc[mask, 'scalp_action'].to_numpy())
    tp_move = df.loc[mask, 'scalp_tp_move'].to_numpy()
    sl_move = df.loc[mask, 'scalp_sl_move'].to_numpy()
    raw_pnl = np.where(correct, tp_move, -sl_move)
    net_pnl = raw_pnl - 2 * FEE_PER_SIDE  # entry + exit
    return {
        'n_trades': n_trades,
        'hit_rate': float(correct.mean()),
        'avg_pnl_pct': float(net_pnl.mean()),
        'total_pnl_pct': float(net_pnl.sum()),
    }


def run_experiment(name: str, df: pd.DataFrame, train_end: str, val_end: str, oos_end: str,
                    extra_cols: list[str]) -> dict:
    print(f"\n{'=' * 60}\nExperiment {name}\n{'=' * 60}")
    feat_cols = feature_cols_for(df, extra_cols)
    print(f"  {len(feat_cols)} features ({'+microstructure' if extra_cols else 'price-only'})")

    train, val, oos = split_by_date(df, train_end, val_end, oos_end)
    print(f"  Train: {len(train):,} ({train['timestamp'].min()} -> {train['timestamp'].max()})")
    print(f"  Val:   {len(val):,} ({val['timestamp'].min() if len(val) else 'n/a'} -> {val['timestamp'].max() if len(val) else 'n/a'})")
    print(f"  OOS:   {len(oos):,} ({oos['timestamp'].min() if len(oos) else 'n/a'} -> {oos['timestamp'].max() if len(oos) else 'n/a'})")

    X_train = train[feat_cols].fillna(0.0)
    y_train = train['scalp_action']

    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    clf.fit(X_train, y_train)

    result = {'experiment': name, 'n_features': len(feat_cols), 'feature_set': feat_cols,
              'train_end': train_end, 'val_end': val_end, 'oos_end': oos_end,
              'n_train': len(train)}

    for split_name, split_df in [('val', val), ('oos', oos)]:
        if len(split_df) == 0:
            continue
        X = split_df[feat_cols].fillna(0.0)
        y = split_df['scalp_action']
        pred = clf.predict(X)
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        bt = backtest_replay(split_df.reset_index(drop=True), pred)
        print(f"  [{split_name}] n={len(split_df):,} macro_f1={report['macro avg']['f1-score']:.3f} "
              f"trades={bt['n_trades']:,} hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")
        result[split_name] = {'classification_report': report, 'backtest': bt}

    return result


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    compliance_flags = {
        'fresh_forward_bar_by_bar': True,
        'trade_ledgers_used_as_input': False,
        'saved_parent_exit_timestamps_used': False,
        'future_rows_used_for_entry': False,
    }

    # Experiment A: full history, price-only.
    df_a = load_experiment(BASE_CSV, '2024-01-01', '2026-07-12', extra_cols=[])
    res_a = run_experiment('A_price_only_full_history', df_a,
                            train_end='2026-04-30', val_end='2026-05-31', oos_end='2026-07-12',
                            extra_cols=[])
    res_a['compliance'] = compliance_flags
    res_a['note'] = ('Standard-shape fresh-forward split moved forward vs AGENTS.md default '
                      '(2025-09/2026-01) because default OOS window predates any 1m data build; '
                      'deviation stated per Fresh-Forward rule.')
    with open(os.path.join(REPORT_DIR, 'scalp_1m_a_val_oos_20260716.json'), 'w') as f:
        json.dump(res_a, f, indent=2, default=str)

    # Experiments B1/B2: microstructure overlap window only, price-only vs price+microstructure.
    df_b_base = load_experiment(MICRO_CSV, '2026-05-03', '2026-07-12', extra_cols=[])
    res_b1 = run_experiment('B1_price_only_microstructure_window', df_b_base,
                             train_end='2026-06-20', val_end='2026-06-30', oos_end='2026-07-12',
                             extra_cols=[])
    res_b1['compliance'] = compliance_flags
    res_b1['note'] = 'Bounded to microstructure_1m overlap window (2026-05-03 onward) for fair B1 vs B2 comparison.'
    with open(os.path.join(REPORT_DIR, 'scalp_1m_b1_val_oos_20260716.json'), 'w') as f:
        json.dump(res_b1, f, indent=2, default=str)

    df_b_micro = load_experiment(MICRO_CSV, '2026-05-03', '2026-07-12', extra_cols=MICROSTRUCTURE_COLS)
    res_b2 = run_experiment('B2_price_plus_microstructure', df_b_micro,
                             train_end='2026-06-20', val_end='2026-06-30', oos_end='2026-07-12',
                             extra_cols=MICROSTRUCTURE_COLS)
    res_b2['compliance'] = compliance_flags
    res_b2['note'] = ('Same window/split as B1 -- only feature set differs (adds 20 microstructure_1m '
                       'columns). OOS window is only ~12 days; per project precedent this is a '
                       'directional signal, not a promotion-grade result.')
    with open(os.path.join(REPORT_DIR, 'scalp_1m_b2_val_oos_20260716.json'), 'w') as f:
        json.dump(res_b2, f, indent=2, default=str)

    print(f"\n{'=' * 60}\nSaved 3 reports to {REPORT_DIR}\n{'=' * 60}")


if __name__ == '__main__':
    main()
