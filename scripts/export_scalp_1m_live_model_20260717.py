"""Exports the trained ETH 1m scalp primary model as a persisted artifact for live use, so the
live runner doesn't retrain on every restart. Trained on ALL available history through the end of
the current dataset (2026-07-12) -- standard "train on everything available, deploy forward" handoff,
matching how every other live model in this project (Omega4.6.1 etc.) is versioned.

Bundles: the fitted HistGradientBoostingClassifier, the exact feature column list (order matters),
the confidence threshold, and the triple-barrier label parameters (HORIZON/ATR/TP-SL) needed to
reproduce the same entry logic live -- so the runtime doesn't need to re-derive any of this from
scratch or risk drifting from what was actually backtested.

Output: data/ensemble/ckpt/scalp_1m_eth_live_v1_20260717.pkl (+ a sibling .json metadata file)
"""
import json
import os
import pickle
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import BASE_CSV, feature_cols_for, HGB_PARAMS
from tune_scalp_1m_levers_20260716 import fit_model
from simulate_maker_entry_scalp_1m_20260716 import LABELS_CSV, OFFSET, FILL_LOOKAHEAD, MAKER_FEE, TAKER_FEE, HORIZON
from build_scalp_1m_tb_labels_20260716 import ATR_LOOKBACK, TP_ATR_MULT, SL_ATR_MULT, TP_BOUNDS, SL_BOUNDS

CKPT_DIR = os.path.join(_ROOT_DIR, 'data', 'ensemble', 'ckpt')
MODEL_ID = 'scalp_1m_eth_live_v1_20260717'

FIXED_THRESHOLD = 0.55  # backtested/walk-forward-validated value; see project memory before changing


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print("Loading full 1m ETH feature + label history...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    print(f"  {len(df):,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    feat_cols = feature_cols_for(df, [])
    print(f"Training final primary model on ALL {len(df):,} available rows ({len(feat_cols)} features)...")
    clf = fit_model(df, feat_cols)

    model_path = os.path.join(CKPT_DIR, f'{MODEL_ID}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': clf, 'feature_cols': feat_cols}, f)

    meta = {
        'model_id': MODEL_ID,
        'trained_through': str(df['timestamp'].max()),
        'n_train_rows': len(df),
        'n_features': len(feat_cols),
        'feature_cols': feat_cols,
        'hgb_params': HGB_PARAMS,
        'confidence_threshold': FIXED_THRESHOLD,
        'entry_fill_model': {
            'offset_bp': OFFSET * 10000, 'fill_lookahead_min': FILL_LOOKAHEAD,
            'maker_fee': MAKER_FEE, 'taker_fee_exit': TAKER_FEE,
        },
        'triple_barrier_label': {
            'horizon_min': HORIZON, 'atr_lookback': ATR_LOOKBACK,
            'tp_atr_mult': TP_ATR_MULT, 'sl_atr_mult': SL_ATR_MULT,
            'tp_bounds': TP_BOUNDS, 'sl_bounds': SL_BOUNDS,
        },
        'backtest_reference': {
            'walkforward_7fold_clean_threshold_0.55_mean_pnl_pct': 3.41,
            'walkforward_7fold_clean_threshold_0.55_all_positive': True,
            'exposure_capped_note': ('Position sizing is NOT baked into this model artifact -- '
                                      'the live runner applies per_trade_pct/max_total_exposure_pct '
                                      'independently at execution time.'),
        },
        'promotion_status': ('NOT promoted -- has not passed CLAUDE.md Omega Artifact Integrity '
                              'Promotion Gate; no live shadow-tracking done yet. Backtest-validated '
                              'only, per project memory project-eth-1m-scalping-microstructure-20260716.md.'),
    }
    meta_path = os.path.join(CKPT_DIR, f'{MODEL_ID}.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == '__main__':
    main()
