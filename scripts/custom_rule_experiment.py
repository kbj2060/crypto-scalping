#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import sys
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ensemble.train_rl_dsac_agent import DSACRouter, GaussianActor, DSAC_STATE_DIM

def load_actor(path: str) -> GaussianActor:
    ckpt = torch.load(path, map_location='cpu')
    actor = GaussianActor(state_dim=int(ckpt.get('state_dim', DSAC_STATE_DIM)))
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    return actor

def compute_regime(row: pd.Series) -> str:
    regime_cols = [
        ('regime_bull', 'bull'),
        ('regime_bear', 'bear'),
        ('regime_chop', 'chop'),
        ('regime_whipsaw', 'whipsaw'),
        ('regime_normal', 'normal'),
    ]
    for col,name in regime_cols:
        if float(row.get(col, 0.0)) >= 0.5:
            return name
    return 'normal'

THRESHOLDS = {
    'bull': {'long': 0.15, 'short': 0.35},
    'bear': {'long': 0.35, 'short': 0.15},
    'normal': {'long': 0.22, 'short': 0.22},
    'chop': {'long': 0.40, 'short': 0.40},
    'whipsaw': {'long': 0.40, 'short': 0.40},
}

def hard_no_go(row: pd.Series, info: dict[str, Any]) -> bool:
    if int(row.get('m7_gate_block', 0)) == 1:
        return True
    if bool(row.get('m7_iso_anom', False)) and bool(row.get('m7_vae_anom', False)):
        return True
    if abs(float(row.get('jump_z', 0.0))) > 3.0:
        return True
    if int(row.get('evt_tail_flag', 0)) == 1:
        return True
    regime = compute_regime(row)
    long_std = float(info.get('long_std', 1.0))
    short_std = float(info.get('short_std', 1.0))
    if regime in ('chop','whipsaw') and long_std > 1.2 and short_std > 1.2:
        return True
    return False

def compute_direction(info: dict[str, Any], row: pd.Series) -> float:
    primary_raw = float(info.get('primary_raw', 0.0))
    long_logit = float(info.get('long_logit', 0.0))
    short_logit = float(info.get('short_logit', 0.0))
    long_std = max(float(info.get('long_std', 1.0)), 1e-6)
    short_std = max(float(info.get('short_std', 1.0)), 1e-6)
    avg_std = (long_std + short_std) / 2
    specialist_diff = (long_logit - short_logit) / avg_std
    m7_conf = float(row.get('m7_quality_pred', 0.0))
    m7_dir = float(row.get('m7_prob_up', 0.0)) - float(row.get('m7_prob_dn', 0.0))
    regime = compute_regime(row)
    regime_bias = 0.0
    if regime == 'bull':
        regime_bias = 1.0
    elif regime == 'bear':
        regime_bias = -1.0
    elif regime == 'normal':
        regime_bias = 0.2
    elif regime in ('chop','whipsaw'):
        regime_bias = 0.0
    direction = (
        0.35 * primary_raw
        + 0.25 * specialist_diff
        + 0.25 * m7_dir * m7_conf
        + 0.15 * regime_bias
    )
    return direction

def simulate(df: pd.DataFrame, router: DSACRouter) -> dict[str, float]:
    fee = 0.0005
    slip = 0.0002
    balance = 1.0
    trades = 0
    wins = 0
    pos = None
    entry = 0.0
    peak = 1.0
    highest = 1.0
    eq = [1.0]
    for i in range(len(df) - 1):
        row = df.iloc[i]
        features = row.drop(labels=['timestamp']).to_dict()
        stats = {k: float(v or 0.0) for k, v in features.items()}
        action_cont, lev_raw, info = router.decide(stats, {
            'type': pos,
            'entry_price': entry,
            'unrealized': 0.0,
            'mdd': 0.0,
            'hold_norm': 0.0,
        })
        if hard_no_go(row, info):
            target_action = 0
        else:
            direction = compute_direction(info, row)
            regime = compute_regime(row)
            table = THRESHOLDS.get(regime, THRESHOLDS['normal'])
            if direction >= table['long']:
                target_action = 1
            elif direction <= -table['short']:
                target_action = 2
            else:
                target_action = 0
        kelly = float(np.clip(lev_raw, 0.0, 1.0))
        # size adjustments
        quality = float(row.get('m7_quality_pred', 0.0))
        kelly *= float(np.clip(1.0 + 0.4 * quality, 0.6, 1.4))
        agreement = float(info.get('agreement', 0.0))
        if agreement >= 0.9:
            kelly *= 1.0
        elif agreement >= 0.5:
            kelly *= 0.75
        else:
            kelly *= 0.45
        kelly = float(np.clip(kelly, 0.0, 0.35))
        price = float(df.iloc[i + 1]['open'])
        if pos is None:
            if target_action == 1 and kelly > 0:
                pos = 'LONG'
                entry = price * (1 + slip)
                balance -= balance * fee * kelly
            elif target_action == 2 and kelly > 0:
                pos = 'SHORT'
                entry = price * (1 - slip)
                balance -= balance * fee * kelly
        else:
            pnl = 0.0
            exit_flag = False
            tp_offset = float(row.get('m7_tp_offset', 0.01))
            sl_offset = float(row.get('m7_sl_offset', 0.01))
            if pos == 'LONG':
                tp = entry * (1 + tp_offset)
                sl = max(entry * (1 - sl_offset), entry * (1 - 0.025))
                if price >= tp or price <= sl:
                    exit_flag = True
                    pnl = ((price * (1 - slip) - entry) / entry)
            else:
                tp = entry * (1 - tp_offset)
                sl = max(entry * (1 + sl_offset), entry * (1 + 0.025))
                if price <= tp or price >= sl:
                    exit_flag = True
                    pnl = ((entry - price * (1 + slip)) / entry)
            if exit_flag:
                pnl *= 0.5
                balance *= 1 + pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                pos = None
            elif target_action == 0 and abs(direction) < 0.05:
                pass
            elif target_action != (1 if pos == 'LONG' else 2):
                pnl = (price - entry) / entry if pos == 'LONG' else (entry - price) / entry
                balance *= 1 + pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                pos = None
        eq.append(balance)
    arr = np.array(eq)
    run_max = np.maximum.accumulate(arr)
    dd = arr / np.maximum(run_max, 1e-12) - 1.0
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(365 * 24 * 12)) if len(rets) > 2 and np.std(rets) > 0 else 0.0
    return {
        'pnl_pct': (arr[-1] - 1.0) * 100.0,
        'trades': trades,
        'wr': wins / trades * 100 if trades else 0.0,
        'mdd_pct': float(np.min(dd)) * 100 if len(dd) else 0.0,
        'sharpe': sharpe,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl-csv', default='data/splits/year_oos/rl_base_2025.csv')
    parser.add_argument('--ckpt', default='data/ensemble/ckpt/best_dsac_agents.pth')
    parser.add_argument('--split', default='2025H1')
    parser.add_argument('--limit', type=int, default=1500)
    args = parser.parse_args()
    df = pd.read_csv(args.rl_cvs if hasattr(args,'rl_cvs') else args.rl_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    mask = (df['timestamp'] >= '2025-01-01') & (df['timestamp'] <= '2025-06-30')
    subset = df[mask].head(args.limit).reset_index(drop=True)
    actor = load_actor(args.ckpt)
    router = DSACRouter(actor, device='cpu')
    metrics = simulate(subset, router)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
