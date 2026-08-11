"""h48_conservative의 min_tp/min_sl floor를 스윕해서, quality_head 타겟이 direction_head 타겟
(zigzag_action)과 스케일 상 얼마나 맞는지 실증적으로 확인한다. tp_mult=1.2/sl_mult=0.8(배포된
conservative 배율), horizon=48은 고정 -- floor 자체만 바꾼다. TP:SL 비율(1.5)은 유지.

핵심 지표(단순 CASH 비율보다 우선):
  - coverage: zigzag가 활성(zigzag_action!=0)인 bar 중, h48 tb_action도 활성인 비율
    (너무 낮으면 quality가 진짜 스윙을 놓친다는 뜻)
  - agreement: 둘 다 활성인 bar 중 방향이 일치하는 비율
    (너무 낮으면 quality가 방향과 무관한 노이즈에 반응한다는 뜻 -- 지금 0.4%/0.6%가 의심되는 지점)
  - specificity: zigzag가 비활성(CASH)인 bar 중 h48도 CASH인 비율
    (너무 낮으면 quality가 진짜 스윙 아닌 구간에서도 계속 거래하라고 한다는 뜻)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_PATH = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_2026.csv"
OUT_CSV = ROOT / "tmp/eth_h48qual_oracle_label_check_20260811/floor_sweep_zigzag_match.csv"

FEE_RATE, SLIP_RATE = 0.0005, 0.0002
FEE_COST = (FEE_RATE + SLIP_RATE) * 2.0 * 3.0
HORIZON, TP_MULT, SL_MULT = 48, 1.2, 0.8
TP_SL_RATIO = 1.5  # = TP_MULT / SL_MULT, 배포된 conservative 비율 유지
SL_FLOOR_GRID = [0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020]  # 0.004=현재 배포값


def _atr_price_move(df: pd.DataFrame) -> np.ndarray:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr / np.where(close != 0, close, np.nan)).rolling(96, min_periods=24).mean().shift(1)
    return atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)


def _reason_and_return(side, entry, fh, fl, fc, tp_move, sl_move):
    if entry <= 0.0:
        return 0.0, "invalid_entry", 0.0
    if side > 0:
        tp_level, sl_level = entry * (1.0 + tp_move), entry * (1.0 - sl_move)
        rel_low = fl / entry - 1.0
        mae = float(np.nanmin(rel_low)) if len(rel_low) else 0.0
        for hi, lo in zip(fh, fl):
            if lo <= sl_level:
                return -float(sl_move), "sl", mae
            if hi >= tp_level:
                return float(tp_move), "tp", mae
        return float(fc[-1] / entry - 1.0), "timeout", mae
    tp_level, sl_level = entry * (1.0 - tp_move), entry * (1.0 + sl_move)
    rel_high = 1.0 - fl / entry
    mae = float(np.nanmin(rel_high)) if len(rel_high) else 0.0
    for hi, lo in zip(fh, fl):
        if hi >= sl_level:
            return -float(sl_move), "sl", mae
        if lo <= tp_level:
            return float(tp_move), "tp", mae
    return float(1.0 - fc[-1] / entry), "timeout", mae


def build_tb_action(df: pd.DataFrame, atr: np.ndarray, min_sl: float) -> np.ndarray:
    min_tp = min_sl * TP_SL_RATIO
    n = len(df)
    open_px = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    last_i = n - HORIZON - 2
    actions = np.zeros(n, dtype=np.int64)
    for i in range(max(last_i, 0)):
        entry_i = i + 1
        end_i = entry_i + HORIZON
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        tp_move = max(min_tp, TP_MULT * vol)
        sl_move = max(min_sl, SL_MULT * vol)
        fh, fl, fc = high[entry_i:end_i + 1], low[entry_i:end_i + 1], close[entry_i:end_i + 1]
        lr, lreas, lmae = _reason_and_return(1, entry, fh, fl, fc, tp_move, sl_move)
        sr, sreas, smae = _reason_and_return(-1, entry, fh, fl, fc, tp_move, sl_move)
        lq = lr - FEE_COST - 0.20 * max(-lmae, 0.0) - 0.003 * int(lreas == "sl")
        sq = sr - FEE_COST - 0.20 * max(-smae, 0.0) - 0.003 * int(sreas == "sl")
        if lq > 0.0 and lq >= sq:
            actions[i] = 1
        elif sq > 0.0:
            actions[i] = 2
    return actions


def main() -> int:
    df = pd.read_csv(LABEL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    zz = df["zigzag_action"].to_numpy()
    atr = _atr_price_move(df)
    n = len(df)

    rows = []
    for min_sl in SL_FLOOR_GRID:
        min_tp = min_sl * TP_SL_RATIO
        tb = build_tb_action(df, atr, min_sl)
        valid = np.arange(n) < (n - HORIZON - 2)
        zz_v, tb_v = zz[valid], tb[valid]

        zz_active = zz_v != 0
        tb_active = tb_v != 0
        coverage = float((tb_active & zz_active).sum() / max(zz_active.sum(), 1))
        both_active = zz_active & tb_active
        zz_side = np.where(zz_v == 1, 1, np.where(zz_v == 2, -1, 0))
        tb_side = np.where(tb_v == 1, 1, np.where(tb_v == 2, -1, 0))
        agreement = float((zz_side[both_active] == tb_side[both_active]).mean()) if both_active.sum() else float("nan")
        specificity = float((~tb_active)[~zz_active].mean()) if (~zz_active).sum() else float("nan")
        cash_rate = float((tb_v == 0).mean())

        rows.append({
            "min_sl": min_sl, "min_tp": round(min_tp, 4),
            "cash_rate": round(cash_rate, 4),
            "coverage_vs_zigzag_active": round(coverage, 4),
            "direction_agreement_when_both_active": round(agreement, 4),
            "specificity_vs_zigzag_cash": round(specificity, 4),
            "n_both_active": int(both_active.sum()),
        })
        print(f"min_sl={min_sl*100:5.2f}%  min_tp={min_tp*100:5.2f}%  CASH={cash_rate*100:5.1f}%  "
              f"coverage={coverage*100:5.1f}%  방향일치={agreement*100:5.1f}%  specificity={specificity*100:5.1f}%  "
              f"(둘다활성 n={int(both_active.sum())})", flush=True)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n저장: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
