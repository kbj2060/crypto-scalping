"""48bar 고정을 포기하고, floor(0.4%/0.6%, 앞선 스윕에서 항상 최선으로 확인됨)는 유지한 채
horizon만 4h~120h로 넓게 스윕해서 zigzag_action과의 방향일치가 어디서 최고인지 찾는다.
coverage/timeout비율도 같이 추적해서 "일치율은 높은데 표본이 거의 안 남는" 함정을 감시한다."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_PATH = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_2026.csv"
OUT_CSV = ROOT / "tmp/eth_h48qual_oracle_label_check_20260811/horizon_wide_sweep.csv"

FEE_RATE, SLIP_RATE = 0.0005, 0.0002
FEE_COST = (FEE_RATE + SLIP_RATE) * 2.0 * 3.0
TP_MULT, SL_MULT = 1.2, 0.8
MIN_SL, MIN_TP = 0.004, 0.006  # 이전 스윕에서 모든 horizon에 걸쳐 항상 최선이었던 floor, 고정
HORIZON_GRID = [48, 96, 144, 216, 288, 384, 480, 576, 720, 864, 1152, 1440]  # 4h ~ 120h(5일)


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


def build_tb_action(df: pd.DataFrame, atr: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    open_px = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    last_i = n - horizon - 2
    actions = np.zeros(n, dtype=np.int64)
    reasons = np.array([""] * n, dtype=object)
    for i in range(max(last_i, 0)):
        entry_i = i + 1
        end_i = entry_i + horizon
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        tp_move = max(MIN_TP, TP_MULT * vol)
        sl_move = max(MIN_SL, SL_MULT * vol)
        fh, fl, fc = high[entry_i:end_i + 1], low[entry_i:end_i + 1], close[entry_i:end_i + 1]
        lr, lreas, lmae = _reason_and_return(1, entry, fh, fl, fc, tp_move, sl_move)
        sr, sreas, smae = _reason_and_return(-1, entry, fh, fl, fc, tp_move, sl_move)
        lq = lr - FEE_COST - 0.20 * max(-lmae, 0.0) - 0.003 * int(lreas == "sl")
        sq = sr - FEE_COST - 0.20 * max(-smae, 0.0) - 0.003 * int(sreas == "sl")
        if lq > 0.0 and lq >= sq:
            actions[i] = 1
            reasons[i] = lreas
        elif sq > 0.0:
            actions[i] = 2
            reasons[i] = sreas
    return actions, reasons


def main() -> int:
    df = pd.read_csv(LABEL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    zz = df["zigzag_action"].to_numpy()
    atr = _atr_price_move(df)
    n = len(df)

    rows = []
    for horizon in HORIZON_GRID:
        tb, reasons = build_tb_action(df, atr, horizon)
        valid = np.arange(n) < (n - horizon - 2)
        zz_v, tb_v, reas_v = zz[valid], tb[valid], reasons[valid]

        zz_active = zz_v != 0
        tb_active = tb_v != 0
        coverage = float((tb_active & zz_active).sum() / max(zz_active.sum(), 1))
        both_active = zz_active & tb_active
        zz_side = np.where(zz_v == 1, 1, np.where(zz_v == 2, -1, 0))
        tb_side = np.where(tb_v == 1, 1, np.where(tb_v == 2, -1, 0))
        agreement = float((zz_side[both_active] == tb_side[both_active]).mean()) if both_active.sum() else float("nan")
        specificity = float((~tb_active)[~zz_active].mean()) if (~zz_active).sum() else float("nan")
        cash_rate = float((tb_v == 0).mean())
        timeout_rate = float((reas_v[tb_active] == "timeout").mean()) if tb_active.sum() else float("nan")

        rows.append({
            "horizon_bars": horizon, "horizon_hours": round(horizon * 5 / 60, 1),
            "cash_rate": round(cash_rate, 4), "coverage_vs_zigzag_active": round(coverage, 4),
            "direction_agreement_when_both_active": round(agreement, 4),
            "specificity_vs_zigzag_cash": round(specificity, 4),
            "timeout_rate_among_active": round(timeout_rate, 4),
            "n_both_active": int(both_active.sum()),
        })
        print(f"h={horizon:5d}bar({horizon*5/60:5.1f}h)  CASH={cash_rate*100:5.1f}%  coverage={coverage*100:5.1f}%  "
              f"방향일치={agreement*100:5.1f}%  specificity={specificity*100:5.1f}%  timeout={timeout_rate*100:5.1f}%  "
              f"(n={int(both_active.sum())})", flush=True)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n저장: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
