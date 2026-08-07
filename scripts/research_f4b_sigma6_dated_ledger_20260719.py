"""F4-B 준비: Sigma6 OOS 원장을 날짜 정보 포함해서 재생성.

원본 scripts/run_sigma6_regime_trend_20260705.py는 trades에 'month' 문자열만 남기고
정확한 entry/exit 타임스탬프를 버린다. 이 스크립트는 원본을 수정하지 않고, 동일한
backtest() 로직을 복제 + entry/exit timestamp 필드만 추가해서 재실행한다.
설정은 docs/model_contracts/sigma6_regime_trend_20260705_contract.md에 명시된
frozen 프로덕션 설정 그대로 사용 (val_regime_frontier.csv로 재확인 완료).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_sigma6_regime_trend_20260705 as sigma6  # noqa: E402
import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

OUT_DIR = ROOT / "data/research"
OOS_START, OOS_END = sigma6.OOS_START, sigma6.OOS_END
PFX = sigma6.PFX

CONFIGS = {
    "sigma6_lev4": dict(thr=0.70, leverage=4.0, margin=0.30, trail_atr=5.0, sl_atr=2.5,
                         min_profit_atr=2.0, max_hold=144, cooldown=3,
                         reg_mode="not_chop", reg_thr=0.42, stab_thr=0.55, fee_mult=1.0),
    "sigma6_lev3": dict(thr=0.70, leverage=3.0, margin=0.30, trail_atr=5.0, sl_atr=2.5,
                         min_profit_atr=2.0, max_hold=144, cooldown=3,
                         reg_mode="not_chop", reg_thr=0.50, stab_thr=0.55, fee_mult=1.0),
}


def backtest_with_dates(tape, *, leverage, margin, trail_atr, sl_atr, min_profit_atr, max_hold,
                         cooldown, reg_mode, reg_thr, stab_thr, fee_mult, start, end):
    """Exact copy of sigma6.backtest() logic, extended to record entry/exit timestamps."""
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    close = sub["close"].to_numpy(np.float64)
    open_ = sub["open"].to_numpy(np.float64)
    side_arr = sub["primary_side"].to_numpy(np.int64)
    atr_arr = sub["atr_pct"].to_numpy(np.float64)
    bull = sub[f"{PFX}bull_prob"].to_numpy(np.float64)
    bear = sub[f"{PFX}bear_prob"].to_numpy(np.float64)
    chop = sub[f"{PFX}chop_prob"].to_numpy(np.float64)
    stab = sub["regime3_cmamba_h6_sidecar_stability_score"].fillna(1.0).to_numpy(np.float64)
    FEE, SLIP = 0.00020 * fee_mult, 0.00050 * fee_mult
    notional = margin * leverage
    cash = peak_eq = 1.0
    mdd = 0.0
    pos = 0
    entry_price = peak_unreal = entry_atr = 0.0
    hold_start = 0
    entry_equity = 1.0
    trades = []
    cooldown_until = -1
    i = 0
    while i < n - 1:
        if pos == 0:
            if i < cooldown_until or side_arr[i] == 0:
                i += 1
                continue
            side = int(side_arr[i])
            ok = True
            if reg_mode == "trend_agree":
                ok = (side > 0 and bull[i] >= reg_thr) or (side < 0 and bear[i] >= reg_thr)
            elif reg_mode == "not_chop":
                ok = chop[i] < reg_thr
            if ok and stab_thr > 0:
                ok = stab[i] >= stab_thr
            if not ok:
                i += 1
                continue
            entry_price = float(open_[min(i + 1, n - 1)]) * (1 + SLIP if side > 0 else 1 - SLIP)
            pos, hold_start, peak_unreal, entry_atr = side, i, 0.0, max(atr_arr[i], 1e-6)
            entry_equity = cash
            cash -= cash * FEE * notional
            i += 1
            continue
        px = close[i]
        raw = (px * (1 - SLIP) - entry_price) / entry_price if pos > 0 else (entry_price - px * (1 + SLIP)) / entry_price
        unreal = raw * notional
        eq = cash * (1 + unreal)
        peak_eq = max(peak_eq, eq)
        mdd = min(mdd, eq / max(peak_eq, 1e-12) - 1)
        peak_unreal = max(peak_unreal, unreal)
        hold = i - hold_start
        reason = ""
        if unreal <= -sl_atr * entry_atr:
            reason = "stop"
        elif peak_unreal >= min_profit_atr * entry_atr and (peak_unreal - unreal) >= trail_atr * entry_atr:
            reason = "trail"
        elif hold >= max_hold:
            reason = "time"
        if reason:
            exit_price = close[i] * (1 - SLIP if pos > 0 else 1 + SLIP)
            rex = (exit_price - entry_price) / entry_price if pos > 0 else (entry_price - exit_price) / entry_price
            before = cash
            cash = cash * (1 + rex * notional)
            cash -= before * FEE * notional
            trades.append({
                "win": cash > entry_equity,
                "entry_timestamp": sub.iloc[hold_start]["timestamp"],
                "exit_timestamp": sub.iloc[i]["timestamp"],
                "reason": reason, "ret": rex * notional,
                "trade_return_on_equity": cash / max(entry_equity, 1e-12) - 1.0,
            })
            pos = 0
            cooldown_until = i + cooldown
        i += 1
    wins = sum(1 for t in trades if t["win"])
    return {"pnl": (cash - 1) * 100, "mdd": mdd * 100, "trades": len(trades),
            "wr": wins / len(trades) if trades else 0.0}, pd.DataFrame(trades)


def main():
    raw = sigma6.load_tape_with_regime()
    for name, cfg in CONFIGS.items():
        thr = cfg.pop("thr")
        tape = v2.apply_quality_threshold(raw, thr)
        result, ledger = backtest_with_dates(tape, start=OOS_START, end=OOS_END, **cfg)
        print(name, result)
        out_csv = OUT_DIR / f"{name}_oos_dated_ledger_20260719.csv"
        ledger.to_csv(out_csv, index=False)
        print("wrote", out_csv, "rows:", len(ledger))


if __name__ == "__main__":
    main()
