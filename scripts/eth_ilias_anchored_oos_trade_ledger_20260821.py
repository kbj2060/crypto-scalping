#!/usr/bin/env python3
"""zigzag/h48qual/cusum(ilias_anchored, TRAIN=2024+2025/EVAL=2026-07-01~08-19, N=6 시드) 전체
18개 run의 OOS 트레이드 원장 재구성. `_metrics_with_ledger`는
`scripts/eth_zigzag_h48qual_cusum_oos_trade_ledger_20260821.py`(이미 report.json과 소수점까지
cross-check 검증된 스크립트)의 함수를 그대로 이식(재구현 아님) -- 유일한 차이는 대상
run 디렉토리/omega.EVAL_CSV 오버라이드가 이 세션의 ilias_anchored full-variant 것이라는 점.
모델 재로드/재추론 불필요 -- 이미 저장된 oos_predictions_qXXX.csv만 읽는다."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))

os.environ["ILIAS_EVAL_VARIANT"] = "full"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eth_directional_change_tabm_training_ilias_anchored_20260821 as ilias_wrapper  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402

omega = ilias_wrapper.omega
omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

SEEDS = [133725056, 325805917, 775149439, 126593178, 286919795, 310216042]
LABELS = ["zigzag", "h48qual", "cusum"]
OUT_ROOT = ROOT / "tmp/causal_regen_20260516"
MODEL_ID = "omega4_3head_parent72_loose_entry_quality_20260620"
LEDGER_OUT_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/oos_trade_ledgers"

FEE, SLIP = omega._load_fee_slip()
COST_MULT = 3.0


def _metrics_with_ledger(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> tuple[dict, pd.DataFrame]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    ts = pd.to_datetime(frame["timestamp"]).to_numpy()
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    cooldown = 0
    next_cooldown = 0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    reasons: dict[str, int] = {}
    ledger: list[dict] = []
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            eq = cash * (1.0 + unreal)
        else:
            unreal = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            hold = int(i) - int(entry_idx)
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "max_hold"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trades += 1
                win = cash > entry_equity
                wins += int(win)
                reasons[reason] = reasons.get(reason, 0) + 1
                ledger.append({
                    "entry_ts": ts[entry_idx], "side": "LONG" if pos > 0 else "SHORT", "entry_price": entry_price,
                    "exit_ts": ts[i], "exit_price": float(exit_px), "exit_reason": reason, "hold_bars": hold,
                    "notional": notional, "leverage": leverage, "trade_pnl_pct": float((cash / before - 1.0) * 100.0),
                    "equity_after": float(cash), "win": bool(win),
                })
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = px
        entry_equity = cash
        entry_idx = int(i)
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        max_hold = int(row.get("max_hold_bars", 0) or 0)
        next_cooldown = int(row.get("cooldown_bars", 0) or 0)
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
    if pos != 0:
        fill_i = len(frame) - 1
        exit_px = omega._fill_price(arrays, fill_i, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades += 1
        win = cash > entry_equity
        wins += int(win)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        ledger.append({
            "entry_ts": ts[entry_idx], "side": "LONG" if pos > 0 else "SHORT", "entry_price": entry_price,
            "exit_ts": ts[fill_i], "exit_price": float(exit_px), "exit_reason": "forced_end", "hold_bars": fill_i - entry_idx,
            "notional": notional, "leverage": leverage, "trade_pnl_pct": float((cash / before - 1.0) * 100.0),
            "equity_after": float(cash), "win": bool(win),
        })
    n_entries = max(long_entries + short_entries, 1)
    agg = {
        "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "avg_notional": float(notional_sum / n_entries), "avg_leverage": float(leverage_sum / n_entries),
        "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
    }
    return agg, pd.DataFrame(ledger)


def main() -> None:
    LEDGER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    oos_raw = pd.read_csv(omega.EVAL_CSV, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
    print(f"oos_raw rows={len(oos_raw)} [{oos_raw['timestamp'].min()}..{oos_raw['timestamp'].max()}]", flush=True)

    summary_rows = []
    for label in LABELS:
        for seed in SEEDS:
            out_dir = OUT_ROOT / f"{MODEL_ID}_label5way_{label}_154feat_ilias_anchored_seed{seed}_20260821"
            report_path = out_dir / "report.json"
            if not report_path.exists():
                print(f"SKIP missing: {label} seed={seed}", flush=True)
                continue
            report = json.loads(report_path.read_text())
            best = report["ranking_by_validation_pnl"][0]
            variant = best["variant"]
            q_file_tag = f"q{int(round(float(best['quality_threshold']) * 100.0)):03d}"
            oos_csv = out_dir / f"oos_predictions_{q_file_tag}.csv"
            if not oos_csv.exists():
                raise RuntimeError(f"{label} seed={seed}: expected {oos_csv} to exist (variant={variant})")

            oos_src = pd.read_csv(oos_csv, parse_dates=["timestamp"])
            label_oos_raw = oos_raw.merge(oos_src[["timestamp"]], on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
            if len(label_oos_raw) != len(oos_src):
                raise RuntimeError(f"{label} seed={seed}: oos_raw join lost rows ({len(label_oos_raw)} vs {len(oos_src)})")
            oos_dec = parent._to_decisions(oos_src, oof=False)
            agg, ledger = _metrics_with_ledger(label_oos_raw, oos_dec, fee=FEE, slip=SLIP, cost_mult=COST_MULT)

            ok_pnl = abs(agg["pnl"] - float(best["oos_pnl"])) < 1e-6
            ok_mdd = abs(agg["mdd"] - float(best["oos_mdd"])) < 1e-6
            ok_trades = agg["trades"] == int(best["oos_trades"])
            ok_wr = abs(agg["wr"] - float(best["oos_wr"])) < 1e-6
            status = "OK" if (ok_pnl and ok_mdd and ok_trades and ok_wr) else "MISMATCH"
            print(f"[{label} seed={seed} variant={variant}] cross-check={status} "
                  f"recon(pnl={agg['pnl']:.4f},trades={agg['trades']}) "
                  f"report(pnl={best['oos_pnl']:.4f},trades={best['oos_trades']})", flush=True)
            if status != "OK":
                raise RuntimeError(f"{label} seed={seed}: reconstructed metrics do not match report.json")

            ledger_path = LEDGER_OUT_DIR / f"{label}_seed{seed}_{variant}_oos_trade_ledger.csv"
            ledger.to_csv(ledger_path, index=False)

            wins_df = ledger[ledger["win"]]
            losses_df = ledger[~ledger["win"]]
            summary_rows.append({
                "label": label, "seed": seed, "variant": variant, "trades": len(ledger),
                "wins": len(wins_df), "losses": len(losses_df), "wr": agg["wr"],
                "avg_win_pct": float(wins_df["trade_pnl_pct"].mean()) if len(wins_df) else float("nan"),
                "avg_loss_pct": float(losses_df["trade_pnl_pct"].mean()) if len(losses_df) else float("nan"),
                "best_trade_pct": float(ledger["trade_pnl_pct"].max()) if len(ledger) else float("nan"),
                "worst_trade_pct": float(ledger["trade_pnl_pct"].min()) if len(ledger) else float("nan"),
                "avg_hold_bars": float(ledger["hold_bars"].mean()) if len(ledger) else float("nan"),
                "exit_reasons": json.dumps(agg["exit_reasons"]),
                "final_pnl_pct": agg["pnl"], "mdd_pct": agg["mdd"],
                "long_entries": agg["long_entries"], "short_entries": agg["short_entries"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = LEDGER_OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nsaved {summary_path} ({len(summary_df)} runs)", flush=True)


if __name__ == "__main__":
    main()
