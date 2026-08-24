#!/usr/bin/env python3
"""zigzag/h48qual/cusum(5-way 비교, seed=133725056로 통일 -- 셋 다 공유했던 첫 시드, "베스트
시드 고르기" 아님) OOS의 실제 트레이드 원장을 재구성. 원본 학습(`eth_tabm_label_logic_5way_
seed_variant_20260820.py`)은 report.json에 집계 통계(pnl/mdd/trades/wr)만 남기고 개별
트레이드는 저장하지 않았다 -- `omega._metrics()`(`train_eval_omega1_2_tabm_diffusion_risk_
20260603.py:574`, 학습된 exit_head 없이 하드 TP/SL/max_hold만 쓰는 그 정확한 함수)를
그대로 복제하되 매 진입/청산마다 트레이드 행을 추가로 기록한다. 모델을 다시 로드할 필요는
없다 -- 이미 저장된 oos_predictions_qXXX.csv(report.json의 ranking_by_validation_pnl[0]가
고른 VAL-베스트 threshold)만 있으면 됨.

정합성 검증: 내 재구성 집계(pnl/mdd/trades/wr)가 원본 report.json의 oos_pnl/oos_mdd/
oos_trades/oos_wr과 정확히 일치해야 한다(fail-loud) -- 다르면 재구성 로직이 원본과
어긋난다는 뜻."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402

omega = canon.omega
# _prepare_frames()의 사이드이펙트를 그대로 재현 -- 원본 학습 전체가 max_hold/cooldown=0으로
# 계산됐는데(omega.BASE_TEMPLATE 기본값은 72/6), 이 스크립트는 _prepare_frames를 안 거치므로
# 명시적으로 맞춰야 한다. 안 맞추면 max_hold 72bar로 강제청산이 훨씬 자주 걸려 트레이드수가
# 원본(예: zigzag 23건)의 8배(195건)까지 부풀려지는 걸 실제로 확인함(cross-check로 발견).
omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

SEED = 133725056
LABELS = ["zigzag", "h48qual", "cusum"]
OUT_ROOT = ROOT / "tmp/causal_regen_20260516"
MODEL_ID = "omega4_3head_parent72_loose_entry_quality_20260620"
LEDGER_OUT_DIR = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad/oos_trade_ledgers")

FEE, SLIP = omega._load_fee_slip()
COST_MULT = 3.0  # train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py --cost-mult default


def _metrics_with_ledger(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> tuple[dict, pd.DataFrame]:
    """omega._metrics()의 정확한 로직 복제(재구현 아님, 라인단위 이식) + 트레이드 행 기록."""
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

    # canonical OOS raw frame (same source the original training used)
    oos_raw = pd.read_csv(omega.EVAL_CSV, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
    print(f"oos_raw rows={len(oos_raw)} [{oos_raw['timestamp'].min()}..{oos_raw['timestamp'].max()}]", flush=True)

    summary_rows = []
    for label in LABELS:
        out_dir = OUT_ROOT / f"{MODEL_ID}_label5way_{label}_154feat_unified_single_model_seed{SEED}_20260820"
        report = json.loads((out_dir / "report.json").read_text())
        best = report["ranking_by_validation_pnl"][0]
        variant = best["variant"]
        q_file_tag = f"q{int(round(float(best['quality_threshold']) * 100.0)):03d}"
        oos_csv = out_dir / f"oos_predictions_{q_file_tag}.csv"
        if not oos_csv.exists():
            raise RuntimeError(f"{label}: expected {oos_csv} to exist (variant={variant})")

        oos_src = pd.read_csv(oos_csv, parse_dates=["timestamp"])
        # ⚠️ zigzag/h48qual의 direction 소스(zigzag_action_labels_20260531)가 실제로는
        # 2026-02-28에서 끊긴다 (파일명의 "20260531"과 무관 -- 빌드시점 원본데이터 자체가
        # 그때까지만 있었던 것으로 추정, 근본원인 미조사). cusum은 전체 51746행(2026-06-30까지)
        # 커버. label별 실제 oos_src 타임스탬프에 oos_raw를 맞춰 조인해야 frame/dec 길이가
        # 어긋나지 않는다 -- 라벨마다 다른 OOS 윈도우 크기 자체를 있는 그대로 드러낸다.
        label_oos_raw = oos_raw.merge(oos_src[["timestamp"]], on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
        if len(label_oos_raw) != len(oos_src):
            raise RuntimeError(f"{label}: oos_raw join lost rows ({len(label_oos_raw)} vs {len(oos_src)})")
        print(f"  [{label}] oos_csv={oos_csv.name} oos_src_rows={len(oos_src)} oos window actually used: "
              f"{label_oos_raw['timestamp'].min()}..{label_oos_raw['timestamp'].max()} ({len(label_oos_raw)} rows)", flush=True)
        oos_dec = parent._to_decisions(oos_src, oof=False)
        agg, ledger = _metrics_with_ledger(label_oos_raw, oos_dec, fee=FEE, slip=SLIP, cost_mult=COST_MULT)

        # correctness cross-check against report.json's own oos numbers for this exact variant
        ok_pnl = abs(agg["pnl"] - float(best["oos_pnl"])) < 1e-6
        ok_mdd = abs(agg["mdd"] - float(best["oos_mdd"])) < 1e-6
        ok_trades = agg["trades"] == int(best["oos_trades"])
        ok_wr = abs(agg["wr"] - float(best["oos_wr"])) < 1e-6
        status = "OK" if (ok_pnl and ok_mdd and ok_trades and ok_wr) else "MISMATCH"
        print(f"\n=== {label} (variant={variant}, seed={SEED}) === cross-check={status}", flush=True)
        print(f"  reconstructed: pnl={agg['pnl']:.4f} mdd={agg['mdd']:.4f} trades={agg['trades']} wr={agg['wr']:.4f}", flush=True)
        print(f"  report.json:   pnl={best['oos_pnl']:.4f} mdd={best['oos_mdd']:.4f} trades={best['oos_trades']} wr={best['oos_wr']:.4f}", flush=True)
        if status != "OK":
            raise RuntimeError(f"{label}: reconstructed metrics do not match report.json -- ledger logic diverges from original")

        ledger_path = LEDGER_OUT_DIR / f"{label}_seed{SEED}_{variant}_oos_trade_ledger.csv"
        ledger.to_csv(ledger_path, index=False)
        print(f"  ledger saved: {ledger_path} ({len(ledger)} trades)", flush=True)

        wins_df = ledger[ledger["win"]]
        losses_df = ledger[~ledger["win"]]
        summary_rows.append({
            "label": label, "variant": variant, "trades": len(ledger),
            "wins": len(wins_df), "losses": len(losses_df), "wr": agg["wr"],
            "avg_win_pct": float(wins_df["trade_pnl_pct"].mean()) if len(wins_df) else float("nan"),
            "avg_loss_pct": float(losses_df["trade_pnl_pct"].mean()) if len(losses_df) else float("nan"),
            "best_trade_pct": float(ledger["trade_pnl_pct"].max()) if len(ledger) else float("nan"),
            "worst_trade_pct": float(ledger["trade_pnl_pct"].min()) if len(ledger) else float("nan"),
            "avg_hold_bars": float(ledger["hold_bars"].mean()) if len(ledger) else float("nan"),
            "exit_reasons": agg["exit_reasons"],
            "final_pnl_pct": agg["pnl"], "mdd_pct": agg["mdd"],
            "long_entries": agg["long_entries"], "short_entries": agg["short_entries"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = LEDGER_OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n=== summary ===\n{summary_df.to_string(index=False)}", flush=True)
    print(f"\nsaved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
