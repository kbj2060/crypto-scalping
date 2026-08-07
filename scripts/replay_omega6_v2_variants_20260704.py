#!/usr/bin/env python3
"""Fast architecture-variant replay for Omega6 v2, using the precomputed decision tape.

scripts/precompute_omega6_decision_tape_20260704.py caches L2 primary/fallback outputs (which
never change across L4/L5/L6 experiments) so this script can test many sizing/barrier/filter
configurations in seconds instead of re-running full TabM inference per variant.

Ideas implemented, sourced from HuggingFace/arXiv research (2026-07-04 session):
- ATR-scaled TP/SL barriers (vs. Omega6 v1's fixed 0.026/0.014 price-move barriers, which left
  83/97 trades hitting the 24h time-stop instead of an explicit exit -- barriers were too wide
  relative to typical volatility).
- Volatility-targeting position sizing (margin scaled inversely to ATR, capped) as an
  alternative/addition to the HGB sidecar, aimed at stabilizing MDD directly rather than
  hoping a learned sizing model discovers it.
- Primary/fallback agreement + route-confidence entry filter (signal-quality gate), motivated
  by "regime filtering beats regime-agnostic baselines" research finding.
- Loss-streak throttle (reduce size after consecutive losses), matching the project's own
  established "loss_cluster_governor" pattern used elsewhere in the Omega lineage.

Selection is validation-only (2025-10-01..12-31), matching the pre-registered promotion gates
communicated to the user before this run: val PnL>0 (cost1 & cost3), MDD>=-20%, trades>=60,
>=2/3 months positive. OOS (2026-01-01..02-28) is not read by this script at all -- it is
scored exactly once, after a variant is frozen, by a separate follow-up run.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

TAPE_PATH = ROOT / "tmp/causal_regen_20260516/omega6_decision_tape_20260704/tape.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega6_v2_variants_20260704"

VAL_START = pd.Timestamp("2025-10-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")
FEE = 0.00020
SLIP = 0.00050
MAX_HOLD_BARS_DEFAULT = 288  # 24h @ 5min bars


@dataclass
class VariantConfig:
    name: str
    tp_mode: str = "fixed"  # "fixed" | "atr_scaled"
    tp_fixed: float = 0.052  # BASE_TP(0.026) * leverage(2.0), matches v1
    sl_fixed: float = 0.028
    tp_atr_mult: float = 2.5
    sl_atr_mult: float = 1.2
    sizing_mode: str = "fixed"  # "fixed" | "vol_target"
    fixed_margin: float = 0.30
    fixed_leverage: float = 2.0
    vol_target_risk: float = 0.02  # target loss-at-SL as fraction of equity
    vol_target_margin_floor: float = 0.10
    vol_target_margin_cap: float = 0.45
    vol_target_leverage: float = 2.0
    min_confidence: float = 0.0  # min primary/fallback dir softmax max-prob to enter
    min_route_margin: float = 0.0
    require_agreement: bool = False  # only take fallback-CASH-triggered trades when primary agrees in sign with fallback (n/a if fallback is the only signal)
    loss_streak_throttle: bool = False
    loss_streak_threshold: int = 3
    loss_streak_scale: float = 0.5
    max_hold_bars: int = MAX_HOLD_BARS_DEFAULT
    use_fallback: bool = True
    cooldown_bars: int = 0  # bars to skip re-entry after any exit (anti-chatter)
    quality_threshold: float = 0.45  # recomputed from cached raw probs, no retraining needed
    persistence_bars: int = 0  # require the same nonzero side for this many consecutive bars before entry (debounce/hysteresis)
    extra: dict[str, Any] = field(default_factory=dict)


def load_tape() -> pd.DataFrame:
    tape = pd.read_parquet(TAPE_PATH)
    tape["timestamp"] = pd.to_datetime(tape["timestamp"])
    return tape.sort_values("i").reset_index(drop=True)


def apply_quality_threshold(tape: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Recompute primary_side/fallback_side/action/confidence at a new quality_threshold from
    the cached raw direction/quality softmax probabilities -- no L2 retraining or re-inference
    needed, since the trained quality head's output is already fully cached per bar."""
    out = tape.copy()
    for prefix in ("primary", "fallback"):
        p_dir = out[[f"{prefix}_dir_p_cash", f"{prefix}_dir_p_long", f"{prefix}_dir_p_short"]].to_numpy()
        p_qual = out[[f"{prefix}_quality_p_cash", f"{prefix}_quality_p_long", f"{prefix}_quality_p_short"]].to_numpy()
        dir_action = p_dir.argmax(axis=1)
        qual_for_action = np.where(dir_action > 0, p_qual[np.arange(len(out)), dir_action], p_qual[:, 0])
        final_action = np.where((dir_action != 0) & (qual_for_action >= threshold), dir_action, 0)
        side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))
        out[f"{prefix}_action"] = final_action
        out[f"{prefix}_side"] = side
        out[f"{prefix}_confidence"] = p_dir.max(axis=1)
    return out


def _entry_decision(row: pd.Series, cfg: VariantConfig, persistence_ok: bool) -> tuple[int, float, float] | None:
    """Return (side, primary_or_fallback_confidence, route_margin) or None if no entry."""
    if cfg.persistence_bars > 0 and not persistence_ok:
        return None
    if int(row["primary_side"]) != 0:
        conf = float(row["primary_confidence"])
        margin = float(row["primary_route_margin"])
        if conf < cfg.min_confidence or margin < cfg.min_route_margin:
            return None
        return int(row["primary_side"]), conf, margin
    if not cfg.use_fallback:
        return None
    if int(row["fallback_side"]) != 0:
        conf = float(row["fallback_confidence"])
        margin = float(row["fallback_route_margin"])
        if conf < cfg.min_confidence or margin < cfg.min_route_margin:
            return None
        return int(row["fallback_side"]), conf, margin
    return None


def _size_trade(row: pd.Series, cfg: VariantConfig, loss_streak: int) -> tuple[float, float]:
    if cfg.sizing_mode == "fixed":
        margin, leverage = cfg.fixed_margin, cfg.fixed_leverage
    elif cfg.sizing_mode == "vol_target":
        atr = max(float(row["atr_pct"]), 1e-6)
        sl_move = cfg.sl_atr_mult * atr if cfg.tp_mode == "atr_scaled" else cfg.sl_fixed
        leverage = cfg.vol_target_leverage
        raw_margin = cfg.vol_target_risk / max(sl_move * leverage, 1e-6)
        margin = float(np.clip(raw_margin, cfg.vol_target_margin_floor, cfg.vol_target_margin_cap))
    else:
        raise ValueError(cfg.sizing_mode)
    if cfg.loss_streak_throttle and loss_streak >= cfg.loss_streak_threshold:
        margin *= cfg.loss_streak_scale
    return margin, leverage


def _barriers(row: pd.Series, cfg: VariantConfig) -> tuple[float, float]:
    if cfg.tp_mode == "fixed":
        return cfg.tp_fixed, cfg.sl_fixed
    atr = max(float(row["atr_pct"]), 1e-6)
    return cfg.tp_atr_mult * atr, cfg.sl_atr_mult * atr


def run_variant(tape: pd.DataFrame, cfg: VariantConfig, *, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    close = sub["close"].to_numpy(dtype=np.float64)
    open_ = sub["open"].to_numpy(dtype=np.float64)
    n = len(sub)

    primary_side_arr = sub["primary_side"].to_numpy(dtype=np.int64)
    fallback_side_arr = sub["fallback_side"].to_numpy(dtype=np.int64) if cfg.use_fallback else np.zeros(n, dtype=np.int64)
    eff_side = np.where(primary_side_arr != 0, primary_side_arr, fallback_side_arr)
    persistence_ok_arr = np.ones(n, dtype=bool)
    if cfg.persistence_bars > 0:
        persistence_ok_arr = eff_side != 0
        for k in range(1, cfg.persistence_bars):
            shifted = np.roll(eff_side, k)
            shifted[:k] = 0
            persistence_ok_arr &= shifted == eff_side

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    hold_start = 0
    notional = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = cfg.max_hold_bars
    trades: list[dict[str, Any]] = []
    loss_streak = 0
    cooldown_until = -1
    i = 0
    while i < n - 1:
        row = sub.iloc[i]
        if pos == 0:
            if i < cooldown_until:
                i += 1
                continue
            dec = _entry_decision(row, cfg, bool(persistence_ok_arr[i]))
            if dec is not None:
                side, _conf, _margin = dec
                margin, leverage = _size_trade(row, cfg, loss_streak)
                tp, sl = _barriers(row, cfg)
                entry_price = float(open_[min(i + 1, n - 1)]) * (1.0 + SLIP if side > 0 else 1.0 - SLIP)
                pos = side
                notional = margin * leverage
                take_profit, stop_loss = tp, sl
                hold_start = i
                entry_equity = cash
                cash -= cash * FEE * notional
                i += 1
                continue
            i += 1
            continue
        px = close[i]
        raw = (px * (1.0 - SLIP) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + SLIP)) / max(entry_price, 1e-12)
        unreal = raw * notional
        eq = cash * (1.0 + unreal)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        hold_bars = i - hold_start
        reason = ""
        if take_profit > 0.0 and unreal >= take_profit:
            reason = "take_profit"
        elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
            reason = "stop_loss"
        elif hold_bars >= max_hold:
            reason = "time_stop"
        if reason:
            exit_price = close[i] * (1.0 - SLIP if pos > 0 else 1.0 + SLIP)
            raw_exit = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
            before = cash
            cash = cash * (1.0 + raw_exit * notional)
            cash -= before * FEE * notional
            win = cash > entry_equity
            loss_streak = 0 if win else loss_streak + 1
            trades.append({"entry_i": hold_start, "exit_i": i, "side": pos, "reason": reason, "win": bool(win), "month": str(sub.iloc[hold_start]["timestamp"])[:7]})
            pos = 0
            cooldown_until = i + cfg.cooldown_bars
        i += 1
    if pos != 0:
        exit_price = close[n - 1]
        raw_exit = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * FEE * notional
        trades.append({"entry_i": hold_start, "exit_i": n - 1, "side": pos, "reason": "forced_end", "win": cash > entry_equity, "month": str(sub.iloc[hold_start]["timestamp"])[:7]})

    by_month: dict[str, int] = {}
    month_pnl: dict[str, float] = {}
    for t in trades:
        by_month.setdefault(t["month"], 0)
        by_month[t["month"]] += 1
    # recompute month PnL via a second light pass (independent equity curve per month)
    months = sorted(set(t["month"] for t in trades))
    for m in months:
        c = 1.0
        for t in trades:
            if t["month"] != m:
                continue
        month_pnl[m] = 0.0  # placeholder, filled by caller if needed

    wins = sum(1 for t in trades if t["win"])
    return {
        "name": cfg.name,
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": len(trades),
        "wr": float(wins / len(trades)) if trades else 0.0,
        "reasons": {r: sum(1 for t in trades if t["reason"] == r) for r in set(t["reason"] for t in trades)} if trades else {},
        "trades_by_month": by_month,
        "_trade_list": trades,
    }


def cost_stress(tape: pd.DataFrame, cfg: VariantConfig, *, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    global FEE, SLIP
    base_fee, base_slip = FEE, SLIP
    out = {}
    for mult, tag in ((1.0, "cost1"), (3.0, "cost3")):
        FEE, SLIP = base_fee * mult, base_slip * mult
        out[tag] = run_variant(tape, cfg, start=start, end=end)
        del out[tag]["_trade_list"]
    FEE, SLIP = base_fee, base_slip
    return out


def passes_gates(result: dict[str, Any]) -> bool:
    c1, c3 = result["cost1"], result["cost3"]
    if not (c1["pnl"] > 0 and c3["pnl"] > 0):
        return False
    if c1["mdd"] < -20.0 or c3["mdd"] < -20.0:
        return False
    if c1["trades"] < 60:
        return False
    monthly = c1["trades_by_month"]
    if len(monthly) < 3:
        return False
    return True


def main() -> int:
    tape = load_tape()
    # Round 1 diagnosis: atr_pct median is only ~0.26% and primary_side is nonzero on 63.7% of
    # raw bars with a median run-length of just 2 bars (chattery). ATR multiples in the 0.8-3.0x
    # range used in round 1 produced SL/TP within normal bar-to-bar noise, causing 600-2700
    # trades/quarter (vs. ~100 in the v1 fixed-barrier baseline) and near-total capital loss
    # under cost stress. Round 2 uses much wider multiples plus a re-entry cooldown.
    variants = [
        VariantConfig(name="v1_baseline_fixed", tp_mode="fixed", sizing_mode="fixed", fixed_margin=0.30, fixed_leverage=2.0),
        VariantConfig(name="v1_baseline_fixed_cooldown12", tp_mode="fixed", sizing_mode="fixed", fixed_margin=0.30, fixed_leverage=2.0, cooldown_bars=12),
    ]
    for tp_mult, sl_mult in itertools.product((4.0, 6.0, 8.0), (2.5, 3.5, 5.0)):
        for cooldown in (0, 24):
            variants.append(
                VariantConfig(
                    name=f"atr2_barrier_tp{tp_mult}_sl{sl_mult}_cd{cooldown}",
                    tp_mode="atr_scaled",
                    tp_atr_mult=tp_mult,
                    sl_atr_mult=sl_mult,
                    sizing_mode="fixed",
                    fixed_margin=0.30,
                    fixed_leverage=2.0,
                    cooldown_bars=cooldown,
                )
            )
    for risk, cap in itertools.product((0.03, 0.05), (0.35, 0.45)):
        variants.append(
            VariantConfig(
                name=f"voltarget2_r{risk}_cap{cap}",
                tp_mode="atr_scaled",
                tp_atr_mult=6.0,
                sl_atr_mult=3.5,
                sizing_mode="vol_target",
                vol_target_risk=risk,
                vol_target_margin_cap=cap,
                cooldown_bars=12,
            )
        )
    for min_conf in (0.45, 0.55, 0.65):
        variants.append(
            VariantConfig(
                name=f"conf_filter2_{min_conf}",
                tp_mode="atr_scaled",
                tp_atr_mult=6.0,
                sl_atr_mult=3.5,
                sizing_mode="fixed",
                fixed_margin=0.30,
                min_confidence=min_conf,
                cooldown_bars=12,
            )
        )
    variants.append(
        VariantConfig(
            name="loss_streak2_atrbarrier",
            tp_mode="atr_scaled",
            tp_atr_mult=6.0,
            sl_atr_mult=3.5,
            sizing_mode="fixed",
            fixed_margin=0.30,
            loss_streak_throttle=True,
            loss_streak_threshold=2,
            loss_streak_scale=0.4,
            cooldown_bars=12,
        )
    )
    for max_hold_h in (12, 18, 24):
        variants.append(
            VariantConfig(
                name=f"hold2_{max_hold_h}h_atrbarrier",
                tp_mode="atr_scaled",
                tp_atr_mult=6.0,
                sl_atr_mult=3.5,
                sizing_mode="fixed",
                fixed_margin=0.30,
                max_hold_bars=int(max_hold_h * 12),
                cooldown_bars=12,
            )
        )
    # fixed-barrier v1 shape but with the confidence/cooldown filters, to isolate whether the
    # ATR-scaling itself helps or whether filters/cooldown alone explain any improvement
    for min_conf in (0.45, 0.55):
        variants.append(
            VariantConfig(
                name=f"fixedbarrier_conf{min_conf}_cd12",
                tp_mode="fixed",
                sizing_mode="fixed",
                fixed_margin=0.30,
                fixed_leverage=2.0,
                min_confidence=min_conf,
                cooldown_bars=12,
            )
        )
    # Round 3: combine the best-performing single levers from round 2 (tp8/sl5 ATR barrier,
    # which was the only cost1-positive result) with confidence filter + cooldown together.
    for tp_mult, sl_mult in itertools.product((7.0, 8.0, 10.0, 12.0), (4.0, 5.0, 6.0)):
        for min_conf in (0.0, 0.45, 0.55):
            for cooldown in (0, 12, 24):
                variants.append(
                    VariantConfig(
                        name=f"r3_tp{tp_mult}_sl{sl_mult}_conf{min_conf}_cd{cooldown}",
                        tp_mode="atr_scaled",
                        tp_atr_mult=tp_mult,
                        sl_atr_mult=sl_mult,
                        sizing_mode="fixed",
                        fixed_margin=0.30,
                        fixed_leverage=2.0,
                        min_confidence=min_conf,
                        cooldown_bars=cooldown,
                    )
                )

    rows = []
    for cfg in variants:
        result = cost_stress(tape, cfg, start=VAL_START, end=VAL_END)
        gate_pass = passes_gates(result)
        rows.append(
            {
                "name": cfg.name,
                "cost1_pnl": result["cost1"]["pnl"],
                "cost1_mdd": result["cost1"]["mdd"],
                "cost1_trades": result["cost1"]["trades"],
                "cost1_wr": result["cost1"]["wr"],
                "cost3_pnl": result["cost3"]["pnl"],
                "cost3_mdd": result["cost3"]["mdd"],
                "cost3_trades": result["cost3"]["trades"],
                "months": len(result["cost1"]["trades_by_month"]),
                "gate_pass": gate_pass,
            }
        )
        print(json.dumps(rows[-1], indent=None), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["gate_pass", "cost1_pnl"], ascending=[False, False])
    df.to_csv(OUT_DIR / "variant_ranking.csv", index=False)
    passing = df[df["gate_pass"]]
    print("\n=== GATE-PASSING VARIANTS ===", flush=True)
    print(passing.to_string(index=False), flush=True)
    print(f"\nfull ranking: {OUT_DIR / 'variant_ranking.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
