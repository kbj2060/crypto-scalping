"""Candidate 5: let-winners-run trailing-stop exit for Omega4.6.1's zig075 component.

Motivation (from the 2026-07-06 upgrade investigation + this session's HF/arXiv research pass):
the model's entire realized edge lives in the `zig075 SHORT` bucket, and Omega4.6.1's exit is
currently a pure static TP(7.5%)/SL(4%) barrier (the exit head sits at an effectively-inert 0.95
threshold). The one demonstrably-working exit lever in this project's history is Sigma6's
let-winners-run trailing stop (OOS cost1 +45.9%, on a different -- 1h trend-following -- signal).
This script tests porting that idea onto zig075's own barrier: once a trade's MFE clears
`arm_frac * take_profit`, the stop is ratcheted up to `mfe - trail_gap` (monotonic, never loosens);
below that MFE the original static stop_loss stays in force (avoids premature stop-outs from early
noise). take_profit remains a hard cap throughout. h48qual keeps its original static barrier
unless explicitly included in a variant (it is net-negative on both VAL and OOS in the baseline
breakdown, so trailing is not expected to rescue it).

Fresh-Forward-aware process (Fresh-Forward Rule / no-cherry-picking discipline, same as candidates
1-4): grid-search (arm_frac, trail_gap) on VAL 2025-10-01..12-31 ONLY, freeze the single winning
config, then score it ONCE on OOS 2026-01-01..06-30. Uses the exact same frozen artifacts / caps /
duration gate as the live model. Stored-ledger based -> DIAGNOSTIC research score, not a
live-promotion claim. trading_bot.py / trading_bot_modules/omega4_6_1_live.py are NOT touched by
this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402
from test_omega4_6_1_drop_h48qual_20260706 import _metrics  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_trailing_exit_20260707"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARM_FRAC_GRID = [0.3, 0.4, 0.5, 0.6]
TRAIL_GAP_GRID = [0.01, 0.015, 0.02, 0.03]


@torch.no_grad()
def greedy_replay_trailing(frame: pd.DataFrame, components: dict, *, fee: float, slip: float,
                            cost_mult: float, device: torch.device, trailing_cfg: dict | None) -> tuple[dict, pd.DataFrame]:
    """Fork of replay_omega4_6_1_greedy_router_20260706.greedy_replay with the exit-barrier block
    replaced by an optional per-component trailing stop. Everything else (entry logic, priority
    routing, caps, fees) is byte-identical to the original -- only the exit condition changes."""
    trailing_cfg = trailing_cfg or {}
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    active_comp = None
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    rows: list[dict] = []
    reasons: dict[str, int] = {}

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            unreal = move * notional
            mfe, mae = max(mfe, move), min(mae, move)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            else:
                trail = trailing_cfg.get(active_comp)
                stop_floor = -abs(stop_loss) if stop_loss > 0.0 else -1e9
                trailing_active = False
                if trail is not None and mfe >= take_profit * trail["arm_frac"]:
                    trail_floor = mfe - trail["trail_gap"]
                    if trail_floor > stop_floor:
                        stop_floor = trail_floor
                        trailing_active = True
                if move <= stop_floor:
                    reason = "trailing_stop" if trailing_active else "stop_loss"
                else:
                    hold = max(i - entry_i, 0)
                    giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                    expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                    prob = sidecar._predict_exit_prob_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=[float(pos), float(hold), float(move), float(mfe), float(mae),
                                    float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move),
                                    float(move + abs(stop_loss)), float(notional), float(leverage_v),
                                    float(notional * leverage_v), float(take_profit), float(stop_loss)],
                        device=device,
                    )
                    if prob >= comp["exit_threshold"]:
                        reason = "exit_head"
            if reason:
                exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({"entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": i,
                             "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                             "exit_timestamp": str(frame["timestamp"].iloc[i]), "side": int(pos),
                             "source_component": active_comp, "reason": reason,
                             "win": int(cash > entry_equity), "trade_return": float(trade_return),
                             "notional": float(notional), "margin_fraction": float(margin_fraction),
                             "leverage": float(leverage_v)})
                pos, active_comp = 0, None
                continue
            continue

        # flat: try priority order (unchanged from the original)
        for name in greedy.PRIORITY:
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
                continue
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            scale = greedy.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
            row_leverage = row_notional / max(row_margin, 1e-12)
            if row_notional <= 0.0:
                continue
            entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            pos, active_comp = side, name
            entry_price, entry_equity = float(entry_px), cash
            entry_i, entry_signal_i = min(i + 1, n - 1), i
            margin_fraction, leverage_v, notional = row_margin, row_leverage, row_notional
            take_profit = float(comp["dec"]["take_profit"].iloc[i])
            stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            break

    return {"reason_counts": reasons}, pd.DataFrame(rows)


def load_components(frame: pd.DataFrame, device, *, val: bool) -> dict:
    components = {}
    for cname, cfg in retest.COMPONENTS.items():
        if val:
            pred = pd.read_csv(valmod.VAL_PRED[cname])
            pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])
            pred = pred[(pred["timestamp"] >= valmod.START) & (pred["timestamp"] <= valmod.END)].reset_index(drop=True)
            common = frame["timestamp"].isin(pred["timestamp"])
            frame = frame[common].reset_index(drop=True)
            pred = pred[pred["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
            tmp = OUT_DIR / f"_val_{cname}_aligned.csv"
            pred.to_csv(tmp, index=False)
            components[cname] = greedy.prepare_component(frame, tmp, cfg, device)
        else:
            pred_csv = retest.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"
            components[cname] = greedy.prepare_component(frame, pred_csv, cfg, device)
    return frame, components


def score(frame, components, *, fee, slip, trailing_cfg) -> dict:
    _, lg = greedy_replay_trailing(frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT,
                                    device=retest.DEVICE, trailing_cfg=trailing_cfg)
    return _metrics(lg, frame, apply_gate=True), lg


def main() -> int:
    device = retest.DEVICE
    fee, slip = omega._load_fee_slip()

    val_frame_raw = valmod.load_val_frame()
    val_frame, val_components = load_components(val_frame_raw, device, val=True)

    baseline_val, _ = score(val_frame, val_components, fee=fee, slip=slip, trailing_cfg=None)
    print(f"VAL baseline (static TP/SL): {baseline_val}", flush=True)

    grid_results = []
    for arm_frac in ARM_FRAC_GRID:
        for trail_gap in TRAIL_GAP_GRID:
            cfg = {"zig075": {"arm_frac": arm_frac, "trail_gap": trail_gap}}
            m, _ = score(val_frame, val_components, fee=fee, slip=slip, trailing_cfg=cfg)
            grid_results.append({"arm_frac": arm_frac, "trail_gap": trail_gap, **m})
            print(f"  zig075-trailing arm_frac={arm_frac:.2f} trail_gap={trail_gap:.3f} -> "
                  f"pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+6.2f}% n={m['trades']:2d} wr={m['wr']:.3f}", flush=True)

    grid_results.sort(key=lambda r: r["pnl"], reverse=True)
    best = grid_results[0]
    print(f"\nBest VAL config: {best}", flush=True)

    both_cfg = {"zig075": {"arm_frac": best["arm_frac"], "trail_gap": best["trail_gap"]},
                "h48qual": {"arm_frac": best["arm_frac"], "trail_gap": best["trail_gap"]}}
    both_val, _ = score(val_frame, val_components, fee=fee, slip=slip, trailing_cfg=both_cfg)
    print(f"VAL (trailing on BOTH components, same params): {both_val}", flush=True)

    frozen_cfg = {"zig075": {"arm_frac": best["arm_frac"], "trail_gap": best["trail_gap"]}}
    adopt = bool(best["pnl"] > baseline_val["pnl"])
    print(f"\nDecision (VAL-only, pre-registered): {'ADOPT' if adopt else 'REJECT'} trailing exit "
          f"(best VAL pnl={best['pnl']:+.2f}% vs baseline {baseline_val['pnl']:+.2f}%)", flush=True)

    # ---- OOS one-shot confirm (frozen config, run regardless of VAL verdict for transparency) ----
    oos_frame_raw = retest.load_frame_current("2026-01-01", "2026-06-30")
    oos_frame, oos_components = load_components(oos_frame_raw, device, val=False)
    baseline_oos, _ = score(oos_frame, oos_components, fee=fee, slip=slip, trailing_cfg=None)
    frozen_oos, oos_ledger = score(oos_frame, oos_components, fee=fee, slip=slip, trailing_cfg=frozen_cfg)
    print(f"\nOOS baseline: {baseline_oos}", flush=True)
    print(f"OOS frozen trailing config: {frozen_oos}", flush=True)

    result = {
        "model_id": "omega4_6_1_trailing_exit_20260707",
        "grid": grid_results,
        "best_val_config": {"arm_frac": best["arm_frac"], "trail_gap": best["trail_gap"]},
        "val": {"baseline": baseline_val, "best_zig075_only": best, "both_components": both_val},
        "oos": {"baseline": baseline_oos, "frozen_zig075_only": frozen_oos},
        "adopt_decision_val_only": adopt,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    oos_ledger.to_csv(OUT_DIR / "oos_ledger_frozen_trailing.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
