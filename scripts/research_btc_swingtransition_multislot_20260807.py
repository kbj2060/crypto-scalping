"""Multi-slot (concurrent positions) replay for the promoted BTC swingtransition model.

Motivation (measured 2026-08-07): OOS slot occupancy is 92% -- the single-position replay/live
path discards ~625 duration-gated quality-passing signal clusters (vs 41 taken trades) because a
position is already held. The sizing lever is closed (multiplier sweep shows current sizing is at
the growth-optimal point; scaling up only buys MDD), so harvesting discarded signals is the one
structural PnL lever left. Same TOTAL margin budget: each of N slots trades at 1/N of the sidecar
sizing, so adopting N>1 does not raise account exposure caps -- it diversifies entry timing.

PRE-REGISTERED RULES (fixed before any result was seen):
- Replay semantics are byte-compatible with sidecar._replay_with_risk at N=1 (same execution
  model, costs x3, exit-head threshold 0.95, same duration-gate-as-ledger-post-filter metric
  convention); the N=1 run is a hard regression gate vs the promoted report (tolerance 0.05pp).
- N in {2, 3} is selected on VAL ONLY: highest VAL gated PnL subject to VAL gated MDD >= -8%.
- The selected N gets ONE OOS look. ADOPT only if ALL of:
    OOS gated PnL >= +13.8% (baseline +10.76% + 3pp)
    OOS gated MDD >= -16.5%
    worst OOS calendar quarter >= -4.0% (baseline worst: -0.87%)
  Otherwise the axis is closed (no re-tuning of slot counts/scales on OOS).
- Caveat logged up front: 513 of the 625 skipped OOS clusters are LONG in a year where longs
  lost money -- the counterfactual can plausibly LOWER PnL. That is what the gate is for.
- Live-wiring note if adopted: concurrent opposite-side slots net out in one-way futures mode;
  live adoption would need hedge-mode or same-side-only slot policy. Recorded now so a pass
  doesn't silently skip the question.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_btc_swingtransition_20260806 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_btc_20260708 as sidecar  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_btc_swingtransition_20260806 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806 as apply_mod  # noqa: E402
from research_btc_swingtransition_trailing_stop_val_oos_20260807 import _compound_metrics, _gate, LIVE_DURATION_THRESHOLD  # noqa: E402

BASELINE_EXPECTED = {
    "val_gated_pnl": 24.226193370361937,
    "val_gated_mdd": -2.4590149966945973,
    "oos_gated_pnl": 10.760766798223663,
    "oos_gated_mdd": -12.410621340770533,
}


def _replay_multislot(frame, base_x, dec, loaded, *, n_slots, risk_margin_fraction, risk_leverage,
                       exit_threshold, fee, slip, cost_mult, device) -> pd.DataFrame:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    route = hard._route_id(frame)
    base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(base_x, loaded)
    dec_side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)

    slots: list[dict[str, Any] | None] = [None] * int(n_slots)
    rows: list[dict[str, Any]] = []

    def close_slot(k, i, exit_px, exit_fee, reason, exit_prob):
        s = slots[k]
        raw_exit = (exit_px - s["entry_price"]) / max(s["entry_price"], 1e-12) if s["pos"] > 0 else (s["entry_price"] - exit_px) / max(s["entry_price"], 1e-12)
        n = s["notional"]
        r = (1.0 - s["entry_fee"] * n) * (1.0 + raw_exit * n) * (1.0 - exit_fee * n) - 1.0
        rows.append({
            "entry_signal_i": s["entry_signal_i"], "entry_i": s["entry_i"], "exit_i": int(i),
            "entry_timestamp": str(frame["timestamp"].iloc[s["entry_signal_i"]]),
            "exit_timestamp": str(frame["timestamp"].iloc[int(i)]),
            "side": int(s["pos"]), "reason": reason, "win": int(r > 0),
            "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(s["mfe"]), "mae_price_move": float(s["mae"]),
            "trade_return": float(r), "notional": float(n), "margin_fraction": float(s["margin"]),
            "leverage": float(s["leverage"]), "exit_prob": float(exit_prob), "slot": int(k),
            "take_profit": float(s["tp"]), "stop_loss": float(s["sl"]),
        })
        slots[k] = None

    for i in range(0, len(frame) - 2):
        exited_this_bar = False
        for k in range(n_slots):
            s = slots[k]
            if s is None:
                continue
            move = sidecar.price_exit._price_move(arrays, int(i), side=s["pos"], entry_price=float(s["entry_price"]), slip_eff=slip_eff)
            s["mfe"] = max(s["mfe"], move)
            s["mae"] = min(s["mae"], move)
            reason = ""
            exit_prob = 0.0
            if s["tp"] > 0.0 and move >= s["tp"]:
                reason = "take_profit"
            elif s["sl"] > 0.0 and move <= -abs(s["sl"]):
                reason = "stop_loss"
            else:
                hold = max(int(i) - int(s["entry_i"]), 0)
                giveback = (s["mfe"] - move) / max(abs(s["mfe"]), 1e-8) if s["mfe"] > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = sidecar._predict_exit_prob_one(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[float(s["pos"]), float(hold), float(move), float(s["mfe"]), float(s["mae"]),
                                float(np.clip(giveback, 0.0, 10.0)), float(s["tp"] - move), float(move + abs(s["sl"])),
                                float(s["notional"]), float(s["leverage"]), float(s["notional"] * s["leverage"]),
                                float(s["tp"]), float(s["sl"])],
                    device=device)
                exit_prob = float(prob)
                if prob >= exit_threshold:
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _ = omega._try_execution(arrays, int(i), s["pos"], entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                close_slot(k, i, exit_px, exit_fee, reason, exit_prob)
                exited_this_bar = True

        if exited_this_bar:
            continue
        free = next((k for k in range(n_slots) if slots[k] is None), None)
        if free is None or not bool(active[i]):
            continue
        side = int(dec_side[i])
        if side == 0:
            continue
        filled, px, fee_paid, _ = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        margin = float(risk_margin_fraction[int(i)]) / float(n_slots)
        leverage = float(risk_leverage[int(i)])
        notional = margin * leverage
        if notional <= 0.0:
            continue
        row = dec.iloc[i]
        slots[free] = {
            "pos": side, "entry_price": float(px), "entry_i": min(int(i) + 1, len(frame) - 1),
            "entry_signal_i": int(i), "entry_fee": float(fee_paid), "margin": margin,
            "leverage": leverage, "notional": notional, "mfe": 0.0, "mae": 0.0,
            "tp": float(row.get("take_profit", 0.0) or 0.0), "sl": float(row.get("stop_loss", 0.0) or 0.0),
        }

    for k in range(n_slots):
        if slots[k] is not None:
            s = slots[k]
            exit_px = omega._fill_price(arrays, len(frame) - 1, s["pos"], slip_eff, entry=False)
            close_slot(k, len(frame) - 1, exit_px, fee_eff, "forced_end", 0.0)

    led = pd.DataFrame(rows)
    if len(led):
        led = led.sort_values("entry_i").reset_index(drop=True)
    return led


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--slots", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/btc_swingtransition_multislot_20260807")
    args = ap.parse_args()
    device = parent._device(str(args.device))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition/true_3head_tabm_bundle.pt"
    sidecar_path = ROOT / "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260806_swingtransition/risk_sidecar.pkl"
    pred_dir = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_freshforward_ext_20260806"

    print("stage=load", flush=True)
    import pickle
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(bundle["models"], device=device)
    with open(sidecar_path, "rb") as f:
        pkl = pickle.load(f)

    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=ROOT / "tmp/causal_regen_20260516/btc_zigzag_action_labels_freshforward_ext_20260802",
        quality_mode="quality_label_action",
        quality_label_dir=ROOT / "tmp/causal_regen_20260516/btc_h48_conservative_padded_freshforward_ext_20260802",
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()

    data = {}
    for split, oof in [("validation", True), ("oos", False)]:
        raw = frames["val_raw" if split == "validation" else "oos_raw"]
        src = sidecar._load_precomputed_prediction(pred_dir, split, "q055", raw)
        x = parent._base_input(raw, base_cols)
        dec_base = parent._to_decisions(src, oof=oof)
        dec, _ = atr_eval._apply_atr_safety_sltp(dec_base, raw, atr_window=192, tp_mult=12.0, sl_mult=6.0,
                                                 min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12)
        atr = atr_eval._atr_pct(raw, 192)
        feats = sidecar._risk_feature_frame(raw, src, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
        x_all, _ = sidecar._feature_matrix(feats, pkl["feature_columns"])
        side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
        score = sidecar._predict_side_split_models(pkl["model"], x_all, side)
        mapping = pkl["selected_mapping"]
        bm = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
        bl = sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS})
        margin, leverage = apply_mod._scaled_margin_leverage(dec, bm, bl, long_scale=0.5, short_scale=2.5)
        ou = raw[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})
        ou["entry_timestamp"] = pd.to_datetime(ou["entry_timestamp"])
        data[split] = dict(raw=raw, x=x, dec=dec, margin=margin, leverage=leverage, ou=ou)

    results = []
    for n in args.slots:
        print(f"stage=replay n_slots={n}", flush=True)
        out = {"n_slots": int(n)}
        for split in ("validation", "oos"):
            d = data[split]
            led = _replay_multislot(d["raw"], d["x"], d["dec"], loaded, n_slots=n,
                                    risk_margin_fraction=d["margin"], risk_leverage=d["leverage"],
                                    exit_threshold=0.95, fee=fee, slip=slip, cost_mult=3.0, device=device)
            led.to_csv(args.out_dir / f"{split}_ledger_n{n}.csv", index=False)
            g = _gate(led, d["ou"])
            out[f"{split}_ungated"] = _compound_metrics(led)
            out[f"{split}_gated"] = _compound_metrics(g)
            if split == "oos" and len(g):
                g2 = g.copy()
                g2["q"] = pd.to_datetime(g2["entry_timestamp"]).dt.to_period("Q")
                out["oos_quarters"] = {str(q): float(((1 + d2["trade_return"]).prod() - 1) * 100) for q, d2 in g2.groupby("q")}
        results.append(out)
        print(json.dumps(out, indent=None), flush=True)
        if n == 1:
            checks = {"val_gated_pnl": out["validation_gated"]["pnl"], "val_gated_mdd": out["validation_gated"]["mdd"],
                      "oos_gated_pnl": out["oos_gated"]["pnl"], "oos_gated_mdd": out["oos_gated"]["mdd"]}
            for k, exp in BASELINE_EXPECTED.items():
                if abs(checks[k] - exp) > 0.05:
                    raise SystemExit(f"N=1 REGRESSION FAIL: {k}={checks[k]:.4f} expected {exp:.4f}")
            print("N=1 regression check PASS (multislot replay reproduces the promoted single-slot numbers)", flush=True)

    multi = [r for r in results if r["n_slots"] > 1]
    eligible = [r for r in multi if r["validation_gated"]["mdd"] >= -8.0]
    verdict: dict[str, Any] = {"selected_n": None, "adopt": False}
    if eligible:
        sel = max(eligible, key=lambda r: r["validation_gated"]["pnl"])
        wq = min(sel.get("oos_quarters", {"none": 0.0}).values())
        adopt = (sel["oos_gated"]["pnl"] >= 13.8 and sel["oos_gated"]["mdd"] >= -16.5 and wq >= -4.0)
        verdict = {"selected_n": sel["n_slots"], "val_gated": sel["validation_gated"],
                   "oos_gated": sel["oos_gated"], "oos_worst_quarter": wq, "adopt": bool(adopt)}
    report = {
        "method": "btc_swingtransition_multislot_equal_budget",
        "preregistered_rule": "select N on VAL gated PnL s.t. VAL MDD>=-8%; adopt iff OOS gated PnL>=13.8 AND MDD>=-16.5 AND worst quarter>=-4.0",
        "results": results, "verdict": verdict,
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VERDICT:", json.dumps(verdict), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
