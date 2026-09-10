"""Phase 1 of the 2026-07-07 improvement roadmap (docs/model_contracts/omega4_6_1_improvement_roadmap_20260707.md):
three no-retrain robustness audits of the FROZEN live Omega4.6.1 config (h48qual+zig075 greedy
router, static TP/SL, duration gate). Does not touch trading_bot.py or search for a new model --
purely quantifies risks already flagged as unverified.

1. Cost stress test (cost_mult in {1,2,3}) on VAL and OOS -- this project's own Alpha1 lesson
   ("wins at cost1, loses at cost3 -> not a real edge") has never been applied to Omega4.6.1.
2. Leave-one-out (jackknife) trade sensitivity on VAL and OOS -- quantifies how much PnL/MDD
   depend on any single trade, given only 22-37 trades per window.
3. Rolling walk-forward diagnostic on additional non-overlapping 2025 quarters (Q1/Q2/Q3, using
   the same frozen artifacts' train_predictions_*.csv) -- purely diagnostic evidence of whether the
   zig075-SHORT edge recurs across regimes or was concentrated in the two windows already used for
   selection/confirmation. NOT used for any selection decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402
from test_omega4_6_1_drop_h48qual_20260706 import _metrics  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_phase1_robustness_20260707"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = retest.DEVICE


def load_val_components(cost_mult_unused=None):
    frame = valmod.load_val_frame()
    components = {}
    for cname, cfg in retest.COMPONENTS.items():
        pred = pd.read_csv(valmod.VAL_PRED[cname])
        pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[(pred["timestamp"] >= valmod.START) & (pred["timestamp"] <= valmod.END)].reset_index(drop=True)
        common = frame["timestamp"].isin(pred["timestamp"])
        frame = frame[common].reset_index(drop=True)
        pred = pred[pred["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
        tmp = OUT_DIR / f"_val_{cname}_aligned.csv"
        pred.to_csv(tmp, index=False)
        components[cname] = greedy.prepare_component(frame, tmp, cfg, DEVICE)
    return frame, components


def load_oos_components():
    frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    components = {}
    for cname, cfg in retest.COMPONENTS.items():
        pred_csv = retest.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"
        components[cname] = greedy.prepare_component(frame, pred_csv, cfg, DEVICE)
    return frame, components


def load_2025_quarter_components(start: str, end: str):
    """Diagnostic-only: reuse each bundle's own TRAIN-window predictions (2025-01-01..09-30
    coverage confirmed) restricted to a quarter, exactly like the VAL loader but pointed at
    train_predictions_*.csv instead of validation_predictions_*.csv."""
    q_pred = {
        "h48qual": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/train_predictions_q050.csv",
        "zig075": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/train_predictions_q075.csv",
    }
    frame = pd.read_csv(valmod.BASE_2025, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(valmod.WIDE24_2025, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].reset_index(drop=True)

    components = {}
    for cname, cfg in retest.COMPONENTS.items():
        pred = pd.read_csv(q_pred[cname])
        pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[(pred["timestamp"] >= start) & (pred["timestamp"] <= end)].reset_index(drop=True)
        common = frame["timestamp"].isin(pred["timestamp"])
        frame = frame[common].reset_index(drop=True)
        pred = pred[pred["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
        tmp = OUT_DIR / f"_q_{cname}_{start}_aligned.csv"
        pred.to_csv(tmp, index=False)
        components[cname] = greedy.prepare_component(frame, tmp, cfg, DEVICE)
    return frame, components


def jackknife(ledger: pd.DataFrame, frame: pd.DataFrame) -> dict:
    if ledger.empty:
        return {"full": {"pnl": 0.0, "mdd": 0.0, "trades": 0}, "loo": []}
    full = _metrics(ledger, frame, apply_gate=True)
    rows = []
    for idx in ledger.index:
        sub = ledger.drop(index=idx)
        m = _metrics(sub, frame, apply_gate=True)
        rows.append({"removed_entry_timestamp": str(ledger.loc[idx, "entry_timestamp"]),
                      "removed_source": str(ledger.loc[idx, "source_component"]),
                      "removed_return": float(ledger.loc[idx, "trade_return"]),
                      "pnl_without": m["pnl"], "mdd_without": m["mdd"]})
    rows.sort(key=lambda r: r["pnl_without"])
    return {"full": full, "loo": rows}


def main() -> int:
    fee, slip = omega._load_fee_slip()
    result: dict = {}

    # ---------------- 1. Cost stress test ----------------
    print("=== 1. Cost stress test (cost_mult 1x/2x/3x) ===", flush=True)
    val_frame, val_components = load_val_components()
    oos_frame, oos_components = load_oos_components()
    cost_stress = {"val": {}, "oos": {}}
    for label, frame, components in [("val", val_frame, val_components), ("oos", oos_frame, oos_components)]:
        for mult in (1.0, 2.0, 3.0):
            greedy.PRIORITY = ("h48qual", "zig075")
            _, lg = greedy.greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=mult, device=DEVICE)
            m = _metrics(lg, frame, apply_gate=True)
            cost_stress[label][f"cost{int(mult)}x"] = m
            print(f"  {label} cost{int(mult)}x -> pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+6.2f}% n={m['trades']:2d} wr={m['wr']:.3f}", flush=True)
    result["cost_stress"] = cost_stress

    # ---------------- 2. Leave-one-out jackknife (cost1x, gated) ----------------
    print("\n=== 2. Leave-one-out (jackknife) trade sensitivity (cost1x) ===", flush=True)
    jk = {}
    for label, frame, components in [("val", val_frame, val_components), ("oos", oos_frame, oos_components)]:
        greedy.PRIORITY = ("h48qual", "zig075")
        _, lg = greedy.greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=1.0, device=DEVICE)
        j = jackknife(lg, frame)
        jk[label] = j
        full = j["full"]
        worst = j["loo"][0] if j["loo"] else None
        best = j["loo"][-1] if j["loo"] else None
        print(f"  {label} FULL: pnl={full['pnl']:+.2f}% mdd={full['mdd']:+.2f}%", flush=True)
        if worst:
            print(f"    removing the trade that helps MOST swings pnl to {worst['pnl_without']:+.2f}% "
                  f"(removed {worst['removed_source']} ret={worst['removed_return']:+.4f} @ {worst['removed_entry_timestamp']})", flush=True)
        if best:
            print(f"    removing the trade that hurts MOST swings pnl to {best['pnl_without']:+.2f}% "
                  f"(removed {best['removed_source']} ret={best['removed_return']:+.4f} @ {best['removed_entry_timestamp']})", flush=True)
    result["jackknife"] = jk

    # ---------------- 3. Rolling walk-forward diagnostic (2025 Q1/Q2/Q3) ----------------
    print("\n=== 3. Rolling walk-forward diagnostic (2025 Q1/Q2/Q3, DIAGNOSTIC ONLY) ===", flush=True)
    quarters = [("2025-01-01", "2025-03-31 23:59:59", "2025-Q1"),
                ("2025-04-01", "2025-06-30 23:59:59", "2025-Q2"),
                ("2025-07-01", "2025-09-30 23:59:59", "2025-Q3")]
    wf = {}
    for start, end, label in quarters:
        frame, components = load_2025_quarter_components(start, end)
        greedy.PRIORITY = ("h48qual", "zig075")
        _, lg = greedy.greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=1.0, device=DEVICE)
        m = _metrics(lg, frame, apply_gate=True)
        if not lg.empty:
            lg["sidestr"] = np.where(lg["side"] > 0, "L", "S")
            breakdown = lg.groupby(["source_component", "sidestr"])["trade_return"].sum().round(4).to_dict()
        else:
            breakdown = {}
        wf[label] = {"metrics": m, "component_side_sum_ret": {str(k): v for k, v in breakdown.items()}}
        print(f"  {label}: pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+6.2f}% n={m['trades']:2d} wr={m['wr']:.3f}  breakdown={breakdown}", flush=True)
    result["rolling_walk_forward_diagnostic"] = wf

    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {OUT_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
