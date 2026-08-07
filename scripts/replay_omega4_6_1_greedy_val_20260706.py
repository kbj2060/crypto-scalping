"""VAL-window (2025-10-01..12-31) greedy replay of Omega4.6.1, reusing the exact same frozen
artifacts, greedy_replay logic, SCALE_MAP, caps and duration gate as the 2026 OOS replay
(replay_omega4_6_1_greedy_router_20260706.py). Purpose: robustness check on the striking OOS
finding that the model's ENTIRE edge came from the `zig075 SHORT` bucket while every other
component/side bucket was net-negative. If that asymmetry is genuine (structural) it should also
appear on VAL, which was NOT used to select any of these components' internals. If it's OOS
small-sample luck, VAL will look different. This is DIAGNOSTIC ONLY -- no promotion decision is
made from stored-ledger VAL numbers per the project's Fresh-Forward Rule.
"""
from __future__ import annotations

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

BASE_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
WIDE24_2025 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
VAL_PRED = {
    "h48qual": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/validation_predictions_q050.csv",
    "zig075": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/validation_predictions_q075.csv",
}
START, END = "2025-10-01", "2025-12-31 23:59:59"


def load_val_frame() -> pd.DataFrame:
    frame = pd.read_csv(BASE_2025, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(WIDE24_2025, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= START) & (frame["timestamp"] <= END)].reset_index(drop=True)
    return frame


def main() -> int:
    device = retest.DEVICE
    frame = load_val_frame()
    print(f"VAL frame rows={len(frame)} range={frame['timestamp'].iloc[0]}..{frame['timestamp'].iloc[-1]}", flush=True)
    fee, slip = omega._load_fee_slip()

    components = {}
    for name, cfg in retest.COMPONENTS.items():
        pred = pd.read_csv(VAL_PRED[name])
        # VAL predictions are OOF-format (omega1_regime3_expertdq_oof_*); rename to the non-oof
        # prefix so the whole downstream pipeline (which reads oof=False columns) works unchanged
        # and produces byte-identical decisions to how it would treat an OOS prediction file.
        pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[(pred["timestamp"] >= START) & (pred["timestamp"] <= END)].reset_index(drop=True)
        # align frame to the intersection of timestamps
        common = frame["timestamp"].isin(pred["timestamp"])
        if not common.all():
            frame_c = frame[common].reset_index(drop=True)
        else:
            frame_c = frame
        pred = pred[pred["timestamp"].isin(frame_c["timestamp"])].reset_index(drop=True)
        # write aligned pred to a temp csv path prepare_component can read
        tmp_pred = ROOT / f"tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/_val_{name}_aligned.csv"
        pred.to_csv(tmp_pred, index=False)
        components[name] = greedy.prepare_component(frame_c, tmp_pred, cfg, device)
        print(f"{name}: prepared, nonzero_side={(components[name]['dec']['side'] != 0).mean():.3f}", flush=True)
        frame = frame_c  # all components share aligned frame

    _, ledger = greedy.greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    if ledger.empty:
        print("NO TRADES on VAL")
        return 0
    ledger["sidestr"] = np.where(ledger["side"] > 0, "L", "S")

    returns = ledger["trade_return"].to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    print(f"\n=== VAL greedy (no gate) pnl={((curve[-1]-1)*100):.2f}% mdd={dd.min()*100:.2f}% trades={len(ledger)} wr={(returns>0).mean():.3f} ===")

    # duration gate
    market = frame[["timestamp", "ou_halflife"]].copy()
    ledger["entry_timestamp_dt"] = pd.to_datetime(ledger["entry_timestamp"])
    ledger = ledger.merge(market.rename(columns={"timestamp": "entry_timestamp_dt"}), on="entry_timestamp_dt", how="left")
    hit = ledger["ou_halflife"] <= greedy.DURATION_THRESHOLD
    gret = np.where(hit, 0.0, ledger["trade_return"])
    curve_g = np.concatenate([[1.0], np.cumprod(1.0 + gret)])
    peak_g = np.maximum.accumulate(curve_g)
    dd_g = curve_g / np.maximum(peak_g, 1e-12) - 1.0
    print(f"=== VAL greedy + duration gate pnl={((curve_g[-1]-1)*100):.2f}% mdd={dd_g.min()*100:.2f}% "
          f"active={int((~hit).sum())} skipped={int(hit.sum())} wr={(gret[~hit]>0).mean() if (~hit).any() else 0:.3f} ===")

    print("\n=== component x side breakdown (no gate) ===")
    g = ledger.groupby(["source_component", "sidestr"]).agg(
        n=("win", "size"), wr=("win", "mean"), mean_ret=("trade_return", "mean"),
        sum_ret=("trade_return", "sum"), avg_notional=("notional", "mean"))
    print(g.round(4).to_string())

    print("\n=== reason breakdown ===")
    print(ledger.groupby("reason").agg(n=("win", "size"), wr=("win", "mean"), sum_ret=("trade_return", "sum")).round(4).to_string())

    ledger.to_csv(ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_VAL.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
