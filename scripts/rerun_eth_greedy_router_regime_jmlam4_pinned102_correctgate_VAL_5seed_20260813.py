"""VAL-window (2025-10-01..12-31) JM-regime full-retrain N=5-seed robustness comparison against the
live HMM baseline. Reuses greedy_replay/prepare_component exactly as
rerun_eth_greedy_router_regime_jmlam4_pinned102_correctgate_20260810.py (the OOS single-seed JM
comparison, correct -25% MDD floor / 0.45-0.95 notional risk gate) and
replay_omega4_6_1_greedy_val_20260706.py (the VAL-window HMM baseline harness) both do. Both parent
bundles per seed (h48qual pinned102 JM q070, zig075 pinned102 JM q080) and their matching correctgate
sidecars must already exist under tmp/causal_regen_20260516/ before running this
(scripts/run_jm_full_retrain_seed_robustness_20260813.sh produces them).

OOS (2026-01-01..02-28) is deliberately NOT touched here -- already read once on 2026-08-10
(single seed). This script is VAL-only per the orchestrator's explicit instruction.

Fresh-Forward note: predictions consumed here are frozen per-bar model outputs the training script
itself generated causally (fixed train/validation/oos date-boundary split, no future leakage into
any row). This script only replays bar-by-bar TP/SL/duration-gate PnL over those frozen decisions --
identical mechanics to every other script in this family (rerun_*_correctgate_20260810.py,
replay_omega4_6_1_greedy_val_20260706.py). Saved trade ledgers this script writes are its own fresh
output, not reused as input to any decision.
"""
from __future__ import annotations

import json
import sys
import traceback
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

OUT_DIR = ROOT / "tmp/eth_greedy_router_regime_jmlam4_pinned102_correctgate_VAL_5seed_20260813"
JM_REGIME3_2025 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2025_maskedname.csv"
SEEDS = [323033734, 50011403, 504028524, 782182142, 393423992]
START, END = valmod.START, valmod.END  # "2025-10-01", "2025-12-31 23:59:59"


def curve_metrics(returns: np.ndarray) -> dict:
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": round(float((curve[-1] - 1.0) * 100.0), 4),
            "mdd": round(float(dd.min() * 100.0), 4),
            "trades": int(len(returns)),
            "wr": round(float((returns > 0).mean()), 4) if len(returns) else 0.0}


def load_frame(overlay_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(valmod.BASE_2025, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(overlay_csv, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= START) & (frame["timestamp"] <= END)].reset_index(drop=True)
    return frame


def load_pred(pred_csv: Path, frame_ts: pd.Series) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    # validation_predictions_*.csv uses the OOF-format prefix (omega1_regime3_expertdq_oof_*);
    # rename to the non-oof prefix so prepare_component's oof=False decision path works unchanged
    # -- see replay_omega4_6_1_greedy_val_20260706.py, which established this exact fix.
    pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    pred = pred[(pred["timestamp"] >= START) & (pred["timestamp"] <= END)]
    pred = pred[pred["timestamp"].isin(frame_ts)].sort_values("timestamp").reset_index(drop=True)
    return pred


def run_router(tag: str, frame_c: pd.DataFrame, component_specs: dict, device) -> dict:
    """component_specs: {name: (cfg, pred_csv)} -- 1 or 2 entries. A component is allowed to be
    absent entirely (e.g. h48qual's sidecar failing to train because its VAL ledger was empty at
    q070 for a given seed -- itself a real, reportable finding, not something to paper over) so the
    router still runs zig075-only rather than skipping the whole seed.

    Prediction CSVs can be a handful of rows short of the frame (seen both in the live HMM
    baseline's h48qual validation_predictions_q050.csv, 6 rows short, and in some JM seeds) --
    replay_omega4_6_1_greedy_val_20260706.py handles this by narrowing the shared frame to
    whichever component's timestamps are sparsest. Doing that per-component in a loop is
    order-dependent (an earlier component's already-prepared tensors wouldn't be re-narrowed by a
    later component's gap), so here the full intersection across the frame AND every component's
    own predictions is computed once upfront, before any component is prepared."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = {}
    ts = frame_c["timestamp"]
    for name, (cfg, pred_csv) in component_specs.items():
        pred = load_pred(pred_csv, ts)
        preds[name] = pred
        ts = ts[ts.isin(pred["timestamp"])]
    frame_c = frame_c[frame_c["timestamp"].isin(ts)].sort_values("timestamp").reset_index(drop=True)

    components = {}
    for name, (cfg, _pred_csv) in component_specs.items():
        pred = preds[name]
        pred = pred[pred["timestamp"].isin(frame_c["timestamp"])].sort_values("timestamp").reset_index(drop=True)
        if len(pred) != len(frame_c) or not pred["timestamp"].equals(frame_c["timestamp"]):
            raise RuntimeError(f"[{tag}] {name}: prediction rows = {len(pred)} != frame rows {len(frame_c)}")
        tmp = OUT_DIR / f"_aligned_{tag}_{name}.csv"
        pred.to_csv(tmp, index=False)
        components[name] = greedy.prepare_component(frame_c, tmp, cfg, device)
        print(f"[{tag}] {name}: prepared nonzero_side={(components[name]['dec']['side'] != 0).mean():.4f} rows={len(frame_c)}", flush=True)

    fee, slip = omega._load_fee_slip()
    _, ledger = greedy.greedy_replay(frame_c, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    components_used = sorted(components.keys())
    if ledger.empty:
        empty = {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
        return {"no_gate": empty, "with_gate": {**empty, "skipped": 0}, "source_component_counts": {},
                "components_used": components_used}

    returns = ledger["trade_return"].to_numpy(dtype=float)
    no_gate = curve_metrics(returns)

    market = frame_c[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp_dt"})
    led = ledger.copy()
    led["entry_timestamp_dt"] = pd.to_datetime(led["entry_timestamp"])
    led = led.merge(market, on="entry_timestamp_dt", how="left")
    hit = led["ou_halflife"] <= greedy.DURATION_THRESHOLD
    gated = curve_metrics(led.loc[~hit, "trade_return"].to_numpy(dtype=float))
    gated["skipped"] = int(hit.sum())

    led.to_csv(OUT_DIR / f"ledger_{tag}.csv", index=False)
    return {"no_gate": no_gate, "with_gate": gated,
            "source_component_counts": ledger["source_component"].value_counts().to_dict(),
            "components_used": components_used}


def main() -> int:
    device = retest.DEVICE
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading baseline (HMM) VAL frame...", flush=True)
    frame_hmm_full = load_frame(valmod.WIDE24_2025)
    print("Loading JM VAL frame...", flush=True)
    frame_jm_full = load_frame(JM_REGIME3_2025)
    common_ts = set(frame_hmm_full["timestamp"]) & set(frame_jm_full["timestamp"])
    print(f"common_ts (hmm frame ∩ jm frame) = {len(common_ts)}  "
          f"(hmm={len(frame_hmm_full)} jm={len(frame_jm_full)})", flush=True)
    frame_hmm = frame_hmm_full[frame_hmm_full["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    frame_jm = frame_jm_full[frame_jm_full["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)

    results: dict = {"window": [START, END], "common_bars": len(common_ts), "seeds": SEEDS}

    print("\n=== baseline_hmm ===", flush=True)
    baseline_specs = {
        "h48qual": (retest.COMPONENTS["h48qual"], valmod.VAL_PRED["h48qual"]),
        "zig075": (retest.COMPONENTS["zig075"], valmod.VAL_PRED["zig075"]),
    }
    results["baseline_hmm"] = run_router("baseline_hmm", frame_hmm, baseline_specs, device)
    print(json.dumps(results["baseline_hmm"], indent=2), flush=True)

    for seed in SEEDS:
        tag = f"jm_seed{seed}"
        print(f"\n=== {tag} ===", flush=True)
        h48_dir = ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime_jmlam4_20260809_h48qual_ext_seed{seed}"
        zig_dir = ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime_jmlam4_20260809_zig075_seed{seed}"
        h48_sidecar = ROOT / f"tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_pinned102_jmlam4_q070_correctgate_seed{seed}_20260813/risk_sidecar.pkl"
        zig_sidecar = ROOT / f"tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_pinned102_jmlam4_q080_correctgate_seed{seed}_20260813/risk_sidecar.pkl"

        # Each component is included independently -- a missing bundle/sidecar for ONE component
        # (e.g. h48qual's sidecar failing to train because its VAL ledger was empty at q070 for
        # this seed) does not disqualify the other; the router just runs on whichever component(s)
        # are actually available, exactly like the live PRIORITY=("h48qual","zig075") router would
        # naturally do if one side produced zero gate-passed trades.
        component_specs = {}
        skipped_components = {}
        if (h48_dir / "true_3head_tabm_bundle.pt").exists() and h48_sidecar.exists():
            h48_cfg = dict(retest.COMPONENTS["h48qual"])
            h48_cfg["bundle"], h48_cfg["sidecar_pkl"] = h48_dir / "true_3head_tabm_bundle.pt", h48_sidecar
            h48_cfg["q_tag"], h48_cfg["quality_threshold"] = "q070", 0.70
            component_specs["h48qual"] = (h48_cfg, h48_dir / "validation_predictions_q070.csv")
        else:
            skipped_components["h48qual"] = "missing_bundle_or_sidecar"
        if (zig_dir / "true_3head_tabm_bundle.pt").exists() and zig_sidecar.exists():
            zig_cfg = dict(retest.COMPONENTS["zig075"])
            zig_cfg["bundle"], zig_cfg["sidecar_pkl"] = zig_dir / "true_3head_tabm_bundle.pt", zig_sidecar
            zig_cfg["q_tag"], zig_cfg["quality_threshold"] = "q080", 0.80
            component_specs["zig075"] = (zig_cfg, zig_dir / "validation_predictions_q080.csv")
        else:
            skipped_components["zig075"] = "missing_bundle_or_sidecar"

        if not component_specs:
            print(f"[{tag}] SKIP: no components available {skipped_components}", flush=True)
            results[tag] = {"error": "no_components_available", "skipped_components": skipped_components}
            continue

        try:
            results[tag] = run_router(tag, frame_jm, component_specs, device)
            if skipped_components:
                results[tag]["skipped_components"] = skipped_components
            print(json.dumps(results[tag], indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001 -- one seed's failure must not kill the other 4
            print(f"[{tag}] ERROR: {exc}", flush=True)
            traceback.print_exc()
            results[tag] = {"error": str(exc)}

    (OUT_DIR / "result.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n\n=== FINAL result.json ===", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
