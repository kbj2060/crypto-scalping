"""czz_trend regime sizing overlay applied to the MULTI-SLOT SHADOW model (2026-08-08).

Shadow config being replayed (scripts/run_btc_multislot_shadow_loop_20260807.py):
N_SLOTS=3, MARGIN_MULT=1.5, cost_mult=3.0, exit threshold 0.95 -- i.e. per-slot margin is
`sidecar_margin * 1.5 / 3`.  Recorded shadow numbers: OOS gated +19.98% / MDD -10.40%.

The overlay (data/research/btc_regime_sizing_risk_overlay_frozen_20260808.json) multiplies the
sidecar margin_fraction by a fixed czz4-regime multiplier (bear 0.5 / chop 1.0 / bull 1.5) looked
up at the ENTRY-SIGNAL bar, before the /N_SLOTS split.  Both components are frozen fixed rules
with no fitted parameters, so this is a MEASUREMENT of an existing combination, not a new
selection: nothing here is tuned, and no promotion claim is attached -- the shadow's own live
record remains the referee for the shadow model.

A regression check runs first: the no-overlay N=3 x1.5 replay must reproduce the recorded shadow
numbers, otherwise the setup is wrong and the comparison is meaningless.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

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
import apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806 as apply_mod  # noqa: E402
from research_btc_swingtransition_multislot_20260807 import _replay_multislot  # noqa: E402
from research_btc_swingtransition_regime_sizing_overlay_20260808 import (  # noqa: E402
    MAPS, overlay_margin, regime_lookup,
)
from research_btc_swingtransition_trailing_stop_val_oos_20260807 import _compound_metrics, _gate  # noqa: E402

OUT_DIR = ROOT / "tmp/btc_multislot_shadow_regime_sizing_20260808"
N_SLOTS, MARGIN_MULT, COST_MULT, EXIT_THRESHOLD = 3, 1.5, 3.0, 0.95
# CORRECTED 2026-08-08.  The previously recorded shadow figure was OOS +19.98% / -10.40%, produced
# by an ad-hoc sweep that RESCALED the N=3 gated ledger returns.  That is invalid here: margin
# feeds the exit head (notional/leverage sit in pos_values), so changing margin changes the exits
# and the ledger itself.  The full causal replay below is the authority, and it is reproduced
# independently by scripts/resweep_btc_multislot_margin_multiplier_fullreplay_20260808.py.
SHADOW_EXPECTED = {"oos_pnl": 25.30, "oos_mdd": -10.77}
SHADOW_EXPECTED_SUPERSEDED = {"oos_pnl": 19.98, "oos_mdd": -10.40,
                              "why": "ledger-rescaling sweep; margin feeds the exit head"}
TOL = 0.35


def prepare(device, margin_mult: float = MARGIN_MULT) -> dict:
    """Load the promoted swingtransition stack and build per-split replay inputs.

    Shared with the multiplier re-sweep so the two paths cannot drift -- a second copy of this
    block is exactly the class of provenance bug this correction exists to fix.
    """
    bundle_path = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition/true_3head_tabm_bundle.pt"
    sidecar_path = ROOT / "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260806_swingtransition/risk_sidecar.pkl"
    pred_dir = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_freshforward_ext_20260806"

    print("stage=load", flush=True)
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
        bm = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"],
                                   **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
        bl = sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"],
                                    **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS})
        margin, leverage = apply_mod._scaled_margin_leverage(dec, bm, bl, long_scale=0.5, short_scale=2.5)
        ou = raw[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})
        ou["entry_timestamp"] = pd.to_datetime(ou["entry_timestamp"])
        data[split] = dict(raw=raw, x=x, dec=dec, margin=np.asarray(margin, dtype=np.float64) * margin_mult,
                           leverage=leverage, ou=ou, reg=regime_lookup(raw))
        print(json.dumps({"prepared": split, "rows": int(len(raw))}), flush=True)
    return {"data": data, "loaded": loaded, "fee": fee, "slip": slip}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--maps", nargs="*", default=["identity", "czz_trend"])
    args = ap.parse_args()
    device = parent._device(str(args.device))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prep = prepare(device)
    data, loaded, fee, slip = prep["data"], prep["loaded"], prep["fee"], prep["slip"]

    results = {}
    for name in args.maps:
        for split in ("validation", "oos"):
            d = data[split]
            m = overlay_margin(d["margin"], d["dec"], d["reg"], MAPS[name])
            led = _replay_multislot(d["raw"], d["x"], d["dec"], loaded, n_slots=N_SLOTS,
                                    risk_margin_fraction=m, risk_leverage=d["leverage"],
                                    exit_threshold=EXIT_THRESHOLD, fee=fee, slip=slip,
                                    cost_mult=COST_MULT, device=device)
            led.to_csv(OUT_DIR / f"{split}_ledger_n{N_SLOTS}_{name}.csv", index=False)
            g = _gate(led, d["ou"])
            entry = {"ungated": _compound_metrics(led), "gated": _compound_metrics(g)}
            if split == "oos" and len(g):
                gg = g.copy()
                gg["q"] = pd.to_datetime(gg["entry_timestamp"]).dt.to_period("Q")
                entry["gated_quarters"] = {str(q): round(float(((1 + s["trade_return"]).prod() - 1) * 100), 2)
                                           for q, s in gg.groupby("q")}
            results[f"{split}|{name}"] = entry
            print(json.dumps({f"{split}|{name}": entry["gated"]}), flush=True)

    reg_ok = None
    if "identity" in args.maps:
        og = results["oos|identity"]["gated"]
        reg_ok = bool(abs(og["pnl"] - SHADOW_EXPECTED["oos_pnl"]) <= TOL
                      and abs(og["mdd"] - SHADOW_EXPECTED["oos_mdd"]) <= TOL)
        print(json.dumps({"regression_vs_recorded_shadow": {"expected": SHADOW_EXPECTED,
                                                            "got": {"pnl": round(og["pnl"], 2),
                                                                    "mdd": round(og["mdd"], 2)},
                                                            "ok": reg_ok}}, indent=2), flush=True)

    out = {"shadow_config": {"n_slots": N_SLOTS, "margin_mult": MARGIN_MULT, "cost_mult": COST_MULT,
                             "exit_threshold": EXIT_THRESHOLD},
           "overlay": "czz_trend (bear 0.5 / chop 1.0 / bull 1.5 on margin_fraction at the entry bar)",
           "regression_vs_recorded_shadow_ok": reg_ok,
           "shadow_baseline_provenance": {"corrected": SHADOW_EXPECTED,
                                          "superseded": SHADOW_EXPECTED_SUPERSEDED},
           "results": results,
           "note": "measurement of two frozen rules in combination; not a selection and not a "
                   "promotion claim. The shadow's own live record remains the referee.",
           "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
           "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
