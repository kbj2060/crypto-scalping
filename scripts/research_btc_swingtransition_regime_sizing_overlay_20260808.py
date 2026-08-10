"""Regime-conditioned SIZING overlay on the promoted BTC swingtransition stack (2026-08-08).

The regime-conditioned ENTRY axis is closed (D2 gate OOS -19.5% on 2026-08-08; the JM/czz
MoE rerun did not clear VAL gates either).  This line moves the regime lever to the one
place a lever is still open: redistributing the ALREADY-PROMOTED parent's sidecar risk
sizing across regimes, without touching entries, exits, or total budget scale.  Failure
mode differs from the closed entry axis: it reweights an existing positive edge instead
of trying to extract a new one.

Mechanics: byte-identical N=1 replay of the promoted model (same _replay_with_risk call
as the promoted report; regression-gated against its VAL/OOS numbers), with
risk_margin_fraction[i] multiplied by a regime multiplier looked up at the ENTRY-SIGNAL
bar from data/research/btc_jm_regime_states_20260808.parquet (causal detectors: JM k3
lam32 and causal-zigzag 4%).  Futures sizing contract respected: notional = margin *
leverage; TP/SL stay price-move targets (notional_scaled_sltp unchanged).

PRE-REGISTERED (2026-08-08 04:30 KST, before any overlay result was seen):
  maps (multiplier by regime {bear, chop, bull} of the entry bar, or side-consensus):
    jm_trend       jm_lam32:   bear 0.5, chop 1.0, bull 1.5
    jm_skip_bear   jm_lam32:   bear 0.0, chop 1.0, bull 1.0
    jm_contra      jm_lam32:   bear 1.5, chop 1.0, bull 0.5  (JM-bear bounce hypothesis)
    czz_trend      czz4:       bear 0.5, chop 1.0, bull 1.5
    czz_consensus  czz4:       x1.25 when entry side matches wave direction else x0.5
    jm_consensus   jm_lam32:   same side rule, chop bars -> x1.0
  selection: VAL gated PnL max, subject to VAL gated MDD >= -8% (same bar as multislot line);
  adoption after ONE OOS read requires ALL: OOS gated PnL >= baseline+2pp (>= 12.76),
  OOS gated MDD >= -14.4 (baseline -12.41 minus 2pp), worst OOS quarter >= -4.0.
  No re-tuning after the OOS read.  Duration gate stays a ledger post-filter (metric
  convention identical to the promoted report).
Fresh-forward flags: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
import apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806 as apply_mod  # noqa: E402
from research_btc_swingtransition_multislot_20260807 import BASELINE_EXPECTED  # noqa: E402
from research_btc_swingtransition_trailing_stop_val_oos_20260807 import _compound_metrics, _gate  # noqa: E402

STATES_PATH = ROOT / "data/research/btc_jm_regime_states_20260808.parquet"
OUT_DIR = ROOT / "tmp/btc_regime_sizing_overlay_20260808"

MAPS: dict[str, dict[str, Any]] = {
    "identity": {"src": None},
    "jm_trend": {"src": "jm_lam32", "mult": {0: 0.5, 1: 1.0, 2: 1.5}},
    "jm_skip_bear": {"src": "jm_lam32", "mult": {0: 0.0, 1: 1.0, 2: 1.0}},
    "jm_contra": {"src": "jm_lam32", "mult": {0: 1.5, 1: 1.0, 2: 0.5}},
    "czz_trend": {"src": "czz4", "mult": {0: 0.5, 1: 1.0, 2: 1.5}},
    "czz_consensus": {"src": "czz4", "consensus": (1.25, 0.5)},
    "jm_consensus": {"src": "jm_lam32", "consensus": (1.25, 0.5)},
}


def regime_lookup(raw: pd.DataFrame) -> pd.DataFrame:
    st = pd.read_parquet(STATES_PATH)[["timestamp", "jm_lam32", "czz4"]]
    st["timestamp"] = pd.to_datetime(st["timestamp"])
    ts = pd.to_datetime(raw["timestamp"]).reset_index(drop=True)
    merged = pd.merge_asof(pd.DataFrame({"timestamp": ts}), st.sort_values("timestamp"),
                          on="timestamp", direction="backward", tolerance=pd.Timedelta("10min"))
    n_missing = int(merged["jm_lam32"].isna().sum())
    if n_missing:
        print(json.dumps({"warn_regime_rows_unmatched": n_missing}), flush=True)
    return merged.fillna(1)


def overlay_margin(margin: np.ndarray, dec: pd.DataFrame, reg: pd.DataFrame, spec: dict) -> np.ndarray:
    out = np.asarray(margin, dtype=np.float64).copy()
    if spec["src"] is None:
        return out
    states = pd.to_numeric(reg[spec["src"]], errors="raise").to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    if "mult" in spec:
        m = np.vectorize(spec["mult"].get)(states).astype(np.float64)
    else:
        hi, lo = spec["consensus"]
        wave_dir = np.where(states == 2, 1, np.where(states == 0, -1, 0))
        m = np.where(wave_dir == 0, 1.0, np.where(side == wave_dir, hi, lo))
    return out * m[: len(out)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--stage", choices=["val", "oos", "risk"], required=True)
    args = ap.parse_args()
    device = parent._device(str(args.device))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        bm = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
        bl = sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS})
        margin, leverage = apply_mod._scaled_margin_leverage(dec, bm, bl, long_scale=0.5, short_scale=2.5)
        ou = raw[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})
        ou["entry_timestamp"] = pd.to_datetime(ou["entry_timestamp"])
        reg = regime_lookup(raw)
        data[split] = dict(raw=raw, x=x, dec=dec, margin=margin, leverage=leverage, ou=ou, reg=reg)

    def replay(split: str, map_name: str) -> dict[str, Any]:
        d = data[split]
        m = overlay_margin(d["margin"], d["dec"], d["reg"], MAPS[map_name])
        metrics, ledger = sidecar._replay_with_risk(
            d["raw"], d["x"], d["dec"], loaded,
            risk_margin_fraction=m, risk_leverage=d["leverage"],
            exit_threshold=0.95, fee=fee, slip=slip, cost_mult=3.0,
            notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device)
        g = _gate(ledger, d["ou"])
        out = {"map": map_name, "split": split,
               "ungated": _compound_metrics(ledger), "gated": _compound_metrics(g)}
        if split == "oos" and len(g):
            g2 = g.copy()
            g2["q"] = pd.to_datetime(g2["entry_timestamp"]).dt.to_period("Q")
            out["quarters"] = {str(q): float(((1 + dd["trade_return"]).prod() - 1) * 100) for q, dd in g2.groupby("q")}
        ledger.to_csv(OUT_DIR / f"{split}_ledger_{map_name}.csv", index=False)
        return out

    if args.stage == "risk":
        # Single pass for the risk-first contract (docs/experiments/btc_regime_sizing_risk_first_20260808.json):
        # replay EVERY map on BOTH splits so the risk analysis has a complete ledger set. The five
        # maps never read on OOS are measured here once; no iteration follows.
        summary = {}
        for split in ("validation", "oos"):
            for name in MAPS:
                r = replay(split, name)
                summary[f"{split}|{name}"] = {"gated": r["gated"], "ungated": r["ungated"]}
                print(json.dumps({f"{split}|{name}": r["gated"]}), flush=True)
        (OUT_DIR / "risk_replays.json").write_text(json.dumps(summary, indent=2))
        print(f"wrote {OUT_DIR / 'risk_replays.json'}")
    elif args.stage == "val":
        base = replay("validation", "identity")
        print(json.dumps(base), flush=True)
        checks = {"val_gated_pnl": base["gated"]["pnl"], "val_gated_mdd": base["gated"]["mdd"]}
        for k in ("val_gated_pnl", "val_gated_mdd"):
            if abs(checks[k] - BASELINE_EXPECTED[k]) > 0.05:
                raise SystemExit(f"identity REGRESSION FAIL: {k}={checks[k]:.4f} expected {BASELINE_EXPECTED[k]:.4f}")
        print("identity regression PASS", flush=True)
        table = [base]
        for name in MAPS:
            if name == "identity":
                continue
            r = replay("validation", name)
            table.append(r)
            print(json.dumps(r), flush=True)
        eligible = [r for r in table if r["map"] != "identity" and r["gated"]["mdd"] >= -8.0
                    and r["gated"]["pnl"] > base["gated"]["pnl"]]
        sel = max(eligible, key=lambda r: r["gated"]["pnl"]) if eligible else None
        out = {"baseline_val": base, "table": table,
               "selected_map": None if sel is None else sel["map"],
               "earns_oos_read": sel is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"selected_map": out["selected_map"], "earns_oos_read": out["earns_oos_read"]}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- no map beat identity on VAL within MDD bar"}))
            return 1
        name = prior["selected_map"]
        base = replay("oos", "identity")
        for k, exp in (("oos_gated_pnl", BASELINE_EXPECTED["oos_gated_pnl"]), ("oos_gated_mdd", BASELINE_EXPECTED["oos_gated_mdd"])):
            v = base["gated"]["pnl"] if k.endswith("pnl") else base["gated"]["mdd"]
            if abs(v - exp) > 0.05:
                raise SystemExit(f"identity OOS REGRESSION FAIL: {k}={v:.4f} expected {exp:.4f}")
        r = replay("oos", name)
        wq = min(r.get("quarters", {"none": 0.0}).values())
        adopt = bool(r["gated"]["pnl"] >= BASELINE_EXPECTED["oos_gated_pnl"] + 2.0
                     and r["gated"]["mdd"] >= -14.4 and wq >= -4.0)
        out = {"stage": "oos", "selected_map": name, "baseline_oos": base, "overlay_oos": r,
               "oos_worst_quarter": wq, "adopt": adopt,
               "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
               "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
        (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
