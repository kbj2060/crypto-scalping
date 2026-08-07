"""Trailing-stop re-validation on the PROMOTED BTC swingtransition model (2026-08-07).

The G1 S3 trailing stop (trail 0.5*SL @ 0.3*TP) is a KEEP-ALIVE lever measured on a SHORT-hold
triple-barrier research ledger (431-763 trades, median holds far under a day). The promoted
swingtransition model is a LONG-hold model (median hold 689-749 bars; winners' median 1397-2390
bars -- long hold IS the winner signature, same profile as ETH Omega4.6.1 where this lever failed
0/6). Pre-registered expectation: UNFAVORABLE prior; this script exists to close the question
empirically on the exact promoted replay chain.

Methodology mirrors the ETH test (research_eth_omega461_router_btc_trailing_val_oos_20260807):
every config reported on BOTH splits, deliberately NO VAL selection.

PRE-REGISTERED ADOPTION RULE (fixed before running): a config is adoptable only if, on BOTH the
duration-gated VAL and duration-gated OOS-extended splits, (a) MDD improves vs baseline AND
(b) compound PnL is no worse than baseline minus 1.0pp. Anything else = keep live config as-is.

Replay chain is byte-identical to apply_final_scale_map_btc_freshforward_ext_swingtransition_
20260806.py (same predictions, sizing, costs, duration gate fixed at the LIVE threshold); the
baseline (trailing=None) run doubles as a regression test that the default-off trailing argument
added to _replay_with_risk is a true no-op.
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

LIVE_DURATION_THRESHOLD = 0.0054143218
CONFIGS: list[tuple[float, float] | None] = [
    None,          # baseline (regression check vs promoted report)
    (0.3, 0.5),    # the BTC G1 operating point
    (0.3, 1.0),
    (0.5, 0.5),
    (0.5, 1.0),
    (0.8, 0.5),
    (0.8, 1.0),
]
BASELINE_EXPECTED = {  # from the promoted report.json (regression tolerance 0.05pp)
    "val_gated_pnl": 24.226193370361937,
    "val_gated_mdd": -2.4590149966945973,
    "oos_gated_pnl": 10.760766798223663,
    "oos_gated_mdd": -12.410621340770533,
}


def _compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if len(ledger) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    eq = (1.0 + ledger["trade_return"]).cumprod()
    peak = eq.cummax()
    return {
        "pnl": float((eq.iloc[-1] - 1.0) * 100.0),
        "mdd": float(((eq / peak) - 1.0).min() * 100.0),
        "trades": int(len(ledger)),
        "wr": float(ledger["win"].mean()),
    }


def _gate(ledger: pd.DataFrame, ou: pd.DataFrame) -> pd.DataFrame:
    led = ledger.copy()
    led["entry_timestamp"] = pd.to_datetime(led["entry_timestamp"])
    led = led.merge(ou, on="entry_timestamp", how="left", validate="one_to_one")
    return led.loc[led["ou_halflife"] > LIVE_DURATION_THRESHOLD].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/btc_swingtransition_trailing_stop_val_oos_20260807")
    args = ap.parse_args()
    device = parent._device(str(args.device))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition/true_3head_tabm_bundle.pt"
    sidecar_path = ROOT / "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260806_swingtransition/risk_sidecar.pkl"
    pred_dir = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_freshforward_ext_20260806"

    print("stage=load", flush=True)
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(bundle["models"], device=device)
    with open(sidecar_path, "rb") as f:
        pkl = pickle.load(f)

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=ROOT / "tmp/causal_regen_20260516/btc_zigzag_action_labels_freshforward_ext_20260802",
        quality_mode="quality_label_action",
        quality_label_dir=ROOT / "tmp/causal_regen_20260516/btc_h48_conservative_padded_freshforward_ext_20260802",
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()

    val_src = sidecar._load_precomputed_prediction(pred_dir, "validation", "q055", frames["val_raw"])
    oos_src = sidecar._load_precomputed_prediction(pred_dir, "oos", "q055", frames["oos_raw"])
    x_val = parent._base_input(frames["val_raw"], base_cols)
    x_oos = parent._base_input(frames["oos_raw"], base_cols)
    val_dec_base = parent._to_decisions(val_src, oof=True)
    oos_dec_base = parent._to_decisions(oos_src, oof=False)

    atr_kwargs = dict(atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12)
    val_dec, _ = atr_eval._apply_atr_safety_sltp(val_dec_base, frames["val_raw"], **atr_kwargs)
    oos_dec, _ = atr_eval._apply_atr_safety_sltp(oos_dec_base, frames["oos_raw"], **atr_kwargs)
    val_atr = atr_eval._atr_pct(frames["val_raw"], 192)
    oos_atr = atr_eval._atr_pct(frames["oos_raw"], 192)

    print("stage=sizing", flush=True)
    val_features = sidecar._risk_feature_frame(frames["val_raw"], val_src, val_dec, base_cols, atr_pct=val_atr, feature_mode=pkl["risk_feature_mode"])
    oos_features = sidecar._risk_feature_frame(frames["oos_raw"], oos_src, oos_dec, base_cols, atr_pct=oos_atr, feature_mode=pkl["risk_feature_mode"])
    x_val_all, _ = sidecar._feature_matrix(val_features, pkl["feature_columns"])
    x_oos_all, _ = sidecar._feature_matrix(oos_features, pkl["feature_columns"])
    val_side = pd.to_numeric(val_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    oos_side = pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    val_score = sidecar._predict_side_split_models(pkl["model"], x_val_all, val_side)
    oos_score = sidecar._predict_side_split_models(pkl["model"], x_oos_all, oos_side)
    mapping = pkl["selected_mapping"]
    val_bm = sidecar._risk_margins(val_dec, val_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    oos_bm = sidecar._risk_margins(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    val_bl = sidecar._risk_leverage(val_dec, val_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS})
    oos_bl = sidecar._risk_leverage(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS})

    # exact scaling used by the apply script
    import apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806 as apply_mod
    val_margin, val_leverage = apply_mod._scaled_margin_leverage(val_dec, val_bm, val_bl, long_scale=0.5, short_scale=2.5)
    oos_margin, oos_leverage = apply_mod._scaled_margin_leverage(oos_dec, oos_bm, oos_bl, long_scale=0.5, short_scale=2.5)

    val_ou = frames["val_raw"][["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})
    oos_ou = frames["oos_raw"][["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})
    val_ou["entry_timestamp"] = pd.to_datetime(val_ou["entry_timestamp"])
    oos_ou["entry_timestamp"] = pd.to_datetime(oos_ou["entry_timestamp"])

    results: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        label = "baseline" if cfg is None else f"trail{cfg[1]}SL@{cfg[0]}TP"
        kw = {} if cfg is None else {"trailing_activate": cfg[0], "trailing_trail": cfg[1]}
        print(f"stage=replay config={label}", flush=True)
        _, val_led = sidecar._replay_with_risk(frames["val_raw"], x_val, val_dec, loaded, risk_margin_fraction=val_margin, risk_leverage=val_leverage, exit_threshold=0.95, fee=fee, slip=slip, cost_mult=3.0, notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device, **kw)
        _, oos_led = sidecar._replay_with_risk(frames["oos_raw"], x_oos, oos_dec, loaded, risk_margin_fraction=oos_margin, risk_leverage=oos_leverage, exit_threshold=0.95, fee=fee, slip=slip, cost_mult=3.0, notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device, **kw)
        val_g = _gate(val_led, val_ou)
        oos_g = _gate(oos_led, oos_ou)
        oos_q1 = oos_g.loc[oos_g["entry_timestamp"] < pd.Timestamp("2026-04-01")].reset_index(drop=True)
        row = {
            "config": label,
            "val_ungated": _compound_metrics(val_led),
            "oos_ungated": _compound_metrics(oos_led),
            "val_gated": _compound_metrics(val_g),
            "oos_gated": _compound_metrics(oos_g),
            "oos_q1_gated": _compound_metrics(oos_q1),
        }
        results.append(row)
        print(json.dumps(row, indent=None), flush=True)
        if cfg is None:
            checks = {
                "val_gated_pnl": row["val_gated"]["pnl"],
                "val_gated_mdd": row["val_gated"]["mdd"],
                "oos_gated_pnl": row["oos_gated"]["pnl"],
                "oos_gated_mdd": row["oos_gated"]["mdd"],
            }
            for k, expected in BASELINE_EXPECTED.items():
                if abs(checks[k] - expected) > 0.05:
                    raise SystemExit(f"BASELINE REGRESSION: {k}={checks[k]:.4f} expected {expected:.4f} -- trailing no-op broken")
            print("baseline regression check PASS (trailing default is a true no-op)", flush=True)

    base = results[0]
    verdicts = []
    for row in results[1:]:
        ok = (
            row["val_gated"]["mdd"] > base["val_gated"]["mdd"]
            and row["oos_gated"]["mdd"] > base["oos_gated"]["mdd"]
            and row["val_gated"]["pnl"] >= base["val_gated"]["pnl"] - 1.0
            and row["oos_gated"]["pnl"] >= base["oos_gated"]["pnl"] - 1.0
        )
        verdicts.append({"config": row["config"], "adoptable_under_preregistered_rule": bool(ok)})
    report = {
        "method": "btc_swingtransition_trailing_stop_val_oos_no_val_selection",
        "preregistered_rule": "adopt only if MDD improves on BOTH gated splits AND gated PnL >= baseline-1.0pp on BOTH splits",
        "preregistered_prior": "unfavorable (long-hold model, same profile as ETH 0/6 failure)",
        "live_duration_threshold": LIVE_DURATION_THRESHOLD,
        "results": results,
        "verdicts": verdicts,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VERDICTS:", json.dumps(verdicts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
