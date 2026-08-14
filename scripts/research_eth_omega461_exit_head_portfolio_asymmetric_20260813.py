#!/usr/bin/env python3
"""RESEARCH ONLY -- portfolio-level VAL backtest of the coordinator's asymmetric-adoption call:
h48qual uses the new live-ATR-relabeled exit head
(scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py, see
docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md), zig075 keeps its
original frozen exit head unchanged. Compared against baseline (both components on their original
frozen exit head).

Every per-component backtest so far in this research thread (h48cons/liveatr docs, the original
research_eth_omega461_exit_sweep_20260721.py harness) simulated h48qual and zig075 as two
INDEPENDENT full-capital ledgers. The real live adapter (trading_bot_modules/omega4_6_1_live.py,
read for reference only -- never imported/touched) shares ONE account-level position slot with
h48qual>zig075 priority: if h48qual has a nonzero-side signal on a bar, zig075 is never even
consulted that bar, and only the position-opening component's own exit-head governs its exit. A
component looking better in isolation does not automatically mean the combined system improves --
this script answers that question using the already-existing, previously-validated single-account
greedy router (scripts/replay_omega4_6_1_greedy_router_20260706.py::greedy_replay/prepare_component,
whose PRIORITY/SCALE_MAP/LEVERAGE_CAP/NOTIONAL_CAP/DURATION_THRESHOLD constants are byte-identical
to the live module), imported and reused unchanged -- not reimplemented.

VAL only: 2025-10-01..2025-12-31 (research_eth_omega461_exit_sweep_20260721.VAL_START/VAL_END).
OOS (2026-01-01..2026-03-31) is never loaded or scored here.

fresh_forward_bar_by_bar=true (greedy_replay is a single causal forward pass, i in increasing
order, only bar i and already-closed history used at bar i). trade_ledgers_used_as_input=false
(ledgers are only ever written as OUTPUT). saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. direction/quality remain frozen (unchanged from the frozen live
bundles); only h48qual's exit_head weights differ between the two compared configurations. No
duration-gate post-filter is applied -- current live runs with the duration gate off (see
docs/experiments/eth_omega461_exit_learning_20260724.md's composition audit and this script's own
predecessors), so the un-gated greedy_replay ledger already matches current live behavior; this
keeps the comparison isolated to the one variable in question (which exit head h48qual uses).

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT overwrite any live checkpoint.
"""
from __future__ import annotations

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
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk_sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_portfolio_asymmetric_20260813"
NEW_H48QUAL_BUNDLE = (
    ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500"
    "/h48qual/true_3head_tabm_bundle.pt"
)
DEVICE = torch.device("cpu")


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _component_cfg(name: str, *, bundle_override: Path | None = None) -> dict[str, Any]:
    cfg = dict(sweep.COMPONENTS[name])
    cfg["exit_threshold"] = 0.95  # trading_bot_modules/omega4_6_1_live.py EXIT_THRESHOLD, fixed
    if bundle_override is not None:
        cfg["bundle"] = bundle_override
    return cfg


def _align_frame_and_predictions(val_frame: pd.DataFrame, q_tags: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Path]]:
    """greedy.prepare_component requires pred['timestamp'].equals(frame['timestamp']) exactly (no
    intersection step, unlike research_eth_omega461_exit_sweep_20260721.prep_component which does
    intersect). sweep.load_frame's raw VAL frame and each component's own
    validation_predictions_qXXX.csv are not guaranteed to cover identical bar sets. Intersect all
    three timestamp sets once, filter/reindex the frame and each component's own predictions to
    that common set (frame-order preserved), and fix pandas>=3.0 StringDtype columns (same
    dtype issue research_eth_omega461_exit_sweep_20260721.prep_component already works around for
    its own harness; greedy.prepare_component reads the CSV itself and does not). Both compared
    variants (baseline/asymmetric) share the SAME h48qual/zig075 prediction CSVs (only the bundle
    path differs), so this alignment only needs to happen once, not once per variant."""
    raw_preds: dict[str, pd.DataFrame] = {}
    keep_ts = set(val_frame["timestamp"])
    for cname, q_tag in q_tags.items():
        pred_csv = sweep.EXT_PRED_DIR / cname / f"validation_predictions_{q_tag}.csv"
        df = pd.read_csv(pred_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        raw_preds[cname] = df
        keep_ts &= set(df["timestamp"])
    aligned_frame = val_frame[val_frame["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
    aligned_paths: dict[str, Path] = {}
    for cname, df in raw_preds.items():
        df = df[df["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
        if len(df) != len(aligned_frame) or not df["timestamp"].equals(aligned_frame["timestamp"]):
            raise RuntimeError(f"{cname}: alignment failed after timestamp intersection")
        for c in df.columns:
            if str(df[c].dtype).lower().startswith("str"):
                df[c] = df[c].astype(object)
        out_path = OUT_DIR / f"_aligned_{cname}_validation_predictions.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        aligned_paths[cname] = out_path
    return aligned_frame, aligned_paths


def _prepare_component_val(frame: pd.DataFrame, pred_csv: Path, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Byte-for-byte copy of replay_omega4_6_1_greedy_router_20260706.prepare_component, except
    `parent._to_decisions(pred, oof=False)` -> `oof=True`. That script's prepare_component
    hardcodes oof=False because its own main() only ever scores OOS predictions (oof=False is the
    fresh, non-out-of-fold scoring convention); VAL predictions
    (validation_predictions_qXXX.csv) were produced out-of-fold and need oof=True
    (research_eth_omega461_exit_sweep_20260721.prep_component does the same for its own,
    per-component-isolated harness). Not modifying the shared greedy-router script itself --
    duplicated here only because its oof flag isn't exposed as a parameter."""
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols, models = bundle["base_cols"], bundle["models"]
    pred = pd.read_csv(pred_csv)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError("timestamp mismatch")

    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(pred, oof=True)
    dec, _ = atr_eval._apply_atr_safety_sltp(dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"],
                                              sl_mult=cfg["sl_mult"], min_tp=cfg["min_tp"], min_sl=cfg["min_sl"],
                                              max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
    atr = atr_eval._atr_pct(frame, cfg["atr_window"])
    loaded = parent._load_payloads(models, device=device)

    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)
    features = risk_sidecar._risk_feature_frame(frame, pred, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = risk_sidecar._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = risk_sidecar._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all))
    mapping = pkl["selected_mapping"]
    margin = risk_sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in risk_sidecar.MARGIN_CFG_KEYS})
    lev = risk_sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in risk_sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(dec))

    base_np, exit_runtime, pos_idx = risk_sidecar._prepare_exit_runtime(x, loaded)
    route = hard._route_id(frame)
    return {
        "dec": dec, "atr": atr, "margin": margin, "leverage": lev, "base_np": base_np,
        "exit_runtime": exit_runtime, "pos_idx": pos_idx, "route": route, "exit_threshold": cfg["exit_threshold"],
    }


def _ledger_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if len(ledger) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0, "exit_reasons": {}, "source_component_counts": {}}
    returns = ledger["trade_return"].to_numpy(dtype=np.float64)
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(dd.min() * 100.0),
        "trades": int(len(ledger)),
        "wr": float((returns > 0).mean()),
        "avg_hold_bars": float((ledger["exit_i"] - ledger["entry_i"]).clip(lower=0).mean()),
        "max_trade_pnl": float(returns.max() * 100.0),
        "reason_counts": ledger["reason"].value_counts().to_dict(),
        "source_component_counts": ledger["source_component"].value_counts().to_dict(),
    }


def run_variant(
    name: str,
    comp_cfgs: dict[str, dict[str, Any]],
    val_frame: pd.DataFrame,
    aligned_pred_paths: dict[str, Path],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    components = {}
    for cname, cfg in comp_cfgs.items():
        components[cname] = _prepare_component_val(val_frame, aligned_pred_paths[cname], cfg, DEVICE)
        print(f"  {cname}: bundle={Path(cfg['bundle']).parent.name} nonzero_side={(components[cname]['dec']['side'] != 0).mean():.3f}", flush=True)
    _diag, ledger = greedy.greedy_replay(val_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=DEVICE)
    ledger.to_csv(OUT_DIR / f"portfolio_ledger_{name}.csv", index=False)
    metrics = _ledger_metrics(ledger)
    print(f"  {name}: {json.dumps({k: v for k, v in metrics.items() if k not in ('reason_counts', 'source_component_counts')})}", flush=True)
    return metrics


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("stage=load_val_frame", flush=True)
    val_frame_raw = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"  VAL frame rows={len(val_frame_raw)} range=[{val_frame_raw['timestamp'].min()}, {val_frame_raw['timestamp'].max()}]", flush=True)
    fee, slip = omega._load_fee_slip()

    print("stage=align_frame_and_predictions", flush=True)
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in ("h48qual", "zig075")}
    val_frame, aligned_pred_paths = _align_frame_and_predictions(val_frame_raw, q_tags)
    print(f"  aligned rows={len(val_frame)} (from raw {len(val_frame_raw)})", flush=True)

    variants = {
        "baseline_both_original": {
            "h48qual": _component_cfg("h48qual"),
            "zig075": _component_cfg("zig075"),
        },
        "asymmetric_h48qual_liveatr_zig075_original": {
            "h48qual": _component_cfg("h48qual", bundle_override=NEW_H48QUAL_BUNDLE),
            "zig075": _component_cfg("zig075"),
        },
    }

    results: dict[str, Any] = {}
    for name, comp_cfgs in variants.items():
        print(f"stage=run_variant name={name}", flush=True)
        results[name] = run_variant(name, comp_cfgs, val_frame, aligned_pred_paths, fee=fee, slip=slip)

    report = {
        "design": (
            "Portfolio-level (single shared position slot, h48qual>zig075 priority) VAL replay via "
            "the existing replay_omega4_6_1_greedy_router_20260706.greedy_replay/prepare_component, "
            "reused unchanged. baseline_both_original = both components on their original frozen "
            "live exit_head bundles. asymmetric_... = h48qual on the new live-ATR-relabeled exit "
            "head bundle, zig075 unchanged -- the coordinator's asymmetric-adoption decision."
        ),
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "oos_opened": False,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "duration_gate_applied": False,
        "new_h48qual_bundle": str(NEW_H48QUAL_BUNDLE),
        "results": results,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
