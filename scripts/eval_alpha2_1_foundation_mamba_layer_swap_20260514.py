#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import build_training_set, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha21  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_mamba_ssm_v41 import DeepAlphaMambaStyleSSM  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402
from scripts.train_eval_hf_v13_multitrack_foundation_parent_v40 import _parent_cfg  # noqa: E402
from scripts.eval_hf_v13_v40_6_full_v31_stack_retrain import _build_v40_6_frames, _load_bundle, _projection_targets  # noqa: E402


MODEL_ID = "alpha2_1_foundation_mamba_layer_swap_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha2_1_foundation_mamba_layer_swap_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha2_1_foundation_mamba_layer_swap_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha2_1_foundation_mamba_layer_swap_20260514_grid.csv"

TEACHER_MODEL = ROOT / "data/ensemble/supervised/alpha1_l2_teacher_deep_parent_20260514/teacher_deep_parent_l2_replay.pt"
V41_MAMBA = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_mamba_ssm_v41_20260512/v41_deep_alpha_mamba_ssm.pt"
V40_6_PARENT = ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512/target_aware_full_bundle.pkl"
V40_6_REPORT = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512_summary.json"
V44_RUNNER = ROOT / "data/ensemble/supervised/alpha2_1_ck_v44_retrain_20260514/v44_retrained_v21_2_runner.pkl"
V44_SCOUT = ROOT / "data/ensemble/supervised/alpha2_1_ck_v44_retrain_20260514/v44_retrained_deep_scout.pt"


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.30 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def _selected_l2_variant(overlay: v31.OverlayConfig | None = None) -> Any:
    base = next(v for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    if overlay is None:
        return base
    return type(base)(
        name="alpha2_1_l2_conservative_fee20_overlay_override",
        layer="conservative_l2_replay",
        overlay=overlay,
        execution_sniper=True,
        sniper_flow_th=base.sniper_flow_th,
        sniper_fee_mult=base.sniper_fee_mult,
        sniper_slip_mult=base.sniper_slip_mult,
    )


def _run_metrics(
    df: pd.DataFrame,
    parent_bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    q: np.ndarray,
    decisions: pd.DataFrame,
    variant: Any,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return {
        f"cost{mult}": l2._run_with_l2_proxy(
            df,
            parent_bundle,
            jackpot_model,
            add_cfg,
            q,
            decisions,
            variant,
            fee,
            slip,
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def _fast_seq_windows(df: pd.DataFrame, cols: list[str], seq_len: int = v31.SEQ_LEN) -> np.ndarray:
    arr = df.loc[:, cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    pad = np.zeros((seq_len - 1, arr.shape[1]), dtype=np.float32)
    padded = np.vstack([pad, arr])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=seq_len, axis=0)
    if windows.shape[1] == arr.shape[1]:
        windows = windows.transpose(0, 2, 1)
    return np.ascontiguousarray(windows)


def _predict_tcn_payload(payload_path: Path, df: pd.DataFrame) -> np.ndarray:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    model = v27.DeepAlphaTCN(len(payload["seq_cols"]))
    model.load_state_dict(payload["state_dict"])
    return _predict_seq_model(model, df, payload["seq_cols"], payload["norm"])


def _predict_mamba_payload(payload_path: Path, df: pd.DataFrame) -> np.ndarray:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    model = DeepAlphaMambaStyleSSM(len(payload["seq_cols"]))
    model.load_state_dict(payload["state_dict"])
    return _predict_seq_model(model, df, payload["seq_cols"], payload["norm"])


def _predict_seq_model(model: torch.nn.Module, df: pd.DataFrame, cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    x = _fast_seq_windows(df, cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            seq = x[start : start + 4096]
            xx = ((seq - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)
            outs.append(model(torch.from_numpy(xx).to(device)).detach().cpu().numpy())
    model.cpu()
    return np.vstack(outs).astype(np.float32)


def _build_encoded_frames(train_all: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    parent_report = json.loads(V40_6_REPORT.read_text(encoding="utf-8"))
    feature_cols = _feature_cols(train_all, eval_df)
    cfg = _parent_cfg()
    x_train, y, _ = build_training_set(train_df, cfg=cfg, stride_bars=48, batch_size=512, feature_cols=feature_cols)
    train_idx = np.arange(0, max(0, len(train_df) - cfg.max_train_horizon_bars - 1), 48, dtype=np.int64)
    if len(train_idx) != len(x_train):
        raise RuntimeError(f"train_idx mismatch for Chronos/Kairos projection: {len(train_idx)} vs {len(x_train)}")
    proj_targets = _projection_targets(y)
    train_feat = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    val_feat = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_feat = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    args = SimpleNamespace(
        train_csv=v31.DEFAULT_TRAIN,
        eval_csv=v31.DEFAULT_EVAL,
        train_stride=48,
        embed_batch=8,
    )
    train_full, val_full, eval_full, meta = _build_v40_6_frames(
        args=args,
        parent_report=parent_report,
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        train_feat=train_feat,
        val_feat=val_feat,
        eval_feat=eval_feat,
        train_idx_sample=train_idx,
        proj_targets=proj_targets,
    )
    return train_full, val_full, eval_full, meta


def _teacher_predictions(model: Any, feature_cols: list[str], norm: dict[str, Any], train_all: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    contract_cols = _feature_cols(train_all, eval_df)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=contract_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=contract_cols)
    return (
        teacher._predict_deep(model, val_features, feature_cols, norm),
        teacher._predict_deep(model, eval_features, feature_cols, norm),
    )


def main() -> int:
    print(f"[{MODEL_ID}] loading Alpha2.1 base stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base_cfg = dict(parent["config"])
    fee = float(base_cfg["fee"])
    slip = float(base_cfg["slip"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    l2_stats = l2._live_l2_stats()
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))

    teacher_payload = torch.load(TEACHER_MODEL, map_location="cpu", weights_only=False)
    teacher_model = alpha21._load_teacher_model(teacher_payload)
    teacher_cols = list(teacher_payload["feature_cols"])
    teacher_norm = teacher_payload["train_meta"]["norm"]
    buckets = tuple(float(x) for x in teacher_payload["buckets"])
    val_teacher_pred, eval_teacher_pred = _teacher_predictions(teacher_model, teacher_cols, teacher_norm, train_all, eval_df)

    runtime = alpha21.Alpha2Runtime("noflip_c0.56_parent_scale1.10", 0.56, 1.10, 2.75)
    l2_variant = _selected_l2_variant()

    print(f"[{MODEL_ID}] preparing original parent decisions and V27/Mamba scout utilities", flush=True)
    val_base_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_base_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_alpha21_dec = alpha21._decisions(val_base_dec, val_teacher_pred, buckets, runtime)
    eval_alpha21_dec = alpha21._decisions(eval_base_dec, eval_teacher_pred, buckets, runtime)
    val_v27_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_v27_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_mamba_q = _predict_mamba_payload(V41_MAMBA, val)
    eval_mamba_q = _predict_mamba_payload(V41_MAMBA, eval_df)

    print(f"[{MODEL_ID}] rebuilding Chronos/Kairos encoded frames", flush=True)
    _, val_ck, eval_ck, ck_meta = _build_encoded_frames(train_all, eval_df)
    ck_parent = _load_bundle(V40_6_PARENT)
    val_ck_dec_raw = predict_policy_frame(ck_parent, val_ck, close=_close(val_ck))
    eval_ck_dec_raw = predict_policy_frame(ck_parent, eval_ck, close=_close(eval_ck))
    val_ck_teacher_dec = alpha21._decisions(val_ck_dec_raw, val_teacher_pred, buckets, runtime)
    eval_ck_teacher_dec = alpha21._decisions(eval_ck_dec_raw, eval_teacher_pred, buckets, runtime)

    print(f"[{MODEL_ID}] loading Chronos/Kairos related-layer retrained V44 runner/scout", flush=True)
    v44_runner_payload = joblib.load(V44_RUNNER)
    v44_runner = v44_runner_payload["cost_runner"]
    v44_add_cfg = CostRunnerConfig(**dict(v44_runner_payload["selected_config"]))
    v44_scout_payload = torch.load(V44_SCOUT, map_location="cpu", weights_only=False)
    v44_overlay = v31.OverlayConfig(**dict(v44_scout_payload["selected_overlay"]))
    v44_l2_variant = _selected_l2_variant(v44_overlay)
    val_v44_q = _predict_tcn_payload(V44_SCOUT, val_ck)
    eval_v44_q = _predict_tcn_payload(V44_SCOUT, eval_ck)

    experiments: list[dict[str, Any]] = []

    def add_experiment(
        name: str,
        df_val: pd.DataFrame,
        df_eval: pd.DataFrame,
        bundle: dict[str, Any],
        runner: dict[str, Any],
        cfg: CostRunnerConfig,
        val_q: np.ndarray,
        eval_q: np.ndarray,
        val_dec: pd.DataFrame,
        eval_dec: pd.DataFrame,
        variant: Any,
        notes: list[str],
    ) -> None:
        print(f"[{MODEL_ID}] selecting/checking {name}", flush=True)
        val_metrics = _run_metrics(df_val, bundle, runner, cfg, val_q, val_dec, variant, fee=fee, slip=slip)
        metrics = _run_metrics(df_eval, bundle, runner, cfg, eval_q, eval_dec, variant, fee=fee, slip=slip)
        score = _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])
        experiments.append(
            {
                "name": name,
                "notes": notes,
                "validation_metrics": val_metrics,
                "metrics": metrics,
                "score": score,
                "variant": asdict(variant),
                "runner_config": asdict(cfg),
            }
        )
        print(
            f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} "
            f"cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )

    add_experiment(
        "alpha2_1_reference",
        val,
        eval_df,
        parent,
        jackpot_model,
        add_cfg,
        val_v27_q,
        eval_v27_q,
        val_alpha21_dec,
        eval_alpha21_dec,
        l2_variant,
        ["Original Alpha2.1 runtime: HGB parent + teacher gate + V27 scout + V21.2 runner + L2 replay."],
    )
    add_experiment(
        "alpha2_1_mamba_v41_scout_replace",
        val,
        eval_df,
        parent,
        jackpot_model,
        add_cfg,
        val_mamba_q,
        eval_mamba_q,
        val_alpha21_dec,
        eval_alpha21_dec,
        l2_variant,
        ["Replace only Frozen V27 TCN scout with previously trained V41 Mamba-style SSM scout."],
    )
    add_experiment(
        "alpha2_1_chronos_kairos_parent_replace_teacher_gate",
        val_ck,
        eval_ck,
        ck_parent,
        jackpot_model,
        add_cfg,
        val_v27_q,
        eval_v27_q,
        val_ck_teacher_dec,
        eval_ck_teacher_dec,
        l2_variant,
        ["Replace parent with V40.6 Chronos/Kairos PLS parent; keep original teacher gate and original V21.2/V27 layers to isolate parent effect."],
    )
    add_experiment(
        "chronos_kairos_v44_related_layers_retrained_l2",
        val_ck,
        eval_ck,
        ck_parent,
        v44_runner,
        v44_add_cfg,
        val_v44_q,
        eval_v44_q,
        val_ck_dec_raw,
        eval_ck_dec_raw,
        v44_l2_variant,
        ["Use V40.6 Chronos/Kairos parent with related V21.2 runner and V27-style scout already retrained in V44, then add L2 replay accounting."],
    )
    add_experiment(
        "chronos_kairos_v44_retrained_plus_alpha2_teacher_gate_l2",
        val_ck,
        eval_ck,
        ck_parent,
        v44_runner,
        v44_add_cfg,
        val_v44_q,
        eval_v44_q,
        val_ck_teacher_dec,
        eval_ck_teacher_dec,
        v44_l2_variant,
        ["V44 related-layer retrained stack plus original Alpha2 teacher gate. This tests whether the old teacher still helps after Chronos/Kairos parent replacement."],
    )

    grid_rows = []
    for exp in experiments:
        m = exp["metrics"]
        grid_rows.append(
            {
                "name": exp["name"],
                "score": exp["score"],
                "cost1_pnl": m["cost1"]["pnl"],
                "cost1_mdd": m["cost1"]["mdd"],
                "cost1_trades": m["cost1"]["trades"],
                "cost1_deep_entries": m["cost1"].get("deep_entries", 0),
                "cost2_pnl": m["cost2"]["pnl"],
                "cost2_mdd": m["cost2"]["mdd"],
                "cost3_pnl": m["cost3"]["pnl"],
                "cost3_mdd": m["cost3"]["mdd"],
            }
        )
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)
    best = max(experiments, key=lambda x: float(x["score"]))
    warnings = list(parent_audit.get("warnings", []))
    blocking = list(parent_audit.get("blocking", []))
    if not l2_stats.get("usable_for_replay", False):
        warnings.append("historical_l2_snapshots_insufficient_conservative_ohlc_replay_only")
    warnings.append("real_live_l2_fill_model_requires_forward_shadow_collection")
    if best["name"] != "alpha2_1_reference":
        ref = next(x for x in experiments if x["name"] == "alpha2_1_reference")
        if best["score"] <= ref["score"]:
            warnings.append("no_foundation_or_mamba_variant_beat_alpha2_1_reference")
    else:
        warnings.append("alpha2_1_reference_remained_best")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "shadow_collect_l2" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31 for runtime/reference checks; V44 related layers were previously selected on 2025Q4",
        "oos_window": "2026 fixed OOS only after layer selection",
        "l2_stats": l2_stats,
        "chronos_kairos_meta": ck_meta,
        "tested_layer_points": [
            "parent replacement with Chronos/Kairos PLS V40.6",
            "deep scout replacement with V41 Mamba-style SSM",
            "related-layer retrained V44 Chronos/Kairos parent + runner + scout",
            "teacher gate compatibility after Chronos/Kairos parent replacement",
        ],
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha2.1 layer insertion/replacement test for Chronos/Kairos and Mamba. The test covers parent-level Chronos/Kairos replacement, V27 scout replacement by Mamba-style SSM, and a related-layer retrained V44 Chronos/Kairos stack with L2 replay accounting.",
        "experiments": experiments,
        "best": best["name"],
        "audit": audit,
        "artifacts": {
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
            "teacher_model": str(TEACHER_MODEL),
            "mamba_model": str(V41_MAMBA),
            "chronos_kairos_parent": str(V40_6_PARENT),
            "v44_runner": str(V44_RUNNER),
            "v44_scout": str(V44_SCOUT),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "best": best["name"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
