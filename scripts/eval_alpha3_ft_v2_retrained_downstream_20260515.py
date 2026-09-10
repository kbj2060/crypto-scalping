#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    FullyLearnedGovernorConfig,
    prepare_features,
    predict_policy_frame,
)
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_alpha3_limit_close_fallback_20260514 as alpha3_close  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_hf_v13_deep_tabular_parent_mdd_20260514 import (  # noqa: E402
    _decisions_from_outputs,
    _normalise_apply,
    _predict_outputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _fill_price, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import (  # noqa: E402
    CostRunnerConfig,
    _addon_utility,
    _grid as _runner_grid,
    _predict_cost_runner,
    _fit_cost_runner,
)


MODEL_ID = "alpha3_ft_v2_retrained_downstream_20260515"
FT_MODEL = ROOT / "data/ensemble/supervised/alpha3_ft_transformer_mtl_parent_v2_20260515/ft_transformer_mtl_parent_v2.pt"
FT_REPORT = ROOT / "data/ensemble/reports/alpha3_ft_transformer_mtl_parent_v2_20260515_summary.json"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_ft_v2_retrained_downstream_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_ft_v2_retrained_downstream_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_ft_v2_retrained_downstream_20260515_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_ft_v2_retrained_downstream_20260515_grid.csv"


def _load_ft_parent(device: torch.device) -> tuple[ft_v2.FTTransformerParentV2, list[str], dict[str, Any], FullyLearnedGovernorConfig]:
    payload = torch.load(FT_MODEL, map_location="cpu", weights_only=False)
    cfg = FullyLearnedGovernorConfig(**dict(payload["config"]))
    feature_cols = list(payload["feature_cols"])
    model = ft_v2.FTTransformerParentV2(len(feature_cols), cfg, d_model=80, n_layers=3)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), feature_cols, dict(payload["normalizer"]), cfg


def _ft_runtime() -> Any:
    report = json.loads(FT_REPORT.read_text(encoding="utf-8"))
    rt = dict(report["experiments"][1]["runtime"])
    return ft_v2.RuntimeConfig(**rt)


def _ft_decisions(df: pd.DataFrame, model: torch.nn.Module, cols: list[str], norm: dict[str, Any], cfg: FullyLearnedGovernorConfig, rt: Any, device: torch.device, batch_size: int) -> pd.DataFrame:
    features = prepare_features(df, side_hint=0, close=_close(df), feature_cols=cols)
    x = _normalise_apply(features, norm)
    outputs = _predict_outputs(model, x, None, device, batch_size, mc_passes=5)
    return _decisions_from_outputs(outputs, cfg, rt, df.index)


def _fit_cost_runner_with_decisions(frame: pd.DataFrame, bundle: dict[str, Any], decisions: pd.DataFrame, *, fee: float, slip: float) -> dict[str, Any]:
    close = _close(frame)
    rows: list[pd.DataFrame] = []
    targets: list[float] = []
    target_cost2: list[float] = []
    target_cost3: list[float] = []
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cash = peak = 1.0
    mfe = mae = 0.0
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(close[i])
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            hold = i - entry_idx
            state = {
                "parent_notional": parent_notional,
                "notional": notional,
                "bars_since_entry": hold,
                "unrealized": unreal,
                "mfe": mfe,
                "mae": mae,
                "drawdown_abs": max(0.0, 1.0 - eq / max(peak, 1e-12)),
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "max_hold": max_hold,
            }
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "tp"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "sl"
            elif max_hold > 0 and hold >= max_hold:
                reason = "hold"
            if unreal >= 0.004 and hold >= 3:
                u1 = _addon_utility(frame, close, pos=pos, entry_idx=entry_idx, snapshot_idx=i, entry_price=entry_price, current_notional=notional, parent_notional=parent_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, add_frac=0.25, fee=fee, slip=slip, cost_mult=1.0)
                u2 = _addon_utility(frame, close, pos=pos, entry_idx=entry_idx, snapshot_idx=i, entry_price=entry_price, current_notional=notional, parent_notional=parent_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, add_frac=0.25, fee=fee, slip=slip, cost_mult=2.0)
                u3 = _addon_utility(frame, close, pos=pos, entry_idx=entry_idx, snapshot_idx=i, entry_price=entry_price, current_notional=notional, parent_notional=parent_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, add_frac=0.25, fee=fee, slip=slip, cost_mult=3.0)
                rows.append(_feature_frame(frame, bundle, decisions, i, state))
                targets.append(float(min(u1, 0.75 * u2, 0.55 * u3)))
                target_cost2.append(float(u2))
                target_cost3.append(float(u3))
            if reason:
                exit_i = min(i + 1, len(frame) - 1)
                exit_px = _fill_price(frame, exit_i, pos, slip, entry=False)
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee * notional
                pos = 0
                continue
        if pos == 0:
            dec = decisions.iloc[i]
            if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
                continue
            fill_i = min(i + 1, len(frame) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(frame, fill_i, pos, slip, entry=True)
            entry_idx = i
            parent_notional = float(dec.notional_exposure)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            cash -= cash * fee * notional
            mfe = mae = 0.0
    if not rows:
        raise RuntimeError("no FT-downstream cost runner snapshots")
    x = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(targets, dtype=np.float64)
    y2 = np.asarray(target_cost2, dtype=np.float64)
    y3 = np.asarray(target_cost3, dtype=np.float64)
    return _fit_cost_runner_from_matrix(x, y, y2, y3)


def _fit_cost_runner_from_matrix(x: pd.DataFrame, y: np.ndarray, y2: np.ndarray, y3: np.ndarray) -> dict[str, Any]:
    # Reuse the exact model family/hyperparameters from V21.2 via a tiny shim.
    # The upstream helper does not expose matrix fitting, so this mirrors its estimator stack.
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    reg = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(max_iter=220, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.12, min_samples_leaf=12, random_state=2020))
    q10 = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="quantile", quantile=0.10, max_iter=220, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.12, min_samples_leaf=12, random_state=2022))
    q90 = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=220, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.12, min_samples_leaf=12, random_state=2023))
    clf = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=200, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.12, min_samples_leaf=12, random_state=2021))
    jackpot_clf = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.16, min_samples_leaf=12, random_state=2024))
    bad_clf = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.16, min_samples_leaf=12, random_state=2025))
    cost3_clf = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.16, min_samples_leaf=12, random_state=2026))
    reg.fit(x, y)
    q10.fit(x, y2)
    q90.fit(x, y2)
    clf.fit(x, (y > 0.0).astype(int))
    jackpot_clf.fit(x, (y2 >= 0.025).astype(int))
    bad_clf.fit(x, (y2 <= -0.004).astype(int))
    cost3_clf.fit(x, (y3 > 0.0).astype(int))
    return {
        "regressor": reg,
        "q10_regressor": q10,
        "q90_regressor": q90,
        "classifier": clf,
        "jackpot_classifier": jackpot_clf,
        "bad_classifier": bad_clf,
        "cost3_classifier": cost3_clf,
        "feature_cols": list(x.columns),
        "snapshot_count": int(len(x)),
        "positive_rate": float((y > 0.0).mean()),
        "jackpot_rate": float((y2 >= 0.025).mean()),
        "bad_rate": float((y2 <= -0.004).mean()),
        "cost3_positive_rate": float((y3 > 0.0).mean()),
        "target_mean": float(y.mean()),
        "target_p10": float(np.quantile(y, 0.10)),
        "target_cost2_p90": float(np.quantile(y2, 0.90)),
        "target_p75": float(np.quantile(y, 0.75)),
        "target_p95": float(np.quantile(y, 0.95)),
    }


def _metrics(df: pd.DataFrame, parent: dict[str, Any], runner: dict[str, Any], cfg: CostRunnerConfig, q: np.ndarray, decisions: pd.DataFrame, overlay: Any, limit_cfg: Any, *, fee: float, slip: float) -> dict[str, Any]:
    return alpha3_close._metrics_signal_limit_close(df, parent, runner, cfg, q, decisions, overlay, limit_cfg, fee=fee, slip=slip)


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def main() -> int:
    p = argparse.ArgumentParser(description="Keep FT v2 parent fixed, retrain teacher gate and V21.2 runner, then backtest Alpha3 corrected execution.")
    p.add_argument("--teacher-epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(20260515)
    np.random.seed(20260515)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{MODEL_ID}] device={device}", flush=True)

    parent = joblib.load(v31.DEFAULT_PARENT)
    cfg = FullyLearnedGovernorConfig(**dict(parent["config"]))
    fee = float(dict(parent["config"])["fee"])
    slip = float(dict(parent["config"])["slip"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    audit_base = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))

    ft_model, ft_cols, ft_norm, ft_cfg = _load_ft_parent(device)
    ft_rt = _ft_runtime()
    print(f"[{MODEL_ID}] fixed FT parent runtime={ft_rt.name}", flush=True)
    train_ft_dec = _ft_decisions(train_df, ft_model, ft_cols, ft_norm, ft_cfg, ft_rt, device, int(args.batch_size))
    val_ft_dec = _ft_decisions(val_df, ft_model, ft_cols, ft_norm, ft_cfg, ft_rt, device, int(args.batch_size))
    eval_ft_dec = _ft_decisions(eval_df, ft_model, ft_cols, ft_norm, ft_cfg, ft_rt, device, int(args.batch_size))

    contract_cols = _feature_cols(train_all, eval_df)
    buckets = tuple(float(x) for x in cfg.notional_buckets)
    print(f"[{MODEL_ID}] retraining teacher gate on FT decisions", flush=True)
    train_features = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=contract_cols)
    train_seq = teacher._seq_tensor(train_features, np.arange(len(train_df), dtype=np.int64), contract_cols)
    y_action = train_ft_dec["action"].astype(int).to_numpy(dtype=np.int64)
    y_quality = pd.to_numeric(train_ft_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_notional = teacher._bucket_labels(train_ft_dec, buckets)
    teacher_model, teacher_meta = teacher._train_teacher_model(train_seq, y_action, y_quality, y_notional, n_buckets=len(buckets), epochs=int(args.teacher_epochs))
    teacher_model_path = OUT_DIR / "ft_v2_teacher_gate.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": teacher_model.state_dict(), "feature_cols": contract_cols, "train_meta": teacher_meta, "buckets": buckets}, teacher_model_path)

    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=contract_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=contract_cols)
    val_pred = teacher._predict_deep(teacher_model, val_features, contract_cols, teacher_meta["norm"])
    eval_pred = teacher._predict_deep(teacher_model, eval_features, contract_cols, teacher_meta["norm"])
    train_pred = teacher._predict_deep(teacher_model, train_features, contract_cols, teacher_meta["norm"])

    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    overlay = next(v.overlay for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    limit_cfg = ft_v2.ft_v1._limit_cfg()
    existing_runner_payload = joblib.load(v31.DEFAULT_JACKPOT)
    existing_runner = existing_runner_payload["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")

    print(f"[{MODEL_ID}] selecting teacher runtime without add-on on 2025Q4", flush=True)
    runtime_rows: list[dict[str, Any]] = []
    selected_rt: alpha2.Alpha2Runtime | None = None
    best_score = -1e18
    for rt in alpha2._runtimes():
        dec = alpha2._decisions(val_ft_dec, val_pred, buckets, rt)
        metrics = _metrics(val_df, parent, existing_runner, noop_cfg, val_q, dec, overlay, limit_cfg, fee=fee, slip=slip)
        score = _score(metrics)
        runtime_rows.append({**asdict(rt), "stage": "teacher_runtime_no_addon", "score": score, "val_cost1_pnl": metrics["cost1"]["pnl"], "val_cost1_mdd": metrics["cost1"]["mdd"], "val_cost2_pnl": metrics["cost2"]["pnl"], "val_cost3_pnl": metrics["cost3"]["pnl"]})
        if score > best_score:
            best_score = score
            selected_rt = rt
            print(f"[{MODEL_ID}] new teacher runtime {rt.name} score={score:.2f} c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f}", flush=True)
    assert selected_rt is not None

    train_final_dec = alpha2._decisions(train_ft_dec, train_pred, buckets, selected_rt)
    val_final_dec = alpha2._decisions(val_ft_dec, val_pred, buckets, selected_rt)
    eval_final_dec = alpha2._decisions(eval_ft_dec, eval_pred, buckets, selected_rt)

    print(f"[{MODEL_ID}] retraining V21.2 runner on FT+teacher decisions", flush=True)
    runner = _fit_cost_runner_with_decisions(train_df, parent, train_final_dec, fee=fee, slip=slip)
    runner_model_path = OUT_DIR / "ft_v2_retrained_v21_2_runner.pkl"

    print(f"[{MODEL_ID}] selecting runner config on 2025Q4", flush=True)
    rows = list(runtime_rows)
    selected_cfg: CostRunnerConfig | None = None
    best_runner_score = -1e18
    for add_cfg in _runner_grid():
        metrics = _metrics(val_df, parent, runner, add_cfg, val_q, val_final_dec, overlay, limit_cfg, fee=fee, slip=slip)
        score = _score(metrics)
        rows.append({"stage": "runner_config", **asdict(selected_rt), "runner_config": add_cfg.name, "score": score, "val_cost1_pnl": metrics["cost1"]["pnl"], "val_cost1_mdd": metrics["cost1"]["mdd"], "val_cost1_trades": metrics["cost1"]["trades"], "val_cost2_pnl": metrics["cost2"]["pnl"], "val_cost3_pnl": metrics["cost3"]["pnl"]})
        if score > best_runner_score:
            best_runner_score = score
            selected_cfg = add_cfg
            print(f"[{MODEL_ID}] new runner config {add_cfg.name} score={score:.2f} c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f}", flush=True)
    assert selected_cfg is not None
    joblib.dump({"model_id": MODEL_ID, "base_parent": "ft_transformer_mtl_parent_v2_fixed", "cost_runner": runner, "selected_config": asdict(selected_cfg), "teacher_runtime": asdict(selected_rt)}, runner_model_path)
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    hgb_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    existing_teacher_model, existing_teacher_cols, existing_teacher_norm, existing_teacher_buckets = ft_v2.ft_v1._load_teacher()
    existing_alpha3_runtime = ft_v2.ft_v1._selected_alpha3_runtime()
    existing_teacher_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=existing_teacher_cols)
    existing_teacher_pred = teacher._predict_deep(existing_teacher_model, existing_teacher_features, existing_teacher_cols, existing_teacher_norm)
    alpha3_current_dec = alpha2._decisions(hgb_dec, existing_teacher_pred, existing_teacher_buckets, existing_alpha3_runtime)
    ft_old_downstream_dec = alpha2._decisions(eval_ft_dec, existing_teacher_pred, existing_teacher_buckets, existing_alpha3_runtime)
    experiments: list[dict[str, Any]] = []
    for name, dec, runner_model, add_cfg in (
        ("alpha3_current_hgb_parent_teacher_downstream", alpha3_current_dec, existing_runner, CostRunnerConfig(**dict(existing_runner_payload["selected_config"]))),
        ("ft_v2_parent_old_downstream", ft_old_downstream_dec, existing_runner, CostRunnerConfig(**dict(existing_runner_payload["selected_config"]))),
        (f"ft_v2_parent_retrained_downstream::{selected_rt.name}::{selected_cfg.name}", eval_final_dec, runner, selected_cfg),
    ):
        metrics = _metrics(eval_df, parent, runner_model, add_cfg, eval_q, dec, overlay, limit_cfg, fee=fee, slip=slip)
        experiments.append({"name": name, "metrics": metrics, "score": _score(metrics), "teacher_runtime": asdict(selected_rt) if name.startswith("ft_v2_parent_retrained") else None, "runner_config": asdict(add_cfg)})
        print(f"[{MODEL_ID}] {name} c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}", flush=True)

    baseline = experiments[0]
    candidate = experiments[-1]
    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if candidate["score"] <= baseline["score"]:
        warnings.append("ft_v2_retrained_downstream_did_not_beat_alpha3_hgb_parent")
    if candidate["metrics"]["cost1"]["pnl"] <= 0:
        warnings.append("ft_v2_retrained_downstream_cost1_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and candidate["score"] > baseline["score"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after teacher/runtime/runner selection",
        "ft_parent_fixed_model": str(FT_MODEL),
        "ft_parent_runtime": asdict(ft_rt),
        "selected_teacher_runtime": asdict(selected_rt),
        "selected_runner_config": asdict(selected_cfg),
        "runner_meta": {k: v for k, v in runner.items() if k not in {"regressor", "q10_regressor", "q90_regressor", "classifier", "jackpot_classifier", "bad_classifier", "cost3_classifier", "feature_cols"}},
        "base_feature_audit": audit_base,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "FT-Transformer v2 parent is frozen. The teacher gate is retrained to imitate FT v2 decisions, teacher runtime is selected on 2025Q4, V21.2 cost runner is retrained on FT+teacher decisions, runner config is selected on 2025Q4, and final OOS is evaluated under Alpha3 corrected next_open_limit_touch0_fee20 execution.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {
            "ft_parent": str(FT_MODEL),
            "teacher_gate": str(teacher_model_path),
            "runner": str(runner_model_path),
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "candidate": candidate}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
