#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_supervised_risk_selector_20260604 as sel  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_supervised_risk_regressor_20260604"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SCORE_THRESHOLDS = [-0.002, 0.0, 0.001, 0.002, 0.003, 0.005]


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _template_features(x: pd.DataFrame, template: dict[str, float]) -> pd.DataFrame:
    out = x.copy().reset_index(drop=True)
    out["risk_tp"] = float(template["tp"])
    out["risk_sl"] = float(template["sl"])
    out["risk_notional"] = float(template["notional"])
    out["risk_leverage"] = float(template["leverage"])
    out["risk_rr"] = float(template["tp"]) / max(float(template["sl"]), 1e-8)
    return out.astype(np.float32)


def _build_regression_dataset(
    frame: pd.DataFrame,
    src: pd.DataFrame,
    risk_x: pd.DataFrame,
    *,
    oof: bool,
    candidate_delta: float,
    fee: float,
    slip: float,
    cost_mult: float,
    max_candidates: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    action = sel._threshold_action(src, oof=oof, thresholds=sel._candidate_thresholds(float(candidate_delta)))
    candidate_idx = np.flatnonzero(action != omega.ACTION_CASH)
    if int(max_candidates) > 0 and len(candidate_idx) > int(max_candidates):
        keep = np.linspace(0, len(candidate_idx) - 1, int(max_candidates)).round().astype(np.int64)
        candidate_idx = candidate_idx[keep]
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    x_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []
    diag_rows: list[dict[str, Any]] = []
    for cls, template in enumerate(sel.RISK_TEMPLATES):
        if cls == 0:
            continue
        scores = np.zeros(len(candidate_idx), dtype=np.float32)
        for out_i, idx in enumerate(candidate_idx):
            act = int(action[int(idx)])
            side = 1 if act == omega.ACTION_LONG else -1
            score, meta = omega._simulate_trade(
                frame,
                arrays,
                int(idx),
                sel._single_dec_row(act, side, template),
                fee=fee,
                slip=slip,
                cost_mult=cost_mult,
            )
            scores[out_i] = float(score)
            if len(diag_rows) < 3000:
                diag_rows.append(
                    {
                        "row": int(idx),
                        "timestamp": str(frame["timestamp"].iloc[int(idx)]),
                        "action": int(act),
                        "template_id": int(cls),
                        "template": str(template["name"]),
                        "score": float(score),
                        "net": float(meta.get("net", 0.0)),
                        "exit_reason": str(meta.get("exit_reason", "")),
                    }
                )
        x_parts.append(_template_features(risk_x.iloc[candidate_idx].reset_index(drop=True), template))
        y_parts.append(scores)
    x_all = pd.concat(x_parts, ignore_index=True)
    y_all = np.concatenate(y_parts).astype(np.float32)
    return x_all, y_all, pd.DataFrame(diag_rows)


def _fit_regressor(x: pd.DataFrame, y: np.ndarray, *, seed: int, iterations: int) -> CatBoostRegressor:
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=int(iterations),
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=10.0,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(x, y)
    return model


def _predict_template_scores(model: CatBoostRegressor, x: pd.DataFrame) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for template in sel.RISK_TEMPLATES[1:]:
        chunks.append(np.asarray(model.predict(_template_features(x, template)), dtype=np.float64))
    return np.stack(chunks, axis=1)


def _risk_decisions_from_scores(
    src: pd.DataFrame,
    scores: np.ndarray,
    *,
    oof: bool,
    action: np.ndarray,
    score_threshold: float,
) -> pd.DataFrame:
    best = scores.argmax(axis=1) + 1
    best_score = scores.max(axis=1)
    risk_class = np.where(best_score >= float(score_threshold), best, 0).astype(np.int64)
    return sel._risk_decision(src, oof=oof, action=action, risk_class=risk_class)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--candidate-delta-train", type=float, default=-0.25)
    ap.add_argument("--candidate-deltas", default="-0.30,-0.25,-0.20,-0.15,-0.10")
    ap.add_argument("--max-candidates", type=int, default=6000)
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260606)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    device = sel._device(args.device)
    out_dir = OUT_DIR if not args.out_suffix.strip() else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix.strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = tabm._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    bundle = torch.load(sel.ZIGZAG_DIR / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    train_x_base, train_src = sel._predict_frame(frames["train_raw"], bundle, oof=True, device=device)
    val_src = sel._read_predictions(sel.ZIGZAG_DIR / "validation_predictions_2025_true3head.csv", frames["val_raw"])
    oos_src = sel._read_predictions(sel.ZIGZAG_DIR / "oos_predictions_2026_true3head.csv", frames["oos_raw"])
    val_x_base = tabm._base_input(frames["val_raw"], list(bundle["base_cols"]))
    oos_x_base = tabm._base_input(frames["oos_raw"], list(bundle["base_cols"]))
    train_x_risk = sel._risk_features(train_x_base, train_src, oof=True)
    val_x_risk = sel._risk_features(val_x_base, val_src, oof=True)
    oos_x_risk = sel._risk_features(oos_x_base, oos_src, oof=False)

    x_train, y_train, diag = _build_regression_dataset(
        frames["train_raw"],
        train_src,
        train_x_risk,
        oof=True,
        candidate_delta=float(args.candidate_delta_train),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_candidates=int(args.max_candidates),
    )
    model = _fit_regressor(x_train, y_train, seed=int(args.seed), iterations=int(args.iterations))
    model.save_model(str(out_dir / "supervised_risk_regressor.cbm"))
    diag.to_csv(out_dir / "risk_regression_label_diagnostics_head.csv", index=False)

    rows: list[dict[str, Any]] = []
    for delta_s in str(args.candidate_deltas).split(","):
        delta = float(delta_s)
        val_action = sel._threshold_action(val_src, oof=True, thresholds=sel._candidate_thresholds(delta))
        oos_action = sel._threshold_action(oos_src, oof=False, thresholds=sel._candidate_thresholds(delta))
        val_candidate = val_action != omega.ACTION_CASH
        oos_candidate = oos_action != omega.ACTION_CASH
        val_scores = np.zeros((len(val_action), len(sel.RISK_TEMPLATES) - 1), dtype=np.float64)
        oos_scores = np.zeros((len(oos_action), len(sel.RISK_TEMPLATES) - 1), dtype=np.float64)
        if bool(val_candidate.any()):
            val_scores[val_candidate] = _predict_template_scores(model, val_x_risk.loc[val_candidate].reset_index(drop=True))
        if bool(oos_candidate.any()):
            oos_scores[oos_candidate] = _predict_template_scores(model, oos_x_risk.loc[oos_candidate].reset_index(drop=True))
        for score_thr in SCORE_THRESHOLDS:
            val_dec = _risk_decisions_from_scores(val_src, val_scores, oof=True, action=val_action, score_threshold=float(score_thr))
            oos_dec = _risk_decisions_from_scores(oos_src, oos_scores, oof=False, action=oos_action, score_threshold=float(score_thr))
            val = omega._metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
            oos = omega._metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
            rows.append(
                {
                    "variant": "supervised_risk_regressor",
                    "candidate_delta": float(delta),
                    "score_threshold": float(score_thr),
                    "train_rows": int(len(x_train)),
                    "train_target_mean": float(np.mean(y_train)),
                    "train_target_p90": float(np.quantile(y_train, 0.90)),
                    "val_candidate_rows": int(val_candidate.sum()),
                    "oos_candidate_rows": int(oos_candidate.sum()),
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_wr": val["wr"],
                    "val_trades": val["trades"],
                    "oos_pnl": oos["pnl"],
                    "oos_mdd": oos["mdd"],
                    "oos_wr": oos["wr"],
                    "oos_trades": oos["trades"],
                    "val_exit_reasons": val.get("exit_reasons", {}),
                    "oos_exit_reasons": oos.get("exit_reasons", {}),
                }
            )
    ranking = pd.DataFrame(rows).sort_values(["val_pnl", "val_wr"], ascending=False)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "CatBoostRegressor predicts template-level future trade score for relaxed Direction/Quality candidates. Runtime chooses the highest predicted risk template or vetoes when below score_threshold.",
        "risk_templates": sel.RISK_TEMPLATES,
        "train": {
            "candidate_delta": float(args.candidate_delta_train),
            "rows": int(len(x_train)),
            "target_mean": float(np.mean(y_train)),
            "target_p50": float(np.quantile(y_train, 0.50)),
            "target_p90": float(np.quantile(y_train, 0.90)),
        },
        "ranking": rows,
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "ranking.csv"),
            "model": str(out_dir / "supervised_risk_regressor.cbm"),
            "label_diag": str(out_dir / "risk_regression_label_diagnostics_head.csv"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.head(20).to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": str(out_dir / "ranking.csv")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
