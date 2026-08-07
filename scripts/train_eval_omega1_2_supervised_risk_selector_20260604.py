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
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_supervised_risk_selector_20260604"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
PRACTICAL_EXPERT_THRESHOLDS = {"bull": 0.72, "bear": 0.64, "chop": 0.65}
PRACTICAL_TP = 0.026
PRACTICAL_SL = 0.012
RISK_TEMPLATES = [
    {"name": "cash", "tp": 0.0, "sl": 0.0, "notional": 0.0, "leverage": 1.0},
    {"name": "scalp_tight_small", "tp": 0.018, "sl": 0.008, "notional": 0.25, "leverage": 2.0},
    {"name": "scalp_tight_mid", "tp": 0.018, "sl": 0.010, "notional": 0.35, "leverage": 2.0},
    {"name": "base_small", "tp": 0.026, "sl": 0.012, "notional": 0.3375, "leverage": 2.0},
    {"name": "base_mid", "tp": 0.026, "sl": 0.012, "notional": 0.405, "leverage": 2.0},
    {"name": "base_full", "tp": 0.026, "sl": 0.012, "notional": 0.45, "leverage": 2.0},
    {"name": "base_large", "tp": 0.026, "sl": 0.012, "notional": 0.55, "leverage": 2.0},
    {"name": "tight_sl_mid", "tp": 0.026, "sl": 0.010, "notional": 0.405, "leverage": 2.0},
    {"name": "loose_runner_mid", "tp": 0.030, "sl": 0.014, "notional": 0.405, "leverage": 2.0},
    {"name": "wide_runner_mid", "tp": 0.034, "sl": 0.014, "notional": 0.405, "leverage": 2.0},
    {"name": "wide_runner_large", "tp": 0.034, "sl": 0.018, "notional": 0.45, "leverage": 2.0},
]
PRED_COLS = [
    "router_confidence",
    "router_margin",
    "dir_p_cash",
    "dir_p_long",
    "dir_p_short",
    "dir_confidence",
    "dir_side_edge",
    "dir_trade_prob",
    "dir_action",
    "quality_p_cash",
    "quality_p_long",
    "quality_p_short",
    "quality_for_action",
]


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _prefix(oof: bool) -> str:
    return "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"


def _read_predictions(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"timestamp contract mismatch: {path}")
    return pred


@torch.no_grad()
def _predict_frame(frame: pd.DataFrame, bundle: dict[str, Any], *, oof: bool, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = list(bundle["base_cols"])
    x = tabm._base_input(frame, base_cols)
    preds = {expert: tabm._predict_payload(bundle["models"][expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = tabm._routed(preds, route, "direction", 3)
    quality = tabm._routed(preds, route, "quality", 3)
    src = tabm._prediction_output(frame, direction, quality, threshold=0.50, prefix=_prefix(oof).rstrip("_"))
    return x, src


def _threshold_action(src: pd.DataFrame, *, oof: bool, thresholds: dict[str, float]) -> np.ndarray:
    prefix = _prefix(oof)
    action = pd.to_numeric(src[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    q = pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    expert = src[f"{prefix}router_expert"].astype(str).to_numpy()
    thr = np.array([float(thresholds.get(str(x).replace("chop_expert", "chop"), 1.0)) for x in expert], dtype=np.float64)
    return np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)


def _candidate_thresholds(delta: float) -> dict[str, float]:
    return {k: float(np.clip(v + float(delta), 0.01, 0.99)) for k, v in PRACTICAL_EXPERT_THRESHOLDS.items()}


def _risk_features(base_x: pd.DataFrame, src: pd.DataFrame, *, oof: bool) -> pd.DataFrame:
    prefix = _prefix(oof)
    out = base_x.copy().reset_index(drop=True)
    for col in PRED_COLS:
        out[f"pred_{col}"] = pd.to_numeric(src[f"{prefix}{col}"], errors="raise").to_numpy(dtype=np.float32)
    route = src[f"{prefix}router_expert"].astype(str).replace({"chop": "chop_expert"})
    for expert in ("bull", "bear", "chop_expert"):
        out[f"router_is_{expert}"] = route.eq(expert).astype(np.float32).to_numpy()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _risk_decision(src: pd.DataFrame, *, oof: bool, action: np.ndarray, risk_class: np.ndarray) -> pd.DataFrame:
    prefix = _prefix(oof)
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    risk_class = np.asarray(risk_class, dtype=np.int64)
    use = (action != omega.ACTION_CASH) & (risk_class > 0)
    router = src[f"{prefix}router_expert"].astype(str).replace({"chop": "chop_expert"}).to_numpy()
    dec = pd.DataFrame(
        {
            "action": np.where(use, action, omega.ACTION_CASH).astype(np.int64),
            "side": np.where(use, side, 0).astype(np.int64),
            "notional_exposure": 0.0,
            "leverage": 1.0,
            "position_fraction": 0.0,
            "take_profit": 0.0,
            "stop_loss": 0.0,
            "max_hold_bars": 0,
            "cooldown_bars": 0,
            "quality_score": pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64),
            "confidence": pd.to_numeric(src[f"{prefix}dir_confidence"], errors="raise").to_numpy(dtype=np.float64),
            "router_expert": router,
        }
    )
    for cls, template in enumerate(RISK_TEMPLATES):
        if cls == 0:
            continue
        mask = use & (risk_class == int(cls))
        dec.loc[mask, "notional_exposure"] = float(template["notional"])
        dec.loc[mask, "position_fraction"] = float(template["notional"])
        dec.loc[mask, "leverage"] = float(template["leverage"])
        dec.loc[mask, "take_profit"] = float(template["tp"])
        dec.loc[mask, "stop_loss"] = float(template["sl"])
    return dec


def _single_dec_row(action: int, side: int, template: dict[str, float]) -> pd.Series:
    return pd.Series(
        {
            "action": int(action),
            "side": int(side),
            "notional_exposure": float(template["notional"]),
            "leverage": float(template["leverage"]),
            "position_fraction": float(template["notional"]),
            "take_profit": float(template["tp"]),
            "stop_loss": float(template["sl"]),
            "max_hold_bars": 72,
            "cooldown_bars": 0,
            "router_expert": "",
        }
    )


def _build_risk_labels(
    frame: pd.DataFrame,
    src: pd.DataFrame,
    *,
    oof: bool,
    candidate_delta: float,
    min_score: float,
    fee: float,
    slip: float,
    cost_mult: float,
    max_candidates: int,
    require_take_profit: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    action = _threshold_action(src, oof=oof, thresholds=_candidate_thresholds(float(candidate_delta)))
    candidate_idx = np.flatnonzero(action != omega.ACTION_CASH)
    if int(max_candidates) > 0 and len(candidate_idx) > int(max_candidates):
        keep = np.linspace(0, len(candidate_idx) - 1, int(max_candidates)).round().astype(np.int64)
        candidate_idx = candidate_idx[keep]
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    labels = np.zeros(len(candidate_idx), dtype=np.int64)
    diag_rows: list[dict[str, Any]] = []
    for out_i, idx in enumerate(candidate_idx):
        act = int(action[int(idx)])
        side = 1 if act == omega.ACTION_LONG else -1
        best_score = -1e9
        best_cls = 0
        best_meta: dict[str, Any] = {}
        for cls, template in enumerate(RISK_TEMPLATES):
            if cls == 0:
                continue
            score, meta = omega._simulate_trade(
                frame,
                arrays,
                int(idx),
                _single_dec_row(act, side, template),
                fee=fee,
                slip=slip,
                cost_mult=cost_mult,
            )
            if float(score) > float(best_score):
                best_score = float(score)
                best_cls = int(cls)
                best_meta = dict(meta)
        if best_score >= float(min_score) and ((not bool(require_take_profit)) or str(best_meta.get("exit_reason", "")) == "take_profit"):
            labels[out_i] = int(best_cls)
        if out_i < 2000:
            diag_rows.append(
                {
                    "row": int(idx),
                    "timestamp": str(frame["timestamp"].iloc[int(idx)]),
                    "action": int(act),
                    "label": int(labels[out_i]),
                    "best_score": float(best_score),
                    "best_net": float(best_meta.get("net", 0.0)),
                    "best_exit_reason": str(best_meta.get("exit_reason", "")),
                }
            )
    return candidate_idx.astype(np.int64), labels, pd.DataFrame(diag_rows)


def _fit_model(x: pd.DataFrame, y: np.ndarray, *, seed: int, iterations: int) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=int(iterations),
        depth=5,
        learning_rate=0.045,
        l2_leaf_reg=8.0,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(x, y)
    return model


def _predict_risk(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(model.predict(x), dtype=np.int64).reshape(-1)
    if not set(np.unique(pred)).issubset(set(range(len(RISK_TEMPLATES)))):
        raise RuntimeError(f"unexpected risk classes: {sorted(np.unique(pred).tolist())}")
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--candidate-deltas", default="-0.30,-0.25,-0.20,-0.15,-0.10")
    ap.add_argument("--candidate-delta-train", type=float, default=-0.25)
    ap.add_argument("--min-score", type=float, default=0.0010)
    ap.add_argument("--max-candidates", type=int, default=6000)
    ap.add_argument("--allow-non-tp-labels", action="store_true")
    ap.add_argument("--iterations", type=int, default=700)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260604)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    device = _device(args.device)
    out_dir = OUT_DIR if not args.out_suffix.strip() else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix.strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = tabm._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    bundle = torch.load(ZIGZAG_DIR / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    train_x_base, train_src = _predict_frame(frames["train_raw"], bundle, oof=True, device=device)
    val_src = _read_predictions(ZIGZAG_DIR / "validation_predictions_2025_true3head.csv", frames["val_raw"])
    oos_src = _read_predictions(ZIGZAG_DIR / "oos_predictions_2026_true3head.csv", frames["oos_raw"])
    val_x_base = tabm._base_input(frames["val_raw"], list(bundle["base_cols"]))
    oos_x_base = tabm._base_input(frames["oos_raw"], list(bundle["base_cols"]))
    train_x_risk = _risk_features(train_x_base, train_src, oof=True)
    val_x_risk = _risk_features(val_x_base, val_src, oof=True)
    oos_x_risk = _risk_features(oos_x_base, oos_src, oof=False)

    train_idx, y_risk, label_diag = _build_risk_labels(
        frames["train_raw"],
        train_src,
        oof=True,
        candidate_delta=float(args.candidate_delta_train),
        min_score=float(args.min_score),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_candidates=int(args.max_candidates),
        require_take_profit=not bool(args.allow_non_tp_labels),
    )
    if len(train_idx) < 200:
        raise RuntimeError(f"not enough risk training candidates: {len(train_idx)}")
    model = _fit_model(train_x_risk.iloc[train_idx].reset_index(drop=True), y_risk, seed=int(args.seed), iterations=int(args.iterations))
    model.save_model(str(out_dir / "supervised_risk_selector.cbm"))
    label_diag.to_csv(out_dir / "risk_label_diagnostics_head.csv", index=False)

    rows: list[dict[str, Any]] = []
    for delta_s in str(args.candidate_deltas).split(","):
        delta = float(delta_s)
        val_action = _threshold_action(val_src, oof=True, thresholds=_candidate_thresholds(delta))
        oos_action = _threshold_action(oos_src, oof=False, thresholds=_candidate_thresholds(delta))
        val_candidate = val_action != omega.ACTION_CASH
        oos_candidate = oos_action != omega.ACTION_CASH
        val_risk = np.zeros(len(val_action), dtype=np.int64)
        oos_risk = np.zeros(len(oos_action), dtype=np.int64)
        if bool(val_candidate.any()):
            val_risk[val_candidate] = _predict_risk(model, val_x_risk.loc[val_candidate].reset_index(drop=True))
        if bool(oos_candidate.any()):
            oos_risk[oos_candidate] = _predict_risk(model, oos_x_risk.loc[oos_candidate].reset_index(drop=True))
        val_dec = _risk_decision(val_src, oof=True, action=val_action, risk_class=val_risk)
        oos_dec = _risk_decision(oos_src, oof=False, action=oos_action, risk_class=oos_risk)
        val = omega._metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos = omega._metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        rows.append(
            {
                "variant": "supervised_risk_selector",
                "candidate_delta": float(delta),
                "train_candidates": int(len(train_idx)),
                "train_label_cash_rate": float((y_risk == 0).mean()),
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
        "design": "Supervised risk selector trained on relaxed Direction/Quality candidates. Labels are argmax over finite TP/SL/notional risk templates, with CASH if no template clears min_score.",
        "risk_templates": RISK_TEMPLATES,
        "train": {
            "candidate_delta": float(args.candidate_delta_train),
            "candidates": int(len(train_idx)),
            "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_risk, minlength=len(RISK_TEMPLATES)))},
            "min_score": float(args.min_score),
        },
        "ranking": rows,
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "ranking.csv"),
            "model": str(out_dir / "supervised_risk_selector.cbm"),
            "label_diag": str(out_dir / "risk_label_diagnostics_head.csv"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": str(out_dir / "ranking.csv")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
