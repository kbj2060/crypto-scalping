#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega_alpha6_style_eqh_overlay_20260608 as eqh_overlay  # noqa: E402
from alpha6_catboost_entry_quality_exit_policy_20260522 import EQEConfig, _build_entry_labels, _predict_entry  # noqa: E402


MODEL_ID = "omega1_2_1_cash_alpha_arch_sleeves_20260608"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

BASELINE_VAL = sleeve.AGGRESSIVE_VAL
BASELINE_OOS = sleeve.AGGRESSIVE_OOS

FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "teacher_")
FORBIDDEN_EXACT = {"tp_sl_action_score"}

RISKS = [
    sleeve.FallbackRisk("micro_tp010_sl007_n020_h96", 0.010, 0.007, 0.20, 2.0, 96),
    sleeve.FallbackRisk("micro_tp010_sl007_n030_h96", 0.010, 0.007, 0.30, 2.0, 96),
    sleeve.FallbackRisk("base_tp026_sl014_n030_h192", 0.026, 0.014, 0.30, 2.0, 192),
    sleeve.FallbackRisk("base_tp026_sl014_n0405_h192", 0.026, 0.014, 0.405, 2.0, 192),
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _forbidden_features(cols: list[str]) -> list[str]:
    return [c for c in cols if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIXES)]


def _build_split(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, src, dec0, prefix = base._build_split(frames, split)
    dec = sleeve._apply_aggressive(dec0)
    feat = sleeve._extra_features(base._feature_frame(frame, src, dec0, prefix), dec)
    bad = _forbidden_features(list(feat.columns))
    if bad:
        raise RuntimeError(f"{split}: forbidden Omega cash sleeve feature columns: {bad}")
    return frame, dec, feat


def _baseline_row(val_frame: pd.DataFrame, val_dec: pd.DataFrame, oos_frame: pd.DataFrame, oos_dec: pd.DataFrame, fee: float, slip: float) -> dict[str, Any]:
    val_m = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    oos_m = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    return {
        "candidate": "aggressive_primary_only",
        "family": "baseline",
        "risk": "none",
        "threshold": 1.0,
        **sleeve._metric_row(
            "val",
            {**val_m, "primary_entries": val_m["long_entries"] + val_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0},
        ),
        **sleeve._metric_row(
            "oos",
            {**oos_m, "primary_entries": oos_m["long_entries"] + oos_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0},
        ),
    }


def _eqh_cash_oof(frame: pd.DataFrame, dec: pd.DataFrame, features: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = EQEConfig()
    valid_rows, y, label_meta = _build_entry_labels(
        frame,
        cfg,
        stride_bars=3,
        batch_size=4096,
        label_preset="current_quality",
        adaptive_sampling=False,
    )
    cash = ~omega._active(dec)
    eligible_pos = np.flatnonzero(cash[valid_rows])
    x_valid = features.to_numpy(dtype=np.float64)[valid_rows]
    action = np.zeros(len(frame), dtype=np.int64)
    conf = np.zeros(len(frame), dtype=np.float64)
    folds: list[dict[str, int]] = []
    n = len(eligible_pos)
    for start_frac, end_frac in ((0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * start_frac)
        val_start = train_end
        val_end = int(n * end_frac)
        if train_end < 500 or val_end <= val_start:
            continue
        train_pos = eligible_pos[:train_end]
        val_pos = eligible_pos[val_start:val_end]
        heads = eqh_overlay._fit_heads(x_valid[train_pos], {k: v[train_pos] for k, v in y.items()}, seed=seed + val_start)
        pred = _predict_entry(heads, x_valid[val_pos], cfg)
        rows = valid_rows[val_pos]
        action[rows] = pd.to_numeric(pred["action"], errors="raise").to_numpy(dtype=np.int64)
        conf[rows] = pd.to_numeric(pred["quality_score"], errors="raise").to_numpy(dtype=np.float64)
        folds.append({"train_cash_valid_rows": int(train_end), "val_cash_valid_rows": int(len(val_pos))})
    return action, conf, {"label_meta": label_meta, "eligible_cash_rows": int(n), "folds": folds}


def _eqh_cash_full_predict(
    train_frame: pd.DataFrame,
    train_dec: pd.DataFrame,
    train_features: pd.DataFrame,
    eval_features: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = EQEConfig()
    valid_rows, y, label_meta = _build_entry_labels(
        train_frame,
        cfg,
        stride_bars=3,
        batch_size=4096,
        label_preset="current_quality",
        adaptive_sampling=False,
    )
    cash = ~omega._active(train_dec)
    eligible_pos = np.flatnonzero(cash[valid_rows])
    heads = eqh_overlay._fit_heads(
        train_features.to_numpy(dtype=np.float64)[valid_rows][eligible_pos],
        {k: v[eligible_pos] for k, v in y.items()},
        seed=seed,
    )
    pred = _predict_entry(heads, eval_features.to_numpy(dtype=np.float64), cfg)
    return (
        pd.to_numeric(pred["action"], errors="raise").to_numpy(dtype=np.int64),
        pd.to_numeric(pred["quality_score"], errors="raise").to_numpy(dtype=np.float64),
        {"label_meta": label_meta, "eligible_cash_rows": int(len(eligible_pos)), "heads": heads["label_distribution"]},
    )


def _zscore_signal(frame: pd.DataFrame, *, window: int, z_abs: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    close = pd.to_numeric(frame["close"], errors="raise")
    mean = close.rolling(window, min_periods=max(12, window // 4)).mean()
    std = close.rolling(window, min_periods=max(12, window // 4)).std().replace(0.0, np.nan)
    z = ((close - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    action = np.zeros(len(frame), dtype=np.int64)
    action[z <= -abs(float(z_abs))] = sleeve.ACTION_LONG
    action[z >= abs(float(z_abs))] = sleeve.ACTION_SHORT
    conf = np.clip(np.abs(z) / max(abs(float(z_abs)), 1.0e-12), 0.0, 3.0)
    return action, conf, {"window": int(window), "z_abs": float(z_abs), "signal_counts": _counts(action)}


def _counts(arr: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in pd.Series(arr).value_counts().sort_index().items()}


def _evaluate_action_family(
    rows: list[dict[str, Any]],
    *,
    family: str,
    risk: sleeve.FallbackRisk,
    val_frame: pd.DataFrame,
    val_dec: pd.DataFrame,
    val_action: np.ndarray,
    val_conf: np.ndarray,
    oos_frame: pd.DataFrame,
    oos_dec: pd.DataFrame,
    oos_action: np.ndarray,
    oos_conf: np.ndarray,
    thresholds: tuple[float, ...],
    fee: float,
    slip: float,
    extra: dict[str, Any] | None = None,
) -> None:
    for threshold in thresholds:
        val_m = sleeve._metrics_with_fallback(val_frame, val_dec, risk, val_action, val_conf, threshold, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = sleeve._metrics_with_fallback(oos_frame, oos_dec, risk, oos_action, oos_conf, threshold, fee=fee, slip=slip, cost_mult=3.0)
        row: dict[str, Any] = {
            "candidate": f"{family}_{risk.name}_thr{threshold:g}",
            "family": family,
            "risk": risk.name,
            "threshold": float(threshold),
            **sleeve._metric_row("val", val_m),
            **sleeve._metric_row("oos", oos_m),
        }
        if extra:
            row.update(extra)
        rows.append(row)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = _build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = _build_split(frames, "oos")

    rows: list[dict[str, Any]] = [_baseline_row(val_frame, val_dec, oos_frame, oos_dec, fee, slip)]
    diagnostics: dict[str, Any] = {
        "model_id": MODEL_ID,
        "feature_contract": {
            "source": "Omega-only base._feature_frame + sleeve._extra_features",
            "feature_count": int(val_features.shape[1]),
            "features": list(val_features.columns),
            "forbidden_feature_audit": {"passed": True, "forbidden": []},
        },
        "val_cash_rows": int(np.count_nonzero(~omega._active(val_dec))),
        "oos_cash_rows": int(np.count_nonzero(~omega._active(oos_dec))),
        "risks": [asdict(r) for r in RISKS],
    }

    print(json.dumps({"stage": "alpha6_eqh_cash"}, ensure_ascii=False), flush=True)
    val_eqh_action, val_eqh_conf, val_eqh_diag = _eqh_cash_oof(val_frame, val_dec, val_features, seed=260608)
    oos_eqh_action, oos_eqh_conf, oos_eqh_diag = _eqh_cash_full_predict(val_frame, val_dec, val_features, oos_features, seed=260608)
    diagnostics["alpha6_eqh_cash"] = {"validation_oof": val_eqh_diag, "oos_full_train": oos_eqh_diag}
    for risk in RISKS:
        _evaluate_action_family(
            rows,
            family="alpha6_eqh_cash",
            risk=risk,
            val_frame=val_frame,
            val_dec=val_dec,
            val_action=val_eqh_action,
            val_conf=val_eqh_conf,
            oos_frame=oos_frame,
            oos_dec=oos_dec,
            oos_action=oos_eqh_action,
            oos_conf=oos_eqh_conf,
            thresholds=(0.0015, 0.0020, 0.0025, 0.0030, 0.0040),
            fee=fee,
            slip=slip,
        )

    print(json.dumps({"stage": "alpha43_style_parent_cash"}, ensure_ascii=False), flush=True)
    val_cash = ~omega._active(val_dec)
    oos_cash = ~omega._active(oos_dec)
    for label_name in ("sltp_edge006", "tb_atr08_h48", "topk2_8h", "reversal_z12_h24"):
        y_val, valid_mask, label_diag = label_family._label_family(label_name, val_frame, val_dec, val_cash, 2025)
        train_mask = val_cash & valid_mask
        diagnostics[f"alpha43_style_{label_name}"] = label_diag
        if len(set(y_val[train_mask].tolist())) < 2:
            continue
        for model_name in ("hgb", "extra", "mlp"):
            val_action, val_conf, oof_diag = label_family._predict_oof(model_name, val_features, y_val, train_mask, seed=260608)
            oos_action, oos_conf, _fitted = label_family._fit_predict(model_name, val_features, y_val, train_mask, oos_features, seed=260608)
            diagnostics[f"alpha43_style_{label_name}_{model_name}_oof"] = oof_diag
            for risk in RISKS:
                _evaluate_action_family(
                    rows,
                    family=f"alpha43_style_{label_name}_{model_name}",
                    risk=risk,
                    val_frame=val_frame,
                    val_dec=val_dec,
                    val_action=val_action,
                    val_conf=val_conf,
                    oos_frame=oos_frame,
                    oos_dec=oos_dec,
                    oos_action=oos_action,
                    oos_conf=oos_conf,
                    thresholds=(0.55, 0.65, 0.75, 0.85, 0.90),
                    fee=fee,
                    slip=slip,
                )

    print(json.dumps({"stage": "micro_mean_reversion_cash"}, ensure_ascii=False), flush=True)
    for window in (48, 96):
        for z_abs in (1.2, 1.6, 2.0):
            val_action, val_conf, val_diag = _zscore_signal(val_frame, window=window, z_abs=z_abs)
            oos_action, oos_conf, oos_diag = _zscore_signal(oos_frame, window=window, z_abs=z_abs)
            diagnostics[f"micro_reversion_w{window}_z{z_abs:g}"] = {"validation": val_diag, "oos": oos_diag}
            for risk in RISKS[:2]:
                _evaluate_action_family(
                    rows,
                    family=f"micro_reversion_w{window}_z{z_abs:g}",
                    risk=risk,
                    val_frame=val_frame,
                    val_dec=val_dec,
                    val_action=val_action,
                    val_conf=val_conf,
                    oos_frame=oos_frame,
                    oos_dec=oos_dec,
                    oos_action=oos_action,
                    oos_conf=oos_conf,
                    thresholds=(1.0, 1.25, 1.5, 2.0),
                    fee=fee,
                    slip=slip,
                )

    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - BASELINE_VAL["pnl"]
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - BASELINE_OOS["pnl"]
    ranking["val_delta_mdd"] = ranking["val_mdd"] - BASELINE_VAL["mdd"]
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - BASELINE_OOS["mdd"]
    ranking["promotable"] = (
        (ranking["family"] != "baseline")
        & (ranking["val_pnl"] > BASELINE_VAL["pnl"])
        & (ranking["oos_pnl"] > BASELINE_OOS["pnl"])
        & (ranking["val_mdd"] >= -12.0)
        & (ranking["oos_mdd"] >= -10.0)
    )
    ranking["score"] = ranking["oos_pnl"] + 0.50 * ranking["val_pnl"] + 0.25 * ranking["oos_mdd"] + 0.25 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "cash_alpha_arch_sleeves_ranking.csv", index=False)
    ranking[ranking["promotable"]].to_csv(OUT_DIR / "cash_alpha_arch_sleeves_promotable.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline": {"model_id": "omega1_2_1_aggressive_compensated_scale200_cap090", "validation": BASELINE_VAL, "oos": BASELINE_OOS},
        "method": "Omega1.2.1 aggressive primary is preserved. Alpha architectures are retrained with Omega-only features and used only when primary is CASH.",
        "diagnostics": diagnostics,
        "best": ranking.iloc[0].to_dict(),
        "promotable_count": int(ranking["promotable"].sum()),
        "top20": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "cash_alpha_arch_sleeves_ranking.csv"),
            "promotable": str(OUT_DIR / "cash_alpha_arch_sleeves_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best": report["best"], "promotable_count": report["promotable_count"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
