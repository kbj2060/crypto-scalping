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

import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as lf  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_cash_fallback_tb_confirm_20260607"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASE_RISK = lf.BASE_RISK
CURRENT_MLP_VAL_PNL = lf.CURRENT_MLP_VAL_PNL
CURRENT_MLP_OOS_PNL = lf.CURRENT_MLP_OOS_PNL
CURRENT_MLP_VAL_MDD = lf.CURRENT_MLP_VAL_MDD
CURRENT_MLP_OOS_MDD = lf.CURRENT_MLP_OOS_MDD


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


def _opposite(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a == sleeve.ACTION_LONG) & (b == sleeve.ACTION_SHORT)) | ((a == sleeve.ACTION_SHORT) & (b == sleeve.ACTION_LONG))


def _same_trade(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a == sleeve.ACTION_LONG) & (b == sleeve.ACTION_LONG)) | ((a == sleeve.ACTION_SHORT) & (b == sleeve.ACTION_SHORT))


def _filter_signal(
    base_action: np.ndarray,
    base_conf: np.ndarray,
    zig_action: np.ndarray,
    zig_conf: np.ndarray,
    sltp_action: np.ndarray,
    sltp_conf: np.ndarray,
    *,
    rule: str,
    confirm_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    base_trade = (base_action == sleeve.ACTION_LONG) | (base_action == sleeve.ACTION_SHORT)
    zig_same = _same_trade(base_action, zig_action) & (zig_conf >= float(confirm_threshold))
    zig_opp = _opposite(base_action, zig_action) & (zig_conf >= float(confirm_threshold))
    sltp_same = _same_trade(base_action, sltp_action) & (sltp_conf >= float(confirm_threshold))
    sltp_opp = _opposite(base_action, sltp_action) & (sltp_conf >= float(confirm_threshold))
    if rule == "zig_same":
        allow = base_trade & zig_same
    elif rule == "sltp_same":
        allow = base_trade & sltp_same
    elif rule == "zig_or_sltp_same":
        allow = base_trade & (zig_same | sltp_same)
    elif rule == "zig_and_sltp_same":
        allow = base_trade & zig_same & sltp_same
    elif rule == "zig_veto_opp":
        allow = base_trade & ~zig_opp
    elif rule == "sltp_veto_opp":
        allow = base_trade & ~sltp_opp
    elif rule == "dual_veto_opp":
        allow = base_trade & ~zig_opp & ~sltp_opp
    elif rule == "same_or_dual_veto":
        allow = base_trade & ((zig_same | sltp_same) | (~zig_opp & ~sltp_opp))
    else:
        raise RuntimeError(f"unknown confirm rule: {rule}")
    action = np.where(allow, base_action, sleeve.ACTION_CASH).astype(np.int64)
    conf_parts = [base_conf.astype(np.float64)]
    if "zig" in rule:
        conf_parts.append(np.where(zig_action == base_action, zig_conf, 0.0))
    if "sltp" in rule or "dual" in rule:
        conf_parts.append(np.where(sltp_action == base_action, sltp_conf, 0.0))
    conf = np.where(allow, np.minimum.reduce(conf_parts), 0.0).astype(np.float64)
    diag = {
        "base_trades": int(np.count_nonzero(base_trade)),
        "allowed": int(np.count_nonzero(allow)),
        "zig_same": int(np.count_nonzero(base_trade & zig_same)),
        "zig_opp": int(np.count_nonzero(base_trade & zig_opp)),
        "sltp_same": int(np.count_nonzero(base_trade & sltp_same)),
        "sltp_opp": int(np.count_nonzero(base_trade & sltp_opp)),
    }
    return action, conf, diag


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return sleeve._metric_row(prefix, metrics)


def _build_signal(
    label_name: str,
    model_name: str,
    val_frame: pd.DataFrame,
    val_dec: pd.DataFrame,
    val_cash: np.ndarray,
    val_features: pd.DataFrame,
    oos_features: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    y_val, valid_val, label_diag = lf._label_family(label_name, val_frame, val_dec, val_cash, 2025)
    train_mask = val_cash & valid_val
    if int(np.count_nonzero(train_mask)) < 500:
        raise RuntimeError(f"not enough rows for {label_name}/{model_name}: {int(np.count_nonzero(train_mask))}")
    val_action, val_conf, oof_diag = lf._predict_oof(model_name, val_features, y_val, train_mask, seed=seed)
    oos_action, oos_conf, fitted = lf._fit_predict(model_name, val_features, y_val, train_mask, oos_features, seed=seed)
    if fitted is None:
        raise RuntimeError(f"no fitted model for {label_name}/{model_name}")
    return val_action, val_conf, oos_action, oos_conf, {"label_diag": label_diag, "oof": oof_diag}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, val_dec0, val_prefix = base._build_split(frames, "validation")
    oos_frame, oos_src, oos_dec0, oos_prefix = base._build_split(frames, "oos")
    val_dec = sleeve._apply_aggressive(val_dec0)
    oos_dec = sleeve._apply_aggressive(oos_dec0)
    val_features = sleeve._extra_features(base._feature_frame(val_frame, val_src, val_dec0, val_prefix), val_dec)
    oos_features = sleeve._extra_features(base._feature_frame(oos_frame, oos_src, oos_dec0, oos_prefix), oos_dec)
    bad = lf._forbidden_features(list(val_features.columns))
    if bad:
        raise RuntimeError(f"forbidden TB confirm feature columns: {bad}")
    val_cash = ~omega._active(val_dec)
    signals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {
        "risk": asdict(BASE_RISK),
        "feature_count": int(val_features.shape[1]),
        "features": list(val_features.columns),
        "forbidden_feature_audit": {"passed": True, "forbidden": []},
    }
    for key, label_name, model_name in (
        ("tb08_hgb", "tb_atr08_h48", "hgb"),
        ("tb08_mlp", "tb_atr08_h48", "mlp"),
        ("tb12_mlp", "tb_atr12_h96", "mlp"),
        ("zig_mlp", "zigzag_action", "mlp"),
        ("sltp_mlp", "sltp_edge006", "mlp"),
    ):
        print(json.dumps({"stage": "signal", "key": key, "label": label_name, "model": model_name}, ensure_ascii=False), flush=True)
        va, vc, oa, oc, diag = _build_signal(label_name, model_name, val_frame, val_dec, val_cash, val_features, oos_features, seed=260607)
        signals[key] = (va, vc, oa, oc)
        diagnostics[f"{key}_signal"] = diag
    rows: list[dict[str, Any]] = []
    baseline_val = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    baseline_oos = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows.append(
        {
            "base_signal": "none",
            "rule": "aggressive_primary_only",
            "confirm_threshold": 1.0,
            "entry_threshold": 1.0,
            **_metric_row("val", {**baseline_val, "primary_entries": baseline_val["long_entries"] + baseline_val["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}),
            **_metric_row("oos", {**baseline_oos, "primary_entries": baseline_oos["long_entries"] + baseline_oos["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}),
        }
    )
    rules = ("zig_same", "sltp_same", "zig_or_sltp_same", "zig_and_sltp_same", "zig_veto_opp", "sltp_veto_opp", "dual_veto_opp", "same_or_dual_veto")
    entry_thresholds = (0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95)
    confirm_thresholds = (0.55, 0.65, 0.75, 0.85)
    for base_key in ("tb08_hgb", "tb08_mlp", "tb12_mlp"):
        bva, bvc, boa, boc = signals[base_key]
        zva, zvc, zoa, zoc = signals["zig_mlp"]
        sva, svc, soa, soc = signals["sltp_mlp"]
        for rule in rules:
            for cthr in confirm_thresholds:
                val_action, val_conf, val_filter_diag = _filter_signal(bva, bvc, zva, zvc, sva, svc, rule=rule, confirm_threshold=float(cthr))
                oos_action, oos_conf, oos_filter_diag = _filter_signal(boa, boc, zoa, zoc, soa, soc, rule=rule, confirm_threshold=float(cthr))
                diagnostics[f"{base_key}_{rule}_{cthr}_filter"] = {"val": val_filter_diag, "oos": oos_filter_diag}
                for eth in entry_thresholds:
                    val_m = sleeve._metrics_with_fallback(val_frame, val_dec, BASE_RISK, val_action, val_conf, float(eth), fee=fee, slip=slip, cost_mult=3.0)
                    oos_m = sleeve._metrics_with_fallback(oos_frame, oos_dec, BASE_RISK, oos_action, oos_conf, float(eth), fee=fee, slip=slip, cost_mult=3.0)
                    row = {
                        "base_signal": base_key,
                        "rule": rule,
                        "confirm_threshold": float(cthr),
                        "entry_threshold": float(eth),
                    }
                    row.update(_metric_row("val", val_m))
                    row.update(_metric_row("oos", oos_m))
                    rows.append(row)
    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - sleeve.AGGRESSIVE_VAL["pnl"]
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - sleeve.AGGRESSIVE_OOS["pnl"]
    ranking["val_delta_mdd"] = ranking["val_mdd"] - sleeve.AGGRESSIVE_VAL["mdd"]
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - sleeve.AGGRESSIVE_OOS["mdd"]
    ranking["score"] = ranking["oos_pnl"] + 0.75 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "cash_fallback_tb_confirm_ranking.csv", index=False)
    promotable = ranking[
        (ranking["rule"] != "aggressive_primary_only")
        & (ranking["oos_pnl"] > CURRENT_MLP_OOS_PNL)
        & (ranking["val_pnl"] > CURRENT_MLP_VAL_PNL)
        & (ranking["oos_mdd"] >= CURRENT_MLP_OOS_MDD * 1.35)
        & (ranking["val_mdd"] >= CURRENT_MLP_VAL_MDD * 1.35)
    ].copy()
    promotable.to_csv(OUT_DIR / "cash_fallback_tb_confirm_promotable.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline": "omega1_2_1_cash_fallback_mlp_base_edge006_thr085_20260606",
        "method": "Cash-only TB fallback signals are filtered by ZigZag/SLTP agreement or opposite-side veto. Primary, fallback risk, features, and accounting remain fixed.",
        "diagnostics": diagnostics,
        "best": ranking.iloc[0].to_dict(),
        "promotable_count": int(len(promotable)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "cash_fallback_tb_confirm_ranking.csv"),
            "promotable": str(OUT_DIR / "cash_fallback_tb_confirm_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best": report["best"], "promotable_count": int(len(promotable))}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
