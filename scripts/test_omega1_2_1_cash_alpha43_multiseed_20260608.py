#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_cash_alpha43_multiseed_20260608"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SEEDS = (260000, 260001, 260002, 260003, 260004, 260005, 260006, 260007, 260008, 260009, 260608, 260780)
LABEL_SPECS = tuple((a, h) for a in (0.6, 0.8, 1.0) for h in (24, 48, 72))
THRESHOLDS = (0.50, 0.55, 0.60, 0.65)
RISK = sleeve.FallbackRisk("tp026_sl014_n0.30_h192", 0.026, 0.014, 0.30, 2.0, 192)
BASELINE_VAL_PNL = 100.542729
BASELINE_OOS_PNL = 72.760041


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


def _row_metrics(prefix: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(m["pnl"]),
        f"{prefix}_mdd": float(m["mdd"]),
        f"{prefix}_wr": float(m["wr"]),
        f"{prefix}_trades": int(m["trades"]),
        f"{prefix}_fallback_entries": int(m.get("fallback_entries", 0)),
        f"{prefix}_primary_takeovers": int(m.get("primary_takeovers", 0)),
        f"{prefix}_reasons": m["exit_reasons"],
    }


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (label, threshold), g in rows.groupby(["label", "threshold"], sort=False):
        item: dict[str, Any] = {
            "label": label,
            "threshold": float(threshold),
            "runs": int(len(g)),
            "val_pnl_mean": float(g["val_pnl"].mean()),
            "val_pnl_median": float(g["val_pnl"].median()),
            "val_pnl_min": float(g["val_pnl"].min()),
            "val_pnl_max": float(g["val_pnl"].max()),
            "oos_pnl_mean": float(g["oos_pnl"].mean()),
            "oos_pnl_median": float(g["oos_pnl"].median()),
            "oos_pnl_min": float(g["oos_pnl"].min()),
            "oos_pnl_max": float(g["oos_pnl"].max()),
            "oos_mdd_mean": float(g["oos_mdd"].mean()),
            "oos_mdd_worst": float(g["oos_mdd"].min()),
            "oos_wr_mean": float(g["oos_wr"].mean()),
            "oos_trades_mean": float(g["oos_trades"].mean()),
            "beat_val_rate": float((g["val_pnl"] > BASELINE_VAL_PNL).mean()),
            "beat_oos_rate": float((g["oos_pnl"] > BASELINE_OOS_PNL).mean()),
            "beat_both_rate": float(((g["val_pnl"] > BASELINE_VAL_PNL) & (g["oos_pnl"] > BASELINE_OOS_PNL)).mean()),
            "oos_above_100_rate": float((g["oos_pnl"] >= 100.0).mean()),
        }
        out.append(item)
    return pd.DataFrame(out).sort_values(
        ["beat_both_rate", "oos_pnl_median", "oos_pnl_mean", "val_pnl_median"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = full._build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = full._build_split(frames, "oos")
    val_cash = ~omega._active(val_dec)

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "seeds": list(SEEDS),
        "labels": [],
        "risk": RISK.__dict__,
        "baseline": {"validation_pnl": BASELINE_VAL_PNL, "oos_pnl": BASELINE_OOS_PNL},
    }
    for atr_mult, label_hold in LABEL_SPECS:
        label = f"label_atr{atr_mult:g}_h{label_hold}"
        y_val, valid_mask, label_diag = label_family._triple_barrier_labels(
            val_frame,
            atr_mult=float(atr_mult),
            max_hold=int(label_hold),
            min_barrier=0.0035,
        )
        train_mask = val_cash & valid_mask
        diagnostics["labels"].append({"label": label, "train_rows": int(np.count_nonzero(train_mask)), "diag": label_diag})
        if int(np.count_nonzero(train_mask)) < 500 or len(np.unique(y_val[train_mask])) < 2:
            continue
        for seed in SEEDS:
            val_action, val_conf, oof_diag = label_family._predict_oof("hgb", val_features, y_val, train_mask, seed=int(seed))
            oos_action, oos_conf, _fitted = label_family._fit_predict("hgb", val_features, y_val, train_mask, oos_features, seed=int(seed))
            for threshold in THRESHOLDS:
                val_m = sleeve._metrics_with_fallback(
                    val_frame,
                    val_dec,
                    RISK,
                    val_action,
                    val_conf,
                    float(threshold),
                    fee=fee,
                    slip=slip,
                    cost_mult=3.0,
                )
                oos_m = sleeve._metrics_with_fallback(
                    oos_frame,
                    oos_dec,
                    RISK,
                    oos_action,
                    oos_conf,
                    float(threshold),
                    fee=fee,
                    slip=slip,
                    cost_mult=3.0,
                )
                rows.append(
                    {
                        "label": label,
                        "atr_mult": float(atr_mult),
                        "label_hold": int(label_hold),
                        "seed": int(seed),
                        "threshold": float(threshold),
                        "oof_rows": int(oof_diag["oof_rows"]),
                        **_row_metrics("val", val_m),
                        **_row_metrics("oos", oos_m),
                    }
                )
            print(json.dumps({"label": label, "seed": int(seed), "done": True}), flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "multiseed_detail.csv", index=False)
    summary = _summarize(detail)
    summary.to_csv(OUT_DIR / "multiseed_summary.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "parent_dir": str(full.PARENT_DIR),
        "diagnostics": diagnostics,
        "top_summary": summary.head(20).to_dict(orient="records"),
        "top_single_runs": detail.sort_values(["oos_pnl", "val_pnl"], ascending=[False, False]).head(30).to_dict(orient="records"),
        "artifacts": {
            "detail": str(OUT_DIR / "multiseed_detail.csv"),
            "summary": str(OUT_DIR / "multiseed_summary.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top_summary": report["top_summary"][:10]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
