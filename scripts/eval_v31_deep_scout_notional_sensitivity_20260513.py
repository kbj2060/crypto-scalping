#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame
from scripts.eval_hf_v13_frozen_v27_rule_exit_overlay_v31 import (
    DEFAULT_EVAL,
    DEFAULT_JACKPOT,
    DEFAULT_PARENT,
    DEFAULT_TRAIN,
    DEFAULT_V27,
    OverlayConfig,
    _audit_contract,
    _close,
    _load_v27,
    _predict_all,
    _read,
    backtest,
)
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig


REPORT_OUT = Path("data/ensemble/reports/v31_deep_scout_notional_sensitivity_20260513.json")
CSV_OUT = Path("data/ensemble/reports/v31_deep_scout_notional_sensitivity_20260513.csv")


def main() -> int:
    bundle = joblib.load(DEFAULT_PARENT)
    jackpot_payload = joblib.load(DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = _load_v27(DEFAULT_V27)
    base = dict(bundle["config"])
    train_all = _read(DEFAULT_TRAIN)
    eval_df = _read(DEFAULT_EVAL)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    print(f"feature_audit={feature_audit.get('status')} eval_rows={len(eval_df)}", flush=True)

    eval_q = _predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))

    base_cfg = OverlayConfig(
        "v31_notional1_time_decay",
        0.010,
        0.004,
        1.0,
        12,
        0.040,
        0.018,
        48,
        1.5,
        2.5,
        1.0,
        0.50,
        18,
        0.025,
        0.075,
        0.036,
    )
    rows: list[dict] = []
    for notional in [0.75, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        cfg = replace(base_cfg, name=f"v31_deep_notional_{notional:g}", notional=float(notional))
        metrics = {}
        for mult in (1, 2, 3):
            metrics[f"cost{mult}"] = backtest(
                eval_df,
                bundle,
                jackpot_model,
                add_cfg,
                eval_q,
                cfg,
                fee=float(base["fee"]),
                slip=float(base["slip"]),
                cost_mult=float(mult),
                decisions=eval_dec,
            )
        rows.append({"config": asdict(cfg), "metrics": metrics})
        print(
            "notional={:.2f} cost1_pnl={:.2f} cost1_mdd={:.2f} cost2_pnl={:.2f} cost3_pnl={:.2f}".format(
                float(notional),
                float(metrics["cost1"]["pnl"]),
                float(metrics["cost1"]["mdd"]),
                float(metrics["cost2"]["pnl"]),
                float(metrics["cost3"]["pnl"]),
            ),
            flush=True,
        )

    flat = []
    for row in rows:
        cfg = dict(row["config"])
        out = {"name": cfg["name"], "notional": float(cfg["notional"])}
        for key, metric in dict(row["metrics"]).items():
            out[f"{key}_pnl"] = float(metric["pnl"])
            out[f"{key}_mdd"] = float(metric["mdd"])
            out[f"{key}_trades"] = int(metric["trades"])
            out[f"{key}_deep_entries"] = int(metric.get("deep_entries", 0))
            out[f"{key}_avg_notional"] = float(metric.get("avg_notional", 0.0))
        flat.append(out)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(CSV_OUT, index=False)
    report = {
        "model_id": "v31_deep_scout_notional_sensitivity_20260513",
        "base_model": "hf_v13_frozen_v27_rule_exit_overlay_v31_20260511",
        "scope": "Only deep_alpha sleeve notional changed. Parent hf_v13_clean_regime_margin110 and V21.2 jackpot are unchanged.",
        "selection_uses_2026": False,
        "oos_window": "2026 fixed OOS, same eval CSV as V31",
        "feature_audit": feature_audit,
        "rows": rows,
        "csv": str(CSV_OUT),
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    best = max(flat, key=lambda x: float(x["cost1_pnl"]))
    print(json.dumps({"report": str(REPORT_OUT), "csv": str(CSV_OUT), "best_cost1": best}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
