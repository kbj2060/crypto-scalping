#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.elite import RegimeEngine  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import retest_alpha3_current_live_guard_20260515 as liveguard  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402


MODEL_ID = "audit_alpha3_backtest_live_pipeline_parity_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/audit_alpha3_backtest_live_pipeline_parity_20260515.json"
HEALTH_PATH = ROOT / "data/live/data_pipeline_health.json"
HEALTH_JSONL = ROOT / "data/live/data_pipeline_health.jsonl"
SNAPSHOT_PATH = ROOT / "data/live/decision_feature_snapshot.json"
LOG_PATH = ROOT / "data/live/trading_bot_stdout.log"
FINAL_GOVERNOR_AI_FEATURE_GROUPS = tuple(
    x.strip().lower()
    for x in os.getenv("FINAL_GOVERNOR_AI_FEATURE_GROUPS", "patchtst,tide,dlinear").split(",")
    if x.strip()
)
AI_FEATURE_COLUMNS = {
    "patchtst": [
        "pred_patchtst",
        "conf_patchtst",
        "ai_dir_edge",
        "ai_dir_p_up",
        "ai_dir_p_down",
        "ai_dir_p_flat",
        "ai_dir_entropy",
        "patchtst_median",
        "patchtst_regime_sim",
    ],
    "tide": [
        "ai_adverse_risk",
        "ai_reward_risk",
        "ai_vol_regime_pct",
        "tide_vol_raw",
        "tide_vol_zscore",
    ],
    "timesnet": [
        "ai_anchor_revert_prob",
        "ai_anchor_overheat",
        "ai_anchor_trend_escape_prob",
        "timesnet_cycle_sin",
        "timesnet_cycle_cos",
        "timesnet_cycle_delta",
    ],
    "dlinear": [
        "ai_flow_pressure",
        "ai_flow_exhaustion",
        "ai_flow_flip_prob",
        "ai_flow_slope",
        "dlinear_smf_ema",
        "dlinear_smf_slope",
    ],
}


def _json_default_safe(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return _json_default(obj)


def _ts(x: Any) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(x)
        if t.tzinfo is not None:
            t = t.tz_convert("Asia/Seoul").tz_localize(None)
        return t
    except Exception:
        return None


def _frame_stats(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": int(len(df)), "cols": int(len(df.columns))}
    if "timestamp" in df.columns and len(df):
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        out.update(
            {
                "first_ts": str(ts.min()),
                "last_ts": str(ts.max()),
                "duplicate_timestamps": int(ts.duplicated().sum()),
                "median_gap_sec": float(ts.sort_values().diff().dt.total_seconds().median()),
                "bad_5m_gaps": int((ts.sort_values().diff().dt.total_seconds().dropna() != 300.0).sum()),
            }
        )
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns):
        last = numeric.iloc[-1]
        out.update(
            {
                "numeric_cols": int(len(numeric.columns)),
                "last_nan": int(last.isna().sum()),
                "last_inf": int(np.isinf(last.astype(float).to_numpy()).sum()),
                "tail_nan_ratio": float(numeric.tail(min(120, len(numeric))).isna().sum().sum() / max(1, numeric.tail(min(120, len(numeric))).size)),
            }
        )
    return out


def _required_contract(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    missing = [c for c in cols if c not in df.columns]
    nonfinite: list[str] = []
    if len(df):
        last = df.iloc[-1]
        for col in cols:
            if col not in df.columns:
                continue
            try:
                val = float(last.get(col, np.nan))
                if not np.isfinite(val):
                    nonfinite.append(col)
            except Exception:
                nonfinite.append(col)
    return {
        "required": int(len(cols)),
        "missing_count": int(len(missing)),
        "missing_cols": missing[:40],
        "last_nonfinite_count": int(len(nonfinite)),
        "last_nonfinite_cols": nonfinite[:40],
    }


def _required_values_contract(values: dict[str, Any], cols: list[str]) -> dict[str, Any]:
    missing = [c for c in cols if c not in values]
    nonfinite: list[str] = []
    for col in cols:
        if col not in values:
            continue
        try:
            val = values.get(col, np.nan)
            if val is None:
                nonfinite.append(col)
                continue
            if not np.isfinite(float(val)):
                nonfinite.append(col)
        except Exception:
            nonfinite.append(col)
    return {
        "required": int(len(cols)),
        "missing_count": int(len(missing)),
        "missing_cols": missing[:40],
        "last_nonfinite_count": int(len(nonfinite)),
        "last_nonfinite_cols": nonfinite[:40],
    }


def _read_latest_health() -> dict[str, Any]:
    if HEALTH_PATH.exists():
        try:
            return json.loads(HEALTH_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _read_latest_snapshot() -> dict[str, Any]:
    if SNAPSHOT_PATH.exists():
        try:
            return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _health_tail(n: int = 12) -> list[dict[str, Any]]:
    if not HEALTH_JSONL.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in HEALTH_JSONL.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]:
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _recent_log_flags() -> dict[str, Any]:
    if not LOG_PATH.exists():
        return {"exists": False}
    text = "\n".join(LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-250:])
    return {
        "exists": True,
        "has_redteam_block_after_recent_restart": "reason=alpha3_redteam_blocked" in text,
        "has_v31_on": "SYSTEM v31_frozen_v27_rule_exit=ON" in text,
        "has_alpha2_on": "SYSTEM alpha2_1=ON" in text,
        "has_governor_ready": "SYSTEM governor=READY" in text,
        "last_pipe_line": next((line for line in reversed(text.splitlines()) if "[PIPE]" in line), ""),
        "last_ai_line": next((line for line in reversed(text.splitlines()) if "[AI]" in line), ""),
    }


def main() -> int:
    print(f"[{MODEL_ID}] loading backtest stack and OOS artifact", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    eval_raw = _read(v31.DEFAULT_EVAL)
    eval_df = RegimeEngine().compute(eval_raw.copy())
    parent_cols = list(stack["parent"].get("feature_cols") or [])
    v27_cols = list(stack["v27_payload"].get("seq_cols") or [])
    teacher_cols = list(stack["teacher_payload"].get("seq_cols") or [])
    close_arr = pd.to_numeric(eval_df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)
    parent_feature_df = prepare_features(eval_df, side_hint=0, close=close_arr, feature_cols=parent_cols)
    teacher_feature_df = prepare_features(eval_df, side_hint=0, close=close_arr, feature_cols=teacher_cols) if teacher_cols else pd.DataFrame(index=eval_df.index)
    ai_cols = sorted(
        {
            col
            for group in FINAL_GOVERNOR_AI_FEATURE_GROUPS
            for col in AI_FEATURE_COLUMNS.get(str(group).lower(), [])
        }
    )
    health = _read_latest_health()
    snapshot = _read_latest_snapshot()
    health_updated = _ts(health.get("updated_at"))
    snapshot_updated = _ts(snapshot.get("created_at"))
    now_kst = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    health_age_sec = float((now_kst - health_updated).total_seconds()) if health_updated is not None else None
    snapshot_age_sec = float((now_kst - snapshot_updated).total_seconds()) if snapshot_updated is not None else None
    snapshot_values = dict(snapshot.get("values") or {})
    snapshot_v27_contract = _required_values_contract(snapshot_values, v27_cols) if snapshot_values else {}
    snapshot_teacher_contract = _required_values_contract(snapshot_values, teacher_cols) if snapshot_values and teacher_cols else {}
    snapshot_ai_contract = _required_values_contract(snapshot_values, ai_cols) if snapshot_values else {}

    backtest_contracts = {
        "eval_csv": str(v31.DEFAULT_EVAL),
        "eval_csv_stats": _frame_stats(eval_raw),
        "eval_after_regime_stats": _frame_stats(eval_df),
        "parent_feature_contract": _required_contract(parent_feature_df, parent_cols),
        "v27_sequence_contract": _required_contract(eval_df, v27_cols),
        "teacher_sequence_contract": _required_contract(teacher_feature_df, teacher_cols),
        "ai_feature_contract": _required_contract(eval_df, ai_cols),
        "execution_contract": {
            "backtest_limit_cfg": liveguard._cfg().__dict__,
            "fill_contract": "optimistic_immediate_limit_touch0_on_5m_ohlc_proxy",
            "bar_contract": "signal_i_fills_next_bar_open_or_limit_proxy",
        },
    }

    live_contracts = {
        "health_path": str(HEALTH_PATH),
        "health_age_sec": health_age_sec,
        "health_status": health.get("status"),
        "health_warnings": health.get("warnings", []),
        "bar_contract": health.get("bar_contract"),
        "raw_eth": health.get("raw_eth", {}),
        "processed": health.get("processed", {}),
        "quality": health.get("quality", {}),
        "ai": health.get("ai", {}),
        "v31": health.get("v31", {}),
        "alpha2_1": health.get("alpha2_1", {}),
        "regime": health.get("regime", {}),
        "decision": health.get("decision", {}),
        "decision_feature_snapshot": {
            "path": str(SNAPSHOT_PATH),
            "exists": bool(snapshot),
            "age_sec": snapshot_age_sec,
            "timestamp": snapshot.get("timestamp"),
            "row_count": snapshot.get("row_count"),
            "column_count": snapshot.get("column_count"),
            "feature_hash_sha256": snapshot.get("feature_hash_sha256"),
            "decision": snapshot.get("decision", {}),
            "health_summary_status": (snapshot.get("health_summary") or {}).get("status"),
            "v27_sequence_contract": snapshot_v27_contract,
            "teacher_sequence_contract": snapshot_teacher_contract,
            "ai_feature_contract": snapshot_ai_contract,
        },
        "recent_log_flags": _recent_log_flags(),
        "health_tail": _health_tail(),
    }

    parity_blocks: list[str] = []
    parity_warnings: list[str] = []
    if backtest_contracts["parent_feature_contract"]["missing_count"] or backtest_contracts["v27_sequence_contract"]["missing_count"]:
        parity_blocks.append("backtest_artifact_missing_required_model_features")
    if health.get("status") != "OK":
        parity_blocks.append("live_pipeline_health_not_ok")
    if (health.get("ai") or {}).get("missing_count", 1) or (health.get("ai") or {}).get("nonfinite_count", 1):
        parity_blocks.append("live_ai_features_missing_or_nonfinite")
    if (health.get("v31") or {}).get("missing_seq_count", 1) or (health.get("v31") or {}).get("nonfinite_seq_count", 1):
        parity_blocks.append("live_v31_sequence_features_missing_or_nonfinite")
    if health_age_sec is None or health_age_sec > 900.0:
        parity_blocks.append("live_health_stale_or_missing")
    if not snapshot:
        parity_blocks.append("live_decision_feature_snapshot_missing")
    elif snapshot_age_sec is None or snapshot_age_sec > 900.0:
        parity_blocks.append("live_decision_feature_snapshot_stale")
    else:
        if snapshot_v27_contract.get("missing_count", 1) or snapshot_v27_contract.get("last_nonfinite_count", 1):
            parity_blocks.append("live_snapshot_v31_sequence_features_missing_or_nonfinite")
        if snapshot_ai_contract.get("missing_count", 1) or snapshot_ai_contract.get("last_nonfinite_count", 1):
            parity_blocks.append("live_snapshot_ai_features_missing_or_nonfinite")
    if str(health.get("bar_contract")) != "signal_close_next_open":
        parity_warnings.append("live_bar_contract_differs_from_expected_signal_close_next_open")
    if float((health.get("quality") or {}).get("tail_nan_ratio", 0.0) or 0.0) > 0.25:
        parity_warnings.append("live_processed_tail_has_high_nan_ratio_even_if_last_row_is_clean")
    if str((health.get("decision") or {}).get("position_reason", "")) == "alpha3_redteam_blocked":
        parity_warnings.append("latest_health_was_written_before_optimistic_execution_restart_cycle")

    structural_mismatches = [
        "Backtest reads a precomputed Jan-Feb 2026 CSV with AI feature columns already materialized; live recomputes AI features every cycle from the current 1200-bar frame.",
        "Backtest evaluates a fixed historical artifact window; live is May 2026 current market and cannot be numerically equal without replaying the exact same historical bars through trading_bot.py.",
        "Backtest fills are simulated inside eval scripts; live journal uses scheduled next-open shadow accounting plus optional exchange/dry-run paths.",
        "The live feature snapshot now persists the final prepared decision row, but numerical parity still requires replaying identical bars and model artifacts through the same runtime path.",
    ]

    verdict = "schema_parity_pass_value_parity_not_provable" if not parity_blocks else "blocked"
    report = {
        "model_id": MODEL_ID,
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "verdict": verdict,
        "blocking": parity_blocks,
        "warnings": parity_warnings,
        "backtest_contracts": backtest_contracts,
        "live_contracts": live_contracts,
        "structural_mismatches": structural_mismatches,
        "required_fix_for_true_parity": [
            "Use the persisted live decision feature snapshot as the seed for a replay parity harness.",
            "Build a replay harness that feeds the same OHLCV rows into trading_bot.FinalGovernorRuntime._prepare_frame and compares model logits/actions against the backtest evaluator at the same timestamps.",
            "Move backtest to that shared runtime path instead of a separate eval script path, then keep eval scripts as wrappers only.",
        ],
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default_safe), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "verdict": verdict, "blocking": parity_blocks, "warnings": parity_warnings}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
