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

import repair_omega1_2_1_tp_runner_clean_baseline_20260613 as clean_repair  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402


MODEL_ID = "omega1_2_1_true_leverage_baseline_redteam_audit_20260613"
BASELINE_ID = "omega1_2_1_true_leverage_price_barrier_scale200_cap090"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DIAG_DIR = ROOT / "tmp/causal_regen_20260516" / "omega1_2_1_true_leverage_diagnostic_20260610"
MANIFEST_PATH = ROOT / "data/ensemble/supervised/omega1_2_1_true_leverage_price_barrier_scale200_cap090/baseline_manifest.json"


FORBIDDEN_PREFIXES = (
    "teacher_",
    "teacher_oof_",
    "regime4_pred_",
    "clean_regime4_",
    "clean_regime_2024_unsup_v4_",
)
FORBIDDEN_NAMES = {"tp_sl_action_score"}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _forbidden_columns(cols: list[str]) -> list[str]:
    return [
        str(c)
        for c in cols
        if str(c) in FORBIDDEN_NAMES or any(str(c).startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    ]


def _feature_audit() -> dict[str, Any]:
    data = clean_repair.legacy_runner._build()
    out: dict[str, Any] = {}
    for split, payload in data.items():
        frame_cols = list(payload["frame"].columns)
        dec_cols = list(payload["dec"].columns)
        state_cols = list(payload["state"].columns)
        out[split] = {
            "frame_forbidden": _forbidden_columns(frame_cols),
            "decision_forbidden": _forbidden_columns(dec_cols),
            "state_forbidden": _forbidden_columns(state_cols),
            "frame_cols": len(frame_cols),
            "decision_cols": len(dec_cols),
            "state_cols": len(state_cols),
            "rows": int(len(payload["frame"])),
            "active_rows": int(np.asarray(base.omega._active(payload["dec"]), dtype=bool).sum()),
            "side_counts": {
                str(k): int(v)
                for k, v in payload["dec"].loc[np.asarray(base.omega._active(payload["dec"]), dtype=bool), "side"].value_counts().to_dict().items()
            },
            "expert_counts": {
                str(k): int(v)
                for k, v in payload["dec"].loc[np.asarray(base.omega._active(payload["dec"]), dtype=bool), "router_expert"].astype(str).value_counts().to_dict().items()
            },
        }
    return out


def _touch_audit(ledger: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"trades": 0, "earlier_touch_count": 0, "different_reason_count": 0}
    arrays = clean_repair._arrays(frame)
    earlier_touch = 0
    different_reason = 0
    deltas: list[int] = []
    for row in ledger.itertuples(index=False):
        side = 1 if str(row.side).upper() == "LONG" else -1
        pos = clean_repair.CleanPosition(
            side=side,
            entry_signal_i=int(row.entry_i) - 1,
            entry_i=int(row.entry_i),
            entry_price=float(row.entry_price),
            notional=float(row.effective_exposure),
            take_profit=float(row.tp_equity_ret),
            stop_loss=abs(float(row.sl_equity_ret)),
            floor_unreal=-abs(float(row.sl_equity_ret)),
        )
        first_i = int(row.entry_i)
        exit_i = int(row.exit_i)
        found_i = exit_i
        found_reason = str(row.exit_reason)
        for i in range(first_i, exit_i + 1):
            best, worst = clean_repair._bar_best_worst(arrays, pos, i, 0.0)
            if worst <= -abs(pos.stop_loss):
                found_i = i
                found_reason = "stop_loss"
                break
            if best >= pos.take_profit:
                found_i = i
                found_reason = "take_profit"
                break
        if found_i < exit_i:
            earlier_touch += 1
            deltas.append(exit_i - found_i)
        if found_reason != str(row.exit_reason):
            different_reason += 1
    return {
        "trades": int(len(ledger)),
        "earlier_touch_count": int(earlier_touch),
        "different_reason_count": int(different_reason),
        "median_early_bars": float(np.median(deltas)) if deltas else 0.0,
        "max_early_bars": int(max(deltas)) if deltas else 0,
    }


def _metric_from_clean(split_payload: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = clean_repair.RunnerConfig(0, "baseline", 0.0, 1.0, 0.0, 0)
    return clean_repair._simulate_clean(split_payload, cfg)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = clean_repair.legacy_runner._build()
    original = {
        "validation": _read_json(DIAG_DIR / "validation_true_leverage_rerun_preserve_price_barrier_metrics.json"),
        "oos": _read_json(DIAG_DIR / "oos_true_leverage_rerun_preserve_price_barrier_metrics.json"),
    }
    same_equity = {
        "validation": _read_json(DIAG_DIR / "validation_true_leverage_rerun_same_equity_tp_sl_metrics.json"),
        "oos": _read_json(DIAG_DIR / "oos_true_leverage_rerun_same_equity_tp_sl_metrics.json"),
    }
    clean_metrics: dict[str, Any] = {}
    touch: dict[str, Any] = {}
    clean_ledgers: dict[str, str] = {}
    for split, payload in data.items():
        metrics, ledger = _metric_from_clean(payload)
        clean_metrics[split] = metrics
        out_path = OUT_DIR / f"{split}_clean_intrabar_taker_ledger.csv"
        ledger.to_csv(out_path, index=False)
        clean_ledgers[split] = str(out_path)
        orig_ledger = pd.read_csv(DIAG_DIR / f"{split}_true_leverage_rerun_preserve_price_barrier_ledger.csv")
        touch[split] = _touch_audit(orig_ledger, payload["frame"])

    manifest = _read_json(MANIFEST_PATH)
    oos_selection_risk = {
        "status": "needs_caution",
        "reason": "risk transform was a manual research choice documented with OOS diagnostic comparisons; no separate untouched post-selection test exists",
        "observed_modes": ["same_equity_tp_sl", "preserve_price_barrier"],
    }
    report = {
        "audit_id": MODEL_ID,
        "baseline_id": BASELINE_ID,
        "verdict": "research_candidate_not_clean_untouched_oos",
        "summary": {
            "direct_forbidden_feature_leak": False,
            "accounting_runtime_equivalence": "partial_fail_original_replay_close_maker_sensitive",
            "clean_replay_available": True,
            "oos_selection_risk": "medium_high",
        },
        "original_reported_preserve_price_barrier": original,
        "clean_intrabar_taker_replay": clean_metrics,
        "failed_same_equity_tp_sl_diagnostic": same_equity,
        "intrabar_touch_audit_on_original_ledgers": touch,
        "feature_audit": _feature_audit(),
        "oos_selection_audit": oos_selection_risk,
        "manifest_snapshot": {
            "status": manifest.get("status"),
            "accounting": (manifest.get("decision_boundary") or {}).get("accounting"),
            "selected_runtime_template": manifest.get("selected_runtime_template"),
        },
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "clean_ledgers": clean_ledgers,
            "source_manifest": str(MANIFEST_PATH),
            "source_diagnostic_dir": str(DIAG_DIR),
        },
        "recommendation": [
            "Do not cite +186.43% as clean untouched OOS.",
            "Use clean intrabar-taker replay numbers for conservative comparison.",
            "If promoted, require fresh forward shadow or later untouched test period.",
        ],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(OUT_DIR / "report.json"),
        "verdict": report["verdict"],
        "original_oos": original["oos"],
        "clean_oos": clean_metrics["oos"],
        "touch_oos": touch["oos"],
    }, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
