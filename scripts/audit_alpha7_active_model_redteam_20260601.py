#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default
from scripts.eval_alpha7_regime3_current_moe_expert_source_mix_20260601 import SOURCES
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import RISK_COLS, _load_frames_with_risk
from scripts.train_alpha7_regime3_expert_moe_20260601 import EXPERT_NAMES, ROUTERS
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import DERIVABLE_FEATURES


MODEL_ID = "alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"
ACTIVE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"
SOURCE_MIX_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_expert_source_mix_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_redteam_audit_20260601"
DOC_PATH = ROOT / "docs/audits/alpha7_active_redteam_audit_20260601.md"

EXPECTED_CANDIDATE = "bull0.85_bear1.15_chop1.25"
EXPECTED_SOURCE_MIX = "bull_practical__bear_risk__chop_practical__conf0.80"
EXPECTED_SCALES = {"bull": 0.85, "bear": 1.15, "chop_expert": 1.25}
SELECTED_SOURCES = {"bull": "practical", "bear": "risk", "chop": "practical"}
ROUTER_NAME = "regime3_current_context"

HARD_FORBIDDEN_PREFIXES = (
    "teacher_",
    "a5dir_",
    "regime3_pred_",
)
HARD_FORBIDDEN_TOKENS = (
    "label",
    "target",
    "future",
    "realized",
    "pnl",
    "wave3",
    "zigzag",
)
WATCH_PREFIXES = (
    "clean_regime4_",
    "regime4_pred_",
)
WATCH_TOKENS = (
    "action_score",
    "tp_sl_action_score",
)


def _issue(severity: str, title: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"severity": severity, "title": title, "detail": detail, **extra}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_decisions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False).reset_index(drop=True)


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    return (action != 0) & (side != 0)


def _close(a: float, b: float, tol: float = 1e-8) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _compare_costs(name: str, recomputed: dict[str, Any], reported: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for cost_name in ("cost1", "cost2", "cost3"):
        for key in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional"):
            rv = recomputed[cost_name][key]
            pv = reported[cost_name][key]
            if key == "trades":
                if int(rv) != int(pv):
                    issues.append(_issue("P0", f"{name} {cost_name}.{key} mismatch", f"reported={pv}, recomputed={rv}"))
            elif not _close(float(rv), float(pv), tol=1e-7):
                issues.append(_issue("P0", f"{name} {cost_name}.{key} mismatch", f"reported={pv}, recomputed={rv}"))
    return issues


def _audit_report_consistency(report: dict[str, Any], source_report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    selected = report["selected"]
    if selected.get("candidate") != EXPECTED_CANDIDATE:
        issues.append(_issue("P0", "unexpected active candidate", f"expected {EXPECTED_CANDIDATE}, got {selected.get('candidate')}"))
    for expert, expected in [("bull", 0.85), ("bear", 1.15), ("chop", 1.25)]:
        key = f"{expert}_scale"
        if not _close(float(selected.get(key)), expected):
            issues.append(_issue("P0", f"unexpected {key}", f"expected {expected}, got {selected.get(key)}"))
    source_selected = source_report["selected"]
    if source_selected.get("candidate") != EXPECTED_SOURCE_MIX:
        issues.append(_issue("P0", "unexpected source-mix candidate", f"expected {EXPECTED_SOURCE_MIX}, got {source_selected.get('candidate')}"))
    for k, v in {"bull_source": "practical", "bear_source": "risk", "chop_source": "practical", "min_conf": 0.80}.items():
        got = source_selected.get(k)
        if isinstance(v, float):
            ok = _close(float(got), v)
        else:
            ok = got == v
        if not ok:
            issues.append(_issue("P0", f"unexpected source-mix {k}", f"expected {v}, got {got}"))
    return issues


def _audit_scale_application(base: pd.DataFrame, active: pd.DataFrame, tag: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if len(base) != len(active):
        return [_issue("P0", f"{tag} row count mismatch", f"base={len(base)} active={len(active)}")]
    active_mask = _active(base)
    required = ["router_expert", "notional_exposure", "position_fraction"]
    missing = [c for c in required if c not in base.columns or c not in active.columns]
    if missing:
        return [_issue("P0", f"{tag} missing scale columns", str(missing))]
    for expert, scale in EXPECTED_SCALES.items():
        mask = active_mask & base["router_expert"].astype(str).eq(expert).to_numpy()
        for col in ("notional_exposure", "position_fraction"):
            expected = pd.to_numeric(base.loc[mask, col], errors="raise").to_numpy(dtype=np.float64) * scale
            got = pd.to_numeric(active.loc[mask, col], errors="raise").to_numpy(dtype=np.float64)
            if len(got) and not np.allclose(got, expected, rtol=1e-10, atol=1e-10):
                diff = float(np.nanmax(np.abs(got - expected)))
                issues.append(_issue("P0", f"{tag} {expert} {col} scale mismatch", f"max_abs_diff={diff}"))
    unchanged_cols = [
        c for c in base.columns
        if c in active.columns and c not in {"notional_exposure", "position_fraction"}
    ]
    for col in unchanged_cols:
        left = base[col]
        right = active[col]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64)
            b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64)
            same = np.allclose(a, b, equal_nan=True, rtol=1e-10, atol=1e-10)
        else:
            same = left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str))
        if not same:
            issues.append(_issue("P0", f"{tag} unexpected changed column", col))
            break
    return issues


def _audit_decision_sanity(dec: pd.DataFrame, tag: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    numeric_cols = [
        "action",
        "side",
        "notional_exposure",
        "position_fraction",
        "leverage",
        "take_profit",
        "stop_loss",
        "max_hold_bars",
        "cooldown_bars",
        "quality_score",
        "confidence",
        "router_confidence",
    ]
    for col in numeric_cols:
        if col not in dec.columns:
            issues.append(_issue("P0", f"{tag} missing decision column", col))
            continue
        values = pd.to_numeric(dec[col], errors="coerce").to_numpy(dtype=np.float64)
        bad = ~np.isfinite(values)
        if bool(bad.any()):
            issues.append(_issue("P0", f"{tag} non-finite decision values", f"{col}: {int(bad.sum())} rows"))
    active = _active(dec)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    router = dec["router_expert"].astype(str).to_numpy()
    if bool((active & (router == "bull") & (side < 0)).any()):
        issues.append(_issue("P0", f"{tag} bull expert emitted short", str(int((active & (router == "bull") & (side < 0)).sum()))))
    if bool((active & (router == "bear") & (side > 0)).any()):
        issues.append(_issue("P0", f"{tag} bear expert emitted long", str(int((active & (router == "bear") & (side > 0)).sum()))))
    cash = ~active
    cash_side_bad = cash & (side != 0)
    if bool(cash_side_bad.any()):
        issues.append(_issue("P0", f"{tag} cash/action side inconsistency", str(int(cash_side_bad.sum()))))
    stats = {
        "rows": int(len(dec)),
        "active_rows": int(active.sum()),
        "policy_counts": {str(k): int(v) for k, v in dec["router_expert"].value_counts().to_dict().items()},
        "max_notional_exposure": float(pd.to_numeric(dec["notional_exposure"], errors="raise").max()),
        "max_position_fraction": float(pd.to_numeric(dec["position_fraction"], errors="raise").max()),
        "max_leverage": float(pd.to_numeric(dec["leverage"], errors="raise").max()),
    }
    return issues, stats


def _audit_router_frames(train_all: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    router_cols = ROUTERS[ROUTER_NAME]["cols"]
    extra_cols = ROUTERS[ROUTER_NAME]["extra_cols"]
    stats: dict[str, Any] = {}
    for tag, df in [("train", train_all), ("eval", eval_df)]:
        block = df[router_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(block).all():
            issues.append(_issue("P0", f"{tag} router non-finite values", "router probability block contains non-finite values"))
        if bool(((block < -1e-9) | (block > 1.0 + 1e-9)).any()):
            issues.append(_issue("P0", f"{tag} router probability out of range", "expected all values in [0, 1]"))
        sums = block.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-5, rtol=1e-5):
            issues.append(_issue("P2", f"{tag} router probabilities do not sum to 1", f"max_abs_diff={float(np.max(np.abs(sums - 1.0))):.6g}"))
        for col in [*extra_cols, *RISK_COLS]:
            values = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                issues.append(_issue("P0", f"{tag} overlay non-finite values", col))
        stats[tag] = {
            "rows": int(len(df)),
            "router_prob_sum_max_abs_diff": float(np.max(np.abs(sums - 1.0))),
            "router_conf_min": float(block.max(axis=1).min()),
            "router_conf_mean": float(block.max(axis=1).mean()),
            "router_conf_max": float(block.max(axis=1).max()),
        }
    return issues, stats


def _selected_artifact_paths() -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {}
    for expert in EXPERT_NAMES:
        source = SELECTED_SOURCES[expert]
        root = SOURCES[source] / expert
        paths[f"{expert}_{source}"] = [
            root / "primary_no_tp/parent.pkl",
            root / "fallback_v2_tp/parent.pkl",
            root / "primary_no_tp/summary.json",
            root / "fallback_v2_tp/summary.json",
        ]
    return paths


def _audit_feature_contract(train_all: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for name, paths in _selected_artifact_paths().items():
        for path in paths:
            if not path.exists():
                issues.append(_issue("P0", "missing selected expert artifact", str(path)))
        for parent_path in [p for p in paths if p.name == "parent.pkl"]:
            if not parent_path.exists():
                continue
            model = joblib.load(parent_path)
            cols = list(model.get("feature_cols", []))
            missing_train = [c for c in cols if c not in train_all.columns and c not in DERIVABLE_FEATURES]
            missing_eval = [c for c in cols if c not in eval_df.columns and c not in DERIVABLE_FEATURES]
            if missing_train or missing_eval:
                issues.append(_issue("P0", f"{name} feature contract missing columns", f"train={missing_train[:20]}, eval={missing_eval[:20]}", artifact=str(parent_path)))
            forbidden = [
                c for c in cols
                if c.startswith(HARD_FORBIDDEN_PREFIXES)
                or any(tok in c.lower() for tok in HARD_FORBIDDEN_TOKENS)
            ]
            watched = [
                c for c in cols
                if c.startswith(WATCH_PREFIXES)
                or any(tok in c.lower() for tok in WATCH_TOKENS)
            ]
            if forbidden:
                issues.append(_issue("P1", f"{name} feature contract contains forbidden-looking columns", ", ".join(forbidden[:40]), artifact=str(parent_path)))
            if watched:
                issues.append(_issue("P2", f"{name} feature contract contains watched legacy/target-adjacent columns", ", ".join(watched[:40]), artifact=str(parent_path)))
            stats[str(parent_path)] = {
                "feature_count": int(len(cols)),
                "forbidden_count": int(len(forbidden)),
                "watched_count": int(len(watched)),
                "watched_cols": watched,
                "first_30_features": cols[:30],
            }
    return issues, stats


def _audit_selection_process(report: dict[str, Any], ranking: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    selected = report["selected"]["candidate"]
    if ranking.empty:
        return [_issue("P0", "empty active ranking", str(ACTIVE_DIR / "ranking.csv"))], {}
    if str(ranking.iloc[0]["candidate"]) != selected:
        issues.append(_issue("P0", "ranking top does not match selected report", f"ranking={ranking.iloc[0]['candidate']}, report={selected}"))
    val_score_rank = int(ranking.index[ranking["candidate"].astype(str).eq(selected)][0]) + 1
    oos_ranked = ranking.sort_values("oos_cost3_pnl", ascending=False).reset_index(drop=True)
    oos_pnl_rank = int(oos_ranked.index[oos_ranked["candidate"].astype(str).eq(selected)][0]) + 1
    issues.append(_issue(
        "P2",
        "OOS metrics are materialized for every grid row",
        "The script sorts by validation score, but ranking.csv/report.json expose OOS for all scale candidates; this is a human process overfit risk, not a direct code selection leak.",
        selected_validation_rank=val_score_rank,
        selected_oos_pnl_rank=oos_pnl_rank,
        grid_rows=int(len(ranking)),
    ))
    return issues, {
        "selected_validation_rank": val_score_rank,
        "selected_oos_cost3_pnl_rank": oos_pnl_rank,
        "grid_rows": int(len(ranking)),
    }


def _write_markdown(report: dict[str, Any]) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    issue_lines = []
    for issue in report["issues"]:
        issue_lines.append(f"- **{issue['severity']} {issue['title']}**: {issue['detail']}")
    if not issue_lines:
        issue_lines.append("- No issues.")
    md = f"""# Alpha7 Active Model Red-Team Audit 2026-06-01

## Scope

- Active model: `{MODEL_ID}`
- Active candidate: `{EXPECTED_CANDIDATE}`
- Source mix: `{EXPECTED_SOURCE_MIX}`

## Verdict

{report['verdict']}

## Findings

{chr(10).join(issue_lines)}

## Recomputed Metrics

- Validation Cost3 PnL/MDD/trades/WR: `{report['metrics_recomputed']['validation']['cost3']['pnl']:.6f}` / `{report['metrics_recomputed']['validation']['cost3']['mdd']:.6f}` / `{report['metrics_recomputed']['validation']['cost3']['trades']}` / `{report['metrics_recomputed']['validation']['cost3']['wr']:.6f}`
- OOS Cost3 PnL/MDD/trades/WR: `{report['metrics_recomputed']['oos']['cost3']['pnl']:.6f}` / `{report['metrics_recomputed']['oos']['cost3']['mdd']:.6f}` / `{report['metrics_recomputed']['oos']['cost3']['trades']}` / `{report['metrics_recomputed']['oos']['cost3']['wr']:.6f}`

## Artifacts

- JSON report: `{OUT_DIR / 'report.json'}`
- Active report: `{ACTIVE_DIR / 'report.json'}`
- Active decisions: `{ACTIVE_DIR / 'validation_decisions.csv'}`, `{ACTIVE_DIR / 'oos_2026_decisions.csv'}`
"""
    DOC_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = _read_json(ACTIVE_DIR / "report.json")
    source_report = _read_json(SOURCE_MIX_DIR / "report.json")
    ranking = pd.read_csv(ACTIVE_DIR / "ranking.csv")
    val_active = _read_decisions(ACTIVE_DIR / "validation_decisions.csv")
    oos_active = _read_decisions(ACTIVE_DIR / "oos_2026_decisions.csv")
    val_base = _read_decisions(SOURCE_MIX_DIR / "validation_decisions.csv")
    oos_base = _read_decisions(SOURCE_MIX_DIR / "oos_2026_decisions.csv")

    issues: list[dict[str, Any]] = []
    issues.extend(_audit_report_consistency(report, source_report))
    issues.extend(_audit_scale_application(val_base, val_active, "validation"))
    issues.extend(_audit_scale_application(oos_base, oos_active, "oos"))

    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    if len(val_df) != len(val_active):
        issues.append(_issue("P0", "validation frame/decision row mismatch", f"frame={len(val_df)}, decisions={len(val_active)}"))
    if len(eval_df) != len(oos_active):
        issues.append(_issue("P0", "oos frame/decision row mismatch", f"frame={len(eval_df)}, decisions={len(oos_active)}"))

    recomputed_val = _combo_metrics(val_df, val_active)
    recomputed_oos = _combo_metrics(eval_df, oos_active)
    issues.extend(_compare_costs("validation", recomputed_val, report["selected"]["validation"]))
    issues.extend(_compare_costs("oos", recomputed_oos, report["selected"]["oos"]))

    val_sanity_issues, val_sanity = _audit_decision_sanity(val_active, "validation")
    oos_sanity_issues, oos_sanity = _audit_decision_sanity(oos_active, "oos")
    issues.extend(val_sanity_issues)
    issues.extend(oos_sanity_issues)
    router_issues, router_stats = _audit_router_frames(train_all, eval_df)
    issues.extend(router_issues)
    feature_issues, feature_stats = _audit_feature_contract(train_all, eval_df)
    issues.extend(feature_issues)
    selection_issues, selection_stats = _audit_selection_process(report, ranking)
    issues.extend(selection_issues)

    hard = [i for i in issues if i["severity"] in {"P0", "P1"}]
    verdict = "PASS_WITH_WARNINGS: no P0/P1 runtime, metric, scale, or contract break found." if not hard else "FAIL: P0/P1 issues require action before active/live use."
    payload = {
        "model_id": MODEL_ID,
        "verdict": verdict,
        "issues": issues,
        "metrics_recomputed": {"validation": recomputed_val, "oos": recomputed_oos},
        "decision_sanity": {"validation": val_sanity, "oos": oos_sanity},
        "router_stats": router_stats,
        "feature_contract": feature_stats,
        "selection_process": selection_stats,
        "overlay": overlay,
        "artifacts_checked": {
            "active_report": str(ACTIVE_DIR / "report.json"),
            "active_ranking": str(ACTIVE_DIR / "ranking.csv"),
            "source_report": str(SOURCE_MIX_DIR / "report.json"),
            "selected_expert_artifacts": {k: [str(p) for p in v] for k, v in _selected_artifact_paths().items()},
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(payload)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "verdict": verdict, "issue_counts": pd.Series([i["severity"] for i in issues]).value_counts().to_dict()}, ensure_ascii=False, indent=2))
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
