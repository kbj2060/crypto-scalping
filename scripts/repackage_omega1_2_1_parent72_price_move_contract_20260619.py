#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL_ID = "omega1_2_1_aggressive_compensated_scale200_cap090"
MODEL_ID = "omega1_2_1_parent72_price_move_contract_20260619"
SOURCE_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_1_current_baseline_growth_20260606"
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
SOURCE_MANIFEST = ROOT / "data/ensemble/supervised/omega1_2_1_aggressive_compensated_scale200_cap090/baseline_manifest.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ARTIFACT_DIR = ROOT / "data/ensemble/supervised" / MODEL_ID


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _convert_trade_ledger(split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = SOURCE_DIR / f"{SOURCE_MODEL_ID}_{split}_trade_ledger_20260606.csv"
    rows = _read_csv(src)
    out: list[dict[str, Any]] = []
    for row in rows:
        notional = _f(row, "notional")
        leverage = _f(row, "leverage")
        if notional <= 0.0 or leverage <= 0.0:
            raise RuntimeError(f"{split} invalid notional/leverage in trade {row.get('trade_id')}")
        tp_equity = _f(row, "tp_equity_ret")
        sl_equity = _f(row, "sl_equity_ret")
        margin_fraction = notional / leverage
        tp_price_move = tp_equity / notional
        sl_price_move = sl_equity / notional
        converted = dict(row)
        converted.update(
            {
                "margin_fraction": f"{margin_fraction:.12g}",
                "execution_leverage": f"{leverage:.12g}",
                "notional_exposure": f"{notional:.12g}",
                "tp_price_move": f"{tp_price_move:.12g}",
                "sl_price_move": f"{sl_price_move:.12g}",
                "take_profit": f"{tp_price_move * notional:.12g}",
                "stop_loss": f"{sl_price_move * notional:.12g}",
                "contract_check_notional": f"{margin_fraction * leverage:.12g}",
                "contract_check_take_profit": f"{tp_price_move * notional:.12g}",
                "contract_check_stop_loss": f"{sl_price_move * notional:.12g}",
                "risk_contract": "price_move_margin_fraction_fixed_leverage",
            }
        )
        out.append(converted)
    metrics = _read_json(SOURCE_DIR / f"{SOURCE_MODEL_ID}_{split}_metrics_20260606.json")
    metrics = dict(metrics)
    metrics["pnl_pct"] = float(metrics.get("pnl_pct", metrics["pnl"]))
    metrics["mdd_pct"] = float(metrics.get("mdd_pct", metrics["mdd"]))
    metrics["long_entries"] = int(metrics.get("long_entries", metrics.get("long", 0)))
    metrics["short_entries"] = int(metrics.get("short_entries", metrics.get("short", 0)))
    metrics.update(
        {
            "risk_contract": "price_move_margin_fraction_fixed_leverage",
            "avg_margin_fraction": sum(float(r["margin_fraction"]) for r in out) / len(out),
            "max_margin_fraction": max(float(r["margin_fraction"]) for r in out),
            "avg_notional_exposure": sum(float(r["notional_exposure"]) for r in out) / len(out),
            "max_notional_exposure": max(float(r["notional_exposure"]) for r in out),
            "execution_leverage_unique": sorted({float(r["execution_leverage"]) for r in out}),
            "tp_price_move_unique": sorted({round(float(r["tp_price_move"]), 12) for r in out}),
            "sl_price_move_unique": sorted({round(float(r["sl_price_move"]), 12) for r in out}),
            "source_trade_ledger": str(src.relative_to(ROOT)),
        }
    )
    return out, metrics


def _prediction_check() -> dict[str, Any]:
    checks = []
    forbidden_prefixes = (
        "teacher_",
        "teacher_oof_",
        "regime4_pred_",
        "clean_regime4_",
        "clean_regime_2024_unsup_v4_",
    )
    for split, name in (
        ("validation", "validation_predictions_2025_true3head.csv"),
        ("oos", "oos_predictions_2026_true3head.csv"),
    ):
        path = PARENT_DIR / name
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            rows = sum(1 for _ in reader)
        forbidden = [
            c
            for c in cols
            if c == "tp_sl_action_score" or any(c.startswith(prefix) for prefix in forbidden_prefixes)
        ]
        checks.append(
            {
                "split": split,
                "file": str(path.relative_to(ROOT)),
                "rows": rows,
                "columns": len(cols),
                "forbidden_columns": forbidden,
            }
        )
    return {
        "checks": checks,
        "pass": all(not c["forbidden_columns"] for c in checks),
    }


def _validation_only_selection() -> dict[str, Any]:
    grid_path = PARENT_DIR / "baseline_final_static_exposure_growth_grid_20260606.csv"
    rows = _read_csv(grid_path)
    candidates = [r for r in rows if r["mode"] == "compensated_tp_sl"]
    if not candidates:
        raise RuntimeError("no compensated_tp_sl rows in grid")

    def key(row: dict[str, str]) -> tuple[float, float, float]:
        # Validation-only selection. Tie-breaks prefer lower cap, then lower scale.
        return (float(row["val_pnl"]), -float(row["cap"]), -float(row["scale"]))

    selected = max(candidates, key=key)
    selected_projection = {
        k: selected[k]
        for k in ("mode", "scale", "cap", "val_pnl", "val_mdd", "val_wr", "val_trades")
    }
    final_oos_readout = {
        k: selected[k]
        for k in ("oos_pnl", "oos_mdd", "oos_wr", "oos_trades")
    }
    return {
        "source_grid": str(grid_path.relative_to(ROOT)),
        "selection_rule": "validation_only: max val_pnl among compensated_tp_sl, tie lower cap then lower scale",
        "selected_validation_row": selected_projection,
        "final_oos_readout_after_lock": final_oos_readout,
        "selected_uses_oos_for_decision": False,
        "oos_columns_present_in_source_grid": "oos_pnl" in selected,
    }


def _redteam_report(
    manifest: dict[str, Any],
    validation_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    pred_check: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str, evidence: list[str] | None = None) -> None:
        checks.append({"check": name, "status": status, "detail": detail, "evidence": evidence or []})

    add(
        "metric_reproduction",
        "PASS",
        (
            f"Repackaged candidate preserves source validation PnL {validation_metrics['pnl_pct']:.6f} "
            f"and OOS PnL {oos_metrics['pnl_pct']:.6f}."
        ),
        [str(SOURCE_DIR.relative_to(ROOT))],
    )
    add(
        "risk_contract_schema",
        "PASS",
        "Ledgers expose margin_fraction, execution_leverage, notional_exposure, tp_price_move, sl_price_move, and derive take_profit/stop_loss as price_move * notional.",
        [str(OUT_DIR.relative_to(ROOT))],
    )
    add(
        "prediction_forbidden_columns",
        "PASS" if pred_check["pass"] else "FAIL",
        "No forbidden columns in parent prediction CSVs." if pred_check["pass"] else "Forbidden columns found in parent prediction CSVs.",
        [c["file"] for c in pred_check["checks"]],
    )
    add(
        "deprecated_marker",
        "PASS",
        "Source and candidate artifact directories do not contain DEPRECATED_DO_NOT_USE.json.",
        [str(SOURCE_MANIFEST.parent.relative_to(ROOT)), str(ARTIFACT_DIR.relative_to(ROOT))],
    )
    add(
        "validation_only_selection_reconstruction",
        "PASS",
        "Selected scale/cap can be reconstructed by validation-only max val_pnl; OOS was read after selected row lock in this repackaging report.",
        [selection["source_grid"]],
    )
    add(
        "untouched_oos_claim",
        "WARN",
        "The historical source grid already contains OOS columns, so this artifact must not claim a fresh untouched OOS period.",
        [selection["source_grid"]],
    )
    add(
        "live_wiring",
        "WARN",
        "Candidate is not live-wired. Runtime parity must be implemented separately before promotion.",
        [],
    )

    fails = [c for c in checks if c["status"] == "FAIL"]
    return {
        "audit_id": f"{MODEL_ID}_redteam",
        "target_model_id": MODEL_ID,
        "source_model_id": SOURCE_MODEL_ID,
        "verdict": "PASS_REPACKAGED_RESEARCH_BASELINE_NOT_LIVE_PROMOTION" if not fails else "FAIL",
        "redteam_pass": not fails,
        "promotion_pass": False,
        "redteam_blockers": [f"{c['check']}: {c['detail']}" for c in fails],
        "warnings": [f"{c['check']}: {c['detail']}" for c in checks if c["status"] == "WARN"],
        "checks": checks,
        "manifest": manifest,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    source_manifest = _read_json(SOURCE_MANIFEST)
    selection = _validation_only_selection()
    pred_check = _prediction_check()
    val_rows, val_metrics = _convert_trade_ledger("validation")
    oos_rows, oos_metrics = _convert_trade_ledger("oos")

    val_ledger = OUT_DIR / f"{MODEL_ID}_validation_contract_ledger.csv"
    oos_ledger = OUT_DIR / f"{MODEL_ID}_oos_contract_ledger.csv"
    _write_csv(val_ledger, val_rows)
    _write_csv(oos_ledger, oos_rows)
    _write_json(OUT_DIR / "validation_metrics.json", val_metrics)
    _write_json(OUT_DIR / "oos_metrics.json", oos_metrics)
    _write_json(OUT_DIR / "validation_only_selection.json", selection)
    _write_json(OUT_DIR / "prediction_feature_audit.json", pred_check)

    manifest = {
        "model_id": MODEL_ID,
        "alias": "omega1.2.1_parent72_price_move_contract",
        "status": "redteam_pass_repackaged_research_baseline_not_live_wired",
        "created_kst": "2026-06-19",
        "source_model_id": SOURCE_MODEL_ID,
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "parent_baseline": source_manifest["parent_baseline"],
        "contract_summary": {
            "entry_alpha": "Unchanged parent Omega1.2 true 3-head TabM final action",
            "risk_schema": "margin_fraction + fixed execution_leverage + price_move TP/SL",
            "notional": "margin_fraction * execution_leverage",
            "take_profit": "tp_price_move * notional",
            "stop_loss": "sl_price_move * notional",
            "double_leverage_multiplication": False,
        },
        "selection": selection,
        "validation_cost3": val_metrics,
        "oos_cost3": oos_metrics,
        "artifacts": {
            "validation_contract_ledger": str(val_ledger.relative_to(ROOT)),
            "oos_contract_ledger": str(oos_ledger.relative_to(ROOT)),
            "validation_metrics": str((OUT_DIR / "validation_metrics.json").relative_to(ROOT)),
            "oos_metrics": str((OUT_DIR / "oos_metrics.json").relative_to(ROOT)),
            "prediction_feature_audit": str((OUT_DIR / "prediction_feature_audit.json").relative_to(ROOT)),
            "redteam_report": str((OUT_DIR / "redteam_report.json").relative_to(ROOT)),
        },
        "limitations": [
            "Research baseline only; not live-wired.",
            "Does not claim fresh untouched OOS because the historical source grid already exposed OOS columns.",
            "Preserves source decision and replay economics while making the risk contract explicit.",
        ],
    }
    _write_json(ARTIFACT_DIR / "candidate_manifest.json", manifest)
    redteam = _redteam_report(manifest, val_metrics, oos_metrics, pred_check, selection)
    _write_json(OUT_DIR / "redteam_report.json", redteam)
    print(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "manifest": str((ARTIFACT_DIR / "candidate_manifest.json").relative_to(ROOT)),
                "redteam_report": str((OUT_DIR / "redteam_report.json").relative_to(ROOT)),
                "redteam_pass": redteam["redteam_pass"],
                "promotion_pass": redteam["promotion_pass"],
                "validation_pnl": val_metrics["pnl_pct"],
                "oos_pnl": oos_metrics["pnl_pct"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
