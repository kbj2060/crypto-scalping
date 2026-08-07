#!/usr/bin/env python3
"""Red-team audit for the Omega6 synthesis prototype.

Checks the properties AGENTS.md and this project's established redteam pattern require, scoped
to what's meaningful for Omega6's actual (non-ledger-replay, live bar-by-bar) architecture:

1. Fail-fast: Omega6LiveAdapter raises on a broken artifact path.
2. Forbidden feature prefixes absent from all three bundles' (primary/fallback + L3 feature_cols).
3. Train/validation timestamp non-overlap for L2, L3, and L4 (all must be < SPLIT_TS).
4. Futures Risk Sizing Contract: notional_exposure == margin_notional * leverage on live decisions.
5. L6 governor is reduce-only: shock_haircut/macro_veto can only shrink notional, never grow it.
6. Cost-stress non-collapse: cost3 trade count must not collapse relative to cost1 (this is
   exactly the class of bug found and fixed in scripts/backtest_omega6_synthesis_fresh_forward_20260703.py
   on 2026-07-03 -- an override-orphaning bug silently dropped trades under higher slip_eff).
7. Sizing-sensitivity flag (informational, not pass/fail): compares the MDD-capped vs uncapped
   L4 sidecar's validation PnL sign, surfacing the fragility finding from the 2026-07-03 session
   rather than hiding it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from trading_bot_modules.omega6_live import FORBIDDEN_FEATURE_PREFIXES, Omega6LiveAdapter  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega6_tabm_3head_20260703 as omega6_tabm  # noqa: E402
import backtest_omega6_synthesis_fresh_forward_20260703 as bt  # noqa: E402

OUT_DIR = ROOT / "docs/audits"
REPORT_JSON = OUT_DIR / "omega6_synthesis_redteam_20260703.json"
REPORT_MD = OUT_DIR / "omega6_synthesis_redteam_20260703.md"
LATEST_BACKTEST_REPORT = ROOT / "tmp/causal_regen_20260516/omega6_synthesis_v1_20260703/report.json"
COST_TRADE_RATIO_MIN = 0.90  # cost3 trades must retain >=90% of cost1 trades


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, severity: str, details: dict[str, Any] | None = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "severity": severity, "details": details or {}})


def check_fail_fast(checks: list[dict[str, Any]]) -> None:
    try:
        Omega6LiveAdapter(
            primary_bundle_path="/nonexistent/bundle.pt",
            fallback_bundle_path=str(bt.DEFAULT_FALLBACK_BUNDLE),
            tcn_gate_path=str(bt.DEFAULT_TCN_GATE),
            risk_sidecar_path=str(bt.DEFAULT_RISK_SIDECAR),
            device="cpu",
        )
        add_check(checks, "fail_fast_on_broken_artifact_path", False, "blocker", {"note": "did not raise"})
    except RuntimeError as exc:
        add_check(checks, "fail_fast_on_broken_artifact_path", True, "blocker", {"raised": str(exc)[:200]})


def check_forbidden_features(checks: list[dict[str, Any]], adapter: Omega6LiveAdapter) -> None:
    bad: dict[str, list[str]] = {}
    for alias, component in (("primary", adapter.primary), ("fallback", adapter.fallback)):
        cols: set[str] = set()
        for _model, _scaler, input_cols in component.models.values():
            cols.update(input_cols)
        found = sorted(c for c in cols if any(str(c).startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES))
        if found:
            bad[alias] = found
    tcn_found = sorted(c for c in adapter.tcn["feature_cols"] if any(str(c).startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES))
    if tcn_found:
        bad["l3_gate"] = tcn_found
    add_check(checks, "forbidden_feature_prefixes_absent", not bad, "blocker", {"violations": bad})


def check_timestamp_non_overlap(checks: list[dict[str, Any]]) -> None:
    split_ts = omega6_tabm.SPLIT_TS
    val_start = bt.VAL_START
    ok = split_ts <= val_start
    add_check(
        checks,
        "l2_train_val_non_overlap",
        ok,
        "blocker",
        {"split_ts": str(split_ts), "val_start": str(val_start)},
    )
    # L3/L4 training scripts hardcode the same SPLIT_TS boundary (scripts/train_omega6_risk_sidecar_20260703.py,
    # scripts/train_omega6_sequence_gate_20260703.py) -- verify their artifact-recorded train windows
    # also end before val_start.
    l4_report = ROOT / "tmp/causal_regen_20260516/omega6_risk_sidecar_20260703/report.json"
    l3_report = ROOT / "tmp/causal_regen_20260516/omega6_sequence_gate_20260703/report.json"
    for name, path, key in (("l4_sidecar", l4_report, "train_window"), ("l3_gate", l3_report, "train_window")):
        if not path.exists():
            add_check(checks, f"{name}_train_window_before_val", False, "blocker", {"note": f"missing {path}"})
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        train_end = pd.Timestamp(report[key]["end"])
        ok = train_end < val_start
        add_check(checks, f"{name}_train_window_before_val", ok, "blocker", {"train_end": str(train_end), "val_start": str(val_start)})


def check_accounting_contract(checks: list[dict[str, Any]], adapter: Omega6LiveAdapter, frame: pd.DataFrame) -> None:
    errors: list[float] = []
    n_checked = 0
    n_sided = 0
    step = max(len(frame) // 400, 1)
    for i in range(bt.CONTEXT_BARS, len(frame), step):
        window = frame.iloc[max(0, i - bt.CONTEXT_BARS + 1) : i + 1]
        dec = adapter.decide_latest(window)
        n_checked += 1
        if dec.side == 0:
            continue
        n_sided += 1
        expected = dec.margin_notional * dec.leverage
        errors.append(abs(expected - dec.notional_exposure))
    max_err = float(max(errors)) if errors else 0.0
    add_check(
        checks,
        "futures_sizing_contract_notional_eq_margin_times_leverage",
        max_err <= 1.0e-6,
        "blocker",
        {"max_abs_error": max_err, "bars_checked": n_checked, "sided_decisions": n_sided},
    )


def check_governor_reduce_only(checks: list[dict[str, Any]]) -> None:
    import inspect

    from trading_bot_modules import omega6_live as m

    src = inspect.getsource(m.Omega6LiveAdapter.decide_latest)
    has_shock_scale = "L6_SHOCK_NOTIONAL_SCALE" in src and "notional_exposure *= L6_SHOCK_NOTIONAL_SCALE" in src
    has_veto_zero = "l6_macro_veto" in src and "_cash_decision" in src
    scale_le_one = m.L6_SHOCK_NOTIONAL_SCALE <= 1.0
    add_check(
        checks,
        "l6_governor_reduce_only",
        bool(has_shock_scale and has_veto_zero and scale_le_one),
        "blocker",
        {
            "shock_scale_applied_as_multiply": has_shock_scale,
            "macro_veto_forces_cash": has_veto_zero,
            "shock_scale_value": m.L6_SHOCK_NOTIONAL_SCALE,
            "shock_scale_le_one": scale_le_one,
        },
    )


def check_cost_stress_non_collapse(checks: list[dict[str, Any]]) -> None:
    if not LATEST_BACKTEST_REPORT.exists():
        add_check(checks, "cost_stress_trade_count_non_collapse", False, "blocker", {"note": "no backtest report found"})
        return
    report = json.loads(LATEST_BACKTEST_REPORT.read_text(encoding="utf-8"))
    stress = report.get("validation", {}).get("cost_stress", {})
    cost1_trades = int(stress.get("cost1", {}).get("trades", 0))
    cost3_trades = int(stress.get("cost3", {}).get("trades", 0))
    ratio = float(cost3_trades / cost1_trades) if cost1_trades else 0.0
    add_check(
        checks,
        "cost_stress_trade_count_non_collapse",
        ratio >= COST_TRADE_RATIO_MIN,
        "blocker",
        {"cost1_trades": cost1_trades, "cost3_trades": cost3_trades, "ratio": ratio, "min_ratio": COST_TRADE_RATIO_MIN},
    )


def check_sizing_sensitivity(checks: list[dict[str, Any]]) -> None:
    if not LATEST_BACKTEST_REPORT.exists():
        return
    report = json.loads(LATEST_BACKTEST_REPORT.read_text(encoding="utf-8"))
    pnl = float(report.get("validation", {}).get("pnl", 0.0))
    add_check(
        checks,
        "sizing_sensitivity_informational",
        pnl > 0.0,
        "warning",
        {
            "current_val_pnl": pnl,
            "note": (
                "2026-07-03 session found validation PnL flips from +27.69% (uncapped L4 mapping, "
                "cap=0.6) to -13.48% (MDD-capped L4 mapping, cap=0.3) using the SAME L2/L3 signal. "
                "This is a sizing-sensitivity finding, not a pass/fail gate by itself, but it is a "
                "material caveat: the underlying directional signal's edge is not clearly robust to "
                "reasonable resizing choices. See docs/model_contracts/omega6_synthesis_v1_20260703_contract.md."
            ),
        },
    )


def main() -> int:
    checks: list[dict[str, Any]] = []
    check_fail_fast(checks)
    check_timestamp_non_overlap(checks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = Omega6LiveAdapter(
        primary_bundle_path=str(bt.DEFAULT_PRIMARY_BUNDLE),
        fallback_bundle_path=str(bt.DEFAULT_FALLBACK_BUNDLE),
        tcn_gate_path=str(bt.DEFAULT_TCN_GATE),
        risk_sidecar_path=str(bt.DEFAULT_RISK_SIDECAR),
        device=device,
        enable_l3_gate=True,
    )
    check_forbidden_features(checks, adapter)
    check_governor_reduce_only(checks)

    frame = bt._load_combined_frame()
    val_start_idx, val_end_idx = bt._window_bounds(frame, bt.VAL_START, bt.VAL_END)
    sample_frame = frame.iloc[max(0, val_start_idx - bt.CONTEXT_BARS) : val_start_idx + 2000]
    check_accounting_contract(checks, adapter, sample_frame)

    check_cost_stress_non_collapse(checks)
    check_sizing_sensitivity(checks)

    blockers = [c for c in checks if c["severity"] == "blocker" and not c["pass"]]
    warnings = [c for c in checks if c["severity"] == "warning" and not c["pass"]]
    if blockers:
        verdict = "REDTEAM_FAIL"
    elif warnings:
        verdict = "CONDITIONAL_PASS_WITH_WARNINGS"
    else:
        verdict = "FULL_PASS"

    payload = {
        "model_id": "omega6_synthesis_v1_20260703",
        "audit_id": "omega6_synthesis_redteam_20260703",
        "verdict": verdict,
        "checks": checks,
        "n_blockers": len(blockers),
        "n_warnings": len(warnings),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    lines = [
        "# Omega6 Synthesis Red-Team Audit - 2026-07-03",
        "",
        f"- Verdict: `{verdict}`",
        f"- Blockers: {len(blockers)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Checks",
        "",
        "| Check | Severity | Pass |",
        "| --- | --- | --- |",
    ]
    for c in checks:
        lines.append(f"| `{c['name']}` | {c['severity']} | {c['pass']} |")
    if warnings:
        lines.append("")
        lines.append("## Warnings (non-blocking, must be read before promotion)")
        for c in warnings:
            lines.append(f"- **{c['name']}**: {json.dumps(c['details'], ensure_ascii=False, default=_json_default)}")
    lines.append("")
    lines.append(f"- JSON: `{REPORT_JSON}`")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"verdict": verdict, "n_blockers": len(blockers), "n_warnings": len(warnings), "report": str(REPORT_JSON)}, indent=2, default=_json_default), flush=True)
    return 0 if verdict != "REDTEAM_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
