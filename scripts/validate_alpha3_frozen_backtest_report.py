#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED = {
    "cost1": {"pnl": 654.9174150098765, "mdd": -29.61731295277763, "trades": 195},
    "cost2": {"pnl": 602.2624624847589, "mdd": -30.093378120960466, "trades": 195},
    "cost3": {"pnl": 456.48201847894717, "mdd": -31.397871677089583, "trades": 198},
}

PNL_TOL = 0.05
MDD_TOL = 0.05
SURFACES = {
    "parent_only",
    "teacher_only",
    "runner_only",
    "deep_scout_only",
    "exit_only",
    "execution_only",
    "parent_plus_downstream_retune",
    "full_stack_retune",
}


def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    if isinstance(report.get("baseline_metrics"), dict):
        return dict(report["baseline_metrics"])
    if isinstance(report.get("experiments"), list):
        for exp in report["experiments"]:
            name = str(exp.get("name", "")).lower()
            cfg = exp.get("config") if isinstance(exp.get("config"), dict) else {}
            cfg_name = str(cfg.get("name", "")).lower()
            if (
                ("alpha3" in name and ("baseline" in name or "current" in name or "original" in name))
                or "next_open_limit_touch0_fee20" in name
                or cfg_name == "next_open_limit_touch0_fee20"
            ):
                if isinstance(exp.get("metrics"), dict):
                    return dict(exp["metrics"])
        first = report["experiments"][0] if report["experiments"] else {}
        if isinstance(first.get("metrics"), dict):
            return dict(first["metrics"])
    if isinstance(report.get("results"), list):
        first = report["results"][0] if report["results"] else {}
        if isinstance(first.get("metrics"), dict):
            return dict(first["metrics"])
    raise KeyError("baseline_metrics")


def validate(path: Path) -> tuple[list[str], list[str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    surface = report.get("primary_mutable_surface")
    if surface is None:
        warnings.append("missing primary_mutable_surface")
    elif str(surface) not in SURFACES:
        errors.append(f"unknown primary_mutable_surface={surface}")

    if report.get("selection_uses_2026") is True:
        errors.append("selection_uses_2026 must be false")

    try:
        baseline = _metrics(report)
    except KeyError:
        errors.append("missing baseline metrics")
        return errors, warnings

    for cost, exp in EXPECTED.items():
        got = baseline.get(cost)
        if not isinstance(got, dict):
            errors.append(f"missing baseline {cost}")
            continue
        try:
            pnl = float(got["pnl"])
            mdd = float(got["mdd"])
            trades = int(got["trades"])
        except Exception as exc:
            errors.append(f"bad baseline {cost}: {exc}")
            continue
        if abs(pnl - float(exp["pnl"])) > PNL_TOL:
            errors.append(f"{cost} pnl mismatch: got {pnl:.6f}, expected {float(exp['pnl']):.6f}")
        if abs(mdd - float(exp["mdd"])) > MDD_TOL:
            errors.append(f"{cost} mdd mismatch: got {mdd:.6f}, expected {float(exp['mdd']):.6f}")
        if trades != int(exp["trades"]):
            errors.append(f"{cost} trades mismatch: got {trades}, expected {int(exp['trades'])}")

    changed = report.get("changed_layers")
    frozen = report.get("frozen_layers")
    if changed is None:
        warnings.append("missing changed_layers")
    if frozen is None:
        warnings.append("missing frozen_layers")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a report against the Alpha3 frozen backtest protocol.")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    errors, warnings = validate(args.report)
    out = {"report": str(args.report), "status": "pass" if not errors else "fail", "errors": errors, "warnings": warnings}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
