#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.elite import RegimeEngine  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.retest_alpha3_current_live_guard_20260515 import (  # noqa: E402
    LIVE_TRAIL_ACTIVATION,
    _cfg as _current_live_cfg,
    backtest_current_live,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402


MODEL_ID = "redteam_alpha3_realistic_execution_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/redteam_alpha3_realistic_execution_20260515.json"
GRID_OUT = ROOT / "data/ensemble/reports/redteam_alpha3_realistic_execution_20260515_grid.csv"
LIVE_L2_AUDIT = ROOT / "tmp/alpha3_live_l2_shadow_audit_now.json"
LIVE_JOURNAL = ROOT / "data/live/trade_journal.jsonl"

TryLimitFn = Callable[..., tuple[bool, float, float, float, str]]


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        val = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(val) if np.isfinite(val) else float(default)


def _open_price(df: pd.DataFrame, signal_i: int) -> float:
    fill_i = int(np.clip(int(signal_i) + 1, 0, len(df) - 1))
    row = df.iloc[fill_i]
    return _safe(row, "open", _safe(row, "close", 0.0))


def _taker_price(df: pd.DataFrame, signal_i: int, side: int, *, entry: bool, slip: float) -> float:
    px = _open_price(df, signal_i)
    if px <= 0.0:
        return 0.0
    if side > 0:
        return float(px * (1.0 + slip if entry else 1.0 - slip))
    return float(px * (1.0 - slip if entry else 1.0 + slip))


def _stable_accept(signal_i: int, side: int, *, entry: bool, fill_prob: float) -> bool:
    # Deterministic pseudo-randomness keeps the report reproducible without
    # giving every same-bar touch an impossible 100% maker queue fill.
    x = (int(signal_i) * 1103515245 + int(side) * 12345 + (17 if entry else 97)) & 0xFFFFFFFF
    u = (x % 1_000_000) / 1_000_000.0
    return bool(u < float(np.clip(fill_prob, 0.0, 1.0)))


def _try_taker_next_open(
    df: pd.DataFrame,
    signal_i: int,
    side: int,
    cfg: alpha3.ImmediateLimitConfig,
    *,
    entry: bool,
    fee: float,
    slip: float,
) -> tuple[bool, float, float, float, str]:
    px = _taker_price(df, signal_i, side, entry=entry, slip=slip)
    if px <= 0.0:
        return False, 0.0, 0.0, 0.0, "next_open_taker_invalid_price"
    return True, px, float(fee), float(slip), "next_open_taker_fee100"


def _make_l2_haircut_try(original: TryLimitFn, fill_prob: float) -> TryLimitFn:
    def _try(
        df: pd.DataFrame,
        signal_i: int,
        side: int,
        cfg: alpha3.ImmediateLimitConfig,
        *,
        entry: bool,
        fee: float,
        slip: float,
    ) -> tuple[bool, float, float, float, str]:
        filled, px, fee_rate, slip_rate, route = original(
            df,
            signal_i,
            side,
            cfg,
            entry=entry,
            fee=fee,
            slip=slip,
        )
        if not filled:
            return filled, px, fee_rate, slip_rate, route
        if "maker" not in str(route):
            return filled, px, fee_rate, slip_rate, route
        if _stable_accept(signal_i, side, entry=entry, fill_prob=fill_prob):
            return True, px, fee_rate, slip_rate, f"{route}_l2_accept_p{fill_prob:.3f}"
        if entry:
            return False, 0.0, 0.0, 0.0, f"{route}_l2_reject_skip_p{fill_prob:.3f}"
        fallback_px = _taker_price(df, signal_i, side, entry=False, slip=slip)
        if fallback_px <= 0.0:
            return False, 0.0, 0.0, 0.0, f"{route}_l2_reject_exit_invalid_p{fill_prob:.3f}"
        return True, fallback_px, float(fee), float(slip), f"{route}_l2_reject_exit_taker_p{fill_prob:.3f}"

    return _try


@contextmanager
def _patched_try_limit(fn: TryLimitFn) -> Iterator[None]:
    original = alpha3._try_immediate_limit
    alpha3._try_immediate_limit = fn
    try:
        yield
    finally:
        alpha3._try_immediate_limit = original


def _metrics_for_contract(
    df: pd.DataFrame,
    stack: dict[str, Any],
    q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    cfg: alpha3.ImmediateLimitConfig,
    try_limit_fn: TryLimitFn,
) -> dict[str, Any]:
    with _patched_try_limit(try_limit_fn):
        return {
            f"cost{mult}": backtest_current_live(
                df,
                stack["parent"],
                stack["jackpot_model"],
                stack["add_cfg"],
                q,
                decisions,
                overlay,
                cfg,
                fee=stack["fee"],
                slip=stack["slip"],
                cost_mult=float(mult),
            )
            for mult in (1, 2, 3)
        }


def _contract_rows(fill_prob: float) -> list[dict[str, Any]]:
    original = alpha3._try_immediate_limit
    return [
        {
            "name": "optimistic_touch0_maker_fee20_original",
            "cfg": _current_live_cfg(),
            "try_fn": original,
            "production_eligible": False,
            "note": "Legacy Alpha3 contract. Next-open touch0 maker fill is almost guaranteed by OHLC.",
        },
        {
            "name": "taker_next_open_fee100_slip100",
            "cfg": replace(_current_live_cfg(), name="taker_next_open_fee100_slip100", maker_fee_mult=1.0),
            "try_fn": _try_taker_next_open,
            "production_eligible": True,
            "note": "No maker assumption. Every entry/add/exit fills at next open with full taker fee and slippage.",
        },
        {
            "name": "post_only_pen2_skip_entry_exit_taker",
            "cfg": replace(
                _current_live_cfg(),
                name="post_only_pen2_skip_entry_exit_taker",
                penetration_bps=2.0,
                entry_miss="skip",
                exit_miss="market_fallback",
            ),
            "try_fn": original,
            "production_eligible": False,
            "note": "Maker requires 2 bps penetration instead of touch0; missed entries are skipped.",
        },
        {
            "name": "post_only_offset2_pen1_skip_entry_exit_taker",
            "cfg": replace(
                _current_live_cfg(),
                name="post_only_offset2_pen1_skip_entry_exit_taker",
                entry_offset_bps=2.0,
                exit_offset_bps=2.0,
                penetration_bps=1.0,
                entry_miss="skip",
                exit_miss="market_fallback",
            ),
            "try_fn": original,
            "production_eligible": False,
            "note": "More realistic passive price improvement requirement; lower fill rate expected.",
        },
        {
            "name": f"l2_haircut_touch0_p{fill_prob:.3f}",
            "cfg": replace(_current_live_cfg(), name=f"l2_haircut_touch0_p{fill_prob:.3f}"),
            "try_fn": _make_l2_haircut_try(original, fill_prob),
            "production_eligible": True,
            "note": "Legacy touch0 maker signal, but accepted only at live L2 compatible ratio; rejected entries skip.",
        },
        {
            "name": "l2_stress_touch0_p0.250",
            "cfg": replace(_current_live_cfg(), name="l2_stress_touch0_p0.250"),
            "try_fn": _make_l2_haircut_try(original, 0.25),
            "production_eligible": True,
            "note": "Stress test for weak post-only queue placement.",
        },
    ]


def _live_l2_compatible_ratio() -> float:
    if not LIVE_L2_AUDIT.exists():
        return 0.4782608695652174
    try:
        audit = json.loads(LIVE_L2_AUDIT.read_text(encoding="utf-8"))
        summary = dict(audit.get("trade_shadow_summary", {}) or {})
        val = float(summary.get("compatible_ratio_on_matched", 0.4782608695652174))
        return float(np.clip(val, 0.0, 1.0))
    except Exception:
        return 0.4782608695652174


def _live_journal_summary() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if LIVE_JOURNAL.exists():
        for line in LIVE_JOURNAL.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    closes = [r for r in rows if isinstance(r.get("pnl_pct"), (int, float))]
    pnl = [float(r["pnl_pct"]) for r in closes]
    recent = [r for r in closes if str(r.get("ts", "")) >= "2026-05-15"]
    alpha3_rows = [r for r in rows if "alpha2_1" in str(r.get("source", "")) or "v31" in str(r.get("source", ""))]
    alpha3_closes = [r for r in alpha3_rows if isinstance(r.get("pnl_pct"), (int, float))]
    return {
        "rows": len(rows),
        "closes": len(closes),
        "pnl_pct_sum": float(sum(pnl)) if pnl else 0.0,
        "pnl_pct_compounded": float((np.prod([1.0 + x / 100.0 for x in pnl]) - 1.0) * 100.0) if pnl else 0.0,
        "pnl_pct_since_2026_05_15": float(sum(float(r["pnl_pct"]) for r in recent)) if recent else 0.0,
        "alpha3_rows": len(alpha3_rows),
        "alpha3_closes": len(alpha3_closes),
        "alpha3_pnl_pct_sum": float(sum(float(r["pnl_pct"]) for r in alpha3_closes)) if alpha3_closes else 0.0,
    }


def _score(metrics: dict[str, Any]) -> float:
    c1, c2, c3 = metrics["cost1"], metrics["cost2"], metrics["cost3"]
    return float(c1["pnl"] + 0.50 * c2["pnl"] + 0.35 * c3["pnl"] - 0.50 * abs(c1["mdd"]))


def _json_default_safe(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return _json_default(obj)


def main() -> int:
    print(f"[{MODEL_ID}] loading frozen Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    eval_df = RegimeEngine().compute(_read(v31.DEFAULT_EVAL).copy())
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)
    overlay = replace(stack["overlay"], notional=2.0, trail_activation=LIVE_TRAIL_ACTIVATION)
    fill_prob = _live_l2_compatible_ratio()

    rows: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    for contract in _contract_rows(fill_prob):
        print(f"[{MODEL_ID}] evaluating {contract['name']}", flush=True)
        metrics = _metrics_for_contract(
            eval_df,
            stack,
            eval_q,
            eval_dec,
            overlay,
            contract["cfg"],
            contract["try_fn"],
        )
        score = _score(metrics)
        row = {
            "contract": contract["name"],
            "production_eligible": bool(contract.get("production_eligible", False)),
            "score": score,
            "cost1_pnl": metrics["cost1"]["pnl"],
            "cost1_mdd": metrics["cost1"]["mdd"],
            "cost1_trades": metrics["cost1"]["trades"],
            "cost1_wr": metrics["cost1"]["wr"],
            "cost2_pnl": metrics["cost2"]["pnl"],
            "cost2_mdd": metrics["cost2"]["mdd"],
            "cost3_pnl": metrics["cost3"]["pnl"],
            "cost3_mdd": metrics["cost3"]["mdd"],
            "cost1_route_counts": json.dumps(metrics["cost1"].get("route_counts", {}), sort_keys=True),
            "cost1_actions": json.dumps(metrics["cost1"].get("runner_actions", {}), sort_keys=True),
            "cost1_exits": json.dumps(metrics["cost1"].get("exits", {}), sort_keys=True),
        }
        rows.append(row)
        experiments.append(
            {
                "name": contract["name"],
                "note": contract["note"],
                "config": asdict(contract["cfg"]),
                "metrics": metrics,
                "score": score,
            }
        )
        print(
            f"[{MODEL_ID}] {contract['name']} c1={row['cost1_pnl']:.2f} "
            f"mdd={row['cost1_mdd']:.2f} trades={row['cost1_trades']} "
            f"c2={row['cost2_pnl']:.2f} c3={row['cost3_pnl']:.2f}",
            flush=True,
        )

    grid = pd.DataFrame(rows).sort_values("score", ascending=False)
    grid.to_csv(GRID_OUT, index=False)
    best = dict(grid.iloc[0]) if len(grid) else {}
    optimistic = next((r for r in rows if r["contract"] == "optimistic_touch0_maker_fee20_original"), {})
    realistic = [r for r in rows if not str(r["contract"]).startswith("optimistic_")]
    production_eligible = [r for r in rows if bool(r.get("production_eligible", False))]
    best_realistic = max(realistic, key=lambda r: float(r["score"])) if realistic else {}
    best_production_eligible = max(production_eligible, key=lambda r: float(r["score"])) if production_eligible else {}
    blocking: list[str] = []
    if not best_production_eligible:
        blocking.append("no_production_eligible_contracts_evaluated")
    if float(best_production_eligible.get("cost1_pnl", -1e18)) <= 0.0:
        blocking.append("no_positive_cost1_under_production_eligible_execution")
    if float(best_production_eligible.get("cost3_pnl", -1e18)) <= 0.0:
        blocking.append("no_positive_cost3_under_production_eligible_execution")
    if float(best_production_eligible.get("cost1_mdd", -100.0)) < -35.0:
        blocking.append("best_production_eligible_mdd_below_minus35")

    report = {
        "model_id": MODEL_ID,
        "selection_uses_2026": False,
        "oos_window": "2026 fixed OOS, Jan-Feb artifact window",
        "decision_stack": {
            "runtime": asdict(stack["runtime"]),
            "parent": str(v31.DEFAULT_PARENT),
            "teacher": str(alpha3.TEACHER_MODEL),
            "v27": str(v31.DEFAULT_V27),
            "jackpot": str(v31.DEFAULT_JACKPOT),
        },
        "live_l2_fill_ratio_used": fill_prob,
        "live_journal_summary": _live_journal_summary(),
        "optimistic_reference": optimistic,
        "best_by_score": best,
        "best_realistic_by_score": best_realistic,
        "best_production_eligible_by_score": best_production_eligible,
        "experiments": experiments,
        "grid": str(GRID_OUT),
        "audit": {
            "status": "blocked" if blocking else "pass",
            "verdict": "do_not_promote_alpha3_until_real_l2_replay_passes" if blocking else "shadow_only_candidate",
            "blocking": blocking,
            "warnings": [
                "Jan-Feb 2026 remains a short OOS window; run Mar-Apr after rebuilding full AI feature stack.",
                "L2 haircut is deterministic approximation, not true queue simulation.",
                "Post-only OHLC penetration contracts are reported but not production-eligible because they still cannot model queue priority or partial fills.",
                "Live bot should not treat optimistic touch0 maker results as production evidence.",
            ],
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default_safe), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_OUT),
                "grid": str(GRID_OUT),
                "status": report["audit"]["status"],
                "best_realistic": best_realistic,
                "blocking": blocking,
            },
            ensure_ascii=False,
            default=_json_default_safe,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
