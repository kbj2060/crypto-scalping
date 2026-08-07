"""Candidate 7: multi-week trailing-trend veto on zig075 SHORT entries.

Motivated by `scripts/diagnose_omega4_6_1_q3_regime_20260707.py`'s finding: across all FIVE
windows examined this project (2025-Q1/Q2/Q3, VAL, OOS), zig075 SHORT's realized sum_ret tracks
the window's broader market trend, not the bar-level regime3 tag (which flipped meaning between
windows -- the same regime3 bull_prob~0.72 tag produced take-profits in Q1 but stop-losses in Q3).
Q1 (market -45%): SHORT +0.617. Q2 (+36%): +0.205. Q3 (+67%, the one window that inverted in
Phase 1): -0.517. VAL (-28%): +0.414. OOS (-46%): +1.092. This is a genuinely cross-window,
monotonic-looking pattern (not a rule reverse-engineered from the single Q3 anomaly), which is why
it is worth testing properly rather than dismissed as overfitting.

Design: veto a zig075 SHORT entry if a CAUSAL trailing-return of `close` over the last
`lookback_days` exceeds `threshold` (i.e., don't short after the market has already rallied hard).
zig075 LONG and h48qual are untouched -- gate zig075's own decision frame's `side` to 0 at vetoed
bars (same `gated_component`-style technique as Candidate 3, reusing the UNMODIFIED greedy_replay
engine, no engine changes).

Protocol (stricter than prior candidates): select (lookback_days, threshold) on TRAIN
2025-01-01..09-30 ONLY (a window even further removed from OOS than VAL is -- Q1/Q2/Q3 predate
VAL). VAL 2025-10-01..12-31 is then an interim OUT-OF-SAMPLE check (not used for reselection). OOS
2026-01-01..06-30 is scored ONCE with the frozen config. trading_bot.py is not touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
from test_omega4_6_1_drop_h48qual_20260706 import _metrics  # noqa: E402
from audit_omega4_6_1_phase1_robustness_20260707 import load_2025_quarter_components, load_val_components, load_oos_components  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_trend_veto_20260707"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BARS_PER_DAY = 288  # 5-minute bars
LOOKBACK_DAYS_GRID = [7, 14, 21, 30, 45, 60]
THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.30]


def gated_zig075_short(comp: dict, frame: pd.DataFrame, lookback_days: int, threshold: float) -> dict:
    close = pd.to_numeric(frame["close"], errors="raise")
    lb = lookback_days * BARS_PER_DAY
    trailing_ret = close.pct_change(periods=lb)  # causal: uses only bars <= i
    veto = (trailing_ret > threshold).fillna(False).to_numpy()
    dec = comp["dec"].copy()
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy()
    mask = veto & (side == -1)
    if mask.any():
        side = side.copy()
        side[mask] = 0
        dec["side"] = side
    out = dict(comp)
    out["dec"] = dec
    return out


def score(frame: pd.DataFrame, components: dict, fee: float, slip: float, lookback_days: int | None, threshold: float | None) -> dict:
    comps = dict(components)
    if lookback_days is not None:
        comps["zig075"] = gated_zig075_short(components["zig075"], frame, lookback_days, threshold)
    greedy.PRIORITY = ("h48qual", "zig075")
    _, lg = greedy.greedy_replay(frame, comps, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=retest.DEVICE)
    return _metrics(lg, frame, apply_gate=True)


def main() -> int:
    fee, slip = omega._load_fee_slip()

    print("Loading TRAIN (2025-01-01..09-30)...", flush=True)
    train_frame, train_components = load_2025_quarter_components("2025-01-01", "2025-09-30 23:59:59")
    baseline_train = score(train_frame, train_components, fee, slip, None, None)
    print(f"TRAIN baseline (no veto): {baseline_train}", flush=True)

    grid = []
    for lb in LOOKBACK_DAYS_GRID:
        for th in THRESHOLD_GRID:
            m = score(train_frame, train_components, fee, slip, lb, th)
            grid.append({"lookback_days": lb, "threshold": th, **m})
            print(f"  lookback={lb:2d}d threshold={th:.2f} -> pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+6.2f}% n={m['trades']:2d} wr={m['wr']:.3f}", flush=True)

    grid.sort(key=lambda r: r["pnl"], reverse=True)
    best = grid[0]
    print(f"\nBest TRAIN config: {best}", flush=True)
    frozen = {"lookback_days": best["lookback_days"], "threshold": best["threshold"]}

    print("\nLoading VAL (interim out-of-sample check, NOT used for selection)...", flush=True)
    val_frame, val_components = load_val_components()
    baseline_val = score(val_frame, val_components, fee, slip, None, None)
    frozen_val = score(val_frame, val_components, fee, slip, frozen["lookback_days"], frozen["threshold"])
    print(f"VAL baseline: {baseline_val}", flush=True)
    print(f"VAL frozen-veto: {frozen_val}", flush=True)

    print("\nLoading OOS (ONE-SHOT confirm)...", flush=True)
    oos_frame, oos_components = load_oos_components()
    baseline_oos = score(oos_frame, oos_components, fee, slip, None, None)
    frozen_oos = score(oos_frame, oos_components, fee, slip, frozen["lookback_days"], frozen["threshold"])
    print(f"OOS baseline: {baseline_oos}", flush=True)
    print(f"OOS frozen-veto (ONE SHOT): {frozen_oos}", flush=True)

    adopt = bool(frozen_val["pnl"] >= baseline_val["pnl"] and frozen_oos["pnl"] >= baseline_oos["pnl"])
    print(f"\nDecision (frozen config must not hurt EITHER out-of-sample window): {'ADOPT' if adopt else 'REJECT'}", flush=True)

    result = {
        "model_id": "omega4_6_1_trend_veto_20260707",
        "grid_train": grid,
        "frozen_config": frozen,
        "train": {"baseline": baseline_train, "best": best},
        "val": {"baseline": baseline_val, "frozen": frozen_val},
        "oos": {"baseline": baseline_oos, "frozen": frozen_oos},
        "adopt_decision": adopt,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
