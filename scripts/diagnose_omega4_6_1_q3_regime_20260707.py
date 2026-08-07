"""Why did 2025-Q3 invert (Phase 1 finding)? Compares market/regime characteristics across all
five windows tested so far (2025-Q1/Q2/Q3, VAL, OOS) using ONLY pre-existing features (no new
feature invented to explain the n=1 bad quarter -- that would be exactly the overfitting trap this
project has repeatedly hit). Also inspects the actual zig075-SHORT entries in each window (regime3
probs, confidence, ou_halflife at entry) to see whether Q3's losing SHORT trades look structurally
different from the winning ones elsewhere."""
from __future__ import annotations

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
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402
from audit_omega4_6_1_phase1_robustness_20260707 import load_2025_quarter_components, load_val_components, load_oos_components  # noqa: E402

REGIME_COLS = ["regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
               "regime3_current_sensitive_wide24_chop_prob", "regime3_current_sensitive_wide24_confidence"]


def market_summary(frame: pd.DataFrame, label: str) -> dict:
    close = pd.to_numeric(frame["close"], errors="raise")
    total_ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
    bar_ret = close.pct_change().dropna()
    ann_vol = float(bar_ret.std() * np.sqrt(288 * 365))
    regime_avg = {c: float(frame[c].mean()) for c in REGIME_COLS if c in frame.columns}
    ou_avg = float(frame["ou_halflife"].mean()) if "ou_halflife" in frame.columns else None
    print(f"  {label}: total_ret={total_ret:+.2%} ann_vol={ann_vol:.2%} regime_avg={regime_avg} ou_halflife_avg={ou_avg}", flush=True)
    return {"total_ret": total_ret, "ann_vol": ann_vol, "regime_avg": regime_avg, "ou_halflife_avg": ou_avg}


def zig075_short_entries(frame: pd.DataFrame, components: dict, fee: float, slip: float) -> pd.DataFrame:
    greedy.PRIORITY = ("h48qual", "zig075")
    _, ledger = greedy.greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=1.0, device=retest.DEVICE)
    if ledger.empty:
        return ledger
    sub = ledger[(ledger["source_component"] == "zig075") & (ledger["side"] < 0)].copy()
    if sub.empty:
        return sub
    ets = pd.to_datetime(sub["entry_timestamp"])
    idx = frame.set_index("timestamp")
    cols = [c for c in REGIME_COLS if c in frame.columns] + (["ou_halflife"] if "ou_halflife" in frame.columns else [])
    for c in cols:
        sub[c] = ets.map(idx[c]).to_numpy()
    return sub[["entry_timestamp", "exit_timestamp", "trade_return", "reason", *cols]]


def main() -> int:
    fee, slip = omega._load_fee_slip()
    print("=== Market/regime summary per window ===", flush=True)
    windows = []
    for start, end, label in [("2025-01-01", "2025-03-31 23:59:59", "2025-Q1"),
                              ("2025-04-01", "2025-06-30 23:59:59", "2025-Q2"),
                              ("2025-07-01", "2025-09-30 23:59:59", "2025-Q3")]:
        frame, components = load_2025_quarter_components(start, end)
        market_summary(frame, label)
        windows.append((label, frame, components))

    val_frame, val_components = load_val_components()
    market_summary(val_frame, "VAL 2025-10..12")
    windows.append(("VAL", val_frame, val_components))

    oos_frame, oos_components = load_oos_components()
    market_summary(oos_frame, "OOS 2026-01..06")
    windows.append(("OOS", oos_frame, oos_components))

    print("\n=== zig075 SHORT entries per window (regime3 probs / ou_halflife AT ENTRY) ===", flush=True)
    for label, frame, components in windows:
        sub = zig075_short_entries(frame, components, fee, slip)
        print(f"\n  --- {label} ({len(sub)} zig075 SHORT trades) ---", flush=True)
        if sub.empty:
            continue
        print(sub.round(4).to_string(index=False), flush=True)
        print(f"  mean bull_prob={sub['regime3_current_sensitive_wide24_bull_prob'].mean():.3f} "
              f"bear_prob={sub['regime3_current_sensitive_wide24_bear_prob'].mean():.3f} "
              f"chop_prob={sub['regime3_current_sensitive_wide24_chop_prob'].mean():.3f} "
              f"confidence={sub['regime3_current_sensitive_wide24_confidence'].mean():.3f} "
              f"ou_halflife={sub['ou_halflife'].mean():.5f}  sum_ret={sub['trade_return'].sum():+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
