#!/usr/bin/env python3
"""RESEARCH ONLY -- barrier-race diagnostic on ACTUAL LIVE-SELECTED trades (not raw zigzag/direction
bars) across all 4 live model lines (h48qual/zig075 ETH, BTC, SOL), using the correct INTRABAR
high/low barrier convention (omega4_6_1_live.py::evaluate_exit's bar_high_move/bar_low_move, SL
checked before TP within a bar per that function's own documented conservative-ordering choice).

Motivation (2026-08-19): today's ModernTCN quality-head investigation found that under the shared
ATR-adaptive TP/SL formula (atr_window=192, tp_mult=12.0, sl_mult=6.0 -- confirmed identical across
h48qual/zig075/BTC/SOL, every components_override call site in trading_bot.py leaves these at
_ComponentConfig defaults), UNCONDITIONAL zigzag-direction entries lose the barrier race to SL ~63%
of the time (cost-independent, confirmed by comparing cost_mult=1.0 vs 3.0 -- near-identical
net_edge_fail rate), using a CLOSE-PRICE-ONLY barrier walk. User asked whether this means the real
live models (which select only a small, quality/veto-gated subset of bars, not every zigzag bar) are
similarly disqualified. This script answers that directly and more accurately: real LIVE-SELECTED
entries only (quality threshold pass via each model's own saved final_action column, PLUS the
ou_halflife <= duration_threshold veto from Omega461LiveAdapter.decide_entry -- final_action alone
under-counts what live actually vetoes), walked forward with the INTRABAR barrier convention that
live actually uses (not the close-only approximation from today's earlier ModernTCN-only check).

Per-asset artifacts (confirmed via repo discovery, 2026-08-19 -- see this script's ASSETS dict for
exact paths/thresholds):
- h48qual (ETH): oos_predictions_q050.csv, quality_threshold=0.50, duration_threshold=0.005417,
  panel=data/splits/year_oos/eth_features_2024_2026_analysis.csv, OOS 2026-01-01..02-28.
- zig075 (ETH): oos_predictions_q075.csv, quality_threshold=0.75, same duration_threshold/panel
  (shared ETH adapter) -- panel reuse is inferred with high but not 100% confidence per discovery.
- BTC: oos_predictions_q055.csv, quality_threshold=0.55, duration_threshold=0.0054143218,
  panel=btc_features_2026_swingtransition.csv, OOS 2026-01-01..07-12.
- SOL: oos_predictions_q070.csv, quality_threshold=0.70, duration_threshold=0.0055208323,
  panel=sol_features_2026.csv (adaptive_squeeze build -- NOT the plain year_oos SOL panel), OOS
  2026-01-01..07-12.

No take_profit/stop_loss columns exist in these prediction CSVs (checked directly) -- TP/SL are
recomputed here from each panel's own OHLC via the shared ATR_CFG formula, evaluated at the signal
bar (matching when the live decision itself computes it), not re-derived per walk-forward bar.

Barrier walk is capped at MAX_WALK_BARS=2016 (1 week of 5-min bars) -- entries that resolve neither
way inside that cap are counted "no_resolution", not walked to the literal end of the panel (unlike
today's earlier ModernTCN close-only check) -- disclosed cap, not silent truncation; reported per
asset alongside true stop_loss/take_profit counts.

fresh_forward_bar_by_bar=true (each entry only walks forward from its own signal bar using already-
confirmed OHLC), trade_ledgers_used_as_input=false (no saved ledgers used, only fresh predictions +
panel OHLC), future_rows_used_for_entry=false. This is a DIAGNOSTIC of an existing artifact's
historical OOS window, not a promotion or model-selection claim -- CLAUDE.md's Omega Artifact
Integrity Promotion Gate does not apply (no promotion claim is being made).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402

ATR_CFG = {"atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0, "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12}
MAX_WALK_BARS = 2016  # 1 week of 5-min bars, disclosed cap

ASSETS = {
    "h48qual_eth": {
        "pred_csv": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/oos_predictions_q050.csv",
        "panel_csv": ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv",
        "duration_threshold": 0.005417,
    },
    "zig075_eth": {
        "pred_csv": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/oos_predictions_q075.csv",
        "panel_csv": ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv",
        "duration_threshold": 0.005417,
    },
    "btc": {
        "pred_csv": ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition/oos_predictions_q055.csv",
        "panel_csv": ROOT / "data/splits/year_oos/btc_features_2026_swingtransition.csv",
        "duration_threshold": 0.0054143218,
    },
    "sol": {
        "pred_csv": ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/oos_predictions_q070.csv",
        "panel_csv": ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2026.csv",
        "duration_threshold": 0.0055208323,
    },
}


def log(msg: str) -> None:
    print(f"[barrier_race] {msg}", flush=True)


def _live_selected_entries(pred_csv: Path, panel: pd.DataFrame, duration_threshold: float) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv, parse_dates=["timestamp"])
    final_action_cols = [c for c in pred.columns if c.endswith("_final_action")]
    if len(final_action_cols) != 1:
        raise RuntimeError(f"{pred_csv}: expected exactly 1 *_final_action column, found {final_action_cols}")
    fa_col = final_action_cols[0]
    sel = pred.loc[pred[fa_col] != 0, ["timestamp", fa_col]].rename(columns={fa_col: "action"}).copy()
    n_quality_selected = len(sel)

    halflife_map = panel.set_index("timestamp")["ou_halflife"]
    sel["ou_halflife"] = sel["timestamp"].map(halflife_map)
    n_missing_halflife = int(sel["ou_halflife"].isna().sum())
    sel = sel.dropna(subset=["ou_halflife"])
    sel = sel[sel["ou_halflife"] > duration_threshold].reset_index(drop=True)
    log(f"  quality_selected={n_quality_selected} missing_halflife={n_missing_halflife} "
        f"after_duration_veto={len(sel)} (dropped {n_quality_selected - n_missing_halflife - len(sel)} by duration gate)")
    return sel


def _barrier_race(panel: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    arrays = {c: pd.to_numeric(panel[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    ts_to_i = {t: i for i, t in enumerate(panel["timestamp"])}
    atr_pct = atr_eval._atr_pct(panel, ATR_CFG["atr_window"])
    tp_move_arr = np.clip(np.maximum(ATR_CFG["min_tp"], atr_pct * ATR_CFG["tp_mult"]), 0.0, ATR_CFG["max_tp"])
    sl_move_arr = np.clip(np.maximum(ATR_CFG["min_sl"], atr_pct * ATR_CFG["sl_mult"]), 0.0, ATR_CFG["max_sl"])
    n = len(panel)

    rows = []
    for _, r in entries.iterrows():
        ts = r["timestamp"]
        i = ts_to_i.get(ts)
        if i is None:
            continue
        side = 1 if int(r["action"]) == 1 else -1
        entry_i = min(i + 1, n - 1)
        if entry_i >= n - 1:
            continue
        entry_price = arrays["open"][entry_i]
        tp, sl = float(tp_move_arr[i]), float(sl_move_arr[i])
        outcome, exit_j = "no_resolution", None
        walk_end = min(entry_i + MAX_WALK_BARS, n - 1)
        for j in range(entry_i, walk_end + 1):
            if side > 0:
                high_move = (arrays["high"][j] - entry_price) / entry_price
                low_move = (arrays["low"][j] - entry_price) / entry_price
            else:
                high_move = (entry_price - arrays["low"][j]) / entry_price
                low_move = (entry_price - arrays["high"][j]) / entry_price
            if sl > 0.0 and low_move <= -abs(sl):
                outcome, exit_j = "stop_loss", j
                break
            if tp > 0.0 and high_move >= tp:
                outcome, exit_j = "take_profit", j
                break
        rows.append({"timestamp": ts, "side": side, "tp": tp, "sl": sl, "outcome": outcome,
                      "hold_bars": (exit_j - entry_i) if exit_j is not None else None})
    return pd.DataFrame(rows)


def main() -> int:
    out_dir = ROOT / "tmp/causal_regen_20260516/research_live_selected_entries_barrier_race_20260819"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for asset, cfg in ASSETS.items():
        log(f"=== {asset} ===")
        t0 = time.time()
        panel = pd.read_csv(cfg["panel_csv"], low_memory=False, parse_dates=["timestamp"])
        panel = panel.sort_values("timestamp").reset_index(drop=True)
        entries = _live_selected_entries(cfg["pred_csv"], panel, cfg["duration_threshold"])
        if len(entries) == 0:
            log("  0 entries after duration veto, skipping")
            continue
        race = _barrier_race(panel, entries)
        race.to_csv(out_dir / f"{asset}_barrier_race.csv", index=False)
        counts = race["outcome"].value_counts().to_dict()
        total = len(race)
        tp_n, sl_n, none_n = counts.get("take_profit", 0), counts.get("stop_loss", 0), counts.get("no_resolution", 0)
        row = {"asset": asset, "n_entries": total, "take_profit": tp_n, "stop_loss": sl_n, "no_resolution": none_n,
               "tp_rate": round(tp_n / max(total, 1), 4), "sl_rate": round(sl_n / max(total, 1), 4),
               "no_res_rate": round(none_n / max(total, 1), 4), "elapsed_sec": round(time.time() - t0, 1)}
        summary_rows.append(row)
        log(f"  {asset}: n={total} TP={tp_n}({row['tp_rate']:.1%}) SL={sl_n}({row['sl_rate']:.1%}) "
            f"no_resolution={none_n}({row['no_res_rate']:.1%}) elapsed={row['elapsed_sec']}s")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    log("\n=== summary (live-selected entries, intrabar barrier race) ===")
    print(summary.to_string(index=False))
    log(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
