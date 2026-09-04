#!/usr/bin/env python3
"""Multi-trigger V자반등 label -- FEEDER-ROLE-OPTIMIZED candidate pool (drops orthogonal_combo,
cluster-dedups kalman_deviation_meanrev/demarker_extreme). LABEL CONSTRUCTION ONLY -- per project
convention (feedback_visual_verification_chart_gate_explain_before_proceed), a 20-example visual
verification must be reviewed and explicitly approved by the user BEFORE any feature-building/
TabPFN-retraining step proceeds. This script does not touch the live 9-trigger pool
(build_eth_5m_v_rebound_multitrigger_labels_20260831.py, unchanged) or the deployed model
(live_eth_sweep_v_rebound_signal_20260829.py, unchanged) -- it only produces a new, separate
candidate-pool CSV for review.

Background: research_eth_v_rebound_multitrigger_feeder_role_screen_20260901.py audited all 9
triggers' NET-NEW contribution (candidates where that trigger fired and no other of the 9 did)
against the V_REBOUND outcome, TRAIN+VAL only. Two findings, both robust (large-N, checked
independently on bottom AND top sides):

  1. orthogonal_combo has ZERO net-new candidates on both sides (100% overlap with the other 8) --
     every bar it fires is already caught by another trigger. Dropping it from the union changes
     the candidate pool NOT AT ALL (same bars, same count) -- pure simplification, zero risk.
  2. kalman_deviation_meanrev and demarker_extreme are consumed as COMPLETELY RAW (undeduped)
     per-bar fires by both this label builder and the live server -- neither ever applies the
     cluster-dedup fix BTC's own 8-trigger V_REBOUND lineage already found necessary for these
     exact two signals (research_btc_v_rebound_8trigger_deduped_metalabel_tabpfn_20260901.py:
     "STATE indicators... can stay pinned past their threshold for many consecutive bars"). The
     GAP sweep (6/12/24/48/96, threshold held at each signal's own already-deployed native cutoff)
     showed a robust, large-sample, plateau-shaped improvement for BOTH signals peaking around
     GAP=6-12: kalman net-new rate +62% (bottom, n=857) / +82% (top, n=1004) relative to GAP=0;
     demarker +49% (bottom, n=417, crosses ABOVE the 14.68% pool baseline) / +80% (top, n=432).
     Threshold-only sweeps (GAP held at 0) showed NO comparable robust trend for either signal --
     dedup, not threshold, is the real lever here (the OPPOSITE of BTC's own kalman finding, where
     GAP barely mattered and threshold was the lever -- confirms even the SAME signal's own
     GAP-vs-threshold sensitivity does not transfer across assets).
  GAP=12 chosen (not 6) because it matches BOTH signals' own already-established live SUSTAIN_BARS_
  OVERRIDE value (live_evidence_signal_dashboard_20260823.py) for kalman_deviation_meanrev, and
  sits on the flat part of demarker's plateau -- a single shared GAP for both signals, not
  independently re-tuned per signal, to keep this a minimal, well-justified change.

  taker_delta_z_climax/short_term_return_z showed smaller threshold-tightening gains in the same
  screen, but their trigger threshold is SHARED with their own standalone evidence-signal/metalabel
  definitions elsewhere on the dashboard (compute_signals()'s delta_z<=-2.0/ret3_z<=-2.5) --
  retuning it here would change those other signals too. Deliberately EXCLUDED from this round;
  their current (native) thresholds are unchanged.

Everything else (outcome/label formula, feature set intent, TRIGGERS list contents, data sources)
is reused verbatim from build_eth_5m_v_rebound_multitrigger_labels_20260831.py -- only the trigger
UNION construction changes (drop one, dedup two). cluster_dedup() imported verbatim from
research_btc_demarker_extreme_metalabel_tabpfn_20260901.py (already used, unmodified, for BTC's own
analogous fix).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_btc_demarker_extreme_metalabel_tabpfn_20260901 import cluster_dedup  # noqa: E402

RECALL_SCRIPT = ROOT / "scripts/research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py"
_spec = importlib.util.spec_from_file_location("recall_check_90d_20260901feeder", RECALL_SCRIPT)
_recall = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_recall)
realized_outcome = _recall.realized_outcome
load_impl = _recall.load_impl

ETH_LOCAL_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_LOCAL_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_feeder_optimized_20260901"
ORIGINAL_REPORT = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/report.json"

# kept from the original 9: orthogonal_combo dropped (0 net-new, see module docstring)
NAMED_SIGNALS = ["taker_delta_z_climax", "short_term_return_z",
                 "smt_divergence", "fib_extension_exhaustion"]
DEDUP_SIGNALS = ["demarker_extreme", "kalman_deviation_meanrev"]
LOCAL_EXTREME_W = 6
DEDUP_GAP = 12  # see module docstring for justification


def load_local(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def dedup_trigger(raw_trig: np.ndarray, extremeness: np.ndarray, most_negative: bool, gap: int) -> tuple[np.ndarray, dict]:
    idx = np.flatnonzero(raw_trig)
    kept_idx = cluster_dedup(idx, extremeness[idx], most_negative=most_negative, gap=gap) if len(idx) else idx
    out = np.zeros(len(raw_trig), dtype=bool)
    out[kept_idx] = True
    return out, {"raw": int(len(idx)), "deduped": int(len(kept_idx))}


def main() -> None:
    t0 = time.time()
    impl = load_impl()
    eth = load_local(ETH_LOCAL_CSV)
    btc = load_local(BTC_LOCAL_CSV)
    print(f"ETH {len(eth)}bars {eth['timestamp'].iloc[0]} ~ {eth['timestamp'].iloc[-1]}")
    print(f"BTC {len(btc)}bars {btc['timestamp'].iloc[0]} ~ {btc['timestamp'].iloc[-1]}")

    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
    sig = compute_signals(eth, btc_df=btc, funding_df=None).reset_index(drop=True)
    sig["atr"] = causal["atr"].to_numpy()
    n = len(sig)

    low, high, close = sig["low"].to_numpy(), sig["high"].to_numpy(), sig["close"].to_numpy()
    level_low, level_high = causal["sweep_level_low"].to_numpy(), causal["sweep_level_high"].to_numpy()
    is_down_sweep = np.where(np.isnan(level_low), False, (low < level_low) & (close > level_low))
    is_up_sweep = np.where(np.isnan(level_high), False, (high > level_high) & (close < level_high))

    down_triggers = {"liquidity_sweep": is_down_sweep}
    up_triggers = {"liquidity_sweep": is_up_sweep}
    for name in NAMED_SIGNALS:
        down_triggers[name] = sig[f"bottom_{name}"].to_numpy()
        up_triggers[name] = sig[f"top_{name}"].to_numpy()

    dedup_stats = {}
    dem, kalman_dev_z = sig["dem"].to_numpy(), sig["kalman_dev_z"].to_numpy()
    for name, extremeness in (("demarker_extreme", dem), ("kalman_deviation_meanrev", kalman_dev_z)):
        raw_bottom = sig[f"bottom_{name}"].to_numpy()
        raw_top = sig[f"top_{name}"].to_numpy()
        down_triggers[name], stats_b = dedup_trigger(raw_bottom, extremeness, most_negative=True, gap=DEDUP_GAP)
        up_triggers[name], stats_t = dedup_trigger(raw_top, extremeness, most_negative=False, gap=DEDUP_GAP)
        dedup_stats[name] = {"bottom": stats_b, "top": stats_t}
        print(f"  {name} cluster-dedup (GAP={DEDUP_GAP}): "
              f"bottom {stats_b['raw']}->{stats_b['deduped']}, top {stats_t['raw']}->{stats_t['deduped']}")

    W = LOCAL_EXTREME_W
    local_low = np.zeros(n, dtype=bool)
    local_high = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        seg_lo, seg_hi = low[i - W:i + W + 1], high[i - W:i + W + 1]
        if low[i] == seg_lo.min():
            local_low[i] = True
        if high[i] == seg_hi.max():
            local_high[i] = True
    down_triggers["local_extreme"] = local_low
    up_triggers["local_extreme"] = local_high

    def build_side(triggers: dict, is_down: bool) -> list[dict]:
        any_fire = np.zeros(n, dtype=bool)
        for arr in triggers.values():
            any_fire |= arr
        rows = []
        for i in np.flatnonzero(any_fire):
            o = realized_outcome(sig, int(i), is_down)
            if o is None or o["partial_window"]:
                continue
            fired = sorted(name for name, arr in triggers.items() if arr[i])
            rows.append({
                "idx": int(i), "timestamp": sig["timestamp"].iloc[i].isoformat(),
                "direction": "downside" if is_down else "upside",
                "triggers": ",".join(fired), "n_triggers": len(fired), **o,
            })
        return rows

    print("스캔 중...")
    rows = build_side(down_triggers, True) + build_side(up_triggers, False)
    labels = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_path = OUT_DIR / "eth_5m_v_rebound_multitrigger_feeder_optimized_labels.csv"
    labels.to_csv(label_path, index=False)

    outcome_counts = labels["outcome"].value_counts().to_dict()
    by_trigger_v_rebound = {}
    for name in list(down_triggers.keys()):
        mask = labels["triggers"].str.contains(name)
        sub = labels[mask]
        by_trigger_v_rebound[name] = {
            "n_candidates": int(len(sub)), "n_v_rebound": int((sub["outcome"] == "V자반등").sum()),
            "rate": None if not len(sub) else round(float((sub["outcome"] == "V자반등").mean()), 4),
        }

    original = json.loads(ORIGINAL_REPORT.read_text())
    report = {
        "change_summary": {
            "dropped_triggers": ["orthogonal_combo"],
            "dedup_applied": {"signals": DEDUP_SIGNALS, "gap": DEDUP_GAP},
            "unchanged_triggers": ["liquidity_sweep"] + NAMED_SIGNALS + ["local_extreme"],
            "source_screen": "research_eth_v_rebound_multitrigger_feeder_role_screen_20260901.py",
        },
        "label_contract": {"V자반등": "outcome==V자반등", "지지/횡보": "outcome==지지/횡보", "제외": "outcome==애매(제외권)"},
        "outcome_formula": "v7b, reused verbatim from research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome, UNCHANGED",
        "triggers": list(down_triggers.keys()),
        "eth_period": {"start": str(eth["timestamp"].min()), "end": str(eth["timestamp"].max())},
        "total_candidates": int(len(labels)),
        "outcome_counts": outcome_counts,
        "outcome_rate": {k: round(v / len(labels), 4) for k, v in outcome_counts.items()} if len(labels) else {},
        "by_trigger": by_trigger_v_rebound,
        "dedup_stats": dedup_stats,
        "comparison_vs_original_9trigger": {
            "original_total_candidates": original["total_candidates"],
            "original_v_rebound_rate": original["outcome_rate"]["V자반등"],
            "new_total_candidates": int(len(labels)),
            "new_v_rebound_rate": round(outcome_counts.get("V자반등", 0) / len(labels), 4) if len(labels) else None,
            "delta_candidates": int(len(labels)) - original["total_candidates"],
        },
        "future_features_used_for_labels": True,
        "output_labels": str(label_path),
        "runtime_sec": round(time.time() - t0, 1),
        "NEXT_STEP_GATE": "label construction only -- per feedback_visual_verification_chart_gate_explain_before_proceed, "
                           "a 20-example visual verification chart must be shown and explained, and the user must "
                           "explicitly approve, before any feature-building/TabPFN-training step proceeds. This is a "
                           "SEPARATE candidate pool from the live 9-trigger one -- nothing live has been changed.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n총 후보: {len(labels)}건 (기존 9트리거: {original['total_candidates']}건, "
          f"delta={len(labels)-original['total_candidates']:+d})")
    for k, v in outcome_counts.items():
        print(f"  {k}: {v}건 ({v/len(labels)*100:.1f}%)")
    print(f"  V자반등 비율: 기존 {original['outcome_rate']['V자반등']*100:.2f}% -> 신규 {outcome_counts.get('V자반등',0)/len(labels)*100:.2f}%")
    print("\n트리거별:")
    for name, s in by_trigger_v_rebound.items():
        print(f"  {name:24s}: 후보 {s['n_candidates']:6d}건, V자반등 {s['n_v_rebound']:5d}건 ({s['rate']*100:.1f}%)")
    print(f"\n산출물: {label_path}, {OUT_DIR}/report.json")
    print(f"실행시간: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
