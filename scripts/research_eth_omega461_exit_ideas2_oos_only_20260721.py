#!/usr/bin/env python3
"""OOS-confirm-only runner for the 19 VAL winners already found by
research_eth_omega461_exit_ideas2_20260721.py (see tmp/research_20260721/exit_ideas2_VAL.csv).
Avoids re-running the full ~9min VAL grid (VAL results are deterministic and already saved).
RESEARCH ONLY, same conventions as the main script (imported, not duplicated).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_ideas2_20260721 as m  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

WINNERS = [
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.6 ret=0.3 mode=hard thr=0.42",
     "kwargs": dict(trailing_activate_frac=0.6, trailing_retain_frac=0.3, regime_mode="hard", regime_thr=0.42, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.6 ret=0.3 mode=hard thr=0.50",
     "kwargs": dict(trailing_activate_frac=0.6, trailing_retain_frac=0.3, regime_mode="hard", regime_thr=0.50, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.6 ret=0.3 mode=soft thr=None",
     "kwargs": dict(trailing_activate_frac=0.6, trailing_retain_frac=0.3, regime_mode="soft", regime_thr=0.50, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.6 ret=0.5 mode=hard thr=0.42",
     "kwargs": dict(trailing_activate_frac=0.6, trailing_retain_frac=0.5, regime_mode="hard", regime_thr=0.42, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.6 ret=0.5 mode=hard thr=0.50",
     "kwargs": dict(trailing_activate_frac=0.6, trailing_retain_frac=0.5, regime_mode="hard", regime_thr=0.50, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.6 ret=0.5 mode=soft thr=None",
     "kwargs": dict(trailing_activate_frac=0.6, trailing_retain_frac=0.5, regime_mode="soft", regime_thr=0.50, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.8 ret=0.3 mode=hard thr=0.42",
     "kwargs": dict(trailing_activate_frac=0.8, trailing_retain_frac=0.3, regime_mode="hard", regime_thr=0.42, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.8 ret=0.3 mode=hard thr=0.50",
     "kwargs": dict(trailing_activate_frac=0.8, trailing_retain_frac=0.3, regime_mode="hard", regime_thr=0.50, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.8 ret=0.3 mode=soft thr=None",
     "kwargs": dict(trailing_activate_frac=0.8, trailing_retain_frac=0.3, regime_mode="soft", regime_thr=0.50, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.8 ret=0.5 mode=hard thr=0.42",
     "kwargs": dict(trailing_activate_frac=0.8, trailing_retain_frac=0.5, regime_mode="hard", regime_thr=0.42, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.8 ret=0.5 mode=hard thr=0.50",
     "kwargs": dict(trailing_activate_frac=0.8, trailing_retain_frac=0.5, regime_mode="hard", regime_thr=0.50, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "1_regime_veto", "component": "h48qual", "label": "act=0.8 ret=0.5 mode=soft thr=None",
     "kwargs": dict(trailing_activate_frac=0.8, trailing_retain_frac=0.5, regime_mode="soft", regime_thr=0.50, chop_arr=None, bull_arr=None, bear_arr=None)},
    {"idea": "2_partial_scaleout", "component": "zig075", "label": "act=0.8 ret=0.4 close_frac=0.5",
     "kwargs": dict(trailing_activate_frac=0.8, trailing_retain_frac=0.4, partial_close_frac=0.5)},
    {"idea": "3_atr_chandelier", "component": "h48qual", "label": "act=0.6 atr_n=2.0",
     "kwargs": dict(trailing_activate_frac=0.6, atr_n=2.0, atr_arr=None)},
    {"idea": "3_atr_chandelier", "component": "h48qual", "label": "act=0.6 atr_n=3.0",
     "kwargs": dict(trailing_activate_frac=0.6, atr_n=3.0, atr_arr=None)},
    {"idea": "3_atr_chandelier", "component": "h48qual", "label": "act=0.6 atr_n=4.0",
     "kwargs": dict(trailing_activate_frac=0.6, atr_n=4.0, atr_arr=None)},
    {"idea": "3_atr_chandelier", "component": "h48qual", "label": "act=0.8 atr_n=4.0",
     "kwargs": dict(trailing_activate_frac=0.8, atr_n=4.0, atr_arr=None)},
    {"idea": "4_time_decay_tp", "component": "h48qual", "label": "grace=400 rate=0.0005",
     "kwargs": dict(decay_grace_bars=400, decay_rate=0.0005, decay_floor=0.5)},
    {"idea": "4_time_decay_tp", "component": "h48qual", "label": "grace=400 rate=0.0010",
     "kwargs": dict(decay_grace_bars=400, decay_rate=0.0010, decay_floor=0.5)},
]


def main() -> int:
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)
    oos_prepped = m.prep_all(sweep.COMPONENTS, oos_frame, sweep.EXT_PRED_DIR, oof=False)

    oos_rows = []
    for w in WINNERS:
        p = oos_prepped[w["component"]]
        kwargs = dict(w["kwargs"])
        if "atr_arr" in kwargs:
            kwargs["atr_arr"] = p["atr"]
        if "chop_arr" in kwargs:
            reg = p["regime"]
            kwargs["chop_arr"] = reg["chop"]
            kwargs["bull_arr"] = reg["bull"]
            kwargs["bear_arr"] = reg["bear"]
        m_cand, _ = m.run_one(p, **kwargs)
        cleared = m.beats_baseline(w["component"], "OOS", m_cand["pnl"], m_cand["mdd"])
        b = m.BASELINES[(w["component"], "OOS")]
        row = {"idea": w["idea"], "component": w["component"], "label": w["label"],
               "oos_pnl": m_cand["pnl"], "oos_mdd": m_cand["mdd"], "oos_trades": m_cand["trades"], "oos_wr": m_cand["wr"],
               "oos_baseline_reference_pnl": b["pnl"], "oos_baseline_reference_mdd": b["mdd"], "cleared_oos": cleared}
        oos_rows.append(row)
        print(f"idea={w['idea']} component={w['component']} {w['label']} -> "
              f"OOS pnl={m_cand['pnl']:.2f}% mdd={m_cand['mdd']:.2f}% trades={m_cand['trades']} wr={m_cand['wr']:.3f} "
              f"(baseline pnl={b['pnl']:.2f}% mdd={b['mdd']:.2f}%) cleared={cleared}", flush=True)

    pd.DataFrame(oos_rows).to_csv(ROOT / "tmp/research_20260721/exit_ideas2_OOS_confirm.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
