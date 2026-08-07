#!/usr/bin/env python3
"""Follow-up to research_eth_omega461_exit_sweep_20260721.py: OOS confirmation for the two VAL
candidates that looked promising (beat baseline on VAL) -- RESEARCH ONLY, no live files touched.

  1) zig075, Experiment A, EXIT_THRESHOLD=0.80 (VAL: pnl 53.7% vs baseline 40.3%, same MDD -13.07%)
  2) h48qual, Experiment B, trailing activate=0.8 retain=0.4 (VAL: pnl 13.5% vs baseline 5.45%,
     MDD -8.00% vs -11.62%, i.e. beat baseline on BOTH metrics)

Reuses prep_component/replay_exit_variant from the main sweep script unmodified.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(Path(r"\\wsl.localhost\ubuntu\home\llewyn\crypto-scalping\scripts")))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

R = Path(r"\\wsl.localhost\ubuntu\home\llewyn\crypto-scalping")


def main() -> int:
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)

    # zig075 candidate: exit_threshold=0.80
    cfg = sweep.COMPONENTS["zig075"]
    pred = sweep.EXT_PRED_DIR / "zig075" / f"oos_predictions_{cfg['q_tag']}.csv"
    print("stage=prep zig075 OOS", flush=True)
    p = sweep.prep_component("zig075", cfg, oos_frame, pred, oof=False)
    print("stage=replay zig075 OOS baseline(0.95)", flush=True)
    m_base, _ = sweep.replay_exit_variant(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        exit_threshold=0.95, fee=p["fee"], slip=p["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    print(f"zig075 OOS baseline(0.95): pnl={m_base['pnl']:.2f}% mdd={m_base['mdd']:.2f}% trades={m_base['trades']} wr={m_base['wr']:.3f}", flush=True)
    print("stage=replay zig075 OOS candidate(0.80)", flush=True)
    m_cand, _ = sweep.replay_exit_variant(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        exit_threshold=0.80, fee=p["fee"], slip=p["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    print(f"zig075 OOS candidate(0.80): pnl={m_cand['pnl']:.2f}% mdd={m_cand['mdd']:.2f}% trades={m_cand['trades']} wr={m_cand['wr']:.3f}", flush=True)

    # h48qual candidate: trailing activate=0.8 retain=0.4
    cfg2 = sweep.COMPONENTS["h48qual"]
    pred2 = sweep.EXT_PRED_DIR / "h48qual" / f"oos_predictions_{cfg2['q_tag']}.csv"
    print("stage=prep h48qual OOS", flush=True)
    p2 = sweep.prep_component("h48qual", cfg2, oos_frame, pred2, oof=False)
    print("stage=replay h48qual OOS baseline(0.95, no trailing)", flush=True)
    m2_base, _ = sweep.replay_exit_variant(
        p2["frame"], p2["x"], p2["dec"], p2["loaded"], risk_margin_fraction=p2["margin"], risk_leverage=p2["leverage"],
        exit_threshold=0.95, fee=p2["fee"], slip=p2["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p2["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    print(f"h48qual OOS baseline(0.95): pnl={m2_base['pnl']:.2f}% mdd={m2_base['mdd']:.2f}% trades={m2_base['trades']} wr={m2_base['wr']:.3f}", flush=True)
    print("stage=replay h48qual OOS candidate(trailing 0.8/0.4)", flush=True)
    m2_cand, _ = sweep.replay_exit_variant(
        p2["frame"], p2["x"], p2["dec"], p2["loaded"], risk_margin_fraction=p2["margin"], risk_leverage=p2["leverage"],
        exit_threshold=0.95, fee=p2["fee"], slip=p2["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p2["notional_scaled_sltp"], device=sweep.DEVICE,
        trailing_activate_frac=0.8, trailing_retain_frac=0.4,
    )
    print(f"h48qual OOS candidate(trailing 0.8/0.4): pnl={m2_cand['pnl']:.2f}% mdd={m2_cand['mdd']:.2f}% trades={m2_cand['trades']} wr={m2_cand['wr']:.3f}", flush=True)

    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
