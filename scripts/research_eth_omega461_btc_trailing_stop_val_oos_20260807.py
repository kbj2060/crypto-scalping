"""RESEARCH ONLY -- apply the BTC gate-G1 trailing stop to ETH Omega4.6.1 and report VAL and OOS.

The trailing stop that survived the 2026-08-07 BTC gate sweep is a FIXED-DISTANCE rule: once
favorable excursion reaches `activate * take_profit`, trail the stop `trail * |stop_loss|` behind
the running favorable extreme. On BTC it lifted win rate 35.5%->55.5%, cut MDD -24%->-14% and
roughly doubled the gross-edge t-stat, while leaving PnL roughly unchanged
([[project-trailing-stop-risk-lever-keep-alive-20260807]]).

This is NOT the configuration this repo already tried on ETH. On 2026-07-21
(`research_eth_omega461_exit_sweep_20260721.py` experiment B) the rule tested was a PROPORTIONAL
giveback (exit once profit falls to `retain * peak MFE`), swept on VAL only, and the single
VAL-selected candidate (h48qual, activate 0.8 / retain 0.4) then FAILED on OOS: 9.49% -> 3.97%.
zig075's trailing variants were never OOS-confirmed at all.

Two things are done differently here, both aimed at not repeating that mistake:

1. The BTC fixed-distance parameterisation is used (`trailing_trail_frac`), including the exact
   BTC operating point activate=0.3, trail=0.5, plus a small grid around it.
2. **Every configuration is reported on BOTH VAL and OOS.** Nothing is selected on VAL and then
   presented as if OOS confirmed it -- with 14-44 trades per window that selection step is what
   produced the 2026-07-21 reversal.

Uses the frozen parent bundles / risk sidecars / prediction CSVs, no retraining, and the certified
bar-by-bar replay loop (fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false).

CAVEATS, both inherited and unavoidable here:
- VAL is 2025-10-01..2025-12-31, one month short of the canonical 2025-09-01 start, because the
  frozen OOF prediction CSVs only exist from 2025-10-01 (2025-09 was inside the parent's TRAIN
  split). Same caveat the original sweep flagged.
- Results are PER COMPONENT (h48qual, zig075). Live Omega4.6.1 runs the combined two-component
  router, so these are not the portfolio-level numbers.
- Trade counts are 14-53 per window. Nothing here is powered enough to promote on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_omega461_btc_trailing_20260807"

# (activate_frac, trail_frac); the BTC operating point is (0.3, 0.5)
TRAILING_GRID = [(0.3, 0.5), (0.3, 1.0), (0.5, 0.5), (0.5, 1.0), (0.8, 0.5), (0.8, 1.0)]


def _run_split(split_name: str, frame, components: dict) -> list[dict]:
    rows = []
    for name, cfg in components.items():
        pred_name = "validation_predictions" if split_name == "val" else "oos_predictions"
        pred = sweep.EXT_PRED_DIR / name / f"{pred_name}_{cfg['q_tag']}.csv"
        print(f"stage=prep {name} {split_name}", flush=True)
        p = sweep.prep_component(name, cfg, frame, pred, oof=(split_name == "val"))
        common = dict(
            risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
            exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"],
            cost_mult=sweep.COST_MULT, notional_scaled_sltp=p["notional_scaled_sltp"],
            device=sweep.DEVICE,
        )
        m, _ = sweep.replay_exit_variant(p["frame"], p["x"], p["dec"], p["loaded"], **common)
        rows.append({"component": name, "split": split_name, "config": "baseline",
                     "activate": None, "trail": None, **_keep(m)})
        print(f"  {name} {split_name} baseline: pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% "
              f"trades={m['trades']} wr={m['wr']:.3f}", flush=True)
        for act, trail in TRAILING_GRID:
            m, _ = sweep.replay_exit_variant(
                p["frame"], p["x"], p["dec"], p["loaded"],
                trailing_activate_frac=act, trailing_trail_frac=trail, **common)
            rows.append({"component": name, "split": split_name,
                         "config": f"trail{trail}@{act}TP", "activate": act, "trail": trail,
                         **_keep(m)})
            print(f"  {name} {split_name} trail{trail}@{act}TP: pnl={m['pnl']:.2f}% "
                  f"mdd={m['mdd']:.2f}% trades={m['trades']} wr={m['wr']:.3f}", flush=True)
    return rows


def _keep(m: dict) -> dict:
    return {"pnl": m["pnl"], "mdd": m["mdd"], "trades": m["trades"], "wr": m["wr"],
            "avg_hold_bars": m["avg_hold_bars"], "exit_reasons": json.dumps(m["exit_reasons"])}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END,
                                 base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END,
                                 base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"VAL rows={len(val_frame)} [{val_frame['timestamp'].min()} .. {val_frame['timestamp'].max()}]")
    print(f"OOS rows={len(oos_frame)} [{oos_frame['timestamp'].min()} .. {oos_frame['timestamp'].max()}]")

    rows = _run_split("val", val_frame, sweep.COMPONENTS) + _run_split("oos", oos_frame, sweep.COMPONENTS)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "eth_omega461_btc_trailing_val_oos.csv", index=False)

    print("\n=== ETH Omega4.6.1 + BTC fixed-distance trailing stop ===")
    for comp in sweep.COMPONENTS:
        print(f"\n[{comp}]")
        hdr = (f"{'config':<18}{'VAL pnl%':>10}{'VAL mdd%':>10}{'VAL tr':>8}{'VAL wr':>8}"
               f"{'OOS pnl%':>10}{'OOS mdd%':>10}{'OOS tr':>8}{'OOS wr':>8}")
        print(hdr)
        print("-" * len(hdr))
        for cfg in ["baseline"] + [f"trail{t}@{a}TP" for a, t in TRAILING_GRID]:
            v = df[(df["component"] == comp) & (df["split"] == "val") & (df["config"] == cfg)]
            o = df[(df["component"] == comp) & (df["split"] == "oos") & (df["config"] == cfg)]
            if v.empty or o.empty:
                continue
            v, o = v.iloc[0], o.iloc[0]
            print(f"{cfg:<18}{v['pnl']:>10.2f}{v['mdd']:>10.2f}{v['trades']:>8.0f}{v['wr']:>8.3f}"
                  f"{o['pnl']:>10.2f}{o['mdd']:>10.2f}{o['trades']:>8.0f}{o['wr']:>8.3f}")

    print("\nVAL = 2025-10-01..2025-12-31 (one month short of canonical, frozen-OOF constraint)")
    print("OOS = 2026-01-01..2026-03-31. Per component, not the combined live router.")
    print("fresh_forward_bar_by_bar=true trade_ledgers_used_as_input=false "
          "saved_parent_exit_timestamps_used=false future_rows_used_for_entry=false")
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
