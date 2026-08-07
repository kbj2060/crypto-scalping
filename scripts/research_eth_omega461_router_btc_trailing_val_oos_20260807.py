"""RESEARCH ONLY -- BTC gate-G1 trailing stop on the COMBINED Omega4.6.1 router, VAL and OOS.

Follow-up to research_eth_omega461_btc_trailing_stop_val_oos_20260807.py, which measured the two
components (h48qual, zig075) independently. Live Omega4.6.1 is not two independent ledgers: it is a
single-account GREEDY priority router (h48qual first, else zig075, one shared position slot), and
the component-level result was ambiguous -- h48qual looked mildly helped at trail=1.0 while zig075
was clearly hurt at every setting. Which of those dominates can only be answered at the router
level, because the router changes which component's trade actually occupies the slot.

Uses `replay_omega4_6_1_greedy_router_20260706.greedy_replay` (the replay that matches what a real
live system can do, as opposed to the two-independent-ledgers-then-reconcile method used elsewhere
in this lineage) with the trailing stop wired in, plus `sweep.prep_component` for artifact prep
because that version handles the VAL (`oof=True`) prediction layout and the pandas-3.0 string-dtype
issue the older prep does not.

As in the component script, EVERY configuration is reported on BOTH VAL and OOS. Nothing is
selected on VAL. Both the raw router number and the duration-gated number (the "honest final"
convention of the original greedy-router script) are reported.

CAVEATS:
- VAL = 2025-10-01..2025-12-31, one month short of canonical, because the frozen OOF prediction
  CSVs start 2025-10-01 (2025-09 was inside the parent's TRAIN split).
- Trade counts are small (roughly 20-50 per window). Nothing here is powered enough to promote on.
- fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
  saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_omega461_router_btc_trailing_20260807"
TRAILING_GRID = [(0.3, 0.5), (0.3, 1.0), (0.5, 0.5), (0.5, 1.0), (0.8, 0.5), (0.8, 1.0)]


def _prep_router_components(frame: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict]:
    """sweep.prep_component filters the frame to the prediction timestamps, so build both
    components against the INTERSECTION of their kept timestamps -- the router walks one shared
    bar index and every component array must line up with it."""
    prepped = {}
    for name, cfg in sweep.COMPONENTS.items():
        pred_name = "validation_predictions" if split == "val" else "oos_predictions"
        pred = sweep.EXT_PRED_DIR / name / f"{pred_name}_{cfg['q_tag']}.csv"
        print(f"stage=prep {name} {split}", flush=True)
        prepped[name] = sweep.prep_component(name, cfg, frame, pred, oof=(split == "val"))

    common_ts = None
    for p in prepped.values():
        ts = set(p["frame"]["timestamp"])
        common_ts = ts if common_ts is None else (common_ts & ts)
    router_frame = frame[frame["timestamp"].isin(common_ts)].reset_index(drop=True)

    components = {}
    for name, p in prepped.items():
        keep = p["frame"]["timestamp"].isin(common_ts).to_numpy()
        if keep.sum() != len(router_frame):
            raise RuntimeError(f"{name}: alignment failed ({keep.sum()} vs {len(router_frame)})")
        dec = p["dec"].loc[keep].reset_index(drop=True)
        x = p["x"][keep]
        base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(x, p["loaded"])
        lev = p["leverage"] if p["leverage"] is not None else np.ones(len(p["dec"]))
        components[name] = {
            "dec": dec, "margin": np.asarray(p["margin"])[keep],
            "leverage": np.asarray(lev)[keep], "base_np": base_np,
            "exit_runtime": exit_runtime, "pos_idx": pos_idx,
            "route": hard._route_id(router_frame),
            "exit_threshold": sweep.BASELINE_EXIT_THRESHOLD,
        }
    return router_frame, components


def _metrics(ledger: pd.DataFrame, frame: pd.DataFrame) -> dict:
    if len(ledger) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0,
                "pnl_gated": 0.0, "mdd_gated": 0.0, "trades_gated": 0}
    r = ledger["trade_return"].to_numpy(dtype=np.float64)
    curve = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    dd = curve / np.maximum(np.maximum.accumulate(curve), 1e-12) - 1.0
    out = {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0),
           "trades": int(len(r)), "wr": float((r > 0).mean()),
           "src": json.dumps(ledger["source_component"].value_counts().to_dict()),
           "reasons": json.dumps(ledger["reason"].value_counts().to_dict())}

    # duration gate, exactly as the original greedy-router script applies it
    market = frame[["timestamp", "ou_halflife"]].copy()
    lg = ledger.copy()
    lg["entry_timestamp_dt"] = pd.to_datetime(lg["entry_timestamp"])
    lg = lg.merge(market.rename(columns={"timestamp": "entry_timestamp_dt"}),
                  on="entry_timestamp_dt", how="left")
    hit = (lg["ou_halflife"] <= greedy.DURATION_THRESHOLD).fillna(False).to_numpy()
    rg = np.where(hit, 0.0, lg["trade_return"].to_numpy(dtype=np.float64))
    cg = np.concatenate([[1.0], np.cumprod(1.0 + rg)])
    ddg = cg / np.maximum(np.maximum.accumulate(cg), 1e-12) - 1.0
    out.update(pnl_gated=float((cg[-1] - 1.0) * 100.0), mdd_gated=float(ddg.min() * 100.0),
               trades_gated=int((~hit).sum()))
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    rows = []
    for split, (start, end, base, wide) in {
        "val": (sweep.VAL_START, sweep.VAL_END, sweep.BASE_2025, sweep.WIDE24_2025),
        "oos": (sweep.OOS_START, sweep.OOS_END, sweep.BASE_2026, sweep.WIDE24_2026),
    }.items():
        frame = sweep.load_frame(start, end, base_csv=base, wide24_csv=wide)
        router_frame, components = _prep_router_components(frame, split)
        print(f"{split}: router bars={len(router_frame)} "
              f"[{router_frame['timestamp'].min()} .. {router_frame['timestamp'].max()}]", flush=True)

        for cfg_name, act, trail in [("baseline", None, None)] + [
                (f"trail{t}@{a}TP", a, t) for a, t in TRAILING_GRID]:
            _, ledger = greedy.greedy_replay(
                router_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT,
                device=sweep.DEVICE, trailing_activate_frac=act, trailing_trail_frac=trail)
            m = _metrics(ledger, router_frame)
            rows.append({"split": split, "config": cfg_name, "activate": act, "trail": trail, **m})
            print(f"  {split} {cfg_name}: pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% "
                  f"tr={m['trades']} wr={m['wr']:.3f} | gated pnl={m['pnl_gated']:.2f}% "
                  f"mdd={m['mdd_gated']:.2f}% tr={m['trades_gated']} | {m['src']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "router_btc_trailing_val_oos.csv", index=False)

    print("\n=== Omega4.6.1 COMBINED greedy router + BTC fixed-distance trailing stop ===")
    hdr = (f"{'config':<18}{'VAL pnl%':>10}{'VAL mdd%':>10}{'VAL tr':>8}{'VAL wr':>8}"
           f"{'OOS pnl%':>10}{'OOS mdd%':>10}{'OOS tr':>8}{'OOS wr':>8}")
    print(hdr)
    print("-" * len(hdr))
    for cfg_name in ["baseline"] + [f"trail{t}@{a}TP" for a, t in TRAILING_GRID]:
        v = df[(df["split"] == "val") & (df["config"] == cfg_name)]
        o = df[(df["split"] == "oos") & (df["config"] == cfg_name)]
        if v.empty or o.empty:
            continue
        v, o = v.iloc[0], o.iloc[0]
        print(f"{cfg_name:<18}{v['pnl']:>10.2f}{v['mdd']:>10.2f}{v['trades']:>8.0f}{v['wr']:>8.3f}"
              f"{o['pnl']:>10.2f}{o['mdd']:>10.2f}{o['trades']:>8.0f}{o['wr']:>8.3f}")

    print("\n--- with duration gate (the 'honest final' convention of the greedy-router script) ---")
    print(hdr.replace("pnl", "pnlG").replace("mdd", "mddG"))
    print("-" * len(hdr))
    for cfg_name in ["baseline"] + [f"trail{t}@{a}TP" for a, t in TRAILING_GRID]:
        v = df[(df["split"] == "val") & (df["config"] == cfg_name)]
        o = df[(df["split"] == "oos") & (df["config"] == cfg_name)]
        if v.empty or o.empty:
            continue
        v, o = v.iloc[0], o.iloc[0]
        print(f"{cfg_name:<18}{v['pnl_gated']:>10.2f}{v['mdd_gated']:>10.2f}{v['trades_gated']:>8.0f}"
              f"{v['wr']:>8.3f}{o['pnl_gated']:>10.2f}{o['mdd_gated']:>10.2f}{o['trades_gated']:>8.0f}{o['wr']:>8.3f}")

    print("\nVAL = 2025-10-01..2025-12-31 (one month short of canonical, frozen-OOF constraint)")
    print("OOS = 2026-01-01..2026-03-31. Single-account greedy router (h48qual > zig075).")
    print("fresh_forward_bar_by_bar=true trade_ledgers_used_as_input=false "
          "saved_parent_exit_timestamps_used=false future_rows_used_for_entry=false")
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
