#!/usr/bin/env python3
"""RESEARCH ONLY -- evaluate the PINNED-102 retrained bundles the same way
research_eth_omega461_retrain_entry_from_bundle_20260727.py evaluated the 2026-07-21 (172-col)
bundles: entry AND exit both sourced from the retrained bundle, replayed at EXIT_THRESHOLD=0.95.

The 172-col run collapsed on VAL for all 12 bundles, but that comparison was confounded -- those
bundles carried m7/NF features the live adapter rejects, and their training frame was 78,509 rows
against live's 183,936. `train_eval_omega4_3head_parent72_pinned102_20260727.py` removes both
defects. This script answers the question that confound blocked:

  control (live labels, pinned contract, fresh training run) vs the live checkpoint
      -> isolates pure retraining variance
  exit-label variants vs that control
      -> isolates the exit-label effect on ENTRY through the shared trunk

VAL-first funnel: OOS is touched only for variants that beat the live control on both pnl and mdd.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.
Does NOT touch any live file or checkpoint. Research artifact only -- no promotion-gate claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as base  # noqa: E402
import research_eth_omega461_retrain_entry_from_bundle_20260727 as efb  # noqa: E402


def _bundle(suffix: str) -> Path:
    return ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_{suffix}/true_3head_tabm_bundle.pt"


# Filled in from whatever pinned102 runs exist on disk.
CANDIDATES: dict[str, dict[str, Path]] = {
    "pinned102_control": {
        "h48qual": _bundle("pinned102_20260727_h48qual_control"),
        "zig075": _bundle("pinned102_20260727_zig075_control"),
    },
    "pinned102_gb045": {
        "h48qual": _bundle("pinned102_20260727_h48qual_gb045"),
        "zig075": _bundle("pinned102_20260727_zig075_gb045"),
    },
    "pinned102_tw08": {
        "h48qual": _bundle("pinned102_20260727_h48qual_tw08"),
        "zig075": _bundle("pinned102_20260727_zig075_tw08"),
    },
    # Same live-equivalent control, but trained on the recovered 2024+2025 tape (183,818 train
    # rows) instead of the 2025-only tape (78,509 train rows) the three variants above used.
    "pinned102_2024tape_control": {
        "h48qual": _bundle("pinned102_2024tape_20260727_h48qual_control"),
        "zig075": _bundle("pinned102_2024tape_20260727_zig075_control"),
    },
    # Same data/seed/labels as pinned102_2024tape_control, but exit_loss_weight=0 during
    # training (exit head receives zero gradient) -- isolates "exit head present vs absent"
    # from both the exit-label axis (round 18) and plain retraining variance (round 19).
    "pinned102_2024tape_noexithead": {
        "h48qual": _bundle("pinned102_2024tape_20260727_h48qual_noexithead"),
        "zig075": _bundle("pinned102_2024tape_20260727_zig075_noexithead"),
    },
    # Reproduction check: identical config to the row above but --seed 260728 instead of the
    # trainer's default 260620.
    "pinned102_2024tape_noexithead_seed2": {
        "h48qual": _bundle("pinned102_2024tape_20260727_h48qual_noexithead_seed2"),
        "zig075": _bundle("pinned102_2024tape_20260727_zig075_noexithead_seed2"),
    },
    "pinned102_2024tape_noexithead_seed_260729": {
        "h48qual": _bundle("pinned102_2024tape_20260727_h48qual_noexithead_seed_260729"),
        "zig075": _bundle("pinned102_2024tape_20260727_zig075_noexithead_seed_260729"),
    },
    "pinned102_2024tape_noexithead_seed_260730": {
        "h48qual": _bundle("pinned102_2024tape_20260727_h48qual_noexithead_seed_260730"),
        "zig075": _bundle("pinned102_2024tape_20260727_zig075_noexithead_seed_260730"),
    },
    "pinned102_2024tape_noexithead_seed_260731": {
        "h48qual": _bundle("pinned102_2024tape_20260727_h48qual_noexithead_seed_260731"),
        "zig075": _bundle("pinned102_2024tape_20260727_zig075_noexithead_seed_260731"),
    },
    # New triple-barrier DIRECTION label (max-density config), replacing zigzag_action, 5 seeds.
    "tripbar_seed_260620": {
        "h48qual": _bundle("pinned102_2024tape_tripbar_20260728_h48qual_seed_260620"),
        "zig075": _bundle("pinned102_2024tape_tripbar_20260728_zig075_seed_260620"),
    },
    "tripbar_seed_260728": {
        "h48qual": _bundle("pinned102_2024tape_tripbar_20260728_h48qual_seed_260728"),
        "zig075": _bundle("pinned102_2024tape_tripbar_20260728_zig075_seed_260728"),
    },
    "tripbar_seed_260729": {
        "h48qual": _bundle("pinned102_2024tape_tripbar_20260728_h48qual_seed_260729"),
        "zig075": _bundle("pinned102_2024tape_tripbar_20260728_zig075_seed_260729"),
    },
    "tripbar_seed_260730": {
        "h48qual": _bundle("pinned102_2024tape_tripbar_20260728_h48qual_seed_260730"),
        "zig075": _bundle("pinned102_2024tape_tripbar_20260728_zig075_seed_260730"),
    },
    "tripbar_seed_260731": {
        "h48qual": _bundle("pinned102_2024tape_tripbar_20260728_h48qual_seed_260731"),
        "zig075": _bundle("pinned102_2024tape_tripbar_20260728_zig075_seed_260731"),
    },
}

# Grid retrain: matched direction+quality triple-barrier pairs (zig075 now gets its OWN quality
# label instead of same_as_direction), 3 barrier widths x 5 seeds x 2 components.
for _grid_cfg in ("dense", "medium", "sparse"):
    for _seed in (260620, 260728, 260729, 260730, 260731):
        CANDIDATES[f"tripbargrid_{_grid_cfg}_seed_{_seed}"] = {
            "h48qual": _bundle(f"pinned102_2024tape_tripbargrid_20260728_{_grid_cfg}_h48qual_seed_{_seed}"),
            "zig075": _bundle(f"pinned102_2024tape_tripbargrid_20260728_{_grid_cfg}_zig075_seed_{_seed}"),
        }

OUT_DIR = ROOT / "tmp/research_20260727/pinned102_entry_eval_20260727"


def available(cname: str) -> dict[str, Path]:
    out = {}
    for name, comp in CANDIDATES.items():
        p = comp.get(cname)
        if p is not None and p.exists():
            out[name] = p
    return out


def run_component(cname: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = dict(base.COMPONENTS[cname])
    cfg["_component"] = cname
    frame_full = efb.load_split(split)
    fsrc = efb.frozen_src(cfg, split)
    frame, fsrc = efb.aligned_frame(frame_full, fsrc)
    oof = split == "VAL"

    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    p_ctrl = efb.prep_from_src(cfg, frame, fsrc, cfg["bundle"], oof=oof)
    m_ctrl = efb.replay(p_ctrl)
    side_live = pd.to_numeric(p_ctrl["dec"]["side"], errors="raise").to_numpy()
    rows.append({"variant": "live_checkpoint", "component": cname, "split": split,
                 "base_feature_count": len(p_ctrl["x"].columns) - 13, **m_ctrl,
                 "exit_reasons": json.dumps(m_ctrl["exit_reasons"]), "side_diff_rows": 0})

    rsrc = efb.bundle_src(cfg["bundle"], frame, quality_threshold=cfg["quality_threshold"], split=split)
    p_regen = efb.prep_from_src(cfg, frame, rsrc, cfg["bundle"], oof=oof)
    m_regen = efb.replay(p_regen)
    side_regen = pd.to_numeric(p_regen["dec"]["side"], errors="raise").to_numpy()
    checks.append({"check": "regen_reproduces_frozen_csv", "component": cname, "split": split,
                   "pnl_frozen": m_ctrl["pnl"], "pnl_regen": m_regen["pnl"],
                   "side_diff_rows": int((side_regen != side_live).sum()),
                   "pnl_close": bool(abs(m_ctrl["pnl"] - m_regen["pnl"]) < 0.01)})

    for name, bpath in available(cname).items():
        vsrc = efb.bundle_src(bpath, frame, quality_threshold=cfg["quality_threshold"], split=split)
        p_var = efb.prep_from_src(cfg, frame, vsrc, bpath, oof=oof)
        m_var = efb.replay(p_var)
        side_var = pd.to_numeric(p_var["dec"]["side"], errors="raise").to_numpy()
        rows.append({"variant": name, "component": cname, "split": split,
                     "base_feature_count": len(p_var["x"].columns) - 13, **m_var,
                     "exit_reasons": json.dumps(m_var["exit_reasons"]),
                     "side_diff_rows": int((side_var != side_live).sum())})
        print(f"  {name:22s} {cname:8s} pnl={m_var['pnl']:8.3f} mdd={m_var['mdd']:8.3f} "
              f"trades={m_var['trades']:3d} side_diff={int((side_var != side_live).sum()):5d}/{len(side_var)}",
              flush=True)
    return pd.DataFrame(rows), pd.DataFrame(checks)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "component", "split", "base_feature_count", "pnl", "mdd", "trades", "wr", "side_diff_rows"]

    comps = [c for c in ("h48qual", "zig075") if available(c)]
    if not comps:
        raise SystemExit("no pinned102 bundles found on disk")
    print(f"stage=components {comps}", flush=True)

    val_rows, val_checks = [], []
    for cname in comps:
        print(f"stage=val component={cname}", flush=True)
        r, c = run_component(cname, "VAL")
        val_rows.append(r)
        val_checks.append(c)
    val = pd.concat(val_rows, ignore_index=True)
    chk = pd.concat(val_checks, ignore_index=True)
    print(chk.to_string(index=False), flush=True)
    print(val[cols].to_string(index=False), flush=True)
    val.to_csv(OUT_DIR / "pinned102_entry_VAL.csv", index=False)
    chk.to_csv(OUT_DIR / "sanity_regen_reproduces_frozen.csv", index=False)

    winners = []
    for cname in comps:
        sub = val[val["component"] == cname]
        ctrl = sub[sub["variant"] == "live_checkpoint"].iloc[0]
        for _, row in sub[sub["variant"] != "live_checkpoint"].iterrows():
            if row["pnl"] > ctrl["pnl"] and row["mdd"] >= ctrl["mdd"] - 1e-9:
                winners.append((cname, row["variant"]))
    print(f"stage=val_winners n={len(winners)} {winners}", flush=True)
    pd.DataFrame(winners, columns=["component", "variant"]).to_csv(OUT_DIR / "val_winners.csv", index=False)

    if not winners:
        print("stage=done no VAL winners -- OOS not touched", flush=True)
        return 0

    oos_rows = []
    for cname in sorted({c for c, _ in winners}):
        print(f"stage=oos component={cname}", flush=True)
        r, _c = run_component(cname, "OOS")
        keep = {v for c, v in winners if c == cname} | {"live_checkpoint"}
        oos_rows.append(r[r["variant"].isin(keep)])
    oos = pd.concat(oos_rows, ignore_index=True)
    print(oos[cols].to_string(index=False), flush=True)
    oos.to_csv(OUT_DIR / "pinned102_entry_OOS.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
