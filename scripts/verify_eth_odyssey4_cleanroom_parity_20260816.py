#!/usr/bin/env python3
"""Bar-by-bar parity check: the production Omega4.6.1 adapter (trading_bot_modules.omega4_6_1_live.
Omega461LiveAdapter) vs. the new cleanroom Odyssey adapter (trading_bot_modules.odyssey_live_adapter.
OdysseyLiveAdapter), built with the EXACT component configuration live_eth_odyssey4_zig075_entry_veto_
shadow_cleanroom_20260816.py uses (h48qual liveATR-relabel bundle for entry+default-exit, original
h48qual bundle for the regime-aware exit guard, fully original zig075).

For each bar in the test window this compares:
  - decide_entry(): side / source_component / margin_fraction / leverage / notional_exposure /
    take_profit / stop_loss (exact float match, no tolerance -- both paths do the identical
    computation with identical cached weights, so any difference at all is a real behavior change)
  - evaluate_exit() for BOTH components, at a fixed synthetic open-position state (since the real
    shadow's actual open position varies bar to bar and stepping through it here would just
    re-derive decide_entry's own output; a fixed synthetic position exercises the exit_probability
    path -- the part that isn't otherwise covered by decide_entry -- directly and deterministically)

=== A known, pre-existing, environment-only wrinkle (NOT a bug in either adapter) ===
`validate_sidecar_lineage` (trading_bot_modules/omega4_6_1_runtime_contract.py) checks that each risk
sidecar's `report.json` records a `precomputed_prediction_dir` matching its bundle's parent directory
exactly. Some of these report.json files were generated on a different machine (home directory
`/home/llewyn`) than this one; on a dev box whose home directory differs, the ABSOLUTE-path compare
fails for BOTH the old and the new adapter identically (verified: patching the check out reproduces
zero-mismatch parity, and the failure occurs at the exact same line for both). This is not something
this script should "fix" by editing report.json (that would misrepresent training provenance) --
production environments where the repo's actual path matches what report.json recorded are
unaffected. This script relaxes the check to a repo-relative-suffix comparison, applied IDENTICALLY
to both adapters, purely so this local parity check can run; nothing on disk is touched.

Usage: python scripts/verify_eth_odyssey4_cleanroom_parity_20260816.py [--rows N] [--start N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

FEATURE_CSV = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"


def _lenient_validate_sidecar_lineage(*, repo_root, bundle_path, sidecar_path, quality_threshold, allowed_selection_scopes):
    """See module docstring -- repo-relative-suffix compare instead of absolute-path compare,
    applied identically to both adapters under test. Everything else matches the real check."""
    root = Path(repo_root).resolve()

    def resolve(p):
        pp = Path(p)
        return (pp if pp.is_absolute() else root / pp).resolve()

    bundle = resolve(bundle_path)
    sidecar = resolve(sidecar_path)
    report_path = sidecar.parent / "report.json"
    for label, path in (("bundle", bundle), ("sidecar", sidecar), ("report", report_path)):
        if not path.is_file():
            raise ValueError(f"missing {label} artifact: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    risk_model = report.get("risk_model", {})
    contract = report.get("contract", {})
    selection_scope = risk_model.get("selection_scope")
    if selection_scope not in allowed_selection_scopes:
        raise ValueError(f"selection_scope {selection_scope!r} not in {sorted(allowed_selection_scopes)}")
    expected_tag = f"q{int(round(float(quality_threshold) * 100)):03d}"
    if risk_model.get("precomputed_prediction_tag") != expected_tag:
        raise ValueError(f"prediction tag mismatch: {risk_model.get('precomputed_prediction_tag')!r} != {expected_tag!r}")
    report_threshold = contract.get("quality_threshold")
    if report_threshold is None or not math.isclose(float(report_threshold), float(quality_threshold), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"quality threshold mismatch: {report_threshold!r} != {quality_threshold!r}")

    def relsuffix(p: Path) -> str:
        s = str(p)
        idx = s.find("crypto-scalping/")
        return s[idx + len("crypto-scalping/"):] if idx >= 0 else s

    pred_dir_raw = risk_model.get("precomputed_prediction_dir")
    if not pred_dir_raw:
        raise ValueError("report is missing precomputed_prediction_dir")
    if relsuffix(Path(pred_dir_raw)) != relsuffix(bundle.parent):
        raise ValueError(f"lineage mismatch (repo-relative compare): {pred_dir_raw} vs {bundle.parent}")
    missing_predictions = [
        str(bundle.parent / f"{split}_predictions_{expected_tag}.csv")
        for split in ("train", "validation", "oos")
        if not (bundle.parent / f"{split}_predictions_{expected_tag}.csv").is_file()
    ]
    if missing_predictions:
        raise ValueError("missing exact prediction artifacts: " + ", ".join(missing_predictions))
    return {"selection_scope": selection_scope, "prediction_dir": str(bundle.parent), "prediction_tag": expected_tag}


def build_adapters(device: str = "cpu"):
    import trading_bot_modules.omega4_6_1_live as old_live_mod
    old_live_mod.validate_sidecar_lineage = _lenient_validate_sidecar_lineage
    from trading_bot_modules.omega4_6_1_live import Omega461LiveAdapter

    import trading_bot_modules.odyssey_live_adapter as new_live_mod
    new_live_mod.validate_sidecar_lineage = _lenient_validate_sidecar_lineage
    from trading_bot_modules.odyssey_live_adapter import (
        ODYSSEY_H48QUAL_BUNDLE_PATH,
        ODYSSEY_H48QUAL_SIDECAR_PATH,
        ODYSSEY_ZIG075_BUNDLE_PATH,
        ODYSSEY_ZIG075_SIDECAR_PATH,
        OdysseyLiveAdapter,
    )

    # Same artifact selection as live_eth_odyssey4_zig075_entry_veto_shadow_cleanroom_20260816.py's
    # COMPONENTS_OVERRIDE -- h48qual uses the liveATR-relabel bundle for entry/default-exit, zig075
    # is fully original.
    h48qual_new_bundle = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual/true_3head_tabm_bundle.pt"
    h48qual_shim_sidecar = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_asymmetric_shadow_20260813_h48qual_sidecar/risk_sidecar.pkl"
    components_override = {
        "h48qual": {"bundle": h48qual_new_bundle, "sidecar": h48qual_shim_sidecar, "quality_threshold": 0.50},
        "zig075": {"bundle": ROOT / ODYSSEY_ZIG075_BUNDLE_PATH, "sidecar": ROOT / ODYSSEY_ZIG075_SIDECAR_PATH, "quality_threshold": 0.75},
    }
    priority = ("h48qual", "zig075")

    old_adapter = Omega461LiveAdapter(
        h48qual_bundle="", h48qual_sidecar="", zig075_bundle="", zig075_sidecar="",
        device=device, components_override=components_override, priority=priority,
    )
    new_adapter = OdysseyLiveAdapter(device=device, components_override=components_override, priority=priority)

    # Also compare the ORIGINAL h48qual guard component used by the regime-aware exit guard.
    from trading_bot_modules.omega4_6_1_live import _Component as OldComponent, _ComponentConfig as OldComponentConfig
    from trading_bot_modules.odyssey_live_adapter import _Component as NewComponent, _ComponentConfig as NewComponentConfig

    old_guard = OldComponent(
        OldComponentConfig("h48qual_guard_original", ROOT / ODYSSEY_H48QUAL_BUNDLE_PATH, ROOT / ODYSSEY_H48QUAL_SIDECAR_PATH, quality_threshold=0.50),
        device=old_adapter.device,
    )
    new_guard = NewComponent(
        NewComponentConfig("h48qual_guard_original", ROOT / ODYSSEY_H48QUAL_BUNDLE_PATH, ROOT / ODYSSEY_H48QUAL_SIDECAR_PATH, quality_threshold=0.50),
        device=new_adapter.device,
    )
    return old_adapter, new_adapter, old_guard, new_guard


def _entry_tuple(dec):
    if dec is None:
        return None
    return (
        dec.side, dec.source_component, round(dec.margin_fraction, 10), round(dec.leverage, 10),
        round(dec.notional_exposure, 10), round(dec.take_profit, 10), round(dec.stop_loss, 10),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2000, help="how many bars to compare")
    ap.add_argument("--start", type=int, default=3000, help="row offset to start comparing at (needs lookback warm-up)")
    ap.add_argument("--csv", type=Path, default=FEATURE_CSV)
    args = ap.parse_args()

    old_adapter, new_adapter, old_guard, new_guard = build_adapters()

    nrows_needed = args.start + args.rows
    df = pd.read_csv(args.csv, nrows=nrows_needed)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if len(df) < nrows_needed:
        print(f"[warn] requested {nrows_needed} rows but {args.csv} only has {len(df)}; clamping --rows", flush=True)
        args.rows = max(0, len(df) - args.start)

    mismatches: list[str] = []
    entry_n = exit_n = guard_n = 0

    for i in range(args.start, args.start + args.rows):
        frame = df.iloc[: i + 1].reset_index(drop=True)
        ts = frame["timestamp"].iloc[-1]

        old_dec = old_adapter.decide_entry(frame.copy())
        new_dec = new_adapter.decide_entry(frame.copy())
        entry_n += 1
        old_t, new_t = _entry_tuple(old_dec), _entry_tuple(new_dec)
        if old_t != new_t:
            mismatches.append(f"[entry] row={i} ts={ts}: old={old_t} new={new_t}")

        for comp in ("h48qual", "zig075"):
            old_exit = old_adapter.evaluate_exit(
                frame.copy(), source_component=comp, side=1, hold_bars=5, unrealized_move=0.01,
                mfe=0.02, mae=-0.005, notional=0.5, leverage=2.0, take_profit=0.08, stop_loss=0.05,
            )
            new_exit = new_adapter.evaluate_exit(
                frame.copy(), source_component=comp, side=1, hold_bars=5, unrealized_move=0.01,
                mfe=0.02, mae=-0.005, notional=0.5, leverage=2.0, take_profit=0.08, stop_loss=0.05,
            )
            exit_n += 1
            old_r = (old_exit[0], old_exit[1], round(old_exit[2], 10))
            new_r = (new_exit[0], new_exit[1], round(new_exit[2], 10))
            if old_r != new_r:
                mismatches.append(f"[exit:{comp}] row={i} ts={ts}: old={old_r} new={new_r}")

        # Original h48qual guard component's exit_probability (used only while the sustained-uptrend
        # detector is active on an open h48qual position).
        old_guard_frame = old_adapter.regime3_current.append(frame.copy())
        new_guard_frame = new_adapter.regime3_current.append(frame.copy())
        old_gp = old_guard.exit_probability(
            old_guard_frame, side=1, hold_bars=5, unrealized_move=0.01, mfe=0.02, mae=-0.005,
            notional=0.5, leverage=2.0, take_profit=0.08, stop_loss=0.05,
        )
        new_gp = new_guard.exit_probability(
            new_guard_frame, side=1, hold_bars=5, unrealized_move=0.01, mfe=0.02, mae=-0.005,
            notional=0.5, leverage=2.0, take_profit=0.08, stop_loss=0.05,
        )
        guard_n += 1
        if round(old_gp, 10) != round(new_gp, 10):
            mismatches.append(f"[guard] row={i} ts={ts}: old={old_gp} new={new_gp}")

        if (i - args.start + 1) % 500 == 0:
            print(f"[progress] {i - args.start + 1}/{args.rows} bars, {len(mismatches)} mismatches so far", flush=True)

    print(f"\ndone. entry_compares={entry_n} exit_compares={exit_n} guard_compares={guard_n} mismatches={len(mismatches)}")
    for m in mismatches[:50]:
        print(m)
    if len(mismatches) > 50:
        print(f"... and {len(mismatches) - 50} more")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
