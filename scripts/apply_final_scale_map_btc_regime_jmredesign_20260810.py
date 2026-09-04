"""Final gated replay (duration gate + scale map + exit threshold) under the redesigned-JM regime.

Thin wrapper over scripts/apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806.py.
That script rebuilds its frames through the BTC omega module, which loads the regime3-current
overlay from its module-level default -- the LIVE wide24 CSVs. Those files are missing from disk
(they vanished mid-session; only the 2024 joblib and a .bak survive), and in any case this replay
must run on the CANDIDATE's regime overlay, not the incumbent's. So the override is applied on the
omega module before the script's main() runs, exactly as the parent and sidecar wrappers do.

This is the stage that produces the number the live BTC promotion was judged on (+10.76% OOS
extended): the sidecar's own `selected` metrics are pre-duration-gate and pre-scale-map, so they
are NOT comparable to it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806 as replay  # noqa: E402

TAG = "jmredesign_20260810"
SUP = ROOT / "data/ensemble/supervised"

# _load_omega_frames() reads the BARE module globals, so the override must land on that exact
# module object; going through an importing script's alias silently does nothing.
import train_eval_omega1_2_tabm_diffusion_risk_btc_swingtransition_20260806 as omega_mod  # noqa: E402

omega_mod.REGIME3_CURRENT_2025 = SUP / f"btc_regime3_current_hmm_{TAG}_2025_maskedname.csv"
omega_mod.REGIME3_CURRENT_2026 = SUP / f"btc_regime3_current_hmm_{TAG}_2026_maskedname.csv"

if __name__ == "__main__":
    raise SystemExit(replay.main())
