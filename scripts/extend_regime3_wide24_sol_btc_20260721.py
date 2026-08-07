"""Extend SOL/BTC's live current-regime wide24 HMM sidecar (frozen 2024-trained joblib, causal
_transform, no retraining) to match the freshly-extended base feature files (through 2026-07-21).
Mirrors apply_regime3_wide24_sidecar_extended_20260713.py's pattern for ETH.

Data-finality buffer policy (see project-btc-features-2026-drift-root-cause-found-20260801.md):
the HMM transform and its rolling-feature inputs are mathematically causal, but the underlying raw
exchange metrics (OI / long-short-ratio / whale-ratio) are provisional for the most recent hours
after collection and get revised on subsequent pulls -- confirmed empirically as a ~17h revision
zone at the tail of every prior extension. Rows older than DATA_FINALITY_BUFFER_HOURS before the
previous run's own tail are expected to be settled and must reproduce exactly; any diff there is a
real bug and fails fast. Rows inside that buffer are expected to still move and only warn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.experiment_regime3_current_hmm_wide24_20260529 import _transform  # noqa: E402
from scripts.train_regime3_hmm_mamba_20260529 import _read  # noqa: E402

# Observed revision zone was ~16.8h (202 x 5min bars); doubled for margin.
DATA_FINALITY_BUFFER_HOURS = 48
SETTLED_DIFF_TOLERANCE = 1e-9

ASSETS = {
    "sol": {
        "joblib": ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707/regime3_current_sensitive_hmm_wide24_2024.joblib",
        "sources": {
            2025: ROOT / "data/splits/year_oos/sol_features_2025.csv",
            2026: ROOT / "data/splits/year_oos/sol_features_2026.csv",
        },
        "out_dir": ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707",
    },
    "btc": {
        "joblib": ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708/regime3_current_sensitive_hmm_wide24_2024.joblib",
        "sources": {
            2025: ROOT / "data/splits/year_oos/btc_features_2025.csv",
            2026: ROOT / "data/splits/year_oos/btc_features_2026.csv",
        },
        "out_dir": ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708",
    },
}


def main() -> int:
    for asset, cfg in ASSETS.items():
        payload = joblib.load(cfg["joblib"])
        for year, src in cfg["sources"].items():
            frame = _read(src)
            sidecar, ev = _transform(payload, frame)
            out_path = cfg["out_dir"] / f"{src.stem}_regime3_current_sensitive_hmm_wide24.csv"
            if out_path.exists() and year == 2026:
                old = pd.read_csv(out_path, parse_dates=["timestamp"])
                merged = sidecar.merge(old, on="timestamp", how="inner", suffixes=("_new", "_old"))
                settled_cutoff = old["timestamp"].max() - pd.Timedelta(hours=DATA_FINALITY_BUFFER_HOURS)
                is_settled = merged["timestamp"] <= settled_cutoff
                settled_maxdiff = 0.0
                provisional_maxdiff = 0.0
                for c in old.columns:
                    if c == "timestamp" or f"{c}_new" not in merged.columns:
                        continue
                    d = (merged[f"{c}_new"].astype(float) - merged[f"{c}_old"].astype(float)).abs()
                    s = float(d[is_settled].max()) if is_settled.any() else 0.0
                    p = float(d[~is_settled].max()) if (~is_settled).any() else 0.0
                    settled_maxdiff = max(settled_maxdiff, s if pd.notna(s) else 0.0)
                    provisional_maxdiff = max(provisional_maxdiff, p if pd.notna(p) else 0.0)
                print(
                    f"{asset} {year}: reproducibility max abs diff -- settled (<= {settled_cutoff}) = "
                    f"{settled_maxdiff:.3e}, provisional (< {DATA_FINALITY_BUFFER_HOURS}h buffer) = "
                    f"{provisional_maxdiff:.3e}",
                    flush=True,
                )
                if settled_maxdiff > SETTLED_DIFF_TOLERANCE:
                    raise RuntimeError(
                        f"{asset} {year}: settled data (older than {DATA_FINALITY_BUFFER_HOURS}h buffer, "
                        f"timestamp <= {settled_cutoff}) changed by {settled_maxdiff:.3e}, exceeding "
                        f"tolerance {SETTLED_DIFF_TOLERANCE:.1e} -- this is unexpected drift outside the "
                        f"known provisional-data window, not the buffer effect. Aborting instead of "
                        f"silently overwriting (AGENTS.md fail-fast contract). Investigate before rerunning."
                    )
                if provisional_maxdiff > 0.0:
                    print(
                        f"{asset} {year}: WARNING -- provisional window revised as expected "
                        f"(data-finality buffer policy, see project memory "
                        f"project-btc-features-2026-drift-root-cause-found-20260801.md). Do not treat "
                        f"predictions/ledgers touching timestamps > {settled_cutoff} as reproducible or "
                        f"promotion-ready until a later run confirms them settled.",
                        flush=True,
                    )
                backup = out_path.with_name(out_path.name + ".bak_pre_extend_20260721")
                old.to_csv(backup, index=False)
            sidecar.to_csv(out_path, index=False)
            print(f"{asset} {year}: wrote {out_path} ({len(sidecar)} rows, {sidecar['timestamp'].iloc[0]}..{sidecar['timestamp'].iloc[-1]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
