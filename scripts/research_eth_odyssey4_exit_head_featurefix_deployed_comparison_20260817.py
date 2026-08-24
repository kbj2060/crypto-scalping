#!/usr/bin/env python3
"""RESEARCH ONLY -- one-off comparison, not a promotion candidate. Completes the 3-way picture for
h48qual's exit_head after the pos_tp/pos_sl feature-barrier mismatch fix
(docs/experiments/eth_odyssey4_exit_head_tpsl_feature_barrier_mismatch_20260817.md): the retrain
job (research_eth_omega461_exit_head_liveatr_relabel_20260813.py --out-suffix full1500_featurefix)
only compared against the ORIGINAL pre-liveATR bundle ("baseline" key in its val_metrics), not
against the CURRENTLY DEPLOYED buggy liveATR-relabel bundle (NEW_H48QUAL_BUNDLE). This script runs
the exact same h48cons._evaluate_val(...) methodology against NEW_H48QUAL_BUNDLE so all three
(original / currently-deployed-buggy / fixed) are on one directly comparable footing. zig075 does
NOT need this -- its live component config has no bundle_override, so its "baseline" IS the
currently-deployed bundle already."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_head_h48cons_relabel_20260813 as h48cons  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402


def main() -> int:
    print(f"evaluating currently-deployed bundle: {portfolio.NEW_H48QUAL_BUNDLE}", flush=True)
    metrics = h48cons._evaluate_val("h48qual", portfolio.NEW_H48QUAL_BUNDLE)
    out = {"component": "h48qual", "bundle": str(portfolio.NEW_H48QUAL_BUNDLE), "currently_deployed_val": metrics["h48cons_relabel"]}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    out_path = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_exit_head_featurefix_deployed_comparison_20260817.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
