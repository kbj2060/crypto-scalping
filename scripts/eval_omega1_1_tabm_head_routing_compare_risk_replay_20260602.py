#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import eval_omega1_regime3_expertdq_tabm_risk_replay_20260602 as tabm_eval


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    tabm_eval.MODEL_ID = "omega1_1_tabm_head_routing_compare_risk_replay_20260602"
    tabm_eval.EXPERTDQ_DIR = ROOT / "tmp/causal_regen_20260516/omega1_1_tabm_head_routing_compare_20260602"
    tabm_eval.OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_1_tabm_head_routing_compare_risk_replay_20260602"
    return tabm_eval.main()


if __name__ == "__main__":
    raise SystemExit(main())
