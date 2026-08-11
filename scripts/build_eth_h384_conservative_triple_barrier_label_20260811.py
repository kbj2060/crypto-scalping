"""quality_head 타겟 재검토판: h48_conservative와 동일한 배리어 공식(tp_mult=1.2/sl_mult=0.8/
min_tp=0.006/min_sl=0.004)에서 horizon만 48bar->384bar(32h)로 바꾼 버전. 오늘 세션의 horizon
스윕(sweep_h48qual_horizon_wide_20260811.py, 자체 재구현/미저장)에서 zigzag_action과의 방향일치가
48bar 89.5% -> 384bar 92.1%(전체 horizon 그리드 중 최고)로 확인되어 채택. 이 스크립트는 그 결과를
캐노니컬 빌더(build_omega1_2_triple_barrier_labels_20260619.py) 경로로 재생성해 디스크에 남기고,
기존 9개 설정(h24/h48/h96 x conservative/balanced/runner)은 이미 tmp/causal_regen_20260516/
omega1_2_triple_barrier_labels_20260619/에 존재하므로 재계산하지 않는다 -- CONFIGS를 h384_conservative
하나로 교체."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_omega1_2_triple_barrier_labels_20260619 as tb  # noqa: E402

tb.CONFIGS = (tb.BarrierConfig("h384_conservative", 384, 1.2, 0.8, 0.006, 0.004),)
tb.OUT_DIR = ROOT / "tmp/eth_h384_conservative_triple_barrier_labels_20260811"

if __name__ == "__main__":
    if "--out-dir" not in sys.argv:
        sys.argv += ["--out-dir", str(tb.OUT_DIR)]
    raise SystemExit(tb.main())
