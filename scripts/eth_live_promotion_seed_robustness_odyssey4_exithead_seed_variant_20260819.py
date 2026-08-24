#!/usr/bin/env python3
"""오디세이4(ETH) 섀도우가 실제로 쓰는 h48qual exit_head(liveATR-relabel, 2026-08-13,
`tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual`)를
다른 시드로 재학습 -- eth_live_promotion_seed_robustness_odyssey4_liveatr_snapshot_20260819.py
(=git HEAD, exit_head position-feature 버그수정 이전 -- 실제 이 아티팩트를 만든 정확한 코드)를
그대로 재사용. 이 스크립트는 encoder/direction_head/quality_head를 라이브 h48qual/zig075
번들에서 얼려서(freeze) exit_head만 재학습하는 구조라(baseline_bundle_path=sweep.COMPONENTS[
component]["bundle"], base_cols도 그 번들에서 그대로 복사) 전체 3-head 부트스트랩(ETH 라이브
h48qual/zig075 자체 재학습)에서 겪은 base_cols 자동유도 드리프트 문제가 원천적으로 없다 --
_build_exit_dataset_entry_label_live_atr_barrier가 frames['feature_cols']를 아예 안 쓰고
BASE_TEMPLATE 고정 risk 상수만 씀(원본 버그 그대로, risk sidecar 불필요). 그래서 canonical
데이터 오버라이드/102-pin 없이 원본 legacy CSV 기본값 그대로 실행.

실제 아티팩트는 h48qual과 zig075 exit_head를 같은 데이터셋에서 동시에 재학습하지만(원본
main()이 두 component를 순회), 이 섀도우가 실제로 쓰는 건 h48qual bundle뿐(zig075는 원본
라이브 zig075를 그대로 씀) -- 둘 다 만들어지지만 h48qual만 평가에 쓴다.

--max-candidates 1500(원본 아티팩트 out-suffix가 "full1500"인 것과 일치, 기본값 2000 아님),
나머지(--epochs 8/--cost-mult 3.0/--max-horizon-bars 6000)는 CLI 기본값 그대로(report.json에
명시적 override 흔적 없음, exit_epochs_ran=6<8은 early stopping으로 추정). 사용: --seed <int>
필수."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eth_live_promotion_seed_robustness_odyssey4_liveatr_snapshot_20260819 as snap  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()
    out_suffix = f"full1500_seedvariant_{known.seed}"
    sys.argv = [
        sys.argv[0],
        "--stage", "full",
        "--max-candidates", "1500",
        "--seed", str(known.seed),
        "--out-suffix", out_suffix,
    ]
    print(f"stage=start seed={known.seed} out_suffix={out_suffix}", flush=True)
    raise SystemExit(snap.main())
