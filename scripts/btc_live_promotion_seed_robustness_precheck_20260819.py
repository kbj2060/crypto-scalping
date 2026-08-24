#!/usr/bin/env python3
"""본 학습(수 분 소요) 돌리기 전에, BTC omega 모듈의 기본(canonical) TRAIN_CSV/EVAL_CSV에서
라이브 BTC h48qual+swingtransition 번들 자신의 152 base_cols가 전부 존재하는지, 자동유도 피쳐
집합이 그 152개와 정확히 일치하는지 가볍게 확인한다 (모델 학습 없음, omega._load_omega_frames +
omega._numeric_feature_cols만 호출).

ETH의 대응 사전점검(scripts/eth_live_promotion_seed_robustness_precheck_20260819.py)과 달리, BTC
omega 모듈(train_eval_omega1_2_tabm_diffusion_risk_btc_swingtransition_20260806.py)의 TRAIN_CSV/
EVAL_CSV 기본값 자체가 이미 라이브 번들과 같은 날(2026-08-06) 만들어진 정적 CSV
(data/splits/year_oos/btc_features_{2025,2026}_swingtransition.csv, 파일시스템 mtime도 2026-08-06)
라서, ETH가 겪은 "7주치 피쳐엔지니어링 누적으로 legacy CSV가 stale해지는" 문제가 애초에 발생할
여지가 훨씬 적다 -- canonicaldata 오버라이드나 152-pin이 필요한지는 가정하지 않고 이 스크립트로
직접 실측한다.

⚠️ 2026-08-19 발견: TRAIN_CSV/EVAL_CSV와 별개로 omega 모듈이 요구하는 REGIME3_CURRENT_2025/2026
오버레이가 공유 정식 경로에서 2025년 파일은 아예 없고 2026년 파일도 백업만 남아있어(dev/server
둘 다 동일), btc_live_promotion_seed_robustness_canonicaldata_20260819.py가 그 오버레이를 별도
scratch 경로에 재생성하는 래퍼로 추가됐다 -- 이 precheck는 (raw 스냅샷이 아니라) 그 래퍼를
import해서 재생성된 오버레이 위에서 실측한다."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import btc_live_promotion_seed_robustness_canonicaldata_20260819 as canon_wrap  # noqa: E402

omega = canon_wrap.omega
LIVE_BUNDLE = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition/true_3head_tabm_bundle.pt"


def main() -> int:
    train_all, eval_df, overlay_report = omega._load_omega_frames()
    full = omega._numeric_feature_cols(train_all, eval_df)
    full_set = set(full)
    print(f"auto-derived full feature count: {len(full)}", flush=True)

    live_cols = list(torch.load(LIVE_BUNDLE, map_location="cpu", weights_only=False)["base_cols"])
    missing = sorted(set(live_cols) - full_set)
    extra = sorted(full_set - set(live_cols))
    print(f"live base_cols count={len(live_cols)} missing_from_auto_derived={len(missing)} sample={missing[:20]}", flush=True)
    print(f"auto_derived_extra_vs_live={len(extra)} (informational only) sample={extra[:10]}", flush=True)
    print(f"ordered_list_identical={full == live_cols}", flush=True)
    print(f"set_identical={full_set == set(live_cols)}", flush=True)
    print("PRECHECK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
