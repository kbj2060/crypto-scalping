#!/usr/bin/env python3
"""라이브 BTC h48qual+swingtransition 번들(tmp/causal_regen_20260516/btc_omega4_3head_parent72_
loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition)을 원본(git HEAD,
2026-08-18 exit_head pos_tp/pos_sl 버그수정 이전 -- 실제 라이브를 학습시킨 그 코드,
btc_live_promotion_seed_robustness_prefix_snapshot_20260819.py)으로 다른 시드 재학습.

ETH 대응 스크립트(eth_live_promotion_seed_robustness_h48qual_seed_variant_20260819.py)와 달리
canonicaldata/102-pin 래퍼가 없다 -- BTC 사전점검(_precheck_20260819.py, 서버에서 실행, 2026-08-19 확인)에서 BTC omega 모듈의 기본
TRAIN_CSV/EVAL_CSV(data/splits/year_oos/btc_features_{2025,2026}_swingtransition.csv, 라이브
번들과 같은 2026-08-06에 만들어진 정적 파일)가 라이브 152 base_cols와 **정확히 일치**함을 실측
확인했다(missing_from_auto_derived=0, auto_derived_extra_vs_live=0, ordered_list_identical=True)
-- ETH가 겪은 "7주치 피쳐엔지니어링 누적 드리프트" 축은 BTC에 해당 없음이 확정됐다.

⚠️ 2026-08-19 발견: 이와 별개로 omega 모듈이 요구하는 REGIME3_CURRENT_2025/2026 오버레이가 공유
정식 경로에서 2025년 파일은 아예 없고 2026년 파일도 백업만 남아있어(dev/server 둘 다 동일, SOL
쪽에서 같은 날 먼저 발견), btc_live_promotion_seed_robustness_canonicaldata_20260819.py가 그
오버레이를 별도 scratch 경로에 재생성하는 래퍼로 추가됐다 -- 이 스크립트는 prefix_snapshot을
직접 import하지 않고 그 canonicaldata 래퍼를 통해 import한다(그 래퍼의 import 시점 side effect로
오버레이가 이미 재생성된 뒤 parent_script.main()이 호출됨). TRAIN_CSV/EVAL_CSV 자체는 이 래퍼가
건드리지 않는다(BTC는 SOL과 달리 이 축의 오버라이드가 불필요 -- 위 문단 참고).

원본과 동일 설정, seed/out-suffix만 바꾼다 (report.json 및 argparse 기본값에서 직접 확인, 아래
FIXED_ARGS 이외는 전부 스크립트 자체 argparse 기본값):
  --direction-label-dir tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708 (스크립트
    자체 기본값 LABEL_DIR과 동일 -- 명시적으로 재전달, 우연한 기본값 변경에 안전하도록)
  --quality-mode quality_label_action (기본값 hard_rule에서 재정의 필요)
  --quality-label-dir tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708
    (기본값 None에서 재정의 필요)
  --device cpu (기본값과 동일, 명시)
  나머지: epochs=4, quality-thresholds="0.40,0.45,0.50,0.55,0.60"(risk sidecar가 참조하는
    precomputed_prediction_tag=q055 포함), max-exit-samples=12000, max-train-rows=30000,
    cost-mult=3.0, exit-label 관련 전부(exit-label-mode/exit-terminal-window/exit-adverse-unreal/
    exit-min-mfe-for-giveback/exit-giveback-min) -- 전부 argparse 기본값. report.json의 간접증거로
    재확인: exit_label.diag.rows=12000(=max-exit-samples 기본값), exit_label.diag.terminal_window=3
    (=exit-terminal-window 기본값), exit_label.diag.min_mfe_for_giveback=0.006/giveback_min=0.65
    (둘 다 기본값), summaries.{bull,bear,chop}.epochs_ran=4(=epochs 기본값과 일치, 조기중단 흔적
    없음), canonical_prediction_contract.risk_sidecar_precomputed_prediction_tag_values=
    [q040,q045,q050,q055,q060](=quality-thresholds 기본값 "0.40,0.45,0.50,0.55,0.60"과 정확히
    일치). report.json에 원본 argv 자체가 저장돼 있지 않아 100% 확정은 아니라는 점은 명시한다
    (ETH 스크립트와 동일한 caveat).

⚠️ seed 가정: report.json에 seed 필드가 없다. argparse 기본값(260620)을 벗어날 명시적 근거를
report.json 어디에서도 찾지 못해 원본 시드=260620으로 가정하고 그대로 재사용(재학습 안 함, ETH와
동일 패턴 -- 기존 라이브 번들 자체를 "seed260620_original"로 평가에 포함).

사용: --seed <int> 필수."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import btc_live_promotion_seed_robustness_canonicaldata_20260819 as canon_wrap  # noqa: E402

parent_script = canon_wrap.parent_script  # == btc_live_promotion_seed_robustness_prefix_snapshot_20260819, with omega.REGIME3_CURRENT_2025/2026 already rebuilt as an import side effect

FIXED_ARGS = [
    "--direction-label-dir", str(ROOT / "tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708"),
    "--quality-mode", "quality_label_action",
    "--quality-label-dir", str(ROOT / "tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708"),
    "--device", "cpu",
]

_orig_fit_expert = parent_script._fit_expert_omega4


def _fit_expert_omega4_logged(*args, **kwargs):
    expert_idx = kwargs.get("expert_idx")
    t0 = time.time()
    print(f"  expert_idx={expert_idx} start", flush=True)
    payload = _orig_fit_expert(*args, **kwargs)
    print(f"  expert_idx={expert_idx} done epochs_ran={payload.get('epochs_ran')} "
          f"best_validation_loss={payload.get('best_validation_loss')} elapsed={time.time() - t0:.1f}s", flush=True)
    return payload


parent_script._fit_expert_omega4 = _fit_expert_omega4_logged

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()
    out_suffix = f"h48qual_20260806_swingtransition_livepromo_seedvariant_{known.seed}"
    sys.argv = [sys.argv[0], *FIXED_ARGS, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start seed={known.seed} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
