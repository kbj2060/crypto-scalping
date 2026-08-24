#!/usr/bin/env python3
"""sol_live_promotion_seed_robustness_prefix_snapshot_20260819(=git HEAD, exit_head 버그수정 이전
원본 코드)의 데이터 소스 두 종류를 오버라이드한 판:

1. omega.TRAIN_CSV/EVAL_CSV -> adaptive_squeeze 재구축 피쳐 (docs/model_contracts/
   sol_adaptive_squeeze_v2_20260720.md 1단계, 기존 committed 래퍼
   scripts/train_eval_omega4_3head_parent72_loose_entry_quality_sol_adaptive_squeeze_20260720.py와
   동일한 오버라이드).

2. omega.REGIME3_CURRENT_2025/2026 -> 이 스크립트가 직접 재생성한 wide24 HMM 리짐 오버레이.
   ⚠️ 2026-08-19 발견: 공유 정식 경로(data/ensemble/supervised/
   sol_regime3_current_hmm_sensitive_wide24_20260707/sol_features_{2025,2026}_regime3_current_
   sensitive_hmm_wide24.csv)에 2025년 파일이 아예 없고 2026년 파일도 `.bak_pre_extend_20260721`
   백업만 남아있다(dev/server 둘 다 동일 -- 실측 확인, 원본 파일이 언제 어떻게 사라졌는지는
   불명). 이 오버레이를 만든 정확한 스크립트(scripts/extend_regime3_wide24_sol_btc_20260721.py)를
   찾아 그 SOL 분기만 재현했다 -- 단, 그 스크립트는 SOL+BTC 공유 정식 출력 경로에 직접 쓰고
   재현성-diff 백업 로직까지 있어(BTC 쪽도 건드림, 동시세션 안전성 미확인) 그대로 실행하지 않고
   여기서 SOL만, 별도 scratch 출력 경로에 다시 구현했다(로직은 재구현 아님 -- 그 스크립트가 쓰는
   `_transform`/`_read`를 동일하게 그대로 호출).

   소스 데이터는 그 스크립트 자신의 `ASSETS["sol"]["sources"]`와 동일하게 LEGACY
   data/splits/year_oos/sol_features_{2025,2026}.csv를 쓴다(adaptive_squeeze 버전이 아님) --
   wide24 HMM이 소비하는 WIDE24_EXTRA_COLS(volatility_z/rsi/macd_hist/bb_width_z/hma_slope/
   wick_ratio/mtf_trend_1h/mtf_trend_4h/breakout_strength/mean_reversion_z/ofi_acceleration/
   taker_acceleration, scripts/experiment_regime3_current_hmm_wide24_20260529.py 확인)는
   adaptive_squeeze 수정이 건드리는 funding/squeeze 피쳐(DOCS_CURRENT_EXTRA_COLS 쪽에만 있음)와
   완전히 무관한 별도 피쳐군이라, legacy 소스로 계산해도 라이브가 실제 학습에 쓴 오버레이와
   내용이 같다 -- adaptive_squeeze 자체가 wide24 HMM 입력에 아무 영향을 주지 않기 때문.

⚠️ 동시성: 새 파일을 생성하므로(regime3 오버레이 CSV 2개) ETH의 canonicaldata 래퍼와 동일하게
os.replace() 원자적 rename으로 병렬 시드 프로세스 간 쓰기 경합을 방지한다."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sol_live_promotion_seed_robustness_prefix_snapshot_20260819 as parent_script  # noqa: E402
from scripts.experiment_regime3_current_hmm_wide24_20260529 import _transform  # noqa: E402
from scripts.train_regime3_hmm_mamba_20260529 import _read  # noqa: E402

omega = parent_script.omega

omega.TRAIN_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2025.csv"
omega.EVAL_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2026.csv"

_SCRATCH_DIR = ROOT / "tmp/causal_regen_20260516/sol_live_promotion_seed_robustness_20260819/regime3_overlay_rebuild"
_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Exact same source/joblib pair as scripts/extend_regime3_wide24_sol_btc_20260721.py's ASSETS["sol"]
# entry -- legacy (non-adaptive_squeeze) year_oos features, see module docstring for why this is
# still faithful to the live bundle's own training-time overlay.
_JOBLIB_PATH = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707/regime3_current_sensitive_hmm_wide24_2024.joblib"
_LEGACY_SOURCES = {
    2025: ROOT / "data/splits/year_oos/sol_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/sol_features_2026.csv",
}


def _rebuild_overlay(year: int, src: Path) -> Path:
    out_path = _SCRATCH_DIR / f"{src.stem}_regime3_current_sensitive_hmm_wide24.csv"
    if out_path.exists():
        return out_path
    payload = joblib.load(_JOBLIB_PATH)
    frame = _read(src)
    sidecar, _ev = _transform(payload, frame)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp{os.getpid()}")
    sidecar.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    print(f"regime3_overlay_rebuild: {year} wrote {out_path} ({len(sidecar)} rows, "
          f"{sidecar['timestamp'].iloc[0]}..{sidecar['timestamp'].iloc[-1]})", flush=True)
    return out_path


omega.REGIME3_CURRENT_2025 = _rebuild_overlay(2025, _LEGACY_SOURCES[2025])
omega.REGIME3_CURRENT_2026 = _rebuild_overlay(2026, _LEGACY_SOURCES[2026])

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
