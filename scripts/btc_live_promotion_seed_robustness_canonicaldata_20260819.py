#!/usr/bin/env python3
"""btc_live_promotion_seed_robustness_prefix_snapshot_20260819(=git HEAD, exit_head 버그수정 이전
원본 코드)의 REGIME3_CURRENT_2025/2026 오버레이를 재생성한 판.

⚠️ 2026-08-19 발견 (BTC 사전점검 중): omega.TRAIN_CSV/EVAL_CSV(data/splits/year_oos/
btc_features_{2025,2026}_swingtransition.csv)는 라이브 번들과 같은 날 만들어진 정적 파일이라
문제가 없었지만(precheck로 실측 확인, 아래 참고), omega 모듈이 별도로 필요로 하는 REGIME3_CURRENT_
2025/2026 오버레이(data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708/
btc_features_{2025,2026}_regime3_current_sensitive_hmm_wide24.csv)는 공유 정식 경로에 2025년
파일이 아예 없고 2026년 파일도 `.bak_pre_extend_20260721` 백업만 남아있다(dev/server 둘 다 동일 --
실측 확인, 원본 파일이 언제 어떻게 사라졌는지는 불명). SOL 쪽에서 같은 날 동일 문제를 먼저 발견해
(scripts/sol_live_promotion_seed_robustness_canonicaldata_20260819.py) 이 오버레이를 만든 정확한
스크립트(scripts/extend_regime3_wide24_sol_btc_20260721.py, SOL+BTC 공유)를 찾아 SOL 분기만
재현했던 것과 동일한 기법을, 여기서는 ASSETS["btc"] 분기로 재현한다 -- 단, 그 스크립트는 SOL+BTC
공유 정식 출력 경로에 직접 쓰고 재현성-diff 백업 로직까지 있어(동시세션 안전성 미확인) 그대로
실행하지 않고 여기서 BTC만, 별도 scratch 출력 경로에 다시 구현했다(로직은 재구현 아님 -- 그
스크립트가 쓰는 `_transform`/`_read`를 동일하게 그대로 호출).

소스 데이터는 extend_regime3_wide24_sol_btc_20260721.py 자신의 ASSETS["btc"]["sources"]와
동일하게 LEGACY data/splits/year_oos/btc_features_{2025,2026}.csv를 쓴다(swingtransition 버전이
아님) -- wide24 HMM이 소비하는 WIDE24_EXTRA_COLS(volatility_z/rsi/macd_hist/bb_width_z/hma_slope/
wick_ratio/mtf_trend_1h/mtf_trend_4h/breakout_strength/mean_reversion_z/ofi_acceleration/
taker_acceleration, scripts/experiment_regime3_current_hmm_wide24_20260529.py 확인)는
swingtransition이 추가하는 swing_transition_prob 피쳐와 완전히 무관한 별도(기존) 피쳐군이라,
legacy 소스로 계산해도 라이브가 실제 학습에 쓴 오버레이와 내용이 같다 -- swingtransition 자체가
wide24 HMM 입력에 아무 영향을 주지 않기 때문(SOL의 adaptive_squeeze/funding 케이스와 동일한 논리,
BTC/SOL 둘 다 이 wide24 HMM 자체는 자산별로 공유되지 않는 독립 산출물이지만 "새 파생피쳐 추가가
기존 wide24 입력열을 안 건드린다"는 구조적 이유는 동일하게 적용된다). 이 정확한 동치 논리를 이
세션에서 BTC 데이터로 직접 재확인하지는 않았다 -- SOL 세션의 동일 스크립트/동일 아키텍처 기반
선례를 신뢰해 그대로 적용한 것으로, 잔여 가정으로 명시한다.

⚠️ 동시성: 새 파일을 생성하므로(regime3 오버레이 CSV 2개) ETH/SOL의 동일 래퍼와 똑같이
os.replace() 원자적 rename으로 병렬 시드 프로세스 간 쓰기 경합을 방지한다."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import btc_live_promotion_seed_robustness_prefix_snapshot_20260819 as parent_script  # noqa: E402
from scripts.experiment_regime3_current_hmm_wide24_20260529 import _transform  # noqa: E402
from scripts.train_regime3_hmm_mamba_20260529 import _read  # noqa: E402

omega = parent_script.omega

_SCRATCH_DIR = ROOT / "tmp/causal_regen_20260516/btc_live_promotion_seed_robustness_20260819/regime3_overlay_rebuild"
_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Exact same source/joblib pair as scripts/extend_regime3_wide24_sol_btc_20260721.py's ASSETS["btc"]
# entry -- legacy (non-swingtransition) year_oos features, see module docstring for why this is
# still faithful to the live bundle's own training-time overlay.
_JOBLIB_PATH = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708/regime3_current_sensitive_hmm_wide24_2024.joblib"
_LEGACY_SOURCES = {
    2025: ROOT / "data/splits/year_oos/btc_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/btc_features_2026.csv",
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
