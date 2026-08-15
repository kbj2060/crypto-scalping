# 오디세이4 새도우 클린룸 재작성 — 결과 계약 (2026-08-16)

이 문서는 계약 문서다 — 결과와 판정만 다룬다. 전체 조사·구현·검증 과정은
`docs/experiments/eth_odyssey_live_cleanroom_dependency_rewrite_20260816.md` 참고.

## 요약

배포 중인 유일한 오디세이 스크립트(`live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py`,
`eth-odyssey4-shadow.service`)가 공유 Omega4.6.1 프로덕션 라이브 스택(`trading_bot.py`의 실거래
경로와 공유)에 과도하게 의존하고 있었다(~6,850줄 로드, 그중 최소 2,583줄은 SOL/BTC/dead-sidecar
import로 100% 미사용, 무관한 Omega5 시스템 크래시 위험까지 동반). 세 개의 새 Odyssey 전용 모듈
(`trading_bot_modules/odyssey_tabm_core.py`, `.odyssey_regime3_live.py`, `.odyssey_live_adapter.py`)
로 재구현하고, 결정 로직이 정확히 동일함을 실제 아티팩트+실제 데이터로 검증했다.

## 판정: `CONFIRMED` (패리티, no-behavior-change)

- **결정 로직**: `decide_entry`/`evaluate_exit`(h48qual+zig075)/원본 h48qual 가드 컴포넌트의
  `exit_probability`까지 실제 과거 피쳐 데이터로 기존 `Omega461LiveAdapter` vs 신규
  `OdysseyLiveAdapter`를 나란히 실행 — 2,000-bar 정식 검증: `entry_compares=2000
  exit_compares=4000 guard_compares=2000`, **mismatch 0건**(총 8,000회 비교, exit code 0).
- **TabM 추론**: 실제 h48qual/zig075 번들의 모든 익스퍼트에서 `payload["config"] == 전역 CFG
  기본값`(버그 수정이 현재 아티팩트엔 no-op), 무작위 입력에 대해 원본 방식 vs 신규
  `build_model`+`predict_proba` 출력 bit-identical.
- **regime3 라이브 라우팅**: 실제 데이터 4,000행에 대해 `regime3_current_sensitive_wide24_*` 6개
  컬럼 전부 bit-identical — 원본 아티팩트/클래스 대비, 그리고 아래 GaussianStateModel 마이그레이션
  전후 둘 다 확인.
- **의존성 축소**: 새 모듈 단독 import 시 SOL/BTC/`omega5_live`/`runtime_config`/dead
  risk-sidecar 학습스크립트/catboost 전부 0건 로드. **`scripts/*` 학습스크립트 의존성도 완전
  0건**(사용자 후속 요청으로 `GaussianStateModel`까지 벤더링 — 아래 참고). `train_eval_omega1_2_
  tabm_3head_20260603.py`/`scripts.retrain_clean_regime_hmm_20260517` 둘 다 강제 차단해도 새
  모듈 3개 전부 정상 로딩됨을 확인.

## GaussianStateModel 벤더링 (regime3 HMM, 마지막 원본 의존성 제거)

pickle은 클래스를 저장 시점의 정확한 모듈 경로로 참조하므로, 클래스 코드를 복사하는 것만으로는
기존 joblib 아티팩트를 새 모듈이 언피클할 수 없다(원본 모듈을 계속 다시 찾음). 해결책은
"재학습이 아니라 파라미터 이전" — 기존 아티팩트에서 학습된 파라미터(`pi_`/`A_`/`mu_`/`var_`/
`log_likelihood_`)를 그대로 읽어 벤더링한 클래스의 새 인스턴스에 옮기고, 같은 payload를 새
파일로 재저장(`scripts/migrate_regime3_hmm_artifact_to_odyssey_native_20260816.py`, 쓰기
전/후 `filter_proba()` bit-identical 자체검증 포함). 새 아티팩트:
`data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/
regime3_current_sensitive_hmm_wide24_2024_odyssey_native.joblib`(원본 파일은 그대로 두고 별도
파일로 추가 — 원본을 쓰는 다른 시스템에 영향 없음).

## 발견된 버그 2건 (수정, 현재 아티팩트엔 no-op으로 검증됨)

1. 원본 `_predict_payload`가 번들 자신의 `config`를 무시하고 전역 싱글턴으로 모델을 재구성하던
   latent trap 수정.
2. 원본이 entry-decision마다 모델을 재생성(캐싱 없음)하던 것을 컴포넌트 초기화 시 1회 빌드+캐싱으로
   수정.

## 범위

- 수정: `scripts/ops/systemd/eth-odyssey4-shadow.service`의 `ExecStart`만(신규 스크립트 경로).
- 신규 코드: `trading_bot_modules/odyssey_tabm_core.py`, `.odyssey_regime3_live.py`,
  `.odyssey_live_adapter.py`, `scripts/live_eth_odyssey4_zig075_entry_veto_shadow_cleanroom_
  20260816.py`, `scripts/verify_eth_odyssey4_cleanroom_parity_20260816.py`,
  `scripts/ops/systemd/cutover_odyssey4_cleanroom_20260816.sh`,
  `scripts/migrate_regime3_hmm_artifact_to_odyssey_native_20260816.py`.
- 신규 아티팩트: `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/
  regime3_current_sensitive_hmm_wide24_2024_odyssey_native.joblib`(원본 파일과 나란히 존재,
  원본은 삭제하지 않음).
- **미수정(전혀 손대지 않음)**: `trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`,
  `trading_bot_modules/omega4_6_2_source_parent_live.py`,
  `trading_bot_modules/runtime_config.py`, `trading_bot_modules/omega5_live.py`, 은퇴한 Odyssey1~3
  새도우 스크립트 3개, SOL/BTC/risk-sidecar/regime3 원본 학습스크립트 자체, 원본 regime3 HMM
  joblib 아티팩트.

## 배포 상태

`섀도우 코드 교체 준비 완료, 서버 cutover 대기` — root 권한이 필요해 코딩 에이전트가 직접 실행
불가. 사용자가 서버에서 `sudo bash scripts/ops/systemd/cutover_odyssey4_cleanroom_20260816.sh`
실행 필요. 실행 전/후로 `data/live/eth_odyssey4_shadow/state.json`은 변경되지 않으며(같은 유닛,
같은 상태 파일), 새도우 이력이 끊기지 않는다.

## 관련 문서

- 전체 과정: `docs/experiments/eth_odyssey_live_cleanroom_dependency_rewrite_20260816.md`
- 원본 아키텍처: `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`
- Odyssey4 계약: `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`
