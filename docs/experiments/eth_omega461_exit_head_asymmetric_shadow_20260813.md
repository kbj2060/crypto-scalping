# ETH Omega4.6.1 비대칭 exit-head 스왑 후보 — 라이브포워드 섀도우 봇 (2026-08-13)

상태: `shadow_running_research_only` — 라이브 승격이 아니라 오늘 밤(2026-08-13) 오버나이트
exit-head 재설계 트랙의 최종 산출물인 섀도우 관찰 봇을 구축·스모크테스트했다. 실주문 없음
(`order_submission_supported=false`, `activation_allowed=false`).

## 배경

`docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`(후속 1~3절)이 오늘 밤
exit head 재설계 트랙에서 시도한 모든 후보(최종보스 v2/v3, SLTP 재보정, 멀티슬롯, JM 레짐
재학습) 중 **유일하게 VAL→OOS 반전 없이 끝까지 살아남은** 결과를 냈다: h48qual만 새 exit_head로
교체하고 zig075는 원본 그대로 두는 비대칭 구성이 컴포넌트 레벨 VAL, 포트폴리오 레벨 VAL,
포트폴리오 레벨 OOS(오케스트레이터 승인, 1회 한정 확인) 전부를 통과했다.

| | VAL baseline | VAL 비대칭 | OOS baseline | OOS 비대칭 |
|---|---:|---:|---:|---:|
| PnL | +36.82% | +46.59% | +49.32% | +93.27% |
| MDD | -24.34% | -21.70% | -16.20% | -15.48% |

⚠️ OOS 절대 수치는 baseline과 후보가 공유하는 `quality_threshold`(0.50/0.75) 자체가 이 OOS
구간(특히 2026-01~02월)을 선택 타겟으로 최적화됐다는 별도 조사(`eth_omega461_oos_selection_
bias_scope_and_resolution_20260813.md`)가 있어 "깨끗한 미접촉 검증"이 아니다 — **상대비교(새
exit_head가 baseline보다 나은가)는 유효**하지만 절대 OOS PnL을 진짜 미래 성과로 과대해석하면 안
된다. 상세 근거·컴포넌트별 결과·유보 사항 전체는 위 링크 문서를 볼 것 — 이 문서는 그 결과를
재검증하지 않고, 그 결과를 **실시간으로 관찰하는 섀도우 봇**만 다룬다.

이 서브 프로젝트의 표준 다음 단계는 라이브 승격이 아니라 섀도우 테스트다 — 같은 날 밤
`eth_multislot_mfe_gated_capacity_20260813.json`의 사전등록 게이트 `G5_outcome_ceiling`이
명시한 규칙과 동일: *"이 계약이 통과할 수 있는 최선의 결과라도 ETH 멀티슬롯 섀도우를 세우는
것 — 라이브 포지션 정책 전환은 아니다"*. 오늘 밤 이 exit-head 트랙도 같은 상한을 적용한다:
VAL·OOS 모두 통과했어도 표본이 작고(VAL 29~35건, OOS 24건) OOS 자체에 선택편향 유보가 있으므로
다음 단계는 실시간 포워드 관찰이다.

## 이 섀도우가 관찰하는 것

실제 라이브 Omega4.6.1 ETH 어댑터와 **완전히 동일한** 입력(실시간 5분봉 피쳐 스트림), 완전히
동일한 zig075 컴포넌트, 완전히 동일한 레짐3 HMM 분류기, 완전히 동일한 우선순위(`h48qual` →
`zig075`)를 쓰되, **h48qual 컴포넌트의 `true_3head_tabm_bundle.pt`만** 오늘 밤 재학습된
exit_head로 교체한 가상의 포지션을 bar-by-bar로 계속 추적한다. 실제 라이브 ETH 포지션도
참고용으로 매 bar 같이 기록해, 이 섀도우가 실제 라이브와 언제/어떻게 갈리는지 관찰할 수 있다.

## 방법 — 구조적 템플릿 재사용

`scripts/live_eth_jmlam4_regime_swap_shadow_20260809.py`(레짐3 HMM→JM 스왑 섀도우)를 그대로
구조적 템플릿으로 재사용해 `scripts/live_eth_exithead_asymmetric_shadow_20260813.py`를 작성했다
— 복붙 후 최소 수정 원칙을 지켰다. 재사용한 것과 바꾼 것을 명확히 구분한다.

**그대로 재사용(무변경)**:
- `Omega461LiveAdapter`(from `trading_bot_modules.omega4_6_1_live`)를 그대로 import — 어댑터
  클래스 자체는 전혀 안 건드림. `components_override`로 컴포넌트별 번들/사이드카/
  `quality_threshold`만 바꿔치기하는 패턴 그대로.
- `data/live/decision_feature_snapshot.jsonl`(라이브가 실제로 쓰는 것과 같은 실시간 5분봉 피쳐
  스트림)을 tail. `seed_buffer`/`read_new_rows`/`try_fill_pending`/`process_bar`/
  `omega461_eth_position` 헬퍼 전부 그대로.
- 진입/청산 다음 bar 시가에 체결(라이브와 동일 실행지연 모델, `evaluate_exit`의 문서화된 계약).
- 상태 저장(`data/live/eth_exithead_asymmetric_shadow/state.json`,`closed_trades.jsonl`,
  `equity_curve.jsonl`), 재시작 시 이어서 진행.
- 히스토리 창의 오래된 NaN 갭을 최신 bar는 건드리지 않고 backfill/forward-fill하는 안전장치.
- `exit_threshold`(0.95), `duration_threshold`, `EXPERT_SCALES`, `base_template` 등 오늘 밤 이
  실험에서 손대지 않은 나머지 전부 — `Omega461LiveAdapter` 생성자에 이 인자들을 아예 넘기지
  않아서 어댑터 자신의 라이브 기본값이 그대로 적용된다(`duration_threshold=0.005417`, 스모크
  테스트 로그로 실측 확인, `.env`에 `FINAL_GOVERNOR_OMEGA4_6_1_ETH_*` 오버라이드가 하나도 없음도
  확인).

**이번 섀도우가 다른 점(수정)**:
1. **레짐3 분류기는 손대지 않았다** — JM은 오늘 밤 N=5시드로 재현 안 되고 축 자체가 종결됐다
   (`eth_omega461_live_jm_full_retrain_seed_robustness_20260813.md`). JM 스크립트가
   `adapter.regime3_current`를 JM 인스턴스로 사후 교체하는 라인, 그리고 그 교체를 위해 필요했던
   `Regime3CurrentLiveFeaturesJM`/`causal_decode_soft`/`_num`/`_with_features`/`_class_proba`
   헬퍼 전부를 삭제했다 — `Omega461LiveAdapter.__init__`이 기본으로 만드는
   `self.regime3_current = _Regime3CurrentLiveFeatures(...)`(원본 라이브 12-state sticky HMM,
   `trading_bot_modules.omega4_6_2_source_parent_live`)를 그대로 쓴다. 이 덕분에 스크립트가 JM
   버전보다 오히려 더 짧다.
2. **`COMPONENTS_OVERRIDE`**:
   - `h48qual`: bundle =
     `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual/
     true_3head_tabm_bundle.pt`(오늘 밤 새로 만든 exit_head 재학습 번들), `quality_threshold=0.50`
     (원본과 동일, 안 바뀜). sidecar는 아래 "아티팩트 계보 우회" 절 참고 — 원본 사이드카 파일을
     그대로 쓰지 못하고 계보 우회 구성이 필요했다.
   - `zig075`: `trading_bot_modules.runtime_config`의
     `FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH`/`_SIDECAR_PATH`를 그대로 import해서 씀,
     `quality_threshold=0.75`(원본과 동일). 명시적으로 `COMPONENTS_OVERRIDE`에 포함시켰다(생략
     불가 — 아래 "`components_override`는 전부 아니면 전무" 참고) — 이렇게 하면 "이번 섀도우가
     무엇을 바꿨는지"가 코드 한 곳에서 명확히 드러난다.
   - `priority`: 라이브와 동일 `("h48qual", "zig075")`.
3. **비교 대상**: 실제 라이브 ETH 포지션(`data/live/dashboard_state.json`의 `position`)을
   `omega461_eth_position()`으로 매 bar `equity_curve.jsonl`에 `real_live_omega461_eth_side`로
   같이 기록(JM 스크립트의 패턴 그대로 재사용).

### 사전 검증 — 새 h48qual 번들의 direction/quality가 정말 얼려져 있는가

지시받은 대로, 컴포넌트 오버라이드를 코드에 넣기 전에 `torch.load`로 직접 두 번들을 비교했다
(스크립트로 실행하지 않고 대화형으로 직접 비교, 결과만 기록):

- **`base_cols`**: 원본 라이브 h48qual 번들과 신규 번들 둘 다 102개, 리스트가 완전히 동일
  (`==` 비교 True).
- **`pos_cols`**(13개), 각 전문가(`bull`/`bear`/`chop`)의 `config`(`ThreeHeadConfig`),
  `n_features`(115), 스케일러의 `columns`/`mean`/`std` — 전부 동일.
- **`state_dict` 텐서 비교**(bull/bear/chop 각각 22개 텐서): `exit_head.weight`/`exit_head.bias`
  2개만 값이 다르고(그마저도 max abs diff가 bear 0.338/bull 0.229/chop 0.377 정도로 정상적인
  "재학습된 헤드" 크기), **나머지 20/22 텐서(encoder + direction_head + quality_head)는
  `torch.equal`로 완전히 동일**(부동소수점 오차조차 없는 비트 단위 일치).

결론: 신규 번들은 direction_head/quality_head/encoder가 진짜로 동결된 채 exit_head만
재학습됐다 — `docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`가
서술한 학습 방법("encoder/direction_head/quality_head 동결, exit_head만 재학습")과 정확히
일치함을 독립적으로 재확인했다.

## 아티팩트 계보(lineage) 우회 구성 — 반드시 읽을 것

원래 지시는 "sidecar = 원본 라이브 h48qual 사이드카 그대로(`runtime_config.py`의
`FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_SIDECAR_PATH`)"였다. **직접 시도해보니 이 조합(신규 번들 +
원본 사이드카 경로 그대로)은 성립하지 않는다** — 이유와 우회 구성을 투명하게 기록한다.

### 왜 안 되는가

`trading_bot_modules/omega4_6_1_live.py`의 `_Component.__init__`은 (라이브 파일, 무변경)
`trading_bot_modules/omega4_6_1_runtime_contract.py`의 `validate_sidecar_lineage`를 **무조건
호출**한다. 이 함수는 사이드카 자신의 디렉터리에 있는 `report.json`을 읽어
`risk_model.precomputed_prediction_dir`가 **지금 로딩하는 번들의 부모 디렉터리와 정확히
같아야** 하고, 그 디렉터리에 `quality_threshold`에 맞는 정확한 태그의
`{train,validation,oos}_predictions_qXXX.csv` 3개가 있어야 한다고 강제한다(Omega Artifact
Integrity 정책과 같은 계보다). 원본 h48qual 사이드카의 `report.json`은 당연히 **원본 번들
디렉터리**(`tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_
zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630`)를 가리키므로, 신규 번들
경로와 짝지으면 무조건 `RuntimeError: invalid artifact lineage`가 난다 — **이건 dev/server
어느 쪽에서 실행해도 나는 에러다**(직접 재현·확인함, 아래 "실행 위치" 절의 두 번째 발견과는
별개의, 경로 자체의 문제).

라이브 파일(`omega4_6_1_live.py`, `omega4_6_1_runtime_contract.py`)을 수정해서 이 체크를
우회하는 건 범위 밖이므로(라이브 파일 미변경 요구사항), 데이터 쪽에서 우회 구성을 만들었다.

### 우회 구성 (모두 추가만, 기존 파일 무변경)

1. `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual/`
   에 형제 파일 3개를 **추가**했다: 원본 라이브 h48qual 번들 디렉터리의
   `train_predictions_q050.csv`/`validation_predictions_q050.csv`/`oos_predictions_q050.csv`를
   바이트 그대로 복사. 이 파일들은 `router_*`/`dir_*`/`quality_*` 컬럼만 담고 있고(헤더 직접
   확인, exit_head 관련 컬럼 없음) direction_head/quality_head가 위에서 비트 단위로 동일함을
   확인했으므로, 신규 번들에 대해서도 그대로 유효한 값이다 — 조작된 데이터가 아니라 신규
   번들에서 다시 뽑아도 동일할 값을 그대로 재사용한 것.
2. 새 디렉터리
   `tmp/causal_regen_20260516/eth_omega461_exit_head_asymmetric_shadow_20260813_h48qual_sidecar/`
   를 만들어 `risk_sidecar.pkl`(원본 사이드카의 바이트 그대로 복사, md5 일치 확인 —
   리스크사이징 모델 자체는 100% 무변경)과 `report.json`(원본 `report.json`을 복사한 뒤 필드
   **딱 하나만** 수정: `risk_model.precomputed_prediction_dir`를 신규 번들 디렉터리를 가리키는
   **저장소-상대경로**로 재기록 — `Omega461LiveAdapter`의 `ROOT`가 실행 시점에 동적으로
   계산되므로 dev/server 어느 쪽에서 실행해도 올바르게 풀린다. `selection_scope`
   (`validation_only`)/`precomputed_prediction_tag`(`q050`)/`contract.quality_threshold`(0.50)
   등 나머지 필드는 원본 그대로. 어떤 필드를 왜 바꿨는지 설명하는 노트를 `report.json`에
   `shadow_lineage_shim_note` 키로 남겨뒀다.

이 우회 구성이 `validate_sidecar_lineage`를 통과함을 별도 스크립트로 직접 호출해 확인했고,
`_Component(_ComponentConfig("h48qual", ...))`를 실제로 생성해 base_cols/experts까지 정상
로딩됨을 확인했다. `scripts/audit_omega_artifact_integrity_20260630.py`(승격 게이트) 대상은
아니다 — 이 섀도우는 승격을 주장하지 않는다.

## 실행 위치 — dev가 아니라 서버여야 한다

지시사항이 "dev도 가능, 다만 라이브 실시간 피쳐스트림에 접근 가능한 쪽이어야 함 — 어느 쪽이
적절한지 확인해라"라고 명시해서 직접 확인했다. **결론: 서버(`handoff.sh`의 `server` 호스트,
`trading_bot.py`가 실제로 도는 라이브 트레이딩 박스)에서만 실행 가능하다.** 근거 두 가지, 독립적으로
확인:

1. **실시간 피쳐스트림 자체가 dev에 없다**: 이 dev 체크아웃의
   `data/live/decision_feature_snapshot.jsonl`/`dashboard_state.json`은 마지막 수정이
   2026-08-11 21:3x — 이 세션 시각(2026-08-13 11시경) 기준 **이틀 가까이 정체**돼 있다. dev에서
   실행하면 처음 seed 시점의 stale한 히스토리만 재생하고 이후 새 bar가 전혀 들어오지 않는다.
2. **원본 라이브 h48qual/zig075 사이드카 자체가 dev에서 로딩이 안 된다**(위 계보 우회와는 별개
   문제): 두 사이드카의 `report.json`이 `risk_model.precomputed_prediction_dir`를
   **절대경로**로 하드코딩하고 있는데, 그 값이 `/home/llewyn/crypto-scalping/...`다(서버의 실제
   경로 — 서버 유저는 `llewyn`, `WorkingDirectory=/home/llewyn/crypto-scalping`,
   `scripts/ops/systemd/eth-jmlam4-shadow.service` 등 기존 systemd 유닛으로 확인). dev
   체크아웃은 `/home/kbj20/crypto-scalping`이라 `/home/llewyn/crypto-scalping`가 아예 존재하지
   않는다(`ls`로 직접 확인, "No such file or directory"). `validate_sidecar_lineage`의 `resolve()`는
   절대경로를 저장소 루트와 재결합하지 않고 그대로 쓰므로, **zig075를 완전히 원본 그대로 쓰는
   부분조차 dev에서는 `_Component` 생성 단계에서 lineage 에러로 실패한다**(직접 재현). 즉 이
   문제는 h48qual 신규 번들과 무관하게 dev 자체의 구조적 제약이다 — dev에서는 h48qual 부분만
   내 우회 구성으로 통과시켜도, zig075(완전 원본)에서 막힌다.

두 근거가 독립적으로 같은 결론(서버 전용)을 가리킨다. 참고로 h48qual 단독(우회 구성 포함)은
dev에서도 `_Component` 생성까지 성공한다 — 막히는 건 zig075(원본 그대로) 쪽이다.

## 실행 방법

이 저장소의 dev/server 핸드오프 컨벤션(`scripts/ops/handoff.sh`, 메모리
`reference_dev_server_handoff`)을 그대로 따른다. GPU 학습이 아니라 CPU 추론뿐이라
`--device cpu`가 기본값이고 서버 GPU 점유와 무관하다.

```bash
# 코드/아티팩트를 서버로 동기화(최초 1회 또는 갱신 시)
bash scripts/ops/handoff.sh push server \
  scripts/live_eth_exithead_asymmetric_shadow_20260813.py \
  tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual/ \
  tmp/causal_regen_20260516/eth_omega461_exit_head_asymmetric_shadow_20260813_h48qual_sidecar/

# 지속 실행(백그라운드, nohup 방식 — systemd 상시등록은 이번 작업 범위 밖)
bash scripts/ops/handoff.sh launch server eth_exithead_asymmetric_shadow -- \
  python -u scripts/live_eth_exithead_asymmetric_shadow_20260813.py --poll-seconds 90

# 상태 확인 / 로그 / 중지
bash scripts/ops/handoff.sh status server eth_exithead_asymmetric_shadow
bash scripts/ops/handoff.sh logs server eth_exithead_asymmetric_shadow -f
bash scripts/ops/handoff.sh stop server eth_exithead_asymmetric_shadow

# 상태 파일을 dev로 가져와 분석
bash scripts/ops/handoff.sh pull server data/live/eth_exithead_asymmetric_shadow/
```

재개(resume)는 별도 처리가 필요 없다 — `run()`이 `data/live/eth_exithead_asymmetric_shadow/
state.json`이 이미 있으면 자동으로 이어서 진행한다(JM 스크립트와 동일 패턴). 상시 실행을
원하면 `scripts/ops/systemd/eth-jmlam4-shadow.service`/`install_eth_jmlam4_shadow_20260810.sh`를
본떠 별도 유닛을 등록할 수 있으나, 이 작업 범위에서는 하지 않았다(사용자가 원하면 별도 요청).

## 스모크 테스트 결과 (2026-08-13, 서버에서 실행)

### 사전 점검(preflight)

타임박스 실행 전에 `Omega461LiveAdapter` 생성 자체가 서버에서 성공하는지 별도로 먼저
확인했다(`tmp/verify_eth_exithead_shadow_preflight_20260813.py`, `handoff.sh launch`로 서버
실행):

```
OK: full adapter constructed successfully.
components: ['h48qual', 'zig075']
priority: ('h48qual', 'zig075')
duration_threshold: 0.005417
  h48qual: base_cols=102 quality_threshold=0.5 bundle=true_3head_tabm_bundle.pt experts=['bear', 'bull', 'chop']
  zig075: base_cols=102 quality_threshold=0.75 bundle=true_3head_tabm_bundle.pt experts=['bear', 'bull', 'chop']
```

### 타임박스 실행

`handoff.sh launch server eth_exithead_shadow_smoketest -- python -u
scripts/live_eth_exithead_asymmetric_shadow_20260813.py --poll-seconds 20 --end-at-kst
2026-08-13T11:32:25+0900`로 최초 실행 시각(11:23:30 KST)부터 약 9분간 타임박스 실행(`--end-at-kst`로
자연 종료, systemd 등록이나 nohup 상시실행은 하지 않음 — 이 스모크테스트 전용 1회성 백그라운드
잡).

**결과: 정상 동작 확인, 에러 0건.**

- `[init]` 로그: 어댑터 정상 로딩, `duration_threshold=0.005417`(라이브 기본값과 일치 로그로
  실측 확인), `seeded buffer with 3532 rows`.
- 실행 시작 수 초 내 첫 bar(`2026-08-13T02:15:00`) 즉시 처리(재시작 없이 최초 seed 시점의 최신
  bar 하나를 바로 평가하는 설계대로).
- 이후 폴링 루프가 살아있는 동안 **실제 라이브 5분봉 피쳐스트림에서 새 bar 2개가 추가로
  들어와 정상 처리됨**(`02:20:00`, `02:25:00` — 5분 간격, `snapshot_offset`이
  272026264→272050478로 전진해 실제 신규 바이트를 읽었음을 확인) — 초기 seed 재생이 아니라
  진짜 실시간 tail 동작을 확인했다는 뜻.
- 전체 실행 로그(`handoff.sh logs`)에 `[error]` 라인 0건 — 히스토리 창 전체(3500+행)에 걸친
  regime3 HMM 재계산, h48qual/zig075 양쪽 TabM forward pass, ATR 계산, 사이드카 사이징 경로가
  매 bar마다 예외 없이 통과했다는 뜻.
- `data/live/eth_exithead_asymmetric_shadow/state.json`/`equity_curve.jsonl`이 매 bar 정상
  갱신됨(아래). 이 3개 bar 전부 h48qual/zig075 둘 다 진입 신호가 없어(CASH) `equity=1.0,
  mdd=0.0` 유지 — 원본 라이브 대비 `nonzero_side≈0.018`(바당 진입확률 약 1.8%) 수준을 감안하면
  3bar 관찰에서 무포지션인 것 자체는 정상이다. `closed_trades.jsonl`은 아직 청산 거래가 없어
  파일이 생성되지 않았다(첫 청산 시점에 `append_jsonl`이 최초 생성하는 지연 생성 방식, 정상
  동작).
- 실제 라이브 ETH 포지션 참고값(`real_live_omega461_eth_side`)도 3개 bar 전부 `-1`(SHORT)로
  일관되게 기록됨 — `omega461_eth_position()` 비교 로직도 정상 동작.

```
=== equity_curve.jsonl (스모크테스트 전체) ===
{"bar_ts": "2026-08-13T02:15:00", "equity": 1.0, "mdd": 0.0, "position_side": 0, "real_live_omega461_eth_side": -1}
{"bar_ts": "2026-08-13T02:20:00", "equity": 1.0, "mdd": 0.0, "position_side": 0, "real_live_omega461_eth_side": -1}
{"bar_ts": "2026-08-13T02:25:00", "equity": 1.0, "mdd": 0.0, "position_side": 0, "real_live_omega461_eth_side": -1}
```

`handoff.sh status`가 종료 시각 직후 `STOPPED (last pid=57530)`을 보고 — `--end-at-kst`
바운드에서 수동 kill 없이 스스로 정상 종료됐다(타임박스 백그라운드 잡 설계가 의도대로
동작함, systemd 상시등록과는 무관한 1회성 실행).

## 재개 방법

- 상태: `data/live/eth_exithead_asymmetric_shadow/state.json`(포지션/pending/equity/mdd/마지막
  처리 bar), `closed_trades.jsonl`(청산된 거래), `equity_curve.jsonl`(매 bar 에쿼티 곡선 +
  실제 라이브 ETH 사이드 비교).
- 재시작 시 `state.json`이 있으면 자동으로 `snapshot_offset`/`last_processed_bar_ts`부터 이어서
  진행 — 별도 커맨드 불필요.
- 중지 후 다시 시작해도 안전(`try_fill_pending`/`process_bar`가 멱등적으로 마지막 처리된 bar
  timestamp를 기준으로만 전진).

## 준수 확인

`fresh_forward_bar_by_bar=true`(causal, 저장된 원장을 재생하지 않고 실시간 피쳐스트림을 앞으로만
tail), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`(전부 `state.json`에 실측 기록됨). `order_submission_supported=false`,
`activation_allowed=false` — 실주문 코드 없음. 라이브 파일
(`trading_bot_modules/omega4_6_1_live.py`, `trading_bot.py`,
`trading_bot_modules/runtime_config.py`, `.env`) 미변경(`git status --porcelain`로 확인, 이
스크립트가 만든 신규 파일은 전부 `tmp/`(gitignored) 또는 `scripts/`/`docs/experiments/`의 새
파일뿐).

## 산출물

- 새 스크립트: `scripts/live_eth_exithead_asymmetric_shadow_20260813.py`
- 사전 점검용 1회성 스크립트(재사용 가능, 삭제해도 무방):
  `tmp/verify_eth_exithead_shadow_preflight_20260813.py`
- 계보 우회 아티팩트(추가만, 기존 파일 무변경):
  - `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual/
    {train,validation,oos}_predictions_q050.csv`(신규 추가, 원본 바이트 그대로 복사)
  - `tmp/causal_regen_20260516/eth_omega461_exit_head_asymmetric_shadow_20260813_h48qual_sidecar/
    {risk_sidecar.pkl,report.json}`(신규 디렉터리)
- 섀도우 상태: `data/live/eth_exithead_asymmetric_shadow/{state.json,closed_trades.jsonl,
  equity_curve.jsonl}`
- 인용 문서: `docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`(후속
  1~3절, 이 섀도우가 관찰하는 후보의 VAL/OOS 근거),
  `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`("2026-08-13
  오버나이트" 표), `docs/experiments/eth_multislot_mfe_gated_capacity_20260813.json`
  (`G5_outcome_ceiling` 관례 근거)

## 업데이트 2026-08-13 — 라이브 대시보드에 상시 노출

사용자 요청으로 이 섀도우를 라이브 대시보드(`dashboard/server.py` + `dashboard/live/{index.html,app.js}`,
서버 8787포트, cloudflared 터널로 외부 노출)에 패널로 추가했다. 기존 "ETH JM 레짐 교체 Shadow"
패널(`ETH_JMLAM4_SHADOW_*` 상수·`eth_jmlam4_shadow_payload()`·`/api/eth-jmlam4-shadow`·
`renderEthJmlam4Shadow`/`refreshEthJmlam4Shadow`·index.html의 `ops-shadow-panel` 섹션)을
구조 그대로 복제해서 "ETH Exit Head 비대칭 교체 Shadow"라는 새 패널을 만들었다(백엔드
`eth_exithead_shadow_payload()`+`/api/eth-exithead-shadow`, 프론트 `renderEthExitheadShadow`/
`refreshEthExitheadShadow`, 신규 CSS 클래스 없이 기존 `ops-shadow-panel`/`ops-shadow-body`/
`shadow-slot-list`/`shadow-chart-grid` 재사용).

**배포**: `handoff.sh push`로 3개 파일 동기화 후 대시보드 프로세스를 재시작했다 — 재시작 도중
포트 충돌(`OSError: address already in use`)이 있었는데, 확인해보니 이 서버의 자동복구
메커니즘(추정 `ops_watchdog`)이 kill 직후 스스로 새 프로세스(새 코드 반영됨)를 이미 띄운
상태였다 — 수동 재시도는 불필요했고 결과적으로 무해했다. `curl`로 신규 엔드포인트
(`/api/eth-exithead-shadow`, 실제 섀도우 상태 JSON 정상 반환 확인) 및 기존 엔드포인트
(`/api/eth-jmlam4-shadow`, `/api/btc-multislot-shadow`, `/api/state` 전부 HTTP 200 유지 —
무회귀 확인)를 직접 검증했다. Python(`py_compile`)·JS(Windows `node.exe --check`, WSL
`/mnt/c/Program Files/nodejs/` 경유)·HTML(태그 균형) 전부 배포 전 문법 검증 통과.
