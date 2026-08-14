# ETH Omega4.6.1 라이브 파이프라인 결함 감사 (2026-08-13)

## 배경

오늘 밤 exit head 문제를 "라벨-런타임 배리어 스케일 불일치"로 root-cause하고 고친 사례([eth_omega461_live_exit_head_liveatr_relabel_20260813.md](eth_omega461_live_exit_head_liveatr_relabel_20260813.md), [eth_omega461_exit_head_asymmetric_shadow_20260813.md](eth_omega461_exit_head_asymmetric_shadow_20260813.md))가 이번 오디세이 세션에서 유일하게 VAL+OOS를 모두 통과한 개선이었다. 반면 SLTP 폭 조정, 멀티슬롯 용량 확대, JM 레짐 재학습은 전부 VAL에서 좋아 보였다가 OOS에서 반전됐다.

이 비대칭(구조 수정은 살아남고 파라미터 튜닝은 반전됨)에 근거해 "새 파라미터를 더 찾기"보다 "exit head와 같은 클래스의 숨은 결함이 라이브 파이프라인 다른 곳에도 있는지"를 파는 방향으로 전환. 재학습 없이 코드 감사만으로 4갈래 병렬 조사를 수행했다.

## 방법

general-purpose 에이전트 4개, 각각 코드 리딩/grep/기존 산출물 대조만 수행 (재학습 없음):
1. zig075 exit head 회귀 원인 재조사
2. h48qual/zig075 quality head의 ATR 배리어 스케일 일치 여부 (exit head와 같은 버그 클래스 재검색)
3. h48qual 사이징 사이드카의 0.31x 리프트가 구조적 결함인지
4. quality_threshold OOS-우선 선정편향 패턴이 라이브 라인리지의 다른 학습 스크립트에도 있는지

## 결과

### 1. zig075 exit head 회귀 — 공유 threshold 미보정 (medium confidence)

relabel 레시피 자체는 h48qual/zig075 간 바이트 단위로 동일 (ATR192/tp_mult12.0/sl_mult6.0/min_tp0.075/min_sl0.040, 동일 학습 프레임에서 동시 생성) — ATR창/배리어배율/피처가용성/학습데이터량 불일치 가능성은 코드 확인으로 전부 배제됨(high confidence).

차이점: `EXIT_THRESHOLD=0.95`가 두 컴포넌트에 공유되고 relabel 후 재보정되지 않음(`_evaluate_val`, `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py` 385/395행). zig075(quality_threshold=0.75)는 h48qual보다 큰 엣지의 트레이드를 선별하는데, `gave_back≥0.65×running peak MFE` 규칙이 "이후 더 높은 peak가 오는지" 확인하지 않는 근시안적 로직이라 zig075의 되돌림-후-재상승 패턴을 조기 청산으로 오판했을 가능성 (medium confidence, 새 추론).

실측: VAL PnL +40.31%→+0.70%, MDD -13.07%→-19.91%, 트레이드 29→65(2.2배), 평균 보유 725.6→275.7바(38%), 승률 48.3%→29.2%(동전던지기 이하), 청산의 75.4%가 새 exit_head 발동. `max_trade_pnl`은 소수점 10자리까지 불변(8.678202...%) — 최대 승자 트레이드가 잘린 게 아님.

제안: (a) 저비용 — 기존 `research_eth_omega461_exit_sweep_20260721.py`의 threshold 그리드를 zig075 relabel 번들 전용으로 재실행. (b) 그래도 안 되면 `gave_back` 조건에 "barrier_end_i 이전에 더 큰 MFE가 없어야 함" 조건 추가.

### 2. h48qual quality head 라벨 미스매치 — CONFIRMED, exit head와 동일 버그 클래스 (high confidence)

| 파라미터 | h48_conservative 라벨 | 라이브 `_ComponentConfig` | 배율 |
|---|---|---|---|
| ATR 창 | 96 | 192 | 2.0x |
| tp_mult | 1.2 | 12.0 | 10.0x |
| sl_mult | 0.8 | 6.0 | 7.5x |
| min_tp floor | 0.006 | 0.075 | 12.5x |
| min_sl floor | 0.004 | 0.040 | 10.0x |
| horizon | 48바 고정 | 없음(무제한) | — |

h48qual의 quality_threshold=0.50 필터는 `BarrierConfig("h48_conservative", horizon=48, tp_mult=1.2, sl_mult=0.8, min_tp=0.006, min_sl=0.004)` (`build_omega1_2_triple_barrier_labels_20260619.py:40`, ATR96)로 "48바 안에 0.6~1% 유리하게 움직였나"를 학습했는데, 실제 라이브는 "시간제한 없이 ≥7.5% TP를 노리는 트레이드"를 걸러내는 데 쓰인다. h48qual은 PRIORITY 1순위 컴포넌트라 이 필터가 진입 여부 자체를 좌우 — exit head보다 파급력이 클 가능성이 있다.

대조군 둘 다 clean 확인: zig075의 quality head는 `quality_mode="same_as_direction"`이라 이 버그 클래스에서 원천 면역(별도 배리어 자체가 없음). direction head(양쪽 공유, `zigzag_action_labels_20260531`)는 zigzag 피벗 라벨이라 tp_mult/sl_mult 개념이 없어 애초에 비교 대상이 아님.

이 감사는 미스매치의 존재만 확인했다 — relabel이 실제 라이브 성과를 바꾸는지는 exit head 때와 같은 전후(VAL+OOS) 비교가 별도로 필요하다.

### 3. 사이징 사이드카 — 학습 목표는 정상, 두 가지 다른 이슈 발견 (medium confidence)

사이징 헤드 자체는 컴포넌트별로 독립 학습되고(`net_per_notional` 타겟, HistGradientBoostingRegressor) exit head식 "잘못된 목표" 결함은 아니다.

**정정**: 어젯밤 사이징-편향 정량화 결과(h48qual 0.31x, zig075 3.41x)는 "flat보다 나쁘다"는 뜻이 아니었다 — h48qual도 실제로 flat보다 낫다(+5.45% vs +5.20% VAL, +9.49% vs +8.66% OOS). 0.31x는 두 양수 개선폭의 비율일 뿐이다.

**방법론 공백**: 그 정량화 스크립트(`research_eth_omega461_val_sizing_bias_quantification_20260813.py`)가 라이브에 실제 적용되는 `SCALE_MAP`(방향별 하드코딩 스케일: h48qual_L=0.38, h48qual_S=2.499, zig075_L=2.446, zig075_S=2.478, blueprint L7)을 빼놓고 계산 — 실제 라이브 경로와 다른 걸 측정했다.

**새 발견 (high confidence)**: h48qual 원시 레버리지(~2.0) × short 스케일(2.499) ≈ 5.0 = `LEVERAGE_CAP` 그 자체. h48qual 포지션의 71~79%가 숏(23/29 VAL, 10/14 OOS)이므로, 대부분의 트레이드에서 사이징 헤드가 뭘 예측하든 라이브 레버리지가 캡에서 잘릴 가능성이 있다. zig075는 스케일이 거의 대칭(2.446/2.478)이라 이 문제에서 상대적으로 자유롭다.

**가설**: h48qual 사이징 헤드는 `risk_feature_mode="parent_outputs"`라 quality head의 confidence만 보고 학습되는데, 그 quality head 자체가 (2번 발견처럼) 스케일이 맞지 않는 라벨로 학습됐다면 사이징 헤드도 비차별적 신호를 물려받는 구조 — 2번과 3번이 같은 근본 원인을 공유할 가능성이 있다.

제안: `--risk-feature-mode all` (기존 옵션)로 h48qual 사이징을 재학습 + SCALE_MAP 포화 완화.

### 4. 선정편향 스윕 — BTC에서 동일 버그의 새 인스턴스 발견 (high confidence)

8개 라이브 관련 학습/선정 스크립트 점검, 2개에서 OOS-primary 정렬 버그(`rows.sort(key=lambda r: (oos_pnl, validation_pnl), reverse=True)`) 확인:
- `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:1173` — 기존에 알려진 ETH 버그.
- `train_eval_omega4_3head_parent72_loose_entry_quality_btc_swingtransition_20260806.py:755` — **신규**. 현재 배포된 BTC 번들(`FINAL_GOVERNOR_OMEGA4_6_1_BTC_BUNDLE_PATH`)을 생성한 스크립트가 동일 패턴(`report.json`이 `ranking_by_oos_pnl`만 저장, ETH 버그와 동일 시그니처).

Clean 확인: SOL 파서(`_sol_20260707.py:765`, VAL-primary), ETH 리스크 사이드카(`validation_only` CLI 강제), BTC/SOL 리스크 사이드카(VAL-primary 정렬, OOS는 3번째 타이브레이커 — BTC는 `validation_only`, SOL은 상대적으로 약한 `validation_oos_guard`를 실제 사용했지만 버그는 아님), 레짐3 HMM 스크립트(argmax는 candidate 승격이 아니라 행 단위 softmax 디코딩), 멀티슬롯 스크립트(사전등록된 단일 OOS-read, `oos_read_spent` 플래그로 보호), 오늘 밤 새로 만든 exit head 스크립트 전부.

Live-irrelevant BUG (연구용/대체됨, 우선순위 낮음): `..._btc_20260708.py:755`, `..._btc_swingtransition_zigzag_20260806.py:755`, `..._btc_exitonly_20260806.py:759`, `..._reduced80_20260724.py:1182`, `train_eval_omega4_separate_risk_tabm_margin_leverage_20260622.py:484`.

## 종합 및 우선순위

exit head 성공 이후 "구조 결함 찾기"가 "파라미터 더 찾기"보다 확실히 더 생산적이었다 — 재학습 없는 20분 코드 감사로 컨펌된 새 결함 1개(h48qual quality head), 컨펌된 새 라이브 메커니즘 이슈 1개(SCALE_MAP 포화), 다른 자산으로 확장된 기존 버그 인스턴스 1개(BTC)를 찾았다.

우선순위:
1. **h48qual quality head relabel** — 가장 파급력 큼(진입 자체를 게이트), exit head와 같은 검증된 레시피 적용 가능. 재학습 필요 → 서버.
2. **zig075 exit threshold 재보정** — 저비용, 재학습 불필요, 즉시 실행.
3. **BTC 선정편향 재점검** — 저비용, 기존 clean-reselection 방법론 재사용.
4. **SCALE_MAP 포화 어블레이션** — 저비용, 재학습 불필요.

## 후속 조치 (2026-08-13, 4개 병렬 에이전트, 전부 완료)

### 후속 1: zig075 exit threshold 재보정 — 부정 (재보정 불가능 확정)

8개 threshold(0.999~0.70) 전수 그리드 스윕. baseline과 동률인 건 0.999 하나뿐인데 그 지점에서 exit_head가 **0번 발동**(사실상 비활성화 — 진짜 수정이 아님). 헤드가 조금이라도 작동하는 모든 threshold(0.97 이하)가 baseline보다 나쁨(+16.11%~-10.75%, 저점에서 최대 216건까지 과매매). **결론: threshold 재보정으로 zig075를 못 구함** — 잘못된 모양의 확률분포는 컷오프로 못 고침. 기존 비대칭 채택(h48qual만 교체) 결정이 옳았음을 재확인.

### 후속 2: h48qual quality head relabel + 서버 재학습 — VAL 강한 통과, 단일 OOS에서 완전 반전

서버 상태 확인(정상) 후 relabel 스크립트 작성 → 재학습(~3분) → 평가 순으로 진행. horizon은 exit head 수정과 같은 철학으로 고정 48바를 제거(무제한, 6000바 계산 안전장치만 유지). 인코더/direction_head/exit_head는 현재 배포 번들에서 그대로 고정, quality_head만 재학습.

**컴포넌트 VAL** (h48qual만): PnL +5.45%→**+20.20%**, MDD -11.62%→-11.97%, 트레이드 29→25, 승률 41.4%→**48.0%**.

**진입 특성 변화**: 게이트 통과율 1.79%→**9.60%**(5.4배), 통과 트레이드의 방향 구성 롱13.3%/숏86.8% → 롱**68.5%**/숏31.5% — 기존에 확정됐던 병리적 숏 쏠림이 줄어든 게 아니라 **반대 극성(롱 쏠림)으로 뒤집혔다**.

**포트폴리오 VAL** (`greedy_replay`, SCALE_MAP 포함, zig075·레짐3 불변): PnL +36.82%→**+74.76%**, MDD -24.34%→**-11.26%**, 공유 슬롯 승리 비중 h48qual 24%→**86%**.

**단일 OOS 확인**(2026-01-01~03-31): PnL +49.32%→**+14.65%**, MDD -16.20%→**-22.12%**, 승률 45.8%→**31.3%** — **전 지표 반전**. 재튜닝 시도 없이 그대로 보고.

**왜 exit head와 다르게 반전됐나**: exit head 수정은 이미 진입이 확정된 포지션을 "언제 닫을지"만 바꿨다(어떤 트레이드를 하는지는 불변). 반면 quality head 수정은 **어떤 트레이드를 할지 자체를 5.4배 넓히고 방향 구성을 뒤집었다** — 배리어 스케일 버그를 고쳤더니 완전히 다른 트레이드 모집단을 선택하게 된 것. 이 서브프로젝트가 2026-08-11~12에 GBDT/TabM/trend-scanning 등 다중 방법론으로 exhaustively 재확인한 기존 결론(`odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`, "quality_head 게이트 편향" 절) — h48qual의 `direction_head`는 always-short 대비 실증된 방향 스킬이 없고, `quality_head` 게이트는 사실상 그 `direction_head`의 confidence 필터일 뿐이다. quality head를 아무리 정확하게 재라벨링해도 "스킬 없는 direction_head의 어느 부분집합을 통과시킬지"만 바뀔 뿐이며, 그 부분집합이 VAL에서 이기는지는 스킬이 아니라 그 구간의 우연한 방향성 베타와의 정렬 문제다. 새 라벨은 VAL(특정 방향성 구간)에서 우연히 유리한 부분집합을 골랐을 뿐, OOS의 다른 레짐에서는 그 정렬이 깨졌다. exit head가 살아남은 이유는 애초에 "어떤 트레이드를 할지"를 전혀 건드리지 않아 이 베타-탑승 함정에서 구조적으로 자유로웠기 때문.

### 후속 3: BTC 선정편향 clean-reselection — 견고함에 가까움, 시급하지 않음

버그 코드는 확인되나 실제 배포값(0.55)은 버그의 1위 선택(0.60)이 아니라 VAL-최적값과 일치. Fresh window(19일) 재점검도 부호반전 없이 좁게 클러스터(-0.97%~+1.34%). 코드는 위생상 고쳐야 하나 이 배포값 자체는 급하지 않음(medium confidence, fresh window가 짧아 확정력 약함).

### 후속 4: SCALE_MAP 포화 어블레이션 — 변경 불필요

포화는 진짜(h48qual 숏 69.6% 캡바인딩, 47.8% 완전고정)이지만 원인은 원본 레버리지 range(1.88~2.18x) 자체가 좁아서 — 캡을 조정해도 3개 후보값 전부 baseline보다 나쁨(선별력 불변, 사이즈만 축소). **권고: h48qual_S=2.499 유지.** 부수 발견: 오늘 밤 컴포넌트-레벨 리플레이 다수(exit_sweep, 사이징편향 등)가 SCALE_MAP 누락 — 상대비교는 유효하나 절대 PnL은 실제 라이브의 절반 수준이었을 가능성(포트폴리오 레벨은 원래부터 정확).

## 결론

4개 후속 전부 완료 — **채택 가능한 변경 0건**. 이 라운드가 시작 근거로 삼았던 가설("구조 결함 찾기가 파라미터 튜닝보다 생산적")은 절반만 맞았다: exit head(트레이드 선택 불변, 청산 타이밍만 변경)는 컨펌된 결함 수정이 OOS까지 살아남은 유일한 사례로 남았지만, quality head(트레이드 선택 자체를 바꿈)는 똑같이 "컨펌된 스케일 불일치"였음에도 반전됐다. 실제 구분선은 "구조냐 파라미터냐"가 아니라 **"이미 확정된 포지션의 부수 로직을 고치는가, 아니면 스킬이 검증 안 된 direction_head의 진입 선택 자체를 바꾸는가"**로 보인다 — 후자는 아무리 잘 고쳐도 h48qual의 근본 미해결 문제(방향 스킬 부재, 2026-08-11/12 확정)를 다시 건드리게 된다.
