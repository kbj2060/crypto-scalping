# Odyssey2 — ETH 라이브 주입 개선 계약 문서 (2026-08-13)

## 상태

| 컴포넌트 | 상태 |
|---|---|
| **Odyssey2 베이스라인 확정** | `locked` — `exit_head` 비대칭 재라벨(h48qual만 교체, zig075 원본 유지)까지가 Odyssey(1)에서 채택된 최종 개선안. 이 시점부터 Odyssey2는 이 상태를 새 비교 기준(reference)으로 삼는다. **주의**: 이 개선안은 아직 섀도우(관찰 전용, 페이퍼)로만 운영 중이며 실제 `trading_bot.py` 라이브 의사결정 경로는 원본 그대로다 — "Odyssey2 베이스라인"은 연구 비교용 기준이지, 실거래 상태를 뜻하지 않는다. |
| `quality_threshold` 선정 코드 | `hygiene_fixed` — OOS-우선 정렬 버그를 h48qual/zig075/BTC를 만드는 2개 스크립트에서 수정(2026-08-13). 배포값 자체는 무변경. |
| `ATR TP/SL floor` 재보정 | `tested_negative_closed` — 배율을 키워 floor 의존도를 낮추는 방향은 VAL에서 결정적으로 부정, OOS 미실행. 이 방향은 닫힘. |
| 다중구간 확인 게이트(`scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`) | `built_and_stress_tested` — 6개 창(2025 Q1~Q3 참고 + VAL + OOS-Q1/OOS-Q2 단일터치) 재사용 모듈 구축, G0 자체검증(4개 스테이지 전부, 세 독립 코드 경로 바이트 단위 렛저 일치 포함) 통과. 기존에 이미 기각된 2개 후보(대기압력·risk-controlled)에 소급 스트레스테스트를 적용해 판정 불변(`REJECTED_SIGN_MISMATCH`)을 직접 확인 — 재심 아님. 앞으로 신규 post-entry 후보는 이 모듈로 심사(아래 "승격 게이트" 절 갱신 참고). 상세: `docs/experiments/eth_omega461_multiwindow_confirmation_gate_20260814.md`. |

## 범위

- 목적: Odyssey(1)이 확정한 베이스라인(라이브 h48qual/zig075 + h48qual exit_head 비대칭 재라벨) 위에서, **Odyssey(1) "독립 재구축" 국면(08-11~12)에서 시도했던 기법들을 direction/quality(진입 선택)가 아니라 post-entry(이미 열린 포지션의 청산·사이징) 컨텍스트로 재적용**해 추가 개선을 찾는다.
- **Odyssey(1)의 핵심 교훈을 그대로 계승**: "이미 확정된 포지션의 부수 로직을 고치는 시도(exit head)는 생존, 스킬 미검증 `direction_head`의 진입 선택 자체를 바꾸는 시도(quality head relabel 등)는 반전" — 이 구분선이 Odyssey2의 우선순위를 결정한다(아래 "Phase 1 아이디어 재적용 트리아지" 절).
- `direction_head`의 방향 스킬 부재는 Odyssey(1)이 GBDT/TabM/오토인코더/TCN/CNN/one-vs-rest/trend-scanning 등 7개 이상의 독립 조합으로 확정한 사실이며, **Odyssey2는 이 결론을 재검증 대상으로 삼지 않는다** — 새로 들어오는 모든 아이디어는 먼저 "이게 진입 선택을 바꾸는가, 청산/사이징만 바꾸는가"로 스크리닝한다.
- 라이브 파일(`trading_bot.py`/`trading_bot_modules/omega4_6_1_live.py`/`runtime_config.py`/`.env`) 미변경 원칙은 Odyssey(1)과 동일하게 유지.
- 승격 절차: Odyssey(1)의 미해결 이슈 12(VAL 구간 신뢰성)·13(섀도우 관찰기간/판단기준 미정)이 Odyssey2에도 그대로 상속된다 — Odyssey2의 새 후보도 같은 제약 아래 평가된다.
- 리소스 레지스트리: `docs/model_contracts/odyssey2_eth_live_injection_data_resources_20260813.md`.
- 선행 계약: `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`(Odyssey 1 — 전체 서사, 결론, 실패 사례 전부 여기 있음. Odyssey2는 이 문서를 대체하지 않고 이어간다).

## 점검 결과 (2026-08-13) — 더 손봐야 할 곳

Odyssey(1) 종료 시점 감사에서 파악된, 아직 열려 있는 항목들:

| # | 항목 | 유형 | 우선순위 |
|---|---|---|---|
| 1 | 레짐별 `quality_threshold` 재보정 (Odyssey1 미해결 이슈 4, 2026-08-11부터 한 번도 착수 안 됨) | 재학습 불필요, 기존 예측 재슬라이싱만 | 최고 — 가장 싸고 가장 오래 방치됨 |
| 2 | 앙상블 불일치(epistemic MI)를 리스크사이징 피처로 사용 — 아키텍처 설계 자체가 "L4 사이징 sidecar 피처 후보로만" 명시했는데 실제로 사이징 모델에 넣어본 적이 없음 | 재학습 필요(사이징 GBM만, TabM 재학습 아님) | 높음 — 설계 의도된 채 미실행 |
| 3 | 오토인코더 latent(139→16) 피처를 사이징/exit_head에 추가 | 재학습 필요 | 높음 |
| 4 | exit_head를 TabM 대신 GBDT(연속회귀)로 — direction에서만 GBDT 시도됨, exit_head 회귀엔 미시도 | 재학습 필요 | 중간 |
| 5 | exit_head에 TCN(시간축 96bar) 적용 — Phase1에서 direction 대상으론 "가장 신호에 가까웠던" 결과, exit엔 미시도 | 재학습 필요, 엔지니어링 부담 중간-높음 | 중간 |
| 6 | zig075 exit_head 개선 — 같은 live-ATR relabel 레시피는 이미 악화로 닫힘(Odyssey1). 다른 접근(개별 재라벨 파라미터, 별도 exit_threshold 등) 미탐색 | 재학습 필요 | 중간 |
| 7 | VAL 구간(2025-10~12) 신뢰성 문제 자체의 근본 원인 규명 (Odyssey1 미해결 이슈 12) | 방법론, 재학습 불필요 | 낮음(오래 걸림, 이번 세션 범위 밖) |
| 8 | 멀티슬롯 상관관계 통제형 배분(Odyssey1에서 가설만 남고 금지됐던 축) | 재학습 불필요하나 원웨이모드 구조적 블로커 있음 | 낮음(배포 불가능한 상태) |

## Phase 1 아이디어 → post-entry 재적용 트리아지

Odyssey(1) "독립 재구축" 국면(08-11~12)에서 시도된 모델/기법 전부를 대상으로, "direction/quality(진입)"가 아니라 "exit_head/사이징(post-entry)"으로 재적용했을 때 새로운 시도인지 아니면 이미 같은 결론에 도달할 게 뻔한 반복인지 재평가:

| Phase 1 기법 | 원래 대상 (Odyssey1 결과) | Post-entry 재적용 후보 | 판정 |
|---|---|---|---|
| GBDT/LightGBM | direction_head 분류 — 8시드×6구간 전패(0/48) | exit_head 연속회귀(TabM 대신) | **재시도 가치 있음** — 분류 실패가 회귀에 그대로 전이된다는 보장 없음, 이미 성공한 live-ATR 라벨 데이터셋 재사용 가능 |
| 오토인코더 latent(139→16) | direction/quality — 분류지표 최대개선이었으나 always-short 못 이김 | exit_head·사이징 GBM에 피처로 추가 | **재시도 가치 있음** — "압축된 넓은 원시 피처"가 청산 타이밍/사이징엔 유효할 수 있음, direction에서 진 건 방향성 베타 문제였지 피처 자체 무가치 때문이 아니었음 |
| TCN(96bar 시퀀스) | direction — Phase1 전체에서 "유일하게 완전 셧아웃 아닌" 결과(VAL 접전), 그러나 정밀 재탐색에서 OOS 0/75로 확정 부정 | exit_head 연속회귀 | **재시도 가치 있음, 최우선급** — "포지션이 이미 열린 뒤 최근 시퀀스가 청산 타이밍에 도움되는가"는 "다음 bar 방향을 맞히는가"와 다른 질문, 이 기법이 Phase1에서 가장 신호에 가까웠다는 점도 고려 |
| CNN 캔들차트 | direction — 처음부터 가장 약한 결과 | exit_head | **낮은 우선순위** — 이미지 인코딩이 position-state 피처(mfe/mae/hold_bars 등) 통합에 자연스럽지 않고, 원래 신호도 제일 약했음 |
| One-vs-rest 전문가(LightGBM×3) | direction — precision 무작위(33%)보다는 높으나 매끄러운 추세를 못 이김 | exit_head는 이미 이진(hold/exit)이라 "one-vs-rest 세분화"가 구조적으로 대응 안 됨 | **해당 없음** — 스킵 |
| MFE 분위수 회귀(quality_head) | quality_head — MI/R² 게이트 유일 통과, always-short 1/5로 패 | 이미 SLTP폭(실패)·멀티슬롯 게이팅(OOS 포트폴리오리스크로 실패)에 두 번 재적용됨 | **거의 소진** — 세 번째 다른 컨텍스트가 명확히 없다면 낮은 우선순위 |
| trend-scanning 라벨 | direction — R²&lt;0으로 결정적 부정 | exit_head는 이미 더 나은 라벨(live-ATR 배리어 해소)을 확보 — 라벨 자체가 전이 안 됨 | **해당 없음** — 스킵 |
| 레짐별 완전분리(hard filter) 학습 | direction — soft/hard 둘 다 부정 | exit_head엔 미적용 — 다만 위 "점검 결과" #1(레짐별 threshold)이 유사한 축을 이미 우선순위에 넣음 | 위 #1과 통합 |

**우선순위 큐 (실행 순서)**:
1. 레짐별 `quality_threshold` 재보정 — 재학습 불필요, 즉시 실행.
2. 앙상블 불일치(epistemic MI) → 사이징 GBM 피처.
3. 오토인코더 latent → 사이징 GBM 피처.
4. GBDT 기반 exit_head 연속회귀.
5. TCN 기반 exit_head 연속회귀.
6. (1~5가 전부 소진되면) 최신 논문 기반 신규 아이디어 탐색.

## 실행 로그

**#1 레짐별 quality_threshold — 완료, 혼재된 결과로 기준선 미달**: h48qual 컴포넌트는 크게
개선(PnL+5.45%→+45.93%, MDD도 개선)됐지만 zig075는 악화, 포트폴리오는 no_gate(PnL대폭개선,
MDD악화)·with_gate(PnL악화, MDD개선)가 서로 반대 방향이라 사전등록 기준(4개 지표 전부
비악화) 미달 — **OOS 미실행**. 게이트를 느슨하게 하는 효과라 direction_head 무스킬 문제와
같은 계열의 트레이드오프로 해석됨. 상세:
`docs/experiments/eth_omega461_regime_specific_quality_threshold_20260813.md`.

**#1 후속(2026-08-14, 사용자 제안) — exit_head와 동일한 비대칭 채택(h48qual만) 테스트,
근접했으나 여전히 기각**: zig075를 완전히 원본(전 레짐 0.75)으로 둔 채 h48qual만 레짐별
threshold 적용. no_gate는 PnL·MDD 둘 다 개선(깔끔한 승리, +36.82%→+67.05%/-24.34%→-17.21%)
했지만 **with_gate의 PnL만 기준선보다 낮음**(+54.88%→+42.27%, MDD는 개선). 사전등록 기준(4개
지표 전부) 중 1개 미달로 **여전히 기각, OOS 미실행** — 근접했다고 사후에 기준을 완화하지
않음. zig075를 안 건드린 게 원 실험의 핵심 문제(zig075 악화·no_gate/with_gate 정반대 신호)를
대부분 해소했다는 점은 유의미. 상세:
`docs/experiments/eth_omega461_regime_specific_quality_threshold_h48qual_only_asymmetric_20260814.md`.

**#2 앙상블 불일치(epistemic MI) → 사이징 피처 — 진행 중**: 추출 인프라 구축 완료
(`scripts/build_eth_ensemble_epistemic_context_features_20260813.py`,
`train_omega1_regime3_routed_expert_direction_quality_20260602`의 raw threshold-무관 컬럼과
기존 `diagnose_eth_h48qual_ensemble_disagreement_20260811.predict_members`를 재사용, 순수
추론만이라 재학습 불필요). **h48qual에서 직접 검증한 예비 발견**: k=8 멤버간 실제 불일치가
존재하긴 하나(std≈0.003) 극히 작다 — TabM의 파라미터 공유형 효율적 앙상블 구조 자체의 특성으로
보임(버그 아님, 직접 확인). 사이징 GBM(`train_eval_omega4_2_risk_sidecar_20260622.py`의
`--risk-context-feature-dir`, 기존에 한 번도 안 쓰인 확장점)으로 실제 재학습 완료 —
**결과: 사실상 무변화(null result)**. h48qual/zig075 둘 다 VAL·OOS 전부 PnL ±0.5%p, MDD
±0.1%p, 거래수 완전 동일 — GBM이 극저분산 피처를 사실상 무시함. 실험 중 h48qual/zig075가
같은 `--out-suffix`를 써서 출력이 충돌하는 버그를 발견·수정(컴포넌트별 고유 접미사로
재실행). Odyssey(1)의 게이팅 무신호(순위상관) 결과와 합쳐 **앙상블 불일치 신호원 전체가
이 프로젝트에서 소진**된 것으로 취급. 상세:
`docs/experiments/eth_omega461_ensemble_epistemic_sizing_feature_20260813.md`.

**#3 오토인코더 latent → 사이징 피처 — 완료, 부정 결과(zig075는 뚜렷한 악화)**: 추출 스크립트
(`scripts/build_eth_autoencoder_latent_context_features_20260813.py`, Odyssey(1)의 139컬럼
풀+동일 아키텍처를 재사용하되 사이드카 자체 프레임 구성에 맞춰 재적합) 완료 후 사이징 GBM
재학습. h48qual은 거의 무변화(소폭 악화), **zig075는 OOS PnL -5.93%p·MDD -1.82%p 둘 다
악화** — 소수 표본(28/13건)에 16차원을 더해 과적합했을 가능성. Odyssey(1)의 "분류지표
개선해도 always-short 못 이김" 패턴이 사이징 맥락에서 "GBM이 피처를 쓰긴 하나 과적합"으로
재현. 상세: `docs/experiments/eth_omega461_autoencoder_latent_sizing_feature_20260813.md`.
원시 피처풀 압축 방향은 direction/quality(Odyssey1)·사이징(Odyssey2) 양쪽에서 소진.

**#4 GBDT 기반 exit_head — 완료, VAL 게이트 실패로 OOS 미실행**: h48qual exit_head를 TabM
대신 LightGBM(레짐별 bull/bear/chop 3개 분리 모델, TabM `_fit_exit_head_only`와 동일한
`balanced × 소프트 라우팅확률` 가중치)로 학습, 라벨을 직접 재확인한 결과 이진(hold=0/exit=1)
분류였다(점검 결과 표의 "연속회귀"는 실제 라벨과 다른 사전 가정이었음 — 직접 확인 후 이진
분류로 정정해 진행). 라이브ATR 재라벨 데이터셋을 TabM `full1500` 런과 동일 시드·후보수로
재구축해 행수/양성개수/사용후보수 3개 지표 전부 일치함을 확인했다. G0 자체검증(기존 코드로
발표된 baseline/TabM 수치 재현) 통과 후 `_predict_exit_prob_one`을 흉내내는 duck-typing
래퍼(`softmax(log(predict_proba))==predict_proba`)로 시뮬레이션 코드 무수정 재사용. **결과:
포트폴리오 레벨(PnL+46.59%→+101.27%, MDD-21.70%→-19.81%)은 둘 다 개선이지만, 컴포넌트
레벨(h48qual 단독, PnL+9.23%→+2.72%, MDD-7.59%→-7.69%)은 둘 다 악화** — 사전등록 게이트(4개
지표 전부 비악화)가 2개 미달이라 **OOS는 열지 않았다**. GBDT exit_head가 TabM보다 훨씬 이르고
잦은 청산(평균 보유기간 210.8→144.9bar)을 학습해 컴포넌트 단독 경제성은 나빠지지만, 그만큼
공유 슬롯을 자주 비워 h48qual 자신의 재진입 기회가 늘어나는 포트폴리오 상호작용(슬롯 승자
13건→16건)이 포트폴리오 지표를 밀어올린 것으로 해석 — #1(레짐별 threshold)과 같은 계열의
"레벨별 반대 방향이라 어느 쪽도 깔끔한 승리가 아님" 판정. 우선순위 큐 이 항목은 이것으로
종결(부정 결과). 상세: `docs/experiments/eth_omega461_gbdt_exit_head_20260813.md`.

**#5 TCN 기반 exit_head — 완료, VAL 게이트 실패로 OOS 미실행(우선순위 큐 최종 항목)**: h48qual
exit_head를 TabM 대신 TCN(시간축 48bar 윈도우, `verify_eth_h48qual_tcn_sequence_model_20260812.py`
/`tune_eth_h48qual_tcn_sequence_model_hpsearch_20260812.py`의 아키텍처와 HP탐색 채택값 재사용)으로
학습, #4(GBDT)가 만든 `_build_dataset` 함수를 무수정 import로 재사용해 동일 데이터셋(1,234,431행)
재현을 코드 재사용으로 보장했다. GBDT와 달리 TCN은 매 bar 결정에 과거 윈도우가 필요해
`_predict_exit_prob_one`의 "단일 행만 모델에 전달" 구조를 그대로 못 썼다 — `_predict_exit_prob_
one`/`replay_exit_variant`/`greedy_replay` 셋 다 무수정, 대신 이름을 바꾼 복사본
(`_predict_exit_prob_one_windowed`/`replay_exit_variant_windowed`/`greedy_replay_windowed`)을 새
스크립트에 만들어 exit_head 호출부만 윈도우 슬라이스로 바꿨다(포트폴리오 리플레이는 h48qual(TCN)
·zig075(원본 TabM)를 같은 루프에서 다루므로 모델별 `IS_WINDOWED` 마커로 동적 분기).
row_i(=`exit_path_entry_i`+`exit_path_hold_bars`) 복원값을 `frame_exit` 타임스탬프 전체 행과
대조(100% 일치)하고 시장피쳐 재구성값을 `cur_` 컬럼과도 대조(10,000셀 무작위 샘플 불일치 0건)해
"같은 데이터에 히스토리만 추가"임을 직접 확인했다. G0 자체검증(4개 지표 정확 일치) 통과 후
CPU-only dev 박스에서 축소 없이 전체 1500후보 데이터셋으로 완주(데이터셋 재구축 638초 + 레짐별
학습 3×491초, held-out AUC≈0.997~0.998로 GBDT와 비슷한 판별력). **결과: 포트폴리오 레벨
(PnL+46.59%→+60.24%, MDD-21.70%→-21.64%)은 개선이지만, 컴포넌트 레벨(h48qual 단독, PnL
+9.23%→-7.74%로 부호 반전, MDD-7.59%→-8.28%)은 GBDT보다 더 크게 악화** — 사전등록 게이트
(4개 지표 전부 비악화)가 2개 미달이라 **OOS는 열지 않았다**(`research_eth_omega461_tcn_exit_
head_oos_20260813.py` 실행 시 `RuntimeError`로 즉시 중단됨을 직접 확인). 평균 보유기간
210.8→11.0bar(-95%, exit_head 발동 100%)로 GBDT(210.8→144.9bar, -31%)보다 훨씬 공격적인
조기청산을 학습했다 — 원장(hold_bars 0~57bar, 표준편차 15.3)에서 뚜렷한 분산을 확인해 상수출력
버그가 아님을 배제했다. TCN이 GBDT보다 42배 많은 입력차원(4,896 vs 115)을 받는다는 점이 VAL
일반화 실패를 더 키웠을 가능성을 해석으로 제시했으나 이번 실험에서 직접 검증하지는 않았다.
우선순위 큐 이 항목은 이것으로 종결(부정 결과). 상세:
`docs/experiments/eth_omega461_tcn_exit_head_20260813.md`.

**우선순위 큐(1~5) 전부 소진** — 계약서 51~57행의 재적용 큐가 이 #5로 완료됐다. 다음 단계는
같은 절의 6번 항목대로 최신 논문 기반 신규 아이디어 탐색으로 전환한다(이번 세션 범위 밖).

**#6 최신 논문 기반 신규 아이디어 탐색(문헌 리서치+랭킹) — 완료, 구현/학습 없음**: 우선순위
큐의 마지막 항목을 실행. GBDT/TCN exit_head가 공통으로 드러낸 "exit를 공격적으로 할수록 컴포넌트
단독 PnL은 악화하는데 공유 슬롯을 자주 비워 포트폴리오 지표는 개선된다"는 패턴(자본 기회비용을
명시적으로 값매기지 않은 채 청산만 최적화한 결과로 해석)을 최우선 질문으로 삼아 4개 방향(자본
기회비용/불확실성 인식 사이징/메타라벨링 2024~2026 발전/RL 기반 exit)을 문헌 조사, Odyssey1의
기존 두 리서치 문서와 중복 배제 확인. **핵심 발견: 이 현상이 OR 문헌의 "retirement formulation"
(Gittins index, 자원 하나를 놓고 경쟁하는 옵션들의 최적 정지)과 정확히 같은 구조**라는 재정식화 —
Dhankhar/Mishra/Bodas(arXiv:2405.01157, 2024)의 Deep RL Gittins index 학습(QGI/DGN)이 이론적
근거, 이를 재학습 없이 근사하는 "대기압력(반대 컴포넌트가 진입 대기 중인가) 후처리 exit 규칙"을
1위로 랭킹(검증비용 최저, 기존 GBDT/TCN 리플레이 하네스의 슬롯 점유 추적 재사용). 2~4위는
Risk-Controlled Post-Processing(Joshi/Wang/Hassani/Dobriban, arXiv:2605.06479, 2026 — 기존
`EXIT_THRESHOLD=0.95`를 위험 제어된 임계값으로 재보정, calibration-only 재학습 불필요),
Conformal Kelly 스타일 구간폭 사이징 스케일(Ryan, arXiv:2608.01494, 2026), Selective Conformal
Risk Control(Xu/Guo/Wei, arXiv:2512.12844). Evidential Deep Learning(Odyssey1이 이미 "신중
취급"했던 candidate B)은 반박 논문("Is EDL a Mirage?" NeurIPS 2024) 이후 2025~2026까지도 활발한
수리 시도(6편 이상)에도 합의된 해법이 없음을 확인해 순위 밖으로 재확인. RL 기반 exit은
기존 bandit 게이트 실패(2026-07-09, skip rate 96.64%)와 질적으로 다른 신규 후보를 못 찾아
낮은 우선순위 유지. **다음 실행 대상(1위 후보)**: 대기압력 후처리 exit 규칙 — 재학습 없이 기존
포트폴리오 리플레이 데이터로 시험 가능. 상세:
`docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`.

**#7 대기압력(Queue Pressure) 조건부 exit_head 임계값 — 완료, VAL 통과 후 OOS 반전으로 부정
결과 종결**: #6 문헌 스카우팅의 1위 후보를 구현. h48qual exit_head **모델은 TabM 라이브ATR
재라벨(현재 확정 베이스라인) 그대로 두고, 고정 `EXIT_THRESHOLD=0.95`만 조건부로 낮췄다** —
h48qual이 슬롯을 보유한 bar에서 zig075의 `dir_action!=CASH & quality_for_action>=0.75`("대기
압력")가 있을 때만 threshold를 후보값(0.80/0.85/0.90 그리드)으로 낮추고, 없으면 0.95 유지.
zig075 자신의 exit 로직/모델/threshold는 전혀 건드리지 않았다(대기압력 신호의 소스로만 읽기
전용). `replay_omega4_6_1_greedy_router_20260706.greedy_replay`는 무수정, 이름 바꾼 복사본
(`greedy_replay_queue_pressure`, exit_head threshold 선택 한 블록만 조건부로 교체)을 새로
만들었다 — GBDT(#4)의 duck-typing, TCN(#5)의 윈도우 슬라이싱 복사본과 같은 패턴. G0(포트폴리오
레벨, 46.59%/-21.70%/35건 재현) 통과 후, 대기압력 마스크를 원시 `dir_action`/`quality_for_action`
컬럼과 이미 계산된 `zig075_dec['side']!=0`(같은 threshold가 이미 반영된 값) 양쪽으로 교차검증해
VAL·OOS 전체 bar에서 불일치 0건 확인. **대기압력 발생 빈도는 h48qual 보유 bar의 6.6~7.9%**로
일관되게 드물지도 무조건적이지도 않음을 진단으로 확인(GBDT/TCN의 "거의 매번 발동"과 질적으로
다름). **VAL 결과: 포트폴리오 레벨 게이트(PnL·MDD 둘 다 비악화)를 threshold=0.80만 통과**
(PnL+46.59%→+52.77%, MDD-21.70%→-21.70%(동일), 38건) — 0.85는 PnL이 더 높았지만(+57.27%) MDD가
악화(-22.40%)해 탈락, 0.90은 PnL 자체가 악화(+43.78%)해 탈락. **OOS 단일 확인(threshold=0.80):
PnL이 +93.27%→+59.08%로 뚜렷이 반전(-34.19pp)**, MDD는 사실상 동률(-15.48%→-15.48%) — 반전의
원인은 전적으로 PnL. GBDT/TCN과 달리 exit_head를 무조건 공격적으로 만든 구조적 결함이 아니라
(대기압력 자체가 조건부·저빈도임을 확인했으므로), 이 프로젝트가 반복 관찰한 "30~40건대 소표본
VAL 승리가 OOS로 일반화되지 않는" 패턴에 더 가까운 것으로 해석된다(추정). 우선순위 큐 6번
항목(최신 논문 기반 신규 아이디어)의 1위 후보는 이것으로 **종결**한다(부정 결과). 상세:
`docs/experiments/eth_omega461_queue_pressure_exit_threshold_20260814.md`.

**완화된 기준 재채점(2026-08-14, 사용자 요청)**: 결과를 보기 전에 원칙적인 새 기준을 먼저
확정(with_gate PnL 개선을 주기준, MDD 3%p 이내 악화 허용, exit_head 모델 자체를 바꾼
실험에는 컴포넌트 50% 상대악화/부호반전 금지 가드레일)한 뒤 레짐threshold(joint·h48qual만)·
GBDT·TCN·대기압력 전부 기계적으로 재채점 — **전부 기각 유지**. GBDT/TCN은 with_gate
포트폴리오만 보면 원래보다 더 극적으로 좋아 보이지만(GBDT+120.20%, TCN+84.87%) 컴포넌트
경제성이 그만큼 더 나빠진 신호였고(TCN은 손실 전환), 가드레일이 정확히 이를 차단했다.
레짐threshold 둘 다 with_gate PnL 자체가 baseline보다 낮아 관대한 기준에서도 미달. 대기압력
OOS 반전도 with_gate로 다시 봐도 그대로(-44% 상대하락). **원 "4개 지표 전부" 기준이
자의적으로 과했던 게 아니라 실제 문제를 걸러내고 있었다는 근거로, 원 기각 판정 신뢰도가
강화됨.** 상세: `docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md`.

**#8 Risk-Controlled Post-Processing of Decision Policies — 완료, VAL 이중 게이트 통과 후 OOS
반전으로 부정 결과 종결(문헌 스카우팅(#6) 2위 후보 소진)**: Joshi/Wang/Hassani/Dobriban
(arXiv:2605.06479, 2026-05-07) 원문을 WebFetch로 직접 읽고 정확한 메커니즘(Theorem 3.1 threshold
정책, Algorithm 1 유한표본 calibration, Theorem 4.2 O(log n/n) 초과위험 보장)을 확인한 뒤 구현.
h48qual exit_head **모델**은 TabM 라이브ATR 그대로 두고(#7과 동일 원칙), 이미 학습된 GBDT
exit_head(#4, `gbdt_exit_bundle.pkl`)를 **자신의 확률을 TabM과 독립적인 위험추정 arbiter로
재사용**해 `Δ(x)=g(π₀(x),x)-g(π*(x),x)`(g(hold,x)=p_GBDT, g(exit,x)=1-p_GBDT)를 구성 — π0·π*가
합의하면 Δ=0(전환불필요), 강하게 불일치할수록 커진다. y(calibration 전용 ground truth)는
GBDT/TCN 문서가 이미 확인한 exit_head 라벨의 98.1%인 `pos_giveback≥0.65 OR pos_unrealized≤
-0.010` 규칙을 매 bar 인과적으로 재사용(신규 피처 0개). Algorithm 1로 VAL(h48qual 컴포넌트 단독,
13,330개 보유 bar)에서만 calibration, 사전등록 ε그리드({0.90,0.70,0.50}×baseline 자체 불일치율)
중 **eps_frac=0.90(τ̂=0.9995)만 원 기준(4개 지표 전부)과 새 기준(with_gate 개선+MDD 3%p+가드레일)
둘 다 통과** — 포트폴리오 3,678개 보유 bar 중 단 7개(0.19%)만 건드리는 매우 좁은 개입이었고,
GBDT/TCN과 달리 **컴포넌트 레벨도 개선**(+9.23%→+9.61%)됐다는 점에서 이 서브프로젝트의 post-entry
후처리 실험 중 최초로 두 기준을 동시 통과. 그러나 **OOS 단일 확인(재보정 없이 τ̂=0.9995 고정
적용)에서 포트폴리오가 뚜렷이 반전**(no_gate PnL+93.27%→+21.18%, MDD-15.48%→-28.70%; with_gate
PnL+67.25%→+4.77%)했다 — 컴포넌트는 OOS에서도 개선(+0.53%→+9.05%)을 유지했으므로 반전의 원인은
컴포넌트 경제성이 아니다. 원장을 직접 대조해, 단 4건의 전환(→exit 3, →hold 1)이 h48qual 자신의
거래는 개선시키면서도 공유슬롯이 zig075에게 풀리는 시점을 미세하게 바꿔 zig075가 이후 다른(더
나쁜 stop_loss 2건 추가) 거래 시퀀스를 잡게 만들었음을 확인 — GBDT/TCN/#7이 반복 관찰한 "슬롯
재순환" 상호작용이지만 이번엔 **컴포넌트에 유리한 전환이 포트폴리오에 불리하게 작용**한 새로운
변형이다. 논문의 Theorem 4.2 보장은 y-규칙(대리 불일치) 위반율에 대한 것이지 실현 PnL에 대한
것이 아니므로, 이 실패가 논문 메커니즘의 결함을 뜻하지는 않는다 — 다만 이 프로젝트의 단일계좌
공유슬롯 구조에서는 극히 좁고 컴포넌트에 유리한 개입도 포트폴리오 레벨 소표본 반전에 취약할 수
있음을 보여준다. 이 과정에서 `docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md`의
"baseline with_gate PnL+54.88%/MDD-31.11%"가 실제로는 `asymmetric_tabm_liveatr`(35건, 이
프로젝트의 진짜 비교기준)가 아니라 `baseline_both_original`(29건, 둘 다 원본) 원장의 with_gate
값이었음을 직접 재현으로 발견 — 과거 GBDT/TCN/레짐threshold/대기압력의 판정 자체는 이와 무관한
별도 사유(가드레일·부호반전·PnL 자체 하락)로 이미 결정적이라 재검토 대상 아니지만, 향후 같은
지표를 재사용할 때 어떤 원장 기준인지 명시할 필요가 있다는 기록으로 남긴다. Odyssey2 문헌
스카우팅(#6) 2위 후보는 이것으로 **종결**한다(부정 결과). 상세: `docs/experiments/
eth_omega461_risk_controlled_post_processing_exit_fallback_20260814.md`.

**방법론 변경(2026-08-14) — 다중구간 확인 게이트 구축**: 이번 항목은 후보 결과가 아니라 이
서브프로젝트가 앞으로 VAL/OOS를 다루는 **방법 자체의 변경**이다. #7(대기압력)·#8(risk-controlled)
둘 다 "VAL 승리 → OOS-Q1 단일 확인 → 반전"으로 끝났는데, 같은 밤 독립적으로 작성된
`docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md`가 다른 4개 후보(최종보스
v2/v3·SLTP 재보정·멀티슬롯 MFE게이팅)에서도 동일 패턴을 발견해 "3개월 단일 OOS 창도 그 자체로는
약한 증거"라고 결론짓고 "VAL+OOS-Q1+OOS-Q2, 가능하면 2025 Q1~Q3까지 포함한 4개 이상의 방향이 섞인
독립 구간에서 부호 일치 확인 전엔 확인됐다고 쓰지 않는다"는 권고를 남겼다. 그 권고를 코드로
구현한 재사용 모듈 `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`를 구축했다 —
`load_all_windows`(6개 창 로더)/`verify_windows`/`align_frame_and_predictions`(범용 정렬 유틸)/
`run_portfolio_variant`(baseline-shape 전용, 후보별 개입 로직은 각 후보 스크립트 재사용)/
`summarize_multiwindow`(표·판정 헬퍼). G0 자체검증 4단계(창 로딩 검증, VAL·OOS-Q1
asymmetric_tabm_liveatr 재현, 2025 Q1/Q2/Q3 baseline_both_original을 두 개의 독립 코드 경로로
재현, `_metrics(apply_gate=True)`↔`_duration_gated` 수학적 동치성) 전부 통과했고, 부가로 baseline
렛저가 세 가지 독립 코드 경로(순수 `greedy_replay`, 대기압력의 퇴화모드, risk-controlled의
퇴화모드)에서 6개 창 전부 바이트 단위로 정확히 일치함을 확인했다. **이미 결정적으로 기각된 대기압력
(threshold=0.80)·risk-controlled(eps_frac=0.90, τ̂ 고정)에 이 모듈을 소급 적용**(재심 아님, 모듈
자체의 스트레스테스트) — 6개 창 전부에서 기존 OOS-Q1 재현 수치가 정확히 일치했고, **둘 다
OOS-Q2만 단독으로 봤다면 오히려 "통과"로 보였을 것**(대기압력 with_gate -1.73%>-12.69%,
risk-controlled with_gate -12.01%>-12.69%, MDD는 사실상 동일)이라는, 이 방법론 자체를 실증적으로
뒷받침하는 사실을 새로 발견했다 — OOS-Q1(반전)과 OOS-Q2(단독으론 통과) 사이의 부호불일치 자체가
단일 OOS 창의 위험성을 보여주는 직접 증거다. 단일터치(oos_q1 AND oos_q2 동시) 판정은 두 후보 다
`REJECTED_SIGN_MISMATCH` — **기존 기각 판정은 바뀌지 않았다**(예상된 결과, `report.json`
`this_module_changes_prior_verdict=false`로 직접 확인). 상세: `docs/experiments/
eth_omega461_multiwindow_confirmation_gate_20260814.md`.

**#9 Conformal Kelly 사이징 스케일 — 완료, VAL 게이트 자체를 통과 못해 OOS 미실행(문헌 스카우팅(#6)
3위 후보 종결)**: Ryan(arXiv:2608.01494, 2026-08-02) 원문을 PDF 직접 fetch로 정독하고 정확한 공식
(`sigma_hat=q_eff/z`, `f=kappa*mu_hat/sigma_hat^2`, `q_eff`=롤링+확장앵커 conformal 분위수 블렌드)을
확인했다. **문헌스카우팅 요약 정정**: "40개 설정 중 다수가 pre-registered holdout에서 저조"는
부정확 — 원문의 "40"은 전부 DEV 윈도우 내부(레버리지컷 다이얼 기각 40+건, 플라시보 검증 40-way)의
숫자이고, 진짜 사전등록 lockbox(2022~2024)는 config 2개만 테스트해 **둘 다** development 값의
약 30%로 급락(연성장 28%대→8%대)해 11개 비교 대상 중 Sharpe·Calmar 꼴찌를 기록했다 — 캘리브레이션
자체(0.745 vs 0.750 목표)는 정직하게 전이됐지만 Kelly 사이징의 경제적 가치는 전이되지 않았다는
게 원 논문 §12 결론. #7·#8과 달리 이번은 **사이징 축**(exit-timing 아님) — 사이징 GBM
(`risk_sidecar.pkl`)의 `score` 출력이 `risk_target_mode="net"`/`target_mae_penalty=0.0`(두 라이브
사이드카 모두 확인)로 `net_per_notional`을 정확히 예측하도록 학습돼 있다는 점을 이용해, `|실현
net_per_notional - 진입시점 score|`를 conformal 잔차로 정의(mu_hat을 처음부터 재도출하는 대안은
사이징 모델의 기존 edge 추정과 이중계산되어 기각). CLAUDE.md Futures Risk Sizing Contract 그대로
`margin = base_sweep.rs._risk_margins(...)` 직후 스케일을 곱하는 지점에 개입했고, 두 라이브
사이드카 모두 `notional_scaled_sltp=False`임을 직접 확인해 TP/SL 재승산(레버리지 이중계산) 경로
자체가 없음을 확인했다. 구현 중 두 가지를 직접 발견해 수정·공개했다: (1) 최초 구현이 창이 바뀔
때만 캘리브레이션 풀을 갱신하는 버그(창 내부에서 스케일이 상수로 고정됨, 로그로 발견해 수정),
(2) `notional_scaled_sltp`와 무관하게 exit_head 모델 자체가 `notional`/`leverage`를 position-state
입력 피처로 받는다는 사실 — margin 스케일링이 청산 타이밍을 bar 단위로 미세하게 바꿀 수 있음을
뜻해 "단일 baseline pass로 거래 타이밍이 스케일과 완전 무관하다"던 최초 가정을 정정했다(2차
근사로 공개, 미래정보 누설은 아님 — 실측 괴리: zig075는 전 창 완전동일, h48qual은 OOS-Q2 동일·
OOS-Q1 2건만 4bar 이내 이동·VAL만 63건 중 다수 재배치). G0(VAL/OOS-Q1 4개 지표, 게이트모듈 경로+
자체 walk-forward 경로 둘 다) **PASS**. VAL: (scale_floor,scale_cap) 3그리드(narrow 0.85~1.20/
medium 0.70~1.40/wide 0.50~2.00) 전부 실현 스케일 범위(0.91~1.04)가 가장 좁은 grid 안에도 완전히
들어가 3후보 결과가 완전히 동일 — **원기준·완화기준 둘 다 3후보 전부 실패**(no_gate
46.59%→39.72%, with_gate 77.31%→52.09%, MDD도 소폭 악화). VAL 자체를 못 넘겨 **OOS-Q1+OOS-Q2
단일터치는 미실행**(단일터치를 시험할 기회조차 없었음) — Odyssey2 #1/#4/#5와 같은 절차. 최종판정
`REJECTED_VAL_GATE`. #7·#8("VAL 승리→단일 OOS 반전")과는 다른 실패 계열(VAL 자체 실패)이며, 원
논문의 진짜 lockbox 결과와 방향이 같다는 점을 판정 서술에 명시했다. 상세: `docs/experiments/
eth_omega461_conformal_kelly_sizing_scale_20260814.md`.

**문헌스카우팅(#6) 랭킹 1~3위 전부 종결**(1위 대기압력=#7 VAL승리 후 OOS반전, 2위
risk-controlled=#8 VAL승리 후 OOS반전, 3위 conformal kelly=#9 VAL 자체 실패) — 4위(Selective
Conformal Risk Control)·5위(Gittins Deep RL 전체판)는 이번 세션 범위 밖으로 보류.

**(2026-08-14 후속 갱신) 문헌스카우팅(#6) 랭킹 1~5위(전체) 종결**: 4위 Selective Conformal Risk
Control=#14(VAL calibration 표본 붕괴로 자체 실패), 5위 Gittins Index Deep RL(QGI/DGN)=#16(VAL
게이트+가드레일 결정적 실패, OOS 미실행) — 문헌 스카우팅 큐 전체가 부정 결과로 마감됐다.

**#10 exit_head liveATR 재라벨 — 지속 상승장 취약성 발견(2026-08-14, 사용자 질문 계기, 신규
후보 아니라 리스크 진단)**: 9건 연속 기각 이후 사용자가 "모든 쿼터가 수익일 순 없지만 상승장에서
못 버는 건 별개 문제"라며 조사를 요청. #8(다중구간 게이트) G0b가 부산물로 남긴, 현재 섀도우
baseline(`asymmetric_tabm_liveatr`)의 2025 Q1/Q2/Q3 수치를 렛저 레벨로 파고든 결과 — Q1·Q2는
재라벨 전 원본 대비 뚜렷한 개선인데, **2025년 유일한 강한 지속 상승장인 Q3(드리프트 +66.63%,
세 분기 중 최저 변동성)에서만 원본보다 4.7배 악화**(no_gate -9.73%→-46.26%). 원인은 h48qual
거래수 폭증(8건→18건, 전부 SHORT) — exit_head 재라벨이 평균 보유기간을 2~3배 단축시키는 "회전
가속기"인데, Q1처럼 거친 구간에서는 거래수가 안 늘지만(8→8) Q3처럼 노이즈 적은 지속 추세에서는
풀린 슬롯이 "이미 알려진 무편향 숏 신호"를 반복 재점화시켜 나쁜 거래 개수 자체가 폭증한다.
zig075 숏의 Q3 약세(-0.517, 어제 조사와 정확히 일치) 자체는 재라벨과 무관한 기존 발견의 재현.
**아직 forward(OOS)로 검증된 적 없는 in-sample 증거**(확보된 OOS 구간엔 지속 상승 레짐이 없음)
— Odyssey(1) 미해결 이슈 13(exit_head 섀도우 관찰기간·승격기준 미정)에 구체적 후보 기준 하나를
제안: 지속 상승장을 최소 한 번 섀도우 관찰하기 전에는 승격하지 않는다(양쪽 계약 문서에 교차기록).
다음 방향(미검증, 제안만): 레짐인지형 exit_head(지속상승 레짐에서만 원본 exit로 되돌리는 조건부
정책) — 오늘 밤 기각된 레짐별 quality_threshold(entry 게이트 축)와는 다른 축(exit 정책 자체를
레짐조건부로). 상세: `docs/experiments/eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md`.

**#11 레짐인지형 exit_head 지속상승장 가드 — #10이 제안한 완화안 검증, 부분 완화 확인되나
승격 근거는 아님(2026-08-14)**: #10이 미검증 제안으로만 남긴 "지속 상승 레짐에서만 h48qual을
원본 exit로 되돌리는 조건부 정책"을 실제 구현·검증. 라이브 regime3 HMM은 어제 조사가 이미
Q3 구분력 없음을 보였으므로 재사용하지 않고, 기존 피처 패널에서 causal 탐지기를 새로 찾았다 —
`regime_persistence`/`chop_index`/`hurst_48` 등 순간값(≤288bar) 기반 후보는 전부 quarter
구분에 실패(regime3와 동일 패턴 재현)했고, **유일하게 `dual_momentum`(이미 1주 lookback 설계)
의 평균값만 quarter 드리프트 방향을 깨끗이 추적**(Q3 평균+0.248, 다음으로 높은 Q2도 -0.014).
단순 조합 1개만 시도: `(dual_momentum>0)`의 rolling(2016bar=1주, `dual_momentum` 자신의 기존
lookback 재사용, 새 숫자 아님).mean(). 임계값(90th percentile)은 **2025 Q1+Q2만으로
캘리브레이션**(Q3 제외, threshold=0.802579)한 뒤에야 Q3 활성화율을 확인하는 순서를 지켰다 —
결과 Q3 43.0% vs 나머지 5개 창 5.4~11.6%(4~8배 차이, 75th/95th percentile로도 재현되는 강건한
분리). 개입은 h48qual의 **보유 중 exit_head 확률을 어느 학습된 모델(원본 vs liveATR 재라벨)로
물을지**만 조건부로 바꾸며(exit_threshold=0.95는 공통), entry/사이징은 가드 상태와 무관하게
항상 liveATR 준비값(두 번들의 dec/margin/leverage가 VAL에서 완전히 동일함을 직접 확인)만
사용 — zig075는 완전 동결. G0(게이트 모듈 재사용 경로 + 신규 렌임드카피 가드-미부착 경로 둘 다)
4개 지표 정확 일치로 **PASS**. 결과: **Q3 손상이 뚜렷이 완화**(no_gate -46.26%→-37.43%, 원본
-35.54%까지 격차의 82.4% 회복; with_gate -18.87%→-15.86%, 격차의 32.9% 회복, MDD는 원본과
정확히 일치) — 완전히 사라지지는 않았다. **Q1/Q2 이득은 100% 보존**(Q1은 가드 0회 발동, Q2는
1,340회 발동했으나 원본·재라벨 모델의 실제 결정이 단 한 번도 갈리지 않아 두 창 모두 재라벨판과
byte-identical). **VAL/OOS-Q1/OOS-Q2 비악화는 원기준·완화기준 둘 다 CONFIRMED**(세 창 모두
재라벨판과 원장 완전 동일). 세 조건이 다 성립해도 **"승격 가능"이라 쓰지 않는다** — 이 결과
전부가 2025 in-sample 또는 재라벨과 무관한 하락/혼조 OOS에서 나온 것이고, 실제 지속상승
레짐에 대한 forward 검증은 여전히 없다(확보된 OOS 구간엔 아직 그런 레짐이 없음, #10과 동일한
한계). 정직한 결론은 **"섀도우 관찰 대상으로 추가할 가치가 있는 후보"** 정도가 최대치. 상세:
`docs/experiments/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.md`.

**#12 레짐별 quality_threshold(h48qual만) 부활 시도 — 사이드분리로 오늘 밤 최고 VAL, 그러나
OOS 양쪽 창 다 결정적 기각(2026-08-14, 사용자 요청)**: #1 후속(with_gate PnL 1개 지표만 근접
미달로 기각)의 렛저를 직접 재생·분해해 원인 규명 — h48qual LONG쪽에 새로 열린 14건(with_gate
활성)이 순이익 거의 0(+0.005)인 채 포트폴리오를 희석시키는 반면 SHORT 18건은 순수 이득(+0.397
활성분)임을 확인. SHORT는 원 실험이 찾은 레짐맵(0.30/0.30/0.35) 그대로 두고 LONG 임계값만
전 그리드(0.30~0.80+완전차단)로 독립 스윕 — **0.35는 양쪽(0.30·0.40)이 실패로 둘러싸인 고립
스파이크라 기각**(단일지점 과최적화 경고신호로 판단, 채택 안 함), 대신 **0.65~1.01 전 구간이
동일 수치로 안정된 고원**(h48qual LONG 사실상 완전 차단 — 이 프로젝트가 반복 확인한 "LONG 방향
스킬 없음" 결론의 직접 적용, 임의 수치 아님)을 채택. G0/G0b PASS. **VAL: 4개 지표 전부 큰 폭
개선**(no_gate +36.82%→+72.31%/-24.34%→-21.57%, with_gate +54.88%→+98.96%/-31.11%→-19.73%) —
오늘 밤 전체 후보 중 최고 VAL 성과. 다중구간 게이트로 **OOS-Q1+OOS-Q2 단일터치 개방 →
`REJECTED_BOTH_WINDOWS_FAIL`**(원기준·완화기준 둘 다, 두 창 모두 실패 — OOS-Q1 no_gate
+49.32%→+26.69%/with_gate +44.48%→+18.27%, OOS-Q2 no_gate +3.13%→-13.63%/with_gate
+9.85%→-0.49%, 전부 PnL·MDD 동시 악화). 원인 추정: LONG 처리는 원칙적으로 고쳤으나, **SHORT
레짐맵(0.30/0.30/0.35) 자체가 원 실험의 컴포넌트 단변량 스윕에서 이미 VAL PnL을 직접 최대화해
선택된 값**이라 그 선택편향은 이번 수정으로 전혀 해소되지 않았다 — `eth_val_oos_regime_mismatch_
investigation_20260813.md`가 경고한 "저표본 VAL 창의 다중 재사용" 문제의 재현. 채택 0건이지만
**h48qual LONG 스킬 부재를 완전히 다른 방법(렛저 사이드 분해)으로 8번째 독립 재확인**했다는
진단 가치는 남는다. 대기압력·risk-controlled에 이은 세 번째 "VAL 클린 통과→OOS 반전" 사례이자
가장 결정적(양쪽 창 다 실패)인 사례. 부수 발견: `research_eth_omega461_regime_specific_quality_
threshold_20260813.portfolio_eval`이 커버리지 갭 있는 창(2025q1/q3)에서 `IndexError`를 내는
잠재 결함을 발견(VAL/OOS/2025q2는 100% 커버리지라 지금까지 안 드러남) — 원본 함수는 무수정,
이번 스크립트가 자체적으로 프레임을 사전 교집합해 우회. 상세: `docs/experiments/
eth_omega461_regime_threshold_h48qual_side_aware_revival_20260814.md`.

**#14 Selective Conformal Risk Control(SCRC) — 문헌스카우팅(#6) 4위 후보, #8과 다른 이유로 VAL
자체 기각(2026-08-14)** (`#13`은 동시 진행 중이던 다른 세션이
`docs/model_contracts/odyssey2_eth_live_injection_data_resources_20260813.md`에 "Odyssey2
#11(레짐인지형 exit_head 지속상승장 가드) 라이브 섀도우 배포" 항목을 먼저 "실행 로그 #13"으로
참조해둔 상태라, 편집 직전 재확인 후 번호 충돌을 피하려고 건너뜀 — 그 세션이 실제 `#13` 항목을
이 절에 채워 넣을 것으로 예상): Xu/Guo/Wei, arXiv:2512.12844 원문(abstract+HTML 전문)을 직접 fetch해
"1단계 선별(feature-only 신뢰도 점수) → 2단계 선별된 부분집합에만 conformal 위험제어" 2단계
구조를 확인하고, #2(Joshi/Wang/Hassani/Dobriban, #8의 근거 논문)를 인용하지 않는 독립 병행 연구임을
related work 절 직접 확인으로 검증(시간상으로도 SCRC가 Joshi보다 먼저 나와 인용 불가능). 이 2단계
구조를 #8과 구별되게 실제로 구현 — 1단계 선별 임계값을 기존 `EXIT_THRESHOLD`(0.95)로 고정(새
VAL-fit 자유도 추가 방지, `eth_val_oos_regime_mismatch_investigation_20260813.md`의 3중
선택편향 경고를 직접 반영), "선별됨"이 정확히 `{a0=1}`(TabM 자신이 이미 확신한 exit 신호)과
일치하도록 설계해 개입이 구조적으로 "확신 있는 exit 취소만" 가능하고 "새 조기exit 유발"은
불가능하게 제한. 2단계 위험제어 수식(Δ(x), Algorithm 1)은 #8(`research_eth_omega461_risk_
controlled_exit_fallback_20260814`)에서 무수정 재사용. G0(오케스트레이터 지정 VAL/OOS-Q1 4수치)·
G0b(2단계 무력화)·G0c(1단계 무력화, 신규 자체검증) 전부 PASS. **Calibration에서 핵심 발견**:
선별 임계값 0.95가 VAL calibration 표본을 13,330개 보유bar 중 52개(0.39%)로 극단적으로 좁혀
Algorithm 1의 eps-비율(baseline 불일치율의 배수) 설계가 **3개 사전등록 eps 전부에서
실행불가능(feasible=0/46)**해졌고, 논문 자신의 "공집합→τ̂=0" fallback이 채택한 정책이 y-규칙
불일치율을 5.66%→81.13%(14배)로 악화시켰다 — 진단 전용 대조(select_threshold=0.50, 게이트
미적용)에서는 calibration이 정상 작동(τ̂=0.9996, #8의 τ̂=0.9995와 거의 일치)해 붕괴 원인이
calibration 수식이 아니라 1단계 선별 폭 자체임을 직접 특정. **VAL: 3개 eps 후보 전부 원기준
FAIL**(컴포넌트 PnL·MDD 둘 다 악화 9.23%→8.53%/-7.59%→-10.33%) **완화기준도 FAIL**(with_gate
PnL은 개선했으나 77.31%→84.63%, MDD가 -21.76%→-25.16%로 3.0%p 슬랙을 3.40%p 초과) — `val_
winner=None`. 방법론(VAL 기각 시 OOS 미개방, #9 Conformal Kelly 전례와 동일)에 따라 **OOS
미실행**(`oos_opened=false`). **#8과 다른 실패**: #8은 VAL을 통과하고 OOS 포트폴리오 레벨에서
공유슬롯 재순환으로 반전됐지만, #14는 VAL 자체에서 calibration 표본 붕괴로 기각됐다 — 오케스트레이터가
사전에 경고한 "같은 실패 패턴 상속" 위험은 현실화되지 않았다(표면적 유사성에도 불구하고 실패
지점·메커니즘 둘 다 다름). Odyssey2 문헌 스카우팅(#6) 4위 후보는 이것으로 부정 결과 종결. 상세:
`docs/experiments/eth_omega461_selective_conformal_risk_control_20260814.md`.

**#13 레짐인지형 exit_head 지속상승장 가드(#11) 라이브 섀도우 배포 — 관찰 개시(2026-08-14)**: #11이
백테스트로만 검증하고 배포는 제안조차 안 했던 조건부 정책을 `scripts/live_eth_regime_aware_exit_
guard_shadow_20260814.py`로 실제 라이브 섀도우 배포(주문 미제출, 관찰 전용). 이미 서버에서 상시
실행 중인 베이스라인 섀도우(`live_eth_exithead_asymmetric_shadow_20260813.py`, h48qual=liveATR
재라벨 exit_head/zig075=원본) 위에, h48qual 보유 포지션의 exit_head 확률 조회만
`SustainedUptrendDetector`(dual_momentum 기반 rolling 2016bar(1주) fracpos,
threshold=0.8025793650793651 — #11 `report.json`에서 전체 정밀도 그대로 복사, 재캘리브레이션
없음)로 매 bar 원본/liveATR 사이에서 조건부 전환한다. **라이브 개입 지점**:
`Omega461LiveAdapter.evaluate_exit`가 TP/SL 통과 후 `self.components[source_component].
exit_probability(...)`를 호출하는 지점 하나 — 신규 `evaluate_exit_guarded`가 이 지점만 가로채
h48qual+탐지기활성 조합일 때만 별도 `_Component`(원본 h48qual 번들/사이드카 =
`trading_bot.py` 실제 라이브 기본값 `FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_{BUNDLE,SIDECAR}_PATH`,
신규 아티팩트 없이 그대로 재사용)의 `exit_probability`를 대신 호출한다. TP/SL·`EXIT_THRESHOLD`
(0.95)·entry/사이징·zig075는 전부 `adapter.evaluate_exit`/`adapter.decide_entry` 원본 경로
그대로(가드 상태와 무관, 이 지점만 조건부).

탐지기는 라이브 제약(bar 하나씩만 도착)에 맞춰 O(1) 증분(`SustainedUptrendDetector`, deque
기반)으로 재구현했다. **배포 전 자체검증**(`scripts/verify_eth_regime_aware_exit_guard_shadow_
detector_20260814.py`, dev에서 실행): #11의 배치 함수 `_rolling_dual_momentum_score`(원본 그대로
재사용, 재구현 없음)와 증분 계산을 2025 전체연도(105,101bar)·2026 재구축본(57,601bar) 두 CSV
전부에서 (1) 처음부터 causal 전체 재생, (2) 4개 임의 지점에서 seed()+update() 재개 재생(라이브
콜드스타트/재시작과 동일 코드경로)으로 대조 — **NaN정렬·점수·활성화플래그 전부 mismatch=0으로
완전 일치**(결과: `tmp/causal_regen_20260516/eth_omega461_regime_aware_exit_guard_shadow_20260814/
detector_verification_report.json`). 이 자체검증 과정에서 이 스크립트 docstring이 인용한 "2025
Q3 -9.73%→-46.26%" 예시가 `with_gate` 원본과 `no_gate` 재라벨을 섞은 취약성 문서 헤드라인
테이블의 부정확한 인용이었음을 발견해 `report.json`의 `no_gate` 수치(PnL -35.54%→-46.26%,
MDD -49.79%→-56.94%, 거래수 25→38)로 직접 수정했다.

**배포**: 신규 모델 아티팩트 없음(원본 h48qual 번들/사이드카는 실제 라이브 기본 경로, liveATR
번들/시밍 사이드카는 #10 배포 때 이미 서버에 존재 — `handoff.sh push`로 스크립트 파일 1개만
동기화). 9분 예정 타임박스 스모크테스트(`--end-at-kst`, poll-seconds=20)로 먼저 검증 — 실제
신규 bar 처리(`04:05→04:10` 전진)를 `state.json` 직접 확인으로 검증(`[error]` 0건,
`detector_bars_seen=2`, `detector_active=false`/`score=0.0` — 현재 시장이 지속상승장이 아니므로
비활성은 정상이며, 이 경우 베이스라인 섀도우와 완전 동일 동작으로 축소됨을 뜻한다). 정상 확인 후
`handoff.sh stop`으로 스모크테스트 종료 → `handoff.sh launch server
eth_regime_aware_exit_guard_shadow -- python -u scripts/live_eth_regime_aware_exit_guard_shadow_
20260814.py --poll-seconds 90`로 상시 실행(nohup) 전환 — 동일 `state.json`을 이어받아 정상
재개(`last_processed_bar_ts=2026-08-14T04:10:00`에서 재개). 전환 직후 최소 10분간 관찰(13:10:28
KST 시작 → 13:20:28 KST 최신 `last_decision_at_kst` 직접 확인, `last_processed_bar_ts`가
`04:10:00`→`04:15:00`으로 실시간 신규 bar를 계속 정상 처리, pid=274710 RUNNING 유지, `[error]`
0건). 참고로 스모크테스트→상시실행 전환 순간 수동 `stop` 타이밍이 04:15 bar의 diagnostic-only
`equity_curve.jsonl` append(상태파일 쓰기 전에 먼저 실행되는, 기존 두 섀도우와 완전 동일한 기존
쓰기 순서)와 겹쳐 그 한 줄만 중복 기록됐다 — `state.json`(재개 판단의 유일한 근거)은 항상 유효한
단일 값만 가졌고 거래/포지션 로직에 영향 없는 순수 진단 로그 중복이라 무해함을 확인, 코드 수정
안 함(기존 두 섀도우와 동일한 기존 패턴이라 이번 작업 범위 밖). `order_submission_supported=
false`(주문 API 호출 경로 자체 없음), 라이브 4개 파일(`trading_bot.py`/`omega4_6_1_live.py`/
`runtime_config.py`/`.env`) 코드 변경 0줄(`git diff` 직접 확인, `.env`는 gitignore 대상이라
mtime으로 세션 내 미접촉 별도 확인). 상세: `docs/experiments/eth_omega461_regime_aware_exit_head_
uptrend_guard_20260814.md`(#11 백테스트), `docs/model_contracts/odyssey2_eth_live_injection_data_
resources_20260813.md`(신규 파일 행).

**#15 zig075 exit_threshold 재보정 — 우선순위 큐 6번 항목, 재학습 없이 VAL 전체 그리드에서
로버스트한 진짜 개선 0건으로 부정 결과 종결(2026-08-14)**: 계약서 51~57행 우선순위 큐 6번
("zig075 exit_head 개선 — 같은 live-ATR relabel 레시피는 이미 악화로 닫힘, 별도
exit_threshold 등 미탐색")을 실행. 오늘 밤 #4/#5/#7/#8/#11/#13/#14 전부 `EXIT_THRESHOLD=0.95`를
h48qual·zig075 양쪽에 동일하게 고정한 채 **h48qual의 exit_head 모델**만 바꿔봤을 뿐, "zig075는
모델을 안 건드리고 그 threshold 숫자 자체만" 바꾸는 축은 이번이 처음이자 재학습이 전혀
필요 없는 축이었다(`replay_omega4_6_1_greedy_router_20260706.greedy_replay`가 이미 컴포넌트별
`exit_threshold`를 설정 딕셔너리에서 읽으므로, 개입 전체가 설정값 오버라이드 하나로 끝남 —
GBDT/TCN/대기압력/risk-controlled와 달리 이름 바꾼 복사본조차 불필요). 가설은 **양방향 다
열어뒀다**: #10이 찾은 "threshold를 낮추면 회전이 빨라져 h48qual처럼 무스킬 컴포넌트엔
유리하다"는 메커니즘이, zig075(이 프로젝트에서 유일하게 검증된 방향별 엣지 보유 컴포넌트)에는
정반대로(threshold를 높여 회전을 늦추는 쪽이 좋은 트레이드를 더 들고 가 유리) 작용할 수도
있다는 경쟁 가설을 사전에 명시하고, 대칭 그리드({0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98,
0.99})로 방향을 미리 정하지 않고 스윕했다. h48qual은 오늘 밤 확정 baseline(`asymmetric_tabm_
liveatr`, liveATR exit_head, `exit_threshold=0.95`)에 완전 고정, zig075의 direction_head/
quality_head/encoder/exit_head **가중치**도 전부 동결(원본 그대로) — exit_threshold 숫자만
스윕했다. G0(포트폴리오 레벨, VAL·OOS-Q1 4수치, `eth_omega461_multiwindow_confirmation_gate_
20260814.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR` 대조) **PASS**, 부가로 zig075 컴포넌트
단독 et=0.95 수치가 2026-07-21에 이미 실행돼 저장된 독립 파일(`tmp/research_20260721/
exit_threshold_sweep_VAL.csv`, zig075는 이 프로젝트에서 한 번도 손댄 적 없어 그 시점 수치가
여전히 유효)과 정확히 일치함도 확인. **핵심 발견 1**: zig075의 exit_head는 이 VAL 구간에서
보유 bar 확신이 사실상 0.90을 넘지 않아, exit_threshold∈[0.90, 0.999] 전 구간이 컴포넌트·
포트폴리오 양쪽에서 baseline과 바이트 단위로 동일한 디제너레이트 동률이다 — 가설의 "회전 둔화"
방향은 이 모델 캘리브레이션상 구조적으로 시험 불가능했다(효과 없음이 아니라 시험할 여지 자체가
없었음). **핵심 발견 2**: 실질적 개선을 보인 유일한 지점(0.80, 컴포넌트 PnL+40.31%→+53.69%,
포트폴리오 no_gate+46.59%→+56.25%, MDD 불변)이 있었지만, 바로 옆 그리드점(0.85)이 컴포넌트
단독 경제성 악화(+40.31%→+36.91%, 포트폴리오 자체는 baseline과 동률이라 그 악화가 실제
공유슬롯에선 발현 안 됨)로 원기준에서 탈락 — 오늘 밤 #12가 확립한 "그리드 이웃도 통과해야
로버스트" 원칙에 따라 **0.80을 고립 스파이크로 명시 기각**(#12의 LONG=0.35 기각과 동일 원칙).
로버스트(이웃 통과) 지점 {0.92, 0.95, 0.97, 0.98, 0.99}은 전부 baseline과의 디제너레이트
동률이라 "후보"에서 별도 제외(동률은 개선이 아니며, 단일터치 OOS 기회를 이런 무변화 설정에
쓰는 것은 낭비라는 판단 — 이번 실험에서 새로 도입한 판정 기준). 결과: **`val_winner=None`,
로버스트하며 동시에 baseline과 실질적으로 다른 후보가 그리드 전체에 0개** — 과제의 "VAL 게이트
통과 후보가 있으면 OOS 단일터치" 조건이 성립하지 않아 **OOS는 열지 않았다**(`oos_opened=false`,
#9/#14와 동일한 "VAL 기각 시 OOS 미개방" 패턴). 우선순위 큐 6번 항목은 이것으로 부정 결과
종결한다 — zig075의 exit_threshold를 단독으로(모델 재학습 없이) 재보정하는 것만으로는 이 VAL
구간에서 유의미한 개선을 찾지 못했다. 라이브 4개 파일 코드 변경 0줄(`git diff` 직접 확인).
상세: `docs/experiments/eth_omega461_zig075_exit_threshold_recalibration_20260814.md`.

**#16 Gittins Index Deep RL(QGI/DGN) 기반 exit_head 재정식화 — 문헌스카우팅(#6) 5위(최종) 후보,
VAL 게이트 자체 실패로 OOS 미실행(2026-08-14)**: Dhankhar/Mishra/Bodas(arXiv:2405.01157, retirement
formulation 기반 QGI/DGN) 원문을 WebFetch로 직접 읽고 정확한 메커니즘(Eq 9-11의 TD 부트스트랩
타겟, `M(x)=Q_θ'(x,x)` 자기참조 은퇴가치, DGN의 (s,x) 쌍 입력 구조)을 확인한 뒤 구현 — 오늘 밤
유일하게 "기존 threshold/모델 재조합"이 아니라 **진짜 새 학습**을 시도한 후보. h48qual exit_head를
"청산확률 분류"가 아니라 "이 포지션을 계속 들고 있는 것의 은퇴가치(continuation/retirement
value) 회귀"로 재정의 — GBDT(#4)/TCN(#5)와 동일한 라이브ATR 데이터셋(`train_eval_omega461_gbdt_
exit_head_liveatr_20260813._build_dataset`, seed=260813, max_candidates=1500)을 무수정 재사용해
`dataset_reference_check` 3개 지표 전부 일치를 재확인했다. candidate별 순차 bar 구조
(`exit_path_hold_bars`/`exit_path_entry_i`)에서 진짜 `(s_t,r_t,s_{t+1})` TD 전이를 복원(보상 =
`exit_path_unrealized`의 bar 증분), 참조상태 x=미니배치 자기 자신(대각 근사), 타겟망 대각선 읽기로
M(x) 구현(Eq-11의 별도 3번째 네트워크를 DQN 타겟망과 병합) — 논문 자체가 다루는 배치 도착·자유
전환 스케줄링 문제와 이 프로젝트(독립 트리거 신호, 전환비용 존재, "은퇴"와 "실제 청산"이 하나로
합쳐짐, 경쟁 arm 간 index 비교 미구현)의 **비동형성을 명시적으로 문서화**했다. 서버 GPU 사용
여부를 실측 기반으로 판단(B×B=65,536쌍/스텝 구조상 CPU 외삽 시 수 시간, 실제 GPU 25초/전문가로
확인 — RTX 3070Ti, `torch.cuda.is_available()=True`를 직접 재확인해 사용). 서버 메모리 안전장치
(`run_jm_full_retrain_seed_robustness_20260813.sh` 패턴 재사용, 8회 `free -h` 체크, 4GB 미만
중단)를 코드화했고 안전 중단 0회, 학습 총 wall time ~8.9분(데이터셋 빌드 7.5분 + 학습 3개 전문가
합산 75초)이었다. G0(컴포넌트 2종 + 포트폴리오 VAL/OOS-Q1 no_gate/with_gate 4종) **PASS**, 6개
지표 전부 오케스트레이터 지정값과 정확히 일치. **VAL: never-trigger 진단 패스로 M(x) 분포를 먼저
수집(우측 치우침, 최솟값≈0, p10=0.026/p25=0.035/p50=0.046)해 사후선택 없이 4점 그리드
`{0.0, p10, p25, p50}`를 구성** — 결과 4개 그리드점 전부 원기준·완화기준·컴포넌트 가드레일 중
하나 이상을 결정적으로 실패(`val_winner=None`). threshold=0.0은 exit_head가 사실상 발동하지 않는
퇴화모드(이미 문서화된 "원본 exit_head가 이 VAL 구간에서 거의 발동 안 함"과 동일 현상 재현, 포트폴리오
수치가 `baseline_both_original`과 정확히 일치)이고, threshold를 p10만큼만 올려도 컴포넌트 exit_head
발동이 2건→70건으로 급증(급격한 상전이, 완만한 중간지대 없음)하며 GBDT/TCN과 동일한 "포트폴리오
개선처럼 보이지만 컴포넌트 PnL 부호반전" 패턴이 재현됐다(가드레일 결정적 실패). 방법론(#9/#14/#15와
동일 원칙, VAL 기각 시 OOS 미개방)에 따라 **OOS 미실행**(`oos_opened=false`). **핵심 진단 가설**:
이 구현이 h48qual의 M(x)를 zig075 현재상태와 비교하지 않고 고정 상수와만 비교해, 논문이 실제로
보장하는 최적성(경쟁 arm 간 index 비교)을 상속받지 못하고 "값 기반으로 재파라미터화된 exit_head
분류기"로 축소됐을 가능성 — 완전히 다른 학습 방법론(TD 회귀 vs 분류/시퀀스 분류)에서도 같은
실패 계열(컴포넌트-vs-포트폴리오 괴리)이 재현된 것은 이 프로젝트의 근본 제약이 모델 종류가 아니라
정책의 형태(전역 상수 threshold 단일 결정)에 있다는 가설을 강화한다. 라이브 4개 파일 코드 변경
0줄(dev·서버 양쪽 `git diff` 직접 확인), 서버 섀도우봇 3개(`eth_exithead_asymmetric_shadow`,
`eth_regime_aware_exit_guard_shadow`, `eth-jmlam4-shadow.service`) 학습·평가 전후 전부 RUNNING
무사 확인. 문헌 스카우팅(#6) 큐 1~5위 전부 종결. 상세: `docs/experiments/
eth_omega461_gittins_index_exit_head_20260814.md`.

**#17 증거 신호 주입 전략 리서치 (2026-08-14, 설계 리서치만 — 구현·학습·OOS 터치 없음)**: 두 개의
독립 구간에서 순위가 재현된 외부 증거 신호 계열(OHLCV+`taker_buy_base` 기반: orthogonal_combo·
liquidity_sweep·volume_wick_climax·short_term_return_z·taker_climax, lift 2.75~4.14)을 이 라이브
모델에 주입하는 전략을 Model Architect 페르소나 dispatch + 리드 세션 검증으로 설계. 핵심 판단:
(1) #1~#16은 전부 내부 파생 신호였고 **외부 증거 신호 계열은 이 모델에 미시도 신규 정보원**,
(2) 진입측 주입은 확정된 direction-스킬 부재 위에서 금지(안티골), (3) 최우선 후보는 무학습 진단
2건(섀도우 관찰 로깅, 냉동예측 순위상관 — 6개 창 중 4개 부호일치 킬 기준) 후 **숏 포지션 반대증거
exit 오버레이**(#11 dual_momentum 가드와 같은 Q3-2025 회전가속 취약점을 다른 신호원·bar 입도로
겨냥하는 보완 후보 — #11 대비 증분 가치로 평가, 재발명 금지), 차순위는 사이징 사이드카 GBM 피쳐
(#3 latent 전례의 소표본 과적합 유의). TP/SL 폭·멀티슬롯·펀딩 계열은 재제안 금지 재확인. 첫 실전
실험 선택(C vs D)·섀도우 통합 여부·4+창 규율 선충족 여부 3건은 사용자 결정 대기. 상세:
`docs/experiments/eth_omega461_evidence_signal_injection_research_20260814.md`.

**#18 Candidate C 실전 구현·검증 — VAL에서 결정적 기각 (2026-08-14, 사용자 지시 "C로 우선
진행")**: h48qual 숏 보유 중 `orthogonal_combo`(가장 강하고 희귀한 증거 신호, 새 임계값 탐색
없이 기존 검증된 정의 그대로 재사용) 발화 시 즉시 강제청산하는 오버레이를 #11의 검증된 템플릿
그대로 구현(`scripts/research_eth_omega461_evidence_veto_exit_overlay_20260814.py`). G0a/G0b
항등성 검증 통과 후 6개 창 실행. **결과: VAL with_gate PnL 77.31%→47.39%(-29.92%p, 게이트
모듈 자체의 row-level 체크가 `pnl_nonworse=False` 직접 반환), Q1/Q2도 각각 -18.83%p/-13.28%p
악화** — 발화는 극히 희귀(VAL 26,209bar 중 6회)했지만 발화당 손상이 매우 커서, precision
43.9%인 신호로 하드 강제청산을 정당화하기엔 비용이 크다는 결론. 목표였던 **Q3는 오히려 원본
초과 회복(no_gate 145%, with_gate 130%)** — 메커니즘 가설 자체는 지지되나, 숏이 실제로 잘
먹히는 VAL/Q1/Q2에서의 손상이 훨씬 커서 순효과 마이너스. OOS-Q1/Q2는 통과했으나 VAL 패배로
승격 근거 안 됨(OOS 통과를 확인으로 쓰지 않음). **절차 결함 자기기록**: 이 스크립트는 VAL을
먼저 판정해 통과할 때만 OOS를 여는 호출자측 게이트를 구현하지 않고 6개 창을 한 번에 실행함 —
이번엔 VAL이 명백히 패배해 결론에 영향 없지만, 앞으로 이 클래스 후보는 VAL 하드게이트를
코드화할 것. 다음 후보안(미착수, 제안만): 즉시청산 대신 N-bar 임계값 완화하는 소프트 변형.
상세: `docs/experiments/eth_omega461_evidence_veto_exit_overlay_20260814.md`.

**#19 exit_head 재학습 피쳐 주입 — 0단계 진단, 재학습 보류 권고 (2026-08-14, 사용자가 "그냥
학습 피쳐로 넣는건 어때" → exit_head 재학습 선택)**: 서버 GPU 다중시드 재학습을 커밋하기 전
재학습 없이 신호-결과 관계부터 확인. **1차 시도(트레이드 레벨: 보유 중 어디선가 발화했는지 vs
최종 수익률)는 모든 창에서 강한 양의 상관을 보였으나(VAL rho=+0.675, p=0.016) 방향이
틀렸음을 직접 잡아냄** — 오래 버틴/이기는 트레이드일수록 bar 수가 많아 희귀 이벤트가 어딘가서
나타날 기회가 느는 지속시간 교란요인이 원인, 폐기. **2차 시도(bar 레벨: 발화 bar 시점부터
forward 12bar 가격이동과 상관, 지속시간 교란 제거)로 재검증**: 5개 창 중 4개가 옳은 방향(발화
= 숏에 불리한 forward 이동 예측)이나 효과크기가 작고(rho -0.02~-0.03), **가장 중요한 선별창인
VAL에서 통계적으로 유의미하게 반대 방향**(rho=+0.056, p=0.0006) — 이게 Candidate C가 VAL에서
실패한 정확한 메커니즘(발화가 숏에 유리한 순간에 일어남)과 일치한다. 합산 상관은 방향은 맞지만
p=0.057로 관례적 유의수준 미달. **권고: 재학습 보류** — GBDT/TCN/Gittins가 반복 걸린
컴포넌트-vs-포트폴리오 가드레일 위험과, VAL에서 이미 반대 방향이 나온다는 사실을 감안하면 이
크기의 상관으로 다중시드 서버 재학습을 정당화하기엔 근거가 약함. 대안 3개 제시(사이징 사이드카로
전환/연속값 피쳐로 재확인/VAL만 다른 이유 규명), 사용자 결정 대기. 상세: `docs/experiments/
eth_omega461_evidence_signal_exit_head_feature_rank_correlation_20260814.md`.

## 미해결 이슈

Odyssey(1)에서 상속(전부 유효):

- VAL 구간(2025-10~12) 자체의 신뢰성 문제 — Odyssey(1) 미해결 이슈 12, 서브프로젝트 최상위 이슈.
- exit_head 섀도우 관찰기간·승격 판단기준 미정 — Odyssey(1) 미해결 이슈 13.
- `quality_threshold` 정렬버그, 동일 코드가 있는 미수정 6개 스크립트 — Odyssey(1) 미해결 이슈 14.
- ATR TP/SL floor가 버그인지 의도인지 — Odyssey(1) 미해결 이슈 15, 재보정 자체는 부정됐으나 "의도였다"는 적극적 증거는 아님.

## 승격 게이트

Odyssey(1)과 동일하게 적용:

- VAL 단독 승리는 승격 근거 아님 — 저비용 사전필터로만.
- 최소 4개 이상 부호가 섞인 독립 구간에서 일치 확인 전엔 "확인됨"이라고 쓰지 않는다. **(2026-08-14
  갱신, 방법론 변경 — 위 실행 로그 참고, 과거 판정 재심 아님)**: 이 원칙은 이제
  `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`로 코드화됐다. **앞으로 신규
  post-entry 후보는 VAL 자체 게이트 통과 후, 공식 OOS 확인을 OOS-Q1+OOS-Q2를 한 실행에서 함께 여는
  단일터치로 심사한다**(순차/반복 확인 금지) — 둘 다 with_gate PnL이 baseline 대비 비악화(또는
  완화기준의 MDD 3%p 허용)여야 통과, 하나만 통과하면 부호불일치로 기각. 2025 Q1~Q3는 참고용 맥락으로
  표에 표시하되 pass/fail 기준에는 넣지 않는다. 설계·G0 자체검증·기존 2개 후보(대기압력·
  risk-controlled) 소급적용 결과: `docs/experiments/eth_omega461_multiwindow_confirmation_gate_20260814.md`.
  대기압력(#7)·risk-controlled(#8)의 기존 기각 판정은 그대로 유효하다 — 이 모듈 도입으로 재심되지
  않았음을 직접 확인했다(위 실행 로그의 "방법론 변경" 항목).
- 재학습 모델은 N≥5개 진짜 다양한 시드(무작위 추출) 없이 신호/노이즈 판정하지 않는다(레포 Seed-Diversity Ensemble Promotion Gate).
- 라이브 파일 무변경, 섀도우 배포 ≠ 승격.
