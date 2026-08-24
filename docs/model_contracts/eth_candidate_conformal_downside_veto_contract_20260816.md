# ETH 진입 conformal 하방-LCB 거부게이트 후보 — 데이터 계약 (2026-08-16)

이 문서는 **공식 Odyssey 계보(Odyssey1~4)에 속하지 않는다** — 버전 번호는 확정된 성과가 있을 때만 올린다는 원칙에 따라, 이 후보는 확정된 성과 없이 축 종결 여부를 판단 중이므로 "Odyssey6"로 명명하지 않는다(2026-08-16, 사용자 결정). 조사·결과 자체는 향후 참고를 위해 그대로 보존한다.

## 상태

| 컴포넌트 | 상태 |
|---|---|
| **ETH conformal veto 후보** | **`HGB 학습 완료(2026-08-16) — zig075 방향 거꾸로(5/5 시드), h48qual 미미(R²≈1%). 축 종결 여부 사용자 판단 대기`**. cheap_gate(애매)→episode 라벨 인접상관(심각)→uniqueness 가중치 정밀재계산(더 심각, zig075 유효표본 학습풀 전체 ~114건)→HGB 회귀 실제 학습(zig075는 VAL 상관관계가 5개 부트스트랩 시드 전부 음수 = 예측 방향이 거꾸로, h48qual은 5/5 양수지만 R²≈1%로 미미)까지 4단계 전부 "이 축이 약하다" 방향으로 쌓였다. 아래 "HGB 학습 결과" 절 참고. |

## 범위

- 모델 id: `eth_candidate_conformal_downside_veto_20260816`
- 목적: Odyssey4까지의 스택은 L3 진입 게이트가 정적 `quality_threshold` 하나뿐이고, 그 확률 예측에
  대한 캘리브레이션된 불확실성이 전혀 없다(`docs/experiments/
  eth_odyssey_internal_architecture_zoo_cross_pollination_survey_20260815.md` Tier 1-2). 이
  계약은 BTC clean_base 계열이 쓰는 "validation-calibrated 하방 신뢰구간(LCB)" 기법을 Odyssey의
  진입 결정에 이식해, quality gate는 통과했지만 예측 하방이 나쁜 진입을 추가로 거부하는 계층을
  설계한다.
- 아키텍처 유형: 지도학습 회귀 2개(HGB, `full`/`adverse` 예측) + validation 잔차 분위수로 계산한
  LCB 임계값. 결정론적 임계값 비교가 아니라 학습 모델이므로, 이 저장소의 N≥5 시드-다양성 게이트가
  **적용 대상**이다(ETH 드로다운 거버너의 결정론적 룰과 다름 — 주의).
- Owner agent: Model Architect(단독, Sonnet).
- 리소스 레지스트리: `docs/model_contracts/eth_candidate_conformal_downside_veto_data_resources_20260816.md`
- 관련 문서: `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`(G0 기준선),
  `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`(L0~L10),
  `docs/experiments/eth_odyssey_internal_architecture_zoo_cross_pollination_survey_20260815.md`(후보
  선정 근거), `docs/model_contracts/eth_candidate_drawdown_budget_governor_contract_20260815.md`(직전
  종결된 후보 — 재진입 처닝·경로효과 교훈 재확인 필요).

## 이식 원본 재조사 — 초기 서베이 요약을 정정한다

원 서베이(`eth_odyssey_internal_architecture_zoo_cross_pollination_survey_20260815.md` Tier 1-2)는
"BTC v1.5 conformal veto sleeve가 L3 진입 게이트에 캘리브레이션된 불확실성을 추가하는 구조"라고
요약했다. 실제 스크립트(`scripts/train_eval_clean_base_causal_sleeve_conformal_veto_v1_5.py`)를
읽어보면 이는 부정확하다:

- v1.5는 **CORE 진입 자체를 거부하지 않는다.** 이미 확정된 core 트레이드 위에 추가로 얹는
  `ADD_SAME_SIDE_15`/`ADD_SAME_SIDE_25`("sleeve") 노출만 거부한다(`CONFORMAL_VETO`). Core
  side/notional/leverage/exit은 원본 계약("Output Invariants")이 명시하듯 절대 불변이다.
- "sleeve"란 이미 열려 있는 core 포지션과 같은 방향으로 **추가 노출을 얹는 피라미딩** 메커니즘이다
  (`train_eval_clean_base_plus_causal_conviction_sleeve_v1_1.py`). **Odyssey에는 이 개념이 아예
  없다** — h48qual/zig075는 진입 시점에 사이징 사이드카가 정한 notional/leverage로 단일
  포지션을 열고, 보유 중 추가 노출을 얹는 메커니즘이 없다(단일 공유 슬롯, 포지션당 1건).
  따라서 v1.5를 "그대로" 포팅할 대상 자체가 Odyssey에 없다.
- v1.5가 실제로 재사용하는 핵심 자산은 **conformal 모델 자체**다 — `scripts/
  train_eval_clean_base_causal_trade_editor_v1_3.py`(v1.3, "Causal Trade Editor")의
  `_train_editor_model`/`_predict_editor`를 그대로 import해서 쓴다. 그 모델은 진입 시점의 정적
  피처(`EDITOR_FEATURES` = sleeve 피처 중 동적 상태 제외)만으로 HistGradientBoostingRegressor
  5개를 학습한다: `full`(전체 보유기간 net return), `h6`/`h12`/`h24`(조기청산 시 net return),
  `adverse`(그 트레이드의 최대 역행폭, MAE 크기). v1.5의 "conformal" 층은 이 위에 딱 두 가지를
  더한다: (1) validation 잔차의 분위수 `residual_q = quantile(|actual_full - pred_full|, q)`를
  계산해 `pred_lcb = pred_full - residual_q`(대칭 잔차 기반 — 진짜 비대칭 conformal 예측구간은
  아니고 "conformal 스타일"의 단순화된 버전임을 정직하게 밝혀둔다), (2) `pred_lcb <=
  lcb_veto_threshold OR pred_adverse >= adverse_veto_cut`이면 그 액션을 `CONFORMAL_VETO`로
  강등.
- **번역**: Odyssey는 sleeve가 없으므로, 이 계약은 같은 "학습된 하방 예측 + validation 잔차
  LCB + 임계값 veto" 기법을 **진입 자체**(sleeve-add가 아니라 core entry)에 적용하도록
  재해석한다. `clean_base_conformal_downside_filter_v1_4`(sleeve 없이 core notional
  shrink+조기청산만 하는 자매 버전, `docs/model_contracts/
  clean_base_conformal_downside_filter_v1_4_contract.md`)가 이 방향에 더 가깝지만, 그것도
  "거부"가 아니라 "축소"다. 사용자가 명시적으로 "veto 게이트"를 선택했으므로, 이 계약은 v1.5의
  veto 프레이밍(액션을 완전히 무효화하는 옵션)을 유지하되 그 대상을 sleeve-add에서 Odyssey의
  L3 진입 결정 자체로 옮긴다. v1.4식 축소(shrink+조기청산)는 미래 후보로 남겨둔다(미해결 이슈
  참고).

## 새로운 핵심 제약: 학습 표본 크기 — 구현 착수 전 반드시 확인

BTC core 트레이드는 하루 6.15건(연 환산 수천 건)이라 회귀모델 학습·잔차 캘리브레이션에 표본이
넉넉하다. Odyssey4 G0의 **실현 트레이드는 창당 10~35건뿐**이다(VAL 26건, OOS-Q1 19건, OOS-Q2
10건) — 이걸로 HGB 2개 학습 + 분위수 캘리브레이션을 하는 건 통계적으로 무의미한 수준이다.

**해법(이 계약이 요구하는 설계, 구현 전 검증 완료)**: 실현 트레이드(우선순위 중재에서 슬롯을
얻은 것)가 아니라 **quality gate를 통과한 모든 신호 episode**(h48qual·zig075 각각, 슬롯을
얻었는지 여부 무관 — 슬롯을 놓친 신호도 "만약 그때 진입했다면"을 causal하게 재구성 가능)를
학습 모집단으로 쓴다. 2026-08-16 직접 측정(코드로 확인, 6개 창):

| 창 | h48qual episode 수 | zig075 episode 수 | 비고 |
|---|---:|---:|---|
| VAL | 254 | 789 | signal_bars 468/1593, 중앙값 episode 길이 1 bar |
| OOS-Q1 | 61 | 573 | signal_bars 102/1268 |
| OOS-Q2 | 93 | 750 | signal_bars 160/1683 |

episode = `side!=0 & active`(quality gate 통과)인 연속 bar 구간 하나. 중앙값 길이가 1 bar라서
"긴 신호 하나를 여러 표본으로 중복 계산"하는 위험이 크지 않지만(최대 길이는 h48qual 16,
zig075 31), **episode 시작 bar만 표본으로 쓰고 같은 episode 내 나머지 bar는 버린다**(uniqueness
원칙, [[feedback_zigzag_segment_id_year_collision]]과 같은 종류의 실수를 피하기 위해). VAL
기준 h48qual 254 + zig075 789 = 1043개 episode-시작 표본은 원본 26건보다 40배 많고, HGB
회귀 + 분위수 캘리브레이션에 통계적으로 타당한 규모다. **이 표본 크기 확보가 이 축의 실현
가능성 자체를 좌우했다 — 실현 트레이드만 썼다면 이 후보는 착수 전에 기각됐을 것이다.**

## Odyssey 삽입 지점 — L3.5 신규 계층

```text
L3 (기존)   : action=argmax(direction), quality[action]>=threshold → 통과
L3.5 (신규) : 통과한 (component, side, entry_bar)에 대해
              pred_full, pred_adverse = conformal_model.predict(entry_time_features)
              pred_lcb = pred_full - residual_q(calibration_quantile)
              if pred_lcb <= lcb_veto_threshold or pred_adverse >= adverse_veto_cut:
                  진입 스킵 (CASH로 강등, 슬롯은 우선순위상 다음 후보에게)
              else:
                  L4(기존 Odyssey4 진입거부)로 그대로 진행
```

- Odyssey4(L4, zig075 SHORT 지속상승장 진입거부)와 우선순위: **L3.5를 L4보다 먼저 평가한다** —
  둘 다 veto-only(스킵만 가능, side/notional 불변)라 순서가 최종 결과에 영향을 주지 않는다(둘
  다 스킵되면 스킵, 어느 하나만 스킵돼도 스킵). 다만 진단 편의상 L3.5를 먼저 적용해 "conformal이
  거부한 것" vs "Odyssey4가 거부한 것"을 분리 기록한다.
- L5(우선순위 중재)·L6(TP/SL)·L7(사이징)·L9(exit)는 전부 무변경.
- **Output Invariant(BTC 원본과 동일 정신)**: side/notional/leverage/TP/SL은 conformal 층이
  절대 건드리지 않는다 — 오직 진입 여부(스킵)만 결정한다. 트레이드를 만들어내거나, 방향을
  뒤집거나, 사이즈를 조정하지 않는다(사이즈 조정은 v1.4식 후속 후보로 남김, 미해결 이슈).

## 상태/피처 계약

- **입력 피처**: Odyssey의 기존 causal 피처(102 base, L0에서 이미 로드됨) 중 **진입 시점에만
  알 수 있는 정적 피처**만 사용 — 보유 중 상태(hold bars, MFE/MAE 등 ETH 드로다운 거버너가 썼던 동적
  피처)는 제외한다(BTC의 `DYNAMIC_FEATURES = {account_dd, daily_dd, loss_streak}` 제외 관례와
  동일 정신). 정확한 서브셋은 구현 시 h48qual/zig075 TabM이 이미 소비하는 102 피처에서 시작해
  feature_analysis로 좁힌다(신규 피처 발명 금지 — 기존 피처만 재사용).
- **라벨(회귀 타깃)**: 각 episode-시작 bar에 대해 "그 컴포넌트/방향/TP/SL/exit_head 규칙으로
  독립적으로 진입했다면" 벌어졌을 net return(`full`)과 최대 역행폭(`adverse`)을 causal하게
  재구성한다 — 슬롯 경쟁·다른 컴포넌트 상태와 무관하게 그 신호 하나만 단독 시뮬레이션(BTC
  `_future_path_stats`와 동형: entry_idx/side/exit 규칙만 있으면 계산 가능, 포트폴리오 전체
  replay가 필요 없음). exit 규칙은 그 컴포넌트의 실제 TP/SL + exit_head 임계값(h48qual은
  레짐가드 전환 포함)을 그대로 재사용 — **저장된 렛저를 라벨로 쓰지 않는다**(CLAUDE.md
  Fresh-Forward 규정 — 이건 매 episode마다 새로 causal 시뮬레이션해서 만드는 라벨이지 과거
  렛저 재사용이 아니다).
- **학습/캘리브레이션/평가 분할**: 2025 Q1~Q3(참고 티어, VAL/OOS 미참조) = 회귀모델 학습.
  VAL = 잔차 분위수 캘리브레이션 + `lcb_veto_threshold`/`adverse_veto_cut`/`residual_quantile`
  그리드 선택. OOS-Q1+OOS-Q2 = 단일터치 확인(Odyssey4와 동일 프로토콜,
  `gate.summarize_multiwindow`).
- **미래 데이터 금지**: 라벨 생성 시 그 episode 자신의 미래 가격 경로만 쓴다(다른 트레이드의
  라벨이나 미래 episode 정보를 피처로 새지 않도록 주의 — 특히 zig075의 787개 episode처럼 밀도가
  높으면 인접 episode끼리 겹치는 시간대의 라벨이 우연히 상관될 수 있다, 미해결 이슈 참고).

## Layer Contract

| Layer | Input | Train label | Output | Artifact |
|---|---|---|---|---|
| L3.5 conformal 모델 학습 | 102 base 피처 중 정적 서브셋, 2025 Q1~Q3 episode-시작 표본 | `full`(net return), `adverse`(MAE) — causal 단독 시뮬레이션 | HGB 회귀 2개 | 신규 스크립트(리소스 레지스트리에 등록) |
| L3.5 캘리브레이션 | VAL episode-시작 표본의 `|actual_full - pred_full|` | - | `residual_q(quantile)` | 동일 |
| L3.5 게이트 | 위 예측 + 잔차분위수 | - | `pred_lcb`, veto 여부(bool) | 동일 |

## Cost/Risk Assumptions

- 이 계층은 notional/leverage/TP/SL 공식에 관여하지 않는다 — CLAUDE.md Futures Risk Sizing
  Contract는 변경 없음.
- Fee/slip: Odyssey4와 동일하게 `omega._load_fee_slip()` 재사용, cost 1x/2x/3x 리포트.
- 회귀모델은 HistGradientBoostingRegressor(BTC 원본과 동일 클래스, `sklearn`) — 신규 아키텍처
  발명 없음.

## Output Contract

렛저 추가 컬럼:

```text
conformal_pred_full
conformal_pred_adverse
conformal_residual_q
conformal_lcb
conformal_veto            # bool
veto_source                # "" | "conformal" | "odyssey4_uptrend_guard" | "both"
```

## Red Team Gates

- [ ] **N≥5 시드-다양성 게이트 적용**(ETH 드로다운 거버너와 달리 이건 학습 모델이다) — HGB 자체는
  결정론적이지만 랜덤시드(`random_state`)에 따라 트리 구조가 달라질 수 있으므로, "이 결과가
  진짜 신호인지 시드 노이즈인지"를 N≥5 무작위시드 재현으로 반드시 확인한다.
  [[tabm_hp_low_signal_pattern]] 참고 — 단일 실행 "승리"를 믿지 않는다.
- [ ] Episode-시작 표본 uniqueness 확인: 같은 episode 내 중복 bar가 학습/캘리브레이션 표본에
  섞여 들어가지 않았는지.
- [ ] 라벨 인접-episode 상관 확인(위 "미래 데이터 금지" 절 마지막 항목).
- [ ] Causality 감사: 라벨 생성이 그 episode 시작 시점 이후 정보만 쓰는지.
- [ ] VAL 캘리브레이션 → OOS-Q1+OOS-Q2 단일터치(Odyssey4 프로토콜 재사용).
- [ ] G0 대비 비교: Odyssey4 G0(VAL with_gate 77.31%/−21.76%/26건)와 나란히 보고, PnL/MDD
  트레이드오프를 정직하게 표로.
- [ ] ETH 드로다운 거버너가 발견한 **재진입/경로효과 위험**을 재확인: 이 게이트가 스킵한 신호 bar 직후
  같은 신호가 다시 뜨면 무한 재평가 루프가 생기는지(ETH 드로다운 거버너의 처닝과는 메커니즘이 다르지만 —
  여긴 강제청산이 아니라 진입 자체를 매 bar 재평가하는 구조라 원칙적으로 처닝 위험은 적다,
  단 확인 필요).

## 필수 저비용 게이트 (cheap_gate, 구현 전 반드시 먼저 통과)

이 저장소의 확립된 방법론(RL 사이징 서브프로젝트, ETH 드로다운 거버너 cheap_gate)을 그대로 적용한다 — 학습
모델을 훈련하기 전에 **이미 존재하는 무료 신호**로 같은 효과를 얻을 수 있는지 먼저 확인한다:

1. **quality_threshold를 그냥 올리기** — h48qual 0.50/zig075 0.75보다 높은 값(예: 0.60/0.85)이
   이미 비슷한 하방-필터링 효과를 내는지 VAL에서 확인. quality 자체가 방향 확신도의 단조
   함수이므로, conformal 모델이 순수하게 quality보다 더 나은 하방 판별력을 갖는지 이 비교로
   먼저 검증해야 한다.
2. **기존 exit_head/quality 확률의 단순 분위수 컷** — 학습된 회귀 2개 없이, 이미 계산되는
   direction/quality 확률의 percentile만으로 유사 veto 규칙을 흉내낼 수 있는지.
3. 위 두 무료 후보가 이미 conformal 모델과 비슷한 효과를 낸다면, 학습 모델 전체를 정당화하기
   어렵다 — 반드시 먼저 통과시켜야 하는 게이트.

## 미해결 이슈

1. **인접 episode 라벨 상관 — 2026-08-16 구현 후 실측, 심각한 문제로 확인됨.** lag-1
   자기상관이 0.55~0.85(전 창·전 컴포넌트)로 매우 높다 — AR(1) 근사 유효표본크기가 원표본의
   6~8배 작다(h48qual 학습풀 원1128→유효~170, zig075 원2732→유효~350). Purge/embargo/uniqueness
   weighting 없이 이 라벨로 잔차분위수를 계산하면 LCB가 체계적으로 과신(낙관)하게 된다. 전체
   진단: `docs/experiments/eth_candidate_conformal_veto_episode_labels_20260816.md`. **이 이슈는
   더 이상 "확인 필요"가 아니라 "확인됨, 다음 단계 전 필수 해결 대상"이다.**
2. **v1.4식 축소(shrink+조기청산) 대안**: 이 계약은 v1.5의 veto 프레이밍을 따르지만, 완전
   거부 대신 notional 축소+exit 앞당김이 더 나을 수도 있다(ETH 드로다운 거버너의 causal trade editor
   후보와 겹치는 영역) — veto 버전이 기각되면 이 대안을 다음 후보로 검토.
3. **h48qual/zig075 라벨 소스가 다르다는 점**: h48qual과 zig075는 서로 다른 라벨 계열(zigzagfix_06
   vs zigzag_action)로 학습된 별도 헤드다 — conformal 모델을 컴포넌트별로 따로 학습할지
   공유할지 결정 필요. 표본 수(254 vs 789)가 크게 달라 공유 학습 시 zig075가 지배할 위험.
   **기본값: 컴포넌트별 독립 학습**(BTC도 sleeve 모델은 core 컴포넌트 단일 기준이라 이 이슈가
   없었음 — Odyssey 고유 이슈).
4. **잔차 대칭 가정의 타당성**: `pred_lcb = pred_full - residual_q`는 잔차가 좌우 대칭이라고
   가정한 단순화다(진짜 conformal prediction interval이 아님, 위 "이식 원본 재조사" 절에
   명시). ETH 트레이드 수익률 분포가 비대칭(스큐)이면 이 가정이 깨질 수 있다 — 구현 시 잔차
   분포의 스큐를 진단하고, 필요하면 하방 전용 분위수(quantile regression)로 대체 검토.

## cheap_gate 결과 (2026-08-16)

전체 과정: `docs/experiments/eth_candidate_conformal_veto_cheap_gate_20260816.md`. 요약:

- h48qual 임계값 0.65+, zig075 임계값 0.90 — 표면적으로 각각 PnL·MDD 동시 개선처럼 보이지만,
  직접 대조 확인 결과 **그 컴포넌트를 완전히 끈 것과 소수점까지 동일**. "품질 하방 필터"가
  아니라 이진 on/off 스위치 — h48qual이 이미 N≥5 시드로 무스킬 확정된 사실의 재확인일 뿐
  신규 발견 아님.
- zig075 임계값 0.80만 진짜 중간 상태(25건, on/off 어느 쪽과도 다른 수치): PnL −17.32pp에
  MDD +6.42pp 개선(비율 ≈2.7pp/1MDDpp) — ETH 드로다운 거버너가 찾은 어떤 트레이드오프보다 낫다.
- **그러나** 이 조작(변하지 않은 신호 위에서 quality_threshold만 재튜닝)은 저장소 전역
  `research_line_registry.json`의 `global_exit_constant_tuning` 항목과 정확히 겹친다 — "21+
  exit rounds and related sweeps did not survive validation/OOS." VAL 단일 창의 숫자 하나가
  이 기록을 뒤집을 근거가 되지 못한다.
- **판단**: cheap_gate는 "무료 필터로 충분하다"도 "무료 필터로 안 된다"도 확정하지 못했다 —
  표면적 숫자는 좋지만 그 종류의 조작 자체가 이미 반복 실패한 축이라 신뢰할 수 없다는
  애매한 결과. Conformal 회귀모델은 quality_score 단일 스칼라가 아니라 더 풍부한 causal
  피처+시뮬레이션된 미래경로를 쓰므로 질적으로 다르지만, **구현 시 모델이 결국 "h48qual
  항상 거부/zig075 선택적 거부"로만 수렴하는지(=quality_score 재현에 불과한지) 반드시
  진단해야 한다**(quality_score를 피처에서 빼고도 비슷한 veto 패턴이 나오는지 확인).

## 다음 단계

1. ~~cheap_gate~~ — 완료(2026-08-16), 결과 애매(위 절). 사용자가 (A) 회귀모델 착수로 결정.
2. ~~episode-시작 라벨 생성~~ — 완료(2026-08-16). h48qual 1128건/zig075 2732건(학습풀
   Q1~Q3) + VAL 캘리브레이션용 254/789건. **그러나 필수 진단(인접-episode 상관)에서 심각한
   문제 발견** — 위 미해결 이슈 1 참고. 전체: `docs/experiments/
   eth_candidate_conformal_veto_episode_labels_20260816.md`.
3. ~~purge/embargo + uniqueness weighting 설계~~ — 완료(2026-08-16, concurrency 기반 Lopez de
   Prado 스타일). **결과가 lag-1 근사보다 훨씬 나쁨** — zig075 유효표본이 학습풀 전체
   기준 ~114건(원 2732건의 4.2%), VAL 캘리브레이션은 창당 ~40건뿐. h48qual은 학습풀 ~166건
   (14.7%)로 상대적으로 낫다. 원인: zig075 median 보유기간(500~688 bar)이 episode 발생
   간격(~25~30 bar)보다 훨씬 길어 평균 20~25개 episode가 항상 동시에 겹침 — 계산이 실측
   가중치 비율(≈1/24)과 정확히 일치해 교차검증됨. 전체 진단:
   `docs/experiments/eth_candidate_conformal_veto_uniqueness_weights_20260816.md`. **이 결과는
   "40배 여유"라던 원래 실현가능성 판단의 핵심 전제를 약화시킨다 — 특히 zig075 캘리브레이션은
   원래 기대만큼 안정적이지 않을 수 있음을 인지한 채로 진행.**
4. ~~HGB 2개 학습(N≥5 시드)~~ — 완료(2026-08-16). 부트스트랩 재표본 시드검증 방법론 버그를
   먼저 발견·수정(`random_state`만으로는 HGB가 결정론적이라 시드가 전부 동일 결과를 냄 —
   `uniqueness_weight` 가중 부트스트랩 재표본으로 교체). **결과: zig075는 VAL
   상관관계(pred vs actual)가 5개 시드 전부 음수(방향이 거꾸로, corr −0.11~−0.10) — veto로
   쓰면 유해할 위험. h48qual은 5/5 양수지만 corr≈0.10~0.14(R²≈1%)로 미미.** 전체:
   `docs/experiments/eth_candidate_conformal_veto_hgb_train_20260816.md`.

## 다음 단계 — 사용자 판단 대기 (축 종결 여부)

cheap_gate(애매) → episode 라벨 인접상관(심각) → uniqueness 가중치 정밀재계산(더 심각) → HGB
실제 학습(zig075 역방향, h48qual 미미) — **4단계 전부가 같은 방향(이 축이 약하다)을 가리킨다.**
남은 단계(VAL 임계값 그리드 선택 → 포트폴리오 백테스트 통합 → OOS-Q1+OOS-Q2 단일터치)는
방법론적으로는 계속 실행 가능하지만, 투입 노력 대비 기대값이 이미 낮다. 권고: zig075는
폐기, h48qual은 계속하더라도 낮은 기대치로. 결정은 사용자 몫 — 계속한다면 결과는
`docs/experiments/eth_candidate_conformal_downside_veto_<date>.md`에 기록.
