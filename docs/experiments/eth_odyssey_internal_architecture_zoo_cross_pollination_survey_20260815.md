# ETH Odyssey 프로젝트 — 저장소 내부 모델 아키텍처 전수조사(Alpha/Omega/MuZero/CSALT/HMM/Sigma 등) 및 이식 후보 (2026-08-15)

상태: **순수 리서치 문서 — 코드 변경 없음(단, 조사 과정에서 발견한 Odyssey4 아키텍처 요약 문서의
사실오류 2건은 같은 세션에서 직접 코드 검증 후 정정함).**

## 요청

사용자: "알파 오메가 등 모든 모델 아키텍쳐를 조사해서 오디세이 프로젝트에 주입할만한 아이디어가
있는지 조사해줘."

Odyssey는 이 저장소의 BTC 위주 Omega4.6.1 스택을 ETH에 옮겨온 것이라, 두 프로젝트가 상당 부분
구조를 공유한다. 이 조사의 진짜 질문은 "완전히 새로운 아이디어를 찾는 것"이 아니라 "이 저장소가
BTC/SOL/기타 자산에서 이미 만들고 검증까지 해본 구조적 메커니즘 중, Odyssey에는 아직 없는 것이
뭔가"였다.

## 방법론

6개 병렬 조사 에이전트로 아래 계열을 각각 훑게 하고(모델명/메커니즘/현재 상태/Odyssey 이식 적합성
형식으로 보고받음), 그 결과 중 사실 검증이 필요한 주장은 리드 세션이 직접 코드/pickle을 열어
재확인했다:

1. Alpha1~8 (BTC)
2. Omega1.x lifecycle/sleeve 계열 (exit-feature lifecycle controller, post-lifecycle bucket
   adapter, cash-sleeve EV veto, TabM Expert-DQ)
3. Omega4.x (4.2~4.6.2) — Odyssey가 직계 상속한 계열
4. clean_base + certified_teacher 리스크/포트폴리오 레이어
5. MuZero/AlphaZero planning 계열 + CSALT(causal rank/DP advantage/policy distillation 등) RL
   인접 계열
6. HMM confluence meta-label + micro_scalp MoE + Sigma10/11 — 전부 ETH 대상

닫힌 축 재탕을 피하기 위해 각 에이전트에게 Odyssey 자체의 닫힌 축(RL 5개 삽입점 전부,
evidence-signal 직접주입, trend-following/vol-targeting, oscillator/AMT/VSA/orderflow)과
저장소 전역 `research_line_registry.json`의 닫힌 라인(BTC RL 방향정책, 일반 기술지표 탐색 소진
등)을 먼저 알려주고 시작했다.

## 0. 조사 중 발견한 문서 버그 2건 (같은 세션에서 정정 완료)

Omega4.x 조사 에이전트가 "Odyssey4 아키텍처 요약 문서가 실제 라이브 코드와 다르다"고 보고했고,
리드 세션이 `trading_bot_modules/omega4_6_1_live.py`와 라이브 sidecar pickle 2개(h48qual q050,
zig075 q075)를 직접 열어 재검증했다:

1. **Duration OU-halflife 리스크 게이트가 다이어그램에서 누락돼 있었다.** `DURATION_FEATURE=
   "ou_halflife"`, `DURATION_THRESHOLD=0.005417`이 `omega4_6_1_live.py`에 하드코딩돼 실제
   라이브 경로에서 h48qual/zig075 공통으로 작동 중이다(BTC
   `omega4_6_1_duration_ou_halflife_risk_gate_20260630_contract.md`에서 이식된 것). "Odyssey에
   주입할 아이디어"로 제안될 뻔했으나 **이미 있었다.**
2. **L7 사이징 사이드카가 "CatBoost"로 잘못 표기돼 있었다.** 라이브 pickle을 직접 열어 확인한 결과
   `model_kind`는 `HistGradientBoostingRegressor`, `selection_objective="log_risk"`(단순 PnL이
   아니라 `log_growth - tail_penalty*tail_excess - liquidation_penalty*liquidation_excess`) —
   BTC Omega4.2/4.3과 완전히 동일한 코드(`scripts/train_eval_omega4_2_risk_sidecar_20260622.py`).

두 건 모두 `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`에 반영해
정정했다(L4.5 레이어 추가, L7 설명 수정). **이 정정 자체가 이번 조사의 첫 번째 성과다** — "새
아이디어"라고 생각했던 것 중 최소 두 개는 이미 배포돼 있었고, 참고 문서만 뒤처져 있었다.

## 1. Odyssey 이식 후보 — Tier 1 (가장 유망: 이미 같은 코드베이스에서 shadow/redteam 통과, Odyssey의
   문서화된 실제 공백을 정확히 겨냥)

### 1-1. Drawdown/MDD Budget Governor
- 출처: `docs/model_contracts/clean_base_deep_gated_drawdown_budget_v5_contract.md`
- 메커니즘: 실현 equity peak 대비 계좌/일일 drawdown 캡, 연패 캡, hard loss stop, profit-only
  trailing lock으로 exposure 버킷을 스로틀. entry side는 불변, exit는 원본보다 당기는 방향으로만
  작동. 미래 데이터 사용 없음(관측된 cash/mark-to-market 경로/과거 peak만 입력).
- 근거: 레드팀 `APPROVED_AS_SHADOW_FRONTIER`(BTC), OOS PnL +382.2%, MDD -18.47%.
- Odyssey 공백: **Odyssey는 트레이드 단위 사이징(L7 HGB sidecar)만 있고, 포트폴리오 equity-peak
  기반 drawdown 예산이 전무하다** — 계약서 자체가 명시하는 구조적 빈틈. 공유 1슬롯 구조라
  "계좌 DD/연패 초과 시 신규 진입 차단 또는 NOTIONAL_CAP 임시 하향"으로 L7 뒤에 바로 붙일 수
  있다.

### 1-2. Conformal Veto Sleeve
- 출처: `docs/model_contracts/clean_base_causal_sleeve_conformal_veto_v1_5_contract.md`
- 메커니즘: validation에서 캘리브레이션한 downside 잔차로 conformal lower-confidence-bound(LCB)를
  산출, LCB가 나쁘면 진입/증액을 거부. entry side/leverage 핵심 값은 불변(veto만 가능).
- 근거: `shadow_candidate`, OOS PnL +210.08%(base 대비 +32.8pp), MDD 거의 불변. `trading_bot.py`/
  `runtime_config.py`에 이미 배선돼 BTC 섀도우로 운영 중 — 포팅 인프라 자체가 이미 공유 런타임에
  존재한다.
- Odyssey 공백: L3 진입 게이트는 정적 `quality>=threshold` 하나뿐이고, 그 확률에 대한 캘리브레이션된
  불확실성이 전혀 없다. quality/exit_head 예측의 validation residual로 LCB를 내어 "threshold는
  넘지만 하방 LCB가 나쁜" 진입을 추가로 거르는 L3.5 게이트 후보.

### 1-3. Causal Trade Editor
- 출처: `docs/model_contracts/clean_base_causal_trade_editor_v1_3_contract.md`
- 메커니즘: 고정된 core entry/side 위에서 causal feature로 `effective_notional`(사이즈)과
  `effective_exit_idx`(조기청산 스케줄)만 사후 학습·조정.
- 근거: `shadow_candidate`, OOS PnL +190.79%(+13.5pp), MDD -18.87%.
- Odyssey 공백: L6 TP/SL이 2025년 세 분기 내내 floor(0.075/0.040)에 관측상 항상 포화돼 있다는
  것이 이미 알려진 미해결 이슈다("ATR-적응형"이라는 이름과 달리 사실상 고정폭). 사후 편집기가
  이 포화를 우회하는 직접적인 대응책이 될 수 있다.

## 2. Tier 2 — zig075의 죽은 exit_head(0/53건 발동)를 정면으로 겨냥

zig075는 exit_head 확률이 세 분기 통틀어 단 한 번도 0.95를 넘은 적이 없다(0/53 관측) — 사실상
TP/SL로만 청산된다. 아래 세 후보는 서로 다른 방식으로 이 죽은 파라미터를 살리려는 시도다.

### 2-1. Cash-Sleeve EV-HGB Veto (가장 구체적인 이식 근거)
- 출처: `docs/model_contracts/omega1_2_3_ev_hgb_cash_sleeve_20260615_contract.md`,
  `trading_bot_modules/omega1_2_3_cash_sleeve.py`
- 메커니즘: parent가 CASH를 내고 오픈 포지션이 없을 때만 개입하는 별도 long/short HGB EV
  회귀모델. `ev_min=0.002` 초과 시에만 fallback 진입, parent 재활성 시 즉시 청산. "게이트
  실패=끝"이 아니라 "게이트 실패=별도 저빈도 모델의 기회"로 재해석하는 프레이밍이 핵심.
- 근거: `walkforward_pass_live_wired`(BTC). 월별 4-fold 중 3-fold 개선, OOS fallback-only
  +3.33%(trades=16, WR 56%, PF 1.33). 더 공격적인 `ev_min=0.004`(+7.91%)는 월별 불안정성으로
  **명시적으로 기각된 이력**이 있어 과적합 방지 사례로도 신뢰도가 있다.
- Odyssey 적용: 같은 "실패한 결정을 별도 모델의 입력으로 재활용" 논리를 zig075의 exit_head에
  적용 — L3 게이트 탈락(quality<threshold) bar들을 모아 별도 EV 모델로 재평가하거나, zig075
  보유 중 exit_head가 침묵하는 구간을 별도 모델이 대신 감시하는 자리.

### 2-2. Exit-Feature Lifecycle Mamba-SAC Controller
- 출처: `docs/model_contracts/omega1_2_exit_feature_lifecycle_baseline_20260604_contract.md`
- 메커니즘: frozen exit_head 출력을 즉시 청산 결정이 아니라 `exit_p_hold/exit_p_exit/exit_edge`
  3개 feature로만 노출. Direction/Quality 출력 + 포지션 상태(side, notional, unrealized return,
  MFE/MAE, giveback, hold bars, TP/SL까지 거리)를 seq_len=64 시퀀스로 Mamba 인코더에 태워 offline
  SAC/AWAC discrete controller가 `hold/enter_base/enter_aggressive/reduce50/full_exit` 중 하나를
  매 bar 선택. Exit Head가 "결정권자"에서 "상태 관측치"로 강등된다.
- 근거: **단일 시드(seed260604)뿐** — 이 프로젝트의 seed-diversity 게이트(N≥5)를 충족하지 못한다.
  OOS PnL +16.07%/MDD -5.40%/WR 65.6%/trades=32는 참고치일 뿐 신뢰 근거로 인용 불가.
- Odyssey 적용: L9의 "exit_head>=0.95 즉시청산" 룰을 대체하는 자리. zig075처럼 exit_head가
  구조적으로 침묵하는 컴포넌트에 정확히 맞는 문제의식이지만, **먼저 N≥5 재시드 재현부터 해야
  근거로 쓸 수 있다.**

### 2-3. Giveback/주문흐름-델타 조건부 27차원 Exit Head 입력
- 출처: `docs/model_contracts/alpha6_entry_quality_exit_5bucket_main_20260522_contract.md`
- 메커니즘: exit_head 입력에 원시 피처 외에 `mfe, mae, giveback, giveback_ratio, hold_frac,
  remaining_frac, target_horizon_frac, ret_atr`와 **side로 부호 반전시킨** 주문흐름 delta(OBI,
  taker, whale, EAI, OI, funding rate)까지 27차원 포지션-상태 벡터로 결합.
- 근거: 연구 candidate, cost1~3 PnL +15.3%/+14.2%/+12.0%, MDD 약 -4.8%. live 미승격.
- Odyssey 적용: zig075의 exit_head가 왜 한 번도 발화하지 않는지에 대한 가설 하나 — **포지션
  이후에만 관측 가능한 신호(giveback, 부호반전 주문흐름 delta) 자체가 입력에 없어서 exit_head가
  "볼 게 없는" 상태일 수 있다.** 위 두 후보(2-1, 2-2)와 결합 가능한 피처 엔지니어링 축.

## 3. Tier 3 — 낮은 우선순위 (근거가 약하거나 시간축이 안 맞음, 참고만)

- **Target-Horizon 5-Bucket Head**(Alpha6, 위와 동일 출처): action/quality와 별도로 예상
  보유기간을 5버킷(6/12/24/48/96bar)으로 분류하는 헤드 — L6 floor 포화 문제에 대한 또 다른
  각도지만 live 미승격, 단일 실험.
- **Post-Lifecycle HGB Bucket Adapter**(`omega1_2_post_lifecycle_bucket_adapter_20260605`) — L6
  포화 대응 후보지만 seed가 `s260692~695`처럼 +1 증분 클러스터라 [[tabm_hp_low_signal_pattern]]과
  같은 가짜-다양성 우려가 있다. 재검증 없이 근거로 쓰지 말 것.
- **Omega4.6.2 cap220: RSI 숏진입 스킵 + 비대칭 노출 + 120시간 강제 time-stop**
  (`omega4_6_2_cap220_short_boost125_time_stop120h_20260630`) — ETH에서 실제로 exit_head가
  발동하지 않은 채 26일간 보유된 SHORT 사례가 관측된 상태라, 무제한 보유를 막는 bounded
  time-stop은 개념적으로 직접 관련 있다. 다만 BTC 단일창 근거이고 상태 자체가
  `conditional_diagnostic_pass_full_live_fail_fresh_holdout_required`(OOS가 후보선정에
  관여돼 clean 승격 근거 아님) — RSI/비대칭노출 부분은 버리고 "무제한 보유 방지" 아이디어만
  독립적으로 재검증하는 게 합리적.
- **Cost-Aware Allocator / Dual-Side Execution Router / Utility Ranker**
  (certified_teacher 계열) — L5의 고정 `PRIORITY=(h48qual, zig075)` 우선순위를 학습형
  배분기로 대체하는 구조 자체는 새롭다. 그러나 이 BTC 인스턴스는 OOS Cost1 PnL이 전부
  음수(-4.4%~-23.7%)로 사실상 폐기 선상 — **메커니즘 아이디어만 참고, 성능 근거로 인용 금지.**
  Odyssey 전용 재검증이 필요하다.
- **Exit Front-Run(체결가 오프셋을 exit 타이밍과 분리)**(`alpha3_exit_front_run_layer_20260514`) —
  L6 관련이지만 Odyssey는 실집행 slippage/limit-offset 최적화 문제가 아직 명확히 정의돼 있지
  않아 우선순위 낮음.

## 4. 검토했으나 재제안 근거가 없는 것 (기록용 — 다시 조사하지 말 것)

| 계열 | 결론 |
|---|---|
| MuZero/AlphaZero planning(entry/exit 전체를 학습된 policy-value 네트워크가 결정) | 기능적으로 이미 닫힌 Odyssey RL축과 같은 삽입점. BTC 자체 clean re-audit(`clean_scope_muzero_az_reaudit_2026.md`)에서 `reject_overlay_keep_base_under_shadow_review` — overlay 제거 base만 생존, overlay는 OOS -68%/MDD -81%로 붕괴. 후속 micro-add loop2~5도 전부 reject 또는 shadow-only. |
| CSALT(causal rank/DP advantage/policy distillation 등 6종) | policy-gradient RL은 아니고 DP-oracle teacher label을 지도학습으로 증류하는 구조지만, `btc_csalt_dp_label_loop_final_20260715.md`에서 6개 전부 `development_fail`(1,320개 후보 중 경제성 게이트 통과 0개) — 저장소 별도 메타발견("DP-oracle 포함 40+ label 방법론 전부 실패")과 정확히 겹침. |
| Feature Max Hazard Firewall(`clean_base_feature_max_hazard_firewall_v6`) | L7 HGB sidecar와 기능 중복(entry-time feature→로컬 노셔널 캡). 신규 가치 낮음. |
| Regime MoE(`certified_teacher_regime_moe_v1`), bull/bear/chop 전문가별 독립 서브넷 | Odyssey에 이미 있음(h48qual/zig075 둘 다 컴포넌트별 3-expert 내장, BTC와 동일 구조). |
| TabM Expert-DQ soft-regime weighting(`omega1_2_softfloor00_tabm_expertdq`) | direction_head 자체가 N≥5로 무스킬 확정된 닫힌 축의 재탕에 가까움. |
| HMM 레짐→메타라벨 조건화(`eth_hmm_confluence_meta_labels_v1/v2`, ETH, Odyssey보다 ~2.5주 이전) | 레짐을 라우팅이 아니라 라벨/타겟 생성 자체에 반영하는 구조적으로 다른 아이디어이긴 하나, v2가 VAL +28.8%→OOS -20.2%로 반증됐고 `promotion_eligible=false`. |
| 재고(inventory) 조건부 다중전문가 합의 거부권(`eth_micro_scalp_inventory_moe_ensemble_v2`, ETH) | 비용게이트 실패(validation -0.71%), 무엇보다 실제 median 보유시간 116~264분으로 "1분봉 스캘핑" 설계의도 자체가 안 지켜짐 — Odyssey(5분봉 스윙성)와 시간축 불일치. |
| 연속보유 기회비용/이탈위험 오버레이(`eth_micro_scalp_opportunity_moe_v3`) | 프로모션 근거 전무, fresh-forward 전 자동 CASH 고정 — 검증 자체가 안 됨. |
| Sigma10 레짐-서브셋 학습, Sigma11 동적 레버리지 | 전자는 인접 threshold에서 노이즈로 반증(chop_thr=0.42 근방 붕괴), 후자는 이미 닫힌 vol-targeted sizing 축과 동일 메커니즘. |
| Day-Opportunity 효용 게이트(Alpha7) | 닫힌 evidence-signal/엔트리게이트 축과 동일 계열, 1차 실험부터 실패(train pass율 82%→OOS 26.5%). |
| Teacher/Deep-Parent 검증 오버레이(Alpha2~4) 구조 자체 | Odyssey3/4의 causal veto와 구조적으로 동일 — 신규성 없음. **단, 재사용 가치 있는 경고**: Alpha4.2/4.3에서 "validation은 teacher 채택을 선택했지만 OOS는 teacher 제거가 압도적으로 우세"였던 사례가 있다 — Odyssey4(zig075 SHORT veto, 지금 OOS 근거가 1건의 trade swap뿐)에 대한 직접적인 신중론으로 재활용해야 한다. |

## 5. 정직한 결론 및 제안 순서

- 가장 확실한 성과는 새 아이디어가 아니라 **문서 버그 정정**(0절)이었다 — "Odyssey에 없다"고
  생각했던 것 중 최소 두 개(duration 리스크 게이트, log-risk HGB 목적함수)는 이미 배포돼
  있었다. 아키텍처 요약 문서를 코드와 대조 없이 신뢰하면 안 된다는 게 이번 조사의 부산물.
- 진짜 신규 후보는 **모두 리스크/포트폴리오/exit 레이어**에 몰려 있다. direction_head/entry
  자체를 건드리는 아이디어는 하나도 살아남지 못했다(HMM meta-label, inventory MoE, Sigma10
  전부 이미 반증됨) — 이는 "게이트는 스킬을 만들지 않는다"는 Odyssey1의 핵심 발견과 일관된다.
- 우선순위 제안: **1-1(drawdown governor) 또는 1-2(conformal veto)를 먼저 프로토타입** — 둘 다
  BTC에서 shadow/redteam 통과 이력이 있고, Odyssey에 구조적으로 전혀 없는 공백(포트폴리오 리스크
  예산, 캘리브레이션된 진입 불확실성)을 정확히 채운다. 그 다음 **2-1(cash-sleeve EV veto)**을
  zig075의 죽은 exit_head 문제에 적용해보는 것을 제안한다. 2-2(lifecycle controller)는 아이디어는
  좋지만 먼저 N≥5 재현부터 통과해야 근거로 쓸 수 있다.
- 이 문서는 구현을 하나도 하지 않았다 — 사용자가 위 우선순위에 동의하면 다음 세션에서 후보를
  하나 골라 Odyssey용 데이터 계약서 작성부터 시작하는 것을 제안한다.

## 관련 문서

- 정정한 아키텍처 요약: `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`
- Odyssey 계약 체인: `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`,
  `odyssey3_eth_regime_guard_baseline_contract_20260814.md`,
  `odyssey4_eth_entry_veto_baseline_contract_20260814.md`
- 같은 날 작성된 RL 축 조사(다른 조사 대상, 상호 참조용):
  `docs/experiments/eth_odyssey4_rl_layer_integration_literature_research_20260815.md`
- 이 조사가 인용한 개별 계약서 30여 건은 본문 각주에 경로로 명시.
