# 일리아스 — h48qual exit_head 수동성(passivity) 근본원인 진단 (2026-08-17)

이 문서는 `docs/model_contracts/ilias_eth_human_direction_risk_management_contract_20260817.md`의
Open Issue (b)("exit_head를 방향 무관 배경신호에서 방향 품질 반응형으로 만드는 방법 미정")에
대한 근본원인 진단이다. **순수 진단 — 신규 학습·신규 백테스트 없음.** 이미 배포된 코드(라벨
정의, feature 구성)와 이미 계산된 산출물(`report.json`, 예측 CSV)만 읽어 답을 찾았다.

## 질문

오디세이4 h48qual의 exit_head(TabM 3-head 중 하나, threshold 0.95)가 왜 방향 품질(포지션이
결국 이기는 방향인지 지는 방향인지)과 거의 무관하게 거의 일정한 비율(21.8~27.7%,
`docs/experiments/ilias_eth_adaptive_exit_direction_quality_signal_design_20260817.md` 배경 절)로
발동하는가?

## 방법

읽은 소스(전부 인용, 추측 없음):
1. `docs/model_contracts/ilias_eth_human_direction_risk_management_contract_20260817.md`,
   `docs/experiments/ilias_eth_adaptive_exit_direction_quality_signal_design_20260817.md`,
   `docs/experiments/eth_odyssey4_random_direction_risk_management_ablation_20260817.md`(오늘
   발견의 1차 출처).
2. `docs/experiments/eth_omega461_exit_head_liveatr_relabel_walkforward_mechanism_diagnosis_20260815.md`(단서
   A, 08-15 진단) + 그 근거 `report.json` 3종.
3. `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`(L0/L9 절, 레짐게이트
   가중치-전환 메커니즘) + `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`
   #10/#11/#13(레짐인지형 exit 가드의 원본 설계·검증·섀도우 배포 기록).
4. **실제 배포 번들 `report.json`의 `label_contract`/`exit_label` 필드 직접 확인**(CRITICAL 규칙
   준수) — h48qual: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_
   zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/report.json`. zig075 대조용:
   `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_
   alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/report.json`. 두 경로는
   `trading_bot_modules/odyssey_live_adapter.py:70,78`, `trading_bot_modules/runtime_config.py:350,358`가
   실제로 참조하는 기본 번들 경로와 바이트 일치함을 확인했다.
5. 라벨 생성 코드: `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:812-964`
   (`_build_exit_dataset_entry_label_terminal_giveback`), `scripts/train_eval_omega1_2_tabm_exit_head_
   20260603.py:353-394`(`_position_feature_row`).
6. 피처 구성 코드(단서 B): 학습 — `scripts/train_eval_omega1_2_tabm_3head_20260603.py:180-195`
   (`_base_input`/`_exit_input_from_position_rows`); 라이브 — `trading_bot_modules/omega4_6_1_live.py:159-260`
   (`entry_decision`/`exit_probability`, **읽기만 함, 무수정**); 리플레이 — `scripts/train_eval_
   omega4_2_risk_sidecar_20260622.py:256-271`(`_predict_exit_prob_one`).
7. 레짐게이트 실제 구현: `scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_
   20260814.py:21-25,207-246`, `scripts/research_eth_omega461_exit_head_portfolio_asymmetric_
   20260813.py:63-67`(`NEW_H48QUAL_BUNDLE` 경로).
8. 오늘 어블레이션 산출물: `tmp/causal_regen_20260516/eth_odyssey4_random_direction_risk_
   management_ablation_20260817/exit_reason_distribution.csv`(pandas로 직접 재확인).
9. 08-15 진단 근거 `report.json`의 `component_h48qual`/`folds` 구조 직접 재확인(측정 단위가
   포트폴리오 레벨인지 컴포넌트 단독인지 검증).

## 결과

### 1. 단서 A(라벨 설계) — CONFIRMED, 그것도 두 겹으로

**h48qual의 "원본(재라벨 전)" exit_head 라벨을 실제 배포 번들 `report.json`에서 직접 확인했다**
(추측 아님, `exit_label` 필드 원문):

```json
{
  "mode": "entry_label_terminal_giveback",
  "diag": {
    "rows": 30000, "positive_rate": 0.0727,
    "continued_exit_reasons": {"hold": 27818, "terminal_window_exit": 2179, "mfe_giveback_exit": 3},
    "used_segments": 732,
    "risk_template": {"notional": 0.45, "leverage": 2.0, "take_profit": 0.026, "stop_loss": 0.014},
    "terminal_window": 3, "adverse_unreal": -0.01, "min_mfe_for_giveback": 0.006, "giveback_min": 0.65
  }
}
```

코드(`_build_exit_dataset_entry_label_terminal_giveback`, 812-964줄)를 직접 읽어 확인한 라벨
구성 로직: `zigzag_action` 라벨이 방향을 유지하는 **한 세그먼트(swing) 전체**를 진입~세그먼트
끝으로 보고, 그 안의 **매 bar마다** 3조건 OR로 라벨링한다 —

```python
terminal   = (세그먼트 끝까지 남은 bar 수) < 3        # 세그먼트 종료 직전 3bar → 항상 양성
adverse    = unreal <= -0.01                          # 미실현손익 -1% 이하 → 양성
gave_back  = mfe >= 0.006 and giveback >= 0.65 and unreal > 0.0  # 되돌림
```

실측 결과: 양성의 **99.86%(2179/2182)가 `terminal_window_exit`**(세그먼트 종료 임박)이고,
`mfe_giveback_exit`는 **0.14%(3/2182)**, `adverse_unreal_exit`는 **0건**이다. 즉 이 라벨은
"거의 항상 hold, 오라클이 정의한 트렌드 세그먼트가 끝나기 직전 3bar에서만 강제 청산"이라는
**순수 시간/세그먼트 경계 기반** 정책이다 — 포지션이 실제로 잘 되고 있는지 여부(unrealized
PnL, 방향 정합성)는 라벨 생성에 사실상 관여하지 않는다(adverse 0건, giveback 0.14%). **이게
바로 방향 품질과 무관한 이유의 1차 근거다**: 라벨 자체가 "이 거래가 이기고 있는가"가 아니라
"오라클 세그먼트가 끝나가는가"만 인코딩한다.

**대조 확인(zig075)**: zig075 번들의 `exit_label.diag`는 h48qual과 **완전 동일**
(positive_rate 0.0727, `continued_exit_reasons` 세 값 전부 동일, `used_segments=732` 동일).
원인은 두 번들의 `direction_label_dir`가 **같은 경로**(`zigzag_action_labels_20260531`)이고,
이 exit 라벨 함수가 `quality_label_dir`가 아니라 `zigzag_action` **direction** 라벨만
소비하기 때문이다 — h48qual/zig075는 quality 라벨만 다를 뿐 exit 라벨 생성 로직·데이터는
바이트 단위로 동일하다. 이것이 zig075 exit_head가 구조적으로 거의 관여하지 않는 이유(0/86,
`docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md` 실행 로그 #1/#6)와
**같은 뿌리**임을 직접 확인했다 — 별개 메커니즘이 아니다.

**h48qual만 추가로 받는 "liveATR 재라벨" 믹스**: `docs/model_contracts/odyssey4_eth_full_stack_
architecture_20260814.md` L9절이 기술한 대로, h48qual 보유 포지션은 지속상승장 탐지기(같은
`dual_momentum` rolling 신호)가 ON이면 위 원본 exit_head를, OFF면 **liveATR 재라벨** exit_head를
조회한다 — 코드로 직접 확인(`research_eth_omega461_regime_aware_exit_head_uptrend_guard_
20260814.py:21-25`): "detector ACTIVE → h48qual's ORIGINAL frozen exit head ... detector
INACTIVE → h48qual's current live-ATR-relabeled exit head". zig075는 이 믹싱 자체가 없다
(`docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md` #11: "zig075는 완전
동결"). liveATR 재라벨 라벨은 08-15 진단(단서 A 원본 문서)이 이미 규명한 대로 **양성의
75.7~79.8%가 `mfe_giveback_exit`**(MFE≥0.6% 후 65%+ 반납, 아직 순양)에서 나온다 — 이것도
"작은 반등 뒤 눌리면 나가라"는 **국소·후향적 노이즈 휴리스틱**이며, 승패와 무관하게 어느
거래에서든 비슷한 빈도로 충족된다(08-15 문서 §3, 4개 재학습 폴드에서 양성률 18.6~19.9%·
giveback 비중 75.7~79.8%가 학습기간 길이(53~296일)와 무관하게 거의 일정).

**결론**: h48qual exit_head가 실제로 조회하는 두 가중치 세트(원본/liveATR재라벨) **둘 다**
설계상 "이 거래가 이기고 있는가"를 라벨에 반영하지 않는다 — 원본은 세그먼트 경계까지 남은
시간만, 재라벨은 국소 되돌림 노이즈만 본다. 방향 품질과 무관한 발동은 **버그가 아니라 라벨
설계의 직접적 귀결**이다.

### 2. 단서 B(feature 가시성) — REFUTED

`docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`의 "102 base + 13 pos(진입
시 0으로 채움) = 115차원" 문구는 **direction/quality head의 진입-시점 입력**에만 해당한다.
exit_head는 별도 경로로 호출되며, `POS_COLS`(13개: `pos_side/pos_hold_bars/pos_unrealized/
pos_mfe/pos_mae/pos_giveback/pos_dist_to_tp/pos_dist_to_sl/pos_notional/pos_leverage/pos_exposure/
pos_tp/pos_sl`, `trading_bot_modules/odyssey_tabm_core.py:45-59`)는 **보유 중 매 bar 실제 값으로
갱신된다** — 학습/라이브/리플레이 3곳 모두에서 동일하게 확인:

- **학습**: `_build_exit_dataset_entry_label_terminal_giveback`(812-964줄)이 매 in-position bar마다
  실시간 mfe/mae/unreal을 누적 갱신하며(878-888줄), `exit_head._position_feature_row`(`scripts/
  train_eval_omega1_2_tabm_exit_head_20260603.py:353-394`)로 그 값을 `pos_*` 컬럼에 채운다 —
  0/placeholder 아님. entry-시점 direction/quality용 `_base_input`(180-184줄)만 `POS_COLS`를
  0으로 채운다(포지션이 없으니 정당함).
- **라이브**(`trading_bot_modules/omega4_6_1_live.py:234-260`, 읽기만 함): `exit_probability`가
  호출자(`trading_bot.py`)로부터 실제 `hold_bars`/`unrealized_move`/`mfe`/`mae`/`notional`/
  `leverage`/`take_profit`/`stop_loss`를 받아 `pos_values` dict(246-253줄)를 구성하고 `POS_COLS`
  전부를 그 값으로 덮어쓴다(255-256줄).
- **리플레이**: `train_eval_omega4_2_risk_sidecar_20260622.py:256-271`(`_predict_exit_prob_one`)도
  호출자가 전달한 `pos_values`를 `pos_idx` 위치에 그대로 주입한다.

세 경로의 `pos_*` 필드 집합과 계산식(giveback 공식 포함)이 사실상 동일해 **train/live/replay
parity가 있다**. 즉 exit_head는 "이 거래가 잘 되고 있는지" 판별할 입력을 **실제로 받는다** —
구조적으로 못 받는 게 아니다. 문제는 입력이 아니라 **그 입력을 어떤 라벨로 지도학습했는가**(§1)다.
`omega4_6_1_extended_oos_20260706/h48qual/validation_predictions_q050.csv`를 pandas로 직접 열어
확인한 결과 이 CSV는 direction/quality 헤드 출력만 담고 있고 exit_head 확률이나 `pos_*` 컬럼
자체가 없다 — exit_head는 정적으로 사전계산되지 않고 보유-bar마다 동적으로 호출되는 구조이기
때문이며, 이는 위 코드 추적 결과와 정합한다(태스크가 제안한 "가벼운 CSV 확인"은 이 CSV엔
적용 대상이 없어 코드 추적으로 대체).

### 3. 오늘 수치(21.8~27.7%) vs 08-15 수치(82~96%) — 왜 다른가

두 수치는 **다른 것을 측정한다**. 산출물을 직접 대조해 확인했다:

| | 08-15 진단(`eth_omega461_exit_head_liveatr_relabel_walkforward_20260814/report.json`) | 오늘 어블레이션(`exit_reason_distribution.csv`) |
|---|---|---|
| 측정 단위 | `component_h48qual`만 — h48qual **단독** 리플레이(같은 report.json 안에 `portfolio_no_gate`/`portfolio_with_gate`가 별도 키로 존재 = component 단독 평가가 명백히 분리된 하위 실험) | **포트폴리오 전체**(h48qual+zig075가 L5 단일슬롯 공유, `real_g0`/각 arm의 KEPT 거래) |
| L4.5 duration gate | 미적용(컴포넌트 단독 수치) | 적용(with_gate KEPT 거래만 집계) |
| zig075 혼입 | 없음 | 있음 — zig075는 exit_head가 구조적으로 0/86이므로 섞이면 비율을 **희석**시킴 |
| VAL 거래수(예시) | 63건(relabel만 100% 적용) | 26건(real_g0, 레짐게이트 적용) |
| VAL exit_head 비중 | 82.5%(52/63) | 26.9%(7/26, `exit_reason_distribution.csv` 직접 재확인) |

`docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md` #11이 이미 밝힌 대로,
VAL/OOS-Q1/OOS-Q2에서는 레짐게이트가 적용된 h48qual 자체의 의사결정이 "liveATR 재라벨 100%
적용"과 **원장 바이트 일치**한다(탐지기 ON 시간이 이 3창에서 워낙 짧고, Q2에서 1,340회
발동해도 원본·재라벨 결정이 한 번도 갈리지 않았음을 직접 검증한 결과) — 즉 **레짐 가중치
전환 자체는 이 3개 판정창에서 h48qual의 exit_head 발동 여부를 바꾸지 않는다.** 그런데도
거래수(63건 vs 26건)와 비중(82.5% vs 26.9%)이 크게 다른 건, 격차의 원인이 "가중치가
섞였다"가 아니라 **측정 모집단 자체가 다르다**는 뜻이다 — h48qual 단독 재현 vs h48qual+zig075가
같은 슬롯을 놓고 경쟁하고 L4.5로 일부가 걸러지는 실제 포트폴리오. zig075가 exit_head를 구조적으로
쓰지 않는다는 §1의 확인과 결합하면, 포트폴리오 집계에 zig075 거래가 섞여 들어갈 때마다 exit_head
비중이 h48qual 단독 수치보다 낮게 나오는 방향은 설명되지만, 63건→26건이라는 정확한 거래수 축소
경로(어느 만큼이 zig075로 넘어갔는지, 어느 만큼이 L4.5로 걸러졌는지)는 이번 세션에서 완전히
분해하지 못했다 — **정직한 한계로 남긴다**(아래 "다음 세션 액션" 참고). 다만 §1의 라벨설계
결론(두 가중치 세트 모두 방향 품질 무관)은 측정 모집단과 무관하게 성립하므로, 이 산술적
미해결이 §1의 근본원인 결론 자체를 흔들지는 않는다.

## 결론

1. **근본원인은 단서 A(라벨 설계) — 강한 근거로 CONFIRMED.** h48qual exit_head가 실제로
   조회하는 두 라벨 세트(원본 `entry_label_terminal_giveback`, liveATR 재라벨) 모두 "이
   거래가 이기고 있는가"를 라벨 정의에 사실상 반영하지 않는다 — 원본은 세그먼트 종료까지
   남은 시간(터미널 윈도우 3bar)이 양성의 99.86%를 결정하고, 재라벨은 국소 MFE-되돌림 노이즈가
   양성의 75~80%를 결정한다. 두 경우 다 승패와 무관하게 어느 거래에서든 구조적으로 비슷한
   빈도로 충족되므로, 발동률이 방향 품질과 무관한 것은 **설계상 당연한 결과**다.
2. **단서 B(feature 가시성)는 REFUTED.** `pos_*` 13개 컬럼은 학습·라이브·리플레이 3곳 모두에서
   보유-bar마다 실제 포지션 상태(미실현손익, MFE, 보유시간, giveback 등)로 갱신된다 —
   train/live parity 확인됨. exit_head는 "잘 되고 있는지" 판별할 입력을 받고 있으나, 그 입력을
   지도학습한 라벨 자체가 방향 품질 무관 신호이므로 방향 품질에 반응하는 법을 배울 기회가
   애초에 없었다.
3. zig075의 exit_head 구조적 무관여(0/86)는 **h48qual의 문제와 같은 뿌리**다 — 두 컴포넌트가
   공유하는 원본 exit 라벨 자체가 거의 항상 hold를 산출하도록 설계돼 있고, zig075는 h48qual과
   달리 liveATR 재라벨 믹싱조차 받지 않아 그 구조가 그대로 노출된다(0%에 가까움). 별도 메커니즘
   재조사는 불필요.
4. Odyssey2의 exit_head 재라벨 실험 15건(`docs/model_contracts/odyssey2_eth_live_injection_
   contract_20260813.md`)은 "다른 라벨/모델로 exit_head PnL을 개선할 수 있는가"를 반복
   검증했고 전부 부정 결과였다 — 그러나 "발동 패턴이 방향 품질에 따라 갈리는가"를 arm-통제
   방식으로 직접 측정한 적은 없다(1차 연구 질문 설계 문서가 이미 지적한 구분과 일치). 이번
   진단은 그 15건과 **중복이 아니다** — 15건은 "다른 라벨 레시피가 이기는가"를, 이번 진단은
   "왜 어느 레시피를 써도 방향 무관성이 반복되는가"라는 한 단계 위의 질문에 답했다.

## 일리아스 1차 연구 질문에의 함의

1차 연구 질문(`docs/experiments/ilias_eth_adaptive_exit_direction_quality_signal_design_
20260817.md`)이 제안한 "새 exit 신호"는 **함정 4(순환 논리)에 이미 정확히 걸려 있던 문제의
근본원인이 이제 확인됐다** — 기존 exit_head의 두 라벨 세트 모두 방향 품질을 전혀 보지 않으므로,
**단순히 라벨 파라미터를 재조정(giveback 문턱 상향, terminal window 조정 등)하는 것으로는
방향 품질 반응성을 만들 수 없다** — 08-15 진단이 이미 "학습구간을 바꿔도 소용없다"를 보였고,
이번 진단은 "왜 애초에 두 레시피 다 방향 품질을 안 보는가"까지 규명했다. 반면 **feature
입력단은 문제가 없다**(단서 B REFUTED) — `pos_unrealized`/`pos_mfe`/`pos_giveback` 등은 이미
실시간으로 정확히 공급되고 있으므로, 1차 연구 질문이 제안한 "진입 시 quality 확신도, 미실현
손익 궤적, MFE-so-far, 보유 bar 수" 같은 입력 후보는 **새로 만들 필요 없이 이미 파이프라인에
존재**한다(`pos_unrealized`=미실현손익, `pos_mfe`=MFE-so-far, `pos_hold_bars`=보유 bar 수;
"진입 시 quality 확신도"만 신규 파생이 필요 — entry 시점 `quality_for_action`을 포지션
전체에서 상수로 들고 있어야 함).

**구체적 방향 제시**: 다음 세션이 설계할 새 exit 신호는 (a) 반드시 **라벨을 "이 거래가
결국 SL로 끝나는가 vs TP로 끝나는가"로 재정의**해야 한다(1차 연구 질문 §3 초안과 일치) —
세그먼트 경계나 국소 되돌림이 아니라 **거래의 최종 귀결**을 직접 예측하는 라벨이어야 방향
품질과 연결될 수 있다. (b) feature는 재구축 불필요 — 기존 `pos_*` 파이프라인을 그대로
재사용하되 라벨만 교체하면 된다(엔지니어링 비용 낮음, feature 배관은 이미 검증됨). (c) 함정 4
(순환 논리)를 피하려면 새 라벨은 exit_head의 발동 여부와 완전히 독립적으로 구성해야 한다 —
"TP/SL 도달까지 시뮬레이션을 연장했다면 어느 쪽이 먼저였을까"를 반사실적으로 재구성하는
1차 연구 질문의 후보안이 이 진단 결과와 정확히 정합한다.

## 준수 확인

`fresh_forward_bar_by_bar=N/A`(신규 리플레이 없음, 순수 사후 진단), `trade_ledgers_used_as_
input=false`(집계 통계·라벨 구성 통계·report.json 필드만 사용), `saved_parent_exit_timestamps_
used=false`, `future_rows_used_for_entry=false`. 라이브 파일(`trading_bot.py`,
`trading_bot_modules/*`, `runtime_config.py`, `.env`)은 읽기만 했고 수정하지 않았다. 이 진단은
promotion·모델 선택 근거가 아니라 순수 진단이며, 1차 연구 질문의 다음 세션 설계에 참고자료로만
쓴다.

## 산출물

신규 학습·신규 스크립트 없음. 인용한 기존 산출물:
- `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/report.json`
- `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/report.json`
- `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_20260814/report.json`
- `tmp/causal_regen_20260516/eth_odyssey4_random_direction_risk_management_ablation_20260817/exit_reason_distribution.csv`
- `tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/h48qual/validation_predictions_q050.csv`(pos_/exit 컬럼 부재 확인)

## 다음 세션 액션

1. §3의 미해결 산술(63건→26건 거래수 축소가 어느 만큼 zig075 슬롯 이관 vs L4.5 필터링인지)을
   완전히 분해하려면, 오늘 어블레이션의 포트폴리오 렛저(component별 태그가 있는 원본 trades
   dataframe, `greedy_replay_entry_veto`가 내부적으로 생성)를 component 컬럼 기준으로 groupby —
   단, 이는 "정확한 산술 규명"이 목적이지 §1의 결론(근본원인=라벨 설계)을 바꾸지 않는다는 걸
   먼저 확인하고 우선순위를 낮게 둘 것.
2. 1차 연구 질문의 라벨 재설계(§3 "TP/SL 최종 귀결" 라벨) 착수 — 이 진단이 feature 배관은
   이미 정상임을 확인했으므로, 다음 세션은 라벨 정의와 함정 4(순환 논리) 회피 설계에만
   집중하면 된다.
