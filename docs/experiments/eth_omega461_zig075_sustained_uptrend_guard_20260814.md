# zig075 지속상승장 가드 — 메커니즘 진단 및 설계 불가 결론 (2026-08-14)

상태: `diagnosed_no_valid_design` — **1단계 진단 결과 exit_head는 이 프로젝트가 관찰한 2025년
세 분기(Q1/Q2/Q3) 전부, 두 변형(baseline_both_original/asymmetric_tabm_liveatr) 전부에서
zig075 SHORT 거래의 청산 사유로 단 한 번도 등장하지 않는다(0/53건, 모든 청산은 stop_loss
아니면 take_profit).** 원칙적으로(Q3를 보지 않고) 캘리브레이션 가능한 exit_threshold 범위는
이미 #15가 VAL에서 전부 훑어(0.80~0.99) 강건한 개선 0개로 닫았고, 그 범위 안에서는 Q3에서도
거의 발동하지 않는다(0.80~0.90 구간 19건 중 0~1건만 발동). 그 아래로 내려가야 부분적 반응이
나오지만(0.60~0.75), 이 구간은 Q3 결과를 본 뒤에야 "효과가 있어 보인다"고 알 수 있는 사후
선택이라 이 프로젝트의 탐지기 캘리브레이션 규율(Q3 배제)을 그대로 위반하며, 그렇게 봐도
Q3 손실거래의 다수(16건 중 11건, 69%)는 여전히 반응하지 않고 Q1의 승리 거래 여러 건을 조기
컷오프해 해친다. **2단계 개입은 설계하지 않는다** — 억지로 만들면 이 프로젝트가 오늘 밤
반복적으로 적용해 온 "목표 구간을 보지 않고 캘리브레이션" 규율을 스스로 어기는 것이기
때문이다.

## 배경

`docs/model_contracts/odyssey3_eth_regime_guard_baseline_contract_20260814.md`(Odyssey3 베이스라인
계약) "다음 점검 대상 #1": h48qual에 성공적으로 적용·섀도우 배포된 레짐인지형 지속상승장 가드
(Odyssey2 #11)를 zig075에도 적용할 수 있는가. 배경 진단:

- `docs/experiments/eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md`
  (Odyssey2 #10): zig075의 2025-Q3 SHORT 손익은 원본/재라벨(h48qual만 재라벨, zig075는 항상
  원본) 무관하게 -0.517/-0.500으로 거의 동일 — h48qual과 달리 거래수 폭증(회전 가속)이 원인이
  아니라는 뜻이라 메커니즘을 직접 재확인해야 한다고 명시.
- `scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py`(Odyssey2 #11):
  h48qual의 가드는 이미 학습된 두 exit_head(원본/liveATR재라벨) 사이를 탐지기로 전환하는
  방식이었다. zig075는 원본 exit_head 모델 하나뿐이라 같은 "모델 전환" 메커니즘을 그대로 못
  쓴다.
- `docs/experiments/eth_omega461_zig075_exit_threshold_recalibration_20260814.md`(Odyssey2 #15):
  zig075의 exit_head 확신도가 VAL 구간에서는 0.90을 거의 안 넘는다는 것을 이미 발견 — 이게 Q3
  에서도 같은지 확인 필요.

## 방법

두 계층, 둘 다 순수 진단용(승격/모델선택 근거 아님 — 아래 "준수 확인" 참고):

1. **렛저 레벨 재집계**(신규 시뮬레이션 없음): 오늘 밤 `eth_omega461_multiwindow_confirmation_
   gate_20260814.py`의 G0b 단계가 이미 생성한 `tmp/causal_regen_20260516/
   eth_omega461_multiwindow_confirmation_gate_20260814/portfolio_ledger_2025q{1,2,3}_
   {baseline_both_original,asymmetric_tabm_liveatr}.csv`를 그대로 재사용해 zig075·SHORT 거래를
   청산사유·보유기간별로 분해.
2. **원시 확률 bar-by-bar 재계산**(신규 causal 계산, #15/기존 스크립트들이 공통으로 쓰는
   `train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one` 호출 패턴 재사용): 1에서
   찾은 각 거래(두 변형의 합집합, entry_signal_i로 중복제거, Q3 19건/Q1 10건/Q2 16건)의
   entry_i~exit_i 구간을 `replay_omega4_6_1_greedy_router_20260706.greedy_replay`와 완전히
   동일한 재구성(move/mfe/mae/pos_values, TP/SL이 먼저 걸리면 모델 조회를 건너뛰는 short-circuit
   포함)으로 bar마다 다시 밟으며 exit_head 확률을 직접 조회. 각 거래는 재구성된 (reason,
   trade_return)이 렛저에 기록된 값과 정확히 일치하는지 자체검증(전체 53건 전부 통과,
   `report.json`의 `all_self_checks_pass=true`).

신규 스크립트: `scripts/research_eth_omega461_zig075_sustained_uptrend_diagnosis_20260814.py`.
출력: `tmp/causal_regen_20260516/eth_omega461_zig075_sustained_uptrend_diagnosis_20260814/
report.json`. 재학습 없음, GPU 불필요(DEVICE=cpu), conda env `quant_ai`.

## 1단계 결과 — 메커니즘 진단

### 발견 1 — exit_head는 zig075 SHORT의 청산 사유로 어느 분기에서도 단 한 번도 등장하지 않는다

| 분기 | 변형 | n | 합계 | 청산사유 | exit_head 횟수 |
|---|---|---:|---:|---|---:|
| Q1 | baseline_both_original | 8 | +0.6172 | take_profit 6 · stop_loss 2 | **0** |
| Q1 | asymmetric_tabm_liveatr | 10 | +0.6531 | take_profit 7 · stop_loss 3 | **0** |
| Q2 | baseline_both_original | 12 | +0.2052 | stop_loss 7 · take_profit 5 | **0** |
| Q2 | asymmetric_tabm_liveatr | 16 | +0.1461 | stop_loss 10 · take_profit 6 | **0** |
| **Q3** | baseline_both_original | 16 | **-0.5173** | stop_loss 14 · take_profit 2 | **0** |
| **Q3** | asymmetric_tabm_liveatr | 18 | **-0.5000** | stop_loss 15 · take_profit 3 | **0** |

53건(6개 셀 합계) 전부 stop_loss 아니면 take_profit으로만 끝난다. 이건 **Q3에 국한된 현상이
아니라 zig075 exit_head(가중치·threshold=0.95 고정) 자체의 분기 불변 구조적 특성**이다 — h48qual
처럼 재라벨로 회전을 가속해도(liveatr 변형이 거래수를 늘리는 이유는 h48qual 회전이 빨라져
공유슬롯이 더 자주 풀리기 때문이지, zig075 자신의 청산 로직이 바뀌어서가 아니다) exit_head가
관여하는 비중은 0%에서 전혀 안 움직인다. 분기 간 실제로 달라지는 건 오직 **TP/SL 도달 비율**
뿐이다: take_profit 비중이 Q1 70%(7/10)·Q2 44%(7/16)·Q3 16%(3/18)로 지속 상승장에 가까울수록
급락한다 — SHORT 포지션이 하락장 베타에 기대는 컴포넌트인 이상 당연한 결과다.

### 발견 2 — MFE는 대부분 "한 번은 유리했다"이지만 규모가 분기마다 크게 다르다

| 분기 | 거래수(합집합) | mfe>0(한번이라도 유리) | stop_loss 거래 중 mfe=0(전혀 유리한 적 없음) | max exit_prob(전체) | 거래별 max_prob 평균 |
|---|---:|---:|---:|---:|---:|
| Q1 | 10 | 10/10 | 0/3 | 0.8148 | 0.6908 |
| Q2 | 16 | 16/16 | 0/10 | 0.8678 | 0.7047 |
| **Q3** | 19 | 17/19 | **2/16** | 0.8960 | **0.6150** |

Q3도 대부분(17/19)은 보유 중 한 번은 진입가보다 유리한 순간이 있었다 — "기회가 아예 없었다"는
아니다. 하지만 정도가 다르다: `stop_loss` 거래의 MFE/StopLoss거리 비율(둘 다 절대값 0.04로
분기 불문 고정, ATR 캡 포화)이 Q3에서 0.000(entry_signal_i=2514, 2558)부터 1.535
(entry_signal_i=16278)까지 넓게 퍼져 있고, 중앙값(0.0163)이 SL거리의 41%에 불과 — 즉 Q3의
손실거래 다수는 손절선 근처까지 거의 안 다가가 본 채(진짜로 유리했던 적이 거의 없이) 밀렸다.
또한 exit_head 확신도 자체는 Q3가 가장 낮다(거래별 max_prob 평균 0.615, Q1/Q2보다 낮음) — Q3의
확신도가 유독 높아서 "임계값만 낮추면 잡힌다"는 그림이 아니라 오히려 **반대**(Q3 확률이 구조적
으로 더 낮다)라는 뜻이다.

### 발견 3 — 반사실 threshold 분석: 검증된 범위는 무력, 그 아래는 원칙 위반이자 비청정 효과

동일 확률 궤적에 사후적으로 다양한 threshold를 적용했을 때 "조기 청산이 실제 청산 대비 얼마나
개선됐는가"(move 기준, +면 개선):

| threshold | Q1 발동/전체(평균개선) | Q2 발동/전체(평균개선) | Q3 발동/전체(평균개선) |
|---|---|---|---|
| 0.95(현행)·0.90 | 0/10 · 0/10 | 0/16 · 0/16 | 0/19 · 0/19 |
| 0.85 | 0/10 | 3/16(+0.0038) | 1/19(**-0.0310**) |
| 0.80(#15가 이미 검증) | 1/10(-0.0175) | 5/16(-0.0079) | 1/19(**-0.0310**) |
| 0.75 | 3/10(-0.0041) | 6/16(-0.0095) | 2/19(+0.0122) |
| 0.70 | 6/10(-0.0100) | 6/16(-0.0180) | 6/19(**+0.0382**) |
| 0.65 | 8/10(-0.0280) | 11/16(-0.0002) | 10/19(+0.0272) |
| 0.60 | 9/10(-0.0278) | 14/16(+0.0067) | 11/19(+0.0288) |

**#15가 이미 원칙적으로(VAL, Q3를 보지 않고) 검증한 범위(0.80~0.99)에서는 Q3에서 사실상 아무
일도 안 일어난다**(19건 중 0~1건). 얼핏 threshold를 0.70 근방까지 내리면 Q3 평균이 양전환되는
것처럼 보이지만, 거래 단위로 뜯어보면(threshold=0.70 예시):

- Q3 stop_loss 16건 중 **5건만 발동**(entry_signal_i 22·3890·4895·11085·11977, 전부 개선
  +0.042~+0.074)하고 **11건(69%)은 여전히 발동조차 안 해** 원래대로 손절까지 그대로 밀린다
  (확률이 보유기간 내내 0.70을 못 넘음 — 발견 2와 일치, 애초에 유리했던 적이 거의 없는
  거래들).
- take_profit 3건 중 1건(entry_signal_i=23887)은 발동해 **-0.0668**(승리거래를 일찍 끊어
  더 작은 이익으로 만듦)로 오히려 해롭다.
- **같은 메커니즘을 Q1에 적용하면**: take_profit 승리거래 4건(1591·5346·8772·22423)이 조기
  발동해 각각 -0.044·-0.028·-0.029·-0.053만큼 이익을 깎는다 — Q1 평균이 전 구간에서
  마이너스인 이유가 바로 이것이다.

즉 0.70 근방은 "Q3를 본 뒤에야 좋아 보인다"는 사후 선택일 뿐 아니라, 그 기준 자체가 Q3
손실거래의 다수는 건드리지도 못하면서 Q1의 승리거래는 여러 건 해친다 — **깨끗한 개선이 아니라
"일부만 돕고 다수는 무반응, 다른 분기 승자는 해치는" 잡음성 효과**다.

### 구조적 이유 — TP/SL 거리도 분기 불변 상수라 "규제 폭" 축소도 원칙 있는 손잡이가 없다

`stop_loss`/`take_profit`는 ATR 캡이 포화돼 세 분기 전부 정확히 0.0400/0.0750(평균=최소=최대)
로 완전히 동일 고정값이다 — 즉 "레짐에 따라 SL 폭이 원래 다르니 그 차이를 조건부로 되돌리자"는
식의, ATR 기반 자연스러운 레짐 신호도 존재하지 않는다. SL 거리를 조건부로 좁히는 개입을
만들려면 그 폭 자체를 처음부터 새로 발명해야 하는데, 이걸 Q3 손실을 줄이는 방향으로 고르면
위와 동일한 "목표 구간을 보고 고른 숫자" 문제에 다시 걸린다.

## 2단계 — 개입 설계하지 않음 (근거)

과제 지시대로, 억지로 설계를 만들지 않고 진단 결과 자체를 최종 결과로 보고한다:

1. **h48qual의 메커니즘(두 개의 이미 학습된 exit_head 사이를 탐지기로 전환)을 zig075에 그대로
   옮길 수 없다** — zig075는 원본 exit_head 모델이 하나뿐이고(오늘 밤 liveATR 재라벨 시도
   자체가 없었음), 두 후보 사이를 "전환"하는 방식은 애초에 threshold=0.95 고정을 유지한 채
   모델 가중치만 바꾸는 것이었기 때문에 새 숫자를 발명할 필요가 없었다. zig075는 이 구조가
   없어 어떤 개입이든 새 threshold 숫자(또는 새 SL 폭)를 만들어내야 한다.
2. **원칙적으로(Q3를 배제하고) 캘리브레이션 가능한 유일한 축인 exit_threshold는 이미 #15가
   VAL 전체 그리드(0.80~0.99)로 훑어 로버스트한 개선 0개로 닫았다** — 그리고 위 발견 3이
   보이듯 그 범위 안에서는 Q3에서도 사실상 무반응이라 "Q3에는 다르게 작동할 것"이라는 가정도
   성립하지 않는다.
3. **그 아래(0.60~0.75)로 내려가는 것은 이 진단 스크립트가 Q3의 실제 결과를 이미 관찰한
   뒤에야 "효과가 있어 보인다"고 알 수 있는 값이다** — 지속상승장 탐지기의 캘리브레이션이
   "Q3를 절대 보지 않고 Q1+Q2만으로 정한다"는 원칙을 지킨 것과 정확히 같은 이유로, exit_
   threshold 숫자도 Q3를 보고 고르면 안 된다. 이 프로젝트 전체가 오늘 밤 이 규율을 15번
   지켰는데 여기서만 어길 이유가 없다.
4. **설사 규율을 무시하고 그 값을 써도 결과가 깨끗하지 않다** — Q3 손실거래의 69%는 여전히
   무반응이고, 같은 메커니즘이 Q1의 승리거래 여러 건을 조기에 깎는다. 즉 원칙을 어겨도 얻는
   게 뒤섞인 잡음일 뿐, h48qual 가드처럼 깨끗하게 방향이 맞는 효과가 아니다.
5. **entry/direction/quality 개입은 과제 지시상 절대 금지**이므로 남은 유일한 통로(exit-side)
   가 막히면 유효한 개입 자체가 없다.

**결론: zig075의 2025-Q3 SHORT 약세는 exit_head의 "둔감함"으로 설명되지 않는다(exit_head는
분기 불문 애초에 거의 관여하지 않는 컴포넌트다). 다수 거래는 순수 진입타이밍 문제(유리했던 적이
거의 없음)이고, 소수는 명목상 "유리했다가 반전"하지만 그 소수를 원칙 있게 골라낼 수 있는
exit-side 손잡이가 없다(이미 검증된 범위는 무력, 그 아래는 사후선택이자 부작용 동반). 이
프로젝트의 "post-entry 개입만 허용" 제약 안에서는 유효한 설계가 없다.** h48qual의 지속상승장
가드와 달리, zig075의 Q3 약세는 이 baseline 구조에서는 미해결로 남는다.

## 3단계 — 해당 없음(개입 설계가 없어 구현·검증 생략)

## 준수 확인

`fresh_forward_bar_by_bar`: 계층 1(렛저 재집계)은 저장된 렛저를 입력으로 재사용 — 레포 정책상
diagnostic/historical-reproduction 용도로 명시적으로 허용되며, 이 문서는 이를 승격·모델선택·
성과 근거로 쓰지 않는다(순수 메커니즘 설명). 계층 2(원시 확률 bar-by-bar 재계산)는 저장된
렛저의 entry_i/exit_i/notional/leverage/margin_fraction만 좌표로 쓰고, 각 bar의 exit_head
확률 자체는 미래 정보 없이 그 bar 시점의 상태만으로 매번 새로 계산했다(`saved_parent_exit_
timestamps_used`에 해당하는 "저장된 exit 시점을 그대로 신뢰"가 아니라 매 bar 재검증) —
`report.json`의 `all_self_checks_pass=true`가 이 재구성이 렛저의 실제 기록과 정확히 일치함을
확인한다. `future_rows_used_for_entry=false`(진입 판단 자체를 다시 하지 않음, 이미 확정된
entry_signal_i를 렛저에서 그대로 읽음). 신규 학습 없음, GPU 불필요(DEVICE=cpu), conda env
`quant_ai`.

`git diff` 확인(0줄): `trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`,
`trading_bot_modules/runtime_config.py`. `.env`는 gitignore 대상이라 세션 중 미접촉을 별도
확인. 기존 스크립트/모듈(`eth_omega461_multiwindow_confirmation_gate_20260814.py`,
`research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py`,
`research_eth_omega461_exit_sweep_20260721.py`, `train_eval_omega4_2_risk_sidecar_20260622.py`,
`train_omega1_regime3_expert_direction_head_volpca_20260602.py`) 전부 임포트 후 읽기만, 수정
없음.

Seed-Diversity Ensemble Promotion Gate: 해당 없음(재학습 모델 없음, 여러 시드 평균/배깅 앙상블
승격 주장 없음). Omega Artifact Integrity Promotion Gate: 해당 없음(신규 parent 예측 아티팩트
없음, 기존 라이브 zig075 parent 아티팩트를 그대로 재사용).

## 산출물

- 신규 스크립트: `scripts/research_eth_omega461_zig075_sustained_uptrend_diagnosis_20260814.py`
  — 렛저 재집계(계층 1) + 원시 확률 bar-by-bar 재계산·자체검증(계층 2) + threshold 반사실
  분석 전부 포함.
- `report.json`: `tmp/causal_regen_20260516/eth_omega461_zig075_sustained_uptrend_diagnosis_
  20260814/report.json`(분기별 렛저 분해, 53건 거래 각각의 bar-by-bar 확률 궤적, 자체검증
  결과, headline 요약 전부 포함).
- 인용 문서: `docs/experiments/eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_
  20260814.md`(Odyssey2 #10, 문제 정의), `scripts/research_eth_omega461_regime_aware_exit_head_
  uptrend_guard_20260814.py`(Odyssey2 #11, h48qual 가드의 구조), `docs/experiments/
  eth_omega461_zig075_exit_threshold_recalibration_20260814.md`(Odyssey2 #15, 이미 검증되고
  닫힌 exit_threshold 범위), `docs/model_contracts/odyssey3_eth_regime_guard_baseline_contract_
  20260814.md`(Odyssey3 계약, "다음 점검 대상 #1").

## 다음 단계 / 미해결

- zig075의 exit_head가 SHORT 포지션에서 왜 구조적으로 거의 관여하지 않는지(확신도가 분기 불문
  낮게 캘리브레이션된 이유) 자체는 이번 진단 범위 밖이다 — #15가 이미 지적한 "zig075 exit_head
  확률 캘리브레이션 자체에 대한 별도 질문"과 동일선상.
  이 raw-probability walk가 재사용 가능한 도구이니, 향후 다른 각도(예: exit_head 재학습)를
  검토할 때 출발점으로 쓸 수 있다.
- Odyssey3 "다음 점검 대상 #1"은 이 문서로 **종결**한다(부정 결과) — zig075의 Q3 약세는
  Odyssey3 베이스라인에 미해결로 남는다(h48qual만 가드로 완화, zig075는 원본 그대로).
