# Odyssey4 exit_head — pos_tp/pos_sl 학습 피쳐와 실제 라벨 배리어 불일치 발견 (2026-08-17)

## 배경

[[eth_odyssey4_tpsl_floor_shrink_exit_head_engagement_20260817]] 진단(floor를 16배 줄이거나
사용자지정 3%/1.5%로 줄여도 exit_head 관여율이 거의 안 바뀜)에 대해 사용자가 "학습을
7%/4.5%로 모두 학습해서 그런거 아니야?"라고 질문 — exit_head가 학습 시점에 본 TP/SL
스케일과 추론(라이브/이 세션의 floor 스윕) 시점에 보는 TP/SL 스케일이 다르면, floor를
아무리 조정해도 exit_head 반응이 안 바뀌는 게 당연하다는 정확한 통찰. 사용자가 제시한
구체적 숫자(7%/4.5%)는 실제 상수와 다르지만(아래), **학습-추론 TP/SL 불일치라는 구조적
지적은 코드로 확인한 결과 정확했다** — 다만 예상보다 더 직접적인 형태의 버그였다.

Ilias 서브프로젝트가 이미 확정한 라벨설계 근본원인
([[eth_odyssey4_exit_head_passivity_root_cause_20260817]], 원본 라벨 양성 99.86%가 오라클
세그먼트 경계 기준)과는 **독립적인, 추가 원인**이다 — 이 세션(Odyssey4 메인라인, Ilias와
별개 진행)에서 새로 발견했다.

## 발견

현재 배포된 h48qual exit_head 번들(`NEW_H48QUAL_BUNDLE` =
`tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/
true_3head_tabm_bundle.pt`)의 학습 스크립트는
`scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py`다.

**1. 라벨(어느 bar가 진짜 TP/SL에 닿았는지)은 ATR-스케일 배리어로 결정된다** (365-368행):
```python
tp_move = min(max(min_tp, a * tp_mult), max_tp)   # min_tp=0.075, tp_mult=12, max_tp=0.22
sl_move = min(max(min_sl, a * sl_mult), max_sl)   # min_sl=0.040, sl_mult=6,  max_sl=0.12
```
이건 이 세션에서 계속 다뤄온 라이브/리플레이 floor 공식과 **완전히 동일**하다(`atr_window=192`
등, `eval_omega4_1_atr_safety_sltp_20260622.py`와 동일 파라미터).

**2. 그러나 exit_head 모델에 실제로 들어가는 입력 피쳐(`pos_tp`/`pos_sl`/`pos_dist_to_tp`/
`pos_dist_to_sl`)는 이 배리어가 아니라 별도의 낡은 고정 상수를 쓴다** (283-285행):
```python
take_profit = float(omega.BASE_TEMPLATE["take_profit"])   # 0.026 (2.6%) 고정
stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])       # 0.014 (1.4%) 고정
```
이 고정값이 그대로 `exit_head._position_feature_row(...)`(`train_eval_omega1_2_tabm_exit_head_
20260603.py:355-391`)에 전달되어:
```python
"pos_dist_to_tp": float(take_profit - unreal),   # = 0.026 - unreal, 실제 배리어와 무관
"pos_dist_to_sl": float(unreal + abs(stop_loss)), # = unreal + 0.014, 실제 배리어와 무관
"pos_tp": float(take_profit),                     # 항상 0.026
"pos_sl": float(stop_loss),                        # 항상 0.014
```
즉 **학습 시 exit_head는 "TP까지 남은 거리"로 항상 `0.026 - unreal`을 봤지, 그 거래를 실제로
종료시킨 진짜 ATR-스케일 배리어(0.075~0.22 사이 어딘가)까지의 거리는 한 번도 본 적이
없다.** 라벨(HOLD/EXIT)은 진짜 배리어 기준으로 만들어졌는데, 그 라벨을 설명해야 할 4개
피쳐(POS_COLS 13개 중)는 완전히 다른(더 작고 고정된) 숫자 체계로 계산됐다 — 지도학습
신호와 입력 피쳐 사이의 내적 불일치.

**3. 대조군: 원본(liveATR 이전) exit_head 학습은 이 문제가 없다.** `train_eval_omega1_2_
tabm_exit_head_20260603.py`가 쓰는 `train_fixed = omega._to_fixed_decisions(train_src,
oof=True)`(`train_eval_omega1_2_tabm_diffusion_risk_20260603.py:304-322`)는 배리어와 피쳐
둘 다 **같은** `BASE_TEMPLATE` 고정값(0.026/0.014)을 쓴다 — ATR 스케일링 자체가 없어서
내적으로는 일관됐다(고정 배리어 자체가 라이브와 다르다는 별개 문제는 있지만). **불일치는
liveATR 재라벨(현재 배포판) 스크립트에서만 새로 생겼다** — ATR 배리어를 도입하면서 피쳐
계산 쪽을 같이 안 바꾼 것으로 보인다.

**4. 추론 시점(라이브 + 이 세션의 floor 스윕)은 실제 배리어를 정확히 반영한다.**
`greedy_replay_entry_veto`의 `pos_values`(`research_eth_omega461_zig075_short_entry_veto_
sustained_uptrend_20260814.py:216-218`)는 `take_profit - move`/`move + abs(stop_loss)`를
그 거래의 **진짜** `dec["take_profit"]`/`["stop_loss"]`(=`_apply_atr_safety_sltp` 결과, 즉
이 세션에서 조정해온 floor 값)로 계산한다. 즉 **학습에서는 상수 0.026/0.014, 추론에서는
실제 0.0047~0.22 사이 값** — 최대 45배 이상 스케일이 다른 입력을 모델이 받는다.

## 결론

사용자의 지적이 정확했다 — floor를 아무리 조정해도 exit_head가 반응하지 않았던 이유
중 하나는 **exit_head가 pos_tp/pos_sl 계열 피쳐로 유의미한 "TP/SL까지 거리" 신호를
애초에 학습한 적이 없기 때문**이다. 학습 시 이 피쳐들은 실제 배리어와 무관한 상수였으므로,
모델이 이 피쳐들에 의미 있는 가중치를 배웠을 가능성 자체가 낮고, 추론 시 floor를 어떻게
바꿔도(이 세션의 스윕 전체) 모델 입장에서는 "한 번도 학습해보지 못한 낯선 스케일의 값"을
받는 것과 마찬가지다. 이는 Ilias가 확정한 라벨설계 근본원인(오라클 세그먼트 경계)과
**독립적이고 추가적인** 원인이며, 둘이 함께 exit_head의 관찰된 수동성을 설명한다.

수정 대상은 명확하다 — `research_eth_omega461_exit_head_liveatr_relabel_20260813.py:284-285`
에서 `take_profit`/`stop_loss`를 `omega.BASE_TEMPLATE`의 고정값이 아니라 그 row의 실제
`tp_move`/`sl_move`(365-368행에서 이미 계산됨)로 바꾸면 된다 — 새 라벨 설계 없이도 시도할
수 있는, 별도의 더 국소적인 수정이다. 단 이건 재학습이 필요한 변경이라(GPU 시간), 이
세션에서 즉시 적용하지 않았다 — 사용자 승인 필요.

## 다음 단계 제안 (초안, 아래 "수정+재학습 실행 결과"로 대체됨)

1. ~~이 버그를 고쳐서(라벨 로직은 그대로, feature 계산만 실제 배리어 참조하도록 수정)
   exit_head를 재학습~~ — 실행 완료, 결과는 아래 참조.
2. 그래도 안 되면 (1)이 문제의 전부가 아니라는 뜻이므로 라벨 자체(오라클 세그먼트 경계)
   재설계로 넘어감 — Ilias 근본원인 진단과 giveback 라벨 실패패턴을 참고.
3. h48cons(원본, ATR-스케일 없음) 변형에는 이 특정 버그가 없으므로 우선순위 낮음.

## 수정+재학습 실행 결과 (2026-08-17, 사용자 승인 "그래")

버그를 고친 뒤(위 커밋), `research_eth_omega461_exit_head_liveatr_relabel_20260813.py
--max-candidates 1500 --out-suffix full1500_featurefix`로 재학습 — 현재 배포본과 동일
후보수(1500)/시드(기본값)를 써서 배포 아티팩트(`_full1500`)는 전혀 건드리지 않고 별도
디렉토리(`_full1500_featurefix`)에 저장. 데이터셋 로그가 배포본과 rows=1234431,
positive_rate≈0.1990으로 소수점까지 일치 — 라벨 로직 자체는 안 건드렸다는 게 재확인됨.

재학습 완료 후, 이미 배포된 (버그 있는) h48qual 번들도 같은 `h48cons._evaluate_val` 방식으로
평가해서 3자 비교(원본/현재배포(버그)/제수정본)를 완성했다(재현:
`scripts/research_eth_odyssey4_exit_head_featurefix_deployed_comparison_20260817.py`).

### h48qual (VAL, 컴포넌트 단독 리플레이)

| 변형 | PnL | MDD | trades | exit_head 발동률 | avg_hold |
|---|---|---|---|---|---|
| 원본(재라벨 이전) | +5.45% | −11.62% | 29 | **0%**(0/29) | 670bar |
| **현재 배포(버그 있음)** | **+9.23%** | **−7.59%** | 63 | **82.5%**(52/63) | 211bar |
| 제 수정본(feature 버그 수정) | +9.21% | −8.69% | 65 | 83.1%(54/65) | 198bar |

### zig075 (VAL, 컴포넌트 단독 리플레이 — zig075는 라이브에서 liveATR 번들 자체를 안 쓰므로
"원본"이 곧 "현재 배포")

| 변형 | PnL | MDD | trades | exit_head 발동률 |
|---|---|---|---|---|
| **원본 = 현재 배포** | **+40.31%** | −13.07% | 29 | 0%(0/29) |
| 제 수정본(liveATR 신규후보) | +6.35% | −10.98% | 61 | 77.0%(47/61) |

### 핵심 발견 — 가장 중요한 건 버그 자체가 아니라 새로 드러난 모순

1. **h48qual: 수정 전후가 사실상 동일하다.** PnL 차이 0.02pp, exit_head 발동률 차이
   0.6pp — 버그는 실재하고 고친 것도 맞지만, **h48qual 단독으로는 이 버그가 실질적
   차이를 거의 안 만들었다.**
2. **더 중요한 발견: 현재 배포 중인(버그 있는 그대로의) h48qual 번들이 컴포넌트 단독
   평가에서는 이미 82.5%로 exit_head가 활발히 발동한다.** 그런데
   [[eth_odyssey4_tpsl_floor_shrink_exit_head_engagement_20260817]]의 **포트폴리오 전체**
   리플레이(zig075와 단일슬롯 공유+L4.5 듀레이션 게이트 포함)에서는 **같은 번들, 같은
   VAL 창인데 exit_head 발동률이 0%**였다. 즉 exit_head는 죽어있지 않다 — **포트폴리오
   레벨의 무언가(zig075 우선순위/슬롯공유, L4.5 게이트)가 발동 기회를 82%→0%까지
   눌러버리고 있다.** 이는 Ilias 근본원인 진단이 이미 발견하고 "완전히 분해하지
   못함"으로 남겨둔 정확히 같은 패턴이다([[eth_odyssey4_exit_head_passivity_root_cause_20260817]]
   의 "수치 불일치 63건→26건" 섹션 참조) — 이번에 이 라인(floor 스윕)에서도 재확인됐다.
3. **zig075에는 이 수정본이 명백히 손해다.** 원래 안 쓰던 liveATR exit head를 새로
   붙이면 VAL PnL이 +40.31%→+6.35%로 급락한다. exit_head가 활발해졌다는 사실 자체가
   성과 개선을 보장하지 않는다는 직접 증거.

### 결론 — 이 버그 수정은 "해결책"이 아니라 "정답 후보 하나를 지운 것"

사용자의 통찰(학습-추론 TP/SL 불일치)은 코드 레벨에서 **확인된 진짜 버그**였고 고쳤다.
그러나 결과는 h48qual의 관찰된 수동성을 설명하지 못한다 — **h48qual 단독으로는 exit_head가
버그 있는 채로도 이미 활발했기 때문이다.** 진짜 병목은 컴포넌트 단독 평가와 포트폴리오
전체 평가 사이의 이 극단적 격차(82%→0%)이며, 이건 Ilias가 이미 표시해둔 미해결 이슈와
같은 것으로 확인됐다. zig075에는 이 수정본을 신규 후보로 붙이면 오히려 손해라는 것도
확인됐다.

**배포 판단 아님** — 두 컴포넌트 다 현재 상태를 바꾸지 않는다: h48qual은 버그 있는 채로도
사실상 동일 성과를 내고 있어 교체 실익이 없고, zig075는 이 수정본이 명백히 더 나쁘다.
promotion gate·Red Team 미실행, live/섀도우 파일 무변경.

## ⚠️ 정정 (2026-08-17, 같은 날 후속 진단) — "82%→0%" 는 두 가지 오차의 합이었다

위 "82.5%→0%"라는 "극단적 격차" 주장은 **부정확했다**. 두 가지를 바로잡아야 한다:

1. **인용한 "0%"의 출처 자체가 버그였다.** [[eth_odyssey4_tpsl_floor_shrink_exit_head_engagement_20260817]]의 floor 스윕은 `_floor_cfg()`가 `bundle_override` 없이 h48qual을 로드해 **재라벨 이전 원본 번들**(exit_head 원래 0%)을 썼다 — 현재 배포 번들이 아니었다. 그 문서를 고쳐 재실행하니 현재 floor(7.5%/4.0%)에서 exit_head는 VAL **26.9%** 발동(0%가 아님).
2. **그 26.9%조차 h48qual 고유 발동률이 아니라 "h48qual+zig075 전체 거래" 기준으로 희석된 값이다.** 전용 진단 스크립트(`scripts/research_eth_odyssey4_exit_head_component_vs_portfolio_gap_diagnosis_20260817.py`, 처음부터 올바른 번들 사용)로 h48qual 거래만 걸러서 다시 보면:

| 조건 | h48qual 거래수 | exit_head 발동률 |
|---|---|---|
| 컴포넌트 단독(`_evaluate_val`) | 63 | 82.5% |
| h48qual 단독, `greedy_replay`, 게이트 없음 (Arm A) | 42 | 64.3% |
| h48qual 단독, `greedy_replay`, L4.5 게이트 (Arm B) | 30 | 56.7% |
| h48qual+zig075 포트폴리오, 게이트 없음, h48qual거래만 (Arm C) | 13 | 69.2% |
| **h48qual+zig075 포트폴리오, L4.5 게이트, h48qual거래만 (Arm D)** | **10** | **70.0%** |

**진짜 격차는 82.5%→70.0%, 완만한 감소이지 "0%로 눌린다"는 극단적 붕괴가 아니다.**
(참고로 Arm A(42건,64.3%)가 컴포넌트단독(63건,82.5%)과도 다른데, 이건 `greedy_replay`와
`sweep.replay_exit_variant`가 서로 다른 리플레이 함수라 거래수 자체가 다르게 나온다는
뜻 — 이 잔여 차이는 아직 완전히 분해 안 됨, 우선순위 낮음.)

zig075 슬롯공유·L4.5 게이트도 h48qual의 exit_head 발동률을 완전히 죽이지 않는다(양쪽
다 60~70%대 유지) — Ilias가 남겨둔 "미해결 격차"는 실재하지만 이번 진단이 시사하는 만큼
극단적이지 않았다. 이 정정으로 아래 "다음 단계"의 "최우선" 항목은 낮은 우선순위로 재조정한다.

## 다음 단계 제안 (2차 갱신)

1. ~~컴포넌트단독(82.5%) vs 포트폴리오전체(0%) 격차 분해~~ — 위 정정으로 82.5%→70.0%의
   완만한 격차로 판명, 긴급성 낮아짐.
2. h48qual단독(`greedy_replay`, Arm A, 64.3%/42건)과 컴포넌트단독평가(`replay_exit_variant`,
   82.5%/63건) 사이의 잔여 차이(리플레이 함수 자체의 구조 차이)는 남은 미해결 항목이나
   우선순위 낮음.
3. h48cons(원본, ATR-스케일 없음) 변형에는 feature-barrier 버그 없으므로 우선순위 낮음.
4. 실질적 결론은 바뀌지 않는다 — h48qual/zig075 둘 다 현재 상태 유지, 이 라인은
   당분간 소진(exhausted)으로 표시. exit_head를 정말 개선하려면 (앞서 floor-shrink
   정정 문서가 확인했듯) 라벨/feature 미세조정보다 근본적인 재설계가 필요해 보인다.

## 재현 (갱신)

- 버그 수정 커밋: `scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py`
  (429-430행)
- 재학습 실행: 위 커맨드, 산출물
  `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500_featurefix/`
- 배포본 비교평가: `scripts/research_eth_odyssey4_exit_head_featurefix_deployed_comparison_20260817.py`,
  산출물 `tmp/causal_regen_20260516/eth_odyssey4_exit_head_featurefix_deployed_comparison_20260817.json`

## 재현/근거 코드 위치

- 배리어 vs 피쳐 불일치: `scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py:
  283-285, 365-368, 429-430`
- 피쳐 계산 함수: `scripts/train_eval_omega1_2_tabm_exit_head_20260603.py:355-391`
- 대조군(원본, 불일치 없음): `scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py:
  304-322`(`_to_fixed_decisions`)
- 추론 시 실제 배리어 반영 확인: `scripts/research_eth_omega461_zig075_short_entry_veto_
  sustained_uptrend_20260814.py:216-218`(`pos_values`)
- 배포 번들 확인: `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_
  full1500/report.json`(`dataset.risk_template.take_profit=0.026`)
