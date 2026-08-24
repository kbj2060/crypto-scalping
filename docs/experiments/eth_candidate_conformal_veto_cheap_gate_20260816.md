# ETH conformal 하방-LCB 거부게이트 후보 — cheap_gate 실행 결과 (2026-08-16)

상태: **cheap_gate 완료. 결과가 표면적으로는 유망해 보이나, 저장소 전역 registry가 이미 닫아둔
축(quality_threshold 재튜닝, 21회+ 실패)과 정확히 겹친다는 걸 발견해 신뢰도를 낮춰 판단한다.**
VAL만 사용, OOS 미개봉.

## 배경

`docs/model_contracts/eth_candidate_conformal_downside_veto_contract_20260816.md`의 cheap_gate
절에 따라, conformal 회귀모델 2개를 학습하기 전에 **이미 계산되는 quality_score를 그냥 더 높은
임계값으로 거르면 비슷한 효과가 나는지** 먼저 확인했다. 스크립트: `scripts/
research_eth_candidate_conformal_veto_cheap_gate_20260816.py`. 각 컴포넌트의 `dec["quality_score"]`
(이미 계산된 값, 신규 모델 없음)를 새 임계값으로 재게이팅해서(기존 임계값보다 높은 값만 유효 —
단조 축소이므로 미래 데이터·재학습 없이 안전) Odyssey4의 causal replay 하네스를 그대로
재사용했다.

## G0 재현

`NOTIONAL_CAP`/threshold 전부 기본값일 때 41.13%/−21.70%/35(no_gate)·77.31%/−21.76%/26(with_gate)
— Odyssey4 VAL 기준값과 정확히 일치. 아래 스윕을 신뢰할 수 있다.

## 결과 1: h48qual 임계값 인상 (기준 0.50 → with_gate 77.31%/−21.76%/26)

| 임계값 | 원 신호 대비 컷 비율 | with_gate PnL | with_gate MDD | trades |
|---:|---:|---:|---:|---:|
| 0.55 | 68.4% | 59.18% | −24.43% | 24 |
| 0.60 | 93.8% | 89.50% | −23.55% | 22 |
| 0.65 | 100.0% | **98.96%** | **−19.73%** | 21 |
| 0.70 | 100.0% | 98.96% | −19.73% | 21 |

0.65/0.70에서 PnL·MDD가 **동시에** 기준선보다 개선된다(77.31%→98.96%, −21.76%→−19.73%) —
ETH 드로다운 거버너의 어떤 실험에서도 못 봤던 진짜 Pareto 개선처럼 보인다. **그런데 직접 확인한 결과, 이건
h48qual을 완전히 끈 것과 소수점까지 정확히 동일하다**(h48qual threshold=1.01로 완전 비활성화 →
동일한 98.96%/−19.73%/21). 즉 이건 "품질 하방을 세밀하게 거르는 필터"가 아니라 **"h48qual을
포트폴리오에서 빼라"는 이진 스위치**일 뿐이다 — h48qual의 direction_head 자체가 N≥5 시드로
무스킬 확정된 상태([[h48qual_label_mismatch_discovered]])이므로, 이 결과는 새로운 발견이
아니라 이미 알려진 사실의 재확인에 가깝다.

## 결과 2: zig075 임계값 인상 (기준 0.75 → with_gate 77.31%/−21.76%/26)

| 임계값 | 원 신호 대비 컷 비율 | with_gate PnL | with_gate MDD | trades |
|---:|---:|---:|---:|---:|
| 0.80 | 70.9% | 59.99% | **−15.34%** | 25 |
| 0.85 | 96.0% | 21.52% | −17.63% | 30 |
| 0.90 | 99.9% | 57.61% | −13.90% | 30 |

0.90은 마찬가지로 **zig075 완전 비활성화와 소수점까지 동일**(zig075 threshold=1.01 →
57.61%/−13.90%/30) — 역시 이진 스위치 아티팩트. 0.85는 0.80/0.90 사이에서 오히려 더 나쁜
비단조(non-monotonic) 결과라 작은 표본의 노이즈로 보인다.

**0.80만 진짜 중간 상태다**(25건, on/off 양쪽과 다른 수치) — PnL 17.32pp를 내주고 MDD를
6.42pp 개선(비율 ≈2.7pp/1MDDpp). ETH 드로다운 거버너가 찾은 어떤 트레이드오프보다 훨씬 낫다(cheap_gate의
NOTIONAL_CAP은 7~16pp/1MDDpp, hard_loss_stop 최선도 7.5pp/1MDDpp였음).

## 왜 이 결과를 그대로 믿으면 안 되는가 — 저장소 registry와의 충돌

`docs/model_contracts/research_line_registry.json`의 `global_exit_constant_tuning` 항목:

> scope: "Exit logic, TP/SL width, **quality threshold**, time exit, and second-slot tuning on an
> unchanged signal"
> reason: "**21+ exit rounds and related sweeps did not survive validation/OOS.**"
> retest_guidance: "A new entry signal or an independently reproduced execution discrepancy makes
> a full-stack retest informative."

**이 cheap_gate가 한 일이 정확히 이 항목이다** — h48qual/zig075라는 "변하지 않은 신호" 위에서
quality_threshold만 재튜닝했다. 이 저장소는 이미 21번 넘게 같은 종류의 튜닝을 시도했고 전부
VAL/OOS 생존에 실패했다는 기록을 갖고 있다. zig075@0.80의 "진짜 중간 상태" 결과가 겉보기엔
가장 그럴듯해 보이지만, **VAL 단일 창짜리 숫자 하나가 21전 0승 기록을 뒤집을 근거가 되지
못한다.** h48qual@0.65/zig075@0.90은 애초에 "튜닝"이 아니라 "컴포넌트 하나를 꺼라"는 이진
결정이라 이 항목의 정신에서 살짝 벗어나지만(재튜닝이라기보다 포트폴리오 구성 문제), 여전히
VAL 단일 창 근거뿐이라 별도 검증 없이 신뢰할 수 없다.

## 결론 — cheap_gate의 목적에 대한 답

cheap_gate의 원래 질문은 "무료 필터로 conformal 모델과 비슷한 효과를 이미 얻는가"였다. 답은
**"표면적으로는 그렇게 보이지만, 그 표면적 결과 자체가 이 저장소에서 이미 21번 실패한 것과
같은 종류의 조작이라 신뢰할 수 없다"**이다. 이건 cheap_gate가 실패한 것도 성공한 것도
아니다 — cheap_gate가 원래 답해야 할 질문("무료 필터로 충분한가")에 대해 "충분해 보이는
숫자가 나왔지만 그 숫자를 믿을 근거가 없다"는 애매한 답을 준 것이다.

이게 conformal 모델(ETH conformal veto 본체) 착수 여부에 주는 함의:

1. **quality_threshold를 그냥 올리는 방식으로는 이 문제를 풀 수 없다는 것이 이미 21번
   확인됐다** — cheap_gate가 그걸 22번째로 재확인했을 뿐이다.
2. ETH conformal veto의 conformal 모델은 quality_score 하나가 아니라 **더 풍부한 causal 피처 +
   episode별 시뮬레이션된 미래 경로(net return/adverse)**를 쓴다 — 단순 threshold 재튜닝과는
   질적으로 다른 정보를 쓴다는 점에서 registry의 "새 진입 신호나 독립적으로 재현된 실행
   불일치가 있어야 재시도 가치가 있다"는 재개조건과 완전히 일치하지는 않지만, "같은 신호를
   다른 문턱값으로 자르는" 문제 유형과는 다르다.
3. 그럼에도 이 cheap_gate가 준 가장 중요한 교훈은 **"h48qual을 포트폴리오에서 빼면 VAL이
   좋아진다"는 사실 자체가 이미 알려진 무스킬 판정의 재확인**이라는 것 — conformal 모델이
   결국 "h48qual quality가 낮으면 거부"만 배우고 끝난다면 이것 역시 같은 아티팩트를 학습 모델
   형태로 재포장한 것에 불과할 위험이 있다. **구현 시 conformal 모델의 예측력이 단순
   quality_score 재현("h48qual 거의 항상 거부, zig075는 선택적")으로 붕괴하지 않는지 반드시
   진단해야 한다**(예: quality_score를 피처에서 제외하고도 유사한 veto 패턴이 나오는지, 또는
   컴포넌트별 veto율이 h48qual≈100%/zig075≈0%로 수렴하지 않는지 확인).

## 아티팩트

- 스크립트: `scripts/research_eth_candidate_conformal_veto_cheap_gate_20260816.py`
- 리포트: `tmp/causal_regen_20260516/eth_candidate_conformal_veto_cheap_gate_20260816/report.json`
