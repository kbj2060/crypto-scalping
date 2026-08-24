# ETH 공유 트렁크 레짐전문가(C3) — N≥5시드 본실험 결과, CLOSED (2026-08-17)

관련 상위 문서: `docs/model_contracts/eth_candidate_shared_trunk_regime_experts_contract_20260816.md`
(계약서, 설계/구현/로컬 sanity check 내역), `docs/experiments/eth_odyssey4_layer_and_parameter_improvement_proposal_20260816.md`
(§C3, 이 후보가 나온 상위 제안 문서 — A1/C1/C2와 같은 계보의 마지막 항목).

## 실행 경위

- 서버 GPU가 `eth_nhits_moderntcn_direction_quality`(N-HiTS/ModernTCN 재검증 작업)에 장시간
  점유돼 있어, `scripts/ops/wait_and_launch_c3_shared_trunk_20260816.sh` 워처를 서버에
  nohup+setsid로 걸어 GPU가 풀리는 즉시 본실험이 자동 시작되도록 함(2026-08-16).
- 2026-08-17, 사용자 판단으로 `eth_nhits_moderntcn_direction_quality`를 직접 중지+삭제 —
  워처가 즉시 감지해 같은 pid로 `exec`, 본실험 시작(2026-08-17T04:37:11Z UTC).
- 실행 커맨드: `python3 scripts/research_eth_odyssey4_shared_trunk_regime_experts_20260816.py
  --epochs 28 --n-seeds 5 --mode both --device cuda --feature-pipeline true`.
- 5개 진짜 무작위 시드(`secrets.randbelow`, 고정간격 아님): `[284220531, 396212443, 418156144,
  662991329, 900276690]`.
- 진짜 라이브 102(+13pos)=115피처 파이프라인(`scripts/eth_odyssey4_true_feature_pipeline_20260816.py`)
  사용, n_train=78,568.
- 완료: `report written to .../eth_odyssey4_shared_trunk_regime_experts_20260816/report.json`
  (서버 mtime 2026-08-17 13:49 KST). 크래시 없음, 전체 5시드×2모드(baseline/shared)×2split(val/oos)
  전부 정상 완료.

## 결과 1 — direction_balanced_accuracy (내부 val split, 계약서 §비교지표 1)

shared_trunk − baseline_independent_trunk, 전문가별 5시드:

| expert | seed1(284220531) | seed2(396212443) | seed3(418156144) | seed4(662991329) | seed5(900276690) | 평균 Δ | 표준편차 | 개선 시드 | 부호일관성 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bull | +0.0114 | +0.0060 | −0.0202 | +0.0006 | +0.0029 | **+0.0001** | **0.0108** | 4/5 | **False** |
| bear | +0.0037 | +0.0059 | +0.0173 | −0.0013 | +0.0009 | **+0.0053** | **0.0065** | 4/5 | **False** |
| chop | −0.0106 | +0.0159 | +0.0054 | −0.0035 | +0.0185 | **+0.0051** | **0.0111** | 3/5 | **False** |

이 세션이 A1/C2에서 반복 적용한 기준(표준편차가 평균보다 훨씬 작아야 진짜 신호)으로 보면
셋 다 명백히 실패한다 — bull은 std(0.0108)가 mean(0.0001)의 **78배**, bear는 1.22배, chop은
2.16배. 세 전문가 전부 부호 일관성 없음(각 5개 중 1~2개는 반대 부호).

## 결과 2 — PnL/MDD (val_raw/oos_raw, `_metrics_with_shared_exit`, exit threshold=0.60)

| seed | VAL baseline pnl/mdd | VAL shared pnl/mdd | OOS baseline pnl/mdd | OOS shared pnl/mdd |
|---:|---:|---:|---:|---:|
| 284220531 | −75.77% / −75.77% | −71.73% / −71.76% | −61.83% / −61.84% | −46.08% / −46.11% |
| 396212443 | −74.51% / −74.51% | −77.67% / −77.68% | −57.34% / −57.55% | −63.21% / −63.21% |
| 418156144 | −75.28% / −75.30% | −75.72% / −75.72% | −57.29% / −57.31% | −54.73% / −54.74% |
| 662991329 | −75.58% / −75.60% | −83.50% / −83.49% | −55.99% / −56.00% | −66.94% / −66.97% |
| 900276690 | −81.73% / −81.73% | −85.35% / −85.35% | −63.74% / −63.75% | −65.20% / −65.21% |
| **평균** | **−76.57% / −76.58%** | **−78.79% / −78.80%** | **−59.24% / −59.29%** | **−59.23% / −59.25%** |

- VAL: shared가 baseline보다 나은 시드는 **1/5**뿐 — 평균도 shared가 −2.2pp 더 나쁨.
- OOS: shared가 baseline보다 나은 시드는 **2/5** — 평균은 거의 동일(−59.24% vs −59.23%, 0.01pp
  차이, 사실상 무의미).

**절대 PnL 규모(−46%~−85%)에 대한 주의**: 이 수치를 "캐노니컬 모델이 실전에서 이 정도로
손실을 낸다"로 오독하면 안 된다. 계약서의 로컬 sanity check(2-epoch, under-trained)에서도
이미 같은 규모(−51%~−77%)가 나왔었고(계약서 §로컬 sanity check 표), 이건 이 스크립트가
h48qual/zig075가 라이브에서 쓰는 quality_threshold 정밀 탐색·zig075 레짐 베토 등 추가 게이팅
레이어 없이, 이번 세션에서 새로 처음부터 학습한 direction/quality head를 고정 threshold(0.45
진입/0.60 청산)로만 거래하기 때문이다 — `scripts/diagnose_eth_h48qual_ungated_direction_vs_
always_short_20260812.py`가 다른 모델 계보(h48qual)에서 이미 확인한 "게이트 없는 원시
direction_head는 always-short/always-long을 못 이긴다"는 패턴과 정합적이다. baseline과
shared_trunk **둘 다 똑같은 조건**(동일 threshold, 동일 프레시 학습 프로토콜)에서 백테스트됐으므로
상대 비교(이 실험의 목적)는 유효하다 — 다만 절대 수치를 승격/배포 판단에 재사용하면 안 된다.

## 판정 — **CLOSED, 반영 안 함**

계약서 Red Team Gates의 두 판정 기준(direction_balanced_accuracy 시드 전반 부호일관성,
PnL/MDD 방향일치) 모두 실패:

1. **direction_balanced_accuracy**: 3개 전문가 전부 표준편차가 평균과 같거나 훨씬 큼(1.22배~78배),
   부호 일관성 없음 — A1(GCE, std가 mean의 3배 이상)·C2 재확인(std가 mean의 4배 초과)과 같은 급의
   명확한 노이즈.
2. **PnL/MDD**: VAL에서는 shared가 평균적으로 더 나쁨(1/5 시드만 개선), OOS에서는 평균 차이가
   사실상 0(2/5 시드만 개선) — 분류지표의 방향과도 일치하지 않고, 그 자체로도 일관된 개선 신호가
   없음.

B1이 실측한 데이터효율 논거(전문가별 유효표본수 28~43%, 트렁크는 3배)는 이론적으로 여전히
타당하지만, 실제 학습 결과에서는 그 효율 이득이 direction_balanced_accuracy나 PnL/MDD 어느
쪽으로도 측정 가능한 개선으로 이어지지 않았다. 파라미터 수는 설계대로 34%로 줄었지만
(baseline_n_params_total=311,976 vs shared_n_params_total=107,080, 5시드 전부 동일), 이 절감이
비용 없는 개선이라는 근거도, 손해라는 근거도 이번 결과로는 나오지 않는다 — 그냥 무관하다(둘 다
노이즈 범위 안).

**결론**: 레짐별 완전 독립 트렁크(현재 캐노니컬 구조)를 공유 트렁크로 바꾸지 않는다. 이걸로
`eth_odyssey4_layer_and_parameter_improvement_proposal_20260816.md`의 A(A1)/C(C1/C2/C3) 전
항목이 CLOSED로 마감된다 — 이 세션이 제안했던 레이어/파라미터 개선 축 중 캐노니컬 스크립트에
실제로 반영되는 것은 하나도 없다.

## fresh-forward 규칙 준수

`fresh_forward_bar_by_bar=true`(val_raw/oos_raw 백테스트는 `_metrics_with_shared_exit`의 기존
bar-by-bar causal walk 그대로 재사용, 스크립트 자체 docstring에 명시), classifier 학습(내부
85/15 val split, direction_balanced_accuracy 비교)은 `fresh_forward_bar_by_bar=n/a`(A1/C1/C2와
동일 방법론), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`.

## 산출물

- 본실험 스크립트: `scripts/research_eth_odyssey4_shared_trunk_regime_experts_20260816.py`.
- 결과 원장: `tmp/causal_regen_20260516/eth_odyssey4_shared_trunk_regime_experts_20260816/report.json`.
- GPU 대기 워처: `scripts/ops/wait_and_launch_c3_shared_trunk_20260816.sh`(서버, job명
  `eth_odyssey4_shared_trunk`) — 임무 완료, 서버 job은 STOPPED 상태로 남아있음(재실행 시
  `handoff.sh launch`로 job명 재사용하면 자동 덮어써짐, 별도 삭제 불필요).
- 캐노니컬 스크립트(`scripts/train_eval_omega1_2_tabm_3head_20260603.py`)는 **미변경**.

## registry 반영

`docs/model_contracts/research_line_registry.json`에 `eth_candidate_shared_trunk_regime_experts`
항목으로 등록 예정 — scope: `ThreeHeadTabM`의 레짐별(bull/bear/chop) 완전독립 트렁크를
공유트렁크+레짐별헤드로 교체(C3), reason: N=5 진짜무작위시드 본실험에서 direction_balanced_accuracy
(3개 전문가 전부 std/mean 비율 1.22~78배, 부호불일치)와 VAL/OOS PnL/MDD(평균적으로 무개선 또는
소폭 악화, 시드별로도 개선 소수) 양쪽 다 신뢰 가능한 개선 신호 없음, retest_guidance: B1의
데이터효율 논거 자체는 이 결과로 반박되지 않았으나(측정 못 함, 노이즈에 묻힘), 재시도하려면
학습 신뢰성(조기종료 타이밍 안정화 등, `feedback_modern_dl_training_checklist` 축)이 먼저
해결된 뒤에나 이 아키텍처 변화의 순수 효과를 분리해 볼 수 있음 — 지금 재시도해도 같은 노이즈
수준에 묻힐 가능성이 높음.
