# ETH 캐시 슬리브 EV-HGB 후보 — 계약 (2026-08-16)

상태: **CLOSED — Stage 1(실제 HGB 학습 + purged CV + 라벨 순열 대조군) 사전등록 기준 FAIL. N≥5 시드
재현/fresh-forward walk-forward로 진행하지 않음.**

이 후보는 **공식 Odyssey 계보(Odyssey1~4)에 속하지 않는다** — 확정된 성과가 있을 때만 버전 번호를
올린다는 원칙에 따라 "Odyssey5"로 명명하지 않는다.

## 범위

- 목적: BTC에서 라이브 검증됐지만 현재 배선되지 않은(dead code) "캐시 슬리브" 메커니즘(PRIMARY가
  CASH/무포지션 상태일 때만 작동하는 별도 EV 회귀 폴백 모델)을 ETH의 PRIMARY(h48qual/zig075 3-Head
  TabM + Odyssey4 잠금 베이스라인)에 이식할 가치가 있는지, **학습 없이** 먼저 확인한다.
- 이 문서는 결과 + 간단한 근거만 담는다. 전체 방법론·수치·경고는
  `docs/experiments/eth_candidate_cash_sleeve_ev_hgb_20260816.md`를 참고.
- 리소스 레지스트리: `docs/model_contracts/eth_candidate_cash_sleeve_ev_hgb_data_resources_
  20260816.md`.
- 구현: `scripts/research_eth_candidate_cash_sleeve_ev_hgb_cheap_gate_20260816.py`.

## cheap_gate 결과 요약 — 오라클/사후확인(hindsight) 측정, 학습된 모델 아님

PRIMARY의 CASH 상태는 Odyssey4 잠금 베이스라인(h48qual regime-aware exit guard + zig075 지속상승장
SHORT 진입거부)의 실제 causal greedy-replay 렛저에서 그대로 읽었다(G0 정합성 체크, val/oos_q1 둘 다
no_gate pnl/mdd/trades 완전 일치로 확인).

| 항목 | VAL(2025-10~12) | OOS-Q1(2026-01~03) |
|---|---:|---:|
| PRIMARY CASH bar 비율 | 26.66% | 27.02% |
| CASH bar 중 `max(long_net,short_net)>ev_min` 비율 | 45.78% | 56.21% |
| 조건 충족 시 평균 net edge | +0.843% | +0.899% |
| 방향성 베이스라인(max(always_long,always_short)) | −1136.88%(가산) | −1013.36%(가산) |

CASH 시간은 사소하지 않고(~27%), 사전 등록된 close-negative 기준(비율<5~10% 또는 평균 엣지 작음)에
못 미치지 않는다 — cheap_gate 자체 기준으로는 종료 근거가 없다.

**그러나 실제 학습에는 착수하지 않았다.** 이유(상세는 실험 문서 참고):

1. 오라클 상한(가산 +2696%/+3487%)은 수천 개의 시간적으로 겹치는 가상 트레이드를 단순 합산한 것으로
   "달성 불가능"임을 그대로 읽어야 하고, 조건 충족률이 높은 것 자체가 "TP:SL≈1.86:1 배리어에서
   사후에 이긴 방향만 고른다"는 오라클의 본질적 특성을 반영하는 것에 가깝다.
2. 이 저장소의 ETH 방향 예측 축은 이미 여러 차례(h48qual/zig075 direction head, evidence-signal
   22종, 오실레이터 confluence, AMT/VSA/iFVG 4종 등) no-skill로 CLOSED됐다 — 같은 시장·같은
   타임프레임에서의 이 강한 사전 정보가 EV-HGB가 이 오라클 상한의 유의미한 부분을 실제로 인과적으로
   포착할 수 있다는 주장에 상당한 회의를 갖게 한다.

cheap_gate 자체 단계에서는 `research_line_registry.json`에 항목을 추가하지 않았다 — 그 시점엔
결정적 부정 결과가 아니라 "헤드룸 있음, 사람 판단 대기" 상태였기 때문이다. (아래 "Stage 1 실제 학습
결과" 절에서 실제 학습이 결정적 부정으로 귀결되어 registry 항목이 추가됐다.)

## Red Team Gates

- [x] cheap_gate(오라클 상한) 먼저 실행 — **완료, 결정적 부정 아님(헤드룸 존재)**.
- [x] G0 정합성 체크(Odyssey4 잠금 베이스라인 재현) — **완료, val/oos_q1 둘 다 정확히 일치**
  (Stage 1 학습 스크립트에서도 val 재확인: 41.13%/−21.70%/35 트레이드, 정확히 일치).
- [x] 학습 전 IC(순위상관) 체크(기존 causal 피처 vs 오라클 타겟, 가격오염 체크 포함) — **완료,
  결정적 부정 아님(12/50 조합이 VAL/OOS-Q1 부호일치+노이즈초과+가격오염<0.5 통과, 상세는 아래 절)**.
- [x] 실제 EV-HGB 학습(purged/embargo CV, causal 피처) — **완료 — Stage 1, FAIL**. purged 5-fold
  CV(purge gap 실측 ≥192bar, `core/event_label_engine.purged_kfold_splits` 재사용), 74피처
  (market 25 + primary-trace 43 + cash-state-history 6), BTC 프로덕션 HGB 하이퍼파라미터 이식,
  단일 시드. `long_model` pooled OOF IC=−0.182(순열 null 대비 z=−3.35, null보다 나쁨),
  `short_model` IC=+0.056(z=+2.13이나 fold별 부호 뒤집힘 — 불안정), 결정 관련 지표(ev_min 필터
  적용 시 실현 edge)는 null과 구분 안 됨(z=−0.18, 부호도 음수). 상세는 실험 문서 "Stage 1" 절.
- [x] 라벨 순열(permutation) 대조군 — **완료 — Stage 1에 포함(fold당 30회, pooled null 비교)**.
- [ ] N≥5 진짜 랜덤 시드 OOS 부호 일치(Seed-Diversity Ensemble Promotion Gate) — **진행 안 함**
  (Stage 1 사전등록 기준 FAIL로 게이트되어 이 단계에 도달하지 않음).
- [ ] 실제 causal walk-forward VAL→OOS 단일터치 확인(Fresh-Forward Rule) — **진행 안 함**
  (동일 사유; OOS-Q1은 Stage 1에서 로드조차 되지 않음 — 온전히 보존됨).

## 학습 전 IC(순위상관) 체크 결과 — 2026-08-16 추가

실험 문서 "권장 다음 단계" 1번(학습 전 더 싼 확인)을 실행했다. HGB 학습 없이, h48qual/zig075가 이미
쓰는 base+wide24 피처 패널에서 모멘텀/추세·변동성·레짐-라우터 후보 25개를 골라 오라클 타겟
`max_net`/`net_diff`와의 Spearman IC(가격추세 오염 체크 + bootstrap/셔플 노이즈 바닥 포함)를 VAL/
OOS-Q1 양쪽에서 쟀다. **결과: 결정적 음성이 아니다** — 50개 (피처×타겟) 조합 중 12개가 양 윈도우
부호 일치 + 노이즈 초과 + 가격오염 IC<0.5를 동시에 통과했다. 가장 유망한 것은 `net_diff`(방향
선호도) 타겟에 대한 `atr_pct_rank_288`(VAL +0.082/OOS +0.121), `compression_score`(-0.057/-0.120),
`volatility_z`(+0.056/+0.134) — 가격오염 IC가 낮고(<0.11) OOS에서 더 강해진다. `max_net` 쪽 강한
신호(변동성 피처 0.14~0.16)는 실재하지만 "둘 중 하나가 이길 확률"이라는 오라클 자체의 구조적 성질을
반영할 가능성이 높고 가격오염 IC도 가장 크다(0.19~0.32)는 경고가 붙는다. IC 크기 자체(0.03~0.16)는
이 저장소가 다른 곳에서 "종종 cost-gate를 못 버틴다"고 본 0.02~0.05 구간보다는 위이지만 "강한 신호"로
봤던 0.3+ 구간에는 못 미치는 애매한 중간 지대다. 전체 방법론·표·경고는 실험 문서의 "후속 조사" 절
참고. 이 IC 체크 단계 자체는 결정적 부정이 아니었다(당시 registry 항목 미추가는 그 시점 기준으로는
맞는 판단이었다) — 아래 "Stage 1 실제 학습 결과" 절에서 실제 학습이 이 애매함을 최종적으로
해소한다.

## Stage 1 실제 학습 결과 — 2026-08-16 추가 (CLOSED)

사용자가 실제 학습 착수를 승인해 진행했다. VAL 윈도우(2025-10-01~12-31)의 CASH bar 6,986개에서
`long_model`/`short_model`(BTC 프로덕션 HGB 하이퍼파라미터 이식, 74피처: market 25 +
h48qual/zig075 primary-trace 43 + cash-state-history 6)을 purged 5-fold CV(purge gap 실측
≥192bar)로 학습하고, fold당 30회 라벨 순열(permutation) 대조군과 비교했다. 단일 시드
(`SEED=20260816`). OOS-Q1은 로드조차 하지 않았다(fresh-forward walk-forward를 위해 온전히 보존).

| 지표 | 실제 | 순열 null 평균±std | z-score | 판정 |
|---|---:|---:|---:|---|
| `long_model` OOF Spearman IC | −0.182 | −0.069 ± 0.034 | **−3.35** | null보다 나쁨(FAIL) |
| `short_model` OOF Spearman IC | +0.056 | −0.013 ± 0.032 | **+2.13** | 개별 통과, but fold별 부호 뒤집힘(불안정) |
| 결정 관련: 선택 bar 평균 실현 edge − 무조건 평균 | −0.099pp | −0.052 ± 0.263pp | **−0.18** | null과 구분 안 됨(FAIL) |

**사전 등록 결합 기준(셋 다 z≥2.0) 결과: FAIL.** `short_model`의 IC만 개별 문턱을 넘었으나 fold별로
부호가 뒤집힐 만큼 불안정했고(fold 0~2 음수, fold 3~4 양수), 실제 트레이딩 판단에 대응하는 결정
관련 지표는 null과 통계적으로 구분되지 않았다(부호도 음수). `long_model`은 라벨을 순열해도 얻지
못할 만큼 나쁜, 순수 노이즈보다도 못한 결과였다. 전체 방법론·fold별 표·해석은 실험 문서의
"Stage 1" 절 참고.

`docs/model_contracts/research_line_registry.json`에 `eth_candidate_cash_sleeve_ev_hgb_stage1_
train_20260816` 항목을 추가했다 — 사전 등록한 합격 기준을 실제 학습이 명시적으로 실패한 결정적
결과이기 때문이다.

## 미해결 이슈

1. ~~실제 causal 학습 착수 여부~~ → **해소됨**: Stage 1이 사전등록 기준 FAIL로 이 후보를 CLOSED
   처리했다.
2. VAL 윈도우 경계가 CLAUDE.md 일반 기본값(2025-09-01 시작)과 한 달 다르다(2025-10-01 시작) — 이유는
   parent 아티팩트의 train/validation split 경계 때문(실험 문서 참고). 이 후보 자체는 종료됐지만,
   같은 parent 아티팩트를 쓰는 다른 향후 서브프로젝트가 이 윈도우 정의를 재사용할 경우 그대로
   유지할 것.

## 다음 단계

**없음 — 이 후보는 CLOSED다.** Stage 1이 사전등록한 결합 기준(순위 품질 IC + 결정 관련 지표, 셋 다
순열 null 대비 z≥2.0)을 명백히 충족하지 못했으므로 N≥5 시드 재현, fresh-forward walk-forward 둘 다
진행하지 않는다. 재시도한다면 이 cheap_gate/IC-check에서 파생된 것과 질적으로 다른 피처 집합·라벨
설계·결정 타겟이 필요하다(`research_line_registry.json`의 해당 항목 `retest_guidance` 참고).
