# BTC Kappa1 — Invariant Composite Policy (설계, 2026-08-07)

**한 줄 요약:** 새 예측기를 학습하지 않는다. 이미 실재가 검증된 4개 신호(P3 exit,
이벤트 게이트, 1m 역추세 flow, 트레일링 엔벨로프)를 **동결된 프리미티브**로 두고,
수십 개 파라미터짜리 부호-제약 합성층 하나만 학습하되, 목적함수를 pooled PnL이 아닌
**최악-분기(worst-quarter) PnL**로 잡고, 선택은 Sigma6에서 검증된 LOWO로만 한다.
진입 비용 문제는 모델이 아니라 **maker-first 실행 프리미티브**로 공략한다.

---

## 1. 왜 "더 복잡한 모델"이 아니라 이 구조인가

이 저장소의 BTC 증거는 일관된다:

| 증거 | 함의 |
|---|---|
| 헤드 교체 무의미 (linear/TabM/tree 전부 64-67%), 용량 스윕 1-16 무효, JEPA/transformer 무효 | 표현·용량 축은 죽었다. 같은 supervised 매핑에 정교함을 더 붓는 것은 5번째가 아니라 25번째 실패가 된다 |
| VAL/OOS 탈상관 5회 재현 (오늘 1h에서도), effective_n ~4천 | 병은 "무엇을 배우나"가 아니라 "어느 윈도우에서 이기나"다. 파라미터 예산은 수십 개 이하여야 하고, 목적함수 자체가 윈도우-불변성을 강제해야 한다 |
| Sigma6-LOWO: held-out 4/5 통과 — 저장소 유일의 생존 선택법 | 불변성 방향으로의 정교화는 이 저장소에서 실증적 근거가 있는 유일한 정교화다 |
| 실재-미수익 신호 4개가 서로 **직교하는 이유**로 실패 | 정밀도(P3) × 실행(이벤트) × 비용(micro) × 리스크(트레일링) — 서로가 서로의 사인을 덮는 조합은 미검증 영역이다 |

**따라서 Kappa1의 정교함은 세 곳에만 투자한다: 합성 구조, 불변성 목적함수, 실행 계층.**
표현 학습에는 0을 투자한다.

## 2. 아키텍처

```
[Layer 0 — 동결 프리미티브 (재학습 없음, 검증된 것만)]
  s1: P3 adverse-pivot exit score        (exit 전용; t=2.2~8.2 검증)
  s2: 이벤트 게이트 stage-1 score        (진입 허가 윈도우; 8/8 롤링 검증)
  s3: 1m 역추세 flow score               (진입 방향/타이밍; 5회 검증)
  s4: GMM/IsoForest anomaly score        (실재-데이터부족 → 단독 게이트 금지, 감산 입력만)
  s5: HMM regime posterior               (라이브 Regime3 그대로)

[Layer 1 — 합성층 (유일한 학습 대상, 파라미터 ~10-20개)]
  진입:  permission = 1{s2 > θ_evt}                     (이벤트 윈도우 밖에선 항상 flat)
        score = Σ w_k · orient_k(s_k),  w_k ≥ 0        (부호는 사전 지식으로 고정, 크기만 학습)
        direction = sign(s3), enter if |score| > θ_in
  사이징: margin_fraction ∈ {0.10, 0.20, 0.30} 버킷    (score 크기 → 버킷, 단조 매핑)
        notional = margin_fraction × leverage(=3)      (CLAUDE.md sizing contract 준수)
  청산:  P3 exit 트리거 OR 트레일링 스탑(0.5·SL @ 0.3·TP, KEEP-ALIVE 레버) OR TP/SL/time

[Layer 2 — 실행 계층 (이벤트 게이트 종결 조건이 요구한 "새 실행 프리미티브")]
  maker-first: post-only 지정가 @ best bid/ask, k bar 내 미체결 시 취소 (추격 금지)
  체결 판정(보수적): 다음 bar의 반대편 극값이 지정가를 "관통"했을 때만 체결 인정
  → 1m micro 엣지를 죽인 taker 비용 자체를 제거하는 것이 목표. 이벤트 윈도우는
    유동성 소진/청산 casacade 순간이라 maker 체결 확률이 구조적으로 높은 시점이다.
```

세 신호의 사인이 서로 덮이는 구조:
- micro flow의 사인(비용) → maker 실행이 제거
- 이벤트 게이트의 사인(실행 불가) → maker+flow가 실행 방법을 제공
- P3의 사인(정밀도 희석) → 여기선 **열린 포지션의 exit 전용**으로만 사용. 오탐의
  비용이 "조기 청산으로 이익 일부 포기"로 유계이며, 실패한 exit 설계들처럼 높은
  정밀도를 요구하지 않는다 (P3가 검증된 바로 그 역할로 강등)

## 3. 학습 목적함수 — 탈상관을 직접 공격

- 환경 분할: 2024-01..2026-08 → 분기 단위 ~10개 environment
- 목적: `maximize  min_q NetPnL_q`  (worst-quarter; 또는 CVaR@25% over quarters)
  — pooled 평균을 이기는 게 아니라 **모든 분기에서 죽지 않는 해**만 통과
- 최적화: CMA-ES (repo에 이미 outcmaes 흔적 있음) — w, θ 약 10-20차원, 미분 불필요
- 선택: 분기 LOWO — 각 분기를 hold-out으로 빼고 나머지에서 최적화 → held-out 분기
  성과 테이블. **단일 VAL 윈도우 선택 전면 금지** (탈상관 5회의 공통 경로였으므로)
- 시드: CMA-ES 초기화 N≥5 diverse random seeds, 시드별 held-out 테이블 부호 일치 요구
  (CLAUDE.md seed-diversity gate 준수)

## 4. 왜 과거 실패와 다른가 (계약의 prior_failure_reassessment 초안)

1. **엔트리 예측기 재시도가 아니다** — Layer 0는 동결, 표현 학습 0. 죽은 축(임베딩
   천장)을 건드리지 않는다.
2. **파라미터 ~10-20개 vs effective_n ~4천** — 과적합 표면이 세 자릿수로 줄어든다.
   부호 제약(w≥0, 방향 사전 고정)이 추가로 해공간을 자른다.
3. **목적함수가 병을 직접 조준** — 과거 25회는 전부 pooled VAL 최적화 후 OOS 사망.
   worst-quarter 목적 + LOWO 선택은 그 경로 자체를 제거하며, LOWO는 이 저장소에서
   held-out을 통과한 유일한 선택법(Sigma6 4/5)이라는 실증 근거가 있다.
4. **이벤트 게이트 종결 조건을 정면으로 충족** — "fundamentally new execution
   primitives 없이 재개 금지" → maker-first post-only가 바로 그 프리미티브다.
5. **비용 문제를 엣지 확대가 아니라 비용 제거로 푼다** — micro 엣지가 5회 실재
   확인되고도 죽은 유일한 이유가 taker 비용이었다.

## 5. 정직한 리스크 (설계 단계에서 미리 등록)

- **R1 체결 시뮬레이션이 최약점.** maker 체결 가정이 낙관적이면 전부 무효.
  보수 규칙(관통 시에만 체결)을 쓰고, Stage 0에서 체결률 자체를 먼저 측정한다.
  체결률이 이벤트 윈도우에서 ~30% 미만이면 거래 수가 붕괴 → 그 자리에서 kill.
- **R2 프리미티브 재현성.** P3/이벤트 게이트/micro score가 라이브-인과 경로로
  재계산 가능한지부터 확인 (저장 원장 재사용 금지 — Fresh-Forward Rule).
- **R3 OOS 소진.** 2026-01..03과 04..08 모두 이미 소비됨. Kappa1의 확정 판정은
  분기 LOWO 테이블 + **2026-08-01 이후 신규 축적 데이터**로만 한다. 과거 단일
  윈도우 성과로 승격 주장 금지 (registry 지침과 일치).
- **R4 거래 수.** 이벤트 윈도우 ∩ maker 체결 ∩ score 문턱이면 저빈도가 필연.
  split당 최소 15 거래를 못 채우면 문턱을 낮추는 게 아니라 결론을 유보한다.

## 6. 실행 계획 (각 단계 = kill 가능 게이트)

- **Stage 0 — 프리미티브·데이터 감사 (모델링 0):** 4개 프리미티브의 인과 재계산
  가능성 + 1m 데이터로 maker 체결률 측정 (이벤트 윈도우 vs 평시). 체결률 붕괴 or
  프리미티브 재계산 불가 → 즉시 종료, 비용 반나절.
- **Stage 1 — 워크벤치 계약:** architecture_workbench init/preflight/analyze-features.
  related_prior_line_ids: btc_event_gate, microstructure, exit-precision, 1h-swing 라인.
- **Stage 2 — 합성층 최적화:** CMA-ES × 5 seeds × 분기 LOWO 테이블. 통과 기준:
  held-out 분기 ≥4/5 net-positive AND 전 시드 부호 일치 AND split당 ≥15 거래.
- **Stage 3 — 신규 데이터 fresh-forward:** 2026-08 이후 축적분에서 bar-by-bar.
  이것만이 승격 근거가 될 수 있다 (Fresh-Forward Rule 플래그 4종 명시).

## 7. 명시적 비목표

- 새 라벨/표현/임베딩 학습 (죽은 축)
- 단일 VAL 윈도우 기반 선택 (탈상관 경로)
- 저장 원장 replay를 성과 근거로 사용 (diagnostic 전용)
- LightGBM 재등판 또는 기존 게이트 그리드의 OOS 재판독 (판독 소진)
