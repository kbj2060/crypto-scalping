# ETH 오디세이4 veto+guard 메커니즘의 BTC/SOL 이식 가능성 — 저비용 사전진단 (2026-08-20)

## 배경 / 질문

전날(`eth_odyssey4_shadow_full_reseed_causal_isolation_20260820.md`) ETH 오디세이4 섀도우의
6/6 부호일치가 `SustainedUptrendDetector` 기반 entry-veto(zig075 SHORT)+exit-guard(h48qual
레짐인지) 메커니즘 자체의 실질적 안정화 효과임이 단일요인격리로 확정됐다.

사용자 질문: "sol과 btc 모델도 veto+guard 붙이고 테스트하면 일정하게 나올까?" → 자산별
detector 재캘리브레이션+veto/guard 구현+정식 N≥5 재테스트로 이어지는 전체 메커니즘을 바로
구축하기 전에, "상승추세 중 SHORT 트레이드가 각 자산의 시드간 분산에 유독 크게 기여하는가"만
저비용으로 먼저 확인하기로 함(사용자 승인: "1번부터 해줘"). BTC/SOL 각각 전담 에이전트를
병렬 실행.

## 방법 (공통)

- **신규 학습/백테스트 없음** — 기존 N=5 시드 라이브 승격 검증(`btc_live_promotion_seed_
  robustness_5seed_20260819.md` / `sol_live_promotion_seed_robustness_5seed_20260819.md`)에서
  이미 생성된 trade ledger를 `handoff.sh pull`로 회수해 순수 사후 재집계만 수행
- ETH `SustainedUptrendDetector`와 수학적으로 동일한 causal 로직(rolling 1주=2016bar,
  `dual_momentum>0` 비율)을 각 자산 자체 `dual_momentum` 피쳐에 적용. threshold는 ETH
  고정값(0.8025793650793651)을 재사용하지 않고 **자산 자신의 2025년 전체 분포 p90**으로
  새로 계산(BTC=0.604167, SOL=0.911210) — 둘 다 진단용 placeholder, 프로덕션 값 아님
- "상승추세 중 SHORT" 트레이드를 제외했을 때 부호일치/분산(std)이 개선되는지, 대조군(C1=동일
  개수 무작위 트레이드 제외, C2=동일 개수 "비상승추세 SHORT" 제외, 각 2000회 부트스트랩)
  대비 유의한지 확인
- 재집계 PnL을 각 자산의 원본 `summary_report.json`/`report.json`과 대조해 방법론 정합성
  검증(BTC: 최대오차 0.0, SOL: no_gate PnL 15칸 전부 일치)

**⚠️ 이 진단 자체는 CLAUDE.md Fresh-Forward 규칙상 diagnostic 전용이다.** 저장 ledger
재집계이므로 promotion/모델선택 근거로 쓸 수 없다. 목적은 "전체 메커니즘을 구축할 가치가
있는지"의 저비용 사전판단뿐이며, 아래 결과는 그 판단 근거로만 쓴다.

## SOL 결과

전체 트레이드 537건(5시드×3창=15칸), SHORT 307건(57.2%). "상승추세중"으로 걸리는 SHORT는
단 **7건(SHORT의 2.3%)** — 15칸 중 11칸은 제외 대상이 0개.

| 창 | baseline std | 제외후 std | C1(무작위) 백분위 | C2(비상승추세SHORT) 백분위 |
|---|---:|---:|---:|---:|
| val | 26.23 | 29.92(악화) | 0.999 | 1.000 |
| oos_q1 | 29.57 | 27.97 | 0.299 | 0.374 |
| oos_q2 | 18.89 | 18.41 | 0.469 | 0.414 |

부호일치 개선 **0/3창**(val/oos_q1/oos_q2 전부 baseline·제외후 둘다 `sign_consistent=False`
유지). val은 대조군 대비 유의하게 **악화**(무작위로 같은 개수를 빼는 것보다 나쁜 선택), 나머지
2창은 대조군과 구분 안 되는 잡음 수준.

## BTC 결과

SHORT 502건(5시드×6창 합계) 중 "상승추세중"은 **35건(약 7%)**, 창당 5시드 합계 2~12건. (참고:
score와 후행 1주 가격수익률 상관 0.340 — 프록시가 순수 노이즈는 아님)

| 창 | baseline 부호 | 제외후 부호 | flip 해소? | std(제외전→후) | C1 성공률 | C2 성공률 |
|---|---|---|---|---:|---:|---:|
| 2025q1 | 일치(+) | 일치(+) | (이미일치) | 12.3→10.2 | 90.1% | 66.3% |
| 2025q2 | **불일치** | **일치(−)** | **해소** | 13.8→10.2 | 8.6% | **0.0%** |
| 2025q3 | 불일치 | 불일치 | 미해소 | 6.2→5.2 | 0.0% | 0.0% |
| val | 일치(+) | 일치(+) | (이미일치) | 6.0→**9.0(악화)** | 100% | 100% |
| oos_q1 | 불일치 | 불일치 | 미해소 | 17.9→**20.8(악화)** | 0.0% | 0.0% |
| oos_q2 | 일치(−) | 일치(−) | (이미일치) | 6.7→6.3 | 35.6% | 0.0% |

baseline에서 flip이던 3창(2025q2/q3, oos_q1) 중 **2025q2만 완전 해소**됐고, 이 해소는 대조군
대비 특정성이 있어 보인다 — 무작위 동일개수 제외(C1)는 2000회 중 8.6%만 부호일치를 달성하는데
실제 개입은 달성했고, "비상승추세 SHORT" 제외(C2)는 2000회 **전부** 실패(0%) — "SHORT라서"가
아니라 "상승추세중 SHORT라서"라는 신호로 읽힌다. 그러나 2025q3·oos_q1의 flip은 전혀 해소되지
않았고, val·oos_q1은 상승추세-SHORT 제외가 대조군보다 std를 더 악화시킨다(87~92백분위).

## 종합 판단

**SOL은 사실상 무의미하다.** 개입 정의가 SHORT 모집단의 2.3%만 건드려서, 설령 완벽히
튜닝해도 3/3 부호플립의 주된 해법이 되기 어렵다.

**BTC는 혼재/불확실이다.** 3개 flip 창 중 1개만 대조군 대비 의미 있게 해소됐고, 나머지 2개는
그대로거나 더 나빠졌다. ETH에서 확인된 "6/6창 전면 안정화"와는 질적으로 다른, 창마다 부호가
뒤바뀌는 패턴이다.

두 자산 모두 "ETH서 됐으니 그대로 이식하면 된다"는 근거는 이번 진단에서 나오지 않았다 —
`eth_candidate_drawdown_governor_closed_20260816.md`(BTC 이식 2회 실패: 사전스로틀 무효+
서킷브레이커 재진입폭주)와 일관된 패턴이다. 다만 BTC 2025q2의 대조군-대비 비대칭
(C1=8.6% vs C2=0%)은 완전한 우연으로 치부하기엔 아깝다 — 순수 노이즈라면 C1/C2가 비슷하게
낮아야 한다.

## 한계

1. 순수 ledger 사후재집계라 실제 veto 엔진의 "제거된 슬롯이 다른 후보에게 재할당되는" 포트폴리오
   동역학을 반영하지 못한다.
2. threshold는 진단용 placeholder(자산 자신의 2025년 전체 분포 p90)다. SOL을 2025 H1만으로
   좁히거나 percentile을 낮추면 커버리지가 달라질 수 있어, "이 특정 정의"의 부정 결과이지
   "모든 가능한 변형"을 배제한 것은 아니다.
3. BTC 2025q2의 특정성은 표본이 매우 작다(창당 uptrend-SHORT 2~12건) — 진짜 신호인지 소표본
   우연인지 이 진단만으로는 확정할 수 없다.
4. 두 에이전트 모두 자체 판단으로 "전체 메커니즘(자산별 detector+veto/guard 구현) 구축은
   시기상조"라고 명시했다.

## 다음 단계 (사용자 판단 대기)

- SOL은 이 축을 닫는 것이 합리적으로 보인다.
- BTC는 (a) 여기서 닫거나, (b) 2025q2가 왜 다른지 정밀조사하거나(포트폴리오 슬롯재할당을
  반영한 정식 재현 포함), (c) 증거가 약함을 인지한 채로 전체 메커니즘을 밀어붙여 정식 N≥5
  테스트까지 가보는 세 갈래가 있다. 어느 쪽으로 갈지는 사용자 선택.

## 산출물

- BTC: 스크립트 `scripts/research_btc_odyssey4_shadow_uptrend_short_variance_diagnostic_
  20260820.py`, 결과 `tmp/causal_regen_20260516/btc_odyssey4_shadow_uptrend_short_variance_
  diagnostic_20260820/report.json`, ledger `tmp/causal_regen_20260516/btc_live_promotion_
  seed_robustness_20260819_eval/`(gitignored, 서버 pull)
- SOL: 스크립트 `scripts/diagnose_sol_uptrend_short_seed_variance_20260820.py`, 결과
  `tmp/causal_regen_20260516/sol_uptrend_short_seed_variance_diagnostic_20260820/report.json`,
  ledger `tmp/causal_regen_20260516/sol_live_promotion_seed_robustness_20260819/`(서버 pull)

신규 학습 없음(기존 N=5 ledger 재집계만). 추적 파일 수정/커밋/푸시 없음.
`trading_bot.py`/`.env`/`runtime_config.py` 미접촉.

**fresh_forward_bar_by_bar=false**(diagnostic 전용, 저장 ledger 사후재집계),
**trade_ledgers_used_as_input=true**(diagnostic 목적 한정 — promotion/모델선택 근거 아님),
**saved_parent_exit_timestamps_used=false**, **future_rows_used_for_entry=false**.
