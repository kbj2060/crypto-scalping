# ETH 증거신호 × 청산맵 지지/저항 근접(confluence) — 2026-08-27

## Context

사용자가 "레짐/trend_score/증거신호(S·B)/청산맵 지지·저항을 함께 그린 3일 차트 아티팩트"를 보고
"레짐 분류는 잘 되는 것 같은데 증거신호를 매매 방향에 어떻게 이용할 수 있을지 연구해달라"고 요청.

같은 날 이미 진행된 두 갈래 연구(①증거신호×레짐(chop) 조건부 lift+비용반영 백테스트 10/10
REJECTED, ②chop-fade 진입/손절 규칙 재설계)와는 다른, **아직 테스트되지 않은 축**인지 먼저 확인—
`docs/`·`scripts/`·`research_line_registry.json` 전체를 검색해 "증거신호 × 청산맵" 조합 연구가
없음을 확인(청산히트맵 자석신호는 독립신호로 CLOSED, 증거신호 결합축은 whale_position_score/obi
등 모델-내부 지표와의 조합만 테스트됨 — 청산맵과의 조합은 처음).

## 데이터 가용성

`compute_spliced_levels()`(`scripts/live_liquidation_map_20260824.py`, 2026-08-26 라이브 확정)는
**24개 1시간봉 tail 윈도우 + current_price만의 순수 함수**다 — 실시간 청산 이벤트 피드나 OI 데이터
의존이 전혀 없다(`_prepare_common()`이 high/low/close/volume/timestamp만 읽음). 즉 최근 수집을
시작한 `liq_magnet_collector`([[eth_liq_magnet_collector_deployed_20260825]], 아직 2일치)를 기다릴
필요 없이, 기존 `data/eth_5m_1year.csv`를 1시간봉으로 리샘플링하면 VAL/OOS 전 구간을 라이브와
동일한 로직으로 인과적으로(미래참조 없이) 재구성할 수 있다. 1시간봉으로 라벨링된 레벨 스냅샷은
`timestamp+1h`로 스탬프한 뒤 5분봉 증거신호 프레임에 `merge_asof(direction="backward")`로 붙여,
"그 순간 라이브 대시보드에 실제로 떠 있었을 지지/저항"만 쓰도록 했다(라이브도 `current_price`가
직전 완결 1시간봉 종가라는 것까지 코드로 확인).

## Part A — 진단(diagnostic lift): 최근접 레벨까지의 거리 tertile

방법: 기존 lineage와 동일한 harness(`event_study`/`load_zigzag_pivots`, K=12bar/1h, VAL 2025-09~12+
OOS 2026-01~02-17, 8개 라이브 증거신호 전부)에 새 분할축만 추가. **이진 분할(레벨 있음/없음)은
무의미**로 판명 — 레버리지 6단계 합성 히트맵 특성상 현재가 5% 이내에 살아남은 레벨이 거의 항상
존재(`has_support`/`has_resistance` 각각 in-window 100.0%). 대신 최근접 동측 레벨까지의 거리를
**tertile(near/mid/far, in-window 데이터 자체의 33/67 백분위 컷 — 사후 임계값 탐색 아님)**로 나눔.

**핵심 결과 — 바닥(B, support) 쪽은 일관된 near>far, 천장(S, resistance) 쪽은 무패턴**:

| 신호 (bottom side) | chop_near lift | chop_mid | chop_far | chop_overall |
|---|---|---|---|---|
| orthogonal_combo | 4.95 | 5.04 | **2.02** | 3.87 |
| taker_delta_z_climax | 3.57 | 3.87 | **2.03** | 3.09 |
| smt_divergence | 3.50 | 3.59 | 2.40 | 3.08 |
| liquidity_sweep | 3.34 | 3.06 | 2.82 | 2.96 |
| short_term_return_z | 3.48 | 3.67 | 2.70 | 3.14 |

8개 신호 중 5~6개가 bottom 쪽에서 near/mid≫far의 같은 방향을 보임(그중 orthogonal_combo가 가장
뚜렷 — chop_far에서 lift가 절반 이하로 붕괴). top(저항) 쪽은 8개 중 일관된 패턴이 없음(mid가
최고인 경우가 더 흔함) — 이 저장소가 이미 여러 번 확인한 "Wyckoff bottom/top 비대칭(천장 신호가
구조적으로 더 약함/노이즈가 많음)"과 방향이 일치하는 독립적 교차확증.

전체 window(레짐 무관)에서도 같은 방향(orthogonal_combo bottom near 4.06 vs far 2.70) — chop
조건과 무관하게 존재하는 별도의 축.

원자료: `tmp/eth_evidence_signal_liquidation_confluence_20260827/lift_by_confluence_{all_regimes,chop_only}.csv`,
`hourly_levels.csv`. 스크립트: `scripts/research_eth_evidence_signal_liquidation_confluence_20260827.py`.

## Part B — 비용반영 백테스트 (orthogonal_combo:bottom만, 가장 강한 후보)

기존 chop-gate 백테스트 엔진(`backtest_eth_evidence_signal_chop_gated_costgate_20260827.py`, TP1.6x:
SL1.0xATR/48bar/3x/10bp, 6윈도우)을 그대로 재사용해 `chop AND near_or_mid_support`(far tertile
제외) 게이트를 추가 레이어로 얹음. **사전에 명시한 기대**: 트리거 수가 더 줄어드니(이미 chop만
게이트해도 손실이던 신호가) always_long/always_short 벤치마크를 이기게 될 가능성은 낮음 — 이 실행의
목적은 통과 여부가 아니라 개선 폭 확인.

| window | ungated 수익률 | chop만 | chop+near/mid | 트레이드수(chop→conf) |
|---|---|---|---|---|
| 2025q1 | -20.87% | -2.37% | -2.81% | 38→24 |
| 2025q2 | -18.58% | -4.92% | **-1.40%** | 38→24 |
| 2025q3 | -17.27% | -4.23% | **-2.11%** | 24→11 |
| val | -29.92% | +0.33% | **+0.66%** | 15→10 |
| oos_q1 | -12.60% | -4.03% | **-0.10%** | 32→13 |
| oos_q2 | -19.65% | +1.43% | +0.47% | 26→10 |

**결과: 예상대로 beats_benchmark 0/6(여전히 REJECTED)** — 강추세 구간 buy&hold(2025q1
always_short +83.32%, 2025q3 always_long +66.63%)를 표본이 더 줄어든 신호가 이길 구조적 방법은
없음(오늘 이미 진단된 것과 동일 원인, 재확인). **단, 6윈도우 중 4곳에서 chop-only 대비 추가
손실축소**(2025q2 +3.52%p, 2025q3 +2.12%p, val +0.33%p, oos_q1 +3.93%p — oos_q1은 사실상
손익분기 -0.10%), 2곳(2025q1, oos_q2)만 소폭 악화(각 -0.44%p/-0.96%p)하고 어느 창도 크게
나빠지지 않음. Stream 1(진입/손절 규칙 재설계, `eth_chop_fade_regime_entry_exit_and_breakout_
predictor_20260827.md`)의 "+1.5~3.5%p 손실축소, 흑자전환 아님"과 같은 성격·같은 크기의 개선.

산출물(1차, orthogonal_combo만): `scripts/backtest_eth_evidence_signal_liquidation_confluence_costgate_20260827.py`,
`tmp/eth_evidence_signal_liquidation_confluence_costgate_20260827/report.json`.

## Part B 확장 — 나머지 4개 후보(taker_delta_z_climax/liquidity_sweep/short_term_return_z/smt_divergence)

같은 날 사용자 승인으로 확장 실행. 위 스크립트를 `CANDIDATES` 리스트로 일반화해 5개 후보 전부
(orthogonal_combo 포함, 재현성 확인용으로 재실행 — 수치 100% 일치) × 3-way(ungated/chop만/
chop+near_or_mid_support)를 한 번에 계산. ungated/chop 베이스라인도 이 스크립트 안에서 새로
계산했다(taker_delta_z_climax는애초 원 chop-gate 스크립트의 5개 후보에 포함된 적이 없었고,
short_term_return_z는 원 스크립트에서 top측으로만 테스트됐었음 — bottom측 3-way 전체가 이번이 처음).

**결과 — 6윈도우 합산 수익률(sum of total_return), 5/5 전부 confluence에서 추가 개선**:

| signal | ungated | chop만 | chop+지지선근접 | delta(conf−chop) |
|---|---|---|---|---|
| orthogonal_combo | -118.89% | -13.80% | **-5.29%** | +8.51%p |
| smt_divergence | -166.23% | -12.12% | **-7.74%** | +4.38%p |
| liquidity_sweep | -267.08% | -33.88% | -21.72% | +12.16%p |
| short_term_return_z | -194.25% | -43.10% | -26.54% | +16.56%p |
| taker_delta_z_climax | -285.20% | -75.87% | -47.55% | +28.32%p |

beats_benchmark는 5후보×3변형×6윈도우 전부(90/90) **False** — 예상대로 벤치마크는 여전히 못
넘는다. 하지만 **delta가 5개 후보 전부 양수**라는 게 핵심: orthogonal_combo만의 우연이 아니라
지지선-근접 필터 자체가 일관되게 손실을 줄이는 방향으로 작동한다는 뜻이다(방향의 부호가
일관된다는 게, 개별 신호 하나의 결과보다 훨씬 신뢰할 수 있는 증거).

**⚠️ smt_divergence 표본 캐비엇 — 액면 순위를 그대로 믿지 말 것**: smt_divergence의 -7.74%(2위)는
일부 창의 표본이 극단적으로 얇아진 결과다 — `oos_q2`는 ungated에서조차 트리거가 0건(이 특정
2026-rebuilt 데이터 슬라이스에서 smt_divergence 조건 자체가 안 걸림, chop/confluence로 갈수록 더
줄어드는 게 아니라 애초에 0), `oos_q1`은 chop만 7건→chop+confluence 3건(그중 1건이 100% 승률을
만들어 그 창만 보면 좋아 보이지만 n=3은 사실상 노이즈). 두 창이 거의 0에 가까운 값을 "합산"에
보태 순위를 올린 것 — orthogonal_combo(모든 창에서 10건 이상 유지)만큼 신뢰할 수 있는 결과가
아니다.

**후보별 해석**: taker_delta_z_climax는 delta 절대값이 가장 크지만(+28.32%p) 여전히 가장 먼
상태(-47.55%)로 남는다 — ungated 자체가 워낙 자주 발동하고(윈도우당 550~645건) 승률이 구조적으로
낮아(33~39%, 이 TP:SL비 손익분기 근처 약 38.5%를 밑돎) 필터 하나로 구조를 못 뒤집는다.
orthogonal_combo가 유일하게 개별 창 단위 흑자/손익분기(val +0.66%, oos_q1 -0.10%, oos_q2 +0.47%)를
낸 후보로 남는다.

산출물(확장): `tmp/eth_evidence_signal_liquidation_confluence_costgate_20260827/report.json`(덮어씀,
5후보 전체 포함).

## Part B 검증 — "chop이 더 잘 맞는다"는 전제가 신호마다 다름

사용자가 "bull/bear에서 chop보다 바닥/천장 신호가 더 잘 맞지 않았냐"고 확인 요청 → 오늘 아침
GBM3 기반 원자료(`tmp/eth_evidence_signal_regime_chop_conditional_20260827/part_a_offense_lift_
by_regime.csv`)를 재조회해 chop vs non_chop(bull+bear) lift를 직접 대조:

| 신호 | chop lift | non_chop lift | 우위 |
|---|---|---|---|
| orthogonal_combo(bottom) | 3.91 | 3.04 | chop |
| liquidity_sweep(bottom) | 3.44 | 2.54 | chop |
| smt_divergence(bottom) | 3.51 | 2.63 | chop |
| volume_wick_climax(bottom) | 3.41 | 2.49 | chop |
| dalton_rule2_balance_edge | 1.49/1.25 | **1.60/1.54** | non_chop |
| taker_delta_z_climax | 2.42/2.02 | **2.46/2.15** | non_chop(근소) |
| short_term_return_z | 2.65/2.91 | 2.59/2.43 | 거의 동률 |

**결론: 사용자 지적이 맞았다 — 단, 8개 신호 전체가 아니라 일부에서만.** 오늘 하루 chop-게이팅의
근거였던 "core 4"(orthogonal_combo/liquidity_sweep/smt_divergence/volume_wick_climax)는 chop
우위가 뚜렷하지만, `dalton_rule2_balance_edge`와 `taker_delta_z_climax`는 정반대로 non_chop이
더 강하다 — 이 세션에서 `taker_delta_z_climax`를 chop으로 게이팅해 confluence 백테스트를 돌린 게
그 신호의 실제 강점 레짐과 안 맞는 프레임이었을 가능성이 있다(5후보 중 유일하게 압도적으로 최악인
결과와 정합적인 설명). **사용자가 이 정도 비대칭이면 "chop이 어느정도 우위"로 판단하고 chop+
지지선근접 설정을 그대로 확정** → non_chop 재게이팅은 보류, 아래 벤치마크 재정의로 진행.

## Part C — 벤치마크 재정의

기존 always_long/always_short(6윈도우 전체 buy&hold) 벤치마크는 chop 바깥의 강추세 구간까지
포함해서, chop 전용 신호가 구조적으로 이길 수 없는 잣대였다([[eth_evidence_signal_chop_regime_conditional_lift_20260827]]에서
사전에 플래그된 문제). 사용자 승인으로 **no-trade(0%, 아무 것도 안 하면 손익 0)**로 교체 — 이미
저장된 `report.json`의 `total_return`(10bp 비용 이미 반영됨)을 재시뮬레이션 없이 그대로
재판정만 다시 했다(`total_return > 0` = pass).

| 후보 | 기존 벤치마크(6윈도) | no-trade 벤치마크(6윈도) |
|---|---|---|
| **orthogonal_combo:chop_confluence** | 0/6 | **2/6** |
| smt_divergence:chop_confluence | 0/6 | 2/6 ⚠️표본(n=3 창 포함) |
| liquidity_sweep / short_term_return_z / taker_delta_z_climax | 0/6 | 0/6(전 변형 포함) |

**orthogonal_combo:bottom:chop_confluence가 no-trade 벤치마크를 통과하는 2개 창**:

| window | n | 승률 | 수익률 | breakeven |
|---|---|---|---|---|
| val | 10 | 50.0% | +0.66% | 17.3bp |
| oos_q2 | 10 | 50.0% | +0.47% | 15.3bp |

두 창 다 breakeven_bp(15~17bp)가 실제 가정 비용(10bp)보다 충분히 높아 여유가 있다(살얼음판
통과 아님). 나머지 4개 창(2025q1/q2/q3, oos_q1)은 no-trade 대비로도 여전히 음수 — **6윈도 중
2/6, 신호 5개 중 1개(orthogonal_combo)만 재현 가능한 수준으로 통과**. `smt_divergence`의 2/6도
액면상 같아 보이지만 그 중 하나(oos_q1, +1.53%)가 n=3 표본이라 신뢰도가 orthogonal_combo와
같지 않다(Part B 확장 절 캐비엇과 동일).

**결론**: 벤치마크를 신호에 맞게 재정의해도 여전히 "6윈도우 전체를 이기는 자동화 엣지"는 아니다
— 다만 always_long/short가 구조적으로 못 넘게 돼 있었다는 진단이 맞았고, 재정의하니
`orthogonal_combo:bottom:chop+지지선근접`이 **특정 두 구간(val, oos_q2)에서는 비용 이후로도
실질적인 여유를 갖고 순양수**를 낸다는, 이전보다 훨씬 구체적이고 검증 가능한 결론을 얻었다.

산출물(Part C): 재시뮬레이션 없음, 기존 `tmp/eth_evidence_signal_liquidation_confluence_costgate_20260827/report.json`을
`total_return > 0` 기준으로 재판정.

## 종합 결론 / How to apply

1. **실질적으로 새로운, 지금까지 테스트 안 된 축이 맞았고, 진단 단계에서는 꽤 뚜렷한 신호가
   나왔다**: 바닥(B) 증거신호는 근처(≈1.6% 이내)에 청산맵 지지선이 있을 때 lift가 확연히 높고,
   지지선이 멀 때(tertile far)는 절반 수준으로 떨어진다. 천장(S) 신호는 저항선 근접이 같은 방식으로
   신뢰도를 높여주지 않는다 — **비대칭적으로만 성립**하는 결과다.
2. **재량 매매(사용자의 실제 chop-fade 관행)에 바로 쓸 수 있는 실용적 결론**: 대시보드에서 B
   마커가 뜰 때, 같은 화면의 파란 지지선이 가까이(근접 tertile) 있으면 더 신뢰하고, 지지선이
   멀거나 안 보이면 신뢰도를 낮춰서 봐도 좋다는 근거가 생겼다. S 마커는 저항선 근접 여부로
   가중치를 조정할 근거가 아직 없다(무패턴).
3. **자동매매 "전체 승격" 근거는 아직 아니지만, 특정 두 구간(val/oos_q2)은 no-trade 대비
   비용 이후 순양수를 여유있게 낸다**: 기존 always_long/short(6윈도우 buy&hold) 벤치마크는
   구조적으로 강추세 구간을 포함해 chop 전용 신호가 못 이기게 돼 있었다(실측 확인, 90/90 False)
   — Part C에서 no-trade(0%)로 재정의하니 `orthogonal_combo:bottom:chop+지지선근접`이 6윈도우
   중 2곳(val +0.66%, oos_q2 +0.47%)에서 breakeven_bp 15~17bp(실제비용 10bp 대비 여유)로
   통과한다. 나머지 4윈도우는 여전히 음수 — "전체 기간 자동 엣지"는 아니지만 "특정 조건에서
   비용을 넘는 순양수"라는, 이전보다 구체적인 결론.
4. **5/5 후보 전부에서 confluence 방향은 재확인됨 — orthogonal_combo만의 우연이 아니다**:
   taker_delta_z_climax/liquidity_sweep/short_term_return_z/smt_divergence 전부 chop-only 대비
   chop+confluence가 손실을 더 줄였다(+4.4%p~+28.3%p). 그러나 no-trade 벤치마크 통과는
   orthogonal_combo(그리고 표본이 얇은 smt_divergence)뿐 — 나머지 3후보는 손실축소는 있어도
   여전히 전 윈도우 순손실.
5. **taker_delta_z_climax/dalton_rule2_balance_edge는 chop이 아니라 non_chop(bull/bear)에서
   원래 더 강한 신호였다**(사용자 확인 요청으로 재검증, chop lift 2.42/1.49 vs non_chop 2.46/1.60)
   — 이번 세션에서 이 둘을 chop으로 게이팅한 게 taker_delta_z_climax가 5후보 중 최악으로 남은
   이유일 수 있다. 사용자가 "chop이 어느정도 우위"로 판단해 재게이팅은 보류했지만, 재제안 시
   이 비대칭을 먼저 확인할 것 — "core 4"(orthogonal_combo/liquidity_sweep/smt_divergence/
   volume_wick_climax)만 chop 우위가 뚜렷하다.

**How to apply**: 청산맵×증거신호 조합 재제안 시 이 문서부터 — (1) support-side 비대칭 결과,
5후보 확장 백테스트, no-trade 벤치마크 재판정 전부 재확인됐으니 재검증 목적 재실행 불필요,
(2) resistance-side는 무패턴이 이미 확인된 결과이니 "저항 근접도 도움될 것" 가정으로 재제안하지
말 것, (3) smt_divergence 순위는 얇은 표본 아티팩트임을 감안할 것, (4) taker_delta_z_climax/
dalton_rule2_balance_edge를 chop-gate 축에 다시 넣기 전에 이 둘이 non_chop 우위 신호라는 점부터
확인, (5) `orthogonal_combo:bottom:chop+지지선근접`이 이 축의 유일한 실질적 결과물 — val/oos_q2
2개 창에서 비용 이후 순양수, 나머지는 여전히 손실.
