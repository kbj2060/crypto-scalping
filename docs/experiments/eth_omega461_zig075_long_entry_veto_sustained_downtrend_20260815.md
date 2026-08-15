# zig075 지속하락장 LONG 진입 거부(entry veto) — Odyssey4 실행 로그 #3 (2026-08-15)

상태: **`CONFIRMED`** — 사전등록 게이트(VAL strict/relaxed + OOS-Q1/OOS-Q2 단일터치 strict/relaxed)
전부 통과. zig075 SHORT/지속상승장판(2026-08-14)과 달리 이번엔 판정 창(OOS-Q2) 자체에서 veto가
실제로 37회 발동해 거래 1건이 교체됐고, 그 교체가 방향이 유리했다(손절 −8.38% 제거, 슬롯이 익절
+13.64%로 대체) — OOS-Q2 `with_gate` PnL이 −12.69% → **+8.30%**로 부호반전, MDD도 −20.76% →
−13.72%로 개선. 다만 VAL·OOS-Q1은 veto 발동 0건(무해성 통과)이라 판정 전체의 근거는 **판정 3창
중 1창, 거래 1건**에 크게 의존한다 — 뒤의 "정직한 한계" 참고.

## 배경

2026-08-14 세션에서 CONFIRMED된 zig075 SHORT 지속상승장 entry veto
(`docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md`)는 외부
causal 레짐 탐지기로 "지속상승장에서 SHORT 진입을 거부"하는 메커니즘이었다. 이 실행 로그는 그
거울상 가설을 검증한다: 대칭적인 "지속하락장" 탐지기로 "지속하락장에서 zig075 LONG 진입을
거부"하면 어떻게 되는가. `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`가
정리한 "외부 causal 레짐 신호로 방향 노출 자체를 관리한다"는 성공 패턴을 반대쪽에 처음 적용한
것이다.

**zig075 SHORT판과의 결정적 차이(정직하게 명시)**: SHORT판은 Odyssey3 실행 로그 #1의 bar-level
손실 메커니즘 진단(Q3 SHORT 손실 10/19건·−0.4089/−0.5440이 탐지기 ON bar에 집중, median MFE
41%)에서 출발한 **진단 확정 설계**였다. 이번 LONG/하락장판은 **그런 진단 없이 구조적 대칭성만으로
설계한 가설 검증**이다 — 실패했다면 다음 단계는 파라미터를 더 스윕하는 게 아니라 LONG 쪽의
동등한 손실 메커니즘 진단을 먼저 하는 것이었다(모듈 docstring "Honest scope note" 참고).

**베이스라인 변경**: SHORT판은 Odyssey3 베이스라인(h48qual 레짐 exit 가드) 위에서 비교했지만, 이
후보는 **Odyssey4 베이스라인**(Odyssey3 + zig075 SHORT 지속상승장 veto, 이미 CONFIRMED) 위에
얹는다 — Odyssey4 계약 문서 자체가 "향후 신규 후보는 이 표를 G0 기준으로 삼는다"고 명시했기
때문이다. 즉 SHORT veto는 이 실험 내내 항상 켜져 있고, 이번에 새로 추가되는 것은 LONG veto뿐이다.

## 탐지기 — 신규 자유변수 0개(레시피는 재사용, 산출된 상수는 신규)

가드 모듈(`research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.build_detector`)이
잠근 레시피를 부호만 뒤집어 그대로 재사용했다: `WEEK_BARS=2016`(dual_momentum 자체의
기존 lookback), `DETECTOR_PERCENTILE=0.90`("상위 10분위" 관행), 캘리브레이션 창=2025 Q1+Q2 전용
(Q3/VAL/OOS 미참조). 유일한 차이는 `dual_momentum>0` 대신 `dual_momentum<0`의 rolling 비율을 쓴다는
것 — 레시피에 새 자유변수는 없지만, 산출되는 임계값 상수는 이번에 처음 계산되는 신규 값이다(상승장
판처럼 미리 잠긴 값과 대조하는 게 아니라, 이 실행이 그 값을 새로 잠근다).

| | 상승장 탐지기(SHORT veto, 기존) | 하락장 탐지기(LONG veto, 신규) |
|---|---|---|
| p75 | 0.5610119047619048 | 0.8501984126984127 |
| **p90(채택)** | **0.8025793650793651**(잠긴 값과 1e-12 내 일치 확인) | **0.9712301587301587**(신규 계산) |
| p95 | 0.8773313492063477 | 1.0 |

하락장 탐지기의 p90/p95가 상승장판보다 훨씬 극단(0.971/1.0 vs 0.803/0.877)인 것은 우연이 아니다
— 2025 Q1+Q2 캘리브레이션 표본 자체에 이미 알려진 극심한 지속 하락 구간이 있었고
([[eth_reversal_evidence_signal_scorecard_20260814]]가 같은 시기를 "sustained ETH downtrend"로
지칭), p95=1.0은 트레일링 2016-bar 전부가 dual_momentum<0이었던 구간이 실제로 존재했다는 뜻이다.

## 개입 정의

flat 상태 진입 루프에서 `component == zig075 && side == LONG && 하락장 탐지기 ON(신호 bar)`이면
그 진입만 스킵. zig075 SHORT veto(기존, 상승장 탐지기 p90)·h48qual(LONG/SHORT + 자체 exit
가드)·모든 모델 헤드·threshold·TP/SL·사이징·priority·캡 전부 무변경. zig075 LONG 신호 자체(방향
예측)는 절대 건드리지 않는다 — 이미 발생한 LONG 신호 중 "하락장 한복판"이라는 레짐 조건에
해당하는 것만 거른다.

## 방법 (사전 등록)

신규 스크립트:
`scripts/research_eth_omega461_zig075_long_entry_veto_sustained_downtrend_20260815.py`. 신규 replay
함수(`greedy_replay_dual_entry_veto`)는 zig075 SHORT veto 스크립트의 `greedy_replay_entry_veto`를
그대로 복사한 뒤 LONG veto 블록 하나만 추가한 것(SHORT 블록은 완전히 동일하게 보존).

1. **detector_build**: 상승장 탐지기는 가드 모듈 재사용 후 잠긴 값과 1e-12 내 일치 확인 —
   **통과**. 하락장 탐지기는 이 스크립트가 신규 계산(위 표).
2. **G0a**: import한 `zveto.greedy_replay_entry_veto`(SHORT veto만 부착)로 Odyssey4 베이스라인
   G0 표(val+oos_q1)를 재현 — **통과**.
3. **G0b**: 이 스크립트의 dual-veto replay 복사본(LONG veto 미부착, SHORT veto만 부착)이 6개 창
   전부에서 Odyssey4 G0 표와 정확히 일치(pnl/mdd ±0.05pp, trades 정확히) + veto_bars_long=0 —
   **통과**. 이 실행이 판정용 베이스라인 렛저를 생성한다.
4. **후보 실행**: p90 LONG veto를 부착(SHORT veto는 항상 유지), 6개 창 전부 단일 실행.
5. **강건성**(참고 전용): LONG veto 임계값 p75/p95로 2025 분기 3개 재실행. SHORT veto·h48qual
   가드는 무변경.
6. **판정**: `gate.summarize_multiwindow` strict(0pp)+relaxed(3pp) — VAL 게이트 통과 후 OOS-Q1+
   OOS-Q2 단일터치. 2025 분기는 참고 티어(판정에 미포함).

## 결과

### G0 — 통과

`gate_pass_g0=True`. G0a(val/oos_q1), G0b(전체 6창, LONG mask 미부착) 전부 Odyssey4 G0 표와 정확히
일치, `veto_bars_long=0`(SHORT veto 발동 수는 기존 Odyssey4 값과 동일: Q1=0/Q2=10/Q3=19/VAL=12/
OOS-Q1=0/OOS-Q2=0).

### 신호 겹침 (실행 전 정적 진단)

| 창 | zig075 LONG 신호 bar | 하락장 탐지기 ON과 겹침 | 탐지기 ON 비율 |
|---|---|---|---|
| 2025-Q1 | 436 | 10 | 7.4% |
| 2025-Q2 | 676 | 20 | 11.8% |
| 2025-Q3 | 364 | 25 | 4.0% |
| VAL | 469 | 44 | 5.2% |
| OOS-Q1 | 498 | 2 | 4.2% |
| OOS-Q2 | 763 | 107 | 4.9% |

신호 bar 겹침일 뿐 실제 체결/슬롯 경합 결과는 fresh-forward replay로만 판정된다 — 아래 판정
표에서 보듯 겹침 bar 수와 실제 veto 발동/거래 교체 건수는 크게 다르다(예: VAL은 겹침 44 bar지만
veto 발동 0건).

### 판정 — strict/relaxed 둘 다 CONFIRMED

| 창 | 티어 | Odyssey4 베이스라인 no_gate / with_gate | LONG veto p90 no_gate / with_gate | veto 발동 | 판정(strict) |
|---|---|---|---|---|---|
| 2025-Q1 | 참고 | 97.70%/−20.62%/28 · 44.98%/−20.62%/20 | 171.56%/−15.00%/28 · **99.15%/−19.92%/20** | 3 bar | (미판정, 큰 개선) |
| 2025-Q2 | 참고 | 65.83%/−14.17%/31 · 5.62%/−23.59%/19 | 36.41%/−14.17%/31 · **동일(5.62%/−23.59%/19)** | 4 bar | (미판정, no_gate만 악화·with_gate 무변화) |
| 2025-Q3 | 참고 | −10.63%/−29.66%/23 · 20.17%/−19.72%/17 | **동일** | 0 bar | (미판정) |
| **VAL** | 판정 | 41.13%/−21.70%/35 · 77.31%/−21.76%/26 | **동일 (0 veto bar)** | 0 bar | **PASS** |
| **OOS-Q1** | 판정 | 93.27%/−15.48%/24 · 67.25%/−15.48%/19 | **동일 (0 veto bar)** | 0 bar | **PASS** |
| **OOS-Q2** | 판정 | −9.55%/−20.76%/13 · −12.69%/−20.76%/10 | **+12.19%/−13.72%/13 · +8.30%/−13.72%/10** | 37 bar | **PASS (부호반전)** |

- VAL 게이트: strict/relaxed 모두 **통과**(`with_gate` 완전 동일, veto 0건 — 무해성).
- OOS 단일터치: strict/relaxed 모두 **CONFIRMED**. OOS-Q1은 무해성(veto 0건), **OOS-Q2는 실제
  개입**: `with_gate` PnL −12.69% → **+8.30%**(+20.99pp, 부호반전), MDD −20.76% → −13.72%(+7.04pp
  개선), `no_gate` PnL −9.55% → +12.19%(+21.74pp), MDD −20.76% → −13.72%.
- **2025-Q1 효과(참고, 판정 미포함)**: `with_gate` 44.98% → **99.15%**(+54.17pp, 2배 이상), MDD
  −20.62% → −19.92%(개선).
- **2025-Q2 비용(참고, 판정 미포함)**: `no_gate`만 65.83% → 36.41%(−29.42pp)로 악화되지만
  `with_gate`는 duration-gate 이후 사실상 무변화(5.6236... → 5.6236..., 12번째 소수점 차이).

### 렛저 diff — 거래 단위로 본 메커니즘

**OOS-Q2 (판정 창, 유일한 실제 효과)**: 2026-05-23 진입 zig075 LONG 손절(−8.38%, 2026-06-02 청산)
1건이 제거되고, 슬롯이 하루 뒤(2026-05-24) 진입한 zig075 SHORT 익절(+13.64%, **같은 날
2026-06-02 청산**)로 대체됐다. 즉 "하락장 한복판에서 거스르는 LONG 대신, 같은 하락장에 순응하는
SHORT가 그 자리를 차지"한 것 — 설계 의도와 정확히 일치하는 단일 사례다.

**2025-Q1 (참고, 개선)**: 2025년 3월 초 하락장에서 zig075 LONG 손절 2건(−8.77%, −8.54%) 제거,
슬롯이 h48qual LONG(사실상 breakeven, +0.05%)과 zig075 SHORT 익절(+14.56%)로 대체 — OOS-Q2와 같은
방향의 메커니즘.

**2025-Q2 (참고, no_gate 비용)**: zig075 LONG 익절(+13.95%, "좋은" 거래) 1건이 제거되고, 슬롯이
h48qual SHORT 손절(−6.27%)로 대체 — 메커니즘의 역방향 사례(하락장 탐지기가 켜져 있었지만 그
LONG은 실제로는 이긴 거래였다). with_gate 수준에서는 이 거래 자체가 duration-gate 대상이 아니라서
최종 성과에는 반영되지 않았다.

### 강건성 — 참고 티어, p95는 사실상 p90과 동일·p75는 방향이 갈림

| LONG veto 임계값 | 2025-Q1 (with_gate) | 2025-Q2 (with_gate) | 2025-Q3 (with_gate) |
|---|---|---|---|
| p75 (0.8502) | 65.91%/−25.41% (5 제거/6 추가, PnL 개선폭 작지만 MDD 악화) | 동일(1 제거/1 추가, p90과 동일) | **5.11%/−19.72%(0 제거/2 추가, p90의 무변화에서 벗어나 악화)** |
| **p90 (0.9712, 채택)** | **99.15%/−19.92%(3 제거/2추가)** | **동일(1 제거/1 추가)** | **동일(0 제거/0 추가)** |
| p95 (1.0) | 동일(0 제거/0 추가, 베이스라인과 동일) | 동일(0 제거/0 추가) | 동일(0 제거/0 추가) |

p95는 이론적 최댓값(1.0)이라 사실상 발동하지 않아 p90과 크게 다르지 않고, p75(더 느슨)는 Q1에서
개선폭이 줄면서 MDD가 악화되고 Q3에서는 p90의 무변화가 실제 악화로 바뀐다 — **임계값을 느슨하게
할수록 이득이 아니라 손해가 커지는 방향**이라, 채택한 p90(=상위 10분위, 관행값)이 우연히 가장
유리한 지점이었을 가능성을 배제할 수 없다. zig075 SHORT판은 p75/p90/p95 전 구간에서 Q3 부호반전이
일관되게 유지됐던 것과 대조적으로, 이번엔 판정에 쓰이지 않는 참고 창에서만이지만 임계값 민감성이
관찰된다.

## 준수 확인

`fresh_forward_bar_by_bar=true`(단방향 단일 pass, 탐지기는 순수 backward rolling, veto는 신호
bar의 mask[i]만 읽음), `trade_ledgers_used_as_input=false`(전부 신규 fresh-forward replay),
`saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`.

`git status` 확인: `trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`,
`trading_bot_modules/runtime_config.py`, `.env` 전부 무변경. 신규 파일은
`scripts/research_eth_omega461_zig075_long_entry_veto_sustained_downtrend_20260815.py`와 이 문서
뿐이며, 기존 임포트 모듈(가드 모듈, zig075 SHORT veto 모듈, 게이트 모듈,
`replay_omega4_6_1_greedy_router_20260706.py` 등)은 전부 읽기 전용으로만 사용했다. 재학습 없음,
GPU 불필요(DEVICE=cpu), conda env `quant_ai`.

Seed-Diversity Ensemble Promotion Gate: 해당 없음(재학습 모델 없음, 결정론적 룰이라 시드 축이
존재하지 않음). Omega Artifact Integrity Promotion Gate: 해당 없음(기존 라이브 h48qual/zig075
parent 예측 아티팩트 그대로 재사용, 신규 아티팩트 없음).

## 정직한 한계

1. **판정 통과의 실질 근거는 판정 3창 중 1창(OOS-Q2), 거래 1건 교체에 크게 의존한다.** VAL·OOS-Q1은
   veto 발동 자체가 0건(무해성만 증명). OOS-Q2의 부호반전(+20.99pp)은 매력적이지만 표본이 극히
   작다(with_gate 거래 10건 중 1건 교체) — zig075 SHORT판의 OOS 판정 3창이 전부 발동 0건("무해성
   증명, 이득 증명 아님")이었던 것과 비교하면 이번엔 최소 하나의 실제 유리한 개입이 관찰됐다는 점에서
   더 강하지만, 단일 거래 표본이라는 근본적 약점은 h48qual SHORT판의 OOS-Q1 "거래 1건 교체"
   케이스와 동일한 수준의 취약성이다.
2. **강건성이 zig075 SHORT판만큼 깨끗하지 않다.** SHORT판은 p75/p90/p95 전 구간에서 Q3 부호반전이
   유지됐지만, 이번엔 p75로 완화하면 Q3(참고 창)가 p90의 무변화에서 실제 악화로 바뀐다. p90(상위
   10분위 관행값)이 사후에 유리한 지점으로 드러났을 가능성을 배제할 수 없다 — 다만 p90은 SHORT
   판과 동일하게 사전에 고정된 관행값이지 사후 스윕으로 고른 게 아니다.
3. **LONG/하락장 손실의 bar-level 메커니즘 진단을 하지 않았다.** zig075 SHORT판은 Odyssey3 실행
   로그 #1의 진단(Q3 손실 10/19건이 탐지기 ON에 집중, MFE 41%)에서 출발한 확정 설계였지만, 이
   후보는 구조적 대칭성만으로 설계했다. 결과가 CONFIRMED로 나왔지만, "왜 하락장 LONG 손실이
   탐지기와 겹치는가"에 대한 독립적인 메커니즘 확인은 없다 — 3건의 거래 diff(위)가 정성적으로
   방향을 지지할 뿐이다.
4. **2025-Q1/Q2/Q3는 참고 티어(in-sample OOF)이며, OOS-Q2(2026-04~06)가 유일한 진짜 forward 판정
   증거다.** 이 창 하나에 판정 전체가 걸려 있다는 점에서 (1)과 같은 우려가 반복된다.
5. **아직 어떤 프로세스에도 배포되지 않았다.** zig075 SHORT veto(이미 CONFIRMED, 섀도우 배포
   스크립트까지 준비됐으나 cutover 대기 중)와 마찬가지로 이 LONG veto도 순수 연구 확정 상태다.

## 종합 판단

메커니즘적으로는 zig075 SHORT/상승장판과 대칭이라는 점에서 사전 타당성이 있고, 판정 게이트를
형식·실질 양쪽에서(적어도 1개 실제 거래 교체를 통해) 통과했다는 점에서 h48qual SHORT판보다는
근거가 강하다. 그러나 "판정 3창 중 1창·거래 1건"이라는 표본 크기는 자체로 결정적 증거라 하기엔
얇다. **배포 판단 이전에 최소한 다음 forward 관찰(섀도우) 또는 추가 독립 판정 표본이 필요하다** —
이 자체를 기각 근거로 보지 않지만, zig075 SHORT veto만큼 확신 있게 "섀도우 배포 최우선 후보"로
격상하기에는 이르다.

## 산출물

- 스크립트: `scripts/research_eth_omega461_zig075_long_entry_veto_sustained_downtrend_20260815.py`
- report: `tmp/causal_regen_20260516/eth_omega461_zig075_long_entry_veto_sustained_downtrend_20260815/report.json`
  (창별 3중 비교, veto 이벤트 전체, per-trade 렛저 diff, 강건성, G0a/G0b 결과 포함)
- 렛저: 동 디렉토리 `portfolio_ledger_<창>_odyssey4_baseline.csv` /
  `portfolio_ledger_<창>_zig075_long_entry_veto_p90.csv`

## 다음 단계 / 미해결

- forward 관찰(섀도우 로깅) 없이는 OOS-Q2 단일 거래 표본 이상의 확신을 얻을 수 없다 — zig075
  SHORT veto 섀도우 배포와 함께 묶어 처리할지는 사용자 판단 필요.
- LONG/하락장 손실의 bar-level 메커니즘 진단(dir_p_long/quality_for_action/MFE 분해, Odyssey3
  실행 로그 #1과 동형)은 이 실행 로그에서 하지 않았다 — 하면 이 후보의 신뢰도를 높이거나 반박할 수
  있다.
- h48qual에도 같은 하락장 LONG veto를 확장할지는 미검토(zig075 SHORT→h48qual SHORT 확장이
  약한 결과였던 선례가 있어 우선순위는 낮다).
- Odyssey4 계약 문서(`docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`)
  실행 로그 갱신은 이 문서 발행 직후 별도로 처리한다.
