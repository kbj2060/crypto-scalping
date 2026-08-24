# h48qual 지속상승장 SHORT 진입 거부(entry veto) — Odyssey4 실행 로그 #2 (2026-08-15)

상태: **`CONFIRMED (약함/한계 있음)`** — 사전등록 게이트(VAL strict/relaxed + OOS-Q1/OOS-Q2 단일터치
strict/relaxed) 전부 통과하지만, zig075판과 달리 이 개입의 존재 이유였던 2025-Q3 참고창 효과가
부호반전 없이 미미하고(+3.24pp) 2025-Q2 참고창 비용이 그보다 크다(−9.09pp) — 참고 3분기 순효과가
오히려 음수(−5.85pp)다. 판정 통과의 실질은 대부분 "무해성"이다: 판정 3창(VAL/OOS-Q1/OOS-Q2) 중
VAL·OOS-Q2는 veto 발동 0건(렛저 완전 동일)이고 OOS-Q1은 발동 1건, 거래 1건 교체로 +1.35pp — 표본
1건짜리 효과다.

## 배경

2026-08-14 세션에서 CONFIRMED된 zig075 SHORT 지속상승장 entry veto
(`docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md`)의 종결부와
Odyssey3 계약 문서가 공통으로 "미검토"로 남긴 항목: 같은 메커니즘을 공유 슬롯의 다른 축인
h48qual SHORT에 확장하는 것(`PRIORITY=(h48qual, zig075)`에서 h48qual이 우선순위 1번). h48qual은
이미 자체 지속상승장 exit 가드(Odyssey3 베이스라인, `sustained_uptrend_mask` 키로 exit_head를
원본 pre-liveATR 버전으로 전환)를 갖고 있어 우선순위가 낮았지만, 검토 자체는 "동일한 자유변수-0
원칙으로 가능"이라고 명시되어 있었다. 이 실행 로그는 그 검토다.

**신규 자유변수 0개**: 탐지기 공식(`dual_momentum>0`의 2016-bar rolling 비율), 캘리브레이션 창
(2025 Q1+Q2 전용), 임계값(p90 = 0.8025793650793651)을 zig075판과 완전히 동일하게 재사용한다.
바뀌는 것은 veto 대상뿐이다(`component=="zig075"` → `component=="h48qual"`, side는 그대로 SHORT).
h48qual의 exit 가드(`sustained_uptrend_mask`)는 이 실험에서 손대지 않고 그대로 유지된다 — 같은
탐지기 신호를 서로 다른 딕셔너리 키(`sustained_uptrend_mask` vs `short_entry_veto_mask`)로 갖는
두 개의 독립 개입이 h48qual 컴포넌트 위에 공존한다.

## 메커니즘 근거 — 신호 bar 탐지기 오버레이 (실행 전 정적 진단)

| 창 | h48qual SHORT 신호 bar | 탐지기 ON과 겹침 | 탐지기 ON 비율 |
|---|---|---|---|
| 2025-Q1 | 400 | 66 | 7.6% |
| 2025-Q2 | 507 | 81 | 11.6% |
| 2025-Q3 | 948 | 264 | 43.0% |
| VAL | 406 | 20 | 7.6% |
| OOS-Q1 | 87 | 8 | 5.4% |
| OOS-Q2 | 111 | 15 | 8.2% |

h48qual은 우선순위 1번이라 zig075보다 신호 bar 수 자체가 훨씬 많다(zig075판 Q3 신호는 19건뿐이었던
것과 대조적으로 h48qual Q3는 948 bar). 단, 이 표는 신호 bar 겹침일 뿐 실제 체결/포트폴리오 슬롯
경합 결과는 fresh-forward replay로만 판정된다.

## 개입 정의

flat 상태 진입 루프에서 `component == h48qual && side == SHORT && 탐지기 ON(신호 bar)`이면 그
진입만 스킵. h48qual LONG·h48qual의 기존 exit 가드·zig075(LONG/SHORT 모두)·모든 모델 헤드·
threshold·TP/SL·사이징·priority·캡 전부 무변경. 비교 베이스라인은 zig075판과 동일한 Odyssey3
베이스라인 전체(asymmetric_tabm_liveatr + h48qual 레짐 가드 p90).

## 방법 (사전 등록)

신규 스크립트:
`scripts/research_eth_omega461_h48qual_short_entry_veto_sustained_uptrend_20260815.py`. 신규 replay
로직 없음 — zig075판 실행 로그가 정의한 `greedy_replay_entry_veto`(이미 컴포넌트 이름에 무관하게
`comp.get("short_entry_veto_mask")`를 읽는 범용 구현)를 그대로 import해서 재사용하고, veto mask를
붙이는 대상만 h48qual로 바꿨다.

1. **detector_build**: 가드 모듈의 `build_detector()` 재사용, 재계산 p90이 잠긴 값과 1e-12 내
   일치 확인 — **통과**(0.8025793650793651, 정확히 일치).
2. **G0a**: 가드 모듈 원본 replay로 Odyssey3 베이스라인(val+oos_q1) 재현 — **통과**.
3. **G0b**: import한 `greedy_replay_entry_veto`(mask 미부착)가 6개 창 전부에서 계약 G0 표와 일치
   (pnl/mdd ±0.05pp, trades 정확히) + veto_bars=0 — **통과**. 이 실행이 판정용 베이스라인 렛저를
   생성한다.
4. **후보 실행**: p90 veto를 h48qual에 부착, 6개 창 전부 단일 실행.
5. **강건성**(참고 전용): veto 임계값 p75/p95로 2025 분기 3개 재실행. zig075는 무변경.
6. **판정**: `gate.summarize_multiwindow` strict(0pp)+relaxed(3pp) — VAL 게이트 통과 후 OOS-Q1+
   OOS-Q2 단일터치. 2025 분기는 참고 티어(판정에 미포함).

## 결과

### G0 — 통과

`gate_pass_g0=True`. G0a(val/oos_q1) 및 G0b(전체 6창, mask 미부착) 전부 6개 창에서 pnl/mdd/trades
정확히 일치, veto_bars=0.

### 판정 — strict/relaxed 둘 다 CONFIRMED

| 창 | 티어 | Odyssey3 베이스라인 no_gate / with_gate | entry-veto p90 no_gate / with_gate | veto 발동 | 판정(strict) |
|---|---|---|---|---|---|
| 2025-Q1 | 참고 | 97.70%/−20.62%/28 · 44.98%/−20.62%/20 | **동일** | 0 bar | (미판정) |
| 2025-Q2 | 참고 | 106.45%/−13.23%/31 · 31.49%/−15.85%/19 | 92.17%/−20.18%/32 · **22.40%/−24.24%/20** | 2 bar | (미판정, **pnl/mdd 둘 다 악화**) |
| 2025-Q3 | 참고 | −37.43%/−51.25%/27 · −15.86%/−44.37%/21 | −35.02%/−49.38%/27 · **−12.63%/−43.45%/21** | 4 bar | (미판정, 소폭 개선·부호반전 없음) |
| **VAL** | 판정 | 46.59%/−21.70%/35 · 77.31%/−21.76%/26 | **동일 (0 veto bar)** | 0 bar | **PASS** |
| **OOS-Q1** | 판정 | 93.27%/−15.48%/24 · 67.25%/−15.48%/19 | 94.83%/−15.48%/24 · **68.60%/−15.48%/19** | 1 bar | **PASS** |
| **OOS-Q2** | 판정 | −9.55%/−20.76%/13 · −12.69%/−20.76%/10 | **동일 (0 veto bar)** | 0 bar | **PASS** |

- VAL 게이트: strict/relaxed 모두 **통과**(`with_gate` 완전 동일, veto 0건).
- OOS 단일터치: strict/relaxed 모두 **CONFIRMED**(OOS-Q1 with_gate PnL +1.35pp/MDD 불변, OOS-Q2
  완전 동일).
- **2025-Q3 효과(참고, 판정 미포함)**: `with_gate` −15.86% → **−12.63%**(+3.24pp, 부호반전 없음),
  MDD −44.37% → −43.45%(+0.92pp). `no_gate` −37.43% → −35.02%(+2.41pp).
- **2025-Q2 비용(참고, 판정 미포함)**: `with_gate` 31.49% → **22.40%**(−9.09pp), MDD −15.85% →
  −24.24%(−8.39pp, 악화). Q3 개선(+3.24pp)보다 Q2 비용(−9.09pp)이 약 2.8배 크다 — 참고 3분기
  순효과는 **−5.85pp로 음수**.

### 렛저 diff — 메커니즘은 실재하지만 효과는 작고 혼재

- **2025-Q2 (비용 발생)**: 05-14 h48qual SHORT 익절(+0.100) 제거, 슬롯이 zig075로 넘어가 06:20
  zig075 SHORT 익절(+0.137)로 대체됐지만 05-17 zig075 LONG 손절(−0.100)도 추가돼 순비용 발생.
  이 2건이 이 창 악화의 전부.
- **2025-Q3 (개선, 소폭)**: 07-18·07-20 h48qual SHORT 손절 2건(합 −0.115) 제거, 슬롯이 zig075
  SHORT 손절로 대체(합 −0.100, 손실폭 소폭 축소); 07-28 h48qual SHORT 익절(+0.096) 제거, zig075
  SHORT 익절(+0.120)로 대체(소폭 개선). 3건 전부 zig075가 유사한 방향/사유로 이어받아 순효과가
  작다 — zig075판(Q3 8건 제거, 대부분 h48qual이 이어받아 손실 축소)과 대칭이지만 폭이 훨씬 작다.
- **OOS-Q1 (판정 창, 표본 1건)**: 03-19 h48qual SHORT exit_head 손절(−0.017) 제거, 슬롯이 03-20
  h48qual LONG 손절(−0.009)로 대체 — 손실이되 더 작은 손실. 표본 1건이라 강한 근거로 보기 어렵다.

### 강건성 — 참고 티어, 임계값에 따라 방향이 흔들림

| veto 임계값 | 2025-Q1 (with_gate) | 2025-Q2 (with_gate) | 2025-Q3 (with_gate) |
|---|---|---|---|
| p75 (0.5610) | 동일 (0 제거) | 22.40%/−24.24% (1 제거/2 추가, p90과 동일) | −9.36%/−43.45% (4 제거/4 추가, p90보다 소폭 개선) |
| **p90 (0.80258, 채택)** | **동일 (0 제거)** | 22.40%/−24.24% (1 제거/2 추가) | −12.63%/−43.45% (3 제거/3 추가) |
| p95 (0.8773) | 동일 (0 제거) | **동일 (0 제거)** | −12.63%/−43.45% (3 제거/3 추가, p90과 동일) |

zig075판과 달리 세 백분위 전부에서 Q3 효과가 여전히 음수 구간에 머문다(부호반전이 한 번도
일어나지 않음). Q2 비용은 p95에서 사라지지만(0 제거), 그 경우 Q3 개선도 p90과 동일해 net은 여전히
약하다. 이는 h48qual SHORT의 Q3 손실이 zig075처럼 "지속상승장 감지기와 강하게 겹치는 진입타이밍
손실"이 아니라 더 분산된 원인을 갖는다는 방증이다.

## 준수 확인

`fresh_forward_bar_by_bar=true`(replay는 import한 `greedy_replay_entry_veto`의 단방향 단일 pass,
탐지기는 순수 backward rolling, veto는 신호 bar의 mask[i]만 읽음), `trade_ledgers_used_as_input=
false`(전부 신규 fresh-forward replay), `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`.

`git status` 확인: `trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`,
`trading_bot_modules/runtime_config.py`, `.env` 전부 무변경(diff 0줄). 신규 파일은
`scripts/research_eth_omega461_h48qual_short_entry_veto_sustained_uptrend_20260815.py`와 이 문서
뿐이며, 기존 임포트 모듈(zig075 entry-veto 스크립트, 가드 모듈, 게이트 모듈,
`replay_omega4_6_1_greedy_router_20260706.py` 등)은 전부 읽기 전용으로만 사용했다. 재학습 없음,
GPU 불필요(DEVICE=cpu), conda env `quant_ai`.

Seed-Diversity Ensemble Promotion Gate: 해당 없음(재학습 모델 없음, 결정론적 룰이라 시드 축이
존재하지 않음). Omega Artifact Integrity Promotion Gate: 해당 없음(기존 라이브 h48qual/zig075
parent 예측 아티팩트 그대로 재사용, 신규 아티팩트 없음).

## 정직한 한계

1. **판정 통과의 실질은 zig075판보다 훨씬 약한 "무해성"이다.** 판정 3창 중 VAL·OOS-Q2는 veto
   발동 자체가 0건(렛저 byte-identical)이고, OOS-Q1의 "개선"(+1.35pp)은 거래 1건 교체에서 나온
   것으로 표본이 너무 작아 노이즈와 구분할 수 없다.
2. **zig075판과 달리 이 개입의 존재 이유(Q3 개선)가 약하고 순효과가 음수다.** zig075 버전은
   2025-Q3 `with_gate`가 −15.86%→+20.17%로 **부호 자체가 반전**했고 그 개선이 Q2 비용(1건,
   −7.61pp)을 압도했다. h48qual 버전은 Q3가 −15.86%→−12.63%로 **여전히 손실 구간에 머물며**
   (+3.24pp에 불과), Q2 비용(−9.09pp)이 그보다 크다 — 참고 3분기 순효과가 −5.85pp로 오히려
   음수다. 강건성 표(p75/p95)에서도 Q3의 부호반전은 한 번도 관찰되지 않는다.
3. **h48qual의 Q3 SHORT 손실은 "지속상승장 진입타이밍 손실"이라는 zig075와 같은 단일 원인으로
   설명되지 않는 것으로 보인다.** 탐지기 ON 겹침 자체는 Q3에서 43.0%로 zig075(Odyssey3 실행
   로그 #1 진단 대상)보다도 높지만, 실제 replay 효과는 훨씬 작고 임계값에 따라 방향이 흔들린다
   — 추가 진단 없이 원인을 확정할 수 없다.
4. **2025-Q3는 참고 티어(in-sample OOF)이며, forward에서 진짜 지속 상승장을 한 번도 겪지 않은
   것은 zig075판과 동일한 한계다.** 진짜 검증은 섀도우 forward 관찰뿐이다.
5. **결론: 사전등록 게이트 문구상으로는 CONFIRMED이지만, zig075판과 같은 지위로 취급해서는
   안 된다.** zig075판은 뚜렷한 참고 티어 근거(부호반전 +36pp급)를 가진 깨끗한 긍정 결과였던
   반면, 이 h48qual판은 판정 창에서 사실상 아무 일도 하지 않으면서 참고 창에서는 순비용을
   발생시키는 개입이다. 배포/섀도우 후보로 우선순위를 둘 근거가 약하다.

## 산출물

- 스크립트: `scripts/research_eth_omega461_h48qual_short_entry_veto_sustained_uptrend_20260815.py`
- report: `tmp/causal_regen_20260516/eth_omega461_h48qual_short_entry_veto_sustained_uptrend_20260815/report.json`
  (창별 3중 비교, veto 이벤트 전체, per-trade 렛저 diff, 강건성, G0a/G0b 결과 포함)
- 렛저: 동 디렉토리 `portfolio_ledger_<창>_odyssey3_baseline.csv` /
  `portfolio_ledger_<창>_h48qual_short_entry_veto_p90.csv`

## 다음 단계 / 미해결

- **배포/섀도우 후보로 권장하지 않는다.** 판정 창에서 사실상 무변화이고 참고 창 순효과가
  음수라, zig075판과 달리 이 확장을 추가로 밀 근거가 약하다.
- h48qual Q3 SHORT 손실의 실제 원인 진단(zig075 실행 로그 #1과 같은 형태의 dir_p_short/
  quality_for_action/MFE 분해)은 이 실행 로그에서 하지 않았다 — 하려면 별도 진단 스크립트가
  필요하다.
- Odyssey3/Odyssey4 계약 문서의 "다음 단계" 갱신은 이 결과를 보고받은 뒤 별도로 처리한다(이
  실행 로그 자체는 계약 문서를 수정하지 않는다).
