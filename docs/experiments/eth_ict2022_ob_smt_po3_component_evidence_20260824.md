# ICT 2022 잔여 3요소 증거연구 — 오더블록 · SMT 다이버전스 · Power of 3

- 일자: 2026-08-24
- 상태: 사전등록 → 실행
- 요청: 사용자 — ICT 2022류 기법 방향 검토에서 "미시도로 남은 세 조각을 시도해보자"
- registry id: `eth_ict2022_ob_smt_po3_component_evidence_20260824`
- 성격: **증거연구(retrospective lift)이지 백테스트/승격 주장이 아님** — 기존 22신호
  스코어카드·AMT/VSA/iFVG 연구와 동일 하네스(`docs/experiments/eth_amt_vsa_footprint_ifvg_strategy_absorption_study_20260815.md`
  의 자매 실행)로, 트리거 bar에서 실제 zigzag 피벗까지의 전방 lift를 잰다. 피벗을 전방으로
  보는 것은 이 연구설계 자체이며 fresh-forward 규칙 위반이 아님(스코어카드 모(母)메모리 참조).

## 선행 이력 (이 3개가 "잔여"인 이유)

ICT 구성요소 기측정: 유동성 스윕 3.01x/2.78x(전체 1위급, 대시보드 편입), FVG 터치
0.88x/0.90x(무효), iFVG 0.48x/0.58x(역예측), 스윕+iFVG 콤보 0.65x(스윕 단독보다 악화),
킬존류 세션 타이밍(비용게이트 실패). 미측정으로 남은 것: 오더블록(OB), SMT 다이버전스
본연형, Power of 3(Judas swing). 사전확률: 셋 다 가격파생이라 낮음 — 목적은 ICT 축을
영수증과 함께 완결하는 것.

## 데이터·하네스 (실행 전 고정)

- ETH: `data/eth_5m_1year.csv` — **UTC-naive 실측 확정**(2026-08-24, ETHUSDT-5m-api.csv와
  동일 타임스탬프 종가 100% 일치, KST 가설 0.01% — 타임존 버그 전례에 따른 필수 선검증).
- BTC(SMT용): `binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv` (UTC-naive, ~2026-08 커버).
- 윈도우: VAL 2025-09-01~12-31 + OOS 2026-01-01~02-17 풀링(자매 연구와 동일 — 기존 lift
  수치와 직접 비교 가능성 확보), 보조로 VAL/OOS 분리 일관성도 보고.
- 지표: `compute_indicators`(atr_price/atr_pct), `load_zigzag_pivots`, `event_study`,
  `excess_move`, K_HORIZONS {1h=12, 4h=48, 8h=96} 전부 기존 모듈에서 임포트(재구현 금지).
- 스윕 정의 재사용: 48-bar causal swing (`add_sweep` 임포트) — F2b/F3의 비교 기준.

## 신호 정의 (사전등록, 변형 탐색 금지)

**F1 오더블록 미티게이션 터치** (ICT: 변위 직전 반대 캔들 존 회귀)
- 강세 OB 형성: bar i가 음봉(close<open) AND close[i+3] > high[i] AND
  (close[i+3]−close[i]) ≥ 1.0×atr_price[i]. 존 = [low[i], high[i]], bar i+4부터 활성,
  수명 48 bar(FVG 관례 일치), bar 몸통종가 < low[i]면 그 bar부터 무효(발화 제외).
- 발화(바텀 신호): 활성 존과 bar 레인지 교차(low[j] ≤ high[i] AND high[j] ≥ low[i]) —
  E1_fvg_touch와 동일한 터치 관례. 약세 OB 미러.

**F2 SMT 다이버전스** (상관자산 스윙 비확인) — 사전등록 서브폼 2개(그리드 아님, AMT 연구의
B1/B2/B3 다중 컴포넌트 관례):
- F2a_smt_raw(바텀): ETH low < ETH 48-bar prior swing low AND BTC low > BTC prior swing low
  (ETH만 저점 갱신 = 다이버전스, ICT 본연형 — 되돌림 미요구).
- F2b_smt_sweep_gated(바텀): ETH sweep_low(관통+종가복귀, 기존 정의 그대로) AND BTC 비확인.
  **결정적 비교: 같은 실행에서 plain sweep_low lift를 나란히 보고** — BTC 게이트가 3.01x를
  올리는가(못 올리면 "컨펌 쌓기 희석" 3번째 사례). 탑 미러.
- BTC 정렬: timestamp inner 기준 left-join, BTC 결측 bar는 발화 불가 처리.

**F3 Power of 3 / Judas swing** (세션 앵커 스윕)
- 아시아 레인지: 같은 UTC 날짜의 00:00~06:55 bar들(완결 요구: 아시아 bar ≥48개)의
  running min/max — 07:00 이후 동결, 인과 계산.
- 발화(바텀): UTC 07:00~13:55 bar j에서 low[j] < asia_low AND close[j] > asia_low
  (아시아 저점 사냥 후 복귀). 탑 미러. bar 단위 발화(기존 관례) + 고유 일수 병기.

**오버랩 점검(의무)**: F2b·F3는 가장 가까운 기측정 신호(plain sweep)와 bar 단위 오버랩
비율을 보고 — 높은 오버랩 + lift 비개선이면 "신규"가 아니라 재라벨링으로 판정
(AMT 연구 How-to-apply 규칙).

## 판정 기준 (사전등록)

- **대시보드 후보**: 풀링 lift ≥ 3x AND n ≥ 100, 양사이드 또는 한사이드 일관 + VAL/OOS
  부호일관 — cvd/vwap 편입 선례 절차로 재량보조 신호 제안.
- **중간 기록**: 1.5x~3x 일관 — 스코어카드 등재만, 액션 없음.
- **CLOSED**: <1.2x 또는 VAL/OOS 비일관 또는 (F2b/F3) plain sweep 대비 비개선.
- 자동매매 재해석 금지: lift가 나와도 "확률 이동 맥락"이지 진입 트리거 아님(top-6 공식
  0/36 전례가 이 계열 전체에 적용됨).

## 결과 (2026-08-24 실행)

스크립트: `scripts/analyze_eth_ict2022_ob_smt_po3_component_evidence_20260824.py`,
산출 테이블: `tmp/eth_ict2022_ob_smt_po3_component_evidence_20260824/`. 윈도우 48,853 bar,
피벗 3,044개, BTC 정렬 커버리지 100%.

### 풀링 lift (1h 호라이즌, 바텀/탑)

| 신호 | n (바텀/탑) | lift 바텀 | lift 탑 | VAL→OOS (바텀) | sweep 오버랩 |
|---|---|---:|---:|---|---:|
| F1 오더블록 터치 | 14,136 / 14,637 | **1.07x** | **1.14x** | 1.02→1.18 | 1.4% |
| F2a SMT raw | 952 / 1,092 | **3.12x** | **2.84x** | 3.34→2.56 | 60.7% |
| F2b SMT 스윕게이트 | 578 / 617 | 2.97x | 2.69x | 3.16→2.50 | 100%(⊂sweep) |
| F3 Po3/Judas | 334 / 374 | 1.34x | **0.96x** | 1.50→**1.05** | 12.3% |
| (기준) plain sweep | 1,257 / 1,124 | 3.01x | 2.78x | 3.14→2.69 | — |

### 판정 (사전등록 기준 적용)

- **F1 오더블록: CLOSED** (<1.2x). 근본 원인이 정량화됨: 1×ATR 변위 기준의 기계적 OB가
  224k bar에서 **35,237개 형성**(13 bar당 1개) — 존이 차트를 뒤덮어 윈도우 bar의 29%가
  "존 터치" 상태, 발화≈배경. SMC 비판론의 "어디에나 존이 있다"가 그대로 실측됨. FVG 터치
  0.88x와 같은 계열(존-회귀 개념 2연속 무효).
- **F2b SMT 스윕게이트: CLOSED** — 구성상 sweep의 진부분집합(오버랩 100%)인데 부분집합
  정밀도(37.2%)가 전체 sweep(37.6%)보다 낮음 = BTC 비확인 게이트는 스윕 위에서 **무가치**.
  "컨펌 쌓기는 희석하거나 무효"(Yush 사다리, 스윕+iFVG에 이은) **3번째 실측 사례**.
- **F3 Po3/Judas: CLOSED** — 바텀 VAL 1.50x가 OOS 1.05x로 붕괴(비일관), 탑은 0.96x로
  랜덤 이하. 스윕과 오버랩 10~12%뿐 = 세션앵커(아시아 레인지) 레벨은 롤링 스윙 레벨과
  다른 대상이고, 정보는 **롤링 스윙 쪽에만** 있음. 킬존류 세션 타이밍 기각과 정합.
- **F2a SMT raw: 사전등록 "대시보드 후보" 기준 문자상 충족** (바텀 3.12x ≥3x, n=952,
  VAL/OOS 일관; 탑 2.84x는 중간기록 구간). **단 정직한 성격 규정**: plain sweep 대비
  정밀도 +1.5pp(39.1% vs 37.6%)는 1 SE(~1.6pp) 이내 = **개선이 아니라 동급**. 실체는
  "종가복귀(reclaim) 필터와 BTC-비확인 필터가 가짜돌파 판별에 대등하며, 서로 ~40% 다른
  bar를 잡는다" — sweep의 **보완형(대체 아님)**. 켜지는 시점이 다르므로(되돌림 전 발화
  가능) 재량 보조 신호로 편입할 가치는 있으나 기존 스윕 신호의 상위호환은 아님.

### 종합

ICT 2022 분해는 이것으로 완결: **살아남는 알맹이는 "직전 스윙 레벨의 스탑 사냥"
단 하나**고(plain sweep 3.01x + 그와 동급인 SMT raw), 존 회귀(OB·FVG), 컨펌 게이트
(iFVG·BTC 다이버전스 게이트), 세션 앵커(킬존·Po3)는 전부 그 알맹이에 아무것도 더하지
못하거나 랜덤 이하다. 자동매매 재해석 금지(top-6 공식 0/36 전례 적용) — F2a 포함 전부
"확률 이동 맥락" 등급.

---

## 후속 라운드 (2026-08-24, 같은 날) — 교차패밀리 조합: ICT 기하 × 주문흐름 클라이맥스

- 요청: 사용자 "ICT/SMC 구성요소들을 조합해서 최고의 신호를 만들 순 없을까?"
- 선행 증거: **같은 패밀리(가격기하)끼리 AND는 3전 3패**(Yush 사다리, 스윕+iFVG 0.65x,
  스윕+BTC게이트 희석 — 본 문서 F2b). 유일한 조합 성공 패턴은 **이종 정보패밀리**:
  orthogonal_combo(가격위치 오실레이터 × 테이커 클라이맥스) 3.51x = 오실레이터 단독 대비
  +54% (22신호 스코어카드). ICT 최강 기하신호(스윕/SMT raw)에 이 패턴을 적용한 것은 미측정.

### 사전등록 (실행 전 고정)

- 조합 2개만, 그리드/윈도우 변형 금지, 같은 bar AND(orthogonal_combo 관례):
  - **C1_sweep×flow**: 바텀 = sweep_low ∧ delta_z≤−2 / 탑 = sweep_high ∧ delta_z≥+2
  - **C2_smt×flow**: 바텀 = smt_raw_bottom ∧ delta_z≤−2 / 탑 = smt_raw_top ∧ delta_z≥+2
- delta_z는 원본 수식 그대로 재사용(`analyze_eth_creative_reversal_evidence_signals_20260814.py`):
  delta = 2×taker_buy_base − volume, z = rolling288(min_periods=288) — 기존 2.75x/3.51x와
  비교 가능성을 위해 min_periods 관례 포함 원본 유지.
- 같은 실행 안에 부모 3개(sweep, smt_raw, taker climax 단독) 동시 측정 — 동일 저울 비교.
- **판정**: 성공 = 조합 lift > max(부모 lift) AND n≥100 AND VAL/OOS 부호일관.
  n<100 = 표본부족(주장 불가). lift ≤ max(부모) = 컨펌희석 4번째 사례로 종결.
- 성공해도 등급은 "재량 보조 맥락"(0/36 자동화 전례 유지) — 트리거 재해석 금지.

### 결과 (같은 날 실행)

스크립트: `scripts/analyze_eth_ict_geometry_x_flow_climax_combo_20260824.py` (부모 3개 동일
실행 내 측정), 테이블: `tmp/eth_ict_geometry_x_flow_climax_combo_20260824/`.

| 신호 (1h, 풀링) | 바텀 lift (n) | 탑 lift (n) | 바텀 VAL→OOS |
|---|---:|---:|---|
| P_sweep | 3.01x (1,257) | 2.78x (1,124) | — |
| P_smt_raw | 3.12x (952) | 2.84x (1,092) | — |
| P_flow_climax | 2.75x (1,194) | 2.29x (1,056) | — |
| **C1 sweep×flow** | **3.42x (147)** | 2.72x (72) | 3.85→2.36 |
| **C2 smt×flow** | **3.45x (199)** | 2.67x (166) | 3.70→2.77 |

**판정 (사전등록 기준)**:
- **바텀: 두 조합 모두 PASS** — 최고 부모를 초과(3.42/3.45 vs 3.01/3.12), n≥100, VAL/OOS
  부호일관. 정밀도 42.9%/43.2%.
- **탑: 전패** — C1은 n=72 표본부족, C2는 2.67x < 부모 2.84x(희석). 기존 Wyckoff 비대칭
  (탑은 모든 신호가 약함) 재확인.

**핵심 발견 — 정밀도 천장 ~43%**: 서로 다른 재료의 바텀 조합 3종이 전부 같은 곳에 수렴 —
orthogonal_combo(오실레이터×flow) 43.9%, C1(스윕×flow) 42.9%, C2(SMT×flow) 43.2%.
"1h 내 진짜 피벗" 예측의 정보 한계가 이 부근이라는 강한 시사 — 레시피를 더 바꿔도 이 위로
뚫릴 전망은 낮다. 기존 챔피언(3.51x)을 **동률로 따라잡은 것이지 넘은 게 아님**
(부모 대비 +4.1pp 정밀도도 ~1.2 SE로 개별 유의 아님 — 단 두 조합·양윈도우 방향 일관).

**조합 규칙 정제**: "컨펌 쌓기는 희석"(같은 패밀리, 이제 4전 4패 — C2 탑 포함)과
"이종 패밀리 AND는 바텀에서 유효"(orthogonal_combo, C1, C2 — 3전 3승)가 공존. 성패를
가르는 건 개수가 아니라 **정보패밀리 상이성 + 바텀 사이드 여부**.

**후속 옵션 (사용자 결정)**: C2 smt×flow 바텀은 대시보드-후보 관례(≥3x·n≥100·일관)를
충족 — F2a와 함께/대신 편입 가능. 단 flow 레그가 orthogonal_combo와 동일하므로 기존
신호와 부분 중복 — 추가 시 "SMT 계열 1개만" 권장. 자동매매 재해석은 여전히 금지
(0/36 전례, 등급은 재량 보조 맥락).
