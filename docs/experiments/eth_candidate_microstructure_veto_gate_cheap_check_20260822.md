# ETH microstructure_1m → zig075/h48qual 거부/게이트 재질문 — cheap-gate 체크 (2026-08-22)

상태: **cheap gate(§4) 완료 후 사용자 승인으로 단일터치 OOS-Q2까지 실행(§10) → 표본부족(N=8)으로
판정보류, OOS-Q2는 이 가설에 대해 소진됨** — 원래 요청받은 "트레이드 원장 조인" 체크는 처음엔
구조적 블로커(§3)로 막혔으나, 사용자가 같은 날 OOS-Q2 예산 사용을 명시적으로 승인해 §10에서
실제 실행했다. 결과는 확인도 반박도 아닌 표본부족 판정보류 — 승격/배포 판단 아님.

## 0. 배경 — 왜 이 재질문인가

`microstructure_1m`(ETH, 3.5개월치 1분봉 오더플로우 파생 34컬럼)은 **엔트리/방향 알파로 이미
4회 기각**됐다(§1). 이 세션에서 사용자가 강하게 요청한 "엣지를 반드시 만들어야 한다" 방향 논의
끝에, 이 데이터를 **방향 예측이 아니라 기존 진입신호(zig075/h48qual) 위에 얹는 거부/게이트**로
재질문하기로 합의했다 — 이 프로젝트에서 20개+ entry-side 실험 중 유일하게 CONFIRMED된
메커니즘(zig075 entry veto, §2)이 정확히 "모델 헤드를 안 건드리고 외부 causal 신호로 노출을
관리"하는 이 계열이었기 때문이다.

## 1. 선행사실 확인 #1 — "4회 기각"과 이번 재질문이 진짜 다른 질문인가

`docs/duckdb_live_data_utilization_design_20260719.md` 전제 #1을 직접 읽었다:

> "1m 단독 알파는 4회 기각됨. 컨트래리언 플로우 신호는 실재(t=-7.6)하지만 0.3~2bps로 비용
> (4~9bps)을 못 넘는다. BTC 병합 lookahead 수정 후 1m HGB 엣지는 완전히 소멸했다. → 이 데이터로
> '새 1m 진입 모델'을 또 만드는 것은 설계에서 제외한다."

즉 기각된 4건은 전부 **`microstructure_1m` 피처로 신규 모델을 학습해 가격 방향/수익률을 직접
예측**하는 시도였다(그 중 최소 1건은 1m HGB로 특정됨) — 목적변수가 "다음 구간 수익률의 부호"인
**direction alpha**였다.

이번 재질문의 목적변수는 다르다: "이미 zig075/h48qual이 쏜 진입 신호를, 그 순간의 microstructure
상태를 보고 거부할지"다 — 방향을 새로 예측하지 않고, 이미 나온 신호의 **부실 확률/역선택
위험(adverse-excursion risk)** 만 판별한다. 구조적으로 이건 direction alpha가 아니라 이 저장소의
유일한 CONFIRMED 계열(외부 causal 신호로 기존 신호의 노출을 조건부 관리)과 같은 종류다.

**§4의 실증 결과가 이 구분을 사후적으로도 지지한다**: 방향성 재검정(신호 부호 vs 향후 수익률
부호)은 전부 무의미로 나왔다(4회 기각과 정합) — 반면 **크기/변동성 조건부 분포**(부호가 아니라
"이 순간 이후 가격이 더 크게 움직이는가")에서는 통계적으로 실재하는 차이가 나왔다. 만약 이번
질문이 단지 기존 4회 기각 테스트의 재탕이었다면 같은 진단(방향 IC)에서 다시 무의미가 나왔어야
한다 — 대신 방향은 여전히 무의미하되 크기 축에서는 다른 결과가 나왔다는 것 자체가, 이게 정말
다른 질문임을 시사한다. **결론: 재질문은 4회 기각과 이름만 다른 재탕이 아니라 구조적으로 다른
질문이다.**

## 2. 선행사실 확인 #2 — 재사용할 CONFIRMED 선례의 실체와 한계 (반드시 재상기)

`docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md` +
`docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md` +
`docs/experiments/eth_omega461_zig075_long_entry_veto_sustained_downtrend_20260815.md`를 읽었다.

**중요**: 이 CONFIRMED 선례의 실제 신호는 **`dual_momentum`(2016-bar rolling 모멘텀 비율, p90
임계값)이었지 오더북/마이크로구조 데이터가 전혀 아니었다.** 재사용 가능한 것은 "게이트/거부
메커니즘 클래스가 이 프로젝트에서 유일하게 작동했다"는 **방법론**뿐이다:

- 신규 자유변수 0개 원칙(임계값을 사전 고정된 방식 — p90 분위수 — 으로 정하고 사후 조정 안 함)
- VAL/OOS-Q1/OOS-Q2 사전등록 판정창 구조, 단일터치
- "판정창 3개 중 1개·거래 1건"이라는 얇은 표본도 정직하게 CONFIRMED라 부르되 한계 명시

**이 CONFIRMED 선례조차 판정 3창 중 2창(VAL·OOS-Q1)은 veto가 0번 발동(무해성만 확인)했고, 단
1개 창(OOS-Q2)에서 거래 1건 스왑으로 부호반전됐을 뿐이다** — SHORT판은 그 대신 참고 티어인
2025-Q3에서 강한 효과(−15.86%→+20.17%)를 보였지만 그건 판정에 포함되지 않는 in-sample 참고
수치다. 즉 "게이트가 확인됐다"는 이 프로젝트의 최고 성공 사례조차 **평상시엔 아무것도 안 하고,
가끔(3번에 1번꼴) 거래 한두 건을 실제로 바꾸는 정도**의 개입이다. 이번 microstructure 재질문이
비슷하게 얇은 결과를 내거나 완전히 기각되더라도, 그건 실패가 아니라 이 프로젝트의 정상적인
결과 분포다 — 기대치를 그 이상으로 잡으면 안 된다.

## 3. 원래 요청받은 체크 — 트레이드 원장 조인 — 는 구조적으로 실행 불가 (핵심 발견)

### 3.1 시간 구간 비중첩

`microstructure_1m`은 **2026-05-03부터만 존재한다**(dev 사본 실측: 129,102행,
2026-05-03 17:36 KST ~ 2026-08-17 13:49 KST — 서버는 136,041행+로 앞서 더 최신이라고 확인된 바
있음, dev/server 지연은 기존에 문서화된 정상 패턴). 반면 이 프로젝트의 등록된 비-OOS 판정/참고
창은 전부 2025년 이하다:

| 창 | 기간 | microstructure_1m과 겹침 |
|---|---|---|
| 2025q1/q2/q3 (context) | 2025-01-01~09-30 | **0%** |
| val | 2025-10-01~12-31 | **0%** |
| oos_q1 | 2026-01-01~03-31 | **0%** |
| oos_q2 | 2026-04-01~06-30 | 2026-05-03~06-30만 (~60%) — **OOS, 이번엔 금지** |
| (미등록) | 2026-07-01~ | 전체 겹침이나 미등록·forward 데이터 |

즉 **"VAL 구간만, 또는 그보다 이전 참고창만 써서" 저비용 진단을 하라는 지시를 문자 그대로
따르면 대상 데이터가 0행이다** — `microstructure_1m`이 존재하는 유일한 달력 구간이 OOS-Q2
내부이거나 그보다 더 미래이기 때문이다. 이건 지시가 틀렸다는 뜻이 아니라, 지시를 설계할 때
"zig075 entry veto 선례처럼 VAL/OOS 창 구조를 재사용"하라는 템플릿이 **2025~2026 캐노니컬
데이터를 쓰는 후보에는 맞지만, 2026-05 이후에만 존재하는 이 특정 라이브 데이터 소스에는
기계적으로 대입이 안 된다**는, 실행 전에는 드러나지 않았던 사실이다.

### 3.2 OOS-Q2조차 기존 예측 아티팩트가 다 커버하지 못함

혹시나 해서 확인: h48qual/zig075의 기존 저장 예측 아티팩트(`oos_predictions_q050.csv`,
`oos_predictions_q075.csv`, `tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/`)는
**2026-01-01 ~ 2026-07-12 09:00까지만 존재**한다(직접 확인, 55,405행 각각). BASE_2026 원시
피처 파일(`training_features_2026_rebuilt.csv`)은 2026-08-19까지 있지만, 이건 모델 추론 결과가
아니라 원시 피처일 뿐이다 — 07-12 이후 구간에서 zig075/h48qual "신호가 떴는가"를 알려면 동결된
TabM/GBDT 체크포인트로 **신규 추론**을 돌려야 한다. 이건 "cheap gate"의 범위를 넘고, Omega
Artifact Integrity 정책(quality-threshold 정합 predictions 아티팩트)과도 얽힐 수 있어 이번엔
하지 않았다.

### 3.3 재사용 가능한 기존 원장 탐색 — 전부 확인, 없음

지시대로 기존 trade-level 원장 재사용 가능성을 먼저 찾아봤다:

- `tmp/ilias_labellogic_recheck_20260821/`을 확인했다 — `oos_trade_ledgers/`,
  `oos_20260701_20260819_regime3_current_states24_sticky090.csv` 등 **이 축 자체가 이미
  2026-07-01~08-19 구간을 건드린 상태**였다(별도 라벨퓨전 축, zigzag_action/cusum/h48_conservative
  기반 — Omega4.6.1 후보의 zig075/h48qual 컴포넌트 형식이 아님). 지시받은 대로 이 디렉토리의
  다른 파일은 열지 않았고, 존재만 확인했다 — 필요한 zig075/h48qual 컴포넌트 레벨 원장이 아니므로
  어차피 재사용 불가.
- 저장소 전체에서 `portfolio_ledger_*zig075*`/`*h48qual*` 파일을 전수 검색(317개 파일) —
  **전부 정확히 등록된 6개 창(2025q1/q2/q3, val, oos_q1, oos_q2) 중 하나에 속하고, 2026-06-30을
  넘어가는 파일은 단 하나도 없다.**

**결론(§3 전체)**: 요청받은 "zig075/h48qual 신호 시점의 microstructure 값 vs 그 거래의 승패"
조인 테스트는, (a) 존재하는 원장이 전부 OOS-Q2 이하 구간에 갇혀 있고 이번 세션은 OOS를
금지했으며, (b) OOS-Q2를 넘는 구간은 기존 예측 아티팩트조차 없어 신규 추론이 필요해 cheap-gate
범위를 벗어나기 때문에, **이번 세션에서는 실행할 수 없다.** 이건 이 재질문 아이디어가 죽었다는
뜻이 아니라, "지금 당장은 이 방법으로 답을 낼 수 없다"는 정직한 진단이다 — §6에서 구체적인
해제 경로 두 가지를 제안한다.

## 4. 대안 — WS-C 방식 조건부 수익률-분포 탐색 스캔 (실행)

### 4.1 방법론이 사전승인된 근거

`docs/duckdb_live_data_utilization_design_20260719.md` WS-C(청산/독성 리스크 오버레이, 이 세션
이전인 2026-07-19에 이미 작성)가 정확히 이 상황을 위해 설계해 둔 방법론을 그대로 썼다:

> "Omega4.6.1은 6개월 24 트레이드라 오버레이 효과를 트레이드 PnL로 검증 불가. 검증 지표는
> **조건부 수익률 분포**로: '톡시시티 상위 상태 직후 N분간 포지션 방향 수익률이 통계적으로
> 나쁜가'를 77일 1m 데이터 **전체**에서 검정(t, 부트스트랩). 조건부 분포 악화가 확인될 때만
> 오버레이 설계로 진행."

이 설계는 **zig075/h48qual의 OOS 트레이드 원장이나 예측 아티팩트를 전혀 쓰지 않는다** — 원시
ETH 가격의 조건부 미래-분포만 본다. 이건 "OMEGA 후보의 OOS 성과를 다시 들여다보는 것"이 아니라
**microstructure_1m이라는 데이터 소스 자체의 특성 진단**이고, 이 세션 이전에 이미 문서화된
정책이라 이번 세션의 "OOS 금지" 지시의 취지(같은 후보를 반복적으로 OOS에 노출시키는 것 방지)와
충돌하지 않는다고 판단해 실행했다. `docs/experiments/eth_candidate_liquidation_feed_features_
cheap_gate_20260817.md`도 같은 취지로 등록된 OOS 경계(oos_q2 끝 2026-06-30) 이후 구간을
"exploratory=true, 비결정" 스캔으로 쓴 선례가 있다 — 다만 이번엔 그 선례보다 더 보수적으로,
가격조건부 **분포** 테스트만 하고 트레이드 단위 승격/판정에는 전혀 쓰지 않는다.

### 4.2 설계

- 데이터: `microstructure_1m` 전체 이력(품질 필터: `data_stale=false`, `valid_taker_flow`,
  `valid_nif`, `warmup_30m_ready` 전부 true) → 123,882행, 2026-05-03 09:05 ~ 2026-08-17 04:49
  (UTC 변환 후), 100 캘린더일.
- 가격: `data/splits/year_oos/training_features_2026_rebuilt.csv`(canonical 5m close, 모델
  예측/라벨 미포함 — 순수 OHLCV 파생) merge_asof(backward, tolerance 6분)로 시점별 price_now/
  price±15·30·60분 결합. `microstructure_1m`의 `ts`는 KST tz-aware로 저장돼 있어 UTC로 변환 후
  조인(이 저장소의 기존 KST-라이브/UTC-캐노니컬 컨벤션과 일치).
- 지표: `fwd_ret_h`(부호, 방향성 재검정용), `abs_fwd_ret_h`(크기, adverse-excursion 프록시,
  h∈{15,30,60}분).
- 검정: 일(day) 단위 block bootstrap(N=2,000, seed=20260822) — 인접 1분봉의 강한 자기상관을
  고려해 개별 행이 아니라 **날짜를 통째로** 재추출(liquidation feed cheap gate와 동일 관행).
  95% CI가 0을 포함하지 않으면 유의로 판정.
- 오염 체크: `spearman(피쳐, 후행 30분 수익률/그 절대값)` — DB에 `price_change_30m`/
  `price_volatility_30m` 컬럼이 실제로는 없음을 `DESCRIBE microstructure_1m`으로 재확인(스캐너
  코드의 반환 dict에는 있지만 영속화되지 않음) → 조인한 가격 시계열로 자체 계산한 후행 30분
  수익률(`bwd_ret_30`, 순수 backward-looking)로 대체.

### 4.3 결과 — 방향은 여전히 무의미, 크기/변동성 축에서 약한 실재 신호

| 그룹 비교 (h=30분) | hi 평균 \|fwd_ret\| | lo 평균 | 차이 | 95% CI (day-block boot) | 유의 | 상대효과 |
|---|---|---|---|---|---|---|
| `shadow_toxicity_regime`=toxic vs normal | 0.2558% | 0.2346% | +0.0211pp | [+0.0130, +0.0302]pp | **예** | +9.0% |
| `shadow_queue_collapse` 상위10% (≥0.905) | 0.2808% | 0.2401% | +0.0407pp | [+0.0283, +0.0534]pp | **예** | +17.0% |
| `shadow_absorption_score` 상위10% (≥0.500) | 0.2478% | 0.2438% | +0.0040pp | [−0.0024, +0.0109]pp | 아니오 | — |
| \|`nif_whale`\| 상위10% (≥0.587) | 0.2009% | 0.2490% | **−0.0481pp** | [−0.0604, −0.0351]pp | **예(역방향)** | **−19.3%** |
| `kelly_mult` 하위10%(≤1.0, n=4,059) | 0.2448% | 0.2257% | +0.0192pp | [+0.0044, +0.0337]pp | **예(약함)** | +8.5% |

방향성(부호) 재검정 — 전부 무의미:
- toxic 레짐의 **부호 있는** fwd_ret_30: 차이 −0.0065pp, CI [−0.0135, +0.0002]pp — **무의미**
- `signal_bias`≠0 vs =0의 부호 있는 fwd_ret_30: 차이 −0.0058pp, CI [−0.0121, +0.0005]pp —
  **무의미**. signal_bias=+1일 때 평균 fwd_ret_30조차 **음수**(−0.0076%), =−1일 때도 음수
  (−0.0029%) — 방향성 스킬의 흔적조차 없음.
- \|nif_whale\| 상위10% 부분집합 내 nif_whale 부호 vs fwd_ret_30의 스피어만 IC = **−0.013**
  (사실상 0, 약한 역행 신호에 불과) — 4회 기각의 컨트래리언 패턴과 결이 비슷하나 이번엔 별도
  IC 산출을 목적으로 설계한 게 아니라 부수 확인.

**해석**: 4회 기각된 direction-alpha와 정확히 같은 결(방향 예측 불가)이 이번 데이터에서도
재확인된다 — 이건 이번 재질문이 "다른 이름의 같은 테스트"가 아님을 보여주는 증거이기도 하다
(§1). 반면 **크기/변동성 조건부 분포**에서는 `shadow_toxicity_regime`과 특히
`shadow_queue_collapse`가 day-block bootstrap 기준으로 통계적으로 실재하는(우연이라 보기
어려운) +9~17% 상대적 크기 증가를 보인다 — VPIN류 독성 지표가 "점프/변동성의 선행지표이지
방향 지표가 아니다"라는 기존 문헌 조사 결론과도 정합적이다.

**예상 밖의 결과**: `nif_whale`(고래-개미 불균형)은 가설(불균형=역선택 위험=변동성↑)과
**반대 방향**으로 유의했다 — 강한 고래 편향 직후엔 오히려 그 다음 15~60분의 절대수익률이
19% 작다. 결정적 고래 플로우가 불확실성을 해소하는 쪽(방향이 이미 정해진 뒤의 조용한 구간)에
가깝다는 해석이 가능하나, 이번 스캔은 이 가설을 검증하도록 설계되지 않았다 — 후속 확인이
필요한 열린 관찰로만 기록한다.

**주의 — 다중비교**: 6개 그룹 가설 × 최대 3개 지평을 봤다(방향성 서브체크 별도). 지평 3개는
같은 그룹분할의 파생이라 독립 검정이 아니므로 실질적으로는 ~6개 독립 가설 중 4개(toxicity,
queue_collapse, nif_whale, kelly_mult)가 다지평 일관된 유의 패턴을 보였다 — 5% 유의수준에서
순전히 우연이라면 기대되는 것보다 많지만, 엄밀한 다중비교 보정(Bonferroni 등)은 하지 않았다 —
이건 존재-신호 탐색 스캔이지 확정 판정이 아니기 때문이다.

### 4.4 오염/중복성 체크

- **오염 없음(P-gate 통과)**: `shadow_toxicity_score`/`shadow_absorption_score`/
  `shadow_queue_collapse`는 후행 30분 수익률(부호)과 거의 무상관(\|ρ\|<0.008) — 최근 추세를
  단순 재진술하는 피쳐가 아니다([[feedback_raw_feature_price_trend_contamination]] 체크리스트
  통과). 후행 절대수익률(최근 변동성)과의 상관도 0.02~0.076로 약함 — degenerate 아님.
- `nif_whale`/`nif_retail`은 후행 30분 수익률(부호)과 중간 정도 상관(0.158/0.122) — 이건
  같은 시간창 안에서 taker 매수/매도 불균형이 그 창의 가격과 기계적으로 동시 움직이는 구조적
  현상이라 우려할 오염은 아니다(방향 예측력이 아니라 동시성).
- **shadow_toxicity_score ≈ shadow_absorption_score (스피어만 0.967, 거의 중복)** — 둘 다
  OBI-vs-taker-flow 발산 계열 공식에서 파생돼 사실상 같은 신호를 두 가지로 라벨링한 것에
  가깝다. `shadow_queue_collapse`는 상대적으로 독립적(그 쌍과 0.37~0.38). 이후 이 축을 계속
  본다면 toxicity/absorption을 별도 피쳐 두 개로 취급하면 안 된다 — 사실상 하나다.

## 5. kelly_mult / signal_bias 정체 확인 (지시받은 코드 고고학)

`microstructure_scanner.py`(729~739행)에서 직접 확인:

```python
nif_bias = -1 if nif_whale < -0.30 and nif_retail > 0.10 else (1 if nif_whale > 0.30 and nif_retail < -0.10 else 0)
eai_bias = -1 if eai > self.eai_threshold and self._fund_rate > 0.0010 else (1 if eai > self.eai_threshold and self._fund_rate < -0.0010 else 0)
kelly_mult = 1.0
if nif_bias == -1: kelly_mult *= 0.40
if spoofing_score > 0.3:
    predicted_bias = spoof_bias + nif_bias + eai_bias
    kelly_mult *= 1.20 if (predicted_bias != 0 and sign(predicted_bias) == sign(spoof_bias)) else 0.70
if eai > self.eai_threshold: kelly_mult *= 1.30
kelly_mult = clip(kelly_mult, 0.30, 2.0)
raw_bias = spoof_bias + nif_bias + eai_bias
signal_bias = sign(raw_bias) if raw_bias != 0 else 0
```

- **`kelly_mult`**: 손으로 짠 규칙 기반 사이징 배율. nif_bias(고래 매도+개미 매수 다이버전스)면
  0.4배, 스푸핑 방향이 나머지 신호와 불일치하면 0.7배, EAI(가격변동 대비 OI변화 이상치)가 크면
  1.3배 등을 곱해 [0.30, 2.0]으로 clip. **한 번도 검증된 적 없는 hand-rolled 공식**이었으나,
  §4.3에서 이 규칙이 "size down"으로 판정한 하위 그룹이 실제로 후행 절대수익률이 유의하게
  높다는 것을 처음으로 확인했다(+8.5%, 약한 신호) — 설계 의도와 방향은 맞지만 검증 근거는
  이번이 처음이다.
- **`signal_bias`**: `sign(spoofing_bias + nif_bias + eai_bias)`, 세 규칙 기반 서브바이어스의
  합의 부호. 방향성 서브체크(§4.3)에서 예측력 없음 확인 — signal_bias=+1일 때조차 평균 후행
  수익률이 음수였다.
- **결정적 발견 — 둘 다 실거래 의사결정 경로에서 사용되지 않는다.** `trading_bot.py`에서
  `kelly_mult`/`signal_bias`를 참조하는 곳은 정확히 2곳(11680~11681행, 14418~14419행)뿐이고,
  둘 다 `state["microstructure"] = {...}` 대시보드/상태 스냅샷 dict에 값을 채워 넣을 뿐이다.
  **`trading_bot_modules/omega4_6_1_live.py`(Omega4.6.1의 실제 라이브 의사결정 엔진)는
  `kelly_mult`/`signal_bias`/`microstructure`를 단 한 번도 참조하지 않는다** — grep 결과 0건.
  즉 이 두 컬럼은 "이미 게이팅/사이징에 쓰이는 검증된 신호"가 아니라, **한 번도 실전에서
  검증되지 않은 채 대시보드 표시용으로만 계산되고 있던 죽은 파생값**이었다. 사용자가 예상한
  "이미 답을 담고 있을 수도 있다"는 가설은 부분적으로만 맞았다 — 답을 담고 있던 게 아니라
  **한 번도 답으로 검증되지 않은 채 방치된 값**이었고, 이번이 그 첫 검증이다.

## 6. 데이터 품질/스코프 확인 (부수)

- `mark_price`/`whale_position_score`: dev 사본 129,102행 중 각각 95,443행(74.0%)/95,851행
  (74.2%) NULL — 사용자가 관찰한 대로 대부분 죽은 컬럼에 가깝다. 이번 분석에서 사용하지 않았다
  (대신 canonical 5m close를 조인).
- `oi_delta_pct`/`funding_rate` vs CLAUDE.md가 언급한 캐노니컬 정확중복쌍(`smart_money_flow`≡
  `oi_change_rate`): `features/engineering.py` 217~221행에서 직접 확인 — 이 캐노니컬 쌍은 둘 다
  `df['sum_open_interest_value'].pct_change().clip(-1,1).fillna(0)`로 **완전히 같은 줄**이다.
  이건 5분봉 OHLCV 파생 OI 컬럼 기반의 **캐노니컬 피처 파이프라인** 산출물이고,
  `microstructure_1m`의 `oi_delta_pct`(라이브 웹소켓 1분 폴링 기반, `microstructure_scanner.py`
  자체 계산)와는 소스·주기·계산식이 전혀 다르다. **같은 데이터의 재탕이 아니다** — 이 재질문은
  캐노니컬 154피쳐셋이 이미 커버한 정보를 다시 게이트로 파는 게 아니라는 뜻이다.
- 품질 플래그: `data_stale` 0.53%, `valid_taker_flow` 99.48%, `valid_nif` 99.38%,
  `warmup_30m_ready` 96.57% — 데이터 자체는 건강하다.
- `signal_bias` 분포: −1(9.4%)/0(81.6%)/+1(9.0%) — 균형적. `kelly_mult`: [0.30, 1.56] 관측
  (설계 상한 2.0 미도달), 평균 0.987, std 0.119 — 저분산이지만 degenerate는 아님.
  `shadow_toxicity_regime`: normal 35.5%/watch 34.2%/toxic 30.3% — 3구간 균형적, degenerate
  아님.

## 7. 종합 판단 및 권고 (승격 판단 아님)

**(a) 4회 기각과 다른 질문인가** — 예, 구조적으로 다르다(§1). 방향 예측이 아니라 크기/변동성
조건부 게이트를 묻는 질문이고, §4의 실증 패턴(방향 무의미·크기 유의)이 이 구분을 사후
지지한다.

**(b) kelly_mult/signal_bias 정체** — 손으로 짠 규칙 기반 파생값, 실거래 의사결정에 전혀
연결 안 됨(대시보드 전용, 죽은 값), 이번이 첫 검증. signal_bias는 방향 무예측력 확인, kelly_mult
는 크기축에서 약한 정합성 처음 확인.

**(c) cheap gate 결과** — **애매/약한 양성**. 원래 요청된 트레이드 원장 조인 테스트는 구조적
블로커로 실행 불가(§3, 이 자체가 이번 체크의 핵심 발견). 대체 시행한 WS-C 조건부 분포 스캔
(§4)에서 `shadow_toxicity_regime`/`shadow_queue_collapse`가 day-block bootstrap 기준 통계적
으로 실재하는 (그러나 경제적으로 작은, +9~17%) 크기/변동성 신호를 보였고, `shadow_absorption_
score`는 toxicity와 96.7% 중복임에도 자체 임계값에서는 유의하지 않았으며, `nif_whale`은
가설과 반대 방향으로 유의했다(고래 플로우 후 오히려 차분해짐) — 열린 관찰. 방향성은 전 축에서
무의미.

**(d) 권고 — 계속 볼 가치는 있으나 지금 만들 단계는 아니다.**
1. **지금 하지 말아야 할 것**: 이 신호로 실제 게이트/베토를 구현하는 것. §4는 "이 데이터 소스가
   완전히 죽지는 않았다"는 존재-신호 확인일 뿐, "zig075/h48qual 특정 트레이드의 승패를 개선
   한다"는 증거가 아니다 — 그 증거를 낼 방법(§3)이 이번 세션엔 막혀 있었다.
2. **다음에 이 축을 열려면(구체적 두 경로)**:
   - **경로 A(권장, 가장 저비용)**: OOS-Q2(2026-04~06)에 대한 zig075/h48qual 예측 아티팩트와
     포트폴리오 원장은 **이미 존재**한다(`tmp/causal_regen_20260516/eth_omega461_zig075_*_entry_
     veto_*/portfolio_ledger_oos_q2_*.csv` 등, 신규 추론 불필요) — `microstructure_1m`과의
     겹침은 2026-05-03~06-30(oos_q2의 약 60%)뿐이지만, 이 세션이 아닌 **다음 세션에 사용자가
     이 특정 가설에 단일터치 OOS-Q2 예산을 쓰기로 명시적으로 결정하면** 가장 적은 신규 계산으로
     실제 트레이드-레벨 검증(승/패, MFE/MAE vs microstructure 상태)이 가능하다. 이번 세션은
     그 예산을 다른 축(일리아스 라벨퓨전)에 이미 썼으므로 실행하지 않았다.
   - **경로 B(자연 해소, 느림)**: `docs/duckdb_live_data_utilization_design_20260719.md`가 이미
     정한 원칙(3개월 누적 후 학습/판정 해금)과 `eth_recency_walkforward_data_split_literature_
     review_20260820` 메모가 언급한 "다음세대 TRAIN~06-30, 새 OOS 2026-07-01~09-30(09-30까지
     대기)" 개편이 확정되면, `microstructure_1m`의 2026-05~06 구간이 정식 TRAIN/참고창 안에
     들어와 OOS를 건드리지 않고도 조인 테스트가 가능해진다 — 09-30 이후 자연 해소.
3. **`shadow_toxicity_score`/`shadow_absorption_score`는 중복이므로 둘 다 병행 피쳐로 쓰지
   말 것**, `shadow_queue_collapse`가 상대적으로 더 독립적이고 이번 스캔에서 가장 강한 상대
   효과(+17%)를 보였다는 점을 다음 시도의 우선순위 참고로 남긴다.

## 8. 준수 확인

`fresh_forward_bar_by_bar`=해당없음(트레이드 replay 아님, 순수 조건부 분포 통계),
`trade_ledgers_used_as_input=false`(§4는 원장을 전혀 읽지 않음, §3.3에서 원장은 "존재 확인"만
하고 내용은 열지 않음), `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`(대상 자체가 zig075/h48qual 엔트리가 아니라 원시 가격
조건부-분포). **oos_predictions_*.csv, portfolio_ledger_oos_q1/oos_q2_*.csv 등 어떤 OOS
결과물도 이번 세션에서 읽지 않았다** — 존재(파일명)만 `ls`로 확인, 내용 미조회.
Seed-Diversity Ensemble Promotion Gate: 해당없음(재학습 모델 없음, 순수 통계 스캔).
Omega Artifact Integrity Gate: 해당없음(신규 predictions 아티팩트 없음).

`git status` 확인: 이 문서와
`docs/model_contracts/eth_candidate_lob_microstructure_contract_20260817.md`(신규 절 추가)만
변경. `trading_bot.py`/`omega4_6_1_live.py`/`runtime_config.py`/`.env` 등 라이브 경로 무변경.

## 9. 정직한 한계

1. **핵심 가설(microstructure 상태가 zig075/h48qual 트레이드 승패를 가르는가)은 이번 세션에
   검증되지 못했다** — §3의 구조적 블로커 때문. §4는 필요조건(신호가 존재하는가) 점검일 뿐
   충분조건(그 신호가 이 특정 후보의 트레이드 품질을 개선하는가) 검증이 아니다.
2. §4의 표본은 3.5개월(100일)뿐이고, 데이터 소스 자체가 단일 레짐일 가능성을 이미 문서(WS
   설계)가 경고했다 — 이 결과가 다른 시장 국면에서 재현될지 불명.
3. 다중비교 보정을 하지 않았다(§4.3 주의 문단) — 6개 그룹 가설 중 4개가 유의했다는 것 자체는
   고무적이지만 엄밀한 통계적 확정은 아니다.
4. `nif_whale`의 역방향 결과는 사후 관찰이지 사전 가설이 아니었다 — 확증이 필요하다.
5. `shadow_absorption_score`가 toxicity와 96.7% 중복이면서도 자체 임계값에서 유의하지 않은
   이유(같은 정보의 다른 파라미터화가 다른 유의성을 내는 것)는 설명하지 않았다 — 열린 질문.
6. §7(d) 경로 A는 이번 세션에 실행하지 않은 제안일 뿐, 사용자 승인 없이 진행하면 안 된다
   (단일터치 OOS 규율).

## 10. ⚠️⚠️ 단일터치 OOS-Q2 실제 실행 결과 (2026-08-22, 같은 날 후속, 사용자 승인)

§7(d) 경로 A를 사용자가 명시적으로 승인해 **같은 날 실행했다 — OOS-Q2는 이 가설에 대해 이제
소진됐다(단일터치, 재조회 금지).**

**방법**: `tmp/causal_regen_20260516/eth_omega461_zig075_short_entry_veto_sustained_uptrend_
20260814/portfolio_ledger_oos_q2_odyssey3_baseline.csv`(zig075 SHORT veto 실험 디렉토리의
baseline, LONG veto 디렉토리의 `..._long_.../portfolio_ledger_oos_q2_odyssey4_baseline.csv`와
바이트 동일함을 확인 — 같은 기저 포트폴리오) 재사용, 신규 추론 없음. OOS-Q2 전체 13개 트레이드
(zig075 9건+h48qual 4건, 진짜 라이브 컴포넌트 우선순위 포트폴리오) 중 `entry_timestamp`가
`microstructure_1m` 커버리지(2026-05-03~08-17) 안에 드는 **8건**만 조인 가능했다(나머지 5건은
04-01~04-14 진입이라 이 데이터 소스 자체가 존재하기 전). `merge_asof`(backward, tolerance
10분)로 진입 직전 최신 microstructure 상태를 붙였다 — 8건 전부 10분 이내 유효값 존재.
사전위원 후보(§4.3에서 이미 지목한 것만): `shadow_toxicity_score`, `shadow_queue_collapse`,
`shadow_absorption_score`, `nif_whale`, `kelly_mult`, `signal_bias` — 이번엔 이 6개만 보고
새로 낚시하지 않았다.

**결과(N=8, 승3/패5)**:

| 피쳐 | 승 평균 | 패 평균 | trade_return과 스피어만(N=8) |
|---|---:|---:|---:|
| shadow_toxicity_score | 0.4466 | 0.4529 | −0.119 |
| shadow_queue_collapse | 0.2462 | 0.5443 | −0.333 |
| shadow_absorption_score | 0.1565 | 0.1594 | −0.119 |
| nif_whale | −0.1885 | +0.0773 | −0.048 |
| kelly_mult | 0.80 | 1.06 | −0.218 |
| signal_bias | −0.333 | 0.000 | −0.412 |

**정직한 판정 — 결정 불가(확인도 반박도 아님)**: N=8은 어떤 방향으로도 통계적 결론을 낼 수
있는 표본이 아니다(spearman 표준오차가 이 N에서 매우 커서, 진짜 상관이 0이어도 ±0.3~0.5대
표본상관이 흔하게 나온다 — p값을 계산하지 않았다, 계산해도 무의미하기 때문). `shadow_queue_
collapse`(−0.333)와 `signal_bias`(−0.412)가 §4.3의 방향(값이 클수록/신호가 있을수록 안 좋음)과
정성적으로는 일치하지만, 이 정도 N에서는 우연과 구분 불가능하다. **결론: "이 게이트가 실제
트레이드 품질을 개선한다"는 핵심 가설은 이번 단일터치로도 검증되지 못했다** — 확인되지도,
반박되지도 않았다. 사용 가능한 유일한 판정창(OOS-Q2 겹침 구간)의 표본 크기 자체가 근본적
한계였다(§9 한계 1 참고, "충분조건 검증 실패" 예측이 실측으로 확인됨).

**의도적으로 하지 않은 것(단일터치 규율 보호)**: 결과가 애매하게 나온 뒤 표본을 늘리려고
BTC/SOL 자매 포트폴리오나 다른 판정창을 추가로 열지 않았다 — 그건 사후에 유의한 결과가 나올
때까지 표본을 바꿔가며 찾는 것과 다를 바 없어 이번 단일터치의 취지를 훼손한다. 더 큰 표본이
필요하면 §7(d) 경로 B(09-30 이후 자연 해소)를 기다리거나, 다음 세대 OOS가 쌓인 뒤 재시도해야
한다.

**다음 단계**: 이 특정 가설(microstructure_1m → zig075/h48qual 진입 게이트)은 이 세션 기준
"계속 지켜볼 근거도 접을 근거도 부족, 표본 부족으로 판정 보류"로 종결한다. 재개하려면 반드시
더 큰 판정 표본(다음 OOS 세대, 또는 09-30 이후 확정되는 TRAIN 확장)이 먼저 필요하다 — 지금
있는 데이터로 또 다른 조인 방식을 시도하는 것은 표본 문제를 풀지 못한다.

## 11. 산출물

- 스크립트(scratchpad, 저장소 외부):
  `/tmp/claude-1000/-home-kbj20-crypto-scalping/4f5e786b-a26d-4f77-b741-1f3c50e13159/scratchpad/microstructure_gate_cheap_check_20260822.py`
- 결과 JSON(scratchpad):
  `/tmp/claude-1000/-home-kbj20-crypto-scalping/4f5e786b-a26d-4f77-b741-1f3c50e13159/scratchpad/microstructure_gate_result.json`
- 이 문서 + `docs/model_contracts/eth_candidate_lob_microstructure_contract_20260817.md` 갱신
  (다음 절 추가)
- (2026-08-22 §11 추가) OOS-Q2 조인 스크립트:
  `/tmp/claude-1000/-home-kbj20-crypto-scalping/4f5e786b-a26d-4f77-b741-1f3c50e13159/scratchpad/oos_q2_microstructure_entry_check.py`,
  결과: `tmp/eth_candidate_microstructure_oos_q2_entry_check_20260822.csv`(저장소 tmp/, 8행)
