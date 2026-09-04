# ETH taker_delta_z_climax 메타라벨링 — v4 최종 확정 (2026-08-29)

## 배경

증거신호 8개(`scripts/live_evidence_signal_dashboard_20260823.py::compute_signals()`) 중
`liquidity_sweep`은 별도 세션에서 V자반등(V_REBOUND) 메타라벨링 프로젝트로 이미 완료됨
(`docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md`, Tier0(22)+rsi(23)
피쳐, TabPFN, VAL 0.642/OOS 0.657/홀드아웃 0.647). 이 문서는 나머지 7개 신호 중 **첫 번째로
완료된 `taker_delta_z_climax`**의 전체 기록이다 — 방법론 재사용 템플릿과 남은 6개 신호의
진행 순서를 포함해 다음 세션이 바로 이어갈 수 있도록 정리한다.

**메타라벨링이란(이 프로젝트의 정의)**: 기존 증거신호는 그대로 두고(재구현하지 않음, trigger
역할), 그 신호가 발동했을 때 "이번 발동이 맞을지"를 별도 모델로 예측하는 2단계 구조
(López de Prado meta-labeling). 신호 자체의 방향은 이미 결정돼 있고, 모델은 신뢰도만 얹는다.

## 최종 채택 결과 (v4)

| 지표 | VAL (2025-09~12) | OOS (2026-01~03) | 홀드아웃 (2026-04~) |
|---|---|---|---|
| AUC | 0.622 | 0.608 | 0.650 |
| Accuracy | 0.595 | 0.582 | 0.622 |
| Balanced Accuracy | 0.584 | 0.562 | 0.608 |
| Naive majority | 0.542 | 0.565 | 0.539 |

- 발동 수: 10,233건(클러스터 앵커링 후, bottom 5,288 / top 4,945)
- 상태값: `exploratory_single_signal_below_promotion_bar` (V자반등과 동급 — 4-독립기간
  부호일치 계약은 시도하지 않음, V자반등도 마찬가지)
- 리포트: `tmp/eth_taker_delta_climax_metalabel_tabpfn_20260829/report.json`
  (`adopted_version: "v4"` 필드로 명시)
- 피쳐 CSV: `data/labels/eth_5m_taker_delta_climax_metalabel_20260829/
  eth_5m_taker_delta_climax_metalabel_features.csv`
- 최종 라벨 스크립트: `scripts/research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py`
  (이 파일이 실행 가능한 v4 코드 그대로 — 재실행하면 위 숫자가 그대로 재현됨, 확인됨)
- 라벨 시각화: `tmp/eth_taker_delta_climax_metalabel_20260829/label_examples_v4_final.png`
  (`scripts/render_eth_5m_taker_delta_climax_metalabel_examples_20260829.py`)

### 정답 라벨(v4) 정확한 정의

```
발동 = compute_signals()의 bottom_taker_delta_z_climax(delta_z<=-2) 또는
       top_taker_delta_z_climax(delta_z>=2), 2024-01-01~ 전체 이력, 클러스터 앵커링 적용
클러스터 앵커링 = 같은 side로 3봉 이내 연속 발동은 하나의 burst로 묶고, 그 중 delta_z가
                 가장 극단적인 봉 하나만 대표로 채택(미래가격 안 봄, 인과적)
entry = 발동봉(대표봉) 자신의 종가
HORIZON = 24봉(2시간)
MFE_pct = bottom: (fire+1~fire+24 구간 intrabar 최고가 - entry)/entry
          top:    (entry - fire+1~fire+24 구간 intrabar 최저가)/entry
hit = MFE_pct >= 2.0 * atr_pct_at_fire   (터치 기반, 지속성/유지 조건 없음)
```

피쳐(23개, 전부 klines+taker_buy_base만으로 계산, 인과적):
`is_bottom, delta_z, atr_pct, atr_percentile_864, hour_utc, weekday, nyse_open_flag, p_fast,
p_slow, ret3_z, vwap_dev_z, cvd_roll_roc_48, vol_z, lower_wick_ratio, upper_wick_ratio, bb_pctb,
adx14, pdi, ndi, bb_width_pctile, er_24, realized_vol_ratio, rsi`

모델: `TabPFNClassifier(device="cuda", random_state=seed)`, 시드 4개
(20260829/141592/271828/577215), 서버(`llewyn@192.168.0.232`) GPU에서만 실행 가능.

분할(이 저장소 Fresh-Forward 기본값과 V자반등 분할 그대로 재사용):
TRAIN=2024-01-01~2025-08-31, VAL=2025-09-01~2025-12-31, OOS=2026-01-01~2026-03-31,
HOLDOUT=2026-04-01~현재(단일 평가).

## 버전 히스토리 (v1 → v5, 왜 각각 실패/성공했는지)

| 버전 | 라벨 정의 | VAL | OOS | 홀드아웃 | 결과 |
|---|---|---|---|---|---|
| v1 | 1h/종가부호만, 로지스틱 | – | 0.489 | – | **NULL** |
| v2 | 30분/종가/0.3×ATR, TabPFN+23피쳐 | 0.509 | 0.511 | 0.551 | 약한 양(+) |
| v3 | 2h/MFE(intrabar)/2.0×ATR | 0.635 | 0.596 | 0.623 | **급등** |
| **v4** | v3 + 클러스터앵커링 | **0.622** | **0.608** | **0.650** | **채택** |
| v5 | v4 + 종료시점 지속성 체크 | 0.562 | 0.561 | 0.606 | **기각**(악화) |

### v1→v2: 왜 sign-only 라벨이 실패했나
`hit = fwd_ret_1h의 부호만 일치`로 정의하면 "hit"의 60%가 0.5% 미만의 노이즈 크기 움직임
(10분위수=0.072%)이었음 — 사실상 동전던지기 크기 잡음을 승패로 세고 있었다. ATR 스케일
임계값(0.3×ATR)을 추가하니(HORIZON은 30분으로 축소) 약하지만 일관된 양의 신호로 개선.

### v2→v3: 왜 윈도우를 2시간으로 늘리고 MFE로 바꿨나
발동봉 기준 ±2시간 넓게 검색해서 "진짜 극값이 어디 있나" 재보니: **발동봉이 정확히 극값인
경우는 14%뿐, 70%는 발동봉 이후에 진짜 극값이 옴**(median 지연 4봉/20분, p90 22봉/110분,
그 사이 추가 역행폭 median 2.9×ATR). 즉 taker_delta_z_climax는 꼭지/바닥에서 정확히
발동하기보다 "그리로 가는 도중"에 자주 발동한다. 30분/종가단일시점 평가는 이 지연을 못
따라갔음. V자반등이 실제로 쓴 방식(`future["high"].max()`, 고정 윈도우 내 MFE — 순환탐색
아님, 윈도우 길이는 사전 확정)을 그대로 적용해 HORIZON=24(2h, p90 근거)+MFE로 전환,
K=2.0으로 재보정 → VAL/OOS/홀드아웃 전부 0.60~0.64로 급등.

### v3→v4: 검증 3종 (전부 클리어)
1. **클러스터 앵커링**: burst 내 최극단 delta_z 봉만 앵커로 채택(미래가격 안 봄, 인과적) —
   중복표집 1.3배(24% 재발동) 해소. 타이밍 정렬 자체는 미미하게만 개선(at-fire-bar 14.0%→
   15.6%, median lag는 4봉으로 불변) — 이것만으론 문제 안 풀림, 하지만 오버샘플링 해소가
   본목적.
2. **룩어헤드 감사**: `compute_indicators`/`add_creative_indicators`/`add_broad_indicators`
   (+헬퍼 `_adx`/`_dmi`) 전체를 한 줄씩 직접 읽음 — `.rolling()`/`.ewm()`/`.diff()`/
   `.shift(양수)`만 사용, `.shift(-N)`/역순인덱싱 0건. 라벨의 미래봉 사용(`high/low[fire+1:
   fire+25]`)은 피쳐 구성(발동봉까지만)과 완전 분리 확인.
3. **변동성 편중 ablation**: permutation importance에서 `atr_percentile_864`가 압도적 1위
   (+0.035, 2위의 거의 2배)였으나, 변동성 관련 3피쳐(atr_pct/atr_percentile_864/
   realized_vol_ratio)를 통째로 제거해도 AUC 하락폭은 전 구간 0.01~0.012뿐 — 신호가 변동성
   레짐 하나에 몰빵된 게 아니라 20개 피쳐에 고르게 분산돼 있음을 확인.

클러스터 앵커링 후 재검증한 결과가 v3보다 오히려 소폭 개선(OOS/홀드아웃)됐다는 건, v3의
급등이 중복표집 아티팩트가 아니었다는 뜻으로도 읽힌다.

### v4→v5: 지속성 체크 시도 → 기각 (중요한 반면교사)
V자반등은 "터치했는가"뿐 아니라 "그 되찾은 레벨을 6봉 내내 유지했는가"라는 지속성 조건이
있었다(스윕이 되찾은 구체적 가격 레벨이 있어 가능). taker_delta_z_climax는 레벨 기반 이벤트가
아니라 이 조건이 없었음 — 실제로 v4의 "터치" 중 17.6%가 윈도우 끝(정확히 fire+24봉)까지
완전히 되돌려져 있었다(진단: 삭제된 스크래치패드 `calibrate_v5_persistence.py`, 재현 필요시
로직은 아래 "재현 방법" 참조).

**시도**: `hit = touched AND end_ret_pct>0`(fire+24봉 "그 한 봉"의 종가가 여전히 진입가 대비
유리한 방향). **결과: AUC가 오히려 전 구간 하락**(0.622/0.608/0.650 → 0.562/0.561/0.606).

**원인 진단**: `end_ret_pct`는 정확히 한 봉(fire+24)의 종가만 보는 **단일시점 평가**다 —
v1/v2를 망쳤던 것과 정확히 같은 종류의 단일시점 노이즈를 지속성 체크를 통해 다시 끌어들인
셈. "터치했는가"는 발동시점 피쳐로 어느 정도 예측 가능했지만, "정확히 그 순간에도 여전히
유리한가"는 그 자체로 예측 불가능한 노이즈에 가까워 예측 가능한 신호를 희석시켰다.

**교훈(다음 신호 적용 시 유의)**: "더 엄격한 라벨 = 더 나은 결과"가 항상 성립하지 않는다.
직관적으로 타당해 보이는 정제가 실제로는 노이즈만 추가할 수 있다 — 반드시 실측 비교 후
채택할 것. 지속성 체크를 다시 시도한다면 **단일 봉이 아니라 여러 봉의 평균/과반** 방식으로
스무딩해야 함(이번 세션에서는 테스트 안 함).

## 재사용 가능한 방법론 템플릿 (남은 6개 신호에 적용)

### 데이터/피쳐 빌더 (klines 전용, 전체 2024~2026 이력 확보)
```python
sys.path에 ROOT, ROOT/scripts 추가 후:
from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators
from live_evidence_signal_dashboard_20260823 import compute_signals

raw = pd.read_csv("binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv", parse_dates=["timestamp"])
frame = compute_indicators(raw); frame = add_creative_indicators(frame); frame = add_broad_indicators(frame)
# + ret3_z(인라인), atr_pct/atr_percentile_864(자체계산), hour_utc/weekday/nyse_open_flag,
#   er_24/realized_vol_ratio, rsi(Wilder-14,인라인) -- 전부 이 스크립트에 이미 구현됨, 복붙 가능
```
이 3개 함수 + 위 추가 피쳐로 23개 피쳐가 나온다. **klines(binance_data/klines/ETHUSDT/
ETHUSDT-5m-api.csv, 2023-12-31~현재, 갭/결측 0)만 쓰므로 전체 2024~2026 이력 사용 가능** —
microstructure_1m 등 duckdb 피쳐(2026-05-03부터만 존재)로는 이 이력 확보가 불가능했음.

### 라벨 설계 체크리스트 (신호별로 반드시 재검증할 것 — 그대로 복붙 금지)
1. **원시 hit rate와 호라이즌 민감도부터 실측**: 여러 호라이즌(15min/30min/1h/2h/4h)에서
   raw hit rate가 어떻게 변하는지 먼저 보고, 노이즈 vs 진짜 신호 감쇠 패턴을 구분.
2. **크기 분포 확인**: "hit"의 상당수가 노이즈 크기(예: <0.5%)면 ATR 스케일 임계값 필요.
3. **발동봉이 실제 극값과 얼마나 어긋나는지 직접 측정**(±2h 넓은 창에서 argmax/argmin 위치
   확인) — 신호마다 다를 수 있음, taker_delta_z_climax는 median 20분 지연이었지만 다른
   신호는 다를 수 있음. 이 실측으로 HORIZON/MFE 필요성을 결정.
4. **연속발동 클러스터링 확인**: 같은 신호가 여러 봉 연속 발동하면 클러스터 앵커링 고려.
5. **지속성 체크를 추가하고 싶다면 단일시점이 아니라 스무딩된 형태로**(v5의 교훈).
6. **10개 예시를 캔들 이미지로 시각 검증**(V자반등 방식 그대로 — `render_eth_5m_sweep_v_
   rebound_label_examples_20260829.py` 참고, ±윈도우, HIT/NO_HIT 각 10개, 진입가+목표선
   표시) — 숫자만으로 놓치는 문제를 실제로 여러 번 잡아냈음(이 신호에서도 v5의 "찍고
   되돌림" 문제를 이미지로 먼저 확인함).

### 모델 사다리
Step 0(naive) → Step1(단일 임계값) → **TabPFN 전체 23피쳐**(V자반등/taker_delta_z_climax
둘 다 TabPFN이 GBM/로지스틱보다 확실히 나았음, GBM은 얕은 신호에서 조기종료로 언더핏되는
경향 관찰됨). 서버(`llewyn@192.168.0.232`, quant_ai conda env, CUDA 확인됨, TabPFN 8.5.0
로컬추론 — TABPFN_TOKEN 불필요, `TabPFNClassifier(device="cuda")`로 바로 동작).

### 필수 검증 3종 (v4에서 확립, 매 신호마다 반복)
1. 룩어헤드 감사 — 재사용 함수 내부를 실제로 한 줄씩 읽고 `.shift(-N)`/역순인덱싱 확인
   (신뢰만 하고 넘어가지 말 것 — 이번에 직접 읽어서 처음으로 검증함).
2. permutation feature importance (hand-rolled, TabPFN엔 sklearn permutation_importance
   래퍼가 안 맞을 수 있어 직접 구현 — 이 스크립트의 `compute_permutation_importance()`
   재사용 가능).
3. 상위 피쳐(특히 변동성/레짐 계열)를 제거한 ablation — 신호가 한 피쳐군에 몰빵됐는지 확인.

## 남은 6개 신호 진행 순서 (제안)

1. ~~`liquidity_sweep`~~ — 완료 (V자반등, 다른 세션)
2. ~~`taker_delta_z_climax`~~ — **완료 (이 문서, v4 채택)**
3. **`short_term_return_z`** (다음 추천) — 가장 단순(ret3_z 단일 임계값 ±2.5), 이미 계산된
   피쳐라 파이프라인 이식이 가장 빠름
4. `volume_wick_climax` — 단순(vol_z>=2 AND wick_ratio>=0.5), 역시 기존 피쳐 재사용
5. `dalton_rule2_balance_edge` — SIGNAL_ORDER 자체 설명에 "실제 VAL/OOS 안정적 lift 있었으나
   고정 TP/SL economic gate만 실패"라고 명시돼 있어 메타라벨링 성공 가능성이 상대적으로 높을
   수 있음. 저변동성 레짐 게이트+48봉 레인지 로직 필요(중간 복잡도).
6. `orthogonal_combo` — 플래그십 신호(펀딩/오실레이터 등 여러 다운스트림 조합에 재사용됨)지만
   자체가 3~4개 조건 결합이라 발동조건 자체의 "발동봉=극값" 여부가 다를 수 있음, funding_df
   없어도 동작(bottom leg가 delta_z-only로 안전하게 축소됨).
7. `smt_divergence` — BTC 교차자산 데이터 필요(`binance_data/klines/BTCUSDT/` 이미 존재,
   추가 난이도 낮음).
8. `fib_extension_exhaustion` — 가장 마지막 추천: SIGNAL_ORDER 자체 설명에 "표본이 얇음
   (n~190)"이라고 명시돼 있어 메타라벨링에 필요한 표본 자체가 부족할 위험이 큼. leg-detection
   로직(48봉 causal argmin/argmax)도 가장 복잡.

## 하지 않은 것 / 캐비엇

- **4-독립기간(2024/2025H1/2025H2/2026) 부호일치 계약**(`docs/model_contracts/evidence_
  signal_quant_use_contract_20260815.md`)은 시도하지 않음 — V자반등도 마찬가지로 이 계약을
  충족하지 않았고, TRAIN/VAL/OOS/HOLDOUT walk-forward 구조 자체가 이 프로젝트 등급에 맞는
  검증 수준이라고 판단(V자반도 이 방식으로만 검증됨).
- **대시보드 배포**: 이 문서 작성 시점엔 taker_delta_z_climax가 연구 단계 산출물만 있었음 →
  **2026-08-30 배포 완료**(다른 세션, short_term_return_z와 함께). V자반등처럼 신규 칩을
  추가하는 대신 **기존 증거신호 칩 자체를 교체**(발동조건 불변, 발동 시 확률만 표시)하는
  방식을 사용자가 지정 — 아래 예측했던 "frozen train-context 재적합" 서빙 패턴 자체는 그대로
  재사용됨(`scripts/live_evidence_signal_metalabel_20260829.py`). 상세:
  `docs/homer/README.md` "배포 방식" 섹션.
- **v5의 스무딩된 지속성 체크는 미시도** — "마지막 몇 봉 평균/과반" 방식은 설계만 하고
  테스트 안 함, 재시도 여지 있음.
- `data/ensemble/reports/`에는 아무것도 쓰지 않음(진단 전용 산출물은 전부 `tmp/`,
  `data/labels/eth_5m_taker_delta_climax_metalabel_20260829/`에만 저장) — 라이브 승격
  레지스트리 무변경.
- `dashboard/server.py`/`app.js`/`trading_bot.py`/`trading_bot_modules/` 전부 무변경(이 문서
  작성 시점 기준 — `dashboard/server.py`/`app.js`는 위 2026-08-30 배포로 이후 변경됨,
  `trading_bot.py`/`trading_bot_modules/`는 여전히 무변경).

## 파일 목록

- `scripts/research_eth_taker_delta_climax_metalabel_phase0_20260829.py` — round 1(NULL,
  klines 10피쳐+로지스틱). 폐기되지 않고 남겨둠(교훈 보존).
- `scripts/research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py` — **최종 v4 코드**
  (round 2~5 전부 이 파일 안에서 반복 수정됨, docstring에 전체 버전 히스토리 기록).
- `scripts/research_eth_taker_delta_climax_metalabel_ablation_vol_20260829.py` — 변동성
  피쳐 ablation (이미 실행 완료, `tmp/.../ablation_vol_regime_report.json`).
- `scripts/render_eth_5m_taker_delta_climax_metalabel_examples_20260829.py` — **v4 최종**
  라벨 시각화 스크립트(재실행 가능).
- `tmp/eth_taker_delta_climax_metalabel_20260829/label_examples_v4_final.png` — 최종 이미지.
  `label_examples_v3.png`/`label_examples_v5.png`는 과거 버전 기록으로 남아있음(폐기된 시도).
- `tmp/eth_taker_delta_climax_metalabel_tabpfn_20260829/report.json` — 최종 v4 리포트
  (`adopted_version: "v4"` 필드로 명시).
- `data/labels/eth_5m_taker_delta_climax_metalabel_20260829/eth_5m_taker_delta_climax_
  metalabel_features.csv` — v4 발동+피쳐+라벨 전체 데이터(10,233행, 재학습/재검증에 재사용
  가능).
- 메모리: `eth_taker_delta_climax_metalabel_phase0_20260829.md`(round1, NULL, 이 문서로
  연결됨), `eth_taker_delta_climax_metalabel_v4_final_20260829.md`(최종, 이 문서 참조).
