# 피쳐 엔지니어링으로 엣지 만들기 — 문헌·리포 통합 연구 (2026-08-17)

## 조사 개요

- **질문**: "내 데이터에 엣지가 없다. 고급 피쳐 엔지니어링으로 엣지를 만들려면 어떻게 해야 하는가?"
- **방법**: arXiv API + OpenAlex API 실검색(2026-08-17 접근, 재현 가능한 쿼리는 문서 말미 provenance 참조) + 이 리포의 종료된 실험 축 전수 대조.
- **자매 문서**: `docs/deep_learning_for_crypto_trading_literature_survey_20260817.md` (모델 아키텍처 축 — "피쳐가 아키텍처를 이긴다"는 결론까지 도달). 이 문서는 그 다음 질문, 즉 **"그래서 어떤 피쳐를, 어떤 데이터에서, 어떻게 만들 것인가"**를 다룬다.

---

## 1. 진단 — 왜 지금 데이터에 엣지가 없는가

이 리포가 이미 스스로 생산한 증거를 먼저 정리한다. 이것이 출발점이다.

1. **5분봉 OHLCV 파생 피쳐의 방향 예측력은 구조적으로 바닥이다.** 2026-05-28 전수 감사에서 방향 후보 220개의 family HGB probe OOS AUC 최고치가 ~0.539였고, 같은 피쳐들이 변동성 예측에서는 0.70–0.73을 기록했다 (`docs/audits/directional_feature_universe_audit_20260528.md`). 즉 **피쳐에 정보가 없는 게 아니라, 방향 정보가 없다.**
2. **라벨 변환으로는 해결되지 않는다.** 40+개 라벨 방법론(triple-barrier/zigzag/meta-label/trend-scan/DP-oracle)이 전부 엄격한 방향-알파 검정에서 실패했고, 원인은 한 번도 barrier 공식이 아니었다.
3. **증거 신호(evidence signal)는 실재하지만 항상 더 단순한 무료 벤치마크에 흡수된다.** 22개 신호 lift 랭킹(최고 3.51x)은 독립 윈도우에서 검증됐으나, 알파는 단기 리버설로, 변동성 게이트는 trailing realized vol로 환원됐다.
4. **아키텍처 교체는 전부 실패했다.** VSN/diffusion/Mamba/Transformer/TCN/TabM-R+S+B, RL 5개 삽입점 — 모두 N≥5 시드에서 종료.

5. **17종 제네릭 피쳐 아이디어 일괄 스윕도 전멸했다.** `research_line_registry.json`의 `eth_overnight_generic_feature_entry_filter_20260809`: 경로 시그니처, OFI 임밸런스(OHLCV/집계 프록시), Hawkes, BTC 리드래그, 크로스섹션 RS, Kaufman ER, 캘린더, 44-피쳐 키친싱크(in-sample AUC 0.9564 → DEV/VAL 0.5166/0.5170), conformal abstention — 17/17 음성.
6. **리포 스스로 이미 같은 결론에 도달해 있다.** quality head 정보 상한 진단(`docs/experiments/eth_h48qual_quality_new_data_source_research_20260811.md`): GBM holdout **R²≈0** (FINAL12과 201-컬럼 REL11 재스크린 모두), 결론 원문 — "문제는 현재 패널의 정보 내용이지 스칼라 추출 방법이 아니다." Knockoff FDR 통제에서도 zig075 tradeability는 **0/138 (fdr=0.10)** — 현 기술적 피쳐 유니버스로는 본질적으로 예측 불가.

**결론적 진단**: 엣지의 부재는 모델·라벨·변환의 문제가 아니라 **입력 데이터의 정보 내용(information content) 문제**다. 같은 5분 캘린더봉 OHLCV를 아무리 다르게 변환해도 그 안에 없는 방향 정보는 생기지 않는다. 이는 문헌과 정확히 일치한다(2절).

### 1.1 현재 배포 피쳐 컨트랙트 현황 (2026-08-17 인벤토리)

배포된 `eth-odyssey4-shadow`의 3개 TabM 번들은 **byte-identical한 102개 base 피쳐 + 13개 포지션 피쳐**를 공유한다 (전체 목록·구축 경로는 별도 인벤토리 조사로 확인; 원천은 `features/engineering.py::FeatureEngineer` + Regime3 HMM 오버레이 6개). 가족 구성: 원시 OHLCV/체결수 9, 파생 포지셔닝 4, BTC 크로스에셋 5, 주문류/고래 프록시 10, 추세/모멘텀 15, 변동성/점프/테일 13, 유동성 프록시 4, funding 12, 시간/세션 7, CVP 5, 프랙탈/레짐 5, 합성 6, Regime3 HMM 6, 캔들구조 1. **전부 5m OHLCV + 5m 집계 파생 데이터에서 나온다 — L2 오더북 유래 피쳐는 0개다.**

품질 관련 기록 사실 (housekeeping 참고용, 성능과는 무관 확인됨):
- 정확히 중복인 쌍 2개(r=1.0000): `smart_money_flow ≡ oi_change_rate`, `funding_z_score ≡ ou_funding_z`. 단 dedup 78-피쳐 재학습(N=5 시드)은 **PnL/스킬 무변화**로 확인됨 — 중복 제거는 엣지와 무관 (`docs/experiments/eth_omega461_dedup78feature_nseed_skill_retest_20260815.md`).
- 가격추세 오염 피쳐 9개가 raw 형태로 live에 들어가 있고, 연구에서 만든 detrended 변형은 미배선.
- live 102에 대한 완결된 per-feature importance 랭킹은 존재하지 않음(2026-07-24 permutation 아티팩트는 baseline만 기록, CSV 미산출).

---

## 2. 문헌의 수렴점 — "피쳐 > 모델", 그러나 "정보 > 변환"

- **Gu, Kelly & Xiu (2020)**, *Empirical Asset Pricing via Machine Learning*, RFS, DOI:10.1093/rfs/hhaa009 (2,394회 인용). ML의 이득은 대부분 **비선형 상호작용의 포착**이지 새 정보의 창조가 아니다. 입력 시그널 셋이 같으면 모델 간 격차는 제한적.
- **DRW 2025 Kaggle 우승자** (자매 서베이 6.3절): "피쳐 품질이 좋으면 선형 모델이 강력하다."
- **Wang (2025)**, arXiv:2506.05764 (크립토 LOB 전용): *"Better Inputs Matter More Than Stacking Another Hidden Layer"* — 층을 더 쌓는 것보다 입력 품질이 지배적.
- **Kolm, Turiel & Westray (2023)**, *Mathematical Finance*, DOI:10.1111/mafi.12413: 나스닥 115종목에서 **OFI 파생 정규화 피쳐가 원시 LOB 상태 입력을 능가** — 도메인 지식 기반 피쳐 엔지니어링이 end-to-end 표현학습을 이긴 대표 사례.
- **TLOB — Berti & Kasneci (2025)**, arXiv:2502.15757: 예측력 자체가 시간에 따라 감소(F1 −6.68pt)하고, **추세 라벨을 평균 스프레드 기준으로 재정의하면(=거래비용 반영) 엣지가 사라진다.**

이 네 갈래를 합치면 실행 가능한 명제가 나온다:

> **엣지는 (1) 남들이 안 보는/못 보는 정보 원천, (2) 그 원천에 대한 도메인 지식 기반 압축(피쳐), (3) 비용 생존이라는 3중 필터를 통과한 교집합에서만 나온다.** 피쳐 엔지니어링의 역할은 (2)이며, (1)이 비어 있으면 (2)는 아무것도 만들지 못한다.

이 리포는 (2)와 (3)의 인프라(검증 규칙, cost gate)는 이미 세계적 수준으로 갖췄고, **(1)이 병목**이다. 따라서 이 문서의 나머지는 "어떤 정보 원천에 방향 엣지가 문헌적으로 확인되어 있고, 각 원천에서 어떤 피쳐를 어떻게 계산하는가"를 원천별로 정리한다.

---

## 3. 엣지의 원천별 상세 — 문헌 증거 강도 순

### 3.1 오더북 마이크로스트럭처 (증거 강도: ★★★★★ — 최우선)

단기(초~분) 방향 예측에서 **문헌적으로 가장 일관되게 확인된 유일한 피쳐 계열**이다.

**핵심 계보:**

| 논문 | 핵심 발견 | 크립토 검증 |
|---|---|---|
| Cont, Kukanov & Stoikov (2014), *J. Financial Econometrics*, DOI:10.1093/jjfinec/nbt003 (arXiv:1011.6402) | **OFI(best bid/ask 수급 불균형)와 단기 가격변화는 선형 관계**, 기울기는 시장 깊이에 반비례. 거래량보다 OFI가 훨씬 강건 | — |
| Gould & Bonart (2015), arXiv:1512.03492 | **큐 임밸런스**(best bid/ask 잔량 비율)만으로 one-tick-ahead 방향 예측 가능 | — |
| Cartea, Donnelly & Jaimungal (2018), *Applied Mathematical Finance*, DOI:10.1080/1350486x.2018.1434009 | 오더북 신호(volume imbalance)를 실행 전략에 결합하면 성과 개선 | — |
| Cont, Cucuringu & Zhang (2023), arXiv:2112.13213 | **멀티레벨 OFI를 단일 통합 OFI로 결합**하면 best-level OFI보다 설명력 우수. **lagged cross-asset OFI는 미래 수익률 예측에 기여**(단기, 빠른 감쇠) | — |
| Kolm, Turiel & Westray (2023), DOI:10.1111/mafi.12413 | OFI 파생 정규화 피쳐 → LSTM/ANN이 원시 LOB 입력보다 우수, 다중 호라이즌 알파 | — |
| Lucchese, Pakkanen & Veraart (2022), arXiv:2211.13777 | 고빈도 중간가 예측력은 "존재하는 정도가 아니라 편재(ubiquitous)". 단 **표현(representation) 선택에 성능이 강하게 의존** — volume representation 권장 | — |
| Silantyev (2019), *Digital Finance*, DOI:10.1007/s42521-019-00007-w | **크립토(BTC)에서 OFI가 trade flow imbalance보다 가격변화 설명력 우수** — Cont 2014의 크립토 재현 | ✅ |
| **Bieganowski & Ślepaczuk (2026), arXiv:2602.00776** | **바이낸스 선물 영구계약 L2+체결, 1초 주기, 2022-01~2025-10.** 같은 엔지니어링 피쳐 셋(OFI/스프레드/adverse selection 계열)이 **시총 10배 차이 나는 5개 자산(BTC, LTC, ETC, ENJ, ROSE)에서 SHAP 중요도·부분효과 형태가 안정적으로 재현** — "크립토 범용 피쳐 라이브러리"의 실증. CatBoost + 방향인지 GMADL 목적함수 + TSCV. taker/maker 백테스트로 tradability 검증, 플래시크래시에서 adverse selection 이론 실증 | ✅ |

**이 리포의 데이터 현황 (인벤토리 확인)**: L2 계열 데이터는 생각보다 많이 이미 있다 — 전부 **모델 미소비** 상태다.

| 자원 | 축적 시작 | 상태 |
|---|---|---|
| raw L2 레벨 (top-20 `[price,qty]`, WS-E E1) | **2026-08-17** (백필 불가) | 라이브 검증 완료, 축적 초기 |
| L2 요약 컬럼 (best bid/ask, mid, `spread_bps`, microprice, `bid/ask_notional_{1,5,10,20}`, `imbalance_{1,5,10,20}`) | **2026-05-13 (~3개월)** | `orderbook_decision_snapshots`에 축적, 미사용 |
| `microstructure_1m` 34컬럼 (OBI, taker_buy_ratio, spoofing/toxicity/absorption 스코어 등) | **2026-05-03** 연속 1m | 미사용. 1m 단독 알파는 비용(0.3–2bp vs 4–9bp)으로 4회 기각 — 게이트/베토/청산타이밍 역할만 후보 |

주의: `orderbook_decision_snapshots`는 의사결정 시점 조건부 샘플링이라 활동-윈도우 편향이 있다. 그리고 이 세 자원 모두 2026-05 이후 시작이라 **캐노니컬 VAL(2025-09~12)/OOS(2026-01~03)와 겹치지 않는다** — 평가는 forward-shadow 또는 새 split 경계 명시(Fresh-Forward 규칙) 필요.

Bieganowski & Ślepaczuk (2026)은 사실상 우리가 가진 것과 같은 데이터(바이낸스 선물 L2)로 "무엇을 계산할지"의 설계도를 제공한다. raw 레벨이 exploratory 임계(09-14)에 도달하면 구현할 구체적 피쳐 후보:

1. **베스트레벨 OFI** (Cont 2014): `e_n = ΔV_bid·1[진입/유지] − ΔV_ask·1[진입/유지]`를 윈도우 합산. 깊이로 정규화.
2. **멀티레벨 통합 OFI** (Cont 2023): 레벨 1~10 OFI의 첫 주성분(PC1) 또는 깊이가중합. 단일 레벨보다 강건.
3. **큐 임밸런스** (Gould & Bonart): `(Q_bid − Q_ask)/(Q_bid + Q_ask)` — 레벨1 및 상위 k레벨 누적판.
4. **스프레드·깊이 상태**: relative spread, 상위 k레벨 총깊이, 깊이 비대칭, 깊이의 시간변화율.
5. **Adverse selection 프록시**: 체결 직후 중간가 이동(realized impact), trade-through 비율, 대형 체결 후 스프레드 반응 (Bieganowski가 SHAP로 확인한 계열).
6. **체결류(trade flow)**: 방향 서명된 체결량 불균형, 체결 강도(빈도), 평균 체결 크기의 z-score — 단 Silantyev에 따르면 OFI보다 약하므로 보조.
7. **VPIN류 독성 지표** (Easley, López de Prado & O'Hara 2012, *RFS*, DOI:10.1093/rfs/hhs053): volume bucket 단위 주문류 독성 — 방향보다는 **리스크/베토 게이트**로 (플래시크래시 선행 지표 논쟁 있음).

**경고 3건 (문헌 내장):**
- TLOB: 스프레드 기준 라벨(=비용 반영)에서 엣지 급감. **모든 L2 피쳐 실험은 처음부터 리포의 breakeven-bp 기준으로 평가**할 것 (이미 [[evidence_signal_quant_use_subproject]] 규칙).
- Lucchese: 표현 선택이 성능을 좌우 — 피쳐 자체보다 정규화·표현 설계에 시간을 쓸 가치가 있다.
- OFI 예측력의 지평은 짧다(초~수분, Cont 2023의 "빠른 감쇠"). **5분봉 진입 게이트보다 진입 타이밍/실행 레이어에 먼저 주입하는 것이 문헌 정합적**이다.

### 3.2 정보 시간 샘플링 — 캘린더봉 탈피 (증거 강도: ★★★☆ — 리포 미탐색 축)

- **Easley, López de Prado & O'Hara (2012)**, *The Volume Clock*, *J. Portfolio Management*, DOI:10.3905/jpm.2012.39.1.019: 고빈도 세계의 시간은 캘린더가 아니라 **거래량 시계**로 흐른다. 정보 도착 속도에 맞춘 샘플링이 통계적 성질(정규성, 이분산)을 개선.
- **López de Prado (2018)**, *Advances in Financial Machine Learning* (Wiley): volume/dollar/imbalance bars — 정보 도착 단위로 봉을 재구성하면 같은 원천 데이터에서도 라벨-피쳐 정렬이 달라진다.
- **Easley, López de Prado & O'Hara (2020)**, *Microstructure in the Machine Age*, *RFS*, DOI:10.1093/rfs/hhaa078: 마이크로스트럭처 피쳐가 ML 시대의 표준 입력이 되어야 한다는 종합 논거.

**이 리포에의 함의**: 이 리포의 **모든** 실험(40+ 라벨 방법론 포함)은 5분 캘린더봉 위에서 수행됐다. 라벨 방법론 축은 소진됐지만 **샘플링 축은 한 번도 시도된 적이 없다.** 주의: 이것은 마법이 아니라 재배열이다 — 방향 정보가 없는 데이터는 재샘플링해도 없다. 그러나 (a) 변동성 클러스터가 봉 경계에 정렬되어 라벨 노이즈가 줄고, (b) 20-23 UTC 같은 세션 효과([[eth_session_split_edge_2023utc_20260817]])가 거래량 시계에서는 자연스럽게 정규화된다는 점에서, **기존 5분봉 파이프라인과 동일 기간 A/B 비교하는 저비용 게이트 1회**의 가치는 있다. 필요 데이터는 aggTrades(체결 틱)이며 바이낸스에서 전 기간 무료 취득 가능 — L2 축적을 기다릴 필요가 없어 **지금 시작 가능한 유일한 신규 축**이다.

### 3.3 파생상품 포지셔닝 — funding/OI/청산/옵션 (증거 강도: ★★★☆)

- **He, Manela, Ross & von Wachter (2022)**, *Fundamentals of Perpetual Futures*, arXiv:2212.06888: 영구선물 가격 = 미래 funding 흐름의 할인 합. funding은 노이즈가 아니라 **포지셔닝 상태변수**.
- **Kim & Park (2025)**, arXiv:2506.08573: funding rate 설계가 시장별로 달라 아비트라지·평균회귀 구조가 거래소마다 다름.
- **Reconciling Open Interest with Traded Volume in Perpetual Swaps** (2023), arXiv:2310.14973: OI 변화와 체결량의 정합 분해 — OI-up-price-down 류 피쳐의 이론적 기반.
- **Alexander & Heck (2020)**, *J. Financial Stability*, DOI:10.1016/j.jfs.2020.100776: **가격발견은 무규제 파생상품 거래소(당시 BitMEX)가 주도** — 파생 포지셔닝 데이터가 현물보다 정보 우위.

**이 리포에의 함의**: funding 계열은 이미 최강 컨텍스트 피쳐군(`last_funding_rate`, `funding_roc_288`, `crowding_pressure` 등)으로 확인되어 있고, 파생 포지셔닝 축은 **부분 개척** 상태다. 인벤토리 확인 결과 하위 원천별 실상태:

- (a) **청산 스트림 — 이미 수집 중, 미사용.** `tail_risk_1m`(long/short 청산 USD, `aftershock_prob`, `liq_event_count_1m`)이 2026-05-03부터 96k+ rows 축적(`data/live/tail_risk.duckdb`, Binance `@forceOrder`). 캐노니컬 VAL/OOS와 비겹침이 평가 블로커.
- (b) **Deribit GEX** — 2026-08-15부터 수집 중, 수 주 필요, 08:00 UTC 만기 톱니 통제 필요, 과거 백필 불가 확인됨.
- (c) **거래소 간 funding 스프레드 — 수집기 존재, 소비자 0.** F4-C altdata collector가 2026-08-10부터 크로스거래소 funding 스프레드 + Fear&Greed 수집 중이나 리포 전체에 소비자가 없다. Kim & Park (2025)의 설계 차이 논거가 이 데이터의 이론적 기반.

### 3.4 크로스마켓 — 현물-선물, 거래소 간, 크로스에셋 (증거 강도: ★★☆)

- **Makarov & Schoar (2019)**, *JFE*, DOI:10.1016/j.jfineco.2019.07.001 (825회): 거래소 간 가격 괴리는 실재하고 수 시간~수일 지속되나 자본 이동 마찰이 원인 — 순수 아비트라지는 어렵지만 **괴리 자체가 수급 상태변수**.
- **Cont, Cucuringu & Zhang (2023)**: lagged **cross-asset OFI**가 미래 수익률 예측에 기여 (단기).
- **Alexander & Heck (2020)**: 파생→현물 방향의 리드가 지배적.

**이 리포에의 함의**: BTC 리드래그 피쳐는 이미 존재하고(`eth_btc_ret_spread_12` 등 ret IC ~0.058), 크로스에셋 상관 cheap-gate는 IC 부호 불안정으로 종료됐다([[eth_candidate_crossasset_correlation_cheap_gate_20260816]]). **미개척 하위 원천은 같은 자산의 시장 간 신호다**: ETH 현물-영구선물 베이시스의 고빈도 다이내믹스(현물 데이터는 무료), 바이낸스-타 거래소 중간가 리드래그. 단 Cont 2023의 감쇠 경고가 동일 적용 — 5분보다 실행 레이어 친화적.

### 3.5 경로 시그니처·고급 변환 (증거 강도: ★☆ — cheap-gate 전용)

- **Chevyrev & Kormilitzin (2016)**, arXiv:1603.03788: 경로 시그니처는 경로의 모든 기하 정보를 위계적으로 압축하는 원리적 피쳐맵.
- 실전 증거는 실행/VWAP 쪽에 편중(예: Genet 2025, arXiv:2503.02680 — signature-enhanced VWAP). **방향 알파에서 GBM 대비 우위를 보인 엄격한 크립토 증거는 검색되지 않았다** (부재가 반증은 아니나 우선순위 하향 근거).
- **리포 내부 반증이 이미 있다**: 2026-08-09 17종 스윕(`eth_overnight_generic_feature_entry_filter_20260809`)에서 경로 시그니처·Hawkes·OFI 프록시(5m/집계 기반)가 전부 음성. 즉 **5m OHLCV 위의 시그니처는 이미 기각**됐다.

**함의**: 남은 질문은 "진짜 L2 스트림 위의 시그니처가 수작업 OFI 피쳐 대비 증분이 있는가"뿐이며, 이는 OFI 피쳐 라이브러리가 베이스라인을 형성한 후에만 성립한다. 지금은 아님.

### 3.6 약하거나 이미 종료된 원천 (재제안 금지 목록과 정합)

- **센티먼트/NLP**: RAML 검토로 종료([[eth_raml_sentiment_regime_gate_literature_review_closed_20260816]]). 문헌도 룩어헤드 오염 반복 확인.
- **온체인**: 검색 결과 대부분 사기탐지/AML 용도(arXiv:2403.17081 매핑 스터디). 5분 스캘핑 지평에서 방향 증거 없음. 일봉 팩터로는 Liu & Tsyvinski (2018, NBER w24877) 네트워크 팩터가 있으나 지평 불일치.
- **캘린더봉 기술지표 재조합**: 오실레이터 합류([[eth_oscillator_confluence_closed_20260814]]), CTA류([[eth_trader_research_gex_infra_start_20260815]]), AMT/VSA/iFVG([[amt_vsa_ifvg_traders_closed_20260815]]) — 전부 종료. 문헌(918-실험 논문)도 동일 결론.

---

## 4. 피쳐 검증 방법론 — 문헌에서 추가로 배울 것

이 리포의 기존 규칙(가격추세 오염 체크 spearman<0.5–0.6, 무료 벤치마크 대비 증분, breakeven-bp cost gate, N≥5 시드, fresh-forward)은 문헌 표준을 이미 상회한다. 문헌에서 **추가로** 채택할 가치가 있는 것:

1. **크로스에셋 SHAP 안정성 게이트** (Bieganowski & Ślepaczuk 2026): 새 피쳐 패밀리는 ETH 단독이 아니라 BTC+2~3개 알트에서 SHAP 중요도 순위·부분효과 형태가 재현되는지 확인. 자산 간 재현 실패 = 오버핏 신호. (기존 60-coin cross-sectional 인프라 재활용 가능.)
2. **방향인지 목적함수 (GMADL)** (동일 논문): 부호 정확도와 크기를 함께 보는 손실 — quality head 재설계 시 참고 후보.
3. **표현 우선 튜닝** (Lucchese 2022): L2 피쳐는 정규화 방식(깊이 정규화, 틱 정규화, volume representation)을 피쳐 추가보다 먼저 스윕.
4. **호라이즌-감쇠 프로파일 명시** (Cont 2023, Kolm 2023): 새 피쳐마다 예측력의 지평별 감쇠 곡선(1/5/15/60분)을 기록 — 피쳐를 어느 레이어(실행/진입/청산)에 꽂을지 데이터가 결정하게.

---

## 5. 실행 로드맵 — 데이터 가용성 기준 우선순위 (인벤토리 반영판)

| 순위 | 축 | 근거 | 데이터 상태 | 시작 가능 시점 |
|---|---|---|---|---|
| 1 | **aggTrades 기반 정보시간 샘플링 A/B** (3.2) | 리포 유일 미탐색 변환 축, 전 기간 무료 취득 가능 → **캐노니컬 VAL/OOS 그대로 평가 가능한 유일한 신규 축** | 취득 필요(무료, `data.binance.vision`) | **지금** |
| 2 | **L2 요약 컬럼 피쳐화** (3.1) — `spread_bps`/microprice/`imbalance_{1,5,10,20}`/depth notional | **이미 ~3개월 축적, 소비자 0.** raw 레벨 임계를 기다릴 필요 없는 선행 cheap-gate. 단 decision-conditional 샘플링 편향 + VAL/OOS 비겹침 → forward 평가 설계 필수 | 축적 중 (2026-05-13→) | 지금 |
| 3 | **청산 피드 피쳐화** (3.3a) | 이미 05-03부터 수집 중, 미사용. Alexander & Heck의 파생 주도 가격발견 논거 | 축적 중 | 지금 (forward 평가 전제) |
| 4 | **raw L2 OFI 피쳐 라이브러리** (3.1) | 문헌 증거 최강, Bieganowski arXiv:2602.00776이 설계도 | WS-E E1 축적 초기 (08-17→) | exploratory 09-14, promotion 11-17 |
| 5 | **현물-선물 베이시스 고빈도** (3.4) | 현물 klines 무료, 미탐색 | 취득 필요(무료) | 지금 |
| 6 | **GEX + 크로스거래소 funding 스프레드 피쳐화** (3.3b,c) | 둘 다 수집 중(08-15/08-10→), 소비자 0 | 축적 중 | 수 주 후 |
| 7 | 경로 시그니처 cheap-gate (3.5) | 5m 판은 이미 기각; **raw L2 위에서만**, OFI 베이스라인 성립 후 | 4번 의존 | 4번 이후 |

**공통 제약 2건:**
1. 2·3·4·6번은 전부 2026-05 이후 시작 데이터라 캐노니컬 VAL(2025-09~12)/OOS(2026-01~03)에 못 얹는다. Fresh-Forward 규칙에 따라 **새 split 경계를 리포트에 명시**하고 forward-shadow 관찰로 평가해야 하며, 이는 [[eth_reversal_evidence_signal_scorecard_20260814]]류의 lift 방법론(윈도우 내 매칭 랜덤 대조)과 결이 맞다.
2. 모든 축은 리포 표준 cheap-gate(무료 벤치마크 대비 증분 + breakeven-bp + N≥5 시드) 통과 전에는 후보 지위 없음. OFI류는 지평 감쇠(3.1 경고)를 감안해 **진입 게이트 이전에 실행/청산 타이밍 레이어 주입을 우선 검토**.

---

## 6. Provenance (재현 정보)

- 접근일: 2026-08-17. 도구: arXiv API (`export.arxiv.org/api/query`, Atom, 3초 간격 직렬), OpenAlex API (`api.openalex.org/works?search=…&mailto=…`).
- 주요 쿼리: `id_list=1011.6402,2112.13213,1808.03668,2502.15757,2211.13777`; `abs:"order flow imbalance" AND abs:cryptocurrency`; `abs:"perpetual futures" AND abs:"funding rate"`; `id_list=1512.03492`; `id_list=2602.00776`; OpenAlex 검색: "flow toxicity liquidity high-frequency world VPIN", "deep order flow imbalance extracting alpha multiple horizons", "the volume clock", "microstructure in the machine age", "trading and arbitrage in cryptocurrency markets", "empirical asset pricing via machine learning", "enhancing trading strategies with order book signals" 등.
- 모든 DOI/arXiv ID는 API 응답에서 직접 추출(기억 인용 아님). 단 López de Prado (2018) *AFML*은 단행본이라 API 미검증 — 표준 서지로 인용.
- 한계: Semantic Scholar는 키 부재로 미사용(429 위험). 경로 시그니처의 "방향 알파 크립토 증거 부재"는 arXiv/OpenAlex 검색 범위 내 부재이며 전수 증명이 아님.
