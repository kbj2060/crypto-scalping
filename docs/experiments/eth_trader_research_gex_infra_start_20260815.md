# "모델에 적합한 트레이더" 조사 확장 — GEX 실시간 수집 착수 (2026-08-15)

상태: **연구 완료, 실행 시작. 백테스트 결과 아직 없음(신규 라이브 수집이라 데이터 누적 필요).**

## 요청

Yush/AMT/VSA/iFVG 조사([[eth_yush_orderflow_strategy_absorption_study_20260815]],
[[eth_amt_vsa_footprint_ifvg_strategy_absorption_study_20260815]])에 이어: "yush와 비슷한
오더플로우 계열이 아니어도 돼. 내 모델에 적합하다고 생각하는 트레이더를 찾아서 조사해줘."

## 1단계 — 기존 시도 전수 조사 (재파생 방지)

새 후보를 추천하기 전, 이 저장소가 이미 무엇을 시도했는지 Explore 에이전트로 전수 조사했다.
결과: **방향성 타이밍 계열 트레이더 철학은 거의 전부 이미 닫혀 있었다.**

| 계열 | 상태 | 근거 |
|---|---|---|
| Donchian 돌파 진입(터틀식) | 근거 없음(lift 0.89~1.06배) | `eth_broad_evidence_signal_sweep_20260814.md` Category B, `eth_evidence_signal_ranking_stability_mar_jul_2026_20260814.md`에서 재확인 |
| ATR 트레일링스탑 exit(Clenow/Carver식) | "분산/MDD 레버일 뿐 PnL 레버 아님", ETH·BTC 둘 다 0/6 | `research_eth_omega461_btc_trailing_stop_val_oos_20260807.py` 등, memory `project-trailing-stop-risk-lever-keep-alive-20260807` |
| 변동성타겟 사이징(Carver식) | 치명적(cost1 -38%, cost3 최대 -83%), 전부 `gate_pass=False` | `replay_omega6_v2_variants_20260704.py`, 레지스트리 `global_vol_targeting`: "재테스트는 새 검증된 리스크 신호 필요" |
| 레짐조건부 사이징 | VAL/OOS 변동성비 0.097→6.49로 부호반전, 사이징 근거 불안정 | `btc_regime_sizing_timeliness_riskchannel_20260808.json` |
| Dalton/VSA/footprint/iFVG(오더플로우 인접) | 4개 전부 근거없음/역신호, 유일한 양성후보(Dalton Rule 2)도 비용게이트 0/6 | [[eth_amt_vsa_footprint_ifvg_strategy_absorption_study_20260815]], [[eth_dalton_rule2_balance_edge_costgate_20260815]] |
| 펀딩비(리버설/캐리/플립) | 전부 신화 파괴 | `eth_deep_evidence_signal_sweep_round2_20260814.md` |
| DVOL(Deribit 집계 IV 레벨) | BTC quality_head에 0/9 | 레지스트리 `btc_dvol_feature_overlay` |
| Fear & Greed 심리지표 | 부정 결과(홀드아웃 비교로 확정) | `eth_fear_greed_backfill_direction_relevance_20260812.md` |

## 2단계 — 남은 축: 딜러 감마 포지셔닝(GEX) 트레이더

**Cem Karsan(Kai Volatility Advisors)** — 옵션 딜러 헤지 메커니즘을 정보원으로 쓰는 전략가.
핵심 개념(SqueezeMetrics가 창안한 **GEX, Gamma Exposure**): 딜러 순감마가 **양(+)** 이면
딜러가 상승엔 팔고 하락엔 사서 변동성을 억제(레인지·pinning 유리). **음(-)** 이면 딜러가 가격
방향을 쫓아가며 변동성을 증폭(추세·모멘텀 유리).

**왜 이 모델에 맞다고 판단했나**: 지금까지 리버설 신호도, 추세추종 진입도 각각 단독으로는
전부 죽었다. GEX는 방향을 맞히는 신호가 아니라 **"둘 중 어느 체제(레인지 vs 추세)를 써야
하는가"를 판정하는 레짐 분류기**다 — 이는 이 저장소가 실제로 검증에 성공한 유일한 패턴
(h48qual = 방향이 아니라 필터, [[h48qual_standalone_replay_invalid]])과 구조적으로 같은
계열이다. 저장소 자체 기록도 이 후보를 "아직 안 죽은 차별점"으로 남겨뒀다(아래 3단계).

## 3단계 — 기존 문서와의 통합 (재검증 아님, 이어받기)

이 축은 완전히 새로운 발견이 아니다. `docs/experiments/
eth_h48qual_quality_new_data_source_research_20260811.md`의 **후보 5**가 2026-08-11에 이미
동일한 개념(옵션체인 스큐/OI/GEX)을 제안했고, **동일한 인프라 문제(과거 옵션체인 조회
불가)로 이미 "보류" 상태였다** — 이번 세션에서 이 사실을 뒤늦게 발견하고 독립적으로
재확인했다(같은 API 호출로 같은 결과: `get_instruments?expired=true`가 어제자 38건만 반환).
**교훈**: 새 후보를 제안하기 전 관련 기존 문서를 더 꼼꼼히 먼저 읽었어야 했다 — 낭비는
크지 않았으나(curl 몇 번) 같은 사실을 두 번 발견한 셈이다.

## 4단계 — 인프라 재확인 + 외부 대안 조사

- Deribit 공개 API: `get_instruments(expired=true)`는 최근 1~2일치만 반환, 과거 그릭스/OI
  히스토리 엔드포인트 없음 — 확인됨(2026-08-11 결론과 동일).
- CryptoDataDownload: "만기별 전체 옵션체인 zip" 광고하지만 실제 무료 티어는 DVOL 지수
  OHLC뿐, 진짜 체인은 **Plus+ 유료**.
- Tardis.dev: 무료분은 **월 1일치만** — VAL+OOS 5개월에 데이터 포인트 6개, bar 단위 방법론에
  못 씀.
- **결론: VAL(2025-09~12)/OOS(2026-01~02) 구간 과거 GEX 복원은 무료로 불가능. 유료
  구독(Tardis.dev/Amberdata)이나 오늘부터 실시간 수집만 남는다.**

## 5단계 — 사용자 결정 및 실행

두 차례 확인 질문 결과: **GEX 인프라 구축 시작(과거 백필 대신 실시간 수집)** 선택.

### 구축한 것

- `scripts/collect_deribit_option_gex_20260815.py`: `get_book_summary_by_currency`(ETH+BTC,
  통화당 1콜)로 라이브 옵션체인 스냅샷 수집. Gamma는 ticker 그릭스 호출(종목당 1회, 700+회/
  스냅샷) 대신 **Black-Scholes로 직접 계산**(r=0, mark_iv를 sigma로 — Deribit ticker의
  `interest_rate=0.0` 관행과 동일). GEX = Σ[gamma·OI·contract_size·S²·0.01], 콜(+)/풋(-) 부호
  (SqueezeMetrics식 단순화, **실제 딜러 포지셔닝은 미검증** — docstring에 명시).
- `data/live/deribit_gex.duckdb`: `option_chain_snapshot`(원시 종목별 행) +
  `gex_summary`(통화당 집계 — total/front_month(≤30일) GEX, spot, 종목수).
- `scripts/run_deribit_gex_collector.sh` + 크론 등록(`0 * * * *`, 매시 정각) — 로그
  `data/research/deribit_gex_collector_cron.log`.

### 첫 스냅샷 검증(퇴화 여부만 확인, 후보5 문서의 "가장 싼 검증 스텝" 그대로)

| 통화 | 종목수 | front_month 종목수 | spot | total GEX | front_month GEX |
|---|---:|---:|---:|---:|---:|
| ETH | 694 | 292 | $1,881.3 | $5,380,856 | $1,678,442 |
| BTC | 818 | 330 | $63,043.9 | $33,512,507 | $17,323,709 |

값이 0이 아니고 콜/풋이 서로 상쇄되는 정상적인 형태 — **퇴화(상수) 아님** 확인. 후보5
문서가 미검증으로 남겨둔 "(1) ETH 옵션 유동성/OI가 스큐·GEX 계산에 충분한가"는 며칠~몇 주
데이터가 쌓인 뒤 재확인이 필요하다(첫 스냅샷 하나로는 판정 불가).

## SOL 확인 (2026-08-15 후속)

사용자가 ETH·BTC·SOL "모두" 요청. 확인 결과 **Deribit에 SOL은 spot 2건만 있고 선물·옵션이
전혀 없다**(`get_instruments?currency=SOL&kind=any`가 `{'spot': 2}`만 반환, `get_currencies`
목록엔 SOL이 있지만 옵션 자체가 개설 안 됨). 즉 SOL은 옵션만이 아니라 **Deribit 파생상품
자체가 없어** GEX 계산이 구조적으로 불가능 — 컬렉터 코드 문제가 아니라 시장 자체의 한계다.
컬렉터는 이미 `CURRENCIES = ("ETH", "BTC")`로 가능한 자산 전부를 수집 중이라 추가 변경 없음.

## 알려진 한계 / 다음 단계

- **아직 신호도 백테스트도 아니다.** 이 문서는 인프라 착수 기록이지 성과 주장이 아니다.
- GEX 부호 컨벤션은 실제 딜러 포지셔닝을 관측한 게 아니라 업계 표준 단순화 가정이다.
- Deribit 옵션은 inverse-settled(ETH/BTC 결제)인데 이 스크립트는 표준 GEX 공식을 그대로 써서
  달러 표기로 근사했다 — 엄밀한 inverse-contract 보정은 하지 않았다(명시적으로 disclosed).
- 몇 주 데이터가 쌓이면: (1) ETH 옵션 유동성 충분성 재확인, (2) GEX 부호 전환과 실현변동성의
  관계를 그때부터 시작되는 기간에 대해서만 causal하게 검증(과거 구간 백테스트는 여전히 불가),
  (3) `evidence_signal_quant_use` 계약의 비용게이트를 그대로 적용.
- 크론이 이 dev 머신에 등록됐다 — 서버가 아니라 dev에 있으므로 dev 세션이 꺼져 있어도 OS
  크론으로 계속 수집된다(별도 프로세스 상주 불필요).

## 수집 상태 점검 (2026-08-15 17:01 KST, 착수 ~2시간 후)

크론 정상 등록·정시 발화 확인(16:00·17:00 KST 2회 + 착수 당시 수동검증 2회, 통화당 총 4라운드,
에러 0건). `data/live/deribit_gex.duckdb` 정상 성장 중(`option_chain_snapshot` 5,960행). 아직
신호 판단은커녕 패턴을 볼 수 있는 데이터량이 전혀 아니다(day 0) — 예정대로 몇 주 더 필요.

**발견 및 원인 확정 — 일일 옵션 만기 아티팩트**: 17:00 스냅샷에서 ETH/BTC 종목수가
front_month에서만 각각 -46/-42로 급감, GEX도 크게 출렁였다(ETH front_month -56%, BTC
front_month -85%, total_gex도 ETH -21%/BTC -48%). 16:00→17:00 사이 사라진 종목을 직접
diff해서 확인한 결과, 사라진 종목 전부(ETH 46개·BTC 42개) `expiration_ts`가 정확히
`2026-08-15 17:00:00+09:00`(=08:00 UTC, Deribit 일일 옵션 표준 만기 시각) 하나로 일치 —
데이터 결함이 아니라 **매일 08:00 UTC마다 그 날 만기 종목군이 통째로 빠지면서 생기는 톱니형
아티팩트**임을 확정했다. 몇 주 뒤 GEX-실현변동성 인과검증 시 이 일일 계단현상을 만기
롤오버로 식별·보정(또는 최소한 통제)하지 않으면 진짜 레짐 신호로 오인할 위험이 있다 —
후속 분석에서 반드시 감안할 것.

## 산출물

- `scripts/collect_deribit_option_gex_20260815.py`
- `scripts/run_deribit_gex_collector.sh`
- `data/live/deribit_gex.duckdb` (매시 갱신)
- `data/research/deribit_gex_collector_cron.log`
