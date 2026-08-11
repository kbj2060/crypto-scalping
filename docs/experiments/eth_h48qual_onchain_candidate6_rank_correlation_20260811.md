# ETH h48qual — 신규 데이터소스 후보 6(CoinMetrics ETH 온체인) 순위상관 진단 (2026-08-11)

## 배경

[신규 데이터소스 리서치](eth_h48qual_quality_new_data_source_research_20260811.md)의 8개 후보 중
1~3번(마이크로구조 toxicity, 청산 캐스케이드, Polymarket)을 먼저 시도했으나, 실제 라이브 수집
데이터가 전부 2026-05-03 이후(Polymarket은 2026-04-21~04-30 9일치)만 있어 기존 VAL(2025-10-01~
2025-12-31)/OOS(2026-01-01~02-28) 예측과 전혀 안 겹친다는 게 확인됐다(사용자 확인 후 5·6번으로
전환). 5번(Deribit 옵션 스큐/GEX)도 라이브로 확인해보니 `get_book_summary_by_currency`는
현재 스냅샷만 주고 `get_instruments(expired=true)`는 만기 계약 메타데이터만 줄 뿐 과거 특정
시점의 스큐를 조회하는 API가 없어 보류 — 이 문서는 6번(ETH 온체인)만 다룬다.

## 방법

### 1단계 — 다운로드

`scripts/download_coinmetrics_onchain_eth_20260811.py`(BTC 전용 원본
`download_coinmetrics_onchain_20260804.py`은 건드리지 않고 ETH용으로 신규 작성). CoinMetrics
Community API(무료, 인증 불필요)에서 일봉 6개 지표 — `AdrActCnt`(활성주소), `CapMVRVCur`(MVRV
비율), `FlowInExNtv`/`FlowOutExNtv`(거래소 유입/유출), `SplyExNtv`(거래소 보유 공급량),
`TxCnt`(트랜잭션 수). BTC 원본 리스트에 있던 `HashRate`는 라이브 확인 결과 ETH가 2022-09 The
Merge로 PoS 전환해 전부 null이라 제외. 2025-01-01~2026-08-10, 587일 다운로드 완료.

### 2단계 — 순위상관 진단 (`scripts/diagnose_eth_h48qual_onchain_rank_correlation_20260811.py`)

이슈 8/candidate C와 동일 방법론: `dir_action`(게이트 전) 기준 pre-gate 트레이드 시뮬레이션,
진입 시점에 온체인 신호를 인과적으로 붙여(`merge_asof(direction='backward')`, entry_date-1일
이하 값만 사용 — 일봉 데이터라 최소 24시간 지연) `spearmanr(신호, trade_return)`. h48orig(5시드)·
h384 v2(15시드) 양쪽, VAL/OOS 전부.

**캐비어트**: CoinMetrics 응답에 `<metric>-status: "flash"`가 붙는다 — 무료tier는 리비전
히스토리를 안 줘서, 이 진단은 "그 날짜의 최종 확정값"에 가깝고 "그 시점에 실제로 게시돼 있던
값"과 정확히 같다는 보장은 없다.

### 3단계 — 오염도 직접 측정 + detrend 재검증 (사용자 지적으로 촉발)

`scripts/diagnose_eth_h48qual_onchain_capmvrv_detrend_20260811.py`. 2단계에서 `CapMVRVCur`가
전 조합(h48orig/h384 × VAL/OOS)에서 방향이 흔들리지 않는 유일한 지표였는데, "MVRV 비율은 느리게
움직이는 평가지표라 장기 레짐/가격추세 프록시일 뿐일 수 있다"는 지적이 있었다. 이 레포엔 이미
같은 패턴의 선례가 있다 — FINAL12 dedup에서 `whale_retail_ratio`가 `corr(close)=+0.561`로 오염
판정돼 detrend 버전(`whale_retail_ratio_dt288`)으로 교체됨(`eth_h48qual_final12_feature_
selection_20260811.md`). 같은 기준으로 (1) 일봉 `spearmanr(지표, close)`/`spearmanr(지표,
시간순번)`을 VAL/OOS 구간에서 직접 측정하고, (2) `CapMVRVCur`의 전일대비 변화(`diff1`)와 7일
변화율(`roc7`) 두 detrend 버전으로 2단계와 동일한 순위상관을 재실행했다.

## 결과

### 2단계 — 6개 지표 원본 레벨 순위상관 (요약, 전체 시드별 수치는 `tmp/eth_h48qual_odyssey_
regression_analysis_20260811/onchain_rank_correlation.csv`)

| 지표 | h48orig VAL | h48orig OOS | h384 VAL | h384 OOS |
|---|---:|---:|---:|---:|
| AdrActCnt | rho=-0.137, p=0.012(음성 유의) | rho=+0.180, p=0.013(양성 유의) | 비유의 | rho=+0.137, p=0.001 |
| **CapMVRVCur** | rho=+0.093, p=0.10 | rho=+0.130, p=0.07 | **rho=+0.123, p=0.0002** | **rho=+0.175, p<0.0001** |
| FlowInExNtv | 비유의 | 비유의 | 비유의 | 비유의 |
| FlowOutExNtv | 비유의 | 비유의 | 비유의 | 비유의 |
| SplyExNtv | 비유의 | 비유의 | **rho=+0.114, p=0.0006** | 비유의(p=0.50) |
| TxCnt | 비유의 | rho=+0.174, p=0.016 | 비유의 | rho=+0.104, p=0.015 |

`CapMVRVCur`만 4개 조합 전부에서 방향이 일관됨(h384 OOS는 15/15 시드 전부 양수) — 24개 검정
Bonferroni 보정(α≈0.002) 적용해도 h384 VAL/OOS 둘 다 생존.

### 3단계 — 오염도 (일봉, VAL/OOS 구간)

| 지표 | VAL corr(close) | VAL corr(시간순번) | OOS corr(close) | OOS corr(시간순번) |
|---|---:|---:|---:|---:|
| AdrActCnt | -0.170 | +0.279 | +0.258 | -0.097 |
| **CapMVRVCur** | **+0.952** | **-0.780** | **+0.973** | **-0.832** |
| FlowInExNtv | +0.163 | -0.276 | -0.337 | +0.298 |
| FlowOutExNtv | +0.195 | -0.320 | -0.364 | +0.347 |
| **SplyExNtv** | **+0.817** | **-0.954** | **+0.868** | **-0.887** |
| TxCnt | -0.217 | +0.235 | +0.186 | -0.058 |

(참고: FINAL12 dedup의 오염 배제 기준은 `whale_retail_ratio`의 `corr(close)=+0.561`.)
`CapMVRVCur`·`SplyExNtv` — 정확히 2단계에서 가장 강한 신호를 보였던 두 지표 — 가 압도적으로
오염돼 있다(0.82~0.97). 나머지 4개는 오염이 약하지만(0.06~0.36), 2단계에서 애초에 신뢰할 만한
상관을 보이지 못했다(무상관이거나 VAL/OOS 부호 반전).

### 3단계 — `CapMVRVCur` detrend 재검증

| 변형 | Split | raw 레벨 | diff1(전일대비) | roc7(7일 변화율) |
|---|---|---:|---:|---:|
| h48orig | VAL | rho=+0.093, p=0.10 | rho=-0.038, p=0.50(**부호반전**) | rho=+0.098, p=0.08 |
| h48orig | OOS | rho=+0.130, p=0.07 | rho=-0.007, p=0.92 | rho=+0.018, p=0.81 |
| h384 | VAL | rho=+0.123, **p=0.0002** | rho=+0.043, p=0.20 | rho=+0.035, p=0.30 |
| h384 | OOS | rho=+0.175, **p<0.0001** | rho=+0.081, p=0.056 | rho=+0.054, p=0.21 |

Detrend하자마자 4개 조합 전부 비유의로 떨어진다(가장 근접한 h384 OOS diff1도 p=0.056로
보정 전 기준으로도 문턱을 못 넘음).

## 해석

**`CapMVRVCur`의 순위상관은 진짜 진입-레벨 신호가 아니라 가격추세 그 자체였다.** `corr(close)
=0.95~0.97`은 사실상 "MVRV 비율이 가격을 거의 그대로 재현한다"는 뜻이고(공식상 당연하다 —
`market_cap = price × supply`이고 supply는 느리게 변하니 `MVRV ≈ f(price)`에 가깝다), 이 세션
전체를 관통한 "하락장 드리프트를 스킬로 착각"하는 패턴이 스칼라 추출(candidate A/C)에 이어
피쳐 후보(candidate 6)에서도 재발했다. `SplyExNtv`도 같은 이유로 오염됐고, 이게 h384 VAL에서만
유의했던(OOS는 아님) 이유도 설명된다 — 국소적인 추세 형태가 두 구간에서 다르게 맞아떨어진
것뿐.

나머지 4개 지표(`AdrActCnt`/`FlowInExNtv`/`FlowOutExNtv`/`TxCnt`)는 오염은 약하지만, 애초에
2단계에서 신뢰할 만한 신호를 보이지 못했다 — `AdrActCnt`는 부호가 스플릿 간 뒤집히고(이 세션
초반 발견한 "부호가 랜덤 init에 좌우되는" 패턴과 유사한 불안정성), `TxCnt`는 OOS에서만
유의(VAL 비유의), Flow 두 개는 어디서도 유의하지 않다.

**결론: CoinMetrics 무료tier ETH 온체인 지표 6개 중 어느 것도 신뢰할 만한 진입-레벨 신호를 주지
않는다** — 강해 보였던 신호는 오염이었고, 오염 없는 나머지는 애초에 신호가 없었다.

## 계약 문서에 미친 영향

신규 데이터소스 후보 6(ETH 온체인)이 부정 결과로 닫힌다. 유료tier(SOPR 등 CoinMetrics
Pro/Glassnode)로 업그레이드할 근거는 없음 — 리서치 문서의 "하지 말아야 할 것" 원칙("무료tier
음성이면 유료 안 삼") 그대로 적용. 이 문서가 새로 확립한 절차 — **새 raw-level 피쳐 후보는
학습 전에 반드시 `corr(price)`/`corr(시간순번)` 오염도부터 확인**(diff/roc 등 detrend 버전과
함께 병행 테스트) — 은 이 프로젝트의 기존 FINAL12 dedup 관행(`funding_pressure_diff1`,
`whale_retail_ratio_dt288`)을 새 데이터소스 리서치 라인에도 명시적으로 적용한 것이며, 앞으로
후보 4(거래소간 basis)·5(옵션 스큐, 인프라 문제로 보류)·8(정식 VPIN) 등 남은 후보에도 동일하게
적용해야 한다.

## 결과 (계약 문서 반영용)

ETH 온체인(CoinMetrics 무료tier 6개 지표) 순위상관 진단 완료. 원본 레벨에서 `CapMVRVCur`가
4개 조합(h48orig/h384 × VAL/OOS) 전부 방향 일관 + h384 양쪽 Bonferroni 생존(p=0.0002/p<0.0001)
으로 이 세션 최초의 강한 양성 신호처럼 보였으나, 오염도 직접 측정 결과 `corr(price)=0.95~0.97`
로 심각하게 오염됨을 확인(FINAL12 배제 기준 0.561보다 훨씬 심함) — 전일대비/7일변화율로
detrend하자 4개 조합 전부 비유의로 붕괴. `SplyExNtv`도 동일 기제로 오염(corr(price)=0.82~0.87).
나머지 4개 지표는 오염은 약하나 애초에 신뢰할 신호 없음(무상관 또는 스플릿간 부호반전).
**결론: 후보 6(온체인) 부정 결과로 닫힘 — 강해 보인 신호는 드리프트-베타였다.**
