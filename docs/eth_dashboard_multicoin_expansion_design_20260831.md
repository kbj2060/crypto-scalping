# 스냅샷 대시보드 멀티코인(BTC/SOL/XRP/HYPE) 확장 설계도서

- 작성일: 2026-08-31
- 목적: "코인 섹션만 바꾸면 ETH와 동일한 대시보드 화면이 BTC/SOL/XRP/HYPE에도 뜨게" 만들기 위한 아키텍처 설계 + 코인별 데이터/모델 가용성 조사
- 근거: 서브에이전트 3건(대시보드 코드 전수조사 / 코인별 데이터수집 현황 / 코인별 모델·파이프라인 현황) + 기존 아티팩트([스냅샷 멀티코인 확장](https://claude.ai/code/artifact/fc6d82be-f60a-46ed-9d07-4e8ff1660226), 2026-08-27) + `~/.claude` 메모리 기록 재확인
- 성격: **설계 제안 + 조사**. 아직 구현하지 않았고, 코드도 수정하지 않았습니다. §9의 결정 사항은 착수 전 사용자 확인이 필요합니다.

> ⚠️ 이 문서의 파일:라인 인용은 2026-08-31 조사 시점 스냅샷입니다. 특히 §3의 "서버측(로컬 미검증)" 표시 항목은 과거 세션이 SSH로 확인했다는 **메모리 기록**을 근거로 삼은 것이고, 이번 조사는 로컬 devmachine 파일로 재확인하지 못했습니다.

---

## 요약

1. **대시보드는 이미 3탭(라이브/운영관리/스냅샷)이고, "라이브" 탭은 이미 ETH/BTC/SOL 멀티코인입니다.** 사용자가 원하는 화면 — 증거신호·모델지표·청산맵·레짐·매크로 등 12개 안팎의 패널 — 은 전부 "스냅샷" 탭 안에 있고, 이 탭만 코인 전환 UI가 전무한 ETH 고정 레이아웃입니다.
2. **"코드 배선"과 "리서치 재수행"은 전혀 다른 비용입니다.** 매크로 캘린더·모델 내부 지표 5종처럼 데이터가 이미 있고 계산 자체는 심볼 무관인 패널은 배선(리팩터)만 하면 됩니다. 반면 증거신호 8종·레짐분류기·특화감지기(V자반등)는 **4개 코인 전부 0건**이고, 임계값(HORIZON/K/ATR배수)이 ETH 변동성에서 실측 보정된 값이라 "코드 수정"이 아니라 코인별 라벨 재설계 + 그리드서치 + TabPFN 재학습 + 홀드아웃까지 이 저장소의 표준 리서치 방법론 전체를 다시 도는 작업입니다.
3. **코인별 준비 상태가 서로 다릅니다.** XRP는 과거 5분봉 데이터가 이미 충분(27만 행대, ETH·BTC와 비슷)한데 학습된 모델은 전무 — 데이터는 준비됐고 리서치만 남은 상태. HYPE는 반대로 **과거 시세 데이터 자체가 존재하지 않음**(60코인 아카이브에도 없음) — 8월 28일부터 라이브로만 쌓이는 중이라 당분간 백테스트가 불가능합니다.
4. **핵심 설계 제안은 "심볼을 코드 곳곳에 다시 하드코딩하지 말고 한 곳(코인 설정 레지스트리)에 모으고, `scripts/live_*.py`의 계산 함수들이 심볼을 인자로 받게 리팩터하는 것"**입니다. 현재 대시보드가 쓰는 계산 함수 10개 중 8개가 심볼 인자 자체가 없습니다(모듈 최상단 `SYMBOL = "ETHUSDT"` 상수, 그마저 3곳에 중복 정의).
5. 사용자가 예시로 든 패널 구성 중 일부는 이미 바뀌었습니다 — "청산 캐스케이드 배너"는 독립 패널이 아니라 청산 물량 게이지 안 한 줄로 흡수됐고(08-27), "꼬리위험 게이지"·"고래 포지션" 칩은 무정보 판정으로 완전히 제거됐습니다(08-30). 현재 실존하는 "모델 내부 지표"는 5개(수급흐름/리테일수급/청산방향압력/베이시스청산압박/청산캐스케이드)뿐입니다.

---

## 1. 목적과 범위

**목표**: 스냅샷 탭에 코인 선택 UI를 추가해, ETH를 고르면 지금 보이는 화면이, BTC/SOL/XRP/HYPE를 고르면 그 코인 버전의 동일한 화면 구성이 뜨게 한다.

**범위에 포함**: 대시보드 표시 계층(server.py API + app.js/index.html 렌더링) 설계, 코인별 데이터 가용성 조사, 코인별 모델 가용성 조사, 단계별 로드맵.

**범위에서 제외(이 문서가 다루지 않음)**: 실거래/자동매매 승격(별개 정책 — 프로젝트 기존 원칙대로 `대시보드 노출기준=IC, 자동매매 승격기준=경제성`을 그대로 적용), `omega4.6.1`/`odyssey4`/`zig075`/`h48qual` 등 실거래 전략 모델의 코인 이식(이건 이미 별도 트랙에서 진행 중이며 대시보드 스냅샷 지표와는 다른 모델 계열).

---

## 2. 현재 구조 (as-is)

### 2.1 탭 구조

단일 `index.html`(412줄) + `app.js`(4,458줄)가 `.hidden` 클래스 토글로 3개 탭을 전환합니다(별도 페이지/라우트 아님). `server.py`(1,797줄)가 유일한 백엔드.

| 탭 | 상태 | 비고 |
|---|---|---|
| 라이브 | **이미 멀티코인** | 자산탭 버튼(`index.html:78-80`) + `setActiveChartAsset()`(`app.js:246-268`) + `/api/market-history?asset=`(`server.py:1524`, `MARKET_SYMBOLS` 딕셔너리로 eth/sol/btc 파라미터화) + `dashboard_state.json`의 `asset_decisions`/`asset_states`에 btc/sol 키 실존 |
| 운영 관리 | 자산 무관 | 헬스체크 리스트만 — 코인 확장과 무관 |
| **스냅샷** | **ETH 고정** | 오늘 설계 대상. 자산탭 UI 자체가 없고, 라이브 탭의 자산탭과도 완전히 무관 — 라이브 탭에서 BTC를 선택해도 스냅샷 탭은 그대로 ETH 데이터를 보여줌 |

### 2.2 스냅샷 탭 현재 패널 인벤토리 (2026-08-31 기준, 코드 직접 확인)

| 패널 | 렌더 함수(app.js) | API(server.py) | 실제 계산/데이터 소스 |
|---|---|---|---|
| 세션·매크로 변동성 경보 배지 | `renderSessionVolatilityAlert`/`renderMacroEventAlert`(2462,2480) | `/api/session-alerts`(1606) | `scripts/live_session_volatility_alert_20260826.py` (순수 계산) |
| 모델 내부 지표 — 수급흐름/리테일수급 | `renderModelIndicatorList`(2195) | `/api/state` | `dashboard_state.json`의 `microstructure` 블록 |
| 모델 내부 지표 — 청산방향압력 | 〃 | `/api/liquidation-direction-signal`(1571) | `scripts/live_liquidation_direction_signal_20260825.py` (`tail_risk.duckdb`) |
| 모델 내부 지표 — 베이시스 청산압박 | 〃 | `/api/basis-liquidation-signal`(1563) | `scripts/live_spot_perp_basis_signal_20260827.py` (현물+perp klines 직접 fetch) |
| 모델 내부 지표 — 청산 캐스케이드(위험도) | 〃 | `/api/state`, `/api/liq-burst-state`(1594) | `dashboard_state.json` `tail_risk` 블록 + `liq_burst_state.json` |
| 증거 신호 8종 칩 | `renderEvidenceSignals`(2512) | `/api/evidence-signals`(1537) 등 | `scripts/live_evidence_signal_dashboard_20260823.py` + `live_evidence_signal_metalabel_20260829.py`(Homer/TabPFN) |
| 특화 감지기 — V자 반등락 | `renderModelIndicatorList(...,"snapSpecializedSignalList")`(4312) | `/api/v-rebound-signal`(1559) | `scripts/live_eth_sweep_v_rebound_signal_20260829.py` |
| 청산맵(캔들+히트맵+S/R리스트+롱숏게이지+5분신호) | `renderLiquidationMapPanel`(2278) 등 | `/api/liquidation-map`, `/api/liquidation-5m-signal`, `/api/regime-wide24` | `scripts/live_liquidation_map_20260824.py`, `live_liquidation_5m_signal_20260825.py`, `live_regime_gbm3_signal_20260826.py` |
| 주요 경제 일정(매크로 캘린더) | `renderMacroCalendar`(2874) | `/api/macro-calendar`(1590) | `scripts/live_macro_calendar_20260826.py` (FRED/EIA/Finnhub, 자산 무관) |

**참고**: `liq_magnet`(청산자석)과 L2 이상감지는 데이터 수집기는 존재하지만 **현재 어떤 탭에도 렌더링 코드가 없습니다**(app.js/index.html grep 0건) — ETH 자신도 아직 대시보드에 노출되지 않은 관찰 단계이므로, 이번 "ETH와 동일하게" 확장 범위에는 해당하지 않습니다. §3에서는 데이터 가용성 질문에 답하기 위해 다루지만, §6 로드맵에서는 "ETH 자체 노출 여부 결정 후" 항목으로 분리했습니다.

### 2.3 하드코딩 현황 (grep 실측)

| 키워드 | server.py | app.js |
|---|---|---|
| ETHUSDT/`"eth"` | 12건 | 10건 |
| BTCUSDT/`"btc"` | 6건 | 4건 |
| SOLUSDT/`"sol"` | 6건 | 2건 |
| XRP / HYPE | 0건 | 0건 |

- **이미 일반화된 부분**: `MARKET_SYMBOLS`(server.py:220, eth/sol/btc), `SCALP_SHADOW_ASSETS`(229-289, asset 키 구조), `ASSET_CONFIG`(app.js:33-37, eth/sol/btc의 label/symbol/priceDigits).
- **여전히 하드코딩**: `EVIDENCE_SIGNAL_SYMBOL = "ETHUSDT"`(server.py:130), `LIQUIDATION_MAP_SYMBOL = "ETHUSDT"`(140) — 그리고 결정적으로, **server.py가 import하는 계산 함수 10개 중 8개가 인자를 0개 받습니다.** `?asset=` 쿼리를 server.py에 추가해도 실제 계산은 `scripts/live_*.py` 내부에 심볼이 박혀 있어 함께 고쳐야 합니다.
  - `scripts/live_evidence_signal_dashboard_20260823.py:136` → `SYMBOL = "ETHUSDT"` (모듈 최상단, CLI 인자 없음)
  - `scripts/live_regime_wide24_signal_20260826.py:51` → `SYMBOL = "ETHUSDT"` 독립 재정의. `live_regime_gbm3_signal_20260826.py:37-38`는 자체 정의도 없이 **여기서 체인 임포트** — 하드코딩이 스크립트 간에 전파됨.
  - `scripts/live_eth_sweep_v_rebound_signal_20260829.py:70` → `SYMBOL = "ETHUSDT"` 세 번째 독립 하드코딩. 공유 config 모듈이 없어 같은 문자열이 최소 3곳에 중복.
  - `compute_basis_liquidation_signal(symbol: str = SYMBOL)`만 파라미터가 있지만, server.py 호출부는 항상 기본값으로만 호출 — "절반만 된" 유일한 사례.

### 2.4 이미 검증된 재사용 패턴 (새로 설계할 필요 없음)

| 패턴 | 내용 |
|---|---|
| `BOT_SYMBOLS` 환경변수 | `duckdb_persist_worker.py`가 콤마구분 값을 읽어 컬렉터를 심볼별로 스태거 기동 — 신규 컬렉터 멀티심볼화의 표준 진입점 |
| 테이블 분기 관용구 | `self._table = TABLE if symbol=='ethusdt' else f'{TABLE}_{suffix}'` — 같은 duckdb 파일 안에서 심볼별 데이터 분리 |
| 파일 분리 해법 | 동시-writer 충돌 시 `tail_risk_btc_sol.duckdb`처럼 별도 파일로 — 다른 프로세스가 소유한 파일일 때 쓰는 해법 |
| `_supervise.sh` 제네릭 wrapper | `<이름> <락파일> <로그prefix> <커맨드…>` — 심볼별 슈퍼바이저 복제, 11개 사례로 검증됨 |
| 프론트 자산탭 템플릿 | `ASSET_CONFIG` + `index.html` 자산탭 버튼 + `setActiveChartAsset()` + `/api/market-history?asset=` — 라이브 탭에서 검증된 UI |

앞 4개는 **데이터 수집 계층**에서 이미 XRP/HYPE까지 실제로 쓰였습니다(§3). 이번 설계가 새로 풀어야 할 것은 5번째 패턴을 스냅샷 탭까지 확장하는 것과, §2.3의 계산 함수 리팩터입니다.

---

## 3. 코인별 데이터 가용성 조사

### 3.1 범례

| 약어 | 실체 |
|---|---|
| MS | `microstructure_scanner.py` — WS depth20/aggTrade + REST OI·펀딩 폴백 |
| TR | `tail_risk_interceptor.py` — WS `@forceOrder`(개별 청산) |
| OI | `oi_lsratio_collector.py` — REST OI/롱숏비율, 5분폴링 |
| L2A | `l2_anomaly_snapshot_collector.py` — 이벤트 트리거시에만 스냅샷 |
| LM | `liq_magnet_collector.py` — `dashboard_state.json` 폴링, **심볼 인자 자체 없음** |
| GEX | `collect_deribit_option_gex_20260815.py` — `CURRENCIES=("ETH","BTC")` 하드코딩 |
| Panel60 | `data/panel/` 60코인 과거 정적 아카이브(라이브 아님) |

### 3.2 데이터 카테고리 × 코인

| 카테고리 | ETH | BTC | SOL | XRP | HYPE |
|---|---|---|---|---|---|
| OHLCV(캔들) | 수집됨 — 라이브 fetcher + Panel60(24-01~) | 수집됨 — 동일 | 수집됨 — 동일 | **부분수집** — 라이브 온디맨드 fetcher 없음(비거래대상)이나 Panel60 존재(24-01-01~26-08-04, 27만행대) | **미수집** — `binance_data/klines/`에 디렉토리 자체 없음, Panel60에도 없음. 학습·백테스트용 과거 데이터가 원천적으로 부재 |
| 펀딩비율 | 수집됨(로컬실측) | 수집됨(로컬실측) | 수집됨(로컬실측) | 수집됨(서버측·미검증) | 수집됨(서버측·미검증) |
| OI | 수집됨(로컬실측 1,347행) | 수집됨(로컬실측 1,346행) | 수집됨(로컬실측 1,346행) | 수집됨(서버측·미검증) | 수집됨(서버측·미검증) |
| 롱숏비율/테이커델타 | 수집됨 | 수집됨 | 수집됨 | 수집됨(서버측·미검증) | 수집됨(서버측·미검증) |
| 청산 개별이벤트 | 수집됨(로컬실측 129,177행) | 수집됨(로컬사본 86행뿐 — 하루 스모크성, 서버 파일은 더 길 것으로 추정) | 수집됨(로컬사본 86행, 동일) | 수집됨(서버측·미검증) | 수집됨(서버측·미검증) |
| 청산맵/매그넷 이력 | **수집됨(ETH 유일)** | **구조적 미수집** — LM 코드에 symbol 인자 자체 없음 | **구조적 미수집** — 동일 | **구조적 미수집** | **구조적 미수집** |
| 청산 캐스케이드 Hawkes 임계값 | 튜닝됨($10,000) | **미튜닝**(기본값 0) | 미튜닝 | 미튜닝 | 미튜닝 |
| L2 오더북 불균형 | 수집됨(로컬실측) | 수집됨(로컬실측) | 수집됨(로컬실측) | 수집됨(서버측·미검증) | 수집됨(서버측·미검증) |
| L2 이상감지 스냅샷 | 수집됨(로컬실측, 이벤트35건) | 수집됨(서버측·미검증, 08-28 신설) | 수집됨(서버측·미검증) | 수집됨(서버측·미검증) | 수집됨(서버측·미검증) |
| 고래/대형포지션 | **프록시만**(OI top-trader비율 + MS 거래규모분류, 전용 수집기 없음) | 동일 프록시 | 동일 프록시 | 동일 프록시(서버측) | 동일 프록시(서버측) |
| 옵션/GEX | 수집됨(로컬실측) | 수집됨 | **미수집**(Deribit에 SOL옵션 실재 — 확장 가능) | **구조적 불가**(유동적 옵션시장 없음) | **구조적 불가**(동일) |

### 3.3 외부 거래소 관점의 가용성 (내부 수집 여부와 별개 질문)

사용자가 물은 "구할 수 있는지"는 두 층위입니다 — ① 이 저장소가 이미 모으고 있는가(위 표), ② 애초에 거래소가 그 데이터를 제공하는가.

- **BTC/SOL/XRP**: 4대 거래소(Binance/Bybit/OKX 등) 전부 최상위 유동성 페어라 OHLCV/펀딩/OI/청산/L2 전부 기술적으로 100% 가능 — 실제로 이미 다 모으고 있음(위 표).
- **HYPE**: `Binance Futures exchangeInfo`에 `HYPEUSDT`(PERPETUAL, status=TRADING)가 실재함을 2026-08-28에 직접 확인했고(메모리 `eth_dashboard_hype_collector_and_l2_anomaly_multicoin_20260828`), 그 근거로 MS/TR/OI/L2A 4종 수집기가 이미 HYPEUSDT로 정상 연결됩니다. 즉 **"앞으로" 데이터는 구할 수 있음이 확인됨** — 문제는 "과거"입니다. HYPE 자체가 상장된 지 오래되지 않았고, 이 저장소도 8월 28일부터만 모으기 시작해 장기 과거 이력이 존재하지 않습니다. 시간이 해결해줄 문제이지 기술적 장벽이 아닙니다.
- **옵션/GEX**: Deribit이 ETH/BTC/SOL 옵션은 상장돼 있으나(SOL은 코드 미반영, 확장 가능), XRP/HYPE는 유동성 있는 옵션 시장 자체가 없어 구조적으로 불가능 — 이건 시간이 아니라 시장 자체의 한계입니다.

### 3.4 가장 큰 격차

1. **HYPE가 전 카테고리에서 가장 얕음** — 라이브(08-28~)만 있고 과거 아카이브가 전무. 8개 데이터 카테고리 중 OHLCV는 코드 진입점 자체가 없는 유일한 항목.
2. **청산맵/매그넷이 구조적으로 ETH 전용 고정** — `liq_magnet_collector.py`가 심볼 인자 없이 `dashboard_state.json`만 읽는 단일 프로세스라, XRP/HYPE처럼 슈퍼바이저만 복제해서는 확장 불가 — 코드 자체를 고쳐야 함.
3. **로컬(devmachine) ≠ 서버(실 라이브) 괴리** — XRP/HYPE 전용 파일 6개, BTC/SOL tail_risk, l2_anomaly 멀티코인 4개, `liq_magnet_history.duckdb` 등 11개 파일이 로컬에 전혀 없어 이번 조사는 "코드+배포기록"까지만 확인했고 "로컬 실측"과 같은 신뢰도가 아닙니다. **착수 전 서버 SSH 재확인을 권장합니다(§6 Step 0).**
4. **BTC/SOL의 tail_risk 로컬 사본은 하루치(86행)뿐** — 서버 파일이 진짜 이력을 갖고 있을 가능성이 높으나 독립 확인 필요.
5. **"고래 포지션"은 전용 수집기 없이 다른 두 카테고리의 파생물** — ETH 포함 5개 코인 전부 동일한 한계이므로, 정밀 고래추적을 원하면 코인 확장과 무관하게 5개 코인 공통 신규 설계가 필요합니다.
6. `docs/` 디렉토리 전체를 검색했으나 멀티코인/XRP/HYPE 관련 기존 문서는 **0건**입니다 — 관련 기록은 전부 로컬 Claude 메모리에만 있었고, 이 문서가 최초의 repo 문서입니다. 가장 가까운 기존 문서는 `docs/panel_universe_coverage_report_20260804.md`(60코인 아카이브, ETH/BTC/SOL/XRP 포함·HYPE 미포함).

---

## 4. 코인별 모델·신호 가용성 조사

### 4.1 대시보드 노출용 모델(증거신호 8종/레짐분류기/특화감지기) — 4코인 전부 0건

| 코인 | 증거신호 8종 | 레짐분류기(GBM2/3/wide24) | 특화감지기(V자반등류) |
|---|---|---|---|
| ETH | 8/8 (Homer 이관 6/8 진행중) | 있음(GBM3 live) | 있음 |
| BTC | 0 | 0 | 0 |
| SOL | 0 | 0 | 0 |
| XRP | 0 | 0 | 0 |
| HYPE | 0 | 0 | 0 |

### 4.2 참고 — 실거래 전략 모델(다른 축, 혼동 주의)

대시보드 스냅샷 지표와는 **다른 모델 계열**이지만, "코인별 인프라 성숙도"를 판단하는 데 참고가 되므로 병기합니다.

| 코인 | omega4.6.1 | regime3_current_hmm | zig075 | h48qual | odyssey4 |
|---|---|---|---|---|---|
| BTC | 풀스택 존재(shadow 가동) | 존재(대시보드 GBM2/3과 다른 모델) | 진단스크립트 2개뿐(ETH 58개 대비 미미) | 추론스크립트 4종 | 이식 시도 → **REJECTED** |
| SOL | 풀스택 **라이브 가동중** | 존재 | **라이브 가동중**(단, R&D 파일 2개뿐, 얕음) | 없음 | 시도 자체 없음 |
| XRP | 없음 | 없음 | 없음 | 없음 | 없음 |
| HYPE | 없음(단, `data/live/hyperliquid_execution_audit.jsonl` 실행감사로그 존재 — §9 참고) | 없음 | 없음 | 없음 | 없음 |

**중요 캐비엇**: "엔진 재사용 가능"과 "결과가 좋다"는 별개입니다. 이 패턴으로 만든 BTC/SOL 라이브 후보는 seed-robustness N=5 재검증에서 탈락했고(BTC 미통과·SOL 최악), `odyssey4`의 BTC 이식은 명시적으로 REJECTED됐습니다 — 엔지니어링 재사용성과 신호의 자산간 실제 전이가능성은 이미 이 저장소 안에서 여러 번 분리된 결론으로 확인됐습니다. 대시보드 지표를 코인별로 이식할 때도 "코드가 돌아간다"와 "IC가 유의미하다"를 반드시 분리해서 판단해야 합니다.

### 4.3 하드코딩 vs 파라미터화 — 구체적 근거

**대시보드 지표 계열(재작업 필요)**:
- 심볼이 최소 3개 파일에 독립적으로 하드코딩(`live_evidence_signal_dashboard_20260823.py:136`, `live_regime_wide24_signal_20260826.py:51`, `live_eth_sweep_v_rebound_signal_20260829.py:70`) — 공유 config 모듈 없음.
- `live_regime_gbm3_signal_20260826.py`는 자체 정의도 없이 `live_regime_wide24_signal_20260826`에서 `SYMBOL`/`BTC_SYMBOL`을 체인 임포트 — 하드코딩이 스크립트 간에 전파되는 구조.
- 파일 경로 자체에 심볼 인코딩: `KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"`, 출력 디렉토리도 `eth_` 접두.
- **argparse/CLI 플래그 전무**(grep 0건). 이식은 "`--symbol=BTCUSDT` 한 줄"이 아니라 **phase1 진단 → HORIZON/GAP 그리드서치 → K(ATR배수) 보정 → TabPFN 학습 → 트레일링스톱 경제성게이트 → 홀드아웃**이라는 표준 리서치 방법론 전체의 재수행을 뜻합니다. K=1.5~4.0×ATR, HORIZON=12~30봉 같은 임계값이 ETH 변동성/미시구조에서 실측 보정된 값이라 전이가 보장되지 않습니다.
- 예외: 청산맵의 핵심 계산 함수(`prepare_cohort_arrays` 등)는 범용 OHLCV+OI 데이터프레임을 받는 **순수함수라 심볼 무관** — 로직 자체는 재사용 가능. 다만 BTC/SOL/XRP/HYPE용 OI+테이커비중+롱숏계정비율 장기 이력 데이터셋을 찾지 못해, 계산 엔진과 별개로 데이터 파이프라인·대시보드 연결부는 여전히 확인·작업이 필요합니다.

**대조 — 이미 파라미터화된 사례(실거래 전략 계열)**:
- `experiment_regime3_current_hmm_wide24_20260529.py`는 진짜 `argparse`(`--train-2024`, `--states`, `--n-iter`, `--sticky`, `--seed`, `--feature-sets`)를 갖춤.
- `apply_final_scale_map_btc_regime_docs42_20260721.py`(31줄)는 ETH 스크립트를 라이브러리로 import해 경로 상수만 BTC 산출물로 오버라이드하는 "엔진 공유 + 자산별 얇은 wrapper" 패턴이 실제로 작동함 — **이게 오늘 설계가 대시보드 지표 계열에도 이식하려는 목표 패턴**입니다.

### 4.4 코인별 준비 상태 요약

| 순위 | 코인 | 상태 |
|---|---|---|
| 1 | XRP | 과거 데이터 이미 충분(27만행대), 학습된 모델 전무 — **리서치만 하면 되는 상태** |
| 2 | BTC/SOL | 대시보드 지표는 없지만 실거래 전략 인프라·데이터는 풍부 — 단, 그 인프라를 대시보드 지표에 재사용할 때 위 캐비엇 유의 |
| 3 | HYPE | 과거 데이터 자체가 없음 — 데이터 축적이 먼저, 리서치는 그 다음 |

---

## 5. 패널별 확장 난이도 매트릭스

배선 난이도(A=즉시, B=배선/리팩터, C=신규 코드·재학습)와 검증 필요도(불요/권장/필수)는 별개 축입니다.

| 패널 | 배선 | 검증 | 코인별 비고 |
|---|---|---|---|
| 매크로 캘린더 | A | 불요 | 자산 무관 — 4코인 즉시 가능, 유일하게 수정 없이 재사용 |
| 세션·매크로 변동성 경보 | A(코드) | **필수**(근거) | 코드는 개장시각 계산이라 심볼 무관이나, "미국장만 2.3배 효과" 근거는 ETH 전용 실측 — 배지만 복제하면 근거 없는 주장이 됨 |
| 모델 내부 지표 5종 | B | 불요 | BTC·SOL 원자재는 로컬 확인됨(단, tail_risk 로컬사본은 하루치뿐). XRP·HYPE는 서버측 수집 중이나 로컬 미검증 |
| 청산맵 + 5분신호 | B(엔진은 순수함수) | 권장 | XRP·HYPE는 OI+테이커+롱숏 장기이력 자체가 없어, 배선해도 신뢰도 낮음 — 데이터 축적 후 권장 |
| 청산자석(liq_magnet) | **C**(코드에 symbol 인자 자체 추가 필요) | 해당없음 | ETH 자신도 대시보드 미노출 — §6에서 별도 트랙으로 분리 |
| 청산 캐스케이드 Hawkes 임계값 | B(재튜닝) | 권장 | ETH만 $10,000로 튜닝, 나머지 전부 미튜닝 기본값(0) — 그대로 켜면 노이즈 폭증 우려 |
| L2 이상감지 | (인프라는 08-28 5코인 배포 완료) | 불요 | ETH 자신도 대시보드 미노출(관찰 단계) — §6에서 별도 트랙으로 분리 |
| 증거신호 8종 | **B/C**(리서치 재수행) | **필수** | 4코인 전부 0 — §4.3 참고, 코드 배선이 아니라 라벨설계부터 재시작 |
| 레짐분류기(GBM2/3/wide24) | **C**(재학습) | **필수** | 4코인 전부 0 — 대시보드 노출용과 실거래용(`regime3_current_hmm`)은 별개 모델이라 후자가 있어도 도움 안 됨 |
| 특화감지기(V자반등) | **B/C** | **필수** | 4코인 전부 0, ETH 전용 이벤트트리거 TabPFN — 증거신호와 유사한 재작업 규모 |
| 고래/대형포지션 프록시 | - | - | 전용 수집기 없음(OI+MS 파생) — ETH 포함 5코인 공통 한계, 코인 확장 이슈가 아님 |
| 옵션/GEX | C(구조적) | 해당없음 | BTC만 이미 있음. SOL 확장 가능(코드 미반영). XRP/HYPE 구조적 불가. 현재 대시보드 패널로 노출되고 있지 않음(연구단계) |

---

## 6. 설계 제안: 아키텍처

핵심 원칙: **§2.3에서 확인한 "심볼이 3곳에 중복 하드코딩"되는 실수를 반복하지 않는다.** 기존 `MARKET_SYMBOLS`(server.py)/`SCALP_SHADOW_ASSETS`(server.py)/`ASSET_CONFIG`(app.js) 세 곳에 흩어진 코인별 설정을 대체하는 게 아니라, 그 세 곳이 참조하는 **단일 소스**를 신설합니다.

### 6.1 코인 설정 레지스트리 (신규)

`scripts/live_*.py`와 `server.py`가 공통으로 import할 수 있는 위치(순환 임포트 방지를 위해 `scripts/` 하위, 예: `scripts/coin_config.py`)에 최소 스키마로 신설:

```python
# scripts/coin_config.py (신규 제안 — 필요한 필드만, 과설계 금지)
COIN_CONFIG = {
    "eth": {
        "binance_symbol": "ETHUSDT",
        "table_suffix": "",              # 기존 관용구 유지: ethusdt는 접미사 없음
        "price_digits": 2,
        "correlated_asset": "BTCUSDT",   # smt_divergence 등 교차자산 신호용
        "liq_hawkes_min_trigger_usd": 10000,  # ETH 실측 튜닝값
        "has_historical_ohlcv": True,
        "evidence_signal_status": "live",     # live / needs_revalidation / not_built
    },
    "btc": {
        "binance_symbol": "BTCUSDT", "table_suffix": "_btc", "price_digits": 1,
        "correlated_asset": None,        # 잠정 미정 — §9 결정 필요
        "liq_hawkes_min_trigger_usd": None,   # 미튜닝, 기본값 0 사용 중이므로 주의
        "has_historical_ohlcv": True,
        "evidence_signal_status": "not_built",
    },
    "sol": { ... },   # btc와 동일 shape
    "xrp": {
        "binance_symbol": "XRPUSDT", "table_suffix": "_xrp", "price_digits": 4,
        "has_historical_ohlcv": True,    # Panel60 아카이브 존재
        "has_live_ohlcv_fetcher": False, # 비거래대상이라 온디맨드 fetcher 없음 — 신규 필요
        "evidence_signal_status": "not_built",
    },
    "hype": {
        "binance_symbol": "HYPEUSDT", "table_suffix": "_hype", "price_digits": 3,  # 자릿수 재확인 필요
        "has_historical_ohlcv": False,   # 상장 후 과거데이터 자체 없음 — 백테스트 불가
        "evidence_signal_status": "not_built",
    },
}
```

기존 `MARKET_SYMBOLS`/`ASSET_CONFIG`는 이 레지스트리에서 파생되는 얇은 view로 남기거나 점진 교체 — 한 번에 다 바꾸지 않고 새 지표를 추가할 때마다 이 레지스트리를 참조하게만 하면 됩니다(surgical, 기존 코드 불필요한 리팩터 금지).

### 6.2 계산 함수 리팩터 (`scripts/live_*.py`)

8개 함수가 인자를 0개 받는 문제를 해결 — 패턴은 이미 절반 존재하는 `compute_basis_liquidation_signal(symbol=...)`을 다른 9개로 확장:

```python
# Before (scripts/live_evidence_signal_dashboard_20260823.py:136)
SYMBOL = "ETHUSDT"
def compute_signals():
    df = fetch_klines(SYMBOL, ...)

# After
def compute_signals(symbol: str = "ETHUSDT"):
    df = fetch_klines(symbol, ...)
```

체인 임포트 문제(`live_regime_gbm3_signal_20260826.py`가 `live_regime_wide24_signal_20260826`에서 `SYMBOL`을 가져오는 구조)는 두 파일 다 `coin_config.COIN_CONFIG`를 직접 참조하도록 고쳐 하드코딩 전파 경로 자체를 끊습니다.

### 6.3 API 계층 (`server.py`)

이미 검증된 `/api/market-history?asset=` 패턴을 스냅샷 탭 엔드포인트로 확장:

```python
@app.get("/api/evidence-signals")
def evidence_signals(asset: str = "eth"):
    cfg = COIN_CONFIG[asset]
    return compute_signals(symbol=cfg["binance_symbol"])
```

### 6.4 프론트 계층

- `ASSET_CONFIG`(app.js:33-37)에 xrp/hype 항목 추가(price_digits 등 — XRP는 소수점 4자리 이상 필요할 가능성 높음).
- **결정 필요(§9-a)**: 코인 선택 상태를 라이브 탭과 스냅샷 탭이 공유할지, 탭별로 독립적으로 가질지. 이 문서의 추천안은 **공유**(코인 하나를 고르면 대시보드 전체가 그 코인 뷰가 되는 것이 사용자가 요청한 "섹션만 넘어가면"에 더 부합) — 단, 라이브 탭은 실거래 포지션(실제로는 ETH/BTC/SOL 3개가 항상 동시 운용 중)을 보여주는 것이라 "선택한 자산만 보여준다"는 개념 자체가 라이브 탭에는 안 맞을 수 있어, 대안으로 스냅샷 탭 전용 독립 코인 스위처(라이브 탭 자산탭 UI를 그대로 복제하되 상태는 분리)도 고려할 수 있습니다.
- 08-30 특화감지기 작업에서 겪었던 캐시 무효화 함정(`lastModelIndicatorHtmlByTarget`, target별로 키를 나누지 않으면 같은 함수를 여러 target에 호출할 때 캐시가 서로를 밀어냄)을 이번엔 **target × 코인** 이중 키로 확장해야 합니다 — 안 그러면 코인 전환마다 불필요한 전체 재렌더가 발생하거나, 반대로 전환했는데 이전 코인 캐시가 그대로 보이는 버그가 생깁니다.

### 6.5 콘텐츠 노출 정책 (Tier 1~4) — "같은 화면 구성"과 "같은 신뢰도"는 별개

코인 탭을 누르면 **레이아웃은 항상 동일하게** 뜨되, 내용의 검증 상태에 따라 배지를 다르게 노출하는 방식을 제안합니다. 이는 프로젝트 기존 원칙(`대시보드 노출기준=IC`)과 정합적입니다.

| Tier | 예시 패널 | 코인별 노출 조건 |
|---|---|---|
| 1 (즉시) | 매크로 캘린더 | 항상 노출 |
| 2 (배선+최소검증) | 모델 내부 지표 5종, 청산맵 | 원자재 데이터 확인 후 노출, 부족하면 "데이터 축적 중" 배지 |
| 3 (재튜닝 필요) | 청산 캐스케이드 임계값, 세션 변동성 경보 근거 | 재튜닝/재검증 전까지 "검증 전" 배지 |
| 4 (연구 재수행) | 증거신호 8종, 레짐분류기, 특화감지기 | 코인별 IC 재검증 통과 전까지 "미지원" 또는 회색 처리, 거짓 신뢰 방지 |

---

## 7. 코인별 특이사항

- **HYPE**: 과거 데이터가 없어 Tier 3~4는 데이터가 축적될 때까지(최소 수 주~수개월) 원천적으로 불가능. Tier 1~2부터 착수하고, 그 사이 라이브 수집이 쌓이길 기다리는 것이 유일한 경로입니다. `data/live/hyperliquid_execution_audit.jsonl`의 존재는 Binance 데이터 수집과 별개로 Hyperliquid 거래소 자체의 실행 계층이 있을 가능성을 시사 — 이번 조사에서 깊이 확인하지 못했습니다(§9-e).
- **XRP**: 데이터가 이미 충분해 "코인-무관 파이프라인" 리팩터를 검증하기 가장 좋은 첫 대상 — 데이터 병목이 없으니 엔지니어링 결과가 순수하게 드러남.
- **BTC/SOL**: 실거래 전략(omega4.6.1 등) 인프라가 성숙해 "이미 다 됐다"고 착각하기 쉬우나, 대시보드 지표 계열은 그것과 다른 모델이라 실제로는 XRP와 같은 0 상태입니다. 게다가 이 저장소 안에서 이미 BTC 이식 실패 사례(odyssey4 REJECTED, N=5 seed-robustness 탈락)가 기록돼 있어, "엔진 재사용 가능 = 신호도 통할 것"이라는 가정은 금물입니다.
- **smt_divergence류 교차자산 신호**: ETH는 BTC를 교차자산 확인용으로 씁니다. BTC 자체 대시보드를 만들 때 무엇을 교차자산으로 쓸지는 정해진 답이 없습니다(§9-d).

---

## 8. 단계별 로드맵

1. **Step 0 — 재확인**: 서버 SSH로 §3의 "서버측·미검증" 11개 파일 로컬 재확인, 미커밋 상태인 `l2_anomaly_snapshot_collector.py`/`liq_magnet_collector.py` 커밋 여부 결정. *(미착수)*
2. **Step 1 — 레지스트리+리팩터**: `coin_config.py` 신설, 계산 함수 8개에 `symbol` 파라미터 추가(§6.1~6.2). 이후 모든 단계의 전제조건이자 가장 leverage 높은 단일 작업. **✅ 2026-08-31 구현 완료(BTC → XRP 순으로 확장)** — `scripts/coin_config.py`에 ETH/BTC/XRP 등록, `live_liquidation_direction_signal_20260825.py`/`live_liquidation_5m_signal_20260825.py`가 `coin` 파라미터로 리팩터됨(8개 중 2개 — 나머지 6개는 Tier4 소속이라 §6.2 원칙대로 보류). XRP 추가 시점엔 `server.py`의 4개 load_* 함수가 이미 `COIN_CONFIG`/`MARKET_SYMBOLS`에만 제네릭 의존하도록 되어 있어 함수 리팩터 자체가 불필요 — 순수 설정 추가 4곳(coin_config.py/MARKET_SYMBOLS/SNAPSHOT_ASSET_KEYS/코인탭 버튼)뿐이었음, 이 패턴이 완전히 기계적임을 재확인.
3. **Step 2 — 프론트 배선**: 스냅샷 탭 코인 스위처 추가(§6.4의 결정 필요 사항 확정 후), `ASSET_CONFIG`에 xrp/hype 추가. **✅ 2026-08-31 구현 완료(ETH/BTC/XRP)** — §9-a 결정: 라이브 탭의 `activeChartAsset`와 공유하지 않고 **독립적인 `activeSnapshotAsset`**로 확정(코드 자체에 이미 "스냅샷 차트는 라이브 탭 자산선택과 무관"이라는 기존 설계의도 주석이 있어 그대로 따름 — §6.4 문서 작성 시의 "공유" 추천은 기각). 이후 라이브 탭 자체가 제거되어 이 구분은 무의미해짐(아래 "후속 2" 참고). hype 프론트 추가는 미착수(과거데이터 없어 Tier1~2도 제한적, §7 참고).
4. **Step 3 — Tier 1~2 패널 배선**: 매크로 캘린더 → 모델 내부 지표 5종 → 청산맵 순. **✅ 2026-08-31 구현 완료(BTC, 이어서 XRP)** — 베이시스청산압박·청산방향압력·청산맵(+캔들차트+5분신호)까지 BTC/XRP 둘 다 배선+실서버 검증 완료. 수급흐름/리테일수급/청산캐스케이드는 `dashboard_state.json`이 trading_bot.py를 통해서만 채워져 **의도적으로 보류**(§2.2 참고, trading_bot.py 미변경 원칙).
5. **Step 4 — Tier 3**: `liq_magnet_collector.py`에 실제 symbol 인자 추가, Hawkes 임계값 코인별 재튜닝, 세션 변동성 경보 타 코인 재검증. *(미착수)*
6. **Step 5 (가장 비쌈, 최후순위)**: 증거신호 8종·레짐분류기·특화감지기 코인별 리서치 재수행 — **XRP부터 권장**(데이터 병목 없음), HYPE는 데이터 축적 후. *(미착수)*

**2026-08-31 구현 검증 방법**: 로컬 서버 기동 후 4개 엔드포인트에 `?asset=btc` curl로 실제 BTC 값 확인(베이시스 z48, 청산맵 현재가 $79,304 등 ETH와 다른 실제 BTC 데이터 확인) + Node vm 하네스로 `app.js`를 실제 서버 데이터에 태워 `setActiveSnapshotAsset("btc")`→`render()` 전체 실행까지 예외 없음 확인. 로컬 devmachine에는 BTC용 `tail_risk_btc_sol.duckdb`가 없어(§3의 서버-로컬 괴리) 청산방향압력·청산5분신호는 로컬에서 `db_missing`으로 우아하게 저하되는 것까지만 확인.

**✅ 2026-08-31 후속 — 커밋(`790c95e`) + `handoff.sh push server`로 실서버 배포 + 재시작 + 실서버 재검증까지 완료.** 서버에는 `tail_risk_btc_sol.duckdb`가 실존해 청산방향압력·청산5분신호도 BTC가 완전히 `warmed_up=true`로 확인됨(로컬의 `db_missing`은 devmachine 한계였을 뿐, 서버에서는 정상 동작). `trading_bot.py`(실거래봇)는 재시작 영향 없음. 자세한 배포 절차·경합 이슈는 memory `eth-dashboard-multicoin-expansion-design-20260831` 참고.

**✅ 2026-08-31 후속 2 — 라이브 탭 제거 + 최적화 2건 + BTC 레짐 리본 버그 수정, 배포 완료.** 이 설계도서와 직접 관련된 부분만 요약:
- BTC 스냅샷 코인 스위처 작업 중 발견된 버그(§ "구현 검증 방법" 상단 문단 참고)를 수정 — `renderCandleSvg()`의 레짐 리본이 `activeSnapshotAsset === "eth"` 조건 없이 그려지고 있어, BTC 선택 시 ETH 전용 `latestRegimeWide24` 데이터를 그대로 겹쳐 그리던 문제를 회색 "미지원" 플레이스홀더로 교체(§6.2에서 이미 "레짐분류기는 Tier4, 코인별 재학습 전엔 노출 금지"라고 정한 원칙을 프론트에서도 실제로 강제하게 됨). 실제 BTC 레짐분류기 학습은 미착수 — memory `eth-dashboard-btc-regime-classifier-not-trained-todo-20260831` 참고.
- 라이브 탭 자체를 제거(스냅샷·운영관리 2탭 체제로 축소)한 것은 이 설계도서의 스코프(코인 확장)와는 별개 결정이었으나, 그 결과 §9-a("라이브 탭·스냅샷 탭의 코인 선택 상태 공유 여부")는 **더 이상 유효한 질문이 아님** — 라이브 탭 자체가 없으므로 스냅샷 탭의 `activeSnapshotAsset`만 유일한 코인 선택 상태로 남음.
- 커밋 `60ab72b` + `handoff.sh push server` + 서버 프로세스 재시작 + curl 기반 실서버 재검증(ETH/BTC 청산방향압력·청산맵 값 구분 확인, SSE 스트림 정상, `trading_bot.py` 무변경) 완료.

**✅ 2026-08-31 후속 3 — XRP를 스냅샷 코인 스위처에 추가, 배포 완료.** 사용자가 위 §9-(b)에서 원래 추천됐던 파일럿 코인(XRP)을 이번에 실제로 진행. `coin_config.py`에 xrp 항목 추가(전용 `tail_risk_xrp.duckdb`/`tail_risk_1m_xrp` — BTC/SOL과 달리 완전히 독립된 워커+파일이라 단일-writer 경합 우려 자체가 없음, 서버 SSH로 5397+ rows·2026-08-27부터 누적 확인) + `server.py`의 `MARKET_SYMBOLS`에 xrp 추가 + 프론트 코인탭 1개 추가, 총 4파일. `server.py`의 4개 load_* 함수가 이미 완전히 제네릭했기 때문에 **함수 리팩터가 전혀 필요 없었음** — Step 1에서 예상한 대로 이 패턴이 진짜 기계적임이 실증됨. 커밋 `170c79a` + 배포 + 실서버 재검증(4개 엔드포인트 전부 `warmed_up=true`, 청산맵 현재가 $1.39 등 실제 XRP 고유값) 완료. §9-(b)는 이제 "XRP 파일럿 검증 완료, 다음은 SOL 또는 HYPE 판단"으로 갱신.

⚠️ **이 저장소를 동시에 건드리는 다른 세션과의 충돌이 이 문서 작업 중에만 4회 발생**(BTC 배포 2회 + 라이브탭 제거 1회 + XRP 1회, 매번 호메로스 증거신호 프로젝트 관련) — 매번 커밋 직전 `git diff`로 훅 단위까지 확인해 분리 커밋 처리함. 이 devmachine에서 대시보드 파일을 커밋하기 전엔 이 확인을 습관화할 것.

**✅ 2026-08-31 후속 4 — SOL을 스냅샷 코인 스위처에 추가, 배포 완료.** XRP보다도 작업량이 적었음 — `server.py`의 `MARKET_SYMBOLS`에 sol이 이미 있어서(옛 라이브 탭 다중자산 차트가 남긴 것) `coin_config.py`에 sol 항목 추가 + 프론트 코인탭 1개, 총 3파일뿐. SOL의 tail-risk는 BTC와 **같은 파일**(`tail_risk_btc_sol.duckdb`)의 별도 테이블(`tail_risk_1m_sol`, 19769+ rows — BTC와 같은 워커가 같은 시점부터 수집해 온 것)이라 이번엔 이 파일이 이미 §3에서 확인된 경로 그대로 재사용됨. 커밋 `a722cda` + 배포 + 실서버 재검증(4개 엔드포인트 전부 `warmed_up=true`, ETH/BTC 회귀 없음, `trading_bot.py` 무변경) 완료. 이걸로 ETH/BTC/SOL/XRP 4코인 중 **HYPE만 미착수**(과거데이터 없어 Tier1~2도 제한적, §7 참고) — Tier 1~2 범위에서는 사실상 완결.

---

## 9. 결정이 필요한 지점 (착수 전 확인)

- ~~(a) 코인 선택 상태 공유 범위~~: **2026-08-31 라이브 탭 제거로 소멸** — 스냅샷 탭의 `activeSnapshotAsset`만 유일한 코인 선택 상태로 남음(위 "후속 2" 참고).
- ~~(b) 4코인 동시 진행 vs 순차 진행~~: **2026-08-31 XRP까지 완료로 사실상 해소** — 실제로는 BTC → XRP 순으로 진행했고 둘 다 Tier 1~2까지 실서버 배포 완료(위 "후속 3" 참고). 다음 후보는 SOL(BTC와 유사한 검증 필요) 또는 HYPE(과거데이터 없어 Tier1~2도 제한적, §7 참고).
- **(c) Tier 4 콘텐츠의 코인별 노출 기준**: IC 재검증 통과를 노출 조건으로 강제할지(§6.5 추천안), 아니면 "참고용" 표시로 미검증 상태로도 먼저 노출할지.
- **(d) 교차자산 신호(smt_divergence류)의 코인별 재정의**: BTC 대시보드의 교차자산 파트너를 무엇으로 할지(ETH? 아니면 폐기?).
- **(e) `hyperliquid_execution_audit.jsonl`의 스코프 포함 여부**: HYPE의 Binance 데이터 수집과는 별개로 보이는 이 실행감사로그가 이번 대시보드 확장과 관련 있는지 확인 필요 — 별도 조사를 원하시면 말씀해주세요.

---

## 부록 — 참고 파일 경로

- 대시보드: `dashboard/server.py`, `dashboard/live/{index.html,app.js,styles.css}`
- 신호 계산: `scripts/live_{evidence_signal_dashboard_20260823,eth_sweep_v_rebound_signal_20260829,evidence_signal_metalabel_20260829,spot_perp_basis_signal_20260827,liquidation_5m_signal_20260825,liquidation_direction_signal_20260825,liquidation_map_20260824,regime_gbm3_signal_20260826,regime_wide24_signal_20260826,session_volatility_alert_20260826,macro_calendar_20260826}.py`
- 데이터 수집: `microstructure_scanner.py`, `tail_risk_interceptor.py`, `oi_lsratio_collector.py`, `l2_anomaly_snapshot_collector.py`, `liq_magnet_collector.py`, `scripts/duckdb_persist_worker.py`, `scripts/ops/supervisor_{xrp_worker,hype_worker,tail_risk_btc_sol_worker,oi_lsratio_worker,l2_anomaly_{btc,sol,xrp,hype}}.sh`
- 참고 문서: `docs/panel_universe_coverage_report_20260804.md`
- 관련 아티팩트: [스냅샷 멀티코인 확장](https://claude.ai/code/artifact/fc6d82be-f60a-46ed-9d07-4e8ff1660226)(2026-08-27, BTC/SOL 스코핑 — 이 문서가 상위호환)
