# 대시보드 오더플로우: 풋프린트(exocharts)·호가 히트맵(bookmap) 설계
2026-09-14 · 상태: **설계안, 착수 전 컨펌 대기**

## 0. 무엇을 만드는가

두 개의 서로 다른 차트다. 데이터 소스도, 저장 포맷도, 렌더 방식도 다르다. 한 덩어리로 취급하면 안 된다.

| | 풋프린트 (exocharts) | 호가 히트맵 (bookmap) |
|---|---|---|
| 보는 것 | **체결된** 거래를 봉×가격빈으로 쪼갠 매수/매도량·델타·POC | **아직 체결 안 된** 지정가 잔량이 시간에 따라 쌓이고 빠지는 모습 |
| 원천 | aggTrade (체결) | L2 호가창 전체 깊이 |
| 과거 재구성 | **가능** (data.binance.vision 일별 zip) | **불가능** — 아래 §2 |
| 저장 | 월별 parquet, ~0.5 MB/일 | 시간별 고정폭 래스터, ~84 MB/일 |
| 렌더 | canvas + 텍스트 | canvas `ImageData` 1장 |

만드는 유일한 정당성: **내 신호·내 주문·내 포지션을 같은 격자에 겹쳐 보기.** 그게 빠지면 exocharts 구독이 더 싸다.

## 1. 이번 세션에서 실측한 숫자 (추정 아님)

```
REST  /fapi/v1/depth?limit=1000   → 40,814 B, 지연 0.10s, 커버 범위 ±0.43%, weight 20
REST  /fapi/v1/depth?limit=500    → 20,476 B,            커버 범위 ±0.21%, weight 10
WS    ethusdt@depth@500ms         → 6.2 KB/s = 0.55 GB/일, 209 레벨갱신/msg (~420/초)
벌크  ETHUSDT-aggTrades-일별.zip   → 14.6 MB/일 (1년 ≈ 5.3 GB 다운로드, 저장은 집계 후 ~180 MB)
기존  binance_data/bookDepth      → ±1~5% **누적** 깊이 30초 간격. 가격별 아님 → 히트맵 불가
기존  depth20@100ms (수집중)       → 상위 20호가 = ETH 에서 폭 약 $0.20 → 히트맵 불가
```

## 2. 하드 제약 3개 (설계를 지배한다)

**(a) 히트맵의 과거는 존재하지 않는다.** 저장소에 있는 L2 는 ① `depth20@100ms`(폭 $0.20) ② `l2_anomaly_snapshots.duckdb`(트리거 순간만, 58,843행/35이벤트) ③ 벌크 `bookDepth`(가격별 아닌 ±%누적) 뿐이다. 셋 다 히트맵을 못 만든다. **오늘 수집을 시작해야 내일부터 보인다** — 그래서 §10 의 P0 가 UI 보다 먼저다.

**(b) REST weight 는 트레이딩 봇과 공유한다.** `depth?limit=1000` 을 1Hz 로 폴링하면 1,200 weight/분(상한 2,400/분/IP)을 혼자 먹는다. 실주문 경로와 같은 IP 다. → **REST 폴링 금지, WS diff 스트림(`@depth@500ms`) + 최초 스냅샷 1회**. weight 비용 ~0, 대역폭 0.55 GB/일.

**(c) `app.js` 는 396 KB 단일 파일이고, 문법오류 하나로 대시보드가 두 번 죽었다**(`},,` 사고). 오더플로우 코드는 **절대 app.js 에 넣지 않는다.** 별도 `flow.js` 로 분리하고 탭 진입 시에만 로드한다 — 버그가 나도 나머지 대시보드는 산다.

## 3. 아키텍처

```
[수집: 서버, 봇과 완전 분리된 단일 프로세스]
  scripts/live_orderflow_collector_20260914.py
    ├ WS ethusdt@depth@500ms ──→ 로컬 호가북(dict) ──1초마다──→ 래스터 1행 append
    │    └ 최초/재동기: REST depth?limit=1000 스냅샷 + pu 체인 검증
    └ WS ethusdt@aggTrade ─┬─→ 분×가격빈 누적 ──1분마다──→ 풋프린트 append
                           └ REST aggTrades?fromId= 갭필 (WS 유실 보정, §8)
        ↓                                    ↓
[저장]  data/live/orderflow/raster/ETHUSDT/  data/live/orderflow/footprint/ETHUSDT/
        2026-09-14T07.f32  (고정폭)          2026-09.parquet
        ↓                                    ↓
[API]   GET /api/flow/heatmap  (octet-stream) GET /api/flow/footprint (json)
        → 계산 없음. seek + read 만.          → parquet 범위 스캔
        ↓
[렌더]  dashboard/live/flow.js — canvas 2장, 공유 시간·가격축
```

**분리 원칙**: 수집기는 트레이딩 봇/기존 수집기의 어떤 파일도 건드리지 않는다(이 저장소의 확립된 규약 — `l2_anomaly_snapshot_collector.py` 도크스트링 참조). 자기 WS, 자기 디렉터리.

## 4. 저장 포맷 — 접근 패턴에 맞춰 **다르게** 간다

### 4.1 히트맵 = 고정폭 플랫파일 (duckdb 아님)

히트맵은 질의 대상이 아니라 **이미지**다. 행 하나 = 1초 = 이미지 한 열.

```
row (976 B, 리틀엔디안):
  int64   ts_ms          그 초의 시작 (UTC)
  int32   bin_lo         맨 아래 빈 인덱스 = round(price/BIN)
  float32 mid            중간가. NaN = 그 초 북이 무효(재동기중/수집중단)
  float32[240] qty       잔량. **부호가 방향**: +매수호가(bid), −매도호가(ask), 0 없음
파일: <UTC시각>.f32, 시간당 3600행 = 3.4 MB
오프셋 = (ts_ms − hour_start) / 1000 × 976   ← 파싱 0, seek 1회
```

기본 격자: `BIN=$0.5`(ETH 에서 0.02%), `240빈 = $120 폭 ≈ ±2.4%`. 하루 84 MB, **보존 14일 = 1.2 GB**. 보존은 `rm` 한 줄(duckdb DELETE 는 파일이 안 줄어든다 — 이게 flat file 을 고른 진짜 이유).

| 노브 | 좁게 | 기본 | 넓게 |
|---|---|---|---|
| BIN | $0.25 | **$0.5** | $1 |
| 빈 수 | 160 | **240** | 320 |
| 폭 | ±0.8% | **±2.4%** | ±6.4% |
| 일 저장 | 56 MB | **84 MB** | 112 MB |

**무효 초도 반드시 한 행을 쓴다**(mid=NaN). 그래야 오프셋 산식이 성립하고, 렌더러가 **빈 열을 회색으로 그린다**. 수집이 끊긴 구간을 보간하면 없는 유동성을 그리는 것이므로 금지.

### 4.2 풋프린트 = 월별 parquet

```
footprint_1m(ts_min int64, price_bin int32, buy_qty f32, sell_qty f32,
             buy_trades i32, sell_trades i32)
```
`buy` = `is_buyer_maker == false`(매수 공격). 델타·POC·VA·불균형(대각 3:1)은 **읽을 때 파생** — 저장하지 않는다. 5분/15분 풋프린트도 1분을 합쳐서 만든다(별도 저장 금지).
ETH 하루 약 3만행 → parquet ~0.5 MB/일 → **1년 ≈ 180 MB**.

## 5. API 계약

```
GET /api/flow/heatmap?symbol=ethusdt&to=<ms>&cols=600&agg=1
  → 200 application/octet-stream
    헤더 32 B: magic"FLOW" u16 ver, f32 bin_size, u16 n_bins, i64 t0_ms, u16 dt_s, u16 cols
    본문: cols × (i32 bin_lo, f32 mid, f32[n_bins] qty)
  agg>1 이면 agg초를 max 로 합친다(numpy reshape-max, 수 ms). 그 외 계산 없음.

GET /api/flow/heatmap?symbol=ethusdt&since=<ms>        ← 라이브 꼬리, 1초 1행(~1 KB)
GET /api/flow/footprint?symbol=ethusdt&tf=1m&from=&to= → json rows
```

**요청 경로에서 모델·pandas 를 돌리지 않는다**(대시보드 사고 전례). 파일 읽기는 공유 to_thread 풀이 아니라 **전용 `ThreadPoolExecutor(max_workers=2)`** 로 격리한다 — 공유 풀 고갈로 이미 한 번 장애가 났다.
라이브 갱신은 **1초 폴링**. `/api/events`(SSE)에 얹지 않는다 — 그 채널은 상태 이벤트용이고, 초당 바이너리를 섞으면 둘 다 망가진다.

## 6. 렌더 설계 (`dashboard/live/flow.js`, canvas 2D)

**히트맵 — 핵심 트릭**: 600열 × 240빈 = 144,000 셀이지만 `createImageData(600,240)` 1장을 채워 `imageSmoothingEnabled=false` 로 `drawImage` 확대하면 **드로우콜 1회**다. 셀당 `fillRect` 를 부르면 모바일에서 즉사한다.
- 색: `log1p(qty)` 를 분위수(그 창의 p99)로 정규화. 선형 스케일은 벽 하나가 전부를 먹어 아무것도 안 보인다.
- 오버레이(이미지 위, 별도 레이어): 체결점(반지름 ∝ 수량, 색 = 공격 방향) · best bid/ask 선 · **내 진입/청산/TP/SL 선** · `/api/chart-markers` 신호 마커.

**풋프린트**: 봉 하나당 가격빈 행. 확대 시 `매도수량 × 매수수량` 텍스트, 축소 시 델타 색막대로 자동 강등(텍스트 측정은 canvas 에서 제일 비싸다). 하단에 누적델타(CVD) 서브플롯.

**공통 인터랙션**: 휠 줌(시간), Shift+휠(가격), 드래그 팬, 핀치(모바일), 십자선 수치 표시, `LIVE` 토글(켜져 있으면 우측 고정).

## 7. 무결성 규율

- **풋프린트 완전성 검증**: 분별 `sum(buy_qty+sell_qty)` 가 같은 분의 kline `volume` 과 일치해야 한다(상대오차 1e-6). 불일치 = 체결 유실 → 그 날은 벌크 zip 으로 덮어쓴다. 체결이 빠지면 델타가 **조용히** 틀어지므로 이 검사는 선택이 아니다.
- **매일 새벽 재조정**: 전날 벌크 zip 이 권위다. 라이브 누적본을 지우고 다시 쓴다.
- **호가북 재동기**: `pu` 체인이 끊기면 즉시 무효 표시 → REST 스냅샷 재취득. 그 사이 초는 mid=NaN.
- **이건 보는 도구지 신호원이 아니다.** 여기서 피쳐를 뽑는 순간 CLAUDE.md 의 사건-라벨 경계 계약이 적용된다(피쳐 창 끝 ≤ 라벨 시작 − 1).

## 8. aggTrade WS 는 이미 신뢰할 수 없다고 문서화돼 있다

`l2_anomaly_snapshot_collector.py` 주석(2026-08-28): aggTrade WS 가 24시간+ 0행이었고 `microstructure_scanner.py` 는 그동안 REST 폴링으로 조용히 메워 왔다. 그러니 처음부터 **WS + REST `fromId` 갭필**을 함께 넣는다. 매분 aggId 연속성을 확인해 구멍이 있으면 그 구간만 REST 로 채운다.

## 9. 배포 규율 (이 저장소에서 실제로 터진 것들)

- 새 파일은 **만든 세션이 그 자리에서 커밋**한다. 서버에만 rsync 하면 watcher stash pop 이 나중에 충돌한다(2026-08-24·09-01 실제 다운 2회).
- main 머지 전 `bash scripts/ops/check_deploy_drift.sh` → 종료코드 0 아니면 머지 금지.
- 서버 파일 덮어쓰기 전 md5 대조(동시 세션이 같은 파일을 배포했을 수 있다).
- `flow.js` 는 `?v=` 캐시버스터 필수(Cloudflare immutable 캐시가 날짜 버스터를 무력화한 전례).
- 수집기는 `scripts/ops/supervisor_orderflow_collector.sh` 로 감시(기존 supervisor 복사). 워처는 `scripts/live_*.py` 변경 시 대시보드를 재시작하지 않으므로 수집기 재시작은 supervisor 몫이다.

## 10. 단계 (각 단계는 **검증 통과** 로 끝난다)

**P0 — 수집만, UI 0줄 (반나절)**
수집기 + supervisor 배포. 히트맵 과거는 만들 수 없으므로 **가장 먼저** 띄운다.
→ 검증: 1시간 뒤 `.f32` 가 정확히 3600행 · NaN 비율 <1% · 파이썬으로 PNG 한 장 렌더해 육안 확인. UI 를 쓰기 전에 데이터부터 눈으로 본다.

**P1 — 풋프린트 (과거 데이터로 바로 가능)**
벌크 백필(우선 최근 90일) + `/api/flow/footprint` + `flow.js` 풋프린트 캔버스.
→ 검증: 볼륨 일치 검사 100% 통과 · 임의 3개 봉을 exocharts 와 눈으로 대조.

**P2 — 히트맵 UI** (P0 가 7일 이상 쌓인 뒤)
→ 검증: 600열 창에서 프레임 16ms 이하(모바일 포함) · 수집 중단 구간이 회색으로 보일 것.

**P3 — 겹치기 (만드는 이유)**
내 포지션/주문/TP·SL, 신호 마커, 청산맵 레벨을 두 차트에 얹는다.

## 11. 열려 있는 결정 (기본값으로 진행 가능)

| 결정 | 기본값 | 바꿀 때의 비용 |
|---|---|---|
| 심볼 | ETH 만 | BTC·SOL 추가 = 저장·대역폭 ×3 |
| 히트맵 보존 | 14일 (1.2 GB) | 30일 = 2.5 GB |
| 가격빈 | $0.5 / 240빈 (±2.4%) | §4.1 표 |
| 풋프린트 백필 | 최근 90일 | 1년 = 5.3 GB 다운로드, 저장 180 MB |
| 시간 단위 | 히트맵 1초 / 풋프린트 1분 | 0.5초 = ×2 |

## 12. 만들지 않는 것

지연/리플레이 모드, 다중 거래소 통합 북, 주문 단위 추적(아이스버그 탐지 — Binance 는 주문ID를 안 준다, 원리적으로 불가), 히트맵 기반 자동매매 신호, 데스크톱 앱.

## 13. 총 코드량 추정

```
scripts/live_orderflow_collector_20260914.py        ~350줄  (WS 2개 + 북 유지 + 래스터/풋프린트 append)
scripts/backfill_footprint_from_bulk_20260914.py    ~120줄
scripts/ops/supervisor_orderflow_collector.sh        ~20줄  (기존 복사)
dashboard/server.py                                  ~60줄  (엔드포인트 2개, 계산 없음)
dashboard/live/flow.js                              ~450줄  (캔버스 2종 + 인터랙션)
index.html 탭 1개 + styles.css                       ~40줄
                                                  ─────────
                                                    ~1,040줄
```
