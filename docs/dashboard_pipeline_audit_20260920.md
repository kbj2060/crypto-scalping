# 대시보드 파이프라인 — 데이터 흐름 지도와 감사 (2026-09-20)

개편(2026-09-15~09-20: 풋프린트 · 1초 수급 · 가격축 프로파일 · 호가 히트맵 · OI 레인이
캔들 SVG 안으로 들어옴) 이후의 전체 경로를 **실측으로** 확인하고, 거기서 나온 결함을
고친 기록이다. 숫자는 전부 재현 방법을 같이 적었다 — 추정치는 «추정»이라고 쓴다.

계측 환경
- 서버 실측: 라이브 서버의 `127.0.0.1:8787` 에 읽기 전용 curl (3회 반복, warm/cold 구분).
- 브라우저 실측: 라이브 서버에서 뜬 **실제 페이로드 30종**을 픽스처로 떠서 dev 에
  목 서버로 되먹이고, headless Chromium(playwright) 으로 CDP `Performance.getMetrics`
  20초 소크 + `performance.now()` 마이크로벤치. 같은 픽스처라 before/after 가 같은 입력이다.

---

## 1. 층 지도

```
   [거래소]                    [수집/워커 프로세스]              [대시보드 서버]            [브라우저]
                                                                 dashboard/server.py       app.js
 Binance @trade  WS ────────────────────────────────────────▶ footprint_state["bars"]  ─┐
   (ETHUSDT)     │                                              (메모리 288봉 링)        │
                 │                                             footprint_state["sec"]   │
                 └──────────────────────────────────────────▶   (메모리 360초 링)       │
                                                                       │                 │
 Binance @trade  WS ──────────────────────────────────────────────────────────────────▶ │ 브라우저 직결
   (현재가/진행봉)                                                     │                 │  (서버 우회)
                                                                       │                 │
 /fapi/v1/openInterest  0.25초 폴링 ─────────────────────────▶ oi_1s{} (360초 링)       │
                                             └── 10초 flush ─▶ data/live/oi_1s.duckdb ──┤
                                                                                         │
 /fapi/v1/klines 5m/1h  ───────────────────────────────────▶ swr 캐시(프레임/청산맵)  ──┤
 /fapi/v1/ticker/price  1초 ────────────────────────────────▶ SSE 팬아웃 ───────────────┤
 /fapi/v2/*, userTrades ───────────────────────────────────▶ binance_account 캐시 ─────┤
                                                                                         │
 depth WS ──▶ orderflow_raster 수집기 ──▶ data/.../*.f32 (seek) ─▶ /api/flow/heatmap ───┤
 forceOrder ─▶ tail_risk_interceptor ──▶ tail_risk.duckdb (r/o) ─▶ 청산 신호·이력 ──────┤
 trading_bot.py ──────────────────────▶ data/live/dashboard_state.json ─▶ SSE ──────────┤
 각종 워커(레짐·극점·V자·게이트·사이징·GEX·거시) ─▶ data/live/*.json ─▶ worker_payload ─┘
```

**규약 셋** (이 저장소가 반복해서 배운 것):
1. **요청 경로에서 모델을 돌리지 않는다.** 무거운 계산은 전부 워커가 하고 서버는
   상태 파일만 읽는다(`worker_payload`). 2026-09-10 에 인라인 채점이 to_thread 풀을
   고갈시켜 전 엔드포인트가 멈춘 실장애가 근거다.
2. **비싼 것은 stale-while-revalidate.** `swr_cached(key, ttl, produce, max_stale=)`.
   `max_stale=0` 은 «낡은 걸 주지 말고 기다려라»(차트 프레임이 그렇다).
3. **화면 경로와 연구 아카이브를 가른다.** 체결은 메모리+json 스냅샷(화면) /
   `trade_tape.duckdb`(연구), 호가는 `.f32` seek(화면) / parquet(연구).
   `deribit_gex.duckdb` 는 아예 안 연다(단일 writer 충돌).

### duckdb 접촉면 (전부)

| 파일 | 대시보드의 역할 | 보호 |
|---|---|---|
| `data/live/oi_1s.duckdb` | **유일한 writer**(10초 flush) + reader(`read_only=True`) | `duckdb_path_lock` in-process 락 |
| `data/live/tail_risk*.duckdb` | reader only (`read_only=True`, `to_thread`) | 별 프로세스가 writer |
| `*_shadow.duckdb` | mtime/size 만(etag) + 섀도우 페이로드 | — |
| `deribit_gex.duckdb` | **열지 않는다** — 수집기 JSON 만 읽는다 | 단일 writer 회피 |
| `trade_tape.duckdb` | **열지 않는다** — 화면 복원은 json 스냅샷(0.8ms vs 250ms) | 단일 writer 회피 |

---

## 2. 실측 — 서버는 병목이 아니다

`/api/*` 3회 연속, gzip 협상 (라이브 서버, 2026-09-20):

| 엔드포인트 | cold | warm | gzip 크기 | 클라 폴링 |
|---|---|---|---|---|
| `/api/footprint?bars=12` | 1.7ms | **1.2ms** | 3.1KB | 400ms |
| `/api/footprint?bars=48` | 3.9ms | **3.7ms** | 12.5KB | 400ms |
| `/api/supply-1s` | — | **1.3ms** | 2.5KB | 1s |
| `/api/supply-profile?bars=12` | — | **0.6ms** | 1.0KB | 5s |
| `/api/flow/heatmap?…&mode=rows` | 14.8ms | **0.6ms** | 5.8KB | 1s (서버 1초 SWR) |
| `/api/state` | — | **1.3ms** | 5.6KB | (SSE 전용) |
| `/api/oi-5m?bars=96` | 15.4ms | **12.1ms** ← 캐시 없음 | 1.5KB | 15s |
| `/api/market-history` | 133ms | **3.6ms** | 1.7KB | 300s |
| `/api/liquidation-map` | 54ms | **2.0ms** | 6.2KB | 60s |
| `/api/chart-markers` | 597ms | **0.7ms** | 0.8KB | 60s |
| `/api/model-indicator-history` | — | 4.2ms | **26.1KB** | 1회(로드 차단) |

**결론: 서버·duckdb 왕복은 전부 밀리초대다.** 체감 지연의 원천은 서버가 아니라
(a) 클라 폴링 주기 (b) 브라우저 렌더 비용 (c) 필요 없는 바이트다. (a)는 사용자가
의도적으로 고른 값이라 손대지 않았고, (b)(c)를 고쳤다.

---

## 3. 고친 것

### 3.1 🔴 톤 띠가 «4시간»이라면서 4시간을 안 보여주고 있었다 (정확성)

`toneHistory`(수급 흐름 · 리테일 수급 · 청산 캐스케이드 · 변동성 수준)는 48칸 × 5분 =
4시간이라고 화면에 적혀 있고, 서버는 그 4시간을 `model_indicator_history.json` 에
**디스크로 남긴다**(배포 재시작이 잦아서 2026-09-12 에 일부러 넣은 것이다).

두 곳이 그 계약을 깨고 있었다.

1. `seedModelIndicatorHistory()` 가 48개 서버 샘플을 밀어 넣으면서 시각을 전부
   `new Date()` 로 찍었다 → **48칸이 모두 같은 순간**. 축이 4시간을 0초로 압축했다.
2. `render()` 의 라이브 푸시가 **SSE 상태 푸시마다** 한 칸씩 밀었다. 서버 실측
   `dashboard_state.json` 변경은 30초에 3회(≈10초 주기)라, 48칸이 **8분**이면 다 차고
   그 과정에서 서버가 남겨 둔 4시간이 통째로 밀려난다.

고침: `pushToneHistory(key, tone, at)` — `at`(씨앗)은 그 시각을 그대로 적고 주기 제한을
안 받는다. 라이브 푸시는 `TONE_PUSH_MIN_MS = 300000`(= 서버 `MODEL_INDICATOR_SAMPLE_SECONDS`)
으로 묶는다.

검증(같은 픽스처, headless): 띠 폭 **0분 → 251분**. 축 라벨
`안정 · 05:50:00~05:50:00` → `안정 · 03:49:19~05:50:00`.
자체점검: `node test/tone_history_cadence_20260920.js` (옛 코드에서는 실패한다).

부수 효과: `render()` 마다 바뀌던 `renderModelIndicatorList` 의 html 이 이제 5분에 한 번만
바뀌어, 6행 innerHTML 재작성이 그만큼 사라진다.

### 3.2 캔들 SVG 가 수급 패널 둘을 초당 2.5번 헛되이 다시 그렸다 (속도)

개편으로 «1초 수급»과 «가격축 프로파일»이 캔들 SVG 안의 중첩 `<svg>` 가 됐다.
캔들 SVG 는 풋프린트 모드에서 **체결이 올 때마다**(400ms 게이트) `innerHTML=""` 후
통째로 다시 만든다 — 두 패널도 같이 따라 만들어졌다. 그런데 그 둘의 입력은
0.2Hz(프로파일) · 1Hz(1초 수급) 로만 바뀐다. 게다가 두 패널은 **자기 폴링으로도**
다시 그린다(히트맵 1초 · 수급 1초 · 프로파일 5초) — 프로파일은 초당 3.7번 그려지고 있었다.

마이크로벤치(1h 탭, 실제 페이로드): `renderSnapshotChart` 6.08ms 중
**프로파일 2.89ms + 1초 수급 0.99ms = 3.87ms (64%)**.

고침:
- 두 패널의 `<svg>` 노드를 **살려 두고 다시 붙인다**(`subPanelCache`). `innerHTML=""` 은
  DOM 에서 떼어낼 뿐 JS 참조가 쥔 서브트리는 살아 있다 — 재부착 실측 **0.01ms**.
- 다시 «그리는» 것은 판번호(`supplyProfileVer` / `flowHeatmapVer` / `supply1sVer`)와
  기하(w·h·진입가)가 바뀐 때뿐이다.
- 현재가는 판번호에 **넣지 않는다** — 그 한 줄만 `updateSupplyProfileNow()` 가 transform 으로
  옮기는 게 원래 설계다. 넣으면 틱마다 캐시가 깨져 최적화가 통째로 무효가 된다.

### 3.3 «box 가 오면 재지 않는다»가 실제로는 재고 있었다 (속도)

`renderSupplyProfileSvg` 는 `box` 를 받으면 그 값을 쓰지만, `svg.getBoundingClientRect()`
**호출 자체는 그대로** 하고 결과만 버렸다. 이 함수는 캔들 렌더 한가운데서 불리므로
그 한 줄이 방금 만든 수천 노드의 레이아웃을 강제로 확정시킨다(forced synchronous layout).
`renderSupply1s` 의 `parentElement.clientWidth` 도 같다. 둘 다 `box` 가 있으면 건너뛴다.

### 3.4 풋프린트 모드에서 청산밀도 이력을 만들어서 버리고 있었다 (속도)

`renderSnapshotChart` 가 `liquidationDensityHistory()`(9 스냅샷 × ~115빈 = 1,000여 개 객체)를
매 렌더 만들고, 바로 다음 줄에서 `footprint ? [] : densityHistory` 로 버렸다. 쓸 때만 만든다.

### 3.5 SSE 가 상태 전체(30KB)를 밀고 있었다 (대역)

`app.js` 의 `render()` 가 읽는 것은 `state.session` · `state.microstructure` ·
`state.tail_risk` 셋뿐이고, `compactState` 는 **인자로 넘어가기만 하고 본문에서 한 번도
안 읽힌다**. 그런데 SSE 는 `dashboard_state.json` 전체를 상태가 바뀔 때마다 그대로 밀었다.
SSE 는 gzip 도 안 걸린다(`json_compress_etag` 는 StreamResponse 를 건드리지 않는다).

실측 **30,109B → 3,498B (8.6배)**. 변경 주기 ≈10초라 클라당 ~3KB/s → ~0.35KB/s.
`/api/state` 는 계약이 다르므로 **그대로 전체**를 준다(테스트가 그쪽을 본다).
⭐지우는 키는 클라가 애초에 안 읽던 것이라 **옛 app.js 와도 호환**된다.
자체점검: `test/test_dashboard_sse_state_view_20260920.py` — 세 이름이 app.js 가 실제로
읽는 이름과 같은지, `compactState` 를 쓰기 시작하지 않았는지까지 본다.

### 3.6 페이지 로드를 막는 응답이 162KB 였다 (대역)

`app.js` 는 `await seedModelIndicatorHistory()` **다음에** SSE 를 연다. 그 응답이
`microstructure`/`tail_risk` 블록 전체 × 48샘플 = 161,915B(gzip 26KB)였는데,
클라이언트가 읽는 것은 `classifyIndicators()` 가 쓰는 **다섯 필드뿐**이다.
나갈 때만 추린다(버퍼·파일은 그대로 — 나중에 다른 필드를 쓸 수 있게):
**161,915B → 11,468B (14배), gzip 26.1KB → 2.5KB.**

### 3.7 코인 전환 경쟁 (정확성)

`refreshLiquidation5mSignal` / `refreshBasisLiquiditySignal` / `refreshLiquidationMap` 이
`?asset=${activeSnapshotAsset}` 로 요청을 걸고, 응답이 왔을 때 **화면이 이미 다른 코인**
이어도 그대로 덮어썼다. 2026-08-31 「ETH 레짐 리본이 BTC 캔들 위에」와 같은 부류다
(그쪽은 변수를 갈라 고쳤고 여기는 «시점»이 문제다). 요청 시점 코인을 들고 가서 다르면 버린다.
⚠️지금은 `DASHBOARD_ASSETS=eth` 하나라 잠복 상태지만, 전환기는 살아 있고 환경변수 하나로 켜진다.

### 3.8 `/api/oi-5m` — 이 저장소에서 매 요청 duckdb 를 여는 유일한 엔드포인트

warm 12~15ms 로 다른 엔드포인트(0.5~4ms)의 4~20배였고, 파일 락을 잡으므로 10초마다 도는
쓰기(`oi_1s_persist`)와 **탭 수만큼** 부딪힌다. TTL 5초 SWR — 클라 폴링(15초)의 1/3이라
단일 탭에서는 늘 미스(신선도 불변)이고, 탭이 늘어도 duckdb 열기는 5초에 한 번으로 묶인다.
값 자체가 5분봉이라 5초는 해상도 아래다.
(부수 관찰: 서버가 «링에서 바로 준다»며 넣은 `openInterest` 필드는 **화면이 안 읽는다**.)

### 3.9 죽은 배선 제거

`refreshLiquidationDirectionSignal()` 은 2026-09-11 에 본문이 비워졌는데 게이트 변수와
호출부 4곳이 남아 매 틱 돌고 있었다. 함수·상수·변수·호출 전부 제거(−25줄).

---

## 4. 전후 실측

### 4.1 브라우저 20초 소크 (CDP `Performance.getMetrics`, 같은 픽스처, 2회 평균)

| 창 | 지표 | before | after | |
|---|---|---|---|---|
| 1h | TaskDuration | 0.953s | **0.803s** | −16% |
| 1h | ScriptDuration | 0.251s | **0.160s** | −36% |
| 1h | LayoutCount | 635 | **446** | −30% |
| 4h | TaskDuration | 1.289s | **1.133s** | −12% |
| 4h | ScriptDuration | 0.385s | **0.300s** | −22% |
| 4h | LayoutCount | 648 | **415** | −36% |

### 4.2 `renderSnapshotChart` 1회 (마이크로벤치, 30회 평균)

| 구성 | before | after | 생성 노드 |
|---|---|---|---|
| 풋프린트 1h | 6.08ms | **4.49ms** | 744 → 551 |
| 풋프린트 4h | 7.56ms | **6.44ms** | 1385 → 1146 |
| 청산맵 1h | 6.22ms | **4.45ms** | 767 → 574 |

### 4.3 회귀 검증 — 픽셀이 아니라 **DOM 을 비교했다**

전체 페이지 스크린샷 픽셀 비교는 글립 안티앨리어싱만으로도 같은 빌드끼리 2% 넘게
흔들려(대조군으로 확인) 쓸 수 없었다. 대신 `#candleSvgSnapshot` · `#liquidationMapList` ·
`#snapModelIndicatorList` · `#snapSpecializedSignalList` 의 `outerHTML` 을 통째로 비교했다.

- **AFTER vs AFTER: 바이트 단위 완전 일치** (내 판이 결정적이다)
- **BEFORE vs AFTER: 다른 것은 톤 띠의 `data-t` 시각과 축 라벨뿐**
  (88 hunk / 182줄, 전부 §3.1 이 의도한 변경). 캔들 셀 · 프로파일 막대 · 호가 막대 ·
  OI/청산 레인 · 청산맵 목록은 **한 글자도 안 바뀌었다**.

---

## 5. 안 고친 것 (근거와 함께)

- **풋프린트 400ms 폴링의 중복.** 브라우저는 이미 자기 WS 로 진행 중인 봉을 쌓는다
  (`footprintLiveAdd` → `footprintMergeLive`). 폴링이 실제로 새로 주는 것은 *닫힌* 봉
  (5분에 한 번) · WS 가 봉 중간에 붙었을 때의 폴백 · `ready` 플래그뿐이다. 4h 탭에서
  12.5KB × 2.5/s = **31KB/s**. 줄이면 데이터 지연이 늘어날 수 있어(사용자 제약) 그대로 뒀다.
  줄이려면 폴링이 아니라 **SSE/WS 푸시**로 바꾸는 쪽이 맞다.
- **`/api/footprint` · `/api/supply-profile` 의 서버 캐시 없음.** 탭 수만큼 곱해진다
  (탭당 3~9.5ms/s). 캐시를 걸면 그만큼 낡아지므로 «느리게 하지 말 것» 제약과 충돌한다.
  입력(체결)이 초당 수백 번 바뀌어 «안 바뀌면 재사용» 키도 성립하지 않는다.
- **캔들 SVG 전체 재생성.** 풋프린트 모드에서 실제로 바뀌는 것은 맨 오른쪽 봉 하나인데
  48봉을 다 다시 만든다. 증분 렌더가 정답이지만 이 함수는 1,300줄이고 좌표 규약이 18곳에
  얽혀 있다 — 지금 손대면 «가장 작은 변경이 잘못된 자리»가 된다. 남은 4.5~6.4ms 는
  2.5Hz 기준 데스크톱 1.1~1.6% 다.
- **낡은 테스트 7건.** `test_dashboard_chip_ids_20260907.py`(4) ·
  `test_dashboard_execution_alert.py`(2) 는 **의도적으로 제거된 기능**(masht_anchor 칩 ·
  EVIDENCE_STRIP_CHIP_IDS · 실행 경보 표면)을 아직 요구한다. HEAD 에서도 똑같이 실패한다
  (확인함) — 코드 버그가 아니라 테스트 부패다. 어느 것을 은퇴시킬지는 사람이 정할 일이라
  손대지 않았다. `test_dashboard_server.py::StaticAssetCacheHeaderTest` 의 실패는
  이 워크트리에 `data/live` 가 없어서다(환경 문제, 서버에서는 통과).

---

## 6. 재현 방법

```bash
# 서버 응답 시간/크기 (읽기 전용)
ssh <server> 'for p in "/api/footprint?bars=48" "/api/oi-5m?bars=96"; do
  for i in 1 2 3; do curl -s -o /dev/null -H "Accept-Encoding: gzip" \
    -w "%{time_total}s gz=%{size_download}B  $p\n" "http://127.0.0.1:8787$p"; done; done'

# 자체점검
node test/tone_history_cadence_20260920.js
node test/render_supply_1s_smoke_20260920.js
node test/render_supply_profile_smoke_20260919.js
python3 -m pytest test/test_dashboard_sse_state_view_20260920.py -q
```

브라우저 소크/DOM 대조 하네스(목 서버 + playwright)는 스크래치패드에만 두었다 —
라이브 페이로드 픽스처가 필요해서 커밋하지 않는다. 다시 만들려면 위 §2 방식대로
30개 엔드포인트를 떠서 정적으로 되먹이면 된다.
