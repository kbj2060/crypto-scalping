# 실시간 수집 데이터 인벤토리 (2026-09-16)

서버(`server` 호스트, `/home/llewyn/crypto-scalping`)에서 **지금 쌓이고 있는** 데이터 전수.
프로세스 목록 · crontab · 파일 mtime · duckdb 행수를 읽기전용으로 직접 대조해 만들었다.
기준 시각 **2026-09-16 19:58 KST**. 숫자는 그 시점 실측이다.

재확인 방법은 §6.

---

## 1. 연속 원천 (WebSocket, 쉬지 않고 쌓임)

| 데이터 | 스트림 | 심볼 | 해상도 / 포맷 | 저장 위치 | 수집 시작 | 실측 증가율 | 보존 |
|---|---|---|---|---|---|---|---|
| 호가 래스터 | `@depth@500ms` | ETH | 1초 × 240빈($0.50), 고정폭 `.f32` 976B/행 | `data/live/orderflow/raster/ETHUSDT/<UTC시각>.f32` | 09-14 21:00 | ~81MB/일 | **14일** (`OF_RETENTION_DAYS`) |
| 최우선 호가 | `@bookTicker` | ETH | 틱 단위(불규칙), 32B 고정폭 + 시각별 gzip | `data/live/orderflow/bookticker/ETHUSDT/<UTC시각>.bt[.gz]` | 09-14 23:00 | ~168MB/일 | **무기한** (§4) |
| 주문단위 호가변경 | `@depth@100ms` | ETH | 전체북 diff, JSONL + 시각별 gzip, 시각당 REST 스냅샷 1건 | `data/live/orderflow/depthdiff/ETHUSDT/<UTC시각>.jsonl[.gz]` | 09-15 00:00 | ~456MB/일 | **무기한** (§4) |
| 체결 테이프 | `@trade` | ETH | 1초 × $0.1빈 | `data/live/trade_tape.duckdb` (`trade_tape_1s`) | **09-16** | ~326k행/일 | 무기한 |
| 마이크로구조 1분 | `@aggTrade` + `@depth20@100ms` + REST | ETH·BTC·SOL·XRP·HYPE | 1분 34컬럼 | `microstructure.duckdb`, `microstructure_{xrp,hype}.duckdb` | ETH 05-03 / BTC·SOL 07-14 / XRP 08-27 / HYPE 08-28 | 1,440행/일/심볼 | 무기한 |
| 테일리스크 1분 | `@forceOrder` | ETH·BTC·SOL·XRP·HYPE | 1분 16컬럼 | `tail_risk.duckdb`, `tail_risk_btc_sol.duckdb`, `tail_risk_{xrp,hype}.duckdb` | ETH 05-03 / BTC·SOL 08-17 / XRP·HYPE 08-27~28 | 1,440행/일/심볼 | 무기한 |

행수(09-16): `microstructure_1m` ETH 179,252 · BTC 89,330 · SOL 89,351 · XRP 28,937 · HYPE 28,068 /
`tail_risk_1m` ETH 179,516 · BTC 43,277 · SOL 43,282 · XRP 28,937 · HYPE 28,068.

`trade_tape` 에는 자체 검증표가 붙어 있다 — `verify_1m`(분별 테이프 물량 vs kline `volume` 상대오차),
`gaps`(WS 끊긴 구간). **연구 쿼리는 `verify_1m` 을 조인해 나쁜 분을 빼고 쓴다.** 구멍은 메우지 않고 기록만 한다.

## 2. 사건 트리거 스냅샷 (평소엔 안 쌓고, 터질 때만)

`l2_anomaly_snapshot_collector.py` — 청산버스트 z 와 가격이동 z 가 **동시에** 임계를 넘을 때만
링버퍼를 디스크로 flush 하고 이후 `POST_SECONDS` 동안 직행 기록. 심볼마다 독립 프로세스·독립 파일.
저장량이 시간이 아니라 **사건 수**에 비례한다.

| 심볼 | 이벤트 | depth행 | trades행 | 파일 크기 |
|---|---|---|---|---|
| ETH | 422 | 417,865 | 710,302 | 196MB |
| BTC | 371 | 339,763 | 707,140 | 120MB |
| SOL | 348 | 356,782 | 209,704 | 133MB |
| XRP | 326 | 333,918 | 196,555 | 130MB |
| HYPE | 264 | 242,502 | 325,180 | 86MB |

시작 2026-08-26. 심볼마다 마지막 행 시각이 몇 십 분씩 다른 것은 **정상**이다 — 트리거가 안 걸린 것이지
수집이 멈춘 게 아니다. 이 표로 «수집 중단»을 판정하지 말 것.

## 3. 주기 폴링 (REST)

| 데이터 | 주기 | 심볼 | 저장 | 시작 | 현재량 | 구동 |
|---|---|---|---|---|---|---|
| OI · 롱숏비율 | 5분 | ETH·BTC·SOL·XRP·HYPE | `oi_lsratio*.duckdb` 19컬럼 | 08-12 | 각 ~10,065행 | `@reboot` supervisor |
| 청산 자석 | 1분 | ETH | `liq_magnet_history.duckdb` | 08-25 | 31,715행 | `@reboot` supervisor |
| Deribit GEX · 옵션체인 | 매시 | ETH | `deribit_gex.duckdb` | ~63일 전 | 요약 1,518 / 체인 1,341,043행 | crontab `0 * * * *` |
| altdata | 매일 01:00 | — | `data/research/altdata.duckdb` | — | 1.0MB | crontab `0 1 * * *` |

OI/롱숏비율을 우리가 직접 모으는 이유: 바이낸스 `openInterestHist` 계열은 `period` 와 무관하게
**항상 500포인트만** 준다(5m 면 ~1.7일). 그 창은 달력과 함께 자라지 않으므로 지금부터 모으는 수밖에 없다.

## 4. bookTicker · depthDiff 는 의도적으로 무기한 보존 (2026-09-16 사용자 결정)

둘 다 코드에 `RETENTION` 상수가 **아예 없다**(raster 만 14일). 이건 누락이 아니라 결정이다 —
방향 축에 쓰일 수 있는 재료이고, **호가는 지금 안 받으면 소급 재구성이 영원히 불가능하다**
(체결은 `data.binance.vision` 일별 zip 으로 언제든 되살릴 수 있지만 호가 잔량은 공개되지 않는다).

용량 전망: 합 ~624MB/일. 2026-09-16 기준 루트 여유 **545GB** ⇒ 약 **2.4년**. 다른 것이 안 늘어난다는
가정이므로, 여유가 100GB 밑으로 내려가면 그때 재검토한다.

## 5. 파생 · 섀도우 원장 (모델이 만드는 실시간 기록)

- `decision_feature_snapshot.jsonl` — 372MB / 28,866행(행당 ~13KB, 235컬럼 프레임). 봇 결정마다 append
- `data_pipeline_health.jsonl` — 107MB / 29,002행
- `exit_ladder_shadow.jsonl` — 3,164행 (09-13~)
- `shadow_1d_top10_20260915/decisions.csv` — 방향 게이트 섀도우 (09-16 실제 시작)
- `maker_fill_shadow.duckdb` — **ETH 만 가동 중**. BTC·SOL·XRP·HYPE 4종은 09-15 06:29 정지(수렴 완료)
- 상태파일(이력 아님, 매번 덮어쓰기): `eth_vol_forecast_state` · `eth_extreme_detector_state` ·
  `evr_gate_state` · `regime_{wide24,btc,xrp}_state` · `eth_v_rebound_state` · `liq_burst_state_*` ·
  `footprint_eth.json` · `dashboard_state.json`

⚠️ **상태파일의 mtime 은 «돌고 있는 코드»를 말하지 않는다.** 옛 판 워커가 계속 갱신할 수 있다.
판정은 키 구성과 프로세스 시작시각으로 한다.

## 6. 함정 · 이 인벤토리를 다시 만드는 법

**좀비 표 — 살아 있는 DB 안의 죽은 표.** 모르고 조인하면 조용히 2행이 나온다:

| 표 | 행수 | 마지막 | 진짜 위치 |
|---|---|---|---|
| `tail_risk.duckdb::tail_risk_1m_btc` / `_sol` | 각 **2** | 2026-08-17 | `tail_risk_btc_sol.duckdb` 로 이사함 |
| `microstructure.duckdb::decision_feature_frame` | 111 | 2026-07-02 | — (정지) |
| `microstructure.duckdb::..._omega5_event_risk_governor_20260702` | 2 | 2026-07-02 | — (정지) |

**연속 수집은 전부 ETH 단독이다.** 래스터·bookTicker·depthDiff·체결테이프 4종 모두 `ethusdt`.
심볼 확장은 환경변수 한 줄(`OF_SYMBOL` / `BT_SYMBOL` / `DD_SYMBOL` / `TAPE_SYMBOL`)이지만 용량이 심볼 수에 비례한다.

**수집기는 전부 트레이딩 봇과 분리돼 있다** — 자기 WS · 자기 파일 · 주문 없음. 죽어도 봇에 영향이 없다.
이 저장소의 확립된 규약이고, 새 수집기도 이 규약을 따른다.

**재확인 절차** (읽기전용, 서버를 건드리지 않는다):

```bash
source scripts/ops/handoff.hosts.conf   # HOSTS[server]
# 1) 무엇이 돌고 있나
ssh -p <port> <user@host> "ps -eo pid,etimes,args --sort=-etimes | grep -E 'collector|worker|_supervise'"
# 2) 무엇이 방금 쓰였나 (2시간 안)
ssh ... "cd <repo> && find data/live -type f -newermt '-2 hours' -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' | sort -k4"
# 3) duckdb 행수/시간범위 — read_only=True 로 열고 즉시 닫는다
```

⚠️ duckdb 는 **프로세스 하나만** 쓸 수 있다. 연결을 붙들고 있으면 워치독의 읽기 검사가 BLOCKED 된다.
`read_only=True` 로 열고 바로 닫을 것. 쓰기 중인 파일은 `Could not set lock` 이 날 수 있는데(수집기가
flush 중) 이건 정상이고 **재시도 루프를 돌리지 말 것** — 잠시 뒤 다시 하면 열린다.

## 7. 2026-09-16 정리 내역

멈춘 micro_scalp 섀도우 duckdb **5개 / 1.06GB** 를 `data/live/` 에서 삭제했다(2026-07-19~21 정지).
백업 드라이브 `/mnt/d/crypto-scalping-backups/data/live/` 에 동일 크기로 남아 있고, 열어서 행수까지
대조한 뒤 지웠다 — `backup_live_data.sh` 는 의도적으로 `--delete` 를 쓰지 않으므로 백업은 유지된다.

```
btc_micro_scalp_shadow.duckdb            450MB
sol_micro_scalp_shadow.duckdb            448MB
eth_micro_scalp_v4_shadow.duckdb         159MB
sol_micro_scalp_entry_shadow.duckdb       43MB
eth_micro_scalp_lifecycle_shadow.duckdb   42MB
```

같이 지운 것: `scripts/run_live_parity_drift_monitor.py` 의 `LIVE_DBS` 에서 이 5개 경로.
남겨두면 `connect_retry` 가 DB 당 56초씩 D3·D4 두 단계에서 재시도해 6시간 cron 이 매번 ~9분을 헛되이 잔다.

**남겨둔 것**(요청 범위 밖, 판단은 다음 세션에):
- `dashboard/server.py` 의 `SCALP_SHADOW_ASSETS` / `SCALP_REUSE_MODES` 와 `/api/scalp-shadow`,
  `/api/scalp-reuse-shadow` 라우트. **프런트엔드는 이미 이 엔드포인트를 부르지 않는다**(`app.js`/html 참조 0건).
  DB 가 사라졌으니 이제 5개 모두 503 을 돌려준다(ETH·lifecycle 은 삭제 전부터 이미 503 이었다).
  테스트 3개가 이 심볼들을 참조하므로 제거하려면 그쪽까지 함께 봐야 한다.
- `.bak` duckdb 2개: `microstructure.duckdb.bak_pre_usdc_purge_20260802`(77MB),
  `deribit_gex.duckdb.bak_pre_dev_orphan_merge_20260823`(21MB)
- micro_scalp 섀도우 로그 `data/live/micro_scalp_shadow_*.{log,err}` 약 55MB
