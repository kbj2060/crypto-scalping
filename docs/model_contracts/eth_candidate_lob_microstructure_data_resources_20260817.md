# ETH 후보 — 오더북(LOB) 마이크로구조 데이터 및 리소스 관리 (2026-08-17)

이 문서는 `docs/model_contracts/eth_candidate_lob_microstructure_contract_20260817.md` 서브
프로젝트에서 실제로 확인한 모든 데이터 소스/리소스를 모은 목록이다. 출발점은
[`docs/deep_learning_for_crypto_trading_literature_survey_20260817.md`](../deep_learning_for_crypto_trading_literature_survey_20260817.md)
5절(오더북/마켓 마이크로구조 DL) — 이 프로젝트가 아직 탐색하지 않은 유일한 DL 축으로 지목됐으나,
"풀뎁스 크립토 LOB 데이터 확보가 선결 과제"라고 명시했었다. 이번 스코핑에서 그 전제 자체가
**부분적으로 틀렸다는 것**을 발견했다 — 이미 이 저장소에 검증까지 끝난 파일럿 인프라가 있다.

상태 값 컨벤션: `활성`, `인프라 확인됨-미착수`, `인프라 차단`, `검증 완료 — 긍정 결과`,
`검증 완료 — 부정 결과`.

## 핵심 발견 (요약)

1. **원시 L2 레벨 저장(raw `bids_json`/`asks_json`)은 설계·구현·격리환경 검증까지 끝나 있었으나
   프로덕션에는 한 달간 배선되지 않은 채 방치돼 있었다.** 2026-08-17 이 세션에서 **E1(기존
   의사결정-조건부 스냅샷에 원시 20레벨 병행 저장) 슬라이스를 사용자 승인 하에 실제
   프로덕션에 배선·배포·검증 완료했다** — 상세는 아래 "프로덕션 배선 완료" 절.
   E2(10초 연속 스냅샷)는 별도 봇 지연 회귀 테스트가 필요해 이번엔 배선하지 않고 보류했다.
2. **이미 프로덕션에서 3개월(ETH)/2주(BTC/SOL) 동안 조건부 L2 요약이 쌓이고 있다** — 원시
   레벨은 아니지만 20레벨 깊이의 depth/imbalance/microprice 요약이다. 다만 의사결정
   시점에서만 기록되는 조건부 샘플링이라 WS-A 설계 문서가 스스로 지적한 편향(활동 구간에
   쏠림)을 안고 있다.
3. **이미 프로덕션에서 3.5개월(ETH) 연속 1분봉 오더플로우 파생 피처**(OBI, taker_buy_ratio,
   NIF, EAI, toxicity, queue-collapse, absorption 등 34컬럼)가 쌓이고 있다 — 이건 문헌
   5.2절의 "raw LOB보다 OFI 파생 피처가 낫다"(Kolm/Turiel/Westray) 및 "피처가 깊이를 이긴다"
   (Wang 2025)는 결론과 정확히 같은 종류의 자원이며, 신규 수집 없이 바로 연구 가능하다.

## 프로덕션 배선 완료 (2026-08-17, 이 세션)

사용자 승인(AskUserQuestion "프로덕션 배선 승인") 하에 WS-E E1 슬라이스를 실제로 배선·배포·검증했다.

- **변경**: `trading_bot_modules/orderbook_recorder.py`의 `summarize_orderbook()`이 이미 fetch된
  bids/asks를 `bids_json`/`asks_json`(JSON VARCHAR)으로 함께 직렬화. `_append_row_duckdb()`가
  기존 테이블에는 `ALTER TABLE ADD COLUMN IF NOT EXISTS`로 컬럼을 추가(기존 행은 NULL 유지),
  신규 테이블은 처음부터 포함. **새 거래소 호출/새 async 루프 없음** — 이미 의사결정 시점에
  fetch하던 오더북을 재사용만 함, 지연 리스크 최소화 근거.
- **검증(배포 전)**: 합성 데이터로 마이그레이션 경로 + 신규 테이블 경로 round-trip 0건 불일치.
  서버에서 받은 **실제 프로덕션 DB 사본**(ETH 14,093행 포함)에 대해서도 마이그레이션+쓰기+
  round-trip 0건 불일치, 기존 행 전부 미변경 확인.
- **배포**: git commit `d047de1` → push → GitHub Actions CI(syntax-check) 성공(57s) →
  `scripts/ops/deploy_watcher.sh`가 `trading-bot.service` 재시작, 60초 헬스체크 통과("deploy
  OK").
- **배포 후 실측 검증(서버 직접 조회)**: 재시작 직전 마지막 행(01:40 KST대)은 `bids_json IS
  NULL`(구코드), 재시작 후 첫 행부터 ETH/BTC/SOL 전부 `bids_json`/`asks_json` 채워짐(길이
  ~356~364자, 20레벨 [price,qty] 배열). 재시작으로 PID 616760→866017 변경 확인, 로그에
  트레이스백/예외 없음, `orderbook_recorder=ON` 정상 부팅 로그 확인.
- **에폭 경계 기록**: `data_epochs.json`에 확정 시각 기록 — ETH 2026-08-17T01:43:28+09:00,
  BTC 2026-08-17T01:43:16+09:00, SOL 2026-08-17T01:43:17+09:00. **이 시각 이전 행은 raw
  레벨 없이 요약만 있다 — 향후 연구가 이 경계를 넘어 raw 레벨을 요구하면 안 된다.**
- **참고**: 배포 시 확인된 바, 현재 라이브 계정은 `binance_execution=OFF dry_run=True
  testnet=True` 상태였다(실제 주문 미체결, 페이퍼 트레이딩) — 이번 재시작의 실질 리스크는
  애초에 기존에 알려진 것보다 낮았다.
- **보류(이번엔 안 함)**: E2(10초 연속 스냅샷, `orderbook_periodic_snapshots` 신규 테이블)는
  독립적인 새 fetch 루프가 필요해 봇 지연 회귀 테스트가 선행돼야 한다 — WS-E 설계 문서 자체의
  요구사항이며 이번 세션에서 검증되지 않았으므로 배선하지 않았다.

## 라이브 수집 데이터 (duckdb)

| 리소스 | 위치 | 커버리지 (2026-08-17 배포 후 실측) | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| `microstructure_1m` (ETH) | `data/live/microstructure.duckdb` | 136,041행+, 2026-05-03~현재(연속 1분봉) | OBI/taker_buy_ratio/spoofing/NIF(고래-개미비)/EAI/OI델타/funding/toxicity/queue-collapse/absorption 등 34컬럼 — OFI파생 피처 자원 | 활성 | 1분 단독 알파는 이미 4회 기각됨(`docs/duckdb_live_data_utilization_design_20260719.md` 전제 #1, contrarian flow t=-7.6이나 0.3~2bps<비용4~9bps) — **엔트리 알파가 아니라 게이트/거부/청산타이밍 피처로만 재탐색 가치 있음** |
| `microstructure_1m_btc`/`_sol` | 동일 파일 | 2026-07-14~현재(~1개월) | 동일 스키마 | 활성이나 미성숙 | ETH 대비 이력 짧음, N≥5시드 통계검증엔 아직 부족 |
| `orderbook_decision_snapshots` (ETH) | 동일 파일 | 14,094행+, 2026-05-13~현재(~96일, **raw 레벨은 2026-08-17T01:43:28+09:00~**) | 의사결정 시점 20레벨 L2: best bid/ask, mid, spread_bps, microprice, `bid/ask_notional_{1,5,10,20}`, `imbalance_{1,5,10,20}` **+ `bids_json`/`asks_json`(신규, 이 시각 이후 행만)** | 활성 — **raw 레벨 배선 완료** | 여전히 **의사결정 조건부 샘플링**(WS-A가 지적한 활동구간 편향 그대로 적용, 연속 아님). raw 레벨은 신규 행부터만 — 과거 14,093행은 NULL(백필 불가, `data_epochs.json` 참고) |
| `orderbook_decision_snapshots_btc`/`_sol` | 동일 파일 | 각 4,085행+/4,902행+, 2026-08-02~현재(~15일, raw 레벨은 동일 배포 시점부터) | 동일 스키마 + raw 레벨 | 활성 — raw 레벨 배선 완료 | 이력이 짧아 아직 연구용으로 부족, raw 레벨 축적은 이제부터 시작 |
| `orderbook_periodic_snapshots`(10초 연속) | — | — | WS-E 설계상 신규 테이블(E2) | **인프라 차단(의도적 보류)** | E1과 별도로 새 fetch 루프 필요, 봇 지연 회귀 테스트 선행 조건 미충족 — 이번 세션에서 의도적으로 배선 안 함 |
| `tail_risk_1m` | `data/live/tail_risk.duckdb` | 96,013행+, 2026-05-03~현재 | 청산 스트림(long/short 청산 USD, aftershock_prob 등) | 활성 | LOB축과 직접 관련은 낮음, 참고용 |

## 인프라

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| WS-E 원시 L2 파일럿 로직 | `scripts/research_ws_e_data_flywheel_20260719.py` | 상위20레벨 `bids_json`/`asks_json` 병행 저장 + round-trip 자가검증 | 검증 완료 — 긍정 결과(격리환경), **이후 프로덕션 이관됨(위 절 참고)** | T-E1(round-trip) PASS, T-E2(72h 소크)는 53.08/72h(73.7%)에서 중단 — 100% coverage, gap p99=10.00초, 오류 0건(`data/research/ws_e_soak_status.json`). 이 로직 자체는 격리 파일럿 전용으로 남아있고, 프로덕션은 `orderbook_recorder.py`의 독립 구현을 씀(설계 동일, 코드는 별도) |
| WS-E 격리 연구 DB | `data/research/ws_e_orderbook_raw_pilot.duckdb` | 위 파일럿의 실제 원시 L2 데이터(19,110건, ETH/USDT 선물) | 활성(연구용 격리 DB, 라이브봇과 무관) | 프로덕션 배선 후에도 별도 소규모 데이터셋으로 유효 — 파이프라인 스모크테스트용 |
| `trading_bot_modules/orderbook_recorder.py` | 프로덕션 라이브 경로 | `orderbook_decision_snapshots`(+btc/sol)에 요약 **+ raw 레벨** 기록 | **배선 완료, 라이브 검증됨** | git `d047de1`, 서버 배포 확인(위 절). E2(연속 스냅샷)는 이 파일에 없음 — 별도 작업 필요 |
| 72h 소크 재개 스크립트 | `scripts/run_ws_e_72h_soak_test.py` | 격리 DB에 대한 72h 연속 소크 테스트 러너 | 인프라 확인됨-미착수(중단됨) | 프로덕션 배선과 별개 — 격리 파일럿 자체의 완결성 확인용으로만 의미 있음, 우선순위 낮음(사용자가 이번엔 선택 안 함) |
| 데이터 에폭 경계 문서 | `data_epochs.json`(repo 루트) | 원시레벨 저장 시작 시점 기록 | **완료** | ETH/BTC/SOL 확정 시각 기록됨(위 절) — 향후 연구는 이 경계를 반드시 확인할 것 |

## 외부 다운로더 / API (풀뎁스 이력 데이터가 추가로 필요할 경우)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| Binance `data.binance.vision` bookDepth | `https://data.binance.vision/data/futures/um/daily/bookDepth/{SYMBOL}/{SYMBOL}-bookDepth-{YYYY-MM-DD}.zip` | 무료, 일별 아카이브. 2026-08-15 ETHUSDT 실제 파일 검증: 컬럼 `timestamp,percentage,depth,notional`, 30초 간격, 고정 12개 %밴드(±0.2/1/2/3/4/5%)별 **누적** depth/notional | 과거 유동성 참고(전진수집 불필요) | **검증 완료 — 부정 결과(용도 제한)** | **원시 레벨 호가가 아니라 %밴드 집계다** — 개별 가격레벨/큐 정보 없음, 30초 간격이라 WS-E의 10초 연속 스냅샷보다도 성김. DeepLOB/TLOB류 raw-LOB 모델링엔 쓸 수 없음. 과거(WS-E 배선 이전) 유동성 히스토리가 필요할 때 보조 참고자료 정도로만 가치 있음 |
| Tardis.dev | tardis.dev, `docs.tardis.dev` | 50+ 거래소(Binance/Bybit/OKX/Deribit/Coinbase 등), tick-level L2 incremental + 상위 5/25레벨 스냅샷, CSV/API/Python·Node 클라이언트 | 이 조사에서 찾은 가장 유연하고 품질 좋은 유료 소스 | 미검증 후보 | 월결제 시 최근 4개월치만 접근, 연결제 시 전체 이력 접근(문서 확인). 가격은 3rd-party 비교글 기준 Professional ~$599/월로 추정(**1차 소스 아님, 실제 가격은 Tardis 자체 견적 폼에서 재확인 필요**) |
| Kaiko Research / Amberdata | kaiko.com / amberdata.io | L1/L2 풀뎁스, 100+ 거래소(Kaiko) | 기관용 벤치마크 데이터 | 미검증 후보 | 두 곳 다 공개 가격표 없음("contact sales") — 5절 서베이에서도 확인됨, 이번 스코핑에서도 별도 견적 없이는 비교 불가. 통상 Tardis보다 고가의 기관 티어로 알려져 있으나 1차 확인 안 됨 |

## 미검증 후보 / 보류

- Tardis.dev/Kaiko/Amberdata 실제 견적은 3rd-party 비교 글만 참고했고 자체 견적 요청은
  하지 않았다 — 유료 데이터가 실제로 필요하다고 결론 나기 전에는 보류(현재는 불필요해 보임 —
  프로덕션 raw 레벨 축적이 이제 막 시작됐고, 얼마나 쌓일 때까지 기다릴지가 먼저 결정할 문제).
- WS-E의 T-E2(72h 소크, 격리환경)는 73.7%에서 중단된 채로 남아 있다 — 재실행 여부는 사용자가
  이번 라운드에선 선택하지 않음, 우선순위 낮음.
- E2(10초 연속 스냅샷, 프로덕션)는 별도 봇 지연 회귀 테스트가 선행돼야 한다 — 다음에 이 축을
  다시 열 때 검토.
- raw 레벨이 얼마나 쌓여야("N일") 최소 스모크 모델링을 시작할 수 있을지는 아직 기준을 정하지
  않았다 — 다음 세션에서 결정.
