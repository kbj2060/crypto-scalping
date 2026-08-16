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

1. **원시 L2 레벨 저장(raw `bids_json`/`asks_json`)은 이미 설계·구현·격리환경 검증까지 끝났지만
   프로덕션에는 배선된 적이 없다** (`docs/test_designs_duckdb_live_20260719/ws_e_data_flywheel.md`,
   2026-07-19 설계, 같은 날 파일럿 실행). Round-trip 자가검증 통과(불일치 0/36), 격리 연구
   DB(`data/research/ws_e_orderbook_raw_pilot.duckdb`)에서 72시간 목표 중 53.08시간
   소크(coverage 100%, gap p99=10.00초, 오류 0건, round-trip 재검증 0건 불일치)까지 진행된
   상태에서 세션이 끝났다. 설계 문서 자체가 "프로덕션 배선은 봇 재시작을 수반하므로 사용자
   승인 후 별도 세션에서 진행"이라고 명시했고, **한 달이 지난 지금(2026-08-17)도 서버 라이브
   DB를 직접 조회해 미배선 상태를 재확인했다** (아래 표).
2. **이미 프로덕션에서 3개월(ETH)/2주(BTC/SOL) 동안 조건부 L2 요약이 쌓이고 있다** — 원시
   레벨은 아니지만 20레벨 깊이의 depth/imbalance/microprice 요약이다. 다만 의사결정
   시점에서만 기록되는 조건부 샘플링이라 WS-A 설계 문서가 스스로 지적한 편향(활동 구간에
   쏠림)을 안고 있다.
3. **이미 프로덕션에서 3.5개월(ETH) 연속 1분봉 오더플로우 파생 피처**(OBI, taker_buy_ratio,
   NIF, EAI, toxicity, queue-collapse, absorption 등 34컬럼)가 쌓이고 있다 — 이건 문헌
   5.2절의 "raw LOB보다 OFI 파생 피처가 낫다"(Kolm/Turiel/Westray) 및 "피처가 깊이를 이긴다"
   (Wang 2025)는 결론과 정확히 같은 종류의 자원이며, 신규 수집 없이 바로 연구 가능하다.

## 라이브 수집 데이터 (duckdb)

| 리소스 | 위치 | 커버리지 (2026-08-17 서버 실측) | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| `microstructure_1m` (ETH) | `data/live/microstructure.duckdb` | 136,041행, 2026-05-03~현재(연속 1분봉) | OBI/taker_buy_ratio/spoofing/NIF(고래-개미비)/EAI/OI델타/funding/toxicity/queue-collapse/absorption 등 34컬럼 — OFI파생 피처 자원 | 활성 | 1분 단독 알파는 이미 4회 기각됨(`docs/duckdb_live_data_utilization_design_20260719.md` 전제 #1, contrarian flow t=-7.6이나 0.3~2bps<비용4~9bps) — **엔트리 알파가 아니라 게이트/거부/청산타이밍 피처로만 재탐색 가치 있음** |
| `microstructure_1m_btc`/`_sol` | 동일 파일 | 2026-07-14~현재(~1개월) | 동일 스키마 | 활성이나 미성숙 | ETH 대비 이력 짧음, N≥5시드 통계검증엔 아직 부족 |
| `orderbook_decision_snapshots` (ETH) | 동일 파일 | 14,090행, 2026-05-13~2026-08-17 01:20 KST(~96일) | 의사결정 시점 20레벨 L2 요약: best bid/ask, mid, spread_bps, microprice, `bid/ask_notional_{1,5,10,20}`, `imbalance_{1,5,10,20}` | 활성 | **원시 레벨 배열 없음**(`bids_json`/`asks_json` 컬럼 미존재, 서버 실측 확인). **의사결정 조건부 샘플링** — WS-A가 스스로 지적한 편향(활동구간 쏠림) 그대로 적용됨. 96일간 14,090행 = 평균 ~10분에 1행꼴로 매우 성김 |
| `orderbook_decision_snapshots_btc`/`_sol` | 동일 파일 | 각 4,081행/4,898행, 2026-08-02~현재(~15일) | 동일 스키마 | 활성이나 미성숙 | 이력이 짧아 아직 연구용으로 부족 |
| `orderbook_periodic_snapshots`(10초 연속) | — | — | WS-E 설계상 신규 테이블(E2) | **인프라 차단** | 서버 실측 확인 결과 테이블 자체가 존재하지 않음 — 설계만 있고 프로덕션 미배선 |
| `tail_risk_1m` | `data/live/tail_risk.duckdb` | 96,013행+, 2026-05-03~현재 | 청산 스트림(long/short 청산 USD, aftershock_prob 등) | 활성 | LOB축과 직접 관련은 낮음, 참고용 |

## 인프라

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| WS-E 원시 L2 파일럿 로직 | `scripts/research_ws_e_data_flywheel_20260719.py` | 상위20레벨 `bids_json`/`asks_json` 병행 저장 + round-trip 자가검증 | **검증 완료 — 긍정 결과(격리환경)** | T-E1(round-trip) PASS, T-E2(72h 소크)는 53.08/72h(73.7%)에서 중단 — 100% coverage, gap p99=10.00초, 오류 0건, round-trip 180건 재검증 0건 불일치(`data/research/ws_e_soak_status.json`). **프로덕션 미배선** — 다음 행 참고 |
| WS-E 격리 연구 DB | `data/research/ws_e_orderbook_raw_pilot.duckdb` | 위 파일럿의 실제 원시 L2 데이터(19,110건, ETH/USDT 선물) | 활성(연구용 격리 DB, 라이브봇과 무관) | 이 파일 자체가 "10초 간격 원시 20레벨 LOB, 53시간, ETH" 데이터셋으로 **즉시 사용 가능** — 소량이지만 DeepLOB류 아키텍처의 스모크테스트/파이프라인 검증엔 충분 |
| `trading_bot_modules/orderbook_recorder.py` | 프로덕션 라이브 경로 | 현재 `orderbook_decision_snapshots` 요약만 기록 | 인프라 확인됨-미착수 | WS-E 설계가 요구하는 변경(신규 컬럼만 추가, 기존 스키마 불변, `serialized_duckdb_access` 경유, 지연 추가 금지)이 코드에 반영되지 않음 — grep 결과 `bids_json`/`asks_json` 문자열이 파일럿 스크립트 2개 외 프로덕션 코드 어디에도 없음 |
| 72h 소크 재개 스크립트 | `scripts/run_ws_e_72h_soak_test.py` | 격리 DB에 대한 72h 연속 소크 테스트 러너 | 인프라 확인됨-미착수(중단됨) | 53.08/72h에서 중단된 이유 미확인(dev 세션 종료 추정) — 재실행하면 나머지 ~19h만 더 채우면 T-E2 격리환경 수락기준 완전 충족 |
| 데이터 에폭 경계 문서 | `docs/feature_contract_manifest.json` 또는 `data_epochs.json` | 원시레벨 저장 시작 시점 기록 | **인프라 차단(미생성)** | WS-E 설계가 필수로 요구했으나 아직 존재하지 않음 — 프로덕션 배선 전 반드시 먼저 만들어야 함(이 저장소의 "파생 아티팩트 동시 연장" 함정과 동일 계열) |

## 외부 다운로더 / API (풀뎁스 이력 데이터가 추가로 필요할 경우)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| Binance `data.binance.vision` bookDepth | `https://data.binance.vision/?prefix=data/futures/um/daily/bookDepth/` | 무료, 일별 아카이브, USD-M/COIN-M 선물 심볼별 | 과거 이력 확보(전진수집 불필요) | 미검증 후보 | **무료지만 정확한 스냅샷 주기·집계방식(레벨별 원시 호가인지 구간 집계인지) 미확인** — 실제 파일 1개를 받아 스키마 검증 필요. Tardis 문서상 Binance는 tick-level 소스로도 별도 지원되므로 무료판과 유료판의 해상도 차이도 확인 대상 |
| Tardis.dev | tardis.dev, `docs.tardis.dev` | 50+ 거래소(Binance/Bybit/OKX/Deribit/Coinbase 등), tick-level L2 incremental + 상위 5/25레벨 스냅샷, CSV/API/Python·Node 클라이언트 | 이 조사에서 찾은 가장 유연하고 품질 좋은 유료 소스 | 미검증 후보 | 월결제 시 최근 4개월치만 접근, 연결제 시 전체 이력 접근(문서 확인). 가격은 3rd-party 비교글 기준 Professional ~$599/월로 추정(**1차 소스 아님, 실제 가격은 Tardis 자체 견적 폼에서 재확인 필요**) |
| Kaiko Research / Amberdata | kaiko.com / amberdata.io | L1/L2 풀뎁스, 100+ 거래소(Kaiko) | 기관용 벤치마크 데이터 | 미검증 후보 | 두 곳 다 공개 가격표 없음("contact sales") — 5절 서베이에서도 확인됨, 이번 스코핑에서도 별도 견적 없이는 비교 불가. 통상 Tardis보다 고가의 기관 티어로 알려져 있으나 1차 확인 안 됨 |

## 미검증 후보 / 보류

- Binance bookDepth 무료 아카이브의 실제 스키마(레벨 수, 집계 방식, 스냅샷 주기)를 파일 1개
  다운로드해서 직접 확인하지 않았다 — 다음 스텝의 최우선 항목.
- Tardis.dev/Kaiko/Amberdata 실제 견적은 3rd-party 비교 글만 참고했고 자체 견적 요청은
  하지 않았다 — 유료 데이터가 실제로 필요하다고 결론 나기 전에는 보류.
- WS-E의 T-E2(72h 소크, 격리환경)는 73.7%에서 중단된 채로 남아 있다 — 재실행 여부는 사용자
  판단 대기(가벼운 작업, 반나절 미만).
- 프로덕션 배선(`orderbook_recorder.py` 수정 + 봇 재시작)은 라이브 트레이딩 시스템에 대한
  운영 변경이므로 **사용자의 명시적 승인 없이는 진행하지 않는다.**
