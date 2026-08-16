# ETH 오더북(LOB) 마이크로구조 DL 후보 — 데이터 계약 (2026-08-17)

이 문서는 **공식 Odyssey 계보(Odyssey1~4)에 속하지 않는다** — 확정된 성과가 있을 때만 번호를
올린다는 원칙에 따라, 아직 데이터소스 스코핑 단계이므로 "Odyssey5"로 명명하지 않는다.

## 상태

| 컴포넌트 | 상태 |
|---|---|
| **데이터소스 스코핑** | **완료(2026-08-17).** 핵심 발견: 원시 L2 저장(WS-E)이 이미 설계·격리검증까지 끝났으나 프로덕션 미배선 상태로 한 달간 방치돼 있었음. 상세는 데이터 리소스 레지스트리 참고. |
| **모델링** | 착수 전. 아래 "다음 단계" 참고 — 사용자 결정 대기 항목이 있어 바로 진행하지 않음. |

## 범위

- 모델 id: `eth_candidate_lob_microstructure_20260817`
- 목적: [DL-for-crypto-trading 전수조사](../deep_learning_for_crypto_trading_literature_survey_20260817.md)
  5절이 지목한, 이 저장소가 유일하게 탐색하지 않은 DL 축(오더북/마켓 마이크로구조)의 실현
  가능성을 데이터 측면에서 먼저 검증한다. TLOB(arXiv:2502.15757)가 비트코인 데이터로 검증된
  드문 사례이고, Wang(2025, arXiv:2506.05764)가 "피처가 깊이를 이긴다"는 이 저장소의 반복된
  결론을 LOB 도메인에서 독립 재현했다는 점이 재탐색 근거다.
- 아키텍처 유형: 미정(데이터 확보 전 단계). 문헌상 후보는 DeepLOB류 CNN+LSTM, TLOB류 이중어텐션
  트랜스포머, 또는 raw LOB 대신 OFI 파생 피처 기반 경량 모델(Kolm/Turiel/Westray 패턴).
- Owner agent: Model Architect(단독, Sonnet).
- 리소스 레지스트리: [`eth_candidate_lob_microstructure_data_resources_20260817.md`](eth_candidate_lob_microstructure_data_resources_20260817.md)
- 관련 문서: [`docs/deep_learning_for_crypto_trading_literature_survey_20260817.md`](../deep_learning_for_crypto_trading_literature_survey_20260817.md) 5절/7절,
  `docs/duckdb_live_data_utilization_design_20260719.md`(기존 라이브 데이터 인벤토리 원본),
  `docs/test_designs_duckdb_live_20260719/ws_e_data_flywheel.md`(원시 L2 저장 설계 원본)

## 데이터소스 스코핑 결론

1. **원시 L2 레벨 저장은 이미 만들어져 있다.** 2026-07-19 WS-E 설계·파일럿에서 round-trip
   자가검증 통과, 격리 연구 DB에서 53.08/72시간 소크(coverage 100%, 오류 0) — 여기서 멈췄다.
   프로덕션(`orderbook_recorder.py`)에는 한 번도 배선되지 않았고, 2026-08-17 서버 직접 조회로
   재확인했다(`bids_json`/`asks_json` 컬럼 없음, `orderbook_periodic_snapshots` 테이블 없음).
2. **신규 수집 없이 바로 쓸 수 있는 자원이 두 개 있다**: (a) ETH 3.5개월치 연속 1분봉
   오더플로우 파생 피처(`microstructure_1m`, 34컬럼 — OFI 파생 피처 패턴, 문헌 5.2절과 정합),
   (b) ETH 96일치 의사결정-조건부 20레벨 L2 요약(`orderbook_decision_snapshots` — 표본이 성기고
   샘플링 편향 있음, 원시 레벨은 없음).
3. 전체 상세는 데이터 리소스 레지스트리 참고.

## 다음 단계 (사용자 결정 필요 — 아직 진행 안 함)

- **A. WS-E 72h 소크 재개**: 격리 연구 DB에서 나머지 ~19시간만 더 채우면 파일럿 자체의 수락
  기준이 완결된다. 라이브 봇과 무관, 낮은 리스크. 재개 여부만 확인되면 바로 실행 가능.
- **B. 프로덕션 배선(`orderbook_recorder.py`에 원시 레벨 컬럼 추가)**: 봇 재시작을 수반하는
  운영 변경이라 **사용자의 명시적 승인 전에는 진행하지 않는다.** 승인 시 이후에만 원시 LOB가
  전진 축적되기 시작한다(백필 불가 — 데이터 에폭 경계 문서화 필수).
- **C. Binance `data.binance.vision` bookDepth 무료 아카이브 스키마 검증**: 파일 1개를 받아
  실제 레벨수/집계방식/주기를 확인 — 저비용, 사용자 승인 불필요한 순수 조사 작업.
- B를 승인받기 전까지는, 이미 쌓여 있는 (a) `microstructure_1m` 오더플로우 피처와 (b) WS-E
  격리 파일럿의 19,110건 원시 스냅샷(53시간)만으로 소규모 스모크 실험을 먼저 해볼 수 있다 —
  다만 (b)는 표본이 작아 방향성 결론을 낼 수준은 아니고 파이프라인 검증용에 가깝다.

## Open Issues

- 데이터 에폭 경계 문서(`data_epochs.json`)가 아직 없음 — B 승인 전에 먼저 만들어야 함.
- WS-E 72h 소크가 왜 53.08h에서 멈췄는지(의도적 중단 vs 세션 종료) 미확인.
- Tardis.dev 등 유료 소스는 A/B/C 결과를 본 뒤, 정말 더 긴 이력이 필요하다고 판단될 때만
  검토 — 현재는 보류.
