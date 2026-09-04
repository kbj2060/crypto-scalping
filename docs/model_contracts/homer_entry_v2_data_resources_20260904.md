# Homer Entry v2 — 데이터 및 리소스 관리 (2026-09-04)

이 문서는 Homer Entry v2 서브 프로젝트(`docs/model_contracts/homer_entry_v2_contract.md`)에서 실제로 만지거나 검토한 모든 데이터 소스/리소스를 한 곳에 모은 목록이다. 계약 문서는 모델/아키텍처 상태만 다루고, 리소스의 위치·커버리지·상태·함정은 여기서 관리한다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것.** 상태 값: `활성`, `인프라 확인됨-미착수`, `인프라 차단`, `검증 완료 — 긍정 결과`, `검증 완료 — 부정 결과`.

## 라벨/예측 데이터

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| ETH 5m klines | `binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv` | 로컬 ~2026-08-28, 서버 ~08-31 | 프레임·라벨 | 활성 | 학습창(≤03-31)엔 충분. HOLDOUT(≥04-01)은 로드하지 않음 |
| ETH 1m klines | `binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv` | 2023-12-31 ~ 2026-07-31 | L2 1분 재구성 | 활성 | 07-31 이후 없음(학습창엔 무관) |
| BTC 5m klines | `binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv` | 로컬 ~2026-08-20 | smt 발동·BTC 레짐 | 활성 | |
| 레짐 OOF (ETH S12_K3 / BTC S24_K3) | `tmp/eth_entry_oof_regime_20260903/regime_oof_{eth,btc}.parquet` | 2024-01 ~ 2026-08-19 | F1/F3 피쳐 | 활성 | 0=bull 1=bear 2=chop, −1=워밍업. **서버에 없음 → launch 시 --sync 필수** |
| 증거신호 메타라벨 OOF (8종) | `tmp/eth_entry_oof_metalabel_20260903/<sig>_oof.csv` | 2024-01 ~ HOLDOUT | F2/F3 재료 | 활성 | TRAIN 워밍업 행은 NaN(제외). 서버에도 존재 |
| v2 산출물 (프레임·게이트·리포트) | 서버 `tmp/homer_entry_v2_20260904/` (JSON은 로컬에도 pull) | TRAIN/VAL/OOS | 재현·감사 | 검증 완료 — 부정 결과(팔 증분 없음) | `frame.parquet`/`fills.csv`는 서버에만, `report_{hgb,tabpfn}.json`·`layer_gates_*.json` 로컬 보유 |
| 인과 발동 모집단 (8종, H/K) | `tmp/eth_causal_population_metalabel_20260902/` | 전 구간 | 재료 horizon 설정 | 활성 | raw 트리거(dedup 없음) |
| 경제라벨 동결 컨텍스트 (F0 라이브 모델) | `data/labels/eth_5m_v_rebound_econ_label_20260902/tabpfn_train_context_frozen_econ_5seed_20260902.csv` | 2024-01-05 ~ 2025-08-31, 5시드×18k | F0 라이브 재현 참조 | 활성 | 이번 v2 F0는 TRAIN 시작 2024-05라 완전 동일 아님 |

## 라이브 수집 데이터 (duckdb 등)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| V자반등 경제라벨 섀도우 원장 | 서버 `data/live/v_rebound_econ_shadow_state.json` | 2026-09-02~ (재기동 09-04) | F0 전진 성과 | 활성 | 09-03 이전 원장은 마크가격 폴링(낙관) — `exit_basis` 구분 |

## 외부 다운로더 / API

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| (없음 — 학습은 저장된 klines만) | | | | | |

## 인프라

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| TabPFN 8.5.0 + CUDA | 서버 quant_ai | 5시드 앙상블 학습/채점 | 활성 | GPU 8GB를 대시보드 TabPFN·경제라벨 섀도우와 공유 — 청크 10k, 시드 순차 |
| 층 게이트 러너 | `scripts/gate_eth_entry_layers_20260903.py` | L4/L1/L2/L2P/L3/T1/T2 | 활성 | L2P용 어댑터 `eth_v_rebound_econ_shadow` 09-04 추가 |
| 증거신호 발동 어댑터 | `scripts/gate_eth_entry_triggers_v1_adapter_20260903.py` | L1 | 활성 | BTC klines 필요(smt) |

## 미검증 후보 / 보류

- 레짐 하드 게이트 팔(예측 chop에서만 진입): 신호별로 정반대 효과였던 선례로 이번엔 피쳐로만 투입. 필요 시 별도 사전등록.
- 재료의 `_proba_cal`(sweep/smt 권장) 대신 `_pct`만 사용 — OOF본에 cal이 없음.
