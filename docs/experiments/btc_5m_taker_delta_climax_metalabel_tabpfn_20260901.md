# BTC 체결 쏠림(taker_delta_z_climax) TabPFN 메타라벨 확인 (2026-09-01)

## 요약

라운드2가 찾은 **종가기준(close_at_h) / H=6 / K=2.0** 라벨로 TabPFN(4시드) 학습·검증.
클러스터 중복제거는 ETH `taker_delta_z_climax` v4의 `cluster_dedup()`(같은방향 3봉이내 묶어
`delta_z` 최극값만 유지)을 그대로 이식.

| 구간 | AUC (mean±std) | n_train | n_eval |
|---|---|---|---|
| VAL | **0.6273 ± 0.0009** | 6,513 | 1,244 |
| OOS | **0.6486 ± 0.0015** | 6,513 | 900 |
| HOLDOUT (1회성) | **0.6276 ± 0.0014** | 6,513 | 1,349 |

ETH 자체 결과(VAL 0.622/OOS 0.608/HOLDOUT 0.650)와 **비교 가능한 수준, OOS는 오히려 BTC가
더 높음**. 라운드2에서 발견한 "BTC는 ETH와 정반대로 짧은 지평(H=6)이 유리하다"는 재탐색이
TabPFN으로도 확인됨 — ETH의 H=24를 그대로 썼다면 이 성능이 안 나왔을 가능성이 높습니다.

## 피쳐 중요도 (VAL, 순열중요도)

변동성/레인지 계열(`atr_percentile_864`/`atr`/`range_width_pct`) 상위, 트리거 자신의 강도
(`delta_z`)는 발동 후엔 낮은 중요도 — 라운드1/2와 일관.

## 다음 단계

TabPFN까지 완료. 경제성 게이트·실거래 판단은 미착수.

## 산출물
- `scripts/research_btc_taker_delta_climax_metalabel_tabpfn_20260901.py` (서버 실행)
- `data/labels/btc_5m_evidence_signal_candidates_20260901/taker_delta_climax_tabpfn_report.json`
