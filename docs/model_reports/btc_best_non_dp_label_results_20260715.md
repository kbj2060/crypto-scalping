# BTC Best Non-DP Label Result — 2026-07-15

Status: `research_label_pack_complete_not_live_promotion`

## 선택 결과

최종 학습 라벨은 BTC-only 1시간 trend-scanning, forward windows
`[3, 6, 12, 24, 36, 48]h`, `|t-value| >= 2.0`이다.

BTC v2의 기존 7-fold threshold 비교에서 이 설정은 테스트한 네 threshold 중 평균 PnL이 가장
높았다.

| threshold | positive folds | mean PnL | PnL std | mean MDD | worst MDD | trades |
|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 4/7 | **+2.83%** | 5.85% | -7.64% | -11.47% | 148 |
| 2.5 | 3/7 | +1.71% | 7.58% | -10.65% | -18.16% | 150 |
| 3.0 | 3/7 | -0.28% | 8.66% | -9.96% | -18.14% | 139 |
| 3.5 | 5/7 | +2.67% | 4.96% | -8.44% | -14.59% | 131 |

PnL 최우선 기준으로 2.0을 선택했다. 안정성 최우선이면 3.5가 더 적절하지만 이번 요청의 목적과
다르다.

## 새로 시험한 CNUL

DP 대신 exact lifecycle net return의 최선 action을 직접 학습하는 Counterfactual Net-Utility
Label(CNUL) 144개 후보도 시험했다.

개발 최선 후보:

- `derived11__net15_f000__p70__sv40__cg0`
- T1 +5.02%, T2 +2.25%, T3 -0.07%, T4 +5.90%
- aggregate normal-cost +13.11%, aggregate 1.5x-cost +11.58%, 42 trades

T5/T6 one-shot에서는 둘 다 모델이 전부 CASH를 출력해 0거래였다. Holdout을 본 뒤 threshold를
낮추지 않았으며 CNUL은 최종 후보에서 제외했다.

## 라벨 artifact

Output: `tmp/causal_regen_20260516/btc_best_mean_pnl_trendscan_labels_20260715`

- `btc_1h_trendscan_t2_labels_2024.parquet`
- `btc_1h_trendscan_t2_labels_2025.parquet`
- `btc_1h_trendscan_t2_labels_2026.parquet`
- `label_charts/btc_1h_trendscan_t2_labels_{year}.png`
- `manifest.json`

각 row는 `timestamp`, `action_id`, `action`, `trend_t_value`, `trend_beta`,
`trend_horizon_hours`를 포함한다. 2024/2025는 각각 8,784/8,760개의 시간 row이며 2026 artifact는
2026-07-13 00:00까지 4,633개 row다.

## 계약과 제한

- 라벨의 forward trend fit은 offline training target 생성에만 쓴다.
- 현재 시점의 label 자체를 live signal이나 feature로 사용할 수 없다.
- runtime에서는 해당 label로 학습한 frozen model이 completed 1h feature를 보고 예측하고,
  다음 1h open부터 action이 가능하다.
- threshold 2.0은 이미 관찰된 7-fold 비교에서 선택됐으므로 selection peeking이 있다.
- 따라서 결과와 artifact는 research용이며 live 승격 근거가 아니다.
- BTC v1 live path, Omega artifact, risk sidecar는 변경하지 않았다.

## 관련 근거

- López de Prado의 fixed-horizon, triple-barrier, meta-labeling 분류 체계:
  https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml
- BTC walk-forward 연구에서 거래비용 기반 실행 필터가 turnover와 순성과를 크게 좌우한다는 결과:
  https://arxiv.org/abs/2606.00060
- 프로젝트 내 BTC threshold 비교 원문:
  `docs/model_contracts/btc_v2_trendscan_threshold_sweep_20260714.md`

