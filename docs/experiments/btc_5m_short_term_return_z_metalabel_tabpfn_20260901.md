# BTC 15분 급변(short_term_return_z) TabPFN 메타라벨 확인 (2026-09-01)

## 요약

라운드2(HIT정의×HORIZON×K 재탐색)가 찾은 **MAE상한 / H=6 / K=2.0**을 그대로 라벨로 써서
TabPFN(4시드, `run_tabpfn_panel`/`compute_permutation_importance` 이식) 학습·검증.

| 구간 | AUC (mean±std) | n_train | n_eval |
|---|---|---|---|
| VAL | **0.6919 ± 0.0007** | 2,372 | 452 |
| OOS | **0.6411 ± 0.0013** | 2,372 | 366 |
| HOLDOUT (1회성) | **0.6443 ± 0.0018** | 2,372 | 529 |

ETH 자체 `short_term_return_z` 결과(VAL 0.674/OOS 0.649/HOLDOUT 0.643)와 **거의 동일하거나
근소 우위** — BTC 고유 파라미터(H=6/K=2.0/MAE상한)로 재탐색한 게 실제로 ETH급 성능을 냈다는
뜻입니다. 라운드1의 "H2 강하지만 불안정 vs H6 약하지만 안정" 트레이드오프가 라운드2에서
MAE상한 방식으로 사실상 해소됐고, 이번 TabPFN 결과가 그 판단을 재확인합니다.

## 클러스터 중복제거

같은 방향 fire가 6봉 이내로 몰려있으면 `|ret3_z|` 최댓값 기준 하나만 남김 — 라운드1/2에서는
빠져있던 절차, 이번에 추가.

## 피쳐 중요도 (VAL, 순열중요도)

상위: 변동성/레인지 계열(`atr_percentile_864`/`atr`/`range_width_pct`, 음의상관) +
`vol_z`(양의상관) — 라운드1/2와 동일 패턴 재확인. `bb_pctb`/`ndi`는 비선형(순열중요도만 높고
단순상관은 거의 0).

## 다음 단계

TabPFN까지 완료. 경제성 게이트(트레일링스톱)·실거래 판단은 별도 요청 필요 — 아직 미착수.

## 산출물
- `scripts/research_btc_short_term_return_z_metalabel_tabpfn_20260901.py` (서버 실행)
- `data/labels/btc_5m_evidence_signal_candidates_20260901/short_term_return_z_tabpfn_report.json`
