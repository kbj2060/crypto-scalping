# BTC CSALT Stage 0/1 and T1 Teacher Smoke Results - 2026-07-15

Status: `stopped_at_T1_teacher_smoke_not_promotion_artifact`

이 실험은 `btc_rl_label_teacher_design_20260715.md`의 구현 순서를 따랐다. Live BTC v1,
Omega bundle, risk sidecar와 runtime config는 변경하지 않았다. 저장 trade ledger와 parent exit
timestamp를 입력으로 사용하지 않았다.

## 결론

CSALT의 미래정보 oracle ceiling은 높았지만, frozen cross-fitted teacher가 그 advantage를 재현하지
못했다. T1 label window에서 최종 N3 gate는 2,168개 event 중 단 한 개만 active로 선택했고 그
거래도 손실이었다. 설계의 중단 규칙에 따라 T1-T6 전체 teacher pack, S1-S4 student selection,
Q4 checkpoint와 Q1 diagnostic은 실행하지 않는다.

## Stage 0: parity and coverage

- 공유 execution/Bellman 단위 테스트: 5/5 통과
- 2024-2025 BTC 5분봉: 연속성/중복/finite 검사 통과
- raw Binance funding: 8시간 event 간격 검사 통과
- T1-T6 decision events: 2,509 / 5,245 / 8,065 / 10,998 / 13,823 / 16,547
- fold별 최소 active-action target coverage: 2,508 이상
- Stage 0 coverage gate: PASS

## Stage 1: oracle ceiling

동일 action/execution에서 continuation `V(next)`만 제거한 N0와 CSALT SMDP oracle을 비교했다.
Oracle은 미래 경로를 아는 ceiling 진단이므로 아래 수치는 예상 live 성과가 아니다.

| fold | N0 MTM MDD | N0 Calmar | CSALT MTM MDD | CSALT Calmar | label change |
|---|---:|---:|---:|---:|---:|
| T1 | -3.11% | 2,831.6 | -2.04% | 13,443.0 | 25.9% |
| T2 | -3.39% | 2,053.5 | -2.95% | 7,271.7 | 23.5% |
| T3 | -4.23% | 1,796.0 | -3.09% | 7,405.1 | 23.5% |
| T4 | -4.01% | 1,381.7 | -3.09% | 6,380.4 | 23.3% |
| T5 | -4.01% | 1,458.6 | -3.09% | 6,620.7 | 23.8% |
| T6 | -4.01% | 1,226.1 | -3.09% | 5,305.4 | 23.4% |

평균 Calmar는 N0 1,791.2, CSALT 7,737.7로 continuation ceiling gate를 통과했다.

같은 label window에 frozen mean-reward HGB causal baseline을 적용했을 때 T1-T6 PnL은 각각
-13.91%, -3.22%, -8.75%, -17.01%, -1.11%, -3.99%였다. 즉 oracle advantage가 존재해도
단순 causal supervised model은 이를 재현하지 못했다.

## T1 q10/q50/q90 teacher smoke

Teacher fit range는 `2024-01-01..2024-03-16`, outcome cutoff는 `2024-03-31`, label window는
`2024-04-16..2024-06-30`이다. 24시간 block bootstrap 5-seed HGB ensemble을 사용했고 label
window realized outcome은 label/gate/weight를 변경하지 않았다.

| family | description | active labels | trades | PnL | MTM MDD | cost 1.5x PnL |
|---|---|---:|---:|---:|---:|---:|
| N0 | immediate lifecycle q10 | 0 | 0 | 0.00% | 0.00% | 0.00% |
| N1 | DP q50 | 873 | 133 | -10.10% | -15.17% | -17.08% |
| N2 | DP q10, no gate | 172 | 32 | -5.35% | -6.80% | -3.88% |
| N3 | DP q10 + vote/cost gate | 1 | 1 | -0.67% | -0.67% | -0.71% |

N3는 `PnL > 0`, `N3 > N0`, `trades >= 20`, `cost 1.5x PnL > 0`를 모두 충족하지 못했다.
T1 teacher smoke gate는 FAIL이다.

## Artifacts

- Stage 0/1 report: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/report.json`
- Causal baseline report: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/causal_baseline/report.json`
- T1 teacher report: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/teacher_T1_smoke/report.json`
- T1 OOF label pack: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/teacher_T1_smoke/T1_oof_rl_label_pack.parquet`
- Label charts: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/**/label_charts/`

이 결과는 research diagnostic이다. Fresh future holdout, Omega promotion 또는 expected live PnL의
근거로 사용할 수 없다.
