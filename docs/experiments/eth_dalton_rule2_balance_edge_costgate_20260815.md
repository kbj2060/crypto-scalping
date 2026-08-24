# Dalton Rule 2(balance-edge-reject) 신호 — 비용 게이트 검증 (2026-08-15)

상태: **완료. 비용 게이트 이전 단계에서 이미 탈락(0/6 창) — 손익분기 비용이 사실상 0bp.**

## 요청과 배경

[[eth_amt_vsa_footprint_ifvg_strategy_absorption_study_20260815]]에서 유일하게 살아남은 후보
(저변동성 레짐 게이트 + 48봉 레인지 가장자리, lift 1.6~1.9배, VAL/OOS 안정)를 실제 causal
TP/SL 백테스트로 검증해달라는 요청. `evidence_signal_quant_use` 계약의 승격 게이트 규칙
("손익분기 비용을 bp로 명시하지 않은 후보는 심사 대상이 아니다")을 이 신호에 적용한다.

## 방법 — 새로 만들지 않고 기존 관행 그대로 재사용

같은 계열의 기존 causal 백테스트(`backtest_eth_evidence_signal_top6_confluence_20260814.py`)와
**완전히 동일한 엔진·상수**를 사용했다. 이 질문을 위해 새로 튜닝한 값은 없다.

- 엔진: `core.causal_futures_backtest.simulate_single_position`/`purged_decision_mask`(entry =
  신호 확정 bar+1 시가, TP/SL bar-by-bar 순방향 확인, 비중복 포지션).
- 상수(그대로 복사): TP=1.6×ATR, SL=1.0×ATR, 최대보유 48봉, 레버리지 3x, 증거금 30%, 왕복비용
  0.1%(기본값), ATR(14).
- 신호: `analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815.add_amt_features`를
  **그대로 import**(재정의 없음) — 저변동성 레짐(48봉 ATR% 롤링 백분위수 ≤30) AND 48봉 레인지
  가장자리(허용오차 = 레인지 폭의 15%) 터치.
- 창: 기존 6개 사전등록 창(`eth_omega461_multiwindow_confirmation_gate_20260814.WINDOW_DEFS`) —
  2025q1/q2/q3(컨텍스트) + val + oos_q1/oos_q2. 새 창 정의 안 함.
- 비용 게이트: 왕복비용 0~200bp를 이분탐색해 수익이 0을 교차하는 지점(손익분기 bp)을 찾는다.

## 결과 — 0/6 창, 손익분기 비용이 사실상 0

| 창 | 거래수 | 승률 | 수익률(10bp 비용) | always_long | always_short | 벤치마크 승 | 손익분기 |
|---|---:|---:|---:|---:|---:|---|---:|
| 2025q1 | 1,145 | 49.0% | -66.27% | -45.45% | +83.32% | ❌ | 0.0bp |
| 2025q2 | 1,058 | 52.7% | -62.63% | +36.22% | -26.59% | ❌ | 0.0bp |
| 2025q3 | 1,152 | 54.1% | -63.11% | +66.63% | -39.99% | ❌ | 0.4bp |
| val | 1,084 | 51.8% | -60.15% | -28.47% | +39.81% | ❌ | 0.6bp |
| oos_q1 | 908 | 48.5% | -56.83% | -31.90% | +46.84% | ❌ | 0.0bp |
| oos_q2 | 955 | 49.3% | -51.23% | -23.34% | +30.45% | ❌ | 1.6bp |

**6/6 창 전부 벤치마크(always_long/always_short) 대비 패배. 손익분기 비용은 0.0~1.6bp** — 왕복
10bp 가정은커녕 **비용을 아예 0으로 둬도 전략이 이긴다.** 즉 이 신호는 비용 문제가 아니라
**방향성 자체의 문제**로 탈락한다.

## 진단 — retrospective lift(1.6~1.9배)가 왜 실제 TP/SL에서 사라지는가

VAL 창의 체결 내역을 직접 열어봤다.

- TP 체결 422건 vs SL 체결 662건 (전체 1,084건). TP 도달률 **422/1084 ≈ 38.9%**.
- TP:SL = 1.6:1의 손익분기 승률은 **≈38.5%**. 즉 정확히 손익분기선 위에 걸쳐 있고, 여기에
  왕복비용을 얹으면 바로 마이너스로 넘어간다.
- 건당 평균 수익: TP 체결 시 +0.18%, SL 체결 시 -0.25%. 1,084건 복리: 이론상
  `(1-0.000846)^1084 ≈ 0.40` → -60% — 실측 -60.15%와 정확히 일치.

원인은 이전 발견 문서(`eth_amt_vsa_footprint_ifvg_strategy_absorption_study_20260815.md`)의 lift
측정 자체에 이미 있었다. 이 신호는 **`median_lead_bars` 4~5봉**, `excess_move_mean_pct`
**-0.44%(바닥)/+0.32%(천장)** — 즉 신호 발화 후 진짜 피벗까지 4~5봉이 더 걸리고, 그 사이 가격이
평균 0.3~0.4%p **더 불리하게** 움직인다. 이 저장소가 TP/SL 백테스트에서 반복 확인해 온 패턴과
동일하다 — "이 신호들은 확률 이동(probability-shift) 맥락이지 진입 트리거가 아니다"
(`eth_evidence_signal_top6_confluence_standalone_backtest_20260814.md`). 고정 1.6×ATR TP는
그 4~5봉의 역행을 못 버티고 대부분 1.0×ATR SL에 먼저 걸린다.

## 결론

**Dalton Rule 2(balance-edge-reject) 신호는 비용 게이트를 통과하지 못한다.** 손익분기가
0~1.6bp라 어떤 현실적 비용 가정에서도 실패하며, 원인은 비용이 아니라 TP/SL 구조와 신호의 실제
리드타임 불일치다. `evidence_signal_quant_use` 승격 게이트("공짜 벤치마크 대비 증분")를 적용할
필요도 없이 이 단계에서 탈락 — always_long/always_short 자체를 못 이긴다.

이 신호는 **폐기한다.** 향후 재검토 시 새 TP/SL을 신호의 실제 리드타임(4~5봉)·역행폭(0.3~0.4%)에
맞춰 재보정하는 방법이 남아 있으나, 그 자체가 이 백테스트를 위한 사후 튜닝이 되므로 별도로
사전등록된 실험으로 취급해야 하며 이 문서의 범위 밖이다.

## 준수 확인

`fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
`saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`. 신규 학습 없음,
GPU 불필요. 라이브 파일 미변경.

## 산출물

- `scripts/backtest_eth_dalton_rule2_balance_edge_costgate_20260815.py`
- `tmp/causal_regen_20260516/eth_dalton_rule2_balance_edge_costgate_20260815/report.json`
