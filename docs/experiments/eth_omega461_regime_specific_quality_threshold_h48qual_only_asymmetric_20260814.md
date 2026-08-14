# ETH Omega4.6.1 레짐별 quality_threshold — h48qual만 비대칭 채택 (2026-08-14, Odyssey2 #1 후속)

## 배경

`eth_omega461_regime_specific_quality_threshold_20260813.md`(Odyssey2 #1)의 "joint" 맵은
zig075의 bear/chop threshold도 같이 낮춰서(0.75→0.30/0.35) zig075 컴포넌트가 악화됐고
(PnL+40.31%→+31.65%), 그게 기준선 미달의 원인 중 하나였다. **사용자 제안**: exit_head
비대칭 채택(h48qual만 교체, zig075는 원본 유지)과 똑같은 패턴을 여기도 적용 — h48qual만
레짐별 threshold(`bull:0.30, bear:0.30, chop:0.35`)를 쓰고 zig075는 전 레짐 0.75로 완전히
그대로 둔다. 이 조합은 원 실험에서 테스트된 적 없다.

## 방법

재학습 불필요, 원 실험 스크립트의 `evaluate_component`/`portfolio_eval`을 그대로 import해
재사용(`scripts/research_eth_omega461_regime_threshold_h48qual_only_asymmetric_20260814.py`).
G0 자체검증 통과, zig075 컴포넌트가 정확히 자기 baseline과 일치함을 sanity check로 재확인
(PnL 40.31%로 완전 동일 — 정말로 안 건드렸다는 검증).

## 결과 — 근접했으나 여전히 기준선 미달

| | baseline no_gate | 비대칭 no_gate | baseline with_gate | 비대칭 with_gate |
|---|---:|---:|---:|---:|
| PnL | +36.82% | **+67.05%**(개선) | +54.88% | **+42.27%**(악화) |
| MDD | -24.34% | **-17.21%**(개선) | -31.11% | **-17.21%**(개선) |
| 거래수 | 29 | 33 | 22 | 27 |

**no_gate 뷰는 PnL·MDD 둘 다 개선 — 깔끔한 승리.** 하지만 **with_gate 뷰는 MDD는 크게
개선되는데 PnL이 54.88%→42.27%로 하락**해, 사전등록 기준(4개 지표 전부 비악화)의 딱 1개를
못 채운다. **OOS는 열지 않았다.**

원 "joint" 실험(no_gate PnL+72.03%/MDD-28.46%(악화), with_gate PnL+34.76%(악화)/MDD-25.02%)과
비교하면 명백히 개선됐다 — zig075를 건드리지 않은 덕분에 no_gate가 완전히 정리됐고, with_gate도
MDD 문제는 해소됐다. 남은 유일한 결함은 with_gate PnL 하나뿐이다.

## 결론

**여전히 기각.** 근접했다고 해서 사전등록 기준을 사후에 완화하지 않는다(이 프로젝트 표준
규율) — with_gate PnL이 baseline보다 낮다는 사실 자체는 바뀌지 않는다. 다만 "zig075를
건드리지 않는 것"이 원 실험의 핵심 문제(zig075 악화, no_gate/with_gate 뷰 간 정반대 신호)를
대부분 해소했다는 점은 유의미한 진단 정보 — h48qual의 레짐별 threshold 자체는 견고한
방향이고, with_gate 특유의 PnL 하락만 별도 원인이 있을 가능성.

## 미해결 / 다음 단계

- with_gate PnL 하락의 정확한 메커니즘(어떤 거래가 duration-gate에 걸려 빠지는지)은 미조사.
- 채택 가능한 변경 0건, 라이브 파일 미변경.

## 준수 확인

- `git diff` 기준 라이브 파일 무변경. 재학습 없음. 스크립트:
  `scripts/research_eth_omega461_regime_threshold_h48qual_only_asymmetric_20260814.py`. 산출물:
  `tmp/causal_regen_20260516/eth_omega461_regime_threshold_h48qual_only_asymmetric_20260814/val_report.json`.
