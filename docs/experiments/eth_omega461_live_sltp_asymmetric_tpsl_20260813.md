# ETH Omega4.6.1 라이브 SLTP — TP/SL 비대칭 분리(SL 고정) 후속실험 (2026-08-13)

이 문서는 `docs/experiments/eth_omega461_live_sltp_mfe_width_20260813.md`(이하 "선행실험")의
직접적인 후속이다. 선행실험은 방향별 MFE 분위수 회귀 예측치에 TP와 SL을 **같은 비율로 묶어서**
비례시켰을 때, 목표 (a)(평균 보유기간 단축+거래수 증가)는 강하게 달성됐지만 목표 (b)(PnL/MDD
비악화)가 전 구간에서 실패했다(승률 41~48%→27~37% 붕괴)고 결론지었다. 오케스트레이터가 제시한
후속 가설: "SL이 TP와 같이 좁아진 게 주범이라면, TP만 좁히고 SL은 원래 라이브 고정폭 그대로 두면
승리거래는 빨라지되 손실거래 리스크는 안 늘어날 것"이라는 가설을 검증한다.

## 방법

- **TP**: 선행실험과 완전히 동일한 메커니즘 — base102(102개 `base_cols`) 피쳐 패널로 학습한 방향별
  MFE 분위수회귀(q=0.5) 예측치 × `tp_scale`, `{1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0}` 동일 그리드.
- **SL**: MFE 예측과 완전히 분리. `prep_component`가 이미 계산해둔 **원본 라이브 ATR-floor
  `stop_loss`**(교체 전 `dec["stop_loss"]`, 사실상 고정 4.0%)에 독립적인 `sl_scale ∈ {1.0, 1.5,
  2.0}`만 곱한다. `sl_scale=1.0`은 SL을 정말 하나도 안 건드린 baseline 그대로다.
  `take_profit = clip(max(FLOOR_TP, width*tp_scale), 0, max_tp)`,
  `stop_loss = clip(baseline_stop_loss * sl_scale, 0, max_sl)` — `max_tp=0.22`/`max_sl=0.12`(라이브
  캡) 불변.
- **피쳐셋**: base102만 사용(final10_latent16은 선행실험에서 이미 확정적으로 열세로 판정돼 반복
  안 함 — 오케스트레이터 지시).
- **격리**: margin/leverage는 여전히 원본 ATR 기반 `dec`에서 먼저 계산(선행실험과 동일 원칙), TP/SL
  교체는 그 계산이 끝난 복사본에서만 수행.
- **그리드**: `tp_scale`(7) × `sl_scale`(3) = 21셀, 각 셀을 컴포넌트 단독(`replay_exit_variant`)과
  우선순위결합 포트폴리오(`greedy_replay`, h48qual>zig075 단일계정) 둘 다에서 확인. 판정 기준: "hold
  개선"=평균보유기간이 baseline의 70% 미만 **AND** 거래수가 baseline의 1.2배 초과, "pnl/mdd 비악화"=
  no_gate·with_gate **양쪽 다** pnl≥baseline pnl **AND** mdd≥baseline mdd. 두 조건 동시 충족 셀을
  탐색.
- 하네스 재사용/제약은 선행실험과 완전 동일: `research_eth_omega461_exit_sweep_20260721.py` +
  `replay_omega4_6_1_greedy_router_20260706.py`, VAL=2025-10-01~12-31, **OOS 미실행**,
  `fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
  saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false`.

## 결과

MFE 모델 자체(TRAIN/VAL R²·spearman)는 선행실험의 base102 결과와 정확히 동일 재현(같은 시드/같은
학습 코드 재사용, 확인용 재학습). 우선순위결합 포트폴리오(VAL, no_gate) 2차원 그리드 전체:

| tp_scale | sl_scale | pnl% | mdd% | trades | avg_hold | wr | hold개선 | pnl/mdd비악화 |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| baseline | - | +36.82 | -24.34 | 29 | 676.5 | 41.4% | - | - |
| 1.0 | 1.0 | +0.73 | **-17.72** | 96 | 174.5 | **79.2%** | O | X |
| 1.5 | 1.0 | +3.39 | **-17.72** | 79 | 203.4 | - | O | X |
| 2.0 | 1.0 | **+5.16** | **-17.72** | 72 | 251.7 | 70.8% | O | X |
| 3.0 | 1.0 | -25.52 | -40.79 | 56 | 313.2 | - | O | X |
| 4.0 | 1.0 | -21.54 | -41.89 | 53 | 365.1 | - | O | X |
| 6.0 | 1.0 | +30.86 | -27.85 | 36 | 538.1 | - | X | X |
| 9.0 | 1.0 | +39.44 | -21.75 | 27 | 634.1 | - | X | X |
| 1.0 | 1.5 | -7.25 | -26.91 | 90 | 194.6 | - | O | X |
| 2.0 | 1.5 | -10.49 | -27.38 | 67 | 274.5 | 76.1% | O | X |
| 9.0 | 1.5 | **+123.68** | **-15.51** | 24 | 737.5 | 62.5% | X | O |
| 1.0 | 2.0 | -9.76 | -27.84 | 83 | 227.4 | - | O | X |
| 2.0 | 2.0 | +5.10 | -22.39 | 52 | 396.3 | 82.7% | O | X |
| 4.0 | 2.0 | +26.05 | -36.11 | 36 | 604.1 | - | X | X |

(전체 21행: `tmp/research_20260813/omega461_live_sltp_asymmetric_tpsl/grid_summary_VAL.csv`, 컴포넌트
단독 28행: `component_variants_VAL.csv`, 원장: `priority_combined_ledger_*_VAL.csv`.)

**두 조건을 동시에 만족하는 셀: 0/21.**

## 해석

**가설은 정확히 확인됐다.** SL을 baseline 그대로 두자(`sl_scale=1.0`) 승률이 41.4%(baseline)는
물론 선행실험의 결합형 scale=1.0(승률 31.2%)보다도 훨씬 높은 **79.2%**로 치솟는다(tp2_sl1은
70.8%, 대부분의 셀이 70~83%). SL을 건드리지 않으면 손실거래의 위험이 안 늘어난다는 가설 그대로,
승리거래만 훨씬 빨리(TP가 좁아서) 확정되니 승률이 급등한 것 — 원인 진단이 정확했다.

**하지만 승률 급등이 곧바로 PnL 개선으로 이어지지 않는다.** TP가 좁아지고(예측 MFE 중앙값 기준
~0.7~1.5%) SL은 그대로 넓으니(~4%), R:R이 1:3~1:6 수준으로 극도로 불리해진다 — 이길 때는 조금만
벌고 질 때는 훨씬 많이 잃는 구조라, 승률이 70~83%로 매우 높아도 기댓값은 간신히 양(+)이거나
여전히 baseline보다 낮다(tp1_sl1 pnl +0.73%, tp2_sl1 +5.16% vs baseline +36.82%). **MDD는 오히려
baseline보다 뚜렷이 개선된다**(tp1_sl1/tp2_sl1 모두 -17.72% vs baseline -24.34%, with_gate도
-18~21% vs -31.11%) — SL을 안 건드렸으니 개별 손실폭이 baseline과 같은데도 승률이 높아 드로다운
누적이 줄어든 것으로 보인다. 즉 **가장 근접한 셀(tp2.0_sl1.0)은 MDD 개선+보유기간 63%
단축+거래수 148%증가를 모두 달성하면서도, 유독 절대 PnL만 이례적으로 강한 baseline(+36.82%/
+54.88%)에 못 미쳐 최종 판정에서 탈락**한다.

**scale를 더 키우면(tp9_sl1.5 등) PnL이 baseline을 크게 능가하기도 하지만(+123.68%), 그 지점은
평균보유기간(737.5바)이 이미 baseline(676.5바)보다 길어져 애초에 풀려던 문제(보유기간 단축)를
전혀 개선하지 못한 상태**다 — 선행실험에서도 나온 "넓히면 PnL은 좋아지지만 문제 해결 자체가
무의미해진다"는 동일 패턴의 재현.

## 결론

**실패 (정직한 보고), 그러나 진단 자체는 유의미하게 정제됐다.** 오케스트레이터의 가설(SL 동시축소가
승률붕괴의 주범)은 승률 지표로 명확히 입증됐다 — TP만 좁히고 SL을 baseline 그대로 두면 승률이
27~37%(선행실험 결합형)에서 70~83%로 급등한다. 하지만 이 실험이 새로 드러낸 문제는 **R:R
역전**이다: SL을 안 건드리면 승률은 살지만 "이길 때 조금, 질 때 많이"라는 구조가 그대로 남아
절대 PnL이 여전히 baseline을 못 넘는다. tp_scale × sl_scale 2차원 그리드 21칸 전부에서 "보유기간이
뚜렷이 짧으면서 PnL/MDD가 baseline 이상"인 지점은 없었다(가장 근접한 tp2.0_sl1.0도 MDD는
이겼지만 PnL이 못 미침).

오케스트레이터 사전 지시에 따라: **이것도 실패이므로 "SLTP 폭 축소로 인한 거래수 감소" 문제는
상수 플로어 재조정(2026-07-28)과 MFE기반 폭 학습(대칭·비대칭 둘 다, 2026-08-13) 두 메커니즘으로는
해결되지 않는 구조적 트레이드오프로 최종 결론짓는다.** 세 번째 메커니즘 탐색은 여기서 멈춘다.

## 산출물

- 스크립트: `scripts/research_eth_omega461_live_sltp_asymmetric_tpsl_20260813.py`
- 결과: `tmp/research_20260813/omega461_live_sltp_asymmetric_tpsl/{report.json,
  grid_summary_VAL.csv, component_variants_VAL.csv, priority_combined_ledger_*_VAL.csv}`
- 선행실험(인용): `docs/experiments/eth_omega461_live_sltp_mfe_width_20260813.md`
