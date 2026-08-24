# 오메가4.6.1 라이브 승격 시드 검증 — Trade-level DSR/PBO/falsification_audit (2026-08-19)

## 배경

[[eth_live_promotion_seed_robustness_3seed_20260819]]가 N=3(이후 N=5)로 라이브(h48qual+zig075
dual, 배포 시드 260620)의 시드 강건성을 확인해 6개 평가창 중 4개에서 부호플립을 발견했다.
사용자 요청으로 `core/selection_stats.py`(DSR/PSR/PBO-CSCV/falsification_audit)를 적용해봤으나
([[eth_live_stack_never_passed_dsr_pbo_20260819]]), 남아있는 산출물이 6개 창 수준 집계 PnL%
뿐이라 `pbo_cscv`는 조합 2개짜리 사실상 무정보 계산이었고 `falsification_audit`은 `n_periods>=10`
요건조차 못 채워 아예 실행 불가했다. 이 문서는 그 한계를 해소하기 위해 시드 94046540/524707103을
재학습하고, trade-level 원장(ledger)까지 보존한 뒤 일별로 재구성한 결과다.

## 방법론

1. **재학습**: `scripts/eth_live_promotion_seed_robustness_{h48qual,zig075}_seed_variant_20260819.py`
   를 시드 94046540/524707103 각각에 대해 그대로 재실행(원본 스크립트 무수정, 원본코드+canonical
   데이터+102-pin 스택, epochs=2, device=cpu). 4개 학습 전부 성공(각 ~278~286초, CPU). 시드
   260620은 실제 라이브 번들을 그대로 재사용(재학습 안 함).
2. **평가 재실행**: `scripts/eth_live_promotion_seed_robustness_eval_3seed_20260819.py`를 그대로
   재실행 — Fresh-Forward bar-by-bar(각 번들 자신의 추론, 저장 ledger 재사용 없음). 이번엔 산출물을
   삭제하지 않고 보존해, `portfolio_ledger_{window}_posfix_canonicaldata.csv`(trade별
   entry/exit timestamp, side, source_component, reason, **trade_return**, notional, margin_fraction,
   leverage)가 3개 시드 × 6개 창 = 18개 파일로 전부 남았다. 재현된 window-level PnL%는 원본
   N=3 결과와 소수점까지 동일(재현성 확인됨).
3. **trade-level → daily 재구성**(`scripts/analyze_eth_live_promotion_seed_dsr_pbo_tradelevel_20260819.py`):
   - `trade_return`은 이미 계좌수준 fractional return(`research_eth_omega461_exit_head_portfolio_
     asymmetric_20260813.py::_ledger_metrics`가 `cumprod(1+returns)`로 그대로 복리화하는 걸 코드로
     확인 — notional을 별도로 곱할 필요 없음).
   - `with_gate`가 실제 라이브 동작이므로, `research_eth_omega461_live_sltp_mfe_width_20260813.py::
     _duration_gated`와 동일한 `ou_halflife <= greedy.DURATION_THRESHOLD` 게이팅을 그대로 재현
     (해당 window의 frame을 다시 로드해 entry_timestamp로 조인).
   - 같은 날 청산된 거래는 로그수익률로 합산(복리 일관성 유지)해 일별 수익률로 압축, 6개 창을
     시간순으로 이어붙인 뒤(2025q1→q2→q3→val→oos_q1→oos_q2는 실제로 연속된 캘린더 구간)
     **2025-01-01~2026-06-30 전체 일별 캘린더(546일)**에 재색인(거래 없는 날은 0으로 채움) —
     3개 시드가 동일한 행(=동일 날짜)을 공유하도록 강제. 이게 `pbo_cscv`/`falsification_audit`의
     "같은 행=같은 기간" 전제를 충족시키는 핵심 단계다.

## 결과

| 지표 | window-level(6개 창, 이전 문서) | **trade-level→daily(546일, 이 문서)** |
|---|---:|---:|
| n_periods | 6 | **546** |
| 260620 관측 Sharpe | 1.114 | 0.0783 |
| noise_floor_sharpe (N=3) | 0.416 | 0.0230 |
| **DSR** | 0.883 | **0.915** |
| passes_95 | False | **False** |
| PBO-CSCV | 0.0 (조합 2개, 사실상 무정보) | **0.444 (조합 252개, 정상 계산)** |
| falsification_audit | 실행 불가(n_periods<10) | **False** (zero-null 94.0%ile, placebo-null 89.2%ile, 둘 다 요건 95%ile 미달) |

세 시드의 일별 Sharpe: 260620=+0.0783, 94046540=+0.0322, 524707103=+0.0308 — 260620이 여전히
최고지만, window-level 계산이 시사했던 것(1.114 vs 0.26~0.27, 약 4배 격차)보다 실제 격차는
훨씬 작다(2.5배).

## 해석

1. **DSR 0.915**: 관례적 통과선(0.95)에는 여전히 못 미치지만, window-level(0.883)보다 통과선에
   더 가까워졌다 — 데이터 해상도가 높아지자 260620의 우위가 아주 근거 없지는 않다는 쪽으로 약간
   움직였다. 그래도 "확정 통과"는 아니다.
2. **PBO 0.444**: 이번엔 진짜 252개 조합에 기반한 유의미한 추정치다(이전 window-level의 조합 2개짜리
   추정과 다름). `pbo_cscv` 자체 docstring이 "PBO 0.5 근방=탐색이 정보를 전혀 안 담고 있다는 뜻"이라
   명시하는데, 0.444는 그 노이즈 기준선에 상당히 가깝다 — "260620을 고른 게 다른 시드를 골랐을
   경우보다 나은 선택이었다"고 자신 있게 말하기 어렵다. 단, 설정(config)이 3개뿐이라 개별 split의
   rank 통계 자체가 이산적/저해상도({1/4, 2/4, 3/4} 근방 몇 개 값)라는 한계는 있다 — 546일 축은
   충분히 강력해졌지만 "3개 시드뿐"이라는 축의 한계는 여전하다.
3. **falsification_audit FAIL**: 가장 결정적인 결과다. 실제 최고 시드(260620)의 Sharpe가 (a) 완전
   무작위(i.i.d. 가우시안) 영벌 대비 94.0백분위, (b) 실제 변동성 군집/자기상관은 보존하되 진짜
   타이밍은 파괴한 microstructure-placebo 영벌 대비 89.2백분위 — 둘 다 요구되는 95백분위에
   못 미친다. 즉 **"이 정확한 탐색(3개 시드 중 최고를 봤다)"이 순수 노이즈만으로도 재현 가능한
   수준의 결과라는 뜻이다.**

## 종합 판정

세 지표가 서로 다른 각도에서 같은 방향을 가리킨다 — **DSR은 통과선 미달, PBO는 노이즈 기준선(0.5)에
근접, falsification_audit은 명시적으로 실패.** window-level 분석 때보다 표본은 훨씬 커졌고(6→546
periods) 계산 자체는 훨씬 신뢰할 만해졌지만, 결론의 방향은 바뀌지 않았다 — 오히려 falsification_audit
이라는, 이전엔 아예 돌릴 수 없었던 가장 엄격한 검정이 처음으로 명시적 FAIL을 냈다.

**한계(정직히 명시)**: (1) N=3(=config 축)은 여전히 CLAUDE.md N≥5 요구에 못 미친다 — PBO/
falsification_audit의 표본기간 축(periods)은 이제 충분하지만 config 축은 아니다. (2) risk sidecar는
3개 시드 전부 원본(260620) 것을 frozen 재사용(이전과 동일한 단순화, 미해결). (3) 일별 재표본화는
거래가 드물어(101~123/546일만 nonzero) 대부분의 "일별 수익률"이 정확히 0이다 — Sharpe 계산 자체가
희소수익률 특유의 통계적 성질(첨도 등)에 민감할 수 있다는 점은 이 문서에서 별도 검증하지 않았다.

## 산출물

- 재학습 스크립트(기존, 무수정 재사용): `scripts/eth_live_promotion_seed_robustness_{h48qual,zig075}_seed_variant_20260819.py`
- 평가 재실행(기존, 무수정 재사용): `scripts/eth_live_promotion_seed_robustness_eval_3seed_20260819.py`
- 신규 trade-level 분석: `scripts/analyze_eth_live_promotion_seed_dsr_pbo_tradelevel_20260819.py`
- 원장: `tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_{seed260620_original,94046540,524707103}/portfolio_ledger_{window}_posfix_canonicaldata.csv`(18개)
- 일별 정렬 행렬: `tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_tradelevel_daily_matrix.csv`(546×3)

`fresh_forward_bar_by_bar=true`(재평가 자체가 원본 eval 파이프라인 그대로), 이 문서의 통계 재구성
단계는 재학습·재평가 없이 기존 산출물만 재가공 — 새로운 promotion/model-selection 근거로 쓰기 위한
것이 아니라(Fresh-Forward Validation/OOS/Test Rule의 "promotion 근거" 대상 아님) 시드 검증 자체의
통계적 엄밀성을 높이기 위한 재해석이다.
