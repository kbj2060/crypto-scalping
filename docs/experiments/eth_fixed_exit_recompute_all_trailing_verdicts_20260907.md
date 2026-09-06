# `sim_exit` 수정 후 트레일링 판정 전수 재계산 (2026-09-07)

선행: [걸 수 없는 스톱 결함](eth_trailing_stop_infeasible_fill_invalidates_exit_edge_20260907.md)

## 한 줄

`sim_exit`을 **거래소가 받아주는 스톱만 인정**하도록 고치고, 그 위에 세워진 판정을 다시 계산했다.
**두 창 모두 양수였던 트레일링 판정 13개 중 살아남은 것 0개.** 배포 섀도우 R은 OOS **+6.06 → −10.82bp**,
자산확장 P3는 **+170.73 → −411.79bp/일**, TRAIN 선별 자산은 **16개 → 0개**가 됐다.

## 1. 수정 내용

`sim_exit(entry, atr, sign, H, L, C, sl, arm, trail, infeasible="exit")`

무장 후 새 스톱 `ns = best − sign·trail·atr`이 **그 봉 종가보다 유리한 쪽**이면 거래소가 거부하는 자리다.

| 모드 | 동작 |
|---|---|
| `exit` (기본) | 즉시 그 봉 **종가에 청산**. 트레일링 스톱 운용의 충실한 모델 |
| `hold` | 스톱을 올리지 않고 직전 스톱 유지 (느슨한 트레일 변형 **정책**) |
| `ignore` | 원문(결함) = `sim_exit_legacy` |

결함 원문은 `sim_exit_legacy`로 **보존**했다(동결 산출물 재현 전용, ⚠️경고 독스트링).

**수정한 파일**
- `scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py` (원문 정의)
- `scripts/research_homer_entry_v2_20260904.py` (사본 정의 — 20+ 스크립트가 여기서 import)
- `scripts/gate_eth_entry_layers_20260903.py` — `trail()` 롱/숏 두 갈래 + `trail_abs()` (층 게이트 표준 청산)
- `scripts/backtest_eth_v_rebound_every_bar_trailing_costgate_20260901.py` (스칼라 참조 + 벡터 엔진 둘 다)

**검증**
- 합성 경로(문서 §4 예시): legacy **+190.0bp** → fixed **+50.0bp** (=종가 100.5), hold +20.0bp
- 무작위 3,000경로: `infeasible="ignore"` == `sim_exit_legacy` **비트 일치**, `fixed ≤ legacy`가 **100.0%** 경로에서 성립
- 프레임 재계산 파리티: legacy 재계산 vs `frame.parquet` 저장값 `net_bp`/`net_bp_flip` **|Δ|max 0.000e+00**
- **네 구현 교차 검증**(무작위 800경로): `sim_exit(fixed)` · 게이트 `trail_abs` · every_bar 스칼라 ·
  every_bar 벡터가 **전부 |Δ|max 0.000e+00**으로 일치 — 서로 다른 코드 경로에서 같은 수정이 같은 값을 낸다.

## 2. 재계산 결과 (bp/건, 순차 포트폴리오 동시 5, 일군집 CI)

### 2-1. F0 프레임 (매 봉 × 양측면 403,190행 = 봉 201,595)

| 판정 | | TRAIN | VAL | OOS |
|---|---|---|---|---|
| **V2 지속 규칙 R** (배포 섀도우) | legacy | +4.49 | +4.41 | **+6.06** |
| n=25,175 첫발동 | **fixed** | **−10.12** | **−9.34** | **−10.82** |
| | *fixed CI95* | *[−11.62,−8.65]* | *[−12.36,−6.23]* | *[−14.07,−7.49]* |
| | hold | −5.85 | −13.15 | +6.32 |
| V3 R 페이드 (대조) | legacy → fixed | −2.95 → −12.18 | −1.05 → −10.56 | +2.09 → −9.09 |
| V1 전체봉 지속 (F0 모집단) | legacy → fixed | +3.12 → −10.97 | +4.94 → −7.72 | +5.25 → −8.69 |
| V1b 전체봉 페이드 | legacy → fixed | +1.78 → −8.85 | −1.10 → −11.72 | +1.72 → −11.67 |

`hold` 변형은 OOS만 +6.32이고 VAL −13.15이라 **두 창 규칙 불통과**다. 구제 아님.

### 2-2. 신호별 × 측면별 × 방향 (32조합)

**두 창(VAL·OOS) 모두 양수: legacy 13개 → fixed 0개.**

legacy 통과 13개: `taker_delta_z_climax/bottom/cont` · `taker_delta_z_climax/top/fade` ·
`short_term_return_z/bottom/cont` · `short_term_return_z/top/cont` · `liquidity_sweep/top/fade` ·
`orthogonal_combo/bottom/cont` · `orthogonal_combo/top/fade` · `smt_divergence/bottom/cont` ·
`fib_extension_exhaustion/{bottom,top}/cont` · `kalman_deviation_meanrev/bottom/{fade,cont}` ·
`kalman_deviation_meanrev/top/cont` → **fixed 통과 0개.**

08-30 "트레일링 비용게이트 돌파/확인"으로 기록된 `taker_delta_z_climax`·`short_term_return_z`가
여기 포함된다. 예: `taker_delta_z_climax/bottom/cont` VAL +8.04→−7.47 · OOS +5.39→−12.78.

### 2-3. 교차자산 4자산 (지속 exp_bp)

| 자산 | TRAIN | VAL | OOS |
|---|---|---|---|
| ETHUSDT | +4.84 → **−10.00** | +4.44 → **−9.16** | +6.87 → **−10.30** |
| BTCUSDT | −1.73 → −10.10 | −1.49 → −9.13 | −2.20 → −11.64 |
| XRPUSDT | +6.47 → −8.14 | +2.15 → −9.57 | +1.46 → −11.06 |
| SOLUSDT | +3.99 → −9.65 | +6.58 → −7.83 | +4.21 → −10.07 |

형식 판정(재현)은 legacy에서도 이미 4/4 False였다 — 이번 수정은 그걸 **더 깊은 음수로** 바꾼다.
fixed에서는 네 자산 전부 세 창 모두 CI 상한이 0 아래다.

### 2-4. 자산 확장 60종 (일 자기자본 bp · 크기 매칭 후)

| 팔 | VAL | OOS |
|---|---|---|
| P0 ETH 단독 | +0.15 → **−11.67** (샤프 +0.07 → −5.78) | +6.36 → **−7.25** (+2.75 → −3.54) |
| P1 ETH+XRP+SOL | +2.00 → **−28.13** (+0.52 → −8.30) | +7.94 → **−23.18** (+1.85 → −5.93) |
| P3 전자산 동일가중 | +129.90 → **−437.29** (+5.11 → −20.10) | +170.73 → **−411.79** (+7.04 → −17.59) |

**TRAIN 일CI 하한>0 선별 자산: 16개 → 0개.**

### 2-5. 09-06 페이드/지속 라벨 (첫발동 18,139건)

| | P(지속 > 페이드) | 평균 지속 | 평균 페이드 |
|---|---|---|---|
| legacy | 0.5432 | +4.95 | −1.93 |
| **fixed** | **0.4887** | **−10.00** | **−11.51** |
| hold | 0.4293 | −4.80 | −11.89 |

⭐**09-06 결론의 핵심 문장이 무너진다.** 그날 "경로상 지속 우세는 모든 지평에서 동전(H200 0.5054
[0.4981,0.5127])인데 pnl 기준 승률은 0.5433이므로 **청산 구조가 동전을 돈으로 바꾼다**"고 썼다.
수정본에서 pnl 기준은 **0.4887**로, 경로상 동전(0.5054)보다 오히려 낮고 두 방향 다 손실이다.
**청산 구조는 아무것도 만들어내지 않았다 — 0.5432가 회계 산물이었다.**
(그날의 다른 축 결론 "방향은 배워지지 않는다 · 크기는 atr_pct 단독으로 AUC 0.82"는 라벨과 무관한
AUC 측정이라 그대로 유효하다.)

## 3. 모델 기반 판정 — 재학습 필요, 여기서는 하한만

선택 규칙이 학습 모델인 판정(호메로스 진입 v2 F0~F3 · F0 V자반등 경제라벨 모델 · 증거신호8 경제성)은
라벨이 바뀌면 **재학습해야** 정확한 값이 나온다. 다만 하한은 명확하다:

- 이 모델들이 뽑는 **모집단 자체**(V1)가 +3.1~+5.3 → **−7.7~−11.0bp**로 이동했다.
- 진입 v2가 주장한 팔 성과는 VAL +2.63 / OOS +8.33bp(F0)였다.
- 즉 같은 주장을 유지하려면 모델이 **13~16bp의 선별 실력**을 새로 보태야 하는데,
  기록된 팔 AUC는 **0.5118 (VAL) / 0.5271 (OOS)**다.

## 4. 라이브 섀도우 러너 4종도 수정 (같은 날, 배포 완료)

`_close(..., p["stop"], ...)`로 스톱가에 청산 기록하던 4개 러너에 같은 검사를 넣었다. 새 사유 `stop_infeasible`로 기록한다.

| 러너 | 서버 실행 상태 |
|---|---|
| `live_eth_fire_cont_shadow_runner_20260904.py` | **3 프로세스**(ETH·XRP·SOL, `--asset` 분기) |
| `live_eth_v_rebound_econ_shadow_runner_20260902.py` | 1 프로세스 |
| `live_eth_retail_shift_b2_shadow_runner_20260905.py` | 1 프로세스 |
| `live_eth_entry_limit_fade_shadow_runner_20260903.py` | 미가동(진입모델 v1 철회로 정지) |

**파리티 검증** — 러너 `manage()` vs 수정된 `sim_exit`(무작위 경로, 원장 bp 소수 2자리 반올림 포함):
`fire_cont` **0/400 불일치** · `v_rebound` **0/250** · `retail_b2` **0/250** (|Δ|max 5e-07 = 반올림).
`entry_limit_fade`는 `manage(s,bars,pol)` 시그니처와 `#post` 합성봉 때문에 스모크만 — 120경로 중
`stop_infeasible` 97건으로 경로 작동 확인. 러너 자체 `--selftest`: `fire_cont` ok · `retail_b2` ok
(나머지 둘은 `--selftest` 플래그 자체가 없다).

**영향 없음이 확인된 러너**: `live_btc_evidence_signal_shadow_runner_20260902.py` ·
`live_xrp_evidence_signal_shadow_runner_20260903.py` — 트레일링 청산 로직 자체가 없는 신호 기록기다.

**진행 중 포지션**: 수정 시점 `fire_cont` 오픈 4건이 전부 `armed=False`(스톱이 진입 −5·ATR 초기값
= 항상 실행 가능)라 규약이 섞이는 구간이 없다. `v_rebound`·`retail_b2`는 오픈 0건이었다.

**⚠️ 수정 이전 원장 기록은 옛 규약이다.** `stop_infeasible` 사유가 없는 과거 `stop` 청산 건은
스톱가에 체결된 것으로 기록돼 있다(`fire_cont` 39건 중 69.2%가 걸 수 없는 자리, 기록 −0.35 →
정직 −14.28bp). 원장을 소급 수정하지 않았다 — 사유 필드로 구분한다.

**배포**: `check_deploy_drift.sh` 종료코드 0 → 서버 md5 4/4 로컬 일치 확인(다른 세션 변경 없음) →
`origin/main` 기준 브랜치에 체리픽 → `push HEAD:main`. 워처가 restart 배선을 가진 유닛은
`trading-bot`/`ops-watchdog`/`prometheus-exporter`/대시보드뿐이고 이번 변경은 `scripts/live_*`만
건드리므로 **워처는 pull만 하고 아무 서비스도 재시작하지 않는다** — 러너 5개는 별도 재시작이 필요하다.

⚠️ 브랜치를 실수로 `origin/main`이 아닌 이전 세션 커밋(`db2a5f1`) 위에 만들었다가, 그대로 밀면
서버에 미추적으로 있던 3개 경로(`run_eth_*_20260906.py`)가 커밋돼 워처 폴백 stash pop이 깨지는
것을 `check_deploy_drift.sh` 경고로 발견하고 `origin/main` 기준으로 다시 만들었다.

그 밖에 자체 트레일링 사본을 가진 과거 백테스트/감사 스크립트 다수가 같은 결함을 갖고 있다.
전부 이미 기각된 판정이거나 위 재계산이 커버하는 모집단이라 개별 수정은 하지 않았다 —
**과거 스크립트의 트레일링 수치를 인용하기 전에 그 파일이 실행가능성 검사를 하는지 먼저 확인할 것.**

## 5. 요약

| | legacy | fixed |
|---|---|---|
| 두 창 양수 (신호×측면×방향 32조합) | 13 | **0** |
| TRAIN 선별 자산 (60종) | 16 | **0** |
| 배포 섀도우 R OOS | +6.06 | **−10.82** |
| F0 모집단 기준선 OOS | +5.25 | **−8.69** |
| 09-06 P(지속>페이드) | 0.5432 | **0.4887** |

**이 저장소에서 트레일링 청산으로 잰 양수 판정 중 실행 가능한 체결 가정에서 살아남는 것은 현재 없다.**

## 산출물
- `scripts/recompute_frame_labels_fixed_exit_20260907.py` → `data/research/eth_fixed_exit_recompute_20260907/{report.json,bar_labels.parquet}`
- `scripts/recompute_signal_verdicts_fixed_exit_20260907.py` → `.../signal_verdicts.json`
- 재실행: `research_crossasset_fire_continuation_replication_20260905.py` · `research_eth_asset_expansion_r_rule_20260906.py`
- legacy 원본 백업: `data/research/_legacy_before_exit_fix_20260907/`
- HOLDOUT 미접촉.
