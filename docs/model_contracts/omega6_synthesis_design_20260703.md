# Omega6 Synthesis Design (Alpha1~Omega5 장점 취합 설계도)

Status: `draft_design_only_not_promoted`

Last updated: 2026-07-03 KST

## 목적

Alpha1~Alpha8, Omega1~Omega5 전 세대에서 **검증되어 살아남은 요소만** 조합하고,
반복된 실패 패턴(funding leakage, selection 오염, accounting drift, hold-time 불일치,
ownership 혼란)을 **계약 수준에서 구조적으로 차단**하는 차기 아키텍처 설계.

이 문서는 설계도이며 구현/승격 문서가 아니다. 구현 시 각 레이어별 개별 contract를
`docs/model_contracts/`에 신규 작성하고 본 문서를 lineage로 참조한다.

## 설계 원칙 (세대별 교훈의 계약화)

| # | 원칙 | 출처 (검증된 세대) | 차단하는 실패 (실패한 세대) |
|---|---|---|---|
| P1 | Parent 방향성은 tabular ensemble이 소유. RL은 parent를 소유하지 않는다 | Alpha4 HGB baseline (+183%/MDD -22%), Omega1.1 TabM | Alpha4 raw DQN 과다거래 (val -18%/MDD -50%), Alpha3 DSAC exit 붕괴 |
| P2 | 소유권 고정: 방향=parent, 사이징=sidecar, 청산=rule+exit head. 재논쟁 금지 | Alpha5 CatBoost deprecation 결정 | Alpha5~Regime3의 반복된 ownership churn |
| P3 | 동적 리스크는 제약된 템플릿만. 자유형 dynamic risk head 금지 | Omega4.4 v18 side-split sidecar (redteam FULL_PROMOTABLE) | Omega3 dynamic cash sleeve "too much flexibility, unstable" (Hold) |
| P4 | 사이징은 `margin_fraction` 예측, leverage는 고정 | AGENTS.md Futures Risk Sizing Contract, Omega1.2.1 true-leverage 수정 | Omega1.2.1 초기 barrier 버그 (val -5.31%), Omega2.1 회계 무효화 (+102%→+33%) |
| P5 | 모델 선택은 train/calibration 데이터로만. 선택 규칙은 사전 동결 | omega5_live_promotion_blocked_20260702 재승격 조건 | Omega5 승격 차단 (validation/test ledger 오염) |
| P6 | 모든 성과 주장은 fresh-forward bar-by-bar walk-forward로만 | AGENTS.md Fresh-Forward Rule | omega5_live_short_momentum_v2 (재검증 시 val -96%) |
| P7 | Feature는 provenance manifest 필수. 이름 기반 cleanliness 검사 금지 | Alpha8 clean funding reset 규율 | Alpha7 funding leak (+198%→+43% 붕괴, `squeeze_power` 등 이름 우회) |
| P8 | Hold-time budget을 승격 게이트에 포함 (max hold ≤ 24h) | Omega4.6.2 time stop, Omega5 8h max hold | Omega4.6 CONDITIONAL_PASS (222h hold vs 24h 목표, 미해결) |
| P9 | 다중 seed 평가 필수, seed 분산이 크면 기각 | Omega2 12-seed ensemble 관행 | Alpha8 Mamba-DSAC seed 민감성 (OOS +8.85%~+65.81%) |
| P10 | 신규 실행 가정은 live shadow 수집으로 먼저 검증 | Alpha2 L2 replay shadow 규율, Omega5 live-only shadow loop | 미검증 체결 가정의 성과 과장 |
| P11 | Fail-fast: contract 불일치 시 즉시 실패. 자동 보정/alias 금지 | AGENTS.md Fail-Fast Rule, Omega1.1 `.cbm` fail-fast, Omega5 governor flag fail-fast | 조용한 legacy 아티팩트 혼입 (Alpha7 stale lineage) |

## 아키텍처 레이어

```
[L0 Data/Label]  ZigZag 3-class canonical label + provenance-manifested features
      │
[L1 Regime Context]  Regime3 (risk/context feature 전용, 방향 소유 없음)
      │
[L2 Parent Policy]  TabM BatchEnsemble 3-head (direction / quality / exit-context)
      │                primary + fallback 이중화 (동일 clean lineage)
      ├─ CASH ──> [L3 Sequence Entry Gate]  TCN 72-bar gate (edge/margin threshold)
      │
[L4 Risk Sizing Sidecar]  side-split log-risk HGB → margin_fraction (템플릿 제약)
      │
[L5 Exit]  true-leverage scaled price barrier (TP/SL) + position-aware exit head
      │      + time stop (hold budget ≤ 24h) + OU-halflife duration gate
      │
[L6 Governors]  event risk governor (매크로 이벤트 veto + shock haircut)
      │           + deep-stop cooldown + loss-cluster throttle
      │
[L7 Execution]  next_open_limit_touch0_fee20 계약, L2 shadow 검증 후 라이브
```

## 레이어별 계약 요약

| Layer | 역할 | 채택 원형 | 원형 출처 |
|---|---|---|---|
| L0 | canonical 3-class ZigZag action label, funding-clean feature frame | omega1_teacher_contract_20260531 | Omega1 |
| L1 | regime feature는 context/risk 입력 전용. 방향/라우팅 소유 금지 | regime3_policy_20260530 | Regime3 정책 |
| L2 | TabM BatchEnsemble direction/quality 헤드 + Alpha6식 3-head 구조, Alpha7식 primary/fallback 이중화 | omega1_1_tabm_expertdq_20260602, alpha6_entry_quality_exit_5bucket_main_20260522, alpha7_live_stack | Omega1.1 + Alpha6 + Alpha7 |
| L3 | parent=CASH일 때만 활성화되는 frozen TCN sequence gate. 초기 범위는 short-only (검증된 방향부터) | alpha1_v31 deep scout gating, omega462_tcn_sequence_entry_gate_short_variants_20260703 | Alpha1 + Omega462 |
| L4 | side-split(LONG/SHORT 분리) log-risk sidecar. 출력은 margin_fraction 버킷 템플릿, leverage 고정 | omega4_3_valonly_logrisk_tail050 (val +30%/OOS +32%), omega4_4_v18_baseline (OOS +43%/MDD -11%) | Omega4.3 + Omega4.4 v18 |
| L5 | `effective_exposure = margin_notional × execution_leverage` 기준 barrier scaling. exit head는 close 신호만 제공, barrier를 대체하지 않음 | omega1_2_1_true_leverage_price_barrier_20260610, alpha6 exit head, omega4_6_1 duration gate, omega4_6_2 time stop | Omega1.2.1 + Alpha6 + Omega4.6.x |
| L6 | 스케줄된 매크로 이벤트(NFP/ISM/PMI/FOMC) 진입 veto + shock notional haircut. governor는 축소만 가능, 확대 불가 | omega5_event_risk_governor_20260702 (설계 자체는 유효, 차단 사유는 selection 오염이었음), alpha7 deep-stop cd18 | Omega5 + Alpha7 |
| L7 | next-open limit 체결 계약. 신규 체결 가정은 shadow 수집 데이터로 검증 후 반영 | alpha3 execution contract, alpha2_teacher_l2_replay_shadow_20260514 | Alpha3 + Alpha2 |

## 명시적으로 배제하는 것 (실패 확정 요소)

- RL(DSAC/DQN/offline-RL)의 parent 또는 exit 직접 소유 — Alpha3/Alpha4에서 반복 실패.
  DSAC는 향후 sizing 보조 후보로만 재평가 가능 (Alpha5 원칙 유지).
- 자유형 dynamic TP/SL/notional/leverage head — Omega3에서 기각.
- CatBoost/기타 모델의 액션 소유 — Alpha5에서 deprecated. causal feature 공급자로만 사용.
- Foundation-model 시퀀스 레이어 교체 (Chronos/Kairos/Mamba parent) — Alpha2.1/Alpha8에서
  기준 미달 또는 seed 불안정.
- 저장 원장(trade ledger) 기반 성과의 승격 근거 사용 — AGENTS.md 규칙 및 Omega5 차단 사유.
- 이름 기반 funding cleanliness 검사 — Alpha7 우회 사례. lineage 기반 검사만 인정.

## 선택/승격 프로토콜 (Omega5 차단 조건의 계약화)

1. **선택 데이터**: hyperparameter/threshold/variant 선택은 train + calibration split만 사용.
   validation/OOS/test 행은 모델 입력·선택 라벨·선택 기준 어디에도 사용 금지.
2. **선택 규칙 동결**: variant 선택 규칙을 문서로 동결한 뒤 fresh holdout을 1회만 소비.
3. **평가**: AGENTS.md Fresh-Forward Rule 준수 (val 2025-09-01~12-31, OOS 2026-01-01~03-31,
   bar-by-bar causal, 리포트에 4개 플래그 명시).
4. **Seed 게이트**: 최소 8 seed, OOS PnL 부호 일치율 ≥ 7/8, PnL 표준편차/평균 비율 상한을
   구현 contract에서 사전 고정.
5. **Artifact 게이트**: `scripts/audit_omega_artifact_integrity_20260630.py` exit 0 +
   `promotion_pass=true`. exact-threshold parent prediction artifact 필수.
6. **Cost stress**: fee/slippage 1x/2x/3x 순위 유지 (기존 red team gate 승계).
   Alpha1의 교훈: cost1에서 +361%여도 cost3에서 +0.58%면 엣지가 아니다.
7. **Hold budget**: validation max hold ≤ 24h. 초과 시 성과와 무관하게 day-trading 승격 불가.
8. **Runtime 게이트**: exact wiring audit + live/backtest parity audit + sizing trace 필수
   (omega5_trading_bot_exact_wiring / omega5_live_backtest_parity / state quarantine 절차 승계).
9. **Shadow 기간**: 라이브 활성화 전 live-only shadow 수집으로 결정 로그를 축적하고
   shadow 결정과 fresh-forward 예측의 parity를 확인.

## Open Issues

- **[v1 오염 감사 결과 2026-07-03]** fresh-forward validation window(AGENTS.md 기본값 2025-09-01)가
  L2 트레이너 자체의 `SPLIT_TS=2025-10-01` 학습 경계와 한 달 겹쳐, 실제로 모델이 학습한 데이터를
  "검증"으로 채점한 오염이 발견/수정됨(window를 2025-10-01로 이동). 또한 재사용 중인 L4 risk
  sidecar의 sizing 템플릿이 원래 선택될 때 관측한 기간이 이 검증 구간과 겹칠 가능성이 있어
  L2만 진짜 out-of-sample이라는 점을 계약 문서에 명시함. 상세:
  `docs/model_contracts/omega6_synthesis_v1_20260703_contract.md`의 "Contamination / Lookahead
  Audit" 절.
- **[v1 구현 결과 2026-07-03]** L3 TCN 게이트는 실제로는 Omega4.6.2 dual-parent 고유의
  decision trace(h48qual_*/zig075_* side/confidence/margin_fraction/router one-hot 등)를
  피처로 학습되어 있어, 범용 시장 피처 게이트가 아니라 Omega4.6.2에 강결합된 아티팩트임이
  구현 중 스모크테스트에서 확인됨. Omega6 자체 L2 parent 출력을 같은 컬럼명으로 대체 투입하는
  것은 fail-fast 원칙 위반(조용한 재보정)이므로, v1 구현체(`trading_bot_modules/omega6_live.py`)는
  `enable_l3_gate=False`로 L3를 기본 비활성화함. 상세: `docs/model_contracts/omega6_synthesis_v1_20260703_contract.md`.
- L3 sequence gate의 long-side 확장 여부 — Omega462는 short-only만 유효 신호
  (best variant val +2.86%/OOS +58.19%). long은 별도 검증 전까지 비활성. (위 결합 문제 해결 후에나 의미 있음)
- L4 margin_fraction 버킷 템플릿 개수 — Omega3 교훈상 적게 시작 (예: 4~6개).
- Alpha2 L2 replay 레이어의 부활 범위 — limit fill 현실화 가치는 확인됐으나
  실제 오더북 shadow 검증이 여전히 미완.
- Omega5 live-only online learning 후보들(logit/bandit v3/v4)의 편입 여부 —
  shadow 수집 결과가 쌓인 뒤 판단.
