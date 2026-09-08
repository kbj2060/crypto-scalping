# Omega4.6.1-RegimeGBM Rebuild Data Contract

Status: `phase1_complete` — Phase 0 완료, Phase 1 진단 = 무신호(VAL 열세/OOS 우세, 표본 12~29건). Phase 2 착수 판단 대기

Last updated: 2026-09-09 KST (Phase 1 결과 반영)

하위 프로젝트 개설: 2026-09-09, 사용자 지시 "오메가 4.6.1을 전신으로 레짐을 대시보드 GBM으로
재구성하자". 이 문서는 이 라인의 **살아있는 단일 진실 소스**다 — 세션 시작 시 먼저 읽고, 상태를
바꾸는 발견이 나오면 같은 턴에 갱신한다. 방법론/중간 수치는 여기 넣지 말고
`docs/experiments/omega461_regimegbm_*_<date>.md`에 쓰고 백틱 경로로 인용한다.
데이터 인벤토리는 `docs/model_contracts/omega461_regimegbm_rebuild_data_resources_20260909.md`.

## Scope

- Model id: `omega461_regimegbm_rebuild_20260909`
- 전신(前身): Omega4.6.1 ETH 라이브 스택 (`omega4_6_1_duration_ou_halflife_risk_gate_20260630`)
- Architecture: 전신과 동일한 ThreeHeadTabM(k=8, hidden 192, 3층) × 2컴포넌트(h48qual/zig075)
  × 3 레짐전문가 + 리스크 사이드카(HGB) + duration gate. **바꾸는 것은 레짐 척추 하나뿐이다.**
- Purpose: 레짐 신호원을 2024년 학습 wide24 HMM에서 대시보드 계열 GBM(S12_K3)으로 교체하고
  부모부터 재학습해, 라우팅 안정성과 레짐 피쳐 품질 개선이 OOS 경제성으로 이어지는지 검증.
- Implementation script: (Phase 0에서 생성)
- Report artifact: (Phase 0에서 생성)
- Model artifacts: (Phase 2에서 생성)
- 워크벤치 계약: `tmp/omega461_regimegbm_rebuild_20260909/contract.json`
  (`pipeline/architecture_workbench.py validate` 통과 필수)

### 왜 이 축인가 — 전신 대비 측정된 차이

같은 봉 66,528개(2026-01-01~08-19) 실측:

| | wide24 HMM (전신) | S12_K3 GBM (대시보드) |
|---|---|---|
| 모델 | 12-state Gaussian HMM + RobustScaler | HistGradientBoostingClassifier |
| 피쳐 | 24 | 136 (wide24 24개의 상위집합) |
| 학습 | 2024년만 | 2024-01-01~2026-06-30 |
| 검증 bal_acc | 0.7638 | 0.8550 |
| flip율 | 0.1213 | 0.0944 |
| 중앙 상태지속 | 3봉 (15분) | 7봉 (35분) |

두 분류기 일치율 **69.3%**. 불일치 30.7% 중 **29.8%p가 추세↔횡보 경계**이고 bull↔bear 정면충돌은
**0.96%(641봉)**뿐이다 — 방향 판단은 사실상 합의하고, "지금이 추세인가"에서 갈린다.
근거: `docs/experiments/omega461_regime_classifier_comparison_20260909.md` (작성 예정).

### 전신에서 레짐이 쓰이는 3곳 (전부 갈아야 함)

1. 부모 TabM의 102개 `base_cols` 중 마지막 6개
   (`regime3_current_sensitive_wide24_{bull_prob,bear_prob,chop_prob,confidence,entropy,margin}`)
2. 전문가 서브넷 라우팅 (`argmax`)
3. 리스크 사이드카 29피쳐 중 `parent_router_margin`
   (= `regime3_current_sensitive_wide24_margin`, `omega4_6_1_live.py:198`)

→ 레짐 교체는 한 줄 교체가 아니라 **부모·사이드카·예측 아티팩트 전면 재구축**이다. 이것이 이
하위 프로젝트를 새로 파는 이유다(전신 아티팩트는 동결 유지, 건드리지 않는다).

## ⚠️ 최우선 미해결 이슈 — 레짐 GBM의 학습구간이 평가창을 삼킨다

대시보드 S12_K3 GBM의 `train_range`는 **2024-01-01 ~ 2026-06-30**이다. 이는 이 라인의
validation(2025-10~12)과 OOS(2026-01~02)를 **전부 포함**한다. 그 모델의 예측을 그대로 부모 피쳐로
쓰면, 라벨 자체는 인과적(er/net/slope 전부 후방참조)이더라도 **적합된 모델이 그 봉들의 라벨을 이미
본 상태**라 downstream 평가가 오염된다.

**Phase 0에서 반드시 해소한다. 미해소 상태로 만든 Phase 1+ 결과는 전부 provisional이며 승격·동결·
보고에 쓰지 않는다.** 채택할 해소안은 Phase 0에서 결정한다:

- (a) **재학습 컷오프**: validation 시작(2025-09-01) 이전까지만 학습한 레짐 GBM을 따로 만든다.
  가장 단순하고 계약이 깨끗하다. 대시보드 배포본과는 다른 아티팩트가 된다.
- (b) **워크포워드 레짐 예측**: 평가창을 구간별로 나눠 각 구간 직전까지만 학습한 모델로 예측.
  라이브 재현성이 가장 높지만 비용이 크다.
- (c) 기각: 대시보드 배포본 그대로 사용 — **불가**. 위 이유로 계약 위반.

## Phase 0 진행 상태 (2026-09-09)

### ✅ 완료 — 컷오프 재학습 (누출 해소)

`scripts/build_omega461_regimegbm_cutoff_sidecar_20260909.py`.
config/136피쳐/medians/라벨정의(S12,K3)/SEED 7529 전부 배포본과 동일, **학습 컷오프만**
2026-06-30 → **2025-09-30**. 라벨 임계값도 같은 마스크에서만 캘리브레이션.

| 지표 | 값 |
|---|---|
| 프레임 | 280,177봉 (2024-01-01~2026-08-30), median 대체 피쳐 **0개** |
| 학습 | 183,985봉, 클래스비중 bull .2298 / bear .2093 / chop .561 |
| 라벨 임계값 | T1 0.286242 · T2 0.224909 (배포본 0.287469 / 0.225296 — 거의 동일) |
| **bal_acc** | in-sample 0.9070 · **OOF 0.8541** · **VAL 0.8531** · OOS 0.8445 |
| 출력 flip율 | 0.0982 (전신 wide24 0.1213, 배포본 0.0965) |

**in-sample 낙관폭 +0.0529** — 컷오프만으로는 부족했다는 실측 근거. TRAIN 창을 purged
시간블록 5-fold OOF 로 채운 결과 **OOF 0.8541 ≈ VAL 0.8531** 로, 부모가 학습·평가에서 보는
레짐 피쳐 품질이 일치한다(Phase 0 의 실제 목표). 아티팩트
`tmp/omega461_regimegbm_rebuild_20260909/regime_cut2509_model.joblib`, 접두사
`regime3_s12k3_cut2509_`, 사이드카 3종 생성 완료.

### ⚠️ 미해결 — 파리티 게이트, **기준 자체가 잘못 설계됨**

`scripts/audit_omega461_regimegbm_offline_live_parity_20260909.py`.
경로 A = 캐노니컬 CSV 의 136피쳐 / 경로 B = 원시 컬럼에서 `FeatureEngineer().process()` 재구성.

| 창 | 워머업 | 1e-6 초과 피쳐 | argmax 일치 |
|---|---:|---:|---:|
| 파일 꼬리 2026-08-24~30 | 4,320 | 1 (`ou_halflife` 9.99e-1) | 99.150% |
| OOS 2026-02-22~28 | 4,320 | 3 (rank 파생) | 99.950% |
| OOS 2026-02-22~28 | 20,000 | 3 (변화 없음) | 99.950% |
| **VAL 2025-12-25~31** | 20,000 | **0** (전부 ≤5.7e-08) | 99.950% |

세 가지가 확인됐다.

1. **파이프라인은 갈리지 않는다.** 캐노니컬 CSV 구간이 깨끗하면 `FeatureEngineer` 와
   136/136 피쳐가 ≤5.7e-08 로 일치한다(VAL 창). 첫 실행의 `ou_halflife` 9.99e-01 은
   **비교 창을 파일 꼬리로 잡아 오염 구간을 잰 것**이었다 — 정상 구간(2026-05) 재비교에서
   두 경로 corr **1.000000**, 최대차 **9.975e-17**.
2. **캐노니컬 2026 파일에 구간 경계 아티팩트가 있다.** `ou_halflife` 는 2026 전체의 3.67%가
   정확히 1.0 으로 클리핑돼 있고 이상 구간이 파일 확장 시점(`bak_pre_extend_*`)과 겹친다.
   OOS 창의 rank 파생 3종(`compression_score`/`bb_width_pct_rank_288`/`compression_release_up`)은
   워머업을 4,320→20,000 으로 늘려도 **정확히 1/288 = 3.472e-03**(한 랭크 단위) 차이가 남는다.
   2025 파일(VAL 창)은 깨끗하다.
3. **게이트 기준이 원리적으로 통과 불가였다.** 나(작성자)는 트리 모델 출력에 연속 허용오차
   1e-6 을 걸었다. HGB 는 split 경계에서 불연속이라 입력이 1e-08 만 달라도 leaf 가 바뀌어
   출력이 도약한다 — 실제로 피쳐가 전부 ≤5.7e-08 로 일치하는 VAL 창에서도 2,000봉 중 1봉의
   argmax 가 뒤집힌다(99.950%). **입력이 bit-identical 이 아닌 한 이 기준은 달성될 수 없다.**

### 판정 — 사용자 결정 (2026-09-09): "피쳐 누수나 룩어헤드 문제가 아니라면 넘어가자"

**잔차는 누수도 룩어헤드도 아니다.** 근거:

- 이 라인의 실제 누수 위험(레짐 GBM 학습구간이 VAL/OOS 를 삼킴)은 컷오프+OOF 로 해소됐고,
  `OOF 0.8541 ≈ VAL 0.8531` 로 학습·평가 피쳐 품질이 일치함이 실측됐다.
- VAL 창 잔차는 **136피쳐가 전부 ≤5.7e-08 로 일치하는데도** argmax 가 2,000봉 중 1봉 뒤집히는
  것으로, HGB 의 split 경계 불연속성이다. 미래 정보 유입과 무관하다.
- OOS 창 rank 파생 3종의 1/288 차이는 캐노니컬 파일 구간 경계의 **stale(낡은) 값** 때문이다.
  낡은 값은 미래 정보가 아니다 — 방향이 lookahead 와 정반대다.

따라서 게이트는 **"통과"가 아니라 "비누수로 판정되어 사용자 결정으로 waive"** 로 기록한다.
잔차 수치(위 표)는 그대로 남기며, Phase 3 최종 평가 시 이 잔차가 결과를 흔들 수 있는지
재확인한다(예: argmax 불일치 봉을 제외한 민감도).

## Dataset Split

전신과 동일 기준을 유지한다(비교 가능성 확보). 날짜 경계가 바뀌면 이 표를 먼저 고친다.

| Split | Source | Timestamp range | Rows | Use |
|---|---|---|---:|---|
| Train | `data/splits/year_oos/training_features_2025.csv` (+2024) | 2025-01-01 ~ 2025-09-30 | 78,510 (부모 후보) | 부모/사이드카 학습 |
| Validation | 동상 | 2025-10-01 ~ 2025-12-31 | 26,490 | 후보 선택 (validation_only) |
| Test/OOS | `data/splits/year_oos/training_features_2026_rebuilt.csv` | 2026-01-01 ~ 2026-02-28 | 16,832 | fresh-forward 최종 평가 |

Audit (Phase 0에서 채운다):

- Timestamp overlap: 미확인
- Duplicate timestamps: 미확인
- Warmup handling: 레짐 GBM은 DAYS_BACK=15 워머업 필요 (라이브 스코어러 관례)
- OOF/embargo: 미결정 — 레짐 GBM 재학습 컷오프와 함께 정한다

## Shared Feature Contract

- Canonical feature source: `data/splits/year_oos/training_features_{2024,2025,2026_rebuilt}.csv`
- Feature count: 102 base + 13 position = 115 (**전신과 동일 개수 유지**; 마지막 6개 레짐 컬럼의
  *생성원*만 교체). 새 접두사는 Phase 0에서 확정하고 여기 기록한다.
- Normalization: 전문가별 RobustScaler/표준화 (전신 번들의 `scaler` 규약 승계)
- Missing fallback: 레짐 GBM은 `feature_medians` 대체 (라이브 스코어러와 동일)
- Live availability: 레짐 GBM 136피쳐 전부 캐노니컬 프레임에 존재함을 확인 완료
  (2026-09-09 실측, median 대체 0개)
- ⚠️ 금지 접두사(`future_`/`label_`/`target_`/`exit_`/`clean_regime4_`/`regime4_pred_`)는
  `pipeline/architecture_workbench.assert_safe_feature_columns`로 강제한다.

Feature list:

```text
(Phase 0에서 확정 후 붙여넣기 — 전신 102 base_cols에서 레짐 6개만 신규 접두사로 치환)
```

## Layer Contracts

| Layer | Input state/features | Train labels | Output | Artifact |
|---|---|---|---|---|
| L0 레짐 | 136 (S12_K3 GBM 피쳐셋) | S12_K3 3-class | bull/bear/chop prob + confidence/entropy/margin | (Phase 0) |
| L1 부모 | 102 base + 13 pos | zigzag_action / h48_conservative / terminal_giveback | direction(3) · quality(3) · exit(2) | (Phase 2) |
| L2 리스크 | 29 (부모출력 + 결정층) | margin/leverage | margin_fraction, leverage | (Phase 3) |
| L3 라우터 | h48qual > zig075 greedy 우선순위 | — | 단일 포지션 결정 | — |

## Label Contract

- Horizon: 전신 승계 — direction/quality는 zigzag/48bar ATR 배리어, exit은 terminal giveback
- Cost included: fee 0.0005 + slip 0.0002 (왕복 14bp)
- Future path usage: 라벨 생성에만 사용, 피쳐 창은 라벨 탐색 시작보다 최소 1봉 앞선다
  (CLAUDE.md 사건 라벨 경계 계약)
- Leakage controls: 위 "최우선 미해결 이슈" 해소가 전제조건
- Known limitations: 전신의 direction_head 무스킬 결론(h48qual 0/5 시드, zig075 N=5 구분불가)은
  **이 라인에서 자동으로 해소되지 않는다.** 레짐 교체가 그 벽을 넘는지가 이 라인의 실질 질문이다.

## Cost/Risk Assumptions

- Fee: 0.0005 / Slippage: 0.0002 (cost_mult 1x/2x/3x 스트레스 보고 필수)
- Max notional exposure: 1.8 / Leverage cap: 5.0 (전신 승계)
- Sizing: `notional = margin_fraction × leverage` (CLAUDE.md Futures Risk Sizing Contract)
- Funding / 청산: 전신과 동일하게 미모델링 — 한계로 명시

## Output Contract

Required decision columns:

```text
action
side
notional_exposure
leverage
position_fraction
quality_score
confidence
```

Required report metrics:

```text
pnl
mdd
trades
trades_per_day
wr
avg_notional
avg_leverage
monthly
cost_stress
```

## Red Team Gates

- [ ] **레짐 GBM 학습구간이 이 라인의 VAL/OOS를 포함하지 않음** (최우선 이슈 해소)
- [ ] Train/validation/test timestamp overlap audit is zero.
- [ ] No bfill/full-sample scaler/future feature enters live state.
- [ ] Fee/slippage 1x/2x/3x ranking is reported.
- [ ] Score/probability buckets are calibrated against realized net PnL.
- [ ] Monthly/weekly walk-forward is reported.
- [ ] Live train state parity is checked (레짐 6컬럼 offline↔live 0-diff).
- [ ] Funding/liquidation limitations are documented.
- [ ] **Seed-Diversity Gate**: N≥5 진짜 무작위 시드, 시드 리스트를 리포트에 기록.
- [ ] **Fresh-Forward**: `fresh_forward_bar_by_bar=true`, 저장 원장 입력 금지.
- [ ] **DSR/PBO/falsification_audit** (`core/selection_stats.py`) — 전신이 한 번도 통과 못 한
      게이트다(DSR 0.915 < 0.95, PBO 0.444, falsification FAIL). 이 라인은 통과를 요구한다.
- [ ] `scripts/audit_omega_artifact_integrity_20260630.py` exit 0 + `promotion_pass=true`.
- [ ] `scripts/audit_position_feature_train_inference_parity_20260818.py` confirmed 항목 증가 없음.

## Phase 1 결과 요약 (2026-09-09) — 무신호

전문: `docs/experiments/omega461_regimegbm_phase1_routing_ab_20260909.md`

| 창 | 라우팅 일치 | A wide24 | B cut2509 | Δ(B−A) |
|---|---:|---|---|---|
| validation (26,496봉) | 69.30% | +36.82% / −24.34% / 29건 | +10.97% / −21.69% / 28건 | **PnL −25.86pp** · MDD +2.65pp |
| oos (16,992봉) | 68.15% | +59.82% / −12.88% / 13건 | +100.84% / −14.38% / 12건 | **PnL +41.03pp** · MDD −1.50pp |

새 레짐은 두 창 모두 **chop 전문가로 약 8pp 더 라우팅**한다(.497→.575, .501→.581).

**판정: 무신호.** 부호가 두 창에서 정반대이고 표본이 29·13건이라 신호와 잡음을 구분할 수 없다.
이 VAL/OOS 불일치 서명은 `omega4_6_1_upgrade_investigation_20260706.md` 의 여섯 후보 중 셋을
기각시킨 것과 같은 서명이다. 계약의 `candidate_selection_scope: validation_only` 만 적용하면
이 후보는 **선택되지 않는다** — OOS 우세는 선택 근거로 쓸 수 없다.

**그러나 Phase 2 자동 기각은 하지 않는다**(사전 기재된 규칙). Phase 1 은 의도적으로 불일치를
도입한 진단이기 때문이다 — 전문가들은 wide24 라우팅으로 학습됐는데 라우팅만 바꾸면 학습 때
못 본 봉 분포를 받는다. Phase 2 의 재학습이 바로 그 불일치를 없앤다. 따라서 이 결과는
"후보가 나쁘다"가 아니라 **"이 진단으로는 판별 불가"**이며, Phase 2 착수는 이 숫자가 아니라
사전 논거와 비용으로 결정한다.

## 단계 계획

| Phase | 내용 | 산출물 | 킬 기준 |
|---|---|---|---|
| 0 | ✅**완료** 컷오프 재학습(2025-09-30)+purged OOF, 사이드카 3종, 파리티 감사 | `regime_cut2509_model.joblib`, 사이드카 3종, 파리티 리포트 | ~~0-diff~~ → 비누수 판정으로 waive (2026-09-09 사용자 결정) |
| 1 | ✅**완료** 동결 부모 라우팅만 교체 `greedy_replay` A/B | `docs/experiments/omega461_regimegbm_phase1_routing_ab_20260909.md` | **무신호** — VAL −25.86pp / OOS +41.03pp, 표본 29·13건. 계약의 validation_only 규칙상 후보 미선택 |
| 2 | 부모 ThreeHeadTabM 재학습 (N≥5 시드, `--pin-` 계약 고정) | 새 번들 | 시드 부호 불일치 시 중단 |
| 3 | 리스크 사이드카 재학습 + fresh-forward VAL/OOS 전체 평가 | 새 사이드카, 원장 | — |
| 4 | 승격 게이트 (DSR/PBO/falsification + artifact integrity) | 감사 리포트 | 미통과 시 승격 불가 |

Phase 1은 진단이다 — 전문가들이 wide24 라우팅으로 학습됐으므로 라우팅만 바꾸면 학습 때와 다른 봉
분포를 받는다. **음성이어도 Phase 2를 자동 기각하지 않는다**(재학습이 바로 그 불일치를 없앤다).
양성이면 강한 순풍 신호로만 쓴다.

## Open Issues

- ~~**[최우선]** 레짐 GBM 학습구간이 VAL/OOS를 삼킴~~ **해소(2026-09-09)** — 컷오프 2025-09-30
  재학습 + TRAIN 창 purged 5-fold OOF. 위 "Phase 0 진행 상태" 참조.
- ~~6컬럼 유도식이 wide24 사이드카와 동일한지 미확인~~ **해소(2026-09-09)** —
  `odyssey_regime3_live.py::_append_current`(372-392행)의 유도식은 **모델 비의존적**이고 확률
  3개만 필요하다(`confidence`=max, `margin`=top1-top2, `entropy`=-Σp·log p/log3). 클래스 순서도
  wide24/S12_K3 둘 다 `['bull','bear','chop']` 로 일치. HMM→GBM 교체가 이 축에서는 무손실.
- **[최우선·신규] 파리티 게이트 기준 재정의 — 사용자 판단 필요.** 아래 셋 중 택일:
  - (A) 게이트를 **결정 수준**으로 재정의: "깨끗한 구간에서 136피쳐 전부 ≤1e-6" + "argmax
    일치율 ≥99.9%". VAL 창은 이미 둘 다 통과, OOS 창은 데이터 아티팩트로 피쳐 기준 미달.
  - (B) **캐노니컬 2026 피쳐를 단일 패스로 재생성**해 구간 경계 아티팩트 자체를 제거. 근본
    해결이고 저장소 전체가 이득이나(2026-07 확장 OOS 재테스트가 지적한 것과 같은 결함 계열)
    비용이 크고 공유 데이터를 건드린다.
  - (C) rank 파생 3종을 레짐 모델 136피쳐에서 제외. 가장 싸지만 대시보드 배포본과 피쳐셋이
    갈라져 "config 동일, 컷오프만 변경"이라는 이 라인의 통제가 깨진다.
  - (D) **사이드카를 라이브와 같은 경로(`FeatureEngineer`)로 생성**한다. 그러면 레짐 6컬럼의
    train/inference 파리티가 구성상 정확해진다. 다만 부모의 나머지 96피쳐는 여전히 캐노니컬
    CSV 에서 오므로 그 축의 갭은 이 라인이 만든 것이 아니라 저장소 기존 조건이다.
- **[신규] 캐노니컬 2026 파일 데이터 결함** — `ou_halflife` 3.67%가 1.0 클리핑, 파일 꼬리
  (2026-08 후반)는 floor 고정. 이 라인의 OOS(2026-01~02)는 꼬리 밖이라 직접 영향은 제한적이나,
  전신 부모의 102 base_cols 와 duration gate 가 같은 컬럼을 쓴다 — 별도 조사 대상.
- 전신 ETH 부모는 재학습 비재현으로 문서화돼 있다(`omega461_eth_frozen_policy_debt_decision_20260807`).
  이 라인은 **전신을 재학습하는 게 아니라 새 라인을 세우는 것**이므로 그 제약에 걸리지 않지만,
  같은 비재현 함정(피쳐 자동탐지 102→172 오염)을 피하려면 `--pin-component` 계열 wrapper가 필수다.
- 전신의 `epochs_ran=2`가 의도인지 미확인 — Phase 2 대조군에 에폭을 포함할지 결정 필요.
- 전신 duration gate는 OOS 과적합으로 서버에서 이미 OFF다. 이 라인의 기본값도 OFF로 시작한다.
