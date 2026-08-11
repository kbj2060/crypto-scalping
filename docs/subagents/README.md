# Trading Bot Subagents

Last updated: 2026-07-02 KST

이 디렉터리는 실시간 코인 선물 트레이딩 봇을 개선할 때 사용할 프로젝트 전용 서브에이전트 정의다. 목표는 단순 PnL 최대화가 아니라, 재현 가능한 OOS 수익률, 제한된 MDD, 충분한 거래 빈도/유동성, 실제 선물 비용 반영을 동시에 만족하는 것이다.

모든 새 모델 설계의 데이터/상태 계약은 [../model_contracts/registry.json](../model_contracts/registry.json)을 기준으로 확인한다.

현재 live-wired Omega 기준은 [Omega5 event-risk governor](../model_contracts/omega5_event_risk_governor_20260702_contract.md)이다. 모델 ID는 `omega5_event_risk_governor_20260702`이며, `trading_bot.py`에서 `Omega5LiveAdapter`가 기본 ON으로 Omega4.6.2 source parent `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701` 신호 위에 검증된 Omega4.6.2 overlay, scheduled macro entry veto, shock notional haircut을 적용한다. source parent는 `trading_bot_modules/omega4_6_2_source_parent_live.py`의 live-native predictive adapter를 사용하며 `FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE=true`가 기본값이다. Omega1.2.1/Omega3 parent substitution은 금지한다. 이전 `omega4_6_plus_t12_nohold_risk1_20260630`은 연구/업그레이드 reference로만 유지한다.

Omega4.6.2 validation/OOS ledger replay는 historical audit 전용이다. `trading_bot_modules/omega4_6_2_runtime_adapter.py`의 `Omega462LedgerReplayAdapter`를 live/future timestamp decision provider로 쓰는 경로는 금지한다. 백테스트 담당 에이전트는 Omega5/Omega4.6.2 성능 재현을 주장하기 전에 새 source parent adapter를 `FinalGovernorRuntime.decide()` 경로로 순차 walk-forward replay해야 한다.

이전 MuZero/AZ rank-1 문서는 historical baseline으로만 유지한다: [current_top_muzero_az_stage2_azexit_2026](../model_contracts/2026-05-06_current_top_muzero_az_stage2_azexit.md).

현재 DSAC 후보 아키텍처는 `state24 HMM current regime + no-mdjd TFT future regime + Router5 CatBoost auxiliary probabilities + DSAC final action owner` 구조다. Regime/TFT/Router5는 모두 DSAC 입력 context이며, live/backtest action owner는 DSAC 하나로 유지한다. Deprecated CatBoost Major/Direction 모델이 직접 `LONG/SHORT/CASH`를 소유하는 구조는 active path에서 금지한다.

현재 Alpha6 메인 연구 후보는 [alpha6_entry_quality_exit_5bucket_main_20260522](../model_contracts/alpha6_entry_quality_exit_5bucket_main_20260522_contract.md)다. 이 후보는 CatBoost `action + quality + 5-bucket target horizon + position-aware exit` 구조이며, entry threshold `0.0034163351358086967`, exit threshold `0.35`를 사용한다. `fixed_notional=0.25`는 평가용 임시값이고, notional/SL/TP는 후속 DSAC 책임으로 남긴다.

## Subagents

| Agent | Definition | Primary ownership |
|---|---|---|
| Model Architect | [model_architect.md](model_architect.md) | 모델 구조, 학습 목표, 지도/비지도/RL 조합, 데이터/상태 계약, 신호/노이즈 필터링, 온라인 업데이트 전략 |
| Data Architect | [data_architect.md](data_architect.md) | `Model Architect`에 통합된 legacy alias |
| Red Team | [red_team.md](red_team.md) | 스트레스 테스트, 비용/슬리피지/레버리지 검증, 데이터 누수/편향 감사, 라이브 승격 차단 |
| Backtest Implementation Maintainer | [implementation_maintainer.md](implementation_maintainer.md) | Alpha3 frozen baseline 계약을 지키는 백테스트 전문가. 한 번에 하나의 레이어만 바꿔 후보를 구현/검증 |
| Docs Manager | [docs_manager.md](docs_manager.md) | `docs/` 폴더 관리자. active/live 명세, 모델 설계도, 트레이딩봇 런타임, 모듈 I/O 계약을 코드 변경과 함께 최신화 |

이 서면 역할들을 실행 권한이 있는 Claude Code 에이전트 팀(opus5 팀장 + sonnet5 팀원)으로 호출하려면 [architecture_team_workflow.md](architecture_team_workflow.md)를 따른다. 서브프로젝트 폴더(`experiments/<name>/`)와 계약 문서, 팀장 승인 워크플로가 거기 정의되어 있다.

## Shared Rules

- 호환성 유지를 위한 alias, fallback prefix, legacy compatibility layer를 active path에 추가하지 않는다.
- feature/state/artifact contract가 바뀌면 런타임은 fail-fast로 에러를 내야 한다. 묵시적 rename, 자동 보정, 조용한 fallback은 금지한다.
- historical reproduction이 아닌 active/live candidate는 legacy contract를 끌고 가지 않는다. 불일치가 나면 모델/데이터/코드를 직접 수정한다.
- Omega/Omega4.x 업그레이드와 live/baseline promotion은 [omega_artifact_integrity_policy_20260630.md](../model_contracts/omega_artifact_integrity_policy_20260630.md)를 따른다. `scripts/audit_omega_artifact_integrity_20260630.py`가 `promotion_pass=true`로 통과하기 전에는 승격하지 않는다.
- Omega parent artifact는 exact-threshold `train_predictions_qXXX.csv`, `validation_predictions_qXXX.csv`, `oos_predictions_qXXX.csv`를 보존해야 하며, risk sidecar는 `risk_model.precomputed_prediction_dir`와 `risk_model.precomputed_prediction_tag`를 report/artifact에 기록해야 한다.
- 저장된 trade ledger나 candidate-event replay는 diagnostic 전용이다. Promotion evidence는 per-bar parent prediction artifact와 fail-fast audit 결과여야 한다.
- 모든 Omega futures risk-sizing 실험은 루트 [AGENTS.md](../../AGENTS.md)의 `Futures Risk Sizing Contract` 중 `PnL = price_move * notional` 해석을 따른다.
- 신규 Omega 리스크 head/sidecar는 account-PnL threshold가 아니라 가격 변동률, `margin_fraction`, `leverage`, `notional`의 의미를 명시적으로 구분해야 한다. 기본 sizing 계약은 `notional = margin_fraction * leverage`이고, 런타임/백테스트에서 `take_profit = tp_price_move * notional`, `stop_loss = sl_price_move * notional`로 account-PnL threshold를 파생한다.
- Leverage가 고정이면 문서에 고정값을 명시하고 `notional`을 margin에서 파생한다. `price_move * notional` 뒤에 leverage를 다시 곱하면 double-count blocker다.
- 새 active/live candidate artifact가 `long_take_profit`, `short_take_profit`, `long_stop_loss`, `short_stop_loss` 같은 account-threshold risk head를 직접 학습 출력으로 쓰면 Red Team blocker다. Historical reproduction은 별도 실험 경로에서만 허용한다.
- 모든 제안은 `PnL`, `MDD`, `trades`, `trades_per_day`, `win rate`, `avg exposure`, `avg leverage`, 비용 가정을 함께 보고한다.
- 과도하게 큰 백테스트 수익률은 성과가 아니라 버그 후보로 먼저 취급한다.
- 학습/검증/평가 구간과 timestamp overlap 여부를 명시하지 않은 결과는 채택하지 않는다.
- DSAC/Router active feature specs must use `clean_regime4_state24_sticky090_v2_*` for current Regime4 state24 features. `clean_regime4_2024_unsup_v1_*` is an ambiguous legacy export prefix and is allowed only for historical reproduction.
- `clean_regime_2024_unsup_v4_*`는 active live/backtest/model-candidate 경로에서 금지한다. 이 prefix는 legacy 5-cluster clean-regime lineage(KMeans cluster semantics + old factor mix)이며 current Regime4 state24 contract와 의미 체계가 다르다. historical reproduction/debug 용도로만 허용한다.
- Docs Manager must mark any `regime_legacy` / legacy regime prefix rows as historical/reference-only and not active inputs, even if an old audit table gave them a usable-looking verdict.
- Current fixed Regime4 DSAC specs live under `tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521/` and must retain legacy prefix count 0.
- Current DSAC candidate architecture is `clean_regime4_state24_sticky090_v2_*` + `a5dir_*` -> DSAC final policy -> execution/ledger. Router5/CatBoost and Regime4 layers are auxiliary context, not action owners. `regime4_pred_*` (TFT future-regime predictor) and M7 (SevenModelEnsemble) were removed from the codebase entirely; do not reintroduce them.
- Next action-classifier/regime redesign policy is `Regime3 + Whipsaw Risk`, documented in `docs/model_contracts/regime3_whipsaw_risk_policy_20260529.md`.
  New action classifiers should use bull/bear/chop as direction/structure regime classes and move whipsaw into risk/veto/sizing context. Do not create new active action targets where `whipsaw` is an independent class.
- Regime3 active policy is `docs/active_live/regime3_policy_20260530.md`.
  `regime3_pred_*` future-class probabilities are removed from active action/direction ownership and must not drive long/short direction, primary/fallback labels, or hard future regime selection.
  Use only `regime3_stability_h6_score`, `regime3_transition_h6_risk_prob`, `regime3_transition_h6_risk_pred`, and `regime3_churn_h6_risk_score` as stability/transition-risk context for veto, size throttle, leverage reduction, and TP/SL/hold tightening.
- Funding-rate cleanliness is a mandatory promotion/audit gate. Any experiment that consumes funding-family columns or derived artifacts that may embed them must prove ETHUSDT-only backward/as-of funding provenance, or explicitly reference `docs/audits/funding_clean_retrain_rescore_20260529.md`.
- Direct CSV patching is not enough: old M7/teacher/regime/policy/Alpha6/Alpha7/Alpha8 candidate artifacts remain suspect unless their manifests or input paths point to the clean funding retrain/rescore run, or they are explicitly regenerated from clean funding inputs.
- Known stale-risk example: `tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/trade_candidates_20*_alpha6_current_tail111_exact.csv` does not match the clean `last_funding_rate` split values and must not be used as clean evidence for Alpha8 promotion.
- 라이브 반영 전에는 수수료, 슬리피지, 포지션 크기, 레버리지, same-side resize, 강제청산 근사, 데이터 지연을 Red Team이 검증해야 한다.
- 실시간 self-training은 shadow 모드에서만 시작한다. 라이브 포지션을 직접 바꾸는 온라인 업데이트는 walk-forward와 cost stress를 통과한 뒤 별도 승인한다.

## Collaboration Flow

1. Model Architect가 모델 구조와 함께 raw data, feature, M7 output, private account state의 상태 계약을 고정하고, 새 모델별 계약서를 `docs/model_contracts/`에 작성한다.
2. Red Team이 의도적으로 데이터/비용/체결/레버리지 이상 상황을 주입해 실패 조건을 찾는다.
3. Backtest Implementation Maintainer가 기준 모델, 데이터, downstream layer, execution, accounting을 먼저 동결하고, 승인된 변경면 하나만 열어 작고 검증 가능한 백테스트로 구현한다.
4. Red Team 차단 항목이 있으면 live default를 올리지 않는다. `compact` 같은 보수적 기준선으로 되돌리는 결정을 우선한다.
5. Docs Manager가 active/live 모델, 트레이딩봇 런타임, 모듈 I/O 계약 변경을 `docs/active_live/`에 반영한다.

## Current Project Anchors

- Current Omega live-wired contract: `docs/model_contracts/omega5_event_risk_governor_20260702_contract.md`
- Current Omega live promotion audit: `docs/audits/omega5_live_promotion_20260701.md`
- Current Omega live stack doc: `docs/active_live/omega5_live_stack.md`
- Current Omega research baseline contract: `docs/model_contracts/omega4_6_plus_t12_nohold_risk1_20260630_contract.md`
- Current Omega research baseline manifest: `data/ensemble/supervised/omega4_6_plus_t12_nohold_risk1_20260630/candidate_manifest.json`
- Current Omega research baseline report: `tmp/causal_regen_20260516/omega_creative_until_10am_20260630/plus_t12_diagnostic_nohold_risk1_20260630/report.json`
- Current Omega4.6 conditional upgrade candidate: `docs/model_contracts/omega4_6_1_duration_ou_halflife_risk_gate_20260630_contract.md`
- Current Omega4.6 upgrade candidate runtime contract: `tmp/causal_regen_20260516/omega4_6_1_duration_ou_halflife_risk_gate_20260630/runtime_contract.json`
- Current Omega4.6 upgrade candidate manifest: `data/ensemble/supervised/omega4_6_1_duration_ou_halflife_risk_gate_20260630/candidate_manifest.json`
- Current Omega live-wired baseline contract: `docs/model_contracts/omega5_event_risk_governor_20260702_contract.md`
- Frozen Alpha3 backtest protocol: `docs/model_contracts/alpha3_frozen_backtest_protocol_20260515.md`
- Current registry: `docs/model_contracts/registry.json`
- Active live docs: `docs/active_live/README.md`
- Current Omega live-wired report: `tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/report.json`
- Current Omega live promotion audit: `tmp/causal_regen_20260516/omega5_live_promotion_20260701/omega5_live_promotion_audit_20260701.json`
- Current Alpha6 main research candidate: `docs/model_contracts/alpha6_entry_quality_exit_5bucket_main_20260522_contract.md`
- Feature contract: `features/schema.py`, `docs/feature_contract_manifest.json`
- Base DSAC compact state: `ensemble/train_rl_dsac_agent.py`
- Controller DSAC state: `ensemble/train_rl_dsac_unified_controller.py`
- Shared execution/reward env: `ensemble/rl_continuous_common.py`
- Live governor/runtime: `trading_bot.py`
- Native 2026 backtest: `scripts/backtest_trading_bot_native_2026.py`
- Current rank-1 model contract: `docs/model_contracts/2026-05-06_current_top_muzero_az_stage2_azexit.md`
- Existing lifecycle stress/walk-forward audits: `scripts/eval_lifecycle_ai_stress.py`, `scripts/eval_lifecycle_walkforward.py`, `scripts/eval_lifecycle_guard_search.py`
