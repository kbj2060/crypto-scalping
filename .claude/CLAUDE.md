CLAUDE.md
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

Omega Artifact Integrity Promotion Gate
Omega/Omega4.x 모델 업그레이드, live 후보, baseline 승격은 scripts/audit_omega_artifact_integrity_20260630.py가 exit status 0과 promotion_pass=true를 반환해야 한다.
Parent artifact는 사용 quality threshold와 정확히 일치하는 train_predictions_qXXX.csv, validation_predictions_qXXX.csv, oos_predictions_qXXX.csv를 포함해야 한다. qXXX = round(quality_threshold * 100) zero-padded 값이다.
Risk sidecar는 report와 artifact에 risk_model.precomputed_prediction_dir와 risk_model.precomputed_prediction_tag를 기록하고, 해당 exact-threshold parent prediction artifact만 사용해야 한다.
저장된 trade ledger, candidate-event replay, 과거 비교 ledger는 diagnostic 전용이다. Per-bar parent prediction artifact를 대신해 promotion 근거로 쓰지 않는다.
정책 원문은 docs/model_contracts/omega_artifact_integrity_policy_20260630.md에 둔다.

Seed-Diversity Ensemble Promotion Gate
여러 학습 시드를 평균/배깅해서 승격을 주장하는 모델(Omega4.6.1의 bull/bear/chop MoE처럼 단일 시드 또는 전문가별 오프셋 구조는 제외)은 N≥5개의 진짜 다양한 시드(고정 간격 증가가 아닌 랜덤 추출)에서 OOS 부호 일치를 보여야 한다.
시드 리스트는 프로모션 리포트에 기록해야 한다.
시드 개수가 너무 적거나 너무 클러스터되어 신호와 시드-분산 노이즈를 구분할 수 없는 프로모션 주장은, 헤드라인 지표와 무관하게 무효다.
정책 배경: 2026-08-01 Sigma3-1h 감사에서 SEEDS=[270705,270710,270715,270720,270725] (한 base값의 +5 증분)로 만든 "5-seed ensemble"이 VAL은 진짜 다양한 8-seed 재앙상블과 거의 일치(+22.99% vs +23.85%)했지만 OOS는 부호가 뒤집혔다(+24.32%→-13.57%, MDD -32.64%).

Fresh-Forward Validation/OOS/Test Rule
Fresh-forward는 고정된 과거 validation/OOS 기간을 5분봉 bar 단위로 처음부터 끝까지 순차 진행하는 causal walk-forward 테스트를 뜻한다.
기본 split은 validation 2025-09-01부터 2025-12-31까지, OOS 2026-01-01부터 2026-03-31까지다. 날짜 경계가 바뀌면 리포트에 명시해야 한다.
각 bar에서는 그 시점까지 확정된 feature/state만 보고 신호를 생성한다. 이후 bar가 도착한 것처럼 한 칸씩 전진하면서 TP/SL/time-exit/PnL을 확정한다.
저장된 trade ledger, candidate-event ledger, parent exit timestamp, 또는 과거 원장의 entry/exit 결과를 입력으로 사용한 성과는 승격/모델 선택/test 근거로 쓰면 안 된다.
trading live path와 동일한 causal feature availability를 써야 하며, 미래 row에서 생성된 label/decision/ledger를 현재 decision에 조인하면 안 된다.
저장 원장 기반 replay는 diagnostic, accounting audit, historical reproduction 전용이다. 모델 선택, 승격, live 후보 성과, baseline 성과로 주장하지 않는다.
리포트는 fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false를 명시해야 한다.
기존 저장 원장 기반 validation/OOS 숫자는 research/dev score로만 취급한다. live promotion 또는 expected live PnL 근거로 쓰지 않는다.
이 규칙을 어기는 평가 결과는 성능 수치와 무관하게 promotion/test 근거로 무효다.
Futures Risk Sizing Contract
Futures sizing must distinguish margin, leverage, and notional explicitly:notional = margin_fraction * leverage
margin_fraction = notional / leverage
PnL = price_move * notional

For new Omega risk-sizing experiments, prefer predicting account risk as margin_fraction rather than predicting leverage directly.
If leverage is fixed, derive notional from margin:leverage = 3
notional = margin_fraction * 3

TP/SL model outputs should be interpreted as price-move targets before converting to account-PnL thresholds:take_profit = tp_price_move * notional
stop_loss = sl_price_move * notional

Canonical example:margin_fraction = 0.30
leverage = 3
notional = 0.90
tp_price_move = 0.04
sl_price_move = 0.015
take_profit = 0.04 * 0.90 = 0.036
stop_loss = 0.015 * 0.90 = 0.0135

Do not multiply TP/SL price lines by leverage again after notional is derived. That double-counts leverage because notional already includes exposure.