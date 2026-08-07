# Project Operating Rules

## Codex Working Rules

- Think before coding:
  - State assumptions before making non-trivial changes.
  - If a request has multiple plausible interpretations, surface the options instead of silently choosing one.
  - If the requested approach looks unnecessarily complex or risky, say so and propose the simpler path.
  - If the task is unclear enough that implementation would be guesswork, stop and ask.
- Keep changes simple:
  - Implement only what was asked.
  - Do not add speculative features, abstractions, configurability, or compatibility layers.
  - Prefer the smallest implementation that satisfies the success criteria.
- Make surgical edits:
  - Touch only files and lines needed for the request.
  - Match existing style, even when a different style would be preferred.
  - Do not refactor, reformat, or delete adjacent code unless the request requires it.
  - Remove only unused code introduced by the current change.
- Work against verifiable goals:
  - For multi-step work, state a brief plan with the check for each step.
  - For bug fixes, prefer reproducing the failure before fixing it when practical.
  - Run the narrowest useful verification after changes and report what passed or could not be run.

## Fail-Fast Feature Contract Rule

- 호환성 유지를 위한 alias, fallback prefix, legacy compatibility layer를 새 기본 전략으로 추가하지 않는다.
- feature/state/artifact contract가 바뀌면 런타임은 즉시 실패해야 한다. 조용한 보정, 자동 호환, 묵시적 rename은 금지한다.
- 불일치가 발생하면 에러를 노출하고, 데이터/모델/코드 중 원인을 직접 수정한다.
- historical reproduction이 필요한 경우에만 별도 실험 경로에서 legacy contract를 유지한다. live path와 active candidate path에는 넣지 않는다.

## Omega Artifact Integrity Promotion Gate

- Omega/Omega4.x 모델 업그레이드, live 후보, baseline 승격은 `scripts/audit_omega_artifact_integrity_20260630.py`가 exit status 0과 `promotion_pass=true`를 반환해야 한다.
- Parent artifact는 사용 quality threshold와 정확히 일치하는 `train_predictions_qXXX.csv`, `validation_predictions_qXXX.csv`, `oos_predictions_qXXX.csv`를 포함해야 한다. `qXXX = round(quality_threshold * 100)` zero-padded 값이다.
- Risk sidecar는 report와 artifact에 `risk_model.precomputed_prediction_dir`와 `risk_model.precomputed_prediction_tag`를 기록하고, 해당 exact-threshold parent prediction artifact만 사용해야 한다.
- 저장된 trade ledger, candidate-event replay, 과거 비교 ledger는 diagnostic 전용이다. Per-bar parent prediction artifact를 대신해 promotion 근거로 쓰지 않는다.
- 정책 원문은 `docs/model_contracts/omega_artifact_integrity_policy_20260630.md`에 둔다.

## Fresh-Forward Validation/OOS/Test Rule

- Fresh-forward는 고정된 과거 validation/OOS 기간을 5분봉 bar 단위로 처음부터 끝까지 순차 진행하는 causal walk-forward 테스트를 뜻한다.
- 기본 split은 validation `2025-09-01`부터 `2025-12-31`까지, OOS `2026-01-01`부터 `2026-03-31`까지다. 날짜 경계가 바뀌면 리포트에 명시해야 한다.
- 각 bar에서는 그 시점까지 확정된 feature/state만 보고 신호를 생성한다. 이후 bar가 도착한 것처럼 한 칸씩 전진하면서 TP/SL/time-exit/PnL을 확정한다.
- 저장된 trade ledger, candidate-event ledger, parent exit timestamp, 또는 과거 원장의 entry/exit 결과를 입력으로 사용한 성과는 승격/모델 선택/test 근거로 쓰면 안 된다.
- trading live path와 동일한 causal feature availability를 써야 하며, 미래 row에서 생성된 label/decision/ledger를 현재 decision에 조인하면 안 된다.
- 저장 원장 기반 replay는 diagnostic, accounting audit, historical reproduction 전용이다. 모델 선택, 승격, live 후보 성과, baseline 성과로 주장하지 않는다.
- 리포트는 `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`를 명시해야 한다.
- 기존 저장 원장 기반 validation/OOS 숫자는 research/dev score로만 취급한다. live promotion 또는 expected live PnL 근거로 쓰지 않는다.
- 이 규칙을 어기는 평가 결과는 성능 수치와 무관하게 promotion/test 근거로 무효다.

## Futures Risk Sizing Contract

- Futures sizing must distinguish margin, leverage, and notional explicitly:
  - `notional = margin_fraction * leverage`
  - `margin_fraction = notional / leverage`
  - `PnL = price_move * notional`
- For new Omega risk-sizing experiments, prefer predicting account risk as `margin_fraction` rather than predicting `leverage` directly.
- If leverage is fixed, derive notional from margin:
  - `leverage = 3`
  - `notional = margin_fraction * 3`
- TP/SL model outputs should be interpreted as price-move targets before converting to account-PnL thresholds:
  - `take_profit = tp_price_move * notional`
  - `stop_loss = sl_price_move * notional`
- Canonical example:
  - `margin_fraction = 0.30`
  - `leverage = 3`
  - `notional = 0.90`
  - `tp_price_move = 0.04`
  - `sl_price_move = 0.015`
  - `take_profit = 0.04 * 0.90 = 0.036`
  - `stop_loss = 0.015 * 0.90 = 0.0135`
- Do not multiply TP/SL price lines by leverage again after notional is derived. That double-counts leverage because notional already includes exposure.
