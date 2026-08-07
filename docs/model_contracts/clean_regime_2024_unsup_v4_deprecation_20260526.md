# clean_regime_2024_unsup_v4 Deprecation Decision (2026-05-26)

## Scope

이 문서는 `clean_regime_2024_unsup_v4_*` prefix를 active 경로에서 계속 사용할지 재판정한 결과를 기록한다.

판정 대상:

- live runtime feature frame
- active backtest/candidate retrain feature contract
- DSAC/Router/Regime context feature inventory

## Re-analysis Findings

1. **라인리지 불일치**
   - `clean_regime_2024_unsup_v4_*`는 `ensemble/certified_teacher_regime_moe.py`의 legacy clean-regime(KMeans 5-cluster) 계열이다.
   - current canonical regime contract는 `clean_regime4_state24_sticky090_v2_*` (HMM state24 sticky 0.90) + `regime4_pred_*` (TFT future)다.
   - 즉 두 prefix는 생성 모델/의미 체계가 다르다.

2. **의미 공간 불일치**
   - legacy는 `cluster`, `cluster_prob_*`, `normal_prob`, `state_code`를 포함한 5-cluster semantic을 갖는다.
   - current contract는 4-class regime + auxiliary factors/risk scores이며 cluster semantics를 active owner 입력으로 사용하지 않는다.

3. **동일 suffix라도 값이 일치하지 않음**
   - 2025 동일 timestamp 기준 old-v4 vs state24-v2 비교 시 일부 factor는 높은 상관을 보이지만,
   - `transition_risk`, `whipsaw_prob`, `chop_prob`, `confidence`, `entropy`는 낮은 상관/큰 괴리를 보인다.
   - 따라서 단순 alias로 간주할 수 없다.

4. **프로비넌스 ambiguity**
   - `clean_regime4_2024_unsup_v1_*` legacy export prefix ambiguity를 해결하기 위해 이미 state24-v2 rename contract가 도입되었다.
   - old-v4 prefix를 active path에 남기면 regime provenance가 다시 혼합되어 재현성과 해석 가능성을 망가뜨린다.

## Verdict

`clean_regime_2024_unsup_v4_*`는 **active contract bug feature**로 분류한다.

- 분류 의미: 단순 성능 저하가 아니라 **계약/프로비넌스 위반 위험**을 가진 legacy surface.
- historical reproduction/debug 용도로는 허용 가능.
- active live/backtest/model-candidate 경로에서는 금지.

## Enforcement Policy

1. Active specs must use:
   - `clean_regime4_state24_sticky090_v2_*` (current regime context)
   - `regime4_pred_*` (future regime context)

2. Active paths forbid:
   - `clean_regime_2024_unsup_v4_*`
   - `clean_regime4_2024_unsup_v1_*` (rename 전 raw export prefix)

3. Any intentional use of old-v4 requires:
   - historical reproduction label
   - explicit scope isolation
   - no promotion to live/candidate-default

## Propagation Targets Updated

- `docs/subagents/README.md`
- `docs/subagents/red_team.md`
- `docs/subagents/model_architect.md`
- `docs/subagents/agent_registry.json`

