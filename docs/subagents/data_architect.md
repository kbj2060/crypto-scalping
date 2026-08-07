# Data Architect Subagent

Status: `merged_into_model_architect`

Last updated: 2026-05-05 KST

이 역할은 [model_architect.md](model_architect.md)에 통합되었다.

앞으로 데이터 파이프라인, feature/state 계약, 신호/노이즈 필터링, train/validation/test split, label/output schema, model contract 작성은 `Model Architect`가 함께 책임진다.

## Invocation Rule

- `데이터 아키텍트 호출`이 들어오면 별도 독립 에이전트를 만들지 않고 `Model Architect` 역할로 처리한다.
- 새 모델 설계가 나오면 `Model Architect`가 `docs/model_contracts/` 계약서와 `docs/model_contracts/registry.json` 등록 필요 여부를 함께 판단한다.
- Red Team과 Implementation Maintainer는 데이터 계약 검토가 필요할 때 `Model Architect` 산출물을 기준으로 삼는다.

## Legacy Responsibilities Now Owned By Model Architect

- `docs/feature_contract_manifest.json`, `features/schema.py`, `features/registry.py` 정합성 확인
- 새 모델별 layer input, dataset split, label, output, artifact 계약 작성
- live/train feature parity, stale/missing/source health/schema_version 규칙 설계
- causal alignment, timestamp overlap, OOF/embargo, future-label 격리 점검
- micro noise, volatility regime, anomaly, tail event, source health 필터링 정책 설계
- private account state와 market state 분리

## Canonical Definition

최신 지침은 항상 [model_architect.md](model_architect.md)를 따른다.

추가 원칙:

- active/live candidate에서는 compatibility shim이나 legacy prefix alias를 유지하지 않는다.
- feature/state contract가 바뀌면 fail-fast로 드러나야 하며, 조용한 보정 대신 upstream/downstream 계약을 직접 수정한다.
- Omega 지도학습 리스크 label/output schema는 account-PnL threshold가 아니라 `tp_price_move`, `sl_price_move`, `margin_fraction`, `leverage`, `notional`의 의미를 명시적으로 구분한다. `notional = margin_fraction * leverage`이고, `take_profit`/`stop_loss`는 데이터 label head가 아니라 `price_move * notional`로 파생되는 실행 필드다.
- Regime redesign must follow `docs/model_contracts/regime3_whipsaw_risk_policy_20260529.md`: bull/bear/chop are structure classes; whipsaw is risk/veto/sizing context. Do not silently alias Regime4 fields into Regime3 fields.
