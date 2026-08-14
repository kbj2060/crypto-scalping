# ETH Omega4.6.1 앙상블 불일치(epistemic MI) → 사이징 GBM 피처 (2026-08-13, Odyssey2 #2)

## 배경

Odyssey(1) 아키텍처 설계 문서 자체가 TabM k=8 앙상블 불일치(Depeweg et al. 2018 MI 분해의
epistemic 성분)를 "L4 리스크사이징 sidecar 피처 후보로만 사용 — 하드 게이트나 재분류엔 절대
사용 안 함"이라고 명시적으로 예약해뒀다. Odyssey(1)은 이 신호를 **게이팅용 순위상관**으로만
검증했고(부정, VAL/OOS 풀링 rho=-0.039/-0.031, 전부 비유의) 실제로 사이징 피처로 넣어본 적은
없었다 — 설계 의도된 채 미실행이었던 항목.

## 방법

**재학습 불필요한 피처 추출**: h48qual/zig075 라이브 번들에서 `.mean(dim=1)` 풀링 전 k=8
멤버별 direction/quality softmax를 직접 뽑아(`diagnose_eth_h48qual_ensemble_disagreement_20260811.
predict_members` 재사용) MI 분해, `trend_ctx_epistemic_direction`/`trend_ctx_epistemic_quality`
2개 컨텍스트 피처 생성(`scripts/build_eth_ensemble_epistemic_context_features_20260813.py`).
파싱 전 재학습 없이 순수 추론.

**사이징 GBM 재학습(실제 재학습)**: `train_eval_omega4_2_risk_sidecar_20260622.py`의
`--risk-context-feature-dir`(기존에 한 번도 안 쓰인 확장점, 모든 기존 배포 사이드카는
`risk_context_feature_dir=null`)로 새 피처를 사이징 GBM(HistGradientBoostingRegressor) 입력에
추가해 재학습. 공정 비교를 위해 baseline도 동일 프레임 구성으로 fresh-predict 재현
(정확한 원본 배포값 재현 확인: h48qual VAL pnl 5.08%/mdd-10.56%, 원본 배포 사이드카 5.20%/
-10.56%와 거의 일치).

**⚠ 실험 중 발견·수정한 버그**: h48qual/zig075에 동일한 `--out-suffix`를 써서 출력
디렉터리가 충돌, zig075 결과가 h48qual 결과를 덮어썼다 — `baseline_bundle` 필드로 발견,
컴포넌트별 고유 접미사로 재실행해 해소.

## 예비 발견 — 신호 자체가 극히 작음(버그 아님, 직접 확인)

k=8 멤버별 출력을 직접 대조한 결과 멤버간 불일치가 존재하긴 하나(softmax 확률 std≈0.003)
**극히 작다** — TabM의 파라미터 공유형 효율적 앙상블(`LinearEfficientEnsemble`, 레이어당 공유
`W` + 멤버별 경량 `r`/`s`/`bias`만 다름) 구조 자체의 특성으로 보인다. Odyssey(1)의 순위상관
무신호 결과와 정합적 — 애초에 값의 분산이 거의 없으면 무엇과도 유의미한 상관을 가질 수 없다.

## 결과 — 사실상 무변화(null result)

| | baseline VAL | epistemic 추가 VAL | baseline OOS | epistemic 추가 OOS |
|---|---:|---:|---:|---:|
| h48qual PnL/MDD/거래 | +5.08% / -10.56% / 29 | +5.31% / -10.56% / 29 | +11.10% / -6.60% / 9 | +11.21% / -6.64% / 9 |
| zig075 PnL/MDD/거래 | +44.04% / -11.30% / 28 | +44.45% / -11.27% / 28 | +31.70% / -6.77% / 13 | +31.62% / -6.12% / 13 |

두 컴포넌트, VAL·OOS 전부 **사실상 동일**(PnL 변화 ±0.5%p 이내, MDD 변화 ±0.1%p 이내, 거래수
완전 동일). GBM이 새 피처를 사실상 무시하는 수준 — 극저분산 피처가 예상대로 예측에 거의
영향을 못 준 결과. 포트폴리오 레벨 리플레이는 컴포넌트 레벨에서 이미 이 정도로 평평한 결과를
확대할 이유가 없어 생략했다.

## 결론

**채택 불가, 그러나 "실패"라기보다 "이 특정 신호원 자체가 사이징 GBM에 넣을 만한 동적범위가
없다"는 정직한 null result.** Odyssey(1)이 남겨둔 미실행 설계 아이디어를 실제로 실행해본 것
자체에 의미가 있다 — 이제 "게이팅 무신호(순위상관 검증됨) + 사이징 무변화(이번 검증)"로
앙상블 불일치 신호 전체가 이 프로젝트에서 소진됐다고 취급할 수 있다. TabM 앙상블 구조를
바꾸지 않는 한(예: 멤버별 완전 독립 가중치 — 별도의 훨씬 큰 아키텍처 변경) 이 신호원 자체의
동적범위를 늘릴 방법이 없어 재시도 우선순위는 낮다.

## 미해결 / 다음 단계

- 채택 가능한 변경 0건, 라이브 파일 미변경.
- 멤버간 불일치가 왜 이렇게 작은지(TabM 논문 자체의 예상 범위인지, 이 특정 학습 레시피의
  특성인지)는 별도 질문으로 미확인.

## 준수 확인

- `git diff` 기준 라이브 파일 무변경. 앙상블 불일치 추출은 순수 추론(재학습 없음), 사이징
  GBM만 실제 재학습(N=1 fit, GBM 자체는 결정론적 HistGradientBoostingRegressor라 신경망
  시드 분산 개념이 적용 안 됨). 스크립트:
  `scripts/build_eth_ensemble_epistemic_context_features_20260813.py`,
  `train_eval_omega4_2_risk_sidecar_20260622.py`(기존 스크립트, 신규 아님). 산출물:
  `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_odyssey2_{h48qual,zig075}_{freshpredict_baseline,ensemble_epistemic}_20260813/`.
