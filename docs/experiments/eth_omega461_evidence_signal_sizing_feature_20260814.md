# ETH Omega4.6.1 증거 신호 → 사이징 GBM 피처 (2026-08-14, Odyssey2 #20)

상태: **완료, 기각(양쪽 컴포넌트 모두 부정 결과)**.

## 배경

Candidate C(하드 exit 오버레이, VAL 기각)와 exit_head 재학습 0단계 진단(VAL에서 반대 방향,
재학습 보류 권고)이 모두 닫힌 뒤, 사용자가 원 리서치 문서의 Candidate D — 증거 신호를 사이징
사이드카 GBM에 학습 피쳐로 넣는 방향으로 진행을 선택했다. Odyssey2 #2(앙상블 불일치)·
#3(오토인코더 latent)가 이미 검증한, **같은 확장점** `train_eval_omega4_2_risk_sidecar_20260622.py`의
`--risk-context-feature-dir`를 재사용 — 진입/청산 타이밍은 전혀 안 건드리고 포지션 크기만
조절하는, 이 축에서 폭발반경이 가장 작은 주입 지점.

## 방법

**피쳐 계산(재학습 전, 결과 확인 전 고정)**: `scripts/build_eth_evidence_signal_context_features_20260814.py`.
불리언 AND 트리거(Candidate C의 실패 원인) 대신, GBM이 스스로 조합을 학습하도록 **연속값
그대로** 6개 컬럼을 넣었다 — `trend_ctx_taker_delta_z`(오더플로우 불균형, 부호 있음),
`trend_ctx_p_fast`/`trend_ctx_p_slow`(오실레이터 percentile), `trend_ctx_ret3_z`(15분 수익률
z-score), `trend_ctx_liquidity_sweep_low`/`_high`(가장 대칭적인 강신호라 불리언 유지 — 두
극이 구조적으로 별개 이벤트라 하나의 부호 있는 값으로 합칠 이유가 없음). `orthogonal_combo`/
`taker_climax`는 의도적으로 제외(이미 포함된 원자 성분의 재조합이라 중복 정보, 이전 세션의
"동일 정보원 지표 결합은 도움 안 됨" 교훈과 이 서브프로젝트 TCN hpsearch의 "넓은 피쳐셋일수록
저SNR에서 더 나쁨" 패턴 반영).

**재학습**: h48qual/zig075 각각 `--risk-context-feature-dir`를 지정해 재실행. 공정 비교를 위해
#2/#3와 동일하게 `risk_feature_mode=parent_outputs`(원시 102피쳐 패널은 애초에 안 씀 — 사이징
GBM이 보는 유일한 raw-ish 정보가 이번에 추가한 6개뿐이라는 뜻, 리던던시 걱정 없음), 동일
`--baseline-bundle`/`--train-csv`/`--eval-csv`/`--direction-label-dir`/`--quality-threshold`
(h48qual 0.50, zig075 0.75)/`--exit-threshold 0.95`/`side-split-model`/`dynamic-leverage`/
`selection-objective log_risk`/`live-exposure-grid`/`min·max-validation-avg-notional 0.45/0.95`을
각 컴포넌트의 기존 `freshpredict_baseline` report.json에서 그대로 재구성(추측 대신 직접 확인).
**시행착오 1건**: 첫 실행이 `--max-validation-mdd-abs`/`--log-tail-penalty` 기본값(8.0/1.0)을
그대로 써서 "적격 매핑 0개"로 실패 — baseline report.json의 실제 채택값(25.0/0.5)을 대조
확인해 재실행, 재현됨.

## 결과

| | 기준선 VAL | 증거피쳐 VAL | 기준선 OOS | 증거피쳐 OOS |
|---|---:|---:|---:|---:|
| **h48qual** PnL/MDD/거래(롱/숏) | +5.076% / -10.556% / 29 (6/23) | **+4.916%** / -10.556% / 29 (6/23) | +11.100% / -6.599% / 9 (1/8) | **+10.803%** / -6.643% / 9 (1/8) |
| **zig075** PnL/MDD/거래(롱/숏) | +44.038% / -11.300% / 28 (7/21) | **+53.337%** / -12.123% / 28 (7/21) | +31.697% / -6.766% / 13 (1/12) | **+28.300%** / -7.368% / 13 (1/12) |

**h48qual: 사실상 무변화, 소폭 일관된 악화.** VAL/OOS 둘 다 PnL이 근소하게 나빠지고(-0.16pp/
-0.30pp), MDD는 VAL 동일·OOS 근소 악화(-0.04pp), 거래수·롱숏 구성은 완전 동일 — 방향/타이밍은
전혀 안 바뀌고 사이징(notional/leverage) 값만 아주 살짝 조정된 결과. #2(앙상블 불일치)의
null result와 같은 계열이나 부호가 일관되게 마이너스 쪽.

**zig075: 정확히 "VAL 승리 → OOS 반전" 패턴 — 이 서브프로젝트가 반복 경고해온 그 함정.** VAL
PnL만 보면 +44.04%→+53.34%(+9.30%p)로 이 세션에서 가장 커 보이는 개선처럼 보인다. 하지만
**OOS는 +31.70%→+28.30%(-3.40%p)로 오히려 악화**하고 OOS MDD도 -6.77%→-7.37%(-0.60%p)
나빠진다 — VAL MDD도 -11.30%→-12.12%로 함께 나빠져, "VAL 개선"이라는 인상 자체가 PnL 한
지표만의 착시였다. 거래수·롱숏 구성은 h48qual과 마찬가지로 완전 동일 — 이번에도 방향/타이밍이
아니라 순수 사이징 크기 조정만으로 이 정도 반전이 나왔다는 뜻이다.

## 해석

이 프로젝트 표준 다중구간 규율(단일 VAL 승리는 승격 근거 아님)을 적용하면 결론은 명확하다:
**두 컴포넌트 모두 기각.** h48qual은 개선도 악화도 아닌 잡음 수준이지만 방향이 나쁜 쪽으로
일관되고, zig075는 VAL만 보고 채택했다면 정확히 이 서브프로젝트가 4번(최종보스 v2/v3,
symmetric_scale9, 멀티슬롯 MFE게이팅) 겪었던 실패를 5번째로 재현했을 것이다. 흥미로운 부수
관찰: 두 컴포넌트 다 거래수·방향 구성이 피쳐 추가 전후 **완전히 동일**하다 — 이는 이 6개
피쳐가 진입 판단(direction/quality_head, parent_outputs 그대로 유지)에는 전혀 관여하지 않고
순수하게 사이징 그리드 선택에만 영향을 준 것이 맞다는 구조적 확인이며(설계대로 작동),
동시에 "숏/롱 비율이 애초에 안 바뀌는데도 이렇게 큰 VAL/OOS 괴리가 나온다"는 사실 자체가
표본(28~29건)이 사이징 그리드 선택에 얼마나 민감한지를 보여준다 —
`eth_val_oos_regime_mismatch_investigation_20260813.md`가 지적한 저표본 취약성의 사이징 축
재현.

## 결론

**채택 불가.** Odyssey2 #2(앙상블 불일치, null)·#3(오토인코더 latent, zig075 악화)에 이어
**세 번째로 사이징 GBM 컨텍스트 피쳐 확장이 부정 결과로 닫혔다** — 신호원이 다 달랐는데도
(모델 내부 불일치, 압축 latent, 외부 오더플로우/오실레이터 이벤트) 매번 같은 곳(무변화 또는
소표본발 VAL/OOS 불일치)에 수렴한다. 사이징 사이드카 자체가 이 표본 크기(28~29 VAL/9~13
OOS)에서 어떤 새 컨텍스트 피쳐를 추가해도 안정적으로 활용할 만한 신호대잡음비를 못 낸다는
가설이 세 번째 독립 증거로 강화된다. 이 서브프로젝트 증거 신호 주입 실험(Candidate C·D 둘 다)은
이걸로 종결.

## 미해결 / 다음 단계

- 채택 가능한 변경 0건, 라이브 파일 미변경.
- 원 리서치 문서(`eth_omega461_evidence_signal_injection_research_20260814.md`)가 제시한 나머지
  옵션은 전부 이미 닫히거나(Candidate C, exit_head 재학습) 지금 이걸로 닫힘 — 증거 신호를
  Omega4.6.1 라이브 모델에 주입하는 시도는 이 세션 기준 3전 3패.
- 표본 크기 자체(사이징 그리드 선택이 28~29건에 극도로 민감)를 별도 문제로 다룰지는 미결정 —
  이 실험 범위 밖.

## 준수 확인

- `git diff` 기준 라이브 파일 무변경. 피쳐 계산은 재학습 없음(순수 pandas). 사이징 GBM만 실제
  재학습(N=1 fit, HGB는 결정론적이라 시드 분산 개념 미적용, #2/#3와 동일 관례).
  `risk_feature_mode=parent_outputs`라 원시 102피쳐 패널은 애초에 미사용. 스크립트:
  `scripts/build_eth_evidence_signal_context_features_20260814.py`,
  `train_eval_omega4_2_risk_sidecar_20260622.py`(기존 스크립트, 신규 아님). 산출물:
  `tmp/causal_regen_20260516/eth_{h48qual,zig075}_evidence_signal_context_20260814/`,
  `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_odyssey2_{h48qual,zig075}_evidence_signal_ctx_20260814/`.
