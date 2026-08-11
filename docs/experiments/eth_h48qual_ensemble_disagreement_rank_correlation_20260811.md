# ETH h48qual — TabM 앙상블 불일치(epistemic) 순위상관 진단 (2026-08-11)

## 배경

계약 미해결 이슈 4 / 연구문서 3-1번 항목, [`quality_for_action` 스칼라 대안 연구](eth_h48qual_quality_scalar_alternatives_research_20260811.md)의
candidate C. [순위상관 진단](eth_h48qual_quality_for_action_rank_correlation_20260811.md)(candidate A)과
[회귀 전환 시도](eth_h48qual_quality_head_regression_conversion_attempt_20260811.md)(candidate E)가
둘 다 부정 결과로 닫힌 뒤, "세 갈래 증거가 전부 스칼라 추출 방법이 아니라 라벨/피쳐 자체에
신호가 없다는 결론으로 수렴" — candidate C(TabM k=8 멤버간 불일치, `quality_head`의 자기 확률이
아닌 별도 신호원)가 이 실패 원인을 공유하지 않는 "사실상 남은 유일한 유력 경로"로 지목되어 있었다.

저장된 예측 CSV는 `.mean(dim=1)` 풀링된 값만 담고 있어(`scripts/train_eval_omega1_2_tabm_3head_
20260603.py:355-357`), per-member 원본 출력을 얻으려면 모델 번들이 필요했다. 기존 v1/v2 스윕은
거의 전부 번들을 저장하지 않았고(예측 CSV + report.json만 존재), 유일하게 저장된 번들은 다른
설정(epoch=4, 초기 baseline)이었다. 그래서 v2와 동일한 5개 시드(260620/481003/26611/903174/
155827)로 재학습하되 이번엔 번들을 저장하도록 했다. dev 환경엔 GPU가 없어(`torch.cuda.
is_available()` False) `scripts/ops/handoff.sh`로 서버(GPU 보유)에 위임 — 5시드 전부
11:30:52~11:38:18UTC에 완료, 번들 5개 전부 확인 후 dev로 pull.

## 방법

스크립트: `scripts/diagnose_eth_h48qual_ensemble_disagreement_20260811.py`. 학습 스크립트:
`scripts/ops/run_h48qual_ensemble_disagreement_5seed_20260811.sh`(서버 실행, `epochs=40
--max-train-rows 30000`, v2와 동일 시드).

- **`predict_members()`**: `.mean(dim=1)` 풀링 **전** `(N,k,3)` direction/quality softmax를
  직접 추출. 입력은 `frame[FINAL12]`가 아니라 `parent._base_input(frame, FINAL12)` —
  `POS_COLS`(exit용 포지션 상태 13개, 신호 생성 시점엔 전부 0)를 base_cols 뒤에 붙여야 스케일러의
  컬럼 계약(`base_cols + POS_COLS`)과 맞는다(`train_eval_omega4_3head_parent72_loose_entry_
  quality_20260620.py:1078-1079`와 동일 패턴). 최초 실행에서 이걸 빠뜨려
  `RuntimeError: 3-head TabM feature column contract mismatch`로 즉시 실패했고, 결과 생성 전에
  발견해 수정했다.
- **`route_combine()`**: `hard._route_id()`로 bull/bear/chop 라우팅 적용 — 라이브와 동일한
  레짐별 전문가 선택 로직.
- **`mi_decomposition()`**: Depeweg et al. 2018 상호정보량 분해 — `total = H[평균분포]`,
  `aleatoric = mean(H[멤버별분포])`, `epistemic = total - aleatoric`. 사이징 후보는 `epistemic`만
  (계약 문서에 이미 명시된 설계).
- **`pre_gate_decisions()` / `trades_with_signal()`**: [이슈 8 진단](eth_h48qual_quality_for_action_rank_correlation_20260811.md)과
  동일하게 `dir_action`(게이트 전, direction_head 원본 픽) 기준으로 포지션을 잡는다 —
  `quality_for_action` 게이트 자체가 이미 편향으로 확정된 상태라 게이트 통과 후만 보면 서바이버십
  왜곡이 생긴다. `omega.BASE_TEMPLATE`(TP/SL/notional, 라이브와 동일) 그대로 사용,
  `cost_mult=3.0`·`max_hold=0`·`cooldown=0` 오버라이드도 이슈 8 스크립트와 동일하게 맞춰 방법론을
  일치시켰다(직접 대조 확인).
- 각 시드·스플릿에서 진입 시점 `epistemic` vs `spearmanr(epistemic, trade_return)`. 시드별 rho가
  1차 근거, 전 시드 풀링 rho는 참고용 — 이슈 8과 동일 원칙.
- **정합성 체크**: seed=260620에서 새로 계산한 `dir_action`을 기존 v2 저장 예측 CSV의
  `dir_action` 컬럼과 비교(VAL/OOS 각각).

## 결과

정합성 체크: seed=260620 VAL/OOS 모두 **일치율 100%(1.0)** — 재구성한 파이프라인(피쳐 조립,
라우팅, forward pass)이 기존 v2 학습 결과와 동일하다는 강한 근거.

| Split | 시드별 rho 평균 | 중앙값 | 양수 시드 | 풀링 rho (n) | 풀링 p |
|---|---:|---:|---:|---:|---:|
| VAL | -0.061 | +0.003 | 3/5 | -0.039 (n=303) | 0.505 |
| OOS | -0.019 | -0.085 | 1/5 | -0.031 (n=192) | 0.671 |

시드별 상세:

| Seed | VAL n | VAL rho | VAL p | OOS n | OOS rho | OOS p |
|---:|---:|---:|---:|---:|---:|---:|
| 260620 | 61 | +0.058 | 0.655 | 41 | -0.127 | 0.428 |
| 481003 | 62 | -0.162 | 0.209 | 37 | -0.130 | 0.443 |
| 26611 | 62 | +0.018 | 0.891 | 42 | +0.288 | 0.064 |
| 903174 | 61 | +0.003 | 0.983 | 38 | -0.040 | 0.812 |
| 155827 | 57 | -0.220 | 0.100 | 34 | -0.085 | 0.634 |

개별 시드 10개(5시드×2스플릿) 중 명목 p<0.05를 넘는 것은 하나도 없다 — 가장 낮은 건
seed=26611 OOS의 p=0.064(양의 방향)이며, 이마저 다중비교 보정 전 기준으로도 비유의하다.

## 해석

**신뢰할 만한 상관이 어느 방향으로도 없다.** [이슈 8 진단](eth_h48qual_quality_for_action_rank_correlation_20260811.md)(candidate A)과
정확히 같은 패턴 — 부호가 스플릿마다(VAL 양수 3/5, OOS 양수 1/5), 시드마다 뒤집히고, 유의미한
개별 시드가 없다. 이건 candidate C를 지지했던 핵심 가설, 즉 "TabM 앙상블 불일치는
`quality_for_action`(quality_head의 자기 확률)과 완전히 다른 신호원이라 앞선 두 후보(A/E)의
실패 원인을 상속하지 않는다"를 실증적으로 기각한다. 다른 신호원인 건 맞지만, 이 신호원 역시
실현 손익과 관계가 없다.

이걸로 quality_scalar_alternatives 연구문서의 후보 A(temperature scaling)·C(앙상블 불일치)·
E(직접 회귀)가 전부 부정 결과로 닫힌다. 남은 건 B(evidential/Dirichlet)뿐인데, 이미 "0단계
결과(순위 자체가 없음)를 감안하면 evidential loss로 재학습해도 없는 신호를 만들어낼 가능성은
낮다"고 신중 취급되고 있었다 — 이번 결과가 그 우려를 한 번 더 뒷받침한다(불일치 기반 신호도
마찬가지로 실패했으므로).

## 계약 문서에 미친 영향

계약의 "남은 유력 경로는 이슈 4/3-1번 항목(TabM 앙상블 불일치를 사이징 피쳐로)" 문구가 이제
성립하지 않는다 — 그 경로도 막혔다. `quality_for_action`/`quality_head` 계열에서 사이징·게이팅
신호를 뽑아내려는 시도는 지금까지 시도된 4갈래(호라이즌 재설계, 회귀 전환, 순위상관 진단,
앙상블 불일치) 전부가 동일한 결론 — **`h48_conservative` 배리어 라벨이 현재 확보한 피쳐
(FINAL12, 201개 풀로 넓혀도)로는 실현 결과와 유의미한 관계를 갖지 않는다** — 로 수렴한다.
새로 남는 선택지는 (a) B(evidential, 문헌상 회의적, 착수 낮은 우선순위) 시도 아니면 (b)
`h48_conservative` 라벨/피쳐 조합 자체를 버리고 새 데이터소스로 가는 것 — 이 프로젝트의 다음
결정 포인트다.

## 결과 (계약 문서 반영용)

TabM k=8 앙상블 불일치(Depeweg MI 분해의 `epistemic`) vs 실현 순수익률 순위상관 진단 완료(5시드,
정합성 체크 100% 일치로 파이프라인 검증됨). VAL/OOS 어느 쪽도 신뢰할 만한 상관 없음(풀링
p=0.51/0.67, 개별 시드 10개 전부 비유의). Candidate C(앙상블 불일치를 L4 사이징 신호로)도 이걸로
닫힘 — "다른 신호원이라 앞선 실패를 상속하지 않는다"는 가설이 기각됨. `quality_for_action`/
`quality_head` 계열 스칼라·사이징 신호 시도 4갈래(호라이즌/회귀/순위상관/앙상블불일치)가 전부
부정 결과로 수렴, 남은 후보는 B(evidential, 신중 취급)뿐이거나 라벨/피쳐 자체 교체.
