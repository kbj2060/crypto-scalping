# ETH 공유 트렁크 레짐전문가 후보 — 데이터 계약 (2026-08-16)

이 문서는 **공식 Odyssey 계보(Odyssey1~4)에 속하지 않는다** — 확정된 성과가 있을 때만 세대
번호를 올린다는 원칙(2026-08-16, 사용자 결정)에 따라, 결과가 확정되기 전까지 "Odyssey5"로
명명하지 않는다. `eth_candidate_*` 명명 규칙을 따른다.

관련 상위 문서: `docs/experiments/eth_odyssey4_layer_and_parameter_improvement_proposal_20260816.md`
(§C3, 이 제안의 가장 실질적인 신규 항목), `docs/experiments/eth_odyssey4_tabm_layer_design_review_20260816.md`
(§7, 레짐전문가 유효표본수 축소 문제 최초 지적).

**⚠️ 데이터 파이프라인 주의사항(중요, 반복 명시, 2026-08-16 갱신)**: `_prepare_frames()`(캐노니컬
스크립트 자체 함수, 실제 라이브 102 base(+13 pos)=115피처 파이프라인)는 dev/서버 양쪽에서 죽은
vsnlstm/chronos AI-context 피처 캐시 때문에 `FileNotFoundError`로 막혀있다(별도 병행 세션도
독립 확인). 이 후보의 최초 로컬 sanity check를 포함해 이 세션의 A1/B1/C1/C2 전부
`_prepare_frames_light()` 우회(`feature_cols`=185개, 진짜 라이브 102피처와 다름)로 학습/평가했다.
**같은 날 병행 세션이 `scripts/eth_odyssey4_true_feature_pipeline_20260816.py`로 진짜 라이브
102피처 계약을 별도 경로로 복구**(죽은 vsnlstm/chronos 캐시를 우회하면서도 실제 라이브 번들과
일치하는 피처 순서/개수 확보, 7개 결측 컬럼은 수식 재현으로 채움, 5/7 완전 일치·2개는 문서화된
연도 경계 콜드스타트 오차). 이 스크립트(`research_eth_odyssey4_shared_trunk_regime_experts_20260816.py`)는
이제 `--feature-pipeline {light,true}` 플래그를 지원하며 **기본값이 `true`**로 바뀌었다 — 아래
"다음 단계"의 서버 실행 명령은 진짜 102피처 파이프라인을 쓴다. 진짜 피처 기준 로컬 sanity check도
**재실행 완료**(2026-08-16, 아래 상태 표 참고) — light 파이프라인 결과는 참고용으로 남겨둔다.

## 상태

| 컴포넌트 | 상태 |
|---|---|
| 아키텍처 설계/구현 | **완료** — `scripts/research_eth_odyssey4_shared_trunk_regime_experts_20260816.py` |
| 독립 리뷰(다른 에이전트, read-only) | **완료, 버그 없음** — tensor shape·prefix/rename 로직·gce 잔재·encode() 동일성·기타 5개 항목 전부 확인, 사소한 스타일 지적 1건(파라미터 카운트 재계산 중복, 무해)만 있었음 |
| 로컬 sanity check — light 185피처(`--epochs 2 --n-seeds 1 --mode both --device cpu --feature-pipeline light`) | **완료, 통과** — 143초, baseline 3모델+shared 1모델 학습·VAL/OOS 백테스트·리포트 저장까지 크래시 없이 전부 성공. PnL 수치 자체는 2-epoch under-trained라 의미 없음(참고용), 목적은 오직 "안 깨지는가" 확인 |
| 로컬 sanity check — true 102(+13pos)=115피처(`--epochs 2 --n-seeds 1 --mode both --device cpu --feature-pipeline true`, 기본값) | **완료, 통과**(2026-08-16) — 226초, baseline 3모델+shared 1모델 학습·VAL/OOS 백테스트·리포트 저장까지 크래시 없이 전부 성공. 진짜 라이브 피처 계약(115차원)으로도 파이프라인 통합에 문제 없음을 확인 — 서버 본실험 착수 전 필수 확인 완료 |
| 서버 N≥5시드 본실험 | **완료(2026-08-17)** — `eth_nhits_moderntcn_direction_quality`를 사용자 판단으로 중지+삭제해 GPU 확보, 워처가 자동 감지해 본실험 실행(2026-08-17T04:37:11Z 시작, 13:49 KST 완료), 5개 진짜무작위시드, 크래시 없음 |
| 판정 | **CLOSED, 반영 안 함** — direction_balanced_accuracy(3개 전문가 전부 std/mean 비율 1.22~78배, 부호불일치) + PnL/MDD(VAL 1/5·OOS 2/5 시드만 개선, 평균 무개선) 양쪽 다 신뢰 가능한 신호 없음. 전체 내용: `docs/experiments/eth_candidate_shared_trunk_regime_experts_n5seed_result_20260817.md` |

### 로컬 sanity check 실측 — light 185피처 (참고용, 2-epoch under-trained 수치)

시드 396997759, epochs=2(본실험은 28):

| 지표 | baseline(독립 트렁크 3개) | shared(공유 트렁크) |
|---|---:|---:|
| 총 파라미터 수 | 355,656 | **121,640**(34%) |
| VAL pnl / mdd / trades | −77.41% / −77.44% / 6777 | −67.10% / −67.15% / 5442 |
| OOS pnl / mdd / trades | −62.32% / −62.33% / 4082 | −50.28% / −50.28% / 3029 |

### 로컬 sanity check 실측 — true 115피처 (참고용, 2-epoch under-trained 수치, 2026-08-16)

시드 208012890, epochs=2(본실험은 28):

| 지표 | baseline(독립 트렁크 3개) | shared(공유 트렁크) |
|---|---:|---:|
| 총 파라미터 수 | 311,976 | **107,080**(34%) |
| VAL pnl / mdd / trades | −72.95% / −72.95% / 5961 | −72.05% / −72.10% / 5852 |
| OOS pnl / mdd / trades | −51.47% / −51.48% / 3482 | −51.84% / −51.84% / 3168 |

두 파이프라인 실행 모두 파라미터 수가 설계대로 약 1/3로 줄었다(정확히 공유 트렁크 하나 + 3세트
헤드만큼 — true 파이프라인은 입력 차원이 light보다 작아서(115 < 185, `in_proj` 크기 차이)
baseline/shared 절대 파라미터 수 자체는 light보다 작다, 비율(34%)은 두 파이프라인에서 동일).
PnL/MDD
숫자 자체는 2-epoch에서 나온 것이라 어느 쪽이 더 나은지 판단 근거로 쓰지 않는다 — 여기선
오직 "백테스트 파이프라인이 두 아키텍처·두 피처파이프라인 전부에 대해 크래시 없이 끝까지
도는가"만 확인했고, 전부 통과했다.

## 배경 — 왜 이 후보가 나왔는가

라이브 `ThreeHeadTabM`(`scripts/train_eval_omega1_2_tabm_3head_20260603.py`, h48qual/zig075가
공유하는 정확히 같은 구조)은 bull/bear/chop 3개 레짐전문가마다 **완전히 독립적인 트렁크**
(`in_proj`+`blocks`+`norms`, 즉 `encode()` 전체)를 처음부터 학습한다
(`main()`의 `for idx, expert in enumerate(hard.EXPERT_NAMES)` 루프, 매 반복마다
`_fit_expert_3head`가 새 `ThreeHeadTabM` 인스턴스를 만들어 학습).

B1 저비용진단(`scripts/diagnose_odyssey4_expert_effective_sample_size_20260816.py`)으로 실측한
결과, 3개 전문가 모두 원시 행 수는 동일(78,568행)하지만 `route_w`(Regime3 HMM 소프트확률)
가중 유효표본수는 훨씬 작다:

| 전문가 | route_w.sum() (유효표본) | len(route_w) (원시행수) | 유효비율 |
|---|---:|---:|---:|
| bull | 22,472.75 | 78,568 | 28.60% |
| bear | 22,018.96 | 78,568 | 28.03% |
| chop | 34,076.28 | 78,568 | 43.37% |

즉 각 트렁크는 **자기 트렁크 전체 파라미터를 갖고도, 실질 학습신호는 원시 데이터의 3분의 1
이하만** 받는다 — 트렁크 파라미터는 3배인데 트렁크당 유효 데이터/파라미터 비율은 통합모델보다
나쁘다. 이 후보는 "트렁크 용량은 그대로 두고 그게 보는 유효 데이터를 3배로 늘리는" 독립적인
데이터효율 논거로 성립한다.

**정정(2026-08-16, 병행 세션)**: 최초 이 문서 초안은 이 후보를 "이미 CLOSED된 R+S+B 완성형
실험(파라미터+6.5%로 N=5시드 전부 악화, `eth_candidate_faithful_tabm_batchensemble`)과 정확히
반대 방향 — R+S+B는 용량-데이터 경계에 있는 모델에 용량을 더 얹은 실패"라고 근거를 댔다. 이후
그 축을 병행 세션이 N≥5시드로 재확인한 결과, 실제 메커니즘은 memorization/capacity 문제가
**아니었다** — baseline_R_only와 full_R_S_B_embed의 **진짜 정점 정확도는 0.003~0.009 차이로
거의 동일**했고, 실전 격차(+0.042~+0.058)는 임베딩 아키텍처의 학습 초반 val_loss 궤적이 더
노이즈가 많아서 조기종료(epoch 1)가 훨씬 자주(~80% vs ~20%) 발동하는 **학습 신뢰성 격차**였다
(`feedback_modern_dl_training_checklist` 메모리, 2026-08-16 최종 갱신). 그래서 이 후보를
"용량-데이터 경계 실패의 반대 방향"으로 프레이밍하지 않는다 — 그 프레임의 근거 자체가 사라졌다.
공유 트렁크 동기(3개 독립 트렁크가 각각 route_w로 희석된 데이터만 보는 것보다, 공유 파라미터가
더 많은 유효 학습신호를 보는 게 나을 것)는 위 B1 수치 자체로 독립적으로 성립하며, 이 정정은
그 동기 자체를 무효화하지 않는다.

## 설계 — 선택한 방식과 이유

**선택**: route_w 소프트라우팅 가중치로 결합한 다중헤드 손실을 하나의 공유 트렁크 위에서
학습(soft-routing-weighted multi-head loss over one shared trunk). 레짐 확률벡터를 헤드
입력에 concat하는 방식(대안)은 채택하지 않음.

이유:
- `route_w`는 이미 지금 코드에서 정확히 이 방식으로 쓰이고 있다 — `_fit_expert_3head`가
  `dir_w = balanced_weight * route_w[:, expert_idx]`로 손실에 가중치를 주는 방식을, 3개의
  분리된 학습 호출 안에서 반복하고 있을 뿐이다. 가장 자연스러운 단일모델 일반화는 이 정확히
  같은 가중치 메커니즘을 유지하면서, `encode(x)`를 배치당 **한 번만** 계산하고 그 결과에 3개
  레짐의 헤드셋을 전부 적용해서 하나의 결합손실(레짐별 합산)로 트렁크를 역전파하는 것 —
  route_w의 의미나 계산 방식은 전혀 바뀌지 않고, 손실이 "어디로 떨어지는가"만 바뀐다(트렁크
  3개 → 1개).
- 레짐 임베딩 조건부 헤드 대안은 채택하지 않았다: (1) 이미 라이브에서 검증된 라우팅 메커니즘
  (Regime3 하드 argmax `route_id`)을 재사용하지 않고 새 라우팅 개념을 만드는 셈이 되고 —
  Odyssey3/4의 CONFIRMED 결과(`eth_odyssey3_zig075_short_entry_veto_uptrend_confirmed`,
  `eth_odyssey4_zig075_long_entry_veto_downtrend_confirmed`)가 전부 이 하드 argmax 라우팅
  위에 서 있다. (2) 기존 `_routed()`(추론/백테스트 코드, 이 저장소 전역에서 재사용됨)는 이미
  "바마다 정확히 한 헤드셋이 유효하다"는 하드 선택 시맨틱을 전제하는데, soft-weighted-loss
  설계는 학습 시점에는 route_w로 부드럽게 가중치를 주면서도 추론/실거래 시점에는 `_routed()`의
  하드 선택을 그대로 유지해 시맨틱이 어긋나지 않는다.

**registry 확인**: `btc_zigzag_as_entry_model_component`(트렁크 공유가 아니라 지그재그를
별도 헤드/피쳐/라우터로 쓰는 BTC 엔트리모델 이슈, 전부 악화로 닫힘)와 질문이 다르다 — 이건
ETH 3-전문가 구조 자체를 다시 묻는 새 질문이라 겹치지 않는다.

## 구현

- 신규 클래스 `SharedTrunkThreeHeadTabM`: `encode()`는 `ThreeHeadTabM`과 완전히 동일(`in_proj`/
  `blocks`/`norms`/BatchEnsemble k=8 게이트, 미변경) — `direction_heads`/`quality_heads`/
  `exit_heads` 각각 3세트(레짐별)로 분리.
- 학습(`_fit_shared_trunk`): 배치당 `encode()` 1회 → 3개 레짐 헤드셋에 각각 적용 → 레짐별
  route_w 가중 plain CE(A1이 시도했던 GCE는 N=5시드 검증에서 이 캐노니컬 스크립트로 전이 안 돼
  되돌려짐 — `docs/experiments/eth_odyssey4_gce_canonical_port_20260816.md` — 그래서 이 후보도
  실제 캐노니컬과 동일한 plain CE를 씀)를 합산한 결합손실로
  1회 역전파. exit_head는 A1과 동일 범위 유지(plain CE). split/CFG/patience 등은 캐노니컬과
  동일(embargo gap 없음 — C1과 독립적으로, "현재 독립트렁크 baseline"과 순수 비교하기 위해).
- 평가: 캐노니컬 스크립트의 기존 백테스트 코드를 **변경 없이 재사용** — `canon._routed`(딕셔너리
  기반이라 예측이 3개 독립모델에서 왔는지 1개 공유모델에서 왔는지 구분 안 함),
  `canon._prediction_output`, `canon._to_decisions`, `canon._metrics_with_shared_exit`,
  `canon._predict_loaded_exit`. `_RegimeView` 래퍼(공유모델+레짐idx를 감싸서
  `forward(x)->{"direction","quality","exit"}` 반환)가 `ThreeHeadTabM`을 duck-type으로
  대체해서, exit_head 백테스트 루프를 한 줄도 다시 짜지 않고 그대로 재사용했다.
- baseline(독립 트렁크)은 캐노니컬 `_fit_expert_3head`를 그대로 3회 호출(= `main()`의 기존
  루프와 동일) — 단, `direction_balanced_accuracy`는 그 함수가 원래 계산하지 않으므로 학습 후
  같은 내부 val split에서 사후 계산(`_direction_val_bacc`)해서 비교 지표를 맞췄다.

## 데이터 계약

- 프레임 준비: `_prepare_frames_light()`(A1/B1/C1/C2와 공유하는 우회 헬퍼, vsnlstm/chronos
  죽은 체인 우회 — `feature_cols`=185개, 실제 라이브 배포 번들의 102개와 다름. 이 후보는
  "트렁크 공유가 아키텍처 지표를 움직이는가"만 보며, 피처 완전 동일성은 범위 밖).
  자세한 내용은 `scripts/diagnose_odyssey4_expert_effective_sample_size_20260816.py`의
  모듈 docstring 참고.
- exit 데이터셋: `max_samples=60000`(원래 무제한) — dev 머신 15GB RAM에서 무제한 빌드가
  13~14GB를 써서 동시 세션과 충돌해 OOM 한 차례 발생, 캡 적용 후 5GB 수준으로 감소 확인.
  exit_head는 두 아키텍처 다 plain CE라 이 캡이 비교의 공정성을 해치지 않는다.
- Split: `SPLIT_TS`(2025-10-01) 기준 train_raw/val_raw, `omega`의 2026 프레임이 oos_raw.
  내부 85/15 direction/quality split은 C1의 embargo gap 없이 캐노니컬과 동일(baseline과
  순수 트렁크-공유 여부만 비교하기 위함).
- N≥5 시드 게이트: **적용 대상** — 스크립트 자체에 `--seeds`(comma) 또는 `--n-seeds`(기본 5,
  `secrets.randbelow` 추출, 고정간격 클러스터 아님) 루프가 내장돼 있다.

## 비교 지표

1. `direction_balanced_accuracy` — 레짐별 내부 val split 기준(A1/C1과 동일 방법론), baseline
   3개 독립모델 vs 공유트렁크 3개 헤드셋.
2. PnL/MDD — `val_raw`/`oos_raw`에 대해 캐노니컬 백테스트 파이프라인(`_metrics_with_shared_exit`)
   그대로, exit threshold=0.60(캐노니컬 스윕 그리드의 대표값) 고정.

## Red Team Gates

- [x] 아키텍처 자체 축이라 cheap_gate로 거를 무료신호 프록시가 없음 — 제안 문서가 이미 이렇게
  명시, 바로 N≥5시드 본실험으로 진행.
- [x] 로컬 sanity check(few epoch, shape/crash) 통과 — **완료**(2026-08-16, epochs=2/n-seeds=1,
  light 185피처 143초 + true 115피처 226초, 둘 다 크래시 없이 끝까지 실행, 결과는 상태 표 참고).
  서버 본실험은 진짜 피처 파이프라인이 기본값이므로 이 재확인이 착수 전 필수 전제였다.
- [x] N≥5 시드로 재현(진짜 무작위 시드, 고정간격 클러스터 아님) — **완료(2026-08-17)**. GPU
  대기 워처가 `eth_nhits_moderntcn_direction_quality` 중지 직후 자동 감지해 실행,
  `[284220531, 396212443, 418156144, 662991329, 900276690]`(secrets.randbelow) 5시드 전부
  크래시 없이 완주.
- [x] direction_balanced_accuracy 개선이 시드 전반에서 부호 일관적인지 확인 — **불일치로 실패**.
  bull/bear/chop 전부 표준편차가 평균 델타와 같거나 훨씬 큼(1.22~78배), 5시드 중 1~2개는
  반대 부호. 상세: `docs/experiments/eth_candidate_shared_trunk_regime_experts_n5seed_result_20260817.md`.
- [x] PnL/MDD가 방향까지 일치하는지 확인(분류지표만으로 승격 근거 삼지 않음) — **불일치로 실패**.
  VAL은 shared가 평균 더 나쁨(1/5 시드만 개선), OOS는 평균 차이가 사실상 0(2/5 시드만 개선).
- [x] fresh-forward 규칙 준수 — val_raw/oos_raw 백테스트는 `_metrics_with_shared_exit`의 기존
  bar-by-bar causal walk 그대로(`fresh_forward_bar_by_bar=true`), 저장 렛저·미래 row 미사용.

## 리스크 (제안 문서에서 이미 명시)

레짐별로 실제로 근본적으로 다른 함수가 필요하다면(트렁크가 표현할 수 없는 레짐별 차이가
존재한다면), 트렁크 공유가 그 차이를 뭉갤 수 있다 — 이건 실증으로만 판단 가능. 헤드만
분리하고 트렁크를 공유하는 이 설계는, 실패하더라도 "레짐별로 얼마나 다른 표현이 필요한가"에
대한 실증적 답을 준다는 점에서 부정 결과도 유용한 정보다.

## 다음 단계

1. ~~로컬 CPU에서 few-epoch sanity check~~ — **완료**(2026-08-16, 통과).
2. **다음 세션에서**: `bash scripts/ops/handoff.sh status server eth_nhits_moderntcn_direction_quality`로
   GPU 상태 재확인 → RUNNING 아니면 위 Red Team Gates 절의 정확한 명령으로 서버에 N≥5시드
   본실험 실행 → `handoff.sh logs server eth_odyssey4_shared_trunk -f`로 실제 epoch 로그가
   찍히는지 확인(단순 PID 생존이 아니라 진짜 학습 중인지).
3. 결과는 이 문서(계약, 상태 요약만)와 별도 `docs/experiments/eth_odyssey4_shared_trunk_regime_experts_<date>.md`
   (전체 실험 과정)에 기록.
