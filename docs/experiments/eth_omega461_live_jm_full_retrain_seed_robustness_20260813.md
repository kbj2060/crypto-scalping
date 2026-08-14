# ETH Omega4.6.1 라이브 — JM 레짐 전체재학습 N=5 시드 강건성 검증 (2026-08-13)

## 배경

사용자 요청: 라이브 레짐3 분류(12-state sticky HMM)를 JM(Statistical Jump Model, k=3, λ=4)으로
전체 레이어(제한된 15피쳐가 아니라 라이브와 동일한 102개 `base_cols` 전체)를 재학습해서 비교.

기존 자산(2026-08-09/08-10, 단일 시드 260620):
- JM 레짐3 분류기: `scripts/build_eth_regime3_jm_lam4_20260809.py`
- h48qual/zig075 pinned102 JM 전체재학습 번들:
  `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime_jmlam4_20260809_{h48qual_ext,zig075}/`
- correctgate 리스크 사이드카(MDD floor -25%, notional 0.45-0.95, 실제 라이브 프로덕션 게이트):
  `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_pinned102_jmlam4_q0{70,80}_correctgate_20260810/`
- OOS(2026-01-01~02-28) 단일시드 백테스트: `tmp/eth_greedy_router_regime_jmlam4_pinned102_correctgate_20260810/result.json`
  — JM이 HMM baseline을 이김(PnL +63.61% vs +42.35%, MDD -12.08% vs -15.48%, 승률 63.6% vs 50.0%)이나
  거래수가 n=11(JM)/14(HMM)뿐이고 전부 zig075에서만 나와(h48qual 0건) 통계적으로 무의미하다고
  이미 닫힌 결과. **이 OOS 결과는 이번 세션에서 다시 열지 않는다 — "이미 한 번 읽힌 것"으로 취급.**

관련 기존 문서: `docs/experiments/eth_zig075_final15_vs_jmlam4_vs_live_comparison_20260812.md`,
`docs/experiments/eth_zig075_jmlam4_candidate_confidence_echo_calibration_check_20260812.md`,
`docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md`(VAL 구간 자체가 여러
모델·레짐분류기에서 공통으로 약한 상관을 보인다는 오늘 밤의 독립적 발견 — 아래 해석에 참고).
**08-09/08-10 JM 전체재학습 자체를 다룬 별도 실험 문서는 존재하지 않는다** — 그 시점 작업은
스크립트+`report.json`/`result.json` 아티팩트로만 남아있고(이 문서가 직접 읽고 검증), 위 08-12
문서들은 이를 다른 각도(zig075 confidence 진단, final15 비교)에서 다룰 뿐이다.

CLAUDE.md Seed-Diversity Ensemble Promotion Gate: N≥5 진짜 다양한(고정 간격 아닌 랜덤) 시드에서
OOS(이 경우 VAL) 부호 일치를 봐야 하고, 시드 리스트를 리포트에 기록해야 한다.

## 방법론

**시드**: `random.SystemRandom().sample(range(1, 1_000_000_000), 5)`(OS 엔트로피, 고정증분 아님)로
생성 — 오늘 밤 `eth_omega461_live_sltp_wide_calibration_seed_robustness_20260813.md`가 확립한 동일
방식(그 문서는 `random.sample`을 OS 엔트로피 기반으로 썼다고 명시). 생성된 5개:
**323033734, 50011403, 504028524, 782182142, 393423992**. h48qual·zig075 양쪽에 동일하게 적용
(사이드카 자체의 내부 시드는 원본 레시피의 고정값 260622 그대로 — 이번 강건성 질문의 대상은
parent TabM 모델의 학습 시드이지 사이드카의 GBM 적합 시드가 아니므로, 사이드카 시드까지 같이
바꾸면 오히려 "정확히 한 가지만 바꾼다"는 원칙에서 벗어난다).

**재사용 레시피(변경 안 함, 시드만 다양화)**: 기존 08-09 단일시드 번들과 동일한 라벨·피쳐·
quality_threshold 스윕·epoch·행수 설정을 그대로 사용했다 — 새 실험을 설계하지 않았다.
- h48qual: `quality_mode=quality_label_action`, `quality_label_dir=.../omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps`,
  `quality_thresholds=0.70,0.75,0.80,0.85,0.90`. **주의**: 이 라벨 디렉토리는 2026-08-11에 발견된
  "corrected" h48qual 라벨 레시피(`h48orig`, `tmp/eth_h48_conservative_orig_padded_to_zigzag_timestamps_20260811`)가
  아니라 그 이전 버전이다 — `zigzag_action_labels_2025.csv`의 약 32% 행이 다르다(별도 검증, 아래
  "인프라 이슈" 절 참고). 기존 h48qual_ext 번들 자체가 이 구버전 라벨로 만들어졌으므로, "동일
  레시피로 시드만 바꾼다"는 원칙에 따라 의도적으로 구버전을 그대로 재사용했다 — 레짐3
  HMM→JM 교체 하나만 격리해서 보기 위함이며, 라벨 보정과 레짐 교체를 동시에 넣지 않는다.
- zig075: `quality_mode=same_as_direction`, `quality_thresholds=0.40~0.90`(11개).
- 공통: `epochs=2`, `max_train_rows=0`(무제한), `max_exit_samples=30000`,
  `exit_label_mode=entry_label_terminal_giveback`, `direction_label_dir=.../zigzag_action_labels_20260531`,
  `--pin-component`으로 라이브 102-`base_cols` 계약에 고정(`train_eval_omega4_3head_parent72_pinned102_20260727.py`).
- 리스크 사이드카: correctgate 게이트 그대로(`--min/max-validation-avg-notional 0.45/0.95`,
  `--max-validation-mdd-abs 25`), h48qual은 `q070`(0.70), zig075는 `q080`(0.80) — 원본 correctgate
  사이드카가 실제로 선택했던 quality_threshold와 동일.

**백테스트**: `scripts/rerun_eth_greedy_router_regime_jmlam4_pinned102_correctgate_20260810.py`(OOS,
전체 우선순위결합 라우터 — `PRIORITY=("h48qual","zig075")`와 동일하게 h48qual·zig075를 함께
`greedy_replay`에 넣는 구조)의 패턴을 VAL 구간(2025-10-01~12-31)에 맞게 새로 작성
(`scripts/rerun_eth_greedy_router_regime_jmlam4_pinned102_correctgate_VAL_5seed_20260813.py`),
`replay_omega4_6_1_greedy_val_20260706.py`의 검증된 VAL 프레임 로딩(`BASE_2025`,
`_expertdq_oof_` 프리픽스 정리)을 그대로 재사용했다. baseline은 라이브 HMM 번들의
`validation_predictions_q050/q075.csv`, JM 쪽은 5개 시드 각각의 `validation_predictions_q070/q080.csv`.
duration 게이트(`DURATION_THRESHOLD`)와 correctgate 리스크사이드카 모두 원본과 동일하게 적용.

**OOS는 건드리지 않았다** — 08-10 결과가 이미 1회 열람된 것으로 처리, 이번 세션은 VAL만 평가.

## 인프라 이슈 및 수정 (실행 전 발견)

1. **`scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py`가 HEAD에 없었다** — 두
   원본 번들 생성(08-09) *이후* 커밋 `4c46d20`("refactor: remove m7 and regime4_pred feature
   families")에서 "~190개 일회성 실험 스크립트" 일괄삭제에 우연히 휩쓸린 것으로 보인다(docstring상
   이 파일 자체는 m7/regime4_pred를 명시적으로 배제하는 목적이라 해당 리팩터와 무관함을 확인).
   `git show f662aa7:scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py`(삭제 직전
   버전, 2026-08-07 커밋, 파일 전체 이력상 유일한 버전)로 복원 후 현재 코드베이스에 대해
   import·`--pin-component` 동작 확인 완료.
2. **h48qual용 JM+pinned102 학습 wrapper가 애초에 존재하지 않았다** — zig075 wrapper의 docstring이
   가리키는 이름(`train_eval_omega4_3head_parent72_pinned102_regime_jmlam4_20260809.py`)은 커밋된
   적이 없다. 기존 h48qual_ext 번들의 `report.json`(라벨 디렉토리, quality_thresholds, epochs 등)과
   복원된 pinned102 스크립트 자체의 사용 예시 docstring을 대조해 정확히 일치하는 값으로
   `scripts/train_eval_omega4_3head_parent72_pinned102_h48qual_regime_jmlam4_20260809.py`를 새로
   작성했다(zig075 wrapper와 동일 패턴).
3. **JM 레짐3 CSV(`eth_regime3_current_hmm_jmlam4_20260809_{2024,2025,2026}_maskedname.csv`)가
   디스크에서 사라져 있었다** — `data/`·`tmp/`가 gitignore 대상이라 백업이 없고, 08-10 이후
   정리(cleanup)로 소실된 것으로 추정. 프로즌 JM 피팅(`eth_regime3_current_jm_jmlam4_20260809_2024.joblib`,
   2024만으로 fit, 시드 고정)은 남아 있어 `scripts/build_eth_regime3_jm_lam4_20260809.py`를
   그대로 재실행해 3개 연도 CSV를 재생성했다(서버, CPU, ~1분) — 결정적 시드라 재현값은
   원본과 사실상 동일할 것으로 판단.
4. **리스크 사이드카 wrapper 2개가 참조하는 "`_gapfilled_20260809`" regime3-risk 파일도 소실**
   — `data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6_gapfilled_20260809.csv`.
   이 파일을 만든 스크립트를 찾지 못했다(원본 생성 경위 불명). 대신 `omega` 모듈 자체의 기본값
   (gapfill 접미사 없는 `training_features_2026_rebuilt_regime3_stability_risk_h6.csv`, 여전히
   존재)이 이 wrapper들이 실제로 쓰는 `EVAL_CSV`(2026-01-01~02-28, 16,897행)와 완전히 일치해
   `dropped_edge_rows=0`으로 병합됨을 직접 검증(`_load_omega_frames()` 직접 호출)한 뒤, 두
   사이드카 wrapper 스크립트의 해당 override 줄을 제거하고 사유를 주석으로 남겼다(모듈 기본값으로
   자연히 폴백).
5. **서버 메모리 사고 경고 반영**: 이 서버에서 직전에 돈 다른 작업(exit head 재학습)이 메모리
   고갈로 서버 전체를 다운시킨 사고가 있었다는 오케스트레이터의 경고를 받고, 20개 작업
   (h48qual/zig075 × 5시드 × {parent, sidecar})을 **전부 순차 실행**(한 번에 정확히 1개)하도록
   드라이버 스크립트를 작성했고, 매 작업 후 `free -h`로 가용 메모리를 로그에 남기고 4GB 미만이면
   즉시 중단하는 안전장치를 넣었다(`scripts/run_jm_full_retrain_seed_robustness_20260813.sh`).
   실측 결과 이 학습은 원래 가벼웠다(전체 20개 작업 중 parent 학습은 CPU에서 개당 ~3.5분,
   sidecar는 ~10초 — RTX 3070Ti GPU조차 불필요, 원본 08-09 레시피 자체가 `--device cpu`
   기본값). 서버 가용 메모리는 실행 내내 22~24GB(전체 31GB 중)로 안정적이었다.
6. **h48qual 사이드카가 다수 시드에서 학습 자체가 실패했다**(버그 아님, 데이터 특성 — 두 가지
   하위 유형 확인). 4번 수정 이후에도 h48qual q070 사이드카가 (a) VAL 구간 게이트 통과 거래 0건
   ("empty ledger for margin replay", 시드 50011403) 또는 (b) 거래는 있으나 side-split 모델을
   피팅하기엔 부족(시드 504028524: `not enough side-split risk samples for side=-1: 10`, 즉 숏
   쪽 10건뿐) 두 가지 형태로 실패하는 시드가 나왔다. 원본 단일시드(260620) 번들의
   `report.json.results`를 다시 보면 이미 `q0p85`/`q0p90`은 VAL·OOS 둘 다 0건이고 `q0p70`도 VAL
   17건뿐이었다 — 애초에 이 게이트가 임계값에 극도로 민감한 스파스한 상태였고, 이번 발견은 그
   연장선(시드를 바꾸면 같은 q070에서도 문턱을 넘는 시드/못 넘는 시드/애매하게 걸치는 시드로
   갈린다)이다.
   **이를 "고치려" 하지 않았다** — quality_threshold를 시드별로 다르게 주면 "레짐3 교체 하나만
   격리"하는 설계 원칙이 깨진다. 대신 `rerun_..._VAL_5seed_20260813.py`가 컴포넌트별로 독립적으로
   가용성을 체크하도록 고쳐서, h48qual이 없는 시드는 zig075 단독 라우터로, 있는 시드는 둘 다 합친
   라우터로 자동 대체되게 했다(라이브 `PRIORITY=("h48qual","zig075")` 라우터가 실제로 h48qual이
   0건일 때 하던 동작과 동일). 어느 시드가 h48qual을 포함했는지는 아래 결과 표에 `components_used`로
   명시한다.
7. **더 중요한 발견: zig075(원래 게이트 통과율이 훨씬 높은 컴포넌트)도 시드 504028524에서
   correctgate 사이드카 학습 자체가 실패했다** — h48qual과 다른 실패 모드로, `grid_risk_mapping`
   단계까지는 도달했으나 `RuntimeError: no eligible validation-only risk mapping: trades >= 46,
   validation_mdd >= -25.0000, min_avg_notional=0.4500, max_avg_notional=0.9500` — 즉 -25% MDD
   floor·0.45-0.95 notional 밴드·거래수 하한(baseline의 95%)을 **동시에** 만족하는 리스크
   매핑이 그리드 전체에 하나도 없었다. 이 정확한 에러 메시지는 새로운 게 아니다 — 기존 사이드카
   wrapper의 주석이 이미 "docs42 fork" 선례로 인용하고 있던 바로 그 실패 모드다(0.45/0.95 밴드가
   모든 후보를 거부해 완화해야 했던 전례, `docs/model_contracts/sol_btc_regime_models_retrain_tuning_20260721.md`).
   즉 correctgate 게이트(실제 라이브 프로덕션 게이트) 자체가 원래 타이트해서 일부 (시드,
   컴포넌트) 조합에서는 정당하게 "적합한 사이즈 매핑 없음"이 나올 수 있다 — h48qual만의 문제가
   아니라 게이트 자체의 성질이며, N=5 시드로 이걸 실제로 잡아낸 것 자체가 이번 강건성 검증의
   목적과 정확히 부합한다. 이 역시 "고치려" 하지 않고 해당 시드의 zig075를 `components_used`에서
   제외하는 것으로 자연스럽게 처리된다(4/6번과 동일한 컴포넌트별 독립 가용성 체크).

## 결과

VAL(2025-10-01~12-31, common_bars=26,496) — `scripts/rerun_eth_greedy_router_regime_jmlam4_pinned102_correctgate_VAL_5seed_20260813.py`
실행 결과(`tmp/eth_greedy_router_regime_jmlam4_pinned102_correctgate_VAL_5seed_20260813/result.json`).
`no_gate`은 duration 게이트 미적용 전체 라우터, `with_gate`는 `DURATION_THRESHOLD` 적용 후.

| 시드 | 사용 컴포넌트 | no_gate PnL | no_gate MDD | 거래수 | 승률 | with_gate PnL | with_gate MDD | 거래수 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline(HMM, 라이브)** | h48qual+zig075 | +36.82% | -24.34% | 29 | 41.4% | +54.88% | -31.11% | 22 |
| 323033734 | zig075만 | +92.39% | -25.74% | 25 | 52.0% | +61.10% | -25.74% | 22 |
| 50011403 | zig075만 | +71.88% | -21.82% | 22 | 54.5% | +78.87% | -21.82% | 19 |
| 504028524 | **(없음)** | — | — | — | — | — | — | — |
| 782182142 | h48qual만 | -1.63% | -2.88% | 13 | 30.8% | -0.59% | -2.45% | 12 |
| 393423992 | h48qual+zig075 | +1.25% | -31.61% | 33 | 33.3% | +31.36% | -18.06% | 25 |

**baseline 대비 "PnL·MDD 둘 다 개선" 시드 카운트** (오케스트레이터가 지정한 판정 기준):
- no_gate 기준: **50011403 1건만** 둘 다 개선(PnL +71.88%>+36.82%, MDD -21.82%>-24.34%). 323033734는
  PnL만 개선(MDD는 -25.74%로 baseline -24.34%보다 오히려 악화). 782182142는 MDD만 개선(PnL은 악화).
  393423992는 둘 다 악화. 504028524는 데이터 없음. → **1/5 (유효 4개 시드 중 1/4)**.
- with_gate 기준: **323033734, 50011403 2건**이 둘 다 개선. 782182142·393423992는 PnL이 악화(MDD만
  개선). → **2/5 (유효 4개 시드 중 2/4)**.

두 기준 모두 오케스트레이터가 제시한 "4/5 이상" 문턱에 크게 못 미친다.

**컴포넌트 가용성 자체가 이미 강한 신호다**: 5개 시드 중 h48qual+zig075가 둘 다 살아있는 시드는
**393423992 단 1개**뿐이고, 그 유일한 "완전한" 비교조차 baseline보다 못하다(no_gate PnL +1.25%<+36.82%,
MDD -31.61%<-24.34%; with_gate PnL +31.36%<+54.88%). 504028524는 아예 거래 가능한 컴포넌트가 없다.
승리처럼 보이는 두 시드(323033734, 50011403)는 **둘 다 zig075 단독**이며, 이는 08-10 단일시드
OOS 결과가 100% zig075에서만 나왔던 것과 같은 패턴 — 하지만 이번엔 h48qual이 관여한 두 시드
(782182142 h48qual만, 393423992 둘 다)가 뚜렷이 더 나쁘다는 대조군이 생겼다.

## 결론

**재현 안됨(NOT reproduced).** 08-10 단일시드(260620) OOS 결과가 보여준 "JM이 HMM baseline을
이긴다"는 헤드라인은, N=5 진짜 무작위 시드로 VAL 구간에 대해 정확히 같은 레시피(라벨·피쳐·
quality_threshold·correctgate 리스크게이트)를 반복했을 때 **재현되지 않는다** — 오케스트레이터가
사전에 정한 판정 기준(4/5 이상 시드에서 PnL·MDD 둘 다 개선)을 no_gate·with_gate 어느 쪽으로 봐도
충족하지 못한다(각각 1/5, 2/5).

**세 가지 독립적인 문제가 겹쳐 있다**:
1. **성과 자체의 시드 분산이 크다** — zig075만 남는 두 시드는 baseline을 확실히 이기지만(with_gate
   기준), h48qual이 관여하는 두 시드는 확실히 진다. "JM이 이긴다"는 원래 헤드라인은 우연히
   zig075-우호적인 단일 시드(260620)를 뽑은 결과였을 가능성이 높다 — 이 세션 전체에서 반복 관찰된
   "N=1 헤드라인이 N≥5에서 사라진다" 패턴([[tabm_hp_low_signal_pattern]])의 또 다른 사례다.
2. **correctgate 리스크 사이드카 자체가 시드에 매우 취약하다** — 10개 사이드카(5시드×2컴포넌트)
   중 정확히 절반(6개, 위 인프라 이슈 6·7번)이 "empty ledger"/"side-split 샘플 부족"/"적합한
   리스크 매핑 없음"으로 학습 자체가 실패했다. 이는 h48qual만의 약점이 아니다 — 원래 게이트
   통과율이 훨씬 높은 zig075도 두 시드(504028524, 782182142)에서 동일하게 실패했다. correctgate
   게이트(실제 라이브 프로덕션 -25% MDD floor + 0.45-0.95 notional)가 JM 레짐+특정 시드 조합에서
   "이 조건을 만족하는 사이즈 매핑이 그리드에 하나도 없다"고 정당하게 판정하는 경우가 드물지
   않다는 뜻이다.
3. **완전한(h48qual+zig075 둘 다 있는) 비교가 사실상 N=1**(393423992)뿐이라 그 자체로는 아무
   결론도 못 낸다. 라이브 실제 동작(`PRIORITY=("h48qual","zig075")` 우선순위 결합)과 가장 가깝게
   비교 가능한 유일한 시드가 baseline보다 나쁘다는 것은, 표본이 작다는 점을 감안해도 "JM이
   확실히 낫다"는 주장에 전혀 힘을 실어주지 못한다.

**h48qual이 부담을 준다는 신호도 함께 나왔다** — h48qual이 관여한 두 시드(782182142 단독,
393423992 결합)가 h48qual이 없는 두 시드(323033734, 50011403 zig075 단독)보다 눈에 띄게 나쁘다.
이는 이번 JM 실험 고유의 문제라기보다, Odyssey 서브프로젝트가 이미 여러 차례 독립적으로 확인한
"h48qual의 quality_head 게이트는 구조적으로 스킬이 없고 종종 부담만 준다"는 결론([[h48qual_standalone_replay_invalid]]
및 `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`)과 일치하는
추가 증거로 읽는 게 더 정확하다.

**`eth_val_oos_regime_mismatch_investigation_20260813.md`와의 접점**: 그 문서는 VAL 구간 자체가
(HMM이든 JM이든) 여러 모델에서 공통으로 약한 상관을 보이는 특성이 있고, "VAL에서 이겼다"는
주장은 리스크사이징·threshold가 VAL에 맞춰 선택된다는 3중 선택편향 문제를 안고 있다고 경고한다.
이번 결과의 거래수(baseline 22~29건, 시드별 12~33건)는 그 경고가 지목한 "저표본 VAL 승리는
약한 증거" 범주에 정확히 들어간다 — 심지어 이번엔 baseline조차 확실히 이기지 못했다.

**권고**: 이 JM 전체재학습 라인은 이 형태로는 승격 근거가 없다. 다음 중 하나를 오케스트레이터가
판단해야 한다 — (a) 여기서 멈추고 HMM 레짐3를 유지, (b) correctgate 게이트의 시드 취약성 자체를
별도로 진단(예: 왜 리스크 매핑 그리드가 특정 시드에서 완전히 비어버리는지), (c) 완전히 새로운
미열람 구간(OOS 아님)에서 다시 확인 — 이 문서는 OOS(2026-01-01~02-28)를 건드리지 않았으므로
그 구간은 여전히 "1회 열람 완료"로 남아 있고, (c)를 택할 경우에도 OOS를 재사용하면 안 된다.

**Fresh-Forward 체크리스트**: `fresh_forward_bar_by_bar=true`(고정 VAL 구간을 5분봉 단위로
causal replay), `trade_ledgers_used_as_input=false`(저장 원장을 입력으로 쓰지 않음, 이번에
생성한 원장은 모두 이 스크립트 자신의 출력), `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`. 라이브 파일(`trading_bot_modules/omega4_6_1_live.py`,
`trading_bot.py`, `runtime_config.py`, `.env`)은 전혀 건드리지 않았다 — 순수 리서치 스크립트만
작성/실행했다.

## 부록: h48qual이 08-10 결과에서 0건이었던 이유

오케스트레이터가 "여유 있으면 확인"하라고 남긴 질문. `tmp/eth_greedy_router_regime_jmlam4_pinned102_correctgate_20260810/result.json`을
직접 다시 읽어 확인했다: **h48qual은 baseline(HMM)·JM 두 쪽 다** `source_component_counts`에
아예 등장하지 않는다(`{"zig075": 14}`, `{"zig075": 11}`) — 즉 이 문제는 JM으로 바꿔서 생긴 게
아니라 **HMM 원본 라이브 레짐에서도 이미 그 2개월(2026-01-01~02-28) 동안 h48qual이 0건**이었다.
원인은 레짐3 분류기 선택이 아니라 h48qual 게이트 자체의 구조적 희소성으로 보인다 — 같은
단일시드(260620) h48qual_ext 번들의 `report.json.results`를 보면 `q070`조차 VAL 17건/OOS 5건뿐이고
`q085`/`q090`은 VAL·OOS 전부 0건이다. Odyssey 서브프로젝트가 이미 반복 확인한 h48qual의
극단적으로 낮은 게이트 통과율(0.68~2.45%)과 정확히 같은 패턴이며, 이번 N=5 시드 실행에서
h48qual 사이드카 학습 자체가 일부 시드에서 "empty ledger"로 실패한 것(위 인프라 이슈 6번)도
같은 근본 원인의 다른 증상이다. 레짐3 분류기(HMM vs JM)와 무관하게, 2~3개월 단위 짧은 평가
구간에서는 h48qual이 종종 0건을 내는 것이 이 게이트의 정상적인(그리고 이미 여러 문서에서 지적된)
동작이라는 결론이다 — 더 깊은 조사는 별도 세션이 필요하면 진행.

## 부록 2 (같은 날 후속, 사용자 "더 파보자" 요청): 사이드카 실패 재검증 + h48qual 언더퍼폼 메커니즘 직접 검증

### 사이드카 실패 수 정정: 6/10이 아니라 5/10(+회복된 일시적 버그 1건)

서버 로그 직접 확인 결과 `zig075_sidecar_seed323033734`은 진짜 실패가 아니라 학습 도중 발생한
일시적 코드 버그(`FileNotFoundError`, 09:49~09:53 구간, 09:53~09:56 사이 래퍼 수정으로 해소)였고
백필 재시도(`jm_full_retrain_seed1_backfill_zig075_20260813`)에서 성공(VAL 35건, `risk_422`,
constraint_pass=true). **진짜 최종 실패는 5/10** — 위 "정확히 절반(6개)"는 정정.

### 실패 메커니즘이 컴포넌트별로 완전히 다르다

- **h48qual 3건**(시드 323033734/50011403/504028524): 그리드 탐색 진입 전에 죽음 — VAL 원장이
  아예 비거나(`empty ledger`) 사이드 스플릿 하드플로어 미달(`side=-1: 샘플 3~10개 < 12`). h48qual의
  기존에 확정된 만성적 게이트 희소성(통과율 0.68~2.45%)과 같은 원인 — 새로운 문제 아님.
- **zig075 2건**(시드 504028524/782182142): 원장에 43~46건의 건강한 트레이드가 있고 사이드
  스플릿도 성공하는데, correctgate의 고정 제약(notional 0.45~0.95 AND MDD≥-25%)을 동시에
  만족하는 그리드 포인트가 하나도 없음. **h48qual과 무관한 별도 문제** — correctgate 그리드가
  HMM baseline의 트레이드 분포로 캘리브레이션돼 있어서, 레짐 분류기를 바꾸면 트레이드 분포도
  바뀌는데 그리드는 안 따라간 것으로 보인다. 원한다면 좁게 고칠 수 있는 지점이지만, "JM이
  이기냐 지냐"와는 별개 축이다.

### h48qual이 JM에서 진 게 레짐 라우팅 문제인지 직접 검증 — 아니다 (moderate-high confidence)

시드782182142(h48qual 단독)·393423992(양쪽 다) 둘 다 트레이드 레벨로 baseline과 직접 대조:

- baseline과 JM 시드의 진입 bar가 **거의 완전히 겹치지 않는다**(정확 타임스탬프 일치 0건) —
  같은 신호를 다른 레짐으로 라우팅한 게 아니라 재학습 자체가 다른 모델을 만든 결과(새 시드+JM
  파생 피처). direction_head 원시 예측 자체도 baseline과 5/13 bar에서 반대 방향으로 뒤집힘.
- 레짐 라벨 불일치율은 22.1%(전체)/28.0%(baseline 게이트 통과 bar)로 상당하지만, **손실과
  상관이 없다** — 오히려 반대 방향(불일치 bar 승률 60% vs 일치 bar 15.4%, 표본 작아 비유의).
- **결정적 증거**: 유일한 "완전한" 비교(393423992)는 5건의 h48qual 트레이드 전부가 레짐
  불일치 0건인데도 순손실이다 — 레짐 라우팅으로 설명할 여지 자체가 없다.
- TP/SL 배리어 상수는 두 설정에서 동일(배제), JM 쪽 낮은 notional(~0.21x vs baseline ~1.3x)은
  사이드카 사이징 차이일 뿐 승패에 무관(배제).
- 손실/수익 트레이드 모두 꼬리 없이 좁게 클러스터(-0.75%~-1.04% 손실, +1.57%~+1.75%
  수익) — n=13/5에서 승률 30.8%/20%는 baseline 자체 승률(42.9%) 대비 딱히 튀지 않는 잡음.

**결론: h48qual이 JM에서 진 것은 레짐 분류기 메커니즘 문제가 아니라, h48qual이 원래 방향
스킬이 없다는 상태에서 새 시드로 재학습하면 나오는 통상적인 표본 잡음이다.** JM이든 HMM이든
h48qual에게 실제 판별력이 없다면 어느 재학습에서도 이런 산포가 나온다 — 이건 이 JM 실험의
고유 문제가 아니라 Odyssey 서브프로젝트의 최상위 미해결 결론(direction_head 방향 스킬 부재)의
또 다른 발현이다.

## 최종 결론 (2026-08-13, 심화 조사 완료)

이 축은 이제 실질적으로 다 팠다. 원래 인상적으로 보였던 08-09/10 단일시드 결과, N=5 재현 실패,
사이드카 실패 원인, h48qual 언더퍼폼 메커니즘까지 — 4개 독립 조사 전부가 같은 결론으로 수렴한다:
**"JM이 모델에 주입되니 성과가 좋아졌다"는 관찰은 h48qual이 거래를 거의/전혀 안 하는 구간에서
zig075 혼자 우연히 좋았던 결과였고, 레짐 분류기 자체의 실질적 우위를 반영하지 않는다.** 남은
열린 조각은 zig075의 correctgate 그리드 캘리브레이션(레짐 분류기와 무관하게 좁게 고칠 수 있는
별도 이슈)뿐이며, 이것도 "JM 채택" 결정과는 분리된 문제다. 새 근거 없이 HMM→JM 마이그레이션을
재제안하지 않는다는 규칙은 이번 심화 조사로 오히려 더 강하게 뒷받침됐다.

## 부록 3 (같은 날, JM과 분리해서 별도 요청): zig075 correctgate 그리드는 결함이 아니라 정상 작동 (moderate-high confidence)

JM 채택 여부와 무관하게, "이 고정 그리드가 실제 라이브 HMM 시스템에도 잠재적 취약점인가"를
별도로 팠다.

**제약의 성격**: `-25% MDD` floor는 이 스크립트만의 임의 값이 아니라, 레포 다른 곳(삭제된
`redteam_omega4_6_1_sol_btc_baselines_20260708.py`의 P1 프로모션 블로커, `docs/audits/
sol_v1_live_config_mdd_gate_recheck_20260720.md`가 "프로젝트의 역사적으로 적용돼온 기준"이라고
명시)에서도 반복 사용되는 실제 리스크 허용치다 — 단, ETH-correctgate 전용 선택이지 범용 상수는
아니다(형제 사이드카 스크립트들은 같은 플래그에 7~100까지 전혀 다른 기본값을 씀). notional band
(0.45~0.95)는 더 유연한 편 — 전부 거부당해서 0.0/0.0으로 완전히 풀어준 선례가 문서에 2건 있음.

**실패한 2개 시드**: `trades>=46/41`은 걸리지 않음(43~46건 정상 확보) — MDD·notional 쪽에서
막힘. report.json이 이 체크 통과 후에만 저장되는 구조라 두 실패 시드 자체의 근접치는 재학습 없이는
복구 불가(범위 밖으로 두지 않음). 통과한 3개 시드로 대리 추정하면 편안(-13.0%, 12pt 여유) 2개 +
아슬아슬(-23.49%, 1.5pt 여유) 1개 — 근소한 차이로 떨어졌을 가능성을 시사(간접 추정).

**실제 라이브 배포 사이드카의 여유 — 핵심 발견**: `runtime_config.py`가 가리키는 실제 프로덕션
zig075 사이드카(HMM, q075)는 VAL MDD -11.59%(풀리플레이 -13.07%) vs -25% 바닥 — **예산의 절반
가까이가 여유**. notional 0.675(0.45~0.95 정중앙), 트레이드 유지율 100%(28/28). h48qual도
마찬가지로 여유(-10.56%, 14.4pt). **두 실제 프로덕션 컴포넌트 다 경계선 근처가 아니라 안전하게
여유 있는 상태.**

**결론**: 이건 캘리브레이션 버그가 아니라 게이트가 제 역할을 한 것으로 읽는 게 더 정확하다 —
`-25%`는 실제로 재사용되는 진짜 안전기준이고, 오늘의 라이브 HMM 시스템은 그 기준을 여유 있게
통과하고 있다. JM의 2개 실패는 "더 나쁜 리스크 프로파일을 게이트가 정당하게 걸러낸 것"에 가깝다.
다만 메커니즘 자체(3축 동시 AND, 부분 완화 없는 all-or-nothing)는 실재하고 지금은 잠자고 있을
뿐이다 — **가벼운 상시 메모**: HMM 쪽 라이브 컴포넌트가 재학습될 때마다 이 여유폭을 재확인할 것.
