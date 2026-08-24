# zigzag/h48qual/cusum 3라벨 — 일리아스 계약 유일 split 규약 재검증 (2026-08-22)

## 배경

2026-08-22, 일리아스 프로젝트 데이터 split 규약을 "분기 앵커드(확장) Walk-Forward" 하나로
통일(계약서 `ilias_eth_human_direction_risk_management_contract_20260817.md` "## Dataset
Split" 절 참고)하면서, 사용자가 "그럼 레짐 분류기와 3라벨 단일 컴포넌트도 새 컨벤션으로
재학습·재검증해야 하는 거 아니야?"라고 질문 → 레짐 분류기는 이미 이 방법론으로 N=5
CONFIRMED 완료돼 재학습 불필요, 3라벨(zigzag/h48qual/cusum)은 기존 chance-level 결론
([[eth_dc_engineered154_feature_set_20260820]], [[eth_label_fusion_combined_model_research_20260821]])이
전부 구식 split(TRAIN=2025/EVAL=2026H1)에서 나왔음을 확인 → 사용자 지시로 재학습·재검증 진행.

## Split

일리아스 계약 유일 규약(2026-08-22 확정) 그대로:
- **TRAIN(fit)**: 2024-01-01 ~ 2026-03-31 (262,609행 중 `parent.SPLIT_TS=2026-04-01` 이전)
- **VAL(판정 티어)**: 2026-04-01 ~ 2026-06-30 (2026 Q2) — 파이프라인 내부에서
  `train_all[ts >= SPLIT_TS]`로 자동 분리
- **OOS(단일터치, 사용자 override로 09-30 대기 없이 즉시실행)**: 2026-07-01 ~ 2026-08-19
  (데이터상 최근일, 14,400행)

레짐 오버레이는 2026-08-21 재확정된 최신 pick(states=24/sticky=0.90)을 사용
(`tmp/ilias_labellogic_recheck_20260821/train_2024_2026H1_...` +
`oos_20260701_20260819_...` concat).

## 파이프라인 구현 중 발견한 이슈 2건 (전부 로컬 오버라이드로 해결, 공유 파일 무수정)

1. **`parent.SPLIT_TS` 하드코딩**(`train_eval_omega1_2_tabm_3head_20260603.py:33`,
   `2025-10-01`): `_prepare_frames()`가 TRAIN_CSV를 이 날짜로 자동으로 fit/validation
   분리하고 EVAL_CSV 전체를 "oos"라 부른다는 걸 처음 몰랐다 — 단순히 TRAIN_CSV/EVAL_CSV
   경로만 계약서 날짜로 바꾸면 "validation"이 실제론 옛 경계(2025-10~2026-03)가 되고
   "oos"가 VAL+OOS 병합이 되는 조용한 오염이 생겼다(1차 테스트에서 실제로 확인). 해결:
   TRAIN_CSV=2024-01~2026-06-30(fit+VAL-Q2 통째로), `parent.SPLIT_TS=2026-04-01`로 로컬
   오버라이드, EVAL_CSV=2026-07-01~순수 OOS만.
2. **라벨 파일 연도 하드코딩**(`_read_labels(dir, 2025)`/`_read_labels(dir, 2026)`): 기존
   "2024merged" 라벨 디렉토리는 "2025.csv"라는 이름 아래 2024+2025만 병합해뒀다 — TRAIN이
   2026 Q2까지 늘어나며 그 구간 라벨이 `_align()`에서 조용히 사라져 val_raw가 텅 비는
   크래시("need at least one array to concatenate")로 이어졌다. 해결: 2026-01~06-30 라벨을
   추가로 병합한 `direction_labels_2024_2026q2merged`/`quality_labels_2024_2026q2merged`
   신규 디렉토리 생성(원본 무수정).

스크립트: `scripts/eth_directional_change_tabm_training_ilias_anchored_canonical_20260822.py`
(로컬 오버라이드 래퍼), `scripts/eth_tabm_label_logic_5way_seed_variant_ilias_anchored_canonical_20260822.py`
(3라벨×N=6시드 러너, 기존 `_ilias_anchored_20260821.py` 러너와 동일 로직).

## 결과 (N=6 시드, 각 라벨의 `ranking_by_validation_pnl[0]` 기준)

VAL(2026Q2) always-long 벤치마크: **-23.34%**(하락장). OOS(2026-07~08-19) always-long
벤치마크: **+21.51%**(강한 상승장) — 두 창이 정반대 레짐이라 레짐의존성 스트레스테스트로 유효.

| 라벨 | VAL pnl 평균±표준편차 | VAL 양성 | OOS pnl 평균±표준편차 | OOS 양성 | VAL·OOS 부호일치 |
|---|---|---:|---|---:|---:|
| zigzag | +3.24% ± 7.47 | 5/6 | -0.38% ± 10.27 | 3/6 | 2/6 |
| h48qual | -2.50% ± 8.29 | 2/6 | +3.89% ± 4.60 | 4/6 | 2/6 |
| cusum | +0.46% ± 6.84 | 4/6 | -1.14% ± 8.24 | 2/6 | 4/6 |

시드별 상세는 `/tmp/label3_aggregate.json`(서버, 스크래치) 참고 — 3라벨×6시드=18회
전체 `ranking_by_validation_pnl[0]` 원자료.

## 추가 검증 — 학습기법 감사 + epoch 어블레이션 (2026-08-22, 같은 날 후속)

사용자 지적("TabM도 DL이니 학습테크닉 다 적용됐는지 확인")으로 위 결과가 쓴 러너
(`eth_tabm_label_logic_5way_seed_variant_ilias_anchored_canonical_20260822.py`)의 학습기법을
감사: AdamW/순수CE/balanced 클래스가중치/gradient clipping(norm=2.0)은 정상 적용, **단
`--epochs 2`가 하드코딩**(원본 `eth_dc_engineered154_training_runner_20260820.py`부터 상속)돼
있어 `CFG.patience=8` 조기종료가 구조적으로 발동 불가능함을 확인(2<8). LR스케줄 자체는 원래
없음(고정 lr=2e-3) — 다만 기존 학습률 격리 실험(`eth_odyssey4_dl_reference_deep_analysis_
20260816.md` §2.5)이 **다른 데이터·다른 split**(2026-08-16 당시 라이브 설정)에서 "lr=2e-3은
best checkpoint가 항상 1에폭째"를 발견해뒀던 게 있어 참고했으나, 사용자가 "그때와 데이터·환경이
다르다"고 정확히 지적해 이 154피쳐/새 split 자체로 직접 재검증했다.

**epoch=2 vs epoch=12(패치언스=8이 실제로 발동 가능한 예산) 직접 비교**(env var `TABM_EPOCHS`로
비침습 오버라이드, 공유 파일 무수정): epochs=12에서 `epochs_ran=9~10`으로 조기종료가 실제로
작동함을 확인(2에폭 강제종료가 아님). 라벨별 VAL↔OOS 부호일치:

| 라벨 | epochs=2 | epochs=12 |
|---|---:|---:|
| zigzag | 2/6 | 4/6 |
| h48qual | 2/6 | 1/6 |
| cusum | 4/6 | 3/6 |
| **합계** | **8/18** | **8/18** |

개별 (라벨,시드) 값은 여러 개 부호가 뒤집히지만(12개 중 5개), **전체 합계는 8/18로 완전히
동일** — 이건 "진짜 학습효과"가 아니라 순수 노이즈 재배치라는 신호다(진짜 신호였다면 총합도
같이 개선됐어야 함). h48qual의 OOS 6/6 양성(epochs=12)도 VAL은 1/6뿐이라 방향예측력이 아니라
롱/숏 편향이 OOS 상승장과 우연히 맞아떨어진 것으로 해석(이 저장소가 이미 반복확인한
long_frac↔PnL 혼입 패턴, [[eth_tabm_label_logic_retest_initiative_20260819]] 참고).

**판정**: epoch 깊이(=학습기법 충분성)는 이 결론의 원인이 아니다 — 154피쳐 데이터셋 자체로
직접 검증했고([[eth_odyssey4_dl_reference_deep_analysis_20260816]] §2.5의 원격 유추가 아님),
결과가 확증됐다.

## 판정

**기존 chance-level/BCE-절편하한 결론이 새 split에서도 재현됨 — 재오픈 근거 없음.**

- 어느 라벨도 VAL↔OOS 부호가 안정적으로 일치하지 않는다(4/6이 최고, zigzag/h48qual은
  동전던지기 수준인 2/6). N=6 표본에서 4/6는 이항검정으로도 유의하지 않다(p≈0.34).
- zigzag는 VAL에서 5/6 양성으로 보이지만 이는 VAL 자체가 -23% 하락장이라 "하락장에서
  숏 편향"이 우연히 맞아떨어졌을 가능성이 높고(이 저장소가 반복 확인한 long_frac↔PnL
  혼입 패턴, [[eth_tabm_label_logic_retest_initiative_20260819]] 참고), OOS(상승장)에서
  바로 3/6으로 무너진다 — 레짐 의존적 우연 그 이상의 근거가 없다.
- h48qual은 VAL·OOS가 통째로 반대 부호 경향(VAL 대부분 음수, OOS 대부분 양수)이라 방향
  자체보다 레짐 자체를 따라가는 패턴에 가깝다.
- 정보이론적 근거(BCE=절편전용 이론하한, [[eth_label_fusion_combined_model_research_20260821]])는
  split 창과 무관하게 성립하는 성질이라 애초에 결론이 뒤집힐 것으로 기대하지 않았고,
  실측도 그 기대와 일치했다.

**결론: 3라벨 단일 컴포넌트 축은 새 split 규약 하에서도 CLOSED 상태를 유지한다.**
이 축을 다시 열려면 154피쳐 자체를 대체할 질적으로 다른 정보원(raw LOB, 청산 tail-risk 등,
이미 진행 중인 별도 축)이 필요하다는 기존 결론([[eth_dc_engineered154_feature_set_20260820]]
"How to apply")도 그대로 유지된다.

## 후속 검증 2 — 진짜 3-expert 독립학습 + LR 스케줄 (2026-08-22, 같은 날 저녁)

사용자 지시("레짐대로 데이터를 분류해서 3 expert에 데이터를 학습시켜줘. LR 스케쥴도 추가하고
lr을 좀 더 작은 수로 시작해줘")에 따라 학습기법을 한 단계 더 올려 재실행했다
(`scripts/eth_tabm_label_logic_5way_regime_expert_lrschedule_20260822.py`, N=6×3라벨 전체).

### 사전 조사에서 발견한 것 — `_route_probs` 몽키패치는 처음부터 no-op였다

기존 러너의 `parent_script._route_probs = _uniform_route_probs` 패치는 **한 번도 작동한 적이
없다.** `_fit_expert_omega4`는 내부에서 `parent._route_probs(...)`(TabM 하위모듈
`train_eval_omega1_2_tabm_3head_20260603`의 속성)를 부르는데, 패치는 omega4 모듈 객체
(`parent_script`)에 걸려 있어 서로 다른 객체였다(probe 스크립트로 실측:
`parent_script._route_probs is uniform → True`, `parent_script.parent._route_probs is
uniform → False`). 즉 **bull expert는 이 스크립트 계열 전체 역사에서 항상 진짜 레짐가중치로
학습돼 왔고**, "uniform regime weight" 로그 문구가 거짓이었다. 실제로 작동한 shortcut은
"bear/chop이 bull 가중치를 복사"(`_fit_expert_omega4_unified`, bare-name 호출이라 패치가
정상 적용) 쪽뿐이다.

### 이번에 실제로 바꾼 것

1. bear/chop-copy shortcut 제거 → 3 expert 전원이 각자의 레짐가중치(states=24/sticky=0.90
   분류기 확률)로 **독립 학습** (단일시드 검증에서 bull/bear/chop의 best epoch·vloss·조기종료
   시점이 전부 달라짐을 확인 — 복사 시절엔 항상 동일했다).
2. `CosineAnnealingLR` 스케줄 추가(원본 `_fit_expert_omega4`는 고정 lr).
3. lr 2e-3 → **2e-4** ([[feedback_modern_dl_training_checklist]] N≥5 검증값 재사용).
4. epoch 예산 4 → 40 (patience=8이 실제로 작동할 여유).

### 결과 (N=6, `ranking_by_validation_pnl[0]`, always-long: VAL −23.34% / OOS +21.51%)

| 라벨 | VAL mean±std | OOS mean±std | 부호일치 |
|---|---|---|---|
| zigzag | +2.92% ± 5.07 | +2.00% ± 3.65 | 2/6 |
| h48qual | +0.95% ± 11.41 | +1.70% ± 4.12 | 1/6 |
| cusum | +2.00% ± 3.98 | −4.10% ± 6.81 | 2/6 |

총 부호일치 5/18 — 동전던지기(9/18)보다도 낮고, 세 라벨 모두 OOS 평균이 always-long
벤치마크(+21.51%)에 크게 미달한다. h48qual seed=286919795의 VAL +24.03%가 눈에 띄지만 같은
시드의 OOS는 −3.30%로 반전 — 전형적인 시드노이즈([[tabm_hp_low_signal_pattern]]).

**판정: 학습기법을 정식으로 다 갖춰도(진짜 3-expert 독립학습 + LR 스케줄 + 낮은 lr + 충분한
epoch) chance-level 결론은 변하지 않는다.** "shortcut/학습기법 때문에 신호를 못 본 것"이라는
반론이 이것으로 소진됐다. 산출물:
`tmp/causal_regen_20260516/omega4_..._label5way_{라벨}_154feat_regime_expert_lrschedule_seed{시드}_20260822/`,
집계는 `scripts/eth_tabm_label_logic_regime_expert_lrschedule_pnl_aggregate_20260822.py`.

## 후속 검증 3 — direction/quality 하이퍼파라미터 튜닝 + long/short bias confound 체크 (2026-08-22, 밤)

사용자 질문("direction과 quality threshold 튜닝은 해야할거 같은데")에 대한 응답. 지금까지
`--quality-min-edge/--quality-max-mae/--quality-min-mfe-mae/--exit-giveback-min/--direction-
focal-gamma`는 전부 파이프라인 고정 기본값이었다. zigzag/h48qual 대상으로 Optuna 20trial
(VAL-pnl 목적함수, 서치 중 OOS 미접근) + 승자 설정 N=6 재검증
(`scripts/eth_tabm_label_logic_regime_expert_quality_direction_optuna_20260822.py`).

### 중요 발견 — 서치 대상 5개 중 4개가 사실상 no-op이었다

`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py` 코드를 추적한 결과:

- `quality-min-edge/quality-max-mae/quality-min-mfe-mae`는 `quality_mode=risk_adjusted_
  barrier_meta_action`일 때만 쓰인다(`_prepare_frames` L346-374). zigzag는
  `quality_mode=same_as_direction`, h48qual은 `quality_mode=quality_label_action` — 둘 다
  이 분기를 안 타므로 **세 파라미터 전부 두 라벨 모두에 대해 완전히 무관**했다.
- `exit-giveback-min`은 `exit_label_mode=entry_label_terminal_giveback`일 때만 쓰인다(L1161,
  1174). 이 축은 COMMON_ARGS로 `exit-label-mode=independent_entry_hold_offsets`를 고정해서
  쓰므로 **이 파라미터도 전체 세션의 모든 실행에서 한 번도 실제로 반영된 적이 없다.**
- 실제로 유효했던 서치 차원은 **`direction-focal-gamma` 하나뿐**이었다(5차원 중 1차원).
  h48qual 재검증 report.json이 baseline과 4/6 seed에서 완전히 동일한 숫자로 나온 것이
  이 사실과 정합적이다(quality-min-edge 등을 바꿔도 값이 안 바뀌는 게 당연함).

Optuna 자체는 이 사실을 몰라도 "관측된 VAL-pnl이 제일 좋은 조합"을 정상적으로 골랐으므로
아래 결과가 무효는 아니지만, "5개 하이퍼파라미터를 튜닝했다"가 아니라 "focal-gamma 1개를
20번 스캔했다"로 정정해서 읽어야 한다. 나머지 4개는 다음에 이 계열 스크립트를 다시 쓸 때
제거하거나(no-op 인자 정리) 최소한 이 문서를 참고해 서치공간에서 빼야 한다.

cusum은 위 no-op 발견을 반영해 처음부터 `direction-focal-gamma` 1개만 서치하도록 스크립트를
정리한 뒤 별도 실행(20trial+N=6 재검증, 나머지 진행 방식은 동일).

### 결과

| 라벨 | 튜닝 전 VAL/OOS(부호일치) | 튜닝 후 VAL/OOS(부호일치) |
|---|---|---|
| zigzag | +2.92%±5.07 / +2.00%±3.65 (2/6) | **+4.80%±2.76 / +3.70%±3.64 (5/6)** |
| h48qual | +0.95%±11.41 / +1.70%±4.12 (1/6) | +0.92%±11.28 / +0.89%±3.25 (1/6, 사실상 동일) |
| cusum | +2.00%±3.98 / −4.10%±6.81 (2/6) | +2.46%±3.62 / −4.87%±5.24 (3/6, N=6에서 유의차 없음) |

zigzag는 숫자만 보면 개선, h48qual과 cusum은 변화 없음(둘 다 20trial 중 VAL-pnl 양수인 best조차
찾았지만 — h48qual best=-1.39%로 그마저 음수, cusum best=-3.39%로 더 나쁨).

### long_frac ↔ pnl confound 체크

[[eth_tabm_label_logic_retest_initiative_20260819]]에서 h48qual 6/6 "완벽한" 부호일치가
사실은 롱비중과 PnL의 상관(0.888)에서 나온 착시였던 전례가 있어, 튜닝 후 승자 설정의 시드별
long_frac(롱진입/전체진입)과 pnl의 상관을 세 라벨 전부에 동일하게 계산했다.

| 라벨 | VAL long_frac↔pnl 상관 | OOS long_frac↔pnl 상관 |
|---|---|---|
| zigzag(튜닝후) | −0.55 | **+0.95** |
| h48qual(튜닝후) | **−0.85** | +0.74 |
| cusum(튜닝후) | +0.01(무관) | **+0.90** |

VAL(2026Q2, always-long −23% 하락장)은 zigzag/h48qual에서 숏비중이 높을수록 pnl이 좋고
(cusum은 이 관계가 없음), OOS(2026-07~, always-long +21% 상승장)는 **세 라벨 전부** 롱비중이
높을수록 pnl이 좋다(r=0.74~0.95) — 2026-08-19 h48qual 사건(0.888)과 정합적인 크기다. cusum은
OOS에서 이 편향이 있는데도 평균 OOS가 오히려 마이너스(−4.87%)인 걸 보면, 상승장의 이득을
롱비중으로도 충분히 못 걷어냈다는 뜻이라 "레짐추종조차 서투르다"는 결론에 가깝다.

### 판정

**"기본값이 잘못 캘리브레이션돼서 신호를 놓쳤다"는 가설은 기각한다.** 실제로 유효했던 유일한
차원(direction-focal-gamma)을 20trial씩 스캔해도: (a) h48qual·cusum은 개선이 없거나 무의미한
범위였고, (b) zigzag의 유일한 개선은 long_frac↔pnl 상관 0.55~0.95라는, 이 저장소가 이미 한 번
데었던 것과 같은 크기의 레짐추종 편향으로 설명된다 — 방향판단력이 좋아진 게 아니라 "하락장엔
숏, 상승장엔 롱" 쪽으로 트레이드 비중이 쏠리는 하이퍼파라미터를 찾은 것뿐이다. OOS
long_frac↔pnl 상관은 세 라벨 모두에서 +0.74~+0.95로 일관되게 강하다 — 우연이라기보다 이
파이프라인(regime-conditioned MoE 라우팅) 자체가 direction-focal-gamma를 통해 "레짐을 더
따라가는 쪽"으로 쉽게 밀리는 구조적 특성으로 보인다.

**direction/quality 튜닝 축도 CLOSED.** 산출물: `tmp/causal_regen_20260516/omega4_..._label5way_
{라벨}_154feat_regime_expert_lrschedule_seed{시드}_20260822_qdirtuned/`, Optuna study는
`tmp/ilias_labellogic_recheck_20260821/quality_direction_optuna_study.db`에 보존(재개 가능하나
위 no-op 발견 때문에 서치공간부터 다시 설계해야 함).

## 후속 검증 4 — BTC-metrics 오염 수정 후 재검증 (2026-08-23)

[[eth_binance_metrics_archive_backfill_canonical_divergence_20260823]]에서 ETH 캐노니컬 2026의
2026-01-20~07-12 구간(OI/롱숏비 및 파생 ~12컬럼, wide24의 `state12_oi_change_rate` 포함)이
BTC 값으로 오염돼 있었음이 발견·수정됨. **VAL(2026Q2) 전체 + OOS 07/1~12이 오염 창과 겹친다.**

### 근본원인 — 수정이 서버에 전파되지 않았다

다른 세션의 수정은 dev 파일시스템에서만 실행됐다. 이 3라벨 축의 실제 학습은 전부 handoff.sh로
서버에서 돌았는데, 서버의 캐노니컬(`training_features_2026_rebuilt.csv`)과 wide24 레짐
오버레이(states24/sticky0.90)는 08-20/08-21 오염 버전 그대로였다 — **어제(08-22) 이 문서의
모든 N=6×3라벨 결과(후속 검증 1~3 전부)가 오염 데이터로 계산됐다는 뜻.** 패치된 3개 파일
(캐노니컬 원본 + wide24 오버레이 train/oos)을 서버로 push해 격차를 해소했다.

### 재검증 — 클린 데이터로 N=6×3라벨 baseline(기본 HP) 재실행

`--out-suffix-tag postfix_datafix_20260823`로 원본 결과와 분리 저장(원본 파일 보존).

| 라벨 | 오염 데이터(어제) | 클린 데이터(오늘) |
|---|---|---|
| zigzag | VAL+2.92%±5.07 / OOS+2.00%±3.65 (2/6) | **VAL+9.13%±3.88 / OOS+4.93%±4.62 (5/6)** |
| h48qual | VAL+0.95%±11.41 / OOS+1.70%±4.12 (1/6) | VAL+3.89%±9.78 / OOS+0.42%±3.71 (2/6) |
| cusum | VAL+2.00%±3.98 / OOS−4.10%±6.81 (2/6) | VAL+8.36%±5.93 / OOS−5.19%±4.82 (**0/6**) |

**숫자 자체는 절대 "미세 변동"이 아니다** — zigzag 단일시드(133725056)만 봐도 VAL −0.61%→
+10.47%로 11pp대 스윙이다. 세 라벨 모두 VAL 평균이 체계적으로 상승했다(레짐 오버레이가
바뀌며 MoE 라우팅이 달라져 거래선택 자체가 바뀐 것으로 보임 — 정확한 인과경로는 미조사).

### long_frac ↔ pnl confound 재확인 — 오히려 더 강해짐

| 라벨 | VAL corr (어제→오늘) | OOS corr (어제→오늘) |
|---|---|---|
| zigzag | −0.55 → **−0.93** | +0.95 → +0.95 |
| h48qual | −0.85 → −0.56 | +0.74 → +0.90 |
| cusum | +0.01 → −0.71 | +0.90 → **+0.98** |

zigzag의 "개선"은 클린 데이터에서 레짐추종 편향이 **더 강해진 것**(VAL −0.93은 사실상 완전
선형관계)으로 설명된다 — 방향판단력이 새로 생긴 게 아니라는 기존 해석이 뒤집히기는커녕
더 뚜렷해졌다. cusum은 부호일치가 2/6→0/6로 오히려 더 나빠졌다.

### 판정

**데이터 오염 수정은 이 축의 정성적 결론(chance-level, long/short 편향으로 설명되는 겉보기
변동)을 바꾸지 않는다 — 오히려 confound 증거가 강해졌다.** [[eth_binance_metrics_archive_
backfill_canonical_divergence_20260823]]의 "판정 소급무효 아님" 평가가 이 축에 대해서는
사실로 확인됐지만, 그 근거("수치 미세변동")는 이 축에서는 부정확했다 — 수치는 크게 변했고,
단지 그 변화의 방향이 결론을 뒤집는 방향이 아니었을 뿐이다. direction/quality 튜닝(Optuna)은
클린 데이터로 재실행하지 않았다 — baseline 확인만으로 confound 해석이 이미 강화됐고, 튜닝의
유일한 유효차원(direction-focal-gamma)이 이 패턴을 바꿀 것으로 기대할 근거가 없기 때문(우선순위
낮음, 필요시 후속 가능).

**CLOSED 유지, 다섯 번째 독립 확증(이번엔 클린 데이터 기준).**

## 최종 판정 (2026-08-22, 2026-08-23 재확증)

zigzag/h48qual/cusum 3라벨 단일 컴포넌트 축은 이번 세션에서 시도한 모든 축 — 새 split 규약,
epoch 깊이, 진짜 3-expert 독립학습+LR스케줄, direction/quality 하이퍼파라미터 튜닝, **그리고
BTC-metrics 오염 수정 후 클린 데이터 재검증** — 에서 전부 chance-level 또는
known-confound-설명가능 결과로 귀결됐다. **CLOSED 유지.** 재오픈하려면 154피쳐 자체를 대체할
질적으로 다른 정보원이 필요하다는 기존 결론([[eth_dc_engineered154_feature_set_20260820]])이
최종적으로 유지된다.
