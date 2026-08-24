# ETH Directional-Change 라벨 — TabM/N-HiTS 단일모델(레짐피쳐) 학습 비교 (2026-08-19)

**상태: 학습+스크리닝 완료 — TabM은 N=5 시드 OOS 부호 불일치로 스크리닝 기각, N-HiTS는
N=1 예비 결과가 always_short 벤치마크에 OOS 패배. 둘 다 2단계(Fresh-Forward 정식평가)
투자 근거 없음. 승격/edge 주장 아님. ⚠️2026-08-20 후속: 조건부 방향정확도가 6개 런
전부 chance(48~51%)로 확정됐고, 시드 불안정성 해소를 노린 4개 기법(HP재조정/ASWA/배깅/
딥앙상블)을 전부 구현+실행해도 동일 — 근본원인은 최적화가 아니라 신호 부재로 결론. dc_theta
파라미터 스윕(경제성+정보량)도 기각, 후보②(CUSUM) 정보량 검증에서 pooled 유의(p=0.000)가
나왔으나 2026단독 재검증에서 소멸(p=0.290) — ①DC+②CUSUM 둘 다 CLOSED. ⚠️2026-08-20 추가
후속: "새 데이터가 없으니 기존 158피쳐를 정리+조합하자"는 요청으로 피쳐셋 자체를 감사 —
158→133개로 정리(죽은 컬럼 11+중복클러스터 14) 후 재학습해도 chance 그대로, 133개
전체쌍(8,778개) 명시적 상호작용 유의성 테스트는 pooled p=0.030(경계선)이 2026단독
재검증에서 p=0.515로 붕괴(CUSUM과 동일 패턴) — 정리·조합 둘 다 CLOSED. 상세 아래
"피쳐셋 연구(정리+조합)" 절.**

## 배경

`docs/experiments/eth_directional_change_labels_20260819.md`(후보①, DC 라벨) 빌드 완료 후,
이 라벨로 TabM 3-head 실제 학습을 진행했다. 도중 "ModernTCN도 비교군으로"라는 요청이 있었고,
비용/리스크(서버 GPU 필요, 라이브봇과 경합, 과거 시간비용으로 중단된 전례)를 확인한 뒤
"N-HiTS 단일시드 사전확인부터"로 범위를 좁혔다. 곧이어 "레짐당 모델 대신 단일모델+레짐피쳐"
지시가 추가돼 TabM(기존 bull/bear/chop 3-expert MoE)과 N-HiTS(regime-hardsplit 계열) 둘 다
이 원칙을 적용했다. 전체 설계는 `/home/kbj20/.claude/plans/1-velvety-whistle.md`(승인됨)
참고.

## 구현

### 1. Dense CASH-fill 라벨 재export
[scripts/build_eth_directional_change_dense_cashfill_labels_20260819.py](../../scripts/build_eth_directional_change_dense_cashfill_labels_20260819.py)
— sparse DC 이벤트(전체 bar의 6~8%)를 그대로 쓰면 `omega._align()`의 inner-join으로 표본이
축소되고 exit_head/`_metrics()`가 행 인덱스를 5분 간격으로 오인하는 문제가 있어(둘 다 코드로
직접 확인), canonical bar 그리드에 재색인 후 비이벤트 bar를 CASH(0)로 채운 dense 버전을
별도로 만들었다(2024/2025/2026 전부, TabM은 2025/2026만 쓰고 N-HiTS는 2024도 씀). 원본
sparse 산출물은 그대로 유지. CASH 91.7~93.4%, LONG/SHORT 각 3~4%대.

### 2. TabM — canonicaldata + 단일모델
- [scripts/eth_directional_change_tabm_training_canonicaldata_20260819.py](../../scripts/eth_directional_change_tabm_training_canonicaldata_20260819.py)
  — `omega.TRAIN_CSV`/`EVAL_CSV`를 legacy(feature drift, 2026-02-28 절단)에서 canonical
  `data/splits/year_oos/training_features_{2025,2026_rebuilt}.csv`로 오버라이드
  (`train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818.py`와 동일 로직,
  별도 placeholder 디렉토리로 동시세션 충돌만 회피).
- [scripts/eth_directional_change_tabm_training_unified_single_model_20260819.py](../../scripts/eth_directional_change_tabm_training_unified_single_model_20260819.py)
  — `parent._route_probs`를 전부 1.0으로 monkeypatch(레짐 라우팅 무력화) + bull expert만
  실제 학습 후 bear/chop에 payload 복사(`train_eth_ilias1_zig075_trial12_unified_single_model_20260819.py`와
  동일 기법). `--quality-mode same_as_direction`(기본값 hard_rule은 DC에 없는 진단컬럼 요구로
  크래시), `--exit-label-mode independent_entry_hold_offsets`(zig075의 entry_label_terminal_giveback은
  DC의 고립된 단일-bar 이벤트에서 세그먼트 스캐너가 거의 전부 skip해 크래시 — dense-fill
  후에도 LONG/SHORT는 여전히 개별 bar지 zigzag_action류 연속구간이 아님). 레짐 피쳐는
  `omega._load_omega_frames()`의 필수 오버레이(`regime3_current_sensitive_wide24_*`)로 이미
  자동 포함 — 새로 계산한 피쳐 없음.

### 3. N-HiTS — 레짐피쳐 + DC 라벨
- [scripts/train_eval_eth_direction_quality_nhits_regimefeature_dc_20260819.py](../../scripts/train_eval_eth_direction_quality_nhits_regimefeature_dc_20260819.py)
  — `train_eval_eth_direction_quality_nhits_moderntcn_20260816.py`(base_nt)를 import로 재사용,
  `SEQ_COLS`(8개 고정 컬럼)에 HMM 레짐확률 6개(`regime3_current_sensitive_wide24_{bull,bear,chop}_prob/
  _confidence/_entropy/_margin`, 2025-01-01부터 커버 — 그 이전 구간은 0.0)를 concat해 14채널로
  확장. `len(SEQ_COLS)`를 참조하는 모든 다운스트림 코드가 자동으로 새 채널 수를 따라감(개별
  패치 불필요). `DIRECTION_LABEL_DIR`/`QUALITY_LABEL_DIR` 둘 다 DC dense-cashfill로 오버라이드.
- [scripts/eth_directional_change_nhits_single_seed_run_20260819.py](../../scripts/eth_directional_change_nhits_single_seed_run_20260819.py)
  — `base_nt.stage_final()`을 N=1 시드로 복제(HP서치/isolation 없이 기본 파라미터, GCE/ELR/mixup
  전부 off).

## 결과

### TabM — N=5 시드 (canonical 데이터, `random.SystemRandom().sample`로 진짜 무작위 추출)

각 시드의 VAL-최적 threshold와 그 OOS PnL:

| seed | VAL-최적 threshold | VAL PnL | VAL 거래수 | OOS PnL | OOS 거래수 | OOS 부호 |
|---:|---:|---:|---:|---:|---:|:---:|
| 758616172 | 0.45 | +13.84 | 43 | **+60.70** | 64 | + |
| 810628369 | 0.60 | +6.43 | 39 | **+23.64** | 64 | + |
| 615897020 | 0.45 | +9.97 | 37 | **+18.07** | 66 | + |
| 176529615 | 0.40 | +10.45 | 45 | **−3.88** | 64 | − |
| 573123622 | 0.45 | +5.49 | 42 | **−9.33** | 69 | − |

**5개 중 3개 양수, 2개 음수 — OOS 부호 불일치.** VAL은 5개 전부 양수로 그럴듯해 보였지만
(특히 첫 시드는 VAL+13.8%/OOS+60.7%로 인상적), 이 저장소가 h48qual/zig075/Sigma3-1h 등에서
반복 확인한 바로 그 패턴("VAL은 일치, OOS는 시드에 따라 뒤집힘")이 DC 라벨에서도 재현됐다.
threshold 선택 자체는 어느 정도 군집(0.4~0.6)돼 순수 노이즈는 아니지만, OOS 부호가 갈리는
이상 스크리닝 통과로 볼 수 없다. 표본수는 canonical 전체 규모로 정상 복원됨(train 78,605~
/val 26,496/oos 51,746행, sparse 시절 6~8% 축소 아님 — dense CASH-fill 수정이 의도대로
작동했음을 확인).

전체 report.json: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_dc_dense_cashfill_unified_single_model_seed{seed}_20260819/report.json`

### N-HiTS — sanity(양 아키텍처) + 단일시드 본실행

**Sanity(크래시 확인용, 2025-06~09 4개월, 2epoch, seed=1 고정)**: ModernTCN/N-HiTS 둘 다
통과. ModernTCN **450초**, N-HiTS **5초** — 소표본 기준으로도 ~90배 차이, 이전에 확인한
"ModernTCN은 CPU에서 사실상 서버GPU 필수" 판단을 재확인. 분류지표는 거의 정확히 chance
수준(둘 다 ~0.33, 3-class 무작위 기대치)이었으나 이건 sanity 자체가 "크래시 안 하는지"만
보는 스테이지라 예상된 결과.

**N-HiTS 단일시드 본실행**(전체 데이터, 30epoch cap, 실제 57초 완주):

| | n | direction_balanced_accuracy | model PnL(cost3) | always_short PnL | always_long PnL | 벤치마크 이김? |
|---|---:|---:|---:|---:|---:|---|
| VAL(2025-10~12) | 26,496 | 0.4924 | +33.92 | +15.01 | −13.39 | **둘 다 이김** |
| OOS(2026-01~02, h48_conservative 라벨 커버리지 한계로 03월 제외) | 16,992 | 0.4993 | **+8.76** | **+17.47** | −17.09 | long만 이김, **short는 짐** |

VAL에서는 always_short/always_long 둘 다 이겼지만, **OOS에서는 always_short(+17.47)가
모델(+8.76)보다 낫다.** direction_balanced_accuracy는 VAL/OOS 둘 다 chance(0.333)보다
뚜렷이 높아(0.49~0.50) 순수 분류 관점에서는 방향 정보를 어느 정도 포착하는 것처럼 보이지만,
PnL 관점에서는 이 저장소가 반복 확인한 "신호는 실재해도 무료 벤치마크(always_short)에
흡수됨" 패턴과 일치한다 — OOS 구간이 단방향 약세장이었다는 기존 기록
(`eth_btc_regime_shift_reopening_candidates_20260819`)과도 정합적이다.

**N=1이므로 이 결과 자체가 결론은 아니다** — TabM처럼 시드에 따라 부호가 뒤집힐 가능성을
배제할 수 없다.

## 스코프 경고

이 문서의 어떤 수치도 방향 예측 edge나 승격 근거로 쓰지 않는다.
`docs/label_methodology_survey_20260815.md`(40+ 선행 라벨 방법론 전부 동일 패턴)와 정확히
같은 결론이 DC 라벨 + 두 아키텍처(TabM/N-HiTS) 조합에서도 재현됐다 — **라벨을 바꿔도,
아키텍처를 바꿔도, 이 저장소의 근본 병목(feature information content 부재)은 그대로
드러난다.** 두 학습 스크립트의 내부 VAL/OOS도 canonical Fresh-Forward 경계(VAL 09-01
시작, OOS 03-31까지)와 정확히 일치하지 않고 실제 리플레이 엔진을 안 거치는 참고용
스크리닝일 뿐이다.

## 마이너스 시드/구간 특징 분석 (2026-08-20 추가)

TabM N=5 중 OOS 부호가 음수인 2개 시드(176529615, 573123622)와 N-HiTS 단일시드(502957522,
동일 seed로 재실행해 재현 확인 — VAL/OOS pnl·trades·wr 전부 원본과 일치)를 대상으로, 각
시드가 실제로 취한 트레이드의 LONG/SHORT 방향 비율·청산사유·월별 지속성을 뜯어봤다. 결론부터:
**"마이너스인 시간 구간"이 따로 있는 게 아니라, 마이너스 시드 자체가 OOS 6개월(2026-01~06)
전체에서 지속적으로 LONG쪽으로 치우쳐 있다.** 이 구간이 always_short 벤치마크가 우세한
하락 성향 구간이라는 점(`eth_btc_regime_shift_reopening_candidates_20260819` 메모리)과 맞물려
방향편향이 그대로 부호를 결정한다.

### TabM — 시드별 OOS 실제 트레이드 방향비율(VAL-최적 threshold 기준)

| seed | OOS pnl | 트레이드 LONG% | wr | mdd | stop_loss% | raw신호 LONG%(매월 최소~최대) |
|---:|---:|---:|---:|---:|---:|---|
| 758616172 | **+60.70** | 26.6% (17/64) | 0.531 | −7.5% | 45.3% | 9~51% |
| 810628369 | **+23.64** | 4.7% (3/64) | 0.453 | −12.8% | 54.7% | 0~1% |
| 615897020 | **+18.07** | 0.0% (0/66) | 0.439 | −12.7% | 56.1% | 8~34% |
| 176529615 | **−3.88** | 50.0% (32/64) | 0.359 | −17.2% | 64.1% | 68~95% |
| 573123622 | **−9.33** | 72.5% (50/69) | 0.333 | −20.7% | 66.7% | 81~93% |

트레이드 단위 LONG 비율로 두 그룹이 완전히 분리된다(양수 그룹 0~27% vs 음수 그룹 50~73%,
그 사이 27~50%는 공백). 원시 신호(threshold 게이팅 전 매 bar 예측)의 월별 LONG 비율도
확인했는데, 음수 시드는 2026-01부터 06까지 6개월 내내 매달 68% 이상 LONG으로 일관됐다
(89%→68%→76%→85%→95%→83% 등) — 특정 주/월에 몰린 이벤트가 아니라 시드 수렴 시점에 고정된
구조적 편향이다.

⚠️ 정확히 말하면 이건 "LONG비율이 낮을수록 좋다"는 선형/단조 관계가 아니라 **문턱 효과**다 —
안전권 3개 시드 안에서는 LONG비율이 가장 높은 758616172(26.6%)가 오히려 wr(0.531)과 PnL도
가장 높다(0%/4.7% LONG인 다른 두 시드보다 우세). wr·stop_loss비율은 **PnL 순위와는** 5개
시드 전부 정확히 단조(순위가 곧 wr 순위)지만, LONG비율과는 안전권 내부에서 순서가 안 맞는다
— "적게 배팅할수록 좋다"가 아니라 "약 30~50% 어딘가의 문턱을 넘기면 급격히 나빠진다"에
가깝다(공백 구간이 그 문턱의 대략적 위치를 시사). 참고로 VAL(2025-10~12)에서는 이 두 음수
시드도 LONG 비율이 상대적으로 높았지만(44%/64%) VAL PnL 자체는 양수였다 — VAL과 OOS 구간의
성향이 다르다는 정황(추가 독립 검증은 안 함, 이 관찰은 탐색적)과 일치한다. VAL 부호일치가
OOS 부호일치를 보장 못하는 이유가 바로 이 방향편향의 구간의존성으로 설명된다.

### N-HiTS — 유일 시드의 방향비율(N=1이라 시드 간 비교 불가, 런 내부만 분석)

동일 seed(502957522)로 재실행해 상세 브레이크다운을 다시 뽑았다(VAL pnl=+33.92, OOS
pnl=+8.76 — 원본과 정확히 일치, 재현성 확인). TabM의 "이긴 시드"들과 달리 이 시드는 원시신호
LONG 비율이 VAL 41~45%, OOS 42~46%로 **양방향 다 균형에 가깝다**(TabM 음수시드의 68~95%나
TabM 최우수시드의 0~9%처럼 한쪽으로 쏠리지 않음). 실제 트레이드도 OOS LONG 16/42=38.1% —
공교롭게도 TabM 5시드가 만든 공백 구간(27~50%)에 정확히 들어간다. PnL도 그 위치에 맞게
TabM 양수시드 3개보다는 낮고 음수시드보다는 높은 딱 중간값(+8.76)이라, 위에서 말한 "문턱"
가설과 정합적이다(다만 N=1이라 이 정합성 자체가 통계적 근거는 아니고 관찰일 뿐).
같은 진입 타이밍에서 방향만 전부 SHORT로 강제하면(`forced_side(dec,-1)`, 이 문서 앞부분의
always_short 벤치마크와 동일 정의 — "매 bar 무조건 숏"이 아니라 "모델이 실제로 진입한 bar에
한해 방향만 숏으로 고정") pnl은 +8.76→+17.47로 뛴다. 즉 이 모델이 스스로 고른 진입 타이밍
자체는 always_short보다 나쁘지 않지만, 그 중 44%를 LONG으로 건 것이 순손실 요인이었다 —
방향 선택이 타이밍 선택보다 약하다는 뜻이고, TabM 음수시드와 같은 방향(과도한 LONG 배분)의
약한 버전이다.

### 조건부 방향정확도 — 6개 런 전부 chance 수준 (2026-08-20 추가)

위 "문턱효과"가 진짜 신호(다만 비선형적인)인지, 애초에 신호가 없는지를 가르기 위해 더 엄격한
질문을 던졌다: **실제 라벨과 모델 예측이 "둘 다 CASH가 아닌"(=둘 다 뭔가 방향을 주장하는)
bar만 놓고, 그 방향(LONG/SHORT)이 서로 얼마나 일치하는가?** 이건 "활성 여부를 맞췄나"와
"방향을 맞췄나"를 분리하는 지표다 — CASH가 라벨의 93%를 차지해서 활성여부 판별만으로도
전체 정확도가 쉽게 부풀려지기 때문. DC 라벨의 실제 이벤트bar 방향분포는 LONG 48.0%/SHORT
52.0%로 거의 균형이라, 이게 chance 기준선이다.

| | 교집합 표본수 | 방향일치율 | chance 기준선 |
|---|---:|---:|---:|
| TabM 758616172(OOS +60.70, 최우수) | 2,699 | 50.6% | ~50~52% |
| TabM 810628369(OOS +23.64) | 2,459 | 51.0% | ~50~52% |
| TabM 615897020(OOS +18.07) | 2,663 | 50.5% | ~50~52% |
| TabM 176529615(OOS −3.88) | 2,392 | 50.3% | ~50~52% |
| TabM 573123622(OOS −9.33, 최악) | 2,840 | 48.2% | ~50~52% |
| N-HiTS 502957522(OOS +8.76, N=1) | 953 | 51.4% | ~52% |

**6개 런 전부 chance 기준선과 통계적으로 구분 안 되는 범위(48.2~51.4%)에 몰려있다 —
OOS PnL이 가장 좋았던 시드(758616172)조차 방향적중률은 정확히 chance(50.6%).** 즉 "문턱을
넘으면 나빠진다"는 위 관찰은 실제로 존재하는 방향판별 능력의 임계치가 아니라, **애초에
방향판별 능력 자체가 없는 상태에서 각 시드가 무작위로 수렴한 편향의 크기가 우연히 OOS
레짐(숏우세)과 얼마나 맞아떨어졌는지**를 다르게 표현한 것뿐이다. N-HiTS의 3-class
direction_balanced_accuracy(VAL 0.49/OOS 0.50, chance 0.333)가 "그래도 chance보단 높다"고
읽힐 여지가 있었는데, 이 세분화된 지표로 보면 그 우위는 거의 전부 "CASH vs 활성" 판별(비교적
쉬운 과제, CASH가 93%)에서 나온 것이고 LONG-vs-SHORT 방향 서브태스크 자체는 TabM과 동일하게
chance 수준이다.

### 종합

두 아키텍처 모두 같은 메커니즘으로 수렴한다 — **방향(LONG/SHORT) 판별 능력 자체가 6개 런
전부에서 확인되지 않았고(조건부 정확도 48~51%, chance 수준), OOS 구간(2026 상반기)이 구조적
으로 숏이 유리했던 상황에서 모델/시드가 우연히 얼마나 숏쪽으로 편향됐는지가 PnL 부호와
크기를 거의 전부 설명한다.** 이건 `h48qual_standalone_replay_invalid` 메모리가 반복 지적한
"always 벤치마크 대조 없이는 편향을 스킬로 착각한다"는 패턴의 직접적인 재확인이며, 위
조건부 방향정확도 결과로 "가능성이 높다"가 아니라 확인된 사실이 됐다 — TabM의 최우수
시드(758616172, +60.70)도 진짜 트레이드 셀렉션이 뛰어난 게 아니라 다른 시드들보다 덜 반대
방향으로(73%~26%LONG 스펙트럼에서 가장 낮은 쪽 근처에) 수렴했을 뿐이다. 이 문서의 REJECTED
스크리닝 판정 자체를 바꾸지는 않지만("왜 시드마다 부호가 갈렸는가"에 대한 설명이며 "그러니
승격해도 된다"는 근거가 아님), 다음에 라벨/아키텍처를 또 바꿔도 같은 패턴(방향편향=벤치마크
우연 일치, 진짜 방향신호 없음)이 재현될 가능성이 높다는 걸 시사한다.

### under-training 배제 테스트 — epoch 2→30, 결과 무변화 (2026-08-20 추가)

사용자가 시드 불안정성 해법으로 ASWA/NASWA류 가중치평균화·앙상블을 제안했으나, 위 chance-level
결과는 "노이즈로 흔들리는 진짜 신호"가 아니라 "신호 자체 부재"의 지문이라는 반론을 제시했다.
다만 TabM 스크리닝 런이 `--epochs 2`(patience=8 CFG 기본값이 발동할 새도 없는 얕은 설정)였다는
잔여 반론은 열려 있었다 — 이걸 직접 닫기 위해 원본 최우수/최악 시드(758616172, 573123622)를
`--epochs 30`(patience=8은 그대로, N-HiTS MAX_EPOCHS_FINAL=30과 동일 예산)으로 재학습했다.

| seed | epochs_ran(원본2→신규) | best_val_loss(원본→신규) | OOS pnl(원본→신규) | 조건부 방향정확도(원본→신규) |
|---:|---:|---:|---:|---:|
| 758616172 | 2 → 10(patience로 조기종료) | 2.566 → 2.160 | +60.70 → +60.70 | 50.6% → 50.6% |
| 573123622 | 2 → 9(patience로 조기종료) | (원본 미기록) → 2.514 | −9.33 → −9.33 | 48.2% → 48.2% |

**검증손실은 두 시드 다 개선됐는데(5배 가까운 추가 학습이 실제로 뭔가에는 도움이 됨 — CASH
vs 활성 판별/보정 쪽으로 추정) 조건부 방향정확도·OOS PnL·트레이드 구성은 소수점까지 완전히
그대로다.** 이는 "학습 부족"이 이 결과의 원인이 아님을 직접 반증한다 — patience=8 기반
early stopping을 정상적으로 거치고도 방향 서브태스크 성능은 한 걸음도 못 움직였다.
`scripts/eth_directional_change_tabm_training_unified_single_model_epoch30test_20260820.py`
(원본 unified_single_model 스크립트와 `--epochs` 값 하나만 다름). 이걸로 ASWA/가중치평균화·
앙상블 제안에 대한 "혹시 학습이 덜 된 것 아니냐"는 마지막 반론도 닫혔다 — 앙상블 투자 여부는
사용자 결정 대기.

### 딥앙상블 실증검증 — 예측대로 신호 없음, chance 유지 (2026-08-20 추가)

사용자가 "그래도 직접 구현해서 실증확인"을 선택해 구현했다. ⚠️ASWA 자체가 아니라 **딥앙상블
(여러 독립 시드의 예측확률을 평균)**을 구현했다 — ASWA/NASWA는 "한 번의 학습 궤적 안"의
체크포인트들을 평균해 궤적 내 진동을 매끈하게 하는 기법인데, 바로 위 epoch30 테스트에서 각
시드가 patience=8 기준 9~10에폭 내에 하나의 고정 편향으로 수렴하고 그 뒤로 전혀 안 흔들린다는
게 확인돼 ASWA가 다룰 "궤적 내 진동" 자체가 없다. 우리 문제(시드마다 다른 편향에 수렴)에
실제로 대응하는 건 여러 독립 모델의 예측을 결합하는 딥앙상블/bagging 쪽이라 이걸 구현했다
(재학습 없음 — 기존 5시드의 direction head 확률을 bar별로 평균 후 argmax,
`scripts/eth_directional_change_tabm_deep_ensemble_verification_20260820.py`).

| 지표 | 개별 5시드 범위 | 딥앙상블(확률평균) | 다수결 앙상블(교차확인) |
|---|---|---|---|
| 조건부 방향정확도 | 48.2~51.4% | **50.7%** | **50.2%** |
| 활성bar 중 LONG비율(raw) | 0~88% | **44.0%** | — |
| OOS PnL | −9.33~+60.70 | **+8.30** | — |
| MDD | −7.5~−20.7% | **−17.33%** | — |
| 승률 | 0.333~0.531 | **0.398** | — |

예측했던 그대로 나왔다 — **조건부 방향정확도는 그대로 chance권(50.7%/50.2%, 개별 시드 범위
안에 그냥 포함됨)이고, 앙상블의 LONG비율(44.0%)은 개별 시드들의 극단(0~88%)이 아니라 라벨
기저비율(48%)에 훨씬 가까운 값으로 수렴했다.** PnL(+8.30)도 최우수 개별시드(+60.70)의 "운좋은
숏편향"이 평균으로 지워지면서 그보다 한참 낮고, MDD(−17.33%)는 오히려 개별 양수시드들
(−7.5~−12.8%)보다 나빠졌다 — "더 안정적이지만 그 중심 자체가 평범하다"는 예측과 정확히
일치. 신호가 없는 상태에서 분산만 줄이면 결과가 시드 간엔 일관돼지지만 기대치 자체는 못
올라간다는 게 실증으로 확인됐다. ASWA/가중치평균화·앙상블 축은 이걸로 닫는다.

### 앙상블 자체의 시드배치간 안정성 — 불안정, 축 완전 CLOSED (2026-08-20 추가)

위 딥앙상블은 원본 5시드 **한 조합**의 결과였다. "앙상블을 다시 만들어도 비슷하게 나오는가"를
직접 확인하기 위해 완전히 새로운 5개 시드(498893814/405866927/492015211/108277116/519733484,
원본과 무교집합)를 학습시켜 두 번째 독립 딥앙상블을 만들었다
(`scripts/eth_directional_change_tabm_deep_ensemble_verification_batch2_20260820.py`).

| | 1차 앙상블(원본5시드) | 2차 앙상블(신규5시드) |
|---|---:|---:|
| LONG 비율 | 44.0% | **65.8%** |
| 조건부 방향정확도 | 50.7% | 50.1%(동일하게 chance) |
| OOS PnL | +8.30 | **−22.65** |
| MDD | −17.33% | **−35.26%(개별 최악시드 −20.67%보다도 나쁨)** |

**앙상블 자체가 시드배치에 따라 완전히 다른 자리로 간다 — 조건부 방향정확도는 두 배치 다
chance로 동일(신호가 없으니 평균내도 안 생김)하지만, LONG비율/PnL/MDD는 44%/+8.30/−17%에서
66%/−22.65/−35%로 뒤집혔다.** 불안정성이 해소된 게 아니라 "개별 시드 뽑기 운"에서 "앙상블
구성 시드 조합 뽑기 운"으로 자리만 옮겼다 — 평균낼 대상 전부가 노이즈면 그 평균도 노이즈다.
ASWA/가중치평균화·앙상블 축 완전 CLOSED.

### 피쳐셋 정보량 점검 — 원본102 vs 신규56, 차이 없음 (2026-08-20 추가)

사용자가 "이전에 쓰던 102개 피쳐셋이 더 성능이 좋았던거 같다"고 언급 — 이 102개는
h48qual/zig075 프로덕션 번들(6/29·6/30)의 원본 피쳐셋으로,
`docs/experiments/eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818.md`
후속세션8/9에서 158/172피쳐 posfix 버전과 Fresh-Forward PnL로 비교된 적이 있다. ⚠️단, 그
비교는 **피쳐셋+risk sidecar 진위+threshold 재튜닝 3개가 동시에 바뀐 단일시드** 결과라
피쳐셋 하나만의 순효과는 분리된 적이 없다(그 문서 자체가 이 한계를 명시). 이번엔 모델·
sidecar·threshold를 전부 빼고 "이 피쳐가 DC 방향라벨과 단독으로 얼마나 상관있는가"만
순수하게 재서 직접 검증했다 — `scripts/eth_dc_feature_set_information_content_20260820.py`,
DC 학습이 실제로 쓴 프레임(2025/2026, REGIME3_CURRENT 오버레이 포함) 기준 이벤트bar
12,193개(LONG 5,916/SHORT 6,277)에서 각 피쳐 단독 AUC-ROC(방향무관) + 라벨셔플
200회 permutation null.

| 그룹 | 피쳐수 | mean AUC | max AUC | 최고피쳐 | permutation p-value |
|---|---:|---:|---:|---|---:|
| 전체158 | 158 | 0.5038 | 0.5141 | ou_halflife | 0.380 |
| 원본102 | 102 | 0.5044 | 0.5141 | ou_halflife(동일) | 0.325 |
| 신규56 | 56 | 0.5025 | 0.5099 | btc_ret_1 | 0.650 |

**세 그룹 다 통계적으로 유의하지 않다(p=0.325~0.650, 관습적 기준 0.05에 한참 못 미침) —
158개 중 가장 잘 나온 피쳐(AUC=0.5141)조차 라벨을 무작위로 섞었을 때 158개 중 최고값과
구분이 안 된다(다중비교 보정 없이 그냥 봤으면 "AUC 0.514면 뭔가 있나?" 착각했을 수치).**
원본102의 mean AUC(0.5044)가 신규56(0.5025)보다 살짝 높긴 하지만 그 차이 자체가 노이즈
수준(둘 다 개별적으로 이미 비유의)이라 "102가 더 정보량이 많다"는 근거로 쓸 수 없다.
즉 **사용자의 recollection("102가 더 성능이 좋았다")은 이 축(DC 방향라벨 정보량)에서는
재현되지 않는다** — liveATR reaudit에서 pinned102가 posfix보다 baseline에 더 가까웠던
건 피쳐셋 자체보다 그 비교에 같이 섞여있던 다른 두 변수(진짜 sidecar/재튜닝된 threshold)나
단일시드 노이즈 쪽에서 왔을 가능성이 높다(단, 그 문서는 다른 라벨(zigzag_action)·다른
모델(h48qual/zig075) 기준이라 이 결과가 그 문서의 열린 질문을 직접 해소하진 않는다 —
어디까지나 DC 라벨 축에서의 교차검증).

### 나머지 3개 안정화 기법 실증 — HP재조정/ASWA/배깅, 전부 무효 (2026-08-20 추가)

사용자가 처음 제안한 4개 기법(하이퍼파라미터/정규화 재조정, ASWA류 가중치평균화, 배깅,
딥앙상블) 중 딥앙상블만 구현했었다는 지적을 받고, 나머지 3개도 전부 실제로 구현+실행했다.
`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py::_fit_expert_omega4`
(원본 학습루프)를 거의 그대로 복제해 변형 지점만 최소수정
(`scripts/eth_directional_change_tabm_training_variants_20260820.py`, 공유모듈 미변경):

- **hp**: lr 2e-3→2e-4([[feedback_modern_dl_training_checklist]] 근거), weight_decay
  2e-4→1e-3(5배), 첫 에폭 선형 warmup.
- **aswa**: burn-in(epoch≥2, epoch30 테스트 실측 수렴시점) 이후 매 에폭 체크포인트를
  전부 누적평균(ThreeHeadTabM은 LayerNorm만 써서 BatchNorm 재보정 이슈 없음, 확인함).
- **bag**: `train_idx`를 복원추출(bootstrap, 같은 크기)로 교체, `val_idx`(조기종료)는
  안 건드림.

3개 변형 다 새 5시드(원본/1차/2차 배치와 전부 무교집합)로 학습(`--epochs 30`, patience=8
그대로 — hp/aswa가 burn-in/수렴에 여러 에폭이 필요해서 스크리닝의 epoch=2가 아니라 이
예산을 씀), 개별 조건부 방향정확도 스프레드와 변형별 5시드 딥앙상블을 원본과 비교했다.

**개별 5시드 조건부 방향정확도 스프레드**:

| | 원본 베이스라인 | hp | aswa | bag |
|---|---|---|---|---|
| 스프레드 | 48.2~51.4% | 48.8~50.7% | 47.9~50.1% | 49.3~51.2% |
| OOS 부호 | 3승2패(혼재) | 4승1패(혼재) | 4승1패(혼재) | 3승2패(혼재) |

세 변형 다 원본과 사실상 같은 chance권(48~51%) 안에 그대로 있고, OOS 부호도 넷 다
혼재(전부 일치하는 경우 없음) — 어떤 기법도 개별 학습을 chance 이상으로 끌어올리지
못했다.

**5시드 딥앙상블 비교** (지금까지 만든 독립 앙상블 5개 전부):

| 앙상블 | LONG% | 조건부정확도 | PnL | MDD |
|---|---:|---:|---:|---:|
| 원본 1차 | 44.0% | 50.7% | +8.30 | −17.33% |
| 원본 2차 | 65.8% | 50.1% | −22.65 | −35.26% |
| hp | 66.3% | 49.8% | +9.50 | −15.10% |
| aswa | 43.7% | 49.3% | −5.13 | −18.86% |
| bag | 56.5% | 49.7% | −23.37 | −30.11% |

**5개 독립 앙상블의 LONG%가 43.7~66.3%로 원본 두 배치가 만든 범위(44.0~65.8%)를 벗어나지
못했고, PnL도 −23.37~+9.50으로 여전히 부호가 갈린다.** 어떤 안정화 기법을 썼는지와 무관하게
결과 스프레드가 거의 동일하다 — hp 앙상블(66.3% LONG)은 원본 2차 배치의 최악 편향(65.8%)과
사실상 같은 자리, bag 앙상블(−23.37)은 원본 2차 배치(−22.65)와 거의 같은 손실 규모.

**결론: 사용자가 제안한 4개 기법(HP재조정/ASWA/배깅/딥앙상블) 전부 구현+실행 완료, 전부
동일한 결과(조건부 방향정확도 불변, 시드/앙상블 배치간 불안정 그대로) — 이 축은 완전히
소진됐다.** 근본 원인은 최적화 방식이 아니라 이 라벨+피쳐셋 조합 자체에 학습 가능한 방향
신호가 없다는 것이고, 어떤 최적화·정규화·앙상블 기법도 없는 신호를 만들어내지 못한다.

### dc_theta 파라미터 스윕 — 경제성·정보량 둘 다 기각, DC축 완전 소진 (2026-08-20 추가)

사용자가 후보①(DC)의 파라미터 조정을 요청 — `dc_theta`(기존 0.004)를 올리는 방향이
이전에 미실행 권장사항으로 남아있었다(`eth_tabm_label_logic_retest_initiative_20260819`
메모리, θ=0.002 축소는 이미 기각됨). `scripts/eth_dc_theta_sweep_20260820.py`로
θ∈{0.004,0.006,0.008,0.010,0.012,0.015} 라벨경제성만 우선 스윕(학습 없음):
θ=0.006→0.008 경계에서 TP폭이 32.7→44.8bp로 뛰지만 이건 정확히 그 지점에서
`calibrate_barriers`가 pt_mult를 1.5→2.0으로 바꾼 그리드 경계 부수효과(cusum_k 스윕에서
이미 확인된 것과 동일 패턴) — 그 이후 θ=0.008→0.015 구간은 이벤트 62%가 더 줄어드는데도
TP폭은 44.8→46.6bp로 거의 안 움직인다. 게다가 애초에 θ=0.004의 <14bp 비율이 이미 2.6%로
문제가 없었으므로(앞선 정정 항목 참고) **경제성을 개선한다는 θ상향의 원래 동기 자체가
허상이었다.**

경제성과 별개인 "큰 스윙=더 많은 정보량" 가설도 직접 테스트했다
(`scripts/eth_dc_theta_raised_feature_information_content_20260820.py`, θ=0.015 —
가장 극단값, DC θ=0.004 검증과 동일 AUC+permutation null 방법론). 결과: n=1,817(θ=0.004의
15%)로 표본이 작아 raw max_auc는 오히려 높아보이지만(0.5293 vs 0.5141), permutation null
대비로는 더 비유의(empirical_p=0.825 vs 0.380) — 표본이 작아진 만큼 우연히 높은 AUC가
나올 여지만 커졌을 뿐 실제 정보량은 늘지 않았다. **dc_theta 축(경제성+정보량 둘 다)
완전 소진 — 이벤트 검출 임계값을 어느 방향으로 튜닝해도 해결책이 아니다.**

### 후보②(CUSUM) 정보량 검증 — 세션 첫 non-chance 결과, 그러나 2026단독 재검증에서 소멸 (2026-08-20 추가)

사용자 지시로 후보②(CUSUM+TB, cusum_k=1.0 기본값, 이미 빌드된 sparse 라벨)에 DC와 동일한
AUC+permutation null 검증을 적용했다(`scripts/eth_cusum_feature_information_content_20260820.py`).
**pooled(2025+2026, 55,851개 이벤트) 결과: max_auc=0.5105(DC보다 숫자는 작음)인데 표본이
커서 permutation empirical_p=0.000 — 이 서브프로젝트 전체에서 처음 나온 non-chance 결과.**
top5 전부 거래량/유동성계열(trades/volume_btc/quote_volume/sum_open_interest_value/
quote_volume_btc).

과대해석을 막기 위해 두 단계로 재검증했다:
1. **연도분할**(`scripts/eth_cusum_volume_signal_temporal_stability_20260820.py`): 2025(train,
   38,430개)는 5개 피쳐 다 "높을수록 SHORT" 방향으로 일관(raw_auc 0.486~0.494)되지만,
   2026(eval, 17,421개)에서는 4/5가 같은 방향이어도 크기가 0.5에 거의 다 붙어버리고
   (0.495~0.500 수준), volume_btc는 아예 방향이 반전(0.4870→0.5004)됐다.
2. **2026단독 독립 permutation null**(`scripts/eth_cusum_2026only_feature_information_content_20260820.py`,
   158개 전체를 2026 데이터만으로 새로 순위매김 — pooled top5로 범위를 미리 좁히지 않음):
   max_auc=0.5123(1위=sum_open_interest_value, pooled top5 중 유일하게 순위 유지)인데
   **empirical_p=0.290 — 비유의, DC θ=0.004(0.380)/θ=0.015(0.825)와 같은 급.** pooled top5
   중 나머지 4개는 2026단독 158개 중 53~132위로 추락 — pooled 유의성이 거의 전부
   2025(표본의 69%)에서 왔고 2026만 떼면 사라짐이 확정됐다.

**결론: "세션 첫 non-chance 결과"도 결국 이 저장소가 반복 확인해온 "VAL은 맞고 OOS는
약화/반전"과 같은 패턴이었다 — CUSUM도 DC와 동일하게 정보량 없음으로 확정, ①DC+②CUSUM
둘 다 CLOSED.**

### CUSUM에도 hp/aswa/bag 전부 적용 — 8개 독립앙상블 전부 chance (2026-08-20 추가)

사용자 지시로 DC에서 이미 검증한 hp/aswa/bag 3기법을 CUSUM에도 그대로 적용
(`scripts/eth_cusum_tabm_training_variant_runner_20260820.py`, 신규 15시드). 개별5시드
cond_acc: hp 49.0~50.5%/aswa 48.6~50.8%/bag 49.4~50.7% — DC와 동급, 전부 chance.
5시드앙상블: hp(LONG41.1%/acc50.7%/pnl+14.08/mdd−13.56), aswa(LONG58.3%/acc50.5%/
pnl−3.97/mdd−28.04), bag(LONG42.4%/acc49.9%/**pnl−30.52/mdd−37.66, 오늘 만든 8개 앙상블
전체 중 최악**). **DC+CUSUM 합쳐 8개 독립 5시드앙상블 전부 cond_acc 49.3~50.8%로 수렴,
PnL은 −30.52~+14.08로 라벨·기법 무관하게 계속 요동.** (분석 중 앙상블 조건부정확도 계산이
sparse 라벨을 reindex해 NaN을 "이벤트있음"으로 오카운트하는 버그를 발견·수정함 —
dense-cashfill 파일로 교체.)

### 레짐피쳐 제거 + 2024-2025 학습 확장 — 9번째 동일결론 (2026-08-20 추가)

사용자가 새 축 요청: 레짐 관련 피쳐 21개(`chop_index`/`cvp_regime`/`regime_trending`/
`regime_persistence` + `regime3_current_sensitive_wide24_*` 6개 + cmamba/stability/
transition/churn 11개)를 전부 제거하고, 학습 데이터를 2025 단독이 아니라 2024+2025로
확장. `scripts/eth_candidate_2024_2025_noregime_canonicaldata_20260820.py`로 구현:
TRAIN_CSV를 2024+2025 concat(210,481행)으로 교체, `_numeric_feature_cols`에 21개 deny-list
적용(158→137 base피쳐), `_read_labels`를 monkeypatch해 하드코딩된 "2025년 라벨" 호출을
2024+2025 결합으로 치환(같은 label_dir에서 두 해 다 읽어 하나로 합침 — 다른 라벨 계열
재사용 시에도 안전). 레짐 피쳐를 전부 버리므로 REGIME3_CURRENT도 CMAMBA/RISK와 동일하게
0-fill 처리 가능해져, 기존에 필요했던 "EVAL_CSV를 REGIME3_CURRENT_2026 실측 커버리지에
맞춰 사전필터링"도 불필요해짐(OOS 51,746→57,601행으로 오히려 복원).

신규5시드 학습 결과, `base_feature_count=137`(정확히 158−21) 확인, train 표본
78,605→183,985행(2.3배 확장) 확인 — 배선은 의도대로 정확히 작동했다. **그런데도 개별5시드
cond_acc=48.4~51.5%(DC원본 158피쳐/2025단독 베이스라인 48.2~51.4%와 사실상 동일), OOS부호
4승1패 혼재, 5시드앙상블 cond_acc=50.6%(chance)/pnl+10.60.** 피쳐를 줄이고 데이터를
2.3배 늘려도 결과가 안 바뀐다 — 라벨(DC/CUSUM)×기법(baseline/hp/aswa/bag/앙상블)×
피쳐셋(158/137)×학습데이터량(1년/2년) 전 조합에서 동일한 chance-level 결론이 9번째로
재확인됐다.

### 피쳐셋 연구(정리+조합) — 10/11번째 동일결론 (2026-08-20 추가)

사용자 요청: "새로운 피쳐셋은 더 이상 구할 수 없으니 정리하고 존재하는 피쳐를 잘 조합하는
수밖에 없다." 착수 전 이 전제 자체를 재확인했다 — raw L2 레벨(WS-E, exploratory 게이트
2026-09-14/승격 게이트 2026-11-17 미도달), 청산 tail_risk_1m(8주 데이터 게이트 2026-09-15
미도달)은 실제로 아직 못 쓰고, `microstructure_1m`(OFI 유래 34컬럼, 05-03부터 3.5개월+
누적)은 "소비자 0"이지만 **1분 단독 엔트리알파로는 이미 4회 기각됨**(`docs/model_contracts/
eth_candidate_lob_microstructure_data_resources_20260817.md` — contrarian flow 0.3~2bp <
비용 4~9bp, "게이트/청산타이밍 피처로만 재탐색 가치") — 방향-라벨링 축(이 서브프로젝트의
범위)에서는 사용자 전제가 맞았다.

**1단계 — 정리(리던던시 감사)**:
[scripts/eth_dc_feature_redundancy_audit_20260820.py](../../scripts/eth_dc_feature_redundancy_audit_20260820.py)
— 158개 base 피쳐를 2025+2026 풀링해 (a) 완전상수(std<1e-12) 컬럼을 먼저 분리, (b) 나머지에
Pearson |corr|≥0.95 union-find 클러스터링. 결과: **완전상수 11개**(`regime3_cmamba_h6_sidecar_*`
7개+`regime3_stability_h6_score`+`regime3_transition_h6_risk_{prob,pred}`+
`regime3_churn_h6_risk_score` — 노레짐 축에서 이미 "0-fill 플레이스홀더"로 알려졌던 바로 그
11개, 재확인) + **중복클러스터 11개**(원소 14개, OHLC 4종 corr=1.0 포함, 대표는 클러스터 내
개별AUC 최고피쳐로 결정론적 선정) → **133개로 정리**(158−11−14, 정리 후 잔존 최고
|corr|=0.9458<0.95 확인).

**2단계 — 조합(상호작용) 유의성 테스트**:
[scripts/eth_dc_feature_interaction_significance_20260820.py](../../scripts/eth_dc_feature_interaction_significance_20260820.py)
— 133개 전체쌍(C(133,2)=8,778개)마다 rank-percentile 상호작용항(z_i×z_j)의
direction-agnostic AUC를 계산(라벨무관 랭크는 1회만 계산 후 재사용, 퍼뮤테이션 200회는
행렬곱으로 일괄처리, 20초 완료). Pooled(2025+2026) 결과: 최고 AUC=0.5244
(`eth_btc_beta_residual_z × oi_up_price_down`), 귀무분포(8,778쌍 다중비교) 대비
**empirical_p=0.030(경계선 유의)**. CUSUM pooled 붕괴 전례를 따라
[scripts/eth_dc_feature_interaction_significance_2026only_20260820.py](../../scripts/eth_dc_feature_interaction_significance_2026only_20260820.py)로
2026 단독 완전독립 재검증(8,778쌍 처음부터 재랭킹, 다른 시드) — **pooled 1위 쌍의 2026단독
AUC=0.5165(거의 chance)로 주저앉고, 2026단독 1위는 아예 다른 쌍**(`cvd_slope_48 × hour_sin`,
auc=0.5368) **empirical_p=0.515(순수 chance 정중앙)**. Pooled 유의성은 2025(train기간)
쏠림 아티팩트로 판정 — CUSUM pooled(p=0.000→2026단독 p=0.290)와 동형.

**3단계 — 정리 효과 실측(실제 재학습)**:
[scripts/eth_dc_pruned133_canonicaldata_20260820.py](../../scripts/eth_dc_pruned133_canonicaldata_20260820.py)
+ [scripts/eth_dc_pruned133_training_runner_20260820.py](../../scripts/eth_dc_pruned133_training_runner_20260820.py)
— DC 원본 canonicaldata 래퍼(TRAIN/EVAL/오버레이 전부 그대로)에 `_numeric_feature_cols`만
133개 allow-list로 교체(피쳐셋 하나만 다른 단일변수 비교), epochs=2로 원본과 동일하게 고정.
신규5시드(N≥5 랜덤): **개별 cond_acc 48.7~49.9%**(DC원본 158피쳐 베이스라인 48.2~51.4%와
사실상 동일, 오히려 분산이 더 좁음), OOS부호 4승1패 혼재, **5시드앙상블 cond_acc=48.4%**
(pnl+14.23) — 정리해도 결과 불변.

**결론**: "정리"(158→133, 죽은컬럼+중복 제거)도 "조합"(명시적 상호작용 8,778쌍 전수조사)도
신호를 만들어내지 못했다. 조합 축에서 유일하게 나온 pooled p=0.030도 이 세션의 다른 모든
pooled-significant 결과(CUSUM p=0.000)와 마찬가지로 out-of-sample 재검증에서 완전히
붕괴했다 — 이 저장소 전체의 메타패턴(VAL/pooled 통과 → OOS/독립 반전)이 피쳐엔지니어링
축에서도 예외 없이 재현됐다.

### 문헌조사 기반 방법론 재검증 — 여전히 동일결론 (2026-08-20 추가)

사용자 요청으로 위 ad-hoc 방법론(correlation 임계값 클러스터링 + rank-product 상호작용) 자체를
문헌으로 검증. 전체 조사+인용 목록은
[docs/feature_redundancy_and_interaction_literature_review_20260820.md](../feature_redundancy_and_interaction_literature_review_20260820.md).
핵심만 요약: (1) 리던던시 제거 — 내 pairwise threshold 방법은 문헌상 single-linkage
클러스터링과 동치이며 스크리닝 목적으론 방어 가능하나, "3개 이상 피쳐에 분산된 다중공선성"은
구조적으로 못 봄(VIF가 필요). 실제 VIF 점검(`scripts/eth_dc_feature_vif_check_20260820.py`)에서
`regime3_current_sensitive_wide24_{bull,bear,chop}_prob` 3개가 VIF~3×10¹³ — 세 확률이 전
구간 예외없이 합=1.0(확률 단체 완전선형종속)인 게 실제로 발견돼 1개 제거(133→132). (2) 조합 —
내 rank-product+permutation-maxT는 문헌상 Aiken&West 곱상호작용항+Westfall-Young maxT와
정확히 동일한 표준기법으로 확인(정당함)됐으나, "매끄러운 단조" 형태만 봐서 비단조/트리형
상호작용은 구조적 사각지대. RIT식 LightGBM 트리구조 채굴(discovery는 2025단독,
`scripts/eth_dc_gbm_interaction_discovery_20260820.py`)로 top-30 후보를 뽑아 완전분리된
2026단독으로 재검증(`scripts/eth_dc_gbm_interaction_2026_validation_20260820.py`) —
**empirical_p=0.400(chance), 최고AUC=0.5232뿐**. 더 강력하고 비단조 형태까지 커버하는 방법으로도
여전히 신호 없음 — 정리+조합(선형+비선형) 전부 문헌 검증까지 마친 상태로 완전 소진.

### 엔지니어링 피쳐셋 구축(정리+조합+문헌표준피쳐) — 신호계산 보류 (2026-08-20 추가)

사용자 지시: "필요없는 피쳐를 제거하고 피쳐 조합으로 새로운 피쳐도 만들어줘, 아직 신호는
계산하지 마" → 이후 "금융시계열/코인 데이터 표준피쳐도 조사해서 추가해줘"로 확장. **이 절
전체는 AUC/permutation-null/PnL 등 신호 계산을 전혀 하지 않는다 — 순수 피쳐셋 구축+구조검증
(NaN/inf/기술통계)만 수행.**

1. **VIF 반복제거**(`scripts/eth_dc_feature_vif_iterative_elimination_20260820.py`):
   133개(리던던시감사) → 확률단체 1개 제거(132) → 매 스텝 최고VIF 피쳐 1개씩 제거+재계산을
   반복(상관행렬 역행렬 대각원소 방식, 133회 개별 OLS보다 빠름) → **VIF<10 수렴, 112개**(20개
   제거). 대표적 제거: `mtf_trend_1h`(VIF259.6), `last_funding_rate`(242.2), `log_return`
   (144.4). 흥미로운 점: 리던던시감사에서 OHLC클러스터 대표로 뽑혔던 `high`도 결국 제거됨(단일
   pairwise가 아니라 나머지 다수 피쳐 조합으로 여전히 거의 완전예측되는 사례, VIF가 정확히
   잡도록 설계된 케이스).
2. **RIT식 조합피쳐 구축**(`scripts/eth_dc_combination_feature_construction_20260820.py`):
   VIF-clean 112개로 LightGBM을 2025단독 학습(discovery가 2026을 오염시키지 않도록), 트리
   조상-자손 공동출현 상위 30쌍에 대해 `combo_a_x_b = raw_a * raw_b` 실제 컬럼 생성(2025+2026
   전체 bar). 구조검증: NaN/inf 없음.
3. **문헌 갭분석**(리서치에이전트, 15회 웹서치+실제 피쳐코드(`features/engineering.py` 등)
   대조 검증): López de Prado 3종(분수차분/SADF/엔트로피), 고전 마이크로구조 추정량
   (Corwin-Schultz/Roll/Kyle's Lambda/VPIN), 고차모멘트(실현semivariance/실현첨도),
   분산비율검정(Lo-MacKinlay), 멀티프랙탈DFA, 전이엔트로피, Hawkes가 158개 중 부재로 확인됨
   (`ofti`/`entropy`처럼 이름이 비슷해 보이는 기존 피쳐도 실제 코드 대조 결과 다른 공식임을
   확인). 고전 기술지표(RSI/MACD/BB/MA)는 2026-08-17에 이미 문헌검증+CLOSED라 재조사 안 함
   (`eth_classical_technical_indicator_literature_check_20260817` 메모리).
4. **구현+구축**(`scripts/eth_dc_financial_ml_feature_construction_20260820.py`): 계산복잡도
   낮은 9개 계열 → 13개 계산했으나 자기상관 확인 후 12개 확정(분수차분 d∈{0.3,0.5,0.7} 3개,
   return-sign 엔트로피, Corwin-Schultz, Roll스프레드, Kyle's Lambda, VPIN근사, 실현
   semivariance비율, 실현첨도, 분산비율 q∈{4,12}). SADF/멀티프랙탈DFA/전이엔트로피/Hawkes
   4종은 구현복잡도+런타임비용으로 이번 패스 제외(향후 후보로 기록). 자기검증(self-test,
   신호 아님): FFD 가중치 부호, 백색잡음 분산비율≈1, Corwin-Schultz 비음수 — 전부 통과.
   구조검증 중 `roll_implied_spread`가 원시가격차분(달러단위)이라 2025/2026 가격대 차이로
   스케일 드리프트하는 문제 발견 → log_return 기반으로 수정(Corwin-Schultz와 동일하게 비례
   단위로 통일).
5. **최종 통합 wrapper**(`scripts/eth_dc_engineered_features_canonicaldata_20260820.py`):
   VIF-clean112 + 조합30 +
   financial-ML12 = **154개**. `omega._load_omega_frames`/`_numeric_feature_cols` 오버라이드로
   실제 학습에 바로 연결 가능. 스모크검증: 154개 전부 로드, train/eval 양쪽 컬럼 누락 없음,
   NaN>1% 컬럼 없음.

**⚠️ 아직 아무 학습/AUC/permutation-null도 실행하지 않았다 — 사용자가 다음에 신호계산을
지시하면 이 wrapper(`eth_dc_engineered_features_canonicaldata_20260820.py`)를 그대로
학습러너에 꽂으면 된다.**

### 154피쳐 엔지니어링셋 신호계산 — 역시 chance, 축 완전종료 (2026-08-20 같은 날 후속)

사용자 지시("그렇게 해줘")로 신호계산 착수. **1단계**: 진짜 신규인 42개(조합30+금융ML12)만
개별 정보량 체크(`scripts/eth_dc_new42_feature_information_content_20260820.py` — 112개
VIF-clean은 158개 전체가 이미 개별 비유의 확인됐으므로 재검증 생략) → **empirical_p=0.460
(chance), 최고AUC=0.5123뿐.** **2단계(결정적 테스트)**: 154피쳐 전체로 실제 TabM 재학습
(`scripts/eth_dc_engineered154_training_runner_20260820.py`, DC원본과 동일 TRAIN_CSV/
EVAL_CSV/epochs=2, 신규5시드) → `scripts/eth_dc_engineered154_analysis_20260820.py`로 분석.

**결과: 개별5시드 cond_acc 48.8~51.0%**(DC원본158피쳐 베이스라인 48.2~51.4%와 사실상 동일),
**OOS부호 3승2패 혼재, 5시드앙상블 cond_acc=49.7%(chance) pnl=−19.96 mdd=−30.27**(이번
세션에서 나온 앙상블 중 하위권).

**정리(리던던시제거+VIF)도, 조합(bilinear 8,778쌍+비단조 RIT 30쌍)도, 신규 문헌표준
피쳐(financial-ML 12종)도 — 개별로도 조합으로도 실제 모델 재학습으로도 — 전부 chance로
수렴했다. 이걸로 "158개 캐노니컬 피쳐 유니버스에서 피쳐 엔지니어링으로 신호를 만들어낼 수
있는가"라는 질문 자체가 문헌검증까지 거쳐 완전히 닫혔다고 판단한다.** 이 154피쳐 wrapper와
구축 스크립트 일체는 향후 참고용으로 보존(리던던시제거/VIF/RIT조합/financial-ML 구축 패턴은
다른 라벨·다른 자산에도 재사용 가능한 일반적 방법론이므로) — 정식 등록:
`docs/model_contracts/research_line_registry.json`(id=`eth_dc_feature_engineering_
redundancy_combination_finml_20260820`), 전체 파생체인:
`docs/model_contracts/eth_dc_engineered_feature_set_lineage_20260820.json`.

**2차 독립 시드배치 재확인(같은 날 후속, 사용자 지시)**: 완전 무교집합 신규5시드로 동일
스크리닝 재실행(`scripts/eth_dc_engineered154_batch2_analysis_20260820.py`). **개별
cond_acc 48.4~49.9%**(1차 48.8~51.0%와 동일 chance대역, 재현됨), **OOS부호 4승1패**(1차
3승2패, 둘다 혼재). **앙상블은 1차(cond_acc49.7%/pnl−19.96/mdd−30.27) vs 2차
(cond_acc49.2%/pnl+25.84/mdd−19.42) — 정확도는 재현되는데 앙상블 PnL은 배치마다 정반대로
뒤집힘**, 이 세션에서 여러 번 확인된 "정확도는 chance로 안정, 앙상블 손익은 시드조합운"
패턴이 154피쳐셋에서도 동일하게 재현됨 — chance 결론의 신뢰도를 한번 더 보강.

## 다음 단계 (미착수)

계획서의 사전등록 기준("N=5 전부 OOS 부호 일치해야 2단계 투자")에 따라 **TabM DC 후보는
여기서 스크리닝 기각**한다. N-HiTS도 N=1 결과가 이미 always_short에 진 상태라 추가 시드
투자 우선순위는 낮다. 라벨(DC/CUSUM)×기법(baseline/hp/aswa/bag/앙상블)×피쳐셋
(158/154/137/133/132/112, 리던던시제거/VIF/레짐제거/bilinear조합/RIT조합/financial-ML문헌
피쳐 전부 포함)×학습데이터량(1년/2년) 전 조합 + 명시적 피쳐 상호작용(선형+비선형) 전수조사 +
문헌검증된 신규 피쳐 구축까지 동일한 chance-level 결론에 도달한 상태다. **피쳐 엔지니어링
축(정리+조합+신규피쳐추가)은 문헌검증과 실제 재학습까지 마치고 완전히 닫혔다.** 남은
선택지:
1. 후보③(분포적 회귀)도 동일한 TabM 단일모델+레짐피쳐 절차로 학습해볼지 — 단, 이 결과를
   보면 라벨을 더 바꿔도 같은 병목(158개 캐노니컬 피쳐 자체의 방향정보 부재, 정리/조합/신규
   피쳐추가로도 안 뚫림)에 부딪힐 가능성이 높다는 걸 감안해야 함.
2. 피쳐셋 자체를 질적으로 다른 정보원으로 바꾸는 축(LOB raw L2 exploratory 게이트
   2026-09-14, 청산 tail_risk_1m 게이트 2026-09-15) — 게이트 도달 전까지는 착수 불가.
3. ModernTCN 풀 검증(서버 GPU, N≥5시드, 레짐분할)은 이 예비 결과들을 볼 때 투자 우선순위가
   더욱 낮아짐 — 사용자 판단 필요.
4. `eth_tabm_label_logic_retest_initiative_20260819` 서브프로젝트 자체를 이 지점에서 잠정
   종료할지 검토.

## 참고

- `docs/experiments/eth_directional_change_labels_20260819.md` — DC 라벨 빌드(후보①)
- `docs/model_contracts/eth_candidate_nhits_moderntcn_direction_quality_contract_20260816.md` —
  N-HiTS/ModernTCN 원 계약(h48qual 라벨, 이번과 다른 축)
- `docs/label_methodology_survey_20260815.md` — 40+ 선행 라벨 방법론 메타발견
- 메모리: `eth_tabm_label_logic_retest_initiative_20260819`
