# ETH Omega4.6.1 TCN 기반 exit_head (2026-08-13, Odyssey2 #5)

상태: `tested_negative_closed` — **VAL 사전등록 게이트(컴포넌트+포트폴리오 레벨, PnL·MDD 넷 다
비악화) 실패로 OOS 미실행.** 포트폴리오 레벨만 보면 baseline(TabM 라이브ATR exit_head) 대비
PnL·MDD 둘 다 개선되지만(+46.59%→+60.24%, -21.70%→-21.64%), **컴포넌트 레벨(h48qual 단독)에서
PnL이 아예 마이너스로 반전**(+9.23%→**-7.74%**, -16.97pp)하고 MDD도 악화(-7.59%→-8.28%)해
사전등록 기준 4개 지표 중 2개를 충족하지 못한다. #4(GBDT)와 같은 계열(컴포넌트 악화·포트폴리오
개선)의 실패지만 컴포넌트 악화 폭이 GBDT보다 훨씬 크다(GBDT는 +9.23%→+2.72%로 양의 영역에
남았으나 TCN은 음전환). 규율대로 OOS는 절대 열지 않았다(코드가 `RuntimeError`로 실행 자체를
거부함을 직접 확인).

## 배경

Odyssey2 우선순위 큐 #5(마지막 항목): h48qual의 exit_head를 현재 확정 베이스라인인 TabM(라이브
ATR 배리어 재라벨, `docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`)
대신 **TCN(temporal convolutional network, 시간축 윈도우 기반)**으로 학습시키면 같은 라벨/
데이터셋에서 다른 결과가 나오는가를 검증한다. TCN은 Odyssey(1) Phase1에서 `direction_head`
3클래스 분류 대상으로 시도됐을 때 "이 세션 전체에서 유일하게 완전 셧아웃이 아니었던"(VAL 접전)
결과였으나, 이후 정밀 재탐색(HP 전체 탐색 150 trial × 5피쳐셋 + N=5시드 최종검증,
`docs/experiments/eth_h48qual_tcn_hpsearch_multivariant_20260812.md`)에서 OOS 0/75로 결정적으로
부정됐다. Odyssey2 계약서의 "Phase 1 아이디어 → post-entry 재적용 트리아지"는 이를
"재시도 가치 있음, 최우선급"으로 분류했다 — "포지션이 이미 열린 뒤 최근 시퀀스가 청산 타이밍에
도움되는가"는 "다음 bar 방향을 맞히는가"와 질적으로 다른 질문이라는 논리다. 방금 완료된 #4
(GBDT)와 동일한 라벨/데이터셋/평가 규율을 그대로 적용하되, 통합 방식만 TCN의 구조적 요구사항에
맞게 다르게 했다(아래 "⚠️ #4와의 핵심 차이점" 절). zig075는 이 실험에서 건드리지 않는다.

비교 대상은 원본 라이브 h48qual 번들이 아니라, #4(GBDT)와 동일하게 **Odyssey(1)이 만든 신규
exit_head 번들**(`tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/
h48qual/true_3head_tabm_bundle.pt`, VAL PnL+46.59%/MDD-21.70% — 현재 Odyssey2가 채택한 확정
베이스라인)이다.

## 방법

### 데이터셋 — #4(GBDT)가 만든 함수를 IMPORT로 재사용, 재구현 없음

`scripts/train_eval_omega461_gbdt_exit_head_liveatr_20260813._build_dataset(1500)`을 **수정 없이
직접 import해서 호출**했다 — 이 함수 자체가 이미
`research_eth_omega461_exit_head_liveatr_relabel_20260813.py`의
`_fast_timescale_checkpoint`/`_build_exit_dataset_entry_label_live_atr_barrier`를 재사용해 TabM
`full1500` 런과 동일 시드(260813)·후보수(1500)로 데이터셋을 재구축하고 원본 `report.json`의
`dataset` 블록과 대조까지 마친 함수이므로, 이걸 그대로 호출하면 #4(GBDT)가 이미 확인한
"행수 1,234,431·양성 245,600·후보 1,500 정확히 일치"가 **코드 재사용을 통해 자동으로 보장**된다
(재실행으로 재확인: 아래 결과 표).

### 윈도잉 — 이 스크립트에서만 추가된 신규 로직

TCN이 필요로 하는 "과거 WINDOW bar 시퀀스"는 GBDT/TabM이 쓰는 단일-행 데이터셋에는 없는
정보이므로, 다음을 새로 구축했다:

1. 데이터셋의 각 행이 속한 **원본 시계열 절대 bar 인덱스**(`row_i`)를
   `exit_path_entry_i + exit_path_hold_bars`로 복원했다(`_build_exit_dataset_entry_label_live_atr_
   barrier`가 두 컬럼 모두 이미 `frame_exit`에 남기고, `row_i in range(entry_i, barrier_end_i+1)`
   루프로 동일 `frame`을 순회하므로 이 합이 정확히 절대 인덱스와 같다). 복원한 `row_i`로
   `frames["train_df"]["timestamp"].iloc[row_i]`를 인덱싱해 `frame_exit["timestamp"]`와
   **전체 행에 대해 벡터화 비교**했고(샘플링 아님), 정확히 일치함을 확인했다(아래 결과 표).
2. `parent._base_input(frames["train_df"], base_cols)`로 전체 TRAIN 구간(2024-06-01~2025-09-30,
   약 4.2만 행)의 102차원 시장피쳐 행렬(`market_np`)을 만들고, 무작위 500행×20컬럼 샘플에서
   `market_np[row_i]`가 GBDT/TabM이 실제로 학습에 쓴 `x_exit_raw`의 `cur_<col>` 값과 정확히
   일치하는지 대조했다(0/10,000 셀 불일치) — "같은 데이터에 히스토리만 추가된 것"임을 원본
   재구현이 아니라 직접 대조로 보장한다.
3. 각 데이터셋 행의 윈도우는 `market_std[row_i-WINDOW+1 : row_i+1]`(표준화 후, 왼쪽 부족분은
   0-패딩)이다. **position-state(13차원, pos_side/pos_hold_bars/...)는 시퀀스에 포함하지
   않는다** — 윈도우 내 과거 bar들에 대해서는 "그 시점에 실제로 열려 있던 포지션의 mfe/mae/
   hold_bars" 값 자체가 애초에 존재하지 않기 때문이다(리플레이 중 현재 열린 포지션에 대해서만
   온라인으로 계산됨). 대신 pos 벡터는 TCN의 pooled 출력과 concat되는 별도 스칼라 브랜치로
   들어간다(아키텍처 절 참고) — 오케스트레이터가 명시적으로 지시한 설계.

### 아키텍처 — Phase1 TCN 재사용 + 신규 pos 브랜치

`CausalConv1d`/`TCNBlock`(dilated causal conv, `scripts/verify_eth_h48qual_tcn_sequence_model_
20260812.py`/`scripts/tune_eth_h48qual_tcn_sequence_model_hpsearch_20260812.py`)을 그대로 재사용.
하이퍼파라미터는 그 전체탐색(Optuna 150 trial × 5피쳐셋, `docs/experiments/
eth_h48qual_tcn_hpsearch_multivariant_20260812.md`)의 `raw_lite`(원본 8피쳐 테마, 이번 102피쳐
시장 시퀀스와 가장 가까운 테마) 채택값을 그대로 썼다: `window=48, hidden=32, n_blocks=5,
kernel_size=5, dropout=0.2998, lr=0.002484, weight_decay=0.000890, batch_size=1024,
optimizer=Adam`. Phase1의 `TCNClassifier`는 3클래스 방향 분류였고 position-state 입력이 없었다 —
`TCNExitClassifier`는 pooled TCN 출력(32차원)에 position-state MLP 브랜치(13→16, ReLU) 출력을
concat한 뒤 작은 MLP 헤드(`Linear(48,32)→ReLU→Dropout→Linear(32,2)`)로 이진(hold/exit) 로짓을
내는 신규 확장이다 — 별도로 튜닝되지 않았음을 명시(오케스트레이터가 "합리적이면 충분, 완벽할
필요 없음"으로 허용한 범위).

레짐별(bull/bear/chop) 3개 분리 모델, `compute_sample_weight(balanced) × 그 레짐의 소프트
Regime3 라우팅확률` 가중치 — #4(GBDT)/TabM `_fit_exit_head_only`와 동일한 스킴.

## ⚠️ #4(GBDT)와의 핵심 차이점 — 왜 duck-typing만으로는 안 됐는가

#4(GBDT)는 `train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one`이 단일 115차원 행
(`row = base_np[row_i]`)만 모델에 넘기는 호출 계약을 그대로 만족했기 때문에(`model(x)` 자리에
GBDT를 duck-typing으로 끼우는 것만으로 충분), `_predict_exit_prob_one` /
`research_eth_omega461_exit_sweep_20260721.replay_exit_variant` /
`replay_omega4_6_1_greedy_router_20260706.greedy_replay` **셋 다 한 글자도 수정하지 않고** 그대로
재사용할 수 있었다.

TCN은 구조적으로 다르다 — 매 bar 결정마다 과거 48bar 윈도우가 필요한데,
`_predict_exit_prob_one`은 `base_np`(전체 시계열)와 `row_i`(현재 인덱스)를 인자로 갖고 있음에도
**모델 호출 직전에 단일 행만 슬라이스**해 넘긴다. 이 함수를 그대로 재사용하면 TCN이 윈도우를 볼
방법이 없다.

**해결**: 기존 세 함수를 수정하지 않고, **이름을 바꾼 복사본**을 `research_eth_omega461_
tcn_exit_head_val_20260813.py`에 만들었다 — `_predict_exit_prob_one_windowed`,
`replay_exit_variant_windowed`, `greedy_replay_windowed`. 로직은 100% 동일하게 유지하되 exit_head
호출부만 `base_np[max(0,row_i-WINDOW+1):row_i+1]` 윈도우 슬라이스로 바꿨다(원본과의 diff를
`diff -u`로 직접 확인 — 함수명/시그니처/docstring/호출부 외 변경 없음, 아래 "준수 확인" 참고).

포트폴리오 레벨에서는 추가 문제가 있었다: `greedy_replay`도 **자기 자신의 호출부에서**
`_predict_exit_prob_one`을 직접 부른다 — 컴포넌트 레벨과 별개의 구조적 제약이다. 게다가
포트폴리오 리플레이는 h48qual(TCN)과 zig075(원본 TabM)를 **같은 루프 안에서** 번갈아 다루므로,
`greedy_replay_windowed`가 무조건 윈도우를 만들면 zig075의 진짜 TabM 모델이 3차원 텐서를 받아
바로 shape 불일치로 죽는다(조용한 오염이 아니라 즉시 크래시). 그래서
`greedy_replay_windowed`는 매 bar 활성 컴포넌트의 모델 객체에 `IS_WINDOWED=True` 마커가 있는지
(`TCNExitHeadWrapper`만 설정)를 보고 윈도우 경로/원본 경로를 **동적으로 분기**한다 — h48qual은
윈도우 경로, zig075는 원본 `rs._predict_exit_prob_one` 그대로.

`TCNExitHeadWrapper`(#4의 `GBDTExitHeadWrapper`와 동일한 "harness는 항등 mean=0/std=1만 넘기고
wrapper가 자체 표준화" 관례)는 윈도우 원시 텐서 `(T=48, 115)`를 받아 시장피쳐 부분(앞 102열,
전체 T)과 position-state 부분(뒤 13열, **마지막 행만**)으로 분리, 각자의 스케일러로 표준화한 뒤
`TCNExitClassifier(seq, pos)`를 호출하고 2-클래스 로짓을 `(batch, k=1, 2)`로 재구성해 TabM용
softmax/앙상블-풀링 로직이 그대로 재현되게 한다(#4가 `log(predict_proba)` 트릭을 썼던 자리에
여기선 진짜 로짓을 그대로 씀 — 더 자연스러움).

**"복사본을 만들어 원본은 무수정으로 남긴다"는 원칙을 실행 전/후 `git diff`로 직접 확인**했다
(아래 "준수 확인").

## 결과

### G0 자체검증 — 100% 기존 코드로 알려진 수치 재현

TCN을 평가하기 전에, 이 스크립트의 하네스가 **기존에 발표된** baseline/TabM-liveATR 수치를
그대로 재현하는지 먼저 확인했다(#4와 동일하게 `h48cons._evaluate_val`,
`research_eth_omega461_exit_head_portfolio_asymmetric_20260813.run_variant`를 그대로 호출).

| | 컴포넌트 baseline(원본) | 컴포넌트 TabM 라이브ATR | 포트폴리오 baseline(원본) | 포트폴리오 TabM 라이브ATR |
|---|---:|---:|---:|---:|
| 발표된 수치 | +5.45% / -11.62% / 29건 | +9.23% / -7.59% / 63건 | +36.82% / -24.34% / 29건 | +46.59% / -21.70% / 35건 |
| 이 하네스 재현값 | +5.45% / -11.62% / 29건 | +9.23% / -7.59% / 63건 | +36.82% / -24.34% / 29건 | +46.59% / -21.70% / 35건 |

4개 지표 전부(PnL·MDD·거래수) **정확히 일치** — G0 통과. 이후 TCN 수치를 신뢰 가능한 것으로
취급했다.

### 데이터셋/윈도잉 자체검증 — 3중 확인

| 확인 항목 | 결과 |
|---|---|
| 데이터셋 레퍼런스 대조(행수/양성개수/사용후보수, GBDT `full1500` report.json 대비) | 전부 일치(1,234,431행 / 245,600양성 / 1,500후보) |
| `row_i`(=`exit_path_entry_i`+`exit_path_hold_bars`) ↔ `frame_exit["timestamp"]` 벡터화 대조 | 전체 1,234,431행 100% 일치 |
| `market_np[row_i]` ↔ `x_exit_raw`의 `cur_<col>` 값 대조(500행×20열 무작위 샘플) | 10,000셀 중 불일치 0건 |

### 학습 진단 — 레짐별 3개 TCN, GBDT와 비슷한 수준의 held-out 판별력

| 전문가 | train/val(eval subsample) 행수 | val AUC | val logloss | 학습 시간 |
|---|---:|---:|---:|---:|
| bull | 1,049,266 / 30,000 | 0.9978 | 0.0423 | 491s |
| bear | 1,049,266 / 30,000 | 0.9980 | 0.0494 | 491s |
| chop | 1,049,266 / 30,000 | 0.9974 | 0.0427 | 491s |

세 전문가 모두 25 epoch(조기종료 미도달, `patience=6` 안에서 최적 epoch로 롤백)까지 학습했고
held-out AUC≈0.997~0.998로 #4(GBDT, AUC≈0.998)와 거의 같은 수준의 판별력을 보인다 — 이 라벨이
단순 임계값 규칙(`pos_giveback≥0.65`, `pos_unrealized≤-0.010` 등)으로 정의돼 있어 표현력 있는
모델이면 거의 다 완벽히 학습 가능한 구조라는 #4의 관찰이 TCN에도 그대로 적용된다. 데이터셋 빌드
638초(#4/TabM 대비 원본 재구현 없이 재현) + 학습 3×491초 ≈ 총 35분(CPU-only, 이 dev 박스에
CUDA 없음 — 학습 스크립트의 module docstring에 실측 처리량 근거 기록: 배치1024/window48 기준
~4,300행/초, epoch당 학습 서브샘플 8만행·held-out 평가 서브샘플 3만행으로 예산 제한, Phase1
TCN 스크립트도 GPU에서조차 동일한 epoch당-서브샘플 방식을 썼음) — **파일럿 축소 없이 전체
1500후보/1,234,431행 데이터셋으로 완주**했다.

### 컴포넌트 레벨(h48qual 단독, VAL 2025-10-01~12-31) — GBDT보다 더 크게 악화

| | TabM 라이브ATR(baseline) | TCN |
|---|---:|---:|
| PnL | +9.23% | **-7.74%**(악화, -16.97pp, **부호 반전**) |
| MDD | -7.59% | **-8.28%**(악화, -0.69pp) |
| 거래수 | 63 | 186 |
| 승률 | 30.2% | 48.4% |
| 평균 보유기간 | 210.8bar | **11.0bar** |
| max_trade_pnl | 4.47% | 0.45% |
| exit_reasons | `exit_head:52, take_profit:8, stop_loss:3` | `exit_head:186`(100%, TP/SL 전무) |

### 포트폴리오 레벨(h48qual+zig075 단일계좌 우선순위, 동일 VAL) — 개선(단 GBDT보다는 작은 폭)

| | baseline(둘 다 원본) | TabM 라이브ATR(현재 확정 베이스라인) | TCN |
|---|---:|---:|---:|
| PnL | +36.82% | +46.59% | **+60.24%**(TabM 대비 개선, +13.65pp) |
| MDD | -24.34% | -21.70% | **-21.64%**(TabM 대비 개선, +0.06pp, 사실상 동일) |
| 거래수 | 29 | 35 | 45 |
| 승률 | 41.4% | 37.1% | 40.0% |
| 평균 보유기간 | 676.5bar | 551.2bar | 415.4bar |
| exit_reasons | `stop_loss:17, take_profit:12` | `take_profit:13, stop_loss:13, exit_head:9` | `exit_head:19, stop_loss:14, take_profit:12` |
| 슬롯 승자(source_component) | `zig075:22, h48qual:7` | `zig075:22, h48qual:13` | `zig075:26, h48qual:19` |

### 사전등록 게이트 판정

| 지표 | 판정 |
|---|---|
| 컴포넌트 PnL 비악화 | **FAIL** |
| 컴포넌트 MDD 비악화 | **FAIL** |
| 포트폴리오 PnL 비악화 | PASS |
| 포트폴리오 MDD 비악화 | PASS |
| **종합 게이트** | **FAIL** (4개 중 2개 미충족) |

`scripts/research_eth_omega461_tcn_exit_head_oos_20260813.py` 실행 결과, VAL report.json에서
`gate_pass=False`를 읽고 OOS 데이터를 전혀 로딩하지 않은 채 즉시 `RuntimeError`로 중단됨을
실행으로 확인했다(#4와 동일한 코드-강제 패턴):

```
RuntimeError: VAL gate_pass=False -- TCN did not beat the TabM live-ATR baseline on VAL
(component+portfolio, PnL+MDD both non-worse). Per this project's methodology discipline, OOS
must not be opened when the VAL gate fails. ...
```

### 리플레이 원장 직접 확인 — 조기청산이 진짜 학습된 행동인지(버그 배제)

극단적인 컴포넌트 결과(평균 보유 11bar, exit_head 100%)가 버그(예: 상수 출력, 인덱싱 오류)가
아니라 실제로 학습된 행동인지 확인하기 위해 포트폴리오 원장(`portfolio_ledger_asymmetric_
h48qual_tcn_zig075_original.csv`)의 h48qual 귀속 거래(19건) `hold_bars`를 직접 봤다: 0~57bar
범위에 평균 12.5·표준편차 15.3으로 **뚜렷한 분산**이 있다(0bar 즉시청산도 있지만 6/20/25/39/57bar
등 다양한 시점도 있음) — 모델이 컨텍스트에 따라 다르게 반응하고 있다는 뜻이므로 상수출력 버그는
배제된다. 데이터셋 레퍼런스 대조·`row_i` 전체행 일치·`cur_` 값 무작위 대조·G0 4개 지표 정확
일치까지 겹쳐 확인했으므로, 이 결과는 하네스 버그가 아니라 TCN이 실제로 학습한 정책으로
취급한다.

## 해석 — 왜 TCN이 GBDT보다 더 공격적으로 조기청산을 학습했는가(추정)

#4(GBDT) 문서가 이미 지적했듯, 이 라벨은 `pos_giveback≥0.65`·`pos_unrealized≤-0.010` 같은 단순
임계값 규칙이 지배적이라(`mfe_giveback_exit` 75.6%·`adverse_unreal_exit` 22.5%) 표현력 있는
모델이면 거의 무엇이든 held-out AUC≈0.998까지 학습 가능한 구조다. GBDT(115차원 단일 행 입력)와
TCN(48bar×102차원=4,896값 시퀀스 입력)이 **거의 동일한 held-out 판별력**에 도달했다는 점이
이것을 뒷받침한다 — 판별력 차이가 아니라 결정 경계의 **위치**가 갈린다는 #4의 결론이 여기서도
반복된다.

다만 TCN의 컴포넌트 악화가 GBDT보다 뚜렷이 더 크다는 사실(평균 보유 210.8→144.9bar(GBDT,
-31%) vs 210.8→11.0bar(TCN, -95%); exit_head 발동비중 82.5%→91.5%(GBDT) vs →100%(TCN))은 입력
차원과 관련 있을 가능성이 있다 — **추정이며 이번 실험 범위에서 직접 검증되지 않았다**: TCN은
GBDT보다 42배 많은 입력값(4,896 vs 115)을 받으므로, TRAIN 구간에서 단순 임계값 규칙과
우연히 상관된 시장-윈도우 패턴을 GBDT보다 더 많이 흡수할 여지가 있고, 이 패턴들이 VAL 구간의
다른 시장 상태에서 재현되지 않으면 결정 경계가 더 쉽게 어긋난다는 가설이다. `raw_lite`(8컬럼)
HP탐색이 이미 direction_head 맥락에서 보여준 "피쳐가 많을수록(=입력 차원이 클수록) VAL 성적이
나빠진다"는 패턴(`docs/experiments/eth_h48qual_tcn_hpsearch_multivariant_20260812.md`)과 같은
방향이라는 점도 이 해석과 정합적이다.

포트폴리오 레벨 개선(+46.59%→+60.24%)은 #4가 확인한 것과 같은 슬롯-재순환 상호작용(h48qual이
슬롯을 더 자주 비워줘 재진입 기회 증가, 슬롯 승자 13건→19건)으로 보이나, TCN 컴포넌트 거래의
`max_trade_pnl`이 0.45%(TabM 4.47%, GBDT 4.20%)까지 짓눌려 있어 GBDT(+54.68pp)보다 개선 폭이
작다(+13.65pp) — 너무 짧게 잡는 거래는 슬롯을 비우는 효과는 있지만 그 거래 자체의 기여가
GBDT보다도 작다는 뜻이다.

## 결론

**채택 불가.** VAL 사전등록 게이트(컴포넌트+포트폴리오, PnL+MDD 넷 다 비악화)를 통과하지
못해 규율대로 OOS를 열지 않았다. 포트폴리오 레벨 수치만 보면 개선(+46.59%→+60.24%)이지만,
컴포넌트 단독 economics가 **부호까지 반전**될 정도로 나빠진 상태에서 나온 결과라 "TCN
exit_head가 더 낫다"는 결론을 지지하지 않는다 — #4(GBDT)와 같은 계열의 "레벨별 반대 방향이라
어느 쪽도 깔끔한 승리가 아니다"이지만, TCN은 GBDT보다 컴포넌트 악화가 훨씬 심하고 포트폴리오
개선은 더 작아 **GBDT보다도 못한 트레이드오프**다. Odyssey2 계약서의 우선순위 큐 #5는 이것으로
**종결**한다(부정 결과) — 이로써 **우선순위 큐 1~5 전부 소진**됐다(계약서 갱신, 아래 참고).

## 미해결 / 다음 단계

- "TCN이 GBDT보다 입력 차원이 커서 더 쉽게 과적합한다"는 해석은 **추정**이며, 이번 실험에서
  직접 검증(예: 윈도우 길이를 줄여가며 컴포넌트 악화 폭이 완화되는지 보는 ablation)하지 않았다
  — 오케스트레이터 지시 범위 밖.
- exit_head 임계값(`EXIT_THRESHOLD=0.95` 고정, 전 모델 공통 적용)을 TCN에 맞게 재보정하면
  컴포넌트 레벨 조기청산이 완화될 가능성이 있으나, 사전등록 기준에 없던 사후 튜닝이라 이번엔
  시도하지 않았다 — 시도한다면 새로운 사전등록·새 VAL-then-OOS 사이클이 필요하다(#4 문서와
  동일한 유보).
- position-state를 pooled TCN 출력과 concat하는 브랜치 구조는 새로 설계된 것으로 별도 HP
  탐색을 거치지 않았다(오케스트레이터가 허용한 범위) — 다른 pos_hidden/head_hidden 조합이나
  concat 대신 FiLM 등 다른 결합 방식이 더 나을 가능성은 미탐색.
- 채택 가능한 변경 0건, 라이브 파일 미변경.
- Odyssey2 계약서 "우선순위 큐"(1~5)가 이 실험으로 전부 소진됨 — 다음 단계는 계약서에 명시된
  대로 "최신 논문 기반 신규 아이디어 탐색"으로 전환한다(이번 세션 범위 밖).

## 준수 확인

`fresh_forward_bar_by_bar=true`(데이터셋 구축은 #4가 이미 검증한 causal 배리어 시뮬레이션을
import로 재사용, VAL/컴포넌트/포트폴리오 리플레이는 `replay_exit_variant_windowed`/
`greedy_replay_windowed` 단일 순방향 루프 — 추가된 윈도우 lookback도 항상 이미 확정된 과거
bar만 포함, 미래 bar 없음). `trade_ledgers_used_as_input=false`.
`saved_parent_exit_timestamps_used=false`. `future_rows_used_for_entry=false`.
direction_head/quality_head/encoder 전부 동결·미변경(TCN은 exit_head만 대체). zig075 미변경.

`git diff`로 확인(작업 시작 전/후 모두 0줄): `scripts/research_eth_omega461_exit_sweep_20260721.py`,
`scripts/train_eval_omega4_2_risk_sidecar_20260622.py`,
`scripts/replay_omega4_6_1_greedy_router_20260706.py`, 라이브 파일
(`trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`,
`trading_bot_modules/runtime_config.py`, `.env`).

direction_head/quality_head/quality_threshold 전부 동결(원본/TabM/GBDT/TCN 네 변형 전부 동일).
`EXIT_THRESHOLD=0.95` 고정 유지. `git diff`로 `scripts/train_eval_omega461_gbdt_exit_head_
liveatr_20260813.py`(import로만 재사용)도 무변경 확인. **OOS(2026-01-01~03-31)는 게이트 실패로
전혀 로딩되지 않았다**(`research_eth_omega461_tcn_exit_head_oos_20260813.py` 실행 시
`RuntimeError`로 즉시 중단됨을 직접 확인, 위 결과 절 인용). Seed-Diversity Ensemble Promotion
Gate는 해당 없음 — #4(GBDT)와 동일하게 레짐별 단일시드 비교이며 다중시드 앙상블 승격을 주장하지
않는다.

## 산출물

- 새 스크립트:
  - `scripts/train_eval_omega461_tcn_exit_head_liveatr_20260813.py` — 데이터셋 재사용(import) +
    row_i 복원/대조 + 윈도우 표준화 + `TCNExitClassifier` 정의 + 레짐별 3개 학습 + 번들 저장.
  - `scripts/research_eth_omega461_tcn_exit_head_val_20260813.py` — G0 자체검증 +
    `TCNExitHeadWrapper`(duck-typing) + `_predict_exit_prob_one_windowed`/
    `replay_exit_variant_windowed`/`greedy_replay_windowed`(무수정 원본의 이름바꾼 복사본) +
    컴포넌트/포트폴리오 VAL 비교 + 게이트 판정.
  - `scripts/research_eth_omega461_tcn_exit_head_oos_20260813.py` — VAL 게이트 통과 시에만
    실행되는 1회용 OOS 확인 스크립트.
- TCN 번들: `tmp/causal_regen_20260516/eth_omega461_tcn_exit_head_liveatr_20260813/h48qual/
  tcn_exit_bundle.pt`(레짐별 `TCNExitClassifier` state_dict 3개 + `base_cols`/`pos_cols`/
  `market_scaler`/`pos_scaler`/`arch` 계약, torch 번들).
- report.json: `tmp/causal_regen_20260516/eth_omega461_tcn_exit_head_liveatr_20260813/report.json`
  (학습, 레짐별 val AUC/logloss 진단 포함), `tmp/causal_regen_20260516/
  eth_omega461_tcn_exit_head_val_20260813/report.json`(G0+VAL 비교+게이트).
- 거래 원장(diagnostic, 참고용): `tmp/causal_regen_20260516/eth_omega461_tcn_exit_head_val_
  20260813/portfolio_ledger_asymmetric_h48qual_tcn_zig075_original.csv`(위 "리플레이 원장 직접
  확인"에서 인용).
- 인용 문서: `docs/experiments/eth_omega461_gbdt_exit_head_20260813.md`(#4, 동일 규율 선례),
  `docs/experiments/eth_h48qual_tcn_hpsearch_multivariant_20260812.md`(TCN 아키텍처/HP 출처),
  `docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`(TabM 베이스라인
  근거), `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`(서브 프로젝트
  계약).
