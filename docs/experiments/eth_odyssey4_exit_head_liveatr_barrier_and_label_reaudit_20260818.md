# liveATR exit_head 배리어 시뮬레이션 + dense 라벨링 재점검 (2026-08-18)

## 배경

[[eth_odyssey4_exit_head_tpsl_feature_barrier_mismatch_20260817]]에서 `pos_tp`/`pos_sl`
피쳐가 실제 배리어와 무관한 고정상수를 썼던 버그를 찾아 고쳤다. 사용자가 같은 스크립트
(`scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py`)의 배리어
시뮬레이션과 dense bar-by-bar 라벨링을 다시 점검해달라고 요청 — 그 수정이 전부가
아니었다. **h48qual과 zig075 둘 다 이 스크립트의 `_build_exit_dataset_entry_label_live_atr_
barrier` 하나를 공유**하므로(`for component in ("h48qual", "zig075")`, 각자 baseline만
다르고 데이터셋은 동일), 아래 발견은 **현재 라이브에 배포된 h48qual exit_head 번들에도
그대로 적용**된다.

## 발견 1(핵심) — `pos_unrealized`/`pos_mfe`/`pos_mae`가 학습 시에는 0.45배로 압축된다

`_build_exit_dataset_entry_label_live_atr_barrier`([research_eth_omega461_exit_head_
liveatr_relabel_20260813.py:282, 407-410, 432-436](scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py:282)):
```python
notional = float(omega.BASE_TEMPLATE["notional"])   # = 0.45 (train_eval_omega1_2_tabm_diffusion_risk_20260603.py:40)
...
raw = (px * (1 - slip_eff) - entry_price) / entry_price   # 순수 가격변동률
unreal = raw * notional                                    # <- 0.45배 스케일
mfe = max(mfe, unreal)
mae = min(mae, unreal)
...
row = exit_head._position_feature_row(..., mfe=mfe, mae=mae, unreal=unreal, ...)
```
`_position_feature_row`([train_eval_omega1_2_tabm_exit_head_20260603.py:377-382](scripts/train_eval_omega1_2_tabm_exit_head_20260603.py:377))는 이 값을 그대로
`pos_unrealized`/`pos_mfe`/`pos_mae`/`pos_dist_to_tp`/`pos_dist_to_sl`에 담는다.

그런데 **추론/실전(`greedy_replay`, `replay_exit_variant`)에서는 이 5개 피쳐가 스케일 없이
그대로 들어간다**([replay_omega4_6_1_greedy_router_20260706.py:125-155](scripts/replay_omega4_6_1_greedy_router_20260706.py:125)):
```python
move = (close[i]*(1-slip_eff) - entry_price) / entry_price   # 순수 가격변동률
mfe, mae = max(mfe, move), min(mae, move)                     # notional 곱셈 없음
...
pos_values=[pos, hold, move, mfe, mae, giveback, take_profit-move,
            move+abs(stop_loss), notional, leverage_v, notional*leverage_v, take_profit, stop_loss]
```
(`unreal = move*notional`는 별도 지역변수로 캐시/MDD 계산에만 쓰이고 `pos_values`엔 안 들어간다.)

**결과: 같은 실제 가격변동에 대해 학습 시 `pos_unrealized`/`pos_mfe`/`pos_mae`는 실제
크기의 45%로 압축된 값을 보고, 추론 시엔 100%(비압축) 값을 본다.** ATR 배리어가
7.5~22%이므로 학습에서 모델이 실제로 본 `pos_mfe` 범위는 대략 0~10%(=0.45×22%)인데,
추론에서는 최대 0~22%까지 들어올 수 있다 — 약 2.2배 큰, 학습 때 한 번도 못 본 입력을
받는 것과 같다.

`pos_dist_to_tp = take_profit(실제 ATR값, 2026-08-17 수정으로 이미 정확) − unreal(0.45배
압축)`이라 두 항의 단위 스케일이 서로 다른 채로 뺄셈된다 — 학습 시 "TP까지 남은 거리"가
실제보다 항상 더 멀게(과대) 계산된다. `pos_dist_to_sl`도 동일 문제.

**`pos_giveback`는 예외 — 영향 없음.** `giveback = (mfe−unreal)/mfe`는 분자분모가 똑같이
0.45배 스케일되므로 비율 자체는 스케일과 무관하게 동일하다. 따라서 이미 확정된
"`giveback_min=0.65` 자체가 너무 늦다"는 [[eth_odyssey4_zig075_exit_head_threshold_
review_20260817]]의 근본원인 진단은 이 버그와 무관하게 그대로 유효하다 — 이번 발견은
그 진단에 **추가되는** 별개 문제다.

**`pos_notional`/`pos_leverage`/`pos_exposure`는 더 심각하다.** 학습 시 이 3개는 매 행마다
`notional=0.45, leverage=2.0, exposure=0.90`으로 **완전히 고정된 상수**다(`BASE_TEMPLATE`
그대로 사용, 실제 후보별 리스크사이징과 무관). 반면 추론 시엔 실제 리스크 사이드카가
계산한 값(대략 notional 0.1~0.9 × leverage 1~5, `RISK_BOUNDS`)이 매 bar 다르게 들어간다.
**모델은 이 3개 피쳐에 대해 분산이 0인 데이터로 학습했으므로, 이 피쳐들에 어떤 의미
있는 가중치도 배울 수 없었다** — 추론 시 처음 보는 변동값이 들어왔을 때 반응이 사실상
정의되지 않은 상태다.

**계보**: 이 `unreal = raw * notional` 스케일링은 liveATR에서 새로 생긴 게 아니라
`research_eth_omega461_exit_head_h48cons_relabel_20260813.py:184, 247-248`(h48cons,
liveATR의 직전 실패작)에서 이미 있었고 liveATR이 그대로 물려받았다. 원조/정본
(`train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622.py`의
`_build_exit_dataset_price_move_terminal_giveback`, 111-244행)은 `move`를 스케일 없이
그대로 쓴다 — **이 버그가 없는 게 원래(정본) 방식**이었다.

## 발견 2 — 배리어 도달 판정이 고가/저가(intrabar) 기준, 실제 청산은 종가 기준

배리어 해소 bar(`barrier_end_i`, `tb_reason`) 판정([research_eth_omega461_exit_head_
liveatr_relabel_20260813.py:378-390](scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py:378)):
```python
hi, lo = arrays["high"][row_i], arrays["low"][row_i]
hit_sl = (lo <= sl_level) if side > 0 else (hi >= sl_level)
hit_tp = (hi >= tp_level) if side > 0 else (lo <= tp_level)
```
이 리포의 실제 리플레이/라이브 엔진은 전부 **종가 기준**이다 — `greedy_replay`
([replay_omega4_6_1_greedy_router_20260706.py:125](scripts/replay_omega4_6_1_greedy_router_20260706.py:125)), `replay_exit_variant`
([research_eth_omega461_exit_sweep_20260721.py:169](scripts/research_eth_omega461_exit_sweep_20260721.py:169)→`price_exit._price_move`), 심지어
**같은 파일의 원조 함수(`_build_exit_dataset_price_move_terminal_giveback`, 181-188행)도
종가만 쓴다.** 즉 liveATR/h48cons만 배리어 판정에 고가/저가를 새로 도입했다.

같은 봉 안에서 고가/저가가 배리어를 찍는 시점은 종가가 배리어를 넘는 시점보다 항상
같거나 빠르다. 그 결과 **`barrier_end_i`(그리고 여기서 파생되는 "terminal window" 라벨
구간)는 실제 종가기준 리플레이가 그 트레이드를 끝낸다고 판단할 시점보다 체계적으로
이르다** — 모델은 "이제 곧 끝난다"는 라벨을, 실제 청산엔진 기준으로는 아직 한참 남은
지점에 대해 학습하게 된다. `tb_reason` 자체는 라벨 조건(terminal/adverse/giveback)에
직접 쓰이진 않지만(메타데이터로만 기록), `barrier_end_i`는 terminal window의 기준점이라
간접적으로 라벨에 영향을 준다.

## 발견 3 — adverse/giveback 임계값이 발견1 스케일링 때문에 표면값과 다르게 작동 (발견1의 파생)

`adverse_unreal=-0.010`, `min_mfe_for_giveback=0.006`([research_eth_omega461_exit_head_
liveatr_relabel_20260813.py:257-258](scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py:257))은 이름상 "−1.0% 미실현손실",
"+0.6% MFE"처럼 보이지만, 발견1의 0.45배 스케일 때문에 실제 가격변동 기준으로는
**−2.22%**(=0.010/0.45), **+1.33%**(=0.006/0.45)가 돼야 발동한다. 게다가 이 상수들은
원래 SL이 고정 1.4%였던 구(舊)레시피 시절 값을 ATR 배리어(SL 4~12%)로 바뀐 뒤에도 한
번도 재보정하지 않았다 — 액면가로 봐도 이미 새 배리어 스케일과 안 맞고, 여기에 스케일
버그까지 겹쳤다.

## 종합

| 항목 | 학습(라벨 생성) | 추론(greedy_replay/live) | 영향 |
|---|---|---|---|
| `pos_unrealized`/`pos_mfe`/`pos_mae` | raw×0.45 | raw×1.0 | 약 2.2배 스케일 불일치 |
| `pos_dist_to_tp`/`pos_dist_to_sl` | 실제배리어 − (raw×0.45) | 실제배리어 − raw | 거리 과대추정 |
| `pos_notional`/`pos_leverage`/`pos_exposure` | 상수(0.45/2.0/0.90) 고정 | 실시간 변동값 | 학습분산 0, 추론시 미학습 입력 |
| `pos_giveback` | 비율(스케일 무관) | 비율(스케일 무관) | **영향 없음** |
| `barrier_end_i`/terminal window 기준점 | 고가/저가 터치 | (참고: 실제 청산은 종가) | terminal 라벨이 실제보다 이른 시점에 앵커링 |

`giveback_min=0.65` 자체가 늦다는 기존 진단([[eth_odyssey4_zig075_exit_head_threshold_
review_20260817]])은 이 버그들과 무관하게 유효하며, 이번 발견들은 **거기 더해지는**
별도 원인이다. 특히 발견1(pos_unrealized/mfe/mae 스케일 불일치, pos_notional/leverage/
exposure 무분산)은 `pos_tp`/`pos_sl` 버그와 같은 부류(학습-추론 피쳐 불일치)이면서 아직
한 번도 고쳐지지 않았고, **현재 라이브 h48qual 번들에 실재한다**.

## 다음 단계 (실행 안 함 — 사용자 확인 필요)

1. `unreal`/`mfe`/`mae`를 `raw`(스케일 없음)로 바꾸고, `pos_notional`/`pos_leverage`를
   후보별 실제 리스크사이징 값으로 교체(또는 최소한 다양한 값으로) — `pos_tp`/`pos_sl`
   수정과 같은 패턴.
2. `adverse_unreal`/`min_mfe_for_giveback`을 새 스케일(비압축) 기준으로 재보정.
3. 배리어 판정을 종가 기준으로 통일할지, 아니면 의도적으로 고가/저가를 쓰되 그 사실을
   인지한 채 재평가할지 결정 — 실제 라이브가 손절/익절을 거래소 stop/limit 주문으로
   내는지, 폴링(종가)으로 처리하는지에 따라 "어느 쪽이 진짜 라이브에 맞는 컨벤션인지"가
   달라지므로, 코드를 고치기 전에 라이브 주문 방식부터 확인 필요.
4. 수정 후 h48qual/zig075 둘 다 재학습 + fresh-forward 6창 재평가(내부 val split 아님).

본 문서는 진단 전용이며 코드 수정을 하지 않았다. live/섀도우 파일 무변경.

## 2026-08-18 후속 세션: 코드 수정 적용 완료

위 "다음 단계" 1~3을 모두 코드 레벨에서 처리했다(4번 재학습은 미실행 — 아래 참고).
`scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py`,
`_build_exit_dataset_entry_label_live_atr_barrier` 기준.

**발견1a(스케일 압축)**: 이전 세션이 이미 `unreal = raw`로 수정해둔 상태(미커밋)를 그대로
유지 — `greedy_replay`(`replay_omega4_6_1_greedy_router_20260706.py:125-126`)와
`_price_move`(`train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622.py:53-64`) 둘 다
슬리피지만 반영한 비압축 `move`를 쓴다는 것을 직접 재확인.

**발견1b(무분산 리스크피쳐) — 실제 라이브 주문 방식 확인 후 진행**: `trading_bot.py`를 직접
확인 — `unrealized`/`gross_unrealized`가 전부 `current_price`(폴링된 마크가격) 기반으로
계산되고(예: 10472-10474행 `_price_return_frac(pos, entry_price, current_price)`), 거래소
STOP_MARKET/TAKE_PROFIT류 주문 생성 코드가 전혀 없다 — 라이브는 종가/마크가격 폴링 방식이지
거래소 stop 주문 방식이 아님을 코드로 확정. 3번 항목의 전제 조건 충족.

실제 리스크사이징 소스는 `research_eth_omega461_exit_sweep_20260721.prep_component`(=
`h48cons.sweep`)가 `h48cons._evaluate_val`이 이 스크립트의 재학습 번들을 채점할 때 쓰는 바로
그 경로 — frozen `risk_sidecar.pkl` + `train_predictions_qXXX.csv`. 신규 헬퍼
`_risk_sizing_for_component(component, frame, seed)`을 추가해 이 경로로 TRAIN 구간
`margin`/`leverage`를 얻는다.

**예상 밖 발견 + 사용자 결정 필요했던 지점**: `margin`을 후보 자신의 signal bar 인덱스로 그대로
읽으면(`risk_margin[i]`), risk sidecar가 parent 자체 결정이 active하지 않은 bar에서는
`margin=0`으로 하드게이트하는데(`train_eval_omega4_2_risk_sidecar_20260622.py:582`,
`margin[~omega._active(dec)] = 0.0`), 이 exit_head 레시피의 후보는 zigzag_action bar
전체(밀도 높은 별개 모집단)에서 뽑으므로 실측 결과 h48qual 3.3%/zig075 8.5%만 겹친다 — 그대로
인덱싱하면 후보 풀이 1500~2000개에서 ~50~170개로 붕괴해 이 레시피의 "밀집 후보" 설계 자체를
훼손한다. 사용자에게 4가지 옵션(실제 active-bar 경험적분포에서 페어보존 리샘플 / 최근접
active-bar 채움 / active-mask 우회 재계산 / 보류)을 제시했고, "실제 active-bar 분포에서
리샘플(추천)"로 결정 — 매 inactive bar에 실제 관측된 (margin, leverage) 페어를 시드고정
`np.random.default_rng(seed).choice`로 복원추출 배정(페어링 보존, leverage 독립추출 아님).
active bar는 원값 유지. `_risk_sizing_for_component`에 구현, 5-candidate 스모크테스트로
100% 양의 notional 확인(h48qual 427종/1500, zig075 80종/1500 unique 값 — 진짜 분산 확보),
같은 seed 재호출시 완전 동일 출력(결정론) 확인.

**발견2(배리어 고가/저가)**: `_fast_timescale_checkpoint`와
`_build_exit_dataset_entry_label_live_atr_barrier` 양쪽의 배리어 도달 판정을 종가 기준으로
변경 — `greedy_replay`, `_price_move`, 그리고 방금 확인한 `trading_bot.py`의
`current_price` 폴링 컨벤션과 통일.

**발견3(파생, 표면값 불일치)**: 발견1a 수정(`unreal=raw`) 하나로 자동 해소됨을 코드로
확인 — `adverse = unreal <= adverse_unreal`이 이제 압축 없는 실제 가격변동과 직접 비교된다.
"새 ATR배리어 스케일(SL 4~12%)에 표면값(-1.0%/+0.6%) 자체가 여전히 적절한가"라는 별도
재보정 질문은 모델링 의사결정이라 이 세션에서 상수를 바꾸지 않음 — 이전 세션이 이미
`scripts/train_eth_zig075_exit_head_barrier_recal_20260818.py`(adverse_unreal=-0.020,
min_mfe_for_giveback=0.015, giveback_min=0.45)로 별도 탐색 스크립트를 만들어둔 상태라 그걸로
충분.

**공유 함수 호출자 5곳 전부 갱신**(시그니처에 `risk_margin`/`risk_leverage` 필수 인자 추가로
호출자 전부 깨지는 걸 확인하고 각각 대응):
- 본 스크립트 `main()`: h48qual/zig075 각자 자기 sidecar로 개별 데이터셋 빌드(기존엔 하나
  공유 — h48qual/zig075 리스크사이징 스케일이 최대 6배 이상 다름을 `greedy_replay`의
  `SCALE_MAP`으로 확인했으므로 공유 불가 판단).
- `scripts/train_eval_omega461_gbdt_exit_head_liveatr_20260813.py`(h48qual 단독) —
  `_risk_sizing_for_component("h48qual", ...)` 추가.
- `scripts/train_eth_zig075_exit_head_barrier_recal_20260818.py`(zig075 단독, 이전 세션이
  방금 만든 후속 스크립트) — 동일 패턴 추가.
- `scripts/train_eth_candidate_unified_phase2_exit_head_giveback_recal_20260817.py` — parent가
  `sweep.COMPONENTS`에 미등록(신규 후보라 risk sidecar 자체가 없음) — `risk_margin=None,
  risk_leverage=None` 명시적 opt-in(고정 BASE_TEMPLATE 폴백, `risk_sizing_source` 필드로
  투명하게 기록됨)으로 대응, 조용한 기본값 아님.
- `scripts/train_eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814.py` — 폴드별
  커스텀 학습기간이라 표준 `train_predictions_qXXX.csv` 커버리지를 벗어날 수 있음(특히 2026년
  걸치는 폴드는 `oos_predictions_qXXX.csv`와의 스티칭이 필요해 별도 설계가 필요) — 지금은
  None 폴백으로 대응, 발견1a/2는 그대로 적용받음.

전부 `python3 -m py_compile` 통과 + 5-candidate 스모크테스트(`_build_exit_dataset_entry_label_
live_atr_barrier`를 실제로 실행, h48qual/zig075/no-sidecar-fallback 3경로 전부)로 end-to-end
검증 완료. **실제 ~1500-2000 후보 전체 재학습은 이 세션에서 실행하지 않음** — 이전 세션이
동일 함수 재학습을 3번 시도해 전부 WSL2 VM 재시작으로 실패했던 이력([[dev_machine_wsl2_
instability_20260816]]) 때문에, 코드 정확성 검증(스모크테스트)까지만 로컬에서 하고 실제
재학습+fresh-forward 재평가는 서버로 위임 권장.

## 2026-08-18: 같은 버그 패턴의 코드베이스 전수조사

사용자 요청으로 이 3가지 버그 패턴이 다른 모델에도 있는지 전수조사(서브에이전트 병렬 조사 +
정적분석 스크립트 `scripts/audit_position_feature_train_inference_parity_20260818.py` 신규
작성). 확인된 것: **h48qual/zig075 라이브 배포 번들은 이 문서가 고친 경로 하나뿐**(zig075는
애초에 liveATR을 라이브에서 안 씀). 그 외 최소 10개 파일이 같은 버그 패턴을 코드 재사용/포팅으로
물려받은 상태이나 전부 라이브 미배포:
- `research_eth_omega461_exit_head_h48cons_relabel_20260813.py` — 이 레시피의 직전 실패작
  원본(발견1의 "계보" 섹션에서 이미 언급됨).
- `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`의
  `_build_exit_dataset_entry_label_path_optimal`/`_build_exit_dataset_entry_label_terminal_
  giveback` — 이 두 함수가 사실상의 "원조" 정의이고, 아래 BTC/SOL/reduced80 변형들이 전부
  이 파일을 복사해 만들어짐.
- `_btc_20260708`/`_btc_exitonly_20260806`/`_btc_swingtransition_20260806`/
  `_btc_swingtransition_zigzag_20260806`/`_sol_20260707`/`_reduced80_20260724` — 위 파일의
  BTC/SOL/변형 포크 6개, 전부 동일 라인(510행 등)에 동일 버그.
- `omega4_exit_label_path_optimal_sol_20260715.py` — "verbatim 포팅"이라고 자기 docstring에
  명시된 SOL 포크. 현재 라이브 SOL 번들(`adaptive_squeeze` 계열)과는 무관.
- `research_eth_omega461_censored_stopping_value_20260724.py` — 발견2(고가/저가) 패턴, RESEARCH
  ONLY로 명시됨.

이 10개 파일은 **이번 세션에서 수정하지 않음** — 전부 비라이브 research 라인이고(일부는
`train_eval_omega4_quality_regression_20260621.py` 등에서 서브에이전트가 확인한 것처럼 자기
평가에서 학습한 exit_head를 아예 사용조차 안 함), "이 3가지 버그 고치기"라는 요청 범위를 넘는
별도 대규모 작업이라 판단. `scripts/audit_position_feature_train_inference_parity_20260818.py`가
이 10개 파일 전부를 confirmed/needs_review로 재현하며, CLAUDE.md의 "Position-Feature
Train/Inference Parity Contract"가 재발 방지 정책으로 추가됨.

⚠️ **위 "전부 비라이브" 결론은 이후 같은 세션에서 정정됨** — 아래 "2026-08-18 후속 세션 2" 참고.

## 2026-08-18 후속 세션 2: 발견2(배리어 컨벤션) 정정 + 라이브 배포 범위 정정

사용자가 "위 10개 research 파일도 정리"를 요청해 4개 서브에이전트로 ETH/BTC/SOL/censored_
stopping_value 각각의 실제 호출자·라이브 배포여부·자산별 risk sidecar 인프라를 조사시킨 결과,
**두 가지 중대한 정정이 필요함이 드러남**.

### 정정 1: 발견2(배리어 고가/저가 vs 종가)의 방향이 틀렸었다

`omega4_6_1_live.py::evaluate_exit`(h48qual/zig075를 실제로 지배하는 함수)를 직접 읽고
`trading_bot.py:9181-9202`의 실제 호출부를 확인한 결과: **TP/SL 하드체크는 intrabar
고가/저가 기준이 맞다** — `bar_high_move`/`bar_low_move`를 방금 완결된 bar의 실제
고가/저가로 계산해 넘기고("resting TP/SL 주문은 종가가 아니라 닿는 즉시 체결, 이미 확정된
bar만 쓰므로 lookahead 아님"이라는 문서화된 설계), `trading_bot.py`가 실제로 None이 아닌
진짜 고가/저가 값을 채워 호출한다.

원래 발견2는 `greedy_replay`/`_price_move`(둘 다 종가 기준)와 `trading_bot.py`의 일반적인
`current_price` 기반 unrealized 체크만 보고 "라이브는 종가 기준"이라 결론 내렸는데,
`omega4_6_1_live.py::evaluate_exit` 자체를 직접 확인하지 않은 게 원인 — greedy_replay가
종가만 쓰는 건 사실이지만 그게 "라이브도 종가"라는 근거는 못 됐다(리플레이 도구가 라이브의
bar_high_move/bar_low_move 개선을 반영 못 한 채 뒤처져 있었을 가능성이 높음).

**정정 조치**: `_fast_timescale_checkpoint`와 `_build_exit_dataset_entry_label_live_atr_
barrier`의 배리어 판정을 intrabar 고가/저가로 되돌림(2026-08-18 최초 수정 이전 원본 코드와
동일) — 스모크테스트로 재검증 완료(rows=4584, 이전 4591에서 소폭 감소는 intrabar가 종가보다
같거나 먼저 배리어에 닿는다는 사실과 정합적). **발견1a(pos_unrealized/mfe/mae 종가기준
스케일)는 정정 대상 아님** — 이는 exit_head 자신의 학습된 피쳐(하드코딩 TP/SL 프리체크를
통과한 뒤에만 평가됨) 이야기이고, 이 피쳐들은 실제로도 종가/마크가격 기준이 맞다
(`trading_bot.py:9178`의 `move`). 같은 라이브 호출 안에 "하드코딩 TP/SL은 intrabar,
학습된 exit_head 입력은 종가"라는 두 컨벤션이 공존하며, 이번 정정 전에는 이 둘을 하나로
통일하려던 게 실수였다. CLAUDE.md 정책 + 감사스크립트의 Pattern C 설명 문구도 "종가만 정답"
주장을 제거하고 "해당 자산의 실제 evaluate_exit류 함수를 직접 확인 후 판단"으로 정정함.

**미해결로 남긴 것**: `research_eth_omega461_censored_stopping_value_20260724.py`도 intrabar
고가/저가로 배리어를 판정하는데(자기 문서에 의도적 선택이라 명시), 이 파일 조사 에이전트가
"오늘 재점검의 발견2 자체가 재정정이 필요할 수 있다"고 독립적으로 먼저 지적함 — 결과적으로
맞는 지적이었다. 이 파일은 비라이브(`development_rejected`, 한 번도 발동 안 함) 확인됐으므로
급하지 않으나, 10개 파일 정리 작업에서 재검토 필요.

### 정정 2: "10개 파일 전부 비라이브"가 틀렸다

4개 조사 에이전트가 각 파일의 실제 라이브 배포 여부를 `runtime_config.py`/`trading_bot.py`
직접 대조로 재확인한 결과:
- **ETH**: `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`의
  `_build_exit_dataset_entry_label_terminal_giveback` — `FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_
  BUNDLE_PATH`/`..._ZIG075_BUNDLE_PATH`(runtime_config.py) 기본값이 이 함수로 학습된 번들을
  가리킴, 두 번들의 report.json에서 `exit_label.mode=="entry_label_terminal_giveback"`,
  `risk_template=={"notional":0.45,"leverage":2.0,...}`(버그 상수 그대로) 직접 확인. **이게
  liveATR 재라벨(이번 세션 초반에 고친 스크립트)이 재학습의 베이스로 삼는 바로 그 baseline
  번들** — 즉 두 스크립트가 서로 다른 레벨에서 같은 라이브 경로에 관여.
- **BTC**: `train_eval_omega4_3head_parent72_loose_entry_quality_btc_swingtransition_
  20260806.py` — `FINAL_GOVERNOR_OMEGA4_6_1_BTC_BUNDLE_PATH`+`OMEGA4_6_1_SHADOW_ASSET_
  CONFIG["btc"]`로 shadow 배포 확인, `trading_bot.py`의 `Omega461LiveAdapter`에 실제 와이어됨.
  실주문 실행은 `FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE` 기본값 False라 현재
  실자본 리스크는 없어 보이나(이 체크아웃 기준, 서버 실제 런타임 env는 확인 불가), 결정은
  계속 계산되고 있음.
- **SOL**: `train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py` —
  `FINAL_GOVERNOR_OMEGA4_6_1_SOL_ENABLE` 기본값 True, `..._SOL_BUNDLE_PATH`가 이 함수로 학습된
  adaptive_squeeze 번들을 가리킴, report.json에서 `label_mode=="entry_label_terminal_
  giveback_every_in_position_bar"`(이 함수만 내는 문자열) 직접 확인. **최초 조사 때 "SOL 라이브와
  무관"이라 결론 낸 건 `omega4_exit_label_path_optimal_sol_20260715.py`(다른 파일, CatBoost라
  Omega461LiveAdapter가 아예 못 읽음, 진짜 비라이브 맞음)에 대한 얘기였는데, 사용자에게 "10개
  전부 비라이브"라고 뭉뚱그려 전달한 게 부정확했음.**

즉 10개 파일 중 최소 3개(ETH 원조정의, BTC swingtransition, SOL 원본)는 라이브/섀도우
배포와 직결되고, 나머지(h48cons, path_optimal류, reduced80, SOL path-optimal 포트,
censored_stopping_value)는 실측대로 비라이브 유지. 사용자 지시: "라이브 3개 먼저 우선순위로
진행", 이후 "나머지 7개도 이어서 진행해줘".

## 2026-08-18 후속 세션 4: 나머지 7개 전부 완료 — 10개 파일 전수 종료

**h48cons**(`research_eth_omega461_exit_head_h48cons_relabel_20260813.py`)는 라이브 3개와 달리
**실제 risk sidecar 소싱이 가능한 구조** — liveATR과 마찬가지로 이미 학습된(이미 risk-
sidecar화된) baseline 위에 exit_head만 재학습하는 패턴이라(`sweep` 직접 import, `_retrain_
component_exit_head`가 `sweep.COMPONENTS[component]["bundle"]`을 로드) 부트스트랩 문제가
없음. 후보밀도(every zigzag_action bar)와 active-bar 겹침률(h48qual 3.3%/zig075 8.5%)까지
실측으로 재확인 — liveATR과 사실상 동일 수치. `_risk_sizing_for_component`를 로컬 복제(순환
import 회피, 이 리포의 기존 컨벤션)해 실제 리사이징 소싱 + main() 컴포넌트별 개별 데이터셋
빌드로 재구성. 스모크테스트로 실데이터 검증 완료(risk_sizing_source="frozen_risk_sidecar_
per_candidate", 실분산 확인).

나머지 6개 파일(7개 함수)은 전부 부트스트랩 패턴이라 `risk_margin=None, risk_leverage=None`
명시적 opt-in + 스케일버그(발견1a) 수정:
- ETH 원조정의 파일의 남은 `_build_exit_dataset_entry_label_path_optimal`(고아 함수, 실제
  학습된 번들 0개 확인됨) — 외부호출자 1개(`build_omega4_entry_label_path_optimal_exit_
  labels_20260620.py`)도 갱신. **주의**: 이 함수는 `notional`이 `_exit_fill_net`의 캐시
  시뮬레이션(라벨 자체를 도출하는 DP suffix-max 계산)에도 쓰이는데, 이건 CLAUDE.md의
  Futures Risk Sizing Contract·기존에 이미 클리어된 다른 BASE_TEMPLATE 백테스트 용법과 같은
  범주(고정가정 기반 백테스트 시뮬레이션)라 판단해 범위 밖으로 유지 — `row_notional`(피쳐용과
  동일 값)을 그대로 넘겨 라벨 산출 로직 자체는 수치적으로 불변.
- reduced80(`_reduced80_20260724.py`)의 path_optimal+terminal_giveback 2개 함수 —
  ETH원조와 바이트동일 확인 후 동일 수정, 외부호출자 0개.
- BTC 3개 파일(`_btc_20260708`, `_btc_exitonly_20260806`, `_btc_swingtransition_zigzag_
  20260806`) — 이미 고친 swingtransition과 바이트동일 확인 후 동일 수정. **`_btc_
  swingtransition_zigzag_20260806.py`는 자기 자신의 의존모듈(`train_eval_omega1_2_tabm_
  diffusion_risk_btc_swingtransition_zigzag_20260806`)이 리포에 아예 존재하지 않아 import
  자체가 실패하는 기존 버그 발견** — git log로 이번 세션과 무관한 기존 문제임을 확인(마지막
  커밋이 이 세션 이전), 별도 이슈라 손대지 않고 기록만 함. `py_compile`(문법)은 통과, 나머지
  4개 파일과 완전 동일한 기계적 포팅이라 실행 검증 없이도 신뢰도 높음.
- SOL path-optimal 포트(`omega4_exit_label_path_optimal_sol_20260715.py`) — ETH원조
  path_optimal과 바이트동일 확인 후 동일 수정(같은 `_exit_fill_net`-notional 범위제한 적용),
  외부호출자 1개(`train_eval_omega4_3head_catboost_parent_sol_path_optimal_20260715.py`) 갱신.

**censored_stopping_value는 최종적으로 아무것도 안 고침** — 재검토 결과 Pattern A/B(스케일/
무분산)는 애초에 없음(이미 실제 per-row margin/leverage 사용 중, 조사에이전트 확인).
Pattern C(intrabar 배리어)는 이번 세션의 발견2 정정(intrabar가 h48qual/zig075의 실제 라이브
컨벤션) 이후 다시 보면 **오히려 올바른 설계일 가능성이 높음** — 자기 문서가 주장하는
"라이브과 동일한 intrabar 순서"가 이번에 직접 코드로 확인한 `omega4_6_1_live.py::
evaluate_exit`의 실제 설계와 정확히 같은 논리(같은 이유: "이미 확정된 bar만 쓰므로 lookahead
아님", SL을 TP보다 먼저 체크). 유일한 불일치(entry source `hazard.greedy`=`greedy_replay`가
종가기준)는 **censored_stopping_value가 아니라 greedy_replay 쪽이 뒤처진 것**으로 재해석 —
이 파일을 종가기준으로 "고치면" 이번 세션이 이미 한 번 저지르고 원복한 실수를 다시
반복하는 셈이라 그대로 둠.

## 최종 검증
전체 수정 파일(10개 원본 파일 중 9개 실제 코드변경 + 3개 외부호출자 파일) `python3 -m
py_compile` 전부 통과. 감사스크립트 재실행: confirmed **14 → 11(라이브3개) → 3(나머지7개)**
— 남은 3건은 전부 이미 검증된 false positive(자기정합적이라 버그 아님, 서브에이전트가
개별 확인). Pattern B는 0건. Pattern C 남은 8건은 전부 censored_stopping_value(위 이유로
올바른 설계 판단) + liveATR 자신의 `_fast_timescale_checkpoint`(이번 세션이 되돌린 intrabar
코드, 정정판) — 둘 다 실제로는 버그 아님. **10개 파일 전수조사 라인 사실상 완결.**

실제 재학습은 여전히 미실행(전 세션 WSL2 3연속 크래시 이력 + 이번 세션은 개발머신에
BTC/SOL 데이터파일 자체가 없음도 재확인) — 서버 위임 권장, 특히 h48cons/ETH원조/BTC
swingtransition/SOL원본 4개는 라이브·섀도우 직결이라 우선순위 높음.

## 2026-08-18 후속 세션 3: 라이브 3개 수정 완료

**발견1a(스케일)+발견1b(notional/leverage 무분산)** 를 3개 파일 모두에 적용, 발견2(배리어)는
해당 없음(아래 이유) — 스코프 확정 근거:

**pos_tp/pos_sl은 이번엔 안 건드림**: 이 3개 함수(`_build_exit_dataset_entry_label_terminal_
giveback`)는 ATR 배리어 시뮬레이션이 전혀 없는 "zigzag 세그먼트 끝까지 보유" 구조라 발견2
(고가/저가 vs 종가)가 원천적으로 적용 안 됨(포지션 종료 시점 자체가 배리어터치가 아니라
세그먼트 경계). 다만 `take_profit`/`stop_loss` 피쳐가 여전히 BASE_TEMPLATE 고정값(0.026/
0.014)인데 실제 라이브는 ATR-adaptive(atr_window=192 등)를 씀 — 이건 08-17에 liveATR
스크립트에서 고친 것과 같은 종류의 별도 버그이나, 이 3개 함수엔 ATR 계산 자체가 없어(이번
전수조사로 신규 발견) 고치려면 이 파일들에 ATR 로직을 새로 들여와야 함. **범위 확장 판단을
피하기 위해 이번엔 안 건드리고 명시적으로 남겨둠** — 사용자에게 별도 보고.

**발견1b(리스크사이징) 소싱 방식 — 중요한 아키텍처 발견**: 이 3개 함수는 번들의 "최초"
parent+exit_head 학습에서 호출된다. Risk sidecar는 그 번들이 이미 존재한 뒤 그 번들의
예측값을 입력으로 별도 스크립트(`train_eval_omega4_2_risk_sidecar_20260622.py`)가 나중에
학습한다(이 파일이 원조 학습 스크립트를 import하지, 역방향 아님 — grep으로 확인). 즉 **이
지점에서는 구조적으로 실제 risk sidecar를 소싱할 방법이 없다**(sidecar가 존재하려면 이
번들이 먼저 있어야 함) — liveATR 스크립트(이미 고친 것)와는 파이프라인 단계가 다르다(그건
이미 학습되고 이미 risk-sidecar화된 baseline 위에 exit_head만 재학습하는 구조라 실제
sidecar 소싱 가능).

따라서 3개 파일 전부 `risk_margin=None, risk_leverage=None` 명시적 opt-in으로 처리(폴백을
`risk_sizing_source` 필드로 투명 기록) — 이게 이 파이프라인 단계에서 가능한 유일하고 정직한
선택. **h48qual/zig075의 리스크사이징 무분산 문제를 실제로 해소하는 건 이미 이번 세션 초반에
고친 liveATR 재라벨 스크립트의 exit-head 재학습**이다 — 단, 그 재학습본이 실제로 라이브에
배포돼 있는지(`runtime_config.py` 기본값은 원본 번들을 가리킴, override 여부는 이 체크아웃
밖이라 확인 불가)는 별도 확인 필요.

**수정 파일**: `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`(ETH 원조,
`_build_exit_dataset_entry_label_terminal_giveback`만, `_build_exit_dataset_entry_label_
path_optimal`은 미수정 orphan이라 제외) + 외부 호출자 6개(`_direction_conditioned_quality`,
`_quality_net_return_regression`, `_5head_margin_leverage`, `_3head_cmamba_replacement`,
`_quality_regression`, `eth_split_oracle_3head`) 전부 `risk_margin=None` 명시 opt-in으로
갱신. `train_eval_omega4_3head_parent72_loose_entry_quality_btc_swingtransition_20260806.py`
(BTC, 외부호출자 없음). `train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py`
(SOL) + 외부 호출자 3개(`omega4_3head_catboost_parent_sol`, `sol_architecture_v2_full_retrain`,
`sol_architecture_v2_pipeline`) 전부 갱신.

**검증**: `python3 -m py_compile` 전체 통과. ETH는 실제 데이터로 스모크테스트(3000-row 샘플,
pos_unrealized 비압축 확인, risk_sizing_source 정확히 기록). BTC/SOL은 **개발 머신에 데이터
파일 자체가 없어서(서버 전용)** 실데이터 스모크테스트 불가 — 대신 3개 함수 모두에 대해 합성
데이터(랜덤워크 가격+zigzag 세그먼트) 테스트 작성, None-폴백 경로와 (미래용) 실제 risk_margin
배열 경로 둘 다 검증 — ETH/BTC/SOL 세 함수가 완전히 동일한 출력을 내는 것까지 확인(byte-
identical 포팅이 실제로 identical 동작함을 재확인). 감사스크립트 재실행 결과 confirmed
14→11(정확히 이 3개 함수만 하락) 확인.

**남은 7개 파일**(h48cons, ETH/reduced80의 orphan path_optimal 2개, BTC 나머지 3개 변형,
SOL path_optimal 포트, censored_stopping_value)은 비라이브 확인됐으므로 이번 라운드에서는
미수정 — 감사스크립트가 계속 재현하므로 유실 없음.

## 2026-08-18 후속 세션 5: pos_tp/pos_sl 고정상수 버그 — 9개 파일 전부 수정

**버그**: 후속 세션 3에서 명시적으로 범위 밖으로 남겨뒀던 문제(`take_profit`/`stop_loss`
피쳐가 `float(omega.BASE_TEMPLATE["take_profit"/"stop_loss"])`, 즉 2.6%/1.4% 고정값으로
`_position_feature_row`에 흘러들어감)를 사용자 요청으로 이번 라운드에서 마저 고침. 실제
라이브는 h48qual/zig075/BTC/SOL 전부 `omega4_6_1_live.py`의 `_ComponentConfig` 기본값
기반 ATR-adaptive 공식을 쓴다(모든 `components_override` 호출부가 이 7개 필드를 그대로
둠, 라이브 코드 `omega4_6_1_live.py:91-97,181-185` 직접 재확인):
```
tp = clip(max(min_tp, atr_pct*tp_mult), 0, max_tp)
sl = clip(max(min_sl, atr_pct*sl_mult), 0, max_sl)
# atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12
```
`atr_pct`는 `eval_omega4_1_atr_safety_sltp_20260622._atr_pct`(자산 무관 순수 OHLC 함수,
`min_periods=1`이라 프레임 시작부에서도 NaN 없이 축소창으로 계산).

**수정 파일 9개**: 기존 8개는 이전 세션에 이미 완료(요약에 기록됨) — ETH원조(`terminal_
giveback`+`path_optimal` 둘 다), BTC swingtransition, BTC 3변형(`_btc_20260708`,
`_btc_exitonly_20260806`, `_btc_swingtransition_zigzag_20260806` — 마지막 파일은 존재하지
않는 모듈을 import하는 사전기존 무관 버그가 있어 문법확인+형제파일과의 기계적 동일성으로만
검증), reduced80(둘 다), SOL원조, SOL path_optimal 포트. 이번에 h48cons
(`research_eth_omega461_exit_head_h48cons_relabel_20260813.py`)를 마저 고쳐 9개 전부 완료.

**h48cons는 구조가 다름**: 이 파일 자신의 라벨 배리어는 h48_conservative CSV(ATR mult
1.2/0.8, floor 0.6%/0.4%)로, 라이브 ATR 공식과 다른 별도 축(라벨 앵커용 튜닝값)이다. 따라서
pos_tp/pos_sl을 이 라벨 배리어에서 유도하면 안 되고, 위 라이브 공식을 **독립적으로 재계산**
해야 한다 — `atr_pct = atr_eval._atr_pct(frame, 192)`를 함수 진입부에서 한 번 계산해두고,
후보 시그널bar 인덱스 `i`(세그먼트 아님, h48cons는 매 zigzag_action bar가 후보)에서
`atr_pct[i]`를 룩업.

**검증 — 중요한 실측 발견**: h48cons는 ETH 기반이라 개발머신에서 실데이터 스모크테스트
가능(BTC/SOL은 여전히 서버전용 데이터라 합성테스트만). 처음 25개 후보로 테스트했더니
pos_tp/pos_sl이 **전부 floor값(0.075/0.040)에 고정**돼 있어 버그처럼 보였음 — 직접
`atr_pct` 분포를 뽑아보니 실제 ETH 5분봉 atr_pct(2025-05~09 구간, atr_window=192bar=16시간)
는 **전체 프레임의 99.7%가 floor 임계값(0.075/12=0.00625) 미만**이었다. 이는 버그가 아니라
**라이브 자체의 실제 특성**(ETH의 이 기간 변동성 레짐에서 ATR-adaptive TP/SL은 대부분
"adaptive"하게 작동하지 않고 거의 항상 floor에 고정됨, 라이브 상수·공식이 위에서 직접
재확인했듯 정확함) — `LIVE_ATR_CFG` 값 자체를 의심할 근거는 없음. 후보 2,000개로 늘려
재검증하니 pos_tp 89종(0.0750~0.0815)/pos_sl 17종(0.0400~0.0408) 관측(floor 초과 비율
각각 6.9%/0.75%) — 드물지만 진짜로 변동한다. 옛 고정값(0.026/0.014)은 재검증 데이터
어디에도 없음을 확인. h48qual/zig075 둘 다 동일 패턴.

**교훈**: 소량 샘플(25개)에서 "분산이 0"으로 보인다고 바로 버그로 단정하면 안 된다 — 공식/
상수를 라이브 소스코드와 직접 대조하고, 표본을 충분히 키운 뒤(2,000개) 재판단해야 실제
버그와 "그 자산/기간의 실제 저변동성 레짐"을 구분할 수 있다.

**감사스크립트 확장**: `audit_position_feature_train_inference_parity_20260818.py`의
Pattern B 탐지 대상(`SIZING_KWARGS`)에 `take_profit`/`stop_loss`를 추가(기존
`notional`/`leverage`와 동일한 탐지 로직 재사용 — BASE_TEMPLATE 상수가 루프 밖에서
할당되고 루프 안의 `_position_feature_row` 호출에 그대로 흘러들어가는 패턴). 확장한
탐지기가 실제로 작동하는지 이 9개 파일 중 하나(reduced80)의 수정 전 git HEAD 버전을 임시로
복원해 스캔해봄 — 두 함수 모두에서 `take_profit`/`stop_loss` Pattern B가 정확히 잡힘(사전
검증 성공). 수정 완료 후 전체 저장소(`scripts/*.py`, 1999개 파일) 재스캔 결과 Pattern B
신규 검출 0건(take_profit/stop_loss 포함) — confirmed 3/needs_review 8로 이전 라운드와
동일(전부 기검증된 false positive/올바른 설계). **9개 파일 전부 실제로 고쳐졌고 남은 게
없다는 걸 이 확장된 자동탐지가 독립적으로 재확인.**

**CLAUDE.md 갱신**: Position-Feature Train/Inference Parity Contract에 pos_tp/pos_sl
전용 항목 추가(라이브 공식·상수, h48cons 특이사항, floor-고정 실측 발견, 감사스크립트
확장 사실 전부 명시).

**최종 상태**: 9개 파일 `python3 -m py_compile` 전부 통과, h48cons는 실데이터로 직접
검증(2,000후보), 나머지 8개는 이전 세션에 합성데이터로 이미 검증됨. 감사스크립트 Pattern B
0건 유지. 실제 재학습은 여전히 미실행(서버 위임 필요, 기존 방침과 동일).

**부록 — liveATR 원본 스크립트의 리포트 dict 잔재 정리**: 위 9개 파일 목록에 liveATR
재라벨 스크립트(`research_eth_omega461_exit_head_liveatr_relabel_20260813.py`) 자신은
포함 안 됨 — 이 파일은 애초에 **08-17(전날) 세션에서 이미 pos_tp/pos_sl을 고쳤던 원본**
이라(`docs/experiments/eth_odyssey4_exit_head_tpsl_feature_barrier_mismatch_20260817.md`
참고, 이번 9개 파일 수정에 그대로 복제한 패턴의 출처), `_position_feature_row` 호출부
(559-562행)는 이미 실제 `tp_move`/`sl_move`(후보별 ATR 계산)를 정확히 쓰고 있었음 —
모델 피쳐 자체는 처음부터 안 깨져 있었다. 다만 최종 점검 중 리포트 dict에 죽은 변수
잔재를 발견: `take_profit = float(omega.BASE_TEMPLATE["take_profit"])`/`stop_loss = ...`
(구 상수 2.6%/1.4%)가 실제로는 아무 피쳐 계산에도 안 쓰이면서 `"take_profit_base_
template": take_profit`/`"stop_loss_base_template": stop_loss`로만 리포트에 남아있어,
report.json을 읽는 사람에게 "이 학습이 고정 2.6%/1.4%를 썼다"는 **잘못된 인상**을 줄 수
있었음(모델 버그는 아니지만 리포트 정확성 문제). 죽은 로컬 제거 + `tp_used`/`sl_used`
추적 추가 + 다른 9개 파일과 동일한 `live_atr_tp_sl` 진단블록으로 교체. 60-candidate
실데이터 스모크테스트로 `pos_tp`/`pos_sl`의 실제 평균이 리포트에 기록된 `tp_mean`/
`sl_mean`과 정확히 일치함을 확인, 구 키(`take_profit_base_template` 등) 완전 제거 확인.
감사스크립트 재실행 결과 findings 불변(Pattern B 0건 유지).

## 2026-08-18 후속 세션 6: h48qual/zig075 재학습 착수 — 사고 발생 및 복구

사용자 질문 "그럼 이제 오디세이4 모델 전체 재학습하면 되나?"에 대해 확인해보니, 답은
"exit_head만"이 아니라 실제로 "전체"가 맞았다 — `_fit_expert_omega4`가 direction/quality/
exit 3개 헤드를 **하나의 공유 스케일러**(`_standardize_fit`, x_dir+x_exit concat 기준
mean/std)와 공유 네트워크로 같이 학습한다. exit_head-only 재학습 경로(`_fit_exit_head_
only`, liveATR/h48cons가 쓰는 방식)는 baseline의 **구(舊) scaler를 그대로 재사용**하고
트렁크 전체를 freeze하므로, pos_tp/pos_sl처럼 x_exit 쪽 피쳐 분포가 바뀐 fix는 제대로
못 반영한다. **직접 확인**: 현재 라이브가 참조하는 h48qual baseline
(`tmp/causal_regen_20260516/..._noctx_padded_..._20260630/report.json`)의
`exit_label.diag.risk_template.take_profit = 0.026, stop_loss = 0.014` — 구버그 상수가
그대로 박제돼 있어, 지금 라이브가 쓰는 exit_head가 버그 있는 코드로 학습된 것을 코드가
아닌 **번들 자신의 기록으로** 재확인했다.

사용자 승인(ETH 먼저, h48qual 먼저 launch 후 확인하고 zig075)에 따라 재학습 커맨드를
정확히 재구성(report.json 필드 기반 역추적, 조사 에이전트 위임 — `--max-train-rows 0`과
`--device cuda`만 근거 없는 추정, 나머지는 report.json에서 직접 확인)하고
`scripts/ops/handoff.sh`로 서버에 launch 준비하던 중 **작업 실수 발생**:
`handoff.sh pull server <path>`가 스크래치가 아니라 **실제 로컬 워킹트리에 직접 덮어쓴다**
는 걸 (그 함수 소스를 직접 읽었음에도) 놓치고 "서버 버전 확인"용으로 pull을 실행 →
이 세션에서 `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`에 적용한
전체 수정(스케일버그+리스크사이징+pos_tp/pos_sl, ~150줄)이 서버의 구버전으로 즉시
덮어써짐(`git diff HEAD` 무출력으로 발각). **복구**: 같은 세션에 이미 검증 완료된 자매
포크 `reduced80`(사전세션 시점에 이 두 함수만은 byte-identical 확인)의 `git diff HEAD`를
템플릿 삼아, reduced80에 섞여있던 무관한 선행 미커밋 변경(VAL-primary ranking, 08-13
축 — ETH원본은 HEAD에 이미 커밋돼있어 무관)은 걸러내고 실제 fix 부분만 수동으로
재적용. 실데이터 스모크테스트(6000-row) 재통과 확인, 감사스크립트 findings 불변(11건,
Pattern B 0), 외부호출자 6개 전부 정상 확인. 교훈 및 재발방지는
`feedback_handoff_pull_overwrites_working_tree` 메모리에 기록.

**h48qual 재학습 launch**: 복구+재검증 완료 후 `handoff.sh push`(pull 아님, 로컬은
안전)로 고쳐진 스크립트를 서버에 동기화하고 `--out-suffix
..._posfix_20260818`(라이브가 읽는 `..._20260630` 경로와 별도, 평가 전 덮어쓰기 방지 —
Omega Artifact Integrity Promotion Gate 취지)로 launch. job명
`h48qual_posfix_retrain`, `--seed 260620`(원본과 동일 시드 — 이번엔 "같은 설정에 버그만
고쳐서" 순수 비교가 목적이라 다중시드 학습은 이후 승격단계에서 별도 검토).
zig075는 h48qual이 정상 진행되는 것 확인 후 이어서 launch 예정(사용자 지정).

**h48qual 완료 + 중요 발견: 피쳐 개수 증가(102→172), "순수 버그수정만" 비교 아님**:
launch 약 2분30초만에 정상종료(트레이스백 없음, 최종 report.json까지 정상 기록) 확인 후
`handoff.sh pull`로 report.json만 받아옴(로컬에 해당 경로 자체가 존재하지 않아 안전 —
이전 사고와 달리 덮어쓸 기존 파일이 없었음). 검증: `risk_sizing.source =
base_template_constant_no_sidecar_available`(notional=0.45/leverage=2.0, 투명기록),
`live_atr_tp_sl`에 실제 ATR값(tp_mean≈0.0750, sl_mean≈0.0400, 변동폭 최대 0.0811/0.0405
— h48cons 실측과 정합), 구 `risk_template`/구상수(0.026/0.014) 완전부재 — pos_tp/pos_sl
fix가 실제 전체재학습에 정확히 반영됨을 최종 확인.

단, **`input_contract.base_feature_count`가 원본 102 → 172로 증가**(pos_cols 13개는
동일, `forbidden_feature_policy`도 두 리포트에서 완전 동일 — 리키지 이슈 아님)를 발견 —
6월30일 이후 약 7주간 이 리포에서 이뤄진 피쳐엔지니어링 리서치(다수)가 `_prepare_frames`가
자동유도하는 base_cols에 그대로 반영된 결과. 즉 **오늘 재학습은 "버그수정 하나만 격리된"
순수비교가 아니라, 버그수정+약 70개 신규피쳐가 함께 반영된 결과물**이다. 사용자에게 보고 후
선택지 3개(현재유지/102개로 고정재실행/둘다) 제시 — **"지금 결과 그대로 유지" 선택**:
재실행 없이 172피쳐 버전을 그대로 인정하고, 증가원인(피쳐진화, 리키지 아님)을 리포트에
명시, "버그수정만의 순수효과"는 별도로 분리하지 않으며 Fresh-Forward VAL/OOS 평가로
**모델 자체의 최종 성능**만 판단 기준으로 삼기로 함. zig075도 동일 방식(현재 파이프라인
그대로)으로 이어서 launch.

⚠️ **후속 승격 판단 시 반드시 참고**: 이 172피쳐 버전이 기존 102피쳐 baseline보다
Fresh-Forward에서 더 낫다는 결과가 나와도, "pos_tp/pos_sl 버그수정 덕분"이라고 단정하면
안 된다 — 신규 70개 피쳐의 기여분과 섞여 있어 분리 불가. 원인 귀속이 중요해지는 시점이
오면(예: 이 버전이 기각됐는데 이유를 알아야 할 때) 102피쳐 고정 재실행이 별도로 필요.

**zig075 launch**: 동일 스크립트+동일 방식(`--quality-mode same_as_direction`,
`--quality-label-dir` 불필요, `--quality-thresholds` 9개값), `--out-suffix
current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_
posfix_20260818`.

**zig075 완료 + 검증**: 약 2분만에 정상종료(최종 print의 `"q045": null`은 버그가 아니라
zig075 sweep이 0.45를 포함 안 해서 나오는 정상 동작 — sweep은 정확히 0.55~0.95 9개값).
report.json 확인: `risk_sizing.source`/`live_atr_tp_sl`/구상수 부재 전부 h48qual과 동일
패턴으로 정상, `base_feature_count`도 동일하게 172(같은 `_prepare_frames` 파이프라인이라
당연히 일치), `quality_mode: same_as_direction` 정확. h48qual과 zig075 둘 다
**pos_tp/pos_sl fix가 실제 전체재학습에 반영됨을 최종 확인** — 여기까지가 "버그수정→실제
재학습 반영"의 1차 완결.

## 2026-08-18 후속 세션 7: Fresh-Forward VAL/OOS 평가 착수 — canonical 데이터 재학습 필요성 발견

사용자 지시 "Fresh-Forward VAL/OOS 평가"에 착수. 기존 Fresh-Forward 인프라 조사(전용
에이전트 위임) 결과: `eval_eth_omega461_exit_head_liveatr_relabel_walkforward_20260814.py`/
`train_eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814.py`는 exit_head-only
재학습 전용(parent 항상 frozen)이라 이번처럼 parent까지 통째로 바뀐 전체재학습엔 구조적으로
안 맞음. `eth_omega461_multiwindow_confirmation_gate_20260814.py`(6개 사전등록창:
2025q1/q2/q3 context + val + oos_q1 + oos_q2, "매 Odyssey2/3 후보가 쓰는" 표준 게이트)의
`run_portfolio_variant`/`align_frame_and_predictions`도 진입 예측 CSV를 `sweep.EXT_PRED_DIR`
(구 번들 예측)에 하드코딩해서 그대로는 못 씀 — 재사용 가능한 하위부품(`sweep.prep_component`/
`greedy.prepare_component`/`greedy.greedy_replay`/`gate.load_all_windows`)만 새 오케스트레이션
스크립트에서 직접 호출하는 방식 확정.

**⚠️ 더 심각한 발견**: posfix 번들들의 `_prepare_frames()`가 실제로 읽는
`omega.TRAIN_CSV`/`EVAL_CSV` 기본값이 `tmp/causal_regen_20260516/
alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_*_alpha6_current_tail111_
exact.csv`(sweep.load_frame/6-window gate가 쓰는 canonical 파일과 다른 "legacy" 파일)임을
발견 — 이 파일은 (a) 다른 스크립트(`build_omega4_6_1_extended_parent_predictions_20260706.py`)
가 이미 "legacy, 일부 피쳐 drift(ou_halflife corr=-0.03 등)"로 문서화, (b) 2026년 데이터가
02-28에서 끊김(Fresh-Forward OOS 3월 전체+oos_q2 전부 불가), (c) base_cols 172개 중 66개
(`m7_*`/`ai_*`/`patchtst_*`/`dlinear_*`/`tide_*`/`sig_ai_squeeze` 등)가 이 legacy 파일에만
있고 canonical 파일엔 없음(evaluation시 0으로 채워질 수밖에 없어 진짜 학습값과 어긋남 —
cmamba/risk 11개 컬럼과 달리 이 66개는 실제 학습시 0이 아니었음). 즉 08-18 앞선 "102→172는
피쳐엔지니어링 진화" 설명은 부정확했음 — 정정: 172는 legacy 파일 자체가 그새 커진 것.

사용자에게 3택 질문("지금 번들로 부분평가/canonical로 재학습후 평가/여기서 캐릭터만 정리하고
중단") → **"canonical로 재학습 후 평가" 선택**.

**canonical 데이터 재학습 방법**: 공유모듈(`omega`) 기본값은 안 건드림(50개 스크립트 의존,
[[omega_cmamba_risk_overlay_dead_code]] 참고) — 기존 확립된 로컬오버라이드 패턴
(`train_eval_omega4_3head_parent72_eth_zig075_liverecipe_20260812.py`)을 그대로 따라 새
래퍼 스크립트(`train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818.py`) 작성:
`omega.TRAIN_CSV`/`EVAL_CSV`를 canonical(`data/splits/year_oos/training_features_2025.csv`/
`_2026_rebuilt.csv`)로 오버라이드, cmamba/risk placeholder를 canonical 타임스탬프로
재생성(기존 placeholder는 legacy EVAL_CSV 기준이라 canonical 프레임의 71%를 못 덮음 — 방치시
`_overlay_required`가 조용히 대량 드롭). 도중 WIDE24_2026(REGIME3_CURRENT_2026, 실제로는
이미 sweep.WIDE24_2026과 동일 canonical 파일)이 자체적으로 2026-02-28 16:05~23:55 95bar
중간 gap이 있다는 걸(gate 모듈이 이미 별도로 문서화한 "이미 알려진 gap") `_overlay_required`
의 엄격한 edge-only 허용치에서 다시 발견 — EVAL_CSV를 REGIME3_CURRENT_2026 커버리지에
정확히 맞춰 사전필터링해서 해결(gate모듈의 `_drop_route_nan` 사후드롭과 동등한 효과를
사전에 적용). 로컬 dry-run(`_prepare_frames()`만 실행, 학습 없이)으로 무오류+158
feature_cols(legacy-only 66개 완전 배제 확인) 검증 후 서버 launch.

**canonical 재학습 결과**: h48qual/zig075 둘 다 정상완료(각 ~2분), `base_cols=158`(102보다
많지만 172보다 훨씬 적음 — 순수 canonical 파이프라인 진화분), `risk_sizing`/`live_atr_tp_sl`
정상(posfix 버그수정 재확인), `sweep.load_frame` 프레임 대비 base_cols 커버리지 100%
(cmamba/risk 11개만 미스매치인데 이건 학습시에도 0이었던 것과 정확히 일치 — 직접 대조 확인,
가정 아님).

**Fresh-Forward 6창 평가 실행**: 새 스크립트
`eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818.py` 작성 — 예측은
`build_omega4_6_1_extended_parent_predictions_20260706.py`와 동일 메커니즘(`_base_input`→
`_predict_payload`→`_routed`→`_prediction_output`)으로 6창 전부 새로 생성(구 번들 예측
재사용 없음), 청산은 `greedy.greedy_replay`(h48qual>zig075 우선순위 라이브 위상 그대로)의
단일 순방향 bar-by-bar 루프, `mfe_width._duration_gated` 사후게이트 적용. **버그 발견+수정**:
`greedy.prepare_component`가 `_to_decisions(..., oof=False)`를 하드코딩하고 있어 oof=True
창(context 3개+val)에 그대로 쓰면 컬럼명이 안 맞아 KeyError — gate 모듈이 이미 확립한 대로
`portfolio._prepare_component_val`(동일 시그니처, oof=True 버전)로 창별 분기(`w["oof"]`
기준)하도록 수정, 소규모 테스트로 재확인 후 전체 실행.

**결과** (모두 `fresh_forward_bar_by_bar=true` 등 CLAUDE.md 필수 필드 명시, no_gate/with_gate
PnL%/MDD%/trades):
| 창 | tier | no_gate | with_gate |
|---|---|---|---|
| 2025q1 | context | -9.66%/-24.82%/31건 | +4.22%/-24.24%/26건 |
| 2025q2 | context | +39.53%/-18.36%/34건 | -0.57%/-24.74%/22건 |
| 2025q3 | context | +19.12%/-20.02%/29건 | +7.70%/-22.39%/22건 |
| val | val | -1.38%/-20.53%/28건 | -11.65%/-24.22%/26건 |
| oos_q1 | oos_confirm | -10.13%/-30.39%/28건 | +3.95%/-21.26%/22건 |
| oos_q2 | oos_confirm | +3.59%/-14.07%/18건 | +11.26%/-14.07%/16건 |

**⚠️ 아직 공식 CONFIRMED/REJECTED 판정 불가**: `summarize_multiwindow`의 사전등록 기준은
"with_gate PnL이 baseline 대비 non-worse AND MDD가 slack 이내, oos_confirm 2창 동시통과"인데
— **이번엔 baseline(원본 6/29·6/30 번들)을 동일 방법론(새 예측생성)으로 아직 안 돌림**. val
with_gate(-11.65%)가 기존 공개된 asymmetric_tabm_liveatr 레퍼런스(+77.31%)보다 크게
낮아보이지만, 그 레퍼런스는 다른 조합(zig075만 원본, h48qual은 이미 다른 liveATR수정판)이라
직접비교 불가 — "posfix 둘 다 vs 원본 둘 다"를 같은 파이프라인으로 새로 돌려야 정식 비교.
risk sidecar도 원본 번들 것 재사용(실제 동적값 확인, 상수 아님 — 최초 report.json 라벨
문구가 부정확했던 것 발견해 정정) — posfix 번들 전용 sidecar는 아직 없음, 참고용 근사치.

## 2026-08-18 후속 세션 8: baseline(버그수정 전) 동일방법론 평가 — 공식 REJECTED_SIGN_MISMATCH

사용자 지시 "버그 수정 전 val oos와 비교해줘"로 baseline(원본 6/29·6/30 번들) 평가 실행.
원본 번들 base_cols(102개) 확인 결과 canonical `sweep.load_frame` 프레임에 100% 포함(cmamba/
risk 컬럼조차 전혀 안 씀, 직접대조) — posfix 평가 때 필요했던 zero-col 예외 없이 그대로
가능. posfix 평가 스크립트를 모듈로 import해 `BUNDLES`/`OUT_DIR`만 오버라이드하는 래퍼
(`eval_eth_odyssey4_baseline_original_freshforward_20260818.py`) 작성 — 로직 자체는 재사용,
각 번들 자신의 진짜 risk sidecar 사용(posfix처럼 빌려쓰는 근사치 아님, 정식 평가).

**교차검증**: 이 실행의 2025q1/q2/q3 with_gate 수치(28.54/-20.62/19건,
39.99/-10.82/15건, -9.73/-44.37/19건)가 `eth_omega461_multiwindow_confirmation_gate_
20260814.py`에 이미 공개된 레퍼런스값과 **소수점까지 정확히 일치** — 이번 세션에 새로 만든
평가 파이프라인(프레임구성+예측생성+replay+게이트) 전체가 이 리포의 기존 감사된 G0
self-check와 완전히 같은 결과를 재현함을 확인, 신뢰도 뒷받침.

**Baseline vs Posfix with_gate 비교** (PnL%/MDD%, posfix-baseline delta):

| 창 | tier | baseline | posfix | PnL delta | MDD delta |
|---|---|---|---|---|---|
| 2025q1 | context | +28.54%/-20.62% | +4.22%/-24.24% | -24.32pp | -3.62pp(악화) |
| 2025q2 | context | +39.99%/-10.82% | -0.57%/-24.74% | -40.55pp | -13.93pp(악화) |
| 2025q3 | context | -9.73%/-44.37% | +7.70%/-22.39% | +17.42pp | +21.98pp(개선) |
| val | val | +54.88%/-31.11% | -11.65%/-24.22% | -66.53pp | +6.89pp(개선) |
| **oos_q1** | **oos_confirm** | **+28.17%/-15.48%** | **+3.95%/-21.26%** | **-24.22pp** | **-5.77pp(악화)** |
| **oos_q2** | **oos_confirm** | **+9.85%/-15.00%** | **+11.26%/-14.07%** | **+1.41pp** | **+0.93pp(개선)** |

**공식 판정 (사전등록 기준, `summarize_multiwindow` 그대로 적용)**: oos_confirm 2창이
동시(single touch)에 "with_gate PnL non-worse AND MDD slack이내"를 통과해야 CONFIRMED.
oos_q1은 PnL -24.22pp/MDD -5.77pp로 strict(0pp)는 물론 relaxed(3pp) slack도 초과해
**명백히 FAIL**. oos_q2는 PASS. 단일창만 통과로는 부족(이 게이트 모듈 자체의 존재이유가
바로 이 "한쪽 OOS창만 반전" 패턴을 잡아내는 것) → **최종판정: REJECTED_SIGN_MISMATCH**.

**⚠️ 이 결과를 "버그수정이 성능을 악화시켰다"로 단순 해석하면 안 되는 이유(교란변수들)**:
1. **단일시드**: `--seed 260620` 하나뿐 — [[tabm_hp_low_signal_pattern]] 메모리:
   "cross-seed std가 전형적 HP효과보다 큼, 단일시드 우승은 노이즈, N≥5 시드평균 필수".
   이 델타가 버그수정의 진짜 효과인지 시드노이즈인지 이 결과 하나로는 구분 불가.
2. **피쳐셋 변경 교란**: posfix는 158피쳐(canonical 파이프라인 자연진화분 포함), baseline은
   102피쳐 — "버그수정만의 효과"가 아니라 "버그수정+56개 신규피쳐"가 섞인 비교.
3. **risk sidecar 근사치**: posfix는 원본 sidecar를 빌려씀(실제 동적값이나, posfix
   번들/피쳐셋 전용으로 학습된 게 아님) — baseline은 자기 정식 sidecar. 사이징 자체의
   차이가 PnL/MDD에 얼마나 기여했는지 미분리.
4. **threshold 재튜닝 없음**: quality_threshold=0.50/0.75는 원본 기준 그대로 재사용 —
   158피쳐로 바뀐 모델에 최적이라는 보장 없음.
[[feedback_dl_needs_optimization_before_failure_verdict]]: "경량튜닝 1회로 '실패' 단정
금지" 원칙과 정확히 같은 상황 — 이 REJECTED 판정은 **"이 특정 단일시드·미세조정없는 posfix
번들"에 대한 판정이지, "pos_tp/pos_sl 등 버그수정 자체가 나쁘다"는 결론이 아니다.**
원인귀속이 필요해지면 (a) 피쳐셋 고정 재실행(102개로), (b) N≥5 시드, (c) posfix 전용
sidecar 재학습, (d) threshold 재튜닝 중 최소 (a)부터 격리해야 함 — 전부 미착수.

## 2026-08-18 후속 세션 9: 교란변수 2·3·4 제거 후 재평가 — 판정 불변(REJECTED), 단 격차 축소

사용자 지시 "2,3,4번 진행해서 다시 테스트해줘"(피쳐셋 고정+진짜 sidecar+threshold 재튜닝,
단일시드는 이번에도 미해결로 명시).

**(2) 피쳐셋 원본 102개로 고정 재학습**: `omega._numeric_feature_cols`를 모듈attribute
레벨에서 monkey-patch(공유모듈 소스 안 건드림, reduced80의 `--base-cols-allowlist-file`과
동일 발상을 `main()` 내부 대신 그 값의 소스함수에 적용 — `main()`의 `base_cols =
list(frames["feature_cols"])`로 자연전파). h48qual/zig075 원본 102개 컬럼이 order까지
완전히 동일함을 먼저 확인 후 하나의 공유 allowlist로 양쪽 다 처리. 로컬 dry-run
(`_prepare_frames()`만) 검증 후 서버재학습 — 둘 다 정상완료, base_cols=102 정확히 일치
확인, live_atr_tp_sl 버그수정 재확인. `quality_threshold_ranking.csv`에서 VAL-1위:
h48qual=0.40(q040), zig075=0.80(q080) — 원본 0.50/0.75와 다름(→ (4)와 자연스럽게 연결).

**(3) 진짜 risk sidecar 학습**: 전용 조사에이전트 위임 — `train_eval_omega4_2_risk_
sidecar_20260622.py`(1698줄)는 GPU불필요/저비용(수 분대, 본질은 소량(~trades수) 지도학습+
벡터화 그리드서치), 단 **동일 legacy TRAIN_CSV/EVAL_CSV 문제를 공유**(`omega4._prepare_
frames` 그대로 호출)하고 스크립트 자체 플래그는 그 중 1/3(TRAIN_CSV/EVAL_CSV)만 커버 —
REGIME3_CMAMBA/RISK/CURRENT 오버라이드는 안 됨. 다행히 **Python 모듈캐시(sys.modules)를
이용해 canonical wrapper를 먼저 import하면 sidecar_script.omega가 완전히 같은 객체라 3부분
오버라이드가 그대로 전파됨**(직접 assert로 확인) — 별도 재구현 불필요. 기존 확립된
`train_eval_omega4_2_risk_sidecar_eth_regime_jmlam4_20260809.py` 패턴을 그대로 모방해 새
래퍼 2개 작성. **1차 시도 둘 다 실패**: "no eligible validation-only risk mapping" —
`--min/max-validation-avg-notional`을 이미 0/0(제약해제)으로 시작했음에도
`--max-validation-mdd-abs`(기본 8.0%) 플로어를 만족하는 그리드 후보가 하나도 없었음(원본
번들에 맞춰진 하이퍼파라미터가 새 번들엔 안 맞는 이미 전례있는 패턴, jmlam4 래퍼 주석에
동일사례 기록됨). 50.0으로 완화 재실행 → 둘 다 성공(`constraint_pass=True,
fallback_used=False`, CLAUDE.md 감사스크립트가 요구하는 조건 충족).

⚠️ **CLAUDE.md Omega Artifact Integrity Gate 전체통과는 별도 이슈로 확인**: 조사에이전트가
`audit_omega_artifact_integrity_20260630.py`를 읽고 발견 — 이 게이트는 `dataset_lineage`
필드를 요구하는데 parent/sidecar 스크립트 둘 다 **애초에 이 필드를 전혀 안 씀**(감사스크립트
자신의 docstring이 "이 게이트 신설 이전 모든 report.json은 의도적으로 전부 fail"이라고
명시 — 원본 h48qual_q050/zig075_q075 sidecar도 예외 아님). 이건 스크립트 자체 수정이
필요한 별도 과제라 이번 라운드(Fresh-Forward 비교 개선) 범위 밖으로 명확히 분리 — "진짜
sidecar 학습완료"와 "공식 promotion_pass=true"는 다른 이야기임을 기록.

**(4) threshold 재튜닝**: 재학습 자체가 이미 전체 sweep을 수행하므로 별도 스윕 불필요 —
`quality_threshold_ranking.csv`(VAL-primary 정렬)의 1위 값을 그대로 채용(위 (2) 참고).

**최종 3자비교** (with_gate, PnL%/MDD%/거래수):

| 창 | tier | baseline(원본) | posfix(158피쳐) | pinned102(3교란변수 제거) | pinned102-baseline |
|---|---|---|---|---|---|
| 2025q1 | context | +28.54/-20.62/19t | +4.22/-24.24/26t | +0.88/-39.94/27t | -27.65pp/-19.32pp |
| 2025q2 | context | +39.99/-10.82/15t | -0.57/-24.74/22t | -9.94/-26.83/24t | -49.93pp/-16.01pp |
| 2025q3 | context | -9.73/-44.37/19t | +7.70/-22.39/22t | -10.72/-38.50/24t | -1.00pp/+5.87pp |
| val | val | +54.88/-31.11/22t | -11.65/-24.22/26t | **+114.03/-25.88/27t** | +59.15pp/+5.23pp |
| **oos_q1** | oos_confirm | +28.17/-15.48/19t | +3.95/-21.26/22t | **+22.75/-19.99/31t** | **-5.42pp/-4.50pp** |
| **oos_q2** | oos_confirm | +9.85/-15.00/10t | +11.26/-14.07/16t | **+36.21/-9.17/14t** | **+26.36pp/+5.83pp** |

**공식판정: 여전히 REJECTED_SIGN_MISMATCH — 단, 격차는 크게 좁혀짐**. oos_q1 PnL delta가
-24.22pp(posfix)에서 -5.42pp(pinned102)로 축소됐지만 여전히 baseline보다 낮고(strict
non-worse 기준 자체가 미달), MDD도 여전히 4.50pp 악화 — strict(0pp)/relaxed(3pp) 둘 다
불충족이라 FAIL 유지. oos_q2는 posfix보다도 더 크게 개선(+26.36pp) — PASS. **양쪽 동시통과
(single touch)가 필요한 기준이라 최종판정은 바뀌지 않음.**

**⚠️ 새로 관찰된 패턴 — VAL 선택편향 심화 가능성**: threshold를 h48qual VAL-PnL 기준으로
재선택했더니 VAL 자체가 극적으로 개선(+59.15pp, 54.88%→114.03%)됐는데, 이건 애초에 VAL에
맞춰 고른 threshold이므로 당연한 결과다. 반면 OOS는 한쪽만 크게 개선(oos_q2)하고 다른쪽은
여전히 못 미침(oos_q1) — 이 비대칭 패턴은 이 게이트 모듈 자신의 존재이유로 명시된
"risk-sizing→quality_threshold→후보"가 전부 같은 VAL 창에 순차적으로 맞춰지는 선택편향
구조(docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md, 이 게이트
모듈의 원출처 문서)와 정확히 같은 모양 — threshold 재튜닝이 교란변수를 하나 줄였다기보다
선택편향 축을 하나 더 추가했을 가능성도 배제 못함. context 3창의 MDD도 전반적으로 크게
악화(2025q1 -19.32pp, 2025q2 -16.01pp)됐는데, threshold 0.40(원본 0.50보다 낮음, 더
느슨한 진입기준)이 트레이드 수 증가+선택성 저하로 이어졌을 가능성.

**남은 유일 교란변수**: 단일시드(`--seed 260620`)뿐 — 이번 라운드에서도 명시적으로 미해결로
남김([[tabm_hp_low_signal_pattern]] 직결, N≥5 시드 없이는 이 -5.42pp/+26.36pp 자체가
시드노이즈 범위 안일 가능성을 배제 못함). 진짜 최종결론을 원하면 다음 단계는 N≥5 시드
평균이 유일하게 남은 방법론적 필요조건.
