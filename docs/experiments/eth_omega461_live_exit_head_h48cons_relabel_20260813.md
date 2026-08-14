# ETH Omega4.6.1 라이브 exit_head — h48_conservative 배리어 재라벨 재학습 (2026-08-13)

상태: `research_only_not_live_promoted` — 결론: **실패 (VAL 성공 기준 (b) 미충족)**

라이브 어댑터(`trading_bot_modules/omega4_6_1_live.py`), `trading_bot.py`, `runtime_config.py`, `.env`,
프로덕션 SLTP/exit_head 번들은 전혀 건드리지 않았다. 순수 리서치 스크립트만 새로 작성했다.

`fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
`saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`. VAL만 실행했고
OOS(2026-01-01~03-31)는 전혀 로딩·평가하지 않았다.

## 배경

라이브로 서빙 중인 h48qual/zig075 3-head TabM parent(`true_3head_tabm_bundle.pt`)의 exit_head는
구조적으로 무력화돼 있다(`scripts/research_eth_omega461_exit_head_retrain_eval_20260721.py`의
`giveback_min` 그리드 재학습이 VAL/OOS에서 완전히 동일한 결과를 낸 것으로 확인). 근본원인은
`scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`의
`_build_exit_dataset_entry_label_terminal_giveback`이 두 라이브 번들의 실제 학습 레시피이고,
"세그먼트"가 `zigzag_action`이 연속되는 구간(학습 구간 전체에서 732~813개뿐)이며, 양성 라벨
2,182건 중 2,179건(99.86%)이 "이 세그먼트 자신의 끝에 가깝다"는 `terminal_window_exit`(지그재그
피벗 임박 신호일 뿐 P&L과 무관)이라는 점이다. `docs/experiments/eth_omega461_exit_learning_20260724.md`가
이미 별도 hazard/rescue 분류기 계열(냉동 SLTP 위에 얹는 방식)을 전부 시도해
`RETIRED_DO_NOT_SHADOW_FOR_PROMOTION`으로 닫았으므로, 이번 시도는 그 축을 반복하지 않고
**exit_head 자체의 라벨 소스를 교체**하는 다른 축을 시험했다.

## 방법

### 새 라벨 함수

`scripts/research_eth_omega461_exit_head_h48cons_relabel_20260813.py`에 새 함수
`_build_exit_dataset_entry_label_h48cons_barrier`를 작성했다(기존
`_build_exit_dataset_entry_label_terminal_giveback`은 수정하지 않음). 두 가지를 바꿨다:

1. **후보 밀도 확대**: 세그먼트의 첫 bar만이 아니라 `zigzag_action ∈ {1,2}`인 **모든 bar**를
   독립 진입 후보로 취급(학습 구간에서 37,164개 vs 기존 732~813개 세그먼트).
2. **"끝" 기준 교체**: 각 후보의 "임박 청산" 기준을 그 세그먼트 자신의 끝(`end_i`) 대신, Odyssey
   서브 프로젝트가 만든 dense per-bar 라벨
   `tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/train_triple_barrier_labels.csv`
   (레시피: `scripts/build_omega1_2_triple_barrier_labels_20260619.py`)의 **그 진입 자신의
   h48_conservative(48bar, ATR 기반) 배리어 해소 지점**으로 바꿨다. 컬럼 의미를 직접 읽어서
   검증: `tb_{long,short}_bars_h48_conservative`는 `entry_i(=i+1)`부터 센 1-index 오프셋(값 1 =
   진입 bar 자신에서 해소)이라서, 절대 해소 인덱스는 `entry_i + bars - 1`이다. 이걸 500개
   무작위 샘플에서 `entry_timestamp_h48_conservative`와 직접 대조해 검증했고 불일치 0건이었다.
   `frame`(omega4 학습 프레임, 41,744행, 2025-05-08~09-30)과 tb CSV(78,470행, 2025-01-01~09-30)는
   타임스탬프로 조인했고, 해소 bar가 `frame` 내에서 정확한 타임스탬프로 재확인되지 않는 87건
   (프레임 갭 6건 + 매칭 실패 81건)은 조용히 클램프하지 않고 건너뛰었다(스킵으로 카운트).

   포지션 피처 생성(`exit_head._position_feature_row`, side/hold_bars/unrealized/mfe/mae/
   giveback/dist_to_tp/dist_to_sl 등)과 adverse-unrealized/mfe-giveback 분기 로직은 원본과
   완전히 동일하게 유지했다 — `terminal_window=3`, `adverse_unreal=-0.010`,
   `min_mfe_for_giveback=0.006`, `giveback_min=0.65` 전부 원본 기본값 그대로. 바뀐 건 "무엇을
   세그먼트의 끝으로 볼 것인가"와 후보 밀도, 두 축뿐이다.

### 재학습/평가 도구

- 재학습: `scripts/train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622.py`의
  `_fit_exit_head_only`를 직접 import해서 재사용(encoder/direction_head/quality_head 동결,
  exit_head만 AdamW로 재학습, class-balance × route-prob 가중 CE loss). 새 코드를 복붙하지
  않고 기존 검증된 함수를 그대로 호출했다.
- 베이스라인 번들: `research_eth_omega461_exit_sweep_20260721.py`의 `COMPONENTS`에 박힌 실제
  라이브 번들 경로(h48qual: `..._zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630`,
  zig075: `..._current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629`)를
  그대로 가져다 썼다 — `docs/model_contracts/odyssey_eth_h48qual_data_resources_20260812.md`의
  "라이브 프로덕션 번들" 표와 경로 일치 확인.
- 평가: `research_eth_omega461_exit_sweep_20260721.py`의 `prep_component`/`replay_exit_variant`를
  그대로 재사용. `EXIT_THRESHOLD=0.95` 고정(라벨-소스 효과와 threshold 효과를 분리하기 위해
  건드리지 않음). direction/quality/side/TP-SL은 원래 인증에 쓰인 냉동 OOF 예측 CSV
  (`tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/{component}/validation_predictions_qXXX.csv`)에서
  그대로 가져오므로, baseline과 new 사이에서 오직 exit_head 가중치만 바뀐다. VAL=2025-10-01~12-31.
- 학습/조인 모두 direction_label_dir=`zigzag_action_labels_20260531`, quality_mode=
  `same_as_direction`로 **한 번만** 빌드해서 h48qual/zig075 둘 다에 재사용했다 — 두 라이브
  번들의 report.json을 직접 대조해 확인한 바, exit_head 학습 입력(`train_df`/`s_train_label`)은
  quality_mode에 의존하지 않고 두 컴포넌트가 완전히 동일한 `direction_label_dir`을 쓰기 때문.

### 실행 환경

dev 머신 GPU 없음(`torch.cuda.is_available()=False`, 12 vCPU) 확인. 800개 후보로 스모크테스트
(2.6초) 후 전체 37,164개 후보로 확장했고, 전체 데이터셋 빌드가 CPU로 약 5분, exit_head 재학습
(3 expert × 2 component, 8 epoch)이 약 20분 안에 끝나 서버 위임 없이 로컬에서 전부 완료했다.

## 1단계 체크포인트 — 라벨 밀도/다양성

| | 원본 레시피(배포됨, `max_exit_samples=30000`에 truncate) | 원본 레시피(전체 미절단 학습구간) | 새 레시피(h48cons, 매 action bar) |
|---|---:|---:|---:|
| 독립 후보 수 | 732 | 813 | **37,158** (스킵 87건) |
| bar-row 수 | 30,000 | (미절단시 더 큼, 미계산) | 540,088 |
| 양성 비율 | 7.27% | - | **19.64%** |
| 양성 사유 구성 | `terminal_window_exit` 2,179건(99.86%), `mfe_giveback_exit` 3건(0.14%), `adverse_unreal_exit` 0건 | - | `near_barrier_resolution_exit` 106,058건(100%), adverse/giveback 0건 |
| 그 "끝"이 의미하는 것 | 지그재그 재라벨링 알고리즘의 스윙 경계(사후적 아티팩트, P&L과 무관) | 〃 | 실제 48bar ATR 트리플배리어 해소(tp 23,150 / sl 11,832 / timeout 2,176건) |

**밀도**: 37,158 vs 732(배포 기준)는 50.8배, vs 813(전체 미절단)는 45.7배 — 명확히 크다.
**다양성**: 후보 개수·양성 비율은 명확히 개선됐지만, 청산 "사유" 카테고리 자체는 원본과 마찬가지로
한 카테고리(`near_barrier_resolution_exit`)가 100%를 차지해 `adverse_unreal_exit`/
`mfe_giveback_exit`는 새 레시피에서도 0건이었다 — **정직하게 밝히면, 카테고리 다양성 지표만
보면 개선되지 않았다.** 다만 그 지배적 카테고리 하나의 **의미 자체**가 "지그재그 재라벨링
아티팩트"에서 "실제 tp/sl/timeout 3갈래 배리어 해소"로 바뀌었다는 점에서 인과적 P&L 근거는
확실히 더 풍부해졌다. 이 체크포인트는 통과로 판단해 2단계(재학습+평가)로 진행했다.

## 재학습 + VAL 평가 결과

`EXIT_THRESHOLD=0.95` 고정, VAL=2025-10-01~12-31, `COST_MULT=1.0`(하네스 기본값).

### h48qual (q050)

| | baseline(냉동 SLTP+exit0.95, 원본 exit_head) | h48cons_relabel |
|---|---:|---:|
| PnL | **+5.45%** | **-3.90%** |
| MDD | -11.62% | -4.22% |
| trades | 29 | 203 |
| WR | 41.4% | 41.9% |
| avg_hold_bars | 670.3 | 6.34 |
| max_trade_pnl | 4.87% | 0.57% |
| p95_trade_pnl | 4.69% | 0.30% |
| exit_reasons | `stop_loss:17, take_profit:11, forced_end:1` (`exit_head:0`) | `exit_head:203` (전부) |

### zig075 (q075)

| | baseline | h48cons_relabel |
|---|---:|---:|
| PnL | **+40.31%** | **-9.94%** |
| MDD | -13.07% | -11.73% |
| trades | 29 | 431 |
| WR | 48.3% | 46.9% |
| avg_hold_bars | 725.6 | 10.24 |
| max_trade_pnl | 8.68% | 0.96% |
| p95_trade_pnl | 6.87% | 0.38% |
| exit_reasons | `stop_loss:15, take_profit:13, forced_end:1` (`exit_head:0`) | `exit_head:431` (전부) |

## 실패 원인 진단

exit_head는 확실히 "발동"한다(baseline `exit_head:0` → 재학습 후 두 컴포넌트 모두 100% `exit_head`) —
더 이상 무력화 상태가 아니다. 하지만 **극단적으로 과발동**한다: 거래 수가 29건→203/431건으로
7~15배 늘고, 평균 보유기간이 670~726bar에서 6~10bar로 붕괴했다. 진입 직후 거의 즉시 청산하는
패턴이 두 컴포넌트·6개 expert 전부에서 일관되게 나타났다(8 epoch 전부 소진, exit validation
loss 0.52~0.53으로 수렴, 특정 expert의 학습 실패가 아니라 구조적 패턴).

원인을 두 단계로 분리해서 확인했다(`/tmp/claude-1000/.../scratchpad/probe_window_length.py`,
벡터화 연산이라 수초 내 재현 가능):

**(1) `terminal_window=3`이 짧은 창 길이와 충돌**: 새 후보의 h48_conservative 해소까지 걸리는
bar 수는 중앙값이 long 10bar/short 9bar(평균 15.06/13.91, 표준편차가 평균과 비슷할 정도로
우측 꼬리가 두꺼움)로, 원본 세그먼트 평균(~41bar)보다 훨씬 짧다. 그 결과 **후보의 17.86%
(6,637/37,164)는 창 길이 자체가 `terminal_window`(3bar) 이하**라서, 그 후보의 모든 행 — 진입
직후(`hold_bars=0`) 첫 행까지 포함 — 이 전부 양성으로 라벨된다. 이 짧은 창들은 전체 540,139행
중 겨우 2.68%(14,495행)만 차지하지만, 서로 다른 6,637개 후보가 동일하게 "막 진입한 상태 →
청산"을 가르치면서 `hold_bars≈0` 근방 특징 공간에 일관되고 학습하기 쉬운 신호를 심는다.

**(2) 더 근본적인 문제 — h48_conservative와 라이브 실제 배리어 사이의 기간 스케일 불일치**:
h48_conservative의 ATR 배율은 `tp_mult=1.2, sl_mult=0.8`, 바닥값은 `min_tp=0.006, min_sl=0.004`
(`scripts/build_omega1_2_triple_barrier_labels_20260619.py`의 `CONFIGS`)인 반면, 실제 라이브
h48qual/zig075가 평가 시점에 쓰는 ATR 동적 SLTP는 `atr_window=192, tp_mult=12.0, sl_mult=6.0,
min_tp=0.075, min_sl=0.040`(`research_eth_omega461_exit_sweep_20260721.py`의 `COMPONENTS`)로,
바닥값 기준 약 12배(TP)·10배(SL) 더 넓다. 이 배리어 폭 차이가 baseline의 실측 평균 보유기간
(670~726bar)과 h48_conservative 후보의 실제 해소 기간(중앙값 9~10bar, 약 67~72배 차이)에
그대로 반영된다. 즉 새 라벨은 **"타이트하고 빠르게 끝나는 가상 거래"의 청산 타이밍을
가르치는데, 실제로 exit_head가 운용되는 라이브 포지션은 그보다 수십 배 더 길게 간다** — `pos_hold_bars`가
모델 입력 피처에 그대로 들어가 있어서, 학습 시 "hold_bars가 두 자릿수 초반만 돼도 이미 청산권"이라고
학습한 모델이 실제로 hold_bars가 그 이상으로 계속 쌓이는 라이브/평가 궤적에서 이르게, 광범위하게
발동한 것으로 해석된다. MDD가 두 컴포넌트 모두 개선된 것(h48qual -11.62%→-4.22%, zig075
-13.07%→-11.73%)도 같은 메커니즘의 부작용이다 — 손실이든 이익이든 거래가 커지기 전에 잘라내니
당연히 낙폭도 작아지지만, `max_trade_pnl`이 4.87%/8.68%→0.57%/0.96%로 붕괴한 데서 보듯 큰
이익 거래까지 함께 잘려서 PnL 자체가 마이너스로 전환됐다. **위험조정수익 개선이 아니라 과발동의
부작용**으로 판단한다.

## 결론

**실패 — 성공 기준 (b) 미충족.** (a)(exit_head 발동)는 기술적으로 충족했지만 병적으로
과발동하는 형태였고, (b)(VAL pnl/mdd가 냉동 baseline 대비 나빠지지 않음)는 두 컴포넌트 모두
PnL이 뚜렷한 양(+5.45%/+40.31%)에서 음(-3.90%/-9.94%)으로 전환되며 명확히 실패했다. 1단계
체크포인트(라벨 밀도·후보 다양성)는 원안 대비 확실히 개선됐고 그 가설 자체는 이 실험으로
검증됐다고 볼 수 있지만, **h48_conservative 배리어를 그대로 가져다 쓰는 이번 구체적 구현은
라이브 SLTP의 실제 배리어 폭·보유기간 스케일과 근본적으로 맞지 않아 채택할 수 없다.**

향후 시도가 있다면(이 문서는 구현하지 않고 방향만 남긴다 — 판단은 오케스트레이터):
h48_conservative 대신 실제 라이브 ATR 동적 배리어 폭(`atr_window=192, tp_mult=12, sl_mult=6,
min_tp=0.075, min_sl=0.040`)으로 재계산한 dense per-bar 배리어 라벨을 쓰거나, `terminal_window`를
고정 bar 수 대신 각 후보 창 길이에 비례하는 값(예: 창 길이의 15~20%)으로 바꿔서 창 길이가
짧은 후보의 "진입 직후 전부 양성" 아티팩트를 없애는 방향이 논리적으로 다음 단계다.

## 산출물

- 새 스크립트: `scripts/research_eth_omega461_exit_head_h48cons_relabel_20260813.py`
  (`_build_exit_dataset_entry_label_h48cons_barrier`, `_retrain_component_exit_head`,
  `_evaluate_val`, CLI `--stage {dataset_only,full}`)
- report.json: `tmp/causal_regen_20260516/eth_omega461_exit_head_h48cons_relabel_20260813/report.json`
  (1단계 체크포인트 전체 + 컴포넌트별 재학습 요약 + VAL 비교 포함)
- 재학습된 번들(연구용, 라이브 미승격):
  `tmp/causal_regen_20260516/eth_omega461_exit_head_h48cons_relabel_20260813/{h48qual,zig075}/true_3head_tabm_bundle.pt`
- 이 문서: `docs/experiments/eth_omega461_live_exit_head_h48cons_relabel_20260813.md`

## 준수 확인

`fresh_forward_bar_by_bar=true`(VAL 리플레이는 `replay_exit_variant`의 단일 순방향 causal
루프), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`(direction/quality/TP-SL은 냉동 OOF 예측 CSV, exit_head만
매 bar 순방향으로 평가). direction_head/quality_head는 전혀 재학습하지 않았다(encoder 포함
동결, exit_head 가중치만 갱신). OOS(2026-01-01~03-31)는 이 실험에서 전혀 로딩되지 않았다.
`docs/experiments/eth_omega461_exit_learning_20260724.md`의 hazard/rescue 축(별도 분류기를
냉동 SLTP 위에 얹는 방식)은 이번 실험과 무관하며 재개하지 않았다.
