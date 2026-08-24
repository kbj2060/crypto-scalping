# ETH ModernTCN/N-HiTS 백본 교체 후보 — direction+quality(h48qual 라벨 계약) 데이터 계약 (2026-08-16)

## 상태

| 컴포넌트 | 상태 |
|---|---|
| 계약 문서 | `draft` — 아래 스테이지 진행에 따라 갱신 |
| `ModernTCNBackbone`/`NHiTSBackbone` 구현 | `implemented, local sanity in progress` — `scripts/train_eval_eth_direction_quality_nhits_moderntcn_20260816.py` |
| `--stage isolation`(GCE/ELR/mixup 격리 검증) | `not started` — 서버 실행 대기 |
| `--stage hpsearch`(Optuna) | `not started` |
| `--stage final`(N≥5 시드) | `not started` |
| TabM 동일조건 대조군 | `not started` |

이 후보는 공식 Odyssey 계보(Odyssey1~4)에 속하지 않는다 — 결과가 확정되기 전까지 "Odyssey5"로 명명하지 않는다(`feedback_subproject_bootstrap_checklist`/사용자 2026-08-16 결정 관례, `eth_candidate_*` 명명 유지).

## 범위

- 목적: 사용자가 검토를 요청한 외부 문헌(`docs/experiments/eth_literature_review_cryptogat_and_918experiments_dl_architecture_20260816.md`, 918실험 벤치마크 arXiv:2603.16886)이 시간봉 멀티자산 회귀 벤치마크에서 ModernTCN(전체 1위, 75%)과 N-HiTS(ETH/USDT 양쪽 horizon 1위)를 찾았다. 사용자 지시(원문): "이것들을 시도해보지 않고 기각하지 말 것 — 모델링 결과는 최적화 과정에 크게 좌우되므로, 우리 데이터에 맞게 적용하는 추가 작업이 필요하다." dev에서 두 아키텍처를 모두 구현하고, 실제 학습은 server에서 진행하기로 확정.
- 모델 id: 없음 — 승격 가능한 아티팩트가 아직 없음 (연구/재테스트 단계).
- 구현 스크립트: `scripts/train_eval_eth_direction_quality_nhits_moderntcn_20260816.py`
- 리포트 아티팩트: `tmp/eth_candidate_nhits_moderntcn_direction_quality_20260816/*.json` (스테이지별 JSON, 아래 "Layer Contracts" 뒤 표 참고)
- 데이터/리소스 레지스트리: `docs/model_contracts/eth_candidate_nhits_moderntcn_direction_quality_data_resources_20260816.md`
- Owner: Model Architect 페르소나 단일 에이전트(Sonnet), `feedback_architect_team_single_agent_sonnet` 관례.

### Registry 중첩 근거 (research.line_id / related_prior_line_ids / prior_failure_reassessment / retest_design)

`pipeline/architecture_workbench.py`의 `validate_contract()`는 `research.line_id`가 기존 레지스트리
항목과 겹치거나 `related_prior_line_ids`가 채워지면 `prior_failure_reassessment`/`retest_design`
필드를 요구한다(스키마 `architecture_experiment_contract_v3`). 이 실험은 JSON 워크벤치 계약을 별도로
생성하지 않지만(대부분의 `research_*.py` 스크립트가 그렇듯 이 문서가 그 역할을 대신한다), 아래에서
동일한 두 질문에 답한다.

- **관련 과거 라인**: 이 저장소에 공식 레지스트리 항목으로 등록된 아키텍처 축은 아니지만(`research_line_registry.json`의 `prior_lines`는 신호/피처/레이블 축 중심이고 아키텍처는 별도 정성 문서로 관리됨), 사실상의 선행 라인은 `docs/experiments/eth_odyssey_dl_rl_architecture_research_20260816.md`("ETH Odyssey DL/RL architecture axis CLOSED", 메모리 `eth_odyssey_dl_rl_architecture_axis_closed_20260816`)다. 이 문서는 VSN/Diffusion/Mamba(3용도)/Transformer/TCN(dilated-causal, 2용도)까지 총 7개 시도가 전부 미승격·반증됐다고 결론짓고, 08-16 구현 충실도 감사에서 TCN 라인이 "가장 깊게 검증됐고 충실했다(N=5시드+150-trial HP서치, 미래정보 누수 없음)"고 재확인했다.
- **prior_failure_reassessment(과거 실패가 이번엔 왜 다를 수 있는가)**: ModernTCN과 N-HiTS는 이 표에 있는 어떤 아키텍처와도 다르다 — grep 확인 결과 "ModernTCN"은 이 저장소에서 이번이 최초 언급이고, "N-HiTS"는 2026-04 NeuralForecast 앙상블팩(`data/nf/NHITS_0.ckpt`, `docs/data_ensemble_cleanup_candidates.md`)의 죽은 체크포인트만 있을 뿐 direction/quality 분류 태스크로 단독 평가된 적이 없다(당시 라이브 추론에는 PatchTST만 쓰였다, iTransformer/TiDE/NHITS는 로드만 되고 미사용). 즉 "과거 실패"가 이 두 아키텍처 자체에 대해서는 존재하지 않는다 — 재테스트가 아니라 최초 테스트다. 다만 CLAUDE.md 규칙과 사용자 지시에 따라, 이미 닫힌 "아키텍처 교체는 근본적으로 안 통한다"는 상위 결론(정보량 병목이 원인)과 어떻게 다른 근거로 재시도하는지는 설명이 필요하다 — 답은: 그 결론은 "합리적인 아키텍처는 모두 시도했다"는 전제 위에 있었는데(`eth_odyssey_dl_rl_architecture_axis_closed_20260816`), 이 두 아키텍처는 그 "모두"에 없었고, 외부 문헌이 구체적으로 이 두 아키텍처를 candidate로 지목했다는 새로운 근거가 있다.
- **retest_design(과거 결과와 구분되는 재검증 설계)**: (1) 동일한 라벨(`zigzag_action`), 동일한 데이터 소스, 동일한 fresh-forward VAL/OOS 규율을 유지해 "정보량 병목" 가설과 직접 대조 가능하게 한다 — 만약 ModernTCN/N-HiTS도 실패하면 이는 세 번째(TCN, GBDT에 이은) 독립 아키텍처 계열의 실패로서 정보량 병목 가설을 더 강화하는 증거가 된다. (2) `feedback_modern_dl_training_checklist`의 전체 체크리스트(purge/embargo, EMA 가중치, sized warmup, label smoothing, GCE/ELR/mixup 격리검증)를 처음부터 적용해, 과거 TCN 시도가 안 가졌던 "최적화 과정 자체의 개선"을 실제로 반영한다 — 사용자가 명시적으로 "모델링 결과는 최적화 과정에 크게 좌우된다"고 지적했으므로, plain-CE 학습 하나만으로 판단하지 않는다. (3) N≥5 진짜 무작위 시드(Seed-Diversity Ensemble Promotion Gate) + Optuna HP 서치(20-30 trial) + VAL-only 후보 선정 + OOS 단일터치를 전부 지킨다.

### ModernTCN이 이미 반증된 plain dilated-causal TCN과 아키텍처적으로 구분되는 이유

"TCN"이라는 이름 때문에 이미 닫힌 `verify_eth_h48qual_tcn_sequence_model_20260812.py`(dilated
causal TCN, OOS 0/75 반증)의 재탕처럼 보일 위험이 있다 — 명시적으로 반박한다. 두 아키텍처는
"1D 컨볼루션을 쓴다"는 것 외에 공유하는 설계가 없다:

| 축 | 반증된 plain TCN | ModernTCN(Luo & Wang, ICLR 2024 Spotlight) |
|---|---|---|
| 커널 | 고정 크기 3, dilation만 지수증가(1,2,4,8,16)로 receptive field 확보 | large-kernel depthwise(9~21) 직접 사용 — dilation이 아니라 "큰 커널 자체"로 ERF 확보(ConvNeXt/RepLKNet 계보) |
| 파라미터화 | 표준 `nn.Conv1d`, 재파라미터화 없음 | **구조적 재파라미터화**: 학습 시 large+small 두 브랜치(각각 BN)를 병렬로 두고, 추론 시 `merge_kernel()`로 단일 conv에 흡수 — RepLKNet/ConvNeXt의 핵심 기법, plain TCN에 대응 개념 자체가 없음 |
| 채널 믹싱 | 없음(순수 시간축 컨볼루션, 채널은 hidden 투영에서만 섞임) | **두 단계 ConvFFN 분리**: ConvFFN1(변수별 독립, `groups=nvars`)이 시간·피처 믹싱, ConvFFN2(`groups=dmodel`)가 **변수 간(cross-variable) 믹싱** — plain TCN은 변수 간 상호작용을 명시적으로 분리해서 모델링하지 않음 |
| 다운샘플링 | 없음(전 레이어가 동일 시간 해상도, dilation만 증가) | **multi-stage**: patch-embedding stem + 스테이지 사이 strided conv로 시간축 해상도를 단계적으로 축소(ConvNeXt/Swin 계보의 계층적 백본 설계) |
| 정규화 | BatchNorm만 | **RevIN**(선택적, 인스턴스별 정규화) — 공식 classification 코드는 정의만 하고 실제로는 안 씀(레포 코드 직접 대조로 확인, 아래 "충실도" 절), 이 구현은 실제로 연결 |

즉 plain TCN이 "동일 커널을 dilation으로 넓게 보는" 고전적 WaveNet 계열이라면, ModernTCN은
"ConvNeXt/RepLKNet의 CNN 현대화 기법(large kernel, 구조적 재파라미터화, 계층적 다운샘플링)을
시계열에 이식"한 완전히 다른 설계 철학이다 — 이름의 "TCN"은 우연한 명명 계승일 뿐, 두 저자가
지수적 dilation 계열 논문을 인용조차 하지 않는다(RevIN·PatchTST·ConvNeXt·RepLKNet만 인용,
README `Acknowledgement` 절 확인). `research_line_registry.json`에 이 프로젝트의 공식 아키텍처
축이 별도 entry로 없어 이 판단을 문서화할 곳이 이 계약 문서 자신이다.

## Dataset Split

두 참조 스크립트(`scripts/verify_eth_h48qual_tcn_sequence_model_20260812.py`,
`scripts/tune_eth_h48qual_tcn_sequence_model_hpsearch_20260812.py`)의 관례를 그대로 따른다 —
CLAUDE.md의 표준 Fresh-Forward 경계(VAL 2025-09-01~12-31/OOS 2026-01-01~03-31)가 **아니다**,
아래 사유로 명시적으로 이탈:

| Split | 구간 | 사유 |
|---|---|---|
| Train | 2024-06-01 ~ 2025-09-30 | 참조 스크립트와 동일. 패널(`data/splits/year_oos/eth_features_2024_2026_analysis.csv`)이 2024-06-01부터 시작 |
| Validation | 2025-10-01 ~ 2025-12-31 | 참조 스크립트와 동일 |
| OOS | 2026-01-01 ~ 2026-02-28 | 참조 스크립트와 동일 — **표준 경계(03-31)보다 짧다**: quality 라벨 원천(`tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/oos_triple_barrier_labels.csv`)이 2026-02-28까지만 존재, 03-31까지 재빌드된 버전이 없음 |

Audit:

- Timestamp overlap: 없음 — 세 구간 경계가 겹치지 않게 필터링(`>=`/`<=` 부등호, 참조 스크립트와 동일 패턴).
- Duplicate timestamps: 패널 로드 시 5분 간격 연속성 assert(`diff==5min`)로 확인, 라벨 CSV는 `drop_duplicates("timestamp", keep="last")`.
- Warmup handling: `idx >= window - 1`인 행만 유효(과거 `window`bar 미만이면 제외).
- OOF/embargo: TRAIN 내부 fit/early-stop 분할에 `EMBARGO_BARS = max(window, 96)`bar 양쪽 purge/embargo 적용(아래 "체크리스트 통합" 절). VAL/OOS 자체는 embargo 없이 참조 스크립트와 동일한 경계 사용(모델은 TRAIN에서만 학습, VAL/OOS는 평가 전용).

## Shared Feature Contract

- Canonical feature source: `data/splits/year_oos/eth_features_2024_2026_analysis.csv` — 참조 TCN 스크립트와 동일 패널.
- Feature count: 8 (`SEQ_COLS`, 참조 스크립트의 "raw_lite" 변형 그대로 재사용 — HP서치 스크립트가 추가로 탐색한 나머지 4개 변형(`final12_seq`/`raw_wide`/`orderflow_funding`/`ohlcv_minimal`)은 그 스크립트 자신의 별도 탐색이지 기본 계약이 아니므로 재사용하지 않음).
- Normalization: TRAIN fit 구간(embargo 적용 후 fit_idx)에서 window 단위로 평균/표준편차 fit, `clip(-10,10)` — 참조 스크립트와 동일 절차.
- Missing fallback: `pd.to_numeric(errors="coerce")` 후 inf/NaN → 0.0.
- Stale handling: 해당 없음(라이브 미배포, 연구 스크립트).
- Live availability: 8개 컬럼 전부 causal 계산(과거 bar만 사용), 라이브 배포 시 검증 필요하나 이 계약 범위 밖.

Feature list:

```text
log_return, volatility_z, rsi, macd_hist, bb_width_z, wick_ratio, net_taker_ratio, cvd_12
```

Window: `WINDOW=96`(8시간) 기본값 — 참조 스크립트 기본값 그대로 유지. `--stage hpsearch`가
`{48, 96, 192}` 범위에서 재탐색(참조 HP서치 스크립트와 동일 categorical 범위)하므로 96이라는
선택 자체가 그냥 assert되는 게 아니라 데이터로 재확인된다. 96을 초기값으로 유지하는 근거: h48qual
quality 라벨의 horizon(48bar)의 2배로, 배리어 터치 시점 이전 맥락을 충분히 담는 동시에 window가
너무 길어 유효 표본 수(TRAIN 시작 이후 warmup bar 수)가 과도히 줄지 않는 절충점.

## Layer Contracts

| Layer | Input state/features | Train labels | Output | Artifact |
|---|---|---|---|---|
| `TabMControlBackbone`(대조군) | SEQ_COLS window의 마지막 timestep(8차원 flat) | 동일 | shared hidden (B,k=8,192) | `final_tabm_control.json` |
| `ModernTCNBackbone` | SEQ_COLS causal window (8, WINDOW) | 동일 | shared hidden (B,1,flatten_dim) | `final_moderntcn.json` |
| `NHiTSBackbone` | SEQ_COLS causal window (8, WINDOW) | 동일 | shared hidden (B,1,repr_dim) | `final_nhits.json` |
| `direction_head`(모든 백본 공유 헤드 설계) | 위 hidden | `zigzag_action`(3-class) | direction logits | - |
| `quality_head`(모든 백본 공유 헤드 설계) | 위 hidden | `h48_conservative`(3-class) | quality logits | - |

손실: `loss_dir + 0.80*loss_qual` (live 3-head 설계의 `+1.15*loss_exit` 항은 스코프 밖 —
"Known limitations" 절 참고). 클래스 균형 샘플가중치: `sklearn.utils.class_weight.compute_sample_weight("balanced", y)`,
direction/quality 각각 독립 계산(h48qual은 두 라벨이 서로 다른 분포이므로 zig075의
`same_as_direction` 설계처럼 하나의 가중치를 공유하지 않음 — 이것도 disclosed adaptation).

## Label Contract

- **direction**: `zigzag_action`, 3-class(CASH=0/LONG=1/SHORT=2). 소스:
  `tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531/zigzag_action_labels_{2024,2025,2026}.csv`.
  참조 스크립트와 완전히 동일한 로딩.
- **quality**: `h48_conservative` — 48-bar horizon, ATR-relative triple barrier(`tp_mult=1.2`,
  `sl_mult=0.8`, `min_tp=0.006`, `min_sl=0.004`), **h48qual이 실제 라이브 배포에 쓰는 원본
  레시피**(384bar 세션-로컬 재설계가 아님). 소스:
  `tmp/eth_h48_conservative_orig_padded_to_zigzag_timestamps_20260811/zigzag_action_labels_{2025,2026}.csv`
  (barrier 원본은 `tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/`, `pad_eth_h48_conservative_orig_labels_to_zigzag_timestamps_20260811.py`가 zigzag_action 타임스탬프 그리드에 결측=CASH로 패딩). 컬럼명이 파일 내부에서 `zigzag_action`으로 저장돼 있는 것은 패딩 스크립트 자체의 명명 관례이지 이 계약에서 만든 버그가 아님 — 로드 시 `h48_conservative`로 즉시 rename.
- Horizon: 48bar(quality), 미고정(zigzag pivot 기반, direction).
- Cost included: 라벨 자체엔 비용 미포함(라벨은 방향/품질 분류일 뿐), PnL 시뮬레이션(`omega._metrics`)이 fee/slip 별도 반영.
- Future path usage: 라벨은 사전 계산된 causal 파일에서 로드(참조 스크립트와 동일) — 이 스크립트 자신은 미래 시점 라벨을 만들지 않음, `_valid_indices`가 `window-1` 이전 구간과 라벨 결측 구간만 제외.
- Leakage controls: window는 `idx-window+1:idx+1`로 현재 bar까지만 사용(causal), embargo로 내부 split 경계 오염 방지.
- Known limitations: h48_conservative 라벨이 2026-02-28까지만 존재 → OOS가 표준 03-31 경계보다 짧음(위 Dataset Split 참고).

## Cost/Risk Assumptions

`train_eval_omega1_2_tabm_diffusion_risk_20260603.py`(`omega`)의 `BASE_TEMPLATE`/`_load_fee_slip()`을
그대로 재사용(참조 스크립트와 동일):

- Fee/Slippage: `omega._load_fee_slip()` 반환값, cost1/2/3(1x/2x/3x) 배율로 스트레스 테스트.
- Max notional exposure / Leverage: `omega.BASE_TEMPLATE["notional"]`/`["leverage"]` — 이 계약이 새로 정의하지 않음.
- Funding/Liquidation: 미반영(참조 스크립트와 동일 범위 밖).
- `max_hold`/`cooldown`: 0으로 강제(참조 스크립트 관례, TP/SL/reverse-signal만으로 청산).

## Output Contract

이 실험은 direction_head의 argmax만 거래 결정으로 변환(`build_dec`, 참조 스크립트와 동일 —
quality_head는 진단 전용, 게이팅에 쓰지 않음 — "Known limitations" 참고):

```text
action, side, notional_exposure, leverage, position_fraction, take_profit, stop_loss, max_hold_bars, cooldown_bars
```

Required report metrics (스테이지별 JSON, `tmp/eth_candidate_nhits_moderntcn_direction_quality_20260816/`):

```text
direction_balanced_accuracy, direction_macro_f1, quality_balanced_accuracy, quality_macro_f1,
model_pnl, model_trades, model_wr, always_short_pnl, always_long_pnl, beats_always_short (cost1/2/3, VAL/OOS)
```

## 아키텍처 구현 충실도 (Known limitations — 명시적 고지, 은폐 없음)

`eth_candidate_faithful_tabm_batchensemble_contract_20260816.md` 사건(TabM이 "쓴다"고 주장한
컴포넌트가 실제로 빠져 있었던 사건) 이후 이 저장소의 요구사항: 모든 단순화는 여기 명시한다.

1. **exit_head 미포함** — direction_head+quality_head만 학습(손실에서 `+1.15*loss_exit` 항 제외).
   exit_head는 포지션-상태 조건부 episodic 학습 문제(`train_eval_omega1_2_tabm_exit_head_20260603.py`의
   `_build_exit_dataset_independent`)로 bar-level window 분류와는 질적으로 다른 문제이고, 이
   저장소의 기존 TCN 축도 direction(`verify_eth_h48qual_tcn_sequence_model_20260812.py`)과
   exit_head(`research_eth_omega461_tcn_exit_head_val_20260813.py`)를 별도 스크립트로 분리했던
   선례를 따름. ModernTCN/N-HiTS 원 논문도 포지션-상태 exit 예측기가 아님.
2. **"라이브 TabM baseline" = 동일조건 대조군, 실제 라이브 번들 아님** — 실제 라이브 h48qual은
   HMM 레짐 라우터 + bull/bear/chop 3개 전문가 서브모델 + FINAL12 피처 엔지니어링을 쓴다. 이걸
   그대로 재현하는 건 백본 교체 질문과 무관한 별도의 무거운 파이프라인이라 범위 밖으로 판단 —
   대신 **동일 SEQ_COLS/window-last-bar/라벨/split/N시드 프로토콜 위에서 학습한 원본
   `ThreeHeadTabM`**(`scripts/train_eval_omega1_2_tabm_3head_20260603.py`의 `encode()`를 그대로
   복사, window의 마지막 timestep만 입력으로 사용)을 대조군으로 씀 — 진짜 라이브 프로덕션
   수치보다 좁은 조건이라는 점을 여기 명시.
3. **ModernTCN — RevIN을 실제로 연결** — 공식 classification 저장소(`ModernTCN-classification/models/ModernTCN.py`,
   2026-08-16 직접 코드 대조로 확인)는 `revin`/`affine`/`subtract_last` 생성자 인자를 받고
   `self.revin_layer`를 만들지만 `forward_feature()`가 그걸 **한 번도 호출하지 않는다** — 공식
   구현 자체의 죽은 코드다. 이 구현은 RevIN을 실제로 연결(토글 가능, HP서치 대상)했다 — 문헌이
   "optional RevIN"을 컴포넌트로 명시하고 이 데이터가 어느 쪽이 나은지 실측할 가치가 있다고
   판단했기 때문. `stem_ratio`/`dw_dims` 생성자 인자도 공식 코드 어디서도 실제로 참조되지 않는
   죽은 파라미터라 이 구현에서는 아예 제거.
4. **ModernTCN — 구조적 재파라미터화는 진짜 구현**(단순화 아님) — `merge_kernel()`이 공식
   `fuse_bn`/`get_equivalent_kernel_bias`/`PaddingTwoEdge1d` 로직을 그대로 이식, large+small
   두 BN'd conv 브랜치를 단일 conv로 흡수한다.
5. **N-HiTS — 분류 태스크 어댑테이션, 반드시 읽을 것** — 원 논문(Challu et al. 2023,
   arXiv:2201.12886)의 출력층은 미래 horizon 포인트 예측(회귀)이고, 이 저장소의 타깃은 3-class
   분류 2개 헤드다. 보존한 것: 스택별 서로 다른 시간 해상도로 pooling하는 계층적 multi-rate
   구조, 블록당 pooling→MLP→theta→basis expansion(선형보간) 메커니즘, doubly-residual
   backcast/forecast 누적(다음 스택이 이전 스택이 못 설명한 잔차만 봄). 바꾼 것: (a) 원 논문은
   단변량 `insample_y`이고 우리는 다변량(8채널) 윈도우라 `insample_y`를 8채널 전체로 일반화(원
   논문의 `hist_exog` pooling/flatten 경로와 동일한 방식으로 채널별 독립 pooling 후 flatten),
   backcast 크기도 스칼라가 아니라 `8×WINDOW`로 일반화. (b) "forecast"(미래 H스텝 예측)를
   고정 크기 `repr_dim` 잠재표현으로 재해석 — `_IdentityBasis`의 선형보간 메커니즘은 그대로
   재사용하되 목표 길이만 "H(미래 시점 수)"에서 "repr_dim(표현 차원)"으로 바꿈, 누적도
   `forecast = forecast + block_forecast`를 `repr_acc = repr_acc + block_repr`로 그대로 유지.
   (c) 원 논문의 "Level with Naive1"(마지막 관측값을 H번 반복하는 초기화)은 표현 공간에 대응
   개념이 없어 제거 — `repr_acc`는 0에서 시작. 상세: 스크립트의 `NHiTSBlock`/`NHiTSBackbone`
   docstring.
6. **quality_head는 PnL 시뮬레이션을 게이팅하지 않는다** — direction_head argmax만 거래로
   변환. 실제 라이브의 `quality_for_action` 파생 + threshold 보정 메커니즘(`cat_dq._prediction_output`)은
   백본 교체 질문과 직교하는 별도 라이브 시스템 로직이라 재구현하지 않음. quality_head는 자체
   classification 지표(balanced_accuracy/macro_f1 vs `h48_conservative`)로만 평가.

## 체크리스트 통합 (mid-task 사용자 추가 지시, `feedback_modern_dl_training_checklist`)

plain-CE 학습 하나만 돌리고 실패로 결론짓지 않는다 — 아래 전부를 학습 루프에 처음부터 내장:

- **Purge/embargo**: TRAIN 내부 fit/early-stop-val 분할 경계 양쪽에서 `max(window,96)`bar 제거(`_split_with_embargo`).
- **EMA 가중치**: `EMAWeights`(decay=0.999) — Polyak 평균 shadow, eval/최종 추론에 사용. ELR의 per-sample soft-target EMA(아래)와는 별개 메커니즘, 둘 다 구현.
- **Sized LR warmup**: 전체 optimizer step의 첫 10%를 lr×0.1→lr로 선형 램프(`_warmup_lr_lambda`).
- **Label smoothing**: eps=0.05, plain-CE 경로와 GCE 경로 양쪽에 동일하게 적용(`_smoothed_target`을 GCE의 `py` 계산에도 재사용 — soft target과 hard-label GCE를 결합하는 방식은 이 구현의 disclosed 확장).
- **GCE/ELR/mixup 격리 검증**(`--stage isolation`, 아키텍처별 N=5시드): 이 저장소 자신의 선행 TabM
  정칙화 연구(`scripts/research_eth_candidate_faithful_tabm_batchensemble_combo_regularizer_20260816.py`,
  `..._regularizer_isolation_20260816.py`, GCE q=0.7/ELR λ=3.0 β=0.7/mixup α=1.0)를 인용 — 그
  연구는 TabM+`zigzag_action`에서 GCE단독이 소폭 우세, ELR단독·mixup단독은 baseline보다
  오히려 악화, 셋 combine은 더 나쁨을 발견했다. **이 결과가 ModernTCN/N-HiTS나 quality 라벨에
  그대로 전이된다는 보장이 없으므로**, `none`/`gce_only`/`elr_only`/`mixup_only` 4-way를
  아키텍처별로 독립 재검증한다 — TabM 결과를 가정하지 않음.

순서(사용자 지시, 하이퍼파라미터 서치와 정칙화 선택을 동시에 흔들지 않기 위해):
`isolation`(정칙화 선택, 문헌 기본 아키텍처 용량) → `hpsearch`(Optuna, isolation 승자 정칙화
고정, 20-30 trial) → `final`(N≥5 시드, hpsearch 최적 HP + isolation 승자 정칙화).

## Red Team Gates

- [x] Train/validation/test timestamp overlap audit is zero. (부등호 필터 + 5분 연속성 assert)
- [x] No bfill/full-sample scaler/future feature enters live state. (TRAIN fit_idx만으로 표준화 fit, causal window)
- [ ] Fee/slippage 1x/2x/3x ranking is reported. (`--stage final` 실행 후 채움)
- [ ] Score/probability buckets are calibrated against realized net PnL. (범위 밖 — quality_head는 진단 전용, 위 "Known limitations" 6 참고)
- [ ] Monthly/weekly walk-forward is reported. (범위 밖 — VAL/OOS 단일 구간만, 참조 스크립트 관례와 동일)
- [ ] Live train state parity is checked. (연구 스크립트, 라이브 미배포)
- [x] Funding/liquidation limitations are documented. (Cost/Risk Assumptions 절)
- [ ] N≥5 진짜 무작위 시드(Seed-Diversity Ensemble Promotion Gate) 확인 — isolation/final 스테이지 실행 후 시드 리스트를 이 표 아래에 기록.
- [ ] `_is_clustered_seed_list` 통과(고정 간격 아님) — `random.SystemRandom().sample` 사용으로 설계상 보장, 실행 후 실제 시드값 기록.

## Open Issues

1. `--stage isolation`/`hpsearch`/`final` 서버 실행 결과 미반영 — 완료 후 이 문서와
   `docs/experiments/eth_candidate_nhits_moderntcn_direction_quality_20260816.md`에 기록.
2. VAL/OOS 경계가 표준 Fresh-Forward 규칙과 다름(위 Dataset Split) — 표준 03-31 OOS까지
   확장하려면 h48_conservative 라벨을 03-31까지 재빌드해야 함, 이 계약 범위 밖.
3. TabM 대조군의 "동일조건" 정의가 라이브 프로덕션(115차원, 레짐 라우팅)보다 훨씬 좁음 — 결과
   해석 시 "ModernTCN/N-HiTS가 이 좁은 TabM 대조군을 이기는가"와 "라이브 h48qual을 이기는가"를
   혼동하지 않을 것.
