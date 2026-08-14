# ETH Omega4.6.1 라이브 SLTP — MFE 분위수 회귀 기반 폭 사이징 (2026-08-13)

## 배경

사용자 지정 개선 과제: "SLTP의 너무 큰 범위로 인한 거래 수 감소". 라이브 `_ComponentConfig`
기본값(`trading_bot_modules/omega4_6_1_live.py:86-98`, `atr_window=192, tp_mult=12.0, sl_mult=6.0,
min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12`)은 `take_profit = clip(max(min_tp,
atr_pct*tp_mult), 0, max_tp)` 식이다. ETH 5분봉 192바 ATR%가 너무 작아(중앙값 0.26%) `min_tp`/
`min_sl` 플로어가 95~98.5%의 시간 동안 그대로 바인딩되어, 사실상 고정 7.5%/4.0% 폭이 되고 평균
보유기간이 366~925바(30~77시간+)에 달해 거래 수를 제한한다(`tmp/research_20260721/
exit_threshold_sweep_VAL.csv`).

**이미 닫힌 축**: `research_eth_omega461_tpsl_floor_sweep_20260728.py`가 플로어 "상수값"을 다르게
고르는 그리드서치를 이미 수행했고, 컴포넌트 단독으로는 승자가 있었지만 포트폴리오 레벨(h48qual+
zig075 우선순위 결합) 재검증에서는 확인되지 않았다(`tmp/research_20260728/
tpsl_floor_portfolio_check/summary.json`). 본 실험은 플로어 상수값 튜닝이 아니라 **다른 메커니즘**
— Odyssey 서브프로젝트가 검증한 방향별 MFE(Maximum Favorable Excursion) 분위수 회귀로 진입별 TP/SL
폭을 예측해서 "고정 상수" 대신 "학습된 예측치에 비례"하는 폭으로 대체하는 시도다
(`docs/experiments/eth_h48qual_mfe_quantile_quality_regression_20260812.md` 1단계: VAL R²=+0.08,
OOS R²=+0.14, spearman +0.28/+0.39, 둘 다 p<0.001 — 이 세션에서 MI/R² 게이트를 결정적으로 통과한
유일한 신호).

## 방법

### 메커니즘

`h48_conservative` 라벨(`tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/`)의
부산물인 `tb_long_mfe_h48_conservative`/`tb_short_mfe_h48_conservative`(48바 호라이즌 방향별 MFE,
매 bar마다 dense)를 타겟으로 HistGradientBoostingRegressor(quantile loss, q=0.5, depth=2,
early-stopping — Odyssey 게이트를 통과한 "강한 정규화" 레시피와 동일)를 롱/숏 방향별로 각 1개씩,
TRAIN 구간(2025-01-01~09-30, VAL 시작 이전으로 cutoff)에서만 학습했다. 라이브 진입 결정이 난 bar에
대해 그 방향의 예측 MFE(`width`)를 구해 다음 식으로 TP/SL을 대체한다(`_apply_atr_safety_sltp`와
동일한 `max(floor, driver*mult)` 클립 구조, driver만 `atr_pct`→`predicted MFE`로 교체):

```
take_profit = clip(max(FLOOR_TP, width * tp_scale), 0, max_tp)          # max_tp=0.22 불변(라이브 캡 유지)
stop_loss   = clip(max(FLOOR_SL, width * tp_scale * (sl_mult/tp_mult)), 0, max_sl)  # max_sl=0.12 불변
```

`FLOOR_TP=0.006`/`FLOOR_SL=0.004`는 스윕 대상이 아니라 `build_omega1_2_triple_barrier_labels_
20260619.py`의 `h48_conservative` 배리어 자체가 쓰는 고정 안전 플로어를 그대로 가져온 앵커값(폭의
주 구동원이 이제 학습된 예측치이므로 플로어는 거의 안 걸리는 백스톱 역할만). `tp_scale`
{1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0} 7개 값은 이 새 메커니즘 하나("예측 MFE 비례 폭")의 공격성을
보정하는 단일 축으로, 2026-07-28에 닫힌 "플로어 상수 자체를 스윕" 축과는 다른 조작변수다.

**변수 격리**: 방향/품질/진입타이밍/margin/leverage 사이징은 전부 라이브 번들 그대로 불변. 리스크
사이드카(`train_eval_omega4_2_risk_sidecar_20260622._risk_feature_frame`)가
`decision_take_profit`/`decision_stop_loss`를 사이징 모델의 **입력 피쳐**로 쓰기 때문에, margin/
leverage는 원본 ATR 기반 `dec`로 먼저 계산한 뒤, 그 계산이 끝난 **복사본**의 `take_profit`/
`stop_loss`만 MFE 예측치로 교체해서 실제 청산 루프(`replay_exit_variant`)에 넘겼다. `max_tp=0.22`/
`max_sl=0.12`(라이브 캡)는 유지. 두 컴포넌트(h48qual/zig075)의 `base_cols`가 완전히 동일(102개,
직접 확인)해서 MFE 모델도 공유 1세트만 학습.

### 두 피쳐 패널 (오케스트레이터 정정 반영)

이 스크립트는 처음부터 Odyssey "최종보스" v1/v2/v3 트랙(`docs/experiments/
eth_h48qual_final_boss_ensemble_20260812.md`)의 FINAL12+오토인코더 latent(16)=28개 축소 피쳐를
쓴 적이 없다 — 그 트랙은 라이브 번들을 대체하는 완전히 새 direction+quality 모델이라 스코프 자체가
다르고(이 실험은 라이브 번들을 안 건드리고 TP/SL 폭만 교체), 처음부터
`research_eth_omega461_exit_sweep_20260721.load_frame()`이 만드는 프레임에서 라이브 h48qual/zig075
번들이 실제로 쓰는 102개 `base_cols`(직접 검증: 전부 존재, all-NaN/all-zero 열 없음)를 그대로
써왔다. 대조군 요청에 따라 두 번째 패널을 추가로 구성해 나란히 비교했다:

- **base102**: 위 102개 `base_cols`(+POS_COLS 13개 0값, `parent._base_input`와 동일 계약).
- **final10_latent16**: FINAL12(h48orig/"최종보스" 트랙의 12개 선별 피쳐) 중 8개는 프레임에 직접
  존재, 2개(`funding_pressure_diff1`, `sum_toptrader_long_short_ratio_dt288`)는 원본 피쳐가 있어
  동일 변환식(diff1/dt288, `train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811.py`
  와 동일 recipe)으로 재구성 가능했지만, **2개(`m7_vae_error_dt288`, `sig_whale_dt288`)는 원본
  피쳐(`m7_vae_error`, `sig_whale`) 자체가 이 파이프라인에 전혀 없어 재구성 불가**했다(h48orig
  전용 별도 피쳐생성 라인, `load_frame()` 소스 CSV엔 없음). 더구나 `m7_vae_error`는
  `omega4_6_1_live.py`의 번들 무결성 체크가 명시적으로 금지하는 접두사(`m7_`)라 라이브 호환
  패널에는 애초에 있을 수 없는 피쳐다 — 단순 누락이 아니라 구조적 배제. 따라서 **FINAL10**(10개
  가용 컬럼)에 오토인코더 latent 16차원(102개 `base_cols` 풀을 입력으로, 64→32→16 인코더, 노이즈
  0.05, ReLU+Dropout(0.1), MSE, Adam lr=1e-3, batch=2048, patience=8, TRAIN 꼬리 15% 조기종료
  홀드아웃, TRAIN-fit 표준화만 — `verify_eth_h48qual_autoencoder_latent_mi_r2_gate_20260812.py`와
  동일 아키텍처)을 얹은 **26개** 피쳐로 근사했다(엄밀한 FINAL12+latent16=28개는 이 파이프라인에서
  재현 불가능함을 명시).

MFE 회귀 모델 레시피(HistGradientBoostingRegressor 설정)는 두 패널에서 완전히 동일 — 피쳐셋만
바뀐 순수 대조.

### 하네스 재사용

- 컴포넌트 단독: `research_eth_omega461_exit_sweep_20260721.py`의 `load_frame`/`prep_component`/
  `replay_exit_variant`(냉동 h48qual/zig075 TabM 번들+리스크 사이드카+냉동 VAL OOF 예측 CSV, causal
  bar-by-bar).
- 우선순위 결합(포트폴리오): `replay_omega4_6_1_greedy_router_20260706.py`의 `greedy_replay`(단일
  계정 공유 포지션 슬롯, h48qual>zig075 우선순위 — 라이브 `PRIORITY=("h48qual","zig075")`와 동일
  메커니즘, `trading_bot_modules/omega4_6_1_live.py`가 실제로 쓰는 그 구조). 이 모듈의
  `prepare_component()`는 `oof=False`로 하드코딩되어 있어(원래 OOS 전용 용도) VAL 예측 CSV에
  그대로 쓰면 잘못된 컬럼 프리픽스를 읽는다(`validation_predictions_q050.csv`는
  `*_oof_*` 컬럼만 있음, 직접 확인). 그래서 `prepare_component`를 그대로 호출하지 않고,
  `prep_component`가 이미 만든 `x`/`loaded`/`frame`에서 `rs._prepare_exit_runtime`/
  `hard._route_id`(둘 다 `prepare_component` 내부가 쓰는 것과 동일 함수)를 직접 호출해서 필요한
  나머지 필드(`base_np`/`exit_runtime`/`pos_idx`/`route`)만 구성했다 — 로직 재구현 없음, oof 처리만
  맞춤.
- VAL = 2025-10-01~12-31(`research_eth_omega461_exit_sweep_20260721.py`의 VAL과 동일 — 냉동 OOF
  예측 CSV가 2025-10-01부터만 존재해 CLAUDE.md 표준(09-01 시작)보다 한 달 짧음, 기존 캐비엇
  그대로). **OOS는 실행하지 않았다.**

`fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false`.

## 결과

### MFE 모델 자체 스코어카드 (VAL sanity gate, active 여부와 무관하게 전체 bar 대상)

| 피쳐셋 | n피쳐 | TRAIN R²(L/S) | TRAIN spearman(L/S) | VAL R²(L/S) | VAL spearman(L/S) |
|---|---:|---|---|---|---|
| **base102** | 115 | +0.148 / +0.194 | +0.478 / +0.494 | **+0.011 / +0.029** | **+0.285 / +0.324** |
| final10_latent16 | 26 | +0.083 / +0.076 | +0.381 / +0.390 | +0.001 / **-0.014** | +0.241 / +0.217 |

base102가 TRAIN·VAL 모든 지표에서 final10_latent16보다 우세(VAL R² short는 final10_latent16이
음수 — 상수예측보다 못함). 두 패널 다 VAL spearman은 양(+)이라 순위 신호 자체는 살아있음(Odyssey
게이트의 FINAL12 단독 패널 VAL spearman +0.28과 base102 결과가 거의 일치).

### 컴포넌트 단독 (VAL, baseline=현재 라이브 ATR-floor)

| 컴포넌트 | 변형 | pnl% | mdd% | trades | avg_hold_bars |
|---|---|---:|---:|---:|---:|
| h48qual | baseline | +5.45 | -11.62 | 29 | 670.3 |
| h48qual | base102 scale1.0 | -4.39 | -4.80 | 138 | 28.1 |
| h48qual | base102 scale2.0 | -3.36 | -5.93 | 97 | 71.7 |
| h48qual | base102 scale4.0 | -4.62 | -9.47 | 56 | 227.0 |
| h48qual | base102 scale9.0 | **+13.08** | -8.94 | 32 | 602.2 |
| h48qual | final10_latent16 scale1.0 | -6.39 | -6.78 | 143 | 28.1 |
| h48qual | final10_latent16 scale9.0 | +7.56 | -12.06 | 35 | 503.5 |
| zig075 | baseline | +40.31 | -13.07 | 29 | 725.6 |
| zig075 | base102 scale1.0 | -8.82 | -14.17 | 279 | 27.7 |
| zig075 | base102 scale2.0 | -2.03 | -12.13 | 164 | 74.8 |
| zig075 | base102 scale9.0 | +80.37 | -11.88 | 29 | 783.3 |
| zig075 | final10_latent16 scale1.0 | -9.88 | -12.85 | 285 | 26.0 |
| zig075 | final10_latent16 scale9.0 | +21.36 | -11.63 | 42 | 488.0 |

(전체 7개 scale × 2피쳐셋 × 2컴포넌트 = 28행은 `tmp/research_20260813/omega461_live_sltp_mfe_width/
component_variants_VAL.csv` 참고.)

### 우선순위 결합 포트폴리오 (h48qual>zig075, VAL, 단일 계정)

| 변형 | no_gate pnl% | no_gate mdd% | no_gate trades | avg_hold_bars | with_gate pnl% | with_gate mdd% | with_gate trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | +36.82 | -24.34 | 29 | 676.5 | **+54.88** | **-31.11** | 22 |
| base102 scale1.0 | -58.71 | -59.28 | 330 | 26.8 | -43.64 | -44.85 | 279 |
| base102 scale2.0 | -37.16 | -39.33 | 188 | 71.6 | -12.09 | -22.44 | 155 |
| base102 scale4.0 | -44.09 | -48.68 | 83 | 199.1 | -16.28 | -29.76 | 65 |
| base102 scale6.0 | -12.36 | -33.69 | 54 | 341.3 | -2.59 | -23.27 | 43 |
| base102 scale9.0 | **+130.25** | **-18.45** | 29 | 629.2 | **+143.53** | **-15.47** | 28 |
| final10_latent16 scale1.0 | -62.52 | -63.17 | 351 | 24.2 | -51.61 | -52.45 | 294 |
| final10_latent16 scale2.0 | -40.06 | -41.11 | 200 | 65.1 | -20.96 | -28.55 | 161 |
| final10_latent16 scale6.0 | -21.54 | -40.61 | 58 | 301.4 | +12.23 | -19.96 | 44 |
| final10_latent16 scale9.0 | +1.61 | -24.53 | 41 | 461.1 | +20.16 | -22.91 | 32 |

(전체 결과: `report.json`, 각 변형 원장은 `priority_combined_ledger_*_VAL.csv`.)

## 해석

**목표 (a) 평균 보유기간 단축 + 거래수 증가는 scale 1.0~6.0 구간에서 확실히 달성된다** —
포트폴리오 레벨 평균 보유기간이 676바→24~341바(2~28배 단축), 거래수가 29건→41~351건(1.4~12배
증가)로, "ATR 플로어가 거래빈도를 제약한다"는 원래 진단이 정확했음을 강하게 재확인한다.

**하지만 목표 (b) PnL/MDD 비악화는 정확히 같은 구간에서 전부 실패한다** — 포트폴리오 레벨
no_gate pnl은 baseline +36.82%에서 scale 1.0~6.0 전 구간 -12~-63%로 추락한다(컴포넌트 단독은
낙폭이 더 작지만 역시 전부 음전환). 원인은 승률 붕괴다: baseline 승률 41~48%가 scale 1.0~2.0
구간에서 27~37%로 떨어진다 — `h48_conservative` 라벨 자체의 SL 히트율(58%, TRAIN)과 정합적인
패턴으로, 48바 단기 MFE 중앙값(~0.75~0.8%)에 맞춰 폭을 좁히면 수수료/슬리피지(왕복 ~0.14%)
대비로는 여유가 있어도, 5분봉의 단기 노이즈에 SL이 TP보다 훨씬 자주 걸리는 구조로 재현된다.

**scale를 계속 키우면(9.0) PnL/MDD가 baseline을 능가하기도 하지만, 그 지점에서는 메커니즘이
사실상 원래 고정폭에 재수렴한다** — scale=9.0의 TP 중앙값은 예측 MFE 중앙값(~0.75%)×9≈6.75%로,
원래 고정 플로어 7.5%에 근접한다. 실제로 base102 scale9.0의 포트폴리오 거래수(29건)·평균
보유기간(629바)이 baseline(29건·676바)과 사실상 동일하다 — "더 넓혀서 baseline과 거의 같아지니
안 나빠졌다"일 뿐, 애초에 풀려던 문제(보유기간/거래수)를 그 지점에서는 더 이상 개선하고 있지
않다. **scale 1.0~9.0 전 구간에서 (a)와 (b)를 동시에 만족하는 지점은 없다** — 좁힐수록 빨라지고
많아지지만 손실이 커지고, 넓힐수록 손실은 줄지만 원래 문제로 되돌아간다는 단조적 트레이드오프
하나만 확인됐다.

**두 피쳐셋 비교**: base102가 VAL R²/spearman과 다운스트림 PnL/MDD 거의 전 구간에서
final10_latent16보다 우세하다(포트폴리오 scale9.0: +130.25% vs +1.61%). FINAL12 원안의 12개 중
2개(`m7_vae_error_dt288`, `sig_whale_dt288`)가 애초에 이 파이프라인에서 재현 불가능한 것과
별개로, 압축(26개, 오토인코더 latent 포함)이 원본 102개 대비 정보 손실을 일으켜 이 특정
회귀타겟(MFE)에서는 손해라는 결론이 이 실험 범위 내에서는 뚜렷하다.

## 결론

**실패 (정직한 보고)** — 목표 (a)는 메커니즘 진단 차원에서 강하게 확인됐지만(ATR 플로어가
거래빈도의 실제 병목이라는 가설이 옳았음), 목표 (b)를 테스트한 scale 1.0~9.0 어디에서도 (a)와
(b)를 동시에 만족하는 지점이 없다. scale을 키워 PnL/MDD를 baseline 수준으로 되돌리면 메커니즘
자체가 사실상 무력화된다(고정폭과 구별 안 됨). "예측 MFE 중앙값에 비례한 폭"이라는 구체적 공식은
이 형태로는 라이브에 얹을 근거가 안 된다. base102(102개 `base_cols`) 피쳐셋이
final10_latent16(FINAL10+오토인코더 latent, 28개 원안 중 26개로 근사)보다 전반적으로 우세하지만,
어느 피쳐셋을 쓰든 결론은 바뀌지 않는다.

향후 시도할 만한 방향(이 실험에서 실행하지 않음): (1) TP와 SL을 같은 비율로 묶지 않고 SL을 MAE
예측(`tb_long_mae_h48_conservative` 등, 이번엔 안 씀)으로 독립적으로 더 넉넉하게 잡아 승률
붕괴를 완화, (2) exit-head 임계값을 폭 축소와 함께 재보정, (3) scale 6~9 사이를 더 촘촘히 스캔해
"보유기간이 baseline 대비 뚜렷이 짧으면서 PnL/MDD가 안 나빠지는" 좁은 구간이 있는지 추가 확인.
전부 이 실험이 격리하려 한 "TP/SL 폭 하나만" 원칙을 벗어나거나(1), 새 자유도를 추가하므로(2, 3),
별도 실험으로 취급해야 한다.

## 산출물

- 스크립트: `scripts/research_eth_omega461_live_sltp_mfe_width_20260813.py`
- 결과: `tmp/research_20260813/omega461_live_sltp_mfe_width/report.json`,
  `component_variants_VAL.csv`, `priority_combined_ledger_*_VAL.csv`(baseline + 피쳐셋×scale별
  14개 변형 원장 — diagnostic 용도, promotion 근거 아님).
