# Omega4.6.1 부모(zig075) — 데이터를 되살려 다시 학습하고 **배포본과 맞붙였다** (2026-09-17)

선행: [A~C](omega461_longwindow_retrain_stages_abc_20260917.md) · [D 깊이 사다리](omega461_longwindow_retrain_stageD_depth_ladder_20260917.md)

## 0. 한 줄

데이터 두 구멍(펀딩·라벨)을 **둘 다 메웠다**. 부모는 잘 학습된다(bacc **0.586**, 방향정확
0.70 — 이 라인 최고). 그런데 **돈은 배포본이 더 번다**: VAL 2026-03~06 에서 1h 총이익
배포본 **+4.94bp** > 내 base **+3.80** > 내 deep **+0.42**. 셋 다 CI 가 비용선을 배제하지
못한다(독립일 122).

## 1. ⭐역공학은 필요 없었다 — 생성기가 커밋돼 있었다

D단계에서 "zigzag 라벨이 2024 부터만 있다"며 역공학을 시도했고 실패했다. 실제로는
**`scripts/build_wave3_action_labels_20260531.py` 가 그 생성기**다 — 파일명이 wave3 라서
「zigzag」로 찾을 때 안 걸렸고, 내부 `DEFAULT_OUT` 이 `zigzag_action_labels_20260531` 이다.

**손대지 않고 그대로 돌린 결과:**

| 연도 | 정본 행 | 재생성 행 | 행 일치율 |
|---|---|---|---|
| 2024 | 105,380 | 105,380 | **1.000000** |
| 2025 | 105,101 | 105,101 | **1.000000** |
| 2026 | 16,897 | 69,696 | 앞 13,601행 **1.000000** |

2026 불일치 42행은 **연속 2덩어리**(2026-02-17, 그리고 마지막 8행)로 전부 정본 절단면
근처다 — 지그재그는 *확정된 피벗*까지만 라벨하므로 입력이 길어지면 절단면의 미확정 파동이
바뀐다. 생성기 불일치가 아니라 라벨 정의의 성질이다.

⇒ **라벨을 2021-12~2026-08 로 확장**했다(파라미터는 다시 적지 않고 재현 실행의 audit JSON
에서 읽는다). ⚠️정본이 **연도별로 따로** 만들어졌으므로 여기서도 연도별로 잘랐다 —
이어붙여 한 번에 돌리면 배포 라벨과 다른 라벨이 된다.

| 연도 | 행 | CASH/LONG/SHORT | segments |
|---|---|---|---|
| 2021(12월) | 8,928 | 0.151 / 0.391 / 0.458 | 233 |
| 2022 | 105,120 | 0.138 / 0.439 / 0.423 | 2,407 |
| 2023 | 105,120 | 0.053 / 0.480 / 0.467 | 1,051 |

## 2. ⭐펀딩 복구 — 채움 324,577행 → **0행**

D §3 의 결함: 펀딩 원천이 `TOTAL_ETHUSDT_fundingRate_2025_2026.csv`(2025~) 뿐이라
**2022·2023·2024 가 전량 중앙값**이었고 파생 9열이 같이 상수였다.
`/fapi/v1/fundingRate` 로 **5,343건(2021-11~2026-09)** 복구 — 저장소의 `_fetch_funding` 을
그대로 재사용(페이징 이미 구현). 5분봉 병합 결측 **0**.

| | 옛 프레임 | 복구본 |
|---|---|---|
| 펀딩 중앙값 채움 | 324,577행 (65.6%) | **0행** |
| 연도간 상수열 | **11개** | **3개** |

잔존 3개(`sum_toptrader_long_short_ratio` `whale_conviction` `ofti`)는 **2022 에만** 상수고,
바이낸스 롱숏비 이력 자체의 한계다. ⭐**연도별 최빈값 점유율 보고를 빌더의 1급 산출물로
넣고 assert 를 걸었다** — 「NaN 0」이 「정보 있음」이 아니라는 D 의 교훈.

B2(balnobb 2022~23 재적합) 2024 일치율 **96.19%**(기준선 80.3%) · 표본외 정확도 94.33%.
C2(walk-forward 레짐) 이음매 Δconfidence **−0.0011** · margin −0.0021 · entropy +0.0055.

## 3. 왜 zig075 인가 (h48qual 아님)

`runtime_contract.json` 을 읽어보니 **두 부모의 방향 라벨이 동일**했다:

| | direction_label_dir | quality_mode | quality_label_dir |
|---|---|---|---|
| h48qual | `zigzag_action_labels_20260531` | `quality_label_action` | `sltp_h48_conservative_padded…` |
| zig075 | `zigzag_action_labels_20260531` | **`same_as_direction`** | None |

⇒ 둘의 차이는 **품질 머리의 타깃뿐**이다. zig075 는 추가 라벨 없이 완결되고, h48qual 은
SL/TP 라벨 파이프라인을 2022~2023 으로 또 돌려야 한다. 그래서 zig075 를 골랐다.

## 4. 학습·테스트

배포 번들의 **115열 계약**(102 base + 13 pos)을 `assert` 로 맞췄다.
⭐추론은 라이브와 같은 **hard routing**(`_route_expert` 가 전문가 하나를 고른다) + q=0.75 게이트
+ softmax 후 k 평균. (D 사다리는 확률 가중 혼합이었다 — 판끼리 비교엔 무해하나 절대 수치를
라이브와 견주려면 안 된다.)

**VAL 2026-03-01~06-30** (35,136봉 · 독립일 **122**). 배포 부모의 TRAIN(2025-01~09)·
VAL(2025-10~12)·OOS(2026-01~02) **전부 밖**이고, 예약된 single-touch OOS(2026-07-01~09-30)
**앞**이다 — 셋 모두에게 공정한 전진 분할. **OOS 는 건드리지 않았다.**

| | TRAIN | 봉 | bacc | 방향정확 | 통과율 | **1h 총이익** | 4h 총이익 |
|---|---|---|---|---|---|---|---|
| **deep** | 2022-01~2026-02 | 437,713 (5.6배) | **0.5803** | 0.7018 | 8.3% | **+0.42bp** [−1.03,+1.85] | +4.96bp |
| **base** | 2025-01~2026-02 | 122,093 | 0.5591 | 0.6887 | 7.7% | **+3.80bp** [+3.13,+4.53] | +8.19bp |
| **배포본** | 2025-01~09 | 78,509 | 0.5686 | **0.7120** | **3.5%** | **+4.94bp** CI[−0.30,+10.25] | +3.48bp |

(deep/base 는 시드 3개 평균·[]는 시드폭. 비용선: USDC 메이커 1.02bp · 배포 peg 5.52bp)

## 5. 읽기

⭐**① 부모는 잘 학습된다.** deep 의 bacc 0.5803(최고 시드 0.5861)은 배포본 0.5686 보다
**높다**. 데이터를 되살리고 5.6배를 주면 **라벨은 확실히 더 잘 맞춘다**. 목표의 "학습 잘하는
하나 잘 만들어서"는 달성됐다.

🔴**② 그런데 돈은 반대로 간다.** deep 은 방향을 **더 자주 맞추는데**(0.7018 vs base 0.6887)
**덜 번다**(+0.42 vs +3.80bp). Δ1h(deep−base) = **−3.38bp, 시드폭 2.88 밖**.
맞추는 빈도가 아니라 **맞출 때 얼마짜리를 잡느냐**가 다르다 —
[[eth_pnl_objective_direction_sample_bound_20260912]]("목적함수를 정확도→순손익으로")의 재확인.

🔴**③ 배포본이 여전히 제일 낫다.** 1h +4.94bp 로 내 두 판을 앞선다. 무기는 **선별성**이다 —
통과율 **3.5%** 로 내 7.7~8.3% 의 절반 이하인데 방향정확은 **0.7120** 으로 가장 높다.
내 품질 머리가 같은 q=0.75 에서 2배 더 통과시킨다 = **보정이 더 헐겁다.**

🔴**④ 셋 다 유의하지 않다.** 배포본 CI[−0.30,+10.25]·base 시드 1개는 CI 하한 −0.97.
독립일 122 로는 이 크기의 차이를 가릴 검정력이 없다. **어느 것도 비용선 돌파를 주장할 수 없다.**

## 6. 결론

- **데이터 복구는 성공했고 재사용 가능한 자산이다** — 펀딩 2021~2026 전량, zigzag 라벨
  2021-12~2026-08, 실펀딩 136열 프레임(490,417행), walk-forward 레짐. 이건 이번 판정과
  무관하게 남는다.
- **부모 재학습으로 배포본을 이기지 못했다.** 깊이는 D 에서 중립이었고, 여기서는 경제적으로
  **해롭다**(deep −3.38bp). 배포 부모를 교체할 근거 없음.
- 남은 레버는 **품질 머리 보정**이다(통과율 3.5% vs 8.3% 가 성과 차의 대부분으로 보인다).
  방향 머리가 아니라 「얼마나 안 쏠까」쪽이다. 다만 독립일 122 에서는 판정이 안 되므로,
  하려면 검정력부터 세야 한다.
- ⚠️학습한 건 direction/quality 두 머리다(exit 는 2025 전용 lifecycle 아티팩트 요구).
  경제 수치는 **총이익 방향 읽기**이지 라이브 경로(ATR 배리어·exit 머리·라우터·사이징)가
  아니다. **승격 근거로 쓰지 않는다.**

## 7. 산출물

- [`backfill_eth_funding_history_20260917.py`](../../scripts/backfill_eth_funding_history_20260917.py) → `tmp/omega461_longwindow_20260917/funding_2021_2026.csv`
- [`build_zigzag_action_labels_2021_2023_20260917.py`](../../scripts/build_zigzag_action_labels_2021_2023_20260917.py) → `.../zigzag_labels_full/` (2021~2026)
- [`build_omega461_longwindow_frame_realfunding_20260917.py`](../../scripts/build_omega461_longwindow_frame_realfunding_20260917.py) → `features_136_2022_2026_realfunding.parquet`
- [`refit_regime_balnobb_2022_realfunding_20260917.py`](../../scripts/refit_regime_balnobb_2022_realfunding_20260917.py) → `regime_balnobb_refit2022_realfunding.joblib`
- [`build_omega461_walkforward_regime_realfunding_20260917.py`](../../scripts/build_omega461_walkforward_regime_realfunding_20260917.py) → `features_with_regime_2022_2026_realfunding.parquet`
- [`train_eval_omega461_parent_zig075_longwindow_20260917.py`](../../scripts/train_eval_omega461_parent_zig075_longwindow_20260917.py) → `stageE/stageE_results.json`
- [`eval_omega461_deployed_parent_benchmark_20260917.py`](../../scripts/eval_omega461_deployed_parent_benchmark_20260917.py) → `stageE/stageE_deployed_benchmark.json`
