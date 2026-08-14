# ETH h48qual — "최종 보스" 통합 설계 (2026-08-12)

## 배경

사용자 지시: always_short 대조는 무시하고, 이 세션에서 절대성능 기준으로 원값 신호가 있었던
요소들만 모아 하나로 합친다("최종 보스"). 선정된 3개 요소:

1. **one-vs-rest 독립 direction**(LONG/SHORT/CASH 각각 독립 LightGBM) — 공유 3-class
   softmax의 confidence 간섭 회피.
2. **방향별 독립 MFE 분위수 회귀 quality**(LightGBM) — 이 세션 유일하게 MI/R² 게이트를
   결정적으로 통과한 타겟.
3. **FINAL12 + 오토인코더 latent(16차원) 병합 피쳐** — latent 단독보다 FINAL12 보강용으로
   쓸 때 원값 개선 확인된 조합.
4. (구조적 추가) **비대칭 게이팅** — 이 세션 전체가 일관되게 확인한 "롱은 어디서나 나쁘다"는
   발견을 반영해 LONG 분위수 컷오프(0.85)를 SHORT(0.60)보다 훨씬 엄격하게.

TCN은 검토 후 제외(재현 안 되는 노이즈, always_short 문제와 별개로 원값 자체가 불안정).
trend-scanning 라벨도 제외(R²가 VAL/OOS 전부 음수로 애초에 원값 신호가 없었음).

## 방법

- 학습 스크립트: `scripts/train_eval_eth_h48qual_final_boss_20260812.py`
- **1단계**(FINAL12 몽키패치 걸리기 전): `omega._load_omega_frames()`(패치 안 된 원본)로
  넓은 원시피쳐풀(172컬럼) 로딩, `verify_eth_h48qual_autoencoder_latent_mi_r2_gate_20260812.py`와
  동일 아키텍처(64→32→16 디노이징 오토인코더, TRAIN-fit 표준화, TRAIN 꼬리 15% 조기종료)로
  시드별 latent 재학습(저장된 latent 없어서 재계산 필요 — 아키텍처/방법론은 동일, 데이터소스는
  h48orig 표준 alpha6_current 패널로 통일, 원 실험의 zig075 패널과 다름).
- **2단계**: h48orig 파이프라인으로 FINAL12 프레임 로딩(`train_eval_omega4_3head_parent72_eth_
  h48qual_final12_h48orig_20260811.py`의 몽키패치 체인 재사용), 사전계산 h48_conservative
  MFE(`build_omega1_2_triple_barrier_labels_20260619.py` 산출물, 재시뮬레이션 없음)와 latent
  병합.
- **Direction**: LightGBM 이진분류기 3개(cash/long/short), argmax 결합.
- **Quality**: LightGBM 회귀 2개(long_mfe/short_mfe), 각자 해당 방향 active bar만 학습.
  TRAIN에서 그 방향으로 실제 선택된 bar들의 예측분포 기준 quantile로 컷오프 산출(LONG q0.85,
  SHORT q0.60) — quantile-relative라 MFE 절대 스케일 상관없이 그대로 적용.
- 평가: `scripts/verify_eth_h48qual_final_boss_pnl_20260812.py`, `omega._metrics` 검증된
  거래시뮬로 cost1/2/3 계산. N=5 진짜 무작위 시드(143618629/474012476/917902448/719688500/
  642783630), 로컬 CPU 실행(LightGBM+소형 오토인코더라 GPU 불필요).

## 결과

| | VAL | OOS |
|---|---:|---:|
| model(cost1) | +6.54±9.17 | +9.59±8.88 |
| model(cost2) | +5.46±9.95 | +8.68±9.06 |
| model(cost3) | +5.38±7.57 | +10.62±9.01 |
| always_short(참고) | +14.4~15.8 | +19.6~20.2 |
| always_long(참고) | -19.2~-19.8 | -18.8~-19.2 |
| 승(vs short, 참고) | 0~1/5 | 0~1/5 |
| 승(vs long, 참고) | 5/5 | 5/5 |

**절대 pnl은 VAL·OOS 둘 다 플러스**(승격 기준, 사용자 지시대로 always_short 대조는 참고용).
시드별 분해(cost3): VAL 4/5 시드 개별 플러스(1개만 음수), OOS도 4/5 시드 개별 플러스(다른
1개만 음수) — 평균만 좋은 게 아니라 대체로 일관됨.

**다만 시너지는 확인 안 됨**: MFE 분위수 회귀 단독(`eth_h48qual_mfe_quantile_quality_
regression_20260812.md`)의 VAL +5.6~6.2%/OOS +9.4~10.2%와 이번 결과가 사실상 동일한
크기다. one-vs-rest 방향헤드와 latent 피쳐를 추가로 얹었는데도 MFE 회귀 단독보다 뚜렷하게
나아지지 않았다 — "요소들을 합치면 각각보다 낫다"는 가설은 이번엔 지지되지 않았다.

## 해석

- always_short 대조를 빼고 봐도, 이 통합 설계 자체가 "MFE 분위수 회귀"라는 핵심 요소 하나가
  성능을 대부분 견인하고 있을 가능성이 높다. one-vs-rest/latent 추가는 (a) 최소한 손해는
  안 봤고(성능이 MFE 단독과 비슷한 수준 유지), (b) 시드간 일관성 면에서는 약간의 이점이
  있어 보이나(4/5 개별 플러스), 결정적 개선 증거는 아니다.
- 여전히 always_short 대조에선 패배 — 이 통합 설계도 결국 롱 배제 비중이 커서(SHORT 비중이
  LONG의 2배 이상) 사실상 "숏 편향 시스템"에 가깝고, 이 세션 전체가 겪은 매끈한 하락장에서
  always_short의 구조적 우위를 넘어서지는 못한다.

## 업데이트 2026-08-13 — 동적 리스크사이징 추가(라이브 추격 시도)

사용자 지시: "우리는 지금 라이브 모델을 뛰어넘어야해" — 라이브 전체 라우터(OOS +145.34%)와의
격차가 사이징 구조(고정 notional=0.45×leverage=2.0=실효노출 0.9 vs 라이브 leverage≤5.0/
notional≤1.8) 때문이라는 가설로 동적 사이징 추가.

**방법**: `train_eval_eth_h48qual_final_boss_20260812.py`에 percentile-rank 기반 매핑 추가
— TRAIN에서 그 방향으로 게이트 통과한 bar들의 예측 MFE 순위를 [0,1]로 정규화해
margin_fraction(0.30~0.90)/leverage(1.5~5.0)에 선형 매핑, notional=margin×leverage를
라이브와 동일 캡(≤1.8)으로 제한.

**버그 발견 및 수정**: 첫 실행에서 거래수가 59건→245건으로 4배 폭증(MDD도 -13%→-42%로
급증) — `omega._metrics`는 `take_profit`/`stop_loss`를 notional-스케일 계정pnl 기준
(`unreal=raw_price_move×notional`과 직접 비교)으로 취급하는데, TP/SL 값을 고정(BASE_
TEMPLATE 그대로)으로 두고 notional만 행마다 바꾸니 notional이 클수록 훨씬 작은 가격변동에도
barrier가 발동해버렸다 — 레포의 Futures Risk Sizing Contract(CLAUDE.md)가 경고하는 정확히
그 "레버리지 이중계산" 패턴. `take_profit`/`stop_loss`를 매 행 notional에 다시 곱해
(`TP_PRICE_MOVE=0.026/0.45≈5.78%`, `SL_PRICE_MOVE=0.014/0.45≈3.11%`를 raw 가격변동 목표로
고정) 수정. 수정 후 fixed/dynamic 두 사이징의 거래 진입·청산 시점이 완전히 동일함을 확인
(사이즈만 바뀌고 barrier 로직은 안 바뀜 — 검증 완료).

**결과 (고정 vs 동적 사이징, 같은 5시드)**:

| | VAL(고정) | VAL(동적) | OOS(고정) | OOS(동적) |
|---|---:|---:|---:|---:|
| model pnl(cost3) | +5.38±7.57 | **+11.79±17.43** | +10.62±9.01 | **+39.97±36.63** |
| MDD | -13.45 | -36.57 | -11.62 | -38.21 |
| always_short(참고) | +14.43 | +34.48 | +19.60 | +71.26 |

**절대 pnl은 상당히 커졌다**(VAL 약 2배, OOS 약 4배) — 라이브 라우터(OOS +145.34%/VAL
+54.88%)와의 격차가 기존 대비 확실히 좁혀졌지만(OOS 기준 격차 15배→3.6배), 여전히 한참
못 미친다. **공짜가 아니다**: MDD가 3배 가까이 커졌다(-11~13%→-36~42%) — 사이징을 키우면
변동성/드로다운도 같이 커진다는, 당연하지만 반드시 명시해야 하는 트레이드오프.

**always_short 대조(참고)도 함께 증폭됐다**(OOS +19.6%→+71.3%) — 이 구간에서 숏 편향
전략 자체가 사이징에 비례해 유리해지기 때문에, 상대 격차(model vs always_short)는 안
좁혀졌다(이 세션의 레짐-베타 결론과 일관).

## 업데이트 2026-08-13 — v2: 레짐전문가 라우팅 추가, VAL 개선·OOS 악화(과적합 패턴)

사용자 지시로 라이브 라우터 구조(bull/bear/chop 레짐전문가)를 v1 위에 얹은 v2 시도.
스크립트: `scripts/train_eval_eth_h48qual_final_boss_v2_regime_routed_20260813.py` — TRAIN을
`hard._route_id`(라이브와 동일 라우팅 컬럼)로 3분할해 레짐마다 완전히 독립적인 direction(3)+
quality(2) 모델 학습(총 15개, v1의 5개 대비 3배). 게이팅 컷오프·사이징 순위도 레짐별 독립
계산. 같은 5시드로 직접 비교.

| | v1(flat) | v2(레짐라우팅) |
|---|---:|---:|
| VAL pnl(고정사이징, cost3) | +5.38±7.57 | **+22.54±11.34**(always_short 4/5 승) |
| OOS pnl(고정사이징, cost3) | +10.62±9.01 | **-0.58±4.94** |
| OOS 시드별(동적사이징, cost3) | 대체로 양수·일관 | **-24.3% / +9.6% / +44.1% / +8.9% / -16.9%** — 2/5 강한 음수 |

**VAL은 확실히 개선(라이브 근접, always_short도 종종 이김)되지만 OOS는 오히려 악화되고
시드별 부호까지 뒤집히는 불안정성이 생겼다.** 레짐별로 TRAIN을 3분할하면서 레짐당 표본이
19,347~38,043행으로 줄어든 것(v1은 78,470행 전체 사용)이 원인으로 보이는 전형적 과적합
패턴 — VAL에 더 맞춰지고 OOS 일반화력은 떨어짐. **v2를 v1보다 나은 결과로 볼 근거 없음**
— 오히려 더 신뢰도가 낮아졌다. 이 세션 전체가 반복 학습한 "그럴듯해 보이는 개선은 반드시
다른 구간/시드로 재검증"이라는 규율이 여기서도 재확인됐다.

## 업데이트 2026-08-13 — v3: 듀얼 컴포넌트 우선순위 결합, v2보다는 낫지만 여전히 불안정

사용자 "1번"(레짐 라우팅 대신 v1 flat 구조 유지 + zig075 역할의 2번째 컴포넌트를 라이브와
동일한 우선순위 결합(`PRIORITY=(h48qual, zig075)`)으로 추가) 지시로 시도.
스크립트: `scripts/train_eval_eth_h48qual_final_boss_v3_dual_component_20260813.py` — 컴포넌트
A(v1과 동일, LONG q0.85/SHORT q0.60)와 컴포넌트 B(더 엄격한 LONG q0.90/SHORT q0.75, 완전히
독립적인 시드)를 각각 전체 78,470행으로 학습(v2처럼 표본을 쪼개지 않음), A가 CASH일 때만 B를
확인하는 라이브와 동일한 우선순위 로직으로 결합. 평가: `scripts/verify_eth_h48qual_final_boss_v3_pnl_20260813.py`.
N=5 진짜 무작위 시드쌍(390489516:767391107/672308585:766277990/628550762:431596439/
178486827:451979108/591986449:769898173).

| | v1(flat, 동적사이징) | v3(듀얼컴포넌트, 동적사이징) |
|---|---:|---:|
| VAL pnl(cost3) | +11.79±17.43 | **+27.44±32.98** |
| VAL MDD | -36.57 | -35.82 |
| OOS pnl(cost3) | **+39.97±36.63** | +15.13±33.07 |
| OOS MDD | -38.21 | -42.64 |
| OOS 시드(쌍)별 | 대체로 양수 | +25.1/**-17.2**/+63.4/+19.1/**-14.6** — 2/5 음수 |

v2만큼 극단적이진 않지만(표본을 안 쪼갰으므로 데이터-희소 과적합은 없음), **동일한 패턴의
불안정성**이 재현됐다 — VAL 개선, OOS 악화, 시드 2/5 음수 전환. B 컴포넌트가 A의 CASH bar
중 매 시드쌍 600~1060건을 "구제"하는데, 이 구제된 거래들이 시드쌍마다 부호가 갈릴 만큼
분산이 크다는 뜻. **v1(flat, 단일 컴포넌트)이 이 세 버전 중 가장 안정적** — v2/v3 둘 다
"라이브 구조를 흉내내면 안정성이 오히려 나빠진다"는 결과를 냈다.

이 결과와 함께 사용자가 방향을 전환: v1/v2/v3처럼 라이브를 처음부터 다시 만드는 "최종 보스"
접근 대신, **실제 라이브 파라미터 번들(h48qual/zig075 TabM parent + risk sidecar)을 그대로
가져와 이 세션에서 찾은 인사이트를 얹는 방식**으로 전환. 후속 작업은
`docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`와
`eth_omega4_6_1_live_model_injection_20260813.md`(신규) 참고. 이 "최종 보스" 트랙(v1/v2/v3)은
여기서 종료 — 추가 변형 시도 안 함.

## 산출물

- `tmp/eth_h48qual_final_boss_20260812/main_5seed/` — v1, `val_decisions_s<seed>.csv`,
  `oos_decisions_s<seed>.csv`, `meta_s<seed>.json`, `pnl_comparison_fixed.csv`,
  `pnl_comparison_dynamic.csv`.
- `tmp/eth_h48qual_final_boss_v2_regime_routed_20260813/main_5seed/` — v2, 동일 구조.
- `tmp/eth_h48qual_final_boss_v3_dual_component_20260813/main_5pairs/` — v3, 동일 구조
  (`s<seed_a>_<seed_b>.csv` 파일명).
