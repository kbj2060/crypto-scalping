# ETH zig075 — quality_head 추세 편향 컨트롤: 실제 라이브 가중치 always-short 대조 (2026-08-11)

## 배경

2026-08-11 세션에서 h48qual(`quality_threshold=0.50`)의 `quality_head` 게이트가 direction_head
원본(게이트 전, 균형 53~59% 숏) 대비 게이트 통과 후(`final_action`) 87~91%까지 숏으로 쏠리는
구조적 추세 편향을 확인했다 — always-short 기준선이 실제 라이브 가중치에서도 VAL/OOS 둘 다
모델을 이겼다. 상세: `docs/experiments/eth_h48qual_quality_trend_bias_h48orig_control_20260811.md`.

이 문서는 같은 진단을 `trading_bot_modules/omega4_6_1_live.py`의 `PRIORITY = ("h48qual", "zig075")`
에서 h48qual 다음으로 체크되는 두 번째 라이브 ETH 컴포넌트 **zig075**(`quality_threshold=0.75`)에
그대로 적용한 결과다. 목적: 이 편향이 h48qual 하나의 문제인지, 아니면 라이브 ETH 전략 전체
(`Omega461LiveAdapter`가 쓰는 quality-gate 구조 자체)의 문제인지 판별.

## zig075가 h48qual과 동일한 구조인지 확인 (재사용 전 검증, 가정 없음)

`trading_bot_modules/omega4_6_1_live.py`의 `_Component` 클래스는 h48qual/zig075 양쪽에 공통으로
쓰인다 — 로딩 코드, `entry_decision()`, `quality_for_action` 게이팅 공식(`qual_for_action =
float(quality[action])`; `final_action = action if (action != 0 and qual_for_action >=
quality_threshold) else 0`, `omega4_6_1_live.py:174-178`) 전부 동일 코드 경로다. 이걸 가정에
그치지 않고 두 번들 파일(`true_3head_tabm_bundle.pt`) 자체를 직접 로드해 확인:

| | h48qual | zig075 |
|---|---|---|
| `state_dict` key 집합(22개) | - | h48qual과 완전 일치 |
| `direction_head`/`quality_head`/`exit_head` shape | `(3,192)`/`(3,192)`/`(2,192)` | 동일 |
| `config`(k/hidden/layers/dropout/loss weight) | `k=8, hidden=192, layers=3, dropout=0.08, quality_loss_weight=0.8, exit_loss_weight=1.15` | 동일 |
| `base_cols`(102개) | - | h48qual과 완전 일치 |
| `n_features`(115 = base 102 + pos 13) | - | 동일 |
| experts | `bull`/`bear`/`chop` | 동일 |

아키텍처·게이팅 공식은 완전히 동일하다. **한 가지 실질적 차이**: `report.json`의
`label_contract.quality_mode`가 h48qual은 `quality_label_action`(독립적 48bar ATR 배리어
`h48_conservative`)인 반면 zig075는 `same_as_direction`(`quality_label_dir=None`) — zig075의
`quality_head`는 `direction_head`와 **동일한 `zigzag_action` 라벨**로 학습된다. 즉 zig075의
게이트는 "독립적인 미래 결과 예측이 direction의 선택에 동의하는가"가 아니라 "같은 라벨을 보고
학습한 두 번째 분류기가 direction의 선택에 (더 높은 확신으로) 동의하는가"를 묻는 구조다. 이
차이는 아래 결과 해석에 영향을 준다.

## 방법

`scripts/verify_eth_h48qual_always_short_baseline_live_bundle_20260811.py`를 최소 수정으로
재사용 — 번들 디렉터리와 예측 파일명(`_q050.csv` → `_q075.csv`)만 zig075 것으로 교체, 나머지
로직(TRAIN_CSV/EVAL_CSV 경로, `omega._metrics()` 시뮬 로직, `BASE_TEMPLATE["max_hold"]=0`/
`["cooldown"]=0`, `dir_action`/`final_action` 컬럼 접근, always-short/always-long 강제 구성)은
전부 동일하다. 새 스크립트: `scripts/diagnose_eth_zig075_quality_trend_bias_20260811.py`.

- 번들: `true_3head_tabm_bundle.pt`(2026-06-29 학습, 102피쳐) — **재학습 없음**, 이미 저장된
  예측(`validation_predictions_q075.csv`/`oos_predictions_q075.csv`, 2026-06-30 export) 그대로
  사용.
- `prediction_export_q075_20260630.json`으로 TRAIN_CSV/EVAL_CSV가 h48qual과 완전히 동일한
  경로(VAL: alpha5 regime4 계열, OOS: alpha6_current 계열, `train_eval_override_used=true`)임을
  확인 — 별도 프레임 정합 작업 불필요.
- 전체 방법론(왜 `dir_action`/`final_action`을 나누는지, always-short 대조가 뭘 뜻하는지,
  `omega._metrics()` 시뮬 루프 검증 방식)은
  `docs/experiments/eth_h48qual_quality_trend_bias_h48orig_control_20260811.md`의 "추세 편향
  직접 측정" 절 참고 — 여기서는 반복하지 않는다.
- 실행 환경: 순수 추론(저장된 예측 CSV + `omega._metrics()` 시뮬 루프)이라 GPU 불필요, dev
  로컬 CPU에서 6초 내 완료(387MB VAL 프레임 로드 포함) — server handoff 불필요.

**실제 데이터 커버리지 주의**: h48qual/zig075 스크립트 둘 다 export된 예측 CSV의 실제 타임스탬프
범위는 VAL 2025-10-01~2025-12-31, OOS 2026-01-01~2026-02-28로, 레포 기본 fresh-forward
윈도우(VAL 2025-09-01~2025-12-31, OOS ~2026-03-31)보다 좁다 — 직접 확인한 값이며 h48qual 원본
스크립트에도 동일하게 존재하던 특성이다. 아래 표 헤더는 h48qual 문서와의 직접 비교를 위해 원
문서의 표기를 그대로 따르되, 실제 커버리지는 이 문단 기준이다. 두 컴포넌트 모두 동일하게
좁혀져 있어 h48qual-zig075 비교 자체는 apples-to-apples다.

## 결과

| | VAL (2025-09-01→2025-12-31) | OOS (2026-01-01→2026-03-31) |
|---|---:|---:|
| Model PnL | +11.03 | +14.71 |
| always_short PnL | +15.57 | +15.63 |
| Short share pre-gate (dir_action) | 57.2% | 53.9% |
| Short share post-gate (final_action) | 70.0% | 65.8% |
| Gate pass rate | 8.53% | 8.78% |
| Trade count | 35 | 13 |

(참고: always_long PnL은 VAL -13.27, OOS -13.85 — 이 구간 자체가 숏에 강하게 유리한 환경이라는
점은 h48qual과 동일하다.)

## h48qual과 비교

| | h48qual VAL | zig075 VAL | h48qual OOS | zig075 OOS |
|---|---:|---:|---:|---:|
| Model PnL | +4.51 | +11.03 | +12.01 | +14.71 |
| always_short PnL | +18.34 | +15.57 | +16.62 | +15.63 |
| always_short − Model | +13.83 | +4.55 | +4.61 | +0.92 |
| 숏비중 게이트 전 | 58.4% | 57.2% | 56.3% | 53.9% |
| 숏비중 게이트 후 | 87.0% | 70.0% | 91.5% | 65.8% |
| 게이트 전→후 상승폭(pp) | +28.6 | +12.8 | +35.2 | +11.9 |
| 게이트 통과율 | 2.45% | 8.53% | 0.68% | 8.78% |
| 거래수 | 29 | 35 | 9 | 13 |

## 해석

**같은 방향의 편향이 재현된다 — 다만 강도는 h48qual의 절반 이하다.**

1. **게이트가 숏 쪽으로 쏠리게 만든다는 패턴은 그대로 재현된다**: direction_head 원본(게이트
   전)은 양쪽 다 53~58%로 균형에 가까운데, quality 게이트를 통과하면 zig075도 65.8~70.0%까지
   숏비중이 올라간다. 다만 상승폭은 zig075가 11.9~12.8pp로, h48qual의 28.6~35.2pp보다 뚜렷하게
   작다.
2. **always_short이 모델을 이긴다는 패턴도 재현된다**: VAL/OOS 둘 다 always_short PnL이 모델
   PnL을 앞선다. 다만 격차가 훨씬 좁다 — VAL은 4.55(h48qual은 13.83), OOS는 0.92(h48qual은
   4.61)로 OOS는 사실상 거의 동률에 가깝다.
3. **게이트 통과율은 오히려 zig075가 더 높다**(8.5~8.8% vs h48qual의 0.68~2.45%) — threshold가
   더 높은데(0.75 vs 0.50) 더 많이 통과하는 역설처럼 보이지만, 위 "구조 확인" 절의
   `quality_mode=same_as_direction` 차이로 설명 가능하다: zig075의 `quality_head`는
   `direction_head`와 동일한 라벨로 학습되므로 두 헤드가 구조적으로 더 자주, 더 높은 확신으로
   일치한다 — 실제로 `report.json`의 `label_quality_summary.*.quality_active_ratio`가 zig075는
   88~89%, h48qual은 57~63%로 확인된다. 즉 zig075의 0.75 threshold는 h48qual의 0.50 threshold
   보다 실질적으로 "더 관대한" 필터일 수 있다.

**종합**: zig075는 h48qual과 **같은 방향의 구조적 편향을 보이지만**(게이트가 방향판별 능력이
아니라 숏 쪽 쏠림을 강화하고, always_short을 못 이긴다), **그 정도는 h48qual보다 뚜렷하게
약하다.** "편향이 h48qual에만 있는 것은 아니다"가 결론이지만, "zig075도 h48qual만큼 심각하다"는
과장이다 — 특히 OOS에서는 모델과 always_short의 차이가 거의 사라질 정도로 작다.

## 유의사항

- **표본 크기**: h48qual과 마찬가지로 거래수가 13~35건뿐인 단일 번들 스냅샷이다 —
  h48qual 문서의 15시드 통계검증(`eth_h48qual_final12_h384_isolated_tuning_sweep_20260811.md`)
  같은 신뢰도는 없다. "방향이 일치한다"는 확인 수준으로만 취급해야 한다(h48qual의 "실제 라이브
  가중치로 확인" 절과 동일한 caveat).
- **quality label 의미 차이**: 위에서 언급했듯 zig075의 `quality_head`는 `same_as_direction`으로
  학습되어 h48qual의 독립적 `h48_conservative` 배리어와 라벨 의미가 다르다. always-short
  대조 자체(포지션 방향만 바꾸고 진입 시점/리스크템플릿은 고정)는 두 컴포넌트 모두 동일하게
  적용 가능한 컨트롤이라 비교는 유효하지만, "왜" 편향이 나타나는지의 메커니즘은 두 컴포넌트가
  다를 수 있다 — h48qual은 서로 다른 라벨을 가진 두 분류기의 "동의"가 우연히 하락장과
  맞아떨어진 것이고, zig075는 동일 라벨로 학습된 두 분류기가 서로의 확신을 강화하는 구조에
  가깝다. 이 메커니즘 차이를 확정하려면 h48qual에서 했던 것과 같은 direction_head 단독
  방향판별력 분석(`dir_action` 기준 순위상관, 시드 다양성 재현 등)이 zig075에도 필요하나, 이번
  진단 범위 밖이다.
- **날짜 커버리지**: "방법" 절 참고 — 실제 검증 구간은 VAL 2025-10-01~2025-12-31, OOS
  2026-01-01~2026-02-28로 표 헤더의 기본 fresh-forward 윈도우보다 좁다(h48qual도 동일).
- 이 결과는 `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`
  (h48qual 전용 계약)를 변경하지 않는다 — zig075는 그 계약의 범위 밖이며, 이 문서는 독립적인
  진단 기록이다.

## 결론 (요약)

zig075(`quality_threshold=0.75`)도 h48qual과 **같은 방향의** `quality_head` 게이트 숏 편향을
보인다 — 게이트 전 균형(53~58% 숏) → 게이트 후 숏 쏠림(66~70%), always_short이 VAL/OOS 둘 다
모델을 이긴다. 그러나 편향의 **강도는 h48qual보다 뚜렷하게 약하다**(게이트 전후 상승폭 절반
이하, always_short 대비 격차가 OOS 기준 h48qual보다 훨씬 작음). 편향은 h48qual 하나만의 문제가
아니라 `Omega461LiveAdapter`의 quality-gate 구조 전반에 어느 정도 내재된 것으로 보이나, zig075는
`same_as_direction` quality 라벨과 더 높은 threshold(0.75)의 조합 덕에 h48qual만큼 심하게
망가지지는 않은 것으로 해석된다.
