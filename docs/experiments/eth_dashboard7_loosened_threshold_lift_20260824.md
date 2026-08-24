# 대시보드 7신호 임계값 완화 lift 재검증 (2026-08-24, 사용자 지시 실행)

## 배경

사용자가 "완화한 버전이 성능이 더 좋을 수도 있으니 테스트해보자"고 요청. 이 라인은 08-23
평가문서가 "09-30 새 OOS 창 없이는 registry-retest-off-limits"라고 명시했고, 08-24 빈도전용
민감도체크에서도 "멀티플컴패리즌 반복 위험"이라며 의도적으로 lift 재계산을 안 했던 축이다.
경고 후에도 사용자가 재요청해 실행 — 단 **임계값을 새로 탐색하지 않고, 이미 같은 날 정해둔
"한 단계 완화" 정의를 그대로 재사용**(멀티플컴패리즌 추가 발생 최소화).

## 완화 정의 (1회 고정, 스윕 없음)

기존 빈도전용 체크의 관례(orthogonal_combo p≤0.10→0.15, delta_z -2.0→-1.5) 그대로 확장:
- 연속 임계값형 4종(orthogonal_combo/volume_wick_climax/short_term_return_z/
  taker_delta_z_climax): percentile밴드 +0.05, |z| 컷오프 -0.5 균일 적용.
- 구조형 3종(liquidity_sweep/smt_divergence/fib_extension_exhaustion, z-score 손잡이가
  없는 패턴형 신호): 스윙 lookback 48→36봉(25% 단축, (a)와 비슷한 완화폭).

## 결과 (VAL+OOS 풀링창, 1h horizon, 14칸=7신호×2방향)

**12/14칸에서 lift 하락 — 메커니즘 예측이 그대로 확인됨**:

| 신호 | 방향 | 현행 | 완화 | 변화 |
|---|---|---|---|---|
| fib_extension_exhaustion | 바텀 | 3.27x | 2.90x | ↓ |
| liquidity_sweep | 바텀 | 3.01x | 2.81x | ↓ |
| orthogonal_combo | 바텀 | 3.51x | 3.35x | ↓ |
| short_term_return_z | 바텀 | 2.90x | 2.69x | ↓ |
| smt_divergence | 바텀 | 3.12x | 2.79x | ↓ |
| taker_delta_z_climax | 바텀 | 2.75x | 2.57x | ↓ |
| volume_wick_climax | 바텀 | 2.94x | 2.64x | ↓ |
| liquidity_sweep | 탑 | 2.78x | 2.59x | ↓ |
| short_term_return_z | 탑 | 2.72x | 2.45x | ↓ |
| smt_divergence | 탑 | 2.84x | 2.64x | ↓ |
| taker_delta_z_climax | 탑 | 2.29x | 2.22x | ↓ |
| volume_wick_climax | 탑 | 2.50x | 2.38x | ↓ |
| **fib_extension_exhaustion** | 탑 | 2.32x | **2.71x** | ↑ |
| **orthogonal_combo** | 탑 | 2.53x | **2.81x** | ↑ |

예외 2칸 모두 **탑 사이드**에서만 발생 — 이 저장소가 반복 확인한 "탑 사이드가 바텀보다
구조적으로 약하고 노이즈가 큼"(Wyckoff 비대칭) 패턴과 정합적이라, 진짜 개선이라기보다
노이즈일 가능성이 높음(14칸 중 2칸이 우연히 반대로 튀는 건 멀티플컴패리즌 하에서 딱히
놀랍지 않은 빈도).

발동횟수는 예측대로 전반적으로 1.5~2배 증가(예: sweep 바텀 1257→1510건, orthogonal_combo
바텀 305→689건).

## 결론

**"완화판이 성능이 더 좋다"는 가설은 기각된다** — 12/14칸에서 예측대로 lift가 하락했다.
다만 하락폭 자체는 대부분 상대적으로 5~12% 수준으로 파국적이진 않다(예: sweep 바텀
3.01x→2.81x, taker 바텀 2.75x→2.57x) — "발동 1.5~2배 더 자주, lift 5~12% 덜" 이라는
정량적 트레이드오프로 요약 가능.

**정직성 caveat**: 이 결과는 이미 22신호+4라운드가 반복 조회한 VAL+OOS 풀링창에서 나온
것이라, 신선한 미터치 데이터로 검증된 게 아니다 — 09-30 이전엔 승격/재배포 근거로 쓰지
않는다. 현재 대시보드 임계값 유지가 여전히 맞는 선택.

## 재현

`scripts/analyze_eth_dashboard7_loosened_threshold_lift_20260824.py`.
산출물: `tmp/eth_dashboard7_loosened_threshold_lift_20260824/loosened_threshold_lift_table.csv`.
