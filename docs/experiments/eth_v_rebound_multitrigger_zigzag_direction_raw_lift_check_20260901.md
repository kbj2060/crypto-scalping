# 9트리거 통합 "지그재그 방향확인"(ZDC) raw-lift 사전점검 — REJECTED (2026-09-01)

## 배경

기존 "V자반등 9트리거 통합모델"(giveback 라벨, HOLDOUT AUC 0.8465, 배포완료)과 다른 라벨/문제정의로
새 방향예측 모델을 만들기 위해, 트리거 발동 이후 지그재그 알고리즘이 확인하는 "다음 스윙" 방향만 라벨로
쓰는 방식을 사전점검했다. 계획 전문: `.claude/plans/swift-doodling-grove.md`(승인됨).

## 방법

`scripts/research_eth_v_rebound_multitrigger_zigzag_direction_raw_lift_check_20260901.py`. 트리거
population은 재계산 없이 기존 9트리거 라벨 CSV(`data/labels/eth_5m_v_rebound_multitrigger_20260831/`)
재사용. 라벨: 트리거 idx에서 지그재그 상태머신(`build_wave3_action_labels_20260531.py::_zigzag_pivots()`의
trend==0 분기)을 새로 시작해 첫 피벗 타입이 트리거의 함의 방향과 일치하는지(bottom→"L" 먼저=1,
top→"H" 먼저=1). 파라미터는 이 저장소 zigzag_action 파이프라인 기존값 그대로 재사용
(min_reversal_pct=0.01, atr_window=14, atr_multiplier=1.0), MAX_LOOKFORWARD_BARS=288(24h).
비-동어반복 베이스라인: 같은 공식을 VAL/OOS 창의 모든 적격봉에 트리거 무관하게 적용.

**셀프체크 통과**: idx=0에서 재현한 첫 피벗이 `_zigzag_pivots()`의 진짜 첫 글로벌 피벗과 정확히
일치(1차 시도에서 불일치 발견 → 확정 시점(i) 대신 실제 극값 발생 봉(low_idx/high_idx)을 반환하도록
수정 후 재확인 — 셀프체크가 실제 구현 버그를 잡아낸 사례).

## 결과

| window | side | n | hit_rate | baseline | lift |
|---|---|---:|---:|---:|---:|
| VAL | bottom(upside) | 4,226 | 0.4927 | 0.5076 | **0.971** |
| VAL | top(downside) | 4,128 | 0.4689 | 0.4924 | **0.952** |
| OOS | bottom(upside) | 2,936 | 0.4703 | 0.4831 | **0.973** |
| OOS | top(downside) | 3,044 | 0.4846 | 0.5169 | **0.938** |

**4칸 전부 lift<1.0** — 트리거 population이 무작위 봉보다 오히려 살짝 못하다(baseline 자체가 이미
~50% 근처인 것도 확인: ZDC는 "어느 방향이 먼저 임계치를 넘는가"라 정보 없는 봉에서는 동전던지기에
수렴하는 게 정상). 이 저장소의 "채택 1.6x+/보류 ~1.0~1.3x" 참고 기준으로 보면 보류권보다도 아래다.

## 해석(확정 아님)

Giveback 라벨(기존 V자반등)은 **트리거 봉 자신의 wick 극값**(`frame["low"/"high"].iloc[idx]`)을
앵커로 쓰는데, 이번 ZDC는 `_zigzag_pivots()` 자체가 종가만 쓰는 함수라 **종가**(`close[idx]`)를
앵커로 썼다. 스윕류 트리거는 wick이 극단을 찍고 종가가 이미 그 방향으로 일부 되돌린 상태로
마감되는 경우가 흔한데, 앵커를 종가로 옮기면 "이미 되돌림이 벌어진 뒤"부터 시계를 다시 재는 셈이라
진짜 엣지가 이미 일부 소진된 시점에서 출발했을 가능성이 있다. 확정된 원인은 아니며, 검증하려면
wick 앵커 버전을 별도로 재점검해야 한다.

## 결론(종가 앵커)

계획서(`swift-doodling-grove.md`) Step A의 중단 기준("lift가 유의미하지 않으면 정직하게 보고하고
중단")에 따라 **Step B(라벨+피쳐 빌드) 이후로 진행하지 않는다.** 이 라벨 정의(종가 앵커 지그재그
방향확인) 자체는 REJECTED.

## 육안검증 (2026-09-01 후속)

`scripts/render_eth_5m_v_rebound_multitrigger_zigzag_direction_20examples_20260901.py`로 HIT
10건+MISS 10건 차트 확인 — 전부 논리적으로 타당함(HIT는 함의방향으로 임계치 선(先)돌파가
뚜렷, MISS는 반대방향 선돌파 또는 급반전이 뚜렷). **REJECTED 판정은 버그가 아니라 진짜 결과임을
재확인.** 부수 발견: 확정까지 걸리는 시간이 매우 길고 가변적(74~179봉=6~15시간대도 흔함) —
giveback(고정 30/60분)과 달리 보유시간 예측이 어렵다는 별도 실용적 약점도 노출.

차트 제작 중 표시 버그 1건 발견·수정: `zdc_first_pivot()`이 반환하는 "피벗 기록봉"(반대편
극값이 안 움직였으면 idx 자신)과 "실제 임계치 확정봉"(그보다 늦을 수 있음, 예: idx는 그대로인데
6봉 뒤에야 확정)을 혼동해 차트에 확정 시점을 잘못 표시했었음 — hit/miss 판정 로직 자체는
영향 없음(그건 pivot_type만 씀), 순수 표시 버그. `confirm_idx`를 별도로 반환하도록 수정.

## Wick 앵커 변형 (2026-09-01 후속 실험)

`scripts/research_eth_v_rebound_multitrigger_zigzag_direction_wick_anchor_raw_lift_check_
20260901.py` — 종가 앵커판과 유일한 차이는 시작 앵커(bottom→low[idx], top→high[idx], giveback과
동일 관례). subsequent 봉 추적은 여전히 종가 기준(바뀐 변수를 하나로 한정). 수동추적 2건으로
로직 검증(idx=396/818, 앵커·확정봉이 손계산과 일치).

| window | side | hit_rate | baseline | lift(종가앵커) | lift(wick앵커) |
|---|---|---:|---:|---:|---:|
| VAL | bottom | 0.5495 | 0.5341 | 0.971 | **1.029** |
| VAL | top | 0.5248 | 0.5192 | 0.952 | **1.011** |
| OOS | bottom | 0.5375 | 0.5127 | 0.973 | **1.048** |
| OOS | top | 0.5526 | 0.5474 | 0.938 | **1.009** |

**가설이 방향적으로 확인됨** — wick 앵커가 4칸 전부 lift를 1.0 이상으로 끌어올렸다(종가 앵커는
4칸 전부 1.0 미만이었음). 다만 **절대 크기(1.01~1.05x)는 이 저장소의 "채택 1.6x+/보류
1.0~1.3x" 기준으로 보면 보류권 최하단** — VAL bottom의 baseline(0.5341)이 트리거 hit_rate의
95% CI 하한(0.5344)에 거의 닿아 있어 통계적으로도 근소함. TabPFN cheap-gate로 넘어갈 만큼
뚜렷한 신호는 아직 아니다.

## 완전 wick 변형 (2026-09-01 후속, subsequent 봉도 wick 기준)

`scripts/research_eth_v_rebound_multitrigger_zigzag_direction_full_wick_raw_lift_check_
20260901.py` — wick-앵커판에서 한 걸음 더: subsequent 봉의 추적도 종가 대신 그 봉의 고가/저가로
바꿈("가격이 먼저 임계치를 넘는다"를 종가확정이 아니라 봉중 터치 기준으로 재정의). **주의**:
giveback(V자반등) 라벨 자체도 "fast move" 진행측정은 여전히 종가 기준이라, 이 변형은 giveback
보다도 더 적극적으로 wick을 쓴다 — 봉중 터치는 실제 체결가능성(스프레드/슬리피지)을 반영 안 함.
수동추적 2건으로 검증(같은 이벤트가 wick-anchor판보다 항상 같거나 빠르게 해상됨을 직접 확인:
idx=396 400→399봉, idx=818 825→820봉 — 추적범위 확장이 threshold 도달을 앞당기거나 유지하는
방향으로만 작용해야 한다는 사실과 일치).

| window | side | lift(종가앵커) | lift(wick앵커,종가추적) | lift(완전wick) |
|---|---|---:|---:|---:|
| VAL | bottom | 0.971 | 1.029 | 1.015 |
| VAL | top | 0.952 | 1.011 | 1.036 |
| OOS | bottom | 0.973 | 1.048 | 1.048 |
| OOS | top | 0.938 | 1.009 | 1.028 |

**완전 wick도 4칸 전부 lift>1.0을 유지하지만, wick-앵커판 대비 뚜렷한 추가 개선은 없다**(평균
1.024→1.032, VAL bottom은 오히려 소폭 하락). subsequent 봉 추적 방식(종가 vs wick)은 이
축에서 2차적 요인 — 결정적인 건 "앵커를 종가에서 wick으로 바꾸는 것" 자체였고, 그 이상의
세부 튜닝은 성과를 크게 못 바꾼다.

## 최종 결론

- 종가 앵커: REJECTED(4칸 전부 lift<1.0).
- wick 앵커(종가추적) / 완전 wick: 둘 다 가설대로 개선되어 4칸 전부 lift>1.0이지만, 절대
  크기가 약함(1.01~1.05x, 이 저장소 "채택 1.6x+/보류 1.0~1.3x" 기준의 보류권 최하단) — 두
  변형 사이 실질적 차이도 없음.
- 3가지 변형(종가/wick-앵커/완전wick) 전부 Step B(TabPFN 파이프라인) 진행 기준을 충족하지
  못함 — 앵커/추적 방식을 더 세밀하게 튜닝해도 이 한계(~1.0~1.05x 천장)를 넘을 근거는 약하다.
  이 세션에서는 이 축을 더 진행하지 않는다. 미착수 대안: giveback 장기호라이즌 버전(4h/8h급,
  라벨 메커니즘 자체가 다름 — fast/full 이중창+giveback비율, 지그재그 단일임계치와 무관).
