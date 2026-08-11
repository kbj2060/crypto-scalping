# ETH h48qual — quality_head 배리어 Floor/Horizon 스윕 (2026-08-11)

## 배경

h48qual의 배포된 `quality_head` 라벨(`h48_conservative`: `tp_mult=1.2×ATR96`, `sl_mult=0.8×ATR96`,
TP:SL=1.5, `horizon=48bar`)이 `direction_head` 타겟(`zigzag_action`)과 스케일상 얼마나 맞는지 실증
검증된 적이 없었다. floor(TP/SL 최소폭)와 horizon을 순서대로 스윕해서 확인했다.

핵심 지표 정의 (`scripts/sweep_h48qual_barrier_floor_zigzag_match_20260811.py`):

- **coverage**: zigzag가 활성(`zigzag_action!=0`)인 bar 중, h48 `tb_action`도 활성인 비율. 너무 낮으면
  quality가 진짜 스윙을 놓친다는 뜻.
- **agreement**: 둘 다 활성인 bar 중 방향이 일치하는 비율. 너무 낮으면 quality가 방향과 무관한
  노이즈에 반응한다는 뜻.
- **specificity**: zigzag가 비활성(CASH)인 bar 중 h48도 CASH인 비율. 너무 낮으면 quality가 진짜
  스윙이 아닌 구간에서도 계속 거래 신호를 낸다는 뜻.

## 1단계 — floor만 스윕 (horizon=48 고정)

Script: `scripts/sweep_h48qual_barrier_floor_zigzag_match_20260811.py`

배포된 floor(`min_sl=0.4%`, `min_tp=0.6%`)가 의심되는 지점이라고 보고 SL floor를 `[0.004, 0.006,
0.010, 0.015, 0.020]`로 넓혀가며 테스트(`tp_mult`/`sl_mult`/TP:SL 비율은 고정).

**결과(역설)**: floor를 넓힐수록 zigzag 방향일치율이 89.5% → 80.0%로 오히려 하락.

## 2단계 — floor+horizon 조인트 스윕 (역설 원인 검증)

Script: `scripts/sweep_h48qual_barrier_floor_horizon_zigzag_match_20260811.py`

**가설**: `horizon=48bar`로 고정한 채 배리어만 넓히면, 짧은 시간 안에 단기 되돌림(pullback)에 먼저
걸릴 기회가 늘어나서 오히려 방향일치가 떨어진다.

`SL_FLOOR_GRID=[0.004, 0.006, 0.010, 0.015, 0.020] × HORIZON_GRID=[48, 96, 144, 216, 288]`
(4h~24h)로 검증. `tp_mult=1.2`/`sl_mult=0.8`(TP:SL=1.5)는 전 구간 고정.

**결과**: floor=0.4%/0.6%가 테스트한 모든 horizon에서 항상 최선으로 확인됨 — 가설 확인. floor는
그대로 두고, horizon 쪽이 진짜 조정 대상이라는 결론으로 3단계로 넘어감.

## 3단계 — horizon 광역 스윕

Script: `scripts/sweep_h48qual_horizon_wide_20260811.py`

floor를 0.4%/0.6%로 고정(2단계에서 확인된 최선값)하고 `HORIZON_GRID=[48, 96, 144, 216, 288, 384,
480, 576, 720, 864, 1152, 1440]` (4h~120h/5일)로 광역 스윕. "일치율은 높은데 표본이 거의 안
남는" 함정을 피하기 위해 coverage/timeout 비율도 같이 추적.

**결과**:

| Horizon | 방향일치 | Specificity | Coverage |
|---|---:|---:|---:|
| 48bar (배포값) | 89.5% | 34.2% | 60.3% |
| 384bar (32h) | **92.1%** (첫 정점) | **65.1%** (~2배) | 38.2% |
| 48h~120h | 91~92.5% (노이즈 수준 정체) | — | 더 하락 |

384bar를 "효율적인 지점"으로 선택: 방향일치가 처음 정점을 찍고 specificity가 거의 2배가 되는데
coverage는 아직 크게 안 깎인 지점. 이보다 더 늘리면 방향일치 개선 없이 노이즈 수준(91~92.5%)에서
정체만 한다.

## 교차검증

위 384bar 수치는 스윕 스크립트 자체의 barrier 로직 재구현에서 나온 것이라, 별도로 캐노니컬
배리어 빌더(`scripts/build_omega1_2_triple_barrier_labels_20260619.py`) 경로로 동일 horizon을
재현: 방향일치 **92.52%** — 스윕 스크립트 수치(92.1%)와 거의 일치. 스윕 스크립트가 캐노니컬 로직과
다른 걸 측정하고 있었던 게 아님을 확인.

## 반영

- `scripts/build_eth_h384_conservative_triple_barrier_label_20260811.py` — 캐노니컬 빌더에
  `horizon=384`만 monkeypatch(배리어 공식·floor는 그대로)해서 실제 학습용 라벨 생성.
- `scripts/pad_eth_h384_conservative_labels_to_zigzag_timestamps_20260811.py` —
  `direction_head`와 같은 zigzag 타임스탬프 그리드로 패딩(미매칭 → CASH).
- `scripts/chart_eth_h48qual_oracle_oos_1week_20260811.py` — 2026 OOS 1주(2026-01-01~02-28
  구간 내 대표 1주)를 대상으로 `zigzag_action` vs `h384` oracle 트레이드를 같은 가격축에 시각적으로
  대조.

## 결과 (계약 문서 반영용)

`quality_head` horizon을 `48bar` → `384bar`(32h)로 변경. 배리어 공식(TP/SL 배율, floor,
SL-priority)은 불변.
