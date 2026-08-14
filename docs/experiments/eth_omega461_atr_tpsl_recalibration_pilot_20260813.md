# ETH Omega4.6.1 ATR TP/SL 재보정 파일럿 (2026-08-13)

## 배경

`docs/experiments/eth_omega4_6_1_atr_tpsl_floor_binding_investigation_20260812.md`가 발견한
문제: 라이브 "ATR 적응형" TP/SL(`tp_mult=12.0`, `sl_mult=6.0`)이 ETH 5분봉 실제 ATR% 규모에
비해 너무 작아, `min_tp=0.075`/`min_sl=0.040` floor가 전체 시간의 95~98.5%에서 그대로
바인딩된다 — 사실상 거의 항상 고정 7.5%/4.0% 타겟이다. 그 문서는 "버그인지 의도된 설계인지"를
열린 질문으로 남겼다. 이 파일럿은 그 판단 자체를 대신하지 않고, **재보정(ATR 스케일링이
실제로 작동하도록 배율을 키우는 것)이 성과에 어떤 영향을 주는지**를 실증으로 확인해 판단에
근거를 제공한다.

## 방법

`tp_mult`/`sl_mult`는 학습된 모델 가중치가 아니라 런타임 실행 상수
(`eval_omega4_1_atr_safety_sltp_20260622._apply_atr_safety_sltp`)이므로 **재학습이 전혀
필요 없다** — 기존 냉동 h48qual/zig075 번들의 저장된 예측을 그대로 재사용해 TP/SL 계산만
바꾸는 순수 백테스트 리플레이다. 재학습이 없으므로 이 특정 축에는 시드 분산이라는 차원 자체가
없다(다른 08-13 트랙들의 N≥5시드 요건과는 다른 성격) — 결정론적 VAL 스윕 1회가 이 메커니즘이
낼 수 있는 가장 확정적인 근거이며, 대신 OOS 단일터치 규율은 다른 모든 08-13 실험과 동일하게
지켰다.

**재사용한 기존 하네스** (신규 작성 없음, 그대로 import):
- `research_eth_omega461_exit_sweep_20260721.py`의 `load_frame`/`prep_component`/
  `replay_exit_variant` — 컴포넌트 레벨.
- `replay_omega4_6_1_greedy_router_20260706.py`의 `greedy_replay` — PRIORITY/SCALE_MAP까지
  라이브와 동일한 포트폴리오 레벨(단일 공유 슬롯, h48qual>zig075).
- `research_eth_omega461_live_sltp_mfe_width_20260813.py`의 `_as_router_component`/
  `_ledger_stats`/`_duration_gated` 헬퍼.

**설계**: `tp_mult:sl_mult` 비율을 라이브 값(12:6=2:1)으로 고정한 채 전체 배율만 스윕해,
TP:SL 리스크·보상 형태는 바꾸지 않고 "ATR 적응형이 얼마나 강하게 작동하는가" 축만 격리했다.
VAL 구간(2025-10-01~12-31) `atr_pct`(window=192) 직접 계산 결과: p50=0.2696%, p90=0.4256%,
p99=0.6685%, max=0.9468%. 배율 12에서는 floor가 p97~98 부근에서만 열리는데(관측 floor
바인딩률과 일치), 배율을 약 28까지 올리면 floor 교차점이 중앙값 부근까지 내려온다.

**그리드**: `(tp_mult, sl_mult)` ∈ {(12,6)=baseline, (16,8), (22,11), (28,14)}.
`exit_threshold=0.95`(불변) — exit_head 축은 건드리지 않고 TP/SL 축만 격리.

**규율**: VAL 전수 스윕 → baseline 대비 pnl·mdd 둘 다 비악화(no_gate·with_gate 둘 다)인
후보가 있어야만 그 최고 후보 하나만으로 OOS 단일 터치. 라이브 파일
(`trading_bot_modules/omega4_6_1_live.py`/`trading_bot.py`/`runtime_config.py`/`.env`) 미변경.
fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.

**G0 자체검증**: baseline 후보(12,6)가 기존 알려진 라이브 baseline 수치(포트폴리오 no_gate
PnL+36.82%/MDD-24.34%/29건)를 정확히 재현하는지 먼저 확인 — 실측 PnL 36.8217%/MDD -24.3392%/
29건으로 **통과**(하네스 정합성 확인, 이후 결과 신뢰 가능).

## 결과 — VAL, 결정적 부정

| 후보 | h48qual PnL/MDD/거래/floor바인딩률 | zig075 PnL/MDD/거래/floor바인딩률 | 포트폴리오 no_gate PnL/MDD/거래 | 포트폴리오 with_gate PnL/MDD/거래 |
|---|---|---|---:|---:|
| **baseline (12,6)** | +5.45% / -11.62% / 29건 / TP 100.0% | +40.31% / -13.07% / 29건 / TP 99.9% | **+36.82% / -24.34% / 29건** | +54.88% / -31.11% / 22건 |
| (16,8) | +5.45% / -11.62% / 29건 / TP 99.6% | +17.69% / -22.80% / 28건 / TP 99.6% | **-7.25% / -36.80% / 28건** | +7.07% / -36.53% / 21건 |
| (22,11) | +4.46% / -12.29% / 26건 / TP 79.7% | +10.11% / -21.27% / 18건 / TP 91.4% | **+17.14% / -23.64% / 22건** | +23.81% / -26.59% / 17건 |
| (28,14) | -2.05% / -15.53% / 17건 / TP 63.9% | -3.72% / -23.77% / 21건 / TP 91.4%→75.0% | **+1.32% / -27.67% / 17건** | -3.04% / -27.69% / 13건 |

**floor 바인딩률은 의도대로 낮아졌다**(h48qual TP 100%→63.9%, zig075 99.9%→75.0%) — 재보정
메커니즘 자체는 정상 작동한다. 하지만 **VAL 후보 3개 전부, no_gate·with_gate 둘 다에서
baseline보다 나쁘다** — 특히 (16,8)은 포트폴리오 PnL이 +36.82%→**-7.25%**로 부호까지
뒤집힌다. 배율이 커질수록 거래수(29→28→22→17)와 승률이 줄고 평균 보유기간이 길어진다
(670→759/779→1066/1340bar) — TP/SL을 실제로 더 넓히면 포지션이 더 오래 열려있어 새 진입
기회가 줄어드는, 이 세션의 SLTP 폭 실험들이 반대 방향(좁히기)에서 이미 겪은 것과 대칭적인
트레이드오프다.

**VAL 통과 후보 0개 — 사전등록 규율에 따라 OOS는 열지 않았다.**

## 결론

이 특정 재보정 방향(배율을 키워 floor 의존도를 낮추는 것)은 **VAL에서 결정적으로 부정**되며,
OOS 확인 없이도 baseline보다 명백히 나쁘다. `tp_mult:sl_mult` 비율을 유지한 스케일 축만
스윕했다는 점에서 결과가 제한적이지만(예: 비율 자체를 바꾸거나, floor/cap 자체를 재조정하는
등 다른 재보정 방향은 미탐색), 적어도 이 방향에서는 **"ATR 적응형"이라는 이름과 다르게
동작하는 현재 상태(사실상 고정폭)가 이름을 명목대로 고치는 것보다 이 백테스트 구간에서 더
나은 성과를 낸다** — 즉 이름과 실제 동작의 불일치는 실재하지만, 그 불일치를 없애는 방향의
수정이 성과를 개선하지 못한다는 근거가 됐다.

**이게 "버그가 아니라 의도된 설계다"를 증명하진 않는다** — floor 값(7.5%/4.0%) 자체가 원래
독립적인 다른 근거로 설계됐고 ATR 배율은 부차적 역할만 의도됐을 가능성(원 조사 문서의
가능성 2)과 일관되지만, 원 설계자의 의도를 직접 확인한 것은 아니다. 다만 **"이름을 실제로
맞추는 수정"이 성과를 해치지 않는지"는 이제 답이 나왔다 — 해친다.**

## 미해결 / 다음 단계

- `tp_mult:sl_mult` 비율을 유지한 스케일 축만 테스트했다 — 비율 자체를 바꾸는 재보정(예: SL만
  독립적으로 넓히기)은 미탐색. 단, 이 세션의 `eth_omega461_live_sltp_asymmetric_tpsl_20260813.md`가
  이미 비슷한 축(SL을 MFE 예측폭과 분리)에서 R:R 역전 문제를 발견한 바 있어 사전 확률은 낮다.
- floor/cap 값(0.075/0.040/0.22/0.12) 자체를 바꾸는 재보정은 미탐색 — 이 파일럿은 배율만
  바꿨다.
- 이 파일럿은 h48qual/zig075 둘 다 같은 배율로 동시에 바꿨다 — 컴포넌트별 독립 재보정(한쪽만
  바꾸기)은 미탐색.
- 채택 가능한 변경 0건, 라이브 파일 미변경, quality_threshold/exit_head 등 다른 축과 무관.

## 준수 확인

- `git diff` 기준 `trading_bot.py`/`trading_bot_modules/omega4_6_1_live.py`/`runtime_config.py`/
  `.env` 무변경.
- 재학습 없음(냉동 예측 재사용), 저장 원장 미사용, VAL-then-단일OOS 규율 준수(이번엔 VAL에서
  전부 탈락해 OOS 자체를 열지 않음).
- 스크립트: `scripts/research_eth_omega461_atr_tpsl_recalibration_pilot_20260813.py`. 산출물:
  `tmp/causal_regen_20260516/eth_omega461_atr_tpsl_recalibration_pilot_20260813/`
  (`report.json`, `component_val.csv`, `portfolio_val.csv` — OOS 미실행이라 `*_oos.csv` 없음).
