# ETH Omega4.6.1 ATR TP/SL floor 독립·컴포넌트별 재보정 (2026-08-15)

## 배경

`eth_omega4_6_1_atr_tpsl_floor_binding_investigation_20260812.md`가 발견한 문제: 라이브
"ATR 적응형" TP/SL(`min_tp=0.075`/`min_sl=0.040`, `tp_mult=12`/`sl_mult=6`)이 ETH 5분봉에서
95~98.5%의 시간 동안 floor에 고정되어 사실상 고정폭 7.5%/4.0% 시스템으로 동작한다.
`eth_omega461_atr_tpsl_recalibration_pilot_20260813.md`가 `tp_mult:sl_mult` 비율(2:1)을 유지한
채 배율만 키워 floor 의존도를 낮추는 방향을 VAL에서 스윕했으나 3후보 전부 baseline보다
나빴다(특히 (16,8)은 포트폴리오 PnL +36.82%→-7.25%로 부호 반전). 그 파일럿이 미탐색으로 남긴
두 축 — (a) floor 절대값 자체를 비율 유지 없이 독립적으로 바꾸기, (b) h48qual/zig075를
컴포넌트별로 따로 재보정하기 — 를 이 실험이 최소 형태로 함께 다룬다.

## 방법

**방향 선택 근거**: 08-13 파일럿은 배율을 올려 floor가 항상 고정하는 값을 "키우는" 방향만
테스트했고, 배율이 커질수록 성과가 단조적으로 나빠졌다(거래수 29→28→22→17, PnL/MDD 둘 다 악화).
floor가 거의 항상 바인딩되므로 floor 값 자체가 곧 실질 거래폭이다. 반대 방향(floor를 낮춰
실질폭을 좁히기)은 아직 정보가 없는 유일한 축이므로 이 실험은 **좁히기(narrowing)만** 테스트한다
— 넓히기는 08-13이 이미 결정적으로 부정했으므로 반복하지 않는다.

**후보 grid 근거**: VAL 구간(2025-10-01~12-31) `atr_pct`(window=192)를 직접 재계산해
percentile을 구했다(n=26,209, 08-13 파일럿이 보고한 p50/p90/p99/max와 정확히 일치):

| percentile | atr_pct | raw_tp(=×12) | raw_sl(=×6) |
|---|---:|---:|---:|
| p25 | 0.2101% | 2.52% | 1.26% |
| p50 | 0.2696% | 3.24% | 1.62% |
| p75 | 0.3486% | 4.18% | 2.09% |
| p90 | 0.4256% | 5.11% | 2.55% |
| p99 | 0.6685% | 8.02% | 4.01% |
| max | 0.9468% | 11.36% | 5.68% |

baseline floor(7.5%/4.0%)는 raw_tp/raw_sl이 p98~p99 부근을 넘을 때만 뚫린다(=floor가 ~98%
바인딩, 08-12 조사의 95~98.5% 실측과 일치). 각 후보는 이 floor를 `raw(atr_pct×mult)`가 각각
p75/p50/p25를 지나는 지점까지 낮춘 것이다(`tp_mult`/`sl_mult`=12/6 및 cap 0.22/0.12는 미변경,
cap은 2025~2026 전체에서 0% 발동이라 범위 밖):

| 후보 | min_tp | min_sl | floor 바인딩률(설계 의도) |
|---|---:|---:|---|
| C1 (p75-cross) | 0.0418 (4.18%) | 0.0209 (2.09%) | ~75% |
| C2 (p50-cross) | 0.0324 (3.24%) | 0.0162 (1.62%) | ~50% |
| C3 (p25-cross) | 0.0252 (2.52%) | 0.0126 (1.26%) | ~25% |

**컴포넌트 격리**: h48qual만 바꾸고 zig075는 라이브 그대로(0.075/0.040) 유지한 경우와, 그
반대(zig075만 바꾸고 h48qual 그대로)를 별도 행으로 각각 테스트했다 — 두 컴포넌트를 동시에
바꾸지 않는다(3후보×2컴포넌트=6셀 + baseline, 사전등록 grid를 작게 유지).

**재사용한 기존 하네스** (신규 작성 없이 import): `research_eth_omega461_exit_sweep_20260721`의
`load_frame`/`prep_component`/`replay_exit_variant`(컴포넌트 레벨), `replay_omega4_6_1_greedy_
router_20260706`의 `greedy_replay`(라이브와 동일한 단일 슬롯 h48qual>zig075 우선순위 포트폴리오
레벨), `research_eth_omega461_live_sltp_mfe_width_20260813`의 `_as_router_component`/
`_ledger_stats`/`_duration_gated` 헬퍼. 재학습 없음(냉동 예측 재사용, 시드 분산 차원 없음).
스크립트: `scripts/research_eth_omega461_atr_tpsl_floor_independent_percomponent_20260815.py`.

**G0 자체검증**: baseline 후보가 기존 알려진 라이브 baseline(포트폴리오 no_gate PnL
+36.82%/MDD -24.34%/29건)을 정확히 재현하는지 확인 — **통과**.

**OOS 데이터 이슈(발견 및 수정, 결과에 무관)**: 최초 실행에서 OOS 리플레이가
`RuntimeError: non-finite Regime3 route probabilities`로 중단됐다. 원인은 이 실험과 무관한
기존 데이터 문제 — `WIDE24_2026`에 2026-02-28 16:05~23:55 구간 95행(0.37%)의 커버리지 공백이
있다(`eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813.py`에 이미 문서화·수정된
선례와 동일). 그 선례와 동일하게 해당 95행을 리플레이 전에 제외했다(라이브 시스템도 regime
확률이 없는 bar는 라우팅할 수 없으므로 causally faithful한 처리).

**규율**: VAL 전수 스윕 → baseline 대비 pnl·mdd 둘 다 비악화(no_gate·with_gate 둘 다)인 후보가
있으면 그중 no_gate PnL 최고 후보 하나만 OOS 단일 터치. 라이브 파일
(`trading_bot_modules/omega4_6_1_live.py`/`trading_bot.py`/`runtime_config.py`/`.env`) 미변경.

## 결과 — VAL (2025-10-01~12-31)

| 후보 | 컴포넌트 | h48qual PnL/MDD/거래/TP바인딩 | zig075 PnL/MDD/거래/TP바인딩 | 포트폴리오 no_gate PnL/MDD/거래 | 포트폴리오 with_gate PnL/MDD/거래 |
|---|---|---|---|---:|---:|
| baseline | - | +5.45%/-11.62%/29/100.0% | +40.31%/-13.07%/29/99.9% | **+36.82%/-24.34%/29** | +54.88%/-31.11%/22 |
| h48qual_p75 | h48qual | -1.30%/-9.19%/46/81.4% | (불변) | **+38.53%/-20.28%/35** | +50.77%/-27.42%/28 |
| **h48qual_p50** | h48qual | +3.09%/-8.63%/51/65.8% | (불변) | **+39.50%/-21.50%/36** | **+59.57%/-24.88%/27** |
| **h48qual_p25** | h48qual | +1.68%/-8.00%/51/24.8% | (불변) | **+43.91%/-19.13%/36** | **+59.88%/-24.84%/28** |
| zig075_p75 | zig075 | (불변) | -13.50%/-18.21%/55/93.5% | **-33.96%/-34.29%/45** | -21.60%/-30.36%/35 |
| zig075_p50 | zig075 | (불변) | -17.91%/-18.51%/71/75.3% | **-42.95%/-43.30%/54** | -9.41%/-20.98%/41 |
| zig075_p25 | zig075 | (불변) | -16.09%/-16.77%/72/47.8% | **-42.31%/-43.54%/56** | -13.98%/-21.02%/42 |

**zig075 단독 재보정은 VAL에서 3후보 전부 결정적으로 나쁘다**(포트폴리오 PnL -34%~-43%,
부호까지 반전) — zig075는 baseline 성과의 대부분(+40.31%, 컴포넌트 PnL 기준)을 담당하는
쪽이라 이 컴포넌트의 floor를 좁히면(거래수 29→55~72로 급증, 평균 보유 726→252~351bar로
급감) 승률 좋은 소수 거래가 승률 낮은 다수 거래로 대체되며 크게 악화된다.

**h48qual 단독 재보정은 VAL에서 3후보 전부 baseline을 pnl·mdd 둘 다, no_gate·with_gate 둘 다
비악화**(qualifiers = `h48qual_p50`, `h48qual_p25` — `h48qual_p75`는 with_gate PnL이
baseline보다 낮아 탈락). h48qual은 baseline 컴포넌트 PnL이 원래 미미했던 쪽(+5.45%)이라
floor를 좁혀 거래수를 29→51로 늘려도 zig075와 달리 포트폴리오에 해를 주지 않고 오히려
no_gate PnL을 +36.82%→+43.91%(`h48qual_p25`)까지 개선했다. no_gate PnL 기준 최고 후보
`h48qual_p25`(min_tp=0.0252, min_sl=0.0126)를 OOS 단일 터치 대상으로 선정.

## 결과 — OOS (2026-01-01~03-31, 단일 터치)

| 후보 | no_gate PnL/MDD/거래 | with_gate PnL/MDD/거래 |
|---|---:|---:|
| baseline | +49.32% / -16.20% / 24 | +44.48% / -15.48% / 20 |
| h48qual_p25 | **+38.16%** / **-28.64%** / 23 | **+19.96%** / -28.64% / 18 |

**OOS에서 결정적으로 반전(REVERSES)**: PnL이 no_gate 기준 +49.32%→+38.16%(악화), MDD가
-16.20%→-28.64%로 거의 2배 악화. with_gate 기준은 더 심하다(+44.48%→+19.96%, MDD 동일하게
악화). VAL에서 보였던 h48qual 단독 floor-narrowing 개선은 OOS로 이어지지 않았다.

## 결론 — REJECTED

**두 열린 축(floor 절대값 독립 변경, 컴포넌트별 분리) 모두 시도했지만 채택 가능한 변경은
없다.**

- **zig075 단독 floor 좁히기**: VAL에서 이미 결정적으로 부정(3/3 후보 전부 baseline보다
  나쁨, OOS 미개봉).
- **h48qual 단독 floor 좁히기**: VAL은 통과(2/3 후보가 baseline 비악화, 최고 후보는 포트폴리오
  PnL +7%p 개선)했으나, 사전등록된 단일 터치 OOS에서 반전됐다(PnL 하락 + MDD 거의 2배 악화).

08-13 파일럿의 "넓히기" 방향과 이번 "좁히기" 방향을 합쳐 보면, floor 값을 baseline(7.5%/4.0%)
에서 어느 방향으로 움직이든(넓히든 좁히든, 두 컴포넌트 중 어느 쪽을 바꾸든) 이 백테스트
구간에서 성과를 안정적으로 개선하는 조합을 찾지 못했다. 특히 h48qual 단독 좁히기가 VAL에서
보인 개선은 표본 크기가 작은 VAL 구간(29→36건)에 대한 과적합/우연일 가능성이 높다 — OOS의
거래수 변화(24→23건, 사실상 동일 표본)에서도 MDD가 거의 2배로 뛴 것은 좁은 floor가 개별
거래의 손절 폭을 줄여 스탑아웃 빈도를 높이는 부작용이 OOS 구간에서 더 강하게 나타났음을
시사한다.

**최종 판단**: "ATR 적응형"이라는 이름과 다르게 동작하는 현재 라이브 floor(0.075/0.040)를
이름에 맞게 고치려는 시도(08-13 넓히기, 08-15 좁히기·컴포넌트분리 모두)가 이 두 실험
전체에서 baseline을 결정적으로 능가하는 사례를 하나도 만들지 못했다. floor/cap 재보정 축은
이것으로 사실상 소진됐다고 본다 — 남은 미탐색 조합(예: TP만 바꾸고 SL은 유지하는 등 tp/sl
floor를 서로 다른 방향으로 독립 이동)은 이론적으로 가능하지만, 두 번 연속 결정적 부정을
낸 뒤라 사전 확률이 낮고 추가 multiple-testing 부담이 크므로 이 세션에서는 더 이상
확장하지 않는다.

## 준수 확인

- `git status`/`git diff` 기준 `trading_bot_modules/omega4_6_1_live.py`/`trading_bot.py`/
  `runtime_config.py`/`.env` 무변경.
- 재학습 없음(냉동 예측 재사용), 저장 원장 미사용, VAL-게이트 후 단일 터치 OOS 규율 준수.
- fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
  saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
- 스크립트: `scripts/research_eth_omega461_atr_tpsl_floor_independent_percomponent_20260815.py`.
  산출물: `tmp/causal_regen_20260516/eth_omega461_atr_tpsl_floor_independent_percomponent_20260815/`
  (`report.json`, `component_val.csv`, `portfolio_val.csv`, `component_oos.csv`,
  `portfolio_oos.csv`).
