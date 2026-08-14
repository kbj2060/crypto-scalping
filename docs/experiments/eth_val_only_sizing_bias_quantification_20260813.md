# ETH Omega4.6.1 — VAL-전용 사이징 편향 정량화 (h48qual/zig075, 2026-08-13)

## 배경 및 후속 관계

이 문서는 같은 날 밤 작성된
`docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md`의 **직접 후속**이다. 그
문서의 5-1절은 실제 라이브 `risk_sidecar.pkl`(h48qual/zig075 둘 다)을 unpickle해
`selection_scope="validation_only"`, `selection_objective="log_risk"`임을 확인하고,
`scripts/train_eval_omega4_2_risk_sidecar_20260622.py:1417-1445`에서 margin_fraction/leverage
매핑(`selected_mapping`)이 VAL(2025-10-01~12-31) 구간의 `log_risk_utility`/`mdd`/`pnl`을 직접
최대화하는 후보로 그리드서치되는 코드를 확인했다. **그 문서는 이 편향을 정성적으로만
지적했고 정량화하지 않았다.** 이 문서는 "VAL에서의 우수한 성과 중 얼마만큼이 사이징 곡선
자체 때문이고 얼마만큼이 진짜 방향/품질 신호 때문인가"를 숫자로 분리한다.

**결론 미리보기**: 컴포넌트별로 답이 다르다. **zig075는 사이징이 VAL 우세폭의 상당 부분을
설명한다**(VAL 우세폭이 OOS의 3.4배). **h48qual은 사이징이 VAL에 특이적이라는 증거가 없다**
(오히려 OOS 우세폭이 VAL보다 절대·상대 양쪽 기준 모두 크다). 즉 "사이징이 편향의 보편적
주범"이라는 명제는 두 컴포넌트에 균일하게 적용되지 않는다.

## 방법론

### 핵심 아이디어 — entry/exit을 완전히 고정한 채 사이징만 재가격

1. **하나의 ledger만 생성한다.** 컴포넌트(h48qual/zig075) × 구간(VAL/OOS) 조합마다,
   `research_eth_omega461_exit_sweep_20260721.py`의 `prep_component()`/`replay_exit_variant()`
   (오늘 밤 이미 최소 4개의 형제 스크립트가 재사용한 바로 그 하네스이며, 인증된
   `train_eval_omega4_2_risk_sidecar_20260622.py`의 `_replay_with_risk` 루프와 구조적으로
   동일함을 그 스크립트 자신의 docstring이 명시)로 **실제 라이브 사이드카의 VAL-선택
   margin/leverage 매핑**(`prep_component()`가 `_Component.entry_decision()`의 사이징 수식을
   `_risk_margins`/`_risk_leverage` + `pkl["selected_mapping"]`로 그대로 재현)을 먹여 인과적
   bar-by-bar 1회 순방향 재생을 수행한다. 이 시점에 entry timing/side, exit timing/reason,
   raw price-move 기반 수익률이 전부 고정된다.
2. **같은 ledger를 두 가지 사이징으로 재가격한다.** 사이드카의 프로모션 파이프라인 자체가
   이미 내부적으로 쓰는 함수인 `train_eval_omega4_2_risk_sidecar_20260622._ledger_metrics_with_margins()`
   (고정 ledger에 대해 entry/exit을 재시뮬레이션하지 않고 margin/leverage 후보만 바꿔 비교하는
   용도로, 그 스크립트의 프로모션 셀렉션 루프 1350~1500행에서 이미 쓰이고 있음)을 그대로
   재사용해 (a) 실제 사이징 그대로(margins=None, pass-through) 재가격하고, (b) flat 사이징으로
   재가격한다. 이 함수는 이미 한 번 계산된 `net_per_notional`(가격변동+수수료만 반영된,
   notional 1단위당 순수익률)에 새 notional을 곱할 뿐 take_profit/stop_loss는 절대 건드리지
   않는다(두 라이브 사이드카 모두 `notional_scaled_sltp=False`로 확인돼 애초에 TP/SL이
   notional에 결합돼 있지도 않다). **두 시나리오가 동일한 ledger 객체를 재사용하므로
   entry_signal_i/exit_i/side/reason이 구조적으로 완전히 동일하다** — 거래수·승패 방향이
   달라질 수 없다. 스크립트는 이걸 assert로도 강제한다(entry/exit 인덱스, reason, win 컬럼이
   두 시나리오 간에 정확히 일치하는지 확인 후 불일치 시 예외 발생).
3. **정합성(integrity) 검증**: ledger-재가격 방식으로 복원한 "실제 사이징" PnL이 원래
   bar-by-bar 재생 자신의 PnL과 일치하는지 매 셀마다 assert했다 — 실제 결과는 4개 셀 전부
   오차 `< 5e-14`pp(부동소수점 오차 수준)로 사실상 완전 일치, 재가격 방법론이 건전함을 확인.

이 설계는 사용자가 명시적으로 경고한 "레버리지 이중계산" 함정(Futures Risk Sizing Contract,
CLAUDE.md)을 원천적으로 피한다 — TP/SL은 이미 고정된 ledger의 `reason`/`raw_exit_price_move`에
녹아있을 뿐 재계산되지 않고, notional은 `price_move * notional`로만 재가격된다.

### Flat 사이징 기준값

`train_eval_omega1_2_tabm_diffusion_risk_20260603.BASE_TEMPLATE`(`notional=0.45`,
`leverage=2.0` → `margin_fraction=0.225`)을 **모든 활성 거래에 side/score/레짐과 무관하게
동일하게** 적용했다(`EXPERT_SCALES`도 side asymmetry도 적용 안 함 — "flat"의 가장 문자
그대로의 해석). 이 값은 이 서브프로젝트의 모든 always-short/always-long 기준선이 이미 쓰는
고정 사이징이며, 사이드카의 VAL 그리드서치가 손댄 적이 없는 상수다. `selected_mapping`의
score-중앙값(z=0) 지점 같은 대안은 **의도적으로 쓰지 않았다** — floor/cap/leverage_min/max
자체가 이미 VAL로 선택된 값이라, 그 커브 위 어느 점을 찍어도 여전히 VAL-fit 정보가 새어
들어오기 때문이다(진짜 VAL-blind 기준이 아님).

### 구간 및 스코프

- VAL = 2025-10-01~12-31, OOS = 2026-01-01~03-31 (`research_eth_omega461_exit_sweep_20260721.py`
  기본값 = CLAUDE.md 표준 OOS 창과 동일; VAL은 parent 모델의 frozen OOF 예측이 2025-10-01부터만
  존재해 표준보다 1개월 늦게 시작 — 오늘 밤 다른 모든 형제 스크립트와 동일한 명시된 편차).
- **컴포넌트 단위 독립 분석**: h48qual/zig075 각각을 라이브의 `PRIORITY` greedy 라우터로
  결합하지 않고, 각자의 전체 active set에 대해 독립적으로 백테스트했다(하네스 스크립트
  자체의 설계 스코프와 동일, 그리고 리서치 대상인 `_ledger_metrics_with_margins`/사이드카
  선택 파이프라인 자체가 컴포넌트 단위로 동작하므로 스코프가 정확히 일치한다). 실제 결합
  운영 시 두 컴포넌트가 우선순위로 나눠 갖는 bar 비중은 여기 포함되지 않는다 — 한계 절 참고.
- **데이터 갭 처리**: `regime3_current_sensitive_wide24` 오버레이 CSV가 OOS 구간 중
  2026-02-28 16:05~23:55(연속 95 bar, 약 7.9시간, base CSV와 직접 diff로 확인)를 갖고 있지
  않아 left-merge 후 NaN이 생기고, `_route_id()`가 이를 정상적으로 거부한다. 이 gap은 이
  스크립트가 만든 게 아니라 오버레이 CSV 자체의 기존 결측(아마 `eth_omega4_6_1_live_risk_
  assessment_20260812.md`의 이슈 5가 기록한 2026-08-12 오버레이 파일 복구의 부작용) —
  공유 오버레이 파일이나 `load_frame`은 건드리지 않고, 이 스크립트 로컬에서만 해당 95개
  bar(OOS 25633건 중 0.37%, 하루 저녁 한 덩어리, 흩어진 구멍 아님)를 걸러냈다.

### Fresh-Forward 준수

`fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`(이 스크립트가 자체
1회 순방향 재생으로 만든 ledger를 같은 run 내에서만 재사용, 과거 저장 ledger를 읽지 않음),
`saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`. 재학습은
전혀 하지 않았으며(frozen parent bundle + frozen risk sidecar만 재생), 사이드카가
frozen artifact이므로 시드 분산 축 자체가 없다(CLAUDE.md의 Seed-Diversity Ensemble
Promotion Gate는 신규로 재학습되는 진짜 시드 앙상블을 겨냥한 것이라 여기엔 해당 없음).

산출물: 스크립트 `scripts/research_eth_omega461_val_sizing_bias_quantification_20260813.py`,
실행 로그 `tmp/research_20260813/omega461_val_sizing_bias_quantification_run.log`, 결과
`tmp/research_20260813/omega461_val_sizing_bias_quantification/{sizing_bias_raw.csv,
sizing_bias_summary.csv, sizing_bias_full.json}`.

---

## 결과

### PnL 우세폭 (핵심 표)

| 컴포넌트 | 구간 | 실제 사이징 PnL | flat 사이징 PnL | 우세폭(실제−flat, pp) | 우세폭(상대, %) | 거래수 |
|---|---|---:|---:|---:|---:|---:|
| h48qual | VAL | +5.45% | +5.20% | **+0.26pp** | +4.9% | 29 |
| h48qual | OOS | +9.49% | +8.66% | **+0.84pp** | +9.7% | 14 |
| **h48qual VAL/OOS 우세폭 비율** | | | | **0.31×** | | |
| zig075 | VAL | +40.31% | +18.80% | **+21.51pp** | +114.4% | 29 |
| zig075 | OOS | +17.12% | +10.81% | **+6.31pp** | +58.4% | 25 |
| **zig075 VAL/OOS 우세폭 비율** | | | | **3.41×** | | |

- **h48qual**: VAL 우세폭(+0.26pp)이 OOS 우세폭(+0.84pp)보다 절대값도, 상대값도(4.9%
  vs 9.7%) 모두 작다 — 비율 0.31배로 **판정 기준의 반대 방향**(사이징이 VAL에 유리하게
  맞춰져 있다면 비율이 1보다 훨씬 커야 하는데, 오히려 1보다 작다). 사이징 자체의
  기여분이 두 구간 모두 실제 PnL의 5~10% 수준으로 작다.
- **zig075**: VAL 우세폭(+21.51pp)이 OOS 우세폭(+6.31pp)의 **3.41배** — 절대·상대(114%
  vs 58%) 양쪽 기준 모두 VAL에 뚜렷하게 편중돼 있다. 다만 OOS 우세폭도 완전히 0은
  아니라서(+6.31pp, 상대 +58%), "사이징이 OOS에서 전혀 안 통한다"는 과장이고 "VAL에서
  특히 부풀려져 있다"가 더 정확하다.

### MDD 우세폭 (보조 지표)

| 컴포넌트 | 구간 | 실제 사이징 MDD | flat 사이징 MDD | 우세폭(실제−flat, pp)\* | 거래수 |
|---|---|---:|---:|---:|---:|
| h48qual | VAL | -10.46% | -8.63% | **-1.82pp** | 29 |
| h48qual | OOS | -6.53% | -5.73% | **-0.80pp** | 14 |
| **h48qual VAL/OOS \|우세폭\| 비율** | | | | **2.28×** | |
| zig075 | VAL | -11.55% | -6.59% | **-4.97pp** | 29 |
| zig075 | OOS | -8.72% | -5.95% | **-2.77pp** | 25 |
| **zig075 VAL/OOS \|우세폭\| 비율** | | | | **1.79×** | |

\* 부호 규약: 양수 = 실제 사이징이 flat보다 MDD가 얕음(더 좋음), 음수 = 실제 사이징이
flat보다 MDD가 깊음(더 나쁨). 4칸 전부 음수 — **실제(VAL-선택) 사이징은 PnL 방향과
무관하게 flat보다 항상 MDD가 나쁘다.** 레버리지/margin을 평균적으로 더 크게 쓰기 때문에
당연한 결과이며(아래 노출 비교 참고), 선택 목적함수가 `log_risk_utility`(성장 항 +
tail/liquidation 페널티 항의 조합)이지 MDD 직접 최소화가 아니므로 모순은 아니다.
흥미로운 점: **MDD의 VAL/OOS 편중 비율(h48qual 2.28×, zig075 1.79×)은 PnL의 편중 비율
(h48qual 0.31×, zig075 3.41×)과 다른 패턴을 보인다** — 특히 h48qual은 PnL 기준으로는
VAL-특이적 편향의 증거가 없는데 MDD 기준으로는 약하게나마 있다. PnL과 MDD가 같은 결론을
내지 않는다는 것을 있는 그대로 기록한다.

### 노출(사이징 크기) 비교 — 왜 두 컴포넌트가 다른가

| 컴포넌트 | 구간 | 평균 실제 notional | 평균 실제 margin_fraction | 평균 실제 leverage | flat notional/margin/leverage |
|---|---|---:|---:|---:|---:|
| h48qual | VAL | 0.544 | 0.266 | 2.046 | 0.45 / 0.225 / 2.0 |
| h48qual | OOS | 0.530 | 0.265 | 1.996 | 〃 |
| zig075 | VAL | 0.687 | 0.321 | 2.126 | 〃 |
| zig075 | OOS | 0.645 | 0.306 | 2.093 | 〃 |

h48qual의 실제 평균 노출은 flat보다 약 18~21% 큰 정도에 그치는 반면, zig075는 약
43~53% 크다 — **사이징 곡선이 flat과 벌어지는 정도 자체가 zig075에서 훨씬 크고, 이게
zig075의 PnL 우세폭이 훨씬 큰 이유와 직접 연결된다.** 참고로 zig075는 LONG 평균 notional
(0.805, VAL·OOS 동일값)이 SHORT(0.642/0.594)보다 뚜렷이 높다 — `selected_mapping`의
`long_leverage_scale=0.95 < short_leverage_scale=1.05`(레버리지는 오히려 숏이 근소하게
우대)와는 반대 방향인데, LONG 표본이 6~8건뿐이라 score 자체의 분포 차이(적게 통과된 만큼
평균적으로 더 극단적인 score)가 leverage-side-scale보다 지배적이었을 가능성이 높다 —
표본이 작아 확정적 설명은 아니며, 이 문서의 핵심 결론에 영향을 주는 디테일은 아니다.

---

## 판정 기준 적용 (사용자 지정)

> VAL 우세폭이 OOS 우세폭보다 훨씬 크면 사이징이 VAL 특이적 편향의 증거, 비슷하면 사이징이
> 주범이 아니다.

컴포넌트별로 답이 갈린다:

- **zig075: 편향 증거 있음.** VAL 우세폭이 OOS의 3.41배(PnL), MDD 저하폭도 VAL이 OOS의
  1.79배로 같은 방향. 사이징 곡선이 VAL 구간에 상당히 특이적으로 맞춰져 있고, OOS로
  일반화되는 부분은 (완전히 0은 아니지만) VAL 대비 눈에 띄게 작다.
- **h48qual: 편향 증거 없음, 오히려 반대 방향.** PnL 기준 VAL/OOS 우세폭 비율이 0.31배로
  1보다 작다 — OOS에서 실제 사이징이 flat보다 이기는 정도가 VAL보다 절대·상대 모두 크다.
  사이징 자체가 h48qual 결과에 기여하는 정도는 두 구간 다 작고(전체 PnL의 5~10%),
  VAL-특이적이라는 신호가 없다.

**종합**: "사이징이 VAL 편향의 보편적 주범"이라는 명제는 기각한다 — 컴포넌트 의존적이다.
`eth_val_oos_regime_mismatch_investigation_20260813.md` 5-1절이 제시한 코드 증거(두
사이드카 모두 `selection_scope="validation_only"`) 자체는 여전히 사실이지만, 그 **선택
절차의 존재**가 곧 **선택 결과의 VAL 특이성**을 보장하지는 않는다는 것이 h48qual에서
드러난다 — h48qual의 사이징 곡선은 VAL로 그리드서치됐음에도 불구하고 실제로는 flat과
크게 다르지 않은 지점에 정착했고(노출 표 참고), 그 결과 우연히도 OOS에 대해서도(사실은
OOS에 대해 더 잘) 일반화된다. zig075는 반대로 사이징 곡선이 flat과 뚜렷이 다른 지점에
정착했고, 그 차이의 상당 부분이 VAL에만 특이적으로 들어맞았다.

---

## 추가 조사: VAL이 파이프라인에서 몇 겹으로 선택에 쓰였는가 (낮은 우선순위)

사이징 외에 `duration_threshold`와 `quality_threshold`도 확인했다(스크립트 검색 +
저장된 선택 아티팩트 CSV를 직접 열어 대조, 재학습 없음). **결과: 원래 가정("사이징 →
quality_threshold → 오늘 밤 신규 후보, 전부 VAL-only")이 틀렸다 — 3개 레이어 중 2개만
VAL-only이고, 나머지 하나(quality_threshold)는 VAL이 아니라 OOS-primary로 선택됐다.**

### duration_threshold — 확인됨: VAL-only, 16개 후보 전수 그리드

`scripts/select_duration_gate_threshold_val_20260706.py`의 docstring이 명시: "Selection is
VALIDATION-ONLY (2025-10-01..12-31, ...)"(1~11행), `VAL_START`/`VAL_END`만 정의되고 OOS
데이터는 스크립트 어디에서도 로드하지 않는다(37행). 후보 그리드는 VAL 내 활성 거래의
`ou_halflife` 분위수 16개(`np.arange(0.05, 0.85, 0.05)`, 89~90행), 목적함수는
`eval_omega4_6_duration_aware_risk_layer_20260630.duration_priority_score`(263~271행,
`2.0*monthly_min_pnl + 0.10*val_pnl + 0.10*max_hold_gain + 0.03*avg_hold_gain`, 전부 VAL
ledger에서만 계산)이며 `mdd>=-20%`/`trades>=0.65*baseline` 게이트 통과 후 점수 내림차순
1위를 선택(109~115행). 저장된 산출물
`tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/duration_threshold_val_selection.csv`을
직접 열어 확인 — 16행 중 `threshold=0.005417, score=88.27`(전체 최고점, pnl=163.30%,
mdd=-10.51%, trades=22)이 라이브 값과 정확히 일치한다. `trading_bot_modules/omega4_6_1_
live.py:63`의 "VAL-reselected 2026-07-06" 주석, `docs/model_contracts/omega4_6_1_full_
architecture_blueprint_20260706.md:219`의 "re-selected via VAL-only grid search"와도
일치. **이 레이어는 사이징과 같은 패턴(VAL-only) — 확정.**

### quality_threshold — 원 문서의 가정과 반대: VAL-only가 아니라 OOS-primary 선택 (신규 발견)

`eth_val_oos_regime_mismatch_investigation_20260813.md`는 "Odyssey 서브프로젝트 메모에
따르면 quality_threshold도 0.40~0.80 VAL 스윕으로 선택됐다"고 적었다. 이걸 코드로 직접
재확인하니 **다른 그림이 나왔다.** 선택 스크립트는 `scripts/train_eval_omega4_3head_
parent72_loose_entry_quality_20260620.py`이며, quality threshold 후보 그리드를 훑어 VAL/OOS
양쪽의 pnl/mdd/wr/trades를 계산하고(`_metric_row`, 602~610행) `quality_threshold_ranking.csv`로
저장하는데, **정렬 키가 `(oos_pnl, validation_pnl)` 내림차순이다** —
1173행: `rows.sort(key=lambda r: (float(r["oos_pnl"]), float(r["validation_pnl"])), reverse=True)`
(직접 grep으로 재확인). **VAL PnL이 아니라 OOS PnL이 1순위 정렬 기준.**
`docs/model_contracts/registry.json:75,80`이 가리키는 실제 아티팩트 디렉터리(이 문서가 쓰는
`research_eth_omega461_exit_sweep_20260721.py`의 `COMPONENTS` 번들 경로와 정확히 일치)에서
두 `quality_threshold_ranking.csv`를 직접 pandas로 열어 재검증했다(표는 저장된 CSV의 실제
행 기준, h48qual 13개 후보/zig075 9개 후보):

| 컴포넌트 | 배포값 | 배포값의 VAL pnl | 배포값의 OOS pnl(전체 1위) | VAL pnl 기준 1위 후보 | 그 후보의 VAL pnl |
|---|---|---:|---:|---|---:|
| h48qual | q=0.50 | 4.58% | **10.65%(1위)** | q=0.35 | **22.47%** |
| zig075 | q=0.75 | 11.09% | **14.77%(1위)** | q=0.55 | **13.37%** |

두 컴포넌트 모두 배포된 threshold는 OOS pnl 기준 전체 후보 중 1위이지만, VAL pnl 기준으로는
1위가 아니다(h48qual은 격차가 특히 크다 — 4.58% vs 22.47%, 거의 5배). **이건 이
서브프로젝트가 지금까지 우려해온 "VAL 과최적화"와는 다른, 더 직접적인 문제다** — OOS 성과
자체가 이 레이어의 threshold 선택 1순위 기준으로 쓰였다는 뜻이므로, quality_threshold가
관여하는 과거 OOS 숫자들에 대해서는 오히려 "OOS가 사실은 순수한 readout이 아니었다"는
우려가 성립한다. `report.json`에는 `selected`/`selection_scope` 필드가 없고,
`docs/model_contracts/omega4_6_1_full_architecture_blueprint_20260706.md:41`도 이 값들을
선택 절차 설명 없이 명시만 한다 — 같은 문서가 몇 절 뒤에서 duration gate는 "VAL-only grid
search"라고 명시적으로 적은 것과 대조적이다. **`eth_val_oos_regime_mismatch_investigation_
20260813.md`의 "quality_threshold도 VAL 스윕으로 선택" 서술은 이 직접 확인으로 정정이
필요하다** — 정확히는 "OOS-primary 정렬로 선택됐고, 그 결과가 VAL-optimal과는 다르다."

### 요약 — 확인된 3개 레이어의 실제 패턴

| 레이어 | 선택 스크립트 | 선택 기준 | VAL-only? |
|---|---|---|---|
| 사이징(margin/leverage 매핑) | `train_eval_omega4_2_risk_sidecar_20260622.py` | `log_risk_utility`(VAL) | **예**(unpickle로 재확인, `selection_scope="validation_only"`) |
| duration_threshold | `select_duration_gate_threshold_val_20260706.py` | `duration_priority_score`(VAL) | **예**(직접 확인) |
| quality_threshold | `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py` | `(oos_pnl, validation_pnl)` 내림차순 | **아니오 — OOS-primary**(직접 확인, 원 문서 가정과 반대) |

"몇 겹이 VAL에 맞춰졌는가"라는 원래 질문의 답은 처음 가정("사이징 → quality_threshold →
오늘 밤 신규 후보, 전부 VAL")보다 복잡하다 — **2개 레이어(사이징, duration)는 VAL-only,
1개 레이어(quality_threshold)는 OOS-primary다.** 이건 "겹의 개수가 줄었다"는 좋은 소식이
아니라 **다른 종류의 문제(OOS 누출)가 하나 더 있다는 소식**이다 — quality_threshold가
관여하는 모든 과거 OOS 성과 숫자(`research_eth_omega461_exit_sweep_20260721.py`를 재사용하는
오늘 밤의 모든 형제 스크립트, 이 문서의 결과 포함)는 "완전히 처음 보는 OOS"가 아니라
"그 threshold를 고르는 데 이미 한 번 쓰인 OOS"라는 뜻이다. 이 발견은 이 문서의 핵심 결론
(사이징 편향의 컴포넌트별 크기)에는 영향을 주지 않는다(이 문서의 사이징 비교는
quality_threshold를 양쪽 시나리오에서 동일하게 고정한 채 실행됐으므로, quality_threshold의
선택 방식과 무관하게 유효하다) — 다만 **서브프로젝트 전체가 "OOS는 순수 readout"이라는
전제로 지금까지 보고해 온 다른 모든 결과에 대해서는 별도의, 더 시급한 후속 조사가
필요함을 시사한다.**

두 번째 낮은 우선순위 항목("VAL의 이례적으로 약한 가격-오류 상관관계가 실제로 특이한지,
2024~2025 전체에서 무작위 13주 구간을 뽑아 분포와 비교")은 이번 세션 시간 예산 내에서
수행하지 않았다 — 새로운 주간 단위 wrong-way-error 계산 파이프라인을 처음부터 구성해야
해서 이 문서의 핵심 정량화(사이징 편향)보다 비용이 훨씬 크고, 사용자가 명시적으로 "우선순위
낮음, 시간 되면만"으로 표시한 항목이다. 미착수로 남긴다.

---

## 스코프와 한계

1. **컴포넌트 단위 독립 분석이지, 라이브의 PRIORITY greedy 라우터 결합이 아니다.** 실제
   운영에서는 h48qual이 우선순위 1이라 h48qual이 신호를 내는 bar는 zig075가 아예 활성화되지
   않는다. 이 문서는 각 컴포넌트를 자기 전체 active set 기준으로 독립 평가했으므로, 결합
   운영 시 zig075가 실제로 차지하는 bar 비중(따라서 zig075의 사이징 편향이 결합 포트폴리오
   PnL에 기여하는 실제 비중)은 이 표만으로 알 수 없다. `research_eth_omega461_exit_sweep_
   20260721.py` 자신의 스코프와 동일하며, 사이드카 프로모션 파이프라인 자체도 컴포넌트
   단위로 동작하므로 이 리서치 질문("사이드카의 VAL 그리드서치가 얼마나 편향됐는가")에는
   정확히 맞는 스코프다.
2. **Flat 기준값은 하나만 시도했다(BASE_TEMPLATE).** 다른 flat 후보(예: `selected_mapping`의
   score-중앙값)는 그 자체가 VAL로 선택된 floor/cap/leverage 경계를 물려받으므로 진짜
   VAL-blind가 아니라고 판단해 의도적으로 배제했다 — 방법론 절 참고. 이 결론이 "정확히 어떤
   상수를 flat으로 골랐는지"에 얼마나 민감한지는 별도로 검증하지 않았다.
3. **MDD는 근사치다(PnL은 정확치).** `_ledger_metrics_with_margins`는 거래 종료 시점과
   `mae_price_move`(거래 중 최대 역행폭) 지점만으로 peak/trough를 마크하고, 거래 도중의
   유리한(MFE) 순간이 새 historical peak를 세울 수 있다는 점은 반영하지 않는다. 이번 4개
   셀 중 "실제 사이징" 시나리오에 한해 진짜 bar-by-bar MDD와 직접 대조한 결과, ledger-재가격
   MDD가 실제보다 0.01~2.29pp 얕게(더 좋게) 나온다(h48qual VAL 1.16pp, h48qual OOS
   0.01pp, zig075 VAL 1.51pp, zig075 OOS 2.29pp 과소평가). **PnL은 두 방법이 정확히 일치하므로
   이 문서의 핵심 판정(PnL 우세폭 비교)에는 영향이 없다.** MDD 표의 절대값은 두 시나리오
   모두 같은 방법으로 계산해 상대 비교(우세폭)는 유효하지만, "실제 MDD가 몇 %냐"의 절대
   숫자는 하한값으로 읽어야 한다.
4. **결정론적 재생, 신뢰구간 없음.** 사이드카가 frozen HistGradientBoostingRegressor이고
   재학습을 하지 않았으므로 각 셀은 정확히 하나의 숫자다(시드 분산으로 인한 불확실성 자체가
   없음). 다만 거래수가 14~29건으로 작아(`eth_val_oos_regime_mismatch_investigation_
   20260813.md`가 이미 지적한 jackknife 민감도 문제) 거래 1~2건의 승패가 우세폭 자체를
   크게 흔들 수 있다는 점은 여전히 유효한 우려다 — 이 문서는 그 우려를 해소하지 않는다.

## 결론 (진단 전용, 채택/승격 제안 아님)

이 결과는 순수 진단이며, 이 결과를 근거로 특정 사이징 방식을 채택하거나 모델을 승격하자는
제안이 아니다. 정량적으로 확인된 것은:

- **사이징 자체가 "VAL 편향의 보편적 주범"은 아니다** — zig075에서는 VAL 우세폭의 상당
  부분(3.4배 편중)을 설명하지만, h48qual에서는 설명하지 못한다(오히려 역방향).
- `eth_val_oos_regime_mismatch_investigation_20260813.md`가 제시한 "3중 VAL 재사용" 가설
  (사이징 → quality_threshold → 오늘 밤 각 신규 후보, 전부 VAL-only)은 이 문서의 추가 조사로
  **정정이 필요하다: 사이징·duration_threshold 2개 레이어는 VAL-only가 맞지만,
  quality_threshold는 VAL-only가 아니라 OOS-primary로 선택된 것이 직접 확인됐다**(위
  "추가 조사" 절). quality_threshold의 OOS-primary 선택은 성격이 반대다 — VAL 성과를
  부풀리는 게 아니라 **OOS 성과 자체를 선택 기준으로 써서 OOS 쪽을 낙관적으로 편향**시키므로,
  "VAL에서 이기고 OOS에서 반전"이라는 이 세션의 관찰 패턴 자체를 설명하는 후보로는 방향이
  맞지 않는다(오히려 그 반전을 더 두드러지게 보이게 만드는 방향). 즉 h48qual 위에 얹힌 후속
  후보들의 VAL 우세-OOS 반전 원인은 사이징(이 문서로 컴포넌트 의존적임이 확인됨)도
  quality_threshold(방향이 안 맞음)도 단독 설명이 못 되며, 후보 자체의 VAL 게이트·저표본
  재사용·그 문서 5-2절이 지적한 "VAL 구간 자체의 구조적 이례성" 쪽에 더 무게가 실린다 —
  다만 이 문서 단독으로 그 인과관계를 확정하지는 않는다. quality_threshold의 OOS-primary
  선택 자체는 이 세션이 지금까지 "OOS는 순수 readout"으로 취급해 온 다른 모든 결과에 대한
  독립적이고 더 시급한 우려로, 별도 후속 조사가 필요하다.
