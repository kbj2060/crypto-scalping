# ATR TP/SL floor 레짐조건부 재보정 — "질적으로 다른 가설"도 OOS 반전 (2026-08-18)

## 배경

[[eth_omega461_atr_tpsl_floor_recalibration_closed_20260815]]는 floor(0.075/0.040)를 넓히기·
좁히기·배율조정·컴포넌트분리 4개 축 전부 baseline을 못 이겨 종결했고, "새로 시도하려면
질적으로 다른 새 가설(예: 레짐조건부 floor)이 필요하다"고 명시했다. 오늘 exit_head 재점검
과정에서 사용자가 "TP 7.5%/SL 4.0%가 너무 크다, ATR 적응형이라 할 수 없다"고 재차 지적 —
08-12 조사(95~98.5% 바인딩률)와 오늘 독립 재계산(97.5%/98.5%)이 일치함을 재확인한 뒤,
사용자가 명시적으로 "레짐조건부 floor" 방향을 선택해 이 실험을 진행했다.

## 문헌 근거

paper-lookup 스킬로 확인(2026-08-18): **Kaminski & Lo (2014), "When Do Stop-Loss Rules Stop
Losses?", Journal of Financial Markets** (Semantic Scholar 83회 인용). 핵심 발견 — Random Walk
Hypothesis 하에서는 손절 규칙이 항상 기대수익을 깎지만, **모멘텀(추세) 존재 시엔 손절 규칙이
가치를 더할 수 있다** — 명시적 레짐전환 모형으로 이를 보인다. 이는 "레짐(추세 vs 횡보)에
따라 손절 폭을 다르게" 조건화해야 한다는 가설을 직접 뒷받침한다 — 단일 전역 상수로는 표현할
수 없는 메커니즘이라 08-15가 요구한 "질적으로 다른 가설" 기준을 충족한다.

## 설계 전 데이터 점검 — 두 가지 사전 가정이 기각됨

1. **레짐별 ATR 분포 자체는 크게 다르지 않다.** bull/bear/chop 3개 레짐(`hard._route_id`)별
   atr_pct(window=192) 중앙값이 각각 0.273%/0.294%/0.254%로 비슷하고, floor는 **세 레짐
   전부에서 96~99% 바인딩**된다(bull 96.6/97.7%, bear 96.3/97.9%, chop 98.7/99.3%). 즉 "특정
   레짐은 원래 변동성이 커서 floor를 자연스럽게 뚫는다"는 근거는 없다 — 순전히 Kaminski-Lo의
   메커니즘(레짐별 손절 "가치")에 의존한 가설이다.
2. **VAL과 OOS-Q1의 레짐 구성비는 거의 같다**(VAL chop 49.4% vs OOS-Q1 chop 50.2%, bull/bear도
   2pp 이내). 08-15의 h48qual 블랭킷 좁히기가 "VAL은 통과, OOS는 반전"됐던 걸 "레짐 구성비가
   창마다 달라서"로 설명할 수 없다는 뜻 — 이 실험은 그 스토리와 무관하게, 메커니즘 자체의
   타당성만으로 진행했다.

## 설계

`scripts/research_eth_odyssey4_atr_tpsl_floor_regime_conditional_20260818.py`. h48qual만
대상(zig075 좁히기는 08-15에서 VAL 3/3 결정적 기각이라 재시도 안 함, multiple-testing 부담
최소화). bull·bear(추세) bar에서만 floor를 좁히고, chop bar는 라이브 floor(7.5%/4.0%) 그대로
유지 — `_apply_atr_safety_sltp`를 라이브 floor·후보 floor로 각각 한 번씩 돌려
`take_profit`/`stop_loss` 컬럼만 레짐 마스크로 스플라이스(위험사이드카 계산은 스플라이스된
`dec` 기준으로 다시 수행, 원본 `prep_component`는 미변경). 후보값은 08-15에서 이미 특성화된
p50-cross(0.0324/0.0162)·p25-cross(0.0252/0.0126)를 그대로 재사용(신규 미검증 값 없음),
p75는 08-15에서 이미 가장 약했던 후보라 제외.

## 결과 — VAL (2025-10-01~12-31)

G0 자기검증 통과(baseline PnL/MDD/거래수가 기존 라이브 baseline과 정확히 일치).

| 후보 | h48qual PnL/MDD/거래 | 포트폴리오 no_gate PnL/MDD/거래 | 포트폴리오 with_gate PnL/MDD/거래 |
|---|---|---:|---:|
| baseline | +5.45%/−11.62%/29 | +36.82%/−24.34%/29 | +54.88%/−31.11%/22 |
| h48qual_trending_p50 | +3.50%/−13.08%/43 | **+37.32%/−21.50%/35** | **+63.36%/−24.88%/26** |
| **h48qual_trending_p25** | +4.23%/−12.49%/43 | **+41.65%/−19.13%/35** | **+63.68%/−24.84%/27** |

**두 후보 다 baseline을 pnl·mdd 둘 다, no_gate·with_gate 둘 다 비악화** — 08-15의 블랭킷
좁히기보다 오히려 더 깔끔하게 VAL을 통과했다(2/2 대 2/3). 최고 후보(`h48qual_trending_p25`)를
사전등록 규율대로 단일터치 OOS 대상으로 선정.

## 결과 — OOS (2026-01-01~03-31, 단일 터치) — 다시 반전

| 후보 | no_gate PnL/MDD/거래 | with_gate PnL/MDD/거래 |
|---|---:|---:|
| baseline | +49.32% / −16.20% / 24 | +44.48% / −15.48% / 20 |
| h48qual_trending_p25 | **+36.61%** / **−28.64%** / 22 | **+18.61%** / −28.64% / 17 |

**REVERSES**: PnL 악화(+49.32%→+36.61%), MDD 거의 2배 악화(−16.20%→−28.64%). **주목할 점 —
이 MDD(−28.64%)가 08-15의 블랭킷(레짐무관) 좁히기 OOS 결과의 MDD와 소수점까지 정확히
일치한다.** VAL의 MDD(−19.13%)도 08-15 블랭킷 버전과 동일하다. 즉 이 OOS/VAL 창에서 실제로
MDD를 만든 거래(들)는 우연히도 trending(bull/bear) 레짐에 걸려 있어서, "chop만 보호"하는
레짐조건화가 그 거래들에는 아무 차이를 못 만들었다 — 레짐조건화가 실질적으로 블랭킷
좁히기와 거의 동일하게 작동한 셈이다.

## 결론 — REJECTED, 문헌근거 있는 "질적으로 다른 가설"도 실패

Kaminski-Lo의 레짐-손절가치 메커니즘 자체는 문헌상 타당하지만, **이 리포의 bull/bear/chop
레짐 정의(wide24 라우터)로 조건화했을 때는 OOS 반전을 막지 못했다.** 이는 08-15가 이미
관찰한 패턴(VAL 과적합/우연 가능성)이 레짐 특이적 원인이 아니라 더 근본적인(예: 표본 크기가
작은 이 창들에서 소수 거래의 꼬리위험) 원인일 가능성을 시사한다.

**최종 판단**: floor/cap 재보정 축은 08-15의 4개 축에 더해 08-15가 직접 권고한 "질적으로
다른 가설"(레짐조건부)까지 포함해 **총 5개 축 전부 기각**됐다. 이 축은 이제 사실상 완전히
소진됐다고 본다 — 남은 이론적 변형(다른 레짐 정의, TP·SL 독립 이동 등)은 사전확률이 더욱
낮고, exit_head 재감사에서 발견된 별도 문제들(pos_unrealized/mfe/mae 스케일, 배리어
고가/저가 불일치)이 더 생산적인 다음 단계로 보인다.

**exit_head 연계 관련**: 이 실험이 실패했으므로 exit_head의 `LIVE_ATR_CFG`(라이브 floor를
그대로 미러링)를 바꿀 이유가 없다 — 현재처럼 라이브와 동일한 단일 7.5%/4.0% floor를 계속
쓰는 게 맞다. exit_head 쪽 별도 이슈(pos_unrealized/mfe/mae raw수정 재학습)는 이 결론과
무관하게 독립적으로 진행한다.

## 준수 확인

- `trading_bot_modules/omega4_6_1_live.py`/`trading_bot.py`/`runtime_config.py`/`.env` 무변경.
- 재학습 없음(냉동 예측 재사용), 저장 원장 미사용, VAL-게이트 후 단일터치 OOS 규율 준수.
- fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
  saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
- 산출물: `tmp/causal_regen_20260516/eth_odyssey4_atr_tpsl_floor_regime_conditional_20260818/`
  (`report.json`, `component_val.csv`, `portfolio_val.csv`, `component_oos.csv`, `portfolio_oos.csv`).
