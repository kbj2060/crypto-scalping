# ETH 추세추종 기술적 셋업 10종 raw rule 리프트 사전점검 (2026-08-31)

## 배경

사용자가 딥러닝 이전 단계로, 고전적 기술적분석 추세추종 셋업 10종(S급 4 + A급 6)의 규칙 기반
로직을 구현하고 raw lift 사전점검까지 진행해달라고 요청:

- **S급**: ① MTF Trend+EMA Pullback ② VWAP Pullback+Price Structure
  ③ Breakout+Volume+OI/CVD ④ BOS+Retest+Volume/CVD
- **A급**: ⑤ EMA Ribbon ⑥ Donchian Breakout ⑦ ADX+DMI+EMA ⑧ VWAP+Volume Profile
  ⑨ Bollinger Squeeze Breakout ⑩ Supertrend+EMA

착수 전 확인한 이 저장소의 관련 선행 결론(둘 다 CLOSED):
- [[eth_trend_continuation_at_evidence_signal_fires_20260831]] — "증거신호 발동 후 추세
  지속" 축. 학습헤드 AUC 0.49~0.55 전멸, 규칙 기반(추세 따라가기+ATR 트레일링)은 통계적
  엣지는 있었으나 물리적 집행 불가능한 폭(스프레드보다 좁은 트레일)에서만 생존해 종결.
- [[eth_breakout_continuation_rejected_20260831]] — "돌파 지속" 축. raw lift 1.04~1.08x
  (무의미), TabPFN도 VAL/OOS AUC 0.54/0.47(무작위 이하)로 완전 실패. "새 특화감지기에서
  돌파류 재검토 금지"로 명시적 종결.

이번 10종은 위 두 축과 **다른 질문**(독립적인 추세추종 기술적 셋업 자체의 raw lift, 반전형
증거신호 발동에 조건화되지 않음)이라 재검토 금지 대상은 아니지만, ③(Breakout+Volume+OI/CVD)과
⑥(Donchian Breakout)은 개념적으로 "돌파 지속"과 겹친다 — 그래서 ⑥을 필터 없는 순수 버전,
③을 거래량+OI/CVD 확인필터가 추가된 버전으로 나란히 설계해, "필터를 추가하면 이미 죽은 돌파가
살아나는가"를 직접 검증하는 대조군으로 삼았다.

## 방법론

코드: `scripts/research_eth_trend_signal_raw_lift_check_20260831.py`.
결과: `tmp/eth_trend_signal_raw_lift_check_20260831/scorecard.csv` (120행).

- **데이터**: `binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv`(OHLCV+taker_buy_base,
  2023-12-31~2026-08-28, gap-free) + `data/TOTAL_ETHUSDT_metrics_2024_2026.csv`(실제
  OI `sum_open_interest`, 2024-01~현재 커버)를 `merge_asof(direction="backward")`로 결합.
  OI가 VAL/OOS 구간을 커버하므로 ③/④에서 OI를 CVD로 대체하지 않고 실제 OI를 그대로 썼다.
- **윈도우**: VAL 2025-09-01~2025-12-31, OOS 2026-01-01~2026-03-31 (CLAUDE.md Fresh-Forward
  기본 윈도우 그대로 — 최신 klines를 쓰므로 후보풀 스크립트처럼 데이터 부족으로 자를 필요 없음).
  단, 이 사전점검은 **벡터화된 forward-return 스크리닝**이며 bar-by-bar 인과적 워크포워드
  시뮬레이션이 아니다 — Fresh-Forward 규칙이 promotion에 요구하는 검증이 아니라
  "research/dev score" 등급의 1차 스크리닝이다(다른 Homer 후보풀 raw-lift-check와 동급).
- **비-동어반복 베이스라인**(`research_eth_breakout_continuation_giveback_check_20260831.py`
  검증된 패턴 그대로): 트리거 유무와 무관하게 "같은 방향 공식"을 윈도우 내 **모든 봉**에
  적용한 것이 베이스라인. 트리거 쪽과 베이스라인 쪽이 다른 규칙을 쓰면 breakout v1의 20배
  버그가 재현된다 — 여기서는 두 population 모두 정확히 같은 `side_dir*(close[i+H]-close[i])
  /atr14_prior[i] >= 0.5` 공식을 쓴다.
- **ATR 자기포함 버그 회피**: 정방향 이동 정규화 분모는 `atr14_prior`(=`atr14.shift(1)`,
  트리거 봉 자신의 true range 미포함)만 사용. 지표 내부(Keltner/Supertrend 밴드)에는 현재봉
  atr14를 그대로 쓴다(밴드는 "그 봉 시점의 상태"를 서술하는 게 맞음 — 트리거 자체가 해당 봉
  종가에 확정되는 것과 동일한 컨벤션).
- **클러스터 디듑**: `first_bar_of_each_run`(breakout_continuation 계열에서 그대로 이식) —
  연속 발동봉을 독립 이벤트 1개로 축소.
- **호라이즌**: 12/48/96봉 = 1h/4h/8h (K_HORIZONS 컨벤션과 동일).
- **hit 정의**: `HIT_THRESHOLD_ATR=0.5`(자유선택, 다음 단계 재검증 없이 승계 금지) — 예측
  방향으로 0.5×ATR 이상 이동하면 hit. `lift = 트리거_hit_rate / 베이스라인_hit_rate`.
- 지표는 이 저장소 기존 컨벤션값 그대로 재사용: Donchian(96)/swing(48)(`analyze_eth_broad_
  evidence_signal_sweep_20260814.py`), Bollinger(20)+폭 864봉 percentile+squeeze, Keltner
  (20,1.5×ATR), ADX 임계 25(업계 표준). 신규 자유선택: RETEST_WINDOW=12/RETEST_ATR_BAND=0.25
  (④), VOL_Z_THRESHOLD=1.0, POC 재계산 간격 12봉/윈도우 288봉(⑧), Supertrend
  ATR_WINDOW=10/MULT=3.0(이 저장소 최초 도입, 업계 표준값).

### 발견 및 수정한 버그: Supertrend NaN 워밍업 전파

1차 실행에서 `10_supertrend_ema`가 VAL/OOS 전 구간에서 n_triggers=0으로 나왔다. ATR(14)
워밍업 구간(첫 13봉)에서 `final_upper/final_lower`가 NaN으로 초기화된 뒤, 래칫 로직의
`ub[i] < final_upper[i-1]` 비교가 NaN과 비교되면 항상 False로 평가돼 **NaN이 전체 시계열
끝까지 영구 전파**되고, 그 결과 `direction`이 최초값 1에서 단 한 번도 안 바뀌는 상태였다
(실측: 20,000봉 서브셋에서 flip=1회, 이후 507회여야 할 것이 0회). 원인은 "ATR이 유효해진
직후 첫 봉"을 워밍업 리셋이 아니라 정상 래칫 단계로 취급한 것 — `final_upper[i-1]`이 아직
NaN인지 별도로 체크하도록 수정(`i==0 or not np.isfinite(atr_np[i]) or not np.isfinite(
final_upper[i-1])` 분기 추가)한 뒤 20,000봉 서브셋에서 flip 507회(정상 범위, ~1.25%
flip rate)로 재확인, 전체 재실행 결과 n_triggers 190~208(윈도우별)로 정상화됨. 코드에
반영 완료(`_supertrend_dir`).

## 결과 요약

전체 스코어카드: signal × side(long/short) × window(VAL/OOS) × horizon(1h/4h/8h) = 120행.
"cells_above_1" = 12칸(2side×2window×3horizon) 중 baseline보다 hit-rate가 높았던 칸 수.

| 순위 | 신호 | cells_above_1/12 | mean lift | median lift | n(총 트리거) |
|---|---|---|---|---|---|
| 1 | ③ Breakout+Volume+OI/CVD | 6 | 1.008 | 0.999 | 1,869 |
| 1 | ① MTF Trend+EMA Pullback | 6 | 0.969 | 0.986 | 7,494 |
| 3 | ④ BOS+Retest+Volume/CVD | 4 | 0.937 | 0.907 | 2,052 |
| 4 | ② VWAP Pullback+Structure | 3 | 0.940 | 0.961 | 2,424 |
| 4 | ⑧ VWAP+Volume Profile | 3 | 0.918 | 0.913 | 1,341 |
| 6 | ⑤ EMA Ribbon | 2 | 0.942 | 0.944 | 6,963 |
| 6 | ⑨ Bollinger Squeeze Breakout | 2 | 0.972 | 0.949 | 1,770 |
| 6 | ⑦ ADX+DMI+EMA | 2 | 0.961 | 0.955 | 12,999 |
| 6 | ⑩ Supertrend+EMA | 2 | 0.938 | 0.915 | 2,877 |
| 10 | ⑥ Donchian Breakout(순수) | 1 | 0.928 | 0.935 | 4,023 |

**VAL과 OOS 양쪽에서 동시에 lift>1.0인 (신호,side) 조합은 H12_1h 기준 20개 중 단 1개
(③ long: VAL 1.003x / OOS 1.059x)뿐** — 나머지 19개는 최소 한쪽 윈도우에서 baseline
이하이거나, 부호 자체가 VAL/OOS 사이에서 뒤집힌다.

### ⑥ vs ③ 대조군: 확인필터가 죽은 돌파를 살리는가

⑥(순수 Donchian, 확인필터 없음)은 12칸 중 1칸만 baseline 상회 — 이미 REJECTED된
breakout_continuation 축과 정확히 같은 결론을 재확인한다. ③(같은 돌파 개념 + 거래량 z-score
+ CVD + OI 확인필터)은 6칸으로 늘지만, 유일하게 VAL·OOS 둘 다 양수인 셀(long)조차
**VAL 1.00x(사실상 baseline과 동일, 엣지 없음)**이고 OOS도 1.06~1.15x 수준의 약한 우위에
불과하다. → **확인필터가 방향은 맞게(순수 돌파보다 낫게) 개선하지만, "무의미하지 않은 수준"에
도달하지는 못한다.** breakout_continuation의 최종 판정("1.04~1.08x=사실상 무의미")과 같은
등급.

### 왜곡분포(mean 양수/median 음수) — 거의 모든 신호에서 반복

breakout_continuation 감사에서 발견된 패턴("평균은 0 근처인데 중앙값은 뚜렷하게 음수 —
소수의 폭발적 케이스가 평균을 끌어올릴 뿐")이 이번 10종 스코어카드 120행 중 **44행**에서
그대로 재현된다(mean_atr_move>0인데 median_atr_move<0). 즉 hit-rate 기준 lift가 1.0 근처로
"나쁘지 않아 보이는" 셀들도, 실제로 각 트리거를 취했을 때 **전형적인(median) 결과는 마이너스**인
경우가 태반이다. hit-rate lift 하나만 보고 판단하면 안 되는 이유가 이번에도 확인됨.

### ① MTF Trend+EMA Pullback short — VAL/OOS 부호 반전

OOS만 보면 3개 호라이즌 전부 lift>1.0(1.03/1.03/1.04x)에 mean_bp_move도 +5.6~+27.1bp로
꽤 그럴듯해 보이지만, **VAL에서는 동일 조합이 3개 호라이즌 전부 lift<0.9(0.89/0.84/0.86x),
mean_bp_move -2.4~-15.6bp로 정반대**다. 이 저장소가 "한쪽 윈도우만 보고 판단하지 말 것"의
근거로 반복 사용해온 불안정성 패턴(zigzag 하이브리드 앵커 REJECTED 사례와 동일 형태)과
일치 — OOS 단독 결과만으로 이 조합을 채택하면 안 된다.

## 해석

10종 전부가 이 저장소의 지배적 사전정보(모멘텀/추세추종형 방향 베팅은 이 자산/시간대에서
거의 항상 실패, 평균회귀형만 반복 성공 — [[eth_trend_continuation_at_evidence_signal_fires_
20260831]], [[eth_breakout_continuation_rejected_20260831]] 및 그 안에 인용된 과거
"144-bar 모멘텀 규칙" always_short baseline 동일 사례)를 **세 번째로 독립 재확인**한다.
raw lift 사전점검 단계에서 이미 20개 중 19개 조합이 탈락하고, 유일한 생존 후보(③ long)조차
VAL에서는 baseline과 구분 안 되는 수준이라 이 프로젝트가 다음 단계(라벨설계+TabPFN)로
진행시켜온 기존 통과 사례들(칼만 2.16~2.36x, DeMarker 1.89~2.12x 등, 전부 top/bottom **양쪽**
윈도우에서 견고)과는 수준 차이가 크다.

## 결론 및 권고

**10종 전부 raw lift 게이트 통과 실패.** 어느 것도 라벨설계/TabPFN 단계로 진행할 근거가 되는
수준의 견고한 엣지를 보이지 않는다:

- ⑥ Donchian(순수 돌파)은 breakout_continuation과 사실상 동일 결론 재확인 — 별도 진행 불필요.
- ③ Breakout+Volume+OI/CVD는 10종 중 가장 낫지만 VAL에서 엣지가 없어(1.00x) OOS 단독
  1.06~1.15x를 신뢰하기 어렵고, 이미 TabPFN까지 실패한 breakout_continuation과 개념적으로
  가까워 같은 결과가 재현될 개연성이 높다.
- ① MTF short는 OOS만 보면 매력적이지만 VAL에서 부호가 반전돼 채택 근거 없음.
- 나머지 6종(②④⑤⑦⑧⑨⑩)은 12칸 중 절반 이하만 baseline을 넘고 대부분 median이 음수라
  약함조차 아니고 사실상 무신호.

이 저장소의 evidence-signal 파이프라인 우선순위(재사용 방법론 템플릿 기준)에 비춰, 이번
10종에 대해 라벨빌더/TabPFN cheap_gate로 넘어가는 것은 권고하지 않는다. 사용자가 특정
신호(예: ③ 또는 ①short)를 그래도 더 파고들고 싶다면, breakout_continuation 선례가 요구했던
수준(giveback 방식 재라벨 + TabPFN cheap_gate)까지 가야 최종 판단이 가능하지만, VAL/OOS
불일치 및 median 음수 패턴을 볼 때 통과 가능성은 낮다고 본다.

## 자유선택 파라미터 (다음 단계 재검증 없이 승계 금지)

`HIT_THRESHOLD_ATR=0.5`, `RETEST_WINDOW=12`/`RETEST_ATR_BAND=0.25`(④),
`VOL_Z_THRESHOLD=1.0`(③④), POC 재계산 간격=12봉/윈도우=288봉(⑧),
`SUPERTREND_ATR_WINDOW=10`/`MULT=3.0`(⑩, 이 저장소 최초 사용 — 업계 표준값이라 상대적으로
안전하나 그리드 검증은 안 됨). 기존 저장소 컨벤션값(Donchian 96, swing 48, Bollinger 20/864,
Keltner 20/1.5, ADX 25)은 자유선택이 아니라 재사용.
