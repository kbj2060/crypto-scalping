# ETH 정보시간 샘플링 A/B — cheap-gate 실험 (사전등록 + 결과, 2026-08-17)

상태: **CLOSED — Gate A 통과, Gate B/C 실패 (사전등록 종료 조건 충족)**
상위 연구: `docs/feature_engineering_edge_research_20260817.md` §3.2 (로드맵 1순위 축)
결과 요약은 §12, 판정은 §13.

---

## 1. 목적과 가설

**질문**: 같은 원천 데이터·같은 피쳐·같은 라벨 공식·같은 모델에서, 봉의 시계(clock)만 캘린더 5분 → 정보시간(달러/거래량/체결수)으로 바꾸면 OOS 방향 스킬과 비용 반영 성과가 개선되는가?

- **H1 (통계, 문헌 재현)**: 정보시간 봉의 수익률 분포는 캘린더봉보다 정규분포에 가깝다(첨도 감소) — Easley, López de Prado & O'Hara (2012), *The Volume Clock*, DOI:10.3905/jpm.2012.39.1.019; López de Prado (2018), *AFML* ch.2.
- **H2 (스킬)**: 정보시간 봉에서 학습한 방향 분류기의 OOS 스킬이 캘린더봉 대비 시드 노이즈를 초과해 개선된다.
- **H3 (경제성)**: 비용 반영 최소 전략이 free benchmark(max(always_long, always_short, flat))를 VAL과 OOS 모두에서 이긴다.
- **Null 시나리오 (명시)**: 방향 정보가 원천 데이터에 없다면(리포 진단) H1은 성립해도 H2/H3는 성립하지 않는다. 이 경우 이 축은 종료된다. **H1의 역할은 treatment 전달 검증이다** — 재샘플링이 기계적으로 작동했는데(H1 pass) 스킬이 안 움직이면(H2/H3 fail), "구현 실패"가 아니라 "정보 부재"로 결론 내릴 수 있다.

이 실험은 이 리포에서 한 번도 시도된 적 없는 유일한 변환 축이며(모든 기존 실험은 5m 캘린더봉), **캐노니컬 VAL/OOS 기간을 그대로 쓸 수 있는 유일한 신규 축**이다(데이터가 전 기간 존재).

## 2. 데이터 (확인 완료)

- 원천: `data/training_features_1m.csv` — ETHUSDT perp 1m, 2024-01-01 ~ 2026-07-12, `timestamp, open, high, low, close, volume, quote_volume, trades, taker_buy_base, taker_buy_quote` 사용 (이후 컬럼은 미사용).
- 무결성(2026-08-17 검증): 2025-01-01~2026-03-31 구간 655,090/655,200 rows (누락 110분, 갭 8개, 최대 56분), `quote_volume` NaN 0. 갭 처리 규칙은 §4.
- **두 군 모두 이 동일한 1m 스트림에서 구축한다.** 기존 5m 파이프라인 데이터를 baseline으로 쓰지 않는다(원천 차이가 교란변수가 되는 것을 차단).
- Phase 1(조건부)에서만 aggTrades 정밀 재구축: `data.binance.vision` futures/um monthly, ~650MB/월(전 기간 ~17GB) — 서버 handoff 대상.

## 3. 실험군 (arms)

| Arm | 봉 정의 | 임계 θ |
|---|---|---|
| A0 (baseline) | 캘린더 5m (1m×5 집계) | — |
| A1 (primary) | **달러봉**: cumsum(`quote_volume`) ≥ θ_d | TRAIN 총 quote_volume ÷ TRAIN 5m 봉 수 |
| A2 | 거래량봉: cumsum(`volume`) ≥ θ_v | 동일 방식 |
| A3 | 체결수봉: cumsum(`trades`) ≥ θ_t | 동일 방식 |

- **θ는 TRAIN(2025-01-01~08-31)에서만 산출한 고정 상수**로 VAL/OOS에 그대로 적용(전체 표본 산출 = 누수 금지). 평균 봉 간격이 5분과 일치하도록 맞춰 A0과 봉 수를 정렬한다.
- 감도 체크(비결정용): A1에 한해 θ×0.5, θ×2 (평균 2.5분/10분 봉) — 방향성 일관성 확인용이며 **선택에 사용하지 않는다**.
- **임밸런스/런 봉은 Phase 0 제외**: 1m 부호류 해상도로는 부적합 + EMA 기대치 기반 구성의 불안정성 문헌 보고. Phase 1(aggTrades) 조건부 편입.
- **결정은 A1(달러봉, θ 기본)에서만 내린다.** A2/A3/감도는 보조 관찰(다중비교 통제).

## 4. 봉 구성 규칙

- 1m 증분을 누적해 cumsum ≥ θ가 되는 **1m 경계에서 봉 마감**(분 내부 분할 없음 — 양자화 한계는 §10).
- 집계: open=첫 1m open, high=max, low=min, close=마지막 close, `volume/quote_volume/trades/taker_buy_*`=합.
- **봉 timestamp = 마지막 1m 봉의 close 시각** (해당 캘린더 시점에 정보 확정 — causality 앵커, 이후 모든 조인·평가의 기준).
- 데이터 갭: 누적을 계속한다(달러봉은 자연스럽게 갭을 관통). A0은 존재하는 1m만 집계, 5분 창에 1m가 하나도 없으면 봉 생략. 갭 관통 봉은 `gap_flag` 기록(진단용, 피쳐 아님).

## 5. 피쳐 컨트랙트 (양 군 동일, 18개, 사전등록)

각 arm의 자기 시계 기준 rolling, 전부 causal (현재 봉 이전 데이터만):

1. `log_return` (1봉)
2. `ret_12` (12봉 누적 로그수익률)
3. `rsi_14`
4. `macd_hist` (12/26/9)
5. `bb_width_z_288` (20봉 BB 폭의 288봉 z)
6. `parkinson_vol_20`
7. `garman_klass_vol_20`
8. `vol_z_288` (20봉 실현변동성의 288봉 z)
9. `taker_imbalance` = (2·taker_buy_quote − quote_volume)/quote_volume
10. `taker_imbalance_z_48`
11. `cvd_48` (부호 체결류 48봉 누적, quote 기준, 288봉 z)
12. `trade_intensity_z_288` (`trades`의 z)
13. `mean_reversion_z_20` (close의 20봉 z)
14. `ema_slope_48` (EMA48 pct-change)
15. `chop_index_14`
16. `wick_ratio`
17. `hour_sin`, `hour_cos` (봉 close 시각 기준 — 세션 효과 통제, [[eth_session_split_edge_2023utc_20260817]] 참고)
18. `amihud_z_288` (|ret|/quote_volume의 z)

funding/OI/BTC 크로스에셋은 Phase 0 제외(as-of 조인 복잡도를 빼고 arm 간 완전 동일성 우선). 라벨 파생·미래 참조·전표본 스케일러 금지(리포 표준). 워밍업 NaN 구간은 drop.

## 6. 라벨 (arm 자기 시계 기준, 공식 고정)

- `fwd_ret_12` = 12봉 앞 close 로그수익률 (A0 기준 약 1시간; 정보시간 arm은 캘린더 길이가 변동 — 그것이 treatment).
- 3-class: LONG if fwd_ret_12 > δ, SHORT if < −δ, else NEUTRAL. **δ = 10bp** (taker 왕복 ~9bp + 슬리피지 고려 비용인지 데드밴드).
- 라벨은 target 전용, 피쳐 입력 금지.

## 7. 모델·시드

- `sklearn.ensemble.HistGradientBoostingClassifier`, HP 고정(사전등록): `max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False`. **HP 튜닝 금지** (튜닝은 A/B 교란).
- 시드 정책([[Seed-Diversity Ensemble Promotion Gate]] 준수): N=5 진짜 랜덤 시드, 2026-08-17 `secrets.randbelow(2**31)`로 사전 추출 — **[1491474210, 163789868, 1345858477, 922652315, 1247871276]**. 모든 arm에 동일 5개 적용.
- 시드 적용 메커니즘(사전등록): HGB는 서브샘플 없이는 사실상 결정적이므로, 각 시드는 (a) TRAIN rows의 90% 무작위 서브샘플 추출과 (b) `random_state`에 적용된다. 시드 간 분산 = 데이터 재표집 분산이며, 게이트 B의 노이즈 기준선이 된다.
- Tabular 1차 패스는 리포 Model Architect 가이드라인("fast, interpretable, easy to ablate") 준수. TabM 통합은 Phase 1 이후.

## 8. 평가 프로토콜

- Split (캐노니컬, 캘린더 시각 기준): TRAIN 2025-01-01~2025-08-31 / VAL 2025-09-01~2025-12-31 / OOS 2026-01-01~2026-03-31. 경계에서 라벨 겹침 purge: 경계 전 12봉의 TRAIN 라벨 제거.
- Fresh-forward 준수: bar-by-bar causal, `fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false` 리포트 명시.
- **스킬 지표** (arm 자기 봉 기준): 3-class bacc, LONG/SHORT OvR AUC, IC = spearman(p_long − p_short, fwd_ret_12). 시드 5개 평균±std.
- **경제성 지표** (캘린더 시간 기준 — arm 간 유일하게 공정한 비교면):
  - 규칙: p_side > τ면 다음 봉 open 진입, 12봉 보유 후 청산(신호 유지 시 연장 없음, 반대 신호 시 flip), 포지션 1개, notional 1.0.
  - 비용: taker 4.5bp + 슬리피지 1bp, 진입·청산 각각 (왕복 11bp) = cost1. cost2 = ×2.
  - τ는 **VAL에서만** {0.40, 0.45, 0.50, 0.55, 0.60} 그리드로 선택, OOS는 블라인드 1회.
  - 산출: PnL, MDD, trades, trades/day, breakeven bp (거래당 평균 엣지).
  - Free benchmark: 같은 기간·같은 비용의 max(always_long, always_short, flat).

## 9. 사전등록 결정 게이트

| 게이트 | 기준 | 역할 |
|---|---|---|
| **A (treatment 검증)** | A1 봉 수익률 첨도 < A0 첨도 (TRAIN/VAL/OOS 전부), 봉당 달러량 변동계수 감소 | 재샘플링이 기계적으로 작동했는지. 실패 시 구현 버그 조사 |
| **B (스킬)** | A1 OOS IC 및 OvR AUC의 5-시드 평균이 A0 대비 +2×(pooled seed std) 초과, VAL·OOS 부호 일치 | 방향 정보 개선 여부 |
| **C (경제성)** | A1 cost1 PnL(5-시드 평균)이 VAL과 OOS 모두에서 free benchmark와 A0을 동시 상회 | 비용 생존 |

- **B와 C 동시 통과** → Phase 1 진행 (aggTrades 정밀 봉 + 임밸런스 봉 + funding as-of 조인 + TabM 파이프라인 통합 검토).
- **A 통과 + B/C 실패** → **축 종료.** "정보시간 샘플링은 treatment가 전달됐음에도 이 데이터의 방향 정보를 늘리지 못함"으로 기록, `research_line_registry.json`에 종료 등록. 재개 조건: 진짜 틱(aggTrades) 해상도에서만 가능한 임밸런스 봉이 질적으로 다른 가설을 제공할 때.
- 부수 관찰(비결정): vol 예측 개선이 관찰되면 리스크 레이어 활용으로 별도 제안(방향 실패와 독립).
- A2/A3/감도 arm이 A1과 상반되면 결과 해석에 기록하되 결정은 불변.

## 10. 알려진 함정과 한계 (사전 명시)

1. **1m 양자화**: 봉 경계가 1m 단위로만 마감 → treatment 희석 가능. 평균 5분 봉 기준 서브 해상도 5:1이라 게이트 수준에선 수용. Gate A가 희석 정도를 정량화한다(첨도 개선폭).
2. **always_short 구조 벤치마크**: 2.083:1 tp/sl 라벨비 아티팩트 전례([[h48qual_standalone_replay_invalid]]) — free benchmark에 always_short 포함으로 통제.
3. **다중비교**: 7개 구성(4 arm + 3 감도) 중 결정은 A1 1개에서만.
4. **정보시간 봉의 비용 집중**: 달러봉은 고변동 구간에 봉이 몰림 → 거래도 고변동 구간에 몰림 → 실제 슬리피지 상회 가능. trades/day 분포와 고변동 구간 거래 비중을 리포트에 명시.
5. **θ 드리프트**: 거래량 성장 시 OOS에서 평균 봉 간격이 짧아짐. 봉 간격 분포를 split별 리포트(정보이며 버그 아님).
6. **시드 노이즈**: [[tabm_hp_low_signal_pattern]] 전례 — 게이트 B의 2×std 기준이 이를 통제.
7. mutual_info 사용 금지([[feedback_forward_fill_mutual_info_degenerate]]) — 스킬 지표는 IC/AUC/bacc만.

## 11. 실행 계획

- 스크립트: `scripts/research_eth_infotime_sampling_ab_cheap_gate_20260817.py` (단일 자기완결: 봉 구축 → 피쳐 → 라벨 → HGB 5시드 → 평가 → JSON+MD 리포트).
- 산출: `tmp/causal_regen_20260516/eth_infotime_sampling_ab_20260817/` 아래 arm별 봉 CSV, summary.json, 본 문서에 결과 추가.
- 예상 자원: ~65만 1m rows → arm당 ~10만 봉, HGB 5시드×7구성 — CPU 수 분, dev 머신에서 실행 가능(서버 handoff 불필요; WSL2 불안정 고려 시 nohup+체크포인트).
- 결과 확정 시: 본 문서에 결과 섹션 추가 + 게이트 판정 + (종료 시) registry 등록 + 메모리 갱신.

---

## 12. 결과 (2026-08-17 실행, `tmp/causal_regen_20260516/eth_infotime_sampling_ab_20260817/summary.json`)

θ (TRAIN 산출): dollar 52.50M USDT, volume 19,161 ETH, tick 19,646 trades (모두 5m-등가).

### Gate A — treatment 전달 (첨도·봉당 달러량 CV)

| Arm | TRAIN kurt | VAL kurt | OOS kurt | qv_cv (TRAIN) | 중앙 봉간격 |
|---|---|---|---|---|---|
| cal5m | 23.80 | 62.66 | 23.86 | 1.459 | 5.0m |
| **dollar_1x** | **16.59** | **26.70** | **12.93** | **0.537** | 4.0m |
| volume_1x | 16.34 | 20.61 | 14.59 | 0.603 | 4–5m |
| dollar_2x | 9.43 | 14.16 | 7.27 | 0.334 | 7–9m |

세 split 전부 첨도 감소 + 봉당 달러량 변동계수 1.46→0.54. **문헌(H1) 재현, treatment 정상 전달 — PASS.**

### Gate B — 스킬 (5-시드 평균±std)

| Arm | VAL IC | OOS IC | OOS AUC_L | OOS AUC_S | OOS bacc |
|---|---|---|---|---|---|
| cal5m | +0.0333±0.0012 | +0.0068±0.0038 | 0.5207 | 0.5291 | 0.5034 |
| **dollar_1x** | **+0.0126±0.0050** | **+0.0064±0.0055** | 0.5065 | 0.5057 | 0.5018 |

결정 arm(dollar_1x)의 OOS IC 델타 = −0.0004 (개선 아님), VAL 델타 = −0.0207 (**악화**). 기준(+2×pooled std, VAL·OOS 부호 일치) 명백 미달 — **FAIL.** 보조 arm(volume +0.0114, tick +0.0110)은 소폭 높지만 2×std 미달이고 dollar_2x는 IC 음수(−0.0137) — θ 감도 비단조 = 노이즈 패턴([[eth_candidate_h48qual_max_hold_cheap_gate_20260816]] 동형).

### Gate C — 경제성 (cost1, τ VAL 선택 = 0.6)

| Arm | VAL PnL | OOS PnL | free bench VAL/OOS | OOS trades | gross 엣지/거래 |
|---|---|---|---|---|---|
| cal5m | −0.5998 | −0.5451 | +0.3888 / +0.3436 (always_short) | 658 | — |
| **dollar_1x** | **−1.4557** | **−0.7574** | 동일 | 781 | VAL −0.68bp, OOS +1.30bp |

전 arm이 free benchmark(always_short — ETH가 양 구간 하락)에 완패. gross 엣지 자체가 VAL에서 음수, OOS에서도 +1.3bp로 왕복비용 11bp에 완전히 잠김 — 리포 기존 관찰(microstructure_1m 단독 알파 0.3–2bp vs 4–9bp 비용) 재현. **FAIL.**

## 13. 판정 (사전등록 §9 그대로)

**A 통과 + B/C 실패 → 축 종료.** 재샘플링은 기계적으로 완벽히 작동했고(첨도 정규화·달러량 균질화) 그럼에도 방향 스킬과 경제성이 전혀 움직이지 않았다. 결론: **정보시간 샘플링은 이 데이터(1m OHLCV+taker flow 해상도)의 방향 정보를 늘리지 못한다** — 상위 연구의 "정보 원천 병목" 진단을 통제된 조건에서 재확인.

- 재개 조건(사전등록 유지): aggTrades 틱 해상도에서만 가능한 **임밸런스/런 봉**이 질적으로 다른 가설을 제공할 때. 단순 달러/거래량 봉의 재시도는 금지.
- 부수 관찰: vol 예측 개선 신호는 별도로 측정하지 않았고 이 결과에서 주장하지 않는다.
- 로드맵 함의: `docs/feature_engineering_edge_research_20260817.md` §5의 1순위 축 종료 → 다음 순위(L2 요약 컬럼 피쳐화, 청산 피드)로 이동.
