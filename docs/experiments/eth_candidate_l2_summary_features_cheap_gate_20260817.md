# ETH L2 요약 컬럼 피쳐화 — cheap-gate 실험 (사전등록 + 결과, 2026-08-17)

상태: **CLOSED — Gate P 통과(편향 없음 확인), Gate B1/B2 실패. raw L2 축은 사전 선언대로 독립 유지**
상위 연구: `docs/feature_engineering_edge_research_20260817.md` §3.1, §5 로드맵 2순위
선행 종료: 로드맵 1순위 정보시간 샘플링 — `docs/experiments/eth_candidate_infotime_sampling_ab_20260817.md` (CLOSED)

---

## 1. 목적과 가설

`orderbook_decision_snapshots`(2026-05-13→, 소비자 0)의 L2 요약 컬럼이 5m~1h 지평의 ETH 방향 예측에 (a) 정보를 갖고 있고, (b) **기존 무료 벤치마크에 대한 증분**이 있는지 검정한다.

- **H-primary (사전 지정)**: `imbalance_5`(상위 5레벨 notional 임밸런스)가 h=1(5m) 전방 수익률과 유의한 양(+)의 IC를 갖는다. 근거: Gould & Bonart (2015, arXiv:1512.03492) 큐 임밸런스의 one-tick-ahead 예측력, Bieganowski & Ślepaczuk (2026, arXiv:2602.00776) 바이낸스 선물에서 임밸런스 계열의 자산 불문 SHAP 안정성, Silantyev (2019, DOI:10.1007/s42521-019-00007-w) 크립토 OFI > trade flow.
- **H-decay**: IC는 h=1 > h=3 > h=12로 감쇠한다 (Cont, Cucuringu & Zhang 2023, arXiv:2112.13213의 빠른 감쇠) — 지평별 감쇠 프로파일은 상위 연구 §4-4의 신규 게이트.
- **H-increment**: L2 요약 피쳐군은 무료 벤치마크(직전 수익률, klines taker_imbalance, microstructure_1m OBI)로 설명되지 않는 증분 예측력을 갖는다 — [[evidence_signal_quant_use_subproject]]의 "무료 벤치마크 흡수" 실패 패턴을 사전 차단.

## 2. 데이터 감사 (2026-08-17 실측)

`data/live/microstructure.duckdb::orderbook_decision_snapshots` (dev 사본, read_only):

- 12,628 rows, 2026-05-13 23:52 KST ~ **2026-08-11 21:35 KST** (dev 사본은 서버 대비 ~6일 지연 — 최종 CONFIRM 전 서버 rsync 여부를 리포트에 명시).
- **케이던스: median 300.0s, p10–p90 = 297.6–303.4s** — 가동 중에는 사실상 정규 5분 루프다. 우려했던 "decision-conditional 활동 편향"은 실측상 **다운타임 문제**로 재정의됨.
- 시간대(UTC hour) 분포 493~558 rows/h로 균일 — 시간대 편향 없음.
- 갭: >10min 606회, >1h 79회. 대형 다운타임 3회: **77.8h**(06-22~26), **101.9h**(07-02~06), **489.4h**(07-13~08-02, 20.4일). 월별 rows: 5월 4,502 / 6월 3,408 / 7월 2,119 / 8월 2,599. 5m 슬롯 커버리지 ≈ 48.7%.
- NULL: spread_bps/imbalance_20/microprice_edge_bps 모두 0.
- 공식(`trading_bot_modules/orderbook_recorder.py`): `imbalance_k = (bid_ntl_k − ask_ntl_k)/(bid_ntl_k + ask_ntl_k)`, `microprice_edge_bps = (microprice − mid)/mid × 1e4`.
- BTC/SOL 스냅샷은 2026-08-02부터만(2.6k/3.4k rows) — 교차자산 재현은 **비결정 관찰**로만.

## 3. Split (Fresh-Forward 경계 명시)

캐노니컬 VAL/OOS와 비겹침이므로 새 경계를 정의한다 (조밀 블록 기준, 시간 분리):

| Split | 기간 | 예상 n |
|---|---|---|
| DEV | 2026-05-14 ~ 2026-06-22 | ~7.6k |
| MID (robustness, 비결정) | 2026-06-26 ~ 2026-07-13 | ~2.4k |
| CONFIRM | 2026-08-02 ~ 2026-08-11 | ~2.5k |

- 검출한계 명시: n≈2.5k에서 h=1 IC의 95% 유의 문턱 ≈ 1.96/√n ≈ **0.039**. CONFIRM은 8월 초 레짐에 국한됨(20일 공백의 구조적 결과) — 한계로 기록.
- h>1 지평은 라벨 중첩으로 유효 n이 줄어든다 — IC 신뢰구간은 1일 블록 부트스트랩으로 계산(중첩 자기상관 보정).

## 4. 피쳐 후보 (12개, 사전등록)

스냅샷 저장 컬럼과 그 1차 변환만 사용 (raw L2 레벨은 이 실험 범위 밖):

| # | 피쳐 | 정의 |
|---|---|---|
| 1–4 | `imbalance_1/5/10/20` | 저장값 |
| 5 | `imb_slope` | `imbalance_1 − imbalance_20` (근접-심층 압력 기울기) |
| 6 | `microprice_edge_bps` | 저장값 |
| 7 | `spread_bps` | 저장값 |
| 8 | `spread_bps_z` | trailing 288 스냅샷 z |
| 9 | `log_depth20_z` | log(bid_ntl_20+ask_ntl_20)의 trailing 288 z |
| 10 | `ofi_proxy_5` | (Δbid_ntl_5 − Δask_ntl_5) / trailing 288 mean(bid_ntl_5+ask_ntl_5) |
| 11 | `d_imbalance_5` | Δ`imbalance_5` |
| 12 | `d_log_depth20` | Δlog(총 depth 20) |

- Δ(delta) 피쳐는 **직전 스냅샷 간격 ≤ 310s일 때만** 계산, 아니면 NaN(갭 관통 delta 금지). NaN은 drop, zero-fill 금지.
- trailing 통계는 available 스냅샷 기준 causal rolling.
- 한계 명시: 5m 스냅샷 간 delta는 진짜 OFI(초 단위 이벤트 합산)의 매우 거친 프록시다. **이 실험이 음성이어도 raw L2 OFI 축(09-14~)을 닫지 않는다** — 해상도가 질적으로 다름.

## 5. 라벨·조인·벤치마크

- 가격/라벨: Binance fapi 공개 5m klines(ETHUSDT, 2026-05-13~08-12) → `fwd_ret_h = log(close[t+h]/close[t])`, h ∈ {1, 3, 12}.
- 조인: 스냅샷 → 다음 5m 봉 경계에 backward as-of, **age ≤ 300s** (넘으면 그 슬롯 미커버 처리). bfill 금지.
- 벤치마크 3종(무료/기존): `lag1_ret`(직전 5m 수익률 — 단기 리버설), `taker_imbalance`(klines의 (2·taker_buy_quote−quote_volume)/quote_volume — 102셋 계열), `obi_1m`(`microstructure_1m.obi`, `data_stale=false` 필터, as-of ≤ 90s).

## 6. 게이트 (사전등록)

| 게이트 | 기준 | 역할 |
|---|---|---|
| **P (품질/오염)** | (a) 전 피쳐 spearman(feature, mid) < 0.5 ([[feedback_raw_feature_price_trend_contamination]]); 위반 피쳐는 detrend 후만 잔류. (b) 커버리지 편향 감사: 커버 슬롯 vs 미커버 슬롯의 \|ret\| 분포 비교 리포트(다운타임-변동성 상관 정량화) | 해석 전제 |
| **B1 (존재)** | primary `imbalance_5` h=1: DEV와 CONFIRM에서 IC 부호 일치 **그리고** CONFIRM \|IC\| ≥ 0.039(검출한계) | 정보 존재 |
| **B2 (증분)** | ridge(벤치마크 3종) vs ridge(벤치마크+12피쳐): CONFIRM 예측-실현 스피어만 상관 증가 Δρ의 1일 블록 부트스트랩(고정 seed 20260817, 2,000회) 95% CI 하한 > 0, h=1 | 무료 벤치마크 흡수 차단 |
| **C-lite (관찰, 비결정)** | 12피쳐 ridge composite의 상/하위 decile 롱숏 거래당 gross bp vs 왕복 11bp | 경제성 위치 확인 |

- 전 피쳐 × 전 지평(12×3) IC 매트릭스와 감쇠 프로파일은 전부 리포트하되 **결정은 B1(1셀)·B2(1셀)에서만** (다중비교 통제).
- **B1·B2 동시 통과** → Phase 2: raw L2 정식 OFI 라이브러리 설계에 요약-컬럼 결과를 prior로 반영 + 지평 감쇠에 따라 주입 레이어(실행/청산 타이밍 우선) 결정.
- **실패** → L2 "요약 컬럼" 축 종료(registry 등록). raw L2 축은 독립 유지(§4 한계 명시대로).
- MID 블록·BTC/SOL은 robustness 관찰로만 기록.

## 7. 함정 (사전 명시)

1. **다운타임-변동성 상관**: 대형 갭에 2026-08-09 크래시루프 시기가 인접([[eth_trading_bot_deprecated_alpha7_v31_crash_risk_20260816]]) — 봇이 죽는 구간이 고변동일 수 있음. Gate P(b)가 정량화하고, 유의하면 "저변동 조건부 결과"로 해석 강등.
2. **CONFIRM 레짐 국한**: 8월 초 9일. 통과해도 "레짐 1개 확인"으로만 기술.
3. **5m 스냅샷의 OFI 과소평가**: 문헌 OFI 지평은 초~분 — h=1(5m)에서도 감쇠 후 잔존분만 측정됨. 음성 결과의 해석 한계로 명시(raw L2 축 유지 근거).
4. 모델 시드: ridge는 결정적 — 시드 이슈 없음. 부트스트랩 seed 고정(재현성).
5. mutual_info 금지, IC/CI만 사용.

## 8. 실행 계획

- 스크립트: `scripts/research_eth_l2_summary_features_cheap_gate_20260817.py` (quant_ai env — duckdb 필요, DB는 read_only 접속).
- 산출: `tmp/causal_regen_20260516/eth_l2_summary_features_20260817/summary.json` + 본 문서 결과 섹션.
- 예상 자원: 12.6k rows, ridge/부트스트랩 — CPU 1~2분, dev에서 실행.

---

## 9. 결과 (2026-08-17 실행, `tmp/causal_regen_20260516/eth_l2_summary_features_20260817/summary.json`)

조인 후 n=12,563 (DEV 6,875 / MID 3,048 / CONFIRM 2,569).

### Gate P — 품질/오염: 통과 (해석 전제 성립)

- (a) 오염: `spread_bps` raw만 |spearman(mid)|>0.5 위반 → z-변형(`spread_bps_z`)만 유효 취급. 나머지 11개 피쳐 청정.
- (b) **커버리지 편향 없음**: 커버 슬롯 |fwd_ret_1| 평균 10.14bp vs 미커버 10.16bp (p90 22.9 vs 22.6bp) — 다운타임은 변동성과 무상관. 결과를 "저변동 조건부"로 강등할 필요 없음.

### Gate B1 — primary `imbalance_5` h=1: 실패

| Split | IC |
|---|---|
| DEV | **−0.0081** |
| MID | +0.0158 |
| CONFIRM | +0.0204 |

부호 불일치(DEV 음 → CONFIRM 양) + CONFIRM |IC| < 검출한계 0.039. h3/h12도 +0.017/+0.029로 미달 — 감쇠 프로파일 이전에 존재 자체가 미확인.

### Gate B2 — 증분: 실패 (오히려 악화)

ridge(벤치마크 3종) CONFIRM ρ=+0.0408 vs ridge(+12피쳐) ρ=+0.0130 → **Δρ = −0.0277**, 1일 블록 부트스트랩 95% CI [−0.060, +0.001]. L2 요약 피쳐 추가는 무료 벤치마크 대비 예측을 **해쳤다**. 유일하게 세 블록 모두 부호 안정인 신호는 벤치마크인 `lag1_ret`(단기 리버설: −0.038/−0.076/−0.039) — [[evidence_signal_quant_use_subproject]]의 "무료 벤치마크만 살아남는" 패턴 정확히 재현.

### C-lite (비결정)

composite 상/하위 decile 롱숏 gross +0.41bp/거래 vs 왕복 11bp — 경제성 없음.

### 비결정 부수 관찰

- `spread_bps_z` CONFIRM IC +0.071(검출한계 상회)이나 DEV 0.000/MID +0.033으로 블록 간 불안정 — 방향 신호가 아니라 유동성/변동성 상태 변수 성격. 리스크 레이어 후보로만 기록.
- `ofi_proxy_5` MID +0.045/CONFIRM +0.031이나 DEV −0.003 — 동일 패턴.

## 10. 판정 (사전등록 §6 그대로)

**B1·B2 실패 → L2 "요약 컬럼" 축 종료.** 5m 스냅샷 해상도의 임밸런스/깊이 요약은 방향 정보를 제공하지 못하며, 무료 벤치마크에 대한 증분도 음수다. 사전 선언(§4, §7-3)대로 **raw L2 OFI 축(WS-E E1, exploratory 09-14)은 독립 유지** — 문헌상 OFI 지평(초~분)은 5m 스냅샷 간 delta로는 구조적으로 측정 불가이므로, 이 결과는 raw L2에 대한 반증이 아니다. 단 이 결과가 주는 prior는 명확하다: raw L2 실험도 **h≤1분 지평 + 실행/청산 타이밍 레이어**로 설계해야 하며, 5m 진입 게이트 지평에서는 기대치를 낮춰야 한다.
