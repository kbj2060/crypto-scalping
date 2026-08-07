# 실시간 DuckDB 데이터 활용 설계 (2026-07-19)

목적: `data/live/*.duckdb`에 실시간으로 쌓이는 마이크로스트럭처/호가/청산/의사결정 데이터를
"어디에, 어떤 순서로, 어떤 검증 기준으로" 활용할지 정의한다.
Hugging Face Papers·arXiv의 최신 LOB/실행/리스크 연구를 근거로 하되,
이 프로젝트에서 이미 검증(및 기각)된 결과를 전제로 한다.

> 워크스트림별 상세 테스트 설계도: [`test_designs_duckdb_live_20260719/`](test_designs_duckdb_live_20260719/README.md)

---

## 0. 전제 — 이미 확정된 프로젝트 결론

설계는 아래 결론과 모순되면 안 된다.

1. **1m 단독 알파는 4회 기각됨.** 컨트래리언 플로우 신호는 실재(t=-7.6)하지만 0.3~2bps로
   비용(4~9bps)을 못 넘는다. BTC 병합 lookahead 수정 후 1m HGB 엣지는 완전히 소멸했다.
   → 이 데이터로 "새 1m 진입 모델"을 또 만드는 것은 설계에서 제외한다.
2. **저주파가 이긴다.** Sigma6(1h), Omega4.6.1(5m)이 실제 후보/라이브 스택이다.
   실시간 데이터의 가치는 *새 알파*가 아니라 **실행(execution)·리스크·비용·검증**에 있다.
3. **MicroExec v1.5** maker-placement는 naive-join +1.3bps/side(t=61)로 인증됐으나
   alpha14 라우터가 이미 라이브에서 수행 중이고, 신규 가치는 adaptive increment
   +0.086bps(t=2.48)뿐. shadow 전용, 재평가 ~2026-10.
4. **Fresh-Forward 규칙**: 저장 원장 replay는 diagnostic 전용. 모든 승격 근거는
   bar-by-bar causal walk-forward여야 한다.

---

## 1. 현재 데이터 인벤토리 (2026-07-19 기준 실측)

### data/live/microstructure.duckdb
| 테이블 | 행수 | 기간 | 내용 |
|---|---|---|---|
| `microstructure_1m` (ETH) | 95,940 | 2026-05-03 → 현재 (~77일) | OBI, taker_buy_ratio, spoofing, NIF(whale/retail), EAI, OI delta, funding, toxicity/queue-collapse/absorption 등 34컬럼 1분 집계 |
| `microstructure_1m_btc` / `_sol` | 각 ~6,000 | 2026-07-14 → 현재 (~4일) | 동일 스키마 |
| `orderbook_decision_snapshots` (ETH) | 11,626 | 2026-05-13 → 현재 | 의사결정 시점 L2 요약(20레벨): best bid/ask, microprice, depth 1/5/10/20 imbalance. **원시 레벨 배열은 저장 안 됨** |
| `orderbook_decision_snapshots_btc` / `_sol` | 각 1,266 | 2026-07-14 → 현재 | 동일 |
| `decision_feature_frame*` | 111 / 1,418 / 2 | ~현재 | 라이브 의사결정 시점 피처 프레임 243~299컬럼 (2026-07-02 스키마 버그 수정 후 정상 기록) |
| `microstructure_features_v1` / `model_ready_v1` | 95,940 / 95,856 | 동상 | 파생 뷰 |

### data/live/tail_risk.duckdb
| 테이블 | 행수 | 기간 | 내용 |
|---|---|---|---|
| `tail_risk_1m` (+features/model_ready) | 96,013 | 2026-05-03 → 현재 | 청산 스트림: long/short 청산 USD, mu/sigma, aftershock_prob, decay_half_life, risk_bucket |

### 섀도우 봇 DB (btc/sol/eth micro_scalp*_shadow.duckdb)
`decisions`(508~819행) + `shadow_pnl`(2,028~3,272행) + `observer_metadata` — 정책별 가상 성과 원장.

### 구조적 한계 (설계가 반드시 반영해야 함)
- **원시 L2 레벨/체결 테이프가 없다.** 요약 통계만 저장 → DeepLOB/TLOB류 raw-LOB 모델은 현재 데이터로 불가능.
- **호가 스냅샷이 의사결정 시점에만 기록**(2초 스로틀) → 연속 시계열이 아니라 조건부 샘플. 체결확률/큐 모델 학습에 부족.
- **BTC/SOL은 4일치뿐** → ETH 외 자산은 당분간 수집만.
- 77일 = 단일 레짐일 수 있음. 이 기간에 학습한 임계값은 레짐 의존성을 가정해야 한다 (Omega4.6.1 Phase1에서 분기 단위 엣지 반전 확인된 전례).

---

## 2. 논문 조사 요약 (Hugging Face Papers / arXiv)

| 분야 | 핵심 논문 | 이 프로젝트에의 시사점 |
|---|---|---|
| Raw-LOB 딥러닝 | [DeepLOB (1808.03668)](https://arxiv.org/abs/1808.03668), [TLOB dual-attention (HF 2502.15757)](https://huggingface.co/papers/2502.15757), [LOBCAST 벤치마크 (HF 2308.01915)](https://huggingface.co/papers/2308.01915), [Deep LOB forecasting microstructural guide (2403.09267)](https://arxiv.org/abs/2403.09267) | LOBCAST의 핵심 결론: **모든 모델이 새 데이터에서 성능 급락**, TLOB 저자들도 시간에 따른 예측력 감소(-6.68 F1) 보고. 우리의 "1m 알파 없음" 결론과 일치. raw LOB 필요 → 현재 데이터로 불가, 수집부터. |
| 체결확률/생존분석 | [Deep attentive survival analysis (2306.05479, Arroyo·Cartea·Zohren)](https://arxiv.org/abs/2306.05479), [KANFormer fill probabilities (2512.05734)](https://arxiv.org/html/2512.05734), [state-dependent fill probabilities (2403.02572)](https://arxiv.org/pdf/2403.02572) | maker 주문의 체결확률·체결시간 분포를 LOB 상태로 추정 → **MicroExec 계열의 정석적 업그레이드 경로.** 큐 포지션 데이터 필요(현재 없음, 근사 가능). |
| 최적 실행 RL | [RL for trade execution with market impact (2507.06345)](https://arxiv.org/html/2507.06345v1), [DRL for crypto limit order placement (EJOR)](https://www.sciencedirect.com/science/article/abs/pii/S0377221721003854), [queue-reactive RL execution (2511.15262)](https://arxiv.org/pdf/2511.15262) | 스프레드 좁을 때 공격적으로, 플로우가 유리할 때 대기 — 우리가 이미 shadow로 인증한 maker-placement 결론과 동일 방향. RL보다 supervised 체결모델+룰이 데이터 효율적. |
| 독성/청산 리스크 | [VPIN (Easley·López de Prado·O'Hara)](https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf), [Bitcoin wild moves: order flow toxicity & jumps (2026)](https://www.sciencedirect.com/science/article/pii/S0275531925004192), [Oct 10–11 2025 crypto liquidation cascade anatomy](https://www.researchgate.net/publication/396645981) | VPIN류 독성 지표는 점프/유동성 위기의 선행지표로 재확인됨. 청산 캐스케이드는 leverage-유동성-변동성 반사 루프 → `tail_risk_1m` + toxicity가 정확히 이 입력. **진입 알파가 아니라 사이징 감쇠/베토용.** |
| LOB 생성모델/시뮬레이션 | [diffusion LOB simulation (2509.05107)](https://arxiv.org/abs/2509.05107), [end-to-end generative LOB (2309.00638)](https://arxiv.org/abs/2309.00638), [DiffLOB counterfactual (2602.03776)](https://arxiv.org/pdf/2602.03776), LOB-Bench | 체결 시뮬레이터의 미래형이지만 raw 메시지 플로우 필요. 현 단계에선 해당 없음 — 수집 파이프라인 확장의 장기 명분. |
| 금융 파운데이션 모델 | [Kronos K-line foundation model (HF 2508.02739, AAAI 2026)](https://huggingface.co/papers/2508.02739) + [HF 체크포인트](https://huggingface.co/NeoQuasar/Kronos-base) | 12B 캔들 사전학습, 5m/1h OHLCV에 바로 적용 가능. 단독 알파 기대는 낮게, **기존 스택의 피처 생성기/prior**로 실험 가치. 컨텍스트 512바 제한. |

---

## 3. 설계 — 5개 워크스트림 (우선순위순)

### WS-A. 실행 비용 모델 보정 (Cost Model Calibration) — 우선순위 1, 공수 소
**왜 먼저인가:** 프로젝트의 모든 병목이 비용 가정이다(cost1 vs cost3에 따라 Sigma3 결론이 뒤집힘).
77일의 실측 스프레드/뎁스/체결 데이터로 비용 상수를 실측 분포로 교체하는 것이 가장 확실한 ROI.

- **입력:** `orderbook_decision_snapshots`(spread_bps, depth notional), `microstructure_1m`(체결 활동),
  `binance_execution_audit.jsonl`(실제 주문 감사 로그).
- **산출물:**
  1. 시간대(KST 세션)×변동성 레짐별 spread_bps / 유효 슬리피지 분포 테이블 (p50/p90/p99).
  2. 백테스트 비용 함수 `cost(asset, hour, vol_regime)` — 고정 상수 대체안.
  3. Omega4.6.1/Sigma6의 기존 백테스트를 실측 비용 분포로 재채점한 감도 리포트.
- **검증:** 실측 분포 vs 백테스트 가정 비교만으로 완결(모델 학습 없음). 승격 게이트와 무관한 diagnostic.
- **논문 근거 불필요** — 순수 실증. 다만 결과가 WS-B/C의 비용 하한을 정의한다.

### WS-B. Maker 체결확률 모델 (Fill-Probability Survival Model) — 우선순위 2, 공수 중
**왜:** 인증된 유일한 마이크로 가치가 maker-placement(+1.3bps/side)다. Arroyo et al.·KANFormer의
생존분석 프레임은 이 룰 기반 오버레이의 정석적 상위호환이고, 진입 알파가 아니라 실행 개선이므로
"1m 알파 기각" 결론과 충돌하지 않는다.

- **입력:** `orderbook_decision_snapshots`(imbalance_1/5/10/20, microprice_edge, spread) +
  `microstructure_1m`(taker flow, queue_collapse, absorption) + 다음 1분 mark_price 이동.
- **1단계(지금 가능):** 스냅샷 시점에 "best bid에 maker 주문을 냈다면 60초 내 체결됐을까 +
  체결 시 adverse selection은 몇 bps였을까"를 라벨링(1m 테이프 근사) → 로지스틱/GBM 베이스라인.
  딥러닝은 데이터가 부족하므로 금지.
- **2단계(데이터 확장 후):** WS-E의 연속 스냅샷 + 체결 테이프가 3개월 쌓이면
  생존분석(체결시간 분포)으로 업그레이드. 이때 Arroyo(2306.05479) 구조 참고.
- **소비처:** MicroExec v1.5 adaptive increment의 대체/보강. alpha14 라우터의 placement 결정에
  체결확률 컷 추가 (p_fill 낮으면 taker 폴백 or 관망).
- **검증:** shadow-only 최소 4주, MicroExec와 동일 프로토콜(naive-join 금지, 시점별 인과 조인).
  성공 기준: adaptive increment의 +0.086bps 대비 통계적으로 유의한 개선(t>3). 실패 시 폐기.

### WS-C. 청산 캐스케이드/독성 리스크 오버레이 — 우선순위 3, 공수 중
**왜:** `tail_risk_1m` 96k행 + toxicity 지표는 이미 쌓여 있고, VPIN/2025-10 캐스케이드 연구가
"독성·청산 불균형은 점프의 선행지표"임을 재확인한다. 라이브 5m/1h 모델(Omega4.6.1, Sigma6)은
청산 스트림 피처를 전혀 안 보므로 **외부 정보**다 — 1m 모델의 regime veto가 실패했던
"모델이 이미 아는 정보의 중복 베토" 문제가 여기엔 없다.

- **입력:** `tail_risk_1m`(aftershock_prob, liq imbalance, decay_half_life) +
  `microstructure_1m`(shadow_toxicity_score, queue_collapse).
- **형태:** 진입 베토가 아니라 **사이징 감쇠**부터. tail-risk 상태 상위 p% 구간에서
  margin_fraction을 α배(예: 0.5) 감쇠. Futures Risk Sizing Contract 준수
  (notional = margin_fraction × leverage, TP/SL은 price-move 기준 유지).
- **주의 — 통계 검정력:** Omega4.6.1은 6개월 24 트레이드라 오버레이 효과를 트레이드 PnL로
  검증 불가. 검증 지표는 **조건부 수익률 분포**로: "tail-risk 상위 상태 직후 N분간
  포지션 방향 수익률이 통계적으로 나쁜가"를 77일 1m 데이터 전체에서 검정(t, 부트스트랩).
  조건부 분포 악화가 확인될 때만 오버레이 설계로 진행.
- **기존 자산 활용:** `tail_risk_interceptor.py`가 이미 존재 — 신규 시스템이 아니라
  인터셉터 임계값의 데이터 기반 재보정으로 프레이밍.
- **검증:** shadow 4주 + fresh-forward 규칙 준수. 승격은 Omega 게이트 정책 그대로.

### WS-D. 라이브/백테스트 패리티 + 드리프트 모니터 — 우선순위 4, 공수 소 (운영 인프라)
**왜:** `decision_feature_frame`(243~299컬럼)은 정확히 이 용도로 만든 테이블이고,
ou_halflife/garch_vol_z 드리프트는 이미 확인된 실제 사고 유형이다.

- **주간 배치 잡:**
  1. 라이브 기록 피처 vs 오프라인 재계산 피처 패리티 체크(허용 오차 초과 컬럼 리포트).
  2. 피처별 PSI/KS 드리프트 스코어 — 학습 분포 대비 최근 7일.
  3. 섀도우 봇 `shadow_pnl` 대비 기대 성과 편차 추적(섀도우 정책별 관제).
- **산출물:** `data/live/parity_report_YYYYMMDD.json` + 대시보드 패널 1개.
- 모델 없음, 순수 SQL/pandas. 사고를 며칠이 아니라 당일 잡는 것이 목적
  (decision_feature_frame 11일 무기록, microstructure 테이블 자산 혼합 등 전례 2건).

### WS-E. 데이터 플라이휠 확장 (지금 심어야 6개월 뒤에 수확) — 우선순위 5, 공수 소
현재 스키마의 구조적 한계를 지금 고쳐야 raw-LOB 계열 연구(TLOB/생존분석 2단계/LOB 시뮬레이션)가
미래에 가능해진다.

1. **원시 L2 레벨 저장:** `orderbook_decision_snapshots`에 상위 20레벨 `bids_json`/`asks_json`
   원시 배열 추가 (요약과 병행, ~수 KB/행 — 스로틀 유지 시 용량 무해).
2. **결정 시점 조건부 샘플링 → 고정 주기 병행:** 10초 주기 연속 스냅샷 테이블 추가
   (별도 테이블, 기존 스키마 불변). 체결확률 모델의 2단계 전제조건.
3. **체결 테이프(aggTrades) 1m 집계 초과분:** 현재 count/notional만 → 방향별 상위 체결
   분포(대구간 히스토그램) 추가 저장.
4. **BTC/SOL 지속 수집 유지** — 최소 3개월 확보 전까지 모델 학습 금지.
5. 월 1회 `VACUUM`/파티션 정리 + backups 정책 확인 (현재 44~71MB로 문제없으나 연속
   스냅샷 추가 시 성장률 재점검).

### WS-F (선택/연구). Kronos 파인튜닝 실험 — 우선순위 최하, 명시적 연구 트랙
[Kronos](https://huggingface.co/papers/2508.02739) (AAAI 2026, HF 체크포인트 공개)를 5m/1h ETH
캔들에 zero-shot/파인튜닝으로 돌려 **기존 Omega/Sigma 피처에 예측 분포 요약(방향 확률,
분위수 폭)을 1~2 컬럼 추가**하는 실험. 단독 알파 주장 금지, 기존 스택의 피처 기여도로만 평가.
LOBCAST/TLOB의 일반화 실패 결론을 감안해 기대치는 낮게. Fresh-forward 규칙 전면 적용,
frozen holdout(≥2026-07-14) 침범 금지 (BTC v3 Stage 0 정책 공유).

---

## 4. 하지 말 것 (Anti-Goals)

- 이 데이터로 **새로운 1m 진입 알파 모델** 구축 (4회 기각, lookahead 수정 후 엣지 소멸 확인).
- raw 레벨 없이 DeepLOB/TLOB류 모델 학습 (입력 데이터 자체가 없음).
- 77일 단일 구간 학습 임계값의 즉시 라이브 적용 (레짐 의존성 — 분기 단위 엣지 반전 전례).
- 저장 shadow_pnl 원장의 replay를 승격 근거로 사용 (Fresh-Forward 규칙 위반).
- BTC/SOL 4일치 데이터로 어떤 모델이든 학습.

## 5. 실행 순서 요약

```
지금:    WS-E(스키마 확장, 반나절) + WS-A(비용 보정, 1~2일)
다음:    WS-D(패리티/드리프트 잡, 1일) → 상시 운영
그다음:  WS-C 1단계(조건부 분포 검정, diagnostic) → 통과 시 오버레이 설계
        WS-B 1단계(체결확률 베이스라인, shadow)
~2026-10: MicroExec 재평가와 함께 WS-B 판정
+3개월:  연속 스냅샷 누적 후 WS-B 2단계(생존분석) / BTC·SOL 편입 검토
연구:    WS-F는 위와 독립적으로 여유 시 진행
```

각 워크스트림의 승격/폐기 판단 기준은 본문에 명시된 검증 기준을 따르며,
모든 성과 주장은 Fresh-Forward 규칙과 Omega Artifact Integrity Promotion Gate를 통과해야 한다.
