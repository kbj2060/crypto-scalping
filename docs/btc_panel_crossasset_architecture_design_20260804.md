# BTC New Architecture Design — Cross-Sectional Panel Model ("Rho1")

작성일: 2026-08-04
상태: **CLOSED** — Stage 0 완료, Stage 0.5(H1) NO-GO, Stage 1(H2) 분포 캘리브레이션만 약하게
성공(8/8 rolling), Stage 2 Fresh-Forward PnL 테스트 전 설정 실패로 라인 종료

## Stage 0 / 0.5 실행 결과 (2026-08-04, 같은 날 업데이트)

**Stage 0 완료**: 60개 USDT 무기한선물 심볼(유동성 상위, 2024-01-01 이전 상장) 5분봉/일별
metrics/월별 funding 전부 다운로드, 커버리지 100%(missing bar 0), `RAW_SOURCE_MANIFEST.json`에
zip 58,560개 등록. 스크립트: `scripts/build_universe_panel_symbols_20260804.py`,
`scripts/download_panel_klines_20260804.py`, `scripts/download_panel_funding_metrics_20260804.py`,
`scripts/build_panel_coverage_report_20260804.py`. 리포트:
`docs/panel_universe_coverage_report_20260804.md`.

**Stage 0.5 NO-GO**: 가격/거래량 횡단면 피처(17개, `scripts/build_btc_panel_marketstate_features_20260804.py`)
+ funding/OI 횡단면 피처(7개, `scripts/build_btc_panel_funding_oi_features_20260804.py`)를 기존
`causalfix_final` 98col(EXCLUDE_COLS 제외 후)에 추가해 기존 dense-nogate LightGBM 아키텍쳐
(`scripts/train_eval_btc_dense_nogate_quality_marketstate_20260804.py`)에 그대로 태운 결과:
- 가격/거래량만 추가: OOS mean_net이 거의 전 구간에서 악화 (예: h48qual OOS thresh=0 -0.451%→-0.486%)
- funding/OI까지 추가: top-20 중요도에 6개 진입(원래 1개)했지만 VAL+OOS 동시 양수 설정 0/9,
  n≥15 기준을 만족하는 개선 셀 없음
- 원래 기준(section 4)대로면 여기서 종료해야 하지만, **사용자가 명시적으로 Stage 1(트랜스포머
  백본) 진행을 선택** — 이유: 이번 테스트는 H1("횡단면 피처를 BTC 단일자산 모델에 컬럼으로
  추가하면 도움되는가")만 검증했고, Rho1의 핵심 가설인 H2("60개 코인을 하나의 학습셋으로
  풀링해 유효 표본을 40~60배 늘리면 도움되는가")는 컬럼 추가가 아니라 실제 패널 풀링/사전학습이
  필요해 이 테스트로는 검증되지 않음. H1 실패가 H2를 논리적으로 기각하지 않는다는 점을 사용자에게
  고지했고, 사용자는 이를 인지한 상태로 진행을 결정함.
- **주의**: 아래 Layer 1 이후 설계는 이제 "H1도 통과했다"가 아니라 "H1은 실패했고 H2는 아직
  미검증 상태에서 사용자 결정으로 진행 중"이라는 전제 위에 있음. Stage 1 결과가 나쁘면 H1 실패와
  일관된 결과(즉 이 축 전체에 신호가 없다는 재확인)로 해석해야지, "모델을 더 키워야 한다"는
  방향으로 계속 밀어붙이지 말 것 — 이는 2026-08-04 CUSUM 재설계 라인에서 이미 반복 확인된 함정
  (모델 패밀리 무관 0/9)과 같은 패턴.

## Stage 1 첫 결과 (2026-08-04, 같은 날)

60개 심볼 풀링 + 심볼 임베딩(완전한 cross-symbol attention 아님, 단순화된 MVP) 트랜스포머
백본을 from-scratch 사전학습(`scripts/train_rho1_panel_backbone_20260804.py`), BTCUSDT 전용
동일 아키텍쳐 베이스라인(`scripts/train_rho1_btconly_backbone_20260804.py`)과 EWMA/Gaussian
제로파라미터 벤치마크를 BTC 캐노니컬 OOS(2026-01-01~03-31)에서 비교
(`scripts/eval_rho1_btc_oos_20260804.py`).

**중간 사고**: 첫 학습 시도는 `taker_long_short_vol_ratio` 피처의 극단치(8.9e7, 저유동성
심볼)로 즉시 NaN 발산 — 피처/타겟에 clip 추가 후 재학습으로 해결.

**결과**: 패널 사전학습 모델이 BTC-only 동일 아키텍쳐 대비 pinball loss -0.66%, EWMA 벤치마크
대비 -2.84% 개선. block-bootstrap 95% CI [-0.000027, -0.000005] — 0을 포함하지 않아 이 단일
윈도우 안에서는 노이즈로 설명 안 되는 개선. 설계 문서 Stage 1 통과 기준을 문자 그대로는 만족.

**그러나 신뢰하지 말아야 할 이유**:
1. 효과 크기가 매우 작음(0.66%) — 트레이딩 가능한 엣지가 아니라 분포 캘리브레이션 지표일 뿐
2. 단일 VAL/OOS 스플릿 — 이 프로젝트 자체 교훈(Sigma6: VAL-selected config가 5개 롤링 윈도우
   중 1개만 통과)에 따르면 단일 스플릿 통과는 증거로 불충분. 8-window rolling stress test
   (event gate에 썼던 방법론) 없이는 이 결과를 믿을 근거가 약함
3. pinball loss 개선 ≠ PnL 개선 (TLOB 논문의 경고와 동일한 함정 가능성)
4. Layer 1 스코프 축소(심볼 임베딩, cross-symbol attention 아님)로 설계 원안보다 단순한 모델

**다음 단계 후보**: (a) 8-window rolling stress test로 이 결과의 안정성부터 확인, (b) 설계대로
Stage 2(횡단면 순위 헤드 + Fresh-Forward PnL 테스트)로 진행. 사용자 결정 대기.

## Stage 1 rolling-window stress test (2026-08-04, 같은 날, 사용자 요청)

`scripts/eval_rho1_rolling_window_20260804.py`: 학습 컷오프(2025-09-01) 제약상 event gate처럼
2025-02~2026-07 전체를 못 돌리고, 학습 이후 순수 구간(2025-09-01~2026-08-01)만으로 4개월폭/
1개월 stride 8개 윈도우 구성 (W1~W4는 VAL 구간과 일부 겹침 — 체크포인트 선택에 쓰인 구간이라
약한 낙관 편향 가능, W5~W8은 순수 OOS).

**결과: 8/8 윈도우 전부 panel이 btconly보다 우수, 8/8 전부 block-bootstrap 유의(CI가 0을 안
포함), 부호 반전 0건.** panel_vs_btconly 범위 -0.54%~-0.87% (매우 좁게 몰림). 순수 OOS만 놓고
봐도(W5-W8: -0.66%, -0.59%, -0.68%, -0.87%) 동일 패턴 — VAL 겹침 편향으로 설명되지 않음.

이건 이 프로젝트 BTC 역사에서 event gate 다음으로 두 번째로 8/8 rolling stress test를 깨끗하게
통과한 결과다. 다만:
1. 효과 크기는 여전히 작음(0.5~0.9%) — pinball loss(분포 캘리브레이션) 개선이지 PnL 개선 아님
2. 진짜 walk-forward 아님 — 매 윈도우마다 재학습한 게 아니라 고정된 모델 하나를 시간축으로
   재생한 것 (event gate 자체의 방법론적 caveat와 동일한 성격)
3. 백본은 여전히 심볼 임베딩 MVP, 완전한 cross-symbol attention 아님

**다음 단계**: 이 정도 안정성이면 Stage 2(횡단면 순위 헤드 + Fresh-Forward PnL 테스트)로 넘어갈
근거는 있음. 단, Stage 2의 실제 PnL 테스트를 통과하기 전까지는 "분포 캘리브레이션이 안정적으로
좋아졌다"이지 "돈 버는 신호를 찾았다"가 아님을 계속 구분해서 보고할 것.

## Stage 2 결과 (2026-08-04, 같은 날) — CLOSED

`scripts/build_panel_rank_labels_20260804.py`(횡단면 순위 라벨) →
`scripts/train_rho1_ranking_head_20260804.py`(순위 헤드 학습) →
`scripts/backtest_rho1_freshforward_btc_20260804.py`(Fresh-Forward bar-by-bar 백테스트,
CLAUDE.md 규칙 준수: VAL 2025-09-01~12-31, OOS 2026-01-01~03-31, 저장 원장/미래 row 미사용).

**순위 헤드 학습부터 사실상 실패**: val MSE 0.08319, 무작위 예측(상수 0.5) 기준선 0.0833과
거의 동일 — 유의미하게 학습된 게 없음. 실제 BTC 예측 score를 뽑아보니 범위가 0.49~0.56,
표준편차 0.019로 사실상 평평함 (원래 계획한 0.7/0.8/0.9 절대 임계값으로는 진입이 단 한 건도
안 나옴 — score의 실제 분포에서 백분위 기반 임계값(80/20, 90/10, 95/5)으로 바꿔서 재실행).

**Fresh-Forward 백테스트: 6개 설정(TP/SL 2종 × 순위 임계값 3종) 전부 VAL·OOS 모두 음수.**
mean_net -0.42%~-0.48%/trade로 매우 촘촘하게 몰림 — 이 세션 초반에 닫힌 CUSUM 재설계 라인의
실패 패턴(-0.33%~-0.50%/trade)과 사실상 동일한 숫자대.

**결론**: Stage 1이 보여준 "패널 사전학습이 BTC-only보다 분포 캘리브레이션이 8/8 윈도우에서
안정적으로 낫다"는 결과는 실제였지만, event gate 때와 정확히 같은 패턴으로 **캘리브레이션
개선이 방향 예측/수익화로 이어지지 않았다.** 순위 헤드는 방향 정보를 사실상 전혀 뽑아내지
못했고(예측 score가 거의 상수), 그 결과 Fresh-Forward PnL은 전 설정에서 일관되게 음수.

**Rho1 라인 CLOSED.** H1(피처 증강)도 실패, H2(패널 풀링)는 분포 캘리브레이션에서만 약하게
성공하고 실제 방향/PnL 헤드에서는 실패 — 두 가설 모두 이 프로젝트의 causal 피처셋으로는 BTC를
움직이지 못한다는 이 세션 전체의 결론과 일치. 완전한 cross-symbol attention(심볼 임베딩이 아닌
진짜 동시간 다중 심볼 attention)을 시도해볼 여지는 이론상 남아있지만, 순위 헤드의 MSE가 이미
무작위 기준선 수준이었다는 건 "더 정교한 attention이 도움될 것"이라는 가설 자체가 근거가 약함
— 이 세션에서 반복 확인된 함정(모델을 더 정교하게 만들어도 안 되는 문제는 안 됨)과 같은 모양.
대체 대상: `causalfix_final` 114-col frame 위의 단일자산 quality-classifier 계열 전부
(h48qual / zig075 / CUSUM+TB / dense-nogate / altmodel / event-gate 파생)

---

## 0. 왜 기존 아키텍쳐를 버려야 하는가 (진단)

지금까지 닫힌 BTC 라인들을 나열하면 전부 같은 형태다:

```
f( x_t^BTC ) -> y_t^BTC ,  x in causalfix_final(114 cols),  t in 271,797 bars
```

바뀐 것은 항상 `f`(LightGBM/RF/ExtraTrees/MLP/TabM/DSAC/REDQ/TQC/CQL), `y`의 기하학
(CUSUM/zigzag/DC/dense, horizon 48~864, TP/SL 1.2/0.8~2.5/1.5), 또는 게이팅 방식뿐이었다.
2026-08-04 Follow-up 3/4에서 결론이 났다: **게이팅도, 라벨 기하학도, 모델 패밀리도 결과를
바꾸지 못한다** (0/10, 0/9, OOS가 전부 -0.33% ~ -0.50%/trade에 촘촘히 몰림).

이건 "모델 용량 부족"의 신호가 아니라 **유효 표본(effective sample) 부족**의 신호다.
271k 바는 명목상 크지만 자기상관이 극심해서 독립적인 레짐 에피소드는 수백 개 수준이고,
BTC의 구조적 특성(ETH 대비 reversal wave 빈도 ~절반, ATR ~30% 낮음) 때문에 실제 학습 이벤트는
더 줄어든다. 그 표본으로 어떤 유연한 학습기를 태워도 VAL에 과적합되고 OOS에서 부호가 뒤집힌다.

탈출구는 두 가지뿐이다:
- (a) **새 원시 데이터** — 마이크로구조/청산 데이터는 2026-09~10 이후에나 축적됨 (시간 게이트)
- (b) **같은 타겟에 대해 유효 표본 자체를 늘리기** — 이 프로젝트에서 **한 번도 시도되지 않은 축**

이 설계는 (b)를 택한다. 그리고 (b)를 실행하려면 지금 당장 받을 수 있는 새 원시 데이터가 하나
있다: **유니버스 확장**. `binance_data/klines`에는 현재 BTCUSDT/ETHUSDT/SOLUSDT 3개뿐이지만,
data.binance.vision에는 40~80개 USDT-perp 심볼의 2024-01~현재 klines/funding/metrics가
**이미 존재**한다. 마이크로구조/청산과 달리 전방 수집이 필요 없다.

두 번째 축: 이 프로젝트 역사상 BTC에서 **유일하게 8/8 rolling window를 통과한 신호**는
event gate(= 큰 움직임이 온다, 크기)였고, 방향 예측(stage 2)은 실패했다. 최신 문헌도 같은 얘기다 —
방향보다 **분포/변동성**이 예측 가능하다. 새 아키텍쳐는 방향을 억지로 짜내는 대신
**예측 가능한 것(분포)을 중심에 놓고 방향은 횡단면 순위로 우회**한다.

---

## 1. 문헌 조사 요약 (2025~2026)

| 논문/모델 | 핵심 주장 | 이 설계에 반영된 부분 |
|---|---|---|
| Re(Visiting) TSFM in Finance (2511.18578) | 범용 TSFM은 금융에서 zero-shot·fine-tune 모두 저조. **금융 데이터로 from-scratch 사전학습**한 모델만 실질적 개선 | 외부 TSFM 체크포인트를 쓰지 않고 **우리 패널로 직접 사전학습** |
| FinCast (2508.19609) | 금융 전용 파운데이션 모델. **Point-Quantile Loss** + token-level sparse MoE로 비정상성/다도메인 대응 | Quantile 헤드 + (선택) 레짐별 MoE |
| TSFM for Multivariate Financial TS (2507.07296) | TTM/Chronos zero-shot이 **변동성 예측**에서는 naive를 이김 (방향은 아님) | 변동성/분포를 1차 타겟으로 |
| iTransformer (2310.06625) / Crossformer | 변수(=자산)를 토큰으로 두고 **cross-variate attention**. 채널 간 의존성을 명시적으로 모델링 | 백본의 cross-symbol attention 단계 |
| TLOB (2502.15757) | LOB 트랜스포머가 BTC에서 SOTA F1이지만, **스프레드 기준 임계값을 적용하면 수익성이 붕괴** | 지표(F1/AUC)가 아니라 **net PnL**로만 게이트 |
| Network Momentum (2501.07135), Cross-Crypto Relationship Mining (2205.00974), Multi-relational Attention for Crypto Return | 알트코인 네트워크/lead-lag 집계가 BTC 수익률 예측에 기여 | 유니버스 breadth/dispersion/lead-lag 피처 |
| Reverso (2602.17634) | long-conv + linear RNN(DeltaNet) 하이브리드가 100배 작은 크기로 트랜스포머급 성능 | 백본 대안 (파라미터 예산이 문제될 때) |

문헌의 일관된 메시지 두 개: **(1) 도메인 패널로 직접 사전학습해라, (2) 방향 분류 지표가
좋아도 비용 임계값을 넘기면 죽는다.** 둘 다 이 프로젝트가 이미 경험적으로 확인한 것과 일치한다.

---

## 2. 새 아키텍쳐: Rho1 (Cross-Sectional Panel Model)

이름 제안: **Rho1** (ρ = 횡단면 상관/패널). Omega/Sigma/Tau 계열과 구분.

```
Layer 0   Universe Panel        40~60 USDT-perp 심볼 × 5m, 2024-01~현재 (신규 원시 데이터)
   │        ├─ 패널 학습 세트 (심볼당 271k bar → ~12M row)
   │        └─ BTC용 market-state 횡단면 피처 (~15 col, 기존 114col에 없음)
   ▼
Layer 1   Backbone              Patch embedding → cross-time attention → cross-symbol attention
   │                            심볼 임베딩 공유, 패널 전체로 from-scratch 사전학습
   ▼
Layer 2   Multi-task Heads      (A) 분포/분위수 헤드  ← 1순위, 예측 가능한 것
   │                            (B) 횡단면 순위 헤드  ← 방향을 우회하는 경로
   │                            (C) 이벤트/변동성 헤드 ← 이미 8/8 통과한 신호, 보조 태스크
   ▼
Layer 3   Execution             TP/SL/사이징을 예측 분포에서 유도 (고정 ATR 배수 폐기)
```

### Layer 0 — 유니버스 패널 (신규 원시 데이터)

`scripts/download_klines_1m_20260716.py` / `download_metrics_funding_generic_20260713.py`를
심볼 리스트로 확장해 재사용. 대상: 2024-01-01 이전 상장 + 최소 유동성 기준을 만족하는
USDT-perp 40~60개 (생존 편향 방지를 위해 **상장 폐지된 심볼도 포함**, 각 심볼의 상장/폐지 구간만
유효 처리).

산출물 2가지:

1. **패널 학습 세트** — 기존 피처 빌더를 심볼별로 그대로 적용. BTC가 자기 전용 모델을 갖는 대신
   공유 모델의 한 row-group이 된다. 유효 표본이 40~60배로 늘고, "어떤 셋업이 작동하는가"라는
   전이 가능한 함수를 학습한다. 이게 이번 설계의 **핵심 변경점**이다.

2. **BTC용 market-state 피처** (기존 114col에 ETH 관련 5개 말고는 전무):
   - breadth: 유니버스 중 자기 VWAP/MA 위에 있는 심볼 비율 (여러 룩백)
   - dispersion: 횡단면 수익률 표준편차 / IQR
   - correlation regime: 평균 쌍별 상관, 상관행렬 1st eigenvalue 비중 ("market mode" 강도)
   - funding cross-section: BTC funding의 횡단면 백분위, 유니버스 funding 중앙값/왜도
   - OI cross-section: 횡단면 OI 변화 백분위, BTC OI 대비 알트 OI 회전
   - dominance: BTC 거래대금 점유율 모멘텀, alt/BTC 베타 로테이션
   - lead-lag: DTW/lagged-corr 기반 알트→BTC 집계 임펄스 (Cross-Crypto Relationship Mining 방식)

   **전부 인과적으로 계산** — 시점 t의 값은 t까지 확정된 바만 사용. 유니버스 정의 자체도 시점별로
   과거 유동성 기준으로 재구성(미래 상장 심볼이 과거 유니버스에 새면 lookahead).

### Layer 1 — 백본

Crossformer/iTransformer 계열의 2단계 어텐션:
- **Patch embedding**: 길이 L 윈도우를 패치로 분할 (PatchTST). 5m/1h/4h 멀티 해상도 입력.
- **Cross-time attention**: 심볼 내부 시간축.
- **Cross-symbol attention**: 동일 타임스탬프에서 심볼 축. lead-lag/market-mode를 수작업 피처가
  아니라 모델이 직접 학습한다.
- **심볼 임베딩** + 레짐 조건부 sparse MoE (FinCast 방식, 선택 사항 — 파라미터 예산 여유가
  있을 때만).
- 파라미터 예산이 문제면 Reverso식 long-conv + linear-RNN 하이브리드로 대체 (100배 작은 모델로
  동급 성능 주장).

**외부 사전학습 체크포인트는 쓰지 않는다** (Re-Visiting TSFM: zero-shot/fine-tune 모두 저조).
우리 패널로 from-scratch 사전학습한다.

### Layer 2 — 헤드 (우선순위 순)

**(A) 분포/분위수 헤드 — 1순위**
H바 전방 수익률의 분위수 {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}를 Point-Quantile Loss로 예측.
이게 왜 중요한가: 구조적 발견 #2(BTC ATR이 ETH보다 ~30% 낮아 고정 TP/SL 바닥이 1.5~2배 늦게
닿음)를 **직접** 공격한다. TP/SL을 고정 ATR 배수가 아니라 예측 분포에서 유도하면 그 문제가
정의상 사라진다. 평가: pinball loss / CRPS를 GARCH·EWMA 벤치마크 대비.

**(B) 횡단면 순위 헤드 — 방향 우회 경로**
BTC의 전방 수익률이 유니버스 안에서 몇 등인지를 listwise/pairwise ranking loss로 학습.
순위 타겟은 절대 수익률보다 훨씬 안정적이고 (비정상성에 강함), "BTC 롱/숏"이 아니라
"BTC가 횡단면 상위/하위 k에 들 때만 진입"으로 조건을 바꾼다. 방향 예측을 정면으로 하지 않으면서
방향성 포지션을 얻는 유일한 경로다.

**(C) 이벤트/변동성 헤드 — 보조 태스크**
이미 8/8 rolling window를 통과한 event gate 라벨을 멀티태스크 보조 손실로 붙인다.
**주의: 이걸 진입 게이트로 다시 쓰지 않는다** — 그 공간(direction/straddle/size-up/size-down)은
2026-08-04에 소진됐다. 여기서는 백본이 "큰 움직임이 온다"는 이미 검증된 신호를 표현하도록
강제하는 **표현 학습용 정규화**로만 쓴다. 실행에는 사이징에만 반영.

### Layer 3 — 실행 (Futures Risk Sizing Contract 준수)

```
margin_fraction = 모델이 예측 (leverage 직접 예측 X)
leverage        = 3 (고정)
notional        = margin_fraction * 3
tp_price_move   = 예측 상단 분위수 (q0.9 등)
sl_price_move   = 예측 하단 분위수 (q0.1 등)
take_profit     = tp_price_move * notional
stop_loss       = sl_price_move * notional
```
notional에 이미 레버리지가 포함되므로 **TP/SL 가격선에 레버리지를 다시 곱하지 않는다.**
보유 시간도 분포에서 유도(예측 분위수 도달 예상 시간), 고정 horizon 아님.

---

## 3. 닫힌 라인과의 대조 (이게 재탕이 아닌 이유)

| 닫힌 라인 | 왜 실패했나 | Rho1에서 무엇이 바뀌나 |
|---|---|---|
| CUSUM/zigzag/DC 이벤트 게이트 (0/92, 0/16, 0/16) | 게이팅 방식은 무관 | 이벤트 게이트 개념 자체를 진입 결정에서 제거 |
| dense no-gate (0/10) | 게이팅 없어도 동일 | 입력 데이터가 바뀜 (패널 + 횡단면) |
| altmodel LGBM/RF/ET/MLP (0/9) | 학습기 무관 | 학습기가 아니라 **표본 수**를 바꿈 (40~60배) |
| advanced RL (REDQ/TQC/CQL/DSAC-T) | 3 seed 전부 OOS 음수 | 강화학습 폐기, 지도 분포 예측으로 |
| event gate 수익화 4종 | 방향/스트래들/사이징 전부 실패 | 게이트를 진입이 아닌 **보조 손실**로 강등 |
| TP/SL 재보정 | 고정 바닥을 다른 고정 바닥으로 교체 | 고정 바닥 폐기, **예측 분포에서 유도** |
| 피처 완결성 3각 검증 (갭 없음) | BTC 자체 피처는 충분 | 부족한 건 BTC 피처가 아니라 **시장 전체 상태** |

**한 문장 요약**: 지금까지는 전부 "같은 입력, 다른 모델/라벨"이었다.
Rho1은 "**다른 입력(유니버스 패널), 표본을 40~60배로**"가 본질이고 모델 변경은 부수적이다.

---

## 4. 단계별 실행 계획 + 킬 크라이테리아

문헌보다 중요한 원칙: **가장 싼 반증을 먼저 한다.** Follow-up 4가 "모델 패밀리는 무관"을
증명했으므로, 새 피처가 기존 LightGBM에서도 아무것도 못 하면 트랜스포머도 못 살린다.
그래서 트랜스포머를 짓기 전에 데이터 가설만 따로 반증한다.

**Stage 0 — 유니버스 다운로드 & 패널 빌드** (~1일)
- 40~60 심볼 klines(5m) + funding + metrics, 2024-01~현재
- 검증: 심볼별 커버리지/갭 리포트, 상장·폐지 구간 마스크, P0 해시 매니페스트 등록,
  유니버스 구성의 시점별 인과성 감사 (미래 상장 심볼 누수 0건)

**Stage 0.5 — 싼 반증 (GO/NO-GO 게이트)** (~1일) ← **먼저 이거부터**
- 기존 `causalfix_final` 114col + 신규 market-state 15col을, **기존 dense-nogate LightGBM
  아키텍쳐 그대로** 태운다 (`train_eval_btc_dense_nogate_quality_20260804.py` 재사용).
- 목적: "새 모델"이 아니라 "새 데이터"만 독립 변수로 놓고 본다.
- **통과 기준**: VAL/OOS 둘 다 양수인 설정이 n≥50 표본에서 최소 1개, 그리고 114col-only
  베이스라인 대비 OOS mean_net이 유의미하게 개선.
- **불통과 시**: 횡단면 정보가 BTC에 증분 신호가 없다는 뜻 → Stage 1~3을 진행하지 않고
  2주가 아니라 **2일 만에** 이 라인을 닫는다. (Layer 1 백본만 별도 재검토)

**Stage 1 — 패널 사전학습 + 분포 헤드** (~4~5일)
- 40~60 심볼 전체 패널로 백본 from-scratch 학습, 분위수 헤드만.
- **통과 기준**: BTC 전용으로 학습한 동일 모델 대비, 그리고 GARCH/EWMA 벤치마크 대비
  BTC OOS pinball loss / CRPS 개선. (아직 PnL 아님 — 캘리브레이션 단계)
- 불통과 시: 패널 전이가 작동하지 않는 것 → 종료.

**Stage 2 — 횡단면 순위 헤드 + Fresh-Forward** (~3일)
- 순위 헤드 추가, CLAUDE.md **Fresh-Forward 규칙 그대로** 적용:
  VAL 2025-09-01~12-31, OOS 2026-01-01~03-31, bar-by-bar, 저장 원장 입력 금지.
  리포트에 `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
  `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false` 명시.
- **통과 기준**: VAL/OOS 둘 다 net 양수 + 거래 수 충분(n≥30) + bar-level MDD 허용범위.

**Stage 3 — 실행/사이징 + 승격 게이트** (~3일)
- 분포 기반 TP/SL/사이징, Futures Risk Sizing Contract 준수.
- **승격 게이트 (전부 필수)**:
  - 8-window rolling replay (event gate와 동일 방법론), 부호 뒤집힘 0
  - 시드 다양성: 앙상블/평균을 쓴다면 N≥5 **무작위 추출** 시드 (등간격 금지),
    시드 리스트 리포트 기재
  - Omega Artifact Integrity 감사 exit 0 + `promotion_pass=true`
  - 비용 스트레스 (TLOB 교훈: 스프레드/수수료 임계값 적용 후에도 살아남는지)

총 예상: Stage 0+0.5까지 2일. 여기서 죽으면 2일 손실. 전부 통과 시 ~2주.

---

## 5. 알려진 리스크 / 솔직한 우려

1. **가장 큰 변수는 아키텍쳐가 아니라 데이터다.** "새 아키텍쳐를 설계해달라"는 요청이었지만,
   닫힌 라인들의 증거(모델 패밀리 무관 0/9)는 트랜스포머 자체가 문제를 풀 확률이 낮다고 말한다.
   그래서 Layer 1을 짓기 **전에** Stage 0.5로 데이터 가설만 분리 검증하도록 설계했다.
   Stage 0.5가 통과하지 않으면 백본은 지어도 의미가 없다.
2. **유니버스 생존 편향**이 이 설계의 1급 lookahead 위험이다. 상장 폐지 심볼 제외, 또는
   미래 상장 심볼을 과거 유니버스에 포함하면 breadth/dispersion 피처가 전부 오염된다.
   Stage 0 검증 항목에 명시적으로 넣었다.
3. **패널 전이가 BTC에 안 통할 수 있다.** 알트코인은 BTC보다 변동성/reversal 빈도가 높아서
   패널이 학습한 함수가 BTC 레짐에 안 맞을 수 있다 (구조적 발견 #1과 같은 이유).
   완화책: 심볼 임베딩 + 변동성 정규화 타겟(수익률을 자기 ATR로 나눔) + 레짐 조건부 MoE.
4. **계산 비용**이 기존 대비 훨씬 크다 (12M row × 트랜스포머). Reverso식 경량 백본을
   대안으로 명시해둔 이유.
5. 이 설계는 **마이크로구조/청산 데이터를 대체하지 않는다.** 2026-09~10에 그 데이터가 쌓이면
   Layer 0에 추가 채널로 들어가는 구조라 서로 배타적이지 않다.
