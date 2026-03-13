# AI 에이전트 학습 로직 — 데이터 흐름 및 아키텍처 명세

## 1. 개요

본 시스템은 **원시 시장 데이터 → 피처 엔지니어링 → 시계열 예측 앙상블 → 강화학습 거래 에이전트**까지 이어지는 오프라인 학습 파이프라인으로 구성된다.  
예측 단계에서는 7대 파운데이션/커스텀 모델이 **Meta Router**로 동적 가중 결합되고, 행동 단계에서는 **4-Agent MoE(Mixture of Experts) + Kelly 동적 레버리지**가 레짐별로 역할을 나누어 매매를 결정한다.

---

## 2. 데이터 흐름 (End-to-End)

```
[원시 데이터]
  ETH 5m OHLCV + funding, OI, taker, top trader ratio 등
  BTC 5m OHLCV (close, volume)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  core/feature_engineering.py — FeatureEngineer.process()         │
│  • _merge_data(ETH, BTC) → merge_asof on timestamp               │
│  • _create_alpha_features (whale, OI, funding_pressure 등)        │
│  • _create_order_flow (net_taker_ratio, taker_acceleration 등)   │
│  • _create_technical (RSI, MACD, BB, HMA, ATR, VWAP 등)          │
│  • _create_advanced_volatility (Rogers-Satchell, Parkinson,      │
│    Amihud illiquidity)                                           │
│  • _create_market_structure (btc_corr_60, fvg_dist, chop)        │
│  • _create_temporal_features (hour_sin/cos, session_*, is_hour)  │
│  • _add_regime_break (volatility_z 기반)                         │
│  • add_cvp_features (CVP POC/VAH/VAL, cluster, regime)          │
│  • QuantSignalFeatures (turtle, dual_momentum, mean_reversion 등) │
│  • FundingRateMomentum (funding_roc, squeeze_risk, divergence)   │
│  • HurstExponentFeatures (hurst_12/48/288, regime_trending)      │
│  • ofi_acceleration (net_taker_ratio EWM + diff(3))              │
│  • _handle_missing (inf→nan, diff계열 0, 나머지 ffill/bfill)     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
  DataFrame with ULTIMATE_FEATURE_COLS (약 60개) + close, log_return 등
        │
        ├──────────────────────────────────┬─────────────────────────────────┐
        ▼                                  ▼                                 ▼
┌───────────────────────┐    ┌─────────────────────────────┐    ┌─────────────────────────────┐
│ train_nf_models.py    │    │ train_rl_agent.py            │    │ (실전 추론 시)               │
│                       │    │ --mode generate_csv          │    │                             │
│ data/                 │    │                              │    │ 동일 피처 DF →              │
│ training_             │    │ INPUT: training_features_5m  │    │ ensemble_router.predict()   │
│ features_5m.csv       │    │                              │    │ + MoEIQNTrader.decide()     │
└───────────┬───────────┘    └──────────────┬──────────────┘    └─────────────────────────────┘
            │                               │
            ▼                               ▼
┌───────────────────────┐    ┌─────────────────────────────┐
│ NeuralForecast 4종    │    │ RL 학습용 CSV 생성            │
│ • PatchTST (단변량)   │    │ • 5대 레짐 계산 (chop,        │
│ • iTransformer (단변) │    │   whipsaw, bull, bear,        │
│ • NHITS (다변량+7α)   │    │   normal)                     │
│ • TiDE (다변량+7α)    │    │ • Elite 13종 시그널           │
│ h=6, input_size=256   │    │ • 7모델 pred/conf 배치 추론   │
│ 저장: data/nf         │    │   (TTM, NF 4종, TimesFM,      │
└───────────────────────┘    │   Chronos)                   │
                              │ OUTPUT: rl_training_data_     │
                              │         full.csv             │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ train_rl_agent.py --mode train│
                              │ • 4개 TradingEnv (Bull, Bear, │
                              │   Support, Resistance)       │
                              │ • TransformerIQN + IQNAgent  │
                              │ • RegimeReplayBuffer (레짐별)│
                              │ 저장: best_moe_agents.pth    │
                              └─────────────────────────────┘
```

---

## 3. 단계별 상세 명세

### 3.1 피처 엔지니어링 (`core/feature_engineering.py`)

| 구분 | 내용 |
|------|------|
| **입력** | `eth_df`, `btc_df` (timestamp, OHLCV, funding, OI, taker 등) |
| **출력** | 단일 DataFrame, `ULTIMATE_FEATURE_COLS` 기준 컬럼 + 제외 컬럼 정리 |
| **핵심 상수** | `ULTIMATE_FEATURE_COLS`: 약 60개 피처 목록 (웨일/리테일, OI, 펀딩, 변동성, 기술적, CVP, 허스트, 퀀트 신호 등) |
| **제외 컬럼** | `EXCLUDE_FEATURE_COLS`: timestamp, close_time, OHLC, volume, trades 등 |
| **필수 피처** | `MUST_INCLUDE_FEATURES`: RSI, mtf_trend, bb_width_z, taker_acceleration 등 (학습/추론 시 누락 방지용) |
| **결측 처리** | diff 계열은 0, 그 외 피처는 ffill/bfill 후 `dropna(subset=feature_cols)` |

데이터 흐름 상 **모든 다운스트림(NeuralForecast, RL CSV 생성, 앙상블 추론)** 은 이 단계의 출력 스키마를 전제로 한다.

---

### 3.2 NeuralForecast 사전 학습 (`ensemble/train_nf_models.py`)

| 구분 | 내용 |
|------|------|
| **입력 CSV** | `data/training_features_5m.csv` (feature_engineering 출력과 동일 스키마 가정) |
| **사용 컬럼** | `timestamp`, `close`(→`y`), `EXOG_COLS` 7개 (session_us, hour_cos, cvp_poc_dist, cvp_volume_imbalance, fvg_dist, breakout_strength, oi_change_rate) |
| **모델 구성** | `NeuralForecast(models=[...], freq='5min')` |
| **단변량** | PatchTST, iTransformer — `h=6`, `input_size=256`, hist_exog 없음 |
| **다변량** | NHITS, TiDE — 동일 h/input_size, `hist_exog_list=EXOG_COLS` |
| **학습** | `nf.fit(df=df_nf, val_size=10000)`, HuberLoss, early_stop_patience_steps=3 |
| **저장** | `data/nf` (4개 모델 통합 저장) |

NF 학습은 **가격 + 7대 알파**만 사용하며, RL/앙상블에서 사용하는 “7대 파운데이션” 중 PatchTST, iTransformer, NHITS, TiDE의 가중치를 이 단계에서 만든다.

---

### 3.3 RL 학습용 CSV 생성 (`ensemble/train_rl_agent.py` — `generate_training_csv`)

| 구분 | 내용 |
|------|------|
| **입력** | `data/training_features_5m.csv` |
| **출력** | `data/ensemble/rl_training_data_full.csv` |
| **필수 컬럼** | `RL_REQUIRED_COLS` = timestamp, close, MODEL_PRED(7), MODEL_CONF(7), ELITE_COLS(13), ALPHA_7_COLS(7), REGIME_COLS(5), log_return |

**레짐 계산 (적응형 5대)**  
- ER(Efficiency Ratio), vol_z, net_change, mtf_trend_1h 기반  
- `regime_chop`: ER<0.20 & vol_z<-1  
- `regime_whipsaw`: ER<0.25 & vol_z>1  
- `regime_bull`: ER≥0.35 & net_change>0 & mtf_trend_1h>0  
- `regime_bear`: ER≥0.35 & net_change<0 & mtf_trend_1h<0  
- `regime_normal`: 위 네 가지가 아닌 경우  

**행 단위 연산**  
- 각 인덱스 `i`에 대해:  
  - Elite 13종: `EliteSignals.compute_all(current, prev, smf_std)`  
  - 7모델 예측/신뢰도:  
    - TTM: `close` 슬라이딩 윈도우 → TinyTimeMixer 추론 → 6스텝 방향성 `get_direction(traj)`  
    - NF 4종: 256봉 윈도우, `close`+ALPHA_7 → NeuralForecast.predict → 동일하게 방향성 추출  
    - TimesFM, Chronos: fallback으로 256봉 DF로 predict(horizon=6) 후 방향성/confidence  

**방향성 함수**  
- `get_direction(traj)`: 6스텝 수익률 궤적의 기울기·양끝 차이로 1.0 / -1.0 / mean 결정  

**이어하기**  
- 이미 존재하는 `rl_training_data_full.csv`의 마지막 timestamp부터 이어서 청크 단위(CHUNK_SIZE=1024)로만 채움.

---

### 3.4 앙상블 라우터 및 예측 (`ensemble/ensemble_router.py`)

**포레스터 계층**  
- 공통 인터페이스: `BaseForecaster.predict(df, horizon=6) -> ForecastOutput(quantiles, median, confidence, model_name)`  
- **커스텀**: TFTForecaster, MacroHFTForecaster (현재 메인 앙상블에서는 사용하지 않을 수 있음)  
- **파운데이션**: ChronosForecaster, KronosForecaster, TimesFMForecaster, MoiraiForecaster  
- **단일 시계열**: TTMForecaster (IBM Granite TTM)  
- **NeuralForecast 래퍼**: UnifiedNFForecaster → PatchTST, iTransformer, NHITS, TiDE (동일 NF 4종, `data/nf` 로드)

**MetaRouter (MoE)**  
- 입력: 현재 시점의 피처 벡터 `x` (TFT feature_cols와 동일 스키마 가정; TFT 미사용 시 빈 리스트 가능)  
- 구조: Feature Attention(Squeeze-Excitation 스타일) → Residual 2층(LayerNorm+GELU+Dropout) → Head → Temperature  
- 학습 시: Noisy Top-K Gating (K=3), Temperature Scaling 후 softmax  
- 출력: 모델별 가중치 벡터 (상위 3개만 비영)

**MetaRouterEnsembleForecaster**  
- `predict(df, horizon)`:  
  - 각 가용 모델의 `ForecastOutput` 수집  
  - `_get_router_weights(df)`로 Meta Router 가중치 계산 (마지막 행 피처 정규화 후 router forward)  
  - `hybrid_weights = router_weights * confidences` 정규화  
  - `ensemble_median`, `ensemble_quantiles`, `direction_consensus`, `model_contributions` 반환  

실제 **RL/트레이딩**에서는 이 앙상블의 단일 호출보다는, `train_rl_agent`에서처럼 **7개 모델을 개별 호출**해 `pred_*`, `conf_*`를 채우는 구조와 맞닿아 있다.

---

### 3.5 강화학습 에이전트 (`ensemble/train_rl_agent.py`)

#### 3.5.1 상태/행동/보상

| 항목 | 내용 |
|------|------|
| **상태 차원** | `STATE_DIM = FEATURE_DIM + 5` (아래 FEATURE_DIM + 포지션 관련 5) |
| **FEATURE_DIM** | MODEL_PRED(7) + MODEL_CONF(7) + 3(stats: mean_pred, std_pred, mean_conf) + ELITE_COLS(13) + ALPHA_7_COLS(7) + REGIME_COLS(5) |
| **포지션 5차원** | long/short/0, entry_price/close-1, unrealized_pnl, max_drawdown, hold_count/MAX_HOLD |
| **행동** | 0=청산/관망, 1=롱, 2=숏 (역할에 따라 1 또는 2만 허용) |
| **에이전트 역할** | bull_sniper, bear_sniper, support_buyer, resistance_seller |

**TradingEnv**  
- 수수료 fee=0.0006, 슬리피지 slip=0.0003, MAX_HOLD phase별 72/144/288  
- 손절/익절/시간초과 시 다음 스텝에서 action을 0으로 덮어써 강제 청산  
- 보상: support_buyer/resistance_seller는 청산 시 실현 PnL에 5x/3x, 그 외는 실현 PnL + 효율 보너스(peak 대비)

#### 3.5.2 신경망 및 학습

- **TransformerIQN**: state [B, STATE_DIM] → (1,1) 임베딩 → 위치 인코딩 → Transformer Encoder → mean(dim=1) → IQN 분위수 샘플링(32) + 코사인 임베딩 → Q(s,a;τ)  
- **IQNAgent**: DQN 스타일 타깃 네트워크, tau=0.005 소프트 업데이트, Quantile Huber Loss (online τ 사용)  
- **RegimeReplayBuffer**: target_regimes 지정 시 해당 레짐이 1인 transition은 100%, 그 외 10%만 저장  
  - Bull → regime_bull  
  - Bear → regime_bear  
  - Support/Resistance → regime_chop, regime_whipsaw  

훈련 루프: 4개 env 동기 스텝, 각 에이전트가 자신의 buffer에만 push, MIN_BUFFER=2048 이상일 때 BATCH=256으로 UPDATE_FREQ=4마다 update. 검증은 MoEIQNTrader로 val 구간 한 번 돌려 PnL/거래 수/WR/평균 레버리지 로깅.

#### 3.5.3 실전 메타 라우터 (MoEIQNTrader)

- **입력**: current_idx, features(행 딕셔너리), pos(type, entry_price, unrealized, mdd, hold_norm)  
- **state 구성**: 위와 동일한 47차원 벡터 (pred, conf, stats, elite, alpha7, regime, pos 5개)  
- **4개 모델**에서 Q 분포 추론 → 각각 mean Q, adv_long = Q[1]-Q[0] (또는 adv_short), std로 Kelly 비율 계산  
- **레짐 분기**  
  - **롱 보유 중**: bull 레짐 & adv_bull>0 → BULL_HOLD; chop & adv_sup>0 → SUP_HOLD; 그 외 → CLOSE_LONG  
  - **숏 보유 중**: bear & adv_bear>0 → BEAR_HOLD; chop & adv_res>0 → RES_HOLD; 그 외 → CLOSE_SHORT  
  - **관망**: chop → sup vs res 중 adv 큰 쪽; bull → BULL_SNIPE; bear → BEAR_SNIPE; normal → ENS_LONG/ENS_SHORT (adv 임계값 0.5/0.2)  
- **진입 허들**: 신규 진입 시 Kelly < 0.2면 action=0 (HOLD_LOW_EDGE)  
- **동적 레버리지**: `leverage_rate = clip(selected_kelly * 0.5, 0.1, 1.0)` (MAX_LEVERAGE=1.0 기준)

---

## 4. 파일·경로 요약

| 용도 | 경로/변수 |
|------|------------|
| 피처 CSV (NF/RL 입력) | `data/training_features_5m.csv` |
| NF 4종 저장 | `data/nf` |
| RL 학습용 CSV | `data/ensemble/rl_training_data_full.csv` |
| 4-Agent 가중치 | `data/ensemble/best_moe_agents.pth` |
| Meta Router 가중치 (선택) | router_path 인자로 전달 (TFT feature_cols 있을 때만 사용) |

---

## 5. 아키텍처 다이어그램 (구성요소)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  FeatureEngineer                         │
                    │  (ETH+BTC merge → Alpha/OrderFlow/Tech/CVP/Quant/...)     │
                    └────────────────────────┬────────────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
     training_features_5m.csv      training_features_5m.csv        실시간 피처 DF
              │                              │                              │
              ▼                              ▼                              │
     train_nf_models.py              generate_training_csv                  │
     (PatchTST,iTrans,               (레짐+Elite+7모델 pred/conf)             │
      NHITS,TiDE)                            │                              │
              │                              ▼                              │
              ▼                     rl_training_data_full.csv                │
             data/nf                         │                              │
              │                              ▼                              │
              │                       train() 4-Agent IQN                   │
              │                              │                              │
              │                              ▼                              │
              │                     best_moe_agents.pth                      │
              │                              │                              │
              └──────────────┬───────────────┴───────────────┬───────────────┘
                             ▼                               ▼
                    MetaRouterEnsembleForecaster      MoEIQNTrader
                    (7 forecasters + router)         (4 agents + regime routing
                     → ensemble_median/quantiles      + Kelly leverage)
```

---

## 6. 정리

- **데이터**: 원시 봉 + 펀딩/OI 등 → `FeatureEngineer` → 단일 피처 테이블(`training_features_5m.csv`)이 모든 학습의 입력이다.  
- **예측**: NF 4종은 이 테이블의 `close`+7대 알파로 사전 학습되고, 실전에서는 TTM/TimesFM/Chronos 등 7개 포레스터가 각각 horizon=6 수익률(또는 방향성)을 내며, 선택적으로 MetaRouter로 가중 결합된다.  
- **행동**: 동일 피처 테이블에서 레짐·Elite·7모델 pred/conf를 넣어 RL용 CSV를 만들고, 4개의 Transformer IQN 에이전트를 레짐별 버퍼로 학습시킨 뒤, MoEIQNTrader가 레짐·Q값·Kelly로 롱/숏/청산과 동적 레버리지를 결정한다.

이 문서는 `core/feature_engineering.py`, `ensemble/ensemble_router.py`, `ensemble/train_nf_models.py`, `ensemble/train_rl_agent.py` 기준으로 작성된 현재 AI 에이전트 학습 로직의 데이터 흐름 및 아키텍처 명세이다.
