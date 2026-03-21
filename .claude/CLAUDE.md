# CLAUDE.md — ETH/USDT 자동매매 시스템 아키텍처 & 개발 가이드

## 프로젝트 개요

ETH/USDT 선물 5분봉 자동매매 시스템.
MoE 6-Agent + GatingNet7이 단독 의사결정 주체이며,
리스크 게이트를 거쳐 거래소 주문을 실행한다.

**핵심 원칙: 두뇌보다 팔다리를 먼저 만든다.**
아무리 정교한 신호도 리스크 관리 없이는 한 번의 블랙스완에 전멸한다.

---

## 7-Layer 아키텍처

```
┌─────────────────────────────────────────────────────┐
│  1. Data Fabric                                      │
│     Binance + fallback, on-chain, health heartbeat  │
├─────────────────────────────────────────────────────┤
│  2. Feature Forge                                    │
│     120+ features, HMM regime, MTF, synthetic alpha │
├─────────────────────────────────────────────────────┤
│  3. Signal Brain (MoE 6-Agent 단독)                  │
│     GatingNet7 → KellySizer → action + leverage     │
├─────────────────────────────────────────────────────┤
│  4. The Gatekeeper  ← 최우선 구축 대상               │
│     Daily cap · DD breaker · Black swan · Vol scale │
├─────────────────────────────────────────────────────┤
│  5. Position Lifecycle                               │
│     PositionManager(단일 진실) · TP/SL · Execution  │
├─────────────────────────────────────────────────────┤
│  6. Trade Journal                                    │
│     SQLite: 모든 거래 기록 (agent/regime/PnL)        │
├─────────────────────────────────────────────────────┤
│  7. Observatory                                      │
│     Drift 감지 · Agent 성능 저하 · Telegram 알림     │
└───────────────────────────────────────┬─────────────┘
                                        │ feedback loop
                                        └──→ Layer 3 가중치 조정
```

### 데이터 흐름 요약

```
Binance API → FeatureEngineer (120+피처)
           → HMM (레짐 4-state)
           → MTF (1h/4h 추세)
           ↓
       MoE 6-Agent + GatingNet7
           ↓
       action(0/1/2) + kelly_leverage
           ↓
       Gatekeeper (6-gate 순차 필터)
           ↓
       approve → PositionManager → Exchange Order
       reject  → log + alert
           ↓
       Trade Journal (SQLite)
           ↓
       Observatory → drift/degradation → feedback
```

---

## Layer 1: Data Fabric

### 현재 상태
- `trading_bot.py` → `BinanceLiveFetcher`
- OHLCV, OI, 펀딩비, 롱숏비율, 체결 데이터 수집
- 5분봉 기준 2500개 캔들 히스토리 유지

### 문제점
- Binance 단일 의존: API 다운 시 봇 전체 정지
- 온체인 데이터 부재: 고래 시그널이 거래소 데이터에만 의존
- 헬스 모니터링 없음: API 지연/장애 감지 불가

### 할 일
- **거래소 이중화**: Binance 장애 시 OKX/Bybit에서 가격+OI fallback
- **온체인 피드**: ETH 거래소 넷플로우 (CryptoQuant API)
- **헬스 하트비트**: API 응답 > 3초 시 알림, 5회 연속 실패 시 봇 정지
---

## Layer 2: Feature Forge

### 현재 상태 (대부분 유지)
- `core/feature_engineering.py` → `FeatureEngineer.process()`
- 120+ 피처: 기술적 지표, 오더플로우, 펀딩비 모멘텀, Hurst, CVP, 합성 알파 4종(OFTI, KEL, MTA, SVPS)
- HMM 4-state 온라인 레짐 감지 (`OnlineHMMDetector`)
- MTF 1h/4h 멀티타임프레임 피처 (`MultiTimeframeFeatures`)
- MDJD (Microstructure-Driven Jump-Diffusion) 시그널

### 피처 차원 구조 (STATE_DIM 분해)

```
STATE_DIM = FEATURE_DIM + 5(pos) + 5(HMM) + 7(MTF)

FEATURE_DIM 내부:
  STATE_PRED  (7)  : pred_{tide,ridge,patchtst,timesfm,chronos,ttm,mdjd}
  STATE_CONF  (7)  : conf_{tide,ridge,ttm,chronos,timesfm,mdjd,patchtst}
  stats       (3)  : preds.mean, preds.std, confs.mean
  STATE_ELITE (9)  : evt_excess_z, sig_orderblock, sig_ai_squeeze, ...
  STATE_ALPHA (9)  : hour_cos, garch_vol, breakout_strength, ...
  REGIME_COLS (5)  : regime_{chop,whipsaw,bull,bear,normal}
  STATE_SYNTH (14) : ou_funding_z, fcsz, vebr, ofti, cada, ...

pos_features (5)   : pos_flag, entry_dist, tanh(unr_pnl), clip(mdd), hold_norm
HMM_DIM     (5)   : 4-state probs + entropy
MTF_DIM     (7)   : 1h_{ret,vol,trend}, 4h_{ret,vol,trend}, alignment

STACK_N = 4 → STACKED_STATE_DIM = STATE_DIM × 4
```

### 개선 사항
- 피처 결측 기본값을 피처별로 지정 (RSI=50, 상관관계=0 등)
- `_handle_missing`의 ffill+0 채우기를 피처 유형별 분기로 교체

---

## Layer 3: Signal Brain (MoE 6-Agent 단독)

### 아키텍처

```
                    Market State (STACKED_STATE_DIM)
                              │
                 ┌────────────┼────────────┐
                 │            │            │
            ┌────┴────┐ ┌────┴────┐ ┌────┴────┐
            │  bull    │ │  bear   │ │chop_long│ ...
            │ (롱전용) │ │ (숏전용) │ │(횡보롱) │
            │ CVaR0.60 │ │ CVaR0.40│ │ CVaR0.50│
            └────┬────┘ └────┬────┘ └────┬────┘
                 │            │            │
                 └────────────┼────────────┘
                              │
                     GatingNet7 (7-way softmax)
                     [flat, bull, bear, cL, cS, nL, nS]
                              │
                    Best agent 선택 + advantage 검증
                              │
                     KellyCriterionSizer
                     (IQN 분위 → leverage_rate)
                              │
                    EpistemicUncertaintyGate
                    (앙상블 헤드 간 std → 진입 필터)
                              │
                     ┌────────┴────────┐
                     │  action (0/1/2) │
                     │  leverage_rate   │
                     └─────────────────┘
```

### 6개 에이전트 구성

| Agent | 방향 | 타겟 레짐 | CVaR 임계값 | 2-Action 매핑 |
|-------|------|-----------|------------|--------------|
| bull | 롱 | regime_bull | 0.60 (공격) | 0=대기/청산, 1=롱진입 |
| bear | 숏 | regime_bear | 0.40 (보수) | 0=대기/청산, 1=숏진입 |
| chop_long | 롱 | regime_chop, whipsaw | 0.50 | 0=대기/청산, 1=롱진입 |
| chop_short | 숏 | regime_chop, whipsaw | 0.50 | 0=대기/청산, 1=숏진입 |
| normal_long | 롱 | regime_normal | 0.50 | 0=대기/청산, 1=롱진입 |
| normal_short | 숏 | regime_normal | 0.50 | 0=대기/청산, 1=숏진입 |

모든 에이전트는 2-Action(대기/진입)으로 통일. 숏 에이전트가 action=1을 내면 GatingRouter7이 global action=2(숏)로 매핑.

### 핵심 모듈 상세

**RobustIQN (모델)**
- MarketAttentionEncoder: 피처 6그룹을 토큰화 → 2-head Self-Attention × 2 layer (~8K 파라미터)
- feat_extractor: Linear(state+attn, 128) → Linear(128, 64)
- context_gate: market_no_pos → sigmoid → feat 곱셈 (시장 상태 기반 동적 게이팅)
- BootstrapEnsembleHeads: N개 독립 (v_head, a_head) → 헤드 간 분산 = epistemic uncertainty
- IQN 코사인 임베딩: tau ~ U(0,1) → cos(tau × [1..64] × π) → phi(64)

**GatingNet7 (라우터)**
- 입력: STACKED_STATE_DIM
- 구조: Linear → LayerNorm → SiLU → Linear → SiLU → Linear(7)
- 출력: softmax(logits / 0.5) → [flat, bull, bear, cL, cS, nL, nS] 가중치
- REINFORCE로 학습 (BnH 알파 기반 리턴)

**KellyCriterionSizer**
- IQN 분위에서 win_rate, payoff_ratio 직접 추정
- f* = (p*b - q) / b → Half-Kelly × uncertainty_penalty × gating_confidence
- 출력: leverage_rate ∈ [0.1, 1.0]

**EpistemicUncertaintyGate**
- 앙상블 헤드 간 Q 표준편차 → epistemic uncertainty
- std < LOW → 정상 / LOW~HIGH → Kelly 축소 / std ≥ HIGH → 진입 거부
- LOW/HIGH 임계값은 running stats로 자동 보정

### 주요 파일
- `ensemble/train_rl_agent.py` → 학습 (MoE 6-Agent + GatingNet7)
- `trading_bot.py` → 라이브 추론 (`MoELiveRouter`, `GatingRouter7`)

---

## Layer 4: The Gatekeeper (최우선 구축 대상)

**현재 상태: 존재하지 않음.**
Kelly + EpistemicGate만으로는 "포트폴리오 전체가 얼마나 위험한가"를 볼 수 없다.

### 6-Gate 순차 필터

Signal Brain에서 `action + kelly_leverage`가 나오면 아래 6개 게이트를 순서대로 통과해야 한다.
어느 하나라도 실패하면 주문이 나가지 않는다.

**Gate 1~3: 하드 차단 (진입 자체를 막음)**

```
Gate 1 — Daily Loss Cap
  조건: 오늘 실현 손실 합산 < -2%
  행동: 24시간 신규 진입 전면 차단
  근거: 프로 트레이더의 가장 기본적인 규율. 현재 코드에 전혀 없음.

Gate 2 — Drawdown Breaker
  조건: 계좌 고점 대비 하락 (peak-to-trough) < -5%
  행동: 전 포지션 즉시 청산 + 24시간 거래 정지
  근거: 드로다운이 깊어질수록 복구 확률이 기하급수적으로 하락.

Gate 3 — Black Swan Detector
  조건: 다음 중 하나라도 충족 시 발동
    - 스프레드가 최근 1h 평균의 5배 이상
    - OI가 5분 내 10% 이상 급변
    - GK 변동성이 5-sigma 초과
  행동: 전 포지션 즉시 청산 + 1시간 쿨다운
  근거: LUNA/FTX급 사태에서 에이전트 판단 자체가 신뢰 불가 (분포 외 데이터).
```

**Gate 4~6: 소프트 조절 (사이즈를 줄임)**

```
Gate 4 — Volatility Size Adjuster
  조건: GK_vol 백분위
  행동:
    - top 20% → kelly *= 0.5
    - top 10% → kelly *= 0.25
  근거: 변동성 높을 때 같은 레버리지는 실질 리스크가 2~4배.

Gate 5 — Consecutive Loss Guard
  조건: 최근 연패 횟수 (Trade Journal에서 조회)
  행동:
    - 3연패 → kelly *= 0.5
    - 5연패 → 다음 1개 신호 스킵
  근거: 연패는 레짐 전환을 시사. 축소 후 관찰이 합리적.

Gate 6 — Time Filter
  조건: 세션 전환 직전/직후 10분 (UTC 20:55~21:05 등)
  행동: 신규 진입 스킵 (기존 포지션은 유지)
  근거: 유동성 공백 구간에서 슬리피지 폭발.
```

### 구현 위치
- 신규 파일: `core/gatekeeper.py`
- `trading_bot.py`에서 Signal Brain 출력과 Execution 사이에 삽입

---

## Layer 5: Position Lifecycle (신규 구축)

**현재 치명적 결함:**
1. `trading_bot.py`에서 `meta_result`를 출력만 하고 실제 `exchange.create_order()`를 호출하지 않음
2. 포지션 상태(`self.pos`)가 `MoELiveRouter`에 독립 존재 → 거래소 실제 포지션과 불일치 가능

### PositionManager (단일 진실의 원천)

```python
class PositionManager:
    """모든 컴포넌트가 이 객체만 참조. 직접 .pos를 관리하는 코드 금지."""

    상태 머신:
      FLAT → (approved long) → LONG
      FLAT → (approved short) → SHORT
      LONG → (signal/TP/SL) → CLOSING → (fill confirmed) → FLAT
      SHORT → (signal/TP/SL) → CLOSING → (fill confirmed) → FLAT

    CLOSING 상태 존재 이유:
      주문 전송 ~ 체결 확인 사이 시간차에서
      중복 주문 방지 (다음 5분봉 사이클이 돌아도 "이미 청산 중"으로 인식)

    필수 속성:
      state: FLAT | LONG | SHORT | CLOSING
      entry_price: float
      entry_time: datetime
      hold_bars: int
      unrealized_pnl: float (거래소 실시간 조회)
      sl_price: float
      tp_price: float
```

### TP/SL 엔진

```
진입 즉시:
  - SL = entry_price × (1 - 2.5%)  (롱 기준)
  - TP = entry_price × (1 + 4.0%)
  → 거래소에 조건부 주문(OCO) 즉시 전송

트레일링:
  - 미실현 PnL > +1.5% → SL을 손익분기점으로 이동
  - 미실현 PnL > +3.0% → SL을 +1.5% 수준으로 이동
  → 거래소 SL 주문 수정 API 호출
```

### Execution Engine

```
주문 흐름:
  1. Gatekeeper 승인 + final_size 수신
  2. 시장가 주문 전송 (작은 사이즈)
     - 큰 사이즈(계좌 대비 50%+)면 TWAP 3회 분할
  3. 체결 확인 대기 (timeout 30초)
  4. 실패 시 3회 재시도 → 포기 시 Telegram 알림
  5. 체결 완료 → PositionManager 상태 갱신 + TP/SL 주문 전송
```

### 구현 위치
- 신규 파일: `core/position_manager.py`
- 신규 파일: `core/execution_engine.py`
- `trading_bot.py`에서 `_run_cycle()` 내 주문 실행 연결

---

## Layer 6: Trade Journal (신규 구축)

**현재 상태: 존재하지 않음.**
학습(train)과 실전(live)의 피드백 루프가 완전히 끊겨 있음.

### SQLite 스키마

```sql
CREATE TABLE trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- 진입
    entry_time      TEXT NOT NULL,        -- ISO 8601
    entry_price     REAL NOT NULL,
    side            TEXT NOT NULL,        -- 'LONG' | 'SHORT'
    size            REAL NOT NULL,        -- 계약 수량
    kelly_lev       REAL,                 -- Kelly가 결정한 레버리지
    -- 컨텍스트
    agent_name      TEXT,                 -- 'bull', 'bear', 'chop_long' 등
    regime          TEXT,                 -- 'bull', 'bear', 'chop', 'whipsaw', 'normal'
    hmm_state       TEXT,                 -- 'bull-trend', 'bear-trend', 'hv-chop', 'lv-range'
    gating_weights  TEXT,                 -- JSON: [flat, bull, bear, cL, cS, nL, nS]
    epist_std       REAL,                 -- epistemic uncertainty
    -- 퇴출
    exit_time       TEXT,
    exit_price      REAL,
    exit_reason     TEXT,                 -- 'signal', 'tp', 'sl', 'trailing', 'gate_force', 'breaker'
    -- 결과
    realized_pnl    REAL,                 -- % 단위
    hold_bars       INTEGER,
    slippage_bps    REAL,                 -- 실제 슬리피지 (bps)
    -- 게이트 기록
    gates_passed    TEXT                  -- JSON: 통과한 게이트 목록
);

CREATE INDEX idx_trades_agent ON trades(agent_name);
CREATE INDEX idx_trades_regime ON trades(regime);
CREATE INDEX idx_trades_time ON trades(entry_time);
```

### 핵심 쿼리

```sql
-- 에이전트별 최근 50거래 승률
SELECT agent_name,
       COUNT(*) as n,
       AVG(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
       AVG(realized_pnl) as avg_pnl
FROM (SELECT * FROM trades WHERE agent_name = ? ORDER BY id DESC LIMIT 50);

-- 레짐별 수익성
SELECT regime, COUNT(*), AVG(realized_pnl), SUM(realized_pnl)
FROM trades GROUP BY regime;

-- 오늘 실현 손실 합산 (Gate 1용)
SELECT COALESCE(SUM(realized_pnl), 0)
FROM trades WHERE date(exit_time) = date('now');
```

### 구현 위치
- 신규 파일: `core/trade_journal.py`

---

## Layer 7: Observatory (신규 구축)

### 핵심 기능 3가지

**1. Agent Degradation Detector**
```
Trade Journal에서 에이전트별 rolling 50거래 승률 조회
  - WR < 40% → 해당 에이전트 GatingNet 라우팅에서 제외
  - WR < 35% 3일 연속 → retrain 트리거 + Telegram 알림
이것은 EpistemicUncertaintyGate의 더 직접적이고 효과적인 대체.
분위 분산보다 "실제로 최근에 돈을 잃고 있는가"가 더 신뢰할 수 있음.
```

**2. Feature Drift Detector**
```
매일 1회, 최근 24h 피처 분포와 학습 데이터 분포 비교
  - KL divergence 또는 PSI (Population Stability Index)
  - PSI > 0.25 인 피처가 5개 이상 → 경고
  - PSI > 0.25 인 피처가 10개 이상 → retrain 트리거
```

**3. System Health + Alerts**
```
모니터링 대상:
  - API 응답 지연 (> 3초)
  - 주문 체결 실패율
  - 봇 사이클 스킵 (5분봉 놓침)
  - Gatekeeper 차단 빈도 (Gate 3이 자주 발동되면 시장 이상)

알림 채널: Telegram Bot API
  - 포지션 진입/퇴출
  - Gate 차단 (사유 포함)
  - 일일 성과 요약
  - 시스템 장애
```

### 구현 위치
- 신규 파일: `core/observatory.py`

---
---

## 코드 레벨 버그 및 주의 사항

### 포지션 상태 관리 규칙

```
금지: 각 컴포넌트가 자체 self.pos를 관리하는 것
허용: PositionManager 단일 객체만 포지션 상태를 소유
      다른 모든 컴포넌트는 PositionManager를 읽기 전용으로 참조

이유: MoELiveRouter.pos와 거래소 실제 포지션이 불일치하면
      "이미 보유 중"으로 판단해 신호를 무시하거나
      "보유 없음"으로 판단해 중복 진입하는 사고 발생
```

---

## 구현 우선순위 (Phase Roadmap)

### Phase 1: Survive (1~2주) ← 여기부터 시작

```
[ ] PositionManager 구현 (단일 진실의 원천)
    - MoELiveRouter의 self.pos 제거, PositionManager 참조로 교체
    - 상태 머신: FLAT → LONG/SHORT → CLOSING → FLAT

[ ] Gatekeeper Gate 1~3 구현 (하드 차단)
    - Gate 1: 일일 실현 손실 -2% 하드캡
    - Gate 2: 드로다운 -5% 서킷 브레이커
    - Gate 3: 블랙스완 감지기 (스프레드/OI/변동성 이상)

[ ] BootstrapEnsembleHeads 5→3 축소
    - N_ENSEMBLE = 3 으로 변경
    - 학습 시간 ~40% 감소, 추론 오버헤드 감소
```

### Phase 2: Execute (3~4주)

```
[ ] Execution Engine 구현
    - exchange.create_order() 실제 연결
    - TP/SL 조건부 주문 자동 전송
    - 트레일링 스톱 로직

[ ] Trade Journal 구현
    - SQLite 테이블 생성
    - 매 거래 진입/퇴출 시 자동 기록
    - Gatekeeper Gate 1, 5가 여기서 데이터 조회
```

### Phase 3: Adapt (5~8주)

```
[ ] Observatory 구현
    - Agent degradation detector (rolling WR < 40% → 비활성화)
    - Feature drift detector (PSI > 0.25 → 경고)
    - Telegram 알림 연동

[ ] Gatekeeper Gate 4~6 구현 (소프트 조절)
    - Gate 4: 변동성 스케일링
    - Gate 5: 연패 가드
    - Gate 6: 시간 필터
```

### Phase 4: Optimize (3개월+)

```
[ ] 데이터 이중화 (OKX/Bybit fallback)
[ ] 온체인 시그널 통합 (거래소 넷플로우)
[ ] 자동 재학습 파이프라인 (Observatory 트리거 → train_rl_agent.py)
[ ] 다중 자산 확장 (BTC/SOL)
```

---

## 주요 파일 맵

```
프로젝트 루트/
├── core/
│   ├── feature_engineering.py  ← Layer 2: 피처 생성
│   ├── cvp.py                  ← Layer 2: CVP 피처
│   ├── gatekeeper.py           ← Layer 4: 리스크 게이트 (신규)
│   ├── position_manager.py     ← Layer 5: 포지션 상태 머신 (신규)
│   ├── execution_engine.py     ← Layer 5: 거래소 주문 실행 (신규)
│   ├── trade_journal.py        ← Layer 6: SQLite 거래 기록 (신규)
│   └── observatory.py          ← Layer 7: 모니터링 (신규)
│
├── ensemble/
│   ├── train_rl_agent.py       ← Layer 3: MoE 6-Agent 학습
│   │   ├── OnlineHMMDetector        (HMM 레짐 감지)
│   │   ├── KellyCriterionSizer      (Kelly 포지션 사이징)
│   │   ├── MultiTimeframeFeatures   (MTF 피처)
│   │   ├── MarketAttentionEncoder   (피처 그룹 Self-Attention)
│   │   ├── BootstrapEnsembleHeads   (앙상블 헤드)
│   │   ├── EpistemicUncertaintyGate (불확실성 필터)
│   │   ├── RobustIQN               (IQN 모델)
│   │   ├── GatingNet7              (7-way 라우터)
│   │   └── GatingRouter7           (라이브 추론 라우터)
│   └── ensemble_router.py     ← 파운데이션 모델 앙상블
│
├── strategies/
│   └── elite_builder.py        ← 엘리트 시그널 생성
│
├── trading_bot.py              ← 메인 봇 루프
│   ├── BinanceLiveFetcher           (데이터 수집)
│   ├── EnsemblePredictor            (6대 파운데이션 앙상블, 대시보드 전용)
│   ├── NFStatePredictor             (NF/TTM 예측, RL state 구성용)
│   ├── RidgeSignalComputer          (Ridge 선형 퀀트)
│   ├── MoELiveRouter                (라이브 MoE 라우터)
│   ├── LLMAnalyzer                  (LLM 분석, 대시보드 전용)
│   └── PolymarketFetcher            (크라우드 확률, 대시보드 전용)
│
└── data/
    ├── ensemble/
    │   ├── rl_training_data_full.csv
    │   └── ckpt/
    │       ├── best_rl_agents.pth   ← 학습된 에이전트 가중치
    │       └── rl_checkpoint.pth    ← 학습 체크포인트
    └── ridge_model.pkl              ← Ridge 퀀트 모델
```

---

## 학습 파라미터 참조

```python
# train_rl_agent.py 핵심 상수
NEP             = 1000          # 총 에피소드
BATCH           = 512           # 배치 크기
UPDATE_FREQ     = 64            # N 스텝마다 업데이트
MIN_BUFFER      = 2048          # 학습 시작 최소 버퍼
EPS_START       = 1.0           # ε-greedy 시작값
EPS_END         = 0.01          # ε-greedy 최종값
EPS_DECAY_STEPS = 400000        # ε 감소 스텝
GAMMA           = 0.99          # 할인율
TAU             = 0.005         # target network soft update
MAX_EPISODE_STEPS = 4096        # 에피소드 최대 길이
MAX_LEVERAGE    = 1.0           # 레버리지 하드캡

# GatingNet 학습
GATING_LR       = 1e-3          # GatingNet 학습률
GATING_FREQ     = ep % 10       # 10 에피소드마다 학습
GATING_START_EP = 50            # 에이전트 충분히 학습 후 시작
N_TRAJECTORIES  = 3             # 궤적 반복 횟수
GATING_N_STEPS  = 1500          # 궤적 길이

# HMM 온라인 업데이트
HMM_UPDATE_FREQ = ep % 50       # 50 에피소드마다
HMM_ONLINE_ITER = 5             # 단축 EM 반복
```

---

## 코딩 컨벤션

- Python 3.10+, PyTorch 2.x
- 타입 힌트 필수 (새 코드)
- 로거: `logging.getLogger(__name__)`
- 상수: UPPER_SNAKE_CASE, 파일 상단에 모아둘 것
- 새 모듈은 반드시 단위 테스트 작성 (특히 Gatekeeper, PositionManager)
- 금액/가격 계산은 float 사용 (Decimal 불필요 — 5분봉 스케일에서 정밀도 충분)