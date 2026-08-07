# DeepScalp-PnL v1 연구 결과 (2026-07-17)

## 결론

규칙 기반 진입, confidence threshold, 고정 TP/SL, cooldown, 고정 보유시간 없이
비용 차감 계좌수익을 직접 최적화하는 TCN-GRU 정책망을 구현하고 학습했다.

누수를 제거한 최종 모델은 4.5bp/노출변경 비용에서 validation과 개발 OOS 모두
`CASH`를 선택했다. 3.25bp 저비용 execution proxy도 validation -0.0819%였고 개발
OOS에서는 `CASH`였다. 따라서 현재 데이터와 비용 가정에서는 live에 투입할 수 있는
딥러닝 스캘핑 정책을 발견하지 못했다.

## 구현

- 120분 causal window
- Base branch: causal TCN 4 block + GRU(96)
- Microstructure branch: causal TCN 3 block
- State branch: 이전 signed notional, 보유시간, 미실현수익, 직전 순수익
- 출력: `SHORT/CASH/LONG` 이산 정책과 `margin_fraction`
- 고정 leverage 3, 최대 margin fraction 0.30
- 목적함수: 비용 차감 log-equity + CVaR + soft drawdown + 보조 분포예측
- 보조 target: 1/2/3/5분 수익, 5분 MFE/MAE/realized volatility
- soft-position curriculum 이후 Gumbel-softmax 이산 정책 학습

계좌 노출 계약은 다음과 같다.

```text
signed_notional = side * margin_fraction * 3
account_return = signed_notional * next_price_return
                 - fee_per_notional * abs(signed_notional - previous_notional)
```

## 데이터 계약

- `data/live/microstructure.duckdb`는 항상 `read_only=True`로 열었다.
- ETH/USDT와 ETH/USDC 오더북은 symbol feature 없이 시간순 단일 스트림으로 합쳤다.
- microstructure는 backward-asof 2분, orderbook은 backward-asof 6분 이내의 실제
  수집시각만 사용했다.
- microstructure coverage는 86.1%, orderbook coverage는 49.3%였다.
- trade ledger, 저장된 exit timestamp, triple-barrier label은 입력하지 않았다.
- `kelly_mult`, `signal_bias`, EAI 및 기존 모델 예측도 입력에서 제외했다.
- tail-risk/liquidation 컬럼은 유효 정보가 없어 제외했다.

## 발견한 semantic look-ahead

첫 학습에서는 validation +134.0%, 개발 OOS +77.3%가 나왔지만 폐기했다.

원인은 기존 1분 feature artifact의 BTC 파생 feature였다. BTC 5분봉 timestamp는 봉의
시작시각인데 기존 builder가 같은 timestamp의 ETH 1분봉에 backward join했다. 이 경우
BTC 5분봉의 close가 확정되기 전 1~5분 동안 미래 BTC 정보를 보게 된다.

`btc_lead_eth_follow_gap_3`, `eth_btc_ret_spread_12`, `btc_ret_3`의 개발 OOS raw IC가
약 0.22~0.23이었고, 해당 입력을 제거하자 과도한 성과가 사라졌다. 기존 truncation
audit는 timestamp 기준으로 현재 BTC 봉을 그대로 보존하기 때문에 이 semantic timing
오류를 검출하지 못했다.

최종 feature contract에서는 모든 BTC 파생 feature를 제외했다. 향후 다시 사용하려면
BTC 5분봉 availability timestamp를 open time + 5분으로 이동한 뒤 전체 feature를
재생성해야 한다.

## 최종 결과

| 후보 | 비용/노출변경 | Validation | 개발 OOS | 선택 행동 |
|---|---:|---:|---:|---|
| Taker 보수 후보 | 4.50bp | 0.000% | 0.000% | 100% CASH |
| Maker+taker 비용 proxy | 3.25bp | -0.0819% | 0.000% | Validation 2회 진입, OOS CASH |
| 기존 HGB | maker fill + triple barrier | +3.739% additive | 기존 report 값 | 직접 비교 불가 |
| 기존 GRU | maker fill + triple barrier | +3.601% additive | 기존 report 값 | 직접 비교 불가 |

기존 HGB/GRU 수치는 겹치는 triple-barrier 거래들의 additive PnL이고, 새 모델은 하나의
연속 계좌 포지션을 복리 계산한다. 따라서 숫자를 portfolio-equivalent 성과처럼 직접
비교하면 안 된다.

## 승격 판정

`promotion_pass=false`다.

- 비용 차감 양의 validation active policy가 없다.
- 7월 데이터는 이미 모델군 연구에 사용되어 untouched OOS가 아니다.
- microstructure 연속 이력이 4개월 promotion 기준보다 짧다.
- 저비용 proxy에는 실제 passive order fill/adverse-selection 데이터가 없다.

## 재현

```bash
venv/bin/python -m pytest -q -s test/test_deepscalp_pnl_20260717.py
venv/bin/python scripts/train_eval_deepscalp_pnl_20260717.py
venv/bin/python scripts/audit_deepscalp_pnl_20260717.py

DEEPSCALP_MODEL_ID=deepscalp_pnl_v1_maker_proxy_20260717 \
  venv/bin/python scripts/train_eval_deepscalp_pnl_20260717.py \
  --fee-per-notional 0.000325
```

## 다음 연구에 필요한 데이터

1. 실제 passive 주문의 제출시각, 가격, queue position, partial fill, cancel, 최종 fill,
   adverse-selection을 별도 artifact로 수집한다.
2. BTC 5분봉을 availability timestamp 기준으로 재생성한다.
3. 고정된 현재 모델 계약으로 2026-07-17 이후 fresh-forward shadow를 누적한다.
4. 최소 4개월 연속 데이터가 확보된 뒤 base-only 대비 microstructure branch의 순증분을
   seed ensemble과 일별 paired bootstrap으로 다시 검정한다.

현재 단계에서는 모델의 CASH 출력을 우회하는 진입 규칙을 추가하면 안 된다. 그렇게
하면 “딥러닝이 수익 거래를 선택했다”는 원래 목표와 검증 계약을 다시 깨게 된다.
