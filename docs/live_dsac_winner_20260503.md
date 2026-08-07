# Choose the Live DSAC Winner

Last updated: 2026-05-03 KST

## Current status

- 후보 3개 비교 대상은 이미 산출물이 있다.
- 현재 `trading_bot.py` 기본 동작은 `COMPACT_MODE=True`, `CONTROLLER_MODE=True`라서, 런타임상 기본 승자는 이미 `controller` 경로다.
- 다만 `controller`의 2026 백테스트 PnL이 비정상적으로 크다. 그대로 라이브 승자로 확정하기 전에 재현 확인이 필요하다.

관련 코드:

- 라이브 기본 체크포인트
  - [trading_bot.py](/home/llewyn/crypto-scalping/trading_bot.py:66)
  - [trading_bot.py](/home/llewyn/crypto-scalping/trading_bot.py:162)
- 라이브 모드 기본값
  - [trading_bot.py](/home/llewyn/crypto-scalping/trading_bot.py:157)
  - [trading_bot.py](/home/llewyn/crypto-scalping/trading_bot.py:158)
- 실제 실행 시 controller 우선 선택
  - [trading_bot.py](/home/llewyn/crypto-scalping/trading_bot.py:3834)
- startup 시 base + compact + controller 모두 로드
  - [trading_bot.py](/home/llewyn/crypto-scalping/trading_bot.py:4476)
  - [trading_bot.py](/home/llewyn/crypto-scalping/trading_bot.py:4485)

## Candidate scorecard

### 1. Exact 2026 eval

Source reports:

- [eval_best_dsac_agent_2026_exact_latest.json](/home/llewyn/crypto-scalping/data/ensemble/reports/eval_best_dsac_agent_2026_exact_latest.json)
- [eval_best_unified_2026_exact_latest.json](/home/llewyn/crypto-scalping/data/ensemble/reports/eval_best_unified_2026_exact_latest.json)
- [eval_best_unified_controller_2026_exact_latest.json](/home/llewyn/crypto-scalping/data/ensemble/reports/eval_best_unified_controller_2026_exact_latest.json)

| Candidate | Ckpt | Score | PnL | MDD | WR | Trades |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | `best_dsac_agents.pth` | 736.73 | 256.17% | -2.68% | 46.42% | 795 |
| compact | `best_dsac_unified.pth` | 962.48 | 320.56% | -2.52% | 62.52% | 731 |
| controller | `best_dsac_unified_controller.pth` | 6157.58 | 2048.36% | -4.64% | 60.48% | 873 |

읽는 법:

- `compact`는 `base`보다 score, PnL, WR이 모두 좋고 MDD도 약간 낫다.
- `controller`는 score와 PnL이 압도적이지만, MDD가 더 나쁘다.
- 따라서 `base`는 현재 승부권 밖이고, 실제 비교축은 `compact vs controller`다.

### 2. Native 2026 backtest

Source reports:

- [backtest_trading_bot_native_2026_base.json](/home/llewyn/crypto-scalping/data/ensemble/reports/backtest_trading_bot_native_2026_base.json)
- [backtest_trading_bot_native_2026_compact.json](/home/llewyn/crypto-scalping/data/ensemble/reports/backtest_trading_bot_native_2026_compact.json)
- [backtest_trading_bot_native_2026_controller.json](/home/llewyn/crypto-scalping/data/ensemble/reports/backtest_trading_bot_native_2026_controller.json)

| Candidate | PnL | MDD | WR | Trades | Avg exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 3070.19% | -2.68% | 68.20% | 805 | 0.540 |
| compact | 2536.08% | -2.04% | 79.47% | 833 | 0.444 |
| controller | 71775.19% | -3.17% | 83.59% | 963 | 0.563 |

해석:

- `compact`는 `base`보다 손익은 낮지만 MDD, WR이 더 좋다.
- `controller`는 다시 압승이지만, `71775%`는 검증 없이 채택하기에는 너무 크다.
- 특히 `controller` 리포트에는 bucket resize 로직과 execution leverage가 포함되어 있어서, 리포트가 실제 라이브 제약을 과하게 낙관하는지 확인이 필요하다.

추가 근거:

- controller bucket 통계 포함 리포트:
  - [backtest_trading_bot_native_2026_controller.json](/home/llewyn/crypto-scalping/data/ensemble/reports/backtest_trading_bot_native_2026_controller.json)
- backtest 구현:
  - [backtest_trading_bot_native_2026.py](/home/llewyn/crypto-scalping/scripts/backtest_trading_bot_native_2026.py:126)
  - [backtest_trading_bot_native_2026.py](/home/llewyn/crypto-scalping/scripts/backtest_trading_bot_native_2026.py:300)

## Working decision

현재 작업 가설은 아래가 맞다.

1. `base`는 탈락
2. `compact`는 보수적 기준선
3. `controller`는 수치상 1위지만, 재현 확인 전까지는 "provisional winner"

즉 지금 바로 필요한 질문은 "`controller`를 최종 라이브 기본값으로 확정할 수 있나?"다.

## Live choice now

현재 코드 기준 라이브 기본값은 이미 `controller`다.

- `COMPACT_MODE=True`
- `CONTROLLER_MODE=True`
- controller runtime이 로드되면 `compact`보다 우선 적용

따라서 이 작업의 목적은 "어떤 후보를 쓸까?"보다 정확히는 아래 두 가지다.

1. 현재 controller 기본값을 유지할지 확정
2. 아니면 compact로 되돌릴지 근거를 만들기

## Repro commands

### Exact eval rerun

```bash
python3 scripts/eval_best_dsac_agent_2026.py \
  --csv-path data/rl_training_2026_unified.csv \
  --ckpt-path data/ensemble/ckpt/best_dsac_agents.pth \
  --config-path data/ensemble/ckpt/dsac_train_config_latest.json \
  --out-json data/ensemble/reports/eval_best_dsac_agent_2026_exact_rerun.json

python3 scripts/eval_best_unified_2026.py \
  --csv-path data/rl_training_2026_unified.csv \
  --ckpt-path data/ensemble/ckpt/best_dsac_unified.pth \
  --config-path data/ensemble/ckpt/dsac_unified_train_config.json \
  --out-json data/ensemble/reports/eval_best_unified_2026_exact_rerun.json

python3 scripts/eval_best_unified_controller_2026.py \
  --csv-path data/rl_training_2026_unified.csv \
  --ckpt-path data/ensemble/ckpt/best_dsac_unified_controller.pth \
  --config-path data/ensemble/ckpt/dsac_unified_controller_train_config.json \
  --out-json data/ensemble/reports/eval_best_unified_controller_2026_exact_rerun.json
```

스크립트 기준:

- [eval_best_dsac_agent_2026.py](/home/llewyn/crypto-scalping/scripts/eval_best_dsac_agent_2026.py:266)
- [eval_best_unified_2026.py](/home/llewyn/crypto-scalping/scripts/eval_best_unified_2026.py:384)
- [eval_best_unified_controller_2026.py](/home/llewyn/crypto-scalping/scripts/eval_best_unified_controller_2026.py:384)

### Native backtest rerun

```bash
python3 scripts/backtest_trading_bot_native_2026.py \
  --csv-path data/rl_training_2026_unified.csv \
  --ckpt-path data/ensemble/ckpt/best_dsac_agents.pth \
  --mode base \
  --report-path data/ensemble/reports/backtest_trading_bot_native_2026_base_rerun.json

python3 scripts/backtest_trading_bot_native_2026.py \
  --csv-path data/rl_training_2026_unified.csv \
  --ckpt-path data/ensemble/ckpt/best_dsac_unified.pth \
  --mode compact \
  --report-path data/ensemble/reports/backtest_trading_bot_native_2026_compact_rerun.json

python3 scripts/backtest_trading_bot_native_2026.py \
  --csv-path data/rl_training_2026_unified.csv \
  --ckpt-path data/ensemble/ckpt/best_dsac_unified_controller.pth \
  --mode controller \
  --report-path data/ensemble/reports/backtest_trading_bot_native_2026_controller_rerun.json
```

스크립트 기준:

- [backtest_trading_bot_native_2026.py](/home/llewyn/crypto-scalping/scripts/backtest_trading_bot_native_2026.py:628)

## Acceptance rule

`controller`를 최종 승자로 확정하려면 최소한 아래를 만족해야 한다.

1. exact eval rerun에서 기존 `score`와 `pnl` 순위가 유지될 것
2. native backtest rerun에서도 `compact` 대비 우위가 재현될 것
3. 과도한 성능 차이가 resize/leverage 구현 버그가 아니라는 설명이 가능할 것

이 셋 중 3번이 무너지면, 라이브 기본값은 `compact`로 내리는 게 맞다.

## Immediate next step

다음 작업은 분석이 아니라 재현이다.

1. 위 6개 명령 재실행
2. rerun 결과를 기존 latest 리포트와 diff
3. `controller`가 여전히 우세하면 라이브 기본값 유지
4. 재현이 흔들리면 `CONTROLLER_MODE=0`으로 내려서 `compact`를 기본값으로 복귀

## If rollback is needed

라이브를 `compact`로 되돌리는 최소 조건:

```bash
export CONTROLLER_MODE=0
export COMPACT_MODE=1
```

코드상 의미:

- `CONTROLLER_MODE`가 꺼지면 controller branch를 타지 않는다.
- 그 상태에서 `COMPACT_MODE`가 켜져 있으면 unified compact가 기본 실행 경로가 된다.
