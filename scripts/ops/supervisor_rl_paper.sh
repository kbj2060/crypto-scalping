#!/usr/bin/env bash
# 모의 매매 엔진(rl_1s_agent.py paper) 코인별 크래시 재기동 래퍼 (2026-10-05).
# 사용: bash scripts/ops/supervisor_rl_paper.sh ETH|SOL|XRP   (프로세스 하나 = 코인 하나, RL_SYMBOL 로 정한다)
# 왜: 10-05 12:10 서버 재부팅 뒤 손으로 띄웠던 엔진 셋이 다시 안 떠 대시보드 맞대결·벽 신호가 «엔진 대기»로 2시간 넘게 멈췄다.
# 세 코인의 명령줄이 같아 pgrep 으로 못 가른다 -- 중복 실행은 코인별 잠금(_supervise.sh flock)이 막는다.
set -uo pipefail
COIN="${1:?코인(ETH|SOL|XRP)}"
case "$COIN" in ETH|SOL|XRP) ;; *) echo "모르는 코인 $COIN" >&2; exit 2 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"
export RL_SYMBOL="${COIN}USDT" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# SOL·XRP 시작 이력 = 서버 엔진 전용 호가 수집 폴더(BT_ROOT/DD_ROOT, crontab 의 supervisor_book_ticker·depth_diff SOL·XRP 줄).
#   아카이브 폴더(data/live/orderflow)의 SOL·XRP 는 Pi 가 1시간 늦게 복제한 것이라 «직전 70분»을 못 채운다 -- 없으면 웹소켓 워밍업 ~63분.
[ "$COIN" = ETH ] || export RL_SEED_ORDERFLOW="$ROOT/data/live/orderflow_engine"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "rl_paper_${COIN}" \
  "$ROOT/data/live/.supervisor_rl_paper_${COIN}.lock" \
  "$ROOT/logs/supervisor/rl_paper_${COIN}" \
  "$PY" -u "$ROOT/scripts/rl_1s_agent.py" paper
