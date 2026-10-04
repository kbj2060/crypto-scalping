#!/usr/bin/env bash
# ollaya(결정 모델 서버, jevk5:4b) GPU 상주 래퍼 (2026-10-05). 뉴스 판정 워커(scripts/live_news_judge_20261005.py)가 쓴다.
# ollaya 는 저장소 밖 사용자 설치(~/.local/bin/ollaya, `curl -fsSL https://ollaya.dev/install.sh | OLLAYA_INSTALL_DIR=$HOME/.local OLLAYA_NO_SERVICE=1 sh`).
# 🔴 WSL 에서 그냥 띄우면 CUDA 러너가 세그폴트한다 -- `apt install nvidia-cuda-toolkit` 이 끌고 온 리눅스용
#   libnvidia-ptxjitcompiler.so.580(WSL 드라이버와 불일치)를 PTX JIT 때 읽기 때문. bwrap 으로 그 파일만 가린다(시스템 변경 없음).
#   VRAM ~5.5GB(라이브 봇 ~0.6GB 와 공유, 8GB 카드). 판정이 30분 없으면 모델이 내려간다(keep_alive).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BIN="${OLLAYA_BIN:-$HOME/.local/bin/ollaya}"
PTXJIT=/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.580.173.02
if pgrep -x ollaya >/dev/null; then
  echo "[$(date -Iseconds)] ollaya 가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi
MASK=(); [[ -e "$PTXJIT" ]] && MASK=(--ro-bind /dev/null "$PTXJIT")
export OLLAYA_DEVICE=cuda OLLAYA_KEEP_ALIVE=30m
exec "$ROOT/scripts/ops/_supervise.sh" \
  "ollaya" \
  "$ROOT/data/live/.supervisor_ollaya.lock" \
  "$ROOT/logs/supervisor/ollaya" \
  nice -n 10 bwrap --dev-bind / / "${MASK[@]}" "$BIN" serve
