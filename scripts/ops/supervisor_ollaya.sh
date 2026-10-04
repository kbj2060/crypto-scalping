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
command -v bwrap >/dev/null || { echo "[$(date -Iseconds)] bwrap 없음 -- apt install bubblewrap" >&2; exit 1; }
# 리눅스용 ptxjit 를 버전 무관하게 찾는다(apt 업그레이드로 파일명이 바뀌면 가리기가 조용히 꺼져 세그폴트 반복이 된다)
PTXJIT="$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.[0-9]*.* 2>/dev/null | head -n 1)"
[[ -n "$PTXJIT" ]] || echo "[$(date -Iseconds)] 리눅스용 libnvidia-ptxjitcompiler 없음 -- 가리지 않고 띄운다(없으면 문제도 없다)" >&2
if pgrep -x ollaya >/dev/null; then
  echo "[$(date -Iseconds)] ollaya 가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi
MASK=(); [[ -n "$PTXJIT" ]] && MASK=(--ro-bind /dev/null "$PTXJIT")
export OLLAYA_DEVICE=cuda OLLAYA_KEEP_ALIVE=30m
exec "$ROOT/scripts/ops/_supervise.sh" \
  "ollaya" \
  "$ROOT/data/live/.supervisor_ollaya.lock" \
  "$ROOT/logs/supervisor/ollaya" \
  nice -n 10 bwrap --die-with-parent --dev-bind / / "${MASK[@]}" "$BIN" serve
