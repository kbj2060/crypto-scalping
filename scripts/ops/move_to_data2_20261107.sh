#!/usr/bin/env bash
# 디렉터리를 6TB(/mnt/data2)로 옮기고 원래 경로에 **디렉터리** 심볼릭 링크를 남긴다 (2026-11-07).
#   copy <src> <dst>   rsync 만(여러 번 돌려도 된다 · 봇·수집기를 안 멈춘다)
#   swap <src> <dst>   마지막 rsync -> 파일 수·바이트 대조 -> src 를 src.moved 로 -> 링크. 쓰는 프로세스는 호출자가 먼저 멈춘다.
#   clean <src>        링크가 살아 있는지 확인하고 src.moved 를 지운다.
#
# 🔴디렉터리만 링크한다(라이브 DB 파일 하나를 링크하지 않는다). 디스크가 안 붙으면 디렉터리 링크는 깨져서
#   mkdir·open 이 **실패**하지만, 파일 링크나 안 붙은 빈 폴더는 새 빈 DB 를 조용히 만든다(10-01 설계의 금지 사유).
set -uo pipefail
cmd="${1:-}"; src="${2:-}"; dst="${3:-}"
DATA2="${MOVE_DATA2:-/mnt/data2}"
BW="${MOVE_BWLIMIT:-25000}"       # KB/s -- 라이브 수집기와 같은 HDD 를 읽는다
die() { echo "🔴 $*" >&2; exit 1; }
[[ -n "$src" ]] || die "usage: $0 copy|swap <src> <dst> / clean <src>"
[[ "${MOVE_SKIP_MOUNT_CHECK:-0}" == "1" ]] || mountpoint -q "$DATA2" || die "$DATA2 not mounted"
count() { find "$1/" -type f | wc -l; }
bytes() { find "$1/" -type f -printf '%s\n' | awk '{s+=$1} END{printf "%.0f", s}'; }

case "$cmd" in
  copy)
    [[ -n "$dst" && -d "$src" && ! -L "$src" ]] || die "copy: src must be a real directory: $src"
    mkdir -p "$dst"
    nice -n 10 rsync -a --bwlimit="$BW" "$src/" "$dst/"
    r=$?; [[ $r -eq 24 ]] && r=0
    echo "copy rc=$r $src -> $dst"; exit $r ;;
  swap)
    [[ -n "$dst" && -d "$src" && ! -L "$src" ]] || die "swap: src must be a real directory: $src"
    mkdir -p "$dst"
    rsync -a "$src/" "$dst/" || die "final rsync failed: $src"
    ns=$(count "$src"); nd=$(count "$dst"); bs=$(bytes "$src"); bd=$(bytes "$dst")
    # 원본에 없는 파일이 사본에 더 있을 수는 있다(이전 시도·보관분) -- 사본이 모자라면 멈춘다.
    (( nd >= ns && bd >= bs )) || die "mismatch $src: files $ns vs $nd, bytes $bs vs $bd"
    mv "$src" "$src.moved" && ln -s "$dst" "$src" || die "swap failed: $src"
    echo "swap OK $src -> $dst (files $ns/$nd, bytes $bs/$bd)" ;;
  clean)
    [[ -L "$src" && -d "$src" ]] || die "clean: $src is not a live symlink"
    [[ -d "$src.moved" ]] || { echo "nothing to clean: $src.moved"; exit 0; }
    rm -rf "$src.moved" && echo "cleaned $src.moved" ;;
  *) die "unknown command: $cmd" ;;
esac
