#!/usr/bin/env bash
# 웹어셈블리 묶음: rtui/dist/rtui-<내용 해시 8>.js · rtui-<해시>_bg.wasm · VERSION(이 해시) -- index.html 이 VERSION 을 읽어 그 이름을 부른다(캐시 = 파일 이름).
# 옛 해시 파일은 지운다. 컴파일 -j 2(개발기 메모리 감시 -- 링크 전 free -m available 4GB 이상 확인).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
# 11-29: 글꼴 부분집합은 소스 글자에서 만든다 -- 새 카드 글자가 빠지면 네모로 그려지니 빌드마다 다시(전체 ttf 가 있을 때만 · 없으면 있는 *.sub.ttf 그대로)
if ls fonts/*-Regular.ttf >/dev/null 2>&1; then bash fonts/subset.sh; fi
cargo build -j 2 --release --lib --target wasm32-unknown-unknown
W=target/wasm32-unknown-unknown/release/rtui.wasm
H=$(sha256sum "$W" | cut -c1-8)
rm -f dist/rtui*.js dist/rtui*_bg.wasm
wasm-bindgen --target web --no-typescript --out-dir dist --out-name "rtui-$H" "$W"
printf '%s\n' "$H" > dist/VERSION
# 응답 서버 /rtweb/ = 이 폴더(rtserve lib.rs) -- 같은 페이지를 묶음 옆 경로로(./dist/ → ./)
sed 's#\./dist/#./#g' index.html > dist/index.html
ls -la dist
