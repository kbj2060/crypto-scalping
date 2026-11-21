#!/usr/bin/env bash
# 공식 배포본(README.md 의 주소·해시) → 화면 글자만 남긴 *.sub.ttf (draw.rs 가 include_bytes 로 묶는다 · ttf 는 저장소에 안 넣는다).
# 글자 집합 = rtui/src/*.rs + dashboard/live/{app.js,index.html} 에 나오는 모든 글자 + ASCII 20~7E + Pretendard 고정폭 숫자 U+E071~E07A(fp.rs tnum).
#   ponytail: 화면 문구를 소스에서 긁는다 -- 서버가 보내는 글자(코인 이름 등)가 소스에 없는 글자면 egui 기본 글꼴로 떨어진다. 그럴 땐 여기 더한다.
# 쓰는 도구 = fonttools pyftsubset(개발기에 이미 있음). 힌팅은 뺀다(egui 는 힌팅을 안 씀 · draw.rs canvas_like 도 끔).
set -euo pipefail
cd "$(dirname "$0")"
W=../..
python3 -I - "$W" > chars.txt <<'EOF'
import pathlib, sys
w = pathlib.Path(sys.argv[1])
fs = list((w / "rtui/src").glob("*.rs")) + [w / "dashboard/live/app.js", w / "dashboard/live/index.html"]
s = set("".join(f.read_text(encoding="utf-8") for f in fs))
s |= {chr(c) for c in range(0x20, 0x7F)} | {chr(c) for c in range(0xE071, 0xE07B)}
s -= {"\n", "\r", "\t"}
print("".join(sorted(s)), end="")
EOF
echo "글자 $(python3 -I -c "print(len(open('chars.txt', encoding='utf-8').read()))")개"
for f in Pretendard-Regular Pretendard-Bold Pretendard-ExtraBold JetBrainsMono-Regular JetBrainsMono-Bold; do
  pyftsubset "$f.ttf" --text-file=chars.txt --output-file="$f.sub.ttf" --no-hinting --layout-features='*' --name-IDs='*' --notdef-outline
  printf '%s  %s → %s 바이트\n' "$f" "$(stat -c %s "$f.ttf")" "$(stat -c %s "$f.sub.ttf")"
done
rm chars.txt
