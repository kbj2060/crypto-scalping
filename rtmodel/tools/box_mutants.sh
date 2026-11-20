#!/usr/bin/env bash
# 양성 대조: src/box_hold.rs 의 식을 하나씩 틀리게 바꿔 tests/box_hold.rs 가 실패하는지 본다. 끝나면(중단돼도) 원본으로 되돌리고 해시로 확인.
#   bash rtmodel/tools/box_mutants.sh      (정답지가 먼저 있어야 한다 -- tests/box_hold.rs 머리말)
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
SRC=src/box_hold.rs
BAK=$(mktemp)
cp "$SRC" "$BAK"; H0=$(md5sum < "$SRC")
trap 'cp "$BAK" "$SRC"; rm -f "$BAK"' EXIT
M=(
  "창 길이 +1(rolling)~rolling_ext(&h, w, true)~rolling_ext(&h, w + 1, true)"
  "lsig 1h 창 60→59~(\"V_lsig1h\", lsig(60))~(\"V_lsig1h\", lsig(59))"
  "σ24 창 1440→1439~npmean(tail(&r2, 1440)).sqrt()~npmean(tail(&r2, 1439)).sqrt()"
  "평균 분모 n−1(ddof 꼴)~npsum(a) / a.len() as f64~npsum(a) / (a.len() - 1) as f64"
  "롤링 NaN 무시~last_nan.map_or(true, |k| k + w <= i)~true"
  "중앙값 짝수 → 위쪽 하나~(v[k - 1] + v[k]) / 2.0~v[k]"
  "발표 부등호 <= → <~(\"M_major_in\", (nm <= wf)~(\"M_major_in\", (nm < wf)"
  "실업수당 부등호 <= → <~(next_min(tc, &claims) <= wf)~(next_min(tc, &claims) < wf)"
  "직전 발표 60분 <= → <~prev_min(tc, &major) <= 60.0~prev_min(tc, &major) < 60.0"
  "R_edge min → max~if 1.0 - p < p~if 1.0 - p > p"
  "p 반올림 4→3자리~\"p\": rnd(p, 4)~\"p\": rnd(p, 3)"
  "half_pct 반올림 3→2자리~rnd(0.5 * rng / px * 100.0, 3)~rnd(0.5 * rng / px * 100.0, 2)"
  "자릿수 문턱 >= 10 → > 10~if px >= 10.0 { 2 }~if px > 10.0 { 2 }"
  "Platt90 신선도 40→45일~40 * 86_400~45 * 86_400"
  "변동성 묵음 15→25분~15.0 * 60_000.0~25.0 * 60_000.0"
  "시각 분 무시~hour as f64 + minute as f64 / 60.0~hour as f64"
  "짝 합 → 차례 합~if n < 8 {~if n < usize::MAX {"
)
for m in "${M[@]}"; do
  IFS='~' read -r name from to <<< "$m"
  cp "$BAK" "$SRC"
  python3 - "$SRC" "$from" "$to" <<'EOF' || { echo "못 바꿈: $name"; continue; }
import sys
p, a, b = sys.argv[1:]
s = open(p).read()
assert s.count(a) == 1, f"{a!r} {s.count(a)}회"
open(p, "w").write(s.replace(a, b))
EOF
  sleep 1; touch "$SRC"   # 같은 초 안에 바뀐 파일을 cargo 가 못 알아채 앞 변형의 실행 파일을 다시 돌린 일이 있었다
  cargo test -q --no-run -j 2 --test box_hold >/dev/null 2>&1 || { echo "컴파일 실패: $name"; continue; }
  out=$( ( ulimit -v 4000000; timeout 300 cargo test -q -j 2 --test box_hold box_hold_parity 2>&1 ) )
  if grep -q "test result: ok" <<< "$out"; then echo "놓침  $name"; else echo "잡음  $name :: $(grep -m1 -E "panicked|assert|오차|불일치|같음|필드" <<< "$out" | grep -v '^thread' | head -c 160)$(grep -m1 -A1 panicked <<< "$out" | tail -1 | head -c 160)"; fi
done
cp "$BAK" "$SRC"
[ "$(md5sum < "$SRC")" = "$H0" ] && echo "원본 복원 확인(md5 같음)"
