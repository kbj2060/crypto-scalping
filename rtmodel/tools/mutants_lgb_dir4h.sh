#!/usr/bin/env bash
# 양성 대조: src/lgb.rs · src/dir4h.rs 의 규칙을 하나씩 틀리게 바꿔 tests/lgb.rs · tests/dir4h.rs 가 실패하는지. 끝나면(중단돼도) 원본으로 되돌리고 해시로 확인.
#   bash rtmodel/tools/mutants_lgb_dir4h.sh        (정답지: 각 시험 머리말)
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
M=(
  "lgb~입력 0 바꿈 없음~let x: Vec<f64> = x.iter().map(|&v| if v.abs() <= K_ZERO { 0.0 } else { v }).collect();~let x: Vec<f64> = x.to_vec();"
  "lgb~NaN→0 없음~if v.is_nan() && miss != 2 {~if false {"
  "lgb~기본 방향 반대~d & 2 != 0 } else~d & 2 == 0 } else"
  "lgb~부등호 <= → <~v <= self.thr[i]~v < self.thr[i]"
  "lgb~영 문턱 f64 1e-35~const K_ZERO: f64 = 1e-35f32 as f64;~const K_ZERO: f64 = 1e-35;"
  "lgb~척도로 나눔 → 역수 곱~s += m.predict(x) / sc;~s += m.predict(x) * (1.0 / sc);"
  "dir4h~rolling mean Kahan 끔~ca = t - sx - y;~ca = 0.0;"
  "dir4h~rz 창 2016→2015~rz(&ft[\"O_doi1h\"], 2016, 288)~rz(&ft[\"O_doi1h\"], 2015, 288)"
  "dir4h~quantile 보간 없음~v[k] + (v[k + 1] - v[k]) * (f - k as f64)~v[k]"
  "dir4h~ewm adjust=False 꼴~old += 1.0;~old = 1.0;"
  "dir4h~nanstd 분모 n−1~(ss / k12 as f64).sqrt()~(ss / (k12 as f64 - 1.0)).sqrt()"
  "dir4h~일봉 경계 <= → <~d0 + DAY <= t~d0 + DAY < t"
  "dir4h~f30 경계 >= → >~tt[i] - d0 >= 1_800_000~tt[i] - d0 > 1_800_000"
  "dir4h~inf→NaN 안 함~if v.is_infinite() { NAN } else { *v }~*v"
  "dir4h~float32 입력 안 함~f.cols[c.as_str()][i] as f32 as f64~f.cols[c.as_str()][i]"
  "dir4h~bp 반올림 2→3~py_round(p * k, 2)~py_round(p * k, 3)"
)
for m in "${M[@]}"; do
  IFS='~' read -r mod name from to <<< "$m"
  [ -n "${1:-}" ] && ! grep -qE "$1" <<< "$name" && continue
  SRC=src/$mod.rs; BAK=$(mktemp); cp "$SRC" "$BAK"; H0=$(md5sum < "$SRC")
  trap 'cp "$BAK" "$SRC"' EXIT
  python3 - "$SRC" "$from" "$to" <<'PY' || { echo "못 바꿈: $name"; cp "$BAK" "$SRC"; continue; }
import sys
p, a, b = sys.argv[1:]
s = open(p).read()
assert s.count(a) == 1, f"{a!r} {s.count(a)}회"
open(p, "w").write(s.replace(a, b))
PY
  sleep 1; touch "$SRC"
  if cargo test -q --release --no-run -j 2 --test "$mod" >/dev/null 2>&1; then
    out=$( ( ulimit -v 6000000; timeout 900 cargo test -q --release -j 2 --test "$mod" 2>&1 ) )
    if grep -q "test result: ok" <<< "$out"; then echo "놓침  [$mod] $name"; else echo "잡음  [$mod] $name :: $(grep -m1 -A1 panicked <<< "$out" | tail -1 | head -c 140)"; fi
  else echo "컴파일 실패: $name"; fi
  cp "$BAK" "$SRC"; [ "$(md5sum < "$SRC")" = "$H0" ] || echo "🔴복원 실패 $SRC"; rm -f "$BAK"; trap - EXIT
done
echo "끝 · 원본 복원 확인"
