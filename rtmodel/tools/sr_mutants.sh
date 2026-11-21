#!/usr/bin/env bash
# 양성 대조: src/sr_lean.rs 의 식·상태 기계 규칙을 하나씩 틀리게 바꿔 tests/sr_lean.rs 가 실패하는지 본다. 끝나면(중단돼도) 원본으로 되돌리고 해시로 확인.
#   bash rtmodel/tools/sr_mutants.sh [이름 정규식]      (정답지가 먼저 있어야 한다 -- tests/sr_lean.rs 머리말 · 정규식 = 그 변형만)
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
SRC=src/sr_lean.rs
BAK=$(mktemp)
cp "$SRC" "$BAK"; H0=$(md5sum < "$SRC")
trap 'cp "$BAK" "$SRC"; rm -f "$BAK"' EXIT
M=(
  "σ 창 1440→1441~roll_std(&dif, 1440, 1200)~roll_std(&dif, 1441, 1200)"
  "σ min_periods 1200→1440~roll_std(&dif, 1440, 1200)~roll_std(&dif, 1440, 1440)"
  "ddof 1→0~ssq / (nobs - 1.0)~ssq / nobs"
  "Kahan 보정 끔(더하기)~ca = t + mean - y;~ca = 0.0;"
  "같은 값 연속 규칙 끔~same as f64 >= nobs~false"
  "σ shift(1) 빠짐~sd[r - 1] * sq60~sd[r] * sq60"
  "프레임 3,000→2,999행(roll_var 시작점)~const KEEP_M: usize = 3000;~const KEEP_M: usize = 2999;"
  "레벨 창 +1~roll_ext(&h, look as usize, minp, true)~roll_ext(&h, look as usize + 1, minp, true)"
  "레벨 min_periods 98%→100%~(look as f64 * 0.98) as usize~look as usize"
  "접근 폭 0.1σ→0.09σ~const D_AP: f64 = 0.1;~const D_AP: f64 = 0.09;"
  "판정 배리어 0.10σ(약 6bp)→0.09σ~const DB: f64 = 0.10;~const DB: f64 = 0.09;"
  "판정 창 300→299초~const HZ: i64 = 300;~const HZ: i64 = 299;"
  "결정 지연 5→4초~const W: i64 = 5;~const W: i64 = 4;"
  "중복 억제 60→59분~const GAP: usize = 60;~const GAP: usize = 59;"
  "피쳐 창 300→301초~const WINS: [usize; 8] = [1, 3, 10, 30, 60, 300, 900, 3600];~const WINS: [usize; 8] = [1, 3, 10, 30, 60, 301, 900, 3600];"
  "돌파 부등호 > → >=~if kind == Kind::Brk { v > th } else { v >= th }~if kind == Kind::Brk { v >= th } else { v >= th }"
  "접근 부등호 <= → <~set(Kind::Apr, -1, &|r| l[r] <= dn(r)~set(Kind::Apr, -1, &|r| l[r] < dn(r)"
  "맥락 부족 문턱 1430→1420~< 1430;~< 1420;"
  "앞 60분 사건 수 창 3600→3599~x >= a - 3600~x >= a - 3599"
  "직전 체결가 찾기 3599→3000초~(a - 3599..a).rev()~(a - 3000..a).rev()"
  "주말 >= 5 → >= 6~.rem_euclid(7) >= 5)~.rem_euclid(7) >= 6)"
  "p 반올림 6→5자리~jf(py_round(p, 6))~jf(py_round(p, 5))"
  "r 반올림 2→3자리~py_round(e.side as f64 * lr * 1e4, 2)~py_round(e.side as f64 * lr * 1e4, 3)"
  "빈 분(v=0) NaN 처리 빠짐~let dead = r.v <= 0.0;~let dead = r.v < 0.0;"
  "다시 확인이 v=0 분도 덮음~if mi <= t0m && self.m.contains_key(&mi) && r.v > 0.0~if mi <= t0m && self.m.contains_key(&mi) && r.v >= 0.0"
  "준비 끝 +60 → +59~t0 + PRE as i64 + PAD + 60~t0 + PRE as i64 + PAD + 59"
  "누적합 첫 값 0.0+x(-0.0 부호)~s = if k == 0 { *v } else { s + v };~s += v;"
)
for m in "${M[@]}"; do
  IFS='~' read -r name from to <<< "$m"
  [ -n "${1:-}" ] && ! grep -qE "$1" <<< "$name" && continue
  cp "$BAK" "$SRC"
  python3 - "$SRC" "$from" "$to" <<'PY' || { echo "못 바꿈: $name"; continue; }
import sys
p, a, b = sys.argv[1:]
s = open(p).read()
assert s.count(a) == 1, f"{a!r} {s.count(a)}회"
open(p, "w").write(s.replace(a, b))
PY
  sleep 1; touch "$SRC"   # 같은 초 안에 바뀐 파일을 cargo 가 못 알아채 앞 변형의 실행 파일을 다시 돌린 일이 있었다
  cargo test -q --release --no-run -j 2 --test sr_lean >/dev/null 2>&1 || { echo "컴파일 실패: $name"; continue; }
  out=$( ( ulimit -v 4000000; timeout 600 cargo test -q --release -j 2 --test sr_lean 2>&1 ) )
  if grep -q "test result: ok" <<< "$out"; then echo "놓침  $name"; else echo "잡음  $name :: $(grep -m1 -A1 panicked <<< "$out" | tail -1 | head -c 160)"; fi
done
cp "$BAK" "$SRC"
[ "$(md5sum < "$SRC")" = "$H0" ] && echo "원본 복원 확인(md5 같음)"
