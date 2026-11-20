#!/usr/bin/env bash
# 양성 대조: src/lgb.rs · src/dir4h.rs · src/risk_radar.rs 의 규칙을 하나씩 틀리게 바꿔 tests/lgb.rs · tests/dir4h.rs 가 실패하는지. 끝나면(중단돼도) 원본으로 되돌리고 해시로 확인.
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
  "risk_radar~거래대금 하한(vclip) 끔~*v = v.max(c);~*v = *v;"
  "risk_radar~긴 창 결측 허용 5%→10%~k[i] <= 0.05 * wf~k[i] <= 0.10 * wf"
  "risk_radar~주말 칸 없음~(hour + 24 * (dow >= 5) as i64)~(hour + 0 * (dow >= 5) as i64)"
  "risk_radar~급변 중복 제거 30→29분(> → >=)~i as i64 - prev > pp.alert_dedupe_min~i as i64 - prev >= pp.alert_dedupe_min"
  "risk_radar~조용 문턱 <= → <~ratio <= pp.quiet~ratio < pp.quiet"
  "risk_radar~B0 창 168→167~roll_std(&hr, 168, 150)~roll_std(&hr, 167, 150)"
  "risk_radar~평소 창 720→719~past[j.saturating_sub(719)..=j]~past[j.saturating_sub(718)..=j]"
  "risk_radar~IV 낡음 7200→3600초~const IV_STALE_S: i64 = 7200;~const IV_STALE_S: i64 = 3600;"
  "risk_radar~IV 분위 같은 값 (n+1)/2 → n/2~(less as f64 + (eq + 1) as f64 / 2.0)~(less as f64 + eq as f64 / 2.0)"
  "risk_radar~형성 중 DVOL 봉도 씀~dv.got[j] >= dv.ts[j] + 3_600_000 &&~true &&"
  "risk_radar~지연 문턱 180→240초~pub const STALE_DATA_S: f64 = 180.0;~pub const STALE_DATA_S: f64 = 240.0;"
  "risk_radar~급변 1분 문턱 4σ → 4σ 직전 분 아님(같은 분)~f15[i - 1].sqrt() / 15f64.sqrt()~f15[i].sqrt() / 15f64.sqrt()"
  "dirshort~분위 lerp 한쪽식만~if g >= 0.5 { b - d * (1.0 - g) } else { a + d * g }~a + d * g"
  "dirshort~분위 가상 색인 n·p+(1−p)−1~let vi = (n - 1) as f64 * p;~let vi = n as f64 * p + (1.0 + p * -1.0) - 1.0;"
  "dirshort~searchsorted left → right~s.partition_point(|&a| a < v)~s.partition_point(|&a| a <= v)"
  "dirshort~문턱 반올림 4→3~py_round(np_quantile(s, 0.2), 4)~py_round(np_quantile(s, 0.2), 3)"
  "dirshort~신호 >= → >~if v >= hi { 1 }~if v > hi { 1 }"
  "dirshort~float32 입력 안 함~f.cols[c][i] as f32 as f64).collect::<Vec<f64>>()~f.cols[c][i]).collect::<Vec<f64>>()"
  "dirrest~20시간 규칙 20→19~now_ms - mx > 20 * 3_600_000~now_ms - mx > 19 * 3_600_000"
  "dirrest~꼬리 300→299봉~if first { 9000 } else { 300 }~if first { 9000 } else { 299 }"
  "dirrest~metrics 밀기 5→0분~(\"openInterestHist\", 0, \"sumOpenInterest\", 5)~(\"openInterestHist\", 0, \"sumOpenInterest\", 0)"
  "dirrest~보관 32→31일~let merged = merge_k(self.k.get(&kk), new, 32);~let merged = merge_k(self.k.get(&kk), new, 31);"
  "dirrest~짧은 봉 공유 안 함~if !self.short.contains_key(&key) {~if true {"
  "dirrest~빈 피쳐 최신 봉 안 뺌~if payload[\"nan_feats\"].as_array().is_some_and(|a| !a.is_empty())~if false"
  "dirrest~중복 결정 다시 씀~.filter(|r| !seen.contains(&r.0))~.filter(|_| true)"
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
