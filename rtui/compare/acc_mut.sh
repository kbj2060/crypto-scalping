#!/usr/bin/env bash
# 양성 대조: 원본 app.js·styles.css 를 한 군데씩 바꿔 정답지를 다시 재면 rtui acc::t::js_parity 가 실패해야 한다 · Rust acc.rs 를 바꿔도 실패해야 한다.
# 변이마다 지어낸 입력이 그 분기에 닿는다(34체결 순환 · 7배 · 증거금 37% · 손실 3건 이름표 · 잘린 순환 파편 …).
# 실행: unshare -rn bash -c 'ip link set lo up; bash rtui/compare/acc_mut.sh'   (작업 폴더 = 워크트리 · 변이 파일은 tmp/mut_acc 에)
set -u
export PATH=$HOME/.cargo/bin:$PATH

W=$PWD; T=$W/tmp/mut_acc; mkdir -p "$T"
A=$W/dashboard/live/app.js; C=$W/dashboard/live/styles.css
cp rtui/compare/fixture/acc_in.json "$T/in.bak"
run() {   # $1 이름 · $2 js|css · $3 sed 식
  local n=$1 kind=$2 ex=$3 aj=$A cs=$C
  if [ "$kind" = js ]; then aj=$T/app.js; sed "$ex" "$A" > "$aj"; cmp -s "$A" "$aj" && { echo "✗ $n: 바뀐 곳 없음"; return; }; fi
  if [ "$kind" = css ]; then cs=$T/s.css; sed "$ex" "$C" > "$cs"; cmp -s "$C" "$cs" && { echo "✗ $n: 바뀐 곳 없음"; return; }; fi
  APP_JS=$aj STYLES_CSS=$cs ACC_OUT=$T/js.json python3 -I rtui/compare/acc_layout.py "$W" > /dev/null || { echo "? $n: 재기 실패"; return; }
  if (cd rtui && ulimit -v 14000000 && ACC_JS=$T/js.json cargo test --release -j 4 --offline --lib acc::t::js_parity > "$T/$n.log" 2>&1); then echo "✗ $n: 못 잡음"; else echo "✓ $n: 잡음 -- $(grep -m1 -A3 'panicked' "$T/$n.log" | tail -n +2 | head -3 | tr '\n' ' ' | cut -c1-110)"; fi
}
run cal_alpha js 's/const a = o ? Math.round((0.1 + 0.6 \* Math.sqrt/const a = o ? Math.round((0.1 + 0.5 * Math.sqrt/'
run pct_dp js 's/Math.abs(p \* 100) < 10 ?/Math.abs(p * 100) < 5 ?/'
run legs30 js 's/if (legs.length <= 30) {/if (legs.length <= 40) {/'
run lab90 js 's/room = labX.every((q) => Math.abs(q - x) > 90)/room = labX.every((q) => Math.abs(q - x) > 30)/'
run pick js 's/(!!cur.truncated_open === !!t.truncated_open \&\& t.entry_time > cur.entry_time)/(t.entry_time > cur.entry_time)/'
run half js 's/Math.min(100, Math.abs(v) \/ m \* 100) \/ 2 : 0/Math.min(100, Math.abs(v) \/ m * 100) \/ 3 : 0/'
run peak6 js 's/t.peak_x > 6 ?/t.peak_x > 7 ?/'
run warn75 js 's/p.usedPct >= 75 ?/p.usedPct >= 30 ?/'
run dur60 js '/const acctDur/s/return m < 60/return m < 30/'
run tiles_gap css 's/.cy-tiles { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px;/.cy-tiles { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 9px;/'
run day_gap css 's/minmax(0, 1fr) auto; gap: 10px; align-items: baseline; padding: 4px 6px;/minmax(0, 1fr) auto; gap: 11px; align-items: baseline; padding: 4px 6px;/'
run as_col css 's/.as-r { display: grid; grid-template-columns: 74px/.as-r { display: grid; grid-template-columns: 76px/'
run mid_min css 's/html.fit1 #acctCard > .acct-mid { flex: .75 1 0; min-height: 300px; }/html.fit1 #acctCard > .acct-mid { flex: .75 1 0; min-height: 310px; }/'
run cal_gap css 's/minmax(0, 1.1fr); gap: 3px; }/minmax(0, 1.1fr); gap: 4px; }/'
run awp_gap css 's/.aw-p { display: grid; grid-template-columns: auto minmax(40px, 1fr) auto; gap: 5px 12px;/.aw-p { display: grid; grid-template-columns: auto minmax(40px, 1fr) auto; gap: 6px 12px;/'
cp "$T/in.bak" rtui/compare/fixture/acc_in.json
rmut() {   # Rust 쪽 변이: acc.rs 를 바꿔 지금 정답지로
  local n=$1 ex=$2 f=rtui/src/acc.rs
  cp $f "$T/acc.rs.bak"; sed -i "$ex" $f
  cmp -s $f "$T/acc.rs.bak" && { echo "✗ $n: 바뀐 곳 없음"; return; }
  if (cd rtui && ulimit -v 14000000 && cargo test --release -j 4 --offline --lib acc::t::js_parity > "$T/$n.log" 2>&1); then echo "✗ $n: 못 잡음"; else echo "✓ $n: 잡음 -- $(grep -m1 -A3 'panicked' "$T/$n.log" | tail -n +2 | head -3 | tr '\n' ' ' | cut -c1-110)"; fi
  cp "$T/acc.rs.bak" $f
}
rmut r_tracks 's/let starts: Vec<f64> = fr.iter().map(|q| { let s0 = fl(x0 + acc);/let starts: Vec<f64> = fr.iter().map(|q| { let s0 = (x0 + acc).round();/'
rmut r_rest 's/let rest = (y - y0) + 10.0 + 24.0;/let rest = (y - y0) + 10.0 + 25.0;/'
rmut r_sub 's/let sy = r0 + 25.0;/let sy = r0 + 26.0;/'
rmut r_wrap 's/let two = cx.w(&tm, mono) > 104.0/let two = cx.w(\&tm, mono) > 124.0/'
rmut r_mid 's/let mid = jround(h \* 0.66);/let mid = (h * 0.66).floor();/'
