#!/usr/bin/env bash
# 양성 대조: 원본 app.js·styles.css 를 한 군데씩 바꿔 정답지를 다시 지으면 rtui mt::t::js_parity 가 실패해야 한다.
# 실행: unshare -rn bash -c 'ip link set lo up; bash rtui/compare/mt_mut.sh'   (작업 폴더 = 워크트리 · 변이 파일은 tmp/mut 에) · 11-29 결과 18/18 잡음
set -u
export PATH=$HOME/.cargo/bin:$PATH

W=$PWD; T=$W/tmp/mut; mkdir -p "$T"
A=$W/dashboard/live/app.js; C=$W/dashboard/live/styles.css
cp rtui/compare/fixture/mt_in.json "$T/in.bak"
run() {   # $1 이름 · $2 js|css · $3 sed 식
  local n=$1 kind=$2 ex=$3 aj=$A cs=$C
  if [ "$kind" = js ]; then aj=$T/app.js; sed "$ex" "$A" > "$aj"; cmp -s "$A" "$aj" && { echo "✗ $n: 바뀐 곳 없음"; return; }; fi
  if [ "$kind" = css ]; then cs=$T/s.css; sed "$ex" "$C" > "$cs"; cmp -s "$C" "$cs" && { echo "✗ $n: 바뀐 곳 없음"; return; }; fi
  TZ=Asia/Seoul APP_JS=$aj OUT=$T/html.json node rtui/compare/mt_js.mjs > /dev/null || { echo "? $n: 정답지 짓기 실패"; return; }
  MT_HTML=$T/html.json STYLES_CSS=$cs MT_OUT=$T/js.json python3 -I rtui/compare/mt_layout.py "$W" > /dev/null || { echo "? $n: 재기 실패"; return; }
  if (cd rtui && ulimit -v 14000000 && MT_JS=$T/js.json MT_HTMLJS=$T/html.json cargo test --release -j 4 --offline --lib mt::t::js_parity > "$T/$n.log" 2>&1); then echo "✗ $n: 못 잡음"; else echo "✓ $n: 잡음 -- $(grep -m1 -A2 'panicked' "$T/$n.log" | tail -n +2 | head -1 | cut -c1-110)"; fi
}
run bp_digits js 's/const d = unit === "bp" ? 0 :/const d = unit === "bp" ? 1 :/'
run gauge_clamp js 's/Math.max(-3, Math.min(3, zv)) \/ 3 \* 50/Math.max(-2.5, Math.min(2.5, zv)) \/ 3 * 50/'
run overlay_lead js 's/off = lead ? 14 : 0, x1 = lead ? 14 : 0/off = lead ? 13 : 0, x1 = lead ? 14 : 0/'
run day_pad js 's/Y = (v) => 3 + (hi - v) \/ rg \* (H - 6)/Y = (v) => 4 + (hi - v) \/ rg * (H - 6)/'
run first_sel js 's/\[a\] || {}).state === "count"));/[a] || {}).state === "zero"));/'
run label_group js 's/if (g \&\& e.t - g.t < 7200)/if (g \&\& e.t - g.t < 3600)/'
run zn_late js 's/lab = age < 1800 ? `${Math.max(1, Math.round(age \/ 60))}분 지연` : "늦음"; if (age >= 1800) st = "bad";/lab = age < 1500 ? `${Math.max(1, Math.round(age \/ 60))}분 지연` : "늦음"; if (age >= 1500) st = "bad";/'
run head_next js 's/\.filter((e) => e.t > Date.now()).sort((a, b) => a.t - b.t)\[0\];/.filter((e) => e.t > Date.now()).sort((a, b) => b.t - a.t)[0];/'
run col_118 css 's/minmax(104px, 1fr) 66px 118px 84px repeat(4, 52px) 96px/minmax(104px, 1fr) 66px 120px 84px repeat(4, 52px) 96px/'
run fam_pad css 's/letter-spacing: .1em; color: var(--muted); padding: 10px 0 4px; }/letter-spacing: .1em; color: var(--muted); padding: 9px 0 4px; }/'
run why_col css 's/grid-template-columns: 74px 40px minmax(0, 1fr); gap: 10px;/grid-template-columns: 76px 40px minmax(0, 1fr); gap: 10px;/'
run meta_lh css 's/.mt-meta { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.55; }/.mt-meta { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.5; }/'
run tile_off css 's/inset 0 0 0 1.5px color-mix(in srgb, var(--warn) 65%, transparent)/inset 0 0 0 1px color-mix(in srgb, var(--warn) 65%, transparent)/'
run head_gap css 's/.fr-mtg { display: grid; grid-template-columns: auto auto auto; gap: 4px 10px;/.fr-mtg { display: grid; grid-template-columns: auto auto auto; gap: 4px 11px;/'
cp "$T/in.bak" rtui/compare/fixture/mt_in.json
rmut() {   # Rust 쪽 변이: mt.rs 를 바꿔 지금 정답지로
  local n=$1 ex=$2 f=rtui/src/mt.rs
  cp $f "$T/mt.rs.bak"; sed -i "$ex" $f
  cmp -s $f "$T/mt.rs.bak" && { echo "✗ $n: 바뀐 곳 없음"; return; }
  if (cd rtui && ulimit -v 14000000 && cargo test --release -j 4 --offline --lib mt::t::js_parity > "$T/$n.log" 2>&1); then echo "✗ $n: 못 잡음"; else echo "✓ $n: 잡음 -- $(grep -m1 -A2 'panicked' "$T/$n.log" | tail -n +2 | head -1 | cut -c1-110)"; fi
  cp "$T/mt.rs.bak" $f
}
rmut r_halfleading 's/line_top + ((lh - cont(z, mono)) \/ 2.0).floor()/line_top + ((lh - cont(z, mono)) \/ 2.0).round()/'
rmut r_table 's/mx.iter().map(|q| q \* dw.max(tot) \/ tot)/mx.iter().map(|q| q + (dw.max(tot) - tot) \/ 7.0)/'
rmut r_break 's/    cjk(a) || cjk(b) || a == /    cjk(a) \&\& cjk(b) || a == /'
rmut r_fr 's/let fr = ((mx_h - auto) \/ nrows as f64).max(22.0);/let fr = ((mx_h - auto) \/ nrows as f64).floor().max(22.0);/'
