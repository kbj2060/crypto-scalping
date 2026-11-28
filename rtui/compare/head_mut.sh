#!/usr/bin/env bash
# 양성 대조: 원본 app.js·styles.css 를 한 군데씩 바꿔 정답지를 다시 지으면(head_js.mjs → head_layout.py) rtui head::t::js_parity 가 실패해야 한다 · Rust 쪽은 head.rs 를 바꿔 지금 정답지로.
# 실행: unshare -rn bash -c 'ip link set lo up; bash rtui/compare/head_mut.sh'   (작업 폴더 = 워크트리 · 변이 파일은 tmp/mut_head 에) · ONLY_R=1 = Rust 변이만
set -u
export PATH=$HOME/.cargo/bin:$PATH

W=$PWD; T=$W/tmp/mut_head; mkdir -p "$T"
A=$W/dashboard/live/app.js; C=$W/dashboard/live/styles.css
cp rtui/compare/fixture/head_in.json "$T/in.bak"
test_() { (cd rtui && ulimit -v 14000000 && if [ -n "$1" ]; then export HEAD_JS=$1; fi && cargo test --release -j 4 --offline --lib head::t::js_parity > "$T/$2.log" 2>&1); }
run() {   # $1 이름 · $2 js|css · $3 sed 식
  local n=$1 kind=$2 ex=$3 aj=$A cs=$C
  if [ "$kind" = js ]; then aj=$T/app.js; sed "$ex" "$A" > "$aj"; cmp -s "$A" "$aj" && { echo "✗ $n: 바뀐 곳 없음"; return; }; fi
  if [ "$kind" = css ]; then cs=$T/s.css; sed "$ex" "$C" > "$cs"; cmp -s "$C" "$cs" && { echo "✗ $n: 바뀐 곳 없음"; return; }; fi
  TZ=Asia/Seoul APP_JS=$aj OUT=$T/html.json node rtui/compare/head_js.mjs > /dev/null || { echo "? $n: 정답지 짓기 실패"; return; }
  HEAD_HTML=$T/html.json STYLES_CSS=$cs HEAD_OUT=$T/js.json python3 -I rtui/compare/head_layout.py "$W" > /dev/null || { echo "? $n: 재기 실패"; return; }
  if test_ "$T/js.json" "$n"; then echo "✗ $n: 못 잡음"; else echo "✓ $n: 잡음 -- $(grep -m1 -A1 'panicked' "$T/$n.log" | tail -1 | cut -c1-110)"; fi
}
if [ -z "${ONLY_R:-}" ]; then
run sess_asia js 's/const sAsiaOn = Number(sess.session_asia || 0) >= 0.5;/const sAsiaOn = Number(sess.session_asia || 0) >= 1.5;/'
run coin_case js 's/const cn = el("fpCoinSym"); if (cn) cn.textContent = String(activeSnapshotAsset || "eth").toUpperCase();/const cn = el("fpCoinSym"); if (cn) cn.textContent = String(activeSnapshotAsset || "eth").toLowerCase();/'
run badge_n js 's/  if (!active.length) { badge.style.display = "none"; return; }/  if (active.length < 2) { badge.style.display = "none"; return; }/'
run spk_show js 's/const SPIKE_SHOW_MIN = 40,/const SPIKE_SHOW_MIN = 38,/'
run spk_same js 's/const same = rs.length === 2 \&\& Math.sign(rs\[0\]\[1\]\[2\]) === Math.sign/const same = rs.length === 2 \&\& Math.sign(rs[0][1][2]) !== Math.sign/'
run pos_dig js 's/^const lineDig = (v) => (v >= 100 ? 2 : 4);/const lineDig = (v) => (v >= 100 ? 2 : 3);/'
run pos_margin js 's/mPct.toFixed(mPct >= 10 ? 0 : 1)/mPct.toFixed(mPct >= 20 ? 0 : 1)/'
run pos_basis js 's/const basis = bb > 0 ? bb : mk > 0/const basis = bb > 0 ? 1 : mk > 0/'
run c_sep css 's/^.session-sep { margin: 0 8px;/.session-sep { margin: 0 7px;/'
run c_coin css 's/font: 800 30px\/1 var(--font-sans); letter-spacing: .06em;/font: 800 30px\/1 var(--font-sans); letter-spacing: .05em;/'
run c_alert_gap css 's/transform: translate(-50%, -50%); display: flex; gap: 8px; }/transform: translate(-50%, -50%); display: flex; gap: 9px; }/'
run c_more css 's/  .snap-head #assetMore { margin-left: -12px; }/  .snap-head #assetMore { margin-left: -10px; }/'
run c_pos_gap css 's/^.fp-pos > div { display: flex; align-items: baseline; gap: 8px; }/.fp-pos > div { display: flex; align-items: baseline; gap: 9px; }/'
run c_chip css 's/letter-spacing: .08em; padding: 3px 6px; border-radius: 4px;/letter-spacing: .08em; padding: 3px 7px; border-radius: 4px;/'
cp "$T/in.bak" rtui/compare/fixture/head_in.json
fi
rmut() {   # Rust 쪽 변이: head.rs 를 바꿔 지금 정답지로
  local n=$1 ex=$2 f=rtui/src/head.rs
  cp $f "$T/head.rs.bak"; sed -i "$ex" $f
  cmp -s $f "$T/head.rs.bak" && { echo "✗ $n: 바뀐 곳 없음"; return; }
  if test_ "" "$n"; then echo "✗ $n: 못 잡음"; else echo "✓ $n: 잡음 -- $(grep -m1 -A1 'panicked' "$T/$n.log" | tail -1 | cut -c1-110)"; fi
  cp "$T/head.rs.bak" $f
}
rmut r_led 's/t(\&mut v, x + 16.0, iy, nm, 11.0, 700, true/t(\&mut v, x + 15.0, iy, nm, 11.0, 700, true/'
rmut r_shrink 's/let (o, ti) = (ma + r.tb - r.s, r.tb - 6.0);/let (o, ti) = (ma + r.tb - r.s, r.tb);/'
rmut r_pos_gap 's/t(\&mut v, cw + 8.0, gt(0.0, 26.0, 26.0, true)/t(\&mut v, cw + 10.0, gt(0.0, 26.0, 26.0, true)/'
rmut r_spk 's/^const SPIKE_SHOW_MIN: f64 = 40.0;/const SPIKE_SHOW_MIN: f64 = 42.0;/'   # 입력 3 = 41분 지난 나스닥(서버 시계 +2초)
rmut r_base 's/let base = ob.max(os);/let base = ob.min(os);/'
