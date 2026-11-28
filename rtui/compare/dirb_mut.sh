#!/usr/bin/env bash
# 양성 대조: 원본 app.js·styles.css 를 한 군데씩 바꿔 정답지를 다시 지으면 rtui dirb::t::js_parity 가 실패해야 한다(바뀐 분기에 닿는 입력이 dirb_in.json 에 있다).
# 실행: unshare -rn bash -c 'ip link set lo up; bash rtui/compare/dirb_mut.sh'   (작업 폴더 = 워크트리 · 변이 파일은 tmp/mut_dirb 에) · 11-29 결과 JS 12/12 · CSS 3/3 잡음
set -u
export PATH=$HOME/.cargo/bin:$PATH
W=$PWD; T=$W/tmp/mut_dirb; mkdir -p "$T"
A=$W/dashboard/live/app.js; C=$W/dashboard/live/styles.css
run() {   # $1 이름 · $2 js|css · $3 sed 식
  local n=$1 kind=$2 ex=$3 aj=$A cs=$C
  if [ "$kind" = js ]; then aj=$T/app.js; sed "$ex" "$A" > "$aj"; cmp -s "$A" "$aj" && { echo "✗ $n: 바뀐 곳 없음"; return; }; fi
  if [ "$kind" = css ]; then cs=$T/s.css; sed "$ex" "$C" > "$cs"; cmp -s "$C" "$cs" && { echo "✗ $n: 바뀐 곳 없음"; return; }; fi
  APP_JS=$aj SRC=$T/src.js node rtui/compare/dirb_js.mjs > /dev/null || { echo "? $n: 자르기 실패"; return; }
  DIRB_SRC=$T/src.js STYLES_CSS=$cs DIRB_OUT=$T/js.json python3 -I rtui/compare/dirb_layout.py "$W" > /dev/null || { echo "? $n: 재기 실패"; return; }
  if (cd rtui && ulimit -v 14000000 && DIRB_JS=$T/js.json cargo test --release -j 4 --offline --lib dirb::t::js_parity > "$T/$n.log" 2>&1); then echo "✗ $n: 못 잡음"; else echo "✓ $n: 잡음 -- $(grep -m1 -A2 'panicked' "$T/$n.log" | tail -n +2 | head -1 | cut -c1-110)"; fi
}
run row_h js 's/w = FPB_W, rh = 23, head = rows.length ? 44 : 30/w = FPB_W, rh = 24, head = rows.length ? 44 : 30/'
run head_44 js 's/w = FPB_W, rh = 23, head = rows.length ? 44 : 30/w = FPB_W, rh = 23, head = rows.length ? 42 : 30/'
run stale_180 js 's/(Number(S.computed_at) || 0) > 180)/(Number(S.computed_at) || 0) > 120)/'
run big_80 js 's/big = l.g \&\& l.g.v >= 80, c = l.dir/big = l.g \&\& l.g.v >= 90, c = l.dir/'
run word_break js 's/\[dns, -1, "깨면"\]\]/[dns, -1, "깨지면"]]/'
run wait_txt js 's/${q.wait ? " 마감 대기" : ""}/${q.wait ? " 대기" : ""}/'
run stripe js '0,/"fill-opacity": i % 2 ? 0 : 0.035/s//"fill-opacity": i % 2 ? 0.035 : 0/'
run room_24 js 's/if (room > 24) mk(rg/if (room > 60) mk(rg/'
run fit_max js 's/measureTextW(t === txt ? t : t + "…", font) > max/measureTextW(t === txt ? t : t + "…", font) > max - 4/'
run tally_gap js 's/width: Math.max(2, uw - (dn ? 1 : 0))/width: Math.max(2, uw - (dn ? 2 : 0))/'
run fold_head js 's/h = fold ? head : hb + dirReadH(R);/h = fold ? head + 4 : hb + dirReadH(R);/'
run hot_grade js 's/l.grade === "근거" || l.grade === "약함"/l.grade === "근거"/'
run panel_css css 's/--panel-strong: rgba(46, 52, 62, 0.92);/--panel-strong: rgba(46, 52, 62, 0.80);/'
run warn_css css '0,/--warn: #e99c57;/s//--warn: #e99c50;/'
run line_css css '0,/--line: rgba(203, 209, 227, 0.18);/s//--line: rgba(203, 209, 227, 0.30);/'
