#!/usr/bin/env bash
# 양성 대조: 원본 app.js·styles.css 를 한 군데씩 바꿔 정답지를 다시 지으면 rtui news::t::js_parity 가 실패해야 한다 · Rust news.rs 를 바꿔도 실패해야 한다.
# 실행: unshare -rn bash -c 'ip link set lo up; bash rtui/compare/news_mut.sh'   (작업 폴더 = 워크트리 · 변이 파일은 tmp/mut_news 에)
set -u
export PATH=$HOME/.cargo/bin:$PATH

W=$PWD; T=$W/tmp/mut_news; mkdir -p "$T"
A=$W/dashboard/live/app.js; C=$W/dashboard/live/styles.css
cp rtui/compare/fixture/news_in.json "$T/in.bak"
run() {   # $1 이름 · $2 js|css · $3 sed 식
  local n=$1 kind=$2 ex=$3 aj=$A cs=$C
  if [ "$kind" = js ]; then aj=$T/app.js; sed "$ex" "$A" > "$aj"; cmp -s "$A" "$aj" && { echo "✗ $n: 바뀐 곳 없음"; return; }; fi
  if [ "$kind" = css ]; then cs=$T/s.css; sed "$ex" "$C" > "$cs"; cmp -s "$C" "$cs" && { echo "✗ $n: 바뀐 곳 없음"; return; }; fi
  TZ=Asia/Seoul APP_JS=$aj OUT=$T/html.json node rtui/compare/news_js.mjs > /dev/null || { echo "? $n: 정답지 짓기 실패"; return; }
  NEWS_HTML=$T/html.json STYLES_CSS=$cs NEWS_OUT=$T/js.json python3 -I rtui/compare/news_layout.py "$W" > /dev/null || { echo "? $n: 재기 실패"; return; }
  if (cd rtui && ulimit -v 14000000 && NEWS_JS=$T/js.json cargo test --release -j 4 --offline --lib news::t::js_parity > "$T/$n.log" 2>&1); then echo "✗ $n: 못 잡음"; else echo "✓ $n: 잡음 -- $(grep -m1 -A2 'panicked' "$T/$n.log" | tail -n +2 | head -1 | cut -c1-110)"; fi
}
run top6 js 's/w(y) - w(x)).slice(0, 6)/w(y) - w(x)).slice(0, 5)/'
run pin5 js 's/const pins = incs.slice(0, 5)/const pins = incs.slice(0, 4)/'
run rel05 js 's/const newsRel = (it) => (it.relevant ?? 0) >= 0.5;/const newsRel = (it) => (it.relevant ?? 0) >= 0.6;/'
run dot_r js 's/const r = 2.2 + 3.6 \* (it.relevant ?? 0)/const r = 2.3 + 3.6 * (it.relevant ?? 0)/'
run imp_bar js 's/imp >= k - 0.5 ? "f" : ""/imp >= k - 0.4 ? "f" : ""/'
run ago_h js 's/`${Math.floor(m \/ 60)}시간${m % 60/`${Math.round(m \/ 60)}시간${m % 60/'
run tl_gap js 's/if (l0 > lastX\[row\] + 8)/if (l0 > lastX[row] + 40)/'
run ev_merge js 's/const same = acc.find((a) => Math.abs(a.t - e.t) < 60e3);/const same = acc.find((a) => Math.abs(a.t - e.t) < 10e3);/'
run row_cols css 's/grid-template-columns: 40px 64px 62px 44px 34px minmax(0, 1fr); gap: 8px;/grid-template-columns: 40px 66px 62px 44px 34px minmax(0, 1fr); gap: 8px;/'
run sum_gap css 's/.nw-sum { min-width: 0; display: flex; flex-direction: column; gap: 10px; }/.nw-sum { min-width: 0; display: flex; flex-direction: column; gap: 11px; }/'
run tilt_gap css 's/.nw-tilt { display: grid; gap: 5px; }/.nw-tilt { display: grid; gap: 6px; }/'
run evenly css 's/.nw-sum .nw-tilts, .nw-sum .nw-top { flex: 1 1 auto; align-content: space-evenly; }/.nw-sum .nw-tilts, .nw-sum .nw-top { flex: 1 1 auto; align-content: space-around; }/'
run top_cols css 's/grid-template-columns: 30px 26px minmax(0, 1fr); grid-template-areas/grid-template-columns: 32px 26px minmax(0, 1fr); grid-template-areas/'
run when_h css 's/html.fit1 #mcWhen { height: 116px;/html.fit1 #mcWhen { height: 120px;/'
run chip_pad css 's/.opt-chip-btn { min-height: 26px; padding: 0 10px;/.opt-chip-btn { min-height: 26px; padding: 0 11px;/'
cp "$T/in.bak" rtui/compare/fixture/news_in.json
rmut() {   # Rust 쪽 변이: news.rs 를 바꿔 지금 정답지로
  local n=$1 ex=$2 f=rtui/src/news.rs
  cp $f "$T/news.rs.bak"; sed -i "$ex" $f
  cmp -s $f "$T/news.rs.bak" && { echo "✗ $n: 바뀐 곳 없음"; return; }
  if (cd rtui && ulimit -v 14000000 && cargo test --release -j 4 --offline --lib news::t::js_parity > "$T/$n.log" 2>&1); then echo "✗ $n: 못 잡음"; else echo "✓ $n: 잡음 -- $(grep -m1 -A2 'panicked' "$T/$n.log" | tail -n +2 | head -1 | cut -c1-110)"; fi
  cp "$T/news.rs.bak" $f
}
rmut r_floor64 's/fn floor64(x: f64) -> f64 { (x \* 64.0).floor() \/ 64.0 }/fn floor64(x: f64) -> f64 { (x * 64.0).ceil() \/ 64.0 }/'
rmut r_baseline 's/let base = |z: f64| c + asc(12.0, false) - asc(z, false);/let base = |_z: f64| c;/'
rmut r_tw 's/as f64 \* 10.5 \* 0.62;/as f64 * 10.5 * 0.6;/'
rmut r_clamp 's/(ls.len().min(2) as f64 \* LH).max(14.0)/(ls.len().min(3) as f64 * LH).max(14.0)/g'
rmut r_grow 's/srcs_h - ngrow as f64 \* 10.0).max(0.0)/srcs_h - ngrow as f64 * 9.0).max(0.0)/'
