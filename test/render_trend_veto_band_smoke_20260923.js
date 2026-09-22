// 추세 veto 밴드 렌더 검증 (2026-09-23, ±2×ATR 등급 추가판).
//   node test/render_trend_veto_band_smoke_20260923.js
// 무엇을 지키는가: ①±2×ATR 점선 두 줄이 실제로 sma±2·atr 에 그려지는가
//   ②등급 경계가 |종가−SMA|/ATR 의 <1 / 1~2 / ≥2 인가(측정한 R1 구간과 같은 경계)
//   ③워밍업(veto=0)에는 등급을 붙이지 않는가 -- 세기는 «허용 측면»의 세기이지 그 자체가 신호가 아니다.
// 🔴시안 하나가 아니라 **상태 전부**를 돈다(형성 중 봉 유무 × 세 등급 × 워밍업).
const fs = require("fs");
const src = fs.readFileSync("dashboard/live/app.js", "utf8");

// 블록을 중괄호 세기로 잘라낸다 -- 앵커 주석 다음의 `  {` 부터 짝이 맞을 때까지.
const anchor = src.indexOf("// ── 추세 veto (2026-09-22");
if (anchor < 0) { console.log("🔴 추세 veto 블록 앵커를 못 찾음"); process.exit(1); }
const open = src.indexOf("\n  {\n", anchor) + 3;
let depth = 0, end = open;
for (; end < src.length; end++) {
  if (src[end] === "{") depth++;
  else if (src[end] === "}" && --depth === 0) { end++; break; }
}
const BLOCK = src.slice(open, end);

const NS = "http://www.w3.org/2000/svg";
function run({ n = 8, atr = 2, closeLast = 100, warmup = false, live = false } = {}) {
  const els = [];
  const mk = (kind) => ({ _a: {}, _kind: kind, kids: [],
    setAttribute(k, v) { this._a[k] = v; }, appendChild(c) { this.kids.push(c); },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; } });
  const document = { createElementNS: (ns, kind) => { const e = mk(kind); els.push(e); return e; } };
  const candles = [];
  for (let i = 0; i < n; i++) {
    const last = i === n - 1;
    const cl = last ? closeLast : 100;
    const row = { close: cl, high: cl + 1, low: cl - 1, open: cl,
                  sma: 100, atr, veto: warmup ? 0 : 1, drop: 100, vn: 144 };
    if (live && last) { delete row.sma; delete row.atr; delete row.veto; }  // 형성 중 봉
    candles.push(row);
  }
  const svg = mk("svg");
  // y 는 «가격 그대로» 로 두면 좌표에서 값을 되읽을 수 있다(부호만 뒤집힌 화면좌표 대신).
  // els 는 document 스텁의 클로저에 이미 쌓인다 -- 블록 자체는 아무것도 반환하지 않는다.
  new Function("candles", "svg", "NS", "document", "xAt", "yAt", "bw", "{" + BLOCK + "}")(
    candles, svg, NS, document, (i) => i * 10, (v) => v, 8);
  return { els, svg };
}

let fail = 0;
const ck = (cond, what) => { if (!cond) { console.log("🔴 " + what); fail++; } else console.log("  ok " + what); };

// ① ±2×ATR 점선 -- sma±2·atr 에 그려졌는가 (yAt 이 항등이라 y 가 곧 가격)
{
  const { els } = run({ atr: 2 });
  const dashed = els.filter((e) => e._kind === "polyline" && e._a["stroke-dasharray"] === "2 4");
  ck(dashed.length === 2, `±2×ATR 점선 2줄 (실제 ${dashed.length})`);
  const ys = dashed.map((e) => Number(e._a.points.split(" ")[0].split(",")[1])).sort((a, b) => a - b);
  ck(ys[0] === 96 && ys[1] === 104, `점선이 sma∓2·atr = 96 / 104 (실제 ${ys})`);
}
// ② 등급 경계 -- |종가−SMA|/ATR 이 <1 / 1~2 / ≥2
for (const [close, want] of [[101.5, "약"], [103, "보통"], [105, "강"], [95, "강"]]) {
  const { els } = run({ atr: 2, closeLast: close });
  const tag = els.filter((e) => e._kind === "text").pop();
  ck(String(tag.textContent).includes("· " + want),
     `종가 ${close} (|z|=${Math.abs(close - 100) / 2}) -> ${want} (실제 "${tag.textContent}")`);
}
// ③ 워밍업에는 등급이 없다
{
  const { els } = run({ warmup: true });
  const tag = els.filter((e) => e._kind === "text").pop();
  ck(String(tag.textContent) === "추세 veto: 워밍업", `워밍업엔 등급 없음 (실제 "${tag.textContent}")`);
}
// ④ 형성 중 봉이 있어도(sma 미확정) 점선과 등급이 모두 나온다
{
  const { els } = run({ atr: 2, closeLast: 105, live: true });
  ck(els.filter((e) => e._a["stroke-dasharray"] === "2 4").length === 2, "형성 중 봉에서도 ±2×ATR 2줄");
  const tag = els.filter((e) => e._kind === "text").pop();
  ck(String(tag.textContent).includes("강"), `형성 중 봉 등급 (실제 "${tag.textContent}")`);
}
process.exit(fail ? 1 : 0);
