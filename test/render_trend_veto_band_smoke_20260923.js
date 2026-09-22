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
function run({ n = 8, atr = 2, closeLast = 100, warmup = false, live = false, vk = 1,
               smaOut = null } = {}) {
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
                  sma: smaOut === null ? 100 : smaOut, atr,
                  veto: warmup ? 0 : 1, drop: 100, vn: 144, vk };
    if (live && last) { delete row.sma; delete row.atr; delete row.veto; }  // 형성 중 봉
    candles.push(row);
  }
  const svg = mk("svg");
  svg.id = "candleSvgSnapshot";
  svg.querySelector = (sel) => els.find((e) => e._kind === "clipPath" && ("clipPath#" + e._a.id) === sel) || null;
  // y 는 «가격 그대로» 로 두면 좌표에서 값을 되읽을 수 있다(부호만 뒤집힌 화면좌표 대신).
  // els 는 document 스텁의 클로저에 이미 쌓인다 -- 블록 자체는 아무것도 반환하지 않는다.
  // 기하: 플롯은 y ∈ [mt, mt+ch] = [0, 200]. yAt 은 항등이라 «가격 = y» 다.
  new Function("candles", "svg", "NS", "document", "xAt", "yAt", "bw", "ml", "mt", "cw", "ch",
               "{" + BLOCK + "}")(
    candles, svg, NS, document, (i) => i * 10, (v) => v, 8, 0, 0, 500, 200);
  return { els, svg };
}

let fail = 0;
const ck = (cond, what) => { if (!cond) { console.log("🔴 " + what); fail++; } else console.log("  ok " + what); };

// ① ±2×ATR 점선 -- sma±2·atr 에 그려졌는가 (yAt 이 항등이라 y 가 곧 가격)
{
  const { els } = run({ atr: 2 });
  const dashed = els.filter((e) => e._kind === "polyline" && e._a["stroke-dasharray"] === "2 4");
  ck(dashed.length === 2, `참조 점선 2줄 (실제 ${dashed.length})`);
  const ys = dashed.map((e) => Number(e._a.points.split(" ")[0].split(",")[1])).sort((a, b) => a - b);
  ck(ys[0] === 96 && ys[1] === 104, `K=1 이면 점선이 **2×ATR**(96/104)에 (실제 ${ys})`);
  // 실선 밴드는 ±1×ATR (98/102) -- 점선과 **다른 자리**여야 한다
  const solid = els.filter((e) => e._kind === "polyline" && !e._a["stroke-dasharray"]
                                  && Number(e._a["stroke-width"]) === 1.25);
  const sy = solid.map((e) => Number(e._a.points.split(" ")[0].split(",")[1])).sort((a, b) => a - b);
  ck(sy[0] === 98 && sy[1] === 102, `실선 밴드는 ±1×ATR = 98/102 (실제 ${sy})`);
}
// ①b K=2 를 서버가 보내면 밴드가 따라가고 점선은 안쪽 1×ATR 로 바뀐다(배선 확인)
{
  const { els } = run({ atr: 2, vk: 2 });
  const solid = els.filter((e) => e._kind === "polyline" && !e._a["stroke-dasharray"]
                                  && Number(e._a["stroke-width"]) === 1.25);
  const sy = solid.map((e) => Number(e._a.points.split(" ")[0].split(",")[1])).sort((a, b) => a - b);
  ck(sy[0] === 96 && sy[1] === 104, `vk=2 -> 실선이 ±2×ATR (실제 ${sy})`);
  const dy = els.filter((e) => e._a["stroke-dasharray"] === "2 4")
                .map((e) => Number(e._a.points.split(" ")[0].split(",")[1])).sort((a, b) => a - b);
  ck(dy[0] === 98 && dy[1] === 102, `vk=2 -> 점선이 안쪽 ±1×ATR (실제 ${dy})`);
}
// ①c vk 가 아예 없으면(옛 서버) **서버 기본값 K=1** 로 떨어져야 한다
{
  const { els } = run({ atr: 2, vk: NaN });
  const solid = els.filter((e) => e._kind === "polyline" && !e._a["stroke-dasharray"]
                                  && Number(e._a["stroke-width"]) === 1.25);
  const sy = solid.map((e) => Number(e._a.points.split(" ")[0].split(",")[1])).sort((a, b) => a - b);
  ck(sy[0] === 98 && sy[1] === 102, `vk 없음 -> 실선이 ±1×ATR (실제 ${sy})`);
}
// ①d 🔴회귀 방지: 밴드·선이 **잘리는 그룹** 안에 있어야 한다
//    (2026-09-23 실사고 — 밴드가 플롯 바닥 y=1032 를 넘어 y=1202 까지 내려와 사분면 막대를 덮었다)
{
  const { els } = run({ atr: 2 });
  const cp = els.find((e) => e._kind === "clipPath");
  ck(!!cp, "clipPath 가 만들어진다");
  const clipped = els.find((e) => e._kind === "g" && e._a["clip-path"]);
  ck(!!clipped && clipped._a["clip-path"] === `url(#${cp._a.id})`,
     `잘리는 하위 그룹이 그 clipPath 를 쓴다 (실제 ${clipped && clipped._a["clip-path"]})`);
  // 밴드 면·실선·점선·SMA 선이 전부 그 그룹 안에 있어야 한다
  const kinds = clipped.kids.map((e) => e._kind);
  ck(kinds.filter((k) => k === "polygon").length === 1, `밴드 면이 그룹 안 (${kinds})`);
  ck(kinds.filter((k) => k === "polyline").length >= 4, `실선2+점선2+SMA 가 그룹 안 (${kinds})`);
  // 🔴꼬리표는 **밖**이어야 한다 -- 밴드가 화면 밖이어도 측면은 읽혀야 하니까
  ck(!kinds.includes("text"), "꼬리표는 잘리는 그룹 밖");
}
// ①e 꼬리표 y 는 플롯 안으로 잡힌다 (SMA 가 플롯 밖이어도)
{
  const { els } = run({ atr: 2, smaOut: 9999 });
  const tag = els.filter((e) => e._kind === "text").pop();
  const y = Number(tag._a.y);
  ck(y >= 0 && y <= 200, `SMA 가 플롯 밖(9999)이어도 꼬리표 y=${y} 가 [0,200] 안`);
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
