// 수급 프로파일 렌더 검증. 🔴node --check 는 **문법만** 본다 -- 2026-09-19 에 헬퍼
// (stack/bar/SEG_*)를 지운 채 문법검사·CI 를 통과해 깨진 화면을 배포했다.
// 여기서는 실제로 그려 보고 ①예외 ②요소 수 ③**상자 경계**를 검사한다.
//   node test/render_supply_profile_smoke_20260919.js
const fs = require("fs");
const src = fs.readFileSync("dashboard/live/app.js", "utf8");
const m = src.match(/function renderSupplyProfileSvg\(svg, profile, currentPrice, entryPrice = 0, box = null\) \{[\s\S]*?\n\}\n/);
if (!m) { console.log("🔴 함수 추출 실패"); process.exit(1); }

const b64 = (s) => { const b = Buffer.from(s, "base64");
                     return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
const SP = JSON.parse(fs.readFileSync("/home/kbj20/crypto-scalping/tmp/sp_probe/sp.json", "utf8"));
const HJ = JSON.parse(fs.readFileSync("/home/kbj20/crypto-scalping/tmp/sp_probe/hm.json", "utf8"));
// 2026-09-20 서버 payload 의 행통계가 2개 -> 5개가 됐다(fe6ab637 «재깔림» 축). 이 픽스처는
// 그 이전에 뜬 것이라 inst/pers 만 있다 -- 없는 축은 **있는 것으로 대체**해 기하·경계 검사를
// 계속 돌린다. 이 테스트가 보는 건 값의 정확성이 아니라 «던지는가 · 상자를 넘는가»다.
// 🔴라이브는 이 모양을 만들 수 없다: refreshFlowHeatmap 의 f4() 가 없는 키에서 던지고
//   try/catch 가 히트맵을 null 로 만든다(그 경로는 아래 «호가 없음» 분기가 덮는다).
//   그래서 이건 픽스처 문제지 버그가 아니다 -- 새 픽스처를 뜨면 이 대체는 저절로 사라진다.
const ROW_STATS = ["inst", "pers", "peak", "refill", "d60"];
const HM = HJ.rows ? { ...HJ, rows: Object.assign({ ...HJ.rows },
  ...ROW_STATS.map((k) => ({ [k]: b64(HJ.rows[`${k}_f4`] || HJ.rows.inst_f4) }))) } : null;

function run(label, profile, hm, w, h, narrow) {
  const rects = [];
  const mk = (kind) => ({ _a: {}, kids: [], _kind: kind,
    setAttribute(k, v) { this._a[k] = v; },
    appendChild(c) { this.kids.push(c); },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    querySelector() { return null; }, getBoundingClientRect() { return { height: h }; } });
  global.document = { createElementNS: (ns, kind) => { const e = mk(kind); if (kind === "rect") rects.push(e); return e; },
                      createElement: () => mk("div"), documentElement: { getAttribute: () => "dark" } };
  global.window = { matchMedia: () => ({ matches: narrow }) };
  global.latestFlowHeatmap = hm;
  global.supplyProfileNow = null;
  global.isMobileChartMode = () => narrow;
  global.fmtNum = (v, d) => String(Number(v).toFixed(d));
  for (const n of ["updateSupplyProfileNow", "footprintShades", "fmtUsdCompact", "escapeHtml",
                   "snapshotAccountPosition"]) {
    if (!global[n]) global[n] = () => (n === "footprintShades" ? ["a", "b", "c", "d"] : null);
  }
  const svg = { _a: {}, innerHTML: "", kids: [], setAttribute(k, v) { this._a[k] = v; },
                appendChild(c) { this.kids.push(c); },
                parentElement: { clientWidth: w, clientHeight: h },
                getBoundingClientRect() { return { height: h }; } };
  try {
    eval(m[0] + "\nrenderSupplyProfileSvg(svg, profile, 2641, 0, {w: W, h: H});"
         .replace("W", w).replace("H", h));
  } catch (e) { console.log(`🔴 ${label}: ${e.constructor.name} — ${e.message}`); return false; }

  // 경계 검사 -- 막대가 상자를 넘으면 잘려 보이거나 옆 패널을 침범한다.
  const bad = [];
  for (const r of rects) {
    const x = +r._a.x, y = +r._a.y, rw = +r._a.width, rh = +r._a.height;
    if (![x, y, rw, rh].every(Number.isFinite)) { bad.push(["NaN", r._a]); continue; }
    if (rw < 0 || rh < 0) bad.push(["음수 크기", r._a]);
    else if (x < -0.5 || x + rw > w + 0.5) bad.push(["가로 넘침", r._a]);
    else if (y < -0.5 || y + rh > h + 0.5) bad.push(["세로 넘침", r._a]);
  }
  if (bad.length) {
    console.log(`🔴 ${label}: 경계 위반 ${bad.length}건 — 예: ${bad[0][0]} ${JSON.stringify(bad[0][1])}`);
    return false;
  }
  const n = svg.kids.length;
  if (profile && (profile.levels || []).length && n < 10) {
    console.log(`🔴 ${label}: 그린 게 너무 적다 (${n})`); return false;
  }
  console.log(`✅ ${label}: 요소 ${n} · rect ${rects.length} · 경계 OK`);
  return true;
}

const empty = { ...SP, levels: [] };
const one = { ...SP, levels: [SP.levels[0]] };
const flat = { ...SP, levels: SP.levels.map((l) => [l[0], 5, 5, 0, 0, 5, 5]) };  // 델타 0
let ok = true;
ok = run("데스크톱 1200x190", SP, HM, 1200, 190, false) && ok;
ok = run("모바일 390x190", SP, HM, 390, 190, true) && ok;
ok = run("좁은상자 320x120", SP, HM, 320, 120, true) && ok;
ok = run("호가 없음", SP, null, 1200, 190, false) && ok;
ok = run("레벨 0개", empty, HM, 1200, 190, false) && ok;
ok = run("레벨 1개", one, HM, 1200, 190, false) && ok;
ok = run("델타 전부 0", flat, HM, 1200, 190, false) && ok;
console.log(ok ? "\nall ok" : "\n🔴 실패 있음");
process.exit(ok ? 0 : 1);
