// 수급 프로파일 렌더 스모크. 🔴node --check 는 **문법만** 본다 -- 2026-09-19 에 헬퍼
// (stack/bar/SEG_*)를 통째로 지운 채 문법검사·CI 를 통과해 **깨진 화면을 배포했다**.
// 미정의 참조는 실행해야 잡힌다. 실 API 응답(tmp/sp_probe/)으로 한 번 그려 본다.
//   node test/render_supply_profile_smoke_20260919.js
const fs = require("fs");
const src = fs.readFileSync("dashboard/live/app.js", "utf8");
const m = src.match(/function renderSupplyProfileSvg\(svg, profile, currentPrice, entryPrice = 0, box = null\) \{[\s\S]*?\n\}\n/);
if (!m) { console.log("🔴 함수 추출 실패"); process.exit(1); }
const mk = () => ({ _a:{}, kids:[], setAttribute(k,v){this._a[k]=v;}, appendChild(c){this.kids.push(c);},
                    set textContent(v){this._t=v;}, get textContent(){return this._t;},
                    querySelector(){return null;}, getBoundingClientRect(){return {height:190};} });
global.document = { createElementNS: () => mk(), createElement: () => mk(),
                    documentElement: { getAttribute: () => "dark" } };
global.window = { matchMedia: () => ({ matches: false }) };
const sp = JSON.parse(fs.readFileSync("/home/kbj20/crypto-scalping/tmp/sp_probe/sp.json","utf8"));
const hj = JSON.parse(fs.readFileSync("/home/kbj20/crypto-scalping/tmp/sp_probe/hm.json","utf8"));
const b64 = (s)=>{const b=Buffer.from(s,"base64");return new Float32Array(b.buffer,b.byteOffset,b.length/4);};
global.latestFlowHeatmap = hj.rows ? {...hj, rows:{...hj.rows, inst:b64(hj.rows.inst_f4), pers:b64(hj.rows.pers_f4)}} : null;
global.supplyProfileNow = null;
global.isMobileChartMode = () => false;
global.fmtNum = (v,d)=>String(v.toFixed(d));
// 모듈 다른 함수는 하네스에 없다 -- 스텁. 호출되면 세어만 둔다(진짜 미정의와 구분하려고).
global.__stubbed = [];
for (const n of ["updateSupplyProfileNow","footprintShades","supplyProfileTip","fmtUsdCompact",
                 "escapeHtml","snapshotAccountPosition","isMobileChartMode"]) {
  if (!global[n]) global[n] = (...a) => { global.__stubbed.push(n); return n==="footprintShades"?["a","b","c","d"]:null; };
}
const svg = { _a:{}, innerHTML:"", kids:[], setAttribute(k,v){this._a[k]=v;}, appendChild(c){this.kids.push(c);},
              parentElement:{clientWidth:1200, clientHeight:190}, getBoundingClientRect(){return {height:190};} };
try {
  eval(m[0] + "\nrenderSupplyProfileSvg(svg, sp, 2641, 0, {w:1200, h:190});");
} catch (e) { console.log("🔴 런타임 오류:", e.constructor.name, "—", e.message); process.exit(1); }
const tags = {};
svg.kids.forEach(k => { const t = k._a.fill || k._a.stroke || "?"; tags[t] = (tags[t]||0)+1; });
if (svg.kids.length < 20) { console.log("🔴 그린 게 너무 적다:", svg.kids.length); process.exit(1); }
console.log(`✅ 예외 없음 · SVG 자식 ${svg.kids.length}개`);
console.log("   색별:", JSON.stringify(tags));
console.log("   스텁 호출:", [...new Set(global.__stubbed)].join(",") || "없음");
console.log("   supplyProfileNow:", global.supplyProfileNow ? "설정됨(행 " + global.supplyProfileNow.keys.length + ")" : "null");
