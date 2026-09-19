// 추가 진입(물타기) 레버리지 고정 + 자동 20배 검증 (2026-09-20).
// 🔴여기는 실주문 경로다. 미리보기 쿼리와 실주문이 **같은 값**을 써야 한다 -- 따로 두면
// «화면은 20배인데 주문은 모델값»이 된다(이 저장소가 여러 번 겪은 파리티 사고의 모양).
//   node test/manual_leverage_lock_20260920.js
const fs = require("fs");
const src = fs.readFileSync("dashboard/live/app.js", "utf8");
const grab = (re, what) => { const m = src.match(re); if (!m) { console.log(`🔴 못 찾음: ${what}`); process.exit(1); } return m[0]; };

const PRE = grab(/const manualLevAuto = [^\n]*\n/, "manualLevAuto")
  + grab(/function manualLevValue\(\) \{[\s\S]*?\n\}\n/, "manualLevValue")
  + grab(/const MANUAL_LEV_AUTO = \d+;/, "MANUAL_LEV_AUTO") + "\n"
  + grab(/const manualLevLocked = \(\) => \{[\s\S]*?\n\};\n/, "manualLevLocked")
  + grab(/function manualLevEffective\(\) \{[\s\S]*?\n\}\n/, "manualLevEffective")
  + grab(/const manualLevQuery = \(\) => \{[\s\S]*?\n\};\n/, "manualLevQuery");

function run({ posLev, auto, gauge }) {
  global.el = (id) => id === "snapLevAuto" ? { checked: auto }
    : id === "snapLevGauge" ? { value: String(gauge) } : null;
  global.snapshotAccountPosition = () => (posLev ? { leverage: posLev } : null);
  return eval(PRE + "[manualLevEffective(), manualLevQuery()];");
}

let ok = true;
const eq = (label, got, want) => {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { console.log(`🔴 ${label}: ${g} != ${w}`); ok = false; }
  else console.log(`✅ ${label}: ${g}`);
};

// 물타기: 게이지를 어디에 두든, 자동이 켜졌든 꺼졌든 **기존 포지션 값**이다.
eq("포지션 ×3 · 자동 · 게이지 50", run({ posLev: 3, auto: true, gauge: 50 }), [3, "&lev=3"]);
eq("포지션 ×3 · 수동 · 게이지 50", run({ posLev: 3, auto: false, gauge: 50 }), [3, "&lev=3"]);
eq("포지션 ×7 · 수동 · 게이지 5 ", run({ posLev: 7, auto: false, gauge: 5 }), [7, "&lev=7"]);
// 신규 진입: 자동은 20배, 수동은 게이지(5단위).
eq("신규 · 자동             ", run({ posLev: 0, auto: true, gauge: 5 }), [20, "&lev=20"]);
eq("신규 · 수동 게이지 35   ", run({ posLev: 0, auto: false, gauge: 35 }), [35, "&lev=35"]);
eq("신규 · 수동 게이지 37   ", run({ posLev: 0, auto: false, gauge: 37 }), [35, "&lev=35"]);

// 실주문 pending 과 미리보기 쿼리가 같은 함수를 보는가(정적).
if (!/lev: manualLevEffective\(\)/.test(src)) {
  console.log("🔴 실주문 pending 이 manualLevEffective 를 안 쓴다 -- 미리보기와 어긋난다"); ok = false;
} else console.log("✅ 실주문 pending 도 manualLevEffective");
if (/manualLevAuto\(\) \? (null|"")/.test(src)) {
  console.log("🔴 «자동이면 안 보낸다» 분기가 남았다 -- 서버가 모델값으로 떨어진다"); ok = false;
} else console.log("✅ «자동이면 안 보낸다» 분기 없음");

console.log(ok ? "✅ 레버리지 고정 OK" : "");
process.exit(ok ? 0 : 1);
