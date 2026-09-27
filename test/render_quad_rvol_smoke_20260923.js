// 사분면 레인 계약 (2026-09-23 RVOL 띠 → 2026-09-28 시안 E).
//   node test/render_quad_rvol_smoke_20260923.js
// 🔴역사: RVOL 은 CVD 레인 -> 전용 레인 -> 사분면 위 띠 -> **제거**(2026-09-28 사용자 «거래량 60분을 제거»).
//   봉별 활발함은 사분면 판 안 **거래대금 선**(원시 USD)이 맡고, RVOL 은 카드 상단 «오늘 거래량» 배지로만 남는다.
//   되살아나거나 배지까지 같이 지워지면 여기서 실패해야 한다.
const fs = require("fs");
const src = fs.readFileSync("dashboard/live/app.js", "utf8");
let fail = 0;
const ck = (c, what) => { if (!c) { console.log("🔴 " + what); fail++; } else console.log("  ok " + what); };
const code = src.split("\n").filter((l) => !l.trim().startsWith("//")).join("\n");   // 주석 속 이름에 속지 않게

ck(!/rvolBar5By|\brvol5\b|rv\.bar5/.test(code), "5분 RVOL 계열이 없다");
ck(!/RVOL_BAND|rvolBy|rvolSig/.test(code), "사분면 위 60분 RVOL 띠(RVOL_BAND·rvolBy·rvolSig)가 없다");
ck(!/"거래량 " \+ rvolMinutes|거래량 60분/.test(code), "«거래량 60분» 꼬리표가 없다");
ck(/"오늘 거래량 " \+ lab/.test(code), "카드 상단 «오늘 거래량» 배지는 남아 있다");
const quad = code.slice(code.indexOf('cachedLayer("quadLane"'), code.indexOf('cachedLayer("cumLane"'));
ck(/stroke", "var\(--turnover\)"/.test(quad) && /r\.turn/.test(quad), "사분면 판 안에 거래대금 선(var(--turnover), r.turn)이 있다");
ck(/fill-opacity", r\.oi == null \? "0\.06" : r\.oi >= 0 \? "0\.2" : "0\.1"/.test(quad), "칸 배경 농도: 신규 진함 · 정리 옅음 · OI 모름 가장 옅음");
ck(!/const hgt = 26/.test(quad), "사분면 막대(높이 |Δ|)가 되살아나지 않았다");

console.log(fail ? `🔴 ${fail}건` : "✅ 사분면 레인 계약 OK");
process.exit(fail ? 1 : 0);
