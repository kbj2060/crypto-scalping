/* 시장 맥락 카드·차트 겹침의 순수 함수 (2026-09-29). 실행: node test/<이 파일>
 * app.js 에서 함수 본문을 떼어 **실제로 돌린다** -- 미국장 시각 · 볼린저. (패턴 판정은 09-30 표식과 함께 제거 · RSI 는 10-06 ④ 줄 제거와 함께 삭제) */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const take = (name) => {
  const i = src.indexOf(`function ${name}(`);
  assert.ok(i > 0, `${name} 를 못 찾았다`);
  return src.slice(i, src.indexOf("\n}\n", i) + 2);
};
const mcUsSession = eval(`(${take("mcUsSession")})`);
const mcBollinger = eval(`(${take("mcBollinger")})`);
const mcLine = { vsess: false };                                  // mcVwapOf 가 읽는 스위치 -- 직접 eval 이라 이 바인딩을 본다
const mcVwapOf = eval(`(${take("mcVwapOf")})`);

// ── VWAP 기준 스위치: 끄면 하루(UTC 날짜가 seg) · 켜면 시장 세션(개장 초가 seg, 이름) · 없는 쪽은 null ──
const row = { time: 1790640000 + 3600, vwap: 2690, vsd: 3, svwap: 2695, svsd: 1, sstart: 1790640000, sname: "아시아" };
assert.deepEqual(mcVwapOf(row), { vwap: 2690, vsd: 3, seg: Math.floor(row.time / 86400), name: "하루" });
mcLine.vsess = true;
assert.deepEqual(mcVwapOf(row), { vwap: 2695, vsd: 1, seg: 1790640000, name: "아시아" });
assert.equal(mcVwapOf({ time: 1, vwap: 1, vsd: 1 }), null);          // 서버가 아직 세션 VWAP 을 안 실었으면 선을 안 그린다
mcLine.vsess = false;

// ── 옵션 흐름 «신규 약 N%» = (V + ΔOI) / 2V · ΔOI 없는 시간은 뺀다 · 0~100% 로 자른다 (2026-10-01) ──
const fmtNum = (v, d) => v.toFixed(d);
const optNewTxt = eval(`(${take("optNewTxt")})`);
assert.equal(optNewTxt([{ cb: 3, cs: 0, pb: 0, ps: 1, doi: 2 }]), " · 미결제 +2 · 신규 약 75%");
assert.equal(optNewTxt([{ cb: 3, cs: 0, pb: 0, ps: 1, doi: -4 }]), " · 미결제 −4 · 신규 약 0%");      // 모두 청산
assert.equal(optNewTxt([{ cb: 3, cs: 0, pb: 0, ps: 1, doi: null }]), "");                            // ΔOI 모름
assert.equal(optNewTxt([{ cb: 1, cs: 1, pb: 0, ps: 0, doi: 2 }, { cb: 9, cs: 9, pb: 0, ps: 0, doi: null }]), " · 미결제 +2 · 신규 약 100%");

// ── 실현 감마(2026-10-02): 12시간 5분 수익률 acf1 · 마감봉만 · 분위·국면 경계 ──
{
  const ri = src.indexOf("const RG_REF ="), RG_REF = eval(src.slice(ri + 14, src.indexOf(";", ri)));
  const optRealizedGamma = eval(`(${take("optRealizedGamma")})`);
  const mk = (rets, t0 = 1e9) => { let p = 2000; const out = [{ time: t0, close: p }];
    rets.forEach((r, i) => { p *= Math.exp(r); out.push({ time: t0 + (i + 1) * 300, close: p }); }); return out; };
  const alt = Array.from({ length: 144 }, (_, i) => (i % 2 ? -0.001 : 0.001));          // 매 봉 되돌림 → acf ≈ −1
  const rev = optRealizedGamma(mk(alt), 1e9 + 145 * 300);
  assert.ok(rev.v < -0.9 && rev.side === "rev" && rev.pct === 0.5, JSON.stringify(rev));
  const runs = Array.from({ length: 144 }, (_, i) => (Math.floor(i / 8) % 2 ? -0.001 : 0.001));   // 8봉씩 이어짐 → acf 양수
  const tr = optRealizedGamma(mk(runs), 1e9 + 145 * 300);
  assert.ok(tr.v > 0.5 && tr.side === "trend" && tr.pct === 99.5, JSON.stringify(tr));
  assert.equal(optRealizedGamma(mk(alt), 1e9 + 144 * 300 + 10), null);                // 마지막 봉이 형성 중 → 마감봉 144개 → 계산 안 함
  const mid = optRealizedGamma(mk(alt).map((c, i) => ({ ...c, close: 2000 * Math.exp(Math.sin(i * 1.3) * 0.001 + i * 0.00001) })), 1e9 + 145 * 300);
  assert.ok(mid && mid.pct > 0 && mid.pct < 100, JSON.stringify(mid));
}

// ── 미국장: 서머타임 13:30 UTC · 주말 건너뜀 · 표준시 14:30 UTC · 장중 판정 ──
const us = (iso) => { const r = mcUsSession(Date.parse(iso)); return [new Date(r.open).toISOString().slice(0, 16), r.live]; };
assert.deepEqual(us("2026-09-29T12:00:00Z"), ["2026-09-29T13:30", false]);
assert.deepEqual(us("2026-09-29T15:00:00Z"), ["2026-09-29T13:30", true]);
assert.deepEqual(us("2026-10-03T12:00:00Z"), ["2026-10-05T13:30", false]);   // 토요일 -> 월요일
assert.deepEqual(us("2026-12-01T12:00:00Z"), ["2026-12-01T14:30", false]);   // 표준시

// ── 볼린저: 상수 종가면 폭 0, 20봉 전엔 값 없음 ──
const cs = Array.from({ length: 25 }, (_, i) => ({ time: i, close: 100 }));
const bb = mcBollinger(cs);
assert.equal(bb.has(18), false); assert.deepEqual(bb.get(19), [100, 100, 100]);
console.log("market ctx patterns ok");

// ── 청산 원 툴팁 거래소별(2026-10-03): 바이낸스 = 합 − 나머지 · 0원 거래소는 줄을 안 쓴다 · 합산 목록 ──
{
  const fmtUsdCompact = (v) => `$${Math.round(v)}`;
  const liqVenueText = eval(`(${take("liqVenueText")})`);
  const t = liqVenueText({ long_usd: 1000, short_usd: 300, events: 4, okx: true, okx_detail: { long_usd: 200, short_usd: 0, n: 1 },
                           bybit: { long_usd: 0, short_usd: 0, n: 0 } });
  assert.ok(t.includes("바이낸스 $1100 3건(롱 $800/숏 $300)") && t.includes("OKX $200 1건"), t);
  assert.ok(!t.includes("Bybit $"), "0원 거래소 줄은 안 쓴다");
  assert.ok(t.endsWith("합산: 바이낸스 · OKX · Bybit"), t);
}
