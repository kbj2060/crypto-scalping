/* 시장 맥락 카드·차트 겹침의 순수 함수 (2026-09-29). 실행: node test/<이 파일>
 * app.js 에서 함수 본문을 떼어 **실제로 돌린다** -- 미국장 시각 · 볼린저 · RSI. (패턴 판정은 09-30 표식과 함께 제거) */
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
const mcRsi = eval(`(${take("mcRsi")})`);
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

// ── 옵션 이벤트 예상 폭(2026-10-01): 평평한 IV → 0 · 이벤트를 품은 만기만 분산이 더 크면 그 몫이 나온다 ──
const optEventMove = eval(`(${take("optEventMove")})`);
{
  const now = Date.parse("2026-10-05T00:00:00Z"), H = 3.6e6, yr = 3.156e10;
  const exp = (h, iv) => ({ exp_ms: now + h * H, atm_i: iv });
  const ev = [{ importance: "high", time_utc: new Date(now + 20 * H).toISOString(), title_ko: "CPI" }];
  assert.deepEqual(optEventMove({ expiries: [exp(12, 40), exp(36, 40), exp(60, 40)] }, now, ev), { nm: "CPI", pct: 0 });
  // 36h 만기에 이벤트 몫 0.01%² 를 얹는다: v36·T36 = 0.16·T36 + x → 기대 = √x
  const x = 0.0001, T36 = 36 * H / yr, iv36 = Math.sqrt((0.16 * T36 + x) / T36) * 100;
  const r = optEventMove({ expiries: [exp(12, 40), exp(36, iv36), exp(60, Math.sqrt((0.16 * 60 * H / yr + x) / (60 * H / yr)) * 100)] }, now, ev);
  assert.ok(Math.abs(r.pct - Math.sqrt(x) * 100) < 1e-6, r);
  assert.equal(optEventMove({ expiries: [] }, now, []), null);                          // 4일 안 일정 없음
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
// ── RSI: 계속 오르면 100, 계속 내리면 0, 모자라면 null ──
assert.equal(mcRsi(Array.from({ length: 30 }, (_, i) => 100 + i)), 100);
assert.equal(mcRsi(Array.from({ length: 30 }, (_, i) => 100 - i)), 0);
assert.equal(mcRsi([1, 2, 3]), null);
console.log("market ctx patterns ok");
