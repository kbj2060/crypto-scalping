/* 현재 상황 18칸 (2026-09-22, 시안 U2). 실행: node test/<이 파일>
 * app.js 에서 명세를 떼어 **실제로 돌린다**. 이 카드의 계약은 하나다:
 *   «무슨 값이 오든 칸 수가 변하지 않는다» -- 그게 깜빡임 제거의 전부다.  */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const a = src.indexOf("const SIT_CLAMP");
const b = src.indexOf("\n}\n", src.indexOf("function situationSignalRow("));
assert.ok(a > 0 && b > a, "명세 블록을 못 찾았다");
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const { SIT_SIGNALS, row } = eval(
  src.slice(a, b + 2) + "\n({ SIT_SIGNALS, row: situationSignalRow })");

const render = (ev) => SIT_SIGNALS.map(([nm, kind, fn]) => {
  let r = null; try { r = fn(ev, ev.thr_ui || {}); } catch (_) { r = null; }
  return row(nm, kind, r);
});

// 실제 페이로드에 가까운 한 벌 (하락 추세)
const FULL = {
  mid: 2706.4, move_bp: -124, move_ratio: 0.53, thr_next: 0.25, dir: -1,
  oi_sum: -318, oi_agree: 0.83, liq_long: 412, liq_short: 96,
  cvd: -480, max_delta: -610, last_delta: 90, climax: true, reject: false,
  breakout_detect: false, breakout_prewarn: true,
  cur_sig: "분배", cur_whale: -140, cur_retail: 260,
  wall: -1, obi: -0.31, persist: 0.34,
  va_lo: 2690, va_hi: 2740, res_bp: 84, sup_bp: 18, near_res: false, near_sup: true,
  no_cushion: false, no_cushion_up: false, hot: true, act_pct: 0.91,
  funding: 0.00018, crowd: 1, trapped: true, basis_d_bp: -1.8, basis_thr_bp: 0.8, lead: 1,
  btc_move_bp: -10, btc_rel: "단독",
  thr_ui: { obi: 0.3, reject: 0.5, act: 0.8, near_bp: 60, cushion: 3,
            persist_thin: 0.3, persist_thick: 0.4, funding: 0.0001 },
};

// ── 계약 1: 칸 수가 절대 안 변한다 ──
const N = SIT_SIGNALS.length;
assert.equal(N, 18, "18칸이어야 한다");
for (const [what, ev] of [["가득", FULL], ["빈 것", {}], ["null 투성이",
    Object.fromEntries(Object.keys(FULL).map((k) => [k, null]))]])
  assert.equal(render(ev).length, N, `${what} 에서 칸 수가 ${N} 이 아니다`);

// ── 계약 2: 세 상태가 **형태로** 갈린다 (색만으로 구분하면 저시력에서 죽는다) ──
const full = render(FULL).join(""), empty = render({}).join("");
assert.ok(/class="sit-sg d"/.test(full), "발산 칸이 없다");
assert.ok(/class="sit-sg"/.test(full), "강도 칸이 없다");
assert.ok(/ off"/.test(full), "조건 미달 칸이 하나도 없다");   // BTC 단독 · 저항 먼 쪽 등
assert.ok(/ none"/.test(empty), "재료 없음 칸이 없다");
assert.ok(/class="z"/.test(full), "조건 미달인데 0 눈금이 없다");
assert.equal((empty.match(/class="f"/g) || []).length, 0, "재료가 없는데 막대를 그렸다");

// ── 계약 3: 막대가 트랙을 안 넘는다 ──
for (const m of full.matchAll(/width:([\d.]+)%/g))
  assert.ok(Number(m[1]) <= 100.001, `막대 폭 ${m[1]}% 가 트랙을 넘는다`);
for (const m of full.matchAll(/class="sit-sg d"[\s\S]*?width:([\d.]+)%/g))
  assert.ok(Number(m[1]) <= 50.001, "발산 막대가 반쪽을 넘는다");

// ── 계약 4: 부호 규약 -- 8칸 전부 «+ 는 롱·상승 쪽» ──
const val = (nm, ev) => { const [, k, fn] = SIT_SIGNALS.find(([x]) => x === nm); void k; return fn(ev, ev.thr_ui || {}); };
assert.ok(val("청산 편중", FULL).v < 0, "롱이 더 청산됐는데 + 쪽이다");
assert.ok(val("청산 편중", { ...FULL, liq_long: 96, liq_short: 412 }).v > 0, "숏 청산이 − 쪽이다");
assert.ok(val("창 CVD", FULL).v < 0, "매도 우위인데 + 쪽이다");
assert.ok(val("고래−리테일", FULL).v < 0, "분배인데 + 쪽이다");
assert.ok(val("호가 OBI", FULL).v < 0, "매도벽인데 + 쪽이다");
assert.ok(val("펀딩 쏠림", FULL).v > 0, "롱 쏠림인데 − 쪽이다");
assert.ok(val("BTC 이동", FULL).v < 0, "BTC 하락인데 + 쪽이다");

// ── 계약 5: 임계는 **서버가 준 값**을 따른다 (클라이언트가 다시 선언하면 조용히 어긋난다) ──
const m1 = val("호가 OBI", FULL).mark;
const m2 = val("호가 OBI", { ...FULL, thr_ui: { ...FULL.thr_ui, obi: 0.5 } }).mark;
assert.equal(m1, 30); assert.equal(m2, 50, "thr_ui 를 안 따른다");

// ── 계약 6: 0 으로 나누거나 NaN 을 뱉지 않는다 ──
for (const ev of [{ ...FULL, max_delta: 0 }, { ...FULL, va_hi: FULL.va_lo },
                  { ...FULL, basis_thr_bp: 0 }, { ...FULL, move_bp: 0 },
                  { ...FULL, res_bp: 0, sup_bp: 0 }, { ...FULL, liq_long: 0, liq_short: 0 }]) {
  const h = render(ev).join("");
  assert.equal(h.length && /NaN|Infinity|undefined/.test(h), false, "NaN/Infinity 가 샜다");
  assert.equal(render(ev).length, N);
}

// ── 계약 7: 1순위 열이 전역 클래스를 쓰면 안 된다 ──
// styles.css 327행의 전역 .top(페이지 헤더)이 display:flex 를 건다. 시안 E 배포 때부터
// 1순위 열만 가로로 눕고 게이지가 폭 0 이었다. 이름이 겹치면 증상이 «레이아웃»으로만
// 나타나 코드만 읽어서는 안 보인다 -- 그래서 여기서 막는다.
const col = src.slice(src.indexOf('`<div class="sit-col$'), src.indexOf('`<div class="sit-col$') + 120);
assert.ok(!/\btop"/.test(col.replace(/sit-top"/g, "")), `1순위 열이 전역 .top 을 쓴다: ${col.slice(0, 80)}`);

console.log(`모두 통과 (${N}칸)`);
