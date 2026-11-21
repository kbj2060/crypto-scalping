// «rt 로 그리기»(app.js rtDraw·rtFpApply·rtFpToScreen·rtSseSupply·rtDrawTick·rtDrawOff·rtDrawReport) -- `node test/test_rt_draw_20261121.mjs`
// 입력 = 진짜 코어가 내보낸 허브 메시지(rtcore/examples/dump_screen_msgs.rs → test/fixtures/rt_screen_msgs.json)와 같은 입력의 원본 응답
// (rtcore 정답지: 원본 supply_1s_payload since=0 · 원본 /api/footprint 봉·칸 전량·이어받기).
// 계약: «rt 로 그린 값 = 옛 경로로 그렸을 값» -- 같은 처리기(applySupply1s · 화면 풋프린트 캐시)가 받는 모양에서 비교한다. 양성 대조 포함.
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";

const src = readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const cut = (a, b) => { const i = src.indexOf(a), j = src.indexOf(b, i); assert.ok(i > 0 && j > i, `app.js 에서 ${a} 를 못 찾았다`); return src.slice(i, j); };
const code = [cut("function applyOi1s", "// ── 밀어주기 스트림"), cut("const rtM = {", "function ensureRtStream"), cut("const RT_TRUST_N", "\n// ── 옵션 카드"),
              cut("let footprintLive = {", "function ensurePriceWs")].join("\n");
const fx = JSON.parse(readFileSync(new URL("./fixtures/rt_screen_msgs.json", import.meta.url), "utf8"));

// app.js 전역 중 이 함수들이 쓰는 것만(같은 이름) -- 그리기 호출은 센다
function load(asset) {
  return new Function("asset", `
    const ASSET_CONFIG = { eth: { symbol: "ETHUSDT" }, sol: { symbol: "SOLUSDT" }, btc: { symbol: "BTCUSDT" } };
    const SUPPLY_1S_SEGMENT = 300, CHART_PAN_BARS = 144, CHART_CANDLE_MIN = 5;
    let activeSnapshotAsset = asset, footprintBars = new Map(), footprintCacheKey = "", latestFootprint = null, footprintLastFetchAt = 123;
    let latestLivePriceByAsset = {}, latestLivePriceTsByAsset = {}, renders = 0, repaints = 0;
    let supply1s = new Map(), oi1s = new Map(), liq1s = new Map(), okxSupply1s = new Map(), okxOi1s = new Map(), okxLiq1s = new Map(), spotSupply1s = new Map(), bybitLiq1s = new Map();
    let supply1sSince = 0, oi1sSince = 0, liq1sSince = 0, okxSupply1sSince = 0, okxOi1sSince = 0, okxLiq1sSince = 0, spotSupply1sSince = 0, bybitLiq1sSince = 0, supply1sVer = 0;
    let supply1sMeta = { now: 0 }, okxMeta = { now: 0 }, spotMeta = { now: 0 };
    const fpWeekFor = () => null, scheduleSnapshotChartRender = () => { renders++; }, maybeRenderSnapshotChartNow = () => { renders++; }, repaintSupply1sPanel = () => { repaints++; };
    ${code}
    return { footprintLiveAdd, footprintMergeLive, fl: () => footprintLive, setFp: (b) => { footprintBars = b; },
             rtM, rtD, rtDraw, rtSseSupply, rtDrawTick, rtDrawOff, rtDrawReport, rtTrusted, rtMeasure, rtSse, applySupply1s,
             setWs: (w) => { rtWs = w; }, st: () => ({ supply1s, oi1s, liq1s, okxSupply1s, okxLiq1s, bybitLiq1s, spotSupply1s, latestFootprint, footprintBars, latestLivePriceByAsset, footprintLastFetchAt, renders, repaints }) };`)(asset);
}
const fpRows = (lf) => lf.bars.map((b) => [b.time, b.levels]);
const mapObj = (m, floor) => JSON.stringify([...m].filter(([k]) => k >= floor).sort((a, b) => a[0] - b[0]));

// ── ① 1초 수급: rt supply1s 메시지 전부 → applySupply1s · 옛 경로 = 원본 since=0 응답 한 번 → applySupply1s ──
{
  const S = fx.supply, rt = load("eth"), old = load("eth");
  rt.rtD.on = true;
  rt.rtDraw(S.snapshot, 1);
  for (const m of S.msgs) rt.rtDraw(m, 1);
  old.applySupply1s(S.want);
  const a = rt.st(), b = old.st(), floor = S.want.now - 2 * 300 - 20;
  assert.ok(a.supply1s.size > 500 && S.msgs.length > 5000, `정답지 크기 ${a.supply1s.size} ${b.supply1s.size}`);
  // ponytail: 원본 응답의 now·partial 과 코어 마지막 밀기는 같은 순간(정답지 끝) -- 창 바닥을 원본 now 로 잡고 그 위 초를 견준다
  assert.equal(mapObj(a.supply1s, floor), mapObj(b.supply1s, floor), "rt 로 쌓은 1초 칸 = 원본 응답으로 쌓은 칸(닫힌 초 + 진행 초)");
  assert.equal(a.repaints, S.msgs.length + 1, "메시지마다 수급 판만 다시 칠한다");
  // 양성 대조: 메시지 하나를 빼면 다르다 · 값 한 칸을 바꾸면 다르다
  const drop = load("eth"); drop.rtD.on = true; drop.rtDraw(S.snapshot, 1);
  const k = S.msgs.findIndex((m) => (m.seconds || []).some((r) => r[0] >= floor + 100));
  S.msgs.forEach((m, i) => { if (i !== k) drop.rtDraw(m, 1); });
  assert.notEqual(mapObj(drop.st().supply1s, floor), mapObj(b.supply1s, floor), "양성 대조: 메시지 하나 빠짐을 잡는다");
  const bad = load("eth"); bad.rtD.on = true; bad.rtDraw(S.snapshot, 1);
  S.msgs.forEach((m, i) => bad.rtDraw(i === k ? { ...m, seconds: m.seconds.map((r) => (r[0] >= floor ? [r[0], r[1], r[2], r[3], r[4], r[5] + 0.001, r[6], r[7]] : r)) } : m, 1));
  assert.notEqual(mapObj(bad.st().supply1s, floor), mapObj(b.supply1s, floor), "양성 대조: 한 칸 값 다름을 잡는다");
  console.log(`① 1초 수급 ok -- rt 메시지 ${S.msgs.length} · 견준 초 ${[...a.supply1s.keys()].filter((x) => x >= floor).length} · 양성 대조 2/2`);
}

// ── ② 풋프린트 + ③ 다른 코인 가격 ──
{
  const F = fx.footprint, rt = load("sol");
  rt.rtD.on = true;
  rt.rtDraw(F.snapshot, 1);
  let n = 0, bars = 0;
  for (const step of F.steps) {
    for (const m of step.msgs) rt.rtDraw(m, 1);
    const lf = rt.st().latestFootprint;
    assert.deepEqual(fpRows(lf), step.full, `대조 ${n}: rt 로 쌓은 봉·칸 = 원본 /api/footprint since=0`);
    assert.deepEqual(fpRows(lf).filter((r) => r[0] >= step.since), step.inc, `대조 ${n}: 이어받기(since=${step.since})`);
    assert.equal(rt.st().latestLivePriceByAsset.sol, step.last_px, `대조 ${n}: 가격(rt prices) = 마지막 체결가`);
    n++; bars += step.full.length;
  }
  // 옛 경로(refreshFootprint)가 같은 응답으로 만드는 캐시와 같은 모양: footprintBars = 봉시각 → {time, levels(, dhl)}
  const last = F.steps[F.steps.length - 1];
  assert.deepEqual([...rt.st().footprintBars.keys()].sort((a, b) => a - b), last.full.map((r) => r[0]), "화면 캐시 봉 = 원본 봉");
  assert.deepEqual(rt.st().latestFootprint.venues, ["binance-perp", "okx-swap"], "OKX 가 붙은 뒤 venues");
  // 양성 대조: 칸 하나 값이 다른 메시지
  const bad = load("sol"); bad.rtD.on = true; bad.rtDraw(F.snapshot, 1);
  // 마지막 풋프린트 메시지의 첫 칸(뒤에서 덮어써 «낫지» 않게)
  const all = F.steps.flatMap((st) => st.msgs);
  const k = all.map((m, i) => (m.type === "footprint" && !m.full && m.bars.length && m.bars[0].levels.length ? i : -1)).filter((i) => i >= 0).pop();
  const hit = k != null;
  all.forEach((m, i) => bad.rtDraw(i !== k ? m : { ...m, bars: [{ ...m.bars[0], levels: [[m.bars[0].levels[0][0], m.bars[0].levels[0][1] + 1, ...m.bars[0].levels[0].slice(2)], ...m.bars[0].levels.slice(1)] }, ...m.bars.slice(1)] }, 1));
  assert.ok(hit);
  assert.notDeepEqual(fpRows(bad.st().latestFootprint), last.full, "양성 대조: 칸 값 하나 다름을 잡는다");
  // 양성 대조: drop·full 을 무시하면(첫 OKX 체결 때 전량) 숨겨야 할 봉이 남는다
  const nofull = load("sol"); nofull.rtD.on = true; nofull.rtDraw(F.snapshot, 1);
  for (const step of F.steps) for (const m of step.msgs) nofull.rtDraw(m.type === "footprint" ? { ...m, full: false } : m, 1);
  assert.notDeepEqual(fpRows(nofull.st().latestFootprint), last.full, "양성 대조: full(갈아 끼움) 무시를 잡는다");
  console.log(`② 풋프린트 ok -- 대조 ${n} 시점 · 봉 ${bars} · ③ 가격 ${n} 시점 · 양성 대조 2/2`);
}

// ── ④ 언제 rt 로 그리나 · 언제 돌아가나 ──
{
  const S = fx.supply, x = load("eth");
  const T = 1_763_700_000_000;
  x.setWs({ readyState: 1 });
  x.rtDraw(fx.footprint.snapshot, T);   // 풋프린트 상태 하나(심볼은 시험 밖)
  const rows = S.want.seconds;
  // 59초 견줌 → 아직 옛 경로
  x.rtMeasure(x.rtM, { type: "supply1s", ex_ms: T, seconds: rows.slice(0, 59) }, T);
  x.rtSse(x.rtM, { seconds: rows.slice(0, 59) }, T);
  x.rtDraw({ type: "prices", rows: [] }, T + 10500);
  x.rtDrawTick(T + 10500);
  assert.equal(x.rtD.on, false, "견준 초 59 -- 아직 안 믿는다");
  x.rtMeasure(x.rtM, { type: "supply1s", ex_ms: T, seconds: [rows[59]] }, T);
  x.rtSse(x.rtM, { seconds: [rows[59]] }, T);
  x.rtDrawTick(T + 10600);
  assert.equal(x.rtD.on, true, "견준 초 60 · 어긋남 0 · 최근 수신 -- rt 로 그린다");
  // SSE 는 계속 받되 oi 만 얹고 들고 있는다
  const before = x.st().supply1s.size;
  x.rtSseSupply({ seconds: [[999, 1, 2, 3, 4, 5, 6, 7]], oi: [[999, 42]], partial: { bn: [1000, 0, 0, 0, 0, 1, 1, 7] } }, T + 10700);
  assert.equal(x.st().supply1s.size, before, "rt 로 그리는 중엔 SSE 1초 칸을 안 얹는다");
  assert.equal(x.st().oi1s.get(999), 42, "rt 에 없는 칸(바이낸스 OI)은 SSE 것");
  // 3초 침묵 → 같은 틱에 옛 경로 · 들고 있던 SSE 를 다시 얹음(진행 초 빼고) · 풋프린트 묻기 바로
  x.rtDrawTick(T + 10500 + 3001);
  assert.equal(x.rtD.on, false, "3초 침묵 -- 옛 경로");
  assert.ok(x.st().supply1s.has(999) && !x.st().supply1s.has(1000), "들고 있던 SSE 닫힌 초는 다시 얹고 진행 초는 뺀다");
  assert.equal(x.st().footprintLastFetchAt, 0, "돌아가면 다음 틱에 /api/footprint");
  // 다시 믿음 → 어긋남 1건이면 즉시 돌아감
  x.rtDraw({ type: "prices", rows: [] }, T + 20000);
  x.rtDrawTick(T + 20000);
  assert.equal(x.rtD.on, true, "수신이 다시 오면 다시 rt");
  const badRow = rows[70].slice(); badRow[5] += 1;
  x.rtMeasure(x.rtM, { type: "supply1s", ex_ms: T, seconds: [badRow] }, T + 5000);
  x.rtSse(x.rtM, { seconds: [rows[70]] }, T + 5000);
  x.rtDraw({ type: "prices", rows: [] }, T + 20100);
  x.rtDrawTick(T + 20100);
  assert.equal(x.rtD.on, false, "어긋남 1건 -- 그 틱에 옛 경로");
  x.rtDrawTick(T + 20200);
  assert.equal(x.rtD.on, false, "이 연결에서는 다시 안 믿는다(새 연결부터)");
  // 끊김 = 연결 없음 → 거짓
  x.rtM.tbad = 0; x.setWs(null); x.rtDrawTick(T + 20300);
  assert.equal(x.rtD.on, false, "연결이 없으면 옛 경로");
  const r = x.rtDrawReport(T + 30000);
  assert.deepEqual(r, { rt_draw_s: 3, rt_back_n: 2 }, "M0: rt 로 그린 초(3.0 + 0.1) · 돌아간 횟수 2");
  assert.deepEqual(x.rtDrawReport(T + 40000), {}, "다음 보고는 비었다");
  // 처리 실패(망가진 메시지)는 믿음을 깬다 -- ensureRtStream onmessage 가 tbad 를 올린다(rtDraw 가 던지면)
  assert.throws(() => x.rtDraw({ type: "footprint", bars: [{ time: 1, levels: 7 }] }, T), "망가진 풋프린트는 던진다(onmessage 가 tbad++)");
  console.log("④ 전환·복귀 ok -- 59→60초 믿음 · SSE 는 oi 만 · 3초 침묵·어긋남 1건·연결 없음에 그 틱 복귀 · 보고 칸 rt_draw_s·rt_back_n");
}

// ── ④-2 켜지는 순간: 그림자 동안 rt 만 받은 초(SSE 는 아직) -- 켜지는 틱에 들고 있던 rt 를 얹는다 ──
{
  const rows = fx.supply.want.seconds, T = 1_763_700_000_000;
  const run = (keepBuf) => {
    const x = load("eth");
    x.setWs({ readyState: 1 });
    x.rtDraw(fx.footprint.snapshot, T);
    x.rtMeasure(x.rtM, { type: "supply1s", ex_ms: T, seconds: rows.slice(0, 60) }, T);
    x.rtSse(x.rtM, { seconds: rows.slice(0, 60) }, T);
    x.rtSseSupply({ seconds: rows.slice(0, 60), now: rows[59][0] + 1 }, T);   // 옛 경로가 그때까지 그린 것
    // 켜지기 직전 3초: rt 만 받음 -- 100~102초 · 101초는 늦은 체결로 두 번(옛 값 → 새 값)
    const old101 = rows[101].slice(); old101[5] -= 0.5;
    const rtOnly = [[rows[100]], [old101], [rows[101]], [rows[102]]];
    rtOnly.forEach((sec, i) => x.rtDraw({ type: "supply1s", seconds: sec, partial: { bn: [rows[102][0] + 1, 0, 0, 0, 0, 1, 1, rows[102][7]] }, now: rows[102][0] + 1 }, T + 8000 + i * 700));
    if (!keepBuf) x.rtD.rtBuf = [];                                                  // 양성 대조: 버퍼 없음
    x.rtDrawTick(T + 10600);
    assert.equal(x.rtD.on, true);
    // 그다음 SSE 가 같은 초를 늦게 준다 -- 켜진 뒤라 oi 만 얹는다
    x.rtSseSupply({ seconds: [rows[100], rows[101], rows[102]], oi: [], now: rows[102][0] + 1 }, T + 11000);
    return x.st().supply1s;
  };
  const y = load("eth");   // 그 구간을 계속 SSE 로만 그렸을 때
  y.applySupply1s({ seconds: rows.slice(0, 60), now: rows[59][0] + 1 });
  y.applySupply1s({ seconds: [rows[100], rows[101], rows[102]], now: rows[102][0] + 1 });
  const pick = (m) => JSON.stringify([100, 101, 102].map((k) => m.get(rows[k][0]) || null));
  assert.equal(pick(run(true)), pick(y.st().supply1s), "켜지기 직전 3초 rt 만 받은 초 = 계속 SSE 로 그렸을 때의 최종 칸(늦은 체결 갱신 포함)");
  assert.notEqual(pick(run(false)), pick(y.st().supply1s), "양성 대조: 버퍼를 빼면 그 초가 빈다");
  console.log("④-2 켜지는 순간 ok -- rt 만 받은 3초(늦은 체결 갱신 포함) = SSE 로만 그린 최종 칸 · 버퍼 뺀 양성 대조 잡음");
}

// ── ⑤ 형성 봉: 직결 @trade 칸과 겹치는가 · ⑥ 꺼지는 순간 /api/footprint since 이어받기 ──
{
  const F = fx.footprint, last = F.steps[F.steps.length - 1];
  const bar = last.full[last.full.length - 1][0];                         // 형성 봉(정답지 끝 봉)
  const evs = JSON.parse(readFileSync(new URL("../rtcore/tests/fixtures/fp_okx_sol.json", import.meta.url), "utf8")).events;
  const bnBar = evs.filter((e) => e[0] === "bn" && Math.floor(e[1] / 300000) * 300 === bar);
  assert.ok(bnBar.length >= 2, "형성 봉 체결");
  const formingOf = (x, byTime, since) => {
    const fl = x.fl(); fl.since = since; fl.bucket = 0.02;
    for (const e of bnBar) x.footprintLiveAdd(e[3], e[4], e[1], e[5], e[2]);
    x.footprintMergeLive(byTime, 0.02);
    return JSON.stringify(byTime.get(bar));
  };
  // (가) 옛 경로: 응답(원본 /api/footprint 의 그 봉) + 직결 누적 · (나) rt 경로: rt 로 쌓은 latestFootprint 의 그 봉 + 같은 직결 누적
  const oldP = load("sol"), rtP = load("sol");
  rtP.rtD.on = true; rtP.rtDraw(F.snapshot, 1); for (const st of F.steps) for (const m of st.msgs) rtP.rtDraw(m, 1);
  const byOld = () => new Map(last.full.map((r) => [r[0], r[1]]));
  const byRt = () => new Map(rtP.st().latestFootprint.bars.map((b) => [b.time, b.levels]));
  for (const [name, since] of [["봉 시작 전부터 붙음(직결로 갈아 끼움)", bar * 1000 - 1], ["봉 중간에 붙음(서버·rt 값 유지)", bar * 1000 + 1]]) {
    const a = formingOf(load("sol"), byOld(), since), b = formingOf(load("sol"), byRt(), since);
    assert.equal(b, a, `형성 봉 칸(${name}): rt 경로 = 옛 경로`);
  }
  // 양성 대조: rt 칸 위에 직결 칸을 «더하면»(이중 합산) 다르다
  const dbl = (() => {
    const x = load("sol"), m = byRt(), fl = x.fl(); fl.since = bar * 1000 - 1; fl.bucket = 0.02;
    for (const e of bnBar) x.footprintLiveAdd(e[3], e[4], e[1], e[5], e[2]);
    const lv = new Map(m.get(bar).map((l) => [l[0], l.slice()]));
    for (const [k, c] of fl.cells) { const p = Math.round(k * 0.02 * 100) / 100, t = lv.get(p) || [p, 0, 0, 0, 0, 0, 0]; for (let j = 0; j < 6; j++) t[j + 1] += c[j]; lv.set(p, t); }
    return JSON.stringify([...lv.values()].sort((u, v) => u[0] - v[0]));
  })();
  assert.notEqual(dbl, formingOf(load("sol"), byOld(), bar * 1000 - 1), "양성 대조: 직결 칸을 rt 칸에 더하면(이중 합산) 잡는다");
  // ⑥ 꺼지는 순간: rt 로 받은 봉 위에 refreshFootprint 가 since = 최신 봉 − 300 으로 이어받는다(봉 객체 통째로 갈아 끼움 -- 같은 봉이 겹쳐 더해지지 않음)
  const since = Math.max(...rtP.st().footprintBars.keys()) - 300;
  const inc = last.full.filter((r) => r[0] >= since);
  const fb = new Map(rtP.st().footprintBars);
  for (const [t, lv] of inc) fb.set(t, { time: t, levels: lv });            // app.js refreshFootprint: footprintBars.set(b.time, b)
  assert.deepEqual([...fb.values()].sort((u, v) => u.time - v.time).map((b) => [b.time, b.levels]), last.full, "꺼진 뒤 이어받은 응답 + rt 봉 = 원본 전량");
  const addWrong = new Map(rtP.st().footprintBars);
  for (const [t, lv] of inc) { const o = addWrong.get(t); addWrong.set(t, { time: t, levels: lv.map((l, i) => (o && o.levels[i] ? l.map((v, j) => (j ? v + o.levels[i][j] : v)) : l)) }); }
  assert.notDeepEqual([...addWrong.values()].sort((u, v) => u.time - v.time).map((b) => [b.time, b.levels]), last.full, "양성 대조: 이어받은 봉을 더하면 잡는다");
  console.log(`⑤ 형성 봉 ok -- 직결 갈아 끼움·봉 중간 붙음 둘 다 rt 경로 = 옛 경로 · 이중 합산 양성 대조 잡음 · ⑥ 꺼진 뒤 since 이어받기 = 원본 전량 · 더하기 양성 대조 잡음`);
}
