"""30m·1h 묶기 · 1주일 잇기 · 휠 줌 끝 처리(app.js) -- 본문을 떼어 node 로 실제로 돌린다 (2026-10-20 리뷰).

    python3 -m pytest -q test/test_fp_tf_week_20261020.py
"""
import pathlib
import re
import subprocess

APP = pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "live" / "app.js"


def _pick(src, name):
    m = re.search(rf"^function {name}\(.*?^}}\n", src, re.S | re.M) or re.search(rf"^const {name} = [^\n]*\n", src, re.M)
    assert m, name
    return m.group(0)


def test_tf_week_zoom():
    src = APP.read_text("utf-8")
    names = ["tfT", "tfCMemo", "tfCandles", "tfLvMemo", "tfLevels", "fpWeekFor", "weekJoinMemo", "weekJoin",
             "chartPanEdge", "chartPanTo", "CHART_WINDOW_MAX", "chartZoomAt"]
    js = ("let chartTfK = 6, chartPanEnd = null, chartPanFrac = 0, chartWindowBars = 12, activeSnapshotAsset = 'eth', fpWeek = null, FULL = [];\n"
          "const assert = require('assert'), scheduleSnapshotChartRender = () => {}, maybeFetchFpWeek = () => {}, renderChartWindowTabs = () => {};\n"
          "const chartTfFull = () => FULL;\n"
          + "".join(_pick(src, n) for n in names) + r"""
const T0 = 1790000000 - 1790000000 % 1800;
const c5 = (i, o, h, l, cl) => ({ time: T0 + i * 300, open: o, high: h, low: l, close: cl });
// 앞 5봉(덜 찬 30m 봉) · 온전한 30m 봉 하나 · 형성 중 2봉
const raw = [];
for (let i = 1; i <= 5; i++) raw.push(c5(i, 1, 9, 0, 1));
for (let i = 6; i < 12; i++) raw.push(c5(i, 10 + i, 30 + i, 5 + i, 20 + i));
raw.push(c5(12, 40, 41, 39, 40.5), c5(13, 40.5, 42, 40, 41));
let tf = tfCandles(raw);
assert.deepStrictEqual(tf.map((c) => c.time), [T0 + 1800, T0 + 3600]);   // 덜 찬 맨 앞 봉은 뺀다
assert.deepStrictEqual([tf[0].open, tf[0].high, tf[0].low, tf[0].close], [16, 41, 11, 31]);
// 형성 5분봉 제자리 갱신 → 같은 배열 · 마지막 tf 봉만 제자리에서 다시 묶임
const keep = tf[1];
raw[12].high = 50; raw[12].close = 49;
tf = tfCandles(raw);
assert.ok(tf[1] === keep && keep.high === 50 && keep.close === 49 && keep.open === 40);
// 새 5분봉(push+shift -- 신원·길이 그대로) → 다시 묶는다
raw.push(c5(14, 49, 49, 48, 48)); raw.shift();
assert.strictEqual(tfCandles(raw)[1].close, 48);

// tfLevels: 5분봉이 모자란 닫힌 tf 봉은 뺀다(형성 중 마지막만 예외) · 칸은 7열 합
const lv = (p, b) => [[p, b, 1, 0, 0, 0, 0]], byT = new Map();
for (let i = 6; i < 12; i++) byT.set(T0 + i * 300, lv(100, 1));
for (let i = 12; i < 17; i++) byT.set(T0 + i * 300, lv(100, 2));   // 하나 빠짐
byT.set(T0 + 18 * 300, lv(100.5, 3));                               // 형성 중
const out = tfLevels(byT, 0.5);
assert.deepStrictEqual([...out.keys()], [T0 + 1800, T0 + 5400]);
assert.deepStrictEqual(out.get(T0 + 1800), [[100, 6, 6, 0, 0, 0, 0]]);

// weekJoin: 라이브가 제자리에서 push+shift 해도 새 봉이 붙는다
fpWeek = { asset: 'eth', candles: [{ time: 1 }, { time: 2 }, { time: 50 }] };
const live = [{ time: 10 }, { time: 20 }];
assert.strictEqual(weekJoin('candles', live, (c) => c.time).length, 4);
live.push({ time: 30 }); live.shift();
assert.deepStrictEqual(weekJoin('candles', live, (c) => c.time).map((c) => c.time), [1, 2, 20, 30]);

// 휠 줌: 실시간이면 넓혔다 좁혀도 실시간 · 12 근처는 12 로 붙는다 · 과거면 커서 밑 봉이 그 자리에
chartTfK = 1; FULL = Array.from({ length: 500 }, (_, i) => ({ time: i * 300 }));
chartZoomAt(30, 0.5); chartZoomAt(12, 0.5);
assert.ok(chartPanEnd === null && chartWindowBars === 12);
chartWindowBars = 12.0004; chartZoomAt(11, 0.5);
assert.strictEqual(chartWindowBars, 12);
chartPanTo(300); chartWindowBars = 24;
const u = chartPanEdge(FULL) - 24 + 0.25 * 24;
chartZoomAt(48, 0.25);
assert.ok(Math.abs(chartPanEdge(FULL) - 48 + 0.25 * 48 - u) < 1e-6);
console.log('ok');
""")
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
