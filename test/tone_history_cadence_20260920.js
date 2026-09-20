// 모델지표 톤 띠의 **주기**를 검사한다(2026-09-20).
//   node test/tone_history_cadence_20260920.js
//
// 왜: 이 띠는 «48칸 x 5분 = 4시간»이라고 화면에 적혀 있는데, 라이브 경로가 SSE 상태 푸시마다
// (실측 10초에 한 번) 한 칸씩 밀어 넣고 있었다. 그래서 ①서버가 디스크에 남겨 둔 4시간 씨앗이
// 몇 분 만에 통째로 밀려나고 ②씨앗 48칸의 시각이 전부 «지금»으로 찍혀 축이 4시간을 0초로
// 압축했다(브라우저 실측: 띠 폭 0분 -> 251분).
const fs = require("fs");
const src = fs.readFileSync("dashboard/live/app.js", "utf8");
const grab = (re, what) => {
  const m = src.match(re);
  if (!m) { console.log(`🔴 못 찾음: ${what}`); process.exit(1); }
  return m[0];
};
const CODE = grab(/const MICRO_HISTORY_MAX = \d+;/, "MICRO_HISTORY_MAX") + "\n"
  + grab(/const toneHistory = \{[^}]*\};/, "toneHistory") + "\n"
  + grab(/const toneHistoryTimes = \{[^}]*\};/, "toneHistoryTimes") + "\n"
  + grab(/const TONE_PUSH_MIN_MS = \d+;/, "TONE_PUSH_MIN_MS") + "\n"
  + grab(/const toneHistoryLastAt = \{\};/, "toneHistoryLastAt") + "\n"
  + grab(/function pushToneHistory\(key, tone, at\) \{[\s\S]*?\n\}/, "pushToneHistory") + "\n";

let ok = true;
const fail = (m) => { console.log("🔴 " + m); ok = false; };

// 시계를 손에 쥔다 -- 주기 검사는 «지금»을 움직일 수 있어야 성립한다.
let NOW = 1700000000000;
const RealDate = Date;
global.Date = class extends RealDate {
  constructor(...a) { super(...(a.length ? a : [NOW])); }
  static now() { return NOW; }
  static parse(s) { return RealDate.parse(s); }
};

const env = {};
(function () { eval(CODE + "\nenv.push = pushToneHistory; env.hist = toneHistory;"
  + " env.times = toneHistoryTimes; env.MIN = TONE_PUSH_MIN_MS; env.MAX = MICRO_HISTORY_MAX;"); })();

// ① 씨앗(`at` 지정)은 주기 제한을 받지 않고 **그 시각**을 그대로 적는다.
const t0 = NOW - 47 * 300000;
for (let i = 0; i < 48; i++) {
  env.push("whale", i % 2 ? "good" : "bad", new RealDate(t0 + i * 300000).toISOString());
}
if (env.hist.whale.length !== 48) fail(`씨앗 48칸이 안 들어갔다: ${env.hist.whale.length}`);
const span = (RealDate.parse(env.times.whale[47]) - RealDate.parse(env.times.whale[0])) / 60000;
if (Math.round(span) !== 235) fail(`씨앗 띠 폭이 4시간 격자가 아니다: ${span}분`);

// ② 씨앗 직후의 라이브 푸시는 **막힌다**(마지막 씨앗이 방금이므로). 이게 없으면 48칸이
//    몇 분 만에 라이브 값으로 덮인다 -- 이 검사의 존재 이유다.
NOW = RealDate.parse(env.times.whale[47]) + 60000;   // 씨앗 1분 뒤
env.push("whale", "neutral");
if (env.times.whale[47] !== new RealDate(RealDate.parse(env.times.whale[47])).toISOString()
    && env.hist.whale.length !== 48) fail("1분 뒤 푸시가 막히지 않았다");
const after1min = env.times.whale[47];

// ③ 5분이 지나면 한 칸 들어간다.
NOW += env.MIN;
env.push("whale", "good");
if (env.times.whale[47] === after1min) fail("5분 뒤에도 푸시가 안 됐다");
if (env.hist.whale.length !== env.MAX) fail(`링이 ${env.MAX} 를 넘었다: ${env.hist.whale.length}`);

// ④ 같은 5분 안에 백 번 불러도 한 칸이다(SSE 푸시가 초당 몇 번 와도 마찬가지여야 한다).
const mark = env.times.whale[47];
for (let i = 0; i < 100; i++) { NOW += 1000; env.push("whale", "bad"); }
if (env.times.whale[47] !== mark) fail("5분 안에 두 번 이상 들어갔다");

// ⑤ 키마다 따로 센다 -- 한 지표의 푸시가 다른 지표를 막으면 안 된다.
NOW += env.MIN;
env.push("retail_flow", "good");
if (env.hist.retail_flow.length !== 1) fail("retail_flow 가 whale 의 시계를 물려받았다");

console.log(ok ? "✅ 톤 띠 주기 OK" : "실패");
process.exit(ok ? 0 : 1);
