// 왼쪽 위 «지표 방향» 판 정답지(1/2): 지금 화면의 실제 코드(app.js FPB_W · fpTipOn · dirReadNow · DRM_TIP · renderDirReadMob · dirReadH · dirTri · renderDirBoard · measureTextW · escapeHtml)를
// 잘라 fixture/dirb_src.js 로 · 지어낸 입력(판 줄 rows·cond · 접힘 · /api/situation 응답 · 코인 · 시각)을 fixture/dirb_in.json 으로. 돌리기·재기는 compare/dirb_layout.py(헤드리스 크로미움 · 원본 styles.css).
// 실행: node rtui/compare/dirb_js.mjs   (바깥 접속 없음) · APP_JS=<app.js 사본> SRC=<출력 js> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
const line = (head) => { const i = src.indexOf("\n" + head); if (i < 0) throw head; return src.slice(i, src.indexOf("\n", i + 1)); };
const code = [fn("escapeHtml"), fn("measureTextW"), line("const FPB_W = "), fn("fpTipOn"), fn("dirReadNow"), line("const DRM_TIP = "), fn("renderDirReadMob"), line("const dirReadH = "),
  fn("dirTri"), fn("renderDirBoard")].join("\n");
fs.writeFileSync(process.env.SRC || here + "fixture/dirb_src.js", code + "\n");

const N = 1795000000;   // 지어낸 시각(초)
const ln = (topic, grade, dir, text, note, g) => ({ topic, grade, dir, text, note, ...(g ? { g } : {}) });
const hot = [
  ln("큰 손", "근거", 1, "큰 손 순매수 — 고래 매수 우위: 5분 +1.2M", "관찰 — 과거 검정 52.1%", { v: 64, on: true }),
  ln("OI·가격", "약함", -1, "OI 늘고 가격 내림 — 새 숏 쌓임: 15분 +0.8%", "해석 — 약한 참고"),
  ln("활동", "근거", 0, "체결 활동 — 양끝 큼: 상위 5%", "크기 — 방향 아님", { v: 85, on: false }),
];
const desc = [ln("청산", "설명", 0, "청산 — 롱 청산 몰림: 3분 2.1M", "설명", { v: 40, on: true }), ln("호가", "설명", 1, "호가 벽 위 두꺼움", "설명"),
  ln("베이시스", "설명", -1, "베이시스 — 낮아짐: 1bp", "설명", { v: 10, on: true }), ln("스프레드", "설명", 0, "스프레드 — 보통", "설명", { v: 20, on: false })];
const S = (lines, age = 5) => ({ computed_at: N - age, read: { lines } });
const rows = [
  { d: 1, name: "일봉 추세", sub: "SMA50 위 · 오늘 내내" }, { d: 1, name: "1h 구조 돌파", sub: "2시간 10분 전" }, { d: -1, name: "OB 저항", sub: "2988.5–2994.0 닿으면" },
  { d: 1, name: "CCI 1h", sub: "35분 전" }, { d: -1, name: "RSI2 눌림", sub: "" },
];
const cond = [{ d: 1, tf: "1h", v: "3021.5", wait: false }, { d: 1, tf: "30m", v: "3008.0", wait: true }, { d: -1, tf: "1h", v: "2951.0", wait: false }];
const long = [ln("아주 긴 이름의 주제 줄", "근거", 1, "아주 긴 이름 — 오른쪽 덧말이 판 폭을 넘어 말줄임표로 잘려야 하는 아주 긴 설명 문장: 끝", "x"),
  ln("짧음", "약함", -1, "콜론 없는 덧말이 아주 길어서 잘려야 함 그리고 더 길게 이어짐", "x"),
  ln("주제 이름이 꽤 길어 덧말 좁음", "근거", 1, "좁은 자리 — 덧말: 끝", "x"), ln("주제 이름이 아주 아주 길어서 덧말 자리가 없음", "약함", 0, "없음 — 안 보임", "x")];   // 덧말 자리 24~60 · 24 미만(안 그림)
const descLong = [ln("하나", "설명", 0, "a", "x", { v: 1, on: true }), ln("둘째 주제 이름", "설명", 0, "b", "x", { v: 1, on: true }), ln("셋째 주제 이름 길게", "설명", 0, "c", "x", { v: 1, on: true }),
  ln("넷째 주제 이름 더 길게", "설명", 0, "d", "x", { v: 1, on: true })];
// 갈래: 다 있음(위·아래 막대 · 줄 홀짝 · 넘으면 둘(마감 대기)·깨면 하나 · 지금 읽기 ▲▼·큰 네모 · 설명(켜진 것)) · 접힘(머리만) · 판 줄 없이 지금 읽기만(말줄임표 · 설명 켜진 것 말줄임) ·
//   줄만(읽기 낡음 200초 → 없음 · 깨면만) · ETH 아님(읽기 있어도 없음 · 위만) · 아무것도 없음(판 없음) · 아래만 + 작은 네모(80 미만·g 없음) + 낡음 경계 150초(있음) · 조건만(머리 30) · 접힘 + 줄 없음 + 읽기
const inputs = [
  { rows, cond, fold: false, S: S([...hot, ...desc]), asset: "eth" },
  { rows, cond, fold: true, S: S([...hot, ...desc]), asset: "eth" },
  { rows: [], cond: [], fold: false, S: S([...long, ...descLong]), asset: "eth" },
  { rows: rows.slice(0, 3), cond: cond.slice(2), fold: false, S: S(hot, 200), asset: "eth" },
  { rows: rows.slice(0, 2), cond: [], fold: false, S: S(hot), asset: "btc" },
  { rows: [], cond: [], fold: false, S: S([]), asset: "eth" },
  { rows: [rows[2], rows[4]], cond: [], fold: false, S: S([ln("활동", "약함", 0, "활동 — 보통", "x", { v: 79 }), ln("깊이", "근거", 0, "깊이 — 얇음", "x"), desc[1]], 150), asset: "eth" },
  { rows: [], cond, fold: false, S: null, asset: "eth" },
  { rows: [], cond: [], fold: true, S: S(hot), asset: "eth" },
].map((s) => ({ ...s, now: N, G: { ml: 60, mt: 30 }, top: 6 }));
fs.writeFileSync(here + "fixture/dirb_in.json", JSON.stringify(inputs) + "\n");
console.log("입력", inputs.length, "· 코드", code.length, "자");
