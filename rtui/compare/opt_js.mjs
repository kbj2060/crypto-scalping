// Option 카드 정답지 1/2: 지금 화면의 실제 함수(app.js renderOptions 와 그 안에서 부르는 optLadderSvg · optLaneSvg · optFlowSvg · optBlkHtml · optRealizedGamma · optTouchModel …)를
// 그대로 잘라 fixture/opt_src.json(함수 묶음 글) · 지어낸 입력 fixture/opt_in.json 을 쓴다. 돌리기·재기는 compare/opt_layout.py(헤드리스 크로미움 · 원본 styles.css · 진짜 DOM -- optColW·clientWidth 를 그대로 잰다).
// 숫자는 전부 지어낸 값. 실행: node rtui/compare/opt_js.mjs   (바깥 접속 없음) · APP_JS=<app.js 사본> SRC_OUT=<묶음 json> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
const line = (head) => { const i = src.indexOf("\n" + head); if (i < 0) throw head; return src.slice(i, src.indexOf("\n", i + 1)); };
const block = (head, end) => { const i = src.indexOf("\n" + head); if (i < 0) throw head; return src.slice(i, src.indexOf(end, i) + end.length); };
const body = [line("const helpAttrs = "), ...["RG_REF", "RG_REF_XRP", "RG_REF_BTC", "RG_REF_SOL"].map((k) => block(`const ${k} = `, "]];\n")), block("const RG_BY = ", "} };\n"),
  line("const rgWord = "), line("const optIv = "), line("const optQ = "), line("const optUsd = "), line("const optPx = "), line("const optSigma = "), line("const optFront = "),
  line("const optHead = "), line("const optPct = "), line("const optPx1 = "), line("const OPT_MON = "), block("const OPT_SHAPE = ", "};\n"), block("const optColW = ", "\n};\n"),
  ...["fmtNum", "fmtUsdCompact", "escapeHtml", "optRealizedGamma", "optData", "optKst", "optChartK", "optLaneSvg", "optCm", "optAsOf", "optNcdf", "optTouchModel", "optMyLevels",
      "optTouchBack", "optTouchFront", "optLadderChips", "optLadderSvg", "optNewTxt", "optFlowSvg", "optBlkShape", "optBlkLegs", "optBlkHtml", "optFlowTodayHtml", "optEqKpiHtml",
      "optRowCols", "optInfoParts", "renderOptions", "optEqFlowSvg"].map(fn)].join("\n");

const N = 1795000000000;   // 지어낸 지금(ms) = 11:06:40 UTC
const H = 3600e3;
const EXP0 = N + 75200e3;   // 다음 08:00 UTC
let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
// 5분봉 n 개(마지막 = 지금 직전 마감) · rho = 수익률 1차 자기상관 쪽(음수 = 되돌림)
const candles = (n, p0, rho) => {
  const t1 = Math.floor(N / 1000 / 300) * 300 - 300;
  let lp = Math.log(p0), e0 = 0;
  return Array.from({ length: n }, (_, i) => { const e = (rnd() - 0.5) * 0.004; lp += e + rho * e0; e0 = e; return { time: t1 - (n - 1 - i) * 300, close: Math.round(Math.exp(lp) * 1e4) / 1e4 }; });
};
const ladder = (px, step, n) => Array.from({ length: n }, (_, i) => { const k = Math.round((px - step * (n / 2) + step * i) / step) * step; return [Math.round(k * 1e4) / 1e4, Math.round((2e6 + 9e6 * rnd()) * (k > px ? 1.4 : 0.5)), Math.round((2e6 + 9e6 * rnd()) * (k < px ? 1.5 : 0.4))]; });
const exps = (pain, iv) => [0, 1, 2, 4, 7].map((d, i) => ({ exp_ms: EXP0 + d * 24 * H, call_oi_usd: [61e6, 23e6, 18.4e6, 140e6, 300e6][i], put_oi_usd: [44e6, 9.1e6, 12e6, 95e6, 210e6][i],
  pain: pain[i], pc: [0.72, 0.39, null, 0.68, 0.9][i], atm_iv: iv[i], oi_share: [0.083, null, null, null, null][i] }));
const flow = (n, doi, liq) => Array.from({ length: n }, (_, i) => ({ h: Math.floor(N / H) * H - (n - 1 - i) * H, cb: Math.round(400 * rnd()), cs: Math.round(400 * rnd()), pb: Math.round(300 * rnd()), ps: Math.round(300 * rnd()),
  dlt: Math.round((rnd() - 0.45) * 900), doi: doi ? Math.round((rnd() - 0.4) * 500) : null, liq: liq && i % 7 === 3 ? Math.round(30 * rnd()) : 0 }));
const blk = (dt, usd, legs, o) => ({ ts_ms: N - dt * 60e3, notional_usd: usd, legs: legs.map(([nm, d, a]) => ({ instrument_name: nm, direction: d, amount: a })), ...o });
const ETHB = [blk(12, 12.4e6, [["ETH-20NOV26-3000-C", "buy", 2500], ["ETH-20NOV26-3000-P", "buy", 2500]], { structure: "straddle", delta_side: "neutral", vol_side: "long_vol" }),
  blk(55, 3.1e6, [["ETH-27NOV26-3200-C", "sell", 1000]], { structure: "single", delta_side: "short", vol_side: "short_vol" }),
  blk(130, 8.8e6, [["ETH-25DEC26-2600-P", "buy", 3000], ["ETH-25DEC26-3600-C", "sell", 3000], ["ETH-25DEC26", "sell", 1500]], { structure: "risk_reversal", hedge: true, delta_side: "short", vol_side: "neutral" }),
  blk(300, 0.95e6, [["ETH-21NOV26-2900-P", "buy", 300], ["ETH-21NOV26-2800-P", "sell", 600], ["ETH-21NOV26-2700-P", "buy", 300], ["ETH-21NOV26-2600-P", "sell", 100]], { structure: "butterfly" }),
  blk(610, 1.6e6, [["ETH-26DEC26-3100-C", "buy", 500]], {}), blk(800, 2.2e6, [["ETH-26DEC26-3100-C", "buy", 700]], { structure: "single", delta_side: "long", vol_side: "long_vol" })];
const opt = (o) => ({ index: 3018.2, dvol: 47.2, iv30: 46.0, rv7: 38.4, recorded_at_utc: new Date(N - 4 * 60e3).toISOString(), cm: { 1: { atm: 52.1 }, 7: { atm: 48.3, rr: -1.24, bf: 0.62 }, 30: { atm: 50.2, rr: 0.41, bf: 0.88 }, 60: { atm: 51.9 } },
  surface: { front: { hours: 20.9, tail: { 2: 0.41, 3: 0.22, 5: 0.0063 } }, 7: { hours: 164.9, tail: { 5: 0.31 } } }, expiries: exps([3000, 2950, 3100, 2900, 3000], [49.8, 47.1, 46.6, 48.9, 50.1]),
  strikes: { week: ladder(3018, 25, 34), front_exp_ms: EXP0 }, ...o });
const smile = (S, coef, xlim) => ({ ok: true, exp_ms: EXP0, S, coef, xlim });
const S = (o) => ({ asset: "eth", now: N, price: null, gex: null, smile: {}, candles: [], ...o });
const inputs = [
  // 0 ETH 다 있음(닿음 등고선 · 실현 감마 되돌림 · 블록 6 · 흐름 25 미결제·강제청산)
  S({ price: 3020.55, candles: candles(160, 3000, -0.35), smile: { ETH: smile(3018.2, [1.2, -0.1, 0.48], 0.15) },
      gex: { available: true, currencies: { ETH: { options: opt({}), opt_hist: { vov24: 3.42 } }, BTC: {} },
             block_trades: { available: true, by_coin: { ETH: ETHB }, n_by_coin: { ETH: 14 }, sum_by_coin: { ETH: 48.2e6 }, max_by_coin: { ETH: { notional_usd: 12.4e6, ts_ms: N - 12 * 60e3 } }, flow_by_coin: { ETH: flow(25, true, true) } } } }),
  // 1 아직 없음
  S({}),
  // 2 옵션 안 모으는 코인
  S({ asset: "skhynix", gex: { available: true, currencies: { ETH: {} }, block_trades: {} } }),
  // 3 BTC: 갱신 지연 · 3시간 묵은 값 · 블록 0 · 블록 수집 지연 · 흐름 없음 · 5분봉 모자람 · 허브 가격 없음(지수)
  S({ asset: "btc", candles: candles(40, 91000, 0),
      gex: { error: "gex 502", available: true, currencies: { BTC: { options: opt({ index: 91234.5, dvol: 41.3, iv30: 40.2, rv7: 44.9, recorded_at_utc: new Date(N - 3 * H - 10 * 60e3).toISOString(), surface: { front: { hours: 20.9, tail: { 2: 0.5, 3: 0.3, 5: 0.995 } } },
        expiries: exps([91000, 90000, 92000, 88000, 95000], [42.1, 40.3, 40.9, 41.2, 43]), strikes: { week: ladder(91234, 500, 30), front_exp_ms: EXP0 } }) } },
        block_trades: { available: false, by_coin: {}, flow_by_coin: {} } } }),
  // 4 SOL: DVOL 없음(30일 IV) · 실현 7일 없음 · 행사가 빈칸 · 만기 하나 · 고정만기 없음 · 꼬리 없음 · 실현 감마 추세(tip rg_sol) · 블록 둘 · 흐름 미결제 없음
  S({ asset: "sol", price: 181.234, candles: candles(150, 180, 0.4),
      gex: { available: true, currencies: { SOL: { options: opt({ index: 180.9, dvol: null, iv30: 88.1, rv7: null, cm: {}, surface: {}, strikes: { week: [] }, expiries: [{ exp_ms: EXP0 + 3 * H, call_oi_usd: 2.1e6, put_oi_usd: 0.4e6, pain: 180, pc: 0.21, atm_iv: 91.2 }] }) } },
             block_trades: { available: true, by_coin: { SOL: [blk(30, 0.42e6, [["SOL_USDC-20NOV26-185-C", "buy", 2000]], { structure: "single", delta_side: "long" }), blk(90, 0.18e6, [["SOL_USDC-27NOV26-170-P", "sell", 1500]], {})] },
                             n_by_coin: { SOL: 2 }, flow_by_coin: { SOL: flow(25, false, false) } } } }),
  // 5 XRP: 작은 가격(자릿수) · 닿음 등고선 · 소수점 행사가 다리 · 흐름 강제청산
  S({ asset: "xrp", price: 0.61234, candles: candles(146, 0.61, 0.05), smile: { XRP: smile(0.6121, [1.5, -0.2, 0.62], 0.2) },
      gex: { available: true, currencies: { XRP: { options: opt({ index: 0.6121, dvol: null, iv30: 63.4, rv7: 58.1, expiries: exps([0.6, 0.62, 0.59, 0.61, 0.6], [61.2, 60.1, 62.4, 63, 64.1]), strikes: { week: ladder(0.612, 0.005, 26), front_exp_ms: EXP0 } }) } },
             block_trades: { available: true, by_coin: { XRP: [blk(44, 0.31e6, [["XRP_USDC-20NOV26-0d62-P", "buy", 250000], ["XRP_USDC-20NOV26-0d65-C", "sell", 250000]], { structure: "risk_reversal", delta_side: "short", vol_side: "neutral" })] },
                             n_by_coin: { XRP: 1 }, sum_by_coin: { XRP: 0.31e6 }, flow_by_coin: { XRP: flow(25, true, true) } } } }),
  // 6 ETH: 22분 묵은 값(25분 문턱 안) · 등고선 없음 · 가격 높음 · 블록 하나 · 흐름 미결제 없음 · 실현 감마 보통
  S({ price: 3090.1, candles: candles(150, 3080, 0.0),
      gex: { available: true, currencies: { ETH: { options: opt({ recorded_at_utc: new Date(N - 22 * 60e3).toISOString(), strikes: { week: ladder(3090, 25, 30), front_exp_ms: EXP0 } }), opt_hist: {} } },
             block_trades: { available: true, by_coin: { ETH: [ETHB[2]] }, flow_by_coin: { ETH: flow(25, false, false) } } } }),
];
fs.writeFileSync(here + "fixture/opt_in.json", JSON.stringify(inputs) + "\n");
fs.writeFileSync(process.env.SRC_OUT || here + "fixture/opt_src.json", JSON.stringify(body) + "\n");
console.log("입력", inputs.length, "함수 묶음", body.length, "글자");
