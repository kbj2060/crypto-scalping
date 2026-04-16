const STATE_URL = "../../data/live/dashboard_state.json";
const POLL_MS = 2500;
const MICRO_HISTORY_MAX = 6; // current + past 5

const el = (id) => document.getElementById(id);
const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj || {}, key);
const microHistory = {
  obi: [],
  whale: [],
  whale_intent: [],
  eai: [],
  kelly: [],
};
let latestState = null;
let latestEvalMap = {};

const PB_INFO = {
  PB_VETO_SHIELD: {
    title: "PB 방어쉴드 (VETO)",
    threshold: "임계값: 펀딩비 극단 역진입 OR 유동성 진공+고독성",
    logic: "로직: 위험장에서는 어떤 시그널도 무시하고 진입을 차단(HOLD)합니다.",
    narration: "내레이션: 손실을 막는 신호가 수익 신호보다 우선입니다.",
  },
  PB_CRISIS_SNIPER: {
    title: "PB 위기 스나이퍼",
    threshold: "임계값: 패닉/가짜돌파/독성함정 중 가장 강한 위기 신호",
    logic: "로직: 극단 이벤트에서 역추세 고확신 타점만 오버라이드 진입합니다.",
    narration: "내레이션: 시장이 과잉반응할 때 가장 날카로운 역방향 기회를 잡습니다.",
  },
  PB_SQUEEZE_SNIPER: {
    title: "PB 스퀴즈 스나이퍼",
    threshold: "임계값: 레버리지 쏠림 + 점화 조건(EAI/펀딩/흡수)",
    logic: "로직: 압축된 포지션이 터지는 순간 스퀴즈 방향을 빠르게 추종합니다.",
    narration: "내레이션: 느린 예측보다 빠른 폭발 추종이 중요한 계층입니다.",
  },
  PB_TREND_SIGNAL: {
    title: "PB 추세 신호",
    threshold: "임계값: 추세 정합(OBI·고래·저독성) 기반 점수 우위",
    logic: "로직: 저독성 환경에서 호가/고래 방향이 정렬되는 건강한 추세를 추종합니다.",
    narration: "내레이션: 왜곡이 적은 추세 구간을 선별해 유지력을 우선합니다.",
  },
  PB_WHALE_SIGNAL: {
    title: "PB 고래 신호",
    threshold: "임계값: 고래 누적체결/잠행흡수 기반 점수 우위",
    logic: "로직: 가격보다 고래 체결 누적의 방향성을 우선 반영합니다.",
    narration: "내레이션: 큰 자금의 발자국이 남는 방향이 결국 중기 방향을 만듭니다.",
  },
  PB_MEAN_REVERT_SIGNAL: {
    title: "PB 평균회귀 신호",
    threshold: "임계값: VWAP 이격/청산 자석/OI 발산 기반 회귀 점수 우위",
    logic: "로직: 과열·연료고갈·청산자석 구간을 평균 복귀 관점으로 통합 판단합니다.",
    narration: "내레이션: 과열과 공포의 꼬리를 중심값으로 되돌리는 계층입니다.",
  },
  PB9_VACUUM_WHIPSAW: {
    title: "PB9 유동성 붕괴 긴급회피",
    threshold: "임계값: queue_collapse > 0.75 && toxicity > 0.85",
    logic: "로직: 호가 공백과 독성이 동시에 높으면 시장가 진입 금지, HOLD/지정가 대응만 허용",
    narration: "내레이션: 시장이 비어있을 때는 방향이 맞아도 체결가가 무너집니다. 먼저 살아남는 모드로 전환합니다.",
  },
  PB_FUNDING_EXTREME_HOLD: {
    title: "PB 펀딩비 극단 진입차단",
    threshold: "임계값: |funding_rate| > 0.002 + 비용 불리한 방향 진입 시도",
    logic: "로직: 펀딩비가 극단적으로 불리하면 해당 방향 신규 진입을 강제 HOLD로 차단",
    narration: "내레이션: 방향이 맞아도 보유비용이 수익을 잠식하는 구간은 진입 자체를 멈춥니다.",
  },
  PB5_MAMMOTH_SNIPER: {
    title: "PB5 폭락장 V반등 저점포착",
    threshold: "임계값: LAI 방어 + 고래 순매수 + 흡수 강화 + 여진 완화",
    logic: "로직: 청산 매물이 쏟아져도 하락이 멈추고 고래가 받아먹는 구간을 LONG 스나이핑",
    narration: "내레이션: 공포가 끝나고 수급이 바닥에서 뒤집히는 순간만 노려 진입합니다.",
  },
  PB13_BREAKOUT_TRAP: {
    title: "PB13 가짜돌파 역추세 저격",
    threshold: "임계값: 소프트 돌파(30분 변동률) + 고독성 + 고래 반대체결",
    logic: "로직: 이산 돌파 플래그 대신 연속형 돌파 점수를 사용해 함정 구간을 역방향 저격",
    narration: "내레이션: 군중이 돌파를 추격할 때, 세력이 던지는 쪽으로 선행 대응합니다.",
  },
  PB8_HOLY_TRINITY_TRAP: {
    title: "PB8 가짜벽 역추적 선행매매",
    threshold: "임계값: OBI-고래체결 다이버전스 + 높은 독성",
    logic: "로직: 호가벽 방향과 실체결 방향이 충돌하면 벽을 무시하고 고래 체결 방향 추종",
    narration: "내레이션: 보이는 벽보다 실제로 누가 시장가를 치는지가 핵심입니다.",
  },
  PB2_SQUEEZE_IGNITION: {
    title: "PB2 응축에너지 돌파추격",
    threshold: "임계값: EAI 고점 + 편향 펀딩 + 흡수 강화",
    logic: "로직: 한쪽으로 쏠린 포지션이 압축될 때 반대방향 스퀴즈 폭발을 추격",
    narration: "내레이션: 스프링이 눌린 구간에서 점화 신호가 뜨면 빠르게 동승합니다.",
  },
  PB7_HOLY_TRINITY_TREND: {
    title: "PB7 저독성 정배열 추세추종",
    threshold: "임계값: OBI와 고래체결 방향 일치 + 낮은 독성",
    logic: "로직: 왜곡이 적고 수급이 정직하게 한쪽으로 정렬되면 추세 지속 구간으로 판단",
    narration: "내레이션: 기만이 적은 장에서는 짧게 자르지 않고 추세를 길게 가져갑니다.",
  },
  PB10_CVD_DIVERGENCE: {
    title: "PB10 고래은닉 매집·분배 추적",
    threshold: "임계값: 30분 고래누적체결(|sum|) 크고 가격변화는 제한적",
    logic: "로직: 가격은 정체인데 체결 누적이 한쪽으로 쌓이면 은닉 매집/분배로 보고 선행 진입",
    narration: "내레이션: 차트보다 체결 누적의 방향을 먼저 믿는 플레이북입니다.",
  },
  PB_LIQUIDATION_MAGNET: {
    title: "PB 청산자석 클러스터 추종",
    threshold: "임계값: 청산클러스터 강도↑ + 클러스터 근접 + 저독성",
    logic: "로직: 현재가 근처 청산 밀집 방향으로 가격이 빨려가는 구간을 짧게 추종",
    narration: "내레이션: 청산이 몰린 가격대는 자석처럼 작동하므로, 가까울수록 반응 속도를 우선합니다.",
  },
  PB_OI_DIVERGENCE: {
    title: "PB OI 발산 역추세 포착",
    threshold: "임계값: 가격 30분 변동 + OI 감소(oi_delta_pct<0) + 저독성",
    logic: "로직: 가격은 진행되는데 OI가 줄면 추세 연료 고갈로 보고 역추세 스냅백을 노림",
    narration: "내레이션: 참여자가 줄어든 추세는 쉽게 꺾이므로 되돌림 구간을 짧게 공략합니다.",
  },
  PB12_FUNDING_SNAPBACK: {
    title: "PB12 펀딩 과열 되돌림 포착",
    threshold: "임계값: 극단 펀딩 + EAI 델타 둔화 + 고래 반대체결",
    logic: "로직: 과열된 포지션 쏠림이 꺾일 때 스냅백 구간을 반대방향으로 공략",
    narration: "내레이션: 시장이 한쪽으로 너무 몰리면 결국 반대쪽 복원탄성이 발생합니다.",
  },
  PB11_TWAP_ABSORPTION: {
    title: "PB11 저변동 잠행 매집 추종",
    threshold: "임계값: 낮은 변동성 + 높은 평균 흡수 + 방향 편향 누적",
    logic: "로직: 큰 흔들림 없이 한 방향 흡수가 계속되면 기관성 TWAP 흐름으로 판단",
    narration: "내레이션: 조용할수록 무거운 자금이 누적될 수 있어 천천히 방향을 맞춥니다.",
  },
  PB15_VWAP_MEAN_REVERSION: {
    title: "PB15 VWAP 탄성 회귀",
    threshold: "임계값: |가격-VWAP 이격| 확대 + 저변동 + 흡수 강화 + 고래 중립",
    logic: "로직: 박스권에서 VWAP 밴드 이탈이 과도하면 중심선 회귀를 노려 역추세 진입",
    narration: "내레이션: 추세가 약한 구간에서 과매수·과매도 스파이크를 짧게 되돌림 수익으로 전환합니다.",
  },
};

function fmtNum(v, d = 2) {
  return Number(v || 0).toFixed(d);
}

function fmtPct(v, d = 2) {
  const n = Number(v || 0);
  const s = n >= 0 ? "+" : "";
  return `${s}${n.toFixed(d)}%`;
}

function clamp01(v) {
  return Math.max(0, Math.min(1, Number(v || 0)));
}

function fmtTs(v) {
  if (!v) return "-";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function fmtShortTs(v) {
  if (!v) return "-";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) {
    const s = String(v);
    return s.length > 16 ? s.slice(0, 16) : s;
  }
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${mo}-${dd} ${hh}:${mm}`;
}

function fmtDateYmd(v) {
  if (!v) return "-";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) {
    const s = String(v);
    return s.length >= 10 ? s.slice(0, 10) : s;
  }
  const yyyy = String(d.getFullYear());
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mo}-${dd}`;
}

function fmtNowClock() {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function buildSessionHtml(sess) {
  const sAsiaOn = Number(sess.session_asia || 0) >= 0.5;
  const sEurOn = Number(sess.session_europe || 0) >= 0.5;
  const sUsOn = Number(sess.session_us || 0) >= 0.5;
  return [
    `<span class="session-item ${sAsiaOn ? "on" : "off"}"><span class="hist-dot session-led ${sAsiaOn ? "on" : "off"}"></span>ASIA ${sAsiaOn ? "ON" : "OFF"}</span>`,
    `<span class="session-sep">|</span>`,
    `<span class="session-item ${sEurOn ? "on" : "off"}"><span class="hist-dot session-led ${sEurOn ? "on" : "off"}"></span>EUROPE ${sEurOn ? "ON" : "OFF"}</span>`,
    `<span class="session-sep">|</span>`,
    `<span class="session-item ${sUsOn ? "on" : "off"}"><span class="hist-dot session-led ${sUsOn ? "on" : "off"}"></span>US ${sUsOn ? "ON" : "OFF"}</span>`,
  ].join("");
}

function actionLabel(a) {
  if (a === 1) return { text: "LONG", icon: "▲", cls: "long" };
  if (a === 2) return { text: "SHORT", icon: "▼", cls: "short" };
  return { text: "HOLD", icon: "⏸", cls: "hold" };
}

function initPlaybookModal() {
  const modal = el("pbModal");
  if (!modal) return;
  const titleEl = el("pbModalTitle");
  const thEl = el("pbModalThreshold");
  const logicEl = el("pbModalLogic");
  const condEl = el("pbModalConditions");
  const narEl = el("pbModalNarration");
  const close = () => {
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
  };
  const open = (pbKey) => {
    const info = PB_INFO[pbKey] || {
      title: pbKey || "플레이북 설명",
      threshold: "임계값: -",
      logic: "로직: -",
      narration: "내레이션: -",
    };
    titleEl.textContent = info.title;
    thEl.textContent = info.threshold;
    logicEl.textContent = info.logic;
    condEl.innerHTML = buildPbConditionReport(pbKey, latestState, latestEvalMap[pbKey]);
    narEl.textContent = info.narration;
    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");
  };

  document.querySelectorAll(".pb-row[data-pb]").forEach((row) => {
    row.addEventListener("click", () => open(row.getAttribute("data-pb") || ""));
  });
  el("pbModalClose")?.addEventListener("click", close);
  modal.querySelector(".pb-modal-backdrop")?.addEventListener("click", close);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) close();
  });
}

function yn(cond) {
  return cond ? "✅" : "❌";
}

function cItem(label, cond, threshold, current) {
  return {
    label: String(label || "-"),
    cond: Boolean(cond),
    threshold: String(threshold || "-"),
    current: String(current || "-"),
    progress: Number.isFinite(Number(arguments[4])) ? clamp01(Number(arguments[4])) : (cond ? 1 : 0),
  };
}

function ratioGE(val, th) {
  const v = Number(val || 0);
  const t = Number(th || 0);
  if (t <= 0) return 0;
  return clamp01(v / t);
}

function ratioLE(val, th) {
  const v = Number(val || 0);
  const t = Number(th || 0);
  if (t <= 0) return 0;
  if (v <= 0) return 1;
  return clamp01(t / v);
}

function ratioABSGE(val, th) {
  return ratioGE(Math.abs(Number(val || 0)), th);
}

function ratioABSLE(val, th) {
  return ratioLE(Math.abs(Number(val || 0)), th);
}

function condIcon(label) {
  if (label.includes("펀딩")) return "💸";
  if (label.includes("휩소") || label.includes("진공")) return "🌪️";
  if (label.includes("스퀴즈")) return "🚀";
  if (label.includes("맘모스") || label.includes("바닥")) return "🩸";
  if (label.includes("가짜돌파")) return "🎯";
  if (label.includes("독성")) return "☣️";
  if (label.includes("추세")) return "🌊";
  if (label.includes("고래")) return "🐋";
  if (label.includes("OI")) return "🧲";
  if (label.includes("VWAP")) return "📏";
  if (label.includes("청산")) return "⚡";
  return "•";
}

function renderConditionCards(items, matched, score) {
  const statusCls = matched ? "ok" : score >= 0.5 ? "warn" : "idle";
  const statusTxt = matched ? "발동" : score >= 0.5 ? "근접" : "대기";
  const head = `<div class="cond-head ${statusCls}">
    <span class="cond-state">${statusTxt}</span>
    <span class="cond-score">${fmtNum(score * 100, 0)}점</span>
  </div>`;
  const rows = (items || []).map((it) => `
    <div class="cond-row ${it.cond ? "ok" : "no"}">
      <div class="cond-title">${condIcon(it.label)} ${it.label}</div>
      <div class="cond-chip ${it.cond ? "ok" : "no"}">${yn(it.cond)}</div>
      <div class="cond-meter-wrap">
        <div class="cond-meter"><span style="width:${fmtNum((it.progress || 0) * 100, 0)}%"></span></div>
        <span class="cond-meter-txt">${fmtNum((it.progress || 0) * 100, 0)}%</span>
      </div>
      <div class="cond-meta">
        <span class="cond-k">기준</span><span class="cond-v">${it.threshold}</span>
      </div>
      <div class="cond-meta">
        <span class="cond-k">현재</span><span class="cond-v">${it.current}</span>
      </div>
    </div>
  `).join("");
  return `${head}<div class="cond-grid">${rows}</div>`;
}

function buildPbConditionReport(pbKey, state, evalObj) {
  if (!state) return `<div class="cond-empty">조건 평가 데이터 대기 중</div>`;
  const micro = state.microstructure || {};
  const tail = state.tail_risk || {};
  const sig = state.signal || {};
  const score = Number((evalObj?.meta || {}).unified_score || 0);
  const matched = Boolean(evalObj?.matched);
  const items = [];

  const funding = Number(micro.funding_rate || 0);
  const collapse = Number(micro.queue_collapse || 0);
  const tox = Number(micro.toxicity_score || 0);
  const eai = Number(micro.eai || 0);
  const absb = Number(micro.absorption_score || 0);
  const oiDelta = Number(micro.oi_delta_pct || 0);
  const obi = Number(micro.obi || 0);
  const whale = Number(micro.nif_whale || 0);
  const price30 = Number(micro.price_change_30m || 0);
  const vol30 = Number(micro.price_volatility_30m || 0);
  const vwapGap = Number(micro.vwap_gap_15m || 0);
  const eaiDelta = Number(micro.eai_delta_15m || 0);
  const nifSum30 = Number(micro.nif_whale_sum_30m || 0);
  const nifStd30 = Number(micro.nif_whale_std_30m || 0);
  const toxAvg30 = Number(micro.toxicity_avg_30m || 0);
  const nifAvg30 = Number(micro.nif_whale_avg_30m || 0);
  const absAvg30 = Number(micro.absorption_avg_30m || 0);
  const biasAvg30 = Number(micro.bias_avg_30m || 0);
  const lai = Number(tail.lai || 0);
  const aft = Number(tail.aftershock_prob || 0);
  const liqDir = Number(tail.liq_cluster_direction || 0);
  const liqStr = Number(tail.liq_cluster_strength || 0);
  const liqDist = Number(tail.distance_to_cluster_pct || 1);
  const baseAction = Number(sig.final_action || 0);
  const nifZ = nifSum30 / (Math.max(nifStd30, 1e-8) * Math.sqrt(30) + 1e-8);

  if (pbKey === "PB_VETO_SHIELD") {
    const fundExtreme = Math.abs(funding) > 0.002;
    const revBlock = (funding > 0 && baseAction === 1) || (funding < 0 && baseAction === 2);
    const fundVeto = fundExtreme && revBlock;
    const vacuum = collapse > 0.75 && tox > 0.85;
    items.push(cItem("펀딩 절대값", fundExtreme, "|funding| > 0.002", fmtNum(Math.abs(funding), 4), ratioABSGE(funding, 0.002)));
    items.push(cItem("역방향 차단 규칙", revBlock, "funding>0→LONG 차단 / funding<0→SHORT 차단", `funding=${fmtNum(funding,4)}, action=${baseAction}`, revBlock ? 1 : 0));
    items.push(cItem("펀딩 VETO 최종", fundVeto, "상기 2조건 동시", `${fundVeto}`, fundVeto ? 1 : Math.min(ratioABSGE(funding, 0.002), revBlock ? 1 : 0)));
    items.push(cItem("큐 붕괴", collapse > 0.75, "collapse > 0.75", fmtNum(collapse, 2), ratioGE(collapse, 0.75)));
    items.push(cItem("독성 급등", tox > 0.85, "tox > 0.85", fmtNum(tox, 2), ratioGE(tox, 0.85)));
    items.push(cItem("진공 VETO 최종", vacuum, "큐 붕괴 & 독성 급등", `${vacuum}`, vacuum ? 1 : Math.min(ratioGE(collapse, 0.75), ratioGE(tox, 0.85))));
    return renderConditionCards(items, matched, score);
  }

  if (pbKey === "PB_CRISIS_SNIPER") {
    const pb5 = lai > 2.25e8 && whale >= 0.35 && absb > 0.62 && aft < 0.65;
    const pb13Breakout = price30 >= 0.015;
    const pb13Breakdown = price30 <= -0.015;
    const pb13 = (pb13Breakout && tox > 0.70 && whale < -0.10) || (pb13Breakdown && tox > 0.70 && whale > 0.10);
    const pb8 = (obi > 0.35 && whale < -0.25 && tox > 0.75) || (obi < -0.35 && whale > 0.25 && tox > 0.75);
    items.push(cItem("PB5-LAI", lai > 2.25e8, "LAI > 2.25억", `${(lai / 1e8).toFixed(2)}억`, ratioGE(lai, 2.25e8)));
    items.push(cItem("PB5-고래 매수", whale >= 0.35, "whale ≥ 0.35", fmtNum(whale, 2), ratioGE(whale, 0.35)));
    items.push(cItem("PB5-흡수", absb > 0.62, "absorption > 0.62", fmtNum(absb, 2), ratioGE(absb, 0.62)));
    items.push(cItem("PB5-여진완화", aft < 0.65, "aftershock < 0.65", fmtNum(aft, 2), ratioLE(aft, 0.65)));
    items.push(cItem("PB5 최종", pb5, "PB5 4개 조건 동시", `${pb5}`, pb5 ? 1 : Math.min(ratioGE(lai, 2.25e8), ratioGE(whale, 0.35), ratioGE(absb, 0.62), ratioLE(aft, 0.65))));
    const brk = pb13Breakout || pb13Breakdown;
    items.push(cItem("PB13-소프트 돌파", brk, "|Δ30| >= 1.5%", `${fmtNum(price30 * 100, 2)}%`, ratioABSGE(price30, 0.015)));
    items.push(cItem("PB13-독성", tox > 0.70, "tox > 0.70", fmtNum(tox, 2), ratioGE(tox, 0.70)));
    items.push(cItem("PB13-고래 역행", Math.abs(whale) > 0.10, "|whale| > 0.10", fmtNum(whale, 2), ratioABSGE(whale, 0.10)));
    items.push(cItem("PB13 최종", pb13, "PB13 3개 조건 동시", `${pb13}`, pb13 ? 1 : Math.min(ratioABSGE(price30, 0.015), ratioGE(tox, 0.70), ratioABSGE(whale, 0.10))));
    items.push(cItem("PB8-호가벽", Math.abs(obi) > 0.35, "|obi| > 0.35", fmtNum(obi, 2), ratioABSGE(obi, 0.35)));
    items.push(cItem("PB8-고래 체결", Math.abs(whale) > 0.25, "|whale| > 0.25", fmtNum(whale, 2), ratioABSGE(whale, 0.25)));
    items.push(cItem("PB8-독성", tox > 0.75, "tox > 0.75", fmtNum(tox, 2), ratioGE(tox, 0.75)));
    items.push(cItem("PB8 최종", pb8, "PB8 3개 조건 동시", `${pb8}`, pb8 ? 1 : Math.min(ratioABSGE(obi, 0.35), ratioABSGE(whale, 0.25), ratioGE(tox, 0.75))));
    return renderConditionCards(items, matched, score);
  }

  if (pbKey === "PB_SQUEEZE_SNIPER") {
    const pb2 = eai > 2.0 && Math.abs(funding) > 0.001 && absb > 0.60 && oiDelta > 0.003;
    const pb12 = Math.abs(funding) > 0.001 && eaiDelta < 0 && Math.abs(whale) > 0.2;
    items.push(cItem("PB2-EAI", eai > 2.0, "EAI > 2.0", fmtNum(eai, 2), ratioGE(eai, 2.0)));
    items.push(cItem("PB2-펀딩 압축", Math.abs(funding) > 0.001, "|funding| > 0.001", fmtNum(funding, 4), ratioABSGE(funding, 0.001)));
    items.push(cItem("PB2-흡수", absb > 0.60, "absorption > 0.60", fmtNum(absb, 2), ratioGE(absb, 0.60)));
    items.push(cItem("PB2-OI 증가", oiDelta > 0.003, "OIΔ > 0.003", fmtNum(oiDelta, 4), ratioGE(oiDelta, 0.003)));
    items.push(cItem("PB2 최종", pb2, "PB2 4개 조건 동시", `${pb2}`, pb2 ? 1 : Math.min(ratioGE(eai, 2.0), ratioABSGE(funding, 0.001), ratioGE(absb, 0.60), ratioGE(oiDelta, 0.003))));
    items.push(cItem("PB12-펀딩 과열", Math.abs(funding) > 0.001, "|funding| > 0.001", fmtNum(funding, 4), ratioABSGE(funding, 0.001)));
    items.push(cItem("PB12-EAI 둔화", eaiDelta < 0, "eaiΔ15 < 0", fmtNum(eaiDelta, 3), clamp01(Math.max(-eaiDelta, 0) / 0.05)));
    items.push(cItem("PB12-고래 반대", Math.abs(whale) > 0.2, "|whale| > 0.2", fmtNum(whale, 2), ratioABSGE(whale, 0.2)));
    items.push(cItem("PB12 최종", pb12, "PB12 3개 조건 동시", `${pb12}`, pb12 ? 1 : Math.min(ratioABSGE(funding, 0.001), clamp01(Math.max(-eaiDelta, 0) / 0.05), ratioABSGE(whale, 0.2))));
    return renderConditionCards(items, matched, score);
  }

  if (pbKey === "PB_TREND_SIGNAL") {
    const pb7 = Math.abs(obi) > 0.25 && Math.abs(whale) > 0.25 && tox < 0.38 && toxAvg30 < 0.30 && Math.abs(nifAvg30) > 0.10;
    items.push(cItem("PB7-호가 정렬", Math.abs(obi) > 0.25, "|obi| > 0.25", fmtNum(obi, 2), ratioABSGE(obi, 0.25)));
    items.push(cItem("PB7-고래 정렬", Math.abs(whale) > 0.25, "|whale| > 0.25", fmtNum(whale, 2), ratioABSGE(whale, 0.25)));
    items.push(cItem("PB7-독성(즉시)", tox < 0.38, "tox < 0.38", fmtNum(tox, 2), ratioLE(tox, 0.38)));
    items.push(cItem("PB7-독성(30m)", toxAvg30 < 0.30, "tox30 < 0.30", fmtNum(toxAvg30, 2), ratioLE(toxAvg30, 0.30)));
    items.push(cItem("PB7-고래평균(30m)", Math.abs(nifAvg30) > 0.10, "|nif_avg_30m| > 0.10", fmtNum(nifAvg30, 2), ratioABSGE(nifAvg30, 0.10)));
    items.push(cItem("PB7 최종", pb7, "PB7 5개 조건 동시", `${pb7}`, pb7 ? 1 : Math.min(ratioABSGE(obi, 0.25), ratioABSGE(whale, 0.25), ratioLE(tox, 0.38), ratioLE(toxAvg30, 0.30), ratioABSGE(nifAvg30, 0.10))));
    return renderConditionCards(items, matched, score);
  }

  if (pbKey === "PB_WHALE_SIGNAL") {
    const pb10 = Math.abs(nifZ) > 2.0 && Math.abs(price30) <= 0.002;
    const pb11 = vol30 > 0 && vol30 < 0.005 && absAvg30 > 0.75 && Math.abs(biasAvg30) > 0.18;
    items.push(cItem("PB10-고래 z", Math.abs(nifZ) > 2.0, "|nif_z_30m| > 2.0", fmtNum(nifZ, 2), ratioABSGE(nifZ, 2.0)));
    items.push(cItem("PB10-가격 고정", Math.abs(price30) <= 0.002, "|Δ30| <= 0.2%", `${fmtNum(price30*100,2)}%`, ratioABSLE(price30, 0.002)));
    items.push(cItem("PB10 최종", pb10, "PB10 2개 조건 동시", `${pb10}`, pb10 ? 1 : Math.min(ratioABSGE(nifZ, 2.0), ratioABSLE(price30, 0.002))));
    items.push(cItem("PB11-저변동", vol30 > 0 && vol30 < 0.005, "0 < vol30 < 0.5%", `${fmtNum(vol30*100,2)}%`, ratioLE(vol30, 0.005)));
    items.push(cItem("PB11-흡수 평균", absAvg30 > 0.75, "abs30 > 0.75", fmtNum(absAvg30, 2), ratioGE(absAvg30, 0.75)));
    items.push(cItem("PB11-편향 평균", Math.abs(biasAvg30) > 0.18, "|bias30| > 0.18", fmtNum(biasAvg30, 2), ratioABSGE(biasAvg30, 0.18)));
    items.push(cItem("PB11 최종", pb11, "PB11 3개 조건 동시", `${pb11}`, pb11 ? 1 : Math.min(ratioLE(vol30, 0.005), ratioGE(absAvg30, 0.75), ratioABSGE(biasAvg30, 0.18))));
    return renderConditionCards(items, matched, score);
  }

  if (pbKey === "PB_MEAN_REVERT_SIGNAL") {
    const pb15 = Math.abs(vwapGap) > 0.004 && vol30 < 0.006 && absb > 0.60 && Math.abs(whale) < 0.20;
    const pbLiq = Math.abs(liqDir) > 0 && liqStr >= 0.22 && liqDist <= 0.005 && tox < 0.40;
    const pbOi = Math.abs(price30) > 0.003 && oiDelta < -0.005 && tox < 0.50;
    items.push(cItem("PB15-VWAP 이격", Math.abs(vwapGap) > 0.004, "|gap| > 0.4%", `${fmtNum(vwapGap*100,2)}%`, ratioABSGE(vwapGap, 0.004)));
    items.push(cItem("PB15-저변동", vol30 < 0.006, "vol30 < 0.6%", `${fmtNum(vol30*100,2)}%`, ratioLE(vol30, 0.006)));
    items.push(cItem("PB15-흡수", absb > 0.60, "absorption > 0.60", fmtNum(absb, 2), ratioGE(absb, 0.60)));
    items.push(cItem("PB15-고래중립", Math.abs(whale) < 0.20, "|whale| < 0.20", fmtNum(whale, 2), ratioABSLE(whale, 0.20)));
    items.push(cItem("PB15 최종", pb15, "PB15 4개 조건 동시", `${pb15}`, pb15 ? 1 : Math.min(ratioABSGE(vwapGap, 0.004), ratioLE(vol30, 0.006), ratioGE(absb, 0.60), ratioABSLE(whale, 0.20))));
    items.push(cItem("PB_LIQ-방향", Math.abs(liqDir) > 0, "|dir| > 0", `${liqDir}`, Math.abs(liqDir) > 0 ? 1 : 0));
    items.push(cItem("PB_LIQ-강도", liqStr >= 0.22, "strength >= 0.22", fmtNum(liqStr, 2), ratioGE(liqStr, 0.22)));
    items.push(cItem("PB_LIQ-거리", liqDist <= 0.005, "dist <= 0.5%", `${fmtNum(liqDist*100,2)}%`, ratioLE(liqDist, 0.005)));
    items.push(cItem("PB_LIQ-독성", tox < 0.40, "tox < 0.40", fmtNum(tox, 2), ratioLE(tox, 0.40)));
    items.push(cItem("PB_LIQ 최종", pbLiq, "PB_LIQ 4개 조건 동시", `${pbLiq}`, pbLiq ? 1 : Math.min((Math.abs(liqDir) > 0 ? 1 : 0), ratioGE(liqStr, 0.22), ratioLE(liqDist, 0.005), ratioLE(tox, 0.40))));
    items.push(cItem("PB_OI-30분 변동", Math.abs(price30) > 0.003, "|Δ30| > 0.3%", `${fmtNum(price30*100,2)}%`, ratioABSGE(price30, 0.003)));
    items.push(cItem("PB_OI-OI 감소", oiDelta < -0.005, "OIΔ < -0.005", fmtNum(oiDelta, 4), clamp01(Math.max(-oiDelta, 0) / 0.005)));
    items.push(cItem("PB_OI-독성 제한", tox < 0.50, "tox < 0.50", fmtNum(tox, 2), ratioLE(tox, 0.50)));
    items.push(cItem("PB_OI 최종", pbOi, "PB_OI 3개 조건 동시", `${pbOi}`, pbOi ? 1 : Math.min(ratioABSGE(price30, 0.003), clamp01(Math.max(-oiDelta, 0) / 0.005), ratioLE(tox, 0.50))));
    return renderConditionCards(items, matched, score);
  }

  return `<div class="cond-empty">조건 규칙이 등록되지 않은 카드입니다.</div>`;
}

function edgeInterpret(v) {
  const x = Number(v || 0);
  if (x >= 0.4) return "LONG 우위 강함";
  if (x >= 0.15) return "LONG 우위 약함";
  if (x <= -0.4) return "SHORT 우위 강함";
  if (x <= -0.15) return "SHORT 우위 약함";
  return "방향 중립";
}

function toxInterpret(v, dir = 0) {
  const x = clamp01(v);
  const dirTxt = dir > 0 ? "매수벽" : dir < 0 ? "매도벽" : "양방향";
  if (x >= 0.7) return `${dirTxt} 독성 높음: 진입 경계`;
  if (x >= 0.4) return `${dirTxt} 독성 주의: 비중 축소`;
  return `${dirTxt} 독성 낮음: 정상 구간`;
}

function aftershockInterpret(v, dir = 0) {
  const x = clamp01(v);
  const dirTxt = dir > 0 ? "상방" : dir < 0 ? "하방" : "양방향";
  if (x >= 0.7) return `${dirTxt} 여진 높음: 진입 경계`;
  if (x >= 0.4) return `${dirTxt} 여진 주의: 비중 축소`;
  return `${dirTxt} 여진 낮음: 정상 구간`;
}

function riskClass(v, inverse = false) {
  const x = Number(v || 0);
  if (inverse) {
    if (x < 0.35) return "good";
    if (x < 0.65) return "warn";
    return "bad";
  }
  if (x > 0) return "good";
  if (x < 0) return "bad";
  return "muted";
}

function obiLabel(v) {
  const x = Number(v || 0);
  if (x >= 0.3) return `강한 매수벽 (${fmtNum(x, 2)})`;
  if (x <= -0.3) return `강한 매도벽 (${fmtNum(x, 2)})`;
  return `호가 균형 (${fmtNum(x, 2)})`;
}

function whaleLabel(micro) {
  const x = Number(micro.nif_whale || 0);
  if (x >= 0.2) return `매집중 (${fmtNum(x, 2)})`;
  if (x <= -0.2) return `털기 (${fmtNum(x, 2)})`;
  return `관망중 (${fmtNum(x, 2)})`;
}

function whaleDescLabel(v) {
  const x = Number(v || 0);
  if (x >= 0.2) return "매수 우위 구간";
  if (x <= -0.2) return "매도 우위 구간";
  return "중립 구간";
}

function splitStateAndScore(label) {
  const s = String(label || "");
  const m = s.match(/^(.*)\s(\([^)]+\))$/);
  if (m) return { state: m[1].trim(), score: m[2].trim() };
  return { state: s, score: "" };
}

function getStatusColor(status) {
  const s = String(status || "");
  if (s.includes("매도") || s.includes("숏")) return "#ff7575";
  if (s.includes("매수") || s.includes("매집")) return "#2fd077";
  return "#66D8FF";
}

function getWhaleIntentColor(status) {
  const s = String(status || "");
  if (s.includes("롱 구축") || s.includes("롱구축") || s.includes("매수 우세")) return "#2fd077";
  if (s.includes("숏 구축") || s.includes("숏구축") || s.includes("매도 우세")) return "#ff7575";
  if (s.includes("중립")) return "#66D8FF";
  return "#66D8FF";
}

function stripLeadingEmoji(s) {
  return String(s || "")
    .replace(/^[\s\uFE0F\u200D\u2600-\u27BF\u{1F300}-\u{1FAFF}]+/gu, "")
    .trim();
}

function whaleIntentLabel(micro) {
  const s = Number(micro.whale_sell_presence_ratio_30m || 0);
  const buyP = Number(micro.whale_buy_presence_ratio_30m || 0);
  const oiDelta = Number(micro.oi_delta_pct || 0);
  const oiCum5m = Number(micro.oi_delta_cum_5m ?? oiDelta);
  const bias = stripLeadingEmoji(micro.whale_position_bias_30m || "중립");
  if (s <= 0.05 && buyP <= 0.05) return `중립(신호 약함)`;
  return `${bias}`;
}

function whaleIntentHistoryLabel(micro) {
  const s = Number(micro.whale_sell_presence_ratio_30m || 0);
  const buyP = Number(micro.whale_buy_presence_ratio_30m || 0);
  const bias = stripLeadingEmoji(micro.whale_position_bias_30m || "중립");
  if (s <= 0.05 && buyP <= 0.05) return "중립(신호 약함)";
  return `${bias}`;
}

function whaleIntentGuideLabel(micro) {
  const bias = stripLeadingEmoji(micro.whale_position_bias_30m || "중립");
  const est = String(micro.whale_position_estimate || "NEUTRAL").toUpperCase();
  const score = Number(micro.whale_position_score ?? 0);
  const longPct = Math.round(clamp01((score + 1) / 2) * 100);
  const shortPct = Math.round((1 - clamp01((score + 1) / 2)) * 100);
  const conf = est === "LONG" ? longPct : est === "SHORT" ? shortPct : Math.max(longPct, shortPct);
  const estText = est === "LONG" ? "롱 우위" : est === "SHORT" ? "숏 우위" : "중립";
  const prefix = `추정 ${estText} ${fmtNum(conf, 0)}%`;
  if (bias.includes("신규 롱 구축")) return `${prefix} · 대응: LONG 동승`;
  if (bias.includes("신규 숏 구축")) return `${prefix} · 대응: SHORT 동승`;
  if (bias.includes("숏 커버링")) return `${prefix} · 대응: 롱 추격 금지`;
  if (bias.includes("기존 롱 청산")) return `${prefix} · 대응: 눌림목 LONG 대기`;
  return `${prefix} · 대응: 관망(HOLD)`;
}

function eaiLabel(v) {
  const x = Number(v || 0);
  if (x >= 2.0) return `에너지 응축 (EAI ${fmtNum(x, 1)})`;
  return `변동성 평온 (EAI ${fmtNum(x, 1)})`;
}

function kellyGuideLabel(v) {
  const x = Number(v || 1);
  if (x >= 1.3) return `공격적 확대 (×${fmtNum(x, 2)})`;
  if (x >= 1.1) return `소폭 확대 (×${fmtNum(x, 2)})`;
  if (x <= 0.6) return `방어적 축소 (×${fmtNum(x, 2)})`;
  return `비중 유지 (×${fmtNum(x, 2)})`;
}

function breakParenLine(s) {
  return String(s || "").replace(/\s+\(/, "\n(");
}

function pbLabel(name, matched) {
  if (!matched) return "미선택";
  const n = String(name || "");
  if (n === "PB_VETO_SHIELD") return "PB 방어쉴드 (VETO)";
  if (n === "PB_CRISIS_SNIPER") return "PB 위기 스나이퍼";
  if (n === "PB_SQUEEZE_SNIPER") return "PB 스퀴즈 스나이퍼";
  if (n === "PB_TREND_SIGNAL") return "PB 추세 신호";
  if (n === "PB_WHALE_SIGNAL") return "PB 고래 신호";
  if (n === "PB_MEAN_REVERT_SIGNAL") return "PB 평균회귀 신호";
  if (n === "PB9_VACUUM_WHIPSAW") return "PB9 유동성 붕괴 긴급회피";
  if (n === "PB_FUNDING_EXTREME_HOLD") return "PB 펀딩비 극단 진입차단";
  if (n === "PB5_MAMMOTH_SNIPER") return "PB5 폭락장 V반등 저점포착";
  if (n === "PB8_HOLY_TRINITY_TRAP") return "PB8 가짜벽 역추적 선행매매";
  if (n === "PB2_SQUEEZE_IGNITION") return "PB2 응축에너지 돌파추격";
  if (n === "PB7_HOLY_TRINITY_TREND") return "PB7 저독성 정배열 추세추종";
  if (n === "PB13_BREAKOUT_TRAP") return "PB13 가짜돌파 역추세 저격";
  if (n === "PB10_CVD_DIVERGENCE") return "PB10 고래은닉 매집·분배 추적";
  if (n === "PB_LIQUIDATION_MAGNET") return "PB 청산자석 클러스터 추종";
  if (n === "PB_OI_DIVERGENCE") return "PB OI 발산 역추세 포착";
  if (n === "PB12_FUNDING_SNAPBACK") return "PB12 펀딩 과열 되돌림 포착";
  if (n === "PB11_TWAP_ABSORPTION") return "PB11 저변동 잠행 매집 추종";
  if (n === "PB15_VWAP_MEAN_REVERSION") return "PB15 VWAP 탄성 회귀";
  if (n === "CLASH_RESOLVED") return "충돌회피 관망결정";
  if (n === "SYNERGY_PERFECT_STORM") return "시너지 풀비중 점화";
  return n || "선택됨";
}

function pbEvalMap(list) {
  const m = {};
  (list || []).forEach((x) => {
    const k = String(x?.name || "");
    if (k) m[k] = x;
  });
  return m;
}

function pickGroupWinnerFromEvals(evals, names) {
  const matched = (evals || []).filter((e) => Boolean(e?.matched) && names.has(String(e?.name || "")));
  if (!matched.length) return { matched: false, name: "NONE", action: 0, kelly: 0.0, priority: 0, reason: "" };
  return matched.reduce((best, cur) => (Number(cur?.priority || 0) > Number(best?.priority || 0) ? cur : best));
}

function pbEvalRender(evalObj) {
  if (!evalObj) return { stage: "대기", cls: "muted", score: 0, impliedAction: 0, reco: "UNKNOWN", dirGap: 0, actionable: false, missing: true };
  const name = String(evalObj.name || "");
  const meta = evalObj.meta || {};
  const matched = Boolean(evalObj.matched);
  const PB_ACTION_SCORE_MIN = 60;
  const PB_DIR_GAP_MIN = 0.12;

  // ── 🧠 1. 스마트 헬퍼 함수 정의 ──
  // 목표값 대비 도달률 (0 ~ 1)
  const hit = (val, target) => clamp01(Math.abs(val) / target);
  const posHit = (val, target) => clamp01(Number(val || 0) / target);
  const negHit = (val, target) => clamp01((-Number(val || 0)) / target);

  // 역방향 도달률: 낮을수록 좋음. target 이하면 만점(1), worst 이상이면 0점
  const invHit = (val, target, worst) => {
    if (val <= target) return 1.0;
    if (val >= worst) return 0.0;
    return clamp01((worst - val) / (worst - target));
  };

  // 병목 앙상블: 60%는 가장 못 미친 조건(Min)에, 40%는 전체 평균(Avg)에 가중치를 두어 가짜 점수 방지
  const calcScore = (arr) => {
    if (!arr.length) return 0;
    const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
    const min = Math.min(...arr);
    return (0.6 * min + 0.4 * avg) * 100;
  };

  // ── 📊 2. 데이터 추출 ──
  const obi = Number(meta.obi || 0);
  const whale = Number(meta.nif_whale || 0);
  const tox = Number(meta.toxicity || 0);
  const collapse = Number(meta.collapse || 0);
  const absb = Number(meta.absorption || 0);
  const eai = Number(meta.eai || 0);
  const funding = Number(meta.funding || 0);
  const lai = Number(meta.lai || 0);
  const aft = Number(meta.aftershock || 0);
  const whaleSum = Number(meta.sum || 0);
  const pChange = Number(meta.price_change || 0);
  const oiDelta = Number(meta.oi_delta_pct ?? meta.oi_delta ?? 0);
  const eaiDelta = Number(meta.eai_delta || 0);
  const liqDir = Number(meta.direction || 0);
  const liqStrength = Number(meta.strength || 0);
  const liqDistance = Number(meta.distance || 1);
  const vol = Number(meta.vol || 0);
  const toxAvg = Number(meta.tox_avg || 0);
  const nifAvg = Number(meta.nif_avg || 0);
  const vwapGap = Number(meta.vwap_gap || 0);

  let score = 0;
  let impliedAction = Number(evalObj.action || 0);
  let dirGap = 0.0;
  const unifiedScore = Number(meta.unified_score);
  const hasUnifiedScore = Number.isFinite(unifiedScore);

  // ── 🎯 3. 플레이북별 최신 Threshold 기반 점수 산출 ──
  if (hasUnifiedScore) {
    score = clamp01(unifiedScore) * 100;
    dirGap = clamp01(Math.abs(unifiedScore - 0.5) * 2.0);
  }
  else if (name === "PB9_VACUUM_WHIPSAW") {
    // 붕괴(0.75)와 독성(0.85)
    score = calcScore([hit(collapse, 0.75), hit(tox, 0.85)]);
  }
  else if (name === "PB5_MAMMOTH_SNIPER") {
    // LAI(3억 * 0.75 = 2.25억), 고래(0.35), 흡수(0.62), 여진(<0.65, 0.9이상이면 0점)
    score = calcScore([
      hit(lai, 225000000),
      hit(whale, 0.35),
      hit(absb, 0.62),
      invHit(aft, 0.65, 0.90)
    ]);
  }
  else if (name === "PB8_HOLY_TRINITY_TRAP") {
    // 방향성 충돌(다이버전스) 계산 (OBI 0.35, Whale 0.25 기준)
    const sShort = calcScore([hit(obi, 0.35), hit(-whale, 0.25)]) / 100;
    const sLong = calcScore([hit(-obi, 0.35), hit(whale, 0.25)]) / 100;
    dirGap = Math.abs(sLong - sShort);

    if (sShort > sLong) impliedAction = 2;
    else if (sLong > sShort) impliedAction = 1;

    // 이긴 방향의 다이버전스 점수와 독성(0.75)의 앙상블
    score = calcScore([Math.max(sShort, sLong), hit(tox, 0.75)]);
  }
  else if (name === "PB2_SQUEEZE_IGNITION") {
    // EAI(2.0), Funding(0.001), 흡수(0.60), OI 델타(0.003)
    score = calcScore([
      hit(eai, 2.0),
      hit(Math.abs(funding), 0.001),
      hit(absb, 0.60),
      hit(oiDelta, 0.003),
    ]);
    dirGap = calcScore([hit(Math.abs(funding), 0.001), hit(oiDelta, 0.003)]) / 100;
    if (funding < 0) impliedAction = 1;
    else if (funding > 0) impliedAction = 2;
  }
  else if (name === "PB7_HOLY_TRINITY_TREND") {
    // OBI(0.25), Whale(0.25) 일치도, 독성(<0.38, 0.7이상이면 0점)
    const sLong = calcScore([hit(obi, 0.25), hit(whale, 0.25)]) / 100;
    const sShort = calcScore([hit(-obi, 0.25), hit(-whale, 0.25)]) / 100;
    dirGap = Math.abs(sLong - sShort);

    if (sLong > sShort) impliedAction = 1;
    else if (sShort > sLong) impliedAction = 2;

    score = calcScore([Math.max(sLong, sShort), invHit(tox, 0.38, 0.70)]);
  }
  else if (name === "PB13_BREAKOUT_TRAP") {
    score = calcScore([hit(tox, 0.70), hit(whale, 0.25)]);
    dirGap = hit(Math.abs(whale), 0.25);
  }
  else if (name === "PB_FUNDING_EXTREME_HOLD") {
    score = calcScore([hit(Math.abs(funding), 0.002)]);
    dirGap = 1.0;
    impliedAction = 0;
  }
  else if (name === "PB10_CVD_DIVERGENCE") {
    const z30 = Number(meta.nif_z_30m || 0);
    const sLong = calcScore([posHit(z30, 2.0), invHit(pChange, 0.002, 0.005)]) / 100;
    const sShort = calcScore([negHit(z30, 2.0), invHit(-pChange, 0.002, 0.005)]) / 100;
    dirGap = Math.abs(sLong - sShort);
    if (sLong > sShort) impliedAction = 1;
    else if (sShort > sLong) impliedAction = 2;
    score = Math.max(sLong, sShort) * 100;
  }
  else if (name === "PB_LIQUIDATION_MAGNET") {
    if (liqDir > 0) impliedAction = 1;
    else if (liqDir < 0) impliedAction = 2;
    score = calcScore([
      hit(liqStrength, 0.22),
      invHit(liqDistance, 0.005, 0.015),
      invHit(tox, 0.40, 0.80),
    ]);
    dirGap = hit(liqStrength, 0.22);
  }
  else if (name === "PB_OI_DIVERGENCE") {
    const oiDelta = Number(meta.oi_delta || 0);
    const sLong = calcScore([negHit(pChange, 0.003), negHit(oiDelta, 0.005), invHit(tox, 0.50, 0.80)]) / 100;
    const sShort = calcScore([posHit(pChange, 0.003), negHit(oiDelta, 0.005), invHit(tox, 0.50, 0.80)]) / 100;
    dirGap = Math.abs(sLong - sShort);
    if (sLong > sShort) impliedAction = 1;
    else if (sShort > sLong) impliedAction = 2;
    score = Math.max(sLong, sShort) * 100;
  }
  else if (name === "PB12_FUNDING_SNAPBACK") {
    score = calcScore([hit(Math.abs(funding), 0.001), hit(-eaiDelta, 0.05), hit(whale, 0.2)]);
    dirGap = calcScore([hit(Math.abs(funding), 0.001), hit(Math.abs(whale), 0.2)]) / 100;
  }
  else if (name === "PB11_TWAP_ABSORPTION") {
    const biasAbs = hit(Number(meta.bias || 0), 0.18);
    const absVal = Number(meta.absorption ?? meta.abs ?? 0);
    score = calcScore([invHit(vol, 0.005, 0.02), hit(absVal, 0.75), biasAbs]);
    dirGap = biasAbs;
  }
  else if (name === "PB15_VWAP_MEAN_REVERSION") {
    if (vwapGap < 0) impliedAction = 1;
    else if (vwapGap > 0) impliedAction = 2;
    score = calcScore([
      hit(vwapGap, 0.004),
      invHit(vol, 0.006, 0.02),
      hit(absb, 0.60),
      invHit(Math.abs(whale), 0.20, 0.60),
    ]);
    dirGap = hit(vwapGap, 0.004);
  }

  if (matched && !hasUnifiedScore) score = 100;
  score = Math.round(score);

  // ── 🎨 4. UI 렌더링 (단계 및 색상) ──
  let stage = "대기";
  if (matched) stage = "발동";
  else if (score >= 75) stage = "고확률";
  else if (score >= 50) stage = "주의";
  else if (score >= 30) stage = "탐색";

  let cls = "muted";
  if (score >= 75) cls = "good";
  else if (score >= 50) cls = "warn";
  else if (score >= 30) cls = "bad";

  let reco = "UNKNOWN";
  if (name === "PB9_VACUUM_WHIPSAW") {
    reco = String(meta.reco || "");
    if (!reco) {
      const pred = String(meta.pred_dir || "UNKNOWN");
      if (pred === "DOWN_TAIL_EXPECTED") reco = "[LIMIT_LONG]";
      else if (pred === "UP_TAIL_EXPECTED") reco = "[LIMIT_SHORT]";
      else reco = "UNKNOWN";
    }
  }

  const actionable =
    (impliedAction === 1 || impliedAction === 2 || (reco && reco !== "UNKNOWN")) &&
    score >= PB_ACTION_SCORE_MIN &&
    (matched || dirGap >= PB_DIR_GAP_MIN);

  return { stage, cls, score, impliedAction, reco, dirGap, actionable, missing: false };
}

function pushMicroHistoryValue(key, ts, text, value) {
  const arr = microHistory[key];
  if (!arr) return;
  const v = Number(value);
  const item = { ts: ts || "-", text: text || "-", value: Number.isFinite(v) ? v : 0 };
  const last = arr[arr.length - 1];
  if (last && last.ts === item.ts) {
    arr[arr.length - 1] = item;
  } else if (!last || last.text !== item.text || last.ts !== item.ts) {
    arr.push(item);
  }
  while (arr.length > MICRO_HISTORY_MAX) arr.shift();
}

function renderMicroHistory(elId, key) {
  const arr = microHistory[key] || [];
  const target = el(elId);
  if (!target) return;
  if (!arr.length) {
    target.textContent = "-";
    return;
  }
  const rows = arr.slice().reverse().map((x, i) => {
    const color = key === "whale_intent" ? getWhaleIntentColor(x.text) : getStatusColor(x.text);
    const dot = `<span class="hist-dot" style="background:${color}"></span>`;
    const cls = i === 0 ? "hist-row now" : "hist-row";
    return `<div class="${cls}"><span class="hist-ts">${x.ts}</span><span class="hist-val">${dot}${x.text}</span></div>`;
  });
  target.innerHTML = rows.join("");
}

function renderWhaleSparkline() {
  const svg = el("whaleSpark");
  if (!svg) return;
  const pts = (microHistory.whale || []).map((x) => Number(x.value || 0));
  if (!pts.length) {
    svg.innerHTML = "";
    return;
  }
  const NS = "http://www.w3.org/2000/svg";
  const w = 320, h = 84;
  const pL = 6, pR = 64, pY = 6;
  const min = Math.min(...pts, -0.2);
  const max = Math.max(...pts, 0.2);
  const span = Math.max(max - min, 1e-6);
  const xAt = (i) => pL + (i * (w - pL - pR)) / Math.max(pts.length - 1, 1);
  const yAt = (v) => pY + ((max - v) * (h - pY * 2)) / span;
  const path = pts.map((v, i) => `${i ? "L" : "M"}${xAt(i)},${yAt(v)}`).join(" ");
  const area = `${path} L${xAt(pts.length - 1)},${h - pY} L${xAt(0)},${h - pY} Z`;
  const lastV = pts[pts.length - 1] || 0;
  const lastX = xAt(pts.length - 1);
  const lastY = yAt(lastV);
  const labelX = Math.min(w - pY - 8, lastX + 8);
  const labelY = Math.max(pY + 10, lastY - 8);
  svg.innerHTML = `
    <defs>
      <linearGradient id="whaleSparkGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#66D8FF" stop-opacity="0.35"></stop>
        <stop offset="100%" stop-color="#66D8FF" stop-opacity="0.02"></stop>
      </linearGradient>
    </defs>
    <path d="${area}" fill="url(#whaleSparkGrad)"></path>
    <path d="${path}" fill="none" stroke="#66D8FF" stroke-width="2"></path>
    <circle cx="${lastX}" cy="${lastY}" r="2.8" fill="#66D8FF"></circle>
    <text x="${labelX}" y="${labelY}" fill="#66D8FF" font-size="10" text-anchor="start" font-family="JetBrains Mono, Roboto Mono, monospace">${fmtNum(lastV, 2)}</text>
  `;
}

function renderWhalePosSparkline() {
  const svg = el("whalePosSpark");
  if (!svg) return;
  const pts = (microHistory.whale_intent || []).map((x) => Number(x.value || 0));
  if (pts.length < 2) {
    svg.innerHTML = "";
    return;
  }
  const w = 320, h = 84;
  const pL = 6, pR = 72, pY = 6;
  const min = 0.0;
  const max = 1.0;
  const span = max - min;
  const xAt = (i) => pL + (i * (w - pL - pR)) / Math.max(pts.length - 1, 1);
  const yAt = (v) => pY + ((max - v) * (h - pY * 2)) / span;

  const longPts = pts.map((s) => clamp01((s + 1) / 2));
  const shortPts = longPts.map((l) => clamp01(1 - l));

  const pathOf = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${xAt(i)},${yAt(v)}`).join(" ");
  const longPath = pathOf(longPts);
  const shortPath = pathOf(shortPts);
  const lastLong = longPts[longPts.length - 1] || 0;
  const lastShort = shortPts[shortPts.length - 1] || 0;
  const lastX = xAt(longPts.length - 1);
  const lastYLong = yAt(lastLong);
  const lastYShort = yAt(lastShort);
  const labelX = Math.min(w - pY - 8, lastX + 8);

  const midY = yAt(0.5);
  svg.innerHTML = `
    <line x1="${pL}" y1="${midY}" x2="${w - pR}" y2="${midY}" stroke="rgba(255,255,255,0.20)" stroke-width="1"></line>
    <path d="${longPath}" fill="none" stroke="#2fd077" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path>
    <path d="${shortPath}" fill="none" stroke="#ff7575" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path>
    <circle cx="${lastX}" cy="${lastYLong}" r="2.6" fill="#2fd077"></circle>
    <circle cx="${lastX}" cy="${lastYShort}" r="2.6" fill="#ff7575"></circle>
    <text x="${labelX}" y="${Math.max(pY + 10, lastYLong - 6)}" fill="#2fd077" font-size="10" text-anchor="start" font-family="JetBrains Mono, Roboto Mono, monospace">L ${Math.round(lastLong * 100)}%</text>
    <text x="${labelX}" y="${Math.min(h - pY, lastYShort + 12)}" fill="#ff7575" font-size="10" text-anchor="start" font-family="JetBrains Mono, Roboto Mono, monospace">S ${Math.round(lastShort * 100)}%</text>
  `;
}

function renderLinearGauge(fillId, value01, color = "#66D8FF", labelId = "", label = "") {
  const fill = el(fillId);
  if (!fill) return;
  const v = clamp01(value01);
  fill.style.height = `${fmtNum(v * 100, 0)}%`;
  fill.style.background = color;
  if (labelId) {
    const lbl = el(labelId);
    if (lbl) lbl.textContent = label;
  }
}

function renderLRGauge(leftFillId, rightFillId, left01, right01, leftTextId = "", rightTextId = "", leftText = "", rightText = "") {
  const l = el(leftFillId);
  const r = el(rightFillId);
  if (l) l.style.width = `${fmtNum(clamp01(left01) * 100, 0)}%`;
  if (r) r.style.width = `${fmtNum(clamp01(right01) * 100, 0)}%`;
  if (leftTextId) {
    const lt = el(leftTextId);
    if (lt) lt.textContent = leftText;
  }
  if (rightTextId) {
    const rt = el(rightTextId);
    if (rt) rt.textContent = rightText;
  }
}

function setMeter(fillId, value01, tone = "good") {
  const fill = el(fillId);
  if (!fill) return;
  fill.style.width = `${Math.round(clamp01(value01) * 100)}%`;
  fill.className = tone;
}

function fmtDateTick(ts) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${mo}-${dd} ${hh}:${mm}`;
}

function obiDescLabel(v) {
  const x = Number(v || 0);
  if (x >= 0.35) return "매수벽 우세 구간";
  if (x <= -0.35) return "매도벽 우세 구간";
  return "균형권";
}

function eaiDescLabel(v) {
  const x = Number(v || 0);
  if (x >= 1.8) return "급변동 경계";
  if (x >= 1.0) return "변동성 확대 중";
  return "저변동 구간";
}

function niceStep(span, targetTicks = 4) {
  if (!Number.isFinite(span) || span <= 0) return 1;
  const rough = span / Math.max(targetTicks, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  const stepNorm = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return stepNorm * mag;
}

function axisTicks(min, max, targetTicks = 4) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0];
  if (min === max) return [min];
  const step = niceStep(max - min, targetTicks);
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step * 0.5; v += step) ticks.push(v);
  return ticks;
}

function renderLineSvg(svg, points) {
  svg.innerHTML = "";
  const vals = points.map((p) => Number(p.equity || 1));
  if (!vals.length) return;
  const w = 800, h = 280;
  const ml = 58, mr = 16, mt = 16, mb = 50;
  const cw = w - ml - mr, ch = h - mt - mb;
  const min = Math.min(...vals), max = Math.max(...vals);
  const pad = Math.max((max - min) * 0.08, 1e-5);
  const yMin = min - pad, yMax = max + pad;
  const ySpan = Math.max(yMax - yMin, 1e-8);
  const xAt = (i) => ml + (i * cw) / Math.max(vals.length - 1, 1);
  const yAt = (v) => mt + ((yMax - v) * ch) / ySpan;

  const NS = "http://www.w3.org/2000/svg";
  const defs = document.createElementNS(NS, "defs");
  const grad = document.createElementNS(NS, "linearGradient");
  grad.setAttribute("id", "equityFillGrad");
  grad.setAttribute("x1", "0");
  grad.setAttribute("y1", "0");
  grad.setAttribute("x2", "0");
  grad.setAttribute("y2", "1");
  const gs1 = document.createElementNS(NS, "stop");
  gs1.setAttribute("offset", "0%");
  gs1.setAttribute("stop-color", "#66d8ff");
  gs1.setAttribute("stop-opacity", "0.35");
  const gs2 = document.createElementNS(NS, "stop");
  gs2.setAttribute("offset", "100%");
  gs2.setAttribute("stop-color", "#66d8ff");
  gs2.setAttribute("stop-opacity", "0.03");
  grad.appendChild(gs1);
  grad.appendChild(gs2);
  defs.appendChild(grad);
  svg.appendChild(defs);

  axisTicks(yMin, yMax, 4).forEach((t) => {
    const y = yAt(t);
    const grid = document.createElementNS(NS, "line");
    grid.setAttribute("x1", ml);
    grid.setAttribute("x2", w - mr);
    grid.setAttribute("y1", y);
    grid.setAttribute("y2", y);
    grid.setAttribute("class", "chart-grid");
    svg.appendChild(grid);

    const lbl = document.createElementNS(NS, "text");
    lbl.setAttribute("x", ml - 8);
    lbl.setAttribute("y", y + 4);
    lbl.setAttribute("text-anchor", "end");
    lbl.setAttribute("class", "axis-tick");
    lbl.textContent = t.toFixed(3);
    svg.appendChild(lbl);
  });

  const idxTicks = [0, Math.floor((vals.length - 1) / 2), vals.length - 1].filter((v, i, a) => a.indexOf(v) === i);
  idxTicks.forEach((idx) => {
    const x = xAt(idx);
    const vline = document.createElementNS(NS, "line");
    vline.setAttribute("x1", x);
    vline.setAttribute("x2", x);
    vline.setAttribute("y1", mt);
    vline.setAttribute("y2", h - mb);
    vline.setAttribute("class", "chart-grid-v");
    svg.appendChild(vline);

    const lbl = document.createElementNS(NS, "text");
    lbl.setAttribute("x", x);
    lbl.setAttribute("y", h - 24);
    lbl.setAttribute("text-anchor", "middle");
    lbl.setAttribute("class", "axis-tick");
    lbl.textContent = fmtDateTick(points[idx]?.ts);
    svg.appendChild(lbl);
  });

  const linePts = vals.map((v, i) => `${xAt(i)},${yAt(v)}`).join(" ");
  const areaPts = `${ml},${h - mb} ${linePts} ${w - mr},${h - mb}`;

  const area = document.createElementNS(NS, "polygon");
  area.setAttribute("points", areaPts);
  area.setAttribute("fill", "url(#equityFillGrad)");
  svg.appendChild(area);

  const poly = document.createElementNS(NS, "polyline");
  poly.setAttribute("class", "line");
  poly.setAttribute("points", linePts);
  svg.appendChild(poly);

  const lastX = xAt(vals.length - 1);
  const lastY = yAt(vals[vals.length - 1]);
  const last = document.createElementNS(NS, "circle");
  last.setAttribute("cx", lastX);
  last.setAttribute("cy", lastY);
  last.setAttribute("r", "3.2");
  last.setAttribute("class", "line-dot");
  svg.appendChild(last);

  const yLabel = document.createElementNS(NS, "text");
  yLabel.setAttribute("x", 14);
  yLabel.setAttribute("y", mt + ch / 2);
  yLabel.setAttribute("transform", `rotate(-90 14 ${mt + ch / 2})`);
  yLabel.setAttribute("class", "axis-label");
  yLabel.textContent = "Equity (x)";
  svg.appendChild(yLabel);

  const xLabel = document.createElementNS(NS, "text");
  xLabel.setAttribute("x", ml + cw / 2);
  xLabel.setAttribute("y", h - 6);
  xLabel.setAttribute("text-anchor", "middle");
  xLabel.setAttribute("class", "axis-label");
  xLabel.textContent = "Date";
  svg.appendChild(xLabel);
}

function renderBarSvg(svg, points) {
  svg.innerHTML = "";
  const vals = points.map((p) => Number(p.pnl_pct || 0));
  if (!vals.length) return;
  const w = 800, h = 280;
  const ml = 58, mr = 16, mt = 16, mb = 50;
  const cw = w - ml - mr, ch = h - mt - mb;
  const min = Math.min(...vals), max = Math.max(...vals);
  const yMin = Math.min(min, 0), yMax = Math.max(max, 0);
  const ySpan = Math.max(yMax - yMin, 1e-8);
  const yAt = (v) => mt + ((yMax - v) * ch) / ySpan;
  const NS = "http://www.w3.org/2000/svg";

  axisTicks(yMin, yMax, 4).forEach((t) => {
    const y = yAt(t);
    const grid = document.createElementNS(NS, "line");
    grid.setAttribute("x1", ml);
    grid.setAttribute("x2", w - mr);
    grid.setAttribute("y1", y);
    grid.setAttribute("y2", y);
    grid.setAttribute("class", Math.abs(t) < 1e-10 ? "chart-zero" : "chart-grid");
    svg.appendChild(grid);

    const lbl = document.createElementNS(NS, "text");
    lbl.setAttribute("x", ml - 8);
    lbl.setAttribute("y", y + 4);
    lbl.setAttribute("text-anchor", "end");
    lbl.setAttribute("class", "axis-tick");
    lbl.textContent = `${t.toFixed(2)}%`;
    svg.appendChild(lbl);
  });

  const idxTicks = [0, Math.floor((vals.length - 1) / 2), vals.length - 1].filter((v, i, a) => a.indexOf(v) === i);
  idxTicks.forEach((idx) => {
    const x = ml + (idx * cw) / Math.max(vals.length - 1, 1);
    const lbl = document.createElementNS(NS, "text");
    lbl.setAttribute("x", x);
    lbl.setAttribute("y", h - 24);
    lbl.setAttribute("text-anchor", "middle");
    lbl.setAttribute("class", "axis-tick");
    lbl.textContent = fmtDateTick(points[idx]?.ts);
    svg.appendChild(lbl);
  });

  const bw = cw / Math.max(vals.length, 1);
  const zeroY = yAt(0);
  vals.forEach((v, i) => {
    const bar = document.createElementNS(NS, "rect");
    const hh = Math.abs(zeroY - yAt(v));
    bar.setAttribute("x", ml + i * bw + 1);
    bar.setAttribute("y", v >= 0 ? zeroY - hh : zeroY);
    bar.setAttribute("width", Math.max(bw - 2, 1));
    bar.setAttribute("height", Math.max(hh, 1));
    bar.setAttribute("rx", "1.5");
    bar.setAttribute("class", v >= 0 ? "bar-pos" : "bar-neg");
    svg.appendChild(bar);
  });

  const yLabel = document.createElementNS(NS, "text");
  yLabel.setAttribute("x", 14);
  yLabel.setAttribute("y", mt + ch / 2);
  yLabel.setAttribute("transform", `rotate(-90 14 ${mt + ch / 2})`);
  yLabel.setAttribute("class", "axis-label");
  yLabel.textContent = "PnL (%)";
  svg.appendChild(yLabel);

  const xLabel = document.createElementNS(NS, "text");
  xLabel.setAttribute("x", ml + cw / 2);
  xLabel.setAttribute("y", h - 6);
  xLabel.setAttribute("text-anchor", "middle");
  xLabel.setAttribute("class", "axis-label");
  xLabel.textContent = "Date";
  svg.appendChild(xLabel);
}

function render(state) {
  const sig = state.signal || {};
  const pos = state.position || {};
  const perf = state.performance || {};
  const ag = state.agents || {};
  const sess = state.session || {};
  const micro = state.microstructure || {};
  const tail = state.tail_risk || {};
  const pb = state.playbook || {};
  const globalStamp = fmtTs(state.updated_at || state.cycle_timestamp_kst);
  const microStampText = fmtTs(micro.updated_at || state.updated_at || state.cycle_timestamp_kst);
  let peakEq = 1;
  let maxDrawdown = 0;
  (state.trades_tail || []).forEach((t) => {
    const eq = Number(t?.equity || 1);
    if (!Number.isFinite(eq) || eq <= 0) return;
    if (eq > peakEq) peakEq = eq;
    const dd = ((peakEq - eq) / peakEq) * 100;
    if (dd > maxDrawdown) maxDrawdown = dd;
  });

  const final = actionLabel(Number(sig.final_action || 0));
  const rl = actionLabel(Number(sig.rl_action || 0));
  const posNow = String(pos.current || "NONE").toUpperCase();
  const reg = String(state.regime || "-").toUpperCase();
  el("dsacDecision").textContent = final.text;
  el("dsacDecision").className = final.text === "LONG" ? "good" : final.text === "SHORT" ? "bad" : "warn";
  el("dsacRl").textContent = rl.text;
  el("dsacRl").className = rl.text === "LONG" ? "good" : rl.text === "SHORT" ? "bad" : "warn";
  el("dsacPricePos").textContent = `${fmtNum(state.price, 2)} / ${posNow}`;
  el("dsacPricePos").className = posNow === "LONG" ? "good" : posNow === "SHORT" ? "bad" : "muted";
  el("dsacRegime").textContent = reg;
  el("dsacRegime").className = reg.includes("BULL") ? "good" : reg.includes("BEAR") ? "bad" : "warn";
  el("dsacKelly").textContent = `×${fmtNum(sig.unified_kelly || 0, 3)}`;
  el("dsacUnreal").textContent = fmtPct(pos.unrealized_pnl_pct || 0);
  el("dsacUnreal").className = riskClass(pos.unrealized_pnl_pct || 0);
  el("dsacPnlMdd").textContent = `${fmtPct(perf.pnl_24h || 0)} / -${fmtNum(maxDrawdown, 2)}%`;
  el("dsacPnlMdd").className = riskClass(perf.pnl_24h || 0);
  el("dsacSource").textContent = String(sig.source || "-");
  el("dsacSource").className = "muted";
  el("dsacStamp").textContent = globalStamp;

  const sessionHtml = buildSessionHtml(sess);
  const opsSession = el("opsSession");
  if (opsSession) opsSession.innerHTML = sessionHtml;
  const topSession = el("topSession");
  if (topSession) topSession.innerHTML = sessionHtml;
  const nowEl = el("opsNow");
  if (nowEl) nowEl.textContent = fmtNowClock();
  const topNowEl = el("topNow");
  if (topNowEl) topNowEl.textContent = fmtNowClock();
  const agents = state.agents || {};
  const agLong = agents.long || {};
  const agShort = agents.short || {};
  const agTracker = agents.tracker || {};
  const agLongTrack = agTracker.long || {};
  const agShortTrack = agTracker.short || {};
  const agDecisionLong = actionLabel(Number(agLong.action ?? 0));
  const agDecisionShort = actionLabel(Number(agShort.action ?? 0));
  const agPosLong = String(agLongTrack.pos || "NONE").toUpperCase();
  const agPosShort = String(agShortTrack.pos || "NONE").toUpperCase();

  el("ensBalDecision").textContent = agDecisionLong.text;
  el("ensBalDecision").className = agDecisionLong.text === "LONG" ? "good" : agDecisionLong.text === "SHORT" ? "bad" : "muted";
  el("ensBalPos").textContent = agPosLong;
  el("ensBalPos").className = agPosLong === "LONG" ? "good" : agPosLong === "SHORT" ? "bad" : "muted";
  el("ensBalKelly").textContent = fmtNum(agLong.kelly_weight ?? 0, 3);
  el("ensBalVotes").textContent = fmtNum(agLongTrack.entry_kelly ?? 0, 3);
  if (hasOwn(agLongTrack, "unrealized_pnl_pct")) {
    el("ensBalWinRate").textContent = fmtPct(agLongTrack.unrealized_pnl_pct ?? 0);
    el("ensBalWinRate").className = riskClass(agLongTrack.unrealized_pnl_pct || 0);
  } else {
    el("ensBalWinRate").textContent = "-";
    el("ensBalWinRate").className = "muted";
  }
  el("ensBalLastPnl").textContent = fmtPct(agLongTrack.last_pnl_pct ?? 0);
  el("ensBalLastPnl").className = riskClass(agLongTrack.last_pnl_pct || 0);
  el("ensBalMdd").textContent = `-${fmtNum(agLongTrack.mdd_pct ?? 0, 2)}%`;
  el("ensBalMdd").className = Number(agLongTrack.mdd_pct ?? 0) > 0 ? "bad" : "muted";
  el("ensBalDecisionAt").textContent = fmtTs(agLong.decision_at || agLongTrack.updated_at || state.updated_at || state.cycle_timestamp_kst);
  el("ensBalDecisionAt").className = "muted";
  el("ensBalTotal").textContent = `누적: ${fmtPct(agLongTrack.total_return_pct ?? 0)} | 승률: ${fmtNum(agLongTrack.win_rate ?? 0, 1)}% | 거래: ${fmtNum(agLongTrack.trades ?? 0, 0)}회`;
  el("ensBalTotal").className = riskClass(agLongTrack.total_return_pct || 0);
  el("ensBalParamMeta").textContent = "";
  el("ensBalStamp").textContent = fmtTs(agLongTrack.updated_at || state.updated_at || state.cycle_timestamp_kst);

  el("ensLowDecision").textContent = agDecisionShort.text;
  el("ensLowDecision").className = agDecisionShort.text === "LONG" ? "good" : agDecisionShort.text === "SHORT" ? "bad" : "muted";
  el("ensLowPos").textContent = agPosShort;
  el("ensLowPos").className = agPosShort === "LONG" ? "good" : agPosShort === "SHORT" ? "bad" : "muted";
  el("ensLowKelly").textContent = fmtNum(agShort.kelly_weight ?? 0, 3);
  el("ensLowVotes").textContent = fmtNum(agShortTrack.entry_kelly ?? 0, 3);
  if (hasOwn(agShortTrack, "unrealized_pnl_pct")) {
    el("ensLowWinRate").textContent = fmtPct(agShortTrack.unrealized_pnl_pct ?? 0);
    el("ensLowWinRate").className = riskClass(agShortTrack.unrealized_pnl_pct || 0);
  } else {
    el("ensLowWinRate").textContent = "-";
    el("ensLowWinRate").className = "muted";
  }
  el("ensLowLastPnl").textContent = fmtPct(agShortTrack.last_pnl_pct ?? 0);
  el("ensLowLastPnl").className = riskClass(agShortTrack.last_pnl_pct || 0);
  el("ensLowMdd").textContent = `-${fmtNum(agShortTrack.mdd_pct ?? 0, 2)}%`;
  el("ensLowMdd").className = Number(agShortTrack.mdd_pct ?? 0) > 0 ? "bad" : "muted";
  el("ensLowDecisionAt").textContent = fmtTs(agShort.decision_at || agShortTrack.updated_at || state.updated_at || state.cycle_timestamp_kst);
  el("ensLowDecisionAt").className = "muted";
  el("ensLowTotal").textContent = `누적: ${fmtPct(agShortTrack.total_return_pct ?? 0)} | 승률: ${fmtNum(agShortTrack.win_rate ?? 0, 1)}% | 거래: ${fmtNum(agShortTrack.trades ?? 0, 0)}회`;
  el("ensLowTotal").className = riskClass(agShortTrack.total_return_pct || 0);
  el("ensLowParamMeta").textContent = "";
  el("ensLowStamp").textContent = fmtTs(agShortTrack.updated_at || state.updated_at || state.cycle_timestamp_kst);

  const ens = state.ensembles || {};
  const ensBal = ens.balanced || {};
  const ensLow = ens.lowfreq || {};
  const ensTrk = ens.tracker || {};
  const ensBalTrk = ensTrk.balanced || {};
  const ensLowTrk = ensTrk.lowfreq || {};
  const profitDecision = actionLabel(Number((ensLow.live || {}).action ?? 0));
  const stableDecision = actionLabel(Number((ensBal.live || {}).action ?? 0));
  const profitPos = String(ensLowTrk.pos || "NONE").toUpperCase();
  const stablePos = String(ensBalTrk.pos || "NONE").toUpperCase();

  el("ultProfitDecision").textContent = profitDecision.text;
  el("ultProfitDecision").className = profitDecision.text === "LONG" ? "good" : profitDecision.text === "SHORT" ? "bad" : "muted";
  el("ultProfitPos").textContent = profitPos;
  el("ultProfitPos").className = profitPos === "LONG" ? "good" : profitPos === "SHORT" ? "bad" : "muted";
  el("ultProfitKelly").textContent = fmtNum((ensLow.live || {}).kelly_weight ?? 0, 3);
  el("ultProfitEntryKelly").textContent = fmtNum(ensLowTrk.entry_kelly ?? 0, 3);
  if (hasOwn(ensLowTrk, "unrealized_pnl_pct")) {
    el("ultProfitUnreal").textContent = fmtPct(ensLowTrk.unrealized_pnl_pct ?? 0);
    el("ultProfitUnreal").className = riskClass(ensLowTrk.unrealized_pnl_pct || 0);
  } else {
    el("ultProfitUnreal").textContent = "-";
    el("ultProfitUnreal").className = "muted";
  }
  el("ultProfitLastPnl").textContent = fmtPct(ensLowTrk.last_pnl_pct ?? 0);
  el("ultProfitLastPnl").className = riskClass(ensLowTrk.last_pnl_pct || 0);
  el("ultProfitMdd").textContent = `-${fmtNum(ensLowTrk.mdd_pct ?? 0, 2)}%`;
  el("ultProfitMdd").className = Number(ensLowTrk.mdd_pct ?? 0) > 0 ? "bad" : "muted";
  el("ultProfitDecisionAt").textContent = fmtTs((ensLow.live || {}).updated_at || ensLowTrk.updated_at || ens.updated_at || state.updated_at || state.cycle_timestamp_kst);
  el("ultProfitDecisionAt").className = "muted";
  el("ultProfitTotal").textContent = `누적: ${fmtPct(ensLowTrk.total_return_pct ?? 0)} | 승률: ${fmtNum(ensLowTrk.win_rate ?? 0, 1)}% | 거래: ${fmtNum(ensLowTrk.trades ?? 0, 0)}회`;
  el("ultProfitTotal").className = riskClass(ensLowTrk.total_return_pct || 0);
  el("ultProfitMeta").textContent = "";
  el("ultProfitStamp").textContent = fmtTs(ensLowTrk.updated_at || ens.updated_at || state.updated_at || state.cycle_timestamp_kst);

  el("ultStableDecision").textContent = stableDecision.text;
  el("ultStableDecision").className = stableDecision.text === "LONG" ? "good" : stableDecision.text === "SHORT" ? "bad" : "muted";
  el("ultStablePos").textContent = stablePos;
  el("ultStablePos").className = stablePos === "LONG" ? "good" : stablePos === "SHORT" ? "bad" : "muted";
  el("ultStableKelly").textContent = fmtNum((ensBal.live || {}).kelly_weight ?? 0, 3);
  el("ultStableEntryKelly").textContent = fmtNum(ensBalTrk.entry_kelly ?? 0, 3);
  if (hasOwn(ensBalTrk, "unrealized_pnl_pct")) {
    el("ultStableUnreal").textContent = fmtPct(ensBalTrk.unrealized_pnl_pct ?? 0);
    el("ultStableUnreal").className = riskClass(ensBalTrk.unrealized_pnl_pct || 0);
  } else {
    el("ultStableUnreal").textContent = "-";
    el("ultStableUnreal").className = "muted";
  }
  el("ultStableLastPnl").textContent = fmtPct(ensBalTrk.last_pnl_pct ?? 0);
  el("ultStableLastPnl").className = riskClass(ensBalTrk.last_pnl_pct || 0);
  el("ultStableMdd").textContent = `-${fmtNum(ensBalTrk.mdd_pct ?? 0, 2)}%`;
  el("ultStableMdd").className = Number(ensBalTrk.mdd_pct ?? 0) > 0 ? "bad" : "muted";
  el("ultStableDecisionAt").textContent = fmtTs((ensBal.live || {}).updated_at || ensBalTrk.updated_at || ens.updated_at || state.updated_at || state.cycle_timestamp_kst);
  el("ultStableDecisionAt").className = "muted";
  el("ultStableTotal").textContent = `누적: ${fmtPct(ensBalTrk.total_return_pct ?? 0)} | 승률: ${fmtNum(ensBalTrk.win_rate ?? 0, 1)}% | 거래: ${fmtNum(ensBalTrk.trades ?? 0, 0)}회`;
  el("ultStableTotal").className = riskClass(ensBalTrk.total_return_pct || 0);
  el("ultStableMeta").textContent = "";
  el("ultStableStamp").textContent = fmtTs(ensBalTrk.updated_at || ens.updated_at || state.updated_at || state.cycle_timestamp_kst);

  const obiNow = obiLabel(micro.obi);
  const whaleNow = whaleLabel(micro);
  const whaleIntentNow = whaleIntentLabel(micro);
  const whaleIntentHistNow = whaleIntentHistoryLabel(micro);
  const whaleIntentWin = Number(micro.whale_position_window_min || 5);
  const eaiNow = eaiLabel(micro.eai);
  el("whaleText").textContent = whaleNow;
  el("whaleStatusText").textContent = whaleDescLabel(micro.nif_whale);
  el("whaleIntentTitle").textContent = `고래포지션(${fmtNum(whaleIntentWin, 0)}m)`;
  el("whaleIntentText").textContent = whaleIntentNow;
  const whaleGuide = whaleIntentGuideLabel(micro);
  const whaleActionText = whaleGuide.includes("·") ? whaleGuide.split("·").pop().trim() : whaleGuide;
  const whaleIntentPctEl = el("whaleIntentPct");
  if (whaleIntentPctEl) whaleIntentPctEl.textContent = whaleActionText;
  el("obiStamp").textContent = microStampText;
  el("whaleStamp").textContent = microStampText;
  el("whaleIntentStamp").textContent = microStampText;
  el("eaiStamp").textContent = microStampText;
  pushMicroHistoryValue("obi", microStampText, obiNow, micro.obi);
  pushMicroHistoryValue("whale", microStampText, whaleNow, micro.nif_whale);
  pushMicroHistoryValue("whale_intent", microStampText, whaleIntentHistNow, micro.whale_position_score);
  pushMicroHistoryValue("eai", microStampText, eaiNow, micro.eai);
  renderMicroHistory("whaleHist", "whale");
  renderMicroHistory("whaleIntentHist", "whale_intent");
  renderWhaleSparkline();
  renderWhalePosSparkline();
  const obi = Number(micro.obi || 0);
  const bidPct = clamp01(0.5 + obi / 2);
  const askPct = clamp01(1.0 - bidPct);
  renderLRGauge(
    "obiGaugeLeftFill", "obiGaugeRightFill",
    askPct, bidPct,
    "obiGaugeLeftTxt", "obiGaugeRightTxt",
    `매도 ${fmtNum(askPct * 100, 0)}%`,
    `매수 ${fmtNum(bidPct * 100, 0)}%`
  );
  el("obiText").textContent = obiNow;
  const eaiV = Number(micro.eai || 0);
  const volHot = clamp01(eaiV / 2.5);
  const volCalm = clamp01(1 - volHot);
  renderLRGauge(
    "eaiGaugeLeftFill", "eaiGaugeRightFill",
    volCalm, volHot,
    "eaiGaugeLeftTxt", "eaiGaugeRightTxt",
    `평온 ${fmtNum(volCalm * 100, 0)}%`,
    `과열 ${fmtNum(volHot * 100, 0)}%`
  );
  el("eaiText").textContent = eaiNow;
  const hftNames = new Set(["PB_VETO_SHIELD", "PB_CRISIS_SNIPER", "PB_SQUEEZE_SNIPER"]);
  const mftNames = new Set(["PB_TREND_SIGNAL", "PB_WHALE_SIGNAL", "PB_MEAN_REVERT_SIGNAL"]);
  const evalList = pb.evaluations || [];
  const pbHft = pickGroupWinnerFromEvals(evalList, hftNames);
  const pbMft = pickGroupWinnerFromEvals(evalList, mftNames);
  const pbHftMatched = Boolean(pbHft.matched);
  const pbHftAction = actionLabel(Number(pbHft.action || 0));
  el("pbName").textContent = pbLabel(pbHft.name, pbHftMatched);
  el("pbAction").textContent = pbHftMatched ? pbHftAction.text : "BASE";
  el("pbAction").className = pbHftMatched ? (pbHftAction.cls === "long" ? "good" : pbHftAction.cls === "short" ? "bad" : "warn") : "muted";
  el("pbKelly").textContent = pbHftMatched ? fmtNum(Number(pbHft.kelly || 0), 3) : "0.000";
  el("pbPriority").textContent = pbHftMatched ? String(pbHft.priority ?? "-") : "-";
  el("pbStamp").textContent = fmtTs(pbHft.updated_at || pb.updated_at || state.updated_at);
  const pbe = pbEvalMap(pb.evaluations || []);
  latestState = state;
  latestEvalMap = pbe;
  const pbVeto = pbEvalRender(pbe.PB_VETO_SHIELD);
  const pbCrisis = pbEvalRender(pbe.PB_CRISIS_SNIPER);
  const pbSqueeze = pbEvalRender(pbe.PB_SQUEEZE_SNIPER);
  const mftTrend = pbEvalRender(pbe.PB_TREND_SIGNAL);
  const mftWhale = pbEvalRender(pbe.PB_WHALE_SIGNAL);
  const mftRevert = pbEvalRender(pbe.PB_MEAN_REVERT_SIGNAL);
  const pbRowDetail = (evalRes) => {
    if (evalRes.missing) return "데이터 대기";
    let displayAction = evalRes.actionable ? actionLabel(evalRes.impliedAction).text : "HOLD";
    if (evalRes.reco && evalRes.reco !== "UNKNOWN") {
      displayAction = evalRes.actionable ? evalRes.reco : "HOLD";
    }
    return `환경 ${fmtNum(evalRes.score, 0)}점 / 실행 ${displayAction} / 방향확신 ${fmtNum((evalRes.dirGap || 0) * 100, 0)}%`;
  };
  el("pbVetoState").textContent = pbVeto.stage; el("pbVetoState").className = pbVeto.cls; el("pbVetoDetail").textContent = pbRowDetail(pbVeto); el("pbVetoGauge").style.width = `${fmtNum(pbVeto.score, 0)}%`; el("pbVetoGauge").className = pbVeto.cls;
  el("pbCrisisState").textContent = pbCrisis.stage; el("pbCrisisState").className = pbCrisis.cls; el("pbCrisisDetail").textContent = pbRowDetail(pbCrisis); el("pbCrisisGauge").style.width = `${fmtNum(pbCrisis.score, 0)}%`; el("pbCrisisGauge").className = pbCrisis.cls;
  el("pbSqueezeState").textContent = pbSqueeze.stage; el("pbSqueezeState").className = pbSqueeze.cls; el("pbSqueezeDetail").textContent = pbRowDetail(pbSqueeze); el("pbSqueezeGauge").style.width = `${fmtNum(pbSqueeze.score, 0)}%`; el("pbSqueezeGauge").className = pbSqueeze.cls;
  el("mftTrendState").textContent = mftTrend.stage; el("mftTrendState").className = mftTrend.cls; el("mftTrendDetail").textContent = pbRowDetail(mftTrend); el("mftTrendGauge").style.width = `${fmtNum(mftTrend.score, 0)}%`; el("mftTrendGauge").className = mftTrend.cls;
  el("mftWhaleState").textContent = mftWhale.stage; el("mftWhaleState").className = mftWhale.cls; el("mftWhaleDetail").textContent = pbRowDetail(mftWhale); el("mftWhaleGauge").style.width = `${fmtNum(mftWhale.score, 0)}%`; el("mftWhaleGauge").className = mftWhale.cls;
  el("mftRevertState").textContent = mftRevert.stage; el("mftRevertState").className = mftRevert.cls; el("mftRevertDetail").textContent = pbRowDetail(mftRevert); el("mftRevertGauge").style.width = `${fmtNum(mftRevert.score, 0)}%`; el("mftRevertGauge").className = mftRevert.cls;

  const pbMftMatched = Boolean(pbMft.matched);
  const pbMftAction = actionLabel(Number(pbMft.action || 0));
  el("mftName").textContent = pbLabel(pbMft.name, pbMftMatched);
  el("mftAction").textContent = pbMftMatched ? pbMftAction.text : "HOLD";
  el("mftAction").className = pbMftMatched ? (pbMftAction.text === "LONG" ? "good" : pbMftAction.text === "SHORT" ? "bad" : "warn") : "muted";
  el("mftKelly").textContent = pbMftMatched ? fmtNum(Number(pbMft.kelly || 0), 3) : "0.000";
  el("mftPriority").textContent = pbMftMatched ? String(pbMft.priority ?? "-") : "-";
  el("mftStamp").textContent = fmtTs(pbMft.updated_at || pb.updated_at || state.updated_at);

  const tox = clamp01(micro.toxicity_score);
  const toxDir = Number(micro.obi || 0) > 0 ? 1 : Number(micro.obi || 0) < 0 ? -1 : 0;
  setMeter("toxFill", tox, tox > 0.65 ? "bad" : tox > 0.35 ? "warn" : "good");
  el("toxText").textContent = `${toxInterpret(tox, toxDir)} (score ${fmtNum(tox, 2)})`;

  const aft = clamp01(tail.aftershock_prob);
  const aftDir = Number(tail.z_bias || 0);
  setMeter("riskFill", aft, aft > 0.65 ? "bad" : aft > 0.35 ? "warn" : "good");
  el("riskText").textContent = `${aftershockInterpret(aft, aftDir)} (p ${fmtNum(aft, 2)})`;

  renderLineSvg(el("equitySvg"), state.trades_tail || []);
  renderBarSvg(el("pnlSvg"), state.trades_tail || []);
}

function tickOpsClock() {
  const nowEl = el("opsNow");
  if (nowEl) nowEl.textContent = fmtNowClock();
  const topNowEl = el("topNow");
  if (topNowEl) topNowEl.textContent = fmtNowClock();
}

async function tick() {
  try {
    const res = await fetch(`${STATE_URL}?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const state = await res.json();
    render(state);
  } catch (_e) {}
}

tick();
setInterval(tick, POLL_MS);
setInterval(tickOpsClock, 1000);
initPlaybookModal();
