
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[ch]);
}


function measureTextW(txt, font) {   // SVG 글자 폭(배지 자리 계산) -- 붙이기 전 텍스트는 getComputedTextLength 가 0 이라 캔버스로 잰다
  if (!measureCtx2d) measureCtx2d = document.createElement("canvas").getContext("2d");
  measureCtx2d.font = font;
  return measureCtx2d.measureText(txt).width;
}


const FPB_W = 268;

function fpTipOn(node, html) {   // 캔들 툴팁 대신 이 설명 -- 버블링을 끊어 봉 툴팁이 덮어쓰지 않게(subSvg 와 같은 수)
  node.setAttribute("pointer-events", "all"); node.style.cursor = "help"; node.classList.add("fp-tip");
  node.addEventListener("mousemove", (e) => { e.stopPropagation(); showTooltip(e.pageX, e.pageY, html); });
  node.addEventListener("mouseleave", hideTooltip);
}


function dirReadNow() {
  const S = latestSituation, r = activeSnapshotAsset === "eth" && S ? S.read : null;
  if (!r || !Array.isArray(r.lines) || !r.lines.length || Date.now() / 1000 - (Number(S.computed_at) || 0) > 180) return null;
  const hot = r.lines.filter((l) => l.grade === "근거" || l.grade === "약함");
  return { hot, desc: r.lines.filter((l) => !hot.includes(l)) };
}


const DRM_TIP = "실시간 데이터 쌍(체결·크기별 흐름·OI·청산·호가·베이시스 …)을 서버가 매초 읽은 것\n▲▼ = 과거 검정에서 방향 근거가 있던 줄만 · 네모 = 움직임 크기 줄(방향 아님)\n효과는 수 bp 수준 -- 메이커로만 비용을 넘는다 · 나머지 줄은 «설명»";

function renderDirReadMob(R) {
  const box = el("dirReadM");
  if (!box) return;
  if (!R || !isMobileChartMode() || fpAltOn()) { if (!box.hidden) { box.hidden = true; box.innerHTML = ""; box._h = ""; } return; }
  const short = (l) => (String(l.text).split(" — ")[1] || String(l.text)).split(":")[0], lines = (t) => String(t || "").split(" — ").join("\n— ");
  const tri = (d) => `<svg viewBox="0 0 10 10" aria-hidden="true"><path d="${d > 0 ? "M1 8.5h8L5 1.5z" : "M1 1.5h8L5 8.5z"}"/></svg>`;
  const btn = (k, cls, title, body, inner) => `<button type="button" class="drm-q ${cls}" ${helpAttrs(k)} title="${escapeHtml(title)}" data-body="${escapeHtml(body)}">${inner}</button>`;
  const nu = R.hot.filter((l) => l.dir > 0).length, nd = R.hot.filter((l) => l.dir < 0).length, on = R.desc.filter((l) => l.g && l.g.on).map((l) => l.topic);
  const h = `<div class="drm-h">${btn("drm-h", "drm-t", "지금 읽기", DRM_TIP, `지금 읽기<span aria-hidden="true">?</span>`)}`
    + `<span class="drm-n${nu ? " up" : ""}" aria-label="위쪽 ${nu}">${tri(1)}${nu}</span><span class="drm-n${nd ? " dn" : ""}" aria-label="아래쪽 ${nd}">${tri(-1)}${nd}</span></div>`
    + R.hot.map((l, i) => btn(`drm${i}`, `drm-r ${l.dir > 0 ? "up" : l.dir < 0 ? "dn" : l.g && l.g.v >= 80 ? "big" : ""}`,
        `${l.topic} · ${l.grade}${l.dir ? (l.dir > 0 ? " · 위쪽" : " · 아래쪽") : " · 크기(방향 아님)"}`, `${lines(l.text)}\n\n${lines(l.note)}`,
        `${l.dir ? tri(l.dir) : '<i aria-hidden="true"></i>'}<b>${escapeHtml(l.topic)}</b><span>${escapeHtml(short(l))}</span><em>${escapeHtml(l.grade)}</em>`)).join("")
    + (R.desc.length ? btn("drm-d", "drm-d", `설명 줄 ${R.desc.length}개 — 지금 일어난 일일 뿐, 다음 방향 근거 없음`, R.desc.map((l) => `${l.topic} · ${short(l)}`).join("\n"),
        `설명 ${R.desc.length}${on.length ? `<span>${escapeHtml(on.join(" · "))}</span>` : ""}`) : "");
  const key = h.replace(/aria-expanded="\w+"/g, "");   // 시트를 여닫는 aria-expanded 만 바뀐 건 다시 쓰지 않는다 -- 닫을 때 포커스가 돌아갈 단추가 사라졌다(검증)
  if (box._h !== key) { box._h = key; box.innerHTML = h; }
  if (box.hidden) box.hidden = false;
}


const dirReadH = (R) => (R ? 30 + R.hot.length * 23 + (R.desc.length ? 22 : 0) + 4 : 0);

function dirTri(g, cx, cy, d, c, r = 5) {   // 위 ▲ / 아래 ▼ 삼각형(그린 아이콘 -- 글자 화살표 대신)
  const e = document.createElementNS("http://www.w3.org/2000/svg", "path");
  e.setAttribute("d", d > 0 ? `M${cx - r} ${cy + r * 0.8}L${cx + r} ${cy + r * 0.8}L${cx} ${cy - r * 0.8}Z` : `M${cx - r} ${cy - r * 0.8}L${cx + r} ${cy - r * 0.8}L${cx} ${cy + r * 0.8}Z`);
  e.setAttribute("fill", c); g.appendChild(e); return e;
}


function renderDirBoard() {   // 왼쪽 위 «지표 방향» 판 -- 줄 = 지금 보이는 표식마다 하나(방향·이름·언제) · 바닥 = 큰 스윙 «넘으면 ▲ / 깨면 ▼» 가격
  const svg = el("candleSvgSnapshot"), G = chartGeo, B = fpBoard;
  svg?.querySelector("#dirBoard")?.remove();
  const R = dirReadNow();
  renderDirReadMob(R);
  if (!svg || !B || fpAltOn() || !G || !(B.rows.length || B.cond.length || R)) return;
  const NSV = "http://www.w3.org/2000/svg", F = "var(--font-sans)", mk = (p, tag, a, txt) => { const e = document.createElementNS(NSV, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; p.appendChild(e); return e; };
  const rows = B.rows, up = rows.filter((r) => r.d > 0).length, dn = rows.filter((r) => r.d < 0).length, ups = B.cond.filter((c) => c.d > 0), dns = B.cond.filter((c) => c.d < 0);
  const fold = dirFold, L = G.ml + 6, T = G.mt + B.top, w = FPB_W, rh = 23, head = rows.length ? 44 : 30, foot = fold ? 0 : (ups.length ? 20 : 0) + (dns.length ? 20 : 0), hb = head + rows.length * rh + (foot ? foot + 10 : 4), h = fold ? head : hb + dirReadH(R);
  const g = mk(svg, "g", { id: "dirBoard" });
  const bg = mk(g, "rect", { x: L, y: T, width: w, height: h, rx: 10, fill: "var(--panel-strong)", "fill-opacity": 0.95, stroke: "var(--line)" });
  bg.addEventListener("mousemove", (e) => { e.stopPropagation(); hideTooltip(); });   // 판 위에선 봉 툴팁 안 띄움
  bg.setAttribute("pointer-events", "all"); bg.classList.add("fp-tip");
  mk(g, "text", { x: L + 12, y: T + 19, fill: "var(--muted)", "font-size": 11, "font-weight": 700, "font-family": F }, "지표 방향");
  {   // 접기 단추(오른쪽 위) -- 접으면 머리줄만
    const fb = mk(g, "g", { role: "button", tabindex: 0, "aria-label": fold ? "지표 방향 펼치기" : "지표 방향 접기", "aria-expanded": String(!fold), class: "dirb-fold" });
    // 검증 11-20: 판은 렌더마다 새로 지어(차트 전체가 먼저 지워진다) 포커스가 body 로 떨어졌다 -- 포커스를 기억해 새 단추에 되돌린다(진짜로 떠난 blur 만 잊는다)
    fb.addEventListener("focus", () => { dirFoldFocus = true; });
    fb.addEventListener("blur", () => setTimeout(() => { if (fb.isConnected) dirFoldFocus = false; }));
    if (dirFoldFocus && document.activeElement === document.body) fb.focus({ preventScroll: true });
    mk(fb, "rect", { x: L + w - 30, y: T + 5, width: 24, height: 20, rx: 6, fill: "var(--text)", "fill-opacity": 0.06 });
    const cx = L + w - 18, cy = T + 15;
    mk(fb, "path", { d: fold ? `M${cx - 4} ${cy - 2}l4 4l4 -4` : `M${cx - 4} ${cy + 2}l4 -4l4 4`, fill: "none", stroke: "var(--muted)", "stroke-width": 1.8, "stroke-linecap": "round", "stroke-linejoin": "round" });
    const flip = (e) => { e.stopPropagation(); e.preventDefault(); dirFold = !dirFold; try { localStorage.setItem("dirBoardFold", dirFold ? "1" : "0"); } catch (err) { /* 기억은 편의 */ } renderSnapshotChart(); el("candleSvgSnapshot")?.querySelector("#dirBoard .dirb-fold")?.focus(); };   // 전체 다시 그림 -- 꼬리표가 접힌 판 자리(fpBoardBox)를 바로 따른다
    fb.setAttribute("pointer-events", "all"); fb.style.cursor = "pointer";
    fb.addEventListener("click", flip); fb.addEventListener("mousedown", (e) => e.stopPropagation());
    fb.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") flip(e); });
    fpTipOn(fb, fold ? "펼치기 -- 지표마다 한 줄" : "접기 -- 개수 줄만 남김");
  }
  if (rows.length) {
    const bx = L + 86, bw = w - 98 - 30, n = up + dn || 1, uw = Math.round(bw * up / n);
    if (up) mk(g, "rect", { x: bx, y: T + 11, width: Math.max(2, uw - (dn ? 1 : 0)), height: 9, rx: 3, fill: "var(--good)" });
    if (dn) mk(g, "rect", { x: bx + uw + (up ? 1 : 0), y: T + 11, width: Math.max(2, bw - uw - (up ? 1 : 0)), height: 9, rx: 3, fill: "var(--bad)" });
    dirTri(g, bx + 5, T + 32, 1, "var(--good)", 4); mk(g, "text", { x: bx + 13, y: T + 36, fill: "var(--good)", "font-size": 11, "font-weight": 750, "font-family": F }, `위 ${up}`);
    dirTri(g, bx + bw - 5, T + 32, -1, "var(--bad)", 4); mk(g, "text", { x: bx + bw - 13, y: T + 36, fill: "var(--bad)", "font-size": 11, "font-weight": 750, "font-family": F, "text-anchor": "end" }, `아래 ${dn}`);
    const tally = mk(g, "rect", { x: bx, y: T + 6, width: bw, height: 34, fill: "transparent" });
    fpTipOn(tally, "<b>지표 방향 개수</b><br>지금 화면에 보이는 표식마다 한 표(오래된 것·효과 다른 것도 같은 한 표)<br>모두 +1~4%p 수준의 약한 참고 -- 개수가 많다고 더 맞는 것 아님");
  }
  if (fold) return;
  rows.forEach((r, i) => {
    const y = T + head + i * rh, c = r.d > 0 ? "var(--good)" : "var(--bad)", rg = mk(g, "g", {});
    mk(rg, "rect", { x: L + 6, y, width: w - 12, height: rh - 2, rx: 5, fill: "var(--text)", "fill-opacity": i % 2 ? 0 : 0.035 });
    dirTri(rg, L + 19, y + (rh - 2) / 2, r.d, c, 5);
    mk(rg, "text", { x: L + 32, y: y + 15, fill: "var(--text)", "font-size": 12, "font-weight": 700, "font-family": F }, r.name);
    mk(rg, "text", { x: L + w - 12, y: y + 15, fill: "var(--muted)", "font-size": 11, "font-weight": 550, "font-family": F, "text-anchor": "end" }, r.sub || "");
    if (r.tip) fpTipOn(rg, r.tip);
  });
  let yc = T + head + rows.length * rh + 4;
  if (foot) mk(g, "line", { x1: L + 10, x2: L + w - 10, y1: yc, y2: yc, stroke: "var(--line)" });
  for (const [arr, d, word] of [[ups, 1, "넘으면"], [dns, -1, "깨면"]]) {
    if (!arr.length) continue;
    const c = d > 0 ? "var(--good)" : "var(--bad)", rg = mk(g, "g", {});
    mk(rg, "rect", { x: L + 6, y: yc + 2, width: w - 12, height: 18, fill: "transparent" });
    mk(rg, "text", { x: L + 12, y: yc + 15, fill: "var(--muted)", "font-size": 11, "font-weight": 650, "font-family": F }, word);
    dirTri(rg, L + 12 + measureTextW(word, '650 11px "Pretendard Variable", Pretendard, sans-serif') + 9, yc + 11, d, c, 4);
    mk(rg, "text", { x: L + 12 + measureTextW(word, '650 11px "Pretendard Variable", Pretendard, sans-serif') + 18, y: yc + 15, fill: "var(--text)", "font-size": 11, "font-weight": 650, "font-family": F },
       arr.map((q) => `${q.tf} ${q.v}${q.wait ? " 마감 대기" : ""}`).join(" · "));
    fpTipOn(rg, `<b>큰 스윙 ${d > 0 ? "고점" : "저점"}(앞뒤 50봉 · 아직 안 깨짐)</b><br>그 봉(${arr.map((q) => q.tf).join("·")}) 종가로 ${word} 구조 돌파<br>돌파 뒤 깬 쪽 2 ATR 먼저 약 53% · 무작위 49%<br>추세 전환 신호 아님 · 참고`);
    yc += 20;
  }
  if (!R) return;
  // ── 지금 읽기 ──
  const ys = T + hb, FT = '"Pretendard Variable", Pretendard, sans-serif', short = (l) => (String(l.text).split(" — ")[1] || String(l.text)).split(":")[0];
  const brk = (t) => escapeHtml(t).split(" — ").join("<br>— ");   // 툴팁은 줄을 안 접는다 -- 관찰 / 해석을 줄로 가른다
  const fit = (txt, font, max) => { let t = txt; while (t.length > 1 && measureTextW(t === txt ? t : t + "…", font) > max) t = t.slice(0, -1); return t === txt ? t : t + "…"; };
  const nu = R.hot.filter((l) => l.dir > 0).length, nd = R.hot.filter((l) => l.dir < 0).length;
  if (rows.length || foot) mk(g, "line", { x1: L + 10, x2: L + w - 10, y1: ys, y2: ys, stroke: "var(--line)" });
  const hg = mk(g, "g", {});
  mk(hg, "rect", { x: L + 6, y: ys + 4, width: w - 12, height: 22, fill: "transparent" });
  mk(hg, "text", { x: L + 12, y: ys + 19, fill: "var(--muted)", "font-size": 11, "font-weight": 700, "font-family": F }, "지금 읽기");
  dirTri(hg, L + w - 64, ys + 15, 1, nu ? "var(--good)" : "var(--muted)", 4); mk(hg, "text", { x: L + w - 56, y: ys + 19, fill: nu ? "var(--good)" : "var(--muted)", "font-size": 11, "font-weight": 750, "font-family": F }, String(nu));
  dirTri(hg, L + w - 32, ys + 15, -1, nd ? "var(--bad)" : "var(--muted)", 4); mk(hg, "text", { x: L + w - 24, y: ys + 19, fill: nd ? "var(--bad)" : "var(--muted)", "font-size": 11, "font-weight": 750, "font-family": F }, String(nd));
  fpTipOn(hg, "<b>지금 읽기</b><br>실시간 데이터 쌍(체결·크기별 흐름·OI·청산·호가·베이시스 …)을 서버가 매초 읽은 것<br>▲▼ = 과거 검정에서 방향 근거가 있던 줄만 · 네모 = 움직임 크기 줄(방향 아님)<br>효과는 수 bp 수준 -- 메이커로만 비용을 넘는다 · 나머지 줄은 «설명»");
  R.hot.forEach((l, i) => {
    const y = ys + 28 + i * rh, big = l.g && l.g.v >= 80, c = l.dir > 0 ? "var(--good)" : l.dir < 0 ? "var(--bad)" : big ? "var(--warn)" : "var(--muted)", rg = mk(g, "g", {});
    mk(rg, "rect", { x: L + 6, y, width: w - 12, height: rh - 2, rx: 5, fill: "var(--text)", "fill-opacity": i % 2 ? 0 : 0.035 });
    if (l.dir) dirTri(rg, L + 19, y + (rh - 2) / 2, l.dir, c, 5); else mk(rg, "rect", { x: L + 15, y: y + 6.5, width: 8, height: 8, rx: 2, fill: c });
    mk(rg, "text", { x: L + 32, y: y + 15, fill: "var(--text)", "font-size": 12, "font-weight": 700, "font-family": F }, l.topic);
    const gw = measureTextW(l.grade, `700 10.5px ${FT}`), room = w - 12 - 32 - measureTextW(l.topic, `700 12px ${FT}`) - gw - 22;
    mk(rg, "text", { x: L + w - 12, y: y + 15, fill: c, "font-size": 10.5, "font-weight": 700, "font-family": F, "text-anchor": "end" }, l.grade);
    if (room > 24) mk(rg, "text", { x: L + w - 12 - gw - 7, y: y + 15, fill: "var(--muted)", "font-size": 11, "font-weight": 550, "font-family": F, "text-anchor": "end" }, fit(short(l), `550 11px ${FT}`, room));
    fpTipOn(rg, `<b>${escapeHtml(l.topic)} · ${escapeHtml(l.grade)}${l.dir ? (l.dir > 0 ? " · 위쪽" : " · 아래쪽") : " · 크기(방향 아님)"}</b><br>${brk(l.text)}<br>${brk(l.note)}`);
  });
  if (R.desc.length) {
    const y = ys + 28 + R.hot.length * rh, on = R.desc.filter((l) => l.g && l.g.on).map((l) => l.topic), dg = mk(g, "g", {}), lab = `설명 ${R.desc.length}`;
    mk(dg, "rect", { x: L + 6, y, width: w - 12, height: 20, fill: "transparent" });
    mk(dg, "text", { x: L + 12, y: y + 14, fill: "var(--muted)", "font-size": 11, "font-weight": 650, "font-family": F }, lab);
    if (on.length) mk(dg, "text", { x: L + 12 + measureTextW(lab, `650 11px ${FT}`) + 8, y: y + 14, fill: "var(--muted)", "font-size": 11, "font-weight": 550, "font-family": F },
                      fit(on.join(" · "), `550 11px ${FT}`, w - 24 - measureTextW(lab, `650 11px ${FT}`) - 8));
    fpTipOn(dg, `<b>설명 줄 ${R.desc.length}개 -- 지금 일어난 일일 뿐, 다음 방향 근거 없음</b><br>${R.desc.map((l) => `${escapeHtml(l.topic)} · ${escapeHtml(short(l))}`).join("<br>")}`);
  }
}

