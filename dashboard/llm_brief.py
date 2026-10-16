"""LLM(Haiku) 해석 입력 요약 (2026-10-17 사용자 «청산맵 1d/7d/30d 와 풋프린트 1h/2h/4h/1d 도 입력으로» → «그렇게 해줘»).

왜: 화면 원본(청산맵 1일 730KB · 풋프린트 일봉 340KB)은 그대로 못 넘긴다 → 화면에서 «읽는» 것만 숫자로 접는다. 전부 순수 함수.
근거: research_llm_haiku_window_read_20261017 -- 스냅샷+윈도우+규칙 사건(C) 이 흐름을 가장 정확히 읽었다(부호 100% · 엄격 92.7%).
🔴방향 판단 재료가 아니다: 청산맵 규칙(자석·쓸기)·풋프린트 패턴(다이버·소진·흡수)은 방향 검정 불통과 -- «지도»로만 서술하게 한다.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from dashboard.market_ctx import value_area

BAND_REL = 0.25          # 띠 = 한쪽 최대의 25% 이상 칸(화면 liqBandEdges 의 t ≥ 0.25 와 같은 문턱)
BAND_RANGE = 0.10        # ±10% 밖은 버린다
IMB_RATIO, IMB_RUN = 3.0, 3   # 쌓인 불균형 = 같은 가격 줄에서 매수 > 매도×3(또는 반대)가 3줄 이상 연속(app.js FOOTPRINT_IMBALANCE_RATIO)
LOOK = 6                 # 패턴 표식(옛 fpPatternMarks, c0979beb): 새 6봉 극값에서만


def r(x, n: int = 3):
    """유효숫자 n 자리(없으면 None) -- 토큰 절약."""
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else float(f"{x:.{n}g}")


def hhmm(ts_s: float) -> str:
    return datetime.fromtimestamp(ts_s, tz=timezone.utc).strftime("%H:%M")


# ── 청산맵 ────────────────────────────────────────────────────────────────────────────
def liqmap_summary(tp: dict | None, px: float) -> dict | None:
    """tier_profile(compute_tier_profile 출력) → 현재가 위(숏 청산)·아래(롱 청산) 띠. unit=oi_usd 면 금액(백만$), share 면 몫."""
    if not tp or not tp.get("tiers") or not px:
        return None
    bw, lo = float(tp["bin_width"]), int(tp["lo"])
    v = np.sum([np.asarray(t["values"], float) for t in tp["tiers"]], axis=0)
    price = (lo + np.arange(len(v)) + 0.5) * bw
    usd = tp.get("unit") == "oi_usd"
    scale = 1e6 if usd else 1.0
    out = {"size_unit": "USD millions" if usd else "share of total", "above_short_liq": [], "below_long_liq": []}
    for side, m in (("above_short_liq", (price > px) & (price <= px * (1 + BAND_RANGE))),
                    ("below_long_liq", (price < px) & (price >= px * (1 - BAND_RANGE)))):
        if not m.any() or v[m].max() <= 0:
            continue
        idx = np.flatnonzero(m & (v >= BAND_REL * v[m].max()))
        bands, cur = [], [int(idx[0])]
        for i in idx[1:]:
            if i - cur[-1] <= 2:            # 한 칸 빈 곳은 같은 띠
                cur.append(int(i))
            else:
                bands.append(cur); cur = [int(i)]
        bands.append(cur)
        rows = []
        for b in bands:
            seg = np.arange(b[0], b[-1] + 1)
            lo_p, hi_p = price[seg[0]] - bw / 2, price[seg[-1]] + bw / 2
            edge = lo_p if side.startswith("above") else hi_p
            rows.append({"from": r(lo_p, 5), "to": r(hi_p, 5), "dist_pct": r((edge / px - 1) * 100),
                         "size": r(v[seg].sum() / scale), "peak": r(price[seg[np.argmax(v[seg])]], 5)})
        rows.sort(key=lambda x: abs(x["dist_pct"]))
        biggest = max(rows, key=lambda x: x["size"])
        out[side] = rows[:3] + ([dict(biggest, largest=True)] if biggest not in rows[:3] else [])
    for w in (3, 5):
        a = v[(price > px) & (price <= px * (1 + w / 100))].sum() / scale
        b = v[(price < px) & (price >= px * (1 - w / 100))].sum() / scale
        out[f"within_{w}pct"] = {"above": r(a), "below": r(b), "above_over_below": r(a / b) if b > 0 else None}
    return out


# ── 풋프린트 ──────────────────────────────────────────────────────────────────────────
def _bar_stats(levels: list, row: float) -> dict | None:
    """레벨 행 [price, buy, sell, whale_buy, whale_sell, retail_buy, retail_sell] → 봉 통계(가격 줄 = row 폭으로 다시 묶음)."""
    if not levels:
        return None
    L = np.asarray(levels, float)
    p, b, s = L[:, 0], L[:, 1], L[:, 2]
    k = np.floor(p / row).astype(np.int64)
    kk, inv = np.unique(k, return_inverse=True)
    B, S = np.bincount(inv, weights=b), np.bincount(inv, weights=s)
    tot = B + S
    return {"k": kk, "buy": B, "sell": S, "delta": float(b.sum() - s.sum()), "vol": float(tot.sum()),
            "whale_delta": float(L[:, 3].sum() - L[:, 4].sum()), "retail_delta": float(L[:, 5].sum() - L[:, 6].sum()),
            "hi": float(p.max()), "lo": float(p.min()), "poc": float((kk[np.argmax(tot)] + 0.5) * row),
            "top_buy": float(B[-1]), "bot_sell": float(S[0])}


def stacked_imbalances(st: dict, row: float) -> list[dict]:
    """한 봉 안 같은 가격 줄 비교: 매수 > 매도×3 이 연속 IMB_RUN 줄 이상 = 매수 쌓임(매도 반대)."""
    out = []
    for side, a, b in (("buy", st["buy"], st["sell"]), ("sell", st["sell"], st["buy"])):
        flag = (a > 0) & (a > IMB_RATIO * b)
        run: list[int] = []
        for i, f in enumerate(list(flag) + [False]):
            if f and (not run or st["k"][i] == st["k"][run[-1]] + 1):
                run.append(i)
                continue
            if len(run) >= IMB_RUN:
                out.append({"side": side, "from": r(st["k"][run[0]] * row, 5), "to": r((st["k"][run[-1]] + 1) * row, 5), "rows": len(run)})
            run = [i] if f else []
    return out


def pattern_marks(stats: list[dict], closes: list) -> list[dict]:
    """옛 fpPatternMarks(c0979beb) 규칙: 새 LOOK 봉 고점에서 div−(누적델타 낮아짐)·exh−(큰 매수델타인데 봉 아래쪽 마감)·abs−(꼭대기 매수 큼·다음 봉이 못 넘음),
    저점은 거울. 검정(09-30): 방향 정보 0 -- 서술용."""
    n = len(stats)
    if n < LOOK + 2:
        return []
    d = np.array([s["delta"] for s in stats]); cv = np.cumsum(d)
    nz = np.abs(d[d != 0]); bigD = np.quantile(nz, .8) if len(nz) else np.inf
    tb = np.array([s["top_buy"] for s in stats]); bs = np.array([s["bot_sell"] for s in stats])
    tbQ = np.quantile(tb[tb > 0], .9) if (tb > 0).any() else np.inf
    bsQ = np.quantile(bs[bs > 0], .9) if (bs > 0).any() else np.inf
    hi = np.array([s["hi"] for s in stats]); lo = np.array([s["lo"] for s in stats])
    out = []
    for i in range(LOOK, n - 1):                # 다음 봉이 있어야 흡수를 판정 -- 마지막 봉은 제외
        c, rng = closes[i], hi[i] - lo[i]
        pos = (c - lo[i]) / rng if c is not None and rng > 0 else 0.5
        t = stats[i]["t"]
        if hi[i] > hi[i - LOOK:i].max():
            jh = i - LOOK + int(np.argmax(hi[i - LOOK:i]))
            if cv[i] < cv[jh]:
                out.append({"at": t, "kind": "bearish_divergence_at_new_high"})
            if d[i] > 0 and d[i] >= bigD and pos <= 0.4:
                out.append({"at": t, "kind": "buy_exhaustion_at_new_high"})
            if tb[i] >= tbQ and hi[i + 1] <= hi[i]:
                out.append({"at": t, "kind": "buy_absorption_at_new_high"})
        if lo[i] < lo[i - LOOK:i].min():
            jl = i - LOOK + int(np.argmin(lo[i - LOOK:i]))
            if cv[i] > cv[jl]:
                out.append({"at": t, "kind": "bullish_divergence_at_new_low"})
            if d[i] < 0 and -d[i] >= bigD and pos >= 0.6:
                out.append({"at": t, "kind": "sell_exhaustion_at_new_low"})
            if bs[i] >= bsQ and lo[i + 1] >= lo[i]:
                out.append({"at": t, "kind": "sell_absorption_at_new_low"})
    return out


def footprint_summary(bars: list[dict], n: int, px: float, close_by_t: dict | None = None, group: int = 1) -> dict | None:
    """최근 n 개 5분 풋프린트 봉 → 창 요약. group = 봉 목록을 몇 개씩 묶어 보여줄지(1h=1 · 2h·4h=3 = 15분)."""
    bars = [b for b in bars if not b.get("agg") and b.get("levels")][-n:]
    if len(bars) < max(3, n // 2) or not px:
        return None
    row = max(0.5, round(px * 0.0004 * 2) / 2)          # ≈ 0.04% (ETH 2,400 → $1)
    stats = []
    for b in bars:
        st = _bar_stats(b["levels"], row)
        if st:
            st["t"], st["t0"] = hhmm(b["time"] + 300), int(b["time"])   # 봉 마감 시각
            stats.append(st)
    allL = np.vstack([np.asarray(b["levels"], float) for b in bars])
    va = value_area(np.floor(allL[:, 0] / row).astype(np.int64), allL[:, 1] + allL[:, 2], row)
    closes = [(close_by_t or {}).get(s["t0"]) for s in stats]
    per = []
    for g in range(0, len(stats), group):
        ch = stats[g:g + group]
        per.append({"end": ch[-1]["t"], "delta": r(sum(c["delta"] for c in ch)), "whale_delta": r(sum(c["whale_delta"] for c in ch)),
                    "retail_delta": r(sum(c["retail_delta"] for c in ch)), "vol": r(sum(c["vol"] for c in ch)),
                    "hi": r(max(c["hi"] for c in ch), 5), "lo": r(min(c["lo"] for c in ch), 5),
                    **({"poc": r(ch[0]["poc"], 5)} if group == 1 else {})})
    imb = [{"bar_end": s["t"], **z} for s in stats for z in stacked_imbalances(s, row)]
    pos = None
    if va:
        pos = "above_value_area" if px > va["vah"] else "below_value_area" if px < va["val"] else "inside_value_area"
    return {"bars_5m": len(stats), "row_usd": row, "delta": r(sum(s["delta"] for s in stats)),
            "whale_delta": r(sum(s["whale_delta"] for s in stats)), "retail_delta": r(sum(s["retail_delta"] for s in stats)),
            "high": r(max(s["hi"] for s in stats), 5), "low": r(min(s["lo"] for s in stats), 5),
            "value_area": {k: r(v, 5) for k, v in va.items()} if va else None, "price_vs_value_area": pos,
            ("per_bar" if group == 1 else f"per_{5 * group}m"): per,
            "stacked_imbalances_recent": imb[-6:], "pattern_marks": pattern_marks(stats, closes)[-6:]}


def footprint_daily_summary(cols: dict, cells: dict | None, row: float, px: float, ndays: int = 3) -> list[dict]:
    """일봉 풋프린트(/api/footprint-daily cols + /cells{day:[lo_px,buy[],sell[]]}) → 최근 ndays 일(오늘은 진행 중)."""
    out = []
    days = list(cols.get("d", []))
    part = cols.get("part") or [0] * len(days)
    for i in range(max(0, len(days) - ndays), len(days)):
        d = days[i]
        rowd = {"date": d[5:], "open": r(cols["o"][i], 5), "high": r(cols["h"][i], 5), "low": r(cols["l"][i], 5), "close": r(cols["c"][i], 5),
                "delta_eth": r(cols["bq"][i] - cols["sq"][i]), "volume_eth": r(cols["bq"][i] + cols["sq"][i]),
                "liq_long_usd_m": r(cols["ll"][i] / 1e6), "liq_short_usd_m": r(cols["ls"][i] / 1e6),
                "oi_chg_pct": r((cols["oi"][i] / cols["oi"][i - 1] - 1) * 100) if i > 0 and cols["oi"][i - 1] else None,
                "in_progress": bool(part[i])}
        c = (cells or {}).get(d)
        if c:
            buy, sell = np.asarray(c[1], float), np.asarray(c[2], float)
            k = int(round(c[0] / row)) + np.arange(len(buy))
            va = value_area(k, buy + sell, row)
            if va:
                rowd["value_area"] = {kk: r(v, 5) for kk, v in va.items()}
                rowd["close_vs_value_area"] = "above" if rowd["close"] > va["vah"] else "below" if rowd["close"] < va["val"] else "inside"
        out.append(rowd)
    return out


# ── 윈도우 · 규칙 사건 ────────────────────────────────────────────────────────────────
def rule_events(w5: dict, tau_oi: float = 0.5, tau_basis: float = 1.5) -> list[str]:
    """5분×24 윈도우 → 사건 문장(research_llm_haiku_window_read_20261017.events_rule 과 같은 규칙)."""
    t = w5["time"]
    ev = []
    px = np.array([np.nan if x is None else x for x in w5["px_pct"]], float)
    mv = np.diff(px)
    if np.isfinite(mv).any():
        j = int(np.nanargmax(np.abs(mv))) + 1
        ev.append(f"largest 5m move {mv[j - 1]:+.2f}% at {t[j]}")
    for key, nm in (("liq_long_usd", "long liquidation"), ("liq_short_usd", "short liquidation")):
        x = np.array([0 if v is None else v for v in w5.get(key, [])], float)
        if len(x):
            thr = max(1e6, 5 * max(np.median(x), 1.0))
            ev += [f"{nm} cluster ${x[i] / 1e6:.1f}M at {t[i]}" for i in np.flatnonzero(x >= thr)]
    vol = np.array([np.nan if v is None else v for v in w5.get("vol", [])], float)
    if np.isfinite(vol).any():
        med = np.nanmedian(vol)
        ev += [f"volume spike {vol[i] / med:.1f}x 2h median at {t[i]}" for i in np.flatnonzero(vol >= 3 * med)]
    oi = [x for x in w5.get("oi_chg_pct", []) if x is not None]
    if len(oi) >= 2 and abs(oi[-1] - oi[0]) >= tau_oi:
        ev.append(f"OI {oi[-1] - oi[0]:+.2f}% over last 2h")
    bas = [x for x in w5.get("basis_bp", []) if x is not None]
    if len(bas) >= 2 and abs(bas[-1] - bas[0]) >= tau_basis:
        ev.append(f"basis {bas[0]:+.1f}→{bas[-1]:+.1f}bp over 2h")
    if np.isfinite(px).any():
        ev.append(f"2h high at {t[int(np.nanargmax(px))]}, 2h low at {t[int(np.nanargmin(px))]}")
    return ev


if __name__ == "__main__":
    # 자체점검 -- 합성 입력
    vals = np.zeros(200); vals[105] = 5e6; vals[106] = 3e6; vals[90] = 9e6; vals[150] = 1e6; vals[60] = 2e5
    tp = {"bin_width": 1.0, "lo": 900, "unit": "oi_usd", "tiers": [{"name": "10", "values": vals.tolist()}]}
    s = liqmap_summary(tp, 1000.0)
    a, b = s["above_short_liq"], s["below_long_liq"]
    assert a[0]["from"] == 1005 and a[0]["to"] == 1007 and abs(a[0]["size"] - 8) < 1e-9, a     # 1005~1007 = 한 띠 $8M
    assert b[0]["to"] == 991 and abs(b[0]["size"] - 9) < 1e-9 and len(b) == 1, b                  # 아래 990 칸 하나 · 0.2M(최대의 2%)은 띠 아님
    assert s["within_3pct"]["above"] == 8 and s["within_3pct"]["below"] == 9, s
    lv = [[1000.2, 10, 1, 0, 0, 0, 0], [1001.2, 9, 2, 0, 0, 0, 0], [1002.3, 8, 1, 0, 0, 0, 0], [1003.1, 1, 1, 0, 0, 0, 0]]
    z = stacked_imbalances(_bar_stats(lv, 1.0), 1.0)
    assert z == [{"side": "buy", "from": 1000.0, "to": 1003.0, "rows": 3}], z
    bars = [{"time": 1_700_000_000 + 300 * i, "levels": [[h - 1, 5 + max(dl, 0), 5 + max(-dl, 0), 0, 0, 0, 0], [h, 1, 1, 0, 0, 0, 0]]}
            for i, (h, dl) in enumerate([(100, 5), (101, 5), (100, -2), (99, -2), (99, -2), (99, -2), (101.5, -8), (100, 1)])]
    stats = []
    for b_ in bars:
        q = _bar_stats(b_["levels"], 1.0); q["t"] = hhmm(b_["time"] + 300); stats.append(q)
    pm = pattern_marks(stats, [None] * len(stats))
    assert any(m["kind"] == "bearish_divergence_at_new_high" and m["at"] == stats[6]["t"] for m in pm), pm
    fs = footprint_summary(bars, 8, 100.0)
    assert fs and fs["value_area"]["val"] <= fs["value_area"]["poc"] <= fs["value_area"]["vah"] and fs["bars_5m"] == 8, fs
    w5 = {"time": [f"10:{m:02d}" for m in range(0, 60, 5)] * 2, "px_pct": list(np.linspace(-1, 0, 24)), "vol": [1] * 23 + [9],
          "liq_long_usd": [0] * 20 + [3e6, 0, 0, 0], "liq_short_usd": [0] * 24, "oi_chg_pct": [0] * 23 + [0.8], "basis_bp": [-5] * 24}
    ev = rule_events(w5)
    assert any("long liquidation cluster $3.0M" in e for e in ev) and any("volume spike" in e for e in ev) and any(e.startswith("OI +0.80%") for e in ev), ev
    print("selftest OK")
