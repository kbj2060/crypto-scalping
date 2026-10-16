#!/usr/bin/env python3
"""**Haiku 가 «흐름»을 맞게 읽나** — 스냅샷(A) vs +윈도우(B) vs +윈도우+규칙 사건(C), 주장 단위 기계 채점 (2026-10-17).

왜: 사용자 «입력 피쳐 더 추가할 필요 없어? 데이터를 윈도우로 주는 게 흐름을 읽을 수 있지 않을까?» → «진행해줘».
기간: 2026-07-20 ~ 08-16(실측 청산 tail_risk 07-18~08-17 · aggTrades 고래/리테일 ~08-22 · OI/롱숏 패널 ~09-14 가 다 있는 구간).
  24 시점(서로 ≥8h): 12 무작위 + 12 «흐름 있음»(직전 2h |가격 변화| 상위 25%). 날짜·절대가격 없음(시각은 UTC HH:MM 만).
입력: A 스냅샷(가격 변화·베이시스+7일 분위·OI+z·롱숏비·흐름·현물·청산 1h·박스 유지 확률+꼬리표·일봉 추세 표·다음 발표·BTC) ·
  B = A + 5분×24(2h)·1h×24(24h) 시계열(가격 %·거래량·선물/현물 테이커 순매수·OI 변화·베이시스·고래/리테일 불균형·롱/숏 청산) ·
  C = B + 규칙이 뽑은 사건 목록. 🔴빠진 것(과거 원천 없음): 다거래소 OI·청산, 청산맵, 뉴스, 옵션.
출력: 한국어 해석 ≤6문장 + claims[{series, span(30m|2h|24h), kind(rise|fall|flat|spike), at(HH:MM, spike 만)}] — 해석 속 흐름 주장을 전부.
채점(기계): 수준 계열(px·oi·basis) = 구간 첫→끝 변화 Δ 로 rise(Δ>τ/2)·fall(Δ<−τ/2)·flat(|Δ|<τ) · 흐름 계열(perp_net·spot_net·whale·retail) =
  구간 합 부호(rise=순매수) · 양 계열(vol·liq_long·liq_short) = 뒤 절반/앞 절반 ≥1.2(rise)·≤1/1.2(fall) · spike = 그 시각 ±5분 값 ≥ 구간 중앙값×2.5.
  + 재현율: 규칙 사건(가장 큰 5분 움직임·청산 몰림·거래량 급증·2h OI 추세) 중 주장으로 짚은 비율(C 는 사건을 받았으니 참고용).
  python scripts/research_llm_haiku_window_read_20261017.py build | eval | --selfcheck
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")
BV = MAIN / "data/binance_vision"
OUT = ROOT / "tmp/haiku_window_20261017"
BDIR = ROOT / "tmp/haiku_window_20261017_inputs"     # 🔴 에이전트가 읽는 폴더 — 원본 시계열(OUT/truth.json)과 분리
START, END = "2026-07-20", "2026-08-16 20:00"
N, PER = 24, 12
SERIES = ("px", "vol", "perp_net", "spot_net", "oi", "basis", "whale_imb", "retail_imb", "liq_long", "liq_short")
TAU = {"px": {"30m": .15, "2h": .3, "24h": 1.0}, "oi": {"30m": .3, "2h": .5, "24h": 1.5}, "basis": {"30m": 1.0, "2h": 1.5, "24h": 3.0}}


def load() -> pd.DataFrame:
    """5분 격자(봉 시작 ts). 모든 칸은 그 봉 마감까지 아는 값."""
    p = pd.read_parquet(BV / "panel/ETHUSDT.parquet").set_index("timestamp").sort_index()
    p = p[(p.index >= "2026-06-01") & (p.index < "2026-08-24")]
    F = pd.DataFrame(index=p.index)
    F["close"] = p.close; F["vol"] = p.volume; F["perp_net"] = 2 * p.taker_buy_base - p.volume
    for c, nm in (("sum_open_interest", "oi"), ("count_long_short_ratio", "ls_global"), ("sum_toptrader_long_short_ratio", "ls_toppos"),
                  ("sum_taker_long_short_vol_ratio", "ls_taker")):
        F[nm] = p[c].shift(1)                                              # metrics 스탬프 규약 -- 한 행 늦춤
    pr = pd.read_parquet(BV / "eth_premium_5m.parquet"); pr.index = pd.to_datetime(pr.open_time, unit="ms")
    F["basis"] = pr.close.reindex(F.index) * 1e4
    sp = pd.read_parquet(BV / "eth_spot_5m.parquet"); sp.index = pd.to_datetime(sp.open_time, unit="ms")
    sp = sp.reindex(F.index); F["spot_net"] = 2 * sp.tb - sp.volume
    ag = pd.concat([pd.read_parquet(f) for f in sorted((BV / "aggfeat").glob("ETHUSDT_2026-0[6-8]-*.parquet"))]).sort_index()
    ag = ag[~ag.index.duplicated()].reindex(F.index)
    F["whale_imb"], F["retail_imb"] = ag.tf_bigimb, ag.tf_smlimb
    c = duckdb.connect(str(MAIN / "data/live/tail_risk.duckdb"), read_only=True)
    L = c.execute("select ts, long_usd_1m, short_usd_1m from tail_risk_1m where ts >= '2026-07-18'").fetchdf()
    L["t"] = pd.to_datetime(L.ts).dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("5min")
    L = L.groupby("t")[["long_usd_1m", "short_usd_1m"]].sum().reindex(F.index).fillna(0)
    F["liq_long"], F["liq_short"] = L.long_usd_1m, L.short_usd_1m
    b = pd.read_parquet(BV / "panel/BTCUSDT.parquet").set_index("timestamp").close.reindex(F.index)
    F["btc"] = b
    return F


def pick(F: pd.DataFrame) -> list[pd.Timestamp]:
    cand = F[(F.index >= START) & (F.index <= END)].copy()
    cand["mv2h"] = (np.log(cand.close) - np.log(F.close.shift(24).reindex(cand.index))).abs()
    cand = cand.dropna(subset=["mv2h", "whale_imb", "basis", "oi"])
    hot = cand[cand.mv2h >= cand.mv2h.quantile(.75)]
    rng = np.random.default_rng(20261017)
    chosen: list[pd.Timestamp] = []
    def take(pool, k):
        for t in pool.index[rng.permutation(len(pool))]:
            if all(abs((t - u).total_seconds()) >= 8 * 3600 for u in chosen):
                chosen.append(t)
                if k and sum(1 for _ in chosen) >= k:
                    return
    take(hot, PER); take(cand, 2 * PER)
    return sorted(chosen[:N])


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else float(f"{x:.{n}g}")


def windows(F: pd.DataFrame, t: pd.Timestamp) -> dict:
    """B·C 시계열 + 채점용 같은 값. 시각 = 봉 마감 UTC HH:MM."""
    k = F.index.get_loc(t)
    w5 = F.iloc[k - 23:k + 1]
    now = F.close.iloc[k]
    lab = lambda idx: [(i + pd.Timedelta("5min")).strftime("%H:%M") for i in idx]  # noqa: E731
    s5 = {"time": lab(w5.index), "px_pct": [r((x / now - 1) * 100) for x in w5.close], "vol": [r(x) for x in w5.vol],
          "perp_net": [r(x) for x in w5.perp_net], "spot_net": [r(x) for x in w5.spot_net],
          "oi_chg_pct": [r((x / w5.oi.iloc[0] - 1) * 100) for x in w5.oi], "basis_bp": [r(x) for x in w5.basis],
          "whale_imb": [r(x) for x in w5.whale_imb], "retail_imb": [r(x) for x in w5.retail_imb],
          "liq_long_usd": [r(x) for x in w5.liq_long], "liq_short_usd": [r(x) for x in w5.liq_short]}
    h = F.iloc[k - 287:k + 1]
    g = (np.arange(len(h)) // 12)
    H = pd.DataFrame({"close": h.close.groupby(g).last().to_numpy(), "vol": h.vol.groupby(g).sum().to_numpy(),
                      "perp_net": h.perp_net.groupby(g).sum().to_numpy(), "spot_net": h.spot_net.groupby(g).sum().to_numpy(),
                      "oi": h.oi.groupby(g).last().to_numpy(), "basis": h.basis.groupby(g).mean().to_numpy(),
                      "liq_long": h.liq_long.groupby(g).sum().to_numpy(), "liq_short": h.liq_short.groupby(g).sum().to_numpy()})
    hl = [(h.index[(gi + 1) * 12 - 1] + pd.Timedelta("5min")).strftime("%H:%M") for gi in range(24)]
    s1 = {"time": hl, "px_pct": [r((x / now - 1) * 100) for x in H.close], "vol": [r(x) for x in H.vol],
          "perp_net": [r(x) for x in H.perp_net], "spot_net": [r(x) for x in H.spot_net],
          "oi_chg_pct": [r((x / H.oi.iloc[0] - 1) * 100) for x in H.oi], "basis_bp": [r(x) for x in H.basis],
          "liq_long_usd": [r(x) for x in H.liq_long], "liq_short_usd": [r(x) for x in H.liq_short]}
    return {"w5": s5, "w1h": s1}


def snapshot(F: pd.DataFrame, t: pd.Timestamp, K1: pd.DataFrame, ev: list) -> dict:
    from dashboard.box_hold import payload
    k = F.index.get_loc(t)
    c = F.close
    ch = lambda n: r((c.iloc[k] / c.iloc[k - n] - 1) * 100)  # noqa: E731
    bas7 = F.basis.iloc[k - 2015:k + 1]
    doi = np.log(F.oi).diff(12)
    oi_z = (doi.iloc[k] - doi.iloc[k - 2016:k].median()) / (1.55 * (doi.iloc[k - 2016:k] - doi.iloc[k - 2016:k].median()).abs().median())
    pct7 = float((bas7 <= F.basis.iloc[k]).mean())
    tag = lambda q: "low (bottom 10% of 7d)" if q <= .1 else "high (top 10% of 7d)" if q >= .9 else "normal"  # noqa: E731
    vz = np.log(F.vol.iloc[k - 11:k + 1].sum()) - np.log(F.vol.iloc[k - 2015:k + 1].rolling(12).sum().median())
    end_ms = int((t + pd.Timedelta("5min")).value // 10**6)
    kk = K1[K1.t < end_ms].tail(1499)
    k1m = pd.DataFrame({"timestamp": pd.to_datetime(kk.t, unit="ms", utc=True), "high": kk.h, "low": kk.l, "close": kk.c,
                        "volume": kk.v, "taker_buy_base": kk.tb}).reset_index(drop=True)
    bh = payload(k1m, ev)
    box = {s: {"p_hold": round(v["p"] * 100), "typical": round(v["base"] * 100),
               "tag": "likely to break (bottom 20%)" if v["p"] < v["q20"] else "likely to hold (top 20%)" if v["p"] > v["q80"] else "normal",
               "half_width_pct": v["half_pct"]} for s, v in bh["scales"].items() if v.get("ok")}
    day = c.resample("1D").last()
    dk = day.index.get_indexer([t.floor("D")])[0] - 1                       # 어제까지 닫힌 일봉
    votes = np.mean([np.sign(day.iloc[dk] - day.iloc[dk - L]) for L in (7, 14, 28, 56, 90) if dk - L >= 0]) if dk > 7 else np.nan
    maj = np.array([e for e in json.loads((ROOT / "tmp/box_improve_20261016/events.json").read_text())["major"]])
    nxt = maj[maj > end_ms]
    return {"time_utc": (t + pd.Timedelta("5min")).strftime("%H:%M %a"),
            "price_chg_pct": {"30m": ch(6), "1h": ch(12), "4h": ch(48), "24h": ch(288)},
            "btc_price_chg_pct": {"1h": r((F.btc.iloc[k] / F.btc.iloc[k - 12] - 1) * 100), "4h": r((F.btc.iloc[k] / F.btc.iloc[k - 48] - 1) * 100)},
            "basis_bp": r(F.basis.iloc[k]), "basis_chg_30m_bp": r(F.basis.iloc[k] - F.basis.iloc[k - 6]),
            "basis_pct_rank_7d": round(pct7, 2), "basis_tag": tag(pct7),
            "oi_chg_pct": {"1h": r((F.oi.iloc[k] / F.oi.iloc[k - 12] - 1) * 100), "24h": r((F.oi.iloc[k] / F.oi.iloc[k - 288] - 1) * 100)},
            "oi_1h_change_robust_z": r(oi_z), "oi_tag": "surging" if oi_z >= 1.5 else "dropping fast" if oi_z <= -1.5 else "normal",
            "long_short_ratio": {"global_accounts": r(F.ls_global.iloc[k]), "global_24h_ago": r(F.ls_global.iloc[k - 288]),
                                 "top_trader_positions": r(F.ls_toppos.iloc[k]), "top_24h_ago": r(F.ls_toppos.iloc[k - 288])},
            "perp_taker_net_1h_eth": r(F.perp_net.iloc[k - 11:k + 1].sum()), "spot_taker_net_1h_eth": r(F.spot_net.iloc[k - 11:k + 1].sum()),
            "whale_imb_1h_avg": r(F.whale_imb.iloc[k - 11:k + 1].mean()), "retail_imb_1h_avg": r(F.retail_imb.iloc[k - 11:k + 1].mean()),
            "volume_1h_vs_7d_median": r(float(np.exp(vz))),
            "liquidations_1h_usd": {"long": r(F.liq_long.iloc[k - 11:k + 1].sum()), "short": r(F.liq_short.iloc[k - 11:k + 1].sum())},
            "box_hold": box, "daily_trend_votes_-1to1": r(votes),
            "next_major_us_macro_release_min": r((nxt[0] - end_ms) / 60000) if len(nxt) else None}


def events_rule(F: pd.DataFrame, t: pd.Timestamp) -> list[dict]:
    """규칙 사건(C 입력 + 재현율 채점 기준)."""
    k = F.index.get_loc(t)
    w = F.iloc[k - 23:k + 1]
    hhmm = lambda i: (i + pd.Timedelta("5min")).strftime("%H:%M")  # noqa: E731
    ev = []
    ret = np.log(w.close).diff()
    j = ret.abs().idxmax()
    ev.append({"series": "px", "kind": "spike", "at": hhmm(j), "text": f"largest 5m move {ret[j] * 100:+.2f}% at {hhmm(j)}"})
    for s in ("liq_long", "liq_short"):
        thr = max(1e6, 5 * max(w[s].median(), 1.0))
        for i in w.index[w[s] >= thr]:
            ev.append({"series": s, "kind": "spike", "at": hhmm(i), "text": f"{s.replace('_', ' ')} cluster ${w.loc[i, s] / 1e6:.1f}M at {hhmm(i)}"})
    vmed = w.vol.median()
    for i in w.index[w.vol >= 3 * vmed]:
        ev.append({"series": "vol", "kind": "spike", "at": hhmm(i), "text": f"volume spike {w.loc[i, 'vol'] / vmed:.1f}x 2h median at {hhmm(i)}"})
    d = (w.oi.iloc[-1] / w.oi.iloc[0] - 1) * 100
    if abs(d) >= TAU["oi"]["2h"]:
        ev.append({"series": "oi", "kind": "rise" if d > 0 else "fall", "span": "2h", "text": f"OI {d:+.2f}% over last 2h"})
    db = w.basis.iloc[-1] - w.basis.iloc[0]
    if abs(db) >= TAU["basis"]["2h"]:
        ev.append({"series": "basis", "kind": "rise" if db > 0 else "fall", "span": "2h", "text": f"basis {w.basis.iloc[0]:+.1f}→{w.basis.iloc[-1]:+.1f}bp over 2h"})
    hi, lo = w.close.idxmax(), w.close.idxmin()
    ev.append({"series": "px", "kind": "info", "text": f"2h high at {hhmm(hi)}, 2h low at {hhmm(lo)}"})
    return ev


PROMPT = """You will interpret the ETH perpetual futures market (Binance ETHUSDT) at several independent snapshots (different days, shuffled).
For EACH snapshot write a Korean interpretation (at most 6 sentences) of the current state AND the recent flow (what happened, in what order).
Rules: cite numbers from the data; never invent data that is not given; do NOT predict future direction.
Then list EVERY factual flow claim your interpretation makes, in this machine-checkable form:
  {{"series": one of px|vol|perp_net|spot_net|oi|basis|whale_imb|retail_imb|liq_long|liq_short,
    "span": "30m"|"2h"|"24h", "kind": "rise"|"fall"|"flat"|"spike", "at": "HH:MM" (spike only, else null)}}
  Meaning: px/oi/basis rise|fall|flat = level change over the span; perp_net/spot_net/whale_imb/retail_imb rise = net BUYING over the span,
  fall = net selling; vol/liq_long/liq_short rise|fall = activity increasing|decreasing over the span; spike = a burst at that time.
{extra}
Output ONLY one JSON array: [{{"id": "...", "interpretation": "...", "claims": [...]}}, ...]

Snapshots:
"""
EXTRA = {"A": "You get only a snapshot (current values and a few changes).",
         "B": "You get the snapshot plus time series: w5 = last 2h in 5-minute bars, w1h = last 24h in 1-hour bars (time = bar close, UTC; px_pct = price vs now in %; oi_chg_pct vs first bar of the series).",
         "C": "You get the snapshot, the time series (w5 = last 2h in 5-minute bars, w1h = last 24h in 1-hour bars; time = bar close UTC; px_pct vs now; oi_chg_pct vs first bar) and a list of rule-detected events."}


def build() -> None:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("imp", ROOT / "scripts/research_eth_box_hold_improve_20261016.py")
    imp = importlib.util.module_from_spec(spec); spec.loader.exec_module(imp)
    K1 = imp.load_1m()
    K1 = K1[(K1.t >= pd.Timestamp("2026-07-10").value // 10**6) & (K1.t < pd.Timestamp("2026-08-24").value // 10**6)]
    evj = json.loads((ROOT / "tmp/box_improve_20261016/events.json").read_text())
    evl = [{"time_utc": pd.Timestamp(int(x), unit="ms", tz="UTC").isoformat(), "category": "fomc"} for x in evj["major"]] + \
          [{"time_utc": pd.Timestamp(int(x), unit="ms", tz="UTC").isoformat(), "category": "econ", "detail": "release_id=180"} for x in evj["claims"]]
    F = load()
    T = pick(F)
    OUT.mkdir(parents=True, exist_ok=True); BDIR.mkdir(parents=True, exist_ok=True)
    truth, items = {}, {}
    for j, t in enumerate(T):
        sid = f"w{j:02d}"
        snap = snapshot(F, t, K1, evl)
        win = windows(F, t)
        ev = events_rule(F, t)
        truth[sid] = {"ts": str(t), "w5": win["w5"], "w1h": win["w1h"], "events": ev}
        items[sid] = {"A": {"id": sid, "snapshot": snap}, "B": {"id": sid, "snapshot": snap, **win},
                      "C": {"id": sid, "snapshot": snap, **win, "events": [e["text"] for e in ev]}}
    (OUT / "truth.json").write_text(json.dumps(truth))
    order = list(items); np.random.default_rng(3).shuffle(order)
    for cond in "ABC":
        for b in range(0, N, PER):
            txt = PROMPT.format(extra=EXTRA[cond]) + "\n".join(json.dumps(items[s][cond], separators=(",", ":")) for s in order[b:b + PER])
            (BDIR / f"{cond}_{b // PER}.txt").write_text(txt)
    print(f"{len(T)} 시점 {T[0]} ~ {T[-1]} · 묶음 {sorted(p.name for p in BDIR.glob('*.txt'))}")
    print("크기:", {p.name: p.stat().st_size for p in sorted(BDIR.glob('*.txt'))})


def sign_wrong(claim: dict, tr: dict) -> bool:
    """«부호 기준» 오답: 수준·흐름 계열은 방향이 반대일 때만, 급증은 시각/크기가 틀릴 때 · flat 은 안 셈(크기 경계)."""
    s, span, kind = claim.get("series"), claim.get("span"), claim.get("kind")
    if kind == "spike":
        return check(claim, tr) is False
    if kind == "flat" or s in ("vol", "liq_long", "liq_short"):
        return False
    key = {"px": "px_pct", "oi": "oi_chg_pct", "basis": "basis_bp"}.get(s, s)
    W = tr["w1h"] if span == "24h" else tr["w5"]
    if key not in W:
        return False
    x = np.array([np.nan if v is None else v for v in W[key]], float)
    x = x[-7:] if span == "30m" else x
    d = (x[-1] - x[0]) if s in ("px", "oi", "basis") else np.nansum(x)
    return bool((kind == "rise" and d < 0) or (kind == "fall" and d > 0))


def check(claim: dict, tr: dict) -> bool | None:
    """주장 하나를 원본 시계열로 채점. None = 채점 불가(형식 밖)."""
    s, span, kind = claim.get("series"), claim.get("span"), claim.get("kind")
    key = {"px": "px_pct", "oi": "oi_chg_pct", "basis": "basis_bp", "liq_long": "liq_long_usd", "liq_short": "liq_short_usd"}.get(s, s)
    if s not in SERIES or span not in ("30m", "2h", "24h") or kind not in ("rise", "fall", "flat", "spike"):
        return None
    W = tr["w1h"] if span == "24h" else tr["w5"]
    if key not in W:
        return None
    x = np.array([np.nan if v is None else v for v in W[key]], float)
    if span == "30m":
        x = x[-7:]                          # 30분 변화 = 6칸 전 → 지금(5분봉 7개) -- 스냅샷 price/basis_chg_30m 과 같은 정의
    if kind == "spike":
        at = claim.get("at")
        if not at or at not in W["time"]:
            near = [i for i, h in enumerate(W["time"]) if at and abs((pd.Timestamp(f"2000-01-01 {h}") - pd.Timestamp(f"2000-01-01 {at}")).total_seconds()) <= 300]
        else:
            near = [W["time"].index(at)]
        if not near:
            return False
        full = np.array([np.nan if v is None else v for v in W[key]], float)
        base = np.nanmedian(np.abs(full)) if s not in ("px",) else np.nanmedian(np.abs(np.diff(full)))
        vals = np.abs(full[near]) if s != "px" else np.abs(np.diff(full))[[max(0, i - 1) for i in near]]
        return bool(np.nanmax(vals) >= 2.5 * max(base, 1e-9))
    if s in ("px", "oi", "basis"):
        d = x[-1] - x[0]
        tau = TAU[s][span]
        return bool(d > tau / 2) if kind == "rise" else bool(d < -tau / 2) if kind == "fall" else bool(abs(d) < tau)
    if s in ("perp_net", "spot_net", "whale_imb", "retail_imb"):
        tot = np.nansum(x)
        scale = np.nanmean(np.abs(x)) * len(x) * 0.1
        return bool(tot > 0) if kind == "rise" else bool(tot < 0) if kind == "fall" else bool(abs(tot) <= scale)
    a, b = np.nansum(x[:len(x) // 2]), np.nansum(x[len(x) // 2:])
    if kind == "flat":
        return bool(max(a, b) <= 1.2 * max(min(a, b), 1e-9))
    return bool(b >= 1.2 * max(a, 1e-9)) if kind == "rise" else bool(b <= max(a, 1e-9) / 1.2)


def covered(e: dict, claims: list) -> bool:
    for c in claims:
        if c.get("series") != e["series"]:
            continue
        if e["kind"] == "spike" and c.get("kind") == "spike" and c.get("at"):
            try:
                if abs((pd.Timestamp(f"2000-01-01 {c['at']}") - pd.Timestamp(f"2000-01-01 {e['at']}")).total_seconds()) <= 600:
                    return True
            except Exception:  # noqa: BLE001
                pass
        if e["kind"] in ("rise", "fall") and c.get("kind") == e["kind"] and c.get("span") in ("2h", "30m"):
            return True
    return False


def evaluate() -> None:
    truth = json.loads((OUT / "truth.json").read_text())
    rows = []
    for cond in "ABC":
        ans = {}
        for f in sorted((OUT / "out").glob(f"{cond}_*.json")):
            for a in json.loads(f.read_text()):
                ans[a["id"]] = a
        for sid, tr in truth.items():
            a = ans.get(sid)
            if not a:
                continue
            res = [check(c, tr) for c in a.get("claims", [])]
            sw = [sign_wrong(c, tr) for c in a.get("claims", [])]
            key_ev = [e for e in tr["events"] if e["kind"] in ("spike", "rise", "fall")]
            rows.append({"cond": cond, "id": sid, "n_claims": len(res), "ok": sum(r is True for r in res), "bad": sum(r is False for r in res),
                         "na": sum(r is None for r in res), "sign_bad": sum(sw), "recall": np.mean([covered(e, a.get("claims", [])) for e in key_ev]) if key_ev else np.nan,
                         "len": len(a.get("interpretation", ""))})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "scored.csv", index=False)
    g = R.groupby("cond").agg(시점=("id", "size"), 주장_평균=("n_claims", "mean"), 맞음=("ok", "sum"), 틀림=("bad", "sum"), 형식밖=("na", "sum"),
                              부호_틀림=("sign_bad", "sum"), 사건_재현율=("recall", "mean"), 글자수=("len", "mean"))
    g["정확도_엄격"] = g.맞음 / (g.맞음 + g.틀림)
    g["정확도_부호"] = 1 - g.부호_틀림 / (g.맞음 + g.틀림)
    print(g.round(3).to_string())
    rng = np.random.default_rng(0)
    for cond in "ABC":
        d = R[R.cond == cond]
        bs = [(lambda s: s.ok.sum() / max(s.ok.sum() + s.bad.sum(), 1))(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
        print(f"  {cond} 정확도 95% CI [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")


def _selfcheck() -> None:
    tr = {"w5": {"time": [f"10:{m:02d}" for m in range(0, 60, 5)] + [f"11:{m:02d}" for m in range(0, 60, 5)],
                 "px_pct": list(np.linspace(-1, 0, 24)), "oi_chg_pct": [0] * 24, "vol": [1] * 23 + [10], "perp_net": [-1] * 24,
                 "basis_bp": [-5] * 24, "liq_long_usd": [0] * 20 + [5e6, 0, 0, 0], "liq_short_usd": [0] * 24, "spot_net": [1] * 24,
                 "whale_imb": [.1] * 24, "retail_imb": [-.1] * 24}, "w1h": {"time": [f"{h:02d}:00" for h in range(24)], "px_pct": [0] * 24}}
    assert check({"series": "px", "span": "2h", "kind": "rise"}, tr) is True          # −1% → 0%
    assert check({"series": "px", "span": "2h", "kind": "fall"}, tr) is False
    assert check({"series": "oi", "span": "2h", "kind": "flat"}, tr) is True
    assert check({"series": "perp_net", "span": "2h", "kind": "fall"}, tr) is True    # 순매도
    assert check({"series": "vol", "span": "2h", "kind": "spike", "at": "11:55"}, tr) is True
    assert check({"series": "vol", "span": "2h", "kind": "spike", "at": "10:30"}, tr) is False
    assert check({"series": "liq_long", "span": "2h", "kind": "spike", "at": "11:40"}, tr) is True
    assert check({"series": "funding", "span": "2h", "kind": "rise"}, tr) is None
    assert covered({"series": "liq_long", "kind": "spike", "at": "11:40"}, [{"series": "liq_long", "kind": "spike", "at": "11:45"}])
    print("selfcheck OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selfcheck" in a:
        _selfcheck()
    elif a[:1] == ["build"]:
        build()
    elif a[:1] == ["eval"]:
        evaluate()
