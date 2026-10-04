"""호가(깊은 벽 dd_imb50) 매매화 실패 경로 겨냥 아이디어 4개 — 탐색(판정 아님) (2026-10-04).

1 코인 간 롱/숏(경로② 드리프트)  2 상태 지속 보유(경로③ 시점 어긋남)  3 역선택 회피 진입(경로①)
4 사용자 재량 체결 × 벽 상태. 전진 사전등록: docs/experiments/orderbook_strategy_ideas_fwd_prereg_20261004.md

데이터는 전부 «이미 본» 구간(ETH 09-15~10-04, 다코인 09-26~10-04)이라 여기 숫자는 탐색이다.
경계 계약: 결정 시각 s 의 신호는 s 초 끝 상태(호가)까지, 결과는 s+1+지연 초 mid 부터.

  서버(nice 19):  python scripts/research_orderbook_strategy_ideas_20261004.py --build   # 패널 → tmp/ob_ideas_20261004/
                  python scripts/research_orderbook_strategy_ideas_20261004.py --fills   # userTrades(서버에서만)
  로컬:           python scripts/research_orderbook_strategy_ideas_20261004.py            # 아이디어 1~4 탐색
                  python scripts/research_orderbook_strategy_ideas_20261004.py --blockcheck   # 판정 통계 vs 매매 크기
                  python scripts/research_orderbook_strategy_ideas_20261004.py --selftest
  전진 판정:      위 모든 명령에 --since 2026-10-04T00:00Z --end <판정일 00Z> (서버 --build·--fills 도 같은 인자)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/ob_ideas_20261004"
COINS = ("ETH", "BTC", "SOL", "XRP", "HYPE")
T = lambda s: int(pd.Timestamp(s).timestamp())  # noqa: E731
_arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None  # noqa: E731
END = T(_arg("--end") or "2026-10-04T00:00Z")           # 전진 판정: --since 2026-10-04T00:00Z --end <판정일>
SINCE = T(_arg("--since")) if _arg("--since") else None
HO3 = "research_deep_wall_ho3_multicoin_20261004.py"
TAKER_HALF, MAKER_HALF = 4.0, 0.705          # 편도 bp: 테이커 왕복 8 · 메이커(USDC peg 실측) 왕복 1.41


def ho3():
    """ho3 모듈(수정 금지)을 찾아 import. 서버에서는 OUT 에 복사본을 둔다(LAKE 경로만 다시 잡는다)."""
    for p in (Path(__file__).parent / HO3, OUT / HO3,
              Path("/home/kbj20/crypto-scalping/.claude/worktrees/trader-strategy-testing-de9fb4/scripts") / HO3):
        if p.exists():
            sys.path.insert(0, str(Path(__file__).parent))
            spec = importlib.util.spec_from_file_location("ho3", p); m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m); m.LAKE = ROOT / "data/lake/binance"
            return m
    raise FileNotFoundError(HO3)


# ── 서버 단계 ────────────────────────────────────────────────────────────────
def build_all():
    from concurrent.futures import ProcessPoolExecutor
    H = ho3(); B = H.B; OUT.mkdir(parents=True, exist_ok=True); meta = {}
    for coin in COINS:
        H.WIN[coin] = (SINCE or (T("2026-09-16T00:00Z") if coin == "ETH" else T("2026-09-27T10:00Z")), END)
        P = H.build(coin, trade_mode=True)
        meta[coin] = H.bin_size(P)
        files = H.hour_files(B.ROOT / "data/live/orderflow/bookticker" / f"{coin}USDT", H.WIN[coin][0] - H.WARM, END)
        with ProcessPoolExecutor(max_workers=6) as ex:
            bt = pd.concat(list(ex.map(B.bt_file, files)))
        P = P.join(bt[~bt.index.duplicated(keep="last")][["bt_qi_mean"]])
        cols = ["mid", "bt_spread_bp", "dd_imb50", "smin", "bmax", "dmid1800", "bt_qi_mean"]
        P[cols].to_parquet(OUT / f"panel_{coin}.parquet")
        print(coin, len(P), f"{P.dd_imb50.notna().mean():.0%} 가격칸 {meta[coin]}", flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta))


def fetch_fills():
    import asyncio
    import time
    import aiohttp
    import os
    from dotenv import load_dotenv
    sys.path.insert(0, str(Path(__file__).parent))
    import live_binance_account_20260910 as A
    load_dotenv(ROOT / ".env"); key, sec = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET_KEY", "")

    async def run():
        rows = []
        async with aiohttp.ClientSession() as s:
            off = await A._clock_offset(s)
            for sym in ("ETHUSDC", "ETHUSDT"):
                for d0 in range(T("2026-09-08T00:00Z"), int(time.time()), 86400):     # 하루 창 = 1000건 한도 안
                    r = await A._get(s, "/fapi/v1/userTrades", {"symbol": sym, "startTime": d0 * 1000,
                                     "endTime": d0 * 1000 + 86400_000 - 1, "limit": 1000}, key, sec, off)
                    if isinstance(r, dict):
                        raise RuntimeError(r)            # 못 읽음 ≠ 없음
                    assert len(r) < 1000, (sym, d0)
                    rows += r
                    await asyncio.sleep(1.5)
        return rows
    rows = asyncio.run(run())
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "fills.json").write_text(json.dumps(rows))
    print("fills", len(rows))


# ── 공용 ────────────────────────────────────────────────────────────────────
def load(coin):
    P = pd.read_parquet(OUT / f"panel_{coin}.parquet")
    P["mid"] = P.mid.ffill(limit=5)
    return P


def zrank(idx, x):
    """초마다 z = 2·분위 − 1. 분위 = 직전 288개 봉마감 x 중 x(t) 보다 작은 비율(그 봉마감 값 제외, ≥144개).
    자기 분포 중심화 — HYPE 처럼 imb50 이 늘 + 로 치우친 코인이 단면 순위를 독점하지 않게."""
    out = np.full(len(idx), np.nan); hist = deque(maxlen=288)
    bc = np.flatnonzero((idx % 300 == 0) & np.isfinite(x))
    for j, i in enumerate(bc):
        k = bc[j + 1] if j + 1 < len(bc) else len(idx)
        if len(hist) >= 144:
            h = np.sort(np.asarray(hist)); v = x[i:k]
            out[i:k] = np.where(np.isfinite(v), 2 * np.searchsorted(h, v) / len(h) - 1, np.nan)
        hist.append(x[i])
    return out


def day_stats(pnl_by_sec: pd.Series, lo: int, hi: int) -> dict:
    """초 단위 손익 → UTC 일 합계, 1h 블록 합계. 일 평균·SD·SE·t."""
    s = pnl_by_sec[(pnl_by_sec.index >= lo) & (pnl_by_sec.index < hi)]
    d = s.groupby(s.index // 86400).sum(); h = s.groupby(s.index // 3600).sum()
    m, sd = d.mean(), d.std(ddof=1)
    return dict(days=len(d), day_mean=m, day_sd=sd, day_se=sd / np.sqrt(len(d)), day_t=m / (sd / np.sqrt(len(d))),
                hour_t=h.mean() / (h.std(ddof=1) / np.sqrt(len(h))), daily={str(pd.to_datetime(k * 86400, unit="s").date()): round(v, 1) for k, v in d.items()})


def fmt(st):
    return (f"{st['days']}일 · 일 {st['day_mean']:+7.1f} ± {st['day_se']:5.1f} (t {st['day_t']:+.2f}, SD {st['day_sd']:.0f}) "
            f"· 1h블록 t {st['hour_t']:+.2f}")


def hyst(u, e_in=0.6, e_out=0.0, min_hold=0):
    """초마다 상태기계: 무포지션 → |u|≥e_in 이면 그 방향, 보유 중 → 부호·u ≤ −e_in 이면 반전, ≤ e_out 이면 청산.
    청산·반전은 진입 뒤 min_hold 초가 지나야 한다."""
    pos = np.zeros(len(u), np.int8); p = 0; t0 = 0
    for i, v in enumerate(u):
        if not np.isfinite(v):
            pos[i] = p; continue
        if p == 0:
            p = 1 if v >= e_in else -1 if v <= -e_in else 0; t0 = i
        elif i - t0 >= min_hold and p * v <= -e_in:
            p = -p; t0 = i
        elif i - t0 >= min_hold and p * v <= e_out:
            p = 0
        pos[i] = p
    return pos


def run_pos(lm, pos_dec, lag, half):
    """결정 위치 pos_dec[t](t 초 끝까지의 정보) → t+1+lag 초 mid 에 실행. 초 손익 bp, 거래 목록(진입·청산 인덱스)."""
    n = len(lm); k = 1 + lag
    ex = np.zeros(n); ex[k:] = pos_dec[:-k]                       # ex[t] = t 초 시작부터 들고 있는 위치
    r = np.diff(lm, append=np.nan) * 1e4; r[~np.isfinite(r)] = 0.0
    pnl = ex * r - np.abs(np.diff(ex, prepend=0.0)) * half
    ch = np.flatnonzero(np.diff(ex, prepend=0.0) != 0)
    trades = []
    for a, b in zip(ch, list(ch[1:]) + [n - 1]):
        if ex[a] != 0:
            path = ex[a] * (lm[a:b + 1] - lm[a]) * 1e4
            trades.append((a, b, ex[a], np.nanmin(path) if np.isfinite(path).any() else np.nan,
                           float(np.nansum(ex[a] * r[a:b]))))
    return pnl, trades


def trade_summary(tr, n_days, half):
    if not tr:
        return "거래 0"
    a = np.array([(t[1] - t[0], t[3], t[4]) for t in tr])
    return (f"거래 {len(tr)/n_days:.1f}/일 · 보유 중앙 {np.median(a[:,0]):.0f}초 · 총bp/건 {a[:,2].mean():+.2f} "
            f"· 순bp/건 {a[:,2].mean() - 2*half:+.2f} · MAE 중앙 {np.nanmedian(a[:,1]):+.1f}")


# ── 1 코인 간 롱/숏 ─────────────────────────────────────────────────────────
def grid(panels, lo, hi):
    idx = np.arange(lo, hi)
    lm = np.column_stack([np.log(p.mid.reindex(idx).ffill(limit=5).to_numpy()) for p in panels.values()])
    u = np.column_stack([pd.Series(zrank(p.index.to_numpy(), p.dd_imb50.to_numpy()), index=p.index).reindex(idx).to_numpy()
                         for p in panels.values()])
    return idx, lm, u


def ls_bars(idx, lm, u, every, lag, gap, half):
    """매 every 초(봉마감 배수) u 최댓값 코인 롱·최솟값 숏(명목 각 1), 다음 결정까지 보유. 다리 바뀔 때만 비용."""
    dec = np.flatnonzero(idx % every == 0)
    w_prev = np.zeros(lm.shape[1]); pnl = pd.Series(0.0, index=idx); per = []
    for j, i in enumerate(dec):
        a, b = i + 1 + lag, (dec[j + 1] + 1 + lag) if j + 1 < len(dec) else len(idx) - 1
        if b >= len(idx):
            break
        v = u[i]; ok = np.isfinite(v)
        w = np.zeros_like(w_prev)
        if ok.sum() >= 4:
            vv = np.where(ok, v, np.nan)
            L, S = np.nanargmax(vv), np.nanargmin(vv)
            if vv[L] >= gap and vv[S] <= -gap:          # gap 0 = 늘 · 0.6 = 둘 다 자기 분포 상·하위 20%
                w[L], w[S] = 1, -1
        cost = np.abs(w - w_prev).sum() * half
        seg = (lm[a:b + 1] - lm[a]) * 1e4
        g = np.nan_to_num(seg[-1]) @ w if w.any() else 0.0
        mae = np.nanmin(np.nan_to_num(seg) @ w) if w.any() else 0.0
        pnl.iloc[a] += g - cost
        if w.any():
            per.append((g, cost, mae, (w != w_prev).any()))
        w_prev = w
    return pnl, per


# ── 3 역선택 회피 메이커 진입(ho3.sim 복사 + 진입 게이트) ─────────────────────
def sim_g(P, x, bs, t0, gate=None, maker=0.0, taker=4.0):
    """ho3.sim 과 같은 규칙(봉마감 문턱 80분위·15분 최소 보유·매초 peg·«뚫고 지나감» 체결). gate(i, side) 가
    False 인 초에는 진입 주문을 걸지 않는다(걸려 있던 진입 주문도 그 초엔 빠진다). gate=None 이면 ho3.sim 과 동일."""
    idx = P.index.to_numpy(); mid = P.mid.ffill().to_numpy()
    hs = P.bt_spread_bp.to_numpy() / 2e4
    bid = P.mid.to_numpy() * (1 - hs); ask = P.mid.to_numpy() * (1 + hs)
    smin = P.smin.to_numpy(); bmax = P.bmax.to_numpy()
    hist: deque = deque(maxlen=288)
    pos, entry, t_fill, real, target, dec = 0, np.nan, -10**9, 0.0, 0, -10**9
    order = None; eq = np.zeros(len(idx)); fills = []; posted = 0
    for i, s in enumerate(idx):
        if s % 300 == 0 and np.isfinite(x[i]):
            if len(hist) >= 144 and s >= t0:
                thr = np.quantile(np.abs(hist), .8)
                target = 1 if x[i] >= thr else -1 if x[i] <= -thr else 0; dec = i
            hist.append(x[i])
        if order and order["kind"] == "enter" and order["side"] != target:
            order = None
        if order is None and target != pos and i >= dec + 2 and (pos == 0 or s - t_fill >= 900):
            order = dict(kind="exit", side=-pos, dl=i + 60) if pos else dict(kind="enter", side=target, dl=i + 300)
            posted += order["kind"] == "enter"
        if order and i >= 1:
            sd = order["side"]; p = bid[i - 1] if sd > 0 else ask[i - 1]
            live = order["kind"] == "exit" or gate is None or gate(i - 1, sd)
            hit = live and np.isfinite(p) and (((smin[i] + 1) * bs <= p) if sd > 0 else (bmax[i] * bs > p))
            if hit:
                fills.append((i, sd, p, True, order["kind"]))
                if order["kind"] == "exit":
                    real += pos * (p / entry - 1) * 1e4 - maker; pos = 0
                else:
                    pos, entry, t_fill = sd, p, s; real -= maker
                order = None
            elif i >= order["dl"]:
                if order["kind"] == "exit" and np.isfinite(bid[i]):
                    px = bid[i] if pos > 0 else ask[i]
                    fills.append((i, -pos, px, False, "exit"))
                    real += pos * (px / entry - 1) * 1e4 - taker; pos = 0
                    order = None
                elif order["kind"] == "enter":
                    order = None
        eq[i] = real + (pos * (mid[i] / entry - 1) * 1e4 if pos else 0.0)
    day = idx // 86400
    last = pd.Series(eq, index=day).groupby(level=0).last()
    daily = last.diff().fillna(last)
    adv = lambda fs: float(np.mean([f[1] * (mid[min(f[0] + 10, len(mid) - 1)] / f[2] - 1) * 1e4 for f in fs])) if fs else np.nan  # noqa: E731
    ent = [f for f in fills if f[3] and f[4] == "enter"]
    return dict(daily=daily, n=len(fills), n_maker=sum(f[3] for f in fills), n_taker=sum(not f[3] for f in fills),
                adv10=adv([f for f in fills if f[3]]), adv10_entry=adv(ent), n_entry=len(ent), posted=posted,
                adv_entry_day=pd.Series([f[1] * (mid[min(f[0] + 10, len(mid) - 1)] / f[2] - 1) * 1e4 for f in ent],
                                        index=[idx[f[0]] // 86400 for f in ent], dtype=float),
                eq=pd.Series(eq, index=idx))


def gates(P):
    hs = P.bt_spread_bp.to_numpy() / 2e4; m = P.mid.to_numpy()
    bid = m * (1 - hs); ask = m * (1 + hs); qi = P.bt_qi_mean.to_numpy()
    db = bid - np.roll(bid, 10); da = ask - np.roll(ask, 10)
    return {"없음": None,
            "튕김(10초 최우선가가 내게서 멂)": lambda i, sd: (db[i] > 0) if sd > 0 else (da[i] < 0),
            "QI 내 쪽": lambda i, sd: (qi[i] > 0) if sd > 0 else (qi[i] < 0)}


# ── 4 사용자 체결 ───────────────────────────────────────────────────────────
def user_fills(P, lo, hi):
    f = pd.DataFrame(json.loads((OUT / "fills.json").read_text()))
    f["t"] = f.time.astype("int64") // 1000; f["dir"] = np.where(f.side == "BUY", 1, -1)
    f["q"] = f.qty.astype(float) * f["dir"]
    f = f.sort_values("time")
    # 2초 안 같은 심볼·같은 방향 체결 = 한 주문
    f["grp"] = ((f.t.diff() > 2) | (f.dir != f.dir.shift()) | (f.symbol != f.symbol.shift())).cumsum()
    g = f.groupby("grp").agg(t=("t", "first"), dir=("dir", "first"), q=("q", "sum"), sym=("symbol", "first"),
                             rpnl=("realizedPnl", lambda s: s.astype(float).abs().sum()), ps=("positionSide", "first"))
    # 심볼(+헤지 측면)별 순포지션을 따라가 진입/추가(물타기)/청산 구분. 09-08 부터 접어 창 시작 위치를 잡는다.
    kinds = []; posd = {}
    for r in g.itertuples():
        k = (r.sym, r.ps); p0 = posd.get(k, 0.0); p1 = p0 + r.q
        kinds.append("청산" if abs(p1) < abs(p0) - 1e-9 else ("추가" if abs(p0) > 1e-9 else "진입"))
        posd[k] = 0.0 if abs(p1) < 1e-9 else p1
    g["kind"] = kinds
    g = g[(g.t >= lo) & (g.t < hi)]
    u = pd.Series(zrank(P.index.to_numpy(), P.dd_imb50.to_numpy()), index=P.index)
    lm = np.log(P.mid)
    g["u"] = u.reindex(g.t - 1).to_numpy()                       # 체결 초 직전 초까지의 벽 상태
    m0 = lm.reindex(g.t).to_numpy()
    for h in (900, 3600):
        g[f"d{h//60}"] = g.dir * (lm.reindex(g.t + h).to_numpy() - m0) * 1e4
    g["mae60"] = [np.nanmin(r.dir * (lm.loc[r.t:r.t + 3600].to_numpy() - lm.get(r.t, np.nan))) * 1e4 for r in g.itertuples()]
    g["state"] = np.where(g.dir * g.u >= .6, "일치", np.where(g.dir * g.u <= -.6, "반대", np.where(g.u.isna(), "결측", "중립")))
    g["day"] = g.t // 86400
    return g


def cl_mean(g, col):
    d = g.groupby("day")[col].mean().dropna()
    return (d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan, len(g[col].dropna()), len(d))


# ── 본 분석 ─────────────────────────────────────────────────────────────────
def main_analysis(lo_override=None):
    meta = json.loads((OUT / "meta.json").read_text())
    pan = {c: load(c) for c in COINS}
    res = {}
    lo_m = lo_override or T("2026-09-27T10:00Z"); hi = END
    lo_e = lo_override or T("2026-09-16T00:00Z"); recent = T("2026-10-01T00:00Z")
    print(f"창: ETH {pd.to_datetime(lo_e,unit='s')}~ · 다코인 {pd.to_datetime(lo_m,unit='s')}~ · 끝 {pd.to_datetime(END,unit='s')} — 전부 탐색(판정 아님)")

    # 1 ─────────────────────────
    print("\n== 1 코인 간 롱/숏 (5코인, z = 2·(직전 24h 봉마감 imb50 중 분위) − 1, 최댓값 롱·최솟값 숏)")
    hi_m = min(hi, max(int(p.index.max()) for p in pan.values()) - 2000)
    idx, lm, u = grid(pan, lo_m, hi_m)
    for sg, sl in ((1, "원"), (-1, "반대")):
        for every in (300, 900):
            for gap in (0.0, 0.6):
                gst = day_stats(ls_bars(idx, lm, sg * u, every, 0, gap, 0.0)[0], lo_m, hi_m)
                for name, half in (("테이커", TAKER_HALF), ("메이커", MAKER_HALF)):
                    pnl, per = ls_bars(idx, lm, sg * u, every, 0, gap, half)
                    st = day_stats(pnl, lo_m, hi_m); a = np.array(per) if per else np.zeros((0, 4))
                    rc = day_stats(pnl, recent, hi_m) if hi_m > recent + 86400 else None
                    print(f"  [{sl}] {every//60:2d}분 문턱{gap:.1f} [{name}] 보유봉 {len(a)/st['days']:.0f}/일 · 다리교체 {a[:,3].mean():.0%} "
                          f"· 총bp/봉 {a[:,0].mean():+.2f} · 비용/봉 {a[:,1].mean():.2f} · MAE 중앙 {np.median(a[:,2]):+.1f} "
                          f"| 총(비용 전) 일 {gst['day_mean']:+.1f} (t {gst['day_t']:+.2f}) | 순 {fmt(st)}"
                          + (f" | 10-01~ 순 일 {rc['day_mean']:+.0f}" if rc else ""))
                    res[f"ls_{sl}_{every}_{gap}_{name}"] = {**st, "gross_day": gst["day_mean"], "gross_t": gst["day_t"]}
        for lag in (30, 120):          # 지연 강건성(15분, gap0)
            g0 = day_stats(ls_bars(idx, lm, sg * u, 900, lag, 0.0, 0.0)[0], lo_m, hi_m)
            pnl, per = ls_bars(idx, lm, sg * u, 900, lag, 0.0, MAKER_HALF)
            print(f"  [{sl}] 15분 문턱0 지연 {lag}초 총 일 {g0['day_mean']:+.1f} (t {g0['day_t']:+.2f}) | [메이커] 순 {fmt(day_stats(pnl, lo_m, hi_m))}")

    # 2 ─────────────────────────
    print("\n== 2 상태 지속 보유 (z 초 단위: 진입 |z|≥0.6=상·하위20%, 청산 z가 0(중앙) 넘음, 반대 ≤−0.6 반전) · +5분 = 최소 보유 300초 · 대조 봉마감+15분")
    for coin in COINS:
        P = pan[coin]; lo = lo_e if coin == "ETH" else lo_m
        P = P[(P.index >= lo - 86400) & (P.index < hi)]
        idx_c = P.index.to_numpy(); lmc = np.log(P.mid.to_numpy())
        uc = zrank(idx_c, P.dd_imb50.to_numpy())
        pos = hyst(uc); pos5 = hyst(uc, min_hold=300)
        # 대조: 봉마감 결정 + 15분 최소 보유(ho3 매매 규칙의 방향만, 체결 모델 없이)
        base = np.zeros(len(uc), np.int8); p, tch = 0, -10**9
        for i, s in enumerate(idx_c):
            if s % 300 == 0 and np.isfinite(uc[i]):
                tg = 1 if uc[i] >= .6 else -1 if uc[i] <= -.6 else 0
                if tg != p and s - tch >= 900:
                    p, tch = tg, s
            base[i] = p
        for lab, pd_ in (("상태", pos), ("상태+5분", pos5), ("봉+15분", base), ("상태·반대", -pos), ("상태+5분·반대", -pos5), ("봉+15분·반대", -base)):
            gst = day_stats(pd.Series(run_pos(lmc, pd_.astype(float), 0, 0.0)[0], index=idx_c), lo, hi)
            print(f"  {coin:4s} {lab:12s} 총(비용 전, 지연0) 일 {gst['day_mean']:+.1f} (t {gst['day_t']:+.2f})")
            for lag in (0, 30, 120):
                for name, half in (("테이커", TAKER_HALF), ("메이커", MAKER_HALF)):
                    if name == "테이커" and lag != 0:
                        continue
                    pnl, tr = run_pos(lmc, pd_.astype(float), lag, half)
                    ps = pd.Series(pnl, index=idx_c); st = day_stats(ps, lo, hi)
                    print(f"  {coin:4s} {lab:12s} 지연{lag:3d} [{name}] {trade_summary([t for t in tr if idx_c[t[0]] >= lo], st['days'], half)} | {fmt(st)}")
                    res[f"st_{coin}_{lab}_{lag}_{name}"] = st
    # 2-LS: 롱숏 상태 지속
    print("  -- 롱숏 상태 지속: max z≥0.6 & min z≤−0.6 이면 짝 진입, 롱 z<0 또는 숏 z>0 이면 청산")
    pos = np.zeros(lm.shape, np.int8); cur = None
    for t in range(len(idx)):
        v = u[t]
        if cur is not None:
            if not (np.isfinite(v[cur[0]]) and np.isfinite(v[cur[1]])) or v[cur[0]] < 0 or v[cur[1]] > 0:
                cur = None
        if cur is None and np.isfinite(v).sum() >= 4:
            L, S = np.nanargmax(v), np.nanargmin(v)
            if v[L] >= .6 and v[S] <= -.6:
                cur = (L, S)
        if cur is not None:
            pos[t, cur[0]], pos[t, cur[1]] = 1, -1
    for sg, sl in ((1, "원"), (-1, "반대")):
      for lag in (0, 30, 120):
        for name, half in (("총", 0.0), ("테이커", TAKER_HALF), ("메이커", MAKER_HALF)):
            if name == "테이커" and lag != 0:
                continue
            tot = np.zeros(len(idx)); trs = []
            for c in range(lm.shape[1]):
                pc, tr = run_pos(lm[:, c], sg * pos[:, c].astype(float), lag, half); tot += pc; trs += tr
            st = day_stats(pd.Series(tot, index=idx), lo_m, hi_m)
            print(f"  [{sl}] LS 상태 지연{lag:3d} [{name}] 다리 {trade_summary(trs, st['days'], half)} | {fmt(st)}")
            res[f"lsst_{sl}_{lag}_{name}"] = st

    # 3 ─────────────────────────
    print("\n== 3 역선택 회피 메이커 진입 (ho3 매매 규칙 + 진입 게이트, 보수적 «뚫고 지나감» 체결)")
    advd = {}                                                      # (게이트) → 코인별 일 평균 진입 역선택
    for coin in COINS:
        P = pan[coin]; lo = lo_e if coin == "ETH" else lo_m
        P = P[(P.index >= lo - 86400) & (P.index < hi)]
        for (gname, gt), (sg, sl) in [(g, f) for f in ((1, "원"), (-1, "반대")) for g in gates(P).items()]:
            r = sim_g(P, sg * P.dd_imb50.to_numpy(), meta[coin], lo, gt)
            st = day_stats(r["eq"].diff().fillna(0.0), lo, hi)
            g0 = day_stats(sim_g(P, sg * P.dd_imb50.to_numpy(), meta[coin], lo, gt, taker=0.0)["eq"].diff().fillna(0.0), lo, hi)
            gname = f"[{sl}] {gname}"
            print(f"  {coin:4s} {gname:28s} 총(테이커 수수료 전) 일 {g0['day_mean']:+.1f} · 진입 시도 {r['posted']} · 진입 체결 {r['n_entry']} ({r['n_entry']/max(r['posted'],1):.0%}) "
                  f"· 진입 10초 역선택 {r['adv10_entry']:+.2f} · 전체 {r['adv10']:+.2f} · 테이커 청산 {r['n_taker']} | {fmt(st)}")
            if sl == "원":
                advd.setdefault(gname, []).append(r["adv_entry_day"].groupby(level=0).mean()[lambda v: v.index >= lo // 86400].rename(coin))
            res[f"g_{coin}_{gname}"] = {**st, "adv10_entry": r["adv10_entry"], "n_entry": r["n_entry"], "posted": r["posted"]}

    base = pd.concat(advd["[원] 없음"], axis=1)
    for gname, lst in advd.items():
        if gname == "[원] 없음":
            continue
        d = (pd.concat(lst, axis=1) - base)                      # 같은 코인·같은 날 짝 차이(+ = 역선택 줄어듦)
        for lab, v in (("ETH", d["ETH"].dropna()), ("5코인 일평균", d.mean(axis=1).dropna())):
            m, se, n = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v))
            print(f"  게이트 효과 {gname} − 없음 [{lab}] 진입 역선택 차 {m:+.2f} ± {se:.2f} bp/건 (t {m/se:+.2f}, {n}일)")

    # 4 ─────────────────────────
    if (OUT / "fills.json").exists():
        print("\n== 4 사용자 재량 체결 × ETH 벽 상태 (체결 직전 초 z, 일치 = 체결 방향 z≥0.6, 2초 안 같은 방향 체결 = 한 주문)")
        g = user_fills(pan["ETH"], lo_e, hi)
        g.to_csv(OUT / "user_fills_tagged.csv", index=False)
        print(f"  주문 {len(g)}개 · 심볼 {g.sym.value_counts().to_dict()} · 종류 {g.kind.value_counts().to_dict()}")
        for kind in ("진입", "추가", "진입+추가"):
            sub = g[g.kind.isin(kind.split("+"))]
            for stt in ("일치", "중립", "반대", "결측"):
                s = sub[sub.state == stt]
                if not len(s):
                    continue
                a, b, c = cl_mean(s, "d15"), cl_mean(s, "d60"), cl_mean(s, "mae60")
                print(f"  {kind:6s} {stt}: n {len(s):3d} ({a[3]}일) · 15분 {a[0]:+6.1f}±{a[1]:4.1f} · 60분 {b[0]:+6.1f}±{b[1]:4.1f} · MAE60 {c[0]:+6.1f}")
        sub = g[g.kind.isin(["진입", "추가"])]
        for h in ("d15", "d60"):
            a = sub[sub.state == "반대"].groupby("day")[h].mean(); b = sub[sub.state == "일치"].groupby("day")[h].mean()
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            print(f"  진입+추가 반대 − 일치 {h}: {a.mean() - b.mean():+.1f} ± {se:.1f} (t {(a.mean() - b.mean())/se:+.2f}, 일 {len(a)}/{len(b)})")
        res["user"] = g.groupby(["kind", "state"])[["d15", "d60", "mae60"]].agg(["mean", "count"]).round(2).to_string()
    (OUT / ("results.json" if lo_override is None else "results_fwd.json")).write_text(json.dumps(res, default=str, ensure_ascii=False, indent=1))


def blockcheck(lo_s="2026-09-26T19:40Z", hi_s="2026-10-03T00:00Z"):
    """HO3 판정 통계(1h 블록 · 양쪽 벽 각 ≥30초 블록만) vs 매매가 받는 풀링 스프레드. 블록 자격은 그 시각의
    «이후» 초까지 보고 정해진다(선택이 미래를 본다) — 자격·비자격 시간을 갈라 본다. 같은 창(HO3)."""
    H = ho3()
    for c in COINS:
        lo0 = SINCE or T(lo_s if c == "ETH" else "2026-09-26T10:00Z")
        P = load(c); P = P[(P.index >= lo0) & (P.index < (END if SINCE else T(hi_s)))].copy()
        P["fwd300"] = (np.log(P.mid.shift(-300)) - np.log(P.mid)) * 1e4
        r = P.dd_imb50.rank(pct=True); P["buy"] = r >= .8; P["sell"] = r <= .2; P["blk"] = P.index // 3600
        blk = H.spread(P, P.buy, P.sell, "fwd300")
        q = (P.groupby("blk").buy.sum() >= 30) & (P.groupby("blk").sell.sum() >= 30); inq = P.blk.isin(q[q].index)
        sp = lambda m: P.fwd300[P.buy & m].mean() - P.fwd300[P.sell & m].mean()  # noqa: E731
        print(f"  {c:4s} 블록 통계(원시) {blk[0]:+.2f}±{blk[1]:.2f} ({blk[2]}블록/{len(q)}) · 풀링 전체 {sp(P.index == P.index):+.2f} "
              f"· 자격 시간 안 {sp(inq):+.2f} · 비자격 {sp(~inq):+.2f} · 자격 시간 벽초 점유 {(P.buy|P.sell)[inq].sum()/(P.buy|P.sell).sum():.0%}")
    print("  -- 인과 풀링 한쪽 방향 수익(z = 직전 24h 분위, |z|≥0.6 인 초, s+1 부터 300초 mid, 일 평균) = 매매가 받는 크기")
    for c in COINS:
        P = load(c); lo = SINCE or T("2026-09-16T00:00Z" if c == "ETH" else "2026-09-27T10:00Z"); P = P[(P.index >= lo - 86400) & (P.index < END)]
        idx = P.index.to_numpy(); z = zrank(idx, P.dd_imb50.to_numpy()); lm = np.log(P.mid.to_numpy())
        f = np.full(len(lm), np.nan); f[:-301] = (lm[301:] - lm[1:-300]) * 1e4
        sg = np.where(z >= .6, 1, np.where(z <= -.6, -1, 0)); ok = np.isfinite(f) & (idx >= lo) & (sg != 0)
        d = pd.Series((sg * f)[ok]).groupby(idx[ok] // 86400).mean(); se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {c:4s} {d.mean():+.2f} ± {se:.2f} bp/5분 (t {d.mean()/se:+.2f}, {len(d)}일) · 10-01~ {d[d.index >= T('2026-10-01T00:00Z') // 86400].mean():+.2f}")


def selftest():
    # hyst: 진입 1, 청산 0.25, 반전 −1
    u = np.array([0, .7, .5, .1, -.1, -.5, -.7, -.3, .1, np.nan, .9])
    assert hyst(u).tolist() == [0, 1, 1, 1, 0, 0, -1, -1, 0, 0, 1], hyst(u)
    assert hyst(u, min_hold=4).tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 1], hyst(u, min_hold=4)
    zz = zrank(np.arange(0, 300 * 300, 1), np.tile(np.arange(300.0), 300))   # 봉마감 값 = 0 → 직전 값 전부 0
    assert np.nanmax(np.abs(zz[300 * 200: 300 * 200 + 5] - np.array([-1, 1, 1, 1, 1]))) < 1e-12
    # run_pos: 지연 0 이면 결정 t → t+1 부터 보유, 비용은 위치 변화마다
    lm = np.log(np.array([100, 100, 101, 102, 102, 102.0]))
    pnl, tr = run_pos(lm, np.array([1, 1, 0, 0, 0, 0.0]), 0, 1.0)
    # ex = [0,1,1,0,0,0] → 수익 1→2 초(100→101→102) 두 칸 보유, 비용 2
    assert abs(pnl.sum() - ((np.log(102) - np.log(100)) * 1e4 - 2)) < 1e-9 and len(tr) == 1, (pnl, tr)
    pnl2, _ = run_pos(lm, np.array([1, 1, 0, 0, 0, 0.0]), 1, 1.0)       # 한 초 늦으면 101→102 만
    assert abs(pnl2.sum() - ((np.log(102) - np.log(101)) * 1e4 - 2)) < 1e-9
    # ls_bars: 최댓값 롱·최솟값 숏, 다리 그대로면 비용 0
    idx = np.arange(0, 1200); lmm = np.zeros((1200, 5)); lmm[:, 0] = np.arange(1200) * 1e-6; lmm[:, 1] = -np.arange(1200) * 1e-6
    uu = np.tile([2.0, -2.0, 0, 0, 0], (1200, 1))
    pnl, per = ls_bars(idx, lmm, uu, 300, 0, 0.0, 1.0)
    assert len(per) == 4 and per[0][1] == 2.0 and per[1][1] == 0.0 and all(p[0] > 0 for p in per), per
    # sim_g(gate=None) == ho3.sim (복사본 검증)
    try:
        H = ho3()
    except FileNotFoundError:
        H = None
    n = 3 * 86400 // 2; ix = np.arange(n) + 86400 * 10; m = 100 + np.sin(np.arange(n) / 500) * 0.5
    rng = np.random.default_rng(1)
    P = pd.DataFrame({"mid": m, "bt_spread_bp": 1.0, "smin": np.where(rng.random(n) < .3, np.floor(m / 0.01) - 3, np.nan),
                      "bmax": np.where(rng.random(n) < .3, np.floor(m / 0.01) + 3, np.nan), "bt_qi_mean": rng.normal(size=n)}, index=ix)
    x = np.sin(np.arange(n) / 3000) + rng.normal(size=n) * .1
    r = sim_g(P, x, 0.01, ix[0])
    if H is not None:
        r0 = H.sim(P, x, 0.01, ix[0])
        assert np.allclose(r0["daily"].to_numpy(), r["daily"].to_numpy()) and r0["n"] == r["n"], (r0["n"], r["n"])
    rg = sim_g(P, x, 0.01, ix[0], lambda i, sd: False)                    # 진입 게이트 항상 닫힘 → 체결 0
    assert rg["n"] == 0
    print("selftest ok" + ("" if H else " (ho3 없음: sim 동일성 건너뜀)"))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--build" in sys.argv:
        build_all()
    elif "--blockcheck" in sys.argv:
        blockcheck()
    elif "--fills" in sys.argv:
        fetch_fills()
    else:
        main_analysis(SINCE)
