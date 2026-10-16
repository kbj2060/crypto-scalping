"""풋프린트 계열 «짧은 지평 불합격» 규칙을 긴 지평(4h·12h·1d·3d)으로 재검정 (2026-10-16).

사전 고정(결과 보기 전) -- 사건 정의는 원 스크립트 그대로, 보유 지평만 바꾼다:
  F1 교과서 imbalance 3종(execution-footprint-imbalance-044af8 의 research_eth_fp_imbalance_textbook_20261005.py)
     5분 × $0.5 칸, 행 0.2×ATR12, DASH(같은 줄 3배) / DIAG(대각 3배 + 칸 ≥ 봉 중앙값), stack ≥ 3줄.
     H1 주도→지속(lag 0) · H2 흡수→반전(lag 0) · H3 극값 소진→반전(다음 봉이 닫혀야 앎, lag 1)
     H1b 지지 재방문: 원판은 구간 끝 지정가 진입 → 여기선 «닿은 봉 j 종가» 진입(긴 지평이라 차이 작음, 명시적 변경).
  F2 표식·위치→사건→반응(agent-a1a6a08b0c163256e 의 research_eth_fp_{pattern_lookback,event_response}_20260930.py)
     L=12 · C=288. div·exh·abs(흡수는 지연 1봉) · eff(효율 흡수).
     A = 사건이 알려진 봉 종가에서 사건 방향(위치 조건 없음 = lookback 판) · B = 위치 있음 + 확인된 반전(15분 안) ·
     C = 위치 있음 + 극값 돌파 실패 → 돌파 방향. eff 는 위치 조건 없음(원판 «전체»). OI 변형·레짐 104칸은 뺀다.
  F3 레포트5 풋프린트/프로파일 규칙(order-flow-five-strategies-735495 의 research_eth_orderflow_report5_20261011.py, S3 위치 버그 수정판)
     S1 지지+stack→돌파 · S2a 델타·가격 일치→지속 · S3 흡수→이탈 진입 · S3 공격 돌파 유지 · S4 HVN 유지(롱)/거부(숏) ·
     S4 LVN→다음 HVN 방향 · S5a 수용 · S5b 거부→POC 방향. 문턱은 원판대로 2023~24 에서 고정. 대조(ev=0)는 안 쓴다(위약으로 대체).
  FD 일봉 재정의(명시적 «재정의», data/footprint_daily $1 칸 2020~): F1 과 같은 코드를 일봉에 -- 행 0.2×ATR12(일), 1d·3d 만.
  공통: 진입 = 사건이 알려진 봉(분)의 종가, 청산 = 진입 + H 시각의 1분 종가(일봉 판은 일 종가). 손절·익절 없음.
        비용 왕복 4bp. 주 = 2025-01-01~, 보조 = 그 전(F1·F3 2023~24, F2 2022~24, FD 2020~24).
        CI = 일 블록(3d 는 주 블록) 포아송 부트스트랩 B=1000. 위약 = 주 기간 안 같은 사건을 ≥7일 원형 이동(방향 유지) 500회.
        통과 = 주 순평균 CI 하한 > 0 · (사건 − 위약) 차 CI 하한 > 0 · 보조 평균 > 0.
        추세 상관 = 규칙 일 손익(보유 노출 × 시간 수익) vs «28일 수익 부호를 1일 보유» 일 손익.

실행: nice -n 10 env OMP_NUM_THREADS=3 python scripts/research_eth_longh_footprint_20261016.py [--selftest] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.market_ctx import session_vwap  # noqa: E402

ROOT = Path(subprocess.run(["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parent).stdout.strip()).resolve().parent
CELLS_DIR = ROOT / ".claude/worktrees/execution-footprint-imbalance-044af8/tmp/footprint_cells/ETHUSDT"
DAY = 86400
HS = {"4h": 4 * 3600, "12h": 12 * 3600, "1d": DAY, "3d": 3 * DAY}
COST = 4.0
MAIN0 = int(pd.Timestamp("2025-01-01").timestamp())
RATIO, STACK_MIN, ATR_N = 3.0, 3, 12
NB, NP = 1000, 500


# ================================================================ 데이터
def load_1m() -> pd.DataFrame:
    root = ROOT / "data/binance_vision"
    fs = sorted(glob.glob(str(root / "klines1m/ETHUSDT-1m-*.parquet"))) + sorted(glob.glob(str(root / "klines1m_daily/ETHUSDT-1m-*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=["t", "h", "l", "c", "v", "tb"]) for f in fs]).drop_duplicates("t").sort_values("t")
    d.index = d.pop("t") // 60_000                                            # 분 번호(시작)
    d = d.reindex(np.arange(d.index.min(), d.index.max() + 1))
    d["delta"] = 2 * d.tb - d.v
    return d


def bars5(m1: pd.DataFrame) -> pd.DataFrame:
    """1분 → 5분봉(인덱스 = 시작 epoch 초). 원판 load_bars/load_5m 합본: open = 직전 종가, top_buy = 고가를 처음 찍은 1분 테이커 매수."""
    x = m1.dropna(subset=["c"])
    b = x.index.to_numpy() * 60 // 300 * 300
    g = x.groupby(b, sort=True)
    o = pd.DataFrame({"high": g.h.max(), "low": g.l.min(), "close": g.c.last(), "v": g.v.sum(), "tb": g.tb.sum()})
    ih, il = g.h.idxmax(), g.l.idxmin()
    o["top_buy"] = x.tb.loc[ih].to_numpy()
    o["bot_sell"] = (x.v.loc[il] - x.tb.loc[il]).to_numpy()
    o = o.reindex(np.arange(o.index.min(), o.index.max() + 300, 300))
    return add_atr(o, 0.5).assign(delta=lambda z: 2 * z.tb - z.v).pipe(
        lambda z: z.assign(dn=z.delta / z.delta.abs().rolling(288, min_periods=200).mean().shift(1)))


def add_atr(o: pd.DataFrame, bucket: float) -> pd.DataFrame:
    if "open" not in o:
        o["open"] = o.close.shift(1)                                          # 1분봉 파일에 o 가 없다 → 직전 종가(원판 규약)
    pc = o.close.shift(1)
    tr = np.maximum(o.high - o.low, np.maximum((o.high - pc).abs(), (o.low - pc).abs()))
    o["atr"] = tr.rolling(ATR_N, min_periods=ATR_N).mean()
    o["rs"] = np.maximum(bucket, np.round(0.2 * o.atr / bucket) * bucket)
    return o


def load_cells() -> pd.DataFrame:
    fs = sorted(glob.glob(str(CELLS_DIR / "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).sort_values(["b", "k"]).reset_index(drop=True)


def load_daily():
    d = pd.read_parquet(ROOT / "data/footprint_daily/ETHUSDT_days.parquet")
    di = (pd.to_datetime(d.day).astype("int64") // 10**9).to_numpy()
    bars = add_atr(pd.DataFrame({"open": d.open.to_numpy(), "high": d.high.to_numpy(), "low": d.low.to_numpy(),
                                 "close": d.close.to_numpy()}, index=di), 1.0)
    bars = bars.reindex(np.arange(di.min(), di.max() + DAY, DAY))
    c = pd.read_parquet(ROOT / "data/footprint_daily/ETHUSDT_cells.parquet")
    cells = pd.DataFrame({"b": (pd.to_datetime(c.day).astype("int64") // 10**9).to_numpy(), "k": c.bin.to_numpy(),
                          "buy": c.buy.to_numpy(), "sell": c.sell.to_numpy()}).sort_values(["b", "k"], ignore_index=True)
    return bars, cells


# ================================================================ F1 / FD 교과서 imbalance (원판 bar_flags, 칸 크기만 인자)
def bar_flags(cells: pd.DataFrame, bars: pd.DataFrame, bucket: float) -> pd.DataFrame:
    rs = bars.rs.reindex(cells.b.to_numpy()).to_numpy()
    c = cells.assign(r=np.floor(cells.k.to_numpy() * bucket / rs + 1e-9), rs=rs).dropna(subset=["r"])
    c = c.groupby(["b", "r"], sort=True, as_index=False).agg(buy=("buy", "sum"), sell=("sell", "sum"), rs=("rs", "first"))
    b, r, bu, se = c.b.to_numpy(), c.r.to_numpy(), c.buy.to_numpy(), c.sell.to_numpy()
    same_prev = np.r_[False, (b[1:] == b[:-1]) & (r[1:] == r[:-1] + 1)]
    same_next = np.r_[same_prev[1:], False]
    sell_below = np.where(same_prev, np.r_[0.0, se[:-1]], 0.0)
    buy_above = np.where(same_next, np.r_[bu[1:], 0.0], 0.0)
    nz = pd.Series(np.r_[bu, se], index=np.r_[b, b])
    med = nz[nz > 0].groupby(level=0).median().reindex(b).to_numpy()
    flags = {"dash_b": (bu > 0) & (bu > RATIO * se), "dash_s": (se > 0) & (se > RATIO * bu),
             "diag_b": (bu > 0) & (bu > RATIO * sell_below) & (bu >= med),
             "diag_s": (se > 0) & (se > RATIO * buy_above) & (se >= med)}
    out = pd.DataFrame(index=bars.index)
    out["has"] = out.index.isin(np.unique(b))
    for name, f in flags.items():
        run_id = np.cumsum(~(f & np.r_[False, f[:-1]] & same_prev))
        R = pd.DataFrame({"b": b, "r": r, "run": run_id, "rs": c.rs.to_numpy()})[f]
        g = R.groupby("run").agg(b=("b", "first"), lo=("r", "min"), hi=("r", "max"), n=("r", "size"), rs=("rs", "first"))
        g = g[g.n >= STACK_MIN]
        g = g.sort_values(["b", "n", "lo"], ascending=[True, False, name.endswith("_b")]).drop_duplicates("b").set_index("b")
        out[f"{name}_stk"] = out.index.isin(g.index)
        out[f"{name}_zlo"] = (g.lo * g.rs).reindex(out.index)
        out[f"{name}_zhi"] = ((g.hi + 1) * g.rs).reindex(out.index)
        X = pd.DataFrame({"b": b, "f": f}).groupby("b").f
        ext = X.last() if name.endswith("_b") else X.first()
        out[f"{name}_ext"] = ext.reindex(out.index, fill_value=False).astype(bool)
    return out


def fam_textbook(o: pd.DataFrame, F: pd.DataFrame, tag: str, bar_sec: int, touch_n: int = 12) -> list:
    ts = o.index.to_numpy()
    op, h, lo, c, atr = (o[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr"))
    n = len(c)
    valid = np.isfinite(atr) & np.isfinite(op) & F.has.to_numpy()
    nxt_h, nxt_l = np.r_[h[1:], np.nan], np.r_[lo[1:], np.nan]
    ev = []

    def add(name, mask, sgn, lag):
        i = np.flatnonzero(mask & valid); i = i[i + lag < n]
        ev.append((name, ts[i + lag] + bar_sec, sgn[i]))
    for mode in ("dash", "diag"):
        M = mode.upper()
        sb, ss = F[f"{mode}_b_stk"].to_numpy(), F[f"{mode}_s_stk"].to_numpy()
        both = sb & ss
        zb_hi, zs_lo = F[f"{mode}_b_zhi"].to_numpy(), F[f"{mode}_s_zlo"].to_numpy()
        up, dn = c > op, c < op
        e1b = sb & ~both & up & (c > zb_hi); e1s = ss & ~both & dn & (c < zs_lo)
        add(f"{tag} {M} H1 주도→지속", e1b | e1s, np.where(e1b, 1, -1), 0)
        e2b = sb & ~both & (c <= op); e2s = ss & ~both & (c > op)
        add(f"{tag} {M} H2 흡수→반전", e2b | e2s, np.where(e2b, -1, 1), 0)
        eb, es = F[f"{mode}_b_ext"].to_numpy(), F[f"{mode}_s_ext"].to_numpy()
        top, bot = (nxt_h <= h) & ~(nxt_l >= lo), (nxt_l >= lo) & ~(nxt_h <= h)
        add(f"{tag} {M} H3 소진→반전", (top & eb) | (bot & es), np.where(top, -1, 1), 1)
        zt = np.where(e1b, zb_hi, np.where(e1s, zs_lo, np.nan))
        tj, sj = [], []
        for i in np.flatnonzero(np.isfinite(zt) & valid):
            s = 1 if e1b[i] else -1
            for j in range(i + 1, min(i + 1 + touch_n, n)):
                if (s > 0 and lo[j] <= zt[i]) or (s < 0 and h[j] >= zt[i]):
                    tj.append(j); sj.append(s); break
        ev.append((f"{tag} {M} H1b 지지 재방문", ts[np.array(tj, int)] + bar_sec, np.array(sj, int)))
    return ev


# ================================================================ F2 표식·반응 (원판 marks / eff_events / levels / respond, OI 뺌)
L2, C2, NEAR_BP, WIN_MIN, SWING_MIN = 12, 288, 10.0, 15, 15


def marks(df: pd.DataFrame, L: int, C: int) -> dict:
    h, lo, c, dl = (df[k].to_numpy(float) for k in ("high", "low", "close", "delta"))
    n = len(h)
    cs = np.nancumsum(np.nan_to_num(dl))
    win = np.lib.stride_tricks.sliding_window_view
    idx = np.arange(L, n)
    Wh, Wl = win(h, L)[:-1], win(lo, L)[:-1]
    ok = np.isfinite(Wh).all(1) & np.isfinite(Wl).all(1) & np.isfinite(h[idx]) & np.isfinite(lo[idx]) & np.isfinite(c[idx])
    jh = idx - L + np.argmax(np.where(np.isfinite(Wh), Wh, -np.inf), 1)
    jl = idx - L + np.argmin(np.where(np.isfinite(Wl), Wl, np.inf), 1)
    nh, nl = ok & (h[idx] > h[jh]), ok & (lo[idx] < lo[jl])
    rng = h[idx] - lo[idx]
    pos = np.where(rng > 0, (c[idx] - lo[idx]) / np.where(rng > 0, rng, 1), 0.5)
    q = lambda s, p: s.rolling(C, min_periods=C).quantile(p, interpolation="lower").to_numpy()   # noqa: E731
    bigD = q(df["delta"].abs(), 0.8)[idx]
    tbQ, bsQ = q(df["top_buy"], 0.9)[idx], q(df["bot_sell"], 0.9)[idx]
    d_i = dl[idx]
    nxt_h, nxt_l = np.append(h[idx[:-1] + 1], np.nan), np.append(lo[idx[:-1] + 1], np.nan)
    div_t = nh & (cs[idx] < cs[jh]); div_b = nl & (cs[idx] > cs[jl])
    exh_t = nh & (d_i > 0) & (d_i >= bigD) & (pos <= 0.4); exh_b = nl & (d_i < 0) & (-d_i >= bigD) & (pos >= 0.6)
    held_t, held_b = nh & (nxt_h <= h[idx]), nl & (nxt_l >= lo[idx])
    tb_i, bs_i = df["top_buy"].to_numpy(float)[idx], df["bot_sell"].to_numpy(float)[idx]
    abs_t, abs_b = held_t & (tb_i >= tbQ), held_b & (bs_i >= bsQ)

    def pack(t, b):
        s = np.zeros(n, np.int8); s[idx[t]] = -1; s[idx[b & ~t]] = 1
        return s
    return {"div": pack(div_t, div_b), "exh": pack(exh_t, exh_b), "abs": pack(abs_t, abs_b)}


def eff_events(df5: pd.DataFrame) -> np.ndarray:
    dl, v = df5["delta"], df5["v"]
    zs = lambda s: (s - s.rolling(C2, min_periods=C2).mean().shift(1)) / s.rolling(C2, min_periods=C2).std().shift(1)   # noqa: E731
    dz, vz = zs(dl), zs(v)
    eff = ((df5["close"] / df5["close"].shift(1) - 1) * 1e4).abs() / dl.abs().clip(lower=1e-9)
    thr = eff.rolling(C2, min_periods=C2).quantile(0.2, interpolation="lower")
    strong = (dz.abs() > 2) & (vz > 2)
    side = -np.sign(dl).fillna(0).astype(int).to_numpy()
    return np.where((strong & (eff <= thr)).to_numpy(), side, 0).astype(np.int8)


def value_area_daily(m1: pd.DataFrame) -> dict:
    tp = ((m1["h"] + m1["l"] + m1["c"]) / 3).to_numpy(); v = m1["v"].to_numpy()
    day = m1.index.to_numpy() // 1440
    cut = np.flatnonzero(np.diff(day)) + 1
    out = {}
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(day)]):
        p, w = tp[a:b], v[a:b]
        s = np.isfinite(p) & np.isfinite(w)
        if s.sum() < 60:
            continue
        p, w = p[s], w[s]
        bw = np.median(p) * 5e-4
        hist = np.bincount(np.floor((p - p.min()) / bw).astype(int), weights=w)
        poc = int(np.argmax(hist)); lo = hi = poc; tot = hist[poc]; target = 0.7 * hist.sum()
        while tot < target and (lo > 0 or hi < len(hist) - 1):
            up = hist[hi + 1] if hi < len(hist) - 1 else -1.0
            dn = hist[lo - 1] if lo > 0 else -1.0
            if up >= dn:
                hi += 1; tot += up
            else:
                lo -= 1; tot += dn
        out[int(day[a])] = (p.min() + (hi + 1) * bw, p.min() + lo * bw)
    return out


def levels(df5: pd.DataFrame, m1: pd.DataFrame):
    ts = df5.index.to_numpy()
    day = pd.Series(ts // DAY)
    dmax, dmin = df5["high"].groupby(day.to_numpy()).max(), df5["low"].groupby(day.to_numpy()).min()
    pdh, pdl = (day - 1).map(dmax).to_numpy(), (day - 1).map(dmin).to_numpy()
    va = value_area_daily(m1)
    vah = (day - 1).map({k: x[0] for k, x in va.items()}).to_numpy(float)
    val = (day - 1).map({k: x[1] for k, x in va.items()}).to_numpy(float)
    h24 = df5["high"].rolling(288, min_periods=200).max().shift(1).to_numpy()
    l24 = df5["low"].rolling(288, min_periods=200).min().shift(1).to_numpy()
    ok = df5["close"].notna().to_numpy()
    vw = np.full(len(df5), np.nan); vs = np.full(len(df5), np.nan)
    a, b = session_vwap(ts[ok].tolist(), df5["high"][ok].tolist(), df5["low"][ok].tolist(), df5["close"][ok].tolist(), df5["v"][ok].tolist())
    vw[ok] = [np.nan if x is None else x for x in a]; vs[ok] = [np.nan if x is None else x for x in b]
    vwu, vwd = pd.Series(vw + 2 * vs).shift(1).to_numpy(), pd.Series(vw - 2 * vs).shift(1).to_numpy()
    hi, lo = df5["high"].to_numpy(), df5["low"].to_numpy()

    def near(x, lv):
        good = np.isfinite(lv) & np.isfinite(x)
        return good & (np.abs(x / np.where(good, lv, 1) - 1) * 1e4 <= NEAR_BP)
    return (near(hi, pdh) | near(hi, vah) | near(hi, h24) | near(hi, vwu),
            near(lo, pdl) | near(lo, val) | near(lo, l24) | near(lo, vwd))


def respond(m1: pd.DataFrame, df5: pd.DataFrame, bars: np.ndarray, sides: np.ndarray, delays: np.ndarray):
    """원판 respond(OI 제외). kind 1 확인된 반전(사건 방향) · 2 실패(돌파 방향). ent = 진입 1분 번호(그 분 종가 진입)."""
    base = int(m1.index[0])
    t0 = df5.index.to_numpy() // 60
    c, h, lo, dl = (m1[k].to_numpy(float) for k in ("c", "h", "l", "delta"))
    hi5, lo5 = df5["high"].to_numpy(float), df5["low"].to_numpy(float)
    n = len(bars)
    kind = np.zeros(n, np.int8); ent = np.full(n, -1, np.int64); edir = np.zeros(n, np.int8)
    for q in range(n):
        k, s, dlb = int(bars[q]), int(sides[q]), int(delays[q])
        j0 = int(t0[k] + 5 * (1 + dlb) - base)
        if j0 - SWING_MIN < 0 or j0 + WIN_MIN >= len(c):
            continue
        pre_l, pre_h = lo[j0 - SWING_MIN:j0], h[j0 - SWING_MIN:j0]
        cc, dd = c[j0:j0 + WIN_MIN], dl[j0:j0 + WIN_MIN]
        if not (np.isfinite(cc).all() and np.isfinite(dd).all() and np.isfinite(pre_l).all() and np.isfinite(pre_h).all()):
            continue
        cd = np.cumsum(dd)
        if s < 0:
            rev, fail = (cc < pre_l.min()) & (cd < 0), cc > hi5[k]
        else:
            rev, fail = (cc > pre_h.max()) & (cd > 0), cc < lo5[k]
        ir = int(np.argmax(rev)) if rev.any() else WIN_MIN
        jf = int(np.argmax(fail)) if fail.any() else WIN_MIN
        if ir == WIN_MIN and jf == WIN_MIN:
            continue
        kind[q], ent[q], edir[q] = (1, j0 + ir + base, s) if ir < jf else (2, j0 + jf + base, -s)
    return kind, ent, edir


def fam_marks(o: pd.DataFrame, m1: pd.DataFrame) -> list:
    ts = o.index.to_numpy()
    near_res, near_sup = levels(o, m1)
    mk = marks(o, L2, C2)
    mk["eff"] = eff_events(o)
    DELAY = {"div": 0, "exh": 0, "abs": 1, "eff": 0}
    ev = []
    for name, s in mk.items():
        d = DELAY[name]
        k = np.flatnonzero(s); k = k[k + d < len(ts)]
        ev.append((f"F2 {name} A 교과서 즉시", ts[k + d] + 300, s[k].astype(int)))
        if name != "eff":
            k = k[np.where(s[k] < 0, near_res[k], near_sup[k])]
        kind, ent, edir = respond(m1, o, k, s[k], np.full(len(k), d))
        loc = "위치" if name != "eff" else "전체"
        for kk, lab in ((1, "B 확인된 반전"), (2, "C 실패→돌파")):
            m = kind == kk
            ev.append((f"F2 {name} {loc}·{lab}", (ent[m] + 1) * 60, edir[m].astype(int)))
    return ev


# ================================================================ F3 레포트5 (원판 Ctx·s1·s2a·s3·s4·s5, 사건만)
def day_profile(price: np.ndarray, vol: np.ndarray) -> dict:
    bw = max(0.5, round(np.median(price) * 5e-4 / 0.5) * 0.5)
    i = np.floor((price - price.min()) / bw).astype(int)
    h = np.bincount(i, weights=vol)
    poc = int(np.argmax(h)); lo = hi = poc; tot = h[poc]
    while tot < 0.7 * h.sum() and (lo > 0 or hi < len(h) - 1):
        up = h[hi + 1] if hi < len(h) - 1 else -1.0
        dn = h[lo - 1] if lo > 0 else -1.0
        if up >= dn:
            hi += 1; tot += up
        else:
            lo -= 1; tot += dn
    s = np.convolve(h, np.ones(5) / 5, mode="same")
    W = 5
    mx = pd.Series(s).rolling(2 * W + 1, center=True, min_periods=1).max().to_numpy()
    mn = pd.Series(s).rolling(2 * W + 1, center=True, min_periods=1).min().to_numpy()
    hv = np.flatnonzero((s == mx) & (s >= 1.2 * s.mean()))
    lv = np.flatnonzero((s == mn) & (s <= 0.6 * s.mean()))
    lv = lv[(lv > hv.min()) & (lv < hv.max())] if len(hv) else lv[:0]
    p = lambda j: price.min() + (np.asarray(j) + 0.5) * bw   # noqa: E731
    return {"poc": float(p(poc)), "vah": float(p(hi) + bw / 2), "val": float(p(lo) - bw / 2),
            "pdh": float(price.max()), "pdl": float(price.min()), "hvn": p(hv), "lvn": p(lv)}


def build_levels(cells: pd.DataFrame) -> dict:
    g = cells.assign(day=cells.b // DAY, v=cells.buy + cells.sell).groupby(["day", "k"], sort=True).v.sum().reset_index()
    prof = {int(d): day_profile(x.k.to_numpy() * 0.5, x.v.to_numpy()) for d, x in g.groupby("day")}
    return {d + 1: p for d, p in prof.items() if d + 1 in prof}                # 전일 프로파일(전일·당일 둘 다 있을 때만, 원판 규약)


def dedup(idx: np.ndarray, gap: int = 12) -> np.ndarray:
    keep, last = [], -10**9
    for i in np.sort(idx):
        if i - last >= gap:
            keep.append(i); last = i
    return np.array(keep, dtype=int)


class Ctx:
    def __init__(self, bars, F, lv, cells):
        self.f = {"stk_b": F.diag_b_stk.to_numpy(), "stk_s": F.diag_s_stk.to_numpy(),
                  "zlo_b": F.diag_b_zlo.to_numpy(), "zhi_b": F.diag_b_zhi.to_numpy(),
                  "zlo_s": F.diag_s_zlo.to_numpy(), "zhi_s": F.diag_s_zhi.to_numpy()}
        self.H, self.L, self.C, self.Op = (bars[k].to_numpy() for k in ("high", "low", "close", "open"))
        self.atr = bars.atr.to_numpy(); self.dn = bars.dn.to_numpy()
        self.ts = bars.index.to_numpy(); self.day = self.ts // DAY
        self.lv = lv
        self.cb = cells.b.to_numpy(); self.ck = cells.k.to_numpy() * 0.5; self.cv = (cells.buy + cells.sell).to_numpy()
        self.rhi = pd.Series(self.H).shift(12).rolling(276, min_periods=200).max().to_numpy()
        self.rlo = pd.Series(self.L).shift(12).rolling(276, min_periods=200).min().to_numpy()
        self.hi12 = pd.Series(self.H).rolling(12).max().to_numpy()
        self.lo12 = pd.Series(self.L).rolling(12).min().to_numpy()
        self.n = len(self.C)
        self.near_ = {s: self._near_all(s) for s in (+1, -1)}

    def _near_all(self, side):
        key = ("pdl", "val") if side > 0 else ("pdh", "vah")
        x = self.L if side > 0 else self.H
        ok = np.zeros(self.n, bool)
        lv = [np.array([self.lv.get(int(d), {}).get(k, np.nan) for d in self.day]) for k in key]
        for L in lv + [self.rlo if side > 0 else self.rhi]:
            ok |= np.abs(x - L) <= 0.5 * self.atr
        return ok

    def vol_beyond_share(self, i0, i1, level):
        a, b = np.searchsorted(self.cb, self.ts[i0]), np.searchsorted(self.cb, self.ts[i1], side="right")
        v, p = self.cv[a:b], self.ck[a:b]
        return float(v[p > level].sum() / v.sum()) if v.sum() > 0 else np.nan


def fam_report5(ctx: Ctx, th: dict) -> list:
    out: dict[str, list] = {}
    put = lambda tag, j, s: out.setdefault(tag, []).append((int(j), int(s)))   # noqa: E731
    # S1
    for sgn, s in ((+1, "b"), (-1, "s")):
        stk, zlo, zhi = ctx.f[f"stk_{s}"], ctx.f[f"zlo_{s}"], ctx.f[f"zhi_{s}"]
        other = ctx.f["stk_" + ("s" if s == "b" else "b")]
        for i in np.flatnonzero(ctx.near_[sgn] & stk):
            if i < 300 or i > ctx.n - 40 or other[i]:
                continue
            ref = ctx.hi12[i] if sgn > 0 else ctx.lo12[i]
            edge = zlo[i] if sgn > 0 else zhi[i]
            for j in range(i + 1, i + 7):
                if (sgn > 0 and ctx.L[j] < edge) or (sgn < 0 and ctx.H[j] > edge):
                    break
                if (sgn > 0 and ctx.C[j] > ref) or (sgn < 0 and ctx.C[j] < ref):
                    put("F3 S1 지지+stack→돌파", j, sgn); break
    # S2a 일치(A)
    d6 = pd.Series(ctx.dn).rolling(6).sum().to_numpy()
    r6 = np.full(ctx.n, np.nan); r6[6:] = (ctx.C[6:] / ctx.C[:-6] - 1) * 1e4
    for i in range(300, ctx.n - 30, 6):
        if not (np.isfinite(d6[i]) and np.isfinite(r6[i]) and r6[i] != 0 and np.isfinite(ctx.atr[i])):
            continue
        sgn = int(np.sign(r6[i]))
        if (d6[i] >= th["d6_hi"] or d6[i] <= th["d6_lo"]) and np.sign(d6[i]) == sgn:
            put("F3 S2a 델타·가격 일치→지속", i, sgn)
    # S3 흡수(위치 수정판) · 공격
    d3 = pd.Series(ctx.dn).rolling(3).sum().to_numpy()
    near3 = {s: ctx.near_[s] | np.r_[False, ctx.near_[s][:-1]] | np.r_[False, False, ctx.near_[s][:-2]] for s in (+1, -1)}
    for sgn in (+1, -1):
        th_big = th["d3_hi"] if sgn < 0 else -th["d3_lo"]
        for i in np.flatnonzero(near3[sgn]):
            if i < 300 or i > ctx.n - 40 or not np.isfinite(ctx.atr[i]):
                continue
            prog = -sgn * (ctx.C[i] - ctx.C[i - 3]) / ctx.atr[i]
            if not np.isfinite(prog) or prog > 0.25 or not (-sgn * d3[i] >= th_big):
                continue
            wh, wl = ctx.H[i - 2:i + 1].max(), ctx.L[i - 2:i + 1].min()
            for j in range(i + 1, i + 7):
                if (sgn < 0 and ctx.H[j] > wh) or (sgn > 0 and ctx.L[j] < wl):
                    break
                if (sgn < 0 and ctx.C[j] < wl) or (sgn > 0 and ctx.C[j] > wh):
                    put("F3 S3 흡수→이탈 진입", j, sgn); break
        s = "b" if sgn > 0 else "s"
        stk = ctx.f[f"stk_{s}"]
        prev = (pd.Series(ctx.H).shift(1).rolling(12).max() if sgn > 0 else pd.Series(ctx.L).shift(1).rolling(12).min()).to_numpy()
        brk = (ctx.C > prev) if sgn > 0 else (ctx.C < prev)
        hold = np.r_[(ctx.L[1:] >= prev[:-1]) if sgn > 0 else (ctx.H[1:] <= prev[:-1]), False]
        big = sgn * ctx.dn >= (th["dn_hi"] if sgn > 0 else -th["dn_lo"])
        for i in dedup(np.flatnonzero(brk & hold)):
            if i < 300 or i > ctx.n - 40 or not np.isfinite(ctx.atr[i + 1]):
                continue
            if big[i] and stk[i]:
                put("F3 S3 공격 돌파 유지→지속", i + 1, sgn)
    # S4
    for i in range(300, ctx.n - 40):
        d = ctx.lv.get(int(ctx.day[i]))
        if d is None or not np.isfinite(ctx.atr[i]) or not np.isfinite(ctx.dn[i]):
            continue
        if abs(ctx.dn[i]) >= th["dn_hi_abs"]:
            sgn = int(np.sign(ctx.dn[i]))
            for L in d["lvn"]:
                if not ((ctx.Op[i] < L < ctx.C[i]) if sgn > 0 else (ctx.Op[i] > L > ctx.C[i])):
                    continue
                if len(d["hvn"][d["hvn"] > ctx.C[i]] if sgn > 0 else d["hvn"][d["hvn"] < ctx.C[i]]):
                    put("F3 S4 LVN 통과→다음 HVN", i, sgn); break
        if ctx.dn[i] < th["dn_hi_abs"]:
            continue
        for Hn in d["hvn"]:
            if not (ctx.H[i] >= Hn - 0.3 * ctx.atr[i] and ctx.C[i - 1] < Hn):
                continue
            j = i + 3
            if ctx.C[j] > Hn:
                put("F3 S4 HVN 위 유지→롱", j, 1)
            elif ctx.C[j] < Hn - 0.2 * ctx.atr[j]:
                put("F3 S4 HVN 거부→숏", j, -1)
            break
    # S5
    for sgn, key in ((+1, "vah"), (-1, "val")):
        seen = set()
        for i in range(300, ctx.n - 40):
            dd = int(ctx.day[i]); d = ctx.lv.get(dd)
            if d is None or (dd, sgn) in seen or not np.isfinite(ctx.atr[i]):
                continue
            L = d[key]
            if not ((ctx.C[i] > L >= ctx.C[i - 1]) if sgn > 0 else (ctx.C[i] < L <= ctx.C[i - 1])):
                continue
            seen.add((dd, sgn))
            s = i + 6
            beyond = sgn * (ctx.C[i + 1:s + 1] - L) > 0
            va = ctx.vol_beyond_share(i + 1, s, L)
            share = va if sgn > 0 else 1 - va
            still = sgn * (ctx.C[s] - L) > 0
            if still and beyond.sum() >= 4 and share >= 0.5:
                put("F3 S5a 수용→바깥 지속", s, sgn)
            if not still:
                put("F3 S5b 거부→POC 방향", s, -sgn)
    return [(tag, ctx.ts[np.array([a for a, _ in v])] + 300, np.array([b for _, b in v])) for tag, v in out.items()]


def report5_thresholds(o: pd.DataFrame) -> dict:
    aux = (o.index >= int(pd.Timestamp("2023-01-01").timestamp())) & (o.index < MAIN0)
    f = lambda x: x[np.isfinite(x)]   # noqa: E731
    dn = f(o.dn.to_numpy()[aux]); d3 = f(o.dn.rolling(3).sum().to_numpy()[aux]); d6 = f(o.dn.rolling(6).sum().to_numpy()[aux])
    return {"dn_hi": np.percentile(dn, 80), "dn_lo": np.percentile(dn, 20), "dn_med_abs": np.percentile(np.abs(dn), 50),
            "dn_hi_abs": np.percentile(np.abs(dn), 80), "d3_hi": np.percentile(d3, 80), "d3_lo": np.percentile(d3, 20),
            "d3_med": np.percentile(np.abs(d3), 50), "d6_hi": np.percentile(d6, 80), "d6_lo": np.percentile(d6, 20)}


# ================================================================ 평가
def price_fn(close: np.ndarray, base: int, step: int):
    """P(t) = t 시각에 끝난 봉(step 초)의 종가. 격자 밖·결측 = NaN."""
    def P(t):
        i = np.asarray(t, np.int64) // step - 1 - base
        ok = (i >= 0) & (i < len(close))
        out = np.full(len(i), np.nan); out[ok] = close[i[ok]]
        return out
    return P


def boot_means(val, blk, B, seed):
    u, inv = np.unique(blk, return_inverse=True)
    s, n = np.bincount(inv, weights=val), np.bincount(inv)
    W = np.random.default_rng(seed).poisson(1.0, (B, len(u))).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ s) / (W @ n)


def circ_shift(t, lo, span, grid, rng):
    u = int(rng.integers(7 * DAY // grid, (span - 7 * DAY) // grid)) * grid
    return lo + ((t - lo + u) % span) // grid * grid


def nonoverlap(te, H):
    keep, last = [], -10**18
    for k in np.argsort(te, kind="stable"):
        if te[k] >= last + H:
            keep.append(k); last = te[k]
    return np.array(keep, int)


class Trend:
    """28일 수익 부호 1일 보유 일 손익 + 규칙 노출(시간 격자) 일 손익."""
    def __init__(self, m1: pd.DataFrame):
        c = m1.c.to_numpy(); base = int(m1.index[0])
        self.h0 = (base + 59) // 60                                          # 첫 온전한 시
        hc = c[self.h0 * 60 - base + 59::60]                                  # 시 h 의 끝 종가
        self.hret = np.r_[np.nan, hc[1:] / hc[:-1] - 1]                       # 시 h 수익(h−1 끝 → h 끝)
        hours = self.h0 + np.arange(len(hc))
        self.hday = hours // 24
        dc = pd.Series(hc).groupby(self.hday).last()
        s28 = np.sign(dc / dc.shift(28) - 1)
        self.base = (s28.shift(1) * (dc / dc.shift(1) - 1))                    # 일 손익(일 인덱스)

    def corr(self, te, sgn, H):
        e = np.zeros(len(self.hret) + 1)
        a = np.clip(-(-te // 3600) - self.h0, 0, len(e) - 1); b = np.clip(-(-(te + H) // 3600) - self.h0, 0, len(e) - 1)
        np.add.at(e, a, sgn); np.add.at(e, b, -sgn)
        pos = np.cumsum(e)[:-1]                                              # 시 h 동안 보유 = 시작 ≤ h 시작 < 끝
        pnl = pd.Series(np.nan_to_num(pos * self.hret)).groupby(self.hday).sum()
        d = pd.concat([pnl, self.base], axis=1, keys=["r", "b"]).dropna()
        d = d[d.index >= MAIN0 // DAY]
        return float(d.r.corr(d.b)) if d.r.std() > 0 else np.nan


def evaluate(te, sgn, H, P, grid, end, rng, trend) -> dict:
    te = np.asarray(te, np.int64); sgn = np.asarray(sgn, float)
    g = sgn * (P(te + H) / P(te) - 1) * 1e4
    ok = np.isfinite(g) & (te + H <= end)
    te, sgn, g = te[ok], sgn[ok], g[ok]
    net = g - COST
    blk = te // (7 * DAY if H >= 3 * DAY else DAY)
    main, aux = te >= MAIN0, te < MAIN0
    yrs = pd.to_datetime(te, unit="s").year
    r = {"n_main": int(main.sum()), "n_aux": int(aux.sum()),
         "gross_main": float(g[main].mean()) if main.any() else np.nan,
         "net_main": float(net[main].mean()) if main.any() else np.nan,
         "net_aux": float(net[aux].mean()) if aux.any() else np.nan,
         "years": {int(y): round(float(net[yrs == y].mean()), 1) for y in np.unique(yrs)}}
    if main.sum() < 30:
        return r
    bm = boot_means(net[main], blk[main], NB, 7)
    r["ci_main"] = [float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5))]
    tm, sm = te[main], sgn[main]
    span = end - H - MAIN0
    pl = np.empty(NP)
    for k in range(NP):
        t2 = circ_shift(tm, MAIN0, span, grid, rng)
        pl[k] = np.nanmean(sm * (P(t2 + H) / P(t2) - 1) * 1e4)
    r["placebo_gross"] = float(pl.mean())
    dif = bm[:NP] - (pl - COST)
    r["diff"] = float(r["net_main"] - (pl.mean() - COST))
    r["ci_diff"] = [float(np.nanpercentile(dif, 2.5)), float(np.nanpercentile(dif, 97.5))]
    if aux.sum() >= 30:
        ba = boot_means(net[aux], blk[aux], NB, 8)
        r["ci_aux"] = [float(np.nanpercentile(ba, 2.5)), float(np.nanpercentile(ba, 97.5))]
    ko = nonoverlap(tm, H)
    r["n_main_nonover"], r["net_main_nonover"] = int(len(ko)), float(net[main][ko].mean())
    ka = nonoverlap(te[aux], H)
    r["n_aux_nonover"] = int(len(ka))
    r["trend_corr"] = trend.corr(tm, sm, H)
    r["long_share"] = float((sm > 0).mean())
    r["PASS"] = bool(r["ci_main"][0] > 0 and r["ci_diff"][0] > 0 and r.get("n_aux", 0) >= 30 and r["net_aux"] > 0)
    return r


# ================================================================ 실행
def all_events(m1, o, cells, dbars, dcells, th=None):
    """(이름, 진입 시각 te 초, 방향) 목록과 가족 태그. th 를 주면 그 문턱(셀프테스트용)."""
    c0, c1 = int(cells.b.min()), int(cells.b.max()) + 300
    o3 = o[(o.index >= c0) & (o.index < c1)]
    F = bar_flags(cells, o3, 0.5)
    ev = [("5m", *e) for e in fam_textbook(o3, F, "F1", 300)]
    ev += [("1m", *e) for e in fam_marks(o, m1)]
    ev += [("5m", *e) for e in fam_report5(Ctx(o3, F, build_levels(cells), cells), th or report5_thresholds(o))]
    if dbars is not None:
        ev += [("1d", *e) for e in fam_textbook(dbars, bar_flags(dcells, dbars, 1.0), "FD 일봉재정의", DAY)]
    return ev


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    m1 = load_1m(); o = bars5(m1); cells = load_cells(); dbars, dcells = load_daily()
    print("1m", m1.shape, "5m", o.shape, "cells", cells.shape, "days", dbars.shape, flush=True)
    ev = all_events(m1, o, cells, dbars, dcells)
    pd.DataFrame([(f, n, t, s) for f, n, T, S in ev for t, s in zip(T, S)], columns=["grid", "rule", "te", "sgn"]).to_parquet(out / "events.parquet")
    Pm = price_fn(m1.c.to_numpy(), int(m1.index[0]), 60)
    Pd = price_fn(dbars.close.to_numpy(), int(dbars.index[0]) // DAY, DAY)
    trend = Trend(m1)
    end_m = (int(m1.index[-1]) + 1) * 60
    end_c = int(cells.b.max()) + 300
    end_d = int(dbars.index[-1]) + DAY
    rng = np.random.default_rng(20261016)
    rows = []
    for grid, name, te, sg in ev:
        for hn, H in HS.items():
            if grid == "1d" and H < DAY:
                continue
            P, gstep, end = (Pd, DAY, end_d) if grid == "1d" else (Pm, 60 if grid == "1m" else 300, end_m if grid == "1m" else end_c)
            r = evaluate(te, sg, H, P, gstep, end, rng, trend)
            rows.append({"rule": name, "H": hn, **r})
            print(name, hn, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "years"}, flush=True)
    res = pd.DataFrame(rows)
    res.to_json(out / "results.json", orient="records", force_ascii=False, indent=1)
    n_cells = int(res.ci_main.notna().sum())
    summ = {"cells": len(res), "cells_evaluated": n_cells, "pass": int(res.PASS.fillna(False).sum()),
            "pass_list": res[res.PASS.fillna(False).astype(bool)][["rule", "H"]].values.tolist(),
            "main_ci_lo_pos": int(res.ci_main.dropna().map(lambda x: x[0] > 0).sum()),
            "diff_ci_lo_pos": int(res.ci_diff.dropna().map(lambda x: x[0] > 0).sum()),
            "main_ci_hi_neg": int(res.ci_main.dropna().map(lambda x: x[1] < 0).sum())}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))


def selftest():
    # 1) 원판 bar_flags 단언
    bars = pd.DataFrame({"rs": [0.5]}, index=[0])
    cells = pd.DataFrame({"b": [0] * 5, "k": [0, 1, 2, 3, 4], "buy": [1.0, 10, 10, 10, 10], "sell": [5.0, 1, 1, 1, 4]})
    F = bar_flags(cells, bars, 0.5)
    assert F.dash_b_stk[0] and F.dash_b_zlo[0] == 0.5 and F.dash_b_zhi[0] == 2.0
    assert not F.dash_s_stk[0] and not F.dash_b_ext[0] and F.dash_s_ext[0]
    assert F.diag_b_stk[0] and F.diag_b_zlo[0] == 1.0 and F.diag_b_ext[0]
    # 2) 가격 함수·원형 이동·겹침 제거
    P = price_fn(np.array([100.0, 101, 102, 103]), 10, 60)                  # 분 10..13 의 종가
    assert np.allclose(P(np.array([11 * 60, 14 * 60])), [100, 103]) and np.isnan(P(np.array([10 * 60]))[0])
    t = np.arange(0, 100 * DAY, 3600); t2 = circ_shift(t, 0, 100 * DAY, 300, np.random.default_rng(0))
    assert len(t2) == len(t) and t2.min() >= 0 and t2.max() < 100 * DAY and (t2 % 300 == 0).all() and not np.array_equal(t, t2)
    assert nonoverlap(np.array([0, 10, 20, 45]), 20).tolist() == [0, 2, 3]
    # 3) 미래참조 없음: 실데이터 조각에서 T 이후 데이터를 바꿔도 te ≤ T 사건은 그대로
    m1 = load_1m()
    m1 = m1[(m1.index >= int(pd.Timestamp("2025-02-20").timestamp()) // 60) & (m1.index < int(pd.Timestamp("2025-03-12").timestamp()) // 60)].copy()
    cells = load_cells(); cells = cells[(cells.b >= m1.index[0] * 60) & (cells.b < (m1.index[-1] + 1) * 60)].reset_index(drop=True)
    dbars, dcells = load_daily()
    T = int(pd.Timestamp("2025-03-06").timestamp())
    o = bars5(m1)
    th = dict(zip(["dn_hi", "dn_lo", "dn_med_abs", "dn_hi_abs", "d3_hi", "d3_lo", "d3_med", "d6_hi", "d6_lo"],
                  [1.0, -1.0, 0.5, 1.2, 2.0, -2.0, 1.0, 3.0, -3.0]))           # 조각엔 2023~24 가 없어 고정값
    base = all_events(m1, o, cells, dbars, dcells, th)
    rng = np.random.default_rng(1)
    m2 = m1.copy(); fut = m2.index.to_numpy() * 60 >= T
    k = rng.uniform(0.8, 1.25, fut.sum())
    for col in ("h", "l", "c"):
        m2.loc[fut, col] = m2.loc[fut, col].to_numpy() * k
    m2.loc[fut, "tb"] = m2.loc[fut, "v"].to_numpy() * rng.uniform(0, 1, fut.sum()); m2["delta"] = 2 * m2.tb - m2.v
    c2 = cells.copy(); cf = c2.b.to_numpy() >= T
    c2.loc[cf, "buy"] *= rng.uniform(0, 5, cf.sum()); c2.loc[cf, "k"] += rng.integers(-40, 40, cf.sum()).astype(np.int32)
    c2 = c2.sort_values(["b", "k"]).groupby(["b", "k"], as_index=False).sum()
    d2 = dbars.copy(); df_ = d2.index.to_numpy() >= T
    d2.loc[df_, ["high", "low", "close", "open"]] *= 1.3; d2 = add_atr(d2.drop(columns=["atr", "rs"]), 1.0)
    dc2 = dcells.copy(); dcf = dc2.b.to_numpy() >= T; dc2.loc[dcf, "sell"] *= 7
    pert = all_events(m2, bars5(m2), c2, d2, dc2, th)
    nchk, ndiff = 0, 0
    for (g1, n1, t1, s1_), (g2, n2, t2_, s2_) in zip(base, pert):
        assert n1 == n2
        a = sorted((int(x), int(y)) for x, y in zip(t1, s1_) if x <= T)
        b = sorted((int(x), int(y)) for x, y in zip(t2_, s2_) if x <= T)
        assert a == b, (n1, len(a), len(b), set(a) ^ set(b))
        nchk += len(a)
        ndiff += sorted(zip(t1.tolist(), s1_.tolist())) != sorted(zip(t2_.tolist(), s2_.tolist()))
    assert nchk > 500 and ndiff > 10, (nchk, ndiff)                           # 교란이 T 뒤 사건은 실제로 바꿨다
    print(f"selftest ok (te≤T 사건 {nchk}건 동일 · T 뒤가 바뀐 규칙 {ndiff}/{len(base)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("tmp/longh_footprint_20261016"))
    a = ap.parse_args()
    selftest() if a.selftest else run(a.out)
