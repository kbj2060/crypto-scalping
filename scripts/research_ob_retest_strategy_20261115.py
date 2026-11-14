#!/usr/bin/env python3
"""15분 오더블록 재진입 지정가 전략 (2026-11-15). 사전등록 docs/prereg/ob_retest_strategy_20261115.md (d9db54cb, 계산 전).

  selftest          손계산 1분봉으로 관통 체결·체결 분 손절·같은 분 둘 다·익절 관통·시간 정리·취소 단언
  dev               개발 2022~25 48설정 · S* · CPCV·PBO·DSR · 대조 A(옆 띠)·B(기하 위약 20벌) · 양성 대조 · 절단 재계산
  lock <cfg> <cost> 관문 통과 설정만 2026 1회 (cost = strict | lenient)
  ledger <trips>    실원장 존 안 정렬 검정
신호·OB·위약은 의미 검증 스크립트(research_smc_ict_meaning_20261115) 함수를 그대로 쓴다. 입력 SCRATCH_K = 최근 Vision 일 파일.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
import research_keltner_squeeze_20261115 as KS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/ob_retest_strategy_20261115"
COINS = SM.COINS
MS15 = 15 * 60_000
DEV = ("2022-01-01", "2026-01-01")
LOCK = ("2026-01-01", "2026-11-14")
MSD = lambda s: int(pd.Timestamp(s, tz="UTC").value // 10**6)
W_PEND = 100
GRID = list(itertools.product((5, 50), (0.5, 1.0, 2.0), (0.0, 0.5), (16, 96), (0, 1)))
NAME = lambda g: f"N{g[0]}_tp{g[1]}_sl{g[2]}_H{g[3]}_tr{g[4]}"
COST = {"strict": (2.0, 2.0, 6.0), "lenient": (0.0, 0.0, 6.0)}   # (진입 메이커, 익절 메이커, 시장가 편도)
SEED = 20261115
R2 = os.environ.get("OB_ROUND") == "2"   # 2라운드 손익비 재설계: docs/prereg/ob_retest_rr_20261115.md (42796489)
if R2:
    OUT = ROOT / "tmp/ob_retest_rr_20261115"
    GRID = list(itertools.product((5, 50), ("mid", "a50", "a25"), ("2R", "3R", "leg"), (16, 96), (0, 1)))
    NAME = lambda g: f"N{g[0]}_sl{g[1]}_tp{g[2]}_H{g[3]}_tr{g[4]}"


def levels(d, btm, top, atr, leg, g, r2=None):
    """지정가·존 먼 쪽·손절·익절. 1라운드 g=(N, 익절 ATR, 손절 여유 ATR, H, 추세) · 2라운드 g=(N, 손절 방식, 익절 방식, H, 추세)."""
    r2 = R2 if r2 is None else r2
    lim = np.where(d > 0, top, btm); far = np.where(d > 0, btm, top)
    a, b = g[1], g[2]
    if not r2:
        return lim, far, far - d * b * atr, lim + d * a * atr
    sl = {"mid": (btm + top) / 2, "a50": lim - d * 0.5 * atr, "a25": lim - d * 0.25 * atr}[a]
    r = np.abs(lim - sl)
    return lim, far, sl, {"2R": lim + d * 2 * r, "3R": lim + d * 3 * r, "leg": leg}[b]


@njit(cache=True)
def leg_ext(h, l, T, d, btm, top):
    """돌파 다리 끝값: OB 봉(가장 최근 [저가,고가] 일치 봉) ~ 돌파 봉 T 의 최고가(롱)/최저가(숏). T 에 알려짐."""
    out = np.empty(len(T))
    for k in range(len(T)):
        t = T[k]; j = t - 1
        while j > 0 and not (l[j] == btm[k] and h[j] == top[k]):
            j -= 1
        out[k] = h[j:t + 1].max() if d[k] > 0 else l[j:t + 1].min()
    return out


# ── 1분 체결 시뮬레이터 ─────────────────────────────────────────────────────
@njit(cache=True)
def sim(t1, h1, l1, c1, m0, sd, lim, far, sl, tp, exp_t, H, leak=0):
    """코인 하나. 신호 p 는 분 m0[p] 부터 유효. 대기 주문은 새 신호로 교체 · 보유 중 신호는 버림.
    반환 행 = [신호, 체결 분, 청산 분, 방향, 진입가, 청산가, 종류(1 손절·2 익절·3 시간)]."""
    n, ns = len(t1), len(m0)
    out = np.empty((ns, 7)); k = 0
    if ns == 0:
        return out[:0]
    p, pend, pos, mf, tex = 0, -1, -1, -1, 0
    m = m0[0]
    while m < n:
        while p < ns and m0[p] <= m:
            if pos < 0:
                pend = p
            p += 1
        if pos >= 0:
            j = pos; d = sd[j]
            if m > mf:
                typ, px = 0, 0.0
                if (d > 0 and l1[m] <= sl[j]) or (d < 0 and h1[m] >= sl[j]):
                    typ, px = 1, min(sl[j], h1[m]) if d > 0 else max(sl[j], l1[m])   # 분 전체가 손절가 너머면 그 분 끝값으로
                elif (d > 0 and h1[m] > tp[j]) or (d < 0 and l1[m] < tp[j]):
                    typ, px = 2, tp[j]
                elif t1[m] >= tex:
                    typ, px = 3, c1[m]
                if typ > 0:
                    out[k, 0] = j; out[k, 1] = mf; out[k, 2] = m; out[k, 3] = d; out[k, 4] = lim[j]; out[k, 5] = px; out[k, 6] = typ
                    k += 1; pos = -1
        elif pend >= 0:
            j = pend; d = sd[j]
            if t1[m] >= exp_t[j]:
                pend = -1
            elif (d > 0 and l1[m] < lim[j]) or (d < 0 and h1[m] > lim[j]):      # 관통 체결
                pos, pend, mf = j, -1, m
                tex = (t1[m] // 900_000) * 900_000 + (H + 1) * 900_000 - 60_000
                if (d > 0 and l1[m] <= sl[j]) or (d < 0 and h1[m] >= sl[j]):    # 체결 분 손절(익절은 안 봄)
                    out[k, 0] = j; out[k, 1] = m; out[k, 2] = m; out[k, 3] = d; out[k, 4] = lim[j]; out[k, 5] = sl[j]; out[k, 6] = 1
                    k += 1; pos = -1
                elif leak and ((d > 0 and h1[m] > tp[j]) or (d < 0 and l1[m] < tp[j])):   # 양성 대조 전용: 체결 분 익절 허용
                    out[k, 0] = j; out[k, 1] = m; out[k, 2] = m; out[k, 3] = d; out[k, 4] = lim[j]; out[k, 5] = tp[j]; out[k, 6] = 2
                    k += 1; pos = -1
            elif (t1[m] + 60_000) % 900_000 == 0 and ((d > 0 and c1[m] < far[j]) or (d < 0 and c1[m] > far[j])):
                pend = -1                                                          # 15분 종가가 존 먼 쪽 밖 = 존 깨짐
        elif p >= ns:
            break
        else:
            m = m0[p]
            continue
        m += 1
    return out[:k]


# ── 신호 ─────────────────────────────────────────────────────────────────────
def coin_data(k1: pd.DataFrame) -> dict:
    B = SM.bars(k1, 15)
    S = {}
    for N in (5, 50):
        ev = SM.structure(B["h"], B["l"], B["c"], SM.pivots(B["h"], N, True), SM.pivots(B["l"], N, False), N)
        T = ev[:, 0].astype(int); d = ev[:, 1]; btm, top = ev[:, 4], ev[:, 5]; atr = B["atr"][T]; cT = B["c"][T]
        ok = np.isfinite(btm) & np.isfinite(top) & (top > btm) & np.isfinite(atr) & (atr > 0) & np.where(d > 0, cT > top, cT < btm)
        S[N] = pd.DataFrame(dict(T=T, d=d, btm=btm, top=top, atr=atr, typ=ev[:, 2], known=B["t"][T] + MS15,
                                 leg=leg_ext(B["h"], B["l"], T, d, btm, top)))[ok].reset_index(drop=True)
    day = pd.Series(k1.c.to_numpy(), index=pd.to_datetime(k1.t, unit="ms")).resample("D").last().ffill()   # 빠진 날이 SMA50 을 50일 지우지 않게
    tr = np.sign(day - day.rolling(50).mean())                      # 일봉 d 마감 뒤에 알려짐
    tr.index = tr.index.values.astype("datetime64[ms]").astype(np.int64) + 86_400_000   # 알려지는 시각 = 그날 마감
    t1 = k1.t.to_numpy()
    for N in S:
        s = S[N]
        ix = np.searchsorted(tr.index.to_numpy(), s.known.to_numpy(), "right") - 1
        s["trend"] = np.where(ix >= 0, tr.to_numpy()[np.maximum(ix, 0)], 0)
        s["m0"] = np.searchsorted(t1, s.known.to_numpy())
    return dict(t1=t1, h1=k1.h.to_numpy(), l1=k1.l.to_numpy(), c1=k1.c.to_numpy(), S=S, B=B)


def run_cfg(D: dict, g, lo: int, hi: int, shift: bool = False, leak: int = 0) -> pd.DataFrame:
    N, tpk, slb, H, trf = g
    s = D["S"][N]
    s = s[(s.known >= lo) & (s.known < hi)]
    if trf:
        s = s[s.trend == s.d]
    s = s.reset_index(drop=True)
    d, btm, top, atr = s.d.to_numpy(), s.btm.to_numpy(), s.top.to_numpy(), s.atr.to_numpy()
    if shift:                                                       # 대조 A: 가격에서 먼 쪽으로 한 칸
        hgt = top - btm
        btm, top = np.where(d > 0, btm - hgt, top), np.where(d > 0, btm, top + hgt)
    lim, far, sl, tp = levels(d, btm, top, atr, s.leg.to_numpy(), g)
    exp_t = s.known.to_numpy() + W_PEND * MS15
    m0 = s.m0.to_numpy(np.int64) if not leak else np.searchsorted(D["t1"], s.known.to_numpy() - MS15).astype(np.int64)   # 양성 대조: 돌파 봉 시작부터 미리 앎
    r = sim(D["t1"], D["h1"], D["l1"], D["c1"], m0, d.astype(float), lim, far, sl, tp, exp_t.astype(np.int64), H, leak)
    T = pd.DataFrame(r, columns=["sig", "mf", "mx", "s", "entry", "exit", "typ"])
    T["sig"] = T.sig.astype(int); T["mf"] = T.mf.astype(int); T["mx"] = T.mx.astype(int)
    T["t_in"] = D["t1"][T.mf]; T["t_out"] = D["t1"][T.mx] + 60_000
    T["gross"] = T.s * (T.exit / T.entry - 1) * 1e4
    T["chtyp"] = s.typ.to_numpy()[T.sig] if len(T) else []
    T["wait_min"] = T.mf - s.m0.to_numpy()[T.sig] if len(T) else []
    T = T[T.t_in < hi].reset_index(drop=True)                      # 구간 끝 뒤 체결은 버림
    T.attrs["n_sig"] = len(s)
    return T


def net(T: pd.DataFrame, cost: str) -> pd.Series:
    e, tpc, mk = COST[cost]
    return T.gross - e - np.where(T.typ == 2, tpc, mk)


def daily(T: pd.DataFrame, cost: str, lo: int, hi: int) -> pd.Series:
    return KS.daily(T.assign(net=net(T, cost)), lo, hi)


def stats(T: pd.DataFrame, cost: str) -> dict:
    x = net(T, cost)
    w, lz = x[x > 0], -x[x <= 0]
    return dict(trades=len(T), fill_rate=round(len(T) / max(T.attrs.get("n_sig", len(T)), 1), 3), net_bp=round(x.mean(), 2), gross_bp=round(T.gross.mean(), 2),
                win=round((x > 0).mean(), 3), be_win=round(lz.mean() / (w.mean() + lz.mean()), 3) if len(w) and len(lz) else None,
                avg_win=round(w.mean(), 1), avg_loss=round(lz.mean(), 1), hold_min=round((T.mx - T.mf).mean(), 1), wait_min=round(T.wait_min.mean(), 1),
                typ_share={int(k): round(v, 3) for k, v in T.typ.value_counts(normalize=True).items()})


def recon(D: dict, g, lo: int, hi: int, rng, n=200) -> str:
    """독립 재구성(pandas, 신호별): sim 이 낸 거래 n 건의 체결 분·청산 분·종류·청산가를 다른 경로로 다시 계산."""
    N, tpk, slb, H, trf = g
    s = D["S"][N]; s = s[(s.known >= lo) & (s.known < hi)]
    s = (s[s.trend == s.d] if trf else s).reset_index(drop=True)
    T = run_cfg(D, g, lo, hi)
    t1 = pd.Series(D["t1"]); bad = 0
    for _, r in T.sample(min(n, len(T)), random_state=int(rng.integers(1 << 30))).iterrows():
        q = s.iloc[int(r.sig)]; d = q.d
        lim, _, sl, tp = (float(x) for x in levels(d, q.btm, q.top, q.atr, q.leg, g))
        w = np.arange(int(q.m0), len(t1))
        hit = (D["l1"][w] < lim) if d > 0 else (D["h1"][w] > lim)
        mf = int(w[np.argmax(hit)])
        stop_now = (D["l1"][mf] <= sl) if d > 0 else (D["h1"][mf] >= sl)
        if stop_now:
            mx, typ = mf, 1
        else:
            tex = pd.Timestamp(D["t1"][mf], unit="ms").floor("15min").value // 10**6 + (H + 1) * MS15 - 60_000
            v = np.arange(mf + 1, len(t1))
            a = (D["l1"][v] <= sl) if d > 0 else (D["h1"][v] >= sl)
            b = (D["h1"][v] > tp) if d > 0 else (D["l1"][v] < tp)
            c_ = D["t1"][v] >= tex
            first = [int(np.argmax(x)) if x.any() else 10**12 for x in (a, b, c_)]
            i = min(first); typ = [1, 2, 3][first.index(i)]; mx = int(v[i])
        bad += not (mf == r.mf and mx == r.mx and typ == r.typ)
    return f"{bad}/{min(n, len(T))}"


# ── 개발 ─────────────────────────────────────────────────────────────────────
RAW: dict = {}


def load_all(placebo_draw: int | None = None) -> dict:
    out = {}
    for ci, coin in enumerate(COINS):
        if coin not in RAW:
            RAW[coin] = SM.load_1m(f"{coin}USDT")
        k = RAW[coin] if placebo_draw is None else SM.placebo(RAW[coin], SEED + 1000 * ci + placebo_draw)
        miss = (pd.Series(1, index=pd.to_datetime(k.t, unit="ms")).resample("D").size() == 0).sum()
        assert miss == 0 and k.t.iloc[-1] >= SM.END - 86_400_000, f"{coin} 빠진 날 {miss} · 끝 {k.t.iloc[-1]} (SCRATCH_K 확인)"
        out[coin] = coin_data(k)
    return out


def portfolio(Dall: dict, g, lo, hi, shift=False, leak=0) -> pd.DataFrame:
    parts = [run_cfg(Dall[c], g, lo, hi, shift, leak).assign(coin=c) for c in COINS]
    T = pd.concat(parts, ignore_index=True)
    T.attrs["n_sig"] = sum(p.attrs["n_sig"] for p in parts)
    return T


def day_boot_mean(x: np.ndarray, t: np.ndarray, rng, B=2000) -> list[float]:
    day = t // 86_400_000; keys, inv = np.unique(day, return_inverse=True)
    sums, cnts = np.bincount(inv, x), np.bincount(inv)
    bs = [sums[ix].sum() / cnts[ix].sum() for ix in rng.integers(0, len(keys), (B, len(keys)))]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def controls(Dall, want: dict, lo, hi, draws=20, select_all=False) -> dict:
    """want = {cost: (g, T)}. A 옆 띠(신호별 짝 차이 달력일 부트스트랩) · B 기하 위약 20벌(위약마다 1분봉 새로 만듦)."""
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for cost, (g, T) in want.items():
        A = portfolio(Dall, g, lo, hi, shift=True)
        a = net(A, cost).groupby([A.coin, A.sig]).first(); r_ = net(T, cost).groupby([T.coin, T.sig]).first()
        both = r_.index.intersection(a.index); dd = (r_[both] - a[both]).to_numpy()
        tt = T.set_index(["coin", "sig"]).t_in.groupby(level=[0, 1]).first()[both].to_numpy()
        out[cost] = dict(select_all=select_all, A_trades=len(A), A_net_bp=round(net(A, cost).mean(), 2), real_minus_A=round(net(T, cost).mean() - net(A, cost).mean(), 2),
                         paired_n=len(both), paired_diff=round(dd.mean(), 2) if len(dd) else None, paired_ci95=day_boot_mean(dd, tt, rng) if len(dd) else None, B=[])
        # 음성 대조: 같은 체결 분·같은 보유 길이, 방향만 무작위 → 실제 시장 움직임 − 비용 ≈ −비용
        rs = rng.choice([-1.0, 1.0], len(T)); px = np.array([Dall[c]["c1"][m] for c, m in zip(T.coin, T.mx)])
        e, tpc, mk = COST[cost]
        out[cost]["negative_random_dir_net_bp"] = round(float(np.mean(rs * (px / T.entry.to_numpy() - 1) * 1e4 - e - mk)), 2)
        L = portfolio(Dall, g, lo, hi, leak=1)                      # 양성 대조: sim 을 거친 누수판(15분 미리 앎 + 체결 분 익절)
        out[cost]["positive_leak_net_bp"] = round(float(net(L, cost).mean()), 2)
    _, _, _, _, sharpe = KS._helpers()
    for dr in range(draws):
        P = load_all(dr)
        for cost, (g, T) in want.items():
            X = portfolio(P, g, lo, hi)
            out[cost]["B"].append(round(float(net(X, cost).mean()), 2) if len(X) else None)
            if out[cost].get("select_all"):                         # 위약에서도 48설정 중 일 샤프 1등을 다시 고름(선택 맞춤)
                XS = {NAME(x): portfolio(P, x, lo, hi) for x in GRID}
                best = max(XS, key=lambda k_: sharpe(daily(XS[k_], cost, lo, hi)))
                out[cost].setdefault("B_selected", []).append(round(float(net(XS[best], cost).mean()), 2))
        print("placebo", dr, flush=True)
    for cost, (g, T) in want.items():
        b = np.array([v for v in out[cost]["B"] if v is not None])
        out[cost]["B_pctile"] = float((b < net(T, cost).mean()).mean() * 100)
        if "B_selected" in out[cost]:
            out[cost]["B_selected_pctile"] = float((np.array(out[cost]["B_selected"]) < net(T, cost).mean()).mean() * 100)
    return out


def dev_run() -> None:
    block_boot, cscv_pbo, dsr, mdd, sharpe = KS._helpers()
    OUT.mkdir(parents=True, exist_ok=True)
    lo, hi = MSD(DEV[0]), MSD(DEV[1])
    Dall = load_all()
    TT = {NAME(g): portfolio(Dall, g, lo, hi) for g in GRID}
    res = {}
    for cost in ("strict", "lenient"):
        M = pd.DataFrame({k: daily(T, cost, lo, hi) for k, T in TT.items()})
        S = max(M.columns, key=lambda k: sharpe(M[k])); g = GRID[[NAME(x) for x in GRID].index(S)]
        T = TT[S]; x = net(T, cost); yr = pd.to_datetime(T.t_out, unit="ms").dt.year
        rng = np.random.default_rng(SEED)
        paths = KS.cpcv(M, sharpe)
        pbo = cscv_pbo(M.to_numpy(), np.minimum((np.arange(len(M)) * 16) // len(M), 15))
        d_s = dsr(M[S].to_numpy(), [M[k].mean() / M[k].std(ddof=1) for k in M.columns if M[k].std() > 0])
        years = {int(y): round(x[yr == y].mean(), 2) for y in range(2022, 2026)}
        coins = {c: [int((T.coin == c).sum()), round(x[T.coin == c].mean(), 2)] for c in COINS}
        yp, cp = sum(v > 0 for v in years.values()), sum(v[1] > 0 for v in coins.values())
        ci = block_boot(M[S].to_numpy(), rng)
        r = dict(S_star=S, stats=stats(T, cost), sharpe=round(sharpe(M[S]), 3), mdd=round(mdd(M[S]), 4), ci95_daily=ci, cpcv=[round(p, 3) for p in paths], pbo=pbo, dsr=d_s,
                 years=years, coins=coins, y2425=round(x[T.t_in >= MSD("2024-01-01")].mean(), 2),
                 bos_choch={int(k): [int(len(v)), round(v.mean(), 2)] for k, v in x.groupby(T.chtyp)})
        if cost == "strict":
            r["gate"] = dict(ci=ci[0] > 0, cpcv=bool(np.median(paths) > 0 and sum(p > 0 for p in paths) >= 6), pbo=pbo < .2, dsr=d_s > .95, yc=yp >= 3 and cp >= 3)
        else:
            r["gate"] = dict(mean=bool(M[S].mean() > 0), cpcv_med=bool(np.median(paths) > 0), pbo=pbo < .5, years=yp >= 3)
        r["pass"] = bool(all(r["gate"].values()))
        dl = sum(KS.SIZE * pd.Series(Dall[c]["B"]["c"], index=pd.to_datetime(Dall[c]["B"]["t"], unit="ms")).resample("D").last().pct_change() * 1e4 for c in COINS)
        dl = dl.reindex(M.index).fillna(0)
        r["sharpe_minus_long"], r["sharpe_minus_short"] = round(sharpe(M[S] - dl), 3), round(sharpe(M[S] + dl), 3)
        res[cost] = r
        print(cost, json.dumps({k: v for k, v in r.items()}, ensure_ascii=False, default=float), flush=True)
    want = {cost: (GRID[[NAME(x) for x in GRID].index(res[cost]["S_star"])], TT[res[cost]["S_star"]]) for cost in res}
    for cost, v in controls(Dall, want, lo, hi, select_all=True).items():
        res[cost]["controls"] = v
    # 절단 재계산 (S* 의 N, ETH)
    N = GRID[[NAME(x) for x in GRID].index(res["strict"]["S_star"])][0]
    B = Dall["ETH"]["B"]; s = Dall["ETH"]["S"][N]
    rng = np.random.default_rng(SEED); mism = 0
    pick = rng.choice(len(s), 200, replace=False)
    for p in pick:
        T0 = int(s["T"][p]); h, l, c = B["h"][: T0 + 1], B["l"][: T0 + 1], B["c"][: T0 + 1]
        ev = SM.structure(h, l, c, SM.pivots(h, N, True), SM.pivots(l, N, False), N)
        hit = ev[(ev[:, 0] == T0)]
        mism += not (len(hit) and hit[0, 1] == s.d[p] and hit[0, 4] == s.btm[p] and hit[0, 5] == s.top[p])
    res["truncation_mismatch"] = f"{mism}/200"
    full = SM.structure(B["h"], B["l"], B["c"], SM.pivots(B["h"], N, True), SM.pivots(B["l"], N, False), 0)   # 양성: 피벗을 그 봉에 바로 안다고 친 누수판
    mis2 = 0
    for p in rng.choice(len(full), 200, replace=False):
        T0 = int(full[p, 0]); h, l, c = B["h"][: T0 + 1], B["l"][: T0 + 1], B["c"][: T0 + 1]
        ev = SM.structure(h, l, c, SM.pivots(h, N, True), SM.pivots(l, N, False), 0)
        hit = ev[ev[:, 0] == T0]
        mis2 += not (len(hit) and hit[0, 1] == full[p, 1] and hit[0, 4] == full[p, 4] and hit[0, 5] == full[p, 5])
    res["truncation_mismatch_leaky_positive"] = f"{mis2}/200"
    gS = GRID[[NAME(x) for x in GRID].index(res["strict"]["S_star"])]
    res["recon_mismatch"] = {c: recon(Dall[c], gS, lo, hi, rng, 50) for c in COINS}
    res["table"] = {k: dict(strict=stats(T, "strict"), lenient_net=round(net(T, "lenient").mean(), 2)) for k, T in TT.items()}
    json.dump(res, open(OUT / "dev.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("truncation", res["truncation_mismatch"])


def lock_run(cfg: str, cost: str) -> None:
    _, _, _, mdd, sharpe = KS._helpers()
    dev = json.load(open(OUT / "dev.json"))
    assert dev[cost]["S_star"] == cfg and dev[cost]["pass"], "관문을 넘은 설정만 잠금을 연다"
    g = GRID[[NAME(x) for x in GRID].index(cfg)]
    lo, hi = MSD(LOCK[0]), MSD(LOCK[1])
    Dall = load_all(); T = portfolio(Dall, g, lo, hi); x = net(T, cost); d = daily(T, cost, lo, hi)
    rng = np.random.default_rng(SEED); half = T.t_in < MSD("2026-06-01")
    res = dict(cfg=cfg, cost=cost, stats=stats(T, cost), ci95=day_boot_mean(x.to_numpy(), T.t_out.to_numpy(), rng), mdd=mdd(d), sharpe=sharpe(d),
               coins={c: [int((T.coin == c).sum()), float(x[T.coin == c].mean())] for c in COINS}, h1=float(x[half].mean()), h2=float(x[~half].mean()),
               controls=controls(Dall, {cost: (g, T)}, lo, hi)[cost])
    json.dump(res, open(OUT / f"lock_{cost}.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, default=float))


# ── 실원장 ───────────────────────────────────────────────────────────────────
def ledger_run(path: str) -> None:
    R = [json.loads(x) for x in open(path) if x.strip()]
    R = [r for r in R if r.get("closed", True)]
    base = lambda s: s.replace("USDT", "").replace("USDC", "")
    rows = []
    for coin in sorted({base(r["symbol"]) for r in R}):
        k = SM.load_1m(f"{coin}USDT"); k = k[k.t >= MSD("2026-01-01")].reset_index(drop=True)
        D = coin_data(k); B = D["B"]; c15, end = B["c"], B["t"] + MS15
        Z = {N: D["S"][N] for N in (5, 50)}
        for r in [r for r in R if base(r["symbol"]) == coin]:
            s = 1 if r["side"] == "LONG" else -1; t0 = r["entry_time"]; px = r["entry_price"]
            row = dict(coin=coin, s=s, t0=t0, net=r["net_pnl"])
            m0 = np.searchsorted(D["t1"], t0 // 60000 * 60000) - 1
            for Hm in (60, 240):
                mh = np.searchsorted(D["t1"], t0 // 60000 * 60000 + Hm * 60000)
                ok = m0 >= 0 and mh < len(D["t1"]) and D["t1"][mh] == t0 // 60000 * 60000 + Hm * 60000
                row[f"fwd{Hm}"] = s * (D["c1"][mh] / D["c1"][m0] - 1) * 1e4 if ok else np.nan
            b = int(np.searchsorted(end, t0, "right")) - 1          # 진입 전 마지막 마감 15분봉
            for N, z in Z.items():
                live = z[(z["T"] <= b) & (z["T"] >= b - W_PEND) & (z.btm <= px) & (px <= z.top)]
                st = "out"
                for _, q in live.iterrows():
                    seg = c15[int(q["T"]) + 1: b + 1]
                    broke = (seg < q.btm).any() if q.d > 0 else (seg > q.top).any()
                    if not broke:
                        st = "aligned" if q.d == s else ("against" if st == "out" else st)
                        if st == "aligned":
                            break
                row[f"N{N}"] = st
            rows.append(row)
    d = pd.DataFrame(rows)
    d["day"] = (d.t0 + 9 * 3_600_000) // 86_400_000
    out = dict(n=len(d), by_state={f"N{N}": {st: dict(n=int(len(g)), fwd60=round(g.fwd60.mean(), 1), fwd240=round(g.fwd240.mean(), 1), net_usd=round(g.net.sum(), 0),
                                                       win=round((g.net > 0).mean(), 3)) for st, g in d.groupby(f"N{N}")} for N in (5, 50)})
    x = d.dropna(subset=["fwd240"]); a = (x.N5 == "aligned").to_numpy(); f = x.fwd240.to_numpy(); days = x.day.to_numpy()
    rng = np.random.default_rng(SEED); ud = np.unique(days); idx = {u: np.flatnonzero(days == u) for u in ud}; bs = []
    for _ in range(2000):
        ix = np.concatenate([idx[u] for u in rng.choice(ud, len(ud))])
        if a[ix].any() and (~a[ix]).any():
            bs.append(f[ix][a[ix]].mean() - f[ix][~a[ix]].mean())
    out["test_N5_aligned_minus_rest_fwd240"] = dict(n_aligned=int(a.sum()), n_rest=int((~a).sum()), diff=round(f[a].mean() - f[~a].mean(), 1) if a.any() else None,
                                                    ci95=[round(v, 1) for v in np.percentile(bs, [2.5, 97.5])] if bs else None)
    out["by_coin_N5"] = {c: {st: [int(len(h)), round(h.fwd240.mean(), 1), round(h.net.sum(), 0)] for st, h in g.groupby("N5")} for c, g in d.groupby("coin")}
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "ledger_rows.csv", index=False)
    json.dump(out, open(OUT / "ledger.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


# ── 자체 점검 ────────────────────────────────────────────────────────────────
def selftest() -> None:
    t = np.arange(40, dtype=np.int64) * 60_000                      # 분 0.. (15분 경계 = 분 14·29 마감)
    h = np.full(40, 103.0); l = np.full(40, 101.0); c = np.full(40, 102.0)
    one = lambda **kw: sim(t, kw.get("h", h), kw.get("l", l), kw.get("c", c), np.array([0]), np.array([1.0]), np.array([100.0]),
                          np.array([99.0]), np.array([kw.get("sl", 99.0)]), np.array([kw.get("tp", 101.0)]), np.array([kw.get("exp", 10**9)], np.int64), kw.get("H", 1))
    assert len(one()) == 0                                          # 지정가 100 미만으로 안 내려옴
    L = l.copy(); L[3] = 100.0
    assert len(one(l=L)) == 0                                       # 닿기만 = 미체결(관통 아님)
    L[3] = 99.9; H_ = h.copy(); H_[3] = 105.0; H_[4] = 101.5
    r = one(l=L, h=H_)
    assert r[0, 1] == 3 and r[0, 2] == 4 and r[0, 6] == 2 and r[0, 5] == 101.0    # 체결 분 고가 105 는 무시, 다음 분 익절
    L2 = L.copy(); L2[3] = 98.5
    r = one(l=L2)
    assert r[0, 1] == r[0, 2] == 3 and r[0, 6] == 1                 # 체결 분 손절
    L3 = L.copy(); L3[5] = 98.0; H3 = h.copy(); H3[5] = 102.0; H3[4] = 100.5; L3[4] = 100.2
    r = one(l=L3, h=H3)
    assert r[0, 2] == 5 and r[0, 6] == 1                            # 같은 분 둘 다 → 손절
    H4 = np.full(40, 100.5); L4 = np.full(40, 100.1); L4[3] = 99.9; C4 = np.full(40, 100.3)
    r = one(l=L4, h=H4, c=C4, H=1)
    assert r[0, 6] == 3 and r[0, 2] == 29                           # 체결 15분봉(0~14) 다음 1봉 마감 = 분 29
    C5 = c.copy(); C5[14] = 98.0; L5 = l.copy(); L5[14] = 97.0; L5[20] = 99.0
    assert len(one(c=C5, l=np.where(np.arange(40) == 14, 100.5, L5))) == 0   # 15분 종가가 존 아래 → 취소(이후 체결 없음)
    L6 = l.copy(); L6[12] = 99.9
    assert len(one(l=L6, exp=12 * 60_000)) == 0                     # 만료 뒤 체결 없음
    # 교체: 두 번째 신호가 대기 주문을 바꿈
    r = sim(t, h, L, c, np.array([0, 2]), np.array([1.0, 1.0]), np.array([90.0, 100.0]), np.array([89.0, 99.0]), np.array([89.0, 99.0]),
            np.array([91.0, 101.0]), np.array([10**9, 10**9], np.int64), 1)
    assert len(r) == 1 and r[0, 0] == 1
    # 비용
    T = pd.DataFrame(dict(gross=[10.0, 10.0], typ=[2, 1]))
    assert net(T, "strict").tolist() == [6.0, 2.0] and net(T, "lenient").tolist() == [10.0, 4.0]
    Hl = h.copy(); Hl[3] = 105.0
    r = sim(t, Hl, L, c, np.array([0]), np.array([1.0]), np.array([100.0]), np.array([99.0]), np.array([99.0]), np.array([101.0]), np.array([10**9], np.int64), 1, 1)
    assert r[0, 2] == 3 and r[0, 6] == 2                            # 누수판만 체결 분 익절
    Lg = L.copy(); Lg[5] = 90.0; Hg = h.copy(); Hg[5] = 95.0
    r = one(l=Lg, h=Hg, tp=200.0)
    assert r[0, 6] == 1 and r[0, 5] == 95.0                         # 분 전체가 손절가 아래면 그 분 고가로 체결
    one_ = lambda *a: [float(x) for x in levels(*a)]
    assert one_(1.0, 98.0, 100.0, 4.0, 110.0, (5, 1.0, 0.5, 16, 0), False) == [100, 98, 96, 104]          # 1라운드 식
    assert one_(1.0, 98.0, 100.0, 4.0, 110.0, (5, "a50", "2R", 16, 0), True) == [100, 98, 98, 104]
    assert one_(1.0, 98.0, 100.0, 4.0, 110.0, (5, "mid", "3R", 16, 0), True) == [100, 98, 99, 103]
    assert one_(1.0, 98.0, 100.0, 4.0, 110.0, (5, "a25", "leg", 16, 0), True) == [100, 98, 99, 110]
    assert one_(-1.0, 100.0, 102.0, 4.0, 90.0, (5, "a25", "2R", 16, 0), True) == [100, 102, 101, 98]
    hh = np.array([5, 9, 7, 6, 8, 10.]); ll = hh - 2; ll[3] = 3.0                               # OB 봉 3 [3,6] · 돌파 봉 5
    assert leg_ext(hh, ll, np.array([5]), np.array([1.0]), np.array([3.0]), np.array([6.0]))[0] == 10.0
    assert len(GRID) == (72 if R2 else 48)
    print("selftest ok")


if __name__ == "__main__":
    cmd, *arg = sys.argv[1:]
    dict(selftest=selftest, dev=dev_run, lock=lock_run, ledger=ledger_run)[cmd](*arg)
