#!/usr/bin/env python3
"""켈트너 수축(TTM 스퀴즈) 돌파 매매 + 실원장 적용 (2026-11-15). 사전등록 docs/prereg/keltner_squeeze_breakout_20261115.md (d3f5eb56, 계산 전).

  selftest          지표·발동·청산·비용 손계산 단언
  dev               개발 2022~25: 6설정 표 · 관문(엄격·후보) · 대조 · 절단 재계산 → tmp/keltner_squeeze_20261115/dev.json
  lock <설정>       2026 잠금 1회 (관문 통과 설정만)
  ledger <trips>    실원장 왕복 × 진입 직전 수축 상태 → ledger.json
입력: data/binance_vision/klines1m{,_daily} (메인 체크아웃, 읽기만) + SCRATCH_K 의 Vision 일 파일(최근분).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
BV = Path("/home/kbj20/crypto-scalping/data/binance_vision")
SK = Path(os.environ.get("SCRATCH_K", "/nonexistent"))
OUT = ROOT / "tmp/keltner_squeeze_20261115"
COINS = ["ETH", "BTC", "SOL", "XRP"]
TFS = {"15m": 15, "1h": 60, "4h": 240}
ENTRIES = ["E1", "E2"]
CONFIGS = [f"{tf}_{e}" for tf in TFS for e in ENTRIES]
DEV = ("2022-01-01", "2025-12-31")
LOCK = ("2026-01-01", "2026-11-13")
L, MB, MK, MINSQ, WIN2, MAXH = 20, 2.0, 1.5, 6, 12, 48
COST = 12.0          # 왕복 bp = (테이커 5 + 미끄러짐 1) × 2
SIZE = 0.25          # 거래당 명목 / 순자산
MS = lambda s: int(pd.Timestamp(s, tz="UTC").value // 10**6)


# ── 자료 ─────────────────────────────────────────────────────────────────────
def load_1m(sym: str) -> pd.DataFrame:
    ps = sorted((BV / "klines1m").glob(f"{sym}-1m-*.parquet")) + sorted((BV / "klines1m_daily").glob(f"{sym}-1m-*.parquet"))
    ps += sorted(SK.glob(f"{sym}-*.parquet"))
    k = pd.concat([pd.read_parquet(p, columns=["t", "h", "l", "c"]) for p in ps])
    return k.drop_duplicates("t").sort_values("t").reset_index(drop=True)


def bars(k: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """1분이 다 찬 봉만, 빠진 봉은 NaN 행으로 격자를 채운다(롤링이 구멍을 건너지 않게)."""
    ms = minutes * 60_000
    g = k.groupby(k.t // ms * ms)
    b = pd.DataFrame({"h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "n": g.t.size()})
    b.loc[b.n < minutes, ["h", "l", "c"]] = np.nan
    grid = np.arange(b.index[0], b.index[-1] + ms, ms)
    b = b.reindex(grid)[["h", "l", "c"]].rename_axis("t").reset_index()
    b["end"] = b.t + ms          # 봉이 마감하는 시각
    return b


def linreg_end(x: pd.Series, n: int) -> np.ndarray:
    """TradingView linreg(x, n, 0): 최근 n 개에 직선 적합한 마지막 점 값 (가중합)."""
    i = np.arange(n); ib = i.mean()
    w = 1 / n + (i - ib) * (n - 1 - ib) / ((i - ib) ** 2).sum()
    return x.rolling(n).apply(lambda a: a @ w, raw=True).to_numpy()


def indicators(b: pd.DataFrame) -> dict:
    c, h, l = b.c, b.h, b.l
    sma = c.rolling(L).mean(); sd = c.rolling(L).std(ddof=0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1, skipna=False)
    rm = tr.rolling(L).mean()
    ubb, lbb, ukc, lkc = sma + MB * sd, sma - MB * sd, sma + MK * rm, sma - MK * rm
    valid = (ubb.notna() & ukc.notna()).to_numpy()
    on = ((lbb > lkc) & (ubb < ukc)).to_numpy() & valid
    val = linreg_end(c - ((h.rolling(L).max() + l.rolling(L).min()) / 2 + sma) / 2, L)
    run = np.zeros(len(b), int)                       # 봉 t 에서 끝나는 연속 수축 봉 수
    for t in range(1, len(b)):
        run[t] = run[t - 1] + 1 if on[t] else 0
    run[0] = int(on[0])
    rel = np.zeros(len(b), bool)                      # 봉 t 에서 수축이 풀림(≥6봉 뒤)
    rel[1:] = on[:-1] & ~on[1:] & (run[:-1] >= MINSQ) & valid[1:] & np.isfinite(val[1:])
    return dict(c=c.to_numpy(), mid=sma.to_numpy(), ukc=ukc.to_numpy(), lkc=lkc.to_numpy(), on=on, val=val, rel=rel,
                valid=valid & np.isfinite(val), end=b.end.to_numpy())


def signals(I: dict, entry: str) -> list[tuple[int, int]]:
    """[(봉 색인, 방향 ±1)] — 봉 t 마감에 알 수 있는 것만."""
    c, val, rel = I["c"], I["val"], I["rel"]
    out = []
    if entry == "E1":
        for t in np.flatnonzero(rel):
            if val[t] != 0:
                out.append((int(t), int(np.sign(val[t]))))
    else:
        last = -10**9
        for t in range(len(c)):
            if rel[t]:
                last, done = t, False
            if t - last < WIN2 and not done and I["valid"][t]:
                if c[t] > I["ukc"][t]:
                    out.append((t, 1)); done = True
                elif c[t] < I["lkc"][t]:
                    out.append((t, -1)); done = True
    return out


def exit_index(I: dict, i: int, s: int) -> int:
    c, val, mid = I["c"], I["val"], I["mid"]
    n = len(c)
    for j in range(i + 1, min(i + MAXH, n - 1) + 1):
        if not (np.isfinite(c[j]) and np.isfinite(val[j]) and np.isfinite(mid[j])):
            return j - 1                                # 구멍 직전 봉에서 정리
        weak = (s * val[j] < s * val[j - 1] < s * val[j - 2])
        cross = s * (c[j] - mid[j]) < 0
        if weak or cross or j == i + MAXH:
            return j
    return min(i + MAXH, n - 1)


def trades(I: dict, sig: list, lo: int, hi: int, dirs=None) -> pd.DataFrame:
    """겹침 없이 차례로. lo ≤ 진입 봉 마감 시각 < hi. dirs 가 있으면 방향을 바꿔 끼움(대조)."""
    rows, busy = [], -1
    for k, (i, s) in enumerate(sig):
        if i <= busy or not (lo <= I["end"][i] < hi):
            continue
        s = s if dirs is None else dirs[k]
        if s == 0:
            continue
        j = exit_index(I, i, s)
        if j <= i:
            continue
        g = s * (I["c"][j] / I["c"][i] - 1) * 1e4
        rows.append((i, j, s, I["end"][i], I["end"][j], g, g - COST, j - i))
        busy = j
    return pd.DataFrame(rows, columns=["i", "j", "s", "t_in", "t_out", "gross", "net", "hold"])


def daily(T: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    days = pd.date_range(pd.to_datetime(lo, unit="ms"), pd.to_datetime(hi - 1, unit="ms"), freq="D").normalize()
    if T.empty:
        return pd.Series(0.0, index=days)
    d = (SIZE * T.net).groupby(pd.to_datetime(T.t_out, unit="ms").dt.normalize()).sum()
    return d.reindex(days, fill_value=0.0)


# ── 판정 도구 (저장소 재사용) ───────────────────────────────────────────────────
def _helpers():
    from research_ema_vwap_session_rules_20261029 import block_boot, cscv_pbo, dsr, mdd, sharpe
    return block_boot, cscv_pbo, dsr, mdd, sharpe


def cpcv(M: pd.DataFrame, sharpe) -> list[float]:
    """6개월 8묶음 · 2묶음 시험 · 28분할 → 7경로. 분할마다 나머지 6묶음 1등을 시험 묶음에."""
    g = ((M.index.year - 2022) * 2 + (M.index.month > 6)).to_numpy()
    res = {}
    for te in itertools.combinations(range(8), 2):
        tr = ~np.isin(g, te)
        best = M.columns[int(np.argmax([sharpe(M[c][tr]) for c in M.columns]))]
        for q in te:
            res.setdefault(q, []).append(M[best][g == q])
    return [sharpe(pd.concat([res[q][p] for q in range(8)])) for p in range(7)]


def build(cfg: str, cache: dict) -> dict:
    tf, e = cfg.split("_")
    out = {}
    for coin in COINS:
        key = (coin, tf)
        if key not in cache:
            cache[key] = indicators(bars(load_1m(f"{coin}USDT"), TFS[tf]))
        I = cache[key]
        out[coin] = (I, signals(I, e))
    return out


def random_entry(I: dict, T: pd.DataFrame, lo: int, hi: int, rng, n=300) -> np.ndarray:
    ok = np.flatnonzero(I["valid"] & (I["end"] >= lo) & (I["end"] < hi))
    c = I["c"]; out = np.empty((n, len(T)))
    for b in range(n):
        h = rng.permutation(T.hold.to_numpy())
        i = rng.choice(ok[ok < len(c) - MAXH - 1], len(T))
        r = T.s.to_numpy() * (c[i + h] / c[i] - 1) * 1e4 - COST
        out[b] = np.where(np.isfinite(r), r, -COST)
    return out


def score(cfg_trades: dict, lo: int, hi: int) -> tuple[pd.Series, pd.DataFrame]:
    T = pd.concat([t.assign(coin=c) for c, t in cfg_trades.items()], ignore_index=True)
    return daily(T, lo, hi), T


def dev_run() -> None:
    block_boot, cscv_pbo, dsr, mdd, sharpe = _helpers()
    OUT.mkdir(parents=True, exist_ok=True)
    lo, hi = MS(DEV[0]), MS("2026-01-01")
    cache, tab, D, TT, B = {}, {}, {}, {}, {}
    for cfg in CONFIGS:
        B[cfg] = build(cfg, cache)
        per = {c: trades(I, sg, lo, hi) for c, (I, sg) in B[cfg].items()}
        D[cfg], TT[cfg] = score(per, lo, hi)
        T = TT[cfg]
        tab[cfg] = dict(trades=len(T), per_day=round(len(T) / len(D[cfg]), 2), gross_bp=round(T.gross.mean(), 2), net_bp=round(T.net.mean(), 2),
                        win=round((T.net > 0).mean(), 3), hold_bars=round(T.hold.mean(), 1), sharpe=round(sharpe(D[cfg]), 3), mdd=round(mdd(D[cfg]), 4),
                        coins={c: [int((T.coin == c).sum()), round(T[T.coin == c].net.mean(), 2)] for c in COINS},
                        years={str(y): round(T[pd.to_datetime(T.t_out, unit="ms").dt.year == y].net.mean(), 2) for y in range(2022, 2026)},
                        y2425_net_bp=round(T[T.t_in >= MS("2024-01-01")].net.mean(), 2))
        print(cfg, tab[cfg])
    M = pd.DataFrame(D)
    S = max(CONFIGS, key=lambda c: sharpe(M[c]))
    rng = np.random.default_rng(20261115)
    ci = block_boot(M[S].to_numpy(), rng)
    paths = cpcv(M, sharpe)
    chunks = np.minimum((np.arange(len(M)) * 16) // len(M), 15)
    pbo = cscv_pbo(M.to_numpy(), chunks)
    daily_sr = [M[c].mean() / M[c].std(ddof=1) for c in CONFIGS]
    d_s = dsr(M[S].to_numpy(), daily_sr)
    T = TT[S]
    yrs_pos = sum(v > 0 for v in tab[S]["years"].values()); coins_pos = sum(v[1] > 0 for v in tab[S]["coins"].values())
    strict = dict(ci95=ci, ci_ok=ci[0] > 0, cpcv_paths=[round(p, 3) for p in paths], cpcv_ok=bool(np.median(paths) > 0 and sum(p > 0 for p in paths) >= 6),
                  pbo=pbo, pbo_ok=pbo < .2, dsr=d_s, dsr_ok=d_s > .95, years_pos=yrs_pos, coins_pos=coins_pos, yc_ok=yrs_pos >= 3 and coins_pos >= 3)
    strict["pass"] = bool(all(strict[k] for k in ("ci_ok", "cpcv_ok", "pbo_ok", "dsr_ok", "yc_ok")))
    lenient = dict(mean_pos=bool(M[S].mean() > 0), cpcv_med_pos=bool(np.median(paths) > 0), pbo_lt_5=pbo < .5, years_ok=yrs_pos >= 3)
    lenient["pass"] = bool(all(lenient.values()))
    # 대조 (S*)
    ctrl = {}
    rnd = np.concatenate([random_entry(I, T[T.coin == c].reset_index(drop=True), lo, hi, rng) for c, (I, _) in B[S].items() if (T.coin == c).any()], axis=1)
    ctrl["random_entry_mean_net"] = [round(float(np.percentile(rnd.mean(1), q)), 2) for q in (5, 50, 95)]
    ctrl["S_pctile_vs_random"] = float((rnd.mean(1) < T.net.mean()).mean() * 100)
    pos, neg = [], []
    for c, (I, sg) in B[S].items():
        fut = [int(np.sign(I["c"][min(i + 12, len(I["c"]) - 1)] - I["c"][i])) if np.isfinite(I["c"][min(i + 12, len(I["c"]) - 1)]) else 0 for i, _ in sg]
        pos.append(trades(I, sg, lo, hi, fut)); neg.append(trades(I, sg, lo, hi, list(rng.choice([-1, 1], len(sg)))))
    ctrl["positive_future_dir_net"] = round(pd.concat(pos).net.mean(), 2); ctrl["negative_random_dir_net"] = round(pd.concat(neg).net.mean(), 2)
    # 절단 재계산 (미래참조 점검)
    tf, e = S.split("_")
    k1 = load_1m("ETHUSDT"); bb = bars(k1, TFS[tf]); full = B[S]["ETH"][1]
    pick = rng.choice(len(full), min(200, len(full)), replace=False)
    mism = 0
    for p in pick:
        i, s = full[p]
        a0 = max(0, i - 600)                                     # 지표는 ~60봉만 거슬러 봄 → 600봉 창으로 잘라 재계산
        sub = signals(indicators(bb.iloc[a0: i + 1].reset_index(drop=True)), e)
        mism += not any(x == (i - a0, s) for x in sub[-3:])
    ctrl["truncation_mismatch"] = f"{mism}/{len(pick)}"
    # 모델−롱·모델−숏 (같은 25% 명목, 4코인)
    dl = sum(SIZE * pd.Series(cache[(c, tf)]["c"], index=pd.to_datetime(cache[(c, tf)]["end"], unit="ms")).resample("D").last().pct_change() * 1e4 for c in COINS)
    dl = dl.reindex(M.index).fillna(0)
    ctrl["sharpe_minus_long"] = round(sharpe(M[S] - dl), 3); ctrl["sharpe_minus_short"] = round(sharpe(M[S] + dl), 3)
    # 필요한 효과 크기 (2026 잠금 317일, 검정력 80%)
    ctrl["needed_daily_bp_lock"] = round(2.8 * M[S].std(ddof=1) / math.sqrt(317), 3)
    ctrl["needed_trade_net_bp_lock"] = round(2.8 * T.net.std(ddof=1) / math.sqrt(max(len(T) / len(M) * 317, 1)), 2)
    res = dict(S_star=S, table=tab, strict=strict, lenient=lenient, controls=ctrl,
               M_daily_sharpe={c: round(sharpe(M[c]), 3) for c in CONFIGS})
    json.dump(res, open(OUT / "dev.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("S_star", "strict", "lenient", "controls", "M_daily_sharpe")}, ensure_ascii=False, indent=1, default=float))


def lock_run(cfg: str) -> None:
    block_boot, _, _, mdd, sharpe = _helpers()
    dev = json.load(open(OUT / "dev.json"))
    assert cfg == dev["S_star"] and (dev["strict"]["pass"] or dev["lenient"]["pass"]), "관문을 넘은 설정만 잠금을 연다"
    lo, hi = MS(LOCK[0]), MS(LOCK[1]) + 86_400_000
    B = build(cfg, {})
    per = {c: trades(I, sg, lo, hi) for c, (I, sg) in B.items()}
    d, T = score(per, lo, hi)
    rng = np.random.default_rng(20261115)
    day = pd.to_datetime(T.t_out, unit="ms").dt.normalize()
    g = T.net.groupby(day.to_numpy())
    keys = g.sum().index; sums, cnts = g.sum().to_numpy(), g.size().to_numpy()
    bs = [sums[ix].sum() / cnts[ix].sum() for ix in rng.integers(0, len(keys), (2000, len(keys)))]
    rnd = np.concatenate([random_entry(I, T[T.coin == c].reset_index(drop=True), lo, hi, rng) for c, (I, _) in B.items() if (T.coin == c).any()], axis=1)
    half = T.t_in < MS("2026-06-01")
    res = dict(cfg=cfg, trades=len(T), net_bp=T.net.mean(), ci95=list(np.percentile(bs, [2.5, 97.5])), pct_vs_random=float((rnd.mean(1) < T.net.mean()).mean() * 100),
               mdd=mdd(d), sharpe=sharpe(d), coins={c: [int((T.coin == c).sum()), float(T[T.coin == c].net.mean())] for c in COINS},
               h1=float(T[half].net.mean()), h2=float(T[~half].net.mean()))
    json.dump(res, open(OUT / "lock.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(res)


# ── 실원장 ───────────────────────────────────────────────────────────────────
def ledger_run(path: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R = [json.loads(x) for x in open(path) if x.strip()]
    R = [r for r in R if r.get("closed", True)]
    K, I = {}, {}
    for coin in sorted({r["symbol"].replace("USDT", "").replace("USDC", "") for r in R}):
        k = load_1m(f"{coin}USDT"); k = k[k.t >= MS("2026-07-01")].reset_index(drop=True)
        K[coin] = k
        for tf, m in TFS.items():
            I[(coin, tf)] = indicators(bars(k, m))
    rows = []
    for r in R:
        coin = r["symbol"].replace("USDT", "").replace("USDC", "")
        s = 1 if r["side"] == "LONG" else -1
        t0 = r["entry_time"]; k = K[coin]
        m0 = k.t.searchsorted(t0 // 60000 * 60000) - 1                # 진입 직전 마감 1분봉
        row = dict(coin=coin, side=r["side"], t0=t0, net=r["net_pnl"])
        for H in (60, 240):
            mh = k.t.searchsorted(t0 // 60000 * 60000 + H * 60000)
            ok = 0 <= m0 and mh < len(k) and k.t.iloc[mh] == t0 // 60000 * 60000 + H * 60000
            row[f"fwd{H}"] = s * (k.c.iloc[mh] / k.c.iloc[m0] - 1) * 1e4 if ok else np.nan
        for tf in TFS:
            J = I[(coin, tf)]
            b = int(np.searchsorted(J["end"], t0, "right")) - 1      # end ≤ t0 인 마지막 봉
            if b < 0 or not J["valid"][b]:
                row[tf] = "na"; continue
            rr = np.flatnonzero(J["rel"][max(0, b - WIN2 + 1): b + 1])
            if J["on"][b]:
                row[tf] = "a_squeeze"
            elif len(rr):
                row[tf] = "b_aligned" if np.sign(J["val"][b]) == s else "c_against"
            else:
                row[tf] = "d_other"
        rows.append(row)
    d = pd.DataFrame(rows)
    d["day"] = (d.t0 + 9 * 3_600_000) // 86_400_000
    out = dict(n=len(d), by_state={})
    for tf in TFS:
        out["by_state"][tf] = {st: dict(n=int(len(g)), fwd60=round(g.fwd60.mean(), 1), fwd240=round(g.fwd240.mean(), 1), net_usd=round(g.net.sum(), 0),
                                        net_win=round((g.net > 0).mean(), 3)) for st, g in d.groupby(tf)}
    x = d.dropna(subset=["fwd240"])
    a = (x["1h"] == "b_aligned").to_numpy(); f = x.fwd240.to_numpy(); days = x.day.to_numpy()
    rng = np.random.default_rng(20261115); ud = np.unique(days); idx = {u: np.flatnonzero(days == u) for u in ud}
    bs = []
    for _ in range(2000):
        ix = np.concatenate([idx[u] for u in rng.choice(ud, len(ud))])
        if a[ix].any() and (~a[ix]).any():
            bs.append(f[ix][a[ix]].mean() - f[ix][~a[ix]].mean())
    out["test_1h_aligned_minus_rest_fwd240"] = dict(n_aligned=int(a.sum()), n_rest=int((~a).sum()), diff=round(f[a].mean() - f[~a].mean(), 1),
                                                    ci95=[round(v, 1) for v in np.percentile(bs, [2.5, 97.5])])
    out["by_coin_1h"] = {c: {st: [int(len(h)), round(h.fwd240.mean(), 1), round(h.net.sum(), 0)] for st, h in g.groupby("1h")} for c, g in d.groupby("coin")}
    d.to_csv(OUT / "ledger_rows.csv", index=False)
    json.dump(out, open(OUT / "ledger.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


# ── 자체 점검 ────────────────────────────────────────────────────────────────
def selftest() -> None:
    # linreg 끝점 = 직선이면 마지막 값
    x = pd.Series(np.arange(30, dtype=float) * 2 + 5)
    assert abs(linreg_end(x, 20)[-1] - x.iloc[-1]) < 1e-9
    y = pd.Series(np.r_[np.zeros(19), 19.0]); i = np.arange(20)
    assert abs(linreg_end(y, 20)[-1] - np.polyval(np.polyfit(i, y.to_numpy(), 1), 19)) < 1e-9
    # 수축 → 풀림: 아주 잔잔한 40봉(BB 가 KC 안) 뒤 큰 상승
    n = 80; c = np.r_[100 + 0.01 * np.sin(np.arange(60)), 100 + np.arange(1, 21) * 1.0]
    h, l = c + 0.5, c - 0.5                                   # 참범위가 커서 KC 가 넓음 → 잔잔할 땐 수축
    b = pd.DataFrame({"t": np.arange(n) * 60_000, "h": h, "l": l, "c": c, "end": np.arange(1, n + 1) * 60_000})
    I = indicators(b)
    assert I["on"][40:60].all() and I["rel"].sum() >= 1
    t = int(np.flatnonzero(I["rel"])[0]); assert t >= 60 and I["val"][t] > 0
    assert signals(I, "E1")[0] == (t, 1)
    e2 = signals(I, "E2"); assert e2 and e2[0][1] == 1 and t <= e2[0][0] < t + WIN2
    # 청산: 모멘텀 2봉 연속 약화 / 중심선 거슬러 / 최대 보유
    J = dict(c=np.array([10, 11, 12, 13, 14.]), val=np.array([1, 3, 2, 1, 0.]), mid=np.zeros(5))
    assert exit_index(J, 1, 1) == 3                            # val[3]<val[2]<val[1]
    J = dict(c=np.array([10, 11, 9, 13.]), val=np.array([1, 2, 3, 4.]), mid=np.full(4, 10.))
    assert exit_index(J, 1, 1) == 2                            # 종가 < 중심선
    J = dict(c=np.arange(100, 200, dtype=float), val=np.arange(100, dtype=float), mid=np.zeros(100), end=np.arange(100))
    assert exit_index(J, 0, 1) == MAXH
    # 비용·겹침
    T = trades(dict(J, end=np.arange(100)), [(0, 1), (10, 1), (60, -1)], 0, 10**9)
    assert len(T) == 2 and T.i.tolist() == [0, 60]
    assert abs(T.net.iloc[0] - ((100 + MAXH) / 100 - 1) * 1e4 + COST) < 1e-9
    print("selftest ok")


if __name__ == "__main__":
    cmd, *arg = sys.argv[1:]
    dict(selftest=selftest, dev=dev_run, lock=lock_run, ledger=ledger_run)[cmd](*arg)
