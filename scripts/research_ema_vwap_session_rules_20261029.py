#!/usr/bin/env python3
"""EMA5·EMA20·세션 VWAP 매매 규칙 → B1 일봉 추세 필터 → B2 LightGBM 메타 라벨 (2026-10-29).

사전등록: docs/prereg/ema_vwap_session_rules_20261029.md (8c6e7347, 계산 전 커밋).
원천: data/live/dirshort_panel/{ETH,BTC,SOL,XRP}USDT.parquet (5분, timestamp = 봉 시작, 시가 = 직전 종가).
결정 = 봉 t 마감 값만 · 체결 = 봉 t+1 안에서만(시장가 = c_t, 지정가 = 관통 체결) · 손절 먼저 · 체결 봉엔 익절 안 봄.

    python scripts/research_ema_vwap_session_rules_20261029.py --selftest
    python scripts/research_ema_vwap_session_rules_20261029.py          # (서버) → tmp/ema_vwap_session_20261029/report.json
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from market_ctx import market_session, session_vwap  # noqa: E402

PANEL = ROOT / "data/live/dirshort_panel"
OUT = ROOT / "tmp/ema_vwap_session_20261029"
COINS = ["ETH", "BTC", "SOL", "XRP"]
TRAIN = ("2024-01-01", "2025-12-31")
VAL = ("2026-01-01", "2026-05-31")
TEST = ("2026-06-01", "2026-10-27")
COST_STRICT = (2e-4, 5e-4)     # (메이커, 테이커) 편도 — 바이낸스 USDT-M VIP0
COST_CAND = (0.0, 5e-4)        # USDC 메이커 0%
GRID = list(itertools.product(("A1", "A2"), ("mkt", "e5", "e20", "vw"), (1, 2), (0.0, 1.0, 2.0), (0.0, 1.5), (0, 1)))
KEYS = ("align", "entry", "exit", "tp", "sl", "sflat")
LOOKBACKS = (7, 14, 28, 56, 90)


# ── 지표 ─────────────────────────────────────────────────────────────────
def load(sym: str) -> pd.DataFrame:
    d = pd.read_parquet(PANEL / f"{sym}USDT.parquet").sort_values("timestamp").reset_index(drop=True)
    assert (d.timestamp.diff().dropna() == pd.Timedelta("5min")).all(), f"{sym} 구멍"
    return d


def t5_signal(ts: pd.Series, c: np.ndarray) -> np.ndarray:
    """봉마다 T5(dashboard/trend_rule.py 같은 식): 어제까지 완결된 UTC 일봉 종가 C 로 mean sign(C[d−1] − C[d−1−L])."""
    day = ts.dt.normalize()
    C = pd.Series(c, index=ts).groupby(day.to_numpy()).last()
    s = sum(np.sign(C - C.shift(L)) for L in LOOKBACKS) / len(LOOKBACKS)
    return s.shift(1).reindex(day.to_numpy()).to_numpy()   # 오늘 봉은 어제 종가까지의 신호


def indicators(d: pd.DataFrame) -> dict:
    c, h, lo, v = (d[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    tsec = (d.timestamp.astype("int64") // 10**9).to_numpy()
    sst = np.array([market_session(int(t))[0] for t in tsec])
    vw, vsd = session_vwap(tsec.tolist(), h.tolist(), lo.tolist(), c.tolist(), v.tolist(), key=lambda t: market_session(t)[0])
    cs = pd.Series(c)
    sig = cs.pct_change().rolling(288).std().to_numpy()
    return dict(c=c, h=h, l=lo, v=v, tsec=tsec, sst=sst,
                e5=cs.ewm(span=5, adjust=False).mean().to_numpy(), e20=cs.ewm(span=20, adjust=False).mean().to_numpy(),
                vw=np.array([np.nan if x is None else x for x in vw]), vsd=np.array([np.nan if x is None else x for x in vsd]),
                sig=sig, s1h=sig * math.sqrt(12), last=np.r_[sst[1:] != sst[:-1], True],
                t5=t5_signal(d.timestamp, c))


def setups(I: dict, align: str):
    c, e5, e20, vw = I["c"], I["e5"], I["e20"], I["vw"]
    if align == "A1":
        L, S = (c > vw) & (e5 > e20), (c < vw) & (e5 < e20)
    else:
        L, S = (e5 > e20) & (e20 > vw), (e5 < e20) & (e20 < vw)
    ok = np.isfinite(I["sig"]) & np.isfinite(vw)
    return L & ok, S & ok


# ── 백테스트 커널 ─────────────────────────────────────────────────────────
@njit(cache=True)
def rec(tr, k, ob, eb, t, side, x):
    tr[k, 0] = ob; tr[k, 1] = eb; tr[k, 2] = t; tr[k, 3] = side; tr[k, 4] = x


@njit(cache=True)
def run(h, l, c, L, S, lvl, s1h, last, mkt, exit_mode, tp_k, sl_k, sflat, cm, ct):
    """→ (봉 손익 비율[n], 거래 (주문봉, 진입봉, 청산봉, 방향, 손익)[k×5]). 시가 o_t = c_{t−1}."""
    n = len(c)
    pnl = np.zeros(n)
    tr = np.zeros((n, 5))
    k = 0
    pos = 0; ep = 0.0; tpp = 0.0; slp = 0.0; tsum = 0.0; ob = 0; eb = 0
    pend = 0; plvl = 0.0; ps1h = 0.0; pob = 0
    for t in range(1, n):
        prev = c[t - 1]
        if pos != 0:
            if sl_k > 0 and ((pos > 0 and l[t] <= slp) or (pos < 0 and h[t] >= slp)):
                x = pos * (slp / prev - 1) - ct; pnl[t] += x; tsum += x
                rec(tr, k, ob, eb, t, pos, tsum); k += 1; pos = 0
            elif tp_k > 0 and ((pos > 0 and h[t] > tpp) or (pos < 0 and l[t] < tpp)):
                px = max(tpp, prev) if pos > 0 else min(tpp, prev)   # 시가가 이미 익절가 너머면 시가에 체결
                x = pos * (px / prev - 1) - cm; pnl[t] += x; tsum += x
                rec(tr, k, ob, eb, t, pos, tsum); k += 1; pos = 0
            else:
                x = pos * (c[t] / prev - 1); pnl[t] += x; tsum += x
        elif pend != 0:
            side = pend; fill = False; cost = 0.0
            if mkt:
                ep = prev; fill = True; cost = ct
            elif (side > 0 and l[t] < plvl) or (side < 0 and h[t] > plvl):
                ep = plvl; fill = True; cost = cm
            if fill:
                ob = pob; eb = t; pos = side
                tpp = ep * (1 + side * tp_k * ps1h); slp = ep * (1 - side * sl_k * ps1h)
                if sl_k > 0 and ((side > 0 and l[t] <= slp) or (side < 0 and h[t] >= slp)):
                    x = side * (slp / ep - 1) - cost - ct; pnl[t] += x
                    rec(tr, k, ob, eb, t, side, x); k += 1; pos = 0
                else:
                    x = side * (c[t] / ep - 1) - cost; pnl[t] += x; tsum = x
        pend = 0
        # 봉 t 마감 결정
        if pos != 0:
            brk = (exit_mode == 1 and ((pos > 0 and not L[t]) or (pos < 0 and not S[t]))) or \
                  (exit_mode == 2 and ((pos > 0 and S[t]) or (pos < 0 and L[t]))) or (sflat and last[t])
            if brk:
                pnl[t] -= ct; tsum -= ct
                rec(tr, k, ob, eb, t, pos, tsum); k += 1; pos = 0
        if pos == 0 and t < n - 1:
            for side in (1, -1):
                if (side > 0 and L[t]) or (side < 0 and S[t]):
                    if mkt or (side > 0 and lvl[t] < c[t]) or (side < 0 and lvl[t] > c[t]):
                        pend = side; plvl = lvl[t]; ps1h = s1h[t]; pob = t
    return pnl, tr[:k]


def backtest(I: dict, cfg: tuple, cost, L=None, S=None, allow=None):
    align, entry, ex, tp, sl, sf = cfg
    if L is None:
        L, S = setups(I, align)
    if allow is not None:
        L, S = L & allow, S & allow
    lvl = I["c"] if entry == "mkt" else I[entry]
    return run(I["h"], I["l"], I["c"], L, S, lvl, I["s1h"], I["last"], entry == "mkt", ex, tp, sl, bool(sf), cost[0], cost[1])


# ── 집계·통계 ─────────────────────────────────────────────────────────────
def daily(days: np.ndarray, pnl: np.ndarray) -> pd.Series:
    return pd.Series(pnl, index=days).groupby(level=0).sum() * 1e4   # bp/일


def sharpe(x) -> float:
    x = np.asarray(x, float); s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(365)) if s > 0 else 0.0


def mdd(x) -> float:
    e = np.cumsum(np.asarray(x, float)) / 1e4
    return float((e - np.maximum.accumulate(np.r_[0.0, e])[1:]).min())


def block_boot(x, rng, B=2000, blk=5):
    x = np.asarray(x, float); n = len(x); nb = -(-n // blk)
    st = rng.integers(0, n, (B, nb))
    idx = ((st[:, :, None] + np.arange(blk)) % n).reshape(B, -1)[:, :n]
    m = x[idx].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def cut(s: pd.Series, per) -> pd.Series:
    return s.loc[per[0]:per[1]]


def summ(s: pd.Series, rng=None) -> dict:
    r = dict(days=len(s), mean_bp=round(float(s.mean()), 3), sharpe=round(sharpe(s), 3), mdd=round(mdd(s), 4),
             mean_ex_top3_bp=round(float(s.drop(s.nlargest(3).index).mean()), 3))
    if rng is not None:
        r["ci95"] = [round(v, 3) for v in block_boot(s.to_numpy(), rng)]
    return r


def cscv_pbo(M: np.ndarray, groups: np.ndarray) -> float:
    """M[일, 설정] · groups 0..5 → PBO = IS 최고 설정의 OOS 상대 순위 ≤ .5 비율 (Bailey 외 2017)."""
    G = np.unique(groups); lam = []
    for isg in itertools.combinations(G, len(G) // 2):
        m = np.isin(groups, isg)
        sr = lambda A: A.mean(0) / (A.std(0, ddof=1) + 1e-12)
        best = int(np.argmax(sr(M[m]))); oos = sr(M[~m])
        w = (oos < oos[best]).sum() / len(oos) + .5 / len(oos)
        lam.append(w <= .5)
    return float(np.mean(lam))


def dsr(x, srs_trials) -> float:
    """Deflated Sharpe (Bailey & López de Prado 2014), 일 단위 SR."""
    from scipy.stats import kurtosis, norm, skew
    x = np.asarray(x, float); sr = x.mean() / x.std(ddof=1); N = len(srs_trials); g = .5772156649
    sr0 = np.std(srs_trials, ddof=1) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * math.e)))
    den = math.sqrt(1 - skew(x) * sr + (kurtosis(x, fisher=False) - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr0) * math.sqrt(len(x) - 1) / den))


# ── 메타 라벨 피쳐 (주문 시각 t 값만) ───────────────────────────────────────
def meta_feats(d: pd.DataFrame, I: dict, btc: dict, coin: int) -> pd.DataFrame:
    c, sig = I["c"], I["sig"]; sp = sig * c; cs = pd.Series(c)
    f = dict(dcv=(c - I["vw"]) / sp, d520=(I["e5"] - I["e20"]) / sp, d20v=(I["e20"] - I["vw"]) / sp,
             zv=(c - I["vw"]) / np.where(I["vsd"] > 0, I["vsd"], np.nan))
    for k in ("e5", "e20", "vw"):
        f[f"s_{k}"] = (I[k] - np.r_[np.full(3, np.nan), I[k][:-3]]) / sp
    f["sbar"] = (I["tsec"] - I["sst"]) / 300
    f["sname"] = sess_no(I["sst"])
    f["hour"] = (I["tsec"] % 86400) // 3600; f["dow"] = (I["tsec"] // 86400 + 4) % 7
    for k in (1, 3, 12, 48, 288):
        f[f"r{k}"] = (c / cs.shift(k).to_numpy() - 1) / sig
    f["sig"] = sig; f["sigr"] = cs.pct_change().rolling(48).std().to_numpy() / sig
    lv = np.log1p(I["v"]); f["vz"] = lv - pd.Series(lv).rolling(288).mean().to_numpy()
    tb = d.taker_buy_base.to_numpy(float)
    f["tb"] = tb / np.where(I["v"] > 0, I["v"], np.nan)
    f["tb12"] = pd.Series(tb).rolling(12).sum().to_numpy() / pd.Series(I["v"]).rolling(12).sum().replace(0, np.nan).to_numpy()
    oi = np.log(d.sum_open_interest.ffill()).shift(1)   # 지표 묶음은 한 봉 늦게(라벨 시점 불확실 → 보수적)
    for k in (12, 48, 288):
        f[f"oi{k}"] = oi.diff(k).to_numpy()
    tt = d.sum_toptrader_long_short_ratio.ffill().shift(1)
    f["tt_z"] = ((tt - tt.rolling(288).mean()) / tt.rolling(288).std()).to_numpy()
    f["tlsv"] = d.sum_taker_long_short_vol_ratio.ffill().shift(1).to_numpy()
    f["btc12"] = 0.0 if coin == 1 else btc["r12"]; f["btc48"] = 0.0 if coin == 1 else btc["r48"]
    f["t5"] = I["t5"]; f["coin"] = coin
    return pd.DataFrame({k: np.broadcast_to(np.asarray(v, float), len(c)) for k, v in f.items()})


def sess_no(sst: np.ndarray) -> np.ndarray:
    """세션 시작 초 → 세션 번호(0 아시아·1 유럽·2 미국) — 시작 시각 UTC 시로 가른다(00·07/08·13/14)."""
    hr = (np.asarray(sst) % 86400) // 3600
    return np.where(hr < 5, 0, np.where(hr < 11, 1, 2))


# ── 본 실행 ───────────────────────────────────────────────────────────────
def main():
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261029)
    D = {s: load(s) for s in COINS}
    I = {s: indicators(D[s]) for s in COINS}
    days = {s: D[s].timestamp.dt.normalize().to_numpy() for s in COINS}
    print("지표 완료", flush=True)

    def port(cfg, cost, allow=None, LS=None):
        out, trades = [], {}
        for s in COINS:
            L, S = (None, None) if LS is None else LS[s]
            p, tr = backtest(I[s], cfg, cost, L, S, None if allow is None else allow[s])
            out.append(daily(days[s], p)); trades[s] = tr
        P = pd.concat(out, axis=1).fillna(0.0)
        P.columns = COINS
        return P.mean(1), P, trades

    rep = {"prereg": "docs/prereg/ema_vwap_session_rules_20261029.md@8c6e7347", "grid": len(GRID),
           "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
           "future_rows_used_for_entry": False, "funding": "무시"}

    def t5_sets(cfg):   # B1: 정렬 ∧ T5 방향
        return {s: tuple(a & b for a, b in zip(setups(I[s], cfg[0]), (I[s]["t5"] > 0, I[s]["t5"] < 0))) for s in COINS}

    def port_b1(cfg, cost, allow=None):
        return port(cfg, cost, allow, t5_sets(cfg))

    def grid_stage(name, LSfn):
        M, rows = [], []
        for cfg in GRID:
            s, P, tr = port(cfg, COST_CAND, None, LSfn(cfg))
            M.append(s); rows.append(dict(zip(KEYS, cfg), train_sharpe=sharpe(cut(s, TRAIN)), train_bp=float(cut(s, TRAIN).mean()),
                                          trades=sum(len(tr[c]) for c in COINS)))
        tr_ = cut(pd.concat(M, axis=1).fillna(0.0), TRAIN)
        grp = np.minimum(((tr_.index.year - 2024) * 12 + tr_.index.month - 1) // 4, 5)
        best = int(np.argmax([r["train_sharpe"] for r in rows]))
        res = dict(best=rows[best], pbo=cscv_pbo(tr_.to_numpy(), grp),
                   dsr=dsr(tr_.iloc[:, best].to_numpy(), (tr_.mean(0) / tr_.std(0, ddof=1)).to_numpy()),
                   n_positive_train=int(sum(r["train_sharpe"] > 0 for r in rows)),
                   top10=sorted(rows, key=lambda r: -r["train_sharpe"])[:10])
        pd.DataFrame(rows).to_csv(OUT / f"grid_{name}.csv", index=False)
        print(name, json.dumps(res["best"], ensure_ascii=False), "PBO", res["pbo"], "DSR", round(res["dsr"], 3), flush=True)
        return GRID[best], res

    cfgA, rep["A"] = grid_stage("A", lambda cfg: None)
    cfgB1, rep["B1"] = grid_stage("B1", t5_sets)

    # B2: B1 설정 학습 거래 → 메타 라벨
    btc = {k: (I["BTC"]["c"] / pd.Series(I["BTC"]["c"]).shift(k).to_numpy() - 1) / I["BTC"]["sig"] for k in (12, 48)}
    btc = {f"r{k}": v for k, v in btc.items()}
    F = {s: meta_feats(D[s], I[s], btc, i) for i, s in enumerate(COINS)}
    _, _, trB1 = port_b1(cfgB1, COST_CAND)
    Xs, ys = [], []
    for s in COINS:
        tr = trB1[s]; ob = tr[:, 0].astype(int)
        m = (D[s].timestamp.iloc[tr[:, 2].astype(int)].to_numpy() <= np.datetime64(TRAIN[1] + "T23:55"))   # 청산까지 학습 안
        m &= D[s].timestamp.iloc[ob].to_numpy() >= np.datetime64(TRAIN[0])
        Xs.append(F[s].iloc[ob[m]]); ys.append(tr[m, 4] > 0)
    X, y = pd.concat(Xs), np.concatenate(ys)
    seeds = [int(x) for x in rng.integers(1, 2**31 - 1, 5)]
    models = [lgb.LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=31, min_child_samples=200, subsample=.5,
                                 subsample_freq=1, colsample_bytree=.8, reg_lambda=10.0, n_jobs=4, verbose=-1, random_state=sd).fit(X, y)
              for sd in seeds]
    prob = {s: np.mean([m.predict_proba(F[s])[:, 1] for m in models], 0) for s in COINS}
    rep["B2"] = dict(n_train_trades=int(len(y)), base_rate=float(y.mean()), seeds=seeds,
                     importance=dict(sorted(zip(X.columns, np.mean([m.feature_importances_ for m in models], 0).round(1)),
                                            key=lambda kv: -kv[1])[:12]))
    thr = {}
    for th in (.50, .55, .60):
        s, _, _ = port_b1(cfgB1, COST_CAND, {c: prob[c] >= th for c in COINS})
        thr[th] = sharpe(cut(s, VAL))
    th_b2 = max(thr, key=thr.get); rep["B2"]["val_sharpe_by_theta"] = thr; rep["B2"]["theta"] = th_b2
    print("B2", rep["B2"], flush=True)

    arms = {"A": lambda cost, LS=None: port(cfgA, cost, None, LS),
            "B1": lambda cost, LS=None: port(cfgB1, cost, None, LS if LS is not None else t5_sets(cfgB1)),
            "B2": lambda cost, LS=None: port(cfgB1, cost, {c: prob[c] >= th_b2 for c in COINS}, LS if LS is not None else t5_sets(cfgB1))}
    # 검증으로 팔 고르기 (시험은 아직 안 봄)
    val = {a: summ(cut(f(COST_CAND)[0], VAL)) for a, f in arms.items()}
    pick = max(val, key=lambda a: val[a]["sharpe"]); rep["val"] = val; rep["pick"] = pick
    print("검증", json.dumps(val, ensure_ascii=False), "→", pick, flush=True)

    # ── 시험 1회 채점 ──
    leg = pd.concat([daily(days[s], np.r_[0.0, I[s]["c"][1:] / I[s]["c"][:-1] - 1]) for s in COINS], axis=1).fillna(0).mean(1)
    test = {}
    for a, f in arms.items():
        sS, PS, trS = f(COST_STRICT); sC, PC, _ = f(COST_CAND)
        T = cut(sS, TEST); TL = cut(leg, TEST)
        test[a] = dict(strict=summ(T, rng), cand=summ(cut(sC, TEST), rng),
                       cand_coin_mean_bp={c: round(float(cut(PC[c], TEST).mean()), 3) for c in COINS},
                       minus_long=dict(mean=round(float((T - TL).mean()), 3), ci95=block_boot((T - TL).to_numpy(), rng)),
                       minus_short=dict(mean=round(float((T + TL).mean()), 3), ci95=block_boot((T + TL).to_numpy(), rng)),
                       trades_test=int(sum(((D[c].timestamp.iloc[trS[c][:, 1].astype(int)] >= TEST[0])).sum() for c in COINS)),
                       val_strict=summ(cut(sS, VAL)), train_strict=summ(cut(sS, TRAIN)), train_cand=summ(cut(sC, TRAIN)))
    # 원형 밀기 위약 (고른 팔만): 정렬 상태 배열만 1~30일 밀기
    cfgP = cfgA if pick == "A" else cfgB1
    null_s, null_c = [], []
    for _ in range(200):
        k = int(rng.integers(288, 30 * 288 + 1))
        base = t5_sets(cfgP) if pick != "A" else {s: setups(I[s], cfgP[0]) for s in COINS}
        LS = {s: (np.roll(base[s][0], k), np.roll(base[s][1], k)) for s in COINS}
        null_s.append(sharpe(cut(arms[pick](COST_STRICT, LS)[0], TEST))); null_c.append(sharpe(cut(arms[pick](COST_CAND, LS)[0], TEST)))
    t = test[pick]
    t["null_pct_strict"] = float((np.array(null_s) < t["strict"]["sharpe"]).mean() * 100)
    t["null_pct_cand"] = float((np.array(null_c) < t["cand"]["sharpe"]).mean() * 100)
    t["null95_strict"] = float(np.quantile(null_s, .95))
    rep["test"] = test
    v = rep["val"][pick]
    rep["verdict_strict"] = bool(v["sharpe"] > 0 and t["strict"]["ci95"][0] > 0 and t["strict"]["sharpe"] > t["null95_strict"]
                                 and t["strict"]["mdd"] > -.20 and t["minus_long"]["mean"] > 0 and t["minus_short"]["mean"] > 0)
    rep["verdict_candidate"] = bool(v["mean_bp"] > 0 and t["cand"]["mean_bp"] > 0 and sum(x > 0 for x in t["cand_coin_mean_bp"].values()) >= 2
                                    and t["null_pct_cand"] >= 50)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("pick", "verdict_strict", "verdict_candidate")}, ensure_ascii=False))
    print(json.dumps(test, ensure_ascii=False, indent=1, default=str))


def selftest():
    n = 12
    c = np.array([100, 100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100.0])
    h, l = c + .5, c - .5
    L = np.zeros(n, bool); L[2:6] = True; S = np.zeros(n, bool)
    last = np.zeros(n, bool); s1h = np.full(n, .01)
    # 시장가: 봉 2 마감 결정 → 봉 3 시가(= c2 101) 진입 · 봉 6 마감에 L 깨짐 → c6 청산 · 봉 손익 = 봉 단순수익 합 − 왕복 테이커
    p, tr = run(h, l, c, L, S, c, s1h, last, True, 1, 0.0, 0.0, False, 0.0, 5e-4)
    assert len(tr) == 1 and tr[0, 1] == 3 and tr[0, 2] == 6, tr
    assert abs(p.sum() - (np.sum(c[3:7] / c[2:6] - 1) - 1e-3)) < 1e-12 and abs(tr[0, 4] - p.sum()) < 1e-12, (p.sum(), tr)
    # 지정가 관통: 평평한 가격(저가 99.5) — 레벨 99.5(저가와 같음)는 안 채워지고 99.6 은 99.6 에 채워짐
    f = np.full(n, 100.0)
    _, tr2 = run(f + .5, f - .5, f, L, S, f - .5, s1h, last, False, 1, 0.0, 0.0, False, 0.0, 5e-4)
    assert len(tr2) == 0, tr2
    _, tr3 = run(f + .5, f - .5, f, L, S, f - .4, s1h, last, False, 1, 0.0, 0.0, False, 2e-4, 5e-4)
    assert len(tr3) == 1 and tr3[0, 1] == 3 and abs(tr3[0, 4] - ((100 / 99.6 - 1) - 2e-4 - 5e-4)) < 1e-12, tr3
    # 손절: 체결 봉에서 저가가 손절 밑이면 그 봉에 손절(익절보다 먼저)
    _, tr4 = run(f + .5, f - .5, f, L, S, f - .4, np.full(n, .001), last, False, 1, 1.0, 1.0, False, 0.0, 5e-4)
    assert tr4[0, 1] == tr4[0, 2] == 3 and tr4[0, 4] < 0, tr4
    # 익절: 메이커 비용으로 tp 가격에 청산
    _, tr6 = run(h, l, c, L, S, c, np.full(n, .005), last, True, 1, 1.0, 0.0, False, 0.0, 5e-4)
    assert abs(tr6[0, 4] - (102 / 101 - 1 - 5e-4)) < 1e-12 and tr6[0, 2] == 4, tr6   # 봉 3 종가 102 > tp 101.505 → 봉 4 시가 102
    # 세션 정리: 봉 4 가 세션 마지막 → 봉 4 마감 청산 · 봉 5 에 다시 진입
    last2 = last.copy(); last2[4] = True
    _, tr5 = run(h, l, c, L, S, c, s1h, last2, True, 1, 0.0, 0.0, True, 0.0, 5e-4)
    assert len(tr5) == 2 and tr5[0, 2] == 4 and tr5[1, 1] == 5, tr5
    # 결정은 봉 t 마감까지만: 봉 9 이후 가격을 바꿔도 그 전 봉 손익 불변
    c2 = c.copy(); c2[9:] *= 1.1
    pa, _ = run(c2 + .5, c2 - .5, c2, L, S, c2, s1h, last, True, 1, 0.0, 0.0, False, 0.0, 5e-4)
    assert np.allclose(pa[:9], p[:9]), (pa, p)
    # T5: 어제까지 종가만 — 오늘 폭락해도 오늘 신호 그대로
    ts = pd.Series(pd.date_range("2024-01-01", periods=200 * 288, freq="5min"))
    cc = 100 * 1.0001 ** np.arange(len(ts)); sg = t5_signal(ts, cc)
    assert np.isnan(sg[:91 * 288]).all() and sg[-1] == 1.0
    cc2 = cc.copy(); cc2[-288:] = 1.0
    assert t5_signal(ts, cc2)[-1] == 1.0
    assert list(sess_no(np.array([0, 7 * 3600, 8 * 3600, 13 * 3600 + 1800, 14 * 3600 + 1800]))) == [0, 1, 1, 2, 2]
    # 세션 VWAP 은 세션 시작에서 다시 센다(대시보드 함수 그대로)
    t0 = int(pd.Timestamp("2026-01-05").timestamp()); st = market_session(t0 + 8 * 3600 + 600)[0]
    tt = [st - 600, st - 300, st, st + 300]
    vw, _ = session_vwap(tt, [11, 11, 21, 21], [9, 9, 19, 19], [10, 10, 20, 20], [1, 1, 1, 1], key=lambda t: market_session(t)[0])
    assert vw[2] == 20 and vw[1] == 10, vw
    # PBO: 정보 없는 무작위 설정 → .5 근처 / 한 설정이 늘 최고 → 0
    r = np.random.default_rng(0); M = r.normal(size=(600, 50)); g = np.repeat(np.arange(6), 100)
    assert .2 < cscv_pbo(M, g) < .8
    M[:, 7] += 1.0; assert cscv_pbo(M, g) == 0.0
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
