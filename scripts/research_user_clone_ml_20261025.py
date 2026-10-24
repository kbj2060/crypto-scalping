#!/usr/bin/env python3
"""내 매매 복제 자동매매 — ML 진입·청산 행동 복제 + 2배·−5% 규칙. 사전 등록 docs/prereg/user_clone_ml_20261025.md (f1041187).

    python3 scripts/research_user_clone_ml_20261025.py --selfcheck   # S1 원장 재생 · S2 피쳐 인과성 · S3 포지션 피쳐 파리티
    python3 scripts/research_user_clone_ml_20261025.py               # G1~G5 (변형 A·B)

학습 = 실원장 ETH 173왕복(2026-08-01~10-21) · 본 시험 = 2022-01-08~2026-07-31(학습 이전). 산출 tmp/user_clone_ml_20261025/.
"""
import argparse, json, math, pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import research_system_trend_ruleb_20261025 as SYS   # noqa: E402
import research_ml_size_stop_20261025 as MS           # noqa: E402

OUT = SYS.ROOT / "tmp" / "user_clone_ml_20261025"
TRAIN = ("2026-08-01", "2026-10-21 23:55"); TEST = ("2022-01-08", "2026-07-31 23:55"); REF = ("2026-08-01", "2026-10-23 23:55")
FEAT_A = ["s", "sr1h", "sr4h", "sr24h", "sr7d", "rv24h", "rv_ratio", "dist_adv", "trend_same", "hsin", "hcos"]
FEAT_B = FEAT_A[1:]
POS_F = ["unr", "mfe", "mae", "loghold", "sr1h", "rv24h"]
CAP, ADD, STOP, TAKER, THRU, WIN, MAXH = 2.0, 0.0075, 0.05, 4e-4, 1e-4, 12, 168 * 12
HOLD_MED_H = 1.42
_G = {}   # 무작위 대조 작업자가 fork 로 물려받는 큰 배열(피클 안 함)


def mkt(x):
    """봉 i 종가까지 시장 피쳐(방향 무관) + 1~4주 추세 부호."""
    f = MS.bar_feats(x); c = x.c
    f["trend"] = np.sign(sum(np.sign(c / c.shift(k * 288) - 1) for k in (7, 14, 21, 28)))
    f["hsin"] = np.sin(2 * np.pi * f.hour / 24); f["hcos"] = np.cos(2 * np.pi * f.hour / 24)
    return f


def side_feats(f, s):
    """방향 s 로 부호 맞춘 진입 피쳐(s 는 배열)."""
    s = np.asarray(s, float)
    return pd.DataFrame({"s": s, "sr1h": s * f.r1h.values, "sr4h": s * f.r4h.values, "sr24h": s * f.r24h.values, "sr7d": s * f.r7d.values,
                         "rv24h": f.rv24h.values, "rv_ratio": f.rv_ratio.values, "dist_adv": np.where(s > 0, f.lo24.values, f.hi24.values),
                         "trend_same": s * f.trend.values, "hsin": f.hsin.values, "hcos": f.hcos.values})


def pos_feats(P0, s, c_now, mfe, mae, k, sr1h, rv24h):
    """청산 모델 포지션 피쳐 — 학습·시뮬 같은 함수(첫 진입가 기준 원시 가격 변동률, 명목·레버리지 곱 안 함)."""
    return np.nan_to_num(np.array([s * (c_now / P0 - 1), mfe, mae, math.log(5.0 * k), sr1h, rv24h]))


class Logit:
    def __init__(self, X, y):
        self.sc = StandardScaler().fit(X); self.m = LogisticRegression(C=0.3, max_iter=5000).fit(self.sc.transform(X), y)
        self.w = self.m.coef_[0] / self.sc.scale_; self.b = float(self.m.intercept_[0] - (self.m.coef_[0] * self.sc.mean_ / self.sc.scale_).sum())

    def p(self, X):
        return 1 / (1 + np.exp(-(np.asarray(X, float) @ self.w + self.b)))


def ledger_bars(x, L):
    """원장 첫 진입이 든 봉(진입 직전에 끝난 봉) · 청산 시각이 든 봉."""
    ends = (x.index + pd.Timedelta("5min")).values
    a = np.array([int(np.searchsorted(ends, np.datetime64(pd.Timestamp(t, unit="ms")), side="right") - 1) for t in L.entry_time])
    e = np.array([int(np.searchsorted(ends, np.datetime64(pd.Timestamp(t, unit="ms")), side="left")) for t in L.exit_time])
    return a, np.maximum(e, a + 1)


def entry_side(m, f, feats, thr):
    pl = m.p(side_feats(f, np.ones(len(f)))[feats].fillna(0).values); ps = m.p(side_feats(f, -np.ones(len(f)))[feats].fillna(0).values)
    side = np.where(np.maximum(pl, ps) >= thr, np.where(pl >= ps, 1, -1), 0)
    side[~np.isfinite(f.rv7d.values) | ~np.isfinite(f.r7d.values)] = 0
    return side, pl, ps


def fit_entry(x, f, L, feats):
    a, _ = ledger_bars(x, L)
    t0, t1 = x.index.get_indexer([pd.Timestamp(TRAIN[0]), pd.Timestamp(TRAIN[1])])
    rng = np.random.default_rng(0); nb = rng.integers(t0, t1 + 1, 20000); ns = rng.choice([-1, 1], 20000)
    X = pd.concat([side_feats(f.iloc[a], L.s.values)[feats], side_feats(f.iloc[nb], ns)[feats]]).fillna(0).values
    m = Logit(X, np.r_[np.ones(len(a)), np.zeros(len(nb))])
    _, pl, ps = entry_side(m, f, feats, 2.0)
    thr = float(np.quantile(np.r_[pl[t0:t1 + 1], ps[t0:t1 + 1]], 1 - len(L) / (t1 - t0 + 1)))
    return m, thr, entry_side(m, f, feats, thr)[0]


def exit_rows(x, f, L):
    a, e = ledger_bars(x, L); c = x.c.values; X, y = [], []
    for i in range(len(L)):
        s, P0 = L.s.values[i], c[a[i]]; mfe = mae = 0.0
        for j in range(a[i] + 1, e[i] + 1):
            u = s * (c[j] / P0 - 1); mfe, mae = max(mfe, u), min(mae, u)
            X.append(pos_feats(P0, s, c[j], mfe, mae, j - a[i], s * f.r1h.values[j], f.rv24h.values[j])); y.append(int(j == e[i]))
    return np.array(X), np.array(y)


def replay_hold(x, f, L, ex, tau):
    """학습 왕복에 청산 모델(+−5% 손절·168h)을 걸었을 때 보유 시간(h) 중앙."""
    a, _ = ledger_bars(x, L); c, h, l = x.c.values, x.h.values, x.l.values; holds = []
    for i in range(len(L)):
        s, P0 = L.s.values[i], c[a[i]]; mfe = mae = 0.0; j = a[i]
        while True:
            j += 1; k = j - a[i]
            if (s > 0 and l[j] <= P0 * (1 - STOP)) or (s < 0 and h[j] >= P0 * (1 + STOP)) or k >= MAXH or j >= len(c) - 1:
                break
            u = s * (c[j] / P0 - 1); mfe, mae = max(mfe, u), min(mae, u)
            if ex.p(pos_feats(P0, s, c[j], mfe, mae, k, s * f.r1h.values[j], f.rv24h.values[j])[None])[0] >= tau:
                break
        holds.append(k * 5 / 60)
    return float(np.median(holds))


def calib_tau(x, f, L, ex):
    lo, hi = 1e-4, 0.9
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if replay_hold(x, f, L, ex, mid) > HOLD_MED_H else (mid, hi)   # 오래 보유하면 문턱을 낮춘다
    return math.sqrt(lo * hi)


def sim(x, f, fund, side, ex, tau, win, cap_frac=CAP, forced_exit=None):
    """5분 순차 한 코인. side[i] = 봉 i 종가 신호(±1/0). forced_exit = {신호 봉: 청산 봉}(S1). 반환 (일 수익률, 거래 목록)."""
    c, h, l = x.c.values, x.h.values, x.l.values; r1h, rv = f.r1h.values, f.rv24h.values
    i0, i1 = x.index.get_indexer([pd.Timestamp(win[0]), pd.Timestamp(win[1])])
    q, C = 0.0, 1.0; pos = 0; trades = []; eq_day = {}; pend = None; st = {}
    for i in range(i0, i1 + 1):
        o = c[i - 1]
        for rate in fund.get(i, ()):
            C -= q * o * rate
        if pos:
            if st["addq"] and ((pos > 0 and l[i] <= st["addpx"] * (1 - THRU)) or (pos < 0 and h[i] >= st["addpx"] * (1 + THRU))):
                q += st["addq"]; C -= st["addq"] * st["addpx"]; st["addq"] = 0.0; st["addi"] = i
            k = i - st["i"]; why = None
            if (pos > 0 and l[i] <= st["stp"]) or (pos < 0 and h[i] >= st["stp"]):
                px, why = (min(o, st["stp"]) if pos > 0 else max(o, st["stp"])), "stop"
            else:
                u = pos * (c[i] / st["P0"] - 1); st["mfe"], st["mae"] = max(st["mfe"], u), min(st["mae"], u)
                if forced_exit is not None:
                    hit = i >= forced_exit.get(st["sig"], 10**12)
                else:
                    hit = ex.p(pos_feats(st["P0"], pos, c[i], st["mfe"], st["mae"], k, pos * r1h[i], rv[i])[None])[0] >= tau
                if hit or k >= MAXH:
                    px, why = c[i], ("model" if hit else "time")
            if why:
                C += q * px - TAKER * abs(q) * px; trades.append((st["sig"], i, pos, why, C / st["C0"] - 1, st["i"], st["P0"], st.get("addi"), px)); q, pos = 0.0, 0   # 뒤 4칸 = 그림용(체결 봉·첫 진입가·물타기 봉·청산가)
        elif pend is not None:
            s_, P0, k0 = pend; hit = (l[i] <= P0 * (1 - THRU)) if s_ > 0 else (h[i] >= P0 * (1 + THRU))
            if hit or i - k0 >= WIN:
                px = P0 if hit else c[i]; cap = cap_frac * C; q1 = s_ * cap / 3 / px
                C0 = C; C -= q1 * px + (0 if hit else TAKER * abs(q1) * px); q, pos = q1, s_
                st = dict(i=i, sig=k0, P0=px, stp=px * (1 - s_ * STOP), addpx=px * (1 - s_ * ADD), mfe=0.0, mae=0.0, C0=C0)
                st["addq"] = s_ * (2 * cap / 3) / st["addpx"]; pend = None
        if not pos and pend is None and side[i]:
            pend = (int(side[i]), c[i], i)
        eq_day[x.index[i].normalize()] = C + q * c[i]
    eq = pd.Series(eq_day); return eq.pct_change().fillna(eq.iloc[0] - 1.0), trades


def fund_idx(x, sym):
    """펀딩 정산 시각 → 그 시각에 시작하는 봉 위치(그 봉 처리 시작 때 낸다)."""
    f = pd.read_parquet(SYS.OUT / f"rest_fund_{sym}.parquet"); pos = x.index.get_indexer(pd.to_datetime(f.t, unit="ms").dt.floor("5min")); g = {}
    for p, r in zip(pos, f.rate):
        if p > 0:
            g.setdefault(int(p), []).append(float(r))
    return g


def stats(r):
    sh = float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else 0.0
    return {"sharpe": sh, "ann": float((1 + r).prod() ** (365 / max(1, len(r))) - 1), "mdd": SYS.mdd(r), "ci": SYS.block_ci(r)}


def random_side(side, i0, i1, seed):
    """같은 신호 수·같은 롱 비율, 시각만 무작위."""
    rng = np.random.default_rng(seed); s = side[i0:i1 + 1]; n = int((s != 0).sum()); pl = float((s > 0).sum() / max(1, n))
    out = np.zeros_like(side); pick = rng.choice(np.arange(i0, i1 + 1), n, replace=False); out[pick] = np.where(rng.random(n) < pl, 1, -1)
    return out


def _null_job(seed):
    g = _G; return stats(sim(g["x"], g["f"], g["fund"], random_side(g["side"], g["i0"], g["i1"], seed), g["ex"], g["tau"], TEST)[0])["sharpe"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", action="store_true"); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    data = {s: SYS.bars5(s)[0] for s in SYS.COINS}; F = {s: mkt(x) for s, x in data.items()}; FU = {s: fund_idx(x, s) for s, x in data.items()}
    x, f = data["ETH"], F["ETH"]; L = SYS.ledger_trips()
    Xe, ye = exit_rows(x, f, L); ex = Logit(Xe, ye)
    if a.selfcheck:
        aa, ee = ledger_bars(x, L); side = np.zeros(len(x), int); side[aa] = L.s.values
        _, tr = sim(x, f, FU["ETH"], side, ex, 0.5, REF, forced_exit=dict(zip(aa, ee)))
        led = dict(zip(aa, L.s.values)); match = [led[t[0]] == t[2] for t in tr]
        print(f"S1 원장 진입 {len(L)} → 시뮬 거래 {len(tr)}(앞 포지션과 겹친 왕복은 건너뜀) · 방향 일치 {np.mean(match):.2f}")
        assert all(match) and len(tr) >= 0.6 * len(L)
        full = mkt(x); rng = np.random.default_rng(1)
        for i in rng.integers(10000, len(x) - 1, 200):
            cut = mkt(x.iloc[i - 9000:i + 1]).iloc[-1]
            assert np.allclose(cut.values.astype(float), full.iloc[i].values.astype(float), rtol=1e-9, equal_nan=True), i
        print("S2 봉 i 에서 자른 데이터 피쳐 = 전체 피쳐 (200 표본)")
        c = x.c.values; s0, a0 = L.s.values[0], aa[0]; u = s0 * (c[a0 + 1] / c[a0] - 1)
        assert np.allclose(Xe[0], pos_feats(c[a0], s0, c[a0 + 1], max(0, u), min(0, u), 1, s0 * f.r1h.values[a0 + 1], f.rv24h.values[a0 + 1]), atol=1e-12)
        print(f"S3 청산 학습 첫 행 = pos_feats 재계산 · 학습 행 {len(Xe)} · 청산 행 {int(ye.sum())}"); print("SELFCHECK PASS"); return
    tau = calib_tau(x, f, L, ex); print(f"청산 문턱 τ {tau:.4f} (학습 왕복 보유 중앙 {replay_hold(x, f, L, ex, tau):.2f}h)", flush=True)
    rep = {"prereg": "docs/prereg/user_clone_ml_20261025.md@f1041187", "tau": tau, "exit_coef": dict(zip(POS_F, ex.m.coef_[0].round(3).tolist()))}
    i0, i1 = x.index.get_indexer([pd.Timestamp(TEST[0]), pd.Timestamp(TEST[1])])
    for name, feats in (("A", FEAT_A), ("B", FEAT_B)):
        m, thr, side = fit_entry(x, f, L, feats)
        r, tr = sim(x, f, FU["ETH"], side, ex, tau, TEST); st = stats(r)
        yr = {str(y): float((1 + v).prod() - 1) for y, v in r.groupby(r.index.year)}
        _G.update(x=x, f=f, fund=FU["ETH"], side=side, ex=ex, tau=tau, i0=i0, i1=i1)
        with ProcessPoolExecutor(10) as pool:
            nulls = list(pool.map(_null_job, range(100)))
        coins = {}
        for s in ("BTC", "SOL", "XRP"):
            ps = entry_side(m, F[s], feats, thr)[0]; coins[s] = stats(sim(data[s], F[s], FU[s], ps, ex, tau, TEST)[0])["sharpe"]
        rr, trr = sim(x, f, FU["ETH"], side, ex, tau, REF)
        g = {"entry_coef": dict(zip(feats, m.m.coef_[0].round(3).tolist())), "thr": thr, "signals_test": int((side[i0:i1 + 1] != 0).sum()),
             "trades": len(tr), "win_rate": float(np.mean([t[4] > 0 for t in tr])) if tr else None,
             "long_share": float(np.mean([t[2] > 0 for t in tr])) if tr else None, "exit_why": pd.Series([t[3] for t in tr]).value_counts().to_dict(),
             **st, "yearly": yr, "null_p975": float(np.percentile(nulls, 97.5)), "null_median": float(np.median(nulls)), "coins_sharpe": coins,
             "train_period_ref": {**stats(rr), "trades": len(trr), "growth": float((1 + rr).prod())}}
        g["gates"] = {"G1": g["ann"] > 0 and g["ci"][0] > 0, "G2": g["mdd"] >= -0.20, "G3": g["sharpe"] > g["null_p975"],
                      "G4": sum(yr.get(str(y), -1) > 0 for y in (2022, 2023, 2024, 2025)) >= 3, "G5": sum(v > 0 for v in coins.values()) >= 2}
        g["pass"] = all(g["gates"].values()); rep[name] = g
        print(name, json.dumps({k: g[k] for k in ("trades", "win_rate", "sharpe", "ann", "mdd", "gates")}, ensure_ascii=False), flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print(json.dumps(rep, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    sys.exit(main())
