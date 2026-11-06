#!/usr/bin/env python3
"""1시간 방향 신호 × 1시간 박자 × 지정가 (사전등록 docs/prereg/dir_hourly_maker_execution_20261107.md).
    python scripts/research_dir_hourly_maker_20261107.py --selftest
    python scripts/research_dir_hourly_maker_20261107.py --checks   # 정렬·양성·음성 대조 (주 판정 채점 안 함)
    python scripts/research_dir_hourly_maker_20261107.py --judge    # 1회만
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import liqlab_harness_20261028 as H  # noqa: E402

COINS = ["ETH", "BTC", "SOL", "XRP"]
OUT = Path("/home/kbj20/crypto-scalping/tmp/dir_hourly_maker_20261107")
HOLD, PHASES, WARM = 12, 24, 288 * 14   # 수정 1: 위상 24개 × 2시간 간격 = 12위상 × 1시간 박자. 한 위상을 1시간 간격으로 두면 청산(t+12~13)이 다음 결정을 막아 2시간 박자가 된다
STRICT, CAND = (2.0, 5.0, 5.0), (0.0, 5.0, 5.0)   # 메이커 · 대체 시장가 · 시장가


def set_cost(c):
    H.MAKER_BP, H.MAKER_FALLBACK_BP, H.TAKER_BP = c


def load(coin):
    p = pd.read_parquet(f"data/live/dirshort_panel/{coin}USDT.parquet", columns=["timestamp", "high", "low", "close"]).set_index("timestamp").sort_index()
    s = pd.read_parquet(f"data/live/{coin.lower()}_dirshort_oos_seed.parquet").set_index("ts").sort_index()
    idx = pd.date_range(s.index.min() - pd.Timedelta("5min"), s.index.max() + pd.Timedelta("3h"), freq="5min")
    d = p.reindex(idx)
    assert d.close.notna().mean() > .999, (coin, "가격 결측", d.close.notna().mean())
    d.attrs["price_nan"] = float(d.close.isna().mean())
    d = d.ffill()
    d["o"] = d.close.shift(1).fillna(d.close)
    # 결정 t 는 행 τ = t−5분의 끝이다
    d["p"] = s.p1h.reindex(d.index + pd.Timedelta("5min")).to_numpy()
    d["p15"] = s.p15m.reindex(d.index + pd.Timedelta("5min")).to_numpy()
    return d


def thresholds(p):
    """직전까지 확장 20·80 분위 -- shift(1) 로 지금 값 제외, 앞 WARM 행은 NaN."""
    lo = p.expanding(WARM).quantile(.2).shift(1)
    hi = p.expanding(WARM).quantile(.8).shift(1)
    return lo, hi


def signal(p):
    lo, hi = thresholds(p)
    return np.where(p >= hi, 1, np.where(p <= lo, -1, 0)).astype(int)


def phase_sig(sig, ph, step=PHASES):   # step=PHASES=24 행(2시간) -- 같은 분·시 위상의 짝수·홀수 시각을 나눠 모든 1시간 결정을 살린다
    out = np.zeros_like(sig)
    out[ph::step] = sig[ph::step]
    return out


def trades(d, sig, hold=HOLD, mode="maker"):
    """위상별 거래를 모은다. 위상마다 한 포지션(보유 중 신호 버림)."""
    rows = []
    for ph in range(PHASES):
        t = H.simulate(phase_sig(sig, ph), d.o.to_numpy(), d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(), H=hold, mode=mode)
        for i, sd, g, n in zip(t["i"], t["side"], t["gross"], t["net"]):
            rows.append((d.index[int(i)], int(sd), float(g), float(n)))
    return pd.DataFrame(rows, columns=["t", "side", "gross", "net"])


def stats(tr):
    if len(tr) < 10:
        return {"n": len(tr)}
    day = tr.t.dt.floor("D").astype("int64").to_numpy()
    dist = H.cluster_boot_mean(tr.net.to_numpy(), day)
    return {"n": len(tr), "days": int(len(np.unique(day))), "mean_net": float(tr.net.mean()), "mean_gross": float(tr.gross.mean()), "ci95": H.ci95(dist),
            "hit": float((tr.net > 0).mean()), "mde80": H.mde80(tr.net.to_numpy(), day)}


def null_means(D, S, P, hold, mode, kind, B=200, seed=11):
    """코인·위상 묶음 위약 평균. 수정 1: 코인 간 동조를 보존한다.
    circular = 네 코인에 같은 이동값 · random = 같은 시각의 신호는 같은 무작위 시각으로(층 = 공통 BTC 시각×변동성)."""
    rng = np.random.default_rng(seed)
    res = np.empty(B)
    n_min = min(len(d) for d in D.values())
    st = D["BTC"]["strata"].to_numpy()
    for b in range(B):
        tot, n = 0.0, 0
        if kind == "circular":
            shift = int(rng.integers(288, n_min - 288))
        else:
            fired = np.unique(np.concatenate([np.flatnonzero(S[c]) for c in D]))
            to = {int(t): P[st[t]][rng.integers(len(P[st[t]]))] for t in fired}
        for c, d in D.items():
            sig = S[c]
            if kind == "circular":
                r = np.roll(sig, shift)
            else:
                r = np.zeros_like(sig)
                for t in np.flatnonzero(sig):
                    r[to[int(t)]] = sig[t]
            for ph in range(PHASES):
                x = H.simulate(phase_sig(r, ph), d.o.to_numpy(), d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(), H=hold, mode=mode)["net"]
                tot += x.sum(); n += len(x)
        res[b] = tot / max(n, 1)
    return res


def prepare():
    D, S = {}, {}
    for c in COINS:
        d = load(c)
        sig = signal(d.p)
        sig[-(HOLD + 40):] = 0
        rv = np.log(d.close).diff().rolling(288).std().shift(1).to_numpy()
        q = np.nanquantile(rv[:WARM + 288], [1 / 3, 2 / 3])   # 층 경계 = 예열 구간
        d["strata"] = H.strata_of(d.index.hour.to_numpy(), rv, q)
        D[c], S[c] = d, sig
    # 공통 후보: 모든 코인에 있는 행(가장 짧은 코인 기준), 층 = BTC
    n_min = min(len(d) for d in D.values())
    assert all(D[c].index[0] == D["BTC"].index[0] for c in COINS), "코인 시작 행이 다름"
    elig = np.ones(n_min, bool); elig[:WARM] = False; elig[-(HOLD + 40):] = False
    st = D["BTC"]["strata"].to_numpy()[:n_min]
    P = {k: np.flatnonzero(elig & (st == k)) for k in np.unique(D["BTC"]["strata"].to_numpy())}
    for k, v in P.items():
        if not len(v):                                       # 드문 층이면 전체 후보로 대체
            P[k] = np.flatnonzero(elig)
    return D, S, P


def checks():
    H.selfcheck()
    D, S, _ = prepare()
    rep = {}
    for c, d in D.items():
        fwd = (d.close.shift(-HOLD) / d.close - 1)
        m = d.p.notna()
        ic = lambda sh: float(d.p[m].shift(sh).corr(fwd[m], method="spearman"))
        rep[c] = {"ic_aligned": ic(0), "ic_shift_past1": ic(1), "ic_shift_future1": ic(-1), "signal_share": float((S[c] != 0).mean()), "price_nan": d.attrs.get("price_nan")}
    # 양성·음성 (실제 가격, ETH): 미래 부호 70% / 예측 섞음
    set_cost(STRICT)
    d, rng = D["ETH"], np.random.default_rng(5)
    fwd = np.sign((d.close.shift(-HOLD - 1) / d.close - 1).fillna(0).to_numpy())
    fire = (S["ETH"] != 0)
    good = np.where(fire, np.where(rng.random(len(d)) < .7, fwd, -fwd), 0).astype(int)
    tr = trades(d, good)
    rep["positive_control"] = stats(tr)
    assert rep["positive_control"]["ci95"][0] > 0, rep["positive_control"]
    shuf = np.where(fire, S["ETH"][rng.permutation(len(d))], 0).astype(int)
    rep["negative_control"] = stats(trades(d, shuf))
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    for c in COINS:   # 수정 1: 씨앗 기록(1시간 IC .05~.07 안팎)과 크게 다르면 중단 · 미래로 민 판이 정렬판보다 낮아야(되돌림 지문)
        assert .02 < rep[c]["ic_aligned"] < .10 and rep[c]["ic_shift_future1"] < rep[c]["ic_aligned"], ("IC 이상", c, rep[c])
    assert abs(rep["negative_control"]["mean_gross"]) < 2.0, rep["negative_control"]
    json.dump(rep, open(OUT / "checks.json", "w"), ensure_ascii=False, indent=1)


def judge():
    vp = OUT / "verdict.json"
    if vp.exists():
        sys.exit(f"판정은 1회만 -- {vp}")
    assert (OUT / "checks.json").exists(), "--checks 먼저"
    D, S, P = prepare()
    rep = {"prereg": "docs/prereg/dir_hourly_maker_execution_20261107.md", "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    for tag, cost in (("strict", STRICT), ("candidate", CAND)):
        set_cost(cost)
        per = {c: trades(D[c], S[c]) for c in COINS}
        all_ = pd.concat([v.assign(coin=c) for c, v in per.items()]).sort_values("t")
        mid = all_.t.iloc[len(all_) // 2]
        rep[tag] = {"fill_rate_of_decisions": len(all_) / max(1, sum(int((S[c] != 0).sum()) for c in COINS) / 12), "pooled": stats(all_), "coins": {c: stats(v) for c, v in per.items()}, "first_half": float(all_[all_.t < mid].net.mean()), "second_half": float(all_[all_.t >= mid].net.mean()),
                    "months": {str(k): round(float(v), 2) for k, v in all_.groupby(all_.t.dt.to_period("M")).net.mean().items()}}
        day_sum = all_.groupby(all_.t.dt.floor("D")).net.sum().sort_values(ascending=False)
        rep[tag]["mean_ex_top3days"] = float(all_[~all_.t.dt.floor("D").isin(day_sum.index[:3])].net.mean())
        nr = null_means(D, S, P, HOLD, "maker", "random")
        nc = null_means(D, S, P, HOLD, "maker", "circular")
        rep[tag]["null_random_pct"] = float((nr < all_.net.mean()).mean() * 100)
        rep[tag]["null_circular_pct"] = float((nc < all_.net.mean()).mean() * 100)
    set_cost(STRICT)
    # 모델−롱·모델−숏: 같은 진입 시각에 항상 롱/숏
    excess = {}
    per = pd.concat([trades(D[c], S[c]).assign(coin=c) for c in COINS])
    for nm, f in (("always_long", lambda s: np.where(s != 0, 1, 0)), ("always_short", lambda s: np.where(s != 0, -1, 0))):
        x = pd.concat([trades(D[c], f(S[c]).astype(int)) for c in COINS])
        excess[nm] = {"mean_net": float(x.net.mean()), "n": len(x), "model_minus": float(per.net.mean() - x.net.mean())}
    rep["vs_market"] = excess
    # 보조: 시장가 · 15분
    tk = pd.concat([trades(D[c], S[c], mode="taker") for c in COINS])
    rep["aux_taker_5bp"] = stats(tk)
    s15 = {c: signal(D[c].p15) for c in COINS}
    for c in COINS:
        s15[c][-(HOLD + 40):] = 0
    t15 = pd.concat([_trades_step(D[c], s15[c], 3, 6) for c in COINS])
    rep["aux_15m_strict"] = stats(t15)
    s, c = rep["strict"], rep["candidate"]
    g1 = s["pooled"]["ci95"][0] > 0
    g2 = s["null_circular_pct"] > 95 and s["null_random_pct"] > 95
    g3 = s["first_half"] > 0 and s["second_half"] > 0
    g4 = sum(v.get("mean_net", -1) > 0 for v in s["coins"].values()) >= 3
    inconclusive = (not g1) and s["pooled"]["ci95"][1] > 0 and s["pooled"]["mde80"] > 2 * 5.9
    rep["verdict"] = {"strict": "통과" if g1 and g2 and g3 and g4 else "판정 불능" if inconclusive else "불통과", "gates": dict(G1=g1, G2=g2, G3=g3, G4=g4),
                      "candidate": "통과 후보" if c["pooled"]["mean_net"] > 0 and c["first_half"] > 0 and c["second_half"] > 0 and sum(v.get("mean_net", -1) > 0 for v in c["coins"].values()) >= 2 and c["null_random_pct"] >= 50 else "불통과"}
    json.dump(rep, open(vp, "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


def _trades_step(d, sig, hold, step):
    rows = []
    for ph in range(step):
        t = H.simulate(phase_sig(sig, ph, step), d.o.to_numpy(), d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(), H=hold, mode="maker")
        rows += [(d.index[int(i)], int(sd), float(g), float(n)) for i, sd, g, n in zip(t["i"], t["side"], t["gross"], t["net"])]
    return pd.DataFrame(rows, columns=["t", "side", "gross", "net"])


def selftest():
    # 문턱은 지금 값을 안 본다: 지금 값을 극단으로 바꿔도 그 행의 문턱은 그대로
    global WARM
    w0, WARM = WARM, 20
    p = pd.Series(np.arange(100.0))
    a, b = thresholds(p)
    q = p.copy(); q.iloc[50] = 1e9
    a2, b2 = thresholds(q)
    assert a[50] == a2[50] and b[50] == b2[50] and np.isnan(a[19]) and not np.isnan(a[20])
    s = signal(pd.Series(np.r_[np.arange(30.0), 1000.0, -1000.0, 15.0]))
    assert s[30] == 1 and s[31] == -1 and s[32] == 0 and s[:20].sum() == 0
    WARM = w0
    # 위상: 24행마다 하나 · 수정 1: 1시간 간격 결정이 하나도 안 버려진다
    sig = np.ones(72, int)
    assert phase_sig(sig, 5).sum() == 3 and phase_sig(sig, 5)[5] == 1 and phase_sig(sig, 5)[6] == 0
    set_cost(STRICT)
    idx0 = pd.date_range("2026-01-01", periods=120, freq="5min")
    c0 = 100 + np.sin(np.arange(120)) * .5
    d0 = pd.DataFrame({"o": np.r_[100, c0[:-1]], "high": c0 + 1, "low": c0 - 1, "close": c0}, index=idx0)
    x0 = np.zeros(120, int); x0[[0, 12, 24, 36]] = 1
    assert sorted(trades(d0, x0).t.tolist()) == [idx0[i] for i in (0, 12, 24, 36)], "1시간 박자 결정이 버려짐"
    # 손계산: 비용 2/5 -- 지정가 100, 다음 봉 저가 99 체결 -> 12봉째 종가 101 -> 그다음 봉 고가 102 > 101 지정가 체결 -> gross 100bp, net 96bp
    set_cost(STRICT)
    n = 40
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    c = np.full(n, 100.0); c[12:] = 101.0
    d = pd.DataFrame({"o": np.r_[100, c[:-1]], "high": c + 1, "low": c - 1, "close": c}, index=idx)
    x = np.zeros(n, int); x[0] = 1
    t = trades(d, x)
    assert len(t) == 1 and abs(t.gross[0] - 100) < 1e-9 and abs(t.net[0] - 96) < 1e-9, t
    print("selftest OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    elif "--checks" in a:
        checks()
    elif "--judge" in a:
        judge()
    else:
        sys.exit(__doc__)
