#!/usr/bin/env python3
"""방향 예측 상위 10% 롱 (사전등록 docs/prereg/dir_top_decile_long_20261107.md).
    python scripts/research_dir_top_decile_20261107.py --selftest
    python scripts/research_dir_top_decile_20261107.py --checks
    python scripts/research_dir_top_decile_20261107.py --judge     # 1회만
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import liqlab_harness_20261028 as H  # noqa: E402

COINS = ["ETH", "SOL", "XRP"]
OUT = Path("/home/kbj20/crypto-scalping/tmp/dir_top_decile_20261107")
HOLD, PHASES, WARM, NULL, SEED = 48, 96, 288 * 90, 200, 20261107
STRICT, CAND = (2.0, 5.0, 5.0), (0.0, 5.0, 5.0)


def set_cost(c):
    H.MAKER_BP, H.MAKER_FALLBACK_BP, H.TAKER_BP = c


def load(coin):
    s = pd.read_parquet(f"data/live/{coin.lower()}_dir4h_oos_seed.parquet").set_index("ts").sort_index().pred
    p = pd.read_parquet(f"data/live/dirshort_panel/{coin}USDT.parquet", columns=["timestamp", "high", "low", "close"]).set_index("timestamp").sort_index()
    idx = pd.date_range(s.index.min() - pd.Timedelta("5min"), s.index.max() + pd.Timedelta("5h"), freq="5min")
    d = p.reindex(idx)
    nan = float(d.close.isna().mean())
    assert nan < .002, (coin, "가격 결측", nan)
    d = d.ffill()
    d["o"] = d.close.shift(1).fillna(d.close)
    d["p"] = s.reindex(d.index).to_numpy()   # 수정 1: 4h 씨앗 ts = 봉 시작(값은 봉 끝 확정, 15분·1시간 씨앗과 다름) -> 행 τ 에 그대로
    return d


def top_signal(p, q=.9, sign=1):
    thr = p.expanding(WARM).quantile(q).shift(1)     # 지금 값 제외
    m = (p >= thr) if sign > 0 else (p <= thr)
    return (m.to_numpy() & thr.notna().to_numpy()).astype(int) * sign


def phase_sig(sig, ph):
    out = np.zeros_like(sig)
    out[ph::PHASES] = sig[ph::PHASES]
    return out


def trades(d, sig, mode="maker"):
    o, h, l, c = d.o.to_numpy(), d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()
    rows = []
    for ph in range(PHASES):
        t = H.simulate(phase_sig(sig, ph), o, h, l, c, H=HOLD, mode=mode)
        rows += [(int(i), float(g), float(n)) for i, g, n in zip(t["i"], t["gross"], t["net"])]
    r = pd.DataFrame(rows, columns=["i", "gross", "net"])
    r["t"] = d.index[r.i.to_numpy()] if len(r) else pd.DatetimeIndex([])
    return r


def stats(tr):
    if len(tr) < 10:
        return {"n": len(tr)}
    day = tr.t.dt.floor("D").astype("int64").to_numpy()
    dist = H.cluster_boot_mean(tr.net.to_numpy(), day)
    return {"n": len(tr), "days": int(len(np.unique(day))), "mean_net": float(tr.net.mean()), "mean_gross": float(tr.gross.mean()), "ci95": H.ci95(dist), "hit": float((tr.net > 0).mean())}


def prepare():
    D = {c: load(c) for c in COINS}
    S = {c: top_signal(D[c].p) for c in COINS}
    for c in COINS:
        S[c][-(HOLD + 80):] = 0
    return D, S


def null_means(D, S, B=NULL):
    """코인 공통 이동값 원형 위약(합침 평균). 코인마다 길이가 달라 가장 짧은 길이 안에서 민다."""
    rng = np.random.default_rng(SEED)
    n_min = min(len(d) for d in D.values())
    out = np.empty(B)
    for b in range(B):
        sh = int(rng.integers(288 * 30, n_min - 288 * 30))
        tot, n = 0.0, 0
        for c, d in D.items():
            x = trades(d, np.roll(S[c], sh))
            tot += x.net.sum(); n += len(x)
        out[b] = tot / max(n, 1)
    return out


def checks():
    H.selfcheck()
    D, S = prepare()
    rep = {}
    set_cost(STRICT)
    for c, d in D.items():
        fwd = d.close.shift(-HOLD) / d.close - 1
        m = d.p.notna() & fwd.notna()
        p = d.p[m]
        rep[c] = {"rows": int(m.sum()), "ic_aligned": float(p.corr(fwd[m], method="spearman")), "ic_future1": float(p.shift(-1).corr(fwd[m], method="spearman")), "ic_past1": float(p.shift(1).corr(fwd[m], method="spearman")),
                  "signal_share": float((S[c] != 0).mean()), "first_signal": str(d.index[np.flatnonzero(S[c])[0]])}
        r1 = np.log(d.close).diff()
        rep[c]["corr_this_bar"] = float(d.p.corr(r1))             # 행 τ 봉 수익: 예측이 본 값(되돌림이라 음수)
        rep[c]["corr_next_bar"] = float(d.p.corr(r1.shift(-1)))   # 체결 봉: 예측이 못 본 값(≈0)
    # 음성: 코인 공통 날짜 섞기(사전등록) -- 신호만 섞고 비용 전 평균을 본다
    n_min = min(len(x) for x in D.values())
    rng = np.random.default_rng(SEED)
    block = np.array_split(np.arange(n_min), n_min // 288)
    perm = np.concatenate([block[i] for i in rng.permutation(len(block))])
    neg = pd.concat([trades(D[c].iloc[:n_min], S[c][:n_min][perm]) for c in COINS])
    rep["negative_control_shuffled_days"] = stats(neg)
    # 양성(ETH 만, 시간 절약)
    d = D["ETH"]
    fwd = (d.close.shift(-HOLD - 1) / d.close - 1).to_numpy()
    good = (fwd >= np.nanquantile(fwd, .9)).astype(int)
    good[-(HOLD + 80):] = 0
    rep["positive_control"] = stats(trades(d, good))
    assert rep["positive_control"]["ci95"][0] > 0, rep["positive_control"]
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    for c in COINS:
        # 수정 1: 정렬판이 IC 정점이어야 한다(과거 쪽도 낮게). 체결 봉 상관 ≈ 0 · 행 봉 상관 음수
        assert 0 < rep[c]["ic_aligned"] < .12 and rep[c]["ic_future1"] < rep[c]["ic_aligned"] and rep[c]["ic_past1"] <= rep[c]["ic_aligned"] + .002, ("IC 이상", c, rep[c])
        assert abs(rep[c]["corr_next_bar"]) < .02 and rep[c]["corr_this_bar"] < -.01, ("정렬 이상", c, rep[c])
    assert abs(rep["negative_control_shuffled_days"]["mean_gross"]) < 3, rep["negative_control_shuffled_days"]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT / "checks.json", "w"), ensure_ascii=False, indent=1)


def judge():
    vp = OUT / "verdict.json"
    if vp.exists():
        sys.exit(f"판정은 1회만 -- {vp}")
    assert (OUT / "checks.json").exists(), "--checks 먼저"
    D, S = prepare()
    rep = {"prereg": "docs/prereg/dir_top_decile_long_20261107.md", "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    for tag, cost in (("strict", STRICT), ("candidate", CAND)):
        set_cost(cost)
        per = {c: trades(D[c], S[c]) for c in COINS}
        all_ = pd.concat([v.assign(coin=c) for c, v in per.items()])
        nl = null_means(D, S)
        day = all_.t.dt.floor("D").astype("int64").to_numpy()
        ex = H.cluster_boot_mean(all_.net.to_numpy() - nl.mean(), day)
        rep[tag] = {"fill_rate": len(all_) / max(1, sum(int((S[c] != 0).sum()) for c in COINS)), "pooled": stats(all_), "null_mean": float(nl.mean()), "null_q95": float(np.quantile(nl, .95)), "null_pct": float((nl < all_.net.mean()).mean() * 100), "excess_over_random_long": float(all_.net.mean() - nl.mean()), "excess_ci95": H.ci95(ex),
                    "years": {str(y): {"mean_net": float(g.net.mean()), "n": len(g)} for y, g in all_.groupby(all_.t.dt.year)}, "coins": {c: stats(v) for c, v in per.items()}}
        ds = all_.groupby(all_.t.dt.floor("D")).net.sum().sort_values(ascending=False)
        rep[tag]["mean_ex_top3days"] = float(all_[~all_.t.dt.floor("D").isin(ds.index[:3])].net.mean())
        if tag == "strict":
            base = pd.concat([trades(D[c], (D[c].p.notna().to_numpy() & (np.arange(len(D[c])) >= WARM) & (np.arange(len(D[c])) < len(D[c]) - HOLD - 80)).astype(int)) for c in COINS])
            rep["always_long_every_bar"] = stats(base)
    s, c = rep["strict"], rep["candidate"]
    ypos = lambda v: sum(x["mean_net"] > 0 for x in v["years"].values())
    g1 = s["pooled"]["ci95"][0] > 0
    g2 = s["excess_ci95"][0] > 0 and s["null_pct"] > 95
    g3 = ypos(s) >= 3
    g4 = sum(x.get("mean_net", -1) > 0 for x in s["coins"].values()) >= 2
    rep["verdict"] = {"strict": "통과" if g1 and g2 and g3 and g4 else "불통과", "gates": dict(G1=g1, G2=g2, G3=g3, G4=g4),
                      "candidate": "통과 후보" if c["pooled"]["mean_net"] > 0 and c["null_pct"] >= 50 and ypos(c) >= 3 else "불통과"}
    # 보조
    set_cost(STRICT)
    deciles = {}
    for coin, d in D.items():
        fwd = (d.close.shift(-HOLD) / d.close - 1) * 1e4
        m = d.p.notna() & fwd.notna() & (np.arange(len(d)) >= WARM)
        q = pd.qcut(d.p[m], 10, labels=False, duplicates="drop")
        deciles[coin] = fwd[m].groupby(q).mean().round(1).tolist()
    rep["aux_deciles_bp_full_sample_bins"] = deciles
    short = pd.concat([trades(D[c], top_signal(D[c].p, .1, -1)) for c in COINS])
    rep["aux_bottom_decile_short"] = stats(short)
    rep["aux_top_decile_taker"] = stats(pd.concat([trades(D[c], S[c], "taker") for c in COINS]))
    json.dump(rep, open(vp, "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(rep["verdict"], ensure_ascii=False))


def selftest():
    global WARM, PHASES, HOLD
    w0, ph0, h0 = WARM, PHASES, HOLD
    WARM = 20
    p = pd.Series(np.arange(100.0))
    q = p.copy(); q.iloc[50] = 1e9
    a = p.expanding(WARM).quantile(.9).shift(1)
    b = q.expanding(WARM).quantile(.9).shift(1)
    assert a[50] == b[50] and np.isnan(a[18])
    s = top_signal(pd.Series(np.r_[np.arange(30.0), 1000.0, 5.0]))
    assert s[30] == 1 and s[31] == 0 and s[:20].sum() == 0
    WARM = w0
    # 위상: 겹치는 결정이 다 산다 · 손계산
    PHASES, HOLD = 4, 2
    n = 30
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    c = np.full(n, 100.0); c[3:] = 101.0
    d = pd.DataFrame({"o": np.r_[100, c[:-1]], "high": c + 1, "low": c - 1, "close": c}, index=idx)
    set_cost(STRICT)
    x = np.zeros(n, int); x[[0, 1, 2, 3, 4]] = 1
    t = trades(d, x)
    assert sorted(t.i.tolist()) == [0, 1, 2, 3, 4], t
    one = np.zeros(n, int); one[1] = 1
    t1 = trades(d, one)
    # 지정가 100, 봉 2 저가 99 체결, 봉 3 종가 101 청산 지정가, 봉 4 고가 102 > 101 체결 → gross 100bp, net 96
    assert abs(t1.gross[0] - 100) < 1e-9 and abs(t1.net[0] - 96) < 1e-9, t1
    PHASES, HOLD = ph0, h0
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
