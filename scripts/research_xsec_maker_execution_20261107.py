#!/usr/bin/env python3
"""20코인 상대 수익 롱숏 지정가 체결 (사전등록 docs/prereg/xsec_maker_execution_20261107.md).
    python scripts/research_xsec_maker_execution_20261107.py --selftest
    python scripts/research_xsec_maker_execution_20261107.py --checks   # 재현·트립와이어 (채점 안 함)
    python scripts/research_xsec_maker_execution_20261107.py --judge    # 1회만
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

os.environ.pop("OMP_NUM_THREADS", None)   # 리지 재현: float32 적합은 스레드 수에 따라 3e-5 달라진다(기본 12 스레드에서 저장본과 차 0.0, 11-07 실측)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_xsec_dl_20coin_20261026 as X  # noqa: E402

XOUT = Path("/home/kbj20/crypto-scalping/tmp/xsec_dl_20261026")
OUT = Path("/home/kbj20/crypto-scalping/tmp/xsec_maker_exec_20261107")
K_MAIN, BOOT, SEED, N_NULL, BAR = 12, 2000, 20261107, 300, pd.Timedelta("5min")
SCORE = {"R1": ("2024-06-01", "2024-10-23", "R1"), "G1": ("2024-10-24", "2024-12-31", "R1"),
         "R2": ("2025-06-01", "2025-10-23", "R2"), "G2": ("2025-10-24", "2025-12-31", "R2"),
         "R3": ("2026-06-01", "2026-09-14", "R3")}


def fill_tables(ts, c5, h5, l5, K):
    """결정×코인: 기준가 p0 · 끝 가격 · 매수/매도 지정가 체결 여부 · K봉째 종가(미체결 시 시장가). c5 등 = 5분 DataFrame(행 = 봉 시작).
    K=0 은 즉시 체결(재현 대조용)."""
    t = pd.to_datetime(ts)
    idx, cols = c5.index, list(c5.columns)
    pos0 = idx.get_indexer(t - BAR)            # 봉 (t−5분) = 기준가
    assert (pos0 >= 0).all(), "기준가 봉 없음"
    C, H, L = c5.to_numpy(), h5.to_numpy(), l5.to_numpy()
    p0 = C[pos0]
    pe = idx.get_indexer(t + pd.Timedelta("3h55min"))
    end = np.where((pe >= 0)[:, None], C[np.clip(pe, 0, len(C) - 1)], np.nan)
    if K == 0:
        return dict(p0=p0, end=end, buy=np.ones_like(p0, bool), sell=np.ones_like(p0, bool), mkt=p0.copy(), missing=0)
    T, N = p0.shape
    buy, sell, mkt = np.zeros((T, N), bool), np.zeros((T, N), bool), np.full((T, N), np.nan)
    for i, p in enumerate(pos0):
        a, b = p + 1, min(p + 1 + K, len(C))     # 봉 t … t+(K−1)×5분
        buy[i] = (L[a:b] < p0[i]).any(0)         # 관통만(같으면 미체결). NaN 비교는 False
        sell[i] = (H[a:b] > p0[i]).any(0)
        mkt[i] = C[b - 1]
    miss = np.isnan(mkt) & np.isfinite(p0)
    mkt = np.where(miss, p0, mkt)                # 봉 결측 -> p0 시장가(건수 보고)
    return dict(p0=p0, end=end, buy=buy, sell=sell, mkt=mkt, missing=int(miss.sum()))


def execute(W, F, maker, taker):
    """W[T,N] 목표 비중. 반환: 시점별 비용 전·순 수익, 회전, 체결 통계."""
    p0, end = F["p0"], F["end"]
    prev = np.vstack([np.zeros(W.shape[1]), W[:-1]])
    D = W - prev
    is_buy = D > 0
    filled = np.where(is_buy, F["buy"], F["sell"]) & (D != 0)
    pf = np.where(filled, p0, F["mkt"])
    r_old = np.nan_to_num(end / p0 - 1)
    r_new = np.nan_to_num(end / pf - 1)
    gross = (prev * r_old + D * r_new).sum(1)
    absd = np.abs(D)
    fee = (absd * np.where(filled, maker, taker)).sum(1)
    return dict(gross=gross, net=gross - fee, turn=absd.sum(1), fill_w=float((absd * filled).sum()), all_w=float(absd.sum()),
                new_filled=float((D * r_new * filled).sum()), new_unfilled=float((D * r_new * (~filled)).sum()))


def sharpe(x):
    return float(x.mean() / x.std() * X.ANN) if x.std() > 0 else 0.0


def block_sharpe_ci(x):
    n, b = len(x), 30
    rng = np.random.default_rng(SEED)
    st = rng.integers(0, n - b + 1, size=(BOOT, int(np.ceil(n / b))))
    s = x[(st[..., None] + np.arange(b)).reshape(BOOT, -1)[:, :n]]
    v = s.mean(1) / s.std(1) * X.ANN
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def ridge_scores(D):
    """창별 리지(verdict2 와 base 의 α·설정). R 창 예측은 base 와 일치하는지 assert."""
    from sklearn.linear_model import Ridge
    ts, y, valid, Xm = D["ts"], D["y"], D["valid"], D["X"]
    Y = X.target(y)
    out = {}
    for w in ("R1", "R2", "R3"):
        s = X.WINDOWS[w]
        tr, te = X.split_idx(ts, s[0], s[1]), X.split_idx(ts, s[4], s[5])
        mu, sd = X.norm_stats(Xm, valid, tr)
        Xn = X.apply_norm(Xm, mu, sd)
        b = np.load(XOUT / f"base_{w}.npz")
        a = json.loads(str(b["meta"]))["B3"]["param"]
        r = Ridge(alpha=a).fit(Xn[tr][valid[tr]], Y[tr][valid[tr]])
        p = r.predict(Xn.reshape(-1, Xn.shape[-1])).reshape(len(ts), Xn.shape[1])
        d = float(np.nanmax(np.abs(p[te] - b["B3_te"])))
        assert d < 1e-6, ("리지 재현 불일치", w, d)
        out[w] = p
    return out


def window_weights(p, ts, valid, start, end, h, band):
    ii = X.split_idx(ts, start, end)
    return ii, X.weights2(X.smooth_rank(p[ii], valid[ii], h), valid[ii], band)


def load_bars():
    c, h, l = {}, {}, {}
    for s in X.COINS:
        d = pd.read_parquet(X.PANEL / f"{s}USDT.parquet", columns=["timestamp", "high", "low", "close"]).set_index("timestamp")
        c[s], h[s], l[s] = d.close, d.high, d.low
    return pd.DataFrame(c), pd.DataFrame(h), pd.DataFrame(l)


def prepare(D, K):
    ts = D["ts"]
    bars = load_bars()
    sel = json.load(open(XOUT / "verdict2.json"))["selected"]
    P = ridge_scores(D)
    tabs = {}
    for w, (a, b, m) in SCORE.items():
        ii, W = window_weights(P[m], ts, D["valid"], a, b, sel["B3"][m]["halflife"], sel["B3"][m]["band"])
        tabs[w] = dict(ii=ii, W=W, F={k: fill_tables(ts[ii], *bars, k) for k in set(K) | {0}})
    return tabs, sel, bars


def checks(D):
    tabs, sel, bars = prepare(D, [K_MAIN])
    v2 = json.load(open(XOUT / "verdict2.json"))
    rep = {"data_npz_sha256": hashlib.sha256((XOUT / "data.npz").read_bytes()).hexdigest()}
    for w in ("R1", "R2", "R3"):
        e = execute(tabs[w]["W"], tabs[w]["F"][0], 0.0, 0.0)
        want = v2["windows"][w]["B3"]
        gb, tu = float(e["gross"].mean() * 1e4), float(e["turn"].mean())
        rep[w] = {"gross_bp": gb, "want": want["gross_bp_4h"], "turn": tu, "want_turn": want["turnover_4h"]}
        assert abs(gb / want["gross_bp_4h"] - 1) < 1e-6 and abs(tu / want["turnover_4h"] - 1) < 1e-6, ("2단계 재현 불일치", rep[w])
    # 트립와이어(검증 구간만 -- 시험 창을 열지 않는다): 한 결정 미래로 민 점수의 비용 전 샤프
    ts, valid = D["ts"], D["valid"]
    P = ridge_scores(D)["R3"]
    s3 = X.WINDOWS["R3"]
    ii = X.split_idx(ts, s3[2], s3[3])
    h, band = sel["B3"]["R3"]["halflife"], sel["B3"]["R3"]["band"]
    F = fill_tables(ts[ii], *bars, 0)
    now = execute(X.weights2(X.smooth_rank(P[ii], valid[ii], h), valid[ii], band), F, 0, 0)
    fut_score = np.vstack([P[ii][1:], P[ii][-1:]])
    # 미래 라벨을 점수로 쓴 판: 장치가 미래를 잡는다는 양성 대조
    oracle = execute(X.weights2(X.smooth_rank(np.nan_to_num(D["lab"][ii]), valid[ii], 0), valid[ii], False), F, 0, 0)
    fut = execute(X.weights2(X.smooth_rank(fut_score, valid[ii], h), valid[ii], band), F, 0, 0)
    rep["tripwire_validation"] = {"current_gross_sharpe": sharpe(now["gross"]), "next_decision_score": sharpe(fut["gross"]), "oracle_label": sharpe(oracle["gross"])}
    assert rep["tripwire_validation"]["oracle_label"] > 10, rep["tripwire_validation"]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT / "checks.json", "w"), indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def concat(tabs, K, maker, taker, names=SCORE):
    es = [execute(tabs[w]["W"], tabs[w]["F"][K], maker, taker) for w in names]
    return np.concatenate([e["net"] for e in es]), np.concatenate([e["gross"] for e in es]), es


def null_dist(tabs, K, maker, taker):
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_NULL):
        nets = []
        for w in SCORE:
            W, F = tabs[w]["W"], tabs[w]["F"][K]
            T = len(W)
            sh = int(rng.integers(42, T - 42))     # ≥ 1주(42 결정)
            nets.append(execute(np.roll(W, sh, axis=0), F, maker, taker)["net"])
        out.append(sharpe(np.concatenate(nets)))
    return np.array(out)


def judge(D):
    vp = OUT / "verdict.json"
    if vp.exists():
        sys.exit(f"판정은 1회만 -- {vp}")
    assert (OUT / "checks.json").exists(), "--checks 먼저"
    tabs, sel, bars = prepare(D, [K_MAIN, 1, 48])
    ts = D["ts"]
    rep = {"prereg": "docs/prereg/xsec_maker_execution_20261107.md", "selected": sel["B3"], "missing_bars": {w: tabs[w]["F"][K_MAIN]["missing"] for w in SCORE}}
    btc = {w: np.nan_to_num(tabs[w]["F"][0]["end"][:, X.COINS.index("BTC")] / tabs[w]["F"][0]["p0"][:, X.COINS.index("BTC")] - 1) for w in SCORE}
    for tag, (mk, tk) in {"strict": (2e-4, 5e-4), "candidate": (0.0, 5e-4)}.items():
        net, gross, es = concat(tabs, K_MAIN, mk, tk)
        nl = null_dist(tabs, K_MAIN, mk, tk)
        g = np.concatenate([e["net"] for w, e in zip(SCORE, es) if w in ("G1", "G2")])
        rep[tag] = {"n_points": len(net), "net_sharpe": sharpe(net), "ci95": block_sharpe_ci(net), "null_pct": float((nl < sharpe(net)).mean() * 100), "null_q95": float(np.quantile(nl, .95)),
                    "net_bp_per_4h": float(net.mean() * 1e4), "gross_bp_per_4h": float(gross.mean() * 1e4), "total_bp": float(net.sum() * 1e4), "gap_sharpe": sharpe(g), "gap_total_bp": float(g.sum() * 1e4),
                    "windows": {w: {"n": len(e["net"]), "net_sharpe": sharpe(e["net"]), "net_bp": float(e["net"].sum() * 1e4), "gross_bp_4h": float(e["gross"].mean() * 1e4), "turn": float(e["turn"].mean()),
                                    "fill_rate": e["fill_w"] / max(e["all_w"], 1e-12), "filled_ret_bp": e["new_filled"] * 1e4, "unfilled_ret_bp": e["new_unfilled"] * 1e4} for w, e in zip(SCORE, es)}}
        s3 = np.sort(pd.Series(net).groupby(np.arange(len(net)) // 6).sum().to_numpy())[::-1]
        rep[tag]["total_bp_ex_top3days"] = float(s3[3:].sum() * 1e4)
        rep[tag]["btc_corr"] = float(np.corrcoef(net, np.concatenate([btc[w] for w in SCORE]))[0, 1])
    st = rep["strict"]
    M1 = st["ci95"][0] > 0 and st["null_pct"] > 95 and sum(v["net_sharpe"] > 0 for v in st["windows"].values()) >= 3 and st["gap_sharpe"] > 0
    ca = rep["candidate"]
    C = ca["net_sharpe"] > 0 and ca["gap_sharpe"] > 0 and ca["null_pct"] >= 50
    rep["verdict"] = {"strict": "통과" if M1 else "불통과", "candidate": "통과 후보" if C else "불통과"}
    # 보조: K·테이커·DL
    aux = {}
    for k in (1, 48):
        n, _, _ = concat(tabs, k, 2e-4, 5e-4)
        aux[f"strict_K{k}"] = {"net_sharpe": sharpe(n), "total_bp": float(n.sum() * 1e4)}
    n0, g0, e0 = concat(tabs, 0, 0, 0)
    aux["taker_immediate_5bp"] = {"net_sharpe": sharpe(np.concatenate([e["gross"] - 5e-4 * e["turn"] for e in e0])), "gross_sharpe": sharpe(g0), "gross_bp_4h": float(g0.mean() * 1e4)}
    v1 = json.load(open(XOUT / "dl_selection.json"))
    dl_net = []
    for w in ("R1", "R2", "R3"):
        a, b, m = SCORE[w]
        c = v1[m]["chosen"]
        pte = np.mean([np.load(XOUT / "dl" / f"{m}_{c}_{sd}.npz")["te"] for sd in X.SEEDS], 0)
        ii = X.split_idx(ts, a, b)
        assert len(ii) == len(pte), (w, len(ii), len(pte))
        s = sel["DL"][m]
        W = X.weights2(X.smooth_rank(pte, D["valid"][ii], s["halflife"]), D["valid"][ii], s["band"])
        dl_net.append(execute(W, tabs[w]["F"][K_MAIN], 2e-4, 5e-4)["net"])
    dl_net = np.concatenate(dl_net)
    aux["dl_strict_R1R2R3"] = {"net_sharpe": sharpe(dl_net), "ci95": block_sharpe_ci(dl_net), "total_bp": float(dl_net.sum() * 1e4)}
    rep["aux"] = aux
    json.dump(rep, open(vp, "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


def selftest():
    idx = pd.date_range("2026-01-01", periods=120, freq="5min")
    c5 = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    h5, l5 = c5.copy(), c5.copy()
    t = idx[60]                          # 결정 시각 = 봉 60 시작 -> 기준가 = 봉 59 종가
    # 매수 A: 봉 61 저가 99.9 < 100 -> 체결. 매도 B: 고가가 100 과 같음 -> 미체결 -> K 봉째 종가 101 시장가
    l5.loc[idx[61], "A"] = 99.9
    c5.loc[idx[71], "B"] = 101.0
    c5.loc[t + pd.Timedelta("3h55min"):, ["A", "B"]] = 100.0
    c5 = c5.reindex(pd.date_range("2026-01-01", periods=400, freq="5min")).ffill()
    h5, l5 = h5.reindex(c5.index).ffill(), l5.reindex(c5.index).ffill()
    c5.loc[t + pd.Timedelta("3h55min"), ["A", "B"]] = [102.0, 99.0]
    F = fill_tables(np.array([t.to_datetime64()]), c5, h5, l5, 12)
    assert bool(F["buy"][0, 0]) and not bool(F["sell"][0, 1]) and F["mkt"][0, 1] == 101.0, F
    W = np.array([[0.5, -0.5]])
    e = execute(W, F, 2e-4, 5e-4)
    # A: +.5×(102/100−1) = +.01 · B: −.5×(99/101−1) = +.0099010 · 수수료 .5×2bp + .5×5bp = 3.5bp
    assert abs(e["gross"][0] - (0.01 + 0.5 * (1 - 99 / 101))) < 1e-12 and abs(e["net"][0] - e["gross"][0] + 3.5e-4) < 1e-12, e
    assert abs(e["fill_w"] / e["all_w"] - .5) < 1e-12
    # 유지분: 변경 없으면 수수료 0·p0 기준 수익
    W2 = np.array([[0.5, -0.5], [0.5, -0.5]])
    F2 = {k: np.vstack([v, v]) if isinstance(v, np.ndarray) else v for k, v in F.items()}
    e2 = execute(W2, F2, 2e-4, 5e-4)
    assert e2["turn"][1] == 0 and abs(e2["gross"][1] - (0.5 * .02 + (-.5) * (100 / 100 - 1) * 0 + (-.5) * (F["end"][0, 1] / 100 - 1))) < 1e-12, e2
    # K=0 재현 모드: 전부 p0 체결
    F0 = fill_tables(np.array([t.to_datetime64()]), c5, h5, l5, 0)
    assert execute(W, F0, 0, 0)["gross"][0] == (W * (F0["end"] / F0["p0"] - 1)).sum()
    # 음성·양성: 무작위 점수 ≈ 0, 미래 점수 ≫ 0
    rng = np.random.default_rng(1)
    T, N = 1500, 20
    R = rng.normal(0, .01, (T, N))
    Fx = dict(p0=np.ones((T, N)), end=1 + R, buy=np.ones((T, N), bool), sell=np.ones((T, N), bool), mkt=np.ones((T, N)), missing=0)
    v = np.ones((T, N), bool)
    assert abs(sharpe(execute(X.weights2(rng.normal(size=(T, N)), v, False), Fx, 0, 0)["gross"])) < 6
    assert sharpe(execute(X.weights2(R, v, False), Fx, 0, 0)["gross"]) > 20
    print("selftest OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    elif "--checks" in a or "--judge" in a:
        D = dict(np.load(XOUT / "data.npz"))
        (checks if "--checks" in a else judge)(D)
    else:
        sys.exit(__doc__)
