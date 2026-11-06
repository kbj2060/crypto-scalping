#!/usr/bin/env python3
"""평활 없는 DL 코인 간 롱숏 + 지정가 (사전등록 docs/prereg/xsec_unsmoothed_maker_20261107.md).
    python scripts/research_xsec_unsmoothed_maker_20261107.py --checks   # 재현·오라클·음성 (시험 합성 수익 안 봄: 음성은 섞은 점수)
    python scripts/research_xsec_unsmoothed_maker_20261107.py --judge    # 1회만
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_xsec_maker_execution_20261107 as M  # noqa: E402   -- OMP 처리·fill_tables·execute·sharpe 재사용
import research_xsec_dl_2021_window_20261026 as Y  # noqa: E402
X = M.X

OUT = Path("/home/kbj20/crypto-scalping/tmp/xsec_unsmoothed_maker_20261107")
YOUT = Path("/home/kbj20/crypto-scalping/tmp/xsec_dl_2021_20261026")
K, NULL = 12, 300


def bars(mod):
    c, h, l = {}, {}, {}
    for s in mod.COINS:
        d = pd.read_parquet(mod.PANEL / f"{s}USDT.parquet", columns=["timestamp", "high", "low", "close"]).set_index("timestamp")
        c[s], h[s], l[s] = d.close, d.high, d.low
    return pd.DataFrame(c), pd.DataFrame(h), pd.DataFrame(l)


def window(mod, out, w, model):
    """저장 시험 예측 → 평활 없는 비중. 반환 ts, W, 라벨, valid."""
    D = dict(np.load(out / "data.npz"))
    s = mod.WINDOWS[w]
    te = mod.split_idx(D["ts"], s[4], s[5])
    if model == "DL":
        c = json.load(open(out / "dl_selection.json"))[w]["chosen"]
        p = np.mean([np.load(out / "dl" / f"{w}_{c}_{sd}.npz")["te"] for sd in mod.SEEDS], 0)
    else:
        p = np.load(out / f"base_{w}.npz")["B3_te"]
    assert len(p) == len(te), (w, len(p), len(te))
    W = mod.weights2(mod.smooth_rank(p, D["valid"][te], 0), D["valid"][te], False)
    return D["ts"][te], W, D["lab"][te], D["valid"][te], p


def tables(mod, out, windows, model="DL"):
    b = bars(mod)
    T = {}
    for w in windows:
        ts, W, lab, valid, p = window(mod, out, w, model)
        T[w] = dict(ts=ts, W=W, lab=lab, valid=valid, p=p, F={k: M.fill_tables(ts, *b, k) for k in (0, K)})
    return T


def null_pct(T, mk, tk, real):
    rng = np.random.default_rng(M.SEED)
    out = []
    for _ in range(NULL):
        nets = [M.execute(np.roll(t["W"], int(rng.integers(42, len(t["W"]) - 42)), axis=0), t["F"][K], mk, tk)["net"] for t in T.values()]
        out.append(M.sharpe(np.concatenate(nets)))
    return float((np.array(out) < real).mean() * 100)


def summarize(T, mk, tk, with_null=True):
    es = {w: M.execute(t["W"], t["F"][K], mk, tk) for w, t in T.items()}
    net = np.concatenate([e["net"] for e in es.values()])
    gross = np.concatenate([e["gross"] for e in es.values()])
    daily = pd.Series(net).groupby(np.arange(len(net)) // 6).sum().sort_values(ascending=False).to_numpy()
    r = {"n": len(net), "net_sharpe": M.sharpe(net), "ci95": M.block_sharpe_ci(net), "net_bp_4h": float(net.mean() * 1e4), "gross_bp_4h": float(gross.mean() * 1e4), "total_bp": float(net.sum() * 1e4), "total_bp_ex_top3days": float(daily[3:].sum() * 1e4),
         "windows": {w: {"n": len(e["net"]), "net_sharpe": M.sharpe(e["net"]), "net_bp": float(e["net"].sum() * 1e4), "gross_bp_4h": float(e["gross"].mean() * 1e4), "turn": float(e["turn"].mean()), "fill_rate": e["fill_w"] / max(e["all_w"], 1e-12),
                         "filled_ret_bp": e["new_filled"] * 1e4, "unfilled_ret_bp": e["new_unfilled"] * 1e4,
                         "mean_abs_4h_bp": float(np.nanmean(np.abs(T[w]["F"][0]["end"] / T[w]["F"][0]["p0"] - 1)) * 1e4)} for w, e in es.items()}}
    if with_null:
        r["null_pct"] = null_pct(T, mk, tk, r["net_sharpe"])
    return r


def checks():
    T = tables(X, M.XOUT, ("R1", "R2", "R3"))
    rep = {}
    for w, t in T.items():
        e = M.execute(t["W"], t["F"][0], 0, 0)
        g, tu = X.pnl2(t["W"], t["lab"])
        rep[w] = {"gross_bp": float(e["gross"].mean() * 1e4), "pnl2_gross_bp": float(g.mean() * 1e4), "turn": float(e["turn"].mean()), "pnl2_turn": float(tu.mean())}
        assert abs(rep[w]["gross_bp"] - rep[w]["pnl2_gross_bp"]) < 1e-3 and abs(rep[w]["turn"] - rep[w]["pnl2_turn"]) < 1e-9, ("재현 불일치", rep[w])
        # 음성: 점수를 시점별로 섞음(비용 전). 수익 합은 보지만 실제 점수 정보는 없다
        rng = np.random.default_rng(M.SEED)
        v = t["valid"]
        shuf = rng.permuted(np.where(v, t["p"], np.nan), axis=1)
        vs = np.isfinite(shuf)
        rep[w]["shuffled_gross_bp"] = float(M.execute(X.weights2(shuf, vs, False), t["F"][0], 0, 0)["gross"].mean() * 1e4)
        assert abs(rep[w]["shuffled_gross_bp"]) < 2, ("음성 대조 이탈", rep[w])
    # 오라클: 검증 구간에서 라벨 순위, K=12 엄격
    D = dict(np.load(M.XOUT / "data.npz"))
    s = X.WINDOWS["R3"]
    ii = X.split_idx(D["ts"], s[2], s[3])
    W = X.weights2(X.smooth_rank(np.nan_to_num(D["lab"][ii]), D["valid"][ii], 0), D["valid"][ii], False)
    o = M.execute(W, M.fill_tables(D["ts"][ii], *bars(X), K), 2e-4, 5e-4)
    rep["oracle_K12_net_sharpe"] = M.sharpe(o["net"])
    assert rep["oracle_K12_net_sharpe"] > 10, rep
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT / "checks.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def judge():
    vp = OUT / "verdict.json"
    if vp.exists():
        sys.exit(f"판정은 1회만 -- {vp}")
    assert (OUT / "checks.json").exists(), "--checks 먼저"
    T = tables(X, M.XOUT, ("R1", "R2", "R3"))
    rep = {"prereg": "docs/prereg/xsec_unsmoothed_maker_20261107.md", "strict": summarize(T, 2e-4, 5e-4), "candidate": summarize(T, 0.0, 5e-4)}
    s, c = rep["strict"], rep["candidate"]
    pos = lambda v: sum(x["net_sharpe"] > 0 for x in v["windows"].values())
    rep["verdict"] = {"strict": "통과" if s["ci95"][0] > 0 and s["null_pct"] > 95 and pos(s) >= 2 else "불통과",
                      "candidate": "통과 후보" if c["net_sharpe"] > 0 and pos(c) >= 2 and c["null_pct"] >= 50 else "불통과"}
    Ty = tables(Y, YOUT, ("W0",))
    rep["aux_2021_strict"] = summarize(Ty, 2e-4, 5e-4)
    rep["aux_2021_candidate"] = summarize(Ty, 0.0, 5e-4, with_null=False)
    Tr = tables(X, M.XOUT, ("R1", "R2", "R3"), "B3")
    rep["aux_ridge_strict"] = summarize(Tr, 2e-4, 5e-4, with_null=False)
    rep["aux_ridge_candidate"] = summarize(Tr, 0.0, 5e-4, with_null=False)
    json.dump(rep, open(vp, "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(rep["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--checks" in a:
        checks()
    elif "--judge" in a:
        judge()
    else:
        sys.exit(__doc__)
