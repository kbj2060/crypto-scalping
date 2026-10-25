#!/usr/bin/env python3
"""메이커 비용 재평가 — 20코인 상대 딥러닝 · RL v19. 사전 등록 docs/prereg/maker_cost_reeval_20261026.md.

    python3 scripts/research_maker_cost_reeval_20261026.py      # 한 번만
모형 재학습 없음(동결 예측·정책). 실주문 없음.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path("/home/kbj20/crypto-scalping/tmp/maker_reeval_20261026")
XS = "/home/kbj20/crypto-scalping/.claude/worktrees/agent-a2389468213c4ebbe/scripts/research_xsec_dl_20coin_20261026.py"
USDC = {"ADA", "ARB", "AVAX", "BNB", "BTC", "DOGE", "ETH", "FIL", "LINK", "LTC", "NEAR", "SOL", "SUI", "XRP"}   # exchangeInfo 10-26
SCEN = {"K0": (4.0, 4.0), "K1": (0.7, 2.93), "K2": (1.5, 3.5)}   # (USDC 코인 bp, USDT 만 bp) — 명목 변화 1 단위당


def load_xs():
    spec = importlib.util.spec_from_file_location("xs", XS); m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(XS).parent)); spec.loader.exec_module(m)
    return m


def costvec(coins, usdc_bp, usdt_bp):
    return np.array([(usdc_bp if c in USDC else usdt_bp) * 1e-4 for c in coins])


def pnl_w(W, lab, cv):
    """가중치 경로 → (총수익, 코인별 비용 합). 첫 시점 진입 비용 포함(1단계 pnl2 와 같은 규칙)."""
    R = np.expm1(np.nan_to_num(lab))
    dW = np.abs(np.diff(np.vstack([np.zeros(W.shape[1]), W]), axis=0))
    return (W * R).sum(1), dW @ cv


def selftest(X):
    rng = np.random.default_rng(3)
    sc, lab = rng.normal(size=(200, 20)), rng.normal(size=(200, 20)) * .01
    v = np.ones_like(sc, bool)
    W = X.weights2(X.smooth_rank(sc, v, 3), v, True)
    g, tu = X.pnl2(W, lab); g2, c2 = pnl_w(W, lab, np.full(20, 4e-4))
    assert np.allclose(g, g2) and np.allclose(4e-4 * tu, c2)          # 일괄 4bp 면 기존 계산과 같음
    print("selftest OK")


def xsec(X):
    D = dict(np.load(X.OUT / "data.npz")); ts, lab, valid = D["ts"], D["lab"], D["valid"]
    sel1 = json.load(open(X.OUT / "dl_selection.json")); rep = {}
    preds = {}
    for w, s in X.WINDOWS.items():
        c = sel1[w]["chosen"]; dl = [np.load(X.OUT / "dl" / f"{w}_{c}_{sd}.npz") for sd in X.SEEDS]
        b, b2 = np.load(X.OUT / f"base_{w}.npz"), np.load(X.OUT / f"base2_va_{w}.npz")
        preds[w] = {"DL": (np.mean([x["va"] for x in dl], 0), np.mean([x["te"] for x in dl], 0)),
                    "B3": (b2["B3"], b["B3_te"]), "B4": (b2["B4"], b["B4_te"])}
    for k, (ub, tb) in SCEN.items():
        cv = costvec(X.COINS, ub, tb); r = {"windows": {}, "combined": {}, "selected": {}}; ser = {}
        for w, s in X.WINDOWS.items():
            va, te = X.split_idx(ts, s[2], s[3]), X.split_idx(ts, s[4], s[5]); r["windows"][w] = {}
            for m, (pva, pte) in preds[w].items():
                val = {}
                for h, band in X.GRID2:
                    g, cc = pnl_w(X.weights2(X.smooth_rank(pva, valid[va], h), valid[va], band), lab[va], cv); val[(h, band)] = X.sharpe(g - cc)
                h, band = max(val, key=val.get); r["selected"].setdefault(m, {})[w] = [h, band]
                g, cc = pnl_w(X.weights2(X.smooth_rank(pte, valid[te], h), valid[te], band), lab[te], cv)
                ser.setdefault(m, []).append((g, cc)); r["windows"][w][m] = round(X.sharpe(g - cc), 3)
        for m, lst in ser.items():
            n = np.concatenate([g - cc for g, cc in lst])
            r["combined"][m] = dict(sharpe=X.sharpe(n), ci=X.sharpe_ci(n), cost_bp_4h=float(np.concatenate([cc for _, cc in lst]).mean() * 1e4),
                                    **{kk: vv for kk, vv in X.ls_stats(n).items() if kk in ("total_bp", "mdd_bp")})
        C = r["combined"]
        r["gates"] = {"M1": bool(C["DL"]["sharpe"] > 0 and C["DL"]["ci"][0] > 0), "M2": bool(C["DL"]["sharpe"] > max(C["B3"]["sharpe"], C["B4"]["sharpe"])),
                      "M3": bool(sum(r["windows"][w]["DL"] > 0 for w in X.WINDOWS) >= 2)}
        rep[k] = r; print("xsec", k, {m: round(v["sharpe"], 2) for m, v in C.items()}, r["gates"], flush=True)
    return rep


def rl():
    sys.path.insert(0, str(HERE)); import research_rl_v14_vol_sizing_20261026 as R   # noqa: E402
    from stable_baselines3 import PPO
    parts = []
    for w, (trp, vap, tep) in R.WIN17.items():
        F, Xf, size, _ = R.load(frame="frame_v18.parquet", train=trp); Xn, ret = Xf.to_numpy(), F.ret.to_numpy(); base = np.sign(F.m30d.to_numpy())
        T = R.teachers(F, size, R.span(F, trp)); m = R.span(F, tep)
        ens = np.mean([R.rollout(PPO.load(R.OUT / f"v19_{w}_{s}" / f"{R.TOTAL}.zip", device="cpu"), Xn, size, m, base) for s in R.SEEDS19], 0)
        parts.append((w, ens, {k: v[m] for k, v in T.items()}, ret[m])); print("rl 경로", w, flush=True)
    rep = {}
    for k, (cbp, _) in SCEN.items():
        c = cbp * 1e-4
        pnl = lambda n, r: n * r - c * np.abs(n - np.r_[0.0, n[:-1]])      # noqa: E731
        per = {w: dict(RL=R.stats(pnl(e, r))["sharpe"], **{t: R.stats(pnl(v, r))["sharpe"] for t, v in T.items()}) for w, e, T, r in parts}
        pool = {nm: R.stats(np.concatenate([pnl(e if nm == "RL" else T[nm], r) for _, e, T, r in parts])) for nm in ("RL", "T1_long", "T2_dirmodel", "T3_tsmom")}
        rng = np.random.default_rng(1); obs = sum(pnl(T["T3_tsmom"] + (e - T["T3_tsmom"]), r).sum() for _, e, T, r in parts)
        null = np.array([sum(pnl(T["T3_tsmom"] + np.roll(e - T["T3_tsmom"], int(rng.integers(168, len(e) - 168))), r).sum() for _, e, T, r in parts) for _ in range(1000)])
        shift = float((null < obs).mean() * 100)
        g = {"H1": bool(pool["RL"]["sharpe"] > pool["T3_tsmom"]["sharpe"] and pool["RL"]["sharpe"] > max(pool["T1_long"]["sharpe"], pool["T2_dirmodel"]["sharpe"])),
             "H2": bool(sum(per[w]["RL"] > per[w]["T3_tsmom"] for w in per) >= 2), "H3": bool(shift > 90)}
        rep[k] = dict(windows=per, pooled={nm: {kk: v[kk] for kk in ("sharpe", "total_bp", "mdd_bp")} for nm, v in pool.items()}, adj_shift_pct=shift, gates=g)
        print("rl", k, {nm: round(v["sharpe"], 2) for nm, v in pool.items()}, shift, g, flush=True)
    return rep


if __name__ == "__main__":
    X = load_xs(); selftest(X)
    rp = OUT / "report.json"; assert not rp.exists(), "판정은 한 번만"
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"scenarios": SCEN, "usdc": sorted(USDC), "xsec": xsec(X), "rl_v19": rl()}
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print("→", rp)
