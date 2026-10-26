#!/usr/bin/env python3
"""20코인 상대 수익 — 새 틈 창 G1·G2 + DL·HGB 결합(E), 메이커 비용. 사전 등록 docs/prereg/xsec_gap_windows_maker_20261026.md.

    python3 scripts/research_xsec_gap_maker_20261026.py --fit      # G1·G2 학습(기준선 + DL 30개)
    python3 scripts/research_xsec_gap_maker_20261026.py --judge    # 한 번만
기존 tmp/xsec_dl_20261026 은 읽기만. 실주문 없음.
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_maker_cost_reeval_20261026 as M   # noqa: E402

X = M.load_xs()
SRC, OUT = X.OUT, Path("/home/kbj20/crypto-scalping/tmp/xsec_gap_20261026")
GAP = {"G1": ("2022-06-01", "2024-05-31", "2024-06-01", "2024-10-23", "2024-10-24", "2024-12-31"),
       "G2": ("2023-06-01", "2025-05-31", "2025-06-01", "2025-10-23", "2025-10-24", "2025-12-31")}
R = ("R1", "R2", "R3")
X.OUT = OUT; X.WINDOWS.update(GAP)                 # dl_job 이 OUT/data.npz(원본 심볼릭 링크)·WINDOWS[창]을 읽음 — fork 로 자식에 전달


def fit():
    (OUT / "dl").mkdir(parents=True, exist_ok=True)
    if not (OUT / "data.npz").exists():
        (OUT / "data.npz").symlink_to(SRC / "data.npz")
    D = dict(np.load(OUT / "data.npz"))
    res = X.fit_baselines(D, wins=GAP, save=False)
    for w in GAP:
        np.savez(OUT / f"base_{w}.npz", **{f"{k}_{s}": res[w][k][s] for k in ("B3", "B4") for s in ("va", "te")})
    jobs = [(w, c, s, False) for w in GAP for c in X.CANDS for s in X.SEEDS if not (OUT / "dl" / f"{w}_{c}_{s}.npz").exists()]
    with ProcessPoolExecutor(5) as ex:
        for tag, ep, vic in ex.map(X.dl_job, jobs):
            print(f"  {tag} best_ep {ep} val IC {vic:+.4f}", flush=True)


def combine(a, b, valid):
    return np.nanmean([X.xs_rank(np.asarray(a, float), valid), X.xs_rank(np.asarray(b, float), valid)], 0)


def preds(ts, y, valid):
    sel1, P = json.load(open(SRC / "dl_selection.json")), {}
    for w, s in X.WINDOWS.items():
        va, te = X.split_idx(ts, s[2], s[3]), X.split_idx(ts, s[4], s[5])
        if w in R:
            c, d = sel1[w]["chosen"], SRC
            b, b2 = np.load(SRC / f"base_{w}.npz"), np.load(SRC / f"base2_va_{w}.npz")
            base = {k: (b2[k], b[f"{k}_te"]) for k in ("B3", "B4")}
        else:
            d = OUT
            ens = {c: np.mean([np.load(OUT / "dl" / f"{w}_{c}_{sd}.npz")["va"] for sd in X.SEEDS], 0) for c in X.CANDS}
            c = max(ens, key=lambda k: np.nanmean(X.ic_series(ens[k], y[va], valid[va])))   # 1단계와 같은 검증 IC 선택
            b = np.load(OUT / f"base_{w}.npz"); base = {k: (b[f"{k}_va"], b[f"{k}_te"]) for k in ("B3", "B4")}
        dl = [np.load(d / "dl" / f"{w}_{c}_{sd}.npz") for sd in X.SEEDS]
        DL = (np.mean([x["va"] for x in dl], 0), np.mean([x["te"] for x in dl], 0))
        P[w] = dict(DL=DL, **base, E=(combine(DL[0], base["B4"][0], valid[va]), combine(DL[1], base["B4"][1], valid[te])), _cand=c)
    return P


def judge():
    rp = OUT / "report.json"; assert not rp.exists(), "판정은 한 번만"
    D = dict(np.load(OUT / "data.npz")); ts, y, lab, valid = D["ts"], D["y"], D["lab"], D["valid"]
    P = preds(ts, y, valid); rep = {"dl_cand": {w: P[w].pop("_cand") for w in P}}
    for k, (ub, tb) in M.SCEN.items():
        cv = M.costvec(X.COINS, ub, tb); r = {"windows": {}, "selected": {}, "combined": {}}; ser = {}
        for w, s in X.WINDOWS.items():
            va, te = X.split_idx(ts, s[2], s[3]), X.split_idx(ts, s[4], s[5]); r["windows"][w] = {}
            for m, (pva, pte) in P[w].items():
                val = {hb: X.sharpe(np.subtract(*M.pnl_w(X.weights2(X.smooth_rank(pva, valid[va], hb[0]), valid[va], hb[1]), lab[va], cv))) for hb in X.GRID2}
                h, band = max(val, key=val.get); r["selected"].setdefault(m, {})[w] = [h, band]
                g, cc = M.pnl_w(X.weights2(X.smooth_rank(pte, valid[te], h), valid[te], band), lab[te], cv)
                ser.setdefault(m, {})[w] = g - cc; r["windows"][w][m] = round(X.sharpe(g - cc), 3)
        for m, by in ser.items():
            for nm, ws in (("all5", list(X.WINDOWS)), ("R", list(R)), ("G", list(GAP))):
                n = np.concatenate([by[w] for w in ws])
                r["combined"].setdefault(nm, {})[m] = dict(sharpe=X.sharpe(n), ci=X.sharpe_ci(n), total_bp=float(n.sum() * 1e4))
        A, G = r["combined"]["all5"], r["combined"]["G"]
        r["gates"] = {"N1": bool(A["E"]["sharpe"] > 0 and A["E"]["ci"][0] > 0), "N2": bool(A["E"]["sharpe"] > max(A[m]["sharpe"] for m in ("DL", "B3", "B4"))),
                      "N3": bool(G["E"]["sharpe"] > 0), "N4": bool(sum(r["windows"][w]["E"] > 0 for w in X.WINDOWS) >= 3)}
        rep[k] = r; print(k, {nm: {m: round(v["sharpe"], 2) for m, v in c.items()} for nm, c in r["combined"].items()}, r["gates"], flush=True)
    old = json.load(open("/home/kbj20/crypto-scalping/tmp/maker_reeval_20261026/report.json"))["xsec"]["K1"]["combined"]
    for m in ("DL", "B3", "B4"):                       # 재현 점검: R 창 단독 = 메이커 재평가 숫자
        assert abs(rep["K1"]["combined"]["R"][m]["sharpe"] - old[m]["sharpe"]) < 1e-6, (m, rep["K1"]["combined"]["R"][m]["sharpe"], old[m]["sharpe"])
    g = rep["K1"]["gates"]; rep["verdict"] = "후보(전진 필요)" if all(g.values()) else "불통과"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print("→", rp, rep["verdict"])


def selftest():
    rng = np.random.default_rng(0); a = rng.normal(size=(50, 20)); v = np.ones_like(a, bool); v[:, 3] = False
    assert np.allclose(np.nan_to_num(combine(a, a, v)), np.nan_to_num(X.xs_rank(a, v)))      # 같은 예측 둘 결합 = 그 순위
    assert np.isnan(combine(a, -a, v)[:, 3]).all() and np.allclose(combine(a, -a, v)[v], 0)  # 정반대 둘 = 0
    print("selftest OK")


if __name__ == "__main__":
    selftest()
    if sys.argv[1:] == ["--fit"]: fit()
    elif sys.argv[1:] == ["--judge"]: judge()
    else: print(__doc__)
