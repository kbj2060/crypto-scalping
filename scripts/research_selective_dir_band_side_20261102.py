"""A 선택적 방향 예측(보류형) · C 띠 이탈 방향 3분류. 사전등록 docs/prereg/selective_dir_band_side_20261102.md (e3e382a7, 5d52230f).
사용: python scripts/research_selective_dir_band_side_20261102.py [align|a|c|all]  (CPU, REST 없음)"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as sw
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("rf", ROOT / "scripts" / "research_regime_f_20261102.py")
rf = importlib.util.module_from_spec(sp)
sp.loader.exec_module(rf)
LIVE, BW = ROOT / "data" / "live", ROOT / "tmp" / "band_warning_20261102"
OUT = ROOT / "tmp" / "selective_dir_band_side_20261102"
VAL0, TEST0 = pd.Timestamp(rf.TRAIN_END), pd.Timestamp(rf.VAL_END)
NB, NPL, SEED = 2000, 200, 20261102
HB = {"p15m": 3, "p1h": 12, "pred": 48}                     # 예측 열 → 지평(5분봉 수)
FILE = {"p15m": "dirshort", "p1h": "dirshort", "pred": "dir4h"}
QS = (0.05, 0.10, 0.20, 0.50, 1.0)


def preds(coin, col):
    p = LIVE / f"{coin.lower()}_{FILE[col]}_oos_seed.parquet"
    if not p.exists():
        return None
    x = pd.read_parquet(p)
    x["ts"] = pd.to_datetime(x.ts)
    return x.set_index("ts")[col]


def trail_pct(a, w=720, mn=168):
    """직전 w개(최소 mn개, 현재 제외) |값| 중 현재 |값|보다 작은 비율."""
    a = np.abs(a)
    win = sw(np.concatenate([np.full(w, np.nan), a]), w)[: len(a)]
    cnt = (~np.isnan(win)).sum(1)
    u = (win < a[:, None]).sum(1) / np.maximum(cnt, 1)
    u[cnt < mn] = np.nan
    return u


def a_frame(coin, col):
    """매시 정각 결정 행: 예측 부호·신뢰도 u·수익(진입 = 시가 t−5분 봉 종가, 청산 = 시가 t+(h−1)×5분 봉 종가)."""
    s = preds(coin, col)
    if s is None:
        return None
    s = s[s.index.minute == 0].sort_index()
    h, c = HB[col], rf.load(coin).c
    pos = c.index.get_indexer(s.index - pd.Timedelta(minutes=5))
    ok = (pos >= 0) & (pos + h < len(c))
    u = trail_pct(s.to_numpy(float))
    ret = np.where(ok, c.to_numpy()[np.minimum(pos + h, len(c) - 1)] / c.to_numpy()[np.maximum(pos, 0)] - 1, np.nan)
    f = pd.DataFrame({"ts": s.index, "coin": coin, "sgn": np.sign(s.to_numpy()), "u": u, "ret": ret})
    f = f[f.ret.notna() & (f.ret != 0) & f.u.notna() & (f.sgn != 0)].copy()
    f["day"] = f.ts.dt.date
    f["up"] = f.ret > 0
    f["hit"] = f.sgn == np.sign(f.ret)
    return f


def align_check(coin, col):
    """선언한 정렬(s=0)에서 IC 가 최대인지 — 한 봉 밀림 점검. 5분 전 행, 시험 창."""
    s = preds(coin, col)
    h, c = HB[col], rf.load(coin).c.to_numpy()
    s = s[(s.index >= TEST0) & (s.index < pd.Timestamp("2026-10-20"))]
    pos = rf.load(coin).c.index.get_indexer(s.index - pd.Timedelta(minutes=5))
    return {k: float(spearmanr(s.to_numpy(), c[pos + k + h] / c[pos + k] - 1)[0]) for k in (-2, -1, 0, 1, 2)}


def day_boot(f, sel, rng):
    code, uniq = pd.factorize(f.day.to_numpy())
    w = lambda m: np.bincount(code, weights=m & sel, minlength=len(uniq))               # noqa: E731
    H, U, N = w(f.hit.to_numpy()), w(f.up.to_numpy()), w(np.ones(len(f), bool))
    idx = rng.integers(0, len(uniq), (NB, len(uniq)))
    h, u, n = H[idx].sum(1), U[idx].sum(1), N[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = h / n - np.maximum(u / n, 1 - u / n)
    return e[np.isfinite(e)]


def excess(f, sel):
    n = sel.sum()
    if n == 0:
        return None
    hit, up = f.hit[sel].mean(), f.up[sel].mean()
    return {"n": int(n), "hit": float(hit), "base": float(max(up, 1 - up)), "E": float(hit - max(up, 1 - up))}


def a_eval(f, rng, label):
    out = {}
    for q in QS:
        sel = (f.u >= 1 - q).to_numpy()
        r = excess(f, sel)
        if r is None:
            continue
        e = day_boot(f, sel, rng)
        r.update(days=int(f.day[sel].nunique()), per_day=float(sel.sum() / max(f.day.nunique(), 1)), p=float((e <= 0).mean()), lo10=float(np.percentile(e, 10)),
                 byc={c: [int(g.hit.to_numpy()[sel[g.index.to_numpy()]].sum()), int(sel[g.index.to_numpy()].sum())] for c, g in f.groupby("coin")})
        out[f"q{int(q * 100)}"] = r
    # 원형 이동 위약(q=10%): 코인별 신뢰도·부호를 수익에 대해 하루 단위로 밀기
    pl, cs = [], {c: g.reset_index(drop=True) for c, g in f.groupby("coin")}
    for _ in range(NPL):
        hit = up = n = 0
        for c, g in cs.items():
            nd = max(g.day.nunique(), 2)
            k = int(rng.integers(1, nd)) * 24
            sg, uu = np.roll(g.sgn.to_numpy(), k), np.roll(g.u.to_numpy(), k)
            s = uu >= 0.90
            hit += (sg[s] == np.sign(g.ret.to_numpy()[s])).sum(); up += g.up.to_numpy()[s].sum(); n += s.sum()
        pl.append(hit / n - max(up / n, 1 - up / n))
    out["placebo_E_q10"] = {"p90": float(np.percentile(pl, 90)), "p95": float(np.percentile(pl, 95)), "mean": float(np.mean(pl))}
    out["label"] = label
    return out


def run_a(res):
    rng = np.random.default_rng(SEED)
    res["align"] = {}
    for col, coins in (("pred", ("ETH", "SOL", "XRP")), ("p1h", rf.COINS), ("p15m", rf.COINS)):
        res["align"][col] = {c: align_check(c, col) for c in coins}
        agg = pd.DataFrame(res["align"][col]).mean(1)
        print("정렬 IC(코인 평균)", col, agg.round(4).to_dict())
        if agg.loc[[-2, -1]].max() > agg[0]:                      # 누수 방향(이미 지난 수익이 창에 들어오는 쪽) 정렬 우위면 중단
            raise SystemExit(f"정렬 점검 실패 {col}: 음의 이동에서 IC 가 더 큼 -- 중단")
        if agg.loc[[1, 2]].max() > agg[0]:
            print("  주의: 양의 이동(선언보다 늦은 창)에서 IC 가 더 큼 -- 선언 정렬(보수적)로 진행, 결과 문서에 기록")
    res["A"] = {}
    for col, coins in (("pred", ("ETH", "SOL", "XRP")), ("p1h", rf.COINS), ("p15m", rf.COINS)):
        f = pd.concat([a_frame(c, col) for c in coins], ignore_index=True)
        wins = {"시험": f[f.ts >= TEST0]}
        if col == "pred":
            wins["검증"] = f[(f.ts >= VAL0) & (f.ts < TEST0)]
            wins["2025"] = f[(f.ts >= "2025-01-01") & (f.ts < VAL0)]
        for w, x in wins.items():
            res["A"][f"{col}|{w}"] = a_eval(x.reset_index(drop=True), rng, f"{col} {w} 행{len(x)} 일{x.day.nunique()}")
            r = res["A"][f"{col}|{w}"]["q10"]
            print(f"A {col} {w}: q10 발화 {r['n']}/{len(x)} 적중 {r['hit']:.3f} 기준 {r['base']:.3f} E {r['E']:+.3f} p {r['p']:.3f} lo10 {r['lo10']:+.3f} | 위약 p95 {res['A'][f'{col}|{w}']['placebo_E_q10']['p95']:+.3f}")


# ---------------- C ----------------
def c_frame(H):
    d = pd.read_parquet(BW / "data.parquet")
    d["day"] = d.ts.dt.date
    parts = []
    for coin in rf.COINS:
        x = d[d.coin == coin].copy()
        c = rf.load(coin).c
        i = c.index.get_indexer(x.ts) - 1                                    # ts = 봉 i 마감 = 봉 i+1 시가
        ok = (i >= 0) & (i + H < len(c))
        Wf = np.full((len(x), H), np.nan)
        Wf[ok] = sw(c.to_numpy(), H)[i[ok] + 1]
        up, dn = Wf > x.R1.to_numpy()[:, None], Wf < x.S1.to_numpy()[:, None]
        fu, fd = np.where(up.any(1), up.argmax(1), H), np.where(dn.any(1), dn.argmax(1), H)
        x["cls"] = np.where(ok, np.where(fu < fd, 1, np.where(fd < fu, 2, 0)), -1)
        p4 = preds(coin, "pred")
        x["p4h"] = x.ts.map(p4) if p4 is not None else np.nan
        parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    k = d[f"y{H}"].notna() & (d.cls >= 0)
    assert ((d.cls[k] > 0) == (d[f"y{H}"][k] == 1)).all(), "이탈 여부가 band_warning 라벨과 불일치"
    return d


def sample_check(d, H, n=400):
    """독립 재계산: 파이썬 루프로 첫 이탈 방향."""
    rng, bad = np.random.default_rng(1), 0
    k = d[(d.cls >= 0) & d.R1.notna() & d.S1.notna()].sample(n, random_state=1)
    for r in k.itertuples():
        c = rf.load(r.coin).c
        i = c.index.get_loc(r.ts) - 1
        side = 0
        for v in c.to_numpy()[i + 1: i + 1 + H]:
            if v > r.R1:
                side = 1; break
            if v < r.S1:
                side = 2; break
        bad += side != r.cls
    return bad


def run_c(res, H=12):
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(SEED)
    d = c_frame(H)
    res["C_side_check_bad_of_400"] = int(sample_check(d, H))
    META = {"coin", "ts", "day", "R1", "S1", "tb", "y6", "y12", "y24", "cls", "p4h"}
    FE0 = [c for c in d.columns if c not in META]
    GEO = ["dR", "dS", "dmin", "width", "bpos"]
    x = d[(d.cls >= 0) & d.dmin.notna()].reset_index(drop=True)
    pur = pd.Timedelta(minutes=5 * 24)
    tr, va, te = x[x.ts < VAL0 - pur], x[(x.ts >= VAL0) & (x.ts < TEST0)], x[x.ts >= TEST0]
    trva = x[x.ts < TEST0 - pur]
    mk = lambda: HGB(max_depth=3, learning_rate=0.03, max_iter=600, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)   # noqa: E731
    mg = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))                                       # noqa: E731
    side = lambda P: P[:, 1] / (P[:, 1] + P[:, 2] + 1e-12)                                                                                                   # noqa: E731
    lgt = lambda s: np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))                                                                    # noqa: E731
    brk = lambda z: z[z.cls > 0]                                                                                                                            # noqa: E731
    arms = {"C0": FE0, "C+": FE0 + ["p4h"]}
    g_tr = mg().fit(tr[GEO], tr.cls)
    sg_va = side(g_tr.predict_proba(va[GEO]))
    b = (va.cls > 0).to_numpy()
    auc_g_va = roc_auc_score((va.cls[b] == 1), sg_va[b])
    val = {}
    for nm, fe in arms.items():
        m = mk().fit(tr[fe], tr.cls)
        val[nm] = float(roc_auc_score((va.cls[b] == 1), side(m.predict_proba(va[fe]))[b]) - auc_g_va)
        print(f"C 검증 {nm}: ΔAUC(M−G) {val[nm]:+.4f}")
    arm = "C+" if val["C+"] > val["C0"] + 1e-9 else "C0"
    fe = arms[arm]
    M, G = mk().fit(trva[fe], trva.cls), mg().fit(trva[GEO], trva.cls)
    Pm, Pg = M.predict_proba(te[fe]), G.predict_proba(te[GEO])
    zm, zg = lgt(side(Pm)), lgt(side(Pg))
    bt = (te.cls > 0).to_numpy()
    yb = (te.cls[bt] == 1).to_numpy()
    aM, aG = roc_auc_score(yb, zm[bt]), roc_auc_score(yb, zg[bt])
    dayc = pd.factorize(te.day.to_numpy()[bt])[0]
    groups = [np.where(dayc == k)[0] for k in range(dayc.max() + 1)]
    delta_b = []
    for _ in range(NB):
        ix = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(yb[ix])) < 2:
            continue
        delta_b.append(roc_auc_score(yb[ix], zm[bt][ix]) - roc_auc_score(yb[ix], zg[bt][ix]))
    delta_b = np.array(delta_b)
    resid, pl = zm - zg, []
    cs = {c: np.where(te.coin.to_numpy() == c)[0] for c in rf.COINS}
    for _ in range(NPL):
        r = np.empty_like(resid)
        for c, ix in cs.items():
            nd = max(len(pd.unique(te.day.to_numpy()[ix])), 2)
            r[ix] = np.roll(resid[ix], int(rng.integers(1, nd)) * 96)
        pl.append(roc_auc_score(yb, (zg + r)[bt]) - aG)
    n_pred = (Pm.argmax(1) == te.cls.to_numpy())
    N_rule = ((te.dR < te.dS).to_numpy()[bt] == yb).mean()
    top = np.abs(zm[bt]) >= np.quantile(np.abs(zm[bt]), 0.9)
    res["C"] = {"H": H, "arm": arm, "val_dAUC": val, "n_test_rows": int(len(te)), "n_break_rows": int(bt.sum()), "days": int(te.day.nunique()),
                "up_share_among_breaks": float(yb.mean()), "AUC_M": float(aM), "AUC_G": float(aG), "dAUC": float(aM - aG), "p": float((delta_b <= 0).mean()),
                "lo10": float(np.percentile(delta_b, 10)), "placebo_p90": float(np.percentile(pl, 90)), "placebo_p95": float(np.percentile(pl, 95)),
                "acc_M_side": float(((zm[bt] > 0) == yb).mean()), "acc_G_side": float(((zg[bt] > 0) == yb).mean()), "acc_nearer_edge_rule": float(N_rule),
                "acc_M_top10conf": float(((zm[bt][top] > 0) == yb[top]).mean()), "n_top10": int(top.sum()),
                "acc_G_top10conf": float(((zg[bt][np.abs(zg[bt]) >= np.quantile(np.abs(zg[bt]), 0.9)] > 0) == yb[np.abs(zg[bt]) >= np.quantile(np.abs(zg[bt]), 0.9)]).mean()),
                "acc_nearer_on_M_top10": float(((te.dR < te.dS).to_numpy()[bt][top] == yb[top]).mean()),
                "logloss_M": float(log_loss(te.cls, Pm, labels=[0, 1, 2])), "logloss_G": float(log_loss(te.cls, Pg, labels=[0, 1, 2])),
                "byc": {c: float(roc_auc_score(yb[te.coin.to_numpy()[bt] == c], zm[bt][te.coin.to_numpy()[bt] == c]) - roc_auc_score(yb[te.coin.to_numpy()[bt] == c], zg[bt][te.coin.to_numpy()[bt] == c])) for c in rf.COINS}}
    print("C 시험:", json.dumps(res["C"], ensure_ascii=False, indent=1))


def verdict(res):
    ps = {"A-4h": res["A"]["pred|시험"]["q10"]["p"], "A-1h": res["A"]["p1h|시험"]["q10"]["p"], "C-1h": res["C"]["p"]}
    plc = {"A-4h": res["A"]["pred|시험"]["q10"]["E"] > res["A"]["pred|시험"]["placebo_E_q10"]["p95"],
           "A-1h": res["A"]["p1h|시험"]["q10"]["E"] > res["A"]["p1h|시험"]["placebo_E_q10"]["p95"],
           "C-1h": res["C"]["dAUC"] > res["C"]["placebo_p95"]}
    out, stop = {}, False
    for rank, (k, p) in enumerate(sorted(ps.items(), key=lambda kv: kv[1])):
        thr = 0.10 / (3 - rank)
        ok = (not stop) and p <= thr and plc[k]
        stop = stop or not ok
        out[k] = {"p": p, "holm_thr": thr, "placebo_pass": bool(plc[k]), "strict_pass": bool(ok)}
    res["verdict"] = out
    print("판정", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.load(open(OUT / "res.json")) if (OUT / "res.json").exists() and mode in ("c", "verdict") else {}
    if mode in ("align", "a", "all"):
        run_a(res)
        json.dump(res, open(OUT / "res.json", "w"), ensure_ascii=False, indent=1)
    if mode in ("c", "all"):
        run_c(res)
        json.dump(res, open(OUT / "res.json", "w"), ensure_ascii=False, indent=1)
    if mode in ("verdict", "all") and "A" in res and "C" in res:
        verdict(res)
        json.dump(res, open(OUT / "res.json", "w"), ensure_ascii=False, indent=1)
