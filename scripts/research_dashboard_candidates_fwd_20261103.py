#!/usr/bin/env python3
"""전진 재검정 (사전등록 부록: docs/prereg/dashboard_candidates_forward_20261103.md, 3dcfbc2a).
원 스크립트 research_dashboard_candidates_20261103 의 동결 사양을 서버 vision 패널(복사본) 위 새 구간에 한 번 적용.
사용: python3 scripts/research_dashboard_candidates_fwd_20261103.py [CTRL T1 T2 T3]
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("R", HERE / "research_dashboard_candidates_20261103.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
ROOT = HERE.parent; PANEL = ROOT / "data/live/dirshort_panel"; OUT = ROOT / "tmp/dashboard_candidates_20261103"
F1 = ("2026-08-23", "2026-11-01 23:55")   # T1·T2 전진 (패널 끝 = 11-01 23:55)
F3 = ("2026-10-24", "2026-11-01 23:55")   # T3 전진


def pan(coin):
    p = pd.read_parquet(PANEL / f"{coin.upper()}USDT.parquet").drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return p


def ctrl():
    """패널 ↔ 로컬 API 대조(겹침 2026-06-01~08-22)."""
    w = slice("2026-06-01", "2026-08-22 23:55"); out = {}
    for c in ["eth", "btc", "sol", "xrp", "ada", "avax", "bnb", "doge", "link"]:
        a, b = pan(c).close[w], R.load(c).close[w].reindex(pan(c).close[w].index)
        out[c] = float((abs(a / b - 1) > 1e-4).mean())
    m = R.load_eth_with_metrics()[w]; q = pan("eth")[w]
    out["eth_oi_gt1pct"] = float((abs(q.sum_open_interest / m.sum_open_interest - 1) > 1e-2).mean())
    out["eth_toplsr_gt1pct"] = float((abs(q.sum_toptrader_long_short_ratio / m.sum_toptrader_long_short_ratio - 1) > 1e-2).mean())
    out["pass"] = bool(all(v <= 0.01 for v in out.values()))
    return out


def t1():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, log_loss
    Bf, Xf, dd = R.feats(); ix = Bf.index
    fin = lambda B, X, d: np.isfinite(B).all(axis=1) & np.isfinite(X).all(axis=1) & d.notna()
    ok = fin(Bf, Xf, dd)
    tr = ok & (ix >= R.TR[0]) & (ix <= R.TR[1] + " 23:55"); trs = tr & (np.arange(len(ix)) % 3 == 0)
    thr = float(np.quantile(dd[tr], .02)); allf = pd.concat([Bf, Xf], axis=1)
    cols_b, cols_f = list(Bf.columns), list(allf.columns)
    fit = {}
    for nm, cols in (("B", cols_b), ("F", cols_f)):
        A = allf[cols]; mu, sd = A[trs].mean(), A[trs].std()
        fit[nm] = (LogisticRegression(C=0.01, max_iter=500).fit(((A - mu) / sd)[trs], (dd[trs] <= thr).astype(int)), mu, sd, cols)
    def score(B, X, d, mask):
        a = pd.concat([B, X], axis=1)[mask]; y = (d[mask] <= thr).astype(int).values
        return y, {nm: m.predict_proba(((a[c] - mu) / sd).values)[:, 1] for nm, (m, mu, sd, c) in fit.items()}
    # 재현 점검: 원 시험 구간 ΔAUC 가 원 결과(+.0211)와 같아야 한다
    te0 = ok & (ix >= R.TE[0]) & (ix <= R.TE[1]); y0, p0 = score(Bf, Xf, dd, te0)
    repro = roc_auc_score(y0, p0["F"]) - roc_auc_score(y0, p0["B"])
    # 전진: 패널 ETH
    k = pan("eth"); B2, X2, dd2 = R.feats_from(k); ix2 = B2.index
    fw = fin(B2, X2, dd2) & (ix2 >= F1[0]) & (ix2 <= F1[1])
    y, p = score(B2, X2, dd2, fw); day = ix2[fw].normalize().values
    ev_days = int(pd.Series(y, index=day).groupby(level=0).max().sum())
    d_auc = roc_auc_score(y, p["F"]) - roc_auc_score(y, p["B"]); d_ll = log_loss(y, p["B"]) - log_loss(y, p["F"])
    ud = np.unique(day); by = {d: np.where(day == d)[0] for d in ud}; bs = []
    for _ in range(R.B):
        ii = np.concatenate([by[d] for d in R.RNG.choice(ud, len(ud))])
        if y[ii].min() != y[ii].max(): bs.append(roc_auc_score(y[ii], p["F"][ii]) - roc_auc_score(y[ii], p["B"][ii]))
    lo90 = float(np.quantile(bs, .10))
    st = "보류(검정력)" if ev_days < 10 else ("통과" if (lo90 > 0 and d_ll > 0) else "불통과")
    return {"repro_dAUC_orig_test(+.0211 기대)": float(repro), "threshold": thr, "rows": int(fw.sum()), "days": len(ud), "events": int(y.sum()),
            "event_days": ev_days, "base_rate": float(y.mean()), "auc_base": float(roc_auc_score(y, p["B"])), "auc_full": float(roc_auc_score(y, p["F"])),
            "dAUC": float(d_auc), "dAUC_lo90": lo90, "dLogLoss": float(d_ll), "P1": st}


def t2():
    cs = ["eth", "btc", "sol", "xrp", "ada", "avax", "bnb", "doge", "link"]
    px = pd.DataFrame({c: pan(c).close for c in cs}); t = px.index[(px.index.hour % 4 == 3) & (px.index.minute == 55)]
    lab = (px.shift(-48) / px - 1).loc[t]; lab = lab.sub(lab.mean(1), axis=0); res = {}
    for nm, lb in (("P2_4h", 48), ("P3_24h", 288), ("P4_72h", 864), ("P5_168h", 2016)):
        f = (px / px.shift(lb) - 1).loc[t]; f = f.sub(f.mean(1), axis=0)
        m = f.notna().all(1) & lab.notna().all(1) & (f.index >= F1[0]) & (f.index <= F1[1])
        fr, lr = f[m].rank(axis=1), lab[m].rank(axis=1); a, b = fr.sub(fr.mean(1), axis=0), lr.sub(lr.mean(1), axis=0)
        ic = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
        d = pd.DataFrame({"d": ic.index.normalize(), "v": ic.values}).groupby("d").v.agg(["sum", "count"]).values
        bs = np.array([(lambda i: d[i, 0].sum() / d[i, 1].sum())(R.RNG.choice(len(d), len(d))) for _ in range(R.B)])
        res[nm] = {"mean_ic": float(ic.mean()), "n": int(len(ic)), "days": len(d), "upper90": float(np.quantile(bs, .90)),
                   "pass": bool(np.quantile(bs, .90) < 0) if nm == "P2_4h" else None}
    return res


def t3():
    import joblib
    import live_eth_dir_short_20261028 as L
    art = joblib.load(ROOT / "data/live/eth_dirshort_model.joblib")
    frames = {s: pd.read_parquet(PANEL / f"{s}USDT.parquet") for s in ["ETH"] + L.d4.COINS}
    book = pd.read_parquet(ROOT / "data/live/dirshort_book.parquet")
    t, X, _, _ = L.dataset(frames, book)
    ts = t                                                         # 씨앗 ts = 결정 시각(봉 마감). 11-03 정정: 처음엔 t-5분(봉 시작)으로 둬 대조가 어긋났다
    okx = np.isfinite(X).all(1); P = L.predict(art, X[okx]); pred = pd.DataFrame({"p15m": P[3], "p1h": P[12]}, index=ts[okx])
    seed = pd.read_parquet(ROOT / "data/live/eth_dirshort_oos_seed.parquet").set_index("ts")
    j = seed.index.intersection(pred.index)
    j = j[(j >= "2026-10-01")]                                      # 모형 학습 경계 뒤 겹침
    par = {c: float(np.abs(pred.loc[j, c] - seed.loc[j, c]).max()) for c in ("p15m", "p1h")} | {"overlap_rows": int(len(j))}
    close = frames["ETH"].drop_duplicates("timestamp").set_index("timestamp").close.sort_index(); res = {"parity_vs_seed_1001_1023": par}
    hist = pd.concat([seed, pred.loc[pred.index > seed.index.max()]])
    for name, col, h in (("P6_15m", "p15m", 3), ("P7_1h", "p1h", 12)):
        ret = (close.shift(-(h - 1)) / close.shift(1) - 1)   # 결정 시각 ts 의 결정가 = 직전 봉(시작 ts-5분) 종가
        ix = pred.index[(pred.index >= F3[0]) & (pred.index <= F3[1])]
        r = ret.reindex(ix); h_ = hist[col]; days = np.unique(ix.normalize()); rows = []
        for d in days:
            past = h_[h_.index < d].values; hi, lo = np.quantile(past, .8), np.quantile(past, .2)
            m = ix.normalize() == d; p = pred.loc[ix[m], col]; rr = r[m]; g = rr.notna()
            rows.append([g.sum(), (rr[g] > 0).sum(), (rr[g] < 0).sum(), (p[g] >= hi).sum(), ((rr > 0) & (p >= hi))[g].sum(), (p[g] <= lo).sum(), ((rr < 0) & (p <= lo))[g].sum()])
        a = np.array(rows, float)
        def parts(i):
            s = a[i].sum(0); return s[4] / s[3] - s[1] / s[0], s[6] / s[5] - s[2] / s[0]
        full = np.arange(len(a)); bs = np.array([np.mean(parts(R.RNG.choice(full, len(full)))) for _ in range(R.B)])
        s = a.sum(0); u, d_ = parts(full)
        res[name] = {"days": len(a), "rows": int(s[0]), "top_hit": f"{int(s[4])}/{int(s[3])}", "top_base_up": s[1] / s[0], "bot_hit": f"{int(s[6])}/{int(s[5])}",
                     "bot_base_dn": s[2] / s[0], "pooled_excess": (u + d_) / 2, "lo90": float(np.quantile(bs, .10)),
                     "daily": [[str(d)[:10], int(r[3]), int(r[4]), int(r[5]), int(r[6])] for d, r in zip(days, rows)],   # 일, 상위 건수·적중, 하위 건수·적중
                     "verdict": "통과" if np.quantile(bs, .10) > 0 else "판정 불가(표본 짧음·하한 ≤ 0)"}
    return res


if __name__ == "__main__":
    sel = [a for a in sys.argv[1:] if a in ("CTRL", "T1", "T2", "T3")] or ["CTRL", "T1", "T2", "T3"]
    OUT.mkdir(parents=True, exist_ok=True)
    for n, fn in {"CTRL": ctrl, "T1": t1, "T2": t2, "T3": t3}.items():
        if n in sel:
            r = fn(); (OUT / f"fwd_{n}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float))
            print(n, json.dumps(r, ensure_ascii=False, indent=1, default=float), flush=True)
            if n == "CTRL" and not r["pass"]: sys.exit("대조 실패 -- 판정 중단")
