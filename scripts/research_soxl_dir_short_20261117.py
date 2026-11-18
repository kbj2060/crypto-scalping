#!/usr/bin/env python3
"""SOXL 15분·1시간 방향 — 배포 65열판 레시피 E·S·J 팔 (사전등록 docs/prereg/soxl_dir_short_20261117.md, 8fb45f02).

레시피·표적·시드·학습 행 = scripts/live_eth_dir_short_20261028.py · 데이터셋 = research_xrp_dir_short_20261031.dataset65 (호가 깊이 없음).
대상 바꾸기 = d4.frames_for(frames, "SOXL") (SOXL 을 ETH 자리에, ETH 는 COINS 밖 자리로).

    python scripts/research_soxl_dir_short_20261117.py --build     # 패널 복사 + vision 일 파일 이어 붙이기(REST 없음)
    python scripts/research_soxl_dir_short_20261117.py --selftest
    python scripts/research_soxl_dir_short_20261117.py             # 학습 → tmp/soxl_dir_short_20261117/preds_<코인>.npz
    python scripts/research_soxl_dir_short_20261117.py --control   # 양성 대조(통과해야 --score)
    python scripts/research_soxl_dir_short_20261117.py --score     # SOXL 채점 → report.json
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir4h_20261015 as d4  # noqa: E402
import live_eth_dir_short_20261028 as ds  # noqa: E402
import research_xrp_dir_short_20261031 as xr  # noqa: E402

OUT = ROOT / "tmp/soxl_dir_short_20261117"
PAN = OUT / "panel"
WT = Path("/home/kbj20/crypto-scalping/.claude/worktrees/dashboard-features-direction-0a3150")
SRC = {**{s: WT / f"data/live/dirshort_panel/{s}USDT.parquet" for s in ["ETH"] + d4.COINS},
       "SOXL": WT / "tmp/stock_dir_short_20261104/SOXLUSDT.parquet"}
CHECK_DAY, END = "2026-11-01", "2026-11-16"
SYMS = ["ETH"] + d4.COINS + ["SOXL"]
TARGETS = ("ETH", "BTC", "SOL", "XRP", "SOXL")
POOL = ("ETH", "BTC", "SOL", "XRP")
MONTHS = [pd.Timestamp(m) for m in ("2026-09-01", "2026-10-01", "2026-11-01")]
VAL = (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-01"))
TEST = (pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-17"))
SOXL_START = pd.Timestamp("2026-05-22")
SMALL = ds.BASE | dict(num_leaves=7, min_child_samples=1000, n_estimators=150)
CANDS = {"E": dict(src=("ETH",), base=ds.BASE, sub=1, w=1.0), "S0": dict(src=("SOXL",), base=ds.BASE, sub=1, w=1.0),
         "S1": dict(src=("SOXL",), base=SMALL, sub=1, w=1.0), "J0": dict(src=POOL + ("SOXL",), base=ds.BASE, sub=4, w=1.0),
         "J1": dict(src=POOL + ("SOXL",), base=ds.BASE, sub=4, w=4.0)}
# 양성 대조: 코인 이식 결과 문서(coin_dir_short_btc_sol_20261031.md · xrp_dir_short_20261031.md) 월별 시험 IC, E 팔(9월, 10월)
CTRL = {("SOL", 3): (.055, .066), ("SOL", 12): (.050, .058), ("BTC", 12): (.067, .063), ("XRP", 12): (.052, .070)}
CTRL_END = pd.Timestamp("2026-10-28")      # 원 연구 데이터 끝(10-27 23:55 봉)
NB, NP = 2000, 200
D: dict = {}


# ── 데이터 ─────────────────────────────────────────────────────────────────
def build():
    from build_binance_vision_panel_20260915 import day
    PAN.mkdir(parents=True, exist_ok=True)
    dates = [str(d.date()) for d in pd.date_range(CHECK_DAY, END, freq="D")]
    jobs = [(s, d) for s in SYMS for d in dates]
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(jobs, ex.map(lambda j: day(f"{j[0]}USDT", j[1]), jobs)))
    rep = {}
    for s in SYMS:
        old = pd.read_parquet(SRC[s])
        assert old.timestamp.max() == pd.Timestamp(f"{CHECK_DAY} 23:55"), (s, old.timestamp.max())
        chk = got[(s, CHECK_DAY)].set_index("timestamp")[old.columns.drop("timestamp")]
        o = old[old.timestamp >= CHECK_DAY].set_index("timestamp")[chk.columns]
        a, b = o.to_numpy(float), chk.reindex(o.index).to_numpy(float)
        mism = int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        new = [got[(s, d)] for d in dates[1:] if got[(s, d)] is not None]
        if s == "SOXL":   # 11-04 사본에 빠진 날(10-28 -- 그때 vision 미공개)을 채운다. 코인 패널은 원 연구 그대로(양성 대조).
            have = set(old.timestamp.dt.normalize())
            new += [x for d in pd.date_range(old.timestamp.min().normalize(), CHECK_DAY) if d not in have
                    for x in [day("SOXLUSDT", str(d.date()))] if x is not None]
        df = pd.concat([old] + new).drop_duplicates("timestamp", keep="first").sort_values("timestamp").reset_index(drop=True)
        df.to_parquet(PAN / f"{s}USDT.parquet")
        rep[s] = {"check_day_mismatch": mism, "rows": len(df), "end": str(df.timestamp.max()), "new_days": len(new)}
        print(s, rep[s], flush=True)
    (OUT / "build.json").write_text(json.dumps(rep, indent=1))
    assert all(r["check_day_mismatch"] == 0 and r["end"] == f"{END} 23:55:00" for r in rep.values()), "패널 대조·끝 불일치"
    x = pd.read_parquet(PAN / "SOXLUSDT.parquet"); n = x.groupby(x.timestamp.dt.normalize()).size()
    assert (n.iloc[1:] == 288).all() and len(n) == (pd.Timestamp(END) - n.index[0]).days + 1, "SOXL 빠진 날·봉"


def frames() -> dict:
    return {s: pd.read_parquet(PAN / f"{s}USDT.parquet") for s in SYMS}


def build_ds(target):
    return xr.dataset65(d4.frames_for(frames(), target))


# ── 학습 ──────────────────────────────────────────────────────────────────
def rows(src, h, month):
    """학습 코인 src 의 학습 행(레시피 ds.train_rows · SOXL 은 05-22~)."""
    t, X, ret, sig = D[src]; t_ns = t.asi8
    m0 = int(np.searchsorted(t_ns, month.value))
    y = ds.fwd(ret, sig, h); ok = np.isfinite(ret) & (sig > 0) & np.isfinite(y)
    tr = ds.train_rows(t_ns, ok, m0, h)
    if src == "SOXL":
        tr = tr[t_ns[tr] >= SOXL_START.value]
    return tr, y


def assemble(cand, h, month):
    """후보 → (X, y, 가중치 | None). J = 코인 [::sub] + SOXL 전부, SOXL 가중 ×w · 15분 반감기 가중(레시피)."""
    c = CANDS[cand]; Xs, ys, ts, ws = [], [], [], []
    for s in c["src"]:
        tr, y = rows(s, h, month)
        if s != "SOXL":
            tr = tr[::c["sub"]]
        Xs.append(D[s][1][tr]); ys.append(y[tr]); ts.append(D[s][0].asi8[tr]); ws.append(np.full(len(tr), c["w"] if s == "SOXL" else 1.0))
    tt, w = np.concatenate(ts), np.concatenate(ws)
    hl = ds.HZ[h]
    if hl is not None:
        w = w * 0.5 ** ((month.value - tt) / ds.DAY / hl)
    assert tt.max() < month.value - (h + 1) * 300_000_000_000 + 1, "학습 행 표적이 그 달 첫 봉 뒤에 확정"
    return np.vstack(Xs), np.concatenate(ys), (None if hl is None and c["w"] == 1.0 else w)


def fit(job):
    import lightgbm as lgb
    cand, h, month = job
    X, y, w = assemble(cand, h, month)
    tg = TARGETS[1:] if cand == "E" else ("SOXL",)   # E 는 SOL·BTC·XRP 도 예측(양성 대조)
    end = month + pd.offsets.MonthBegin(1)
    sl = {g: slice(int(np.searchsorted(D[g][0].asi8, month.value)), int(np.searchsorted(D[g][0].asi8, end.value))) for g in tg}
    out = {g: [] for g in tg}
    base = CANDS[cand]["base"]
    for seed in ds.SEEDS:
        acc = {g: 0.0 for g in tg}
        for k in ds.ENS:
            m = lgb.LGBMRegressor(**base, **({"objective": "huber", "alpha": 1.0} if k == "huber" else {}), n_jobs=1, verbose=-1,
                                  random_state=seed).fit(X, ds.target(y, k), sample_weight=w)
            sc = float(m.predict(X[::50]).std() + 1e-12)
            for g in tg:
                acc[g] = acc[g] + m.predict(D[g][1][sl[g]]) / sc
        for g in tg:
            out[g].append(acc[g])
    print(f"  {cand} h{h} {month.date()} 학습 {len(y):,}행", flush=True)
    return cand, h, {g: (sl[g].start, np.array(v)) for g, v in out.items()}


def run():
    with get_context("fork").Pool(1, maxtasksperchild=1) as pool:   # 패널 21개는 버리는 자식에서 읽는다
        for s, d in zip(TARGETS, pool.imap(build_ds, TARGETS)):
            D[s] = d; print("dataset", s, d[1].shape, d[0][0], d[0][-1], flush=True)
    t, X = D["SOXL"][0], D["SOXL"][1]
    m = (t >= VAL[0]) & (t < TEST[1])
    nan = dict(zip(d4.FEATS, np.isnan(X[m]).mean(0).round(4).tolist()))
    (OUT / "nan_share_val_test.json").write_text(json.dumps(nan, indent=1))
    jobs = [(c, h, mo) for c in ("J0", "J1", "E", "S0", "S1") for h in ds.HZ for mo in MONTHS]
    P = {(g, c, h): np.full((len(ds.SEEDS), len(D[g][0])), np.nan) for c in CANDS for h in ds.HZ for g in (TARGETS[1:] if c == "E" else ("SOXL",))}
    with get_context("fork").Pool(3) as pool:
        for c, h, preds in pool.imap_unordered(fit, jobs):
            for g, (s0, pr) in preds.items():
                P[(g, c, h)][:, s0:s0 + pr.shape[1]] = pr
    for g in TARGETS[1:]:
        np.savez_compressed(OUT / f"preds_{g}.npz", t=D[g][0].asi8, ret=D[g][2], sig=D[g][3],
                            **{f"{c}_{h}": v for (gg, c, h), v in P.items() if gg == g})
        print("saved", g, flush=True)


# ── 대조·채점 ─────────────────────────────────────────────────────────────
def control():
    rep, ok = {}, True
    for (g, h), refs in CTRL.items():
        z = np.load(OUT / f"preds_{g}.npz"); t = pd.DatetimeIndex(z["t"]); y = ds.fwd(z["ret"], z["sig"], h); p = z[f"E_{h}"].mean(0)
        for mo, ref, end in ((MONTHS[0], refs[0], MONTHS[1]), (MONTHS[1], refs[1], CTRL_END - pd.Timedelta(minutes=5 * (h - 1)))):
            mm = (t >= mo) & (t < end) & np.isfinite(y)
            ic = xr.spear(p[mm], y[mm]); d = ic - ref
            rep[f"{g}_{ds.NAMES[h]}_{mo:%m}"] = {"ic": round(ic, 5), "doc": ref, "diff": round(d, 5), "n": int(mm.sum())}
            ok &= abs(d) <= .003
    rep["pass"] = bool(ok)
    (OUT / "control.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1)); print(json.dumps(rep, ensure_ascii=False, indent=1))
    return ok


def blocks(day, blk=5):
    ud = np.unique(day)
    return [np.flatnonzero(np.isin(day, ud[s:s + blk])) for s in range(0, len(ud), blk)]


def boot(fn, bl, rng, nb=NB):
    return np.array([fn(np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))])) for _ in range(nb)])


def q(a, p):
    return float(np.quantile(a, p))


def score():
    assert json.loads((OUT / "control.json").read_text())["pass"], "양성 대조 불통과 — 채점 금지(사전등록)"
    from research_soxl_session_vol_20261112 import SESS, clock
    z = np.load(OUT / "preds_SOXL.npz"); t = pd.DatetimeIndex(z["t"]); ret, sig = z["ret"], z["sig"]
    rng = np.random.default_rng(20261117); rep = {"prereg": "docs/prereg/soxl_dir_short_20261117.md@8fb45f02",
                                                 "nan_share_val_test": json.loads((OUT / "nan_share_val_test.json").read_text())}
    for h, nm in ds.NAMES.items():
        y = ds.fwd(ret, sig, h)
        mval = (t >= VAL[0]) & (t < VAL[1]) & np.isfinite(y)
        mtest = (t >= TEST[0]) & (t < TEST[1]) & np.isfinite(y)
        arms = {c: z[f"{c}_{h}"].mean(0) for c in CANDS}
        for c, p in arms.items():
            assert np.isfinite(p[mval]).all() and np.isfinite(p[mtest]).all(), f"{c} 예측 구멍"
        val_ic = {c: xr.spear(p[mval], y[mval]) for c, p in arms.items()}
        pick = max(val_ic, key=val_ic.get)
        yt, day = y[mtest], t[mtest].normalize().asi8; bl = blocks(day)
        r = {"val_ic": val_ic, "pick": pick, "n_val": int(mval.sum()), "n_test": int(mtest.sum()),
             "val_days": int(len(np.unique(t[mval].normalize()))), "test_days": int(len(np.unique(day)))}
        for c, p in arms.items():
            pt = p[mtest]; bs = boot(lambda i: xr.spear(pt[i], yt[i]), bl, rng)
            r[c] = {"test_ic": xr.spear(pt, yt), "p10": q(bs, .10), "p05": q(bs, .05), "p95": q(bs, .95),
                    "seed_ic": [xr.spear(z[f"{c}_{h}"][i][mtest], yt) for i in range(len(ds.SEEDS))],
                    "month_ic": {f"{mo:%m}": xr.spear(p[mm], y[mm]) for mo in MONTHS[1:] for mm in [(t >= mo) & (t < mo + pd.offsets.MonthBegin(1)) & mtest]}}
        ic = r[pick]["test_ic"]; pt = arms[pick][mtest]
        # 위약: 시험 창 안 날짜 원형 이동(1~일수−1 전부) · 7의 배수 · 무작위
        idx = np.flatnonzero((t >= TEST[0]) & (t < TEST[1])); ok = np.isfinite(y[idx]); pf = arms[pick][idx]; nday = len(idx) // 288
        circ = {k: xr.spear(np.roll(pf, k * 288)[ok], y[idx][ok]) for k in range(1, nday)}
        rnd = np.array([xr.spear(rng.standard_normal(ok.sum()), y[idx][ok]) for _ in range(NP)])
        cv = np.array(list(circ.values()))
        r["placebo"] = {"n_circ": len(cv), "circ_q90": q(cv, .90), "circ_q95": q(cv, .95), "circ_pct": float((cv < ic).mean()),
                        "circ_7x": {k: v for k, v in circ.items() if k % 7 == 0}, "rand_q95": q(rnd, .95)}
        # 표시 규칙(검증 20/80) · 모델−늘 롱 · 모델−늘 숏
        lo, hi = np.quantile(arms[pick][mval], [.2, .8]); up = yt > 0; dn = yt < 0
        tail = (pt <= lo) | (pt >= hi); hit = np.where(pt >= hi, up, dn)
        ti = np.flatnonzero(tail); tday = day[ti]; tbl = blocks(tday)
        hT, uT, dT = hit[ti].astype(float), up[ti].astype(float), dn[ti].astype(float)
        bl_ = boot(lambda i: hT[i].mean() - uT[i].mean(), tbl, rng); bs_ = boot(lambda i: hT[i].mean() - dT[i].mean(), tbl, rng)
        r["vs_always"] = {"tail_hits": f"{int(hT.sum())}/{len(ti)}", "hit": float(hT.mean()), "always_long": float(uT.mean()), "always_short": float(dT.mean()),
                          "m_minus_long": float(hT.mean() - uT.mean()), "mL_p10": q(bl_, .10), "mL_p90": q(bl_, .90),
                          "m_minus_short": float(hT.mean() - dT.mean()), "mS_p10": q(bs_, .10), "mS_p90": q(bs_, .90)}
        r["display"] = {"thr": [float(lo), float(hi)], "long_hits": f"{int(up[pt >= hi].sum())}/{int((pt >= hi).sum())}",
                        "short_hits": f"{int(dn[pt <= lo].sum())}/{int((pt <= lo).sum())}", "mid_rate": float((np.sign(pt[~tail] - np.median(arms[pick][mval])) == np.sign(yt[~tail])).mean()),
                        "base_up": float(up.mean())}
        sess = clock(t[mtest].asi8 // 10**9, h * 5).sess.to_numpy()
        r["session_ic"] = {SESS[s]: {"ic": xr.spear(pt[sess == s], yt[sess == s]), "n": int((sess == s).sum()),
                                     "days": int(len(np.unique(day[sess == s])))} for s in range(len(SESS)) if (sess == s).sum() > 100}
        r["tripwire_ic_gt_08"] = bool(max(r[c]["test_ic"] for c in CANDS) > .08)
        v = r["vs_always"]
        strict = r[pick]["p10"] > 0 and ic > r["placebo"]["circ_q95"] and ic > r["placebo"]["rand_q95"] and v["mL_p10"] > 0 and v["mS_p10"] > 0
        cand = (not strict) and ic > 0 and ic > r["placebo"]["circ_q90"]
        r["verdict"] = "엄격 통과" if strict else ("통과 후보" if cand else "불통과")
        rep[nm] = r
        print(nm, "pick", pick, {c: round(x, 4) for c, x in val_ic.items()}, "test", round(ic, 4), "p10", round(r[pick]["p10"], 4),
              "circ95", round(r["placebo"]["circ_q95"], 4), "mL", round(v["mL_p10"], 4), "mS", round(v["mS_p10"], 4), r["verdict"], flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


def selftest():
    f = {s: x[x.timestamp >= "2026-05-01"] for s, x in frames().items()}
    t, X, ret, sig = xr.dataset65(d4.frames_for(f, "SOXL"))
    raw = f["SOXL"].set_index("timestamp").close; lc = np.log(raw)
    j, i = d4.FEATS.index("P_r5m"), 20000
    want = (lc[t[i] - pd.Timedelta("5min")] - lc[t[i] - pd.Timedelta("10min")]) * 1e4   # 결정 t 피쳐 = 봉 t−5분(대상 SOXL 자신)
    assert abs(X[i, j] - want) < 1e-3, (X[i, j], want)
    assert abs(ret[i] - (raw[t[i]] / raw[t[i] - pd.Timedelta("5min")] - 1)) < 1e-12           # 표적 첫 봉 = 결정 뒤 봉
    jb = d4.FEATS.index("X_btc_r5m"); lb = np.log(f["BTC"].set_index("timestamp").close)
    assert abs(X[i, jb] - (lb[t[i] - pd.Timedelta("5min")] - lb[t[i] - pd.Timedelta("10min")]) * 1e4) < 1e-3   # BTC 자리 = BTC
    D["SOXL"] = (t, X, ret, sig)
    for h in ds.HZ:
        tr, _ = rows("SOXL", h, MONTHS[0])
        assert t[tr[0]] >= SOXL_START and t[tr[-1]] + pd.Timedelta(minutes=5 * (h + 1)) <= MONTHS[0], (t[tr[0]], t[tr[-1]])
    D["ETH"] = D["SOXL"]                                                                         # 가중치 조립만 본다(가짜 코인 행)
    CANDS["_t"] = dict(src=("ETH", "SOXL"), base=ds.BASE, sub=4, w=4.0)
    Xa, ya, w = assemble("_t", 12, MONTHS[0]); n_e = len(rows("ETH", 12, MONTHS[0])[0][::4])
    assert len(ya) == n_e + len(rows("SOXL", 12, MONTHS[0])[0]) and (w[:n_e] == 1).all() and (w[n_e:] == 4).all()
    _, _, w3 = assemble("_t", 3, MONTHS[0]); assert w3[n_e:].max() <= 4 and w3[n_e - 1] < 1
    del CANDS["_t"]
    rng = np.random.default_rng(0); yy = rng.standard_normal(3000); bl = blocks(np.repeat(np.arange(30), 100))
    assert abs(np.median(boot(lambda k: xr.spear(yy[k], yy[k]), bl, rng, 50)) - 1) < 1e-9
    xx = rng.standard_normal(3000); assert abs(np.median(boot(lambda k: xr.spear(xx[k], yy[k]), bl, rng, 200))) < .03
    print("selftest ok")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--build" in sys.argv:
        build()
    elif "--selftest" in sys.argv:
        selftest()
    elif "--control" in sys.argv:
        sys.exit(0 if control() else 1)
    elif "--score" in sys.argv:
        score()
    else:
        run()
