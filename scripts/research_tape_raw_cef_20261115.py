"""체결 원본(바이낸스 선물 aggTrades) — C 가격 충격 · E 자기 흥분(Hawkes) · F VPIN 이 가격·체결 기준선 ⓪ 위에 더 주나 (2026-11-15).

사용자 «C·E·F 원본으로» · «사전등록하지 말고 테스트 결과는 없어?» → 결과부터. 결과 문서: docs/experiments/tape_raw_cef_20261115.md
단계: --dl COIN(일별 zip → 초 표) → --frame COIN(1분 격자 피쳐·대상) → --fit ARM → --score. --selftest.
시각 규약: 격자 행 t = 초 s 의 끝(=다음 초 시작). 피쳐는 초 ≤ s(체결 시각 < t) · 대상은 초 s+1..s+h(체결 시각 ≥ t).
분할: 학습 2024-01~2025-12 · 검증 2026-01~05(조기 종료만) · 시험 2026-06-01~11-14 (경계 앞 1시간 버림).
"""
import argparse, io, os, sys, urllib.error, urllib.request, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path(os.environ.get("TAPERAW_OUT", ROOT / "tmp/tape_raw_20261115"))
COINS = ("ETH", "BTC", "SOL", "XRP")
D0, D1 = "2024-01-01", "2026-11-14"
TAUS = (1, 10, 60, 600)                       # Hawkes 지수 커널 시간(초)
H = (10, 60, 300, 900)
FAM = {"dir": "bin", "jump": "bin", "rv": "reg", "cnt": "reg", "flow": "reg"}
TARGETS = [f"y_{fm}{h}" for fm in FAM for h in H]
ARMS = {"0": "p", "C": "pc", "E": "pe", "F": "pf", "Z": "pz", "ALL": "pcef"}
SPLIT = (int(pd.Timestamp("2026-01-01").timestamp()), int(pd.Timestamp("2026-06-01").timestamp()))
SEED = 20261115
URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/{s}USDT/{s}USDT-aggTrades-{d}.zip"
N = 86400
NAN = np.nan


# ---------- 1. 일별 원본 → 초 표 ----------
def pack(ts, px, qty, bm):
    """ts = 그 날 0시부터 ms · bm = is_buyer_maker(True ⇒ 공격자 매도). 초별: 체결 수·명목($)·마지막 가격·Hawkes 초 안 기여."""
    o = np.argsort(ts, kind="stable"); ts, px, qty, bm = ts[o], px[o], qty[o], bm[o]
    s = ts // 1000; q = px * qty; out = {}
    for nm, m in (("b", ~bm), ("s", bm)):
        out[f"n_{nm}"] = np.bincount(s[m], minlength=N).astype(np.float32)
        out[f"q_{nm}"] = np.bincount(s[m], q[m], minlength=N).astype(np.float32)
        for tau in TAUS:   # 초 끝 시각에서 본 그 초 체결들의 exp(−경과/τ) 합 → 초 사이는 lfilter 로 정확히 이어짐
            out[f"h{tau}_{nm}"] = np.bincount(s[m], np.exp(-((s[m] + 1) * 1000 - ts[m]) / (1000.0 * tau)), minlength=N).astype(np.float32)
    last = np.searchsorted(s, np.arange(N), side="right") - 1
    has = np.bincount(s, minlength=N) > 0
    out["px"] = np.where(has, px[np.clip(last, 0, None)], NAN)
    return pd.DataFrame(out)


def sec_day(args):
    coin, d = args
    f = OUT / "sec" / coin / f"{d}.parquet"
    if f.exists():
        return d, "have"
    try:
        raw = urllib.request.urlopen(URL.format(s=coin, d=d), timeout=180).read()
    except urllib.error.HTTPError as e:
        return d, f"http{e.code}"
    z = zipfile.ZipFile(io.BytesIO(raw)); name = z.namelist()[0]
    head = z.open(name).readline()
    df = pd.read_csv(z.open(name), header=0 if head[:1].isalpha() else None,
                     names=["id", "price", "qty", "f", "l", "ts", "bm"], usecols=["price", "qty", "ts", "bm"])
    bm = df.bm.values if df.bm.dtype == bool else df.bm.astype(str).str.lower().eq("true").values
    t0 = int(pd.Timestamp(d).timestamp()) * 1000
    ts = df.ts.values.astype(np.int64) - t0
    k = (ts >= 0) & (ts < N * 1000)
    f.parent.mkdir(parents=True, exist_ok=True)
    pack(ts[k], df.price.values[k].astype(float), df.qty.values[k].astype(float), bm[k]).to_parquet(f)
    return d, f"ok {k.sum()}"


# ---------- 2. 초 표 → 1분 격자 피쳐·대상 ----------
def C(x):
    return np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))


def win(cs, g, k):      # 초 (g−k, g] 합
    lo = g + 1 - k
    return np.where(lo >= 0, cs[g + 1] - cs[np.clip(lo, 0, None)], NAN)


def fwd(cs, g, h):      # 초 (g, g+h] 합
    hi = g + 1 + h
    return np.where(hi < len(cs), cs[np.clip(hi, 0, len(cs) - 1)] - cs[g + 1], NAN)


def ratio(a, b):
    with np.errstate(all="ignore"):
        return np.where(b > 0, a / b, NAN)


def vpin(vol, f, g, dstart, div, ns):
    """부피 시계 VPIN(Easley·López de Prado·O'Hara 2012) — 통 크기 = 전날 명목/div · 공격 방향은 aggTrades 실제 부호.
    반환 {n: 최근 완성 통 n개의 |매수−매도|/통 크기 평균}, 부호판, 최근 10통 걸린 초."""
    Cs, Ds = np.cumsum(vol, dtype=np.float64), np.cumsum(f, dtype=np.float64)
    B = []
    for i in range(1, len(dstart)):
        a = dstart[i]; e = dstart[i + 1] if i + 1 < len(dstart) else len(vol)
        V = (Cs[a - 1] - (Cs[dstart[i - 1] - 1] if dstart[i - 1] > 0 else 0.0)) / div
        if V > 0 and Cs[e - 1] > Cs[a - 1]:
            B.append(Cs[a - 1] + V * np.arange(1, int((Cs[e - 1] - Cs[a - 1]) // V) + 1))
    B = np.concatenate(B) if B else np.array([0.0, 1.0])
    sel = vol > 0
    DB = np.interp(B, Cs[sel], Ds[sel]); TB = np.interp(B, Cs[sel], np.flatnonzero(sel).astype(float))
    dB, dd = np.diff(B), np.diff(DB)
    A, Sg = C(np.abs(dd) / dB), C(dd / dB)
    k = np.searchsorted(B, Cs[g], side="right") - 1          # 완성된 통 수
    out = {}
    for n in ns:
        lo = k - n; ok = lo >= 0
        out[n] = np.where(ok, (A[np.clip(k, 0, None)] - A[np.clip(lo, 0, None)]) / n, NAN)
        out[f"s{n}"] = np.where(ok, (Sg[np.clip(k, 0, None)] - Sg[np.clip(lo, 0, None)]) / n, NAN)
    lo = k - 10
    out["rate"] = np.where(lo >= 0, g - TB[np.clip(lo, 0, None)], NAN)
    return out


def build(S, ok, dstart):
    """S: 연속 초 배열 dict · ok: 그 초의 날 파일 있음 · dstart: 날 시작 초 인덱스. 반환: 격자 인덱스 g, 열 dict."""
    n_all = len(ok); g = np.arange(59, n_all, 60)
    qb, qs = S["q_b"].astype(float), S["q_s"].astype(float)
    n = S["n_b"] + S["n_s"]; vol = (qb + qs) / 1e6; f = (qb - qs) / 1e6         # $M
    lp = np.log(pd.Series(S["px"]).ffill().values)
    r1 = np.nan_to_num(np.diff(lp, prepend=NAN)) * 1e4                           # bp
    cok = C(ok); M = {}

    def okw(k):
        return win(cok, g, k) == k

    def ws(x, k):
        return np.where(okw(k), win(C(x), g, k), NAN)

    o = {}
    # ⓪ 가격·체결 기준선(분모 포함: 거래량·체결 수를 같은 창으로)
    for k in (1, 10, 60, 300, 900, 3600):
        o[f"p_r{k}"] = np.where(okw(k + 1) & (g - k >= 0), (lp[g] - lp[np.clip(g - k, 0, None)]) * 1e4, NAN)
    for k in (60, 300, 3600):
        o[f"p_rv{k}"] = np.sqrt(ws(r1 ** 2, k))
    for k in (1, 10, 60, 300):
        o[f"p_imb{k}"] = ratio(ws(qb - qs, k), ws(qb + qs, k))
    for k in (10, 60, 300, 3600):
        o[f"p_vol{k}"] = ws(vol, k); o[f"p_cnt{k}"] = ws(n, k)
    # C 가격 충격 상태 — 초 수익(bp)을 초 부호 명목($M)에 원점 회귀(Kyle λ)
    rf, ff, rr = r1 * f, f * f, r1 * r1
    for k in (60, 300, 1800):
        o[f"c_kyle{k}"] = ratio(ws(rf, k), ws(ff, k))
    o["c_kr2_300"] = ratio(ws(rf, 300) ** 2, ws(ff, 300) * ws(rr, 300))
    fp, fn = np.maximum(f, 0), np.minimum(f, 0)
    o["c_kasym300"] = ratio(ws(r1 * fp, 300), ws(fp * fp, 300)) - ratio(ws(r1 * fn, 300), ws(fn * fn, 300))
    o["c_amihud300"] = ratio(ws(np.abs(r1), 300), ws(vol, 300))
    for k in (60, 300):
        o[f"c_absorb{k}"] = o[f"p_r{k}"] - o["c_kyle1800"] * ws(f, k)          # 흐름이 설명 못 한 가격 움직임
    # E 자기 흥분 — 지수 커널 강도(ms 시각 그대로), 체결/초
    lam = {}
    for tau in TAUS:
        a = np.exp(-1.0 / tau); m3 = okw(3 * tau)
        for sd in "bs":
            Hs = lfilter([1.0], [1.0, -a], S[f"h{tau}_{sd}"].astype(float))
            lam[tau, sd] = np.where(m3, Hs[g] / tau, NAN)
        o[f"e_lb{tau}"], o[f"e_ls{tau}"] = lam[tau, "b"], lam[tau, "s"]
    tot = {t: lam[t, "b"] + lam[t, "s"] for t in TAUS}
    for tau in (1, 10, 60):
        o[f"e_imb{tau}"] = ratio(lam[tau, "b"] - lam[tau, "s"], tot[tau])
    o["e_burst1_60"], o["e_burst10_600"] = ratio(tot[1], tot[60]), ratio(tot[10], tot[600])
    c10 = win(C(n), np.arange(n_all), 10)[9::10]                                 # 10초 통 체결 수
    j = (g - 9) // 10; c1, c2 = C(np.nan_to_num(c10)), C(np.nan_to_num(c10) ** 2)
    lo = np.clip(j + 1 - 360, 0, None)
    mu = (c1[j + 1] - c1[lo]) / 360; var = (c2[j + 1] - c2[lo]) / 360 - mu ** 2
    with np.errstate(all="ignore"):   # 분지 비율 근사 n = 1 − √(평균/분산)(Hardiman·Bercot·Bouchaud 2013 식의 1시간판)
        o["e_nhat"] = np.where(okw(3600) & (var > mu) & (mu > 0), 1 - np.sqrt(mu / var), np.where(okw(3600), 0.0, NAN))
    # F VPIN — 하루 50통(원 논문 설정) + 하루 1000통(분 단위 지평용)
    m = okw(3600)
    va, vb = vpin(vol, f, g, dstart, 50, (50,)), vpin(vol, f, g, dstart, 1000, (10, 50))
    o["f_vpin_d50"] = np.where(m, va[50], NAN)
    o["f_vpin_k50"], o["f_vpin_k10"] = np.where(m, vb[50], NAN), np.where(m, vb[10], NAN)
    o["f_svpin_k50"], o["f_rate_k10"] = np.where(m, vb["s50"], NAN), np.where(m, vb["rate"], NAN)
    v = pd.Series(o["f_vpin_k50"])
    o["f_vpinz_k50"] = ((v - v.rolling(1440, min_periods=720).mean()) / v.rolling(1440, min_periods=720).std()).values
    # 대상 (g, g+h]
    cq, cs_, cn, crr = C(qb), C(qs), C(n), C(rr)
    rv_sig = o["p_rv3600"] / 60.0                                                 # 초당 σ(bp)
    for h in H:
        okf = (fwd(cok, g, h) == h) & ok[g]
        hi = np.clip(g + h, 0, n_all - 1)
        r = np.where(okf, (lp[hi] - lp[g]) * 1e4, NAN)
        o[f"y_dir{h}"] = np.where(r > 0, 1.0, np.where(r < 0, 0.0, NAN))
        o[f"y_jump{h}"] = np.where(np.isfinite(r) & np.isfinite(rv_sig), (np.abs(r) > 3 * rv_sig * np.sqrt(h)).astype(float), NAN)
        o[f"y_rv{h}"] = np.where(okf, np.log(fwd(crr, g, h) + 0.01), NAN)
        o[f"y_cnt{h}"] = np.where(okf, np.log1p(fwd(cn, g, h)), NAN)
        o[f"y_flow{h}"] = np.where(okf, ratio(fwd(cq, g, h) - fwd(cs_, g, h), fwd(cq, g, h) + fwd(cs_, g, h)), NAN)
    return g, o


SEC_COLS = ["n_b", "n_s", "q_b", "q_s", "px"] + [f"h{t}_{s}" for t in TAUS for s in "bs"]


def frame_coin(coin):
    days = pd.date_range(D0, D1)
    parts = []
    for m0 in pd.date_range(D0, D1, freq="MS"):
        mdays = days[(days >= m0) & (days < m0 + pd.offsets.MonthBegin(1))]
        cd = pd.date_range(mdays[0] - pd.Timedelta(days=2), mdays[-1] + pd.Timedelta(days=1))
        S = {c: [] for c in SEC_COLS}; ok = []
        for d in cd:
            fp = OUT / "sec" / coin / f"{d.date()}.parquet"
            x = pd.read_parquet(fp) if fp.exists() else None
            for c in SEC_COLS:
                S[c].append(x[c].values if x is not None else np.full(N, NAN if c == "px" else 0.0, np.float32))
            ok.append(np.full(N, x is not None))
        S = {c: np.concatenate(v) for c, v in S.items()}; ok = np.concatenate(ok)
        g, o = build(S, ok, np.arange(len(cd)) * N)
        df = pd.DataFrame(o).astype(np.float32)
        for c in [c for c in df if c[:2] in ("c_", "e_", "f_")]:
            df["z_" + c] = df[c].shift(1440)                                      # 위약: 같은 피쳐의 하루 전 값
        df.insert(0, "t", int(cd[0].timestamp()) + g + 1)
        keep = (df.t >= int(mdays[0].timestamp()) + 60) & (df.t <= int(mdays[-1].timestamp()) + N) & ok[g]
        parts.append(df[keep.values]); print(coin, m0.date(), int(keep.sum()), flush=True)
    F = pd.concat(parts, ignore_index=True)
    F.to_parquet(OUT / f"frame_{coin}.parquet")
    print(coin, "rows", len(F), "target notna", {y: int(F[y].notna().sum()) for y in ("y_dir10", "y_jump900", "y_flow10")})


# ---------- 3. 적합·채점 ----------
def load(cols=None):
    return pd.concat([pd.read_parquet(OUT / f"frame_{c}.parquet", columns=cols).assign(coin=i) for i, c in enumerate(COINS)],
                     ignore_index=True)


def split_of(t):
    return np.select([t < SPLIT[0] - 3600, (t >= SPLIT[0]) & (t < SPLIT[1] - 3600), t >= SPLIT[1]], [0, 1, 2], -1)


def loss(kind, y, p):
    if kind == "bin":
        p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return (y - p) ** 2


def fit(arm):
    import lightgbm as lgb
    import pyarrow.parquet as pq
    pre = {x + "_" for x in ARMS[arm]}
    X = [c for c in pq.read_schema(OUT / f"frame_{COINS[0]}.parquet").names if c[:2] in pre] + ["coin"]
    df = load(["t"] + [c for c in X if c != "coin"] + TARGETS)
    sp = split_of(df.t.values)
    od = OUT / "fit"; (od / "models").mkdir(parents=True, exist_ok=True)
    out = []
    for y in TARGETS:
        kind = FAM[y[2:].rstrip("0123456789")]
        mf = od / "models" / f"arm{arm}_{y}.txt"
        yy = df[y].values; has = np.isfinite(yy)
        if mf.exists():
            m = lgb.Booster(model_file=str(mf))
        else:
            tr, va = has & (sp == 0), has & (sp == 1)
            P = dict(objective="binary" if kind == "bin" else "l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
                     feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1, lambda_l2=1.0, seed=SEED,
                     num_threads=int(os.environ.get("TAPERAW_THREADS", 10)), verbose=-1)
            dtr = lgb.Dataset(df.loc[tr, X], yy[tr], categorical_feature=["coin"], free_raw_data=True)
            dva = lgb.Dataset(df.loc[va, X], yy[va], reference=dtr)
            m = lgb.train(P, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            m.save_model(str(mf), num_iteration=m.best_iteration); del dtr, dva
        ev = has & (sp == 2)
        p = m.predict(df.loc[ev, X])
        out.append(pd.DataFrame({"t": df.t.values[ev], "coin": df.coin.values[ev].astype(np.int8), "target": y,
                                 "y": yy[ev].astype(np.float32), "p": p.astype(np.float32),
                                 "loss": loss(kind, yy[ev], p).astype(np.float32)}))
        print(arm, y, "iters", m.current_iteration(), "test n", int(ev.sum()), "loss", float(out[-1].loss.mean()), flush=True)
    R = pd.concat(out, ignore_index=True); R["target"] = R.target.astype("category")
    R.to_parquet(od / f"pred_{arm}.parquet")


def score(arms=("C", "E", "F", "ALL", "Z")):
    from sklearn.metrics import roc_auc_score
    from research_ob_info_map_gbm_20261115 import boot, holm
    od = OUT / "fit"; B = pd.read_parquet(od / "pred_0.parquet")
    rows, plain = [], []
    for a in ("0",) + tuple(arms):
        A = B if a == "0" else pd.read_parquet(od / f"pred_{a}.parquet")
        for y in TARGETS:
            x = A[A.target == y]
            kind = FAM[y[2:].rstrip("0123456789")]
            met = roc_auc_score(x.y, x.p) if kind == "bin" else 1 - ((x.y - x.p) ** 2).sum() / ((x.y - x.y.mean()) ** 2).sum()
            plain.append(dict(arm=a, target=y, metric="AUC" if kind == "bin" else "R2", value=float(met)))
            if a == "0":
                continue
            m = x.merge(B[B.target == y], on=["t", "coin"], suffixes=("_a", "_b"))
            d = (m.loss_a - m.loss_b).values; day = m.t.values // 86400
            mean, up, p = boot(d, day)
            cs = m.assign(d=d).groupby("coin").d.mean()
            rows.append(dict(arm=a, target=y, n=len(m), days=len(np.unique(day)), base=float(m.loss_b.mean()), d=mean,
                             rel=mean / float(m.loss_b.mean()), up90=up, p=p, coins_better=int((cs < 0).sum()),
                             **{f"d_{COINS[i]}": float(v) for i, v in cs.items()}))
    R = pd.DataFrame(rows)
    R["holm"] = False
    for a in R.arm.unique():
        i = R.arm == a; R.loc[i, "holm"] = holm(R.p[i].values)
    dz = R[R.arm == "Z"].set_index("target").d
    R["beats_z"] = R.d < R.target.map(dz)
    R["strict"] = R.holm & (R.up90 < 0) & (R.beats_z | (R.arm == "Z"))
    R["candidate"] = (R.up90 < 0) & (R.coins_better >= 3)
    R.to_csv(od / "score.csv", index=False)
    pd.DataFrame(plain).pivot(index="target", columns="arm", values="value").to_csv(od / "plain.csv")
    print(R.groupby("arm")[["strict", "candidate"]].sum())
    print(R[R.strict | R.candidate][["arm", "target", "rel", "up90", "coins_better", "strict", "beats_z"]].to_string())
    print(pd.read_csv(od / "plain.csv").to_string())


# ---------- 자체점검 ----------
def selftest():
    rng = np.random.default_rng(0)
    # (1) Hawkes 초 기여 + lfilter = ms 단위 무차별 합
    ts = np.sort(rng.integers(0, 50_000, 400)); px = 100 + rng.normal(0, .1, 400).cumsum()
    bm = rng.random(400) < .4; P = pack(ts, px, np.ones(400), bm)
    for tau in (1, 10):
        Hs = lfilter([1.0], [1.0, -np.exp(-1 / tau)], P[f"h{tau}_b"].values.astype(float))
        for s in (3, 17, 41):
            tt = ts[(~bm) & (ts < (s + 1) * 1000)]
            assert abs(Hs[s] - np.exp(-((s + 1) * 1000 - tt) / (1000 * tau)).sum()) < 1e-4
    # (2) 인과: 초 > g 를 바꿔도 피쳐 불변 · 대상은 (g, g+h] 만 봄
    n3 = 3 * N
    S = {"n_b": rng.poisson(2, n3).astype(np.float32), "n_s": rng.poisson(2, n3).astype(np.float32)}
    S["q_b"] = (S["n_b"] * rng.uniform(1e3, 1e5, n3)).astype(np.float32); S["q_s"] = (S["n_s"] * rng.uniform(1e3, 1e5, n3)).astype(np.float32)
    S["px"] = 100 * np.exp(rng.normal(0, 1e-4, n3).cumsum())
    for t in TAUS:
        for sd in "bs":
            S[f"h{t}_{sd}"] = (S[f"n_{sd}"] * .5).astype(np.float32)
    ok = np.ones(n3, bool); ds = np.arange(3) * N
    g, o = build(S, ok, ds)
    i = np.searchsorted(g, 2 * N - 1); gi = g[i]
    S2 = {k: v.copy() for k, v in S.items()}
    for k in S2:
        S2[k][gi + 1:] = S2[k][gi + 1:] * 3 + (1 if k != "px" else 0)
    _, o2 = build(S2, ok, ds)
    for c in o:
        if c[:2] != "y_":
            a, b = o[c][i], o2[c][i]
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-6 * max(1, abs(a)), c
    S3 = {k: v.copy() for k, v in S.items()}
    for k in S3:
        S3[k][gi + 61:] = S3[k][gi + 61:] * 3 + (1 if k != "px" else 0)
    _, o3 = build(S3, ok, ds)
    for c in ("y_dir10", "y_rv60", "y_cnt60", "y_flow10"):
        assert o[c][i] == o3[c][i] or (np.isnan(o[c][i]) and np.isnan(o3[c][i])), c
    # (3) VPIN: 매수 몫이 늘 φ 이면 |2φ−1|
    vol = np.ones(n3); fl = np.full(n3, 0.3); vp = vpin(vol, fl, g, ds, 50, (50,))
    assert np.nanmax(np.abs(vp[50] - 0.3)) < 1e-9 and np.isfinite(vp[50][-1])
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl"); ap.add_argument("--frame"); ap.add_argument("--fit"); ap.add_argument("--score", action="store_true")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.dl:
        jobs = [(a.dl, str(d.date())) for d in pd.date_range(D0, D1)]
        with ProcessPoolExecutor(a.workers) as ex:
            for d, st in ex.map(sec_day, jobs):
                if not (st == "have" or st.startswith("ok")) or d.endswith("-01"):
                    print(a.dl, d, st, flush=True)
    if a.frame:
        frame_coin(a.frame)
    if a.fit:
        fit(a.fit)
    if a.score:
        score()
