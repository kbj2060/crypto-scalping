"""5분마다 «현재가 위·아래로 얼마나 가까이 얼마나 많은 청산액» 피쳐 → 방향 규칙·모델 (2026-10-10).
사용자 «매 5분마다 현재가에서 위/아래로 얼마나 가까이 얼마나 많은 금액이 청산액이 걸려있는지를 피쳐로 만들어서 방향에 대한 규칙이나 모델을 만들어줘».

지도 = 대시보드 1일 청산맵 그대로(dashboard/server.py load_liquidation_map → compute_tier_profile(지난 24 시간봉, 마지막은 형성 중)).
매 5분 마감 T: 시간봉 23개(닫힘) + 형성 중 시간봉(그 시간 시작~T 의 1분봉) · 기준가 = T 의 1분 종가. 값 = 칸 몫(합 1 = 2×OI 의 비율).
라벨: T 다음 1분 종가 진입 → 30·60·240분 뒤 로그수익(피쳐 끝 T < 라벨 시작 T+1분, 사건 라벨 경계 계약).

사전등록(결과 전 고정)
- 구간: TRAIN 2022-01~2024-06 · VAL 2024-07~12 · TEST(주 판정) 2025-01~ (최근 우선).
- 피쳐 LIQ: 거리 D∈{0.3,0.5,1,2,3,5}% 안 위/아래 몫과 로그비 · 누적 1%/3% 몫이 쌓이는 거리 위/아래 · 5% 안 최대 칸 거리·몫 · 거리역가중 합 로그비.
  피쳐 BASE(가격만): 수익률 5m·30m·1h·4h·24h · 실현변동 1h·24h · 24h 범위 안 위치.
- 모델: HGB 분류(부호) BASE / LIQ / BOTH, TRAIN 학습. 판정 = TEST 에서 BOTH−BASE AUC 일 블록 부트스트랩 CI 하한 > 0.
- 규칙: lr_1(1% 안 위/아래 로그비)·lr_3·lr_mag 의 TRAIN 20/80 분위 밖 → 많은 쪽으로(자석) 진입, 4+4bp 비용 후 TEST 평균 CI 하한 > 0 이면 통과
  (반대 부호면 «반대(페이드)» 로 기록만 — 사후라 판정 아님).
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · future_rows_used_for_entry=false.
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_liquidation_map_20260824 as LM  # noqa: E402

K1 = "/home/kbj20/crypto-scalping/data/binance_vision/klines1m"
OUT = "tmp/liqmap_5m_direction_20261010"
DIST = (0.3, 0.5, 1, 2, 3, 5)
HOR = (30, 60, 240)
COST = 8.0                                                                   # 시장가 4+4bp
SPL = {"TRAIN": ("2022-01-01", "2024-07-01"), "VAL": ("2024-07-01", "2025-01-01"), "TEST": ("2025-01-01", "2027-01-01")}
HM, M5 = 3_600_000, 300_000


def load_1m():
    k = pd.concat([pd.read_parquet(f"{K1}/{f}", columns=["t", "h", "l", "c", "v"]) for f in sorted(os.listdir(K1))])
    k = k.drop_duplicates("t").sort_values("t")
    full = pd.DataFrame({"t": np.arange(k.t.iloc[0], k.t.iloc[-1] + 1, 60_000)}).merge(k, on="t", how="left")
    full["c"] = full.c.ffill(); full["h"] = full.h.fillna(full.c); full["l"] = full.l.fillna(full.c); full["v"] = full.v.fillna(0.0)
    return full


def features(tp, cp):
    v = np.sum([t["values"] for t in tp["tiers"]], axis=0)
    p = (tp["lo"] + np.arange(len(v))) * tp["bin_width"]
    d = (p - cp) / cp * 100
    r = {}
    for side, m, dd in (("up", d > 0, d), ("dn", d < 0, -d)):
        ds, vs = dd[m], v[m]
        o = np.argsort(ds); ds, vs = ds[o], vs[o]
        for D in DIST:
            r[f"{side}_{D}"] = float(vs[ds <= D + 1e-9].sum())
        cs = np.cumsum(vs)
        for q in (0.01, 0.03):
            j = np.searchsorted(cs, q)
            r[f"{side}_dcum{int(q * 100)}"] = float(ds[j]) if j < len(ds) else 10.0
        w5 = ds <= 5
        j = int(np.argmax(np.where(w5, vs, -1))) if w5.any() else -1
        r[f"{side}_topd"], r[f"{side}_topw"] = (float(ds[j]), float(vs[j])) if j >= 0 else (5.0, 0.0)
        r[f"{side}_mag"] = float((vs[w5] / np.maximum(ds[w5], 0.1)).sum())
    for D in DIST:
        r[f"lr_{D}"] = float(np.log((r[f"up_{D}"] + 1e-4) / (r[f"dn_{D}"] + 1e-4)))
    r["lr_mag"] = float(np.log((r["up_mag"] + 1e-4) / (r["dn_mag"] + 1e-4)))
    return r


def _chunk(args):
    Ts, H, P = args
    rows = []
    for T in Ts:
        h = (T - 1) // HM                                                     # 형성 중 시간 번호
        i = np.searchsorted(H["hour"], h)
        if i < 23 or i >= len(H["hour"]) or H["hour"][i] != h or H["hour"][i - 23] != h - 23:   # 구멍 있는 창 건너뜀
            continue
        j = (T - 60_000 - P["t0"]) // 60_000                                   # T 에 끝나는 1분봉
        df = pd.DataFrame({"timestamp": pd.to_datetime(np.r_[H["hour"][i - 23:i], h] * HM, unit="ms", utc=True),
                           "high": np.r_[H["high"][i - 23:i], P["ph"][j]], "low": np.r_[H["low"][i - 23:i], P["pl"][j]],
                           "close": np.r_[H["close"][i - 23:i], P["c"][j]], "volume": np.r_[H["vol"][i - 23:i], P["pv"][j]]})
        cp = float(P["c"][j])
        tp = LM.compute_tier_profile(df, cp)
        if tp:
            rows.append({"T": int(T), "cp": cp, **features(tp, cp)})
    return rows


def build():
    os.makedirs(OUT, exist_ok=True)
    k = load_1m()
    t = k.t.to_numpy(); hb = t // HM
    g = k.groupby(hb)
    H = {"hour": g.t.first().index.to_numpy(), "high": g.h.max().to_numpy(), "low": g.l.min().to_numpy(),
         "close": g.c.last().to_numpy(), "vol": g.v.sum().to_numpy()}
    # 형성 중 봉: 그 시간 안 1분 j 까지의 고가 최대·저가 최소·거래량 합
    P = {"ph": g.h.cummax().to_numpy(), "pl": g.l.cummin().to_numpy(), "pv": g.v.cumsum().to_numpy(), "c": k.c.to_numpy(), "t0": int(t[0])}
    Ts = np.arange((t[0] // M5 + 1) * M5 + 24 * HM, t[-1] + 60_000 + 1, M5)
    with Pool(8) as pool:
        rows = [r for part in pool.map(_chunk, [(p, H, P) for p in np.array_split(Ts, 64)]) for r in part]
    F = pd.DataFrame(rows)
    F.to_parquet(f"{OUT}/feat5m.parquet")
    print(len(F), pd.to_datetime(F["T"].min(), unit="ms"), pd.to_datetime(F["T"].max(), unit="ms"))


def panel():
    F = pd.read_parquet(f"{OUT}/feat5m.parquet").sort_values("T").reset_index(drop=True)
    k = load_1m()
    c, t = k.c.to_numpy(), k.t.to_numpy()
    j = (F["T"].to_numpy() - 60_000 - t[0]) // 60_000                          # T 에 끝나는 1분봉
    assert np.allclose(c[j], F.cp)                                              # 정렬 확인
    lc = np.log(c)
    for m, nm in ((5, "r5m"), (30, "r30m"), (60, "r1h"), (240, "r4h"), (1440, "r24h")):
        F[nm] = lc[j] - lc[np.maximum(j - m, 0)]
    cs2 = np.cumsum(np.r_[0, np.diff(lc)] ** 2)
    F["rv1h"] = np.sqrt(cs2[j] - cs2[j - 60]); F["rv24h"] = np.sqrt(cs2[j] - cs2[j - 1440])
    hi = pd.Series(k.h.to_numpy()).rolling(1440).max().to_numpy()[j]; lo = pd.Series(k.l.to_numpy()).rolling(1440).min().to_numpy()[j]
    F["pos24"] = (c[j] - lo) / np.maximum(hi - lo, 1e-9)
    e = j + 1                                                                   # 다음 1분 종가 진입
    for h in HOR:
        ok = e + h < len(c)
        F[f"y{h}"] = np.where(ok, (lc[np.minimum(e + h, len(c) - 1)] - lc[np.minimum(e, len(c) - 1)]) * 1e4, np.nan)
    ts = pd.to_datetime(F["T"], unit="ms")
    F["split"] = np.select([(ts >= a) & (ts < b) for a, b in SPL.values()], list(SPL), "-")
    F["day"] = F["T"] // 86_400_000
    return F


BASE = ["r5m", "r30m", "r1h", "r4h", "r24h", "rv1h", "rv24h", "pos24"]


def liq_cols(F):
    return [c for c in F.columns if c.startswith(("up_", "dn_", "lr_"))]


def boot(day, fn, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ud = np.unique(day); g = {u: np.flatnonzero(day == u) for u in ud}
    return np.quantile([fn(np.concatenate([g[u] for u in rng.choice(ud, len(ud))])) for _ in range(n)], [.025, .975])


def test():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    F = panel()
    LIQ = liq_cols(F)
    out = {"n": {s: int((F.split == s).sum()) for s in SPL}, "flags": {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
           "future_rows_used_for_entry": False}, "models": {}, "rules": {}, "ic": {}}
    tr, va, te = ((F.split == s).to_numpy() for s in SPL)
    day = F.day.to_numpy()
    for h in HOR:                                                               # 단변량 IC(TEST 스피어만)
        out["ic"][h] = {c: round(float(F.loc[te, c].corr(F.loc[te, f"y{h}"], method="spearman")), 4) for c in LIQ + BASE}
    for h in HOR:
        y = F[f"y{h}"].to_numpy(); ok = np.isfinite(y); yb = (y > 0).astype(int)
        P = {}
        for nm, cols in (("BASE", BASE), ("LIQ", LIQ), ("BOTH", BASE + LIQ)):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                                               l2_regularization=1.0, random_state=0)
            m.fit(F.loc[tr & ok, cols], yb[tr & ok])
            P[nm] = m.predict_proba(F[cols])[:, 1]
        w = te & ok; d = day[w]; yt = yb[w]
        res = {}
        for nm, p in P.items():
            pt = p[w]
            q20, q80 = np.quantile(p[tr & ok], [.2, .8])                         # 매매: TRAIN 예측 20/80 분위 밖 → 방향
            s = np.where(pt >= q80, 1, np.where(pt <= q20, -1, 0)); sel = s != 0
            pnl = (s * y[w] - COST)[sel]
            res[nm] = {"auc_val": round(float(roc_auc_score(yb[va & ok], p[va & ok])), 4), "auc_test": round(float(roc_auc_score(yt, pt)), 4),
                       "trade_n": int(sel.sum()), "trade_bp": round(float(pnl.mean()), 2),
                       "trade_ci": [round(float(x), 2) for x in boot(d[sel], lambda ix: pnl[ix].mean())]}
        pb, pB = P["BASE"][w], P["BOTH"][w]
        diff = boot(d, lambda ix: roc_auc_score(yt[ix], pB[ix]) - roc_auc_score(yt[ix], pb[ix]), n=300)
        res["BOTH_minus_BASE_auc"] = [round(res["BOTH"]["auc_test"] - res["BASE"]["auc_test"], 4), [round(float(x), 4) for x in diff]]
        res["pass"] = bool(diff[0] > 0)
        out["models"][h] = res
        print(h, json.dumps(res, ensure_ascii=False), flush=True)
    for c in ("lr_1", "lr_3", "lr_mag"):                                         # 규칙: 많은 쪽으로(자석)
        q20, q80 = np.quantile(F.loc[tr, c], [.2, .8])
        s = np.where(F[c] >= q80, 1, np.where(F[c] <= q20, -1, 0))
        for h in HOR:
            y = F[f"y{h}"].to_numpy(); r = {}
            for sp, msk in zip(SPL, (tr, va, te)):
                w = msk & (s != 0) & np.isfinite(y)
                g_ = s[w] * y[w]; pnl = g_ - COST
                r[sp] = {"n": int(w.sum()), "gross_bp": round(float(g_.mean()), 2), "net_bp": round(float(pnl.mean()), 2),
                         "gross_ci": [round(float(x), 2) for x in boot(day[w], lambda ix: g_[ix].mean())]}
            r["pass"] = bool(r["TEST"]["gross_ci"][0] > COST)
            out["rules"][f"{c}_{h}"] = r
            print(c, h, json.dumps(r), flush=True)
    json.dump(out, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1)


def selftest():
    # 위 1% 칸 0.2, 아래 1% 칸 0.1 → up_1·dn_1·lr_1 · 누적 1% 거리 · 최대 칸 거리
    tp = {"lo": 990, "bin_width": 1.0, "tiers": [{"values": [0.1] + [0] * 19 + [0.2]}]}       # 칸 990(아래 1%)·1010(위 1%)
    r = features(tp, 1000.0)
    assert abs(r["up_1"] - 0.2) < 1e-12 and abs(r["dn_1"] - 0.1) < 1e-12 and r["up_0.5"] == 0
    assert abs(r["lr_1"] - np.log(0.2001 / 0.1001)) < 1e-12 and r["up_dcum1"] == 1.0 and r["dn_topd"] == 1.0
    print("selftest OK")


if __name__ == "__main__":
    {"--build": build, "--test": test, "--selftest": selftest}[sys.argv[1]]()
