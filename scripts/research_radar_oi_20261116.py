"""B2 레이더(HAR-1m + 1분봉 거래대금 5창) 위에 OI 변화 크기를 얹으면 15분·1시간 폭 예측이 나아지나 (2026-11-16).

사용자 «OI 변화 레이더 증분 검정» — 코인글래스 세션에서 |OI 1시간 변화|+거래대금비가 시간봉 HAR 위 1h −1.29% 였다.
레이더 식·표적·분할·QLIKE·거래대금 창은 angry-wilson 워크트리의 scripts/research_realtime_risk_radar_20261115.py 그대로(import).
OI = Binance Vision metrics sum_open_interest(계약 수, 5분 스냅숏) · 스냅숏 create_time T 는 T+5분에 안다고 본다(보수적).
결정 분 m(마감 (m+1)·60초)에서 알려진 마지막 스냅숏 j: O5·O15·O60·O240 = log(|log OI_j − log OI_{j−k}| + 1e-4), k = 1·3·12·48.
스냅숏이 15분 넘게 낡았거나 j−k 가 빠졌으면 NaN.
팔: HV(B2 기준) · HVO(+O) · HVOZ(+O 하루 전 = 위약) · H · HO(참고).
판정(계산 전 고정): 엄격 = 4코인 합동 HVO−HV 날짜 블록 단측 90% 상한 < 0 · 코인 Holm(α .10) ≥ 3/4 · HVO 개선 > HVOZ 개선.
  실용 = 엄격 + 합동 상대 개선 ≤ −0.5%(Hawkes 고유 몫 −0.6% 를 «작음»으로 본 것과 같은 잣대). 15분·1시간 각각.
  python scripts/research_radar_oi_20261116.py selftest | run
"""
import glob, io, json, os, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/kbj20/crypto-scalping/.claude/worktrees/angry-wilson-bbf351/scripts")
import research_realtime_risk_radar_20261115 as RR

MET = Path("/home/kbj20/crypto-scalping/data/vision_cg_20261116/met")
OUT = Path(__file__).resolve().parents[1] / "tmp/radar_oi_20261116"
LAGS = {"O5": 1, "O15": 3, "O60": 12, "O240": 48}
KNOWN_S, STALE_S = int(os.getenv("RADAR_OI_KNOWN_S", 300)), 900   # 민감도: 0 = 스냅숏 즉시 앎(낙관)


def load_oi(c):
    p = OUT / f"oi_{c}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    rows = []
    for f in sorted(glob.glob(str(MET / f"{c}USDT-metrics-*.zip"))):
        with zipfile.ZipFile(f) as z:
            rows.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), usecols=["create_time", "sum_open_interest"]))
    d = pd.concat(rows)
    d = pd.DataFrame({"T": pd.to_datetime(d.create_time).astype("int64") // 10**9, "oi": d.sum_open_interest.astype(float)})
    d = d[d.oi > 0].drop_duplicates("T").sort_values("T").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True); d.to_parquet(p)
    return d


def oi_feats(mins, T, oi):
    """결정 분 m 마다 (m+1)·60 − KNOWN_S 이하의 마지막 스냅숏 기준 |Δ log OI| 창들. T·oi 는 정렬된 스냅숏."""
    T = np.asarray(T, dtype=np.int64); lo = np.log(np.asarray(oi, dtype=float))
    j = np.searchsorted(T, (mins + 1) * 60 - KNOWN_S, "right") - 1
    fresh = (j >= 0) & (((mins + 1) * 60 - KNOWN_S) - T[np.maximum(j, 0)] <= STALE_S)
    out = {}
    for nm, k in LAGS.items():
        jk = j - k; ok = fresh & (jk >= 0)
        ok &= np.where(ok, T[np.maximum(j, 0)] - T[np.maximum(jk, 0)] == 300 * k, False)   # 사이 스냅숏 빠짐 없음
        d = np.abs(lo[np.maximum(j, 0)] - lo[np.maximum(jk, 0)])
        out[nm] = np.where(ok, np.log(d + 1e-4), np.nan)
    return out


ARMS = {"H": lambda W, V, O, Z: W, "HO": lambda W, V, O, Z: W + O, "HV": lambda W, V, O, Z: W + V,
        "HVO": lambda W, V, O, Z: W + V + O, "HVOZ": lambda W, V, O, Z: W + V + Z}
CMP = [("HVO", "HV"), ("HVOZ", "HV"), ("HO", "H"), ("HV", "H")]


def run():
    rng = np.random.default_rng(RR.SEED); R = {}; pooled = {}
    for c in RR.COINS:
        cl, hi, lo, m0 = RR.load(c)
        F = RR.features(cl, m0); M, day = RR.masks(F["mins"]); tr, te = M["train"], M["test"]
        RR.vol_features(F, RR.load_qv(c))
        d = load_oi(c); X = oi_feats(F["mins"], d["T"].values, d["oi"].values); F.update(X)
        for k in LAGS:
            F["z" + k] = np.r_[np.full(1440, np.nan), F[k][:-1440]]
        W, V, O, Z = list(RR.WIN), list(RR.VWIN), list(LAGS), ["z" + k for k in LAGS]
        common = np.all([np.isfinite(F[k]) for k in O + Z], 0)          # 모든 팔이 같은 행으로 학습·채점
        R[c] = {"oi_cover_test": float(common[te].mean())}
        for H in (15, 60):
            y = RR.target(F["r2"], H)
            f = {a: RR.fit_predict(F, y, tr & common, g(W, V, O, Z))[0] for a, g in ARMS.items()}
            ok = te & common & np.isfinite(y) & np.all([np.isfinite(v) for v in f.values()], 0)
            q = {a: RR.qlike(y[ok] ** 2, v[ok]) for a, v in f.items()}
            r = {"n": int(ok.sum()), "days": int(len(np.unique(day[ok]))), "qlike": {a: float(v.mean()) for a, v in q.items()}}
            for a, b in CMP:
                dd = q[a] - q[b]; dm = pd.Series(dd).groupby(day[ok]).mean()
                bs = np.array([rng.choice(dm.values, len(dm)).mean() for _ in range(RR.B)])
                r[f"{a}-{b}"] = dict(rel=float(dd.mean() / q[b].mean()), up90=float(np.quantile(bs, .9)),
                                     p=float((bs >= 0).mean()), days_better=float((dm < 0).mean()))
                pooled.setdefault((H, a, b), []).append(pd.DataFrame({"day": day[ok], "d": dd, "base": q[b]}))
            R[c][H] = r
            print(c, H, json.dumps({k: (round(v["rel"] * 100, 2), round(v["up90"], 5)) for k, v in r.items() if "-" in k}), flush=True)
    P = {}
    for (H, a, b), parts in pooled.items():
        D = pd.concat(parts); g = D.groupby("day").d.agg(["sum", "count"]); idx = rng.integers(0, len(g), (RR.B, len(g)))
        m = g["sum"].values[idx].sum(1) / g["count"].values[idx].sum(1)
        P[f"{H}:{a}-{b}"] = dict(rel=float(D.d.mean() / D.base.mean()), up90=float(np.quantile(m, .9)),
                                coins_better=int(sum(R[c][H][f"{a}-{b}"]["rel"] < 0 for c in RR.COINS)))
    for H in (15, 60):
        ps = [R[c][H]["HVO-HV"]["p"] for c in RR.COINS]; o = np.argsort(ps)
        holm = {RR.COINS[i]: bool(all(ps[o[j]] <= .10 / (4 - j) for j in range(r + 1))) for r, i in enumerate(o)}
        e, z = P[f"{H}:HVO-HV"], P[f"{H}:HVOZ-HV"]
        strict = bool(e["up90"] < 0 and sum(holm.values()) >= 3 and e["rel"] < z["rel"])
        P[f"{H}:verdict"] = dict(holm_coins=holm, strict=strict, practical=bool(strict and e["rel"] <= -0.005),
                                 candidate=bool(e["up90"] < 0 and e["coins_better"] >= 3))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(per_coin=R, pooled=P), open(OUT / f"result_known{KNOWN_S}.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(P, indent=1, ensure_ascii=False))


def selftest():
    T = np.arange(0, 300 * 100, 300); oi = np.exp(np.cumsum(np.r_[0, np.random.default_rng(0).normal(0, .01, 99)]))
    mins = np.arange(60, 400)
    a = oi_feats(mins, T, oi)
    oi2 = oi.copy(); oi2[60:] *= 3.0                                  # 스냅숏 60(T=18,000초) 부터 바꿈
    b = oi_feats(mins, T, oi2)
    known = (mins + 1) * 60 - KNOWN_S < 18000                          # 그 전에 결정된 분은 값이 같아야 한다
    for k in LAGS:
        assert np.allclose(np.nan_to_num(a[k][known]), np.nan_to_num(b[k][known])), k
        assert not np.allclose(np.nan_to_num(a[k][~known]), np.nan_to_num(b[k][~known])), k
    m = np.array([100]); j = (101 * 60 - 300) // 300                  # 결정 6,060초 → 알려진 스냅숏 T=5,700 (j=19)
    assert np.isclose(oi_feats(m, T, oi)["O5"][0], np.log(abs(np.log(oi[j]) - np.log(oi[j - 1])) + 1e-4))
    T3 = np.delete(T, 15); oi3 = np.delete(oi, 15)                     # 스냅숏 하나 빠짐 → 그 구간 창은 NaN
    g = oi_feats(np.array([100]), T3, oi3)
    assert np.isnan(g["O60"][0]) and np.isfinite(g["O15"][0])        # T=4,500 이 O60 창 안·O15 창 밖
    assert np.isnan(oi_feats(np.array([2000]), T, oi)["O5"][0])        # 15분 넘게 낡음
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "selftest"]()
