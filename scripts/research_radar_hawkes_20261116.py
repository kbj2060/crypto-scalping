"""배포 레이더(A10 HAR-1m) 위에 체결 원본 Hawkes 강도를 얹으면 15분·1시간 폭 예측이 나아지나 (2026-11-16).

사용자 «레이더 위 재검». 레이더 식·표적·분할·QLIKE 는 scripts/research_realtime_risk_radar_20261115.py 그대로(같은 함수 import).
얹는 재료(분 t 마감까지, scripts/research_tape_raw_cef_20261115.py 의 frame_<COIN>.parquet):
  E = log 강도(매수+매도, τ=1·10·60·600초) + 분지 비율 근사 · N(대조) = log 체결 수·거래량 10초~1시간 · Z(위약) = E 의 하루 전 값.
결과: docs/experiments/radar_hawkes_20261116.md
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_realtime_risk_radar_20261115 as RR
from research_tape_raw_cef_20261115 import OUT as TR

TAUS = (1, 10, 60, 600)
OUTF = TR / "radar_hawkes.json"


def extra(c, mins):
    """frame 행 t(분 끝 시각) → 레이더 분 인덱스 m = t/60 − 1 에 맞춘 E·N·Z 배열."""
    cols = ["t", "e_nhat", "z_e_nhat"] + [f"{p}e_l{s}{t}" for p in ("", "z_") for s in "bs" for t in TAUS] \
        + [f"p_{k}{w}" for k in ("cnt", "vol") for w in (10, 60, 300, 3600)]
    F = pd.read_parquet(TR / f"frame_{c}.parquet", columns=cols)
    idx = pd.Index(mins).get_indexer(F.t.values // 60 - 1); k = idx >= 0
    out = {}
    def put(name, v):
        a = np.full(len(mins), np.nan); a[idx[k]] = v[k]; out[name] = a
    for p, tag in (("", "E"), ("z_", "Z")):
        for t in TAUS:
            put(f"{tag}{t}", np.log(F[f"{p}e_lb{t}"].values + F[f"{p}e_ls{t}"].values + 0.01))
        put(f"{tag}n", F[f"{p}e_nhat"].values)
    for kk in ("cnt", "vol"):
        for w in (10, 60, 300, 3600):
            put(f"N{kk}{w}", np.log(F[f"p_{kk}{w}"].values + (1 if kk == "cnt" else 1e-6)))
    return out


ARMS = {"H": [], "HN": ["N"], "HE": ["E"], "HNE": ["N", "E"], "HZ": ["Z"]}
CMP = [("HE", "H"), ("HNE", "HN"), ("HN", "H"), ("HZ", "H")]


def run():
    rng = np.random.default_rng(RR.SEED); R = {}; pooled = {}
    for c in RR.COINS:
        cl, hi, lo, m0 = RR.load(c)
        F = RR.features(cl, m0); M, day = RR.masks(F["mins"]); tr, te = M["train"], M["test"]
        X = extra(c, F["mins"]); F.update(X)
        R[c] = {}
        for H in (15, 60):
            y = RR.target(F["r2"], H)
            f = {a: RR.fit_predict(F, y, tr, list(RR.WIN) + [k for k in X if k[0] in "".join(g)] if g else list(RR.WIN))[0]
                 for a, g in ARMS.items()}
            ok = te & np.isfinite(y) & np.all([np.isfinite(v) for v in f.values()], 0)
            q = {a: RR.qlike(y[ok] ** 2, v[ok]) for a, v in f.items()}
            r = {"n": int(ok.sum()), "days": int(len(np.unique(day[ok]))), "qlike": {a: float(v.mean()) for a, v in q.items()}}
            for a, b in CMP:
                d = q[a] - q[b]; dm = pd.Series(d).groupby(day[ok]).mean()
                bs = np.array([rng.choice(dm.values, len(dm)).mean() for _ in range(RR.B)])
                r[f"{a}-{b}"] = dict(mean=float(d.mean()), rel=float(d.mean() / q[b].mean()), up90=float(np.quantile(bs, .9)),
                                     p=float((bs >= 0).mean()), days_better=float((dm < 0).mean()))
                pooled.setdefault((H, a, b), []).append(pd.DataFrame({"day": day[ok], "d": d, "base": q[b]}))
            R[c][H] = r
            print(c, H, json.dumps({k: (round(v["rel"] * 100, 2), round(v["up90"], 5)) for k, v in r.items() if "-" in k}), flush=True)
    P = {}
    for (H, a, b), parts in pooled.items():                       # 4코인 합동 · 날짜 블록
        D = pd.concat(parts); g = D.groupby("day").d.agg(["sum", "count"])
        idx = rng.integers(0, len(g), (RR.B, len(g)))
        m = g["sum"].values[idx].sum(1) / g["count"].values[idx].sum(1)
        P[f"{H}:{a}-{b}"] = dict(rel=float(D.d.mean() / D.base.mean()), up90=float(np.quantile(m, .9)), p=float((m >= 0).mean()),
                                coins_better=int(sum(R[c][H][f"{a}-{b}"]["mean"] < 0 for c in RR.COINS)))
    for H in (15, 60):
        ps = [R[c][H]["HE-H"]["p"] for c in RR.COINS]; o = np.argsort(ps)
        holm = {RR.COINS[i]: bool(all(ps[o[j]] <= .10 / (4 - j) for j in range(r + 1))) for r, i in enumerate(o)}
        e, ne, z = P[f"{H}:HE-H"], P[f"{H}:HNE-HN"], P[f"{H}:HZ-H"]
        P[f"{H}:verdict"] = dict(holm_coins=holm, strict=bool(e["up90"] < 0 and ne["up90"] < 0 and e["rel"] < z["rel"] and sum(holm.values()) >= 3),
                                 candidate=bool(e["up90"] < 0 and e["coins_better"] >= 3))
    json.dump(dict(per_coin=R, pooled=P), open(OUTF, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(P, indent=1, ensure_ascii=False))


def selftest():
    mins = np.arange(100, 110)
    F = pd.DataFrame({"t": (mins + 1) * 60})                     # frame 행 t = 분 m 끝 → m = t/60 − 1
    assert (pd.Index(mins).get_indexer(F.t.values // 60 - 1) == np.arange(10)).all()
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "selftest"]()
