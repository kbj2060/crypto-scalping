"""B2 레이더(HAR-1m + 1분봉 거래대금 5창, vclip 하한판) 위에 IV(Deribit DVOL)를 얹으면 15분·1시간 폭 예측이 나아지나 (2026-11-16).

사용자 «IV 증분 검정도 진행해줘» — 코인글래스 세션 C15: IV 분위(DVOL 30일)가 시간봉 HAR 위 1h −0.87%·4h −1.25%(사후 선택, ETH·BTC).
레이더 식·표적·분할·QLIKE·거래대금 창은 scripts/research_realtime_risk_radar_20261115.py 그대로(같은 워크트리 · vclip = dashboard/risk_radar_params_20261116.json).
IV = data/vision_cg_20261116/dvol/<COIN>_dvol_hourly.csv(Deribit DVOL 시간봉, timestamp = 봉 시작) · 종가는 봉 끝(시작+1시간)에 안다.
결정 분 m(마감 (m+1)·60초)에서 끝난 마지막 시간봉 j: I_l = log DVOL 종가 · I_p = 직전 720시간(자기 포함) 중 분위(코인글래스 pct30 과 같은 식) · 2시간 넘게 낡으면 NaN.
팔: HV(B2 기준) · HVI(+I) · HVIZ(+I 하루 전 = 위약) · H · HI(참고). 코인 = ETH·BTC(DVOL 있는 둘).
판정(계산 전 고정): 엄격 = 두 코인 합동 HVI−HV 날짜 블록 단측 90% 상한 < 0 · 코인 Holm(α .10) 2/2 · HVI 개선 > HVIZ 개선.
  실용 = 엄격 + 합동 상대 개선 ≤ −0.5%. 15분·1시간 각각.
  python scripts/research_radar_iv_20261116.py selftest | run
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_realtime_risk_radar_20261115 as RR

DV = Path("/home/kbj20/crypto-scalping/data/vision_cg_20261116/dvol")
OUT = Path(__file__).resolve().parents[1] / "tmp/radar_iv_20261116"
COINS = ["ETH", "BTC"]
COLS = ["I_l", "I_p"]
STALE_S = 7200
VCLIP = json.loads((Path(__file__).resolve().parents[1] / "dashboard/risk_radar_params_20261116.json").read_text())["coins"]


def load_iv(c):
    d = pd.read_csv(DV / f"{c}_dvol_hourly.csv", parse_dates=["timestamp"])
    d = d[d.close > 0].drop_duplicates("timestamp").sort_values("timestamp")
    T = d.timestamp.astype("int64").to_numpy() // 10**9 + 3600          # 종가를 아는 시각 = 봉 끝
    x = d.close.to_numpy(float)
    pct = pd.Series(x).rolling(720, min_periods=240).rank(pct=True).to_numpy() * 100
    return T, np.log(x), pct


def iv_feats(mins, T, lx, pct):
    """결정 분 m 마다 (m+1)·60 이하에 끝난 마지막 시간봉의 log DVOL·30일 분위. 2시간 넘게 낡으면 NaN."""
    now = (mins + 1) * 60
    j = np.searchsorted(T, now, "right") - 1
    ok = (j >= 0) & (now - T[np.maximum(j, 0)] <= STALE_S)
    return {"I_l": np.where(ok, lx[np.maximum(j, 0)], np.nan), "I_p": np.where(ok, pct[np.maximum(j, 0)], np.nan)}


ARMS = {"H": lambda W, V, O, Z: W, "HI": lambda W, V, O, Z: W + O, "HV": lambda W, V, O, Z: W + V,
        "HVI": lambda W, V, O, Z: W + V + O, "HVIZ": lambda W, V, O, Z: W + V + Z}
CMP = [("HVI", "HV"), ("HVIZ", "HV"), ("HI", "H"), ("HV", "H")]


def run():
    rng = np.random.default_rng(RR.SEED); R = {}; pooled = {}
    for c in COINS:
        cl, hi, lo, m0 = RR.load(c)
        F = RR.features(cl, m0); M, day = RR.masks(F["mins"]); tr, te = M["train"], M["test"]
        RR.vol_features(F, RR.load_qv(c), VCLIP[c]["vclip"])
        X = iv_feats(F["mins"], *load_iv(c)); F.update(X)
        for k in COLS:
            F["z" + k] = np.r_[np.full(1440, np.nan), F[k][:-1440]]
        W, V, O, Z = list(RR.WIN), list(RR.VWIN), list(COLS), ["z" + k for k in COLS]
        common = np.all([np.isfinite(F[k]) for k in O + Z], 0)          # 모든 팔이 같은 행으로 학습·채점
        R[c] = {"iv_cover_test": float(common[te].mean())}
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
                                coins_better=int(sum(R[c][H][f"{a}-{b}"]["rel"] < 0 for c in COINS)))
    for H in (15, 60):
        ps = [R[c][H]["HVI-HV"]["p"] for c in COINS]; o = np.argsort(ps)
        holm = {COINS[i]: bool(all(ps[o[j]] <= .10 / (2 - j) for j in range(r + 1))) for r, i in enumerate(o)}
        e, z = P[f"{H}:HVI-HV"], P[f"{H}:HVIZ-HV"]
        strict = bool(e["up90"] < 0 and sum(holm.values()) >= 2 and e["rel"] < z["rel"])
        P[f"{H}:verdict"] = dict(holm_coins=holm, strict=strict, practical=bool(strict and e["rel"] <= -0.005),
                                 candidate=bool(e["up90"] < 0 and e["coins_better"] >= 2))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(per_coin=R, pooled=P), open(OUT / "result.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(P, indent=1, ensure_ascii=False))


def selftest():
    T = np.arange(3600, 3600 * 200, 3600); lx = np.log(np.linspace(50, 80, len(T))); pct = np.arange(len(T), dtype=float)
    mins = np.arange(60, 3000)
    a = iv_feats(mins, T, lx, pct)
    lx2 = lx.copy(); lx2[20:] += 1.0                                    # 시간봉 20(끝 = 21·3600초) 부터 바꿈
    b = iv_feats(mins, T, lx2, pct)
    known = (mins + 1) * 60 < T[20]                                      # 그 전에 결정된 분은 값이 같아야 한다
    assert np.allclose(np.nan_to_num(a["I_l"][known]), np.nan_to_num(b["I_l"][known])) and not np.allclose(a["I_l"][~known], b["I_l"][~known])
    m = np.array([3600 * 5 // 60 - 1]); j = 4                            # 결정 18,000초 = 봉 4(끝 18,000초) 마감 순간 → 봉 4 사용
    assert np.isclose(iv_feats(m, T, lx, pct)["I_l"][0], lx[j])
    assert np.isnan(iv_feats(np.array([(T[-1] + 7300) // 60]), T, lx, pct)["I_l"][0])   # 2시간 넘게 낡음
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "selftest"]()
