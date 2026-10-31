#!/usr/bin/env python3
"""XRP 박스 유지 — Q1 1h 굴림 Platt90 · Q2 4h QX4 σ 등조 지도. 사전등록 docs/prereg/xrp_box_hold_platt90_qx4_20261031.md.

    python scripts/research_xrp_box_hold_platt90_qx4_20261031.py --selftest
    python scripts/research_xrp_box_hold_platt90_qx4_20261031.py --run      # 한 번만(report.json 있으면 거절)
입력 tmp/xrp_box_20261031/in/(xrp_rows.parquet = 10-20 XRP 박스 연구 features.parquet 사본 · pred/report_r4·r5W1·r5W2_XRP = 변동성 경주 동결 예측).
라벨·피쳐·보정 함수·p_stay 는 기존 연구 함수 그대로 import. 실주문 없음.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_box_hold_calibration_drift_20261025 as C   # noqa: E402 -- labels·rolling·metrics·day_boot·auc_boot·logit·sig
import research_box_hold_vs_qx4i_20261029 as R                  # noqa: E402 -- p_stay·selftest
import research_vol4h_forecast_horserace_20261025 as V          # noqa: E402 -- qlike
from dashboard.box_hold import BUNDLES                           # noqa: E402

OUT = ROOT / "tmp/xrp_box_20261031"
IN = OUT / "in"
R.IN = IN
XB = BUNDLES["xrp"]
BV = Path("/home/kbj20/crypto-scalping/data/binance_vision")
END = pd.Timestamp("2026-10-20")                 # 1분봉 끝 10-19 23:59 → 라벨 창 끝(ts + W + 1분) ≤ 이 시각
MIN = pd.Timedelta("1min")
J = {"J1": ("2025-02-01", "2026-01-01"), "J2": ("2026-01-01", END)}
AUX1 = {"val_2026_01_05": ("2026-01-01", "2026-06-01"), "test_2026_06": ("2026-06-01", END)}
WIN4 = {"val": ("2026-01-01", "2026-06-01"), "test": ("2026-06-01", END)}
QL_TEST = 0.2165861748790505                     # report_r4_XRP QX4 test_qlike (재현 관문 P1)
ETH_ISO = json.loads((ROOT / "dashboard/box_hold_qx4i_iso_20261029.json").read_text())


def rows(scale):
    cf = XB["scales"][scale]
    A = pd.read_parquet(IN / "xrp_rows.parquet", filters=[("scale", "==", scale)]).replace([np.inf, -np.inf], np.nan)
    A = A.dropna(subset=cf["cols"]).sort_values("ts").reset_index(drop=True)
    W = C.WIN[scale]
    A = A[(A.ts + (W + 1) * MIN <= END) & (np.exp(A.log_rng) > 0)].reset_index(drop=True)
    assert set(A.hold.unique()) <= {0, 1}
    A["p"] = cf["model"].predict_proba(A[cf["cols"]])[:, 1]
    return A, cf, W


def gate_L1(A1, A4):
    """1분봉에서 라벨 독립 재계산 → 행 100% 일치."""
    fs = sorted(glob.glob(f"{BV}/klines1m/XRPUSDT-1m-*.parquet")) + sorted(glob.glob(f"{BV}/klines1m_daily/XRPUSDT-1m-*.parquet"))
    K = pd.concat([pd.read_parquet(f)[["t", "h", "l", "c"]] for f in fs]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    assert (np.diff(K.t) == 60000).all(), "1분봉 빈칸"
    pos = pd.Series(np.arange(len(K)), index=K.t)
    out = {}
    for nm, A, W in (("1h", A1, 60), ("4h", A4, 240)):
        y = C.labels(*(K[k].to_numpy(float) for k in ("h", "l", "c")), W)
        i = pos.reindex(A.ts.astype("int64").to_numpy() // 10 ** 6).to_numpy()
        assert np.isfinite(i).all()
        same = float((y[i.astype(int)] == A.hold.to_numpy()).mean())
        out[nm] = {"rows": len(A), "match": same}
        assert same == 1.0, (nm, same)
    return out


def dmean(a, b, y, days):
    d = (a - y) ** 2 - (b - y) ** 2
    lo, hi = C.day_boot(d, days, alpha=0.2)
    return {"mean": float(d.mean()), "one_sided90": [lo, hi], "ci95": C.day_boot(d, days)}


def q1(A, cf, W):
    A = A[A.ts >= "2025-01-01"].reset_index(drop=True)
    ts, y, pA = A.ts.to_numpy(), A.hold.to_numpy(float), A.p.to_numpy()
    pB = C.rolling("platt", ts, pA, y, W, 90, t0="2025-01-01")
    res, acc = {}, []
    for k, (a, b) in {**J, **AUX1}.items():
        m = (A.ts >= a) & (A.ts + (W + 1) * MIN <= pd.Timestamp(b))
        m = m.to_numpy(); d = pd.DatetimeIndex(ts[m]).normalize().to_numpy()
        mA, mB = C.metrics(y[m], pA[m], cf["q20"], cf["q80"]), C.metrics(y[m], pB[m], cf["q20"], cf["q80"])
        flip = float(((pA[m] < cf["q20"]) != (pB[m] < cf["q20"])).mean() + ((pA[m] > cf["q80"]) != (pB[m] > cf["q80"])).mean())
        res[k] = {"n": int(m.sum()), "days": int(len(np.unique(d))), "hold_rate": float(y[m].mean()), "A": mA, "B": mB,
                  "dbrier": dmean(pB[m], pA[m], y[m], d), "mean_abs_change_pp": float(np.abs(pB[m] - pA[m]).mean() * 100), "label_flip": flip}
        if k in J:
            acc.append(m)
    m = acc[0] | acc[1]; d = pd.DatetimeIndex(ts[m]).normalize().to_numpy()
    mA, mB = C.metrics(y[m], pA[m], cf["q20"], cf["q80"]), C.metrics(y[m], pB[m], cf["q20"], cf["q80"])
    db = dmean(pB[m], pA[m], y[m], d)
    v = {"V1": db["one_sided90"][1] < 0, "V2": mB["logloss"] <= mA["logloss"], "V3": mB["ece_pp"] <= mA["ece_pp"],
         "V4": mB["auc"] >= mA["auc"] - 0.002, "V5": res["J1"]["dbrier"]["mean"] < 0 and res["J2"]["dbrier"]["mean"] < 0}
    lenient = db["mean"] < 0 and all(v[k] for k in ("V2", "V3", "V4", "V5"))
    monthly = pd.DataFrame({"m": pd.DatetimeIndex(ts).to_period("M").astype(str), "A": pA - y, "B": pB - y})[m].groupby("m").mean().mul(100).round(2)
    return {"pooled": {"n": int(m.sum()), "days": int(len(np.unique(d))), "A": mA, "B": mB, "dbrier": db}, "windows": res, "verdict": v,
            "pass": all(v.values()), "candidate": lenient, "monthly_bias_pp": monthly.to_dict("index")}


def qx(name, col, c):
    P = pd.read_parquet(IN / name)
    assert (P.index.minute == 55).all(), name
    return pd.Series(P[col].to_numpy() + c, index=P.index + pd.Timedelta("5min")), P


def q2(A, cf):
    A = A.assign(tau=A.ts + MIN).set_index("tau")
    A["a"] = np.exp(A.log_rng) / 2e4
    tr = []
    for w in ("5W1", "5W2"):
        c = json.loads((IN / f"report_r{w}_XRP.json").read_text())["models"]["QX4"]["c"]
        tr += [qx(f"pred_val_r{w}_XRP.parquet", "v_QX4", c)[0], qx(f"pred_test_r{w}_XRP.parquet", "t_QX4", c)[0]]
    L = pd.concat(tr); L = L[~L.index.duplicated()]
    T = A.join(L.rename("lrv"), how="inner")
    T = T[(T.index >= "2024-01-01") & (T.index < "2026-01-01")]
    assert (T.index + pd.Timedelta("4h")).max() <= pd.Timestamp("2026-01-01")
    T["p_raw"] = R.p_stay(np.sqrt(np.exp(T.lrv)) / T.a)
    iso = IsotonicRegression(y_min=.001, y_max=.999, out_of_bounds="clip").fit(T.p_raw, T.hold)
    c4 = json.loads((IN / "report_r4_XRP.json").read_text())["models"]["QX4"]["c"]
    rep = {"train": {"n": len(T), "days": int(T.index.normalize().nunique()), "range": [str(T.index.min()), str(T.index.max())],
                     "hold_rate": float(T.hold.mean()), "p_raw_mean": float(T.p_raw.mean())}, "c_r4": c4, "P1": {}, "win": {}}
    S = {}
    for part, wk in (("val", "v_QX4"), ("test", "t_QX4")):
        s, P = qx(f"pred_{part}_r4_XRP.parquet", wk, c4)
        if part == "test":
            ql = float(V.qlike(P.y.to_numpy(), P[wk].to_numpy() + c4).mean())
            rep["P1"] = {"qlike": ql, "want": QL_TEST, "ok": abs(ql - QL_TEST) < 1e-9}
            assert rep["P1"]["ok"], rep["P1"]
        S[part] = s
    rows_ = []
    for part, (a, b) in WIN4.items():
        D = A.join(S[part].rename("lrv"), how="inner")
        D = D[(D.index >= a) & (D.index < pd.Timestamp(b)) & np.isfinite(D.lrv)].copy()
        D["p_raw"] = R.p_stay(np.sqrt(np.exp(D.lrv)) / D.a)
        D["p_iso"] = iso.predict(D.p_raw)
        D["p_ethmap"] = np.interp(D.p_raw, ETH_ISO["x"], ETH_ISO["y"])
        D["p_mix"] = C.sig((C.logit(D.p.to_numpy()) + C.logit(D.p_iso.to_numpy())) / 2)
        D["p_const"] = D.hold.mean()
        y, days = D.hold.to_numpy(float), D.index.normalize().to_numpy()
        M = {"HGB": "p", "QX4원판": "p_raw", "QX4등조": "p_iso", "ETH지도": "p_ethmap", "섞은것": "p_mix", "상수": "p_const"}
        r = {"n": len(D), "days": int(len(np.unique(days))), "range": [str(D.index.min()), str(D.index.max())], "hold_rate": float(y.mean()), "models": {}}
        for k, col in M.items():
            p = D[col].to_numpy(float)
            if k == "상수":
                e = {"brier": float(np.mean((p - y) ** 2))}
            else:
                m = C.metrics(y, p, cf["q20"], cf["q80"]); e = {x: m[x] for x in ("brier", "logloss", "auc", "ece_pp", "calmax_pp", "bias_pp")}
                e["dbrier_vs_const"] = dmean(p, D.p_const.to_numpy(), y, days)
            if k != "HGB":
                e["dbrier_vs_HGB"] = dmean(p, D.p.to_numpy(), y, days)
                if k != "상수":
                    e["dauc_vs_HGB_ci95"] = C.auc_boot(y, p, D.p.to_numpy(), days)
            r["models"][k] = e
        r["monthly_dbrier_iso_vs_hgb"] = {str(k): round(float(v), 5) for k, v in
                                          pd.Series((D.p_iso - D.hold) ** 2 - (D.p - D.hold) ** 2).groupby(D.index.to_period("M")).mean().items()}
        rep["win"][part] = r
        rows_.append(D.assign(win=part)[["win", "hold", "a", "lrv", "p", "p_raw", "p_iso", "p_ethmap", "p_mix"]])
    t, vv = rep["win"]["test"]["models"]["QX4등조"], rep["win"]["val"]["models"]["QX4등조"]
    g = {"G1": t["dbrier_vs_HGB"]["one_sided90"][1] < 0, "G2": t["dbrier_vs_const"]["one_sided90"][1] < 0,
         "G3": all(x["auc"] >= .72 and x["calmax_pp"] <= 6 for x in (t, vv))}
    rep["verdict"], rep["pass"] = g, all(g.values())
    rep["candidate"] = t["dbrier_vs_HGB"]["mean"] < 0 and t["dbrier_vs_const"]["mean"] < 0 and g["G3"]
    rep["iso"] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    pd.concat(rows_).to_parquet(OUT / "scored_4h.parquet")
    return rep


def selftest():
    R.selftest()
    rng = np.random.default_rng(0)                       # 굴림 Platt: 치우친 p 를 고친다 · 해결 안 된 행은 적합에 안 씀(rolling 의 L5 assert)
    ts = (pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(60000) * 5, "min")).to_numpy()
    q = rng.uniform(.05, .6, len(ts)); y = (rng.uniform(size=len(ts)) < q).astype(float); p = C.sig(C.logit(q) + 0.5)
    pb = C.rolling("platt", ts, p, y, 60, 90, t0="2025-01-01")
    late = ts >= np.datetime64("2025-03-01")
    assert np.mean((pb[late] - y[late]) ** 2) < np.mean((p[late] - y[late]) ** 2) and abs(pb[late].mean() - y[late].mean()) < .01
    assert (pb[ts < np.datetime64("2025-02-01")] == p[ts < np.datetime64("2025-02-01")]).all()   # 첫 달 = 항등
    print("selftest OK — p_stay · 굴림 Platt 치우침 교정 · 첫 달 항등")


def run():
    selftest()
    rp = OUT / "report.json"; assert not rp.exists(), "판정은 한 번만"
    A1, cf1, W1 = rows("1h"); A4, cf4, _ = rows("4h")
    rep = {"prereg": "docs/prereg/xrp_box_hold_platt90_qx4_20261031.md", "L1": gate_L1(A1, A4)}
    print("L1", rep["L1"], flush=True)
    rep["Q1_1h"] = q1(A1, cf1, W1)
    rep["Q2_4h"] = q2(A4, cf4)
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: rep[k] for k in ("L1",)} | {"Q1": {x: rep["Q1_1h"][x] for x in ("pooled", "verdict", "pass", "candidate")},
                      "Q2": {x: rep["Q2_4h"][x] for x in ("train", "P1", "verdict", "pass", "candidate")}}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    run() if "--run" in sys.argv else selftest() if "--selftest" in sys.argv else print(__doc__)
