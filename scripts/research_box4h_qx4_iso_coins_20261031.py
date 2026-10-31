#!/usr/bin/env python3
"""BTC·SOL(+XRP 재채점) 박스 4h — QX4 σ → P_stay → 등조 지도, 팔 S(자기 지도) vs 팔 E(ETH 지도) 검증으로 고르기 → 시험 1회.
사전등록 docs/prereg/btc_sol_box4h_qx4_iso_20261031.md.

    OMP_NUM_THREADS=3 python scripts/research_box4h_qx4_iso_coins_20261031.py --selftest
    OMP_NUM_THREADS=3 python scripts/research_box4h_qx4_iso_coins_20261031.py --run      # 한 번만(report.json 있으면 거절)
입력 tmp/box4h_qx4_20261031/in/(동결 QX4 예측·report 사본, md5 ../inputs.md5) · 박스 행 = 10-31 BTC·SOL 박스 연구 features.parquet · XRP xrp_rows.parquet.
XRP Q2(research_xrp_box_hold_platt90_qx4_20261031.q2)를 코인 인자로 넓힌 것 — 같은 함수(qx·dmean·p_stay·metrics·day_boot)를 import. 실주문·REST 없음.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import research_xrp_box_hold_platt90_qx4_20261031 as P   # noqa: E402 -- qx · dmean · C(metrics·auc_boot·labels) · R(p_stay) · V(qlike)
import dashboard.box_hold as bh                           # noqa: E402

C, R, V = P.C, P.R, P.V
OUT = ROOT / "tmp/box4h_qx4_20261031"
IN = OUT / "in"
P.IN = IN
MIN = pd.Timedelta("1min")
XRP_REP = ROOT / "docs/experiments/xrp_box_hold_platt90_qx4_20261031_report.json"
ETH_ISO = P.ETH_ISO
COINS = {"XRP": (IN / "xrp_rows.parquet", "xrp", "2026-10-20"),
         "BTC": (ROOT / "tmp/box_btc_sol_20261031/btc/features.parquet", "btc", "2026-10-28"),
         "SOL": (ROOT / "tmp/box_btc_sol_20261031/sol/features.parquet", "sol", "2026-10-28")}
WIN = {"val": ("2026-01-01", "2026-06-01"), "test": ("2026-06-01", "2026-10-24")}
ARMS = {"S": "자기지도", "E": "ETH지도"}


def rows(coin):
    """P.rows("4h") 와 같은 행 · τ 인덱스 · 반폭 a."""
    path, a, end = COINS[coin]
    cf = bh.BUNDLES[a]["scales"]["4h"]
    A = pd.read_parquet(path, filters=[("scale", "==", "4h")]).replace([np.inf, -np.inf], np.nan)
    A = A.dropna(subset=cf["cols"]).sort_values("ts").reset_index(drop=True)
    A = A[(A.ts + 241 * MIN <= pd.Timestamp(end)) & (np.exp(A.log_rng) > 0)].reset_index(drop=True)
    assert set(A.hold.unique()) <= {0, 1}
    A["p"] = cf["model"].predict_proba(A[cf["cols"]])[:, 1]
    A = A.assign(tau=A.ts + MIN).set_index("tau")
    A["a"] = np.exp(A.log_rng) / 2e4
    return A, cf


def gate_L1(coin, A):
    """1분봉에서 4h 라벨 독립 재계산 → 100% 일치."""
    import research_btc_sol_box_hold_20261031 as B
    K = B.load_1m(f"{coin}USDT")
    y = C.labels(*(K[k].to_numpy(float) for k in ("h", "l", "c")), 240)
    i = pd.Series(np.arange(len(K)), index=K.t).reindex(A.ts.astype("int64").to_numpy() // 10 ** 6).to_numpy()
    assert np.isfinite(i).all()
    same = float((y[i.astype(int)] == A.hold.to_numpy()).mean())
    assert same == 1.0, (coin, same)
    return {"rows": len(A), "match": same}


def self_map(coin, A):
    """XRP q2 와 같은 학습: W1·W2 표본 밖 QX4(각 창 c) · τ ∈ [2024, 2026) · 등조."""
    tr = []
    for w in ("5W1", "5W2"):
        c = json.loads((IN / f"report_r{w}_{coin}.json").read_text())["models"]["QX4"]["c"]
        tr += [P.qx(f"pred_val_r{w}_{coin}.parquet", "v_QX4", c)[0], P.qx(f"pred_test_r{w}_{coin}.parquet", "t_QX4", c)[0]]
    L = pd.concat(tr); L = L[~L.index.duplicated()]
    T = A.join(L.rename("lrv"), how="inner")
    T = T[(T.index >= "2024-01-01") & (T.index < "2026-01-01")]
    assert (T.index + pd.Timedelta("4h")).max() <= pd.Timestamp("2026-01-01")
    T["p_raw"] = R.p_stay(np.sqrt(np.exp(T.lrv)) / T.a)
    iso = IsotonicRegression(y_min=.001, y_max=.999, out_of_bounds="clip").fit(T.p_raw, T.hold)
    info = {"n": len(T), "days": int(T.index.normalize().nunique()), "range": [str(T.index.min()), str(T.index.max())],
            "hold_rate": float(T.hold.mean()), "p_raw_mean": float(T.p_raw.mean())}
    return iso, info


def score(coin):
    A, cf = rows(coin)
    iso, info = self_map(coin, A)
    r4 = json.loads((IN / f"report_r4_{coin}.json").read_text())["models"]["QX4"]
    rep = {"train": info, "c_r4": r4["c"], "win": {}}
    S = {}
    for part, wk in (("val", "v_QX4"), ("test", "t_QX4")):
        s, Pf = P.qx(f"pred_{part}_r4_{coin}.parquet", wk, r4["c"])
        if part == "test":
            ql = float(V.qlike(Pf.y.to_numpy(), Pf[wk].to_numpy() + r4["c"]).mean())
            rep["P0b"] = {"qlike": ql, "want": r4["test_qlike"], "ok": abs(ql - r4["test_qlike"]) < 1e-9}
        S[part] = s
    scored = []
    for part, (lo, hi) in WIN.items():
        D = A.join(S[part].rename("lrv"), how="inner")
        D = D[(D.index >= lo) & (D.index < pd.Timestamp(hi)) & np.isfinite(D.lrv)].copy()
        D["p_raw"] = R.p_stay(np.sqrt(np.exp(D.lrv)) / D.a)
        D["p_self"] = iso.predict(D.p_raw)
        D["p_eth"] = np.interp(D.p_raw, ETH_ISO["x"], ETH_ISO["y"])
        D["p_const"] = D.hold.mean()
        y, days = D.hold.to_numpy(float), D.index.normalize().to_numpy()
        r = {"n": len(D), "days": int(len(np.unique(days))), "range": [str(D.index.min()), str(D.index.max())], "hold_rate": float(y.mean()), "models": {}}
        for k, col in {"HGB": "p", "QX4원판": "p_raw", "자기지도": "p_self", "ETH지도": "p_eth", "상수": "p_const"}.items():
            p = D[col].to_numpy(float)
            if k == "상수":
                e = {"brier": float(np.mean((p - y) ** 2))}
            else:
                m = C.metrics(y, p, cf["q20"], cf["q80"]); e = {x: m[x] for x in ("brier", "logloss", "auc", "ece_pp", "calmax_pp", "bias_pp")}
                e["dbrier_vs_const"] = P.dmean(p, D.p_const.to_numpy(), y, days)
            if k != "HGB":
                e["dbrier_vs_HGB"] = P.dmean(p, D.p.to_numpy(), y, days)
                if k != "상수":
                    e["dauc_vs_HGB_ci95"] = C.auc_boot(y, p, D.p.to_numpy(), days)
            r["models"][k] = e
        rep["win"][part] = r
        scored.append(D.assign(win=part)[["win", "hold", "a", "lrv", "p", "p_raw", "p_self", "p_eth"]])
    D = pd.concat(scored)
    v, t = rep["win"]["val"]["models"], rep["win"]["test"]["models"]
    ok = lambda e: e["auc"] >= .72 and e["calmax_pp"] <= 6  # noqa: E731
    good = [k for k in ARMS.values() if ok(v[k])]
    arm = min(good, key=lambda k: v[k]["brier"]) if good else min(ARMS.values(), key=lambda k: v[k]["calmax_pp"])
    g = {"G1": t[arm]["dbrier_vs_HGB"]["one_sided90"][1] < 0, "G2": t[arm]["dbrier_vs_const"]["one_sided90"][1] < 0,
         "G3": ok(v[arm]) and ok(t[arm])}
    T = D[D.win == "test"]
    col = {"자기지도": "p_self", "ETH지도": "p_eth"}[arm]
    rep.update(arm=arm, arm_rule="val G3 통과 팔 중 Brier 최소" if good else "둘 다 val G3 ✗ → 보정 최대 차 최소", verdict=g,
               strict=all(g.values()) and coin != "XRP",
               candidate=bool(t[arm]["dbrier_vs_HGB"]["mean"] < 0 and t[arm]["dbrier_vs_const"]["mean"] < 0 and g["G3"]),
               monthly_dbrier_test={str(k): round(float(x), 5) for k, x in
                                    ((T[col] - T.hold) ** 2 - (T.p - T.hold) ** 2).groupby(T.index.to_period("M")).mean().items()},
               iso={"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()})
    D.to_parquet(OUT / f"scored_{coin}.parquet")
    print(f"{coin}: 팔 {arm}({rep['arm_rule']}) · " + " · ".join(
        f"{w} {k} Brier {rep['win'][w]['models'][k]['brier']:.5f} AUC {rep['win'][w]['models'][k]['auc']:.4f} 최대차 {rep['win'][w]['models'][k]['calmax_pp']:.2f}"
        for w in ("val", "test") for k in ("HGB", "자기지도", "ETH지도")) + f" · {g} 엄격 {rep['strict']} 후보 {rep['candidate']}", flush=True)
    return rep, A


def control(rep_x):
    """P0a: XRP 10-31 Q2 숫자 재현."""
    w = json.loads(XRP_REP.read_text())["Q2_4h"]
    d = {"train_n": rep_x["train"]["n"] == w["train"]["n"]}
    for part in ("val", "test"):
        for k_new, k_old in (("HGB", "HGB"), ("QX4원판", "QX4원판"), ("자기지도", "QX4등조"), ("ETH지도", "ETH지도")):
            d[f"{part}_{k_new}"] = abs(rep_x["win"][part]["models"][k_new]["brier"] - w["win"][part]["models"][k_old]["brier"])
        d[f"{part}_n"] = rep_x["win"][part]["n"] == w["win"][part]["n"]
    d["self_dbrier"] = abs(rep_x["win"]["test"]["models"]["자기지도"]["dbrier_vs_HGB"]["mean"] - w["win"]["test"]["models"]["QX4등조"]["dbrier_vs_HGB"]["mean"])
    d["ok"] = all(v is True or (not isinstance(v, bool) and v <= 1e-12) for v in d.values())
    print("P0a", d, flush=True)
    return d


def selftest():
    R.selftest()
    assert all(abs(np.interp(g, ETH_ISO["x"], ETH_ISO["y"]) - v) < 1e-12 for g, v in ETH_ISO["parity"])
    assert {a for _, a, _ in COINS.values()} <= set(bh.BUNDLES)
    print("selftest OK — p_stay · ETH 지도 보간 = 내보낼 때 적은 값")


def run():
    selftest()
    rp = OUT / "report.json"; assert not rp.exists(), "판정은 한 번만"
    rep = {"prereg": "docs/prereg/btc_sol_box4h_qx4_iso_20261031.md"}
    rep["XRP"], _ = score("XRP")
    rep["P0a"] = control(rep["XRP"])
    assert rep["P0a"]["ok"] and rep["XRP"]["P0b"]["ok"], "양성 대조 실패 → 채점 안 함"
    for coin in ("BTC", "SOL"):
        r, A = score(coin)
        assert r["P0b"]["ok"], r["P0b"]
        r["L1"] = gate_L1(coin, A)
        rep[coin] = r
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({c: {k: rep[c][k] for k in ("arm", "verdict", "strict", "candidate", "P0b")} for c in COINS}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    run() if "--run" in sys.argv else selftest() if "--selftest" in sys.argv else print(__doc__)
