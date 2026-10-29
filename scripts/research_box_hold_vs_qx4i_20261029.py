#!/usr/bin/env python3
"""4h 박스 유지: 박스 HGB vs QX4I σ(반사 원리 무접촉 확률) — 사전등록 docs/prereg/box_hold_vs_qx4i_20261029.md (bd9f8325).

    BOXVOL_IN=<입력 폴더> BOXVOL_OUT=<출력 폴더> python scripts/research_box_hold_vs_qx4i_20261029.py --selftest
    BOXVOL_IN=... BOXVOL_OUT=... python scripts/research_box_hold_vs_qx4i_20261029.py --run      # 한 번만(report.json 있으면 거절)
입력 폴더: rows.parquet(tmp/box_calib_20261025) · pred_{val,test}_r4_ETH · pred_{val,test}_r5W{1,2}_ETH · report_r5W{1,2}_ETH.json
(tmp/vol4h_forecast_20261025) · deribit/trades/*.parquet(tmp/deribit_optflow_20261025). 라벨·피쳐·QX4·QX4I 는 기존 산출·함수 그대로. 실주문 없음.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_box_hold_calibration_drift_20261025 as C   # noqa: E402 -- BH(배포 박스 HGB)·metrics·day_boot·auc_boot·logit·sig
import research_vol4h_round12_option_iv_20261025 as R12        # noqa: E402 -- qx4i(동결 BETA_I)·iv_hourly
import research_vol4h_forecast_horserace_20261025 as V          # noqa: E402 -- qlike

IN = Path(os.environ.get("BOXVOL_IN", ROOT / "tmp/boxvol_20261029/in"))
OUT = Path(os.environ.get("BOXVOL_OUT", ROOT / "tmp/boxvol_20261029"))
C_QX4I = 0.2831459942982983                      # 12라운드 report.json models.QX4I.c (검증 2026-01~05 동결)
QL_TEST, QL_VAL = 0.26857101946570916, 0.2831459942982983   # 12라운드 test_qlike·val_qlike (재현 관문 P1)
WINS = {"val": ("2026-01-01", "2026-06-01"), "test": ("2026-06-01", "2026-10-24")}
TR_END = pd.Timestamp("2026-01-01")
K = np.arange(400)[None, :]
CF = C.BH["scales"]["4h"]


def p_stay(s):
    """구간 [−a, a] 가운데 출발 · 표류 0 브라운 운동이 시간 T 동안 두 벽에 안 닿을 확률, s = σ√T / a."""
    s = np.asarray(s, float)[:, None]
    p = 4 / np.pi * np.sum((-1.0) ** K / (2 * K + 1) * np.exp(-(2 * K + 1) ** 2 * np.pi ** 2 * np.maximum(s, 0.02) ** 2 / 8), 1)
    return np.where(s[:, 0] < 0.02, 1.0, np.clip(p, 0, 1))


def qx(name, col, c=0.0):
    """동결 예측 → (결정 시각 τ = 5분봉 시작 + 5분) 인덱스의 ŷ + c 와 원 예측 프레임."""
    P = pd.read_parquet(IN / name)
    assert (P.index.minute == 55).all(), name
    return pd.Series(P[col].to_numpy() + c, index=P.index + pd.Timedelta("5min")), P


def selftest():
    assert p_stay([0.0])[0] == 1.0 and p_stay([5.0])[0] < 1e-6
    ps = p_stay(np.linspace(0.05, 3, 60)); assert (np.diff(ps) <= 1e-12).all() and abs(ps[0] - 1) < 1e-9, ps[:5]   # 작은 s 는 1 에 붙는다
    rng = np.random.default_rng(0); n, m = 4000, 4000                     # 몬테카를로(이산 감시 → 약간 높게 나옴)
    for s in (0.5, 1.0, 1.5):
        x = np.cumsum(rng.normal(0, s / np.sqrt(m), (n, m)), 1)
        mc = float((np.abs(x).max(1) < 1).mean())
        assert abs(mc - p_stay([s])[0]) < 0.025, (s, mc, p_stay([s])[0])
    print("selftest OK — p_stay 경계·단조·몬테카를로 일치")


def fit_iso():
    """박스 4h 행(HGB p 포함) · 학습 창 행 · 등조 지도."""
    A = pd.read_parquet(IN / "rows.parquet", filters=[("scale", "==", "4h")])
    assert (A.ts.dt.minute == 59).all()
    A["tau"] = A.ts + pd.Timedelta("1min")
    A = A.replace([np.inf, -np.inf], np.nan).dropna(subset=CF["cols"]).set_index("tau")
    A["a"] = np.exp(A.log_rng) / 2e4                                      # 반폭 = ½·rng/c
    A["p_hgb"] = CF["model"].predict_proba(A[CF["cols"]])[:, 1]
    # 학습 창: W1·W2 표본 밖 QX4 (각 창 c) → 등조 지도
    tr = []
    for w in ("5W1", "5W2"):
        c = json.loads((IN / f"report_r{w}_ETH.json").read_text())["models"]["QX4"]["c"]
        tr += [qx(f"pred_val_r{w}_ETH.parquet", "v_QX4", c)[0], qx(f"pred_test_r{w}_ETH.parquet", "t_QX4", c)[0]]
    L = pd.concat(tr); L = L[~L.index.duplicated()]
    T = A.join(L.rename("lrv"), how="inner")
    T = T[(T.index >= "2024-01-01") & (T.index < TR_END)]
    assert (T.index + pd.Timedelta("4h")).max() <= TR_END
    T["p_raw"] = p_stay(np.sqrt(np.exp(T.lrv)) / T.a)
    return A, T, IsotonicRegression(y_min=.001, y_max=.999, out_of_bounds="clip").fit(T.p_raw, T.y)


def run():
    selftest()
    rp = OUT / "report.json"; assert not rp.exists(), "판정은 한 번만"
    OUT.mkdir(parents=True, exist_ok=True)
    A, T, iso = fit_iso()
    # 검증·시험: QX4I = 12라운드 동결 식
    R12.O.OUT = IN / "deribit"
    iv = R12.iv_hourly(int(pd.Timestamp("2026-10-25").value // 10**6))
    rep = {"prereg": "docs/prereg/box_hold_vs_qx4i_20261029.md", "train": {"n": len(T), "days": int(T.index.normalize().nunique()),
           "range": [str(T.index.min()), str(T.index.max())], "hold_rate": float(T.y.mean()), "p_raw_mean": float(T.p_raw.mean())}, "P1": {}, "win": {}}
    S = {}
    for part, wk in (("val", "v_QX4"), ("test", "t_QX4")):
        _, P = qx(f"pred_{part}_r4_ETH.parquet", wk)
        lq = R12.qx4i(P.index, P[wk].to_numpy(), iv) + C_QX4I
        ql = float(V.qlike(P.y.to_numpy(), lq).mean()); want = QL_VAL if part == "val" else QL_TEST
        rep["P1"][part] = {"qlike": ql, "want": want, "ok": abs(ql - want) < 1e-9}
        assert rep["P1"][part]["ok"], rep["P1"]
        S[part] = pd.Series(lq, index=P.index + pd.Timedelta("5min"))
    rows = []
    for part, (a, b) in WINS.items():
        D = A.join(S[part].rename("lrv"), how="inner")
        D = D[(D.index >= a) & (D.index < b) & np.isfinite(D.lrv)].copy()
        D["p_raw"] = p_stay(np.sqrt(np.exp(D.lrv)) / D.a)
        D["p_iso"] = iso.predict(D.p_raw)
        D["p_mix"] = C.sig((C.logit(D.p_hgb.to_numpy()) + C.logit(D.p_iso.to_numpy())) / 2)
        D["p_const"] = D.y.mean()
        y, days = D.y.to_numpy(float), D.index.normalize().to_numpy()
        M = {"HGB": D.p_hgb, "QX4I원판": D.p_raw, "QX4I등조": D.p_iso, "섞은것": D.p_mix, "상수": D.p_const}
        r = {"n": len(D), "days": int(len(np.unique(days))), "range": [str(D.index.min()), str(D.index.max())], "hold_rate": float(y.mean()), "models": {}}
        for k, p in M.items():
            p = p.to_numpy(float); m = C.metrics(y, p, CF["q20"], CF["q80"])
            e = {x: m[x] for x in ("brier", "auc", "ece_pp", "bias_pp")}
            if k == "상수":
                e["ece_pp"] = 0.0                                          # 분위 칸 1개(qcut NaN) -- 창 평균이라 정의상 0
            if k != "HGB":
                d = (p - y) ** 2 - (D.p_hgb.to_numpy() - y) ** 2
                lo, hi = C.day_boot(d, days, alpha=0.2)
                e["dbrier_vs_HGB"] = {"mean": float(d.mean()), "one_sided90": [lo, hi], "ci95": C.day_boot(d, days),
                                      "verdict": f"{k} 더 정확(통과)" if hi < 0 else "HGB 더 정확(통과)" if lo > 0 else "구분 못 함(불통과)"}
                if k != "상수":
                    e["dauc_vs_HGB"] = {"mean": e["auc"] - r["models"]["HGB"]["auc"], "ci95": C.auc_boot(y, p, D.p_hgb.to_numpy(), days)}
            if k != "상수":
                d = (p - y) ** 2 - (D.p_const.to_numpy() - y) ** 2
                e["dbrier_vs_const"] = {"mean": float(d.mean()), "one_sided90": C.day_boot(d, days, alpha=0.2)}
            r["models"][k] = e
        rep["win"][part] = r
        rows.append(D.assign(win=part)[["win", "y", "a", "lrv", "p_hgb", "p_raw", "p_iso", "p_mix"]])
    t, v = rep["win"]["test"]["models"], rep["win"]["val"]["models"]
    ok = [k for k in ("QX4I등조", "섞은것") if t[k]["dbrier_vs_HGB"]["one_sided90"][1] < 0]
    rep["main"] = t["QX4I등조"]["dbrier_vs_HGB"]["verdict"]; rep["sub"] = t["섞은것"]["dbrier_vs_HGB"]["verdict"]
    rep["chart"] = min(ok, key=lambda k: v[k]["brier"]) if ok else "HGB(현행 유지)"
    pd.concat(rows).to_parquet(OUT / "scored.parquet")
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=float))


def aux():
    """채점 뒤 보조(판정에 안 씀): 옵션 IV 몫(QX4 등조 = 같은 지도에 IV 없는 σ) · 5일 블록 95%(V.day_ci) · 월별 ΔBrier."""
    D = pd.read_parquet(OUT / "scored.parquet"); D = D[D.win == "test"]
    c4 = json.loads((IN / "report_r12.json").read_text())["models"]["QX4"]["c"]          # 12라운드 QX4 c(같은 검증 창)
    q, _ = qx("pred_test_r4_ETH.parquet", "t_QX4", c4)
    iso = fit_iso()[2]
    D["p_q4"] = iso.predict(p_stay(np.sqrt(np.exp(q.reindex(D.index).to_numpy())) / D.a))
    assert np.allclose(iso.predict(D.p_raw), D.p_iso)                                     # 지도 재적합 = 본 채점과 같음
    y, days = D.y.to_numpy(float), D.index.normalize().to_numpy()
    L = lambda p: (D[p].to_numpy() - y) ** 2  # noqa: E731
    out = {}
    for name, a, b in (("QX4등조−HGB", "p_q4", "p_hgb"), ("QX4I등조−QX4등조", "p_iso", "p_q4"), ("QX4I등조−HGB", "p_iso", "p_hgb"), ("섞은것−HGB", "p_mix", "p_hgb")):
        d = L(a) - L(b)
        out[name] = {"mean": float(d.mean()), "one_sided90_1d": C.day_boot(d, days, alpha=0.2), "ci95_5d": list(V.day_ci(D.index, d))[1:],
                     "monthly": {str(k): round(float(v), 5) for k, v in pd.Series(d, index=D.index).groupby(D.index.to_period("M")).mean().items()}}
    out["QX4등조_brier"] = float(L("p_q4").mean())
    (OUT / "aux.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    run() if "--run" in sys.argv else aux() if "--aux" in sys.argv else selftest() if "--selftest" in sys.argv else print(__doc__)
