#!/usr/bin/env python3
"""BTC·SOL 4h 변동성 QX4 vs HAR · BTC QX4I 재현 (2026-10-31). 사전등록 docs/prereg/btc_sol_vol4h_qx4_20261031.md.

    python3 -I scripts/research_btc_sol_vol4h_qx4_20261031.py --selftest
    python3 -I scripts/research_btc_sol_vol4h_qx4_20261031.py --run        # 한 번만(report.json 있으면 거절)
입력 = 변동성 경주 동결 예측 사본(tmp/vol4h_btcsol_20261031/in). 학습 없음(QX4 는 원래 4코인 모델). 실주문 없음.
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "tmp" / "vol4h_btcsol_20261031"
IN = OUT / "in"
SRC = Path("/home/kbj20/crypto-scalping/.claude/worktrees/chart-indicators-tuning-ecfa25/tmp/vol4h_forecast_20261025")
MAIN_TMP = Path("/home/kbj20/crypto-scalping/tmp")
COINS, TAGS, REPL = ("ETH", "XRP", "BTC", "SOL"), ("r4", "r5W1", "r5W2", "r14W0"), ("r14W0", "r5W1", "r5W2")


def qlike(y, yh):
    e = y - yh
    return np.exp(e) - e - 1


def block_q(x, qs, b=5, n=4000, seed=7):
    """research_eth_dir4h_reverify_20261019.block_ci 와 같은 재표집(같은 seed → 같은 표본), 분위만 qs."""
    rng = np.random.default_rng(seed); k = int(np.ceil(len(x) / b))
    st = rng.integers(0, len(x) - b + 1, size=(n, k))
    m = x[(st[:, :, None] + np.arange(b)).reshape(n, -1)[:, :len(x)]].mean(1)
    return [float(np.quantile(m, q)) for q in qs]


def day_stats(ts, d):
    """일 평균 계열 → [평균, 2.5%, 97.5%, 90%(단측 상한)], 독립 일수."""
    s = pd.Series(d, index=ts).groupby(ts.normalize()).mean().to_numpy()
    lo, hi, u90 = block_q(s, (.025, .975, .90))
    return {"mean": float(s.mean()), "ci95": [lo, hi], "upper90": u90, "days": len(s)}


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def score(Pv, Pt, arms):
    """arms {이름: (val 예측, test 예측)} → 모델별 지표 + 짝 차(이름 − M0, B − A)."""
    yv, yt, ts = Pv.y.to_numpy(), Pt.y.to_numpy(), Pt.index
    c = {m: float(np.log(np.mean(np.exp(yv - v)))) for m, (v, _) in arms.items()}
    L = {m: qlike(yt, t + c[m]) for m, (_, t) in arms.items()}
    S = {m: (yt - t) ** 2 for m, (_, t) in arms.items()}
    rep = {"n_test": len(yt), "models": {}}
    for m, (v, t) in arms.items():
        r = {"c": c[m], "val_qlike": float(qlike(yv, v + c[m]).mean()), "test_qlike": float(L[m].mean()), "test_mse": float(S[m].mean()),
             "r2_logrv": r2(yt, t), "corr": float(np.corrcoef(yt, t)[0, 1])}
        if m != "M0":
            r["dq_vs_M0"], r["dmse_vs_M0"] = day_stats(ts, L[m] - L["M0"]), day_stats(ts, S[m] - S["M0"])
        rep["models"][m] = r
    if "B" in arms:
        rep["B_minus_A"] = {"dq": day_stats(ts, L["B"] - L["A"]), "dmse": day_stats(ts, S["B"] - S["A"])}
        rep["monthly_dq_A_vs_M0"] = {str(k): round(float(v), 4) for k, v in pd.Series(L["A"] - L["M0"], index=ts).groupby(ts.to_period("M")).mean().items()}
    return rep


def arms_of(Pv, Pt, extra=True):
    a = {"M0": (Pv.v_M0.to_numpy(), Pt.t_M0.to_numpy()), "A": (Pv.v_QX4.to_numpy(), Pt.t_QX4.to_numpy())}
    if extra:
        a["B"] = (Pv[["v_M3", "v_M6"]].mean(1).to_numpy(), Pt[["t_M3", "t_M6"]].mean(1).to_numpy())
        a["M3"] = (Pv.v_M3.to_numpy(), Pt.t_M3.to_numpy())
    return a


def load(tag, s):
    return tuple(pd.read_parquet(IN / f"pred_{k}_{tag}_{s}.parquet") for k in ("val", "test"))


def copy_inputs():
    IN.mkdir(parents=True, exist_ok=True); lines = []
    for f in sorted([SRC / f"pred_{k}_{t}_{s}.parquet" for k in ("val", "test") for t in TAGS for s in COINS] + [SRC / f"report_r4_{s}.json" for s in COINS]):
        shutil.copy2(f, IN / f.name); lines.append(f"{hashlib.md5(f.read_bytes()).hexdigest()}  {f.name}")
    (OUT / "inputs.md5").write_text("\n".join(lines) + "\n")


def p0():
    """양성 대조: ETH·XRP 4라운드 «QX4 − M0» 재현 · BTC·SOL 입력 무결성."""
    out, ok = {}, True
    for s in COINS:
        ref = json.loads((IN / f"report_r4_{s}.json").read_text())["models"]
        Pv, Pt = load("r4", s); r = score(Pv, Pt, arms_of(Pv, Pt, extra=False))["models"]
        d = {"qlike_QX4": abs(r["A"]["test_qlike"] - ref["QX4"]["test_qlike"]), "qlike_M0": abs(r["M0"]["test_qlike"] - ref["M0"]["test_qlike"])}
        if s in ("ETH", "XRP"):
            got = [r["A"]["dq_vs_M0"]["mean"], *r["A"]["dq_vs_M0"]["ci95"]]
            d["dq_vs_M0"] = float(np.max(np.abs(np.subtract(got, ref["QX4"]["dq_vs_M0"]))))
        out[s] = d; ok &= max(d.values()) < 1e-9
    out["pass"] = bool(ok)
    return out


def q2_r0():
    """12라운드 코드 재실행 → ETH·BTC report 1e-9 재현. 판정 = 12b 그대로(단측 90% 상한만 같은 일 계열에서 기재)."""
    import research_vol4h_forecast_horserace_20261025 as V
    import research_vol4h_round12_option_iv_20261025 as R12
    V.OUT, R12.O.OUT = IN, MAIN_TMP / "deribit_optflow_20261025"   # R12.main 은 pred_*_r4_{sym} 를 V.OUT 에서 읽는다
    res = {}
    for sym, ref_dir in (("ETH", MAIN_TMP / "vol4h_r12_20261025"), ("BTC", MAIN_TMP / "vol4h_r12b_btc_20261026")):
        o = OUT / f"r12_{sym}"; o.mkdir(parents=True, exist_ok=True)
        if sym == "BTC" and not (o / "trades").exists():
            (o / "trades").symlink_to(ref_dir / "trades")
        R12.main(sym, o)
        got, ref = json.loads((o / "report.json").read_text()), json.loads((ref_dir / "report.json").read_text())
        diff = max([abs(a - b) for a, b in zip(got["beta_I"], ref["beta_I"])] +
                   [abs(got["models"][m]["test_qlike"] - ref["models"][m]["test_qlike"]) for m in ref["models"]] +
                   [abs(a - b) for m in ("QX4I", "QX4O") for a, b in zip(got["models"][m]["dq_vs_QX4"], ref["models"][m]["dq_vs_QX4"])])
        res[sym] = {"max_abs_diff": diff, "pass": diff < 1e-9, "beta_I": got["beta_I"], "dq_QX4I_vs_QX4_ci95": got["models"]["QX4I"]["dq_vs_QX4"]}
    # 단측 90% 상한: 같은 손실 차 일 계열 다시 만들기(R12.main 과 같은 c·예측)
    for sym in res:
        rep = json.loads((OUT / f"r12_{sym}" / "report.json").read_text())
        res[sym]["dq_upper90"] = _qx4i_upper90(sym, rep, V, R12)
        res[sym]["strict"] = bool(res[sym]["pass"] and res[sym]["dq_upper90"] < 0 and rep["beta_I"][1] > 0)
    return res


def _qx4i_upper90(sym, rep, V, R12):
    Pt = pd.read_parquet(IN / f"pred_test_r4_{sym}.parquet")
    tdir = R12.O.OUT / "trades" if sym == "ETH" else OUT / f"r12_{sym}" / "trades"
    tr = pd.concat([pd.read_parquet(f) for f in sorted(tdir.glob("*.parquet"))], ignore_index=True)
    tr = tr[tr.instrument_name.str.match(rf"^{sym}-\d+[A-Z]{{3}}\d{{2}}-\d+-[CP]$") & ~tr.block & ~tr.liq].reset_index(drop=True)
    exp = pd.to_datetime(tr.instrument_name.str.split("-").str[1], format="%d%b%y").astype("int64") // 10**6 + 8 * R12.H1
    tr["T"] = (exp - tr.timestamp) / 86_400_000; tr["d"] = R12.O.deltas(tr); tr = tr[np.isfinite(tr.d)]
    tr["dnb"] = np.where(tr.direction == "buy", 1, -1) * tr.d * tr.amount
    grid = np.arange(pd.Timestamp("2025-12-01").value // 10**6, pd.Timestamp("2026-10-25").value // 10**6, R12.H1)
    iv = R12.hourly(tr, grid).iv
    d = (Pt.index + pd.Timedelta("5min")).astype("int64") // 10**6
    yh, yt = Pt.t_QX4.to_numpy(), Pt.y.to_numpy(); b = rep["beta_I"]
    x1 = np.nan_to_num(np.log((iv.reindex(d).to_numpy() / 100) ** 2 * 4 / 8766) - yh)
    LI = V.qlike(yt, yh + b[0] + b[1] * x1 + rep["models"]["QX4I"]["c"]); L0 = V.qlike(yt, yh + rep["models"]["QX4"]["c"])
    st = day_stats(Pt.index, LI - L0)
    assert abs(st["mean"] - rep["models"]["QX4I"]["dq_vs_QX4"][0]) < 1e-9, "일 계열 재구성 불일치"
    return st["upper90"]


def run():
    rp = OUT / "report.json"; assert not rp.exists(), "판정은 한 번만"
    copy_inputs()
    rep = {"prereg": "docs/prereg/btc_sol_vol4h_qx4_20261031.md", "P0": p0()}
    print("P0", json.dumps(rep["P0"]), flush=True)
    assert rep["P0"]["pass"], "양성 대조 실패 — 채점 안 함"
    rep["Q1"] = {}
    for s in ("BTC", "SOL", "ETH", "XRP"):   # ETH·XRP 는 같은 표(참고)
        Pv, Pt = load("r4", s); r = score(Pv, Pt, arms_of(Pv, Pt)); A, M0 = r["models"]["A"], r["models"]["M0"]
        r["repl"] = {}
        for t in REPL:
            qv, qt = load(t, s); rr = score(qv, qt, arms_of(qv, qt, extra=False))["models"]
            r["repl"][t] = {"dq_vs_M0": rr["A"]["dq_vs_M0"], "r2_A": rr["A"]["r2_logrv"], "r2_M0": rr["M0"]["r2_logrv"]}
        n4 = sum(r["repl"][t]["dq_vs_M0"]["mean"] < 0 for t in REPL)
        g = {"G1": A["dq_vs_M0"]["upper90"] < 0, "G2": A["dmse_vs_M0"]["upper90"] < 0, "G3": A["r2_logrv"] > M0["r2_logrv"], "G4": n4 >= 2, "G4_windows": f"{n4}/3"}
        cand = A["dq_vs_M0"]["mean"] < 0 and A["dmse_vs_M0"]["mean"] < 0 and g["G3"] and g["G4"]
        r["gates"] = g; r["verdict"] = "엄격 통과" if g["G1"] and g["G2"] and g["G3"] and g["G4"] else "통과 후보" if cand else "불통과"
        r["val_pick"] = min(("A", "B"), key=lambda m: r["models"][m]["val_qlike"])
        sig = np.exp(Pt.t_QX4.to_numpy() / 2) * 100          # 중앙값형 σ4h %(서빙 sig4h_pct 와 같은 눈금)
        r["sig4h_pct_test_q"] = {q: round(float(np.quantile(sig, q)), 3) for q in (.05, .25, .5, .75, .95)}
        rep["Q1"][s] = r
        print(s, r["verdict"], g, "dq", A["dq_vs_M0"], "R2", round(A["r2_logrv"], 4), round(M0["r2_logrv"], 4), flush=True)
    rep["Q2"] = q2_r0(); print("Q2", json.dumps(rep["Q2"]), flush=True)
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))


def selftest():
    rng = np.random.default_rng(0)
    y = rng.normal(-7, .5, 600); assert abs(r2(y, y) - 1) < 1e-12 and abs(qlike(y, y).sum()) < 1e-12
    ts = pd.date_range("2026-06-01", periods=600, freq="h")
    good = day_stats(ts, qlike(y, y + rng.normal(0, .1, 600)) - qlike(y, y + rng.normal(0, .6, 600)))
    assert good["upper90"] < 0 < good["mean"] + 1 and good["ci95"][1] >= good["upper90"]   # 좋은 예측 − 나쁜 예측 < 0
    from research_eth_dir4h_reverify_20261019 import block_ci
    x = rng.normal(0, 1, 50); assert block_q(x, (.025, .975)) == list(block_ci(x))          # 같은 재표집
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else run() if "--run" in sys.argv else print(__doc__)
