"""XRP 판 «4h 변동성 σ ±% · 확대 확률» 모델 (2026-10-20) -- ETH 배포판(scripts/live_eth_sizing_vol_model_20260912.py)과 같은 피쳐·타깃·GBM·시드.

ETH 라이브 파일은 고치지 않는다: build_features·predict_vol·fit_expand_logit 을 그대로 가져다 쓴다(학습·라이브 같은 함수 계약).
검증(결과 보기 전 고정):
  분할 = 학습 XRP 2021-12 ~ 2024-12-31(타깃 4h 가 2025 로 넘지 않게 정화) · 표본외 2025-01 ~ (주 판정 2025~ · 2026 따로).
  비교 = (a) HAR-RV(로그 rv48·rv288·rv2016 OLS, XRP 같은 학습 창) (b) ETH 레시피를 ETH ≤2024 로 학습해 XRP 에 적용(공정)
         (c) 배포된 ETH 아티팩트를 XRP 에 그대로(학습 끝 2026-10-14 -- 시기 겹침, 진단 전용).
  지표 = 로그 변동성 R²(표본외 평균 기준) · QLIKE(분산: y²/p² − ln − 1) · 스피어만 · 중앙 편향. 차이는 UTC 일 재표집 1,000회 CI.
  확대 확률 = P(y ≥ 1.3·rv48) = sigmoid(a + b·ln(pred/rv48)) -- 2025 표본외로 적합 → 2026 에서 십분위 보정 오차 · ETH VOL_P 를 그대로 쓴 경우와 비교.
  산출 아티팩트 = 전 기간 학습(ETH --monthly 와 같은 구조) + expand_logit(2025~ 표본외 적합).
실행: nice python -I scripts/research_xrp_vol4h_20261020.py <vision 5m zip 폴더> [--selftest]
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.live_eth_sizing_vol_model_20260912 import (EXPAND_LABEL, FEATURES, HOLD, SEEDS, build_features,  # noqa: E402
                                                        fit_expand_logit, predict_vol)

OUT = Path("/home/kbj20/crypto-scalping/tmp/xrp_parity_20261020/vol4h")
ETH_LONG = Path.home() / "backups/crypto-scalping-server/data/live/vol_panel/ETHUSDT_5m.parquet"      # 서버 사본(읽기 전용)
ETH_ART = Path.home() / "backups/crypto-scalping-server/data/live/eth_sizing_vol_model.joblib"
TEST0, Y26 = np.datetime64("2025-01-01"), np.datetime64("2026-01-01")
VOL_P_ETH = (-1.4950, 5.1671)


def load_zips(folder: Path) -> pd.DataFrame:
    """vision 선물 5분봉 zip(월·일) → ETH LONG 과 같은 열."""
    parts = []
    for p in sorted(folder.glob("*.zip")):
        with zipfile.ZipFile(p) as z:
            txt = z.read(z.namelist()[0]).decode()
        k = pd.read_csv(io.StringIO(txt), header=None)
        if not str(k.iat[0, 0]).isdigit():
            k = k.iloc[1:]
        parts.append(pd.DataFrame({"timestamp": pd.to_datetime(k[0].astype("int64"), unit="ms"), "high": k[2].astype(float),
                                   "low": k[3].astype(float), "close": k[4].astype(float), "quote_volume": k[7].astype(float),
                                   "trades": k[8].astype(float)}))
    return pd.concat(parts).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def frame(d: pd.DataFrame):
    X = build_features(d["timestamp"], d["close"].to_numpy(float), d["quote_volume"].to_numpy(float),
                       d["trades"].to_numpy(float), d["high"].to_numpy(float), d["low"].to_numpy(float))
    lr = np.diff(np.log(np.maximum(d["close"].to_numpy(float), 1e-12)), prepend=0.0)
    s = pd.Series(lr)
    y = s.rolling(HOLD, min_periods=HOLD).std().shift(-HOLD).to_numpy()
    rv2016 = np.log(np.maximum(s.rolling(2016, min_periods=500).std().to_numpy(), 1e-12))       # HAR 주간 항
    ok = np.isfinite(y) & (y > 0) & np.isfinite(X.to_numpy(float)).all(1) & np.isfinite(rv2016)
    return X, y, rv2016, ok, d["timestamp"].to_numpy()


def fit_gbm(X, y, m):
    from sklearn.ensemble import HistGradientBoostingRegressor
    A = np.nan_to_num(X.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    return [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=int(s)).fit(A[m], np.log(y[m]))
            for s in SEEDS]


def har(X, rv2016):
    return np.column_stack([np.ones(len(X)), X["rv48"], X["rv288"], rv2016])


def metrics(y, p):
    ly, lp = np.log(y), np.log(p)
    from scipy.stats import spearmanr
    v = (y / p) ** 2
    return {"r2_logvol": float(1 - np.sum((ly - lp) ** 2) / np.sum((ly - ly.mean()) ** 2)), "qlike": float(np.mean(v - np.log(v) - 1)),
            "msle": float(np.mean((ly - lp) ** 2)), "spearman": float(spearmanr(p, y).statistic),
            "median_ratio_y_over_p": float(np.exp(np.median(ly - lp))), "n": int(len(y))}


def diff_ci(y, pa, pb, day, rng, n=1000):
    """(b − a) 일 재표집 CI -- 양수면 a(XRP GBM)가 낫다. QLIKE 와 제곱 로그 오차."""
    out = {}
    for nm, f in (("qlike", lambda p: (y / p) ** 2 - np.log((y / p) ** 2) - 1), ("sq_log_err", lambda p: (np.log(y) - np.log(p)) ** 2)):
        g = pd.Series(f(pb) - f(pa)).groupby(day).agg(["sum", "count"])
        S, C = g["sum"].to_numpy(), g["count"].to_numpy()
        idx = rng.integers(0, len(S), (n, len(S)))
        bs = S[idx].sum(1) / C[idx].sum(1)
        out[nm] = {"mean": float(S.sum() / C.sum()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    return out


def calib(p, lab, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": lab, "q": q}).groupby("q").mean()
    from sklearn.metrics import roc_auc_score
    return {"max_abs_err_pp": float((t.p - t.y).abs().max() * 100), "mean_abs_err_pp": float((t.p - t.y).abs().mean() * 100),
            "brier": float(np.mean((p - lab) ** 2)), "auc": float(roc_auc_score(lab, p)), "base_rate": float(lab.mean()),
            "mean_p": float(p.mean()), "deciles": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(t.p, t.y)]}


def sig(a, b, x):
    return 1 / (1 + np.exp(-(a + b * x)))


def main(zdir: Path) -> None:
    import joblib
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261020)
    d = load_zips(zdir)
    gap = d.timestamp.diff().max()
    d.to_parquet(OUT / "XRPUSDT_5m.parquet")   # 라이브 --monthly 의 LONG 시작점으로 쓸 수 있는 같은 열
    print(f"XRP 5분봉 {len(d):,}행 {d.timestamp.min()} ~ {d.timestamp.max()} · 최대 간격 {gap}", flush=True)
    assert gap <= pd.Timedelta("1h"), gap
    X, y, r2016, ok, ts = frame(d)
    tr = ok & (ts < TEST0 - np.timedelta64(HOLD * 5, "m"))
    te = ok & (ts >= TEST0)
    per = {"2025": te & (ts < Y26), "2026": te & (ts >= Y26), "2025~": te}

    e = pd.read_parquet(ETH_LONG)
    Xe, ye, _, oke, tse = frame(e)
    tre = oke & (tse < TEST0 - np.timedelta64(HOLD * 5, "m"))
    print(f"학습 XRP {tr.sum():,} · ETH {tre.sum():,} · 표본외 XRP {te.sum():,}", flush=True)

    P = {}
    m_x = fit_gbm(X, y, tr); P["xrp_gbm"] = predict_vol(m_x, X)
    m_e = fit_gbm(Xe, ye, tre); P["eth_gbm_le2024_on_xrp"] = predict_vol(m_e, X)
    Xp, yp = pd.concat([X[tr], Xe[tre]], ignore_index=True), np.concatenate([y[tr], ye[tre]])   # ETH+XRP 합쳐 학습(≤2024)
    P["pooled_eth_xrp_gbm"] = predict_vol(fit_gbm(Xp, yp, np.ones(len(yp), bool)), X)
    art_e = joblib.load(ETH_ART); P["eth_deployed_on_xrp(diag)"] = predict_vol(art_e["models"], X)
    H = har(X, r2016); beta = np.linalg.lstsq(H[tr], np.log(y[tr]), rcond=None)[0]; P["har_rv"] = np.exp(H @ beta)
    P["naive_rv48"] = np.exp(X["rv48"].to_numpy())
    res: dict = {"data": {"rows": int(len(d)), "start": str(d.timestamp.min()), "end": str(d.timestamp.max()), "max_gap": str(gap),
                          "train_rows_xrp": int(tr.sum()), "train_rows_eth": int(tre.sum()), "har_beta": [float(b) for b in beta]},
                 "contract": {"features": FEATURES, "hold_bars": HOLD, "seeds": list(SEEDS), "train_end": "2024-12-31 (purged 4h)",
                              "oos": "2025-01-01 ~", "causal_features": "build_features self-check (ETH script) -- 같은 함수"},
                 "metrics": {}, "diff_vs_xrp_gbm": {}}
    for pn, msk in per.items():
        res["metrics"][pn] = {k: metrics(y[msk], v[msk]) for k, v in P.items()}
        day = ts[msk].astype("datetime64[D]")
        res["diff_vs_xrp_gbm"][pn] = {k: diff_ci(y[msk], P["xrp_gbm"][msk], v[msk], day, rng) for k, v in P.items() if k != "xrp_gbm"}
    # 확대 확률: 2025 로 적합 → 2026 평가 · ETH VOL_P 그대로
    rv48 = np.exp(X["rv48"].to_numpy())
    lab = (y / rv48 >= EXPAND_LABEL).astype(int)
    x_ = np.log(P["xrp_gbm"] / rv48)
    a25 = fit_expand_logit(P["xrp_gbm"][per["2025"]], rv48[per["2025"]], y[per["2025"]])
    a_all = fit_expand_logit(P["xrp_gbm"][te], rv48[te], y[te])
    m26 = per["2026"]
    res["expand"] = {"label": f"y >= {EXPAND_LABEL} * rv48", "logit_fit_2025": a25, "logit_fit_2025~(exported)": a_all,
                     "eval_2026_fit2025": calib(sig(*a25, x_[m26]), lab[m26]),
                     "eval_2026_eth_VOL_P_on_xrp_model": calib(sig(*VOL_P_ETH, x_[m26]), lab[m26]),
                     "eval_2026_eth_VOL_P_on_eth_deployed_model": calib(sig(*VOL_P_ETH, np.log(P["eth_deployed_on_xrp(diag)"][m26] / rv48[m26])), lab[m26]),
                     "eval_2025~_insample_logit": calib(sig(*a_all, x_[te]), lab[te])}
    for nm in ("eth_gbm_le2024_on_xrp", "pooled_eth_xrp_gbm", "eth_deployed_on_xrp(diag)"):   # 다른 판도 XRP 로 다시 맞춘 상수(2025 적합 → 2026)
        a_ = fit_expand_logit(P[nm][per["2025"]], rv48[per["2025"]], y[per["2025"]])
        res["expand"][f"refit2025_eval2026[{nm}]"] = {"logit": a_, **calib(sig(*a_, np.log(P[nm][m26] / rv48[m26])), lab[m26])}
        res["expand"][f"logit_fit_2025~[{nm}]"] = fit_expand_logit(P[nm][te], rv48[te], y[te])
    # 화면 눈금: σ4h % 중앙·분위 (XRP 모델 표본외)
    s4 = P["xrp_gbm"][te] * np.sqrt(HOLD) * 100
    res["sig4h_pct_oos_quantiles"] = {q: float(np.quantile(s4, q)) for q in (.05, .25, .5, .75, .95)}
    s4e = predict_vol(art_e["models"], Xe[oke & (tse >= TEST0)]) * np.sqrt(HOLD) * 100
    res["eth_sig4h_pct_oos_quantiles(deployed, diag)"] = {q: float(np.quantile(s4e, q)) for q in (.05, .25, .5, .75, .95)}

    # 배포용 아티팩트: 전 기간(ETH --monthly 와 같은 구조)
    models = fit_gbm(X, y, ok)
    art = {"models": models, "features": FEATURES, "ref_pred": float(np.median(predict_vol(models, X[ok]))), "seeds": list(SEEDS),
           "train_end": str(pd.Timestamp(ts[ok].max()).date()), "hold_bars": HOLD, "n_train": int(ok.sum()),
           "train_start": str(pd.Timestamp(ts[ok].min()).date()), "expand_logit": [round(a_all[0], 4), round(a_all[1], 4)], "symbol": "XRPUSDT"}
    joblib.dump(art, OUT / "xrp_sizing_vol_model.joblib")
    lc = np.corrcoef(np.log(predict_vol(models, X[te])), np.log(P["xrp_gbm"][te]))[0, 1]
    res["artifact"] = {**{k: v for k, v in art.items() if k != "models"}, "path": str(OUT / "xrp_sizing_vol_model.joblib"),
                       "corr_log_vs_le2024_model_on_oos": float(lc), "VOL_P_xrp": [round(a_all[0], 4), round(a_all[1], 4)]}
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("metrics", "expand", "artifact")}, ensure_ascii=False, indent=1, default=str)[:6000])


def export_pooled() -> None:
    """표본외 1위(ETH+XRP 합쳐 학습: 2025~ 로그 R² .680 · 2026 QLIKE·제곱 로그 오차 모두 XRP 단독보다 CI 0 배제)를 전 기간으로 다시 학습해 배포 구조로.
    expand_logit = 합친 판(≤2024)의 2025~ XRP 표본외 적합값(report.json)."""
    import joblib
    rep = json.loads((OUT / "report.json").read_text())
    a = rep["expand"]["logit_fit_2025~[pooled_eth_xrp_gbm]"]
    X, y, _, ok, ts = frame(pd.read_parquet(OUT / "XRPUSDT_5m.parquet"))
    Xe, ye, _, oke, tse = frame(pd.read_parquet(ETH_LONG))
    Xp, yp = pd.concat([X[ok], Xe[oke]], ignore_index=True), np.concatenate([y[ok], ye[oke]])
    models = fit_gbm(Xp, yp, np.ones(len(yp), bool))
    art = {"models": models, "features": FEATURES, "ref_pred": float(np.median(predict_vol(models, X[ok]))), "seeds": list(SEEDS),
           "train_end": str(pd.Timestamp(ts[ok].max()).date()), "hold_bars": HOLD, "n_train": int(len(yp)),
           "train_start": str(pd.Timestamp(ts[ok].min()).date()), "expand_logit": [round(a[0], 4), round(a[1], 4)],
           "symbol": "XRPUSDT", "train_symbols": ["XRPUSDT", "ETHUSDT"], "eth_train_end": str(pd.Timestamp(tse[oke].max()).date())}
    joblib.dump(art, OUT / "xrp_sizing_vol_model.joblib")
    rep["artifact"] = {**{k: v for k, v in art.items() if k != "models"}, "path": str(OUT / "xrp_sizing_vol_model.joblib"), "VOL_P_xrp": art["expand_logit"],
                       "chosen": "pooled_eth_xrp_gbm (fair OOS best); XRP-only model metrics kept as xrp_gbm"}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    print(rep["artifact"])


def selftest() -> None:
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(-6, .4, 5000)); day = np.repeat(np.arange(50), 100)
    m = metrics(y, y)
    assert abs(m["qlike"]) < 1e-12 and abs(m["r2_logvol"] - 1) < 1e-12
    c = diff_ci(y, y, y * 1.5, day, rng, 200)
    assert c["qlike"]["ci95"][0] > 0, c        # 1.5배 과대 예측은 정답보다 나쁘다
    p = rng.random(20000); lab = (rng.random(20000) < p).astype(int)
    assert calib(p, lab)["max_abs_err_pp"] < 3.5
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else export_pooled() if "--export-pooled" in sys.argv else main(Path(sys.argv[1]))
