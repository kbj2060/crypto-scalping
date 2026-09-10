#!/usr/bin/env python3
"""중기·장기 지평 예측 가능성 스크린 (2026-09-10) -- 사용자: "대시보드 재료는 전부 5분봉 단기다. 중기·장기로 만들 수 있나?
단·중·장기 흐름을 예측하고 싶다."

질문을 셋으로 나눈다.
 Q1 5분 재료를 상위 타임프레임으로 올릴 수 있나 -- (a) 5분 발동을 일 단위로 **집계**(하루/7일 바닥·천장 표수)
    (b) 같은 코드를 **1h/4h 봉에서 재계산**(compute_signals 는 봉 수 기반이라 타임프레임 무관하게 돈다)
 Q2 그 재료 + 일 단위 정보(OI·롱숏비·테이커·DVOL·HAR-RV·펀딩)가 **지평별 방향**을 예측하나 -- 1d/3d/7d/14d/30d
 Q3 같은 재료가 **지평별 변동성 확장**을 예측하나 -- 같은 지평, 앞 RV / 뒤 RV > 1.2

평가: 앵커 = 일봉 종가(UTC). TRAIN ≤2025-08-31 / TEST 2025-09-01~. 표본이 작다(일 ~980, 테스트 ~375일,
H=30 이면 독립 관측 ~12). 그래서 AUC 에 **원형 블록 부트(블록=H일) 95% CI** 를 붙이고, 단변량 최대 AUC 를 같이 낸다.
4h 앵커(n≈5,900)로 중기(1d/3d)를 한 번 더 잰다.
출력 tmp/eth_multi_horizon_20260910/report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_exit_synth_dataset_20260910 as BD  # noqa: E402  (klines/dvol 로더 재사용)
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
import exit_synth_features_20260910 as EF  # noqa: E402

OUT = ROOT / "tmp/eth_multi_horizon_20260910"
SPLIT = "2025-09-01"
HORIZONS_D = [1, 3, 7, 14, 30]
ANN_D = np.sqrt(365) * 100


def log(m):
    print(f"[mh {time.strftime('%H:%M:%S')}] {m}", flush=True)


def resample(kl: pd.DataFrame, rule: str) -> pd.DataFrame:
    kl = kl.copy()
    for c in ("quote_volume", "trades"):
        kl[c] = pd.to_numeric(kl[c], errors="coerce")      # 연장 parquet 은 문자열
    g = kl.set_index("timestamp").resample(rule, label="left", closed="left")
    o = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
               "taker_buy_base": "sum", "quote_volume": "sum", "trades": "sum"}).dropna(subset=["close"])
    return o.reset_index()


def fires_by_bar(sig: pd.DataFrame) -> pd.DataFrame:
    b = np.zeros(len(sig)); t = np.zeros(len(sig))
    for s in EF.SIGNALS:
        b += sig[f"bottom_{s}"].fillna(False).to_numpy(bool); t += sig[f"top_{s}"].fillna(False).to_numpy(bool)
    return pd.DataFrame({"timestamp": pd.to_datetime(sig["timestamp"]), "fb": b, "ft": t})


def block_boot_auc(y, p, block, n=500, seed=0):
    rng = np.random.default_rng(seed); n_ = len(y); out = []
    for _ in range(n):
        starts = rng.integers(0, n_, int(np.ceil(n_ / block)))
        idx = np.concatenate([np.arange(s, s + block) % n_ for s in starts])[:n_]
        if y[idx].min() == y[idx].max():
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def daily_frame(eth, btc, dv, met_df, fires5, fires1h, fires4h) -> pd.DataFrame:
    d = resample(eth, "1D"); db = resample(btc, "1D")[["timestamp", "close"]].rename(columns={"close": "btc"})
    d = d.merge(db, on="timestamp", how="left")
    r5 = np.log(eth["close"]).diff()
    rv_d = (r5.groupby(eth["timestamp"].dt.floor("D")).std() * np.sqrt(288)).rename("rv1d")   # 일 실현변동성(일 단위 σ)
    d = d.merge(rv_d.reset_index().rename(columns={"timestamp": "timestamp"}), on="timestamp", how="left")
    c = np.log(d["close"]); F = pd.DataFrame({"timestamp": d["timestamp"]})
    for k in (1, 3, 7, 14, 30, 90):
        F[f"ret{k}"] = c.diff(k)
    F["btc_ret7"] = np.log(d["btc"]).diff(7); F["btc_ret30"] = np.log(d["btc"]).diff(30)
    F["eth_btc_sp7"] = F["ret7"] - F["btc_ret7"]; F["eth_btc_sp30"] = F["ret30"] - F["btc_ret30"]
    for k in (7, 30, 90):
        F[f"rv{k}"] = d["rv1d"].rolling(k, min_periods=k).mean()
    F["rv_ratio_7_90"] = F["rv7"] / F["rv90"]
    F["dist_hi30"] = c - np.log(d["high"].rolling(30).max()); F["dist_lo30"] = c - np.log(d["low"].rolling(30).min())
    F["dist_hi90"] = c - np.log(d["high"].rolling(90).max()); F["dist_lo90"] = c - np.log(d["low"].rolling(90).min())
    F["taker_ratio1"] = d["taker_buy_base"] / d["volume"].replace(0, np.nan)
    F["taker_ratio7"] = F["taker_ratio1"].rolling(7).mean()
    F["vol_z30"] = (np.log(d["quote_volume"]) - np.log(d["quote_volume"]).rolling(30).mean()) / np.log(d["quote_volume"]).rolling(30).std()
    # DVOL: 일 마지막 시간봉(다음날 0시에 알려짐 = 그날 종가 시점엔 23시 봉까지) -> 그날 23:00 봉 close 사용
    dvd = dv.set_index("timestamp")["dvol"].resample("1D").last().rename("dvol").reset_index()
    F = F.merge(dvd, on="timestamp", how="left")
    F["vrp30"] = F["dvol"] - F["rv30"] * ANN_D; F["dvol_chg7"] = F["dvol"].diff(7)
    # 메트릭(일 마지막 값)
    if met_df is not None:
        md = met_df.set_index("ts").resample("1D").last().reset_index().rename(columns={"ts": "timestamp"})
        F = F.merge(md, on="timestamp", how="left")
        for c_ in ("oi", "ttp", "ttc", "retail", "tkv"):
            F[f"{c_}_d7"] = F[c_].diff(7)
        F["oi_d1"] = F["oi"].diff(1); F["oi_d30"] = F["oi"].diff(30)
    # 5분 증거신호 집계 + 상위 타임프레임 재계산 발동
    for nm, fr in (("f5", fires5), ("f1h", fires1h), ("f4h", fires4h)):
        g = fr.set_index("timestamp")[["fb", "ft"]].resample("1D").sum().reset_index()
        g = g.rename(columns={"fb": f"{nm}_b1", "ft": f"{nm}_t1"})
        F = F.merge(g, on="timestamp", how="left")
        F[f"{nm}_net1"] = F[f"{nm}_b1"] - F[f"{nm}_t1"]
        F[f"{nm}_net7"] = F[f"{nm}_net1"].rolling(7).sum(); F[f"{nm}_net30"] = F[f"{nm}_net1"].rolling(30).sum()
    F["dow"] = F["timestamp"].dt.weekday
    F["_close"] = c; F["_rv1d"] = d["rv1d"]
    return F


def targets(F: pd.DataFrame, H: int) -> tuple[np.ndarray, np.ndarray]:
    c = F["_close"]; fwd = c.shift(-H) - c
    y_dir = (fwd > 0).astype(float); y_dir[fwd.isna()] = np.nan
    rv_fwd = F["_rv1d"].shift(-1).rolling(H).mean().shift(-(H - 1))
    rv_back = F["_rv1d"].rolling(max(H, 2), min_periods=max(2, H // 2)).mean()
    y_vol = (rv_fwd / rv_back > 1.2).astype(float); y_vol[(rv_fwd.isna()) | (rv_back.isna())] = np.nan
    return y_dir.to_numpy(), y_vol.to_numpy()


def evaluate(F, cols, y, H, split, label):
    ok = np.isfinite(y)
    X = F[cols].to_numpy(float)
    tr = ok & (F["timestamp"] < split).to_numpy(); te = ok & (F["timestamp"] >= split).to_numpy()
    # 라벨이 split 을 넘어보는 TRAIN 행은 제외(퍼지)
    tr &= (F["timestamp"] < pd.Timestamp(split) - pd.Timedelta(days=H)).to_numpy()
    if tr.sum() < 100 or te.sum() < 30 or len(np.unique(y[te])) < 2:
        return None
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "n_test_independent": int(te.sum() // H), "base_rate_test": float(y[te].mean())}
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[tr], y[tr])
    hg = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=2.0).fit(X[tr], y[tr])
    for nm, m in (("logit", lr), ("hgb", hg)):
        p = m.predict_proba(X[te])[:, 1]
        res[nm] = {"auc": float(roc_auc_score(y[te], p)), "ci95_block": block_boot_auc(y[te], p, block=max(H, 2))}
    # 단변량 최대(테스트) -- 방향 부호 무관
    best = ("", 0.5)
    for j, c_ in enumerate(cols):
        x = X[te, j]; m_ = np.isfinite(x)
        if m_.sum() < 30 or len(np.unique(y[te][m_])) < 2:
            continue
        a = roc_auc_score(y[te][m_], x[m_]); a = max(a, 1 - a)
        if a > best[1]:
            best = (c_, float(a))
    res["univariate_max_test"] = best
    log(f"{label:22s} H={H:2d}d  logit {res['logit']['auc']:.3f} {res['logit']['ci95_block']}  hgb {res['hgb']['auc']:.3f}  "
        f"단변량 {best[0]} {best[1]:.3f}  n_test {res['n_test']} (독립≈{res['n_test_independent']})")
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    eth = BD.load_klines("eth", "ETHUSDT"); btc = BD.load_klines("btc", "BTCUSDT"); dv = BD.load_dvol()
    m = pd.read_csv(ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv", parse_dates=["create_time"])
    met_df = pd.DataFrame({"ts": m["create_time"], "oi": np.log(m["sum_open_interest_value"]),
                           "ttp": np.log(m["sum_toptrader_long_short_ratio"]), "ttc": np.log(m["count_toptrader_long_short_ratio"]),
                           "retail": np.log(m["count_long_short_ratio"]), "tkv": np.log(m["sum_taker_long_short_vol_ratio"])})
    # Q1: 5분 발동 집계 + 1h/4h 재계산
    sig5 = compute_signals(eth, btc_df=btc, funding_df=None); f5 = fires_by_bar(sig5)
    e1, b1 = resample(eth, "1h"), resample(btc, "1h"); f1h = fires_by_bar(compute_signals(e1, btc_df=b1, funding_df=None))
    e4, b4 = resample(eth, "4h"), resample(btc, "4h"); f4h = fires_by_bar(compute_signals(e4, btc_df=b4, funding_df=None))
    q1 = {tf: {"bars": int(len(fr)), "fires_per_bar_bottom": float(fr.fb.mean()), "fires_per_bar_top": float(fr.ft.mean()),
               "fires_per_day": float((fr.fb + fr.ft).sum() / max(1, (fr.timestamp.max() - fr.timestamp.min()).days))}
          for tf, fr in (("5m", f5), ("1h", f1h), ("4h", f4h))}
    log(f"Q1 발동률: {json.dumps(q1)}")
    F = daily_frame(eth, btc, dv, met_df, f5, f1h, f4h)
    F = F[F["timestamp"] >= "2024-04-01"].reset_index(drop=True)          # 90일 워밍업
    groups = {
        "price_only": [c for c in F.columns if c.startswith(("ret", "btc_ret", "eth_btc", "rv", "dist_", "taker_ratio", "vol_z", "dow"))],
        "deriv_metrics": ["oi", "ttp", "ttc", "retail", "tkv", "oi_d1", "oi_d7", "oi_d30", "ttp_d7", "ttc_d7", "retail_d7", "tkv_d7", "dvol", "vrp30", "dvol_chg7"],
        "evidence_5m_aggregated": [c for c in F.columns if c.startswith("f5_")],
        "evidence_1h_4h_recomputed": [c for c in F.columns if c.startswith(("f1h_", "f4h_"))],
    }
    groups["all"] = sum(groups.values(), [])
    rep = {"q1_fire_rates": q1, "split": SPLIT, "n_days": int(len(F)), "span": [str(F.timestamp.iloc[0]), str(F.timestamp.iloc[-1])],
           "direction": {}, "vol_expansion": {}}
    for H in HORIZONS_D:
        yd, yv = targets(F, H)
        rep["direction"][H] = {g: evaluate(F, cols, yd, H, SPLIT, f"dir/{g}") for g, cols in groups.items()}
        rep["vol_expansion"][H] = {g: evaluate(F, cols, yv, H, SPLIT, f"vol/{g}") for g, cols in groups.items()}
    # 4h 앵커로 중기(6·18·42봉 = 1·3·7일) 방향 한 번 더 -- 표본 6배
    c4 = np.log(e4["close"]); F4 = pd.DataFrame({"timestamp": e4["timestamp"], "_close": c4})
    for k in (1, 6, 18, 42, 180):
        F4[f"ret{k}"] = c4.diff(k)
    r4 = c4.diff(); F4["rv42"] = r4.rolling(42).std(); F4["rv180"] = r4.rolling(180).std(); F4["rv_ratio"] = F4["rv42"] / F4["rv180"]
    F4["dist_hi180"] = c4 - np.log(e4["high"].rolling(180).max()); F4["dist_lo180"] = c4 - np.log(e4["low"].rolling(180).min())
    F4 = F4.merge(f4h.rename(columns={"fb": "f4_b", "ft": "f4_t"}), on="timestamp", how="left")
    F4["f4_net6"] = (F4["f4_b"] - F4["f4_t"]).rolling(6).sum(); F4["f4_net42"] = (F4["f4_b"] - F4["f4_t"]).rolling(42).sum()
    F4["_rv1d"] = r4.abs()
    F4 = F4[F4["timestamp"] >= "2024-04-01"].reset_index(drop=True)
    cols4 = [c for c in F4.columns if not c.startswith("_") and c != "timestamp"]
    rep["direction_4h_anchor"] = {}
    for Hb, nm in ((6, "1d"), (18, "3d"), (42, "7d")):
        fwd = F4["_close"].shift(-Hb) - F4["_close"]; y = (fwd > 0).astype(float); y[fwd.isna()] = np.nan
        rep["direction_4h_anchor"][nm] = evaluate(F4, cols4, y.to_numpy(), Hb, SPLIT, f"dir4h/{nm}")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"→ {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
