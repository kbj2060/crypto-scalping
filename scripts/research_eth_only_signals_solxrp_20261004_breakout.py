#!/usr/bin/env python3
"""대시보드 «추세 전환 탐지기·경보기»(ETH 전용) → SOL·XRP 5분봉 같은 규약 검정 (2026-10-04).

정의는 09-30 감사 replay.py(tmp/bd_verify_20260930) 그대로:
  탐지기  z288(거래대금·체결건수) ≥ 후행 2016봉 q90(현재 봉 포함, 라이브 _thr_all_at 규약) AND
  경보기  F33 피쳐 → HGB 5시드 평균 → 후행 2016 q90(shift 1) 이상이면 발동(커버 10%)
  타깃    (t, t+30분] 안에 탐지 발동
  대조    경보기 ← «z_qv288 + z_n288» 합을 같은 규약(후행 2016 q90 shift 1)으로 발동
          탐지기 ← 같은 발동률의 «봉 가격폭 상위»(후행 2016 분위) — 앞 1h 최대이탈 비교
사전등록(분석 전 고정):
  경보기 통과 = 주 구간에서 정밀도(모델) − 정밀도(z합 대조) > 0, 일 블록 부트스트랩 95% CI 하한 > 0
  탐지기 통과 = 주 구간에서 앞 1h 최대이탈(발동) − (봉폭 상위 대조) > 0, CI 하한 > 0
  주 구간 = 2025-01-01~ (재학습 경보기는 학습 밖인 2025-09-01~). 2022~24 는 보조.
  코인 판정 pass = 경보기(전이·재학습 중 하나라도) 통과 AND 탐지기 통과 → 둘 다 의미가 있을 때만 화면에 올릴 근거.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eth_breakout_features33_20260911 as F33  # noqa: E402

DATA = Path("/home/kbj20/crypto-scalping/data")
BDV = Path("/home/kbj20/crypto-scalping/.claude/worktrees/options-sentiment-analysis-a1b431/tmp/bd_verify_20260930")
OUT = ROOT / "tmp" / "eth_only_signals_solxrp"
QWIN, H, B = 2016, 6, 2000
SEEDS = (30474, 663233, 730273, 425331, 154778)          # 빌더와 같은 시드
COLS = ["timestamp", "open", "high", "low", "close", "quote_volume", "trades", "taker_buy_quote"]


def trail_q(x, q, shift=0):
    t = pd.Series(x).rolling(QWIN, min_periods=200).quantile(q)
    return (t.shift(shift) if shift else t).to_numpy()


def zcol(d, col, w=288):
    s = d[col].astype(float)
    return ((s - s.rolling(w).mean()) / s.rolling(w).std()).to_numpy()


def load(coin: str) -> pd.DataFrame:
    parts = [pd.read_csv(DATA / f"{coin}_5m_{y}.csv", usecols=COLS, parse_dates=["timestamp"])
             for y in ("2022_2023", "2024_2026")]
    for f in sorted(glob.glob(str(OUT / "vision" / f"{coin.upper()}USDT-5m-*.csv"))):
        v = pd.read_csv(f)
        v = v[pd.to_numeric(v.open_time, errors="coerce").notna()]
        parts.append(pd.DataFrame({"timestamp": pd.to_datetime(v.open_time.astype("int64"), unit="ms"),
                                   "open": v.open, "high": v.high, "low": v.low, "close": v.close,
                                   "quote_volume": v.quote_volume, "trades": v["count"],
                                   "taker_buy_quote": v.taker_buy_quote_volume}).astype({c: float for c in COLS[1:]}))
    d = pd.concat(parts).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    return d


def detector(d):
    zq, zn = zcol(d, "quote_volume"), zcol(d, "trades")
    fire = np.isfinite(zq) & np.isfinite(zn) & (zq >= trail_q(zq, .9)) & (zn >= trail_q(zn, .9))
    return fire, zq, zn


def fut_any(fire):
    """(t, t+H] 안에 발동 — 자기 봉 제외. 끝 H봉은 NaN."""
    w = np.lib.stride_tricks.sliding_window_view(fire[1:].astype(float), H).max(axis=1)
    out = np.full(len(fire), np.nan); out[:len(w)] = w
    return out


def fwd_exc(d, k=12):
    c, h, l = (d[x].to_numpy(float) for x in ("close", "high", "low"))
    hw = np.lib.stride_tricks.sliding_window_view(h[1:], k).max(axis=1)
    lw = np.lib.stride_tricks.sliding_window_view(l[1:], k).min(axis=1)
    exc = np.full(len(d), np.nan); ret = np.full(len(d), np.nan); n = len(hw)   # 끝 k봉 NaN
    exc[:n] = np.maximum(hw[:n] / c[:n] - 1, 1 - lw[:n] / c[:n]) * 1e4
    ret[:n] = (c[k:k + n] / c[:n] - 1) * 1e4
    return exc, ret


def boot(day, num_a, den_a, num_b, den_b, seed=0):
    """일 블록 부트스트랩: sum(num_a)/sum(den_a) − sum(num_b)/sum(den_b) 의 점추정·95% CI."""
    g = pd.DataFrame({"d": day, "na": num_a, "da": den_a, "nb": num_b, "db": den_b}).groupby("d").sum().to_numpy()
    est = g[:, 0].sum() / g[:, 1].sum() - g[:, 2].sum() / g[:, 3].sum()
    idx = np.random.default_rng(seed).integers(0, len(g), (B, len(g)))
    s = g[idx].sum(axis=1)
    bs = s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return round(float(est), 4), [round(float(lo), 4), round(float(hi), 4)], int(len(g))


def ens_score(models, feats, F):
    X = F[feats].to_numpy(np.float32); good = np.isfinite(X).all(axis=1)
    sc = np.full(len(F), np.nan)
    sc[good] = np.mean([m.predict_proba(X[good])[:, 1] for m in models], axis=0)
    return sc


def prewarn_eval(ts, mask_base, fut, sc, zsum, periods):
    pw = np.isfinite(sc) & (sc >= trail_q(sc, .9, 1))
    zs = np.where(np.isfinite(sc), zsum, np.nan)                    # 같은 봉 집합에서 대조
    cw = np.isfinite(zs) & (zs >= trail_q(zs, .9, 1))
    day = ts.dt.floor("D").to_numpy()
    res = {}
    for name, (a, b) in periods.items():
        m = mask_base & np.isfinite(fut) & np.isfinite(sc) & np.isfinite(zs)
        m &= ((ts >= a) if a else True) & ((ts <= b) if b else True)
        m = np.asarray(m); y = fut[m]
        if m.sum() < 500:
            continue
        est, ci, nd = boot(day[m], y * pw[m], pw[m], y * cw[m], cw[m])
        res[name] = {"bars": int(m.sum()), "days": nd, "base_rate": round(float(y.mean()), 4),
                     "auc_model": round(roc_auc_score(y, sc[m]), 4), "auc_zsum": round(roc_auc_score(y, zs[m]), 4),
                     "cov_model": round(float(pw[m].mean()), 4), "prec_model": round(float(y[pw[m]].mean()), 4),
                     "cov_ctrl": round(float(cw[m].mean()), 4), "prec_ctrl": round(float(y[cw[m]].mean()), 4),
                     "diff": est, "diff_ci95": ci}
    return res


def detector_eval(d, ts, warm, fire, periods):
    exc, ret = fwd_exc(d)
    c = d.close.to_numpy(float)
    rng = (d.high.to_numpy(float) - d.low.to_numpy(float)) / c * 1e4
    barret = np.r_[np.nan, np.diff(np.log(c))] * 1e4
    day = ts.dt.floor("D").to_numpy()
    res = {}
    for name, (a, b) in periods.items():
        m = np.asarray(warm & np.isfinite(exc) & ((ts >= a) if a else True) & ((ts <= b) if b else True))
        if m.sum() < 500:
            continue
        rate = fire[m].mean()
        base = m & (rng >= trail_q(rng, 1 - rate))                   # 같은 발동률의 봉폭 상위(후행 분위)
        f = m & fire
        ex = np.nan_to_num(exc)
        est, ci, nd = boot(day[m], (ex * f)[m], f[m], (ex * base)[m], base[m])
        cont = lambda s: float((np.sign(barret[s]) == np.sign(ret[s])).mean())
        res[name] = {"days": nd, "fires_per_day": round(float(rate * 288), 1), "ctrl_per_day": round(float(base[m].mean() * 288), 1),
                     "exc1h_fire_bp": round(float(np.nanmean(exc[f])), 1), "exc1h_ctrl_bp": round(float(np.nanmean(exc[base])), 1),
                     "exc1h_all_bp": round(float(np.nanmean(exc[m])), 1), "diff_bp": round(est, 1), "diff_ci95": [round(x, 1) for x in ci],
                     "cont_fire": round(cont(f), 4), "cont_ctrl": round(cont(base), 4)}
    return res


def retrain(d, F, out: Path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    # 빌더 detector_fires 그대로(라벨용: 후행 분위 shift 1)
    fire = np.ones(len(F), bool)
    for col in ("qv", "n"):
        x = F[f"z_{col}_288"].to_numpy(float)
        thr = trail_q(x, .9, 1)
        fire &= np.isfinite(x) & np.isfinite(thr) & (x >= thr)
    fwd = pd.Series(fire[::-1]).rolling(H, min_periods=1).max()[::-1].to_numpy().astype(bool)
    y = np.r_[fwd[1:], False]
    X = F.to_numpy(np.float32)
    tr = np.flatnonzero(np.isfinite(X).all(axis=1) & (d.timestamp <= "2025-08-31").to_numpy())[:-2 * H]
    out.mkdir(parents=True, exist_ok=True)
    models = []
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.15, random_state=sd).fit(X[tr], y[tr])
        joblib.dump(m, out / f"hgb_{sd}.joblib", compress=3); models.append(m)
    (out / "meta.json").write_text(json.dumps({"features": list(F.columns), "seeds": list(SEEDS), "qwin": QWIN,
                                               "fire_quantile": .9, "train_end": "2025-08-31", "train_rows": int(len(tr)),
                                               "base_rate_train": float(y[tr].mean())}, ensure_ascii=False, indent=2))
    return models, int(len(tr))


def REASON(det, tr_, rt):
    a, b, c = det["주 2025-01~"], tr_["주 2025-01~"], rt["주(학습밖) 2025-09~"]
    return (f"탐지기 1h 이탈 발동−봉폭대조 {a['diff_bp']:+.1f}bp CI {a['diff_ci95']} → {'통과' if a['diff_ci95'][0] > 0 else '불통과'} · "
            f"경보기 전이 정밀도 차 {b['diff']*100:+.2f}pp CI {[round(x*100, 2) for x in b['diff_ci95']]} · "
            f"재학습(25-09~) {c['diff']*100:+.2f}pp CI {[round(x*100, 2) for x in c['diff_ci95']]} → 경보기 통과. "
            "경보기는 탐지기를 맞히는 모델이라 탐지기 불통과면 화면 근거 없음(pass=false).")


def self_check():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000); t = 3000
    full, cut = trail_q(x, .9), trail_q(x[:t + 1], .9)
    assert abs(full[t] - cut[t]) < 1e-12, "후행 분위가 미래를 본다"
    f = np.zeros(20, bool); f[10] = True
    fa = fut_any(f)
    assert fa[10] == 0 and fa[4] == 1 and fa[9] == 1 and fa[3] == 0, "타깃 창이 (t, t+6] 가 아니다"
    print("self-check OK")


def main():
    self_check()
    A = BDV / "artifact"; meta = json.load(open(A / "meta.json"))
    eth_models = [joblib.load(A / f"hgb_{s}.joblib") for s in meta["seeds"]]
    result = {}

    # ── 1. 양성 대조: ETH (09-30 감사 데이터·구간 그대로)
    d = pd.read_parquet(BDV / "eth5m_full.parquet").sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    ts = d.timestamp; warm = np.arange(len(d)) >= QWIN + 300
    fire, zq, zn = detector(d); fut = fut_any(fire)
    sc = ens_score(eth_models, meta["features"], F33.build_features(d))
    P_eth = {"TRAIN≤25-08": (None, "2025-08-31 23:59"), "VAL 25-09~12": ("2025-09-01", "2025-12-31 23:59"),
             "OOS 26-01~03": ("2026-01-01", "2026-03-31 23:59"), "FWD 26-04~09-10": ("2026-04-01", "2026-09-10 23:59"),
             "배포후 09-11~09-28": ("2026-09-11", "2026-09-28 23:59")}
    zsum = np.nan_to_num(zq) + np.nan_to_num(zn)
    result["eth_control"] = {"prewarn": prewarn_eval(ts, warm, fut, sc, zsum, P_eth),
                             "detector": detector_eval(d, ts, warm, fire, P_eth)}
    print(json.dumps(result["eth_control"], ensure_ascii=False, indent=1), flush=True)

    # ── 2·3. SOL·XRP
    P = {"보조 2022~24": (None, "2024-12-31 23:59"), "25-01~08": ("2025-01-01", "2025-08-31 23:59"),
         "VAL 25-09~12": ("2025-09-01", "2025-12-31 23:59"), "OOS 26-01~03": ("2026-01-01", "2026-03-31 23:59"),
         "FWD 26-04~09-10": ("2026-04-01", "2026-09-10 23:59"), "배포후 09-11~10-03": ("2026-09-11", None),
         "주 2025-01~": ("2025-01-01", None), "주(학습밖) 2025-09~": ("2025-09-01", None)}
    result["constants_if_applied"] = {}
    for coin in ("sol", "xrp"):
        d = load(coin); ts = d.timestamp; warm = np.arange(len(d)) >= QWIN + 300
        gaps = int((ts.diff().dt.total_seconds().iloc[1:] != 300).sum())
        F = F33.build_features(d)
        fire, zq, zn = detector(d); fut = fut_any(fire)
        zsum = np.nan_to_num(zq) + np.nan_to_num(zn)
        det = detector_eval(d, ts, warm, fire, P)
        tr_ = prewarn_eval(ts, warm, fut, ens_score(eth_models, meta["features"], F), zsum, P)
        models, ntr = retrain(d, F, OUT / f"artifact_{coin}")
        rt = prewarn_eval(ts, warm, fut, ens_score(models, list(F.columns), F), zsum, P)
        ok = lambda r: r["diff_ci95"][0] > 0
        det_pass = ok(det["주 2025-01~"]); tr_pass = ok(tr_["주 2025-01~"]); rt_pass = ok(rt["주(학습밖) 2025-09~"])
        result[coin] = {"data": {"bars": len(d), "from": str(ts.iloc[0]), "to": str(ts.iloc[-1]), "gaps": gaps, "train_rows": ntr},
                        "detector": det, "prewarn_transfer": tr_, "prewarn_retrain": rt,
                        "verdict": {"detector_pass": det_pass, "prewarn_transfer_pass": tr_pass, "prewarn_retrain_pass": rt_pass},
                        "pass": bool(det_pass and (tr_pass or rt_pass)),
                        "reason": REASON(det, tr_, rt)}
        zr = np.asarray(ts >= "2025-01-01")
        result["constants_if_applied"][coin] = {
            "detector": "ETH 와 같은 규칙 — z288(qv,n) ≥ 후행 2016 q90(현재 봉 포함) AND · 코인별 상수 없음",
            "z288_q90_thr_median_2025on": {"qv": round(float(np.nanmedian(trail_q(zq, .9)[zr])), 3),
                                           "n": round(float(np.nanmedian(trail_q(zn, .9)[zr])), 3)},
            "prewarn_model_dir": str(OUT / f"artifact_{coin}"), "prewarn_fire": "후행 2016 q90 shift 1"}
        print(coin, json.dumps(result[coin]["verdict"]), flush=True)
    (OUT / "breakout.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    print("저장", OUT / "breakout.json")


if __name__ == "__main__":
    main()
