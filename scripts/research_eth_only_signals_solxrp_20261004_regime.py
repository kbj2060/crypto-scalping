#!/usr/bin/env python3
"""SOL 레짐 리본 — XRP(S96_K9) 파이프라인을 SOL 에 적용할 수 있나 + «모델 vs close>SMA288 한 줄» (2026-10-04).

  python scripts/research_eth_only_signals_solxrp_20261004_regime.py build XRPUSDT   # 원시 프레임 → FeatureEngineer → 캐시
  python scripts/research_eth_only_signals_solxrp_20261004_regime.py build SOLUSDT
  python scripts/research_eth_only_signals_solxrp_20261004_regime.py run             # 워크포워드 + 측정 → regime_sol.json
  python scripts/research_eth_only_signals_solxrp_20261004_regime.py selftest

파이프라인 = XRP 배포본(삭제된 build_xrp_raw_frame / build_xrp_regime_s96k9_model, f2eb2377^) 그대로:
교차 슬롯(_btc)에 BTC · 피처 열·중앙값 = ETH GBM3 페이로드 · 라벨 S96_K9(학습창 분위 임계) · HGB seed 7529.
배포 아티팩트(학습 2024-01~2026-06)는 로컬에 없고 있어도 2025~ 가 표본 안이라, 같은 레시피를 분기마다
확장창으로 다시 적합해 2025-01~ 를 표본 밖으로 재생한다.

측정 = 메모리 eth_czz_trend_nowcast_loses_to_sma_control_20260922 와 같은 규약:
매일 23:55 봉 종가에서 상태를 읽고 그 종가로 진입, 다음 봉부터 288봉 TP1.5%/SL0.7%(같은 봉 동시 = 손절),
겹치지 않는 24h 창 · 월 블록 부트스트랩 95% CI. 순−역 = (상태 방향 거래 EV) − (상태 반대 방향 거래 EV).
데이터 = 로컬 CSV/zip 만 (REST 호출 없음).
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/kbj20/crypto-scalping")
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT_DIR = ROOT / "tmp/eth_only_signals_solxrp"
CACHE = OUT_DIR / "regime_cache"
GBM3 = DATA / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"
CLEAN_PRED = {"BTCUSDT": DATA / "tmp/btc_regime_s24k3_clean_20260902/predictions.parquet",
              "ETHUSDT": DATA / "tmp/eth_regime_s12k3_clean_20260902/predictions.parquet"}
GBM3_HP = dict(max_depth=10, learning_rate=0.04, max_iter=400, l2_regularization=2.0)
SEED, SCALE, K = 7529, 96, 9
FIT_START = pd.Timestamp("2024-01-01")
REFITS = [pd.Timestamp(s) for s in ("2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01",
                                    "2026-01-01", "2026-04-01", "2026-07-01")]
RECENT = pd.Timestamp("2025-01-01")
CLEAN_OOS = pd.Timestamp("2025-09-01")        # BTC/ETH clean 모델 학습 종료 다음날
TP, SL, H, SMA_N, B = 0.015, 0.007, 288, 288, 4000


def log(m): print(f"[regime-sol] {m}", flush=True)


PIPELINE = {
    "steps": [
        {"step": "1 원시 프레임", "script": "build_xrp_raw_frame_20260903.py (f2eb2377 에서 삭제, 이 파일 build() 로 포팅)",
         "inputs": "binance_data/klines/{SYM}/{SYM}-5m-api.csv · metrics/{SYM}-metrics-*.zip(일별, OI·top/acct 롱숏비) · "
                   "funding_rate_other/{SYM}-fundingRate-*.zip(월별) · BTCUSDT 5m klines(교차 슬롯)",
         "note": "첫 완전 커버 봉부터(펀딩 zip 이 2024-01 부터라 2024-01-01). metrics 는 create_time 그대로(배포본 파리티)"},
        {"step": "2 피처", "script": "build_xrp_features_20260903.py → FeatureEngineer().process(subj, cross) + _with_raw_state12",
         "trap": "교차 컬럼명은 close_btc/volume_btc/quote_volume_btc 하드코딩. ETH=BTC · BTC=ETH · XRP=BTC · SOL=BTC(2026-07-07 파일럿과 같음)"},
        {"step": "3 라벨", "script": "research_xrp_regime_extended_label_phase3_clean_20260903.make_label",
         "rule": "S96_K9: ER(96)/ER(192) 임계를 학습창에서 RegimeEngine 발동률(ER24≥.20, ER48≥.16)에 분위 맞춤 → "
                 "추세 & net192 부호 & EWM96 기울기 부호 → bull/bear, 나머지 chop → 9봉 연속 확인 디바운스",
         "causal": "라벨 자체가 과거 종가만 쓰는 규칙(미래 참조 0) — 라이브에서 모델 없이도 계산 가능"},
        {"step": "4 학습", "script": "build_xrp_regime_s96k9_model_20260903.py",
         "spec": "HGB(max_depth=10, lr=.04, max_iter=400, l2=2.0, seed 7529) · 피처 136열·결측 중앙값 = ETH GBM3 페이로드",
         "deployed_train": "2024-01-01 ~ 2026-06-30 (2025~ 가 표본 안)"},
        {"step": "5 라이브", "script": "scripts/live_regime_xrp_signal_20260903.py",
         "note": "서버 REST(klines·OI·롱숏비·펀딩) → 같은 FeatureEngineer → model.joblib → bull/bear/chop 확률 · 152봉 이력"}],
    "artifact_local": "tmp/xrp_regime_s96k9_20260903/model.joblib 은 로컬에 없다(서버에만). 스크립트 1~4 는 f2eb2377 에서 삭제, git 에서 복원 가능",
    "parity_check": "같은 레시피로 2024-01~2026-03 적합 → 2026-04~06 예측 bal_acc 0.8444 (배포 문서 확인창 0.8439)",
    "sol_data": {"klines_5m": "2021-12-01 ~ 2026-08-04 03:25 (로컬, XRP 와 같은 끝)", "metrics": "일별 zip ~2026-08-31",
                 "funding": "2024-01 ~ 2026-07 월별 zip", "frame_rows": 272490, "missing_feature_cols": 0},
    "runtime_measured_local": {"features_per_asset_min": "3.5~3.9 (FeatureEngineer 272,490봉)",
                               "single_fit_262k_bars_sec": 40, "walk_forward_7_fits_min": 3,
                               "sol_full_pipeline_min": "≈5 (피처 3.5 + 적합 0.7)"},
    "grid_research_estimate": "S·K 격자 재탐색(30라벨 × Phase3 적합 ~25초 ≈ 13분 + Phase2/3b 증거신호·피벗 SOL 판 신규 생성) ≈ 1시간 안팎 — 안 함",
}

CONSTANTS = {
    "if_sma288_ribbon": {"SYMBOL": "SOLUSDT", "SMA_N": 288, "bar": "5m", "state": "sign(close − SMA288), 봉 종가 기준",
                         "note": "디바운스/기권 밴드는 SOL 에서 안 쟀다(ETH 화면은 SMA144±1·ATR 히스테리시스)"},
    "if_model_ribbon": {"SYMBOL": "SOLUSDT", "CROSS_SYMBOL": "BTCUSDT", "SCALE": 96, "DEBOUNCE_K": 9, "SEED": 7529,
                        "GBM3_HP": GBM3_HP, "feature_cols": "tmp/eth_regime_gbm3_independent_20260826/model.joblib",
                        "TRAIN": "2024-01-01 ~ 직전 완결 월", "MODEL_PATH": "tmp/sol_regime_s96k9_<date>/model.joblib",
                        "HISTORY_BARS_RETURNED": 152, "live_script": "live_regime_xrp_signal_20260903.py 의 SYMBOL·MODEL_PATH 만 교체"},
}


# ── 1. 원시 프레임 + 피처 (build_xrp_raw_frame_20260903 포팅) ─────────────────────────────
def _zips(pattern: str, col: str, unit=None) -> pd.DataFrame:
    fr = []
    for p in sorted((DATA / "binance_data").glob(pattern)):
        if "metrics" in p.name and p.name.split("-metrics-")[1][:7] < "2023-12":
            continue
        with zipfile.ZipFile(p) as z, z.open(z.namelist()[0]) as f:
            fr.append(pd.read_csv(f))
    d = pd.concat(fr, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d[col], unit=unit)
    return d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _klines(sym: str) -> pd.DataFrame:
    k = pd.read_csv(DATA / f"binance_data/klines/{sym}/{sym}-5m-api.csv", low_memory=False)
    k["timestamp"] = pd.to_datetime(k["timestamp"])
    return k.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def build(sym: str) -> None:
    import joblib
    from features.engineering import FeatureEngineer
    from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12
    t0 = time.time()
    met = _zips(f"metrics/{sym}-metrics-*.zip", "create_time")[
        ["timestamp", "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]]
    fund = _zips(f"funding_rate_other/{sym}-fundingRate-*.zip", "calc_time", "ms")[["timestamp", "last_funding_rate"]]
    cross = _klines("BTCUSDT")[["timestamp", "close", "volume", "quote_volume"]].rename(
        columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"})
    # ponytail: metrics 는 배포본처럼 create_time 그대로 붙인다(2024-03-04 이후 한 봉 이른 OI). 24h 지평엔 무시 가능.
    f = _klines(sym).merge(met, on="timestamp", how="left")
    f = pd.merge_asof(f, fund, on="timestamp", direction="backward")
    f = f.merge(cross, on="timestamp", how="left")
    ff = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
          "close_btc", "volume_btc", "quote_volume_btc"]
    f[ff] = f[ff].ffill()
    subj = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "sum_open_interest_value", "sum_toptrader_long_short_ratio",
            "count_long_short_ratio", "last_funding_rate"]
    cc = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]
    ok = f[subj + cc[1:]].notna().all(axis=1)
    f = f.loc[ok.idxmax():].reset_index(drop=True)
    log(f"{sym} raw {len(f):,}행 {f.timestamp.iloc[0]} ~ {f.timestamp.iloc[-1]} ({time.time()-t0:.0f}s)")
    feats = _with_raw_state12(FeatureEngineer().process(f[subj].copy(), f[cc].copy()))
    cols = joblib.load(GBM3)["feature_cols"]
    keep = ["timestamp", "open", "high", "low", "close"] + [c for c in cols if c not in ("timestamp",)]
    miss = [c for c in cols if c not in feats.columns]
    out = feats[[c for c in keep if c in feats.columns]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE / f"{sym}_features.parquet", index=False)
    log(f"{sym} features {out.shape} 결측열 {miss} -> 캐시 ({time.time()-t0:.0f}s)")


# ── 2. 라벨·워크포워드 (research_xrp_regime_extended_label_phase3_clean.make_label 포팅) ──
# research_eth_regime_scalping_label_geometry_20260902 의 세 함수 원문 복사 -- 그 모듈은 import 때
# ROOT/tmp 의 GBM3 를 읽어 워크트리에서 죽는다(아티팩트는 메인 체크아웃에만 있다).
def _debounce(raw: np.ndarray, k_bars: int) -> np.ndarray:
    n = len(raw)
    confirmed = np.empty(n, dtype=int)
    confirmed[0] = raw[0]
    candidate, streak = raw[0], 0
    for t in range(1, n):
        if raw[t] == confirmed[t - 1]:
            candidate, streak = confirmed[t - 1], 0
        else:
            streak = streak + 1 if raw[t] == candidate else 1
            candidate = raw[t]
        confirmed[t] = candidate if streak >= k_bars else confirmed[t - 1]
    return confirmed


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    diff_abs = close.diff().abs()
    net = (close - close.shift(n)).abs()
    return (net / (diff_abs.rolling(n, min_periods=max(2, n // 6)).sum() + 1e-12)).fillna(0.0)


def scaled_label(close: pd.Series, s: int, t1: float, t2: float) -> np.ndarray:
    er_s, er_2s = efficiency_ratio(close, s), efficiency_ratio(close, 2 * s)
    net_2s = close - close.shift(2 * s)
    slope = close.ewm(span=s, adjust=False).mean().pct_change().fillna(0.0)
    trend = (er_s >= t1) | (er_2s >= t2)
    y = np.full(len(close), 2, dtype=int)
    y[(trend & (net_2s > 0) & (slope > 0)).to_numpy()] = 0
    y[(trend & (net_2s < 0) & (slope < 0)).to_numpy()] = 1
    return y


def make_label(close: pd.Series, fit: np.ndarray, s: int = SCALE, k: int = K) -> np.ndarray:
    c = close[fit]
    r1 = float((efficiency_ratio(c, 24) >= 0.20).mean())
    r2 = float((efficiency_ratio(c, 48) >= 0.16).mean())
    t1 = float(efficiency_ratio(c, s).quantile(1.0 - r1))
    t2 = float(efficiency_ratio(c, 2 * s).quantile(1.0 - r2))
    y0 = scaled_label(close, s, t1, t2)
    return y0 if k == 1 else _debounce(y0, k)


def walk_forward(sym: str) -> tuple[pd.DataFrame, list[dict]]:
    """분기마다 [2024-01-01, refit) 로 적합 → [refit, 다음 refit) 예측. 라벨은 과거만 보는 규칙이라 퍼지 불필요."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    cp = CACHE / f"{sym}_wf_pred.parquet"
    if cp.exists():
        return pd.read_parquet(cp), json.loads((CACHE / f"{sym}_wf_meta.json").read_text())
    pay = joblib.load(GBM3)
    cols, med = pay["feature_cols"], pay["feature_medians"]
    df = pd.read_parquet(CACHE / f"{sym}_features.parquet")
    df = df[df.timestamp >= FIT_START].reset_index(drop=True)
    x = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(pd.Series(med)).fillna(0.0)
    ts = df.timestamp
    pred = np.full(len(df), -1)
    rule = np.full(len(df), -1)
    meta = []
    for i, r0 in enumerate(REFITS):
        r1 = REFITS[i + 1] if i + 1 < len(REFITS) else ts.iloc[-1] + pd.Timedelta(minutes=5)
        fit = (ts < r0).to_numpy()
        ev = ((ts >= r0) & (ts < r1)).to_numpy()
        y = make_label(df.close, fit)
        t0 = time.time()
        m = HistGradientBoostingClassifier(random_state=SEED, **GBM3_HP).fit(x[fit], y[fit])
        pred[ev] = m.predict(x[ev])
        rule[ev] = y[ev]
        meta.append({"refit": str(r0.date()), "fit_bars": int(fit.sum()), "pred_bars": int(ev.sum()),
                     "bal_acc_vs_rule": float(_bal_acc(y[ev], pred[ev])), "fit_sec": round(time.time() - t0, 1)})
        log(f"{sym} refit {r0.date()} fit {fit.sum():,} pred {ev.sum():,} bal_acc {meta[-1]['bal_acc_vs_rule']:.4f} "
            f"({meta[-1]['fit_sec']}s)")
    m = pred >= 0
    out = pd.DataFrame({"timestamp": ts[m].values, "model": pred[m], "rule": rule[m]})
    out.to_parquet(cp, index=False)
    (CACHE / f"{sym}_wf_meta.json").write_text(json.dumps(meta))
    return out, meta


def _bal_acc(y, p) -> float:
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


# ── 3. 측정 ────────────────────────────────────────────────────────────────────────────
def barrier_ev(high, low, close, t, side) -> float:
    """t 종가 진입, t+1..t+H 선착. 같은 봉에서 둘 다 닿으면 손절. 반환 bp."""
    e = close[t]
    for j in range(t + 1, t + H + 1):
        if side > 0:
            if low[j] <= e * (1 - SL): return -SL * 1e4
            if high[j] >= e * (1 + TP): return TP * 1e4
        else:
            if high[j] >= e * (1 + SL): return -SL * 1e4
            if low[j] <= e * (1 - TP): return TP * 1e4
    return side * (close[t + H] / e - 1) * 1e4


def windows(sym: str, hourly: bool = False) -> pd.DataFrame:
    """hourly=False: 하루 1창(23:55, 겹치지 않음 = 사전등록 주 판정). True: 매시 :55(24위상, 창이 겹침 —
    월 블록 부트스트랩이 의존을 흡수) = 위상 운 민감도."""
    k = _klines(sym)
    k = k[k.timestamp >= pd.Timestamp("2021-12-01")].reset_index(drop=True)
    hi, lo, cl = (k[c].to_numpy(float) for c in ("high", "low", "close"))
    sma = k["close"].rolling(SMA_N).mean().to_numpy()
    idx = np.where((k.timestamp.dt.minute == 55) & ((k.timestamp.dt.hour == 23) | hourly))[0]
    idx = idx[(idx >= SMA_N) & (idx + H < len(k))]
    # 연속성 가드: 창 안에 결측 봉이 있으면 288봉 ≠ 24h 이므로 뺀다
    span = (k.timestamp.to_numpy()[idx + H] - k.timestamp.to_numpy()[idx]) / np.timedelta64(5, "m")
    idx = idx[span == H]
    return pd.DataFrame({"timestamp": k.timestamp.to_numpy()[idx],
                         "sma": np.where(cl[idx] > sma[idx], 1, -1),
                         "L": [barrier_ev(hi, lo, cl, t, 1) for t in idx],
                         "S": [barrier_ev(hi, lo, cl, t, -1) for t in idx]})


def _stat(L, S, d) -> np.ndarray:
    """[롱 순−역, 숏 순−역, 평균]. d=+1/−1/0(기권)."""
    with np.errstate(invalid="ignore"):
        lo = L[d > 0].mean() - L[d < 0].mean() if (d > 0).any() and (d < 0).any() else np.nan
        so = S[d < 0].mean() - S[d > 0].mean() if (d > 0).any() and (d < 0).any() else np.nan
    return np.array([lo, so, (lo + so) / 2])


def compare(w: pd.DataFrame, srcs: dict[str, np.ndarray], seed: int = 20261004) -> dict:
    """각 상태원의 순−역 + 모델/규칙 − SMA288 쌍대 차(자기 커버리지 · 같은 창). 월 블록 부트스트랩."""
    L, S = w.L.to_numpy(), w.S.to_numpy()
    month = w.timestamp.dt.to_period("M").astype(str).to_numpy()
    blocks = [np.where(month == m)[0] for m in np.unique(month)]
    sma = w.sma.to_numpy()

    def stats(ix):
        out = {}
        for n, d in srcs.items():
            out[n] = _stat(L[ix], S[ix], d[ix])
            if n != "sma288":
                nz = d[ix] != 0
                out[f"{n}_minus_sma288"] = out[n] - _stat(L[ix], S[ix], sma[ix])
                out[f"{n}_minus_sma288_same_windows"] = (_stat(L[ix][nz], S[ix][nz], d[ix][nz])
                                                         - _stat(L[ix][nz], S[ix][nz], sma[ix][nz]))
        return out

    point = stats(np.arange(len(w)))
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in point}
    for _ in range(B):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        for k, v in stats(ix).items():
            boot[k].append(v)
    res = {"n_windows": int(len(w)), "n_months": len(blocks),
           "range": [str(w.timestamp.min().date()), str(w.timestamp.max().date())]}
    for k, v in point.items():
        b = np.array(boot[k])
        lo_, hi_ = np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)
        res[k] = {nm: {"bp": round(float(v[i]), 2), "ci95": [round(float(lo_[i]), 2), round(float(hi_[i]), 2)]}
                  for i, nm in enumerate(("long", "short", "both"))}
    for n, d in srcs.items():
        res[f"{n}_coverage"] = round(float(np.mean(d != 0)), 4)
        if n != "sma288":
            nz = d != 0
            res[f"{n}_disagree_with_sma288"] = round(float(np.mean(d[nz] != sma[nz])), 4)
    return res


def _dir(c: np.ndarray) -> np.ndarray:
    return np.select([c == 0, c == 1], [1, -1], 0)          # 0 bull · 1 bear · 2 chop


def yearly_sma(w: pd.DataFrame) -> dict:
    out = {}
    for y, g in w.groupby(w.timestamp.dt.year):
        out[str(y)] = {"n": len(g), **{k: round(float(v), 2) for k, v in
                                        zip(("long", "short", "both"), _stat(g.L.to_numpy(), g.S.to_numpy(), g.sma.to_numpy()))}}
    return out


def asset_block(sym: str, preds: pd.DataFrame | None, start: pd.Timestamp) -> dict:
    w, wh = windows(sym), windows(sym, hourly=True)
    full = {"sma288_all_history": compare(w, {"sma288": w.sma.to_numpy()}),
            "sma288_all_history_hourly": compare(wh, {"sma288": wh.sma.to_numpy()}),
            "sma288_by_year_point": yearly_sma(w)}
    old = w[w.timestamp < RECENT].reset_index(drop=True)
    full["sma288_2021_2024_aux"] = compare(old, {"sma288": old.sma.to_numpy()})
    if preds is None:
        r = w[w.timestamp >= start].reset_index(drop=True)
        full["recent"] = compare(r, {"sma288": r.sma.to_numpy()})
        return full
    r = w.merge(preds, on="timestamp", how="inner")
    r = r[r.timestamp >= start].reset_index(drop=True)
    srcs = {"sma288": r.sma.to_numpy(), "model": _dir(r.model.to_numpy())}
    if "rule" in r:
        srcs["rule_s96k9"] = _dir(r.rule.to_numpy())
    full["recent"] = compare(r, srcs)
    rh = wh.merge(preds, on="timestamp", how="inner")
    rh = rh[rh.timestamp >= start].reset_index(drop=True)
    full["recent_hourly"] = compare(rh, {k: (rh.sma.to_numpy() if k == "sma288" else
                                             _dir(rh[k.replace("_s96k9", "")].to_numpy())) for k in srcs})
    if start < CLEAN_OOS:
        s2 = r[r.timestamp >= CLEAN_OOS].reset_index(drop=True)
        full["since_2025_09"] = compare(s2, {k: v[r.timestamp >= CLEAN_OOS] for k, v in srcs.items()})
        h2 = rh[rh.timestamp >= CLEAN_OOS].reset_index(drop=True)
        full["since_2025_09_hourly"] = compare(h2, {k: (h2.sma.to_numpy() if k == "sma288" else
                                                       _dir(h2[k.replace("_s96k9", "")].to_numpy())) for k in srcs})
    return full


def verdict(rec: dict) -> str:
    own, same, sma = rec["model_minus_sma288"]["both"], rec["model_minus_sma288_same_windows"]["both"], rec["sma288"]["both"]
    if own["ci95"][0] > 0 and same["ci95"][0] > 0:
        return "SOL 모델 채택"
    if sma["ci95"][0] > 0:
        return "SMA288 한 줄이면 충분"
    return "검정력 부족"


def run() -> None:
    t0 = time.time()
    out = {"pipeline": PIPELINE, "constants_if_applied": CONSTANTS, "preregistration": {
        "written_before_analysis": True,
        "metric": "겹치지 않는 24h 창(매일 23:55 봉 종가 진입) · TP1.5%/SL0.7% · 288봉 · 순−역 bp(롱·숏·평균) · 월 블록 부트스트랩 B=4000 95% CI",
        "primary_window": "2025-01-01 ~ 데이터 끝(2026-08-03), 모델 예측은 분기별 확장창 재적합의 표본 밖 값",
        "adopt_model_if": "SOL 모델 − SMA288 (평균 순−역)의 CI 하한 > 0 — 자기 커버리지·같은 창(모델 비-chop 창) 둘 다",
        "else_sma_enough_if": "SOL SMA288 평균 순−역 CI 하한 > 0",
        "else": "검정력 부족",
        "grid": "S96_K9 고정, 격자 재탐색 안 함"}}
    xrp_pred, xrp_meta = walk_forward("XRPUSDT")
    sol_pred, sol_meta = walk_forward("SOLUSDT")
    xrp = asset_block("XRPUSDT", xrp_pred, RECENT)
    xrp["walk_forward"] = xrp_meta
    ctrl = {"XRPUSDT": xrp}
    for s, p in CLEAN_PRED.items():
        pr = pd.read_parquet(p).rename(columns={"regime": "model"})
        ctrl[s] = asset_block(s, pr, CLEAN_OOS)
        ctrl[s]["note"] = f"{p.parent.name}: 학습 2024-01~2025-08-31 단일 적합, 2025-09~ 표본 밖. ETH 는 배포 리본(GBM3 balnobb)이 아니라 같은 계열 S12_K3"
    sol = asset_block("SOLUSDT", sol_pred, RECENT)
    sol["walk_forward"] = sol_meta
    out["control_xrp"] = ctrl
    out["sol"] = {"model": sol, "sma288": {k: sol[k] for k in ("sma288_all_history", "sma288_2021_2024_aux", "sma288_by_year_point")},
                  "verdict": verdict(sol["recent"])}
    out["control_xrp"]["XRPUSDT"]["verdict_same_rule"] = verdict(xrp["recent"])
    out["runtime_sec"] = round(time.time() - t0, 1)
    p = OUT_DIR / "regime_sol.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)                                      # pipeline / constants_if_applied 는 손으로 채운 칸을 보존
    p.write_text(json.dumps(old, ensure_ascii=False, indent=2))
    log(f"-> {p} ({out['runtime_sec']}s) verdict {out['sol']['verdict']}")


def selftest() -> None:
    n = H + 2
    cl = np.full(n, 100.0); hi = cl.copy(); lo = cl.copy()
    hi[5] = 101.6                                        # 롱 TP
    assert barrier_ev(hi, lo, cl, 0, 1) == TP * 1e4
    assert barrier_ev(hi, lo, cl, 0, -1) == -SL * 1e4    # 숏은 같은 봉의 +1.6% 에 손절
    lo[5] = 99.2                                         # 같은 봉 양쪽 → 손절 우선
    assert barrier_ev(hi, lo, cl, 0, 1) == -SL * 1e4
    hi2, lo2, cl2 = np.full(n, 100.0), np.full(n, 100.0), np.full(n, 100.0); cl2[H] = 100.5
    assert abs(barrier_ev(hi2, lo2, cl2, 0, 1) - 50.0) < 1e-9   # 시간 청산
    L, S, d = np.array([10., 10, -5, -5]), np.array([-5., -5, 10, 10]), np.array([1, 1, -1, -1])
    assert np.allclose(_stat(L, S, d), [15, 15, 15])
    assert np.isnan(_stat(L, S, np.zeros(4))[2])        # 전부 기권이면 정의 안 됨
    assert list(_dir(np.array([0, 1, 2]))) == [1, -1, 0]
    c = pd.Series(np.r_[np.linspace(1, 2, 400), np.linspace(2, 1, 400)])
    y = make_label(c, np.ones(len(c), bool), 24, 3)
    assert (y[300:390] == 0).all() and (y[700:790] == 1).all()     # 단조 상승=bull, 하락=bear
    print("selftest ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    {"build": lambda: build(sys.argv[2]), "run": run, "selftest": selftest}[cmd]()
