"""ETH 4시간 방향 예측 — 실시간 파이프라인 (2026-10-15, 대시보드 ⑤ 예측 판 «4h 예측»).

모델 = research_eth_best_direction_model_20261010 의 «--nomd» 판(65피쳐: 가격 P · 체결 T · OI·롱숏 O · 교차 X · 시간 C · 개장 U).
  마이크로(M)·깊이(D)·베이시스(B)·현물(S) 묶음은 실시간 원천이 없어 뺐다 — 연구에서 같은 65피쳐로 ETH 결합 1.47(전체 1.56).
  매달 1일 재학습: 학습 창 = 직전 365일(정화 4h), HGB 시드 0·1·2 평균, 문턱 = 학습 예측 20/80 분위.
  예측 ≥ 상위 문턱 → 상승 · ≤ 하위 문턱 → 하락 · 사이 → 보류. 결정 = 5분봉 마감, 보유 4h. 참고용(전진 판정 전).

학습과 실시간이 같은 `build_features` 하나를 쓴다. 입력은 둘 다 data.binance.vision 패널 형식이다:
  timestamp(5분봉 시작) · high · low · close · volume · trades · taker_buy_base · metrics 5열.
  🔴metrics 시각 규약(2026-10-15 서버 REST 원장 ↔ vision 576행 대조): vision 행 t = REST 스탬프 t+5분
  (OI·상위 포지션·계정·전체 롱숏), 테이커 비 = 같은 스탬프. `rest_frames` 가 REST 를 이 규약으로 옮긴다.

    python scripts/live_eth_dir4h_20261015.py --selftest            # 연구 패널과 피쳐 일치(로컬 vision 패널)
    python scripts/live_eth_dir4h_20261015.py --train [--month 2026-10-01]   # vision 갱신 + 학습 → data/live/eth_dir4h_model.joblib
    python scripts/live_eth_dir4h_20261015.py --parity              # (서버) REST 피쳐 ↔ vision 피쳐 어제 하루 대조
    python scripts/live_eth_dir4h_20261015.py                       # (서버) 지금 예측 한 줄
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data" / "live" / "eth_dir4h_model.joblib"
PANEL = ROOT / "data" / "live" / "dir4h_panel"          # 코인별 vision 패널(최근 480일 = 학습 365 + 추세 5표 90 + 여유, 매달 이어 받기)
COINS = ["ADA", "APT", "ARB", "ATOM", "AVAX", "BNB", "BTC", "DOGE", "DOT", "FIL", "INJ", "LINK", "LTC", "NEAR", "OP", "SOL", "SUI", "TRX", "XRP"]
MCOLS = ["sum_open_interest", "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
         "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
FEATS = ["P_r5m", "P_r15m", "P_r1h", "P_r4h", "P_r1d", "P_r7d", "P_r28d", "P_rv1h", "P_rv1d", "P_vratio", "P_rsi14", "P_pctb",
         "P_pos1d", "P_veto_z", "P_ac144", "P_trend5", "T_imb5m", "T_imb15m", "T_imb1h", "T_imb4h", "T_cvd1h_z", "T_vol_z",
         "T_trades_z", "T_size", "O_doi5m", "O_doi1h", "O_doi4h", "O_doi1d", "O_doi1h_z", "O_doi1h_x_r", "O_toppos",
         "O_toppos_d1h", "O_toppos_z", "O_topacct", "O_topacct_d1h", "O_topacct_z", "O_glob", "O_glob_d1h", "O_glob_z",
         "O_takerls", "O_takerls_d1h", "O_takerls_z", "O_top_minus_glob", "X_btc_r5m", "X_btc_r15m", "X_btc_r1h", "X_btc_r4h",
         "X_btc_imb1h", "X_btc_doi1h", "X_breadth1h", "X_breadth1d", "X_eth_rel1h", "X_eth_rel1d", "X_eth_rank1d", "X_disp1h",
         "C_hsin", "C_hcos", "C_dow", "C_to_funding", "C_weekend", "U_eth_since_open", "U_eth_f30_day", "U_btc_since_open",
         "U_btc_f30_day", "U_min_since_open"]   # 연구 feat_cols 순서(M·D·B·S 뺌) -- 모델 입력 순서 계약
H = 48
HGB = dict(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=1000, l2_regularization=1.0)


# ── 연구 스크립트와 같은 보조 함수 ────────────────────────────────────────
def lag(x, k):
    x = np.asarray(x, dtype=float)
    return np.r_[np.full(k, np.nan), x[:-k]] if k > 0 else x


def rsum(x, k):
    c = np.r_[0.0, np.nancumsum(np.nan_to_num(np.asarray(x, dtype=float)))]
    o = np.full(len(x), np.nan); o[k - 1:] = c[k:] - c[:-k]
    return o


def rz(x, w=2016, mp=288):
    s = pd.Series(x).shift(1)
    med, q1, q3 = s.rolling(w, min_periods=mp).median(), s.rolling(w, min_periods=mp).quantile(.25), s.rolling(w, min_periods=mp).quantile(.75)
    return ((pd.Series(x) - med) / ((q3 - q1) / 1.349).replace(0, np.nan)).to_numpy()


def known(p, col):
    """metrics 열을 그 봉 마감에 아는 값으로(vision 2024-03-04 이후 규약 = 한 행 늦춤). 학습 창은 전부 그 이후다."""
    return lag(p[col].to_numpy(dtype=float), 1)


def grid(p: pd.DataFrame, ts: pd.Series | None = None) -> pd.DataFrame:
    p = p.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if ts is None:
        ts = pd.Series(pd.date_range(p.timestamp.min(), p.timestamp.max(), freq="5min"))
    return pd.DataFrame({"timestamp": ts.to_numpy()}).merge(p, on="timestamp", how="left")


def build_features(frames: dict[str, pd.DataFrame], daily_close: pd.Series | None = None) -> pd.DataFrame:
    """frames = {"ETH": …, "BTC": …, 나머지 COINS …} vision 형식. 결과 = ts + FEATS + y_4h (행 = 5분봉, 결정 = 봉 마감)."""
    p = grid(frames["ETH"])
    ts = p.timestamp; c = p.close.ffill().to_numpy(); lc = np.log(c)
    F = pd.DataFrame({"ts": ts})
    F["y_4h"] = np.r_[(lc[H:] - lc[:-H]) * 1e4, np.full(H, np.nan)]
    r1 = np.r_[np.nan, np.diff(lc)] * 1e4
    for k, nm in ((1, "5m"), (3, "15m"), (12, "1h"), (48, "4h"), (288, "1d"), (2016, "7d"), (8064, "28d")):
        F[f"P_r{nm}"] = (lc - lag(lc, k)) * 1e4
    F["P_rv1h"] = pd.Series(r1).rolling(12).std().to_numpy(); F["P_rv1d"] = pd.Series(r1).rolling(288).std().to_numpy()
    F["P_vratio"] = F.P_rv1h / F.P_rv1d
    d = pd.Series(c).diff()
    F["P_rsi14"] = (100 - 100 / (1 + d.clip(lower=0).ewm(alpha=1 / 14).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14).mean())).to_numpy()
    F["P_pctb"] = ((pd.Series(c) - pd.Series(c).rolling(20).mean()) / (2 * pd.Series(c).rolling(20).std())).to_numpy()
    hi, lo = p.high.to_numpy(), p.low.to_numpy()
    h288, l288 = pd.Series(hi).rolling(288).max().to_numpy(), pd.Series(lo).rolling(288).min().to_numpy()
    F["P_pos1d"] = (c - l288) / (h288 - l288)
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - lag(c, 1)), np.abs(lo - lag(c, 1))))
    F["P_veto_z"] = (c - pd.Series(c).rolling(144).mean().to_numpy()) / pd.Series(tr).ewm(alpha=1 / 144).mean().to_numpy()
    F["P_ac144"] = pd.Series(r1).rolling(144).apply(lambda v: np.corrcoef(v[:-1], v[1:])[0, 1], raw=True).to_numpy()
    if daily_close is None:   # 그날 마지막 5분봉 종가 = 일봉 종가 → 다음날 00:00 에 앎
        daily_close = pd.Series(c, index=ts).groupby(ts.dt.floor("D").to_numpy()).last()
    dk = np.searchsorted(daily_close.index.to_numpy() + np.timedelta64(1, "D"), (ts + pd.Timedelta("5min")).to_numpy(), "right") - 1
    dv = daily_close.to_numpy()
    votes = [np.where(dk - L >= 0, np.sign(dv[np.maximum(dk, 0)] - dv[np.maximum(dk - L, 0)]), np.nan) for L in (7, 14, 28, 56, 90)]
    F["P_trend5"] = np.mean(votes, axis=0)
    v, tb = p.volume.to_numpy(), p.taker_buy_base.to_numpy()
    dl = 2 * tb - v
    for k, nm in ((1, "5m"), (3, "15m"), (12, "1h"), (48, "4h")):
        F[f"T_imb{nm}"] = rsum(dl, k) / rsum(v, k)
    F["T_cvd1h_z"] = rsum(dl, 12) / pd.Series(rsum(dl, 12)).shift(1).rolling(2016, min_periods=288).std().to_numpy()
    F["T_vol_z"] = rz(np.log(rsum(v, 12)))
    F["T_trades_z"] = rz(np.log(rsum(p.trades.to_numpy(dtype=float), 12)))
    F["T_size"] = np.log(rsum(v, 12) / rsum(p.trades.to_numpy(dtype=float), 12))
    oi = known(p, "sum_open_interest")
    for k, nm in ((1, "5m"), (12, "1h"), (48, "4h"), (288, "1d")):
        F[f"O_doi{nm}"] = (np.log(oi) - np.log(lag(oi, k))) * 1e4
    F["O_doi1h_z"] = rz(F.O_doi1h.to_numpy())
    F["O_doi1h_x_r"] = F.O_doi1h * np.sign(F.P_r1h)
    for col, nm in (("sum_toptrader_long_short_ratio", "toppos"), ("count_toptrader_long_short_ratio", "topacct"),
                    ("count_long_short_ratio", "glob"), ("sum_taker_long_short_vol_ratio", "takerls")):
        x = np.log(known(p, col))
        F[f"O_{nm}"] = x; F[f"O_{nm}_d1h"] = x - lag(x, 12); F[f"O_{nm}_z"] = rz(x, 8064, 2016)
    F["O_top_minus_glob"] = F.O_toppos - F.O_glob
    rets12, rets288 = [], []
    for other in COINS:
        q = grid(frames[other], ts) if other in frames else pd.DataFrame({"close": np.full(len(ts), np.nan)})
        lq = np.log(q.close.ffill().to_numpy())
        rets12.append((lq - lag(lq, 12)) * 1e4); rets288.append((lq - lag(lq, 288)) * 1e4)
        if other == "BTC":
            for k, nm in ((1, "5m"), (3, "15m"), (12, "1h"), (48, "4h")):
                F[f"X_btc_r{nm}"] = (lq - lag(lq, k)) * 1e4
            F["X_btc_imb1h"] = rsum(2 * q.taker_buy_base.to_numpy() - q.volume.to_numpy(), 12) / rsum(q.volume.to_numpy(), 12)
            boi = known(q, "sum_open_interest")
            F["X_btc_doi1h"] = (np.log(boi) - np.log(lag(boi, 12))) * 1e4
    R12, R288 = np.vstack(rets12), np.vstack(rets288)
    with np.errstate(all="ignore"):
        F["X_breadth1h"] = np.nanmean(R12, 0); F["X_breadth1d"] = np.nanmean(R288, 0)
        F["X_disp1h"] = np.nanstd(R12, 0)
    F["X_eth_rel1h"] = F.P_r1h - F.X_breadth1h; F["X_eth_rel1d"] = F.P_r1d - F.X_breadth1d
    F["X_eth_rank1d"] = (R288 < F.P_r1d.to_numpy()).mean(0)
    tt = ts + pd.Timedelta("5min")
    hr = tt.dt.hour + tt.dt.minute / 60
    F["C_hsin"], F["C_hcos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    F["C_dow"] = tt.dt.dayofweek; F["C_to_funding"] = (8 - hr % 8) % 8
    F["C_weekend"] = (tt.dt.dayofweek >= 5).astype(float)
    d0 = tt.dt.floor("D")
    for sym, pref in (("ETH", "U_eth"), ("BTC", "U_btc")):   # 시각 t 의 가격 = t−5분 봉 종가
        s = grid(frames[sym]).set_index("timestamp").close.ffill()
        px = lambda t: s.reindex((t - pd.Timedelta("5min")).to_numpy()).to_numpy()
        now, o, o30 = px(tt), px(d0), px(d0 + pd.Timedelta("30min"))
        F[f"{pref}_since_open"] = np.log(now / o) * 1e4
        F[f"{pref}_f30_day"] = np.where((tt - d0) >= pd.Timedelta("30min"), np.log(o30 / o) * 1e4, np.nan)
    F["U_min_since_open"] = ((tt - d0).dt.total_seconds() / 60).to_numpy()
    F.replace([np.inf, -np.inf], np.nan, inplace=True)
    return F[["ts", "y_4h"] + FEATS]


# ── 학습 (vision 패널) ─────────────────────────────────────────────────────
def vision_update(end_day: str, days: int = 480, workers: int = 8) -> dict[str, pd.DataFrame]:
    """PANEL/{SYM}USDT.parquet 를 end_day 까지 이어 받고 최근 days 일만 남긴다(data.binance.vision 정적 아카이브 -- REST 아님)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from concurrent.futures import ThreadPoolExecutor
    from build_binance_vision_panel_20260915 import day
    PANEL.mkdir(parents=True, exist_ok=True)
    out = {}
    for sym in ["ETH"] + COINS:
        f = PANEL / f"{sym}USDT.parquet"
        old = pd.read_parquet(f) if f.exists() else None
        if old is not None and len(old) and old.timestamp.min() > pd.Timestamp(end_day) - pd.Timedelta(days=days - 2):
            old = None   # 잘린 패널(10-15 --parity days=45 가 480일 → 45일로 잘랐다) -- 처음부터 다시 받는다
        start = (old.timestamp.max().normalize() + pd.Timedelta("1D")) if old is not None and len(old) else pd.Timestamp(end_day) - pd.Timedelta(days=days)
        dates = [x.strftime("%Y-%m-%d") for x in pd.date_range(start, end_day, freq="D")]
        with ThreadPoolExecutor(workers) as ex:
            parts = [x for x in ex.map(lambda dd: day(f"{sym}USDT", dd), dates) if x is not None]
        df = pd.concat(([old] if old is not None else []) + parts).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        df = df[df.timestamp >= pd.Timestamp(end_day) - pd.Timedelta(days=days)].reset_index(drop=True)
        df.to_parquet(f)
        out[sym] = df
        print(f"  {sym} {len(df):,}행 ~{df.timestamp.max()} (+{len(dates)}일)", flush=True)
    return out


def train(frames: dict[str, pd.DataFrame], month: pd.Timestamp, seeds=(0, 1, 2)) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    F = build_features(frames)
    ts, y = F.ts.to_numpy(), F.y_4h.to_numpy()
    cut = np.datetime64(month) - np.timedelta64(5 * H, "m")
    tr = (ts < cut) & (ts >= cut - np.timedelta64(365, "D")) & np.isfinite(y)
    X = F[FEATS].to_numpy(dtype=np.float32)
    if tr.sum() < 100_000:   # 365일 = 105,120행 -- 짧은 패널로 조용히 학습하지 않게
        raise ValueError(f"학습 행 부족 {int(tr.sum())}")
    models, pt = [], []
    for s in seeds:
        m = HistGradientBoostingRegressor(random_state=s, **HGB).fit(X[tr][::2], y[tr][::2])
        models.append(m); pt.append(m.predict(X[tr][::7]))
    lo, hi = np.quantile(np.mean(pt, 0), [0.2, 0.8])
    return {"models": models, "features": FEATS, "lo": float(lo), "hi": float(hi), "month": str(month.date()),
            "train_window": [str(pd.Timestamp(ts[tr].min())), str(pd.Timestamp(ts[tr].max()))], "n_train": int(tr.sum()), "seeds": list(seeds)}


def predict(art: dict, F: pd.DataFrame) -> np.ndarray:
    X = F[art["features"]].to_numpy(dtype=np.float32)
    return np.mean([m.predict(X) for m in art["models"]], axis=0)


# ── 실시간 (바이낸스 REST → vision 형식) ─────────────────────────────────────
_CACHE: dict = {}
FAPI = "https://fapi.binance.com"
METRIC_EP = {   # REST 끝점 → (vision 열, 값 키, REST 스탬프 → vision 행 밀기)
    "openInterestHist": ("sum_open_interest", "sumOpenInterest", 5),
    "topLongShortPositionRatio": ("sum_toptrader_long_short_ratio", "longShortRatio", 5),
    "topLongShortAccountRatio": ("count_toptrader_long_short_ratio", "longShortRatio", 5),
    "globalLongShortAccountRatio": ("count_long_short_ratio", "longShortRatio", 5),
    "takerlongshortRatio": ("sum_taker_long_short_vol_ratio", "buySellRatio", 0),
}


def _get(path: str, params: dict):
    import requests   # 서버 안에선 binance_ban_guard 가 requests 를 감싼다(밴 중 전송 안 함)
    r = requests.get(FAPI + path, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _klines(sym: str, n: int, now_ms: int) -> pd.DataFrame:
    rows, end = [], None
    while len(rows) < n:
        q = {"symbol": f"{sym}USDT", "interval": "5m", "limit": min(1000, n - len(rows) + 1), **({"endTime": end} if end else {})}
        got = _get("/fapi/v1/klines", q)
        if not got:
            break
        rows = got + rows; end = int(got[0][0]) - 1
        if len(got) < q["limit"]:
            break
    rows = [r for r in rows if int(r[6]) < now_ms]   # 형성 중 봉 버림
    return pd.DataFrame({"timestamp": pd.to_datetime([int(r[0]) for r in rows], unit="ms"), "high": [float(r[2]) for r in rows],
                         "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows], "volume": [float(r[5]) for r in rows],
                         "trades": [float(r[8]) for r in rows], "taker_buy_base": [float(r[9]) for r in rows]})


def _metric(sym: str, ep: str, days: float) -> pd.DataFrame:
    col, key, shift = METRIC_EP[ep]
    rows, end, start = [], None, int(time.time() * 1000 - days * 86_400_000)
    while True:
        q = {"symbol": f"{sym}USDT", "period": "5m", "limit": 500, **({"endTime": end} if end else {})}
        got = _get(f"/futures/data/{ep}", q)
        if not got:
            break
        rows = got + rows; end = int(got[0]["timestamp"]) - 1
        if len(got) < 500 or int(got[0]["timestamp"]) <= start:
            break
    t = pd.to_datetime([int(r["timestamp"]) for r in rows], unit="ms") - pd.Timedelta(minutes=shift)
    return pd.DataFrame({"timestamp": t, col: [float(r[key]) for r in rows]})


def _merge(old: pd.DataFrame | None, new: pd.DataFrame, keep_days: float) -> pd.DataFrame:
    df = new if old is None else pd.concat([old, new]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return df[df.timestamp >= df.timestamp.max() - pd.Timedelta(days=keep_days)].reset_index(drop=True)


def rest_frames(now_ms: int | None = None, others_n: int = 340) -> dict[str, pd.DataFrame]:
    """REST → vision 형식 frames. 첫 호출 = ETH 31일(klines 9천·metrics 5종) 받기, 다음부터 꼬리만 이어 붙인다(_CACHE)."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    first = "ETH_k" not in _CACHE or pd.Timestamp(now_ms, unit="ms") - _CACHE["ETH_k"].timestamp.max() > pd.Timedelta(hours=20)   # 꼬리 300봉=25h -- 그보다 오래 안 불리면 구멍 대신 처음부터
    _CACHE["ETH_k"] = _merge(_CACHE.get("ETH_k"), _klines("ETH", 9000 if first else 300, now_ms), 32)
    met = {}
    for ep, (col, _, _) in METRIC_EP.items():
        _CACHE[f"ETH_{ep}"] = _merge(_CACHE.get(f"ETH_{ep}"), _metric("ETH", ep, 30.5 if first else 0.5), 32)
        met[col] = _CACHE[f"ETH_{ep}"]
    eth = _CACHE["ETH_k"]
    for col, m in met.items():
        eth = eth.merge(m, on="timestamp", how="left")
    frames = {"ETH": eth}
    for sym in COINS:   # 교차 = 1일 수익(288봉)·BTC 4h·OI 1h 면 충분
        k = _klines(sym, others_n, now_ms)
        if sym == "BTC":
            k = k.merge(_metric("BTC", "openInterestHist", others_n / 288 + 0.1), on="timestamp", how="left")
        frames[sym] = k
    for f in frames.values():
        for c in MCOLS:
            if c not in f:
                f[c] = np.nan
    return frames


def daily_closes(now_ms: int) -> pd.Series:
    got = _get("/fapi/v1/klines", {"symbol": "ETHUSDT", "interval": "1d", "limit": 120})
    got = [r for r in got if int(r[6]) < now_ms]
    return pd.Series([float(r[4]) for r in got], index=pd.to_datetime([int(r[0]) for r in got], unit="ms"))


def live_payload(art: dict | None = None) -> dict:
    import joblib
    art = art or joblib.load(ARTIFACT)
    now_ms = int(time.time() * 1000)
    F = build_features(rest_frames(now_ms), daily_closes(now_ms))
    last = F.iloc[-1:]
    p = float(predict(art, last)[0])
    nan = [c for c in FEATS if not np.isfinite(last[c].iloc[0])]
    tail = F.tail(48)
    hist = predict(art, tail)
    sig = 1 if p >= art["hi"] else -1 if p <= art["lo"] else 0
    return {"ok": True, "decided_at_ms": int(pd.Timestamp(last.ts.iloc[0]).value // 10**6) + 300_000, "pred_bp": round(p, 2),
            "lo": round(art["lo"], 2), "hi": round(art["hi"], 2), "signal": sig, "model_month": art["month"],
            "train_window": art["train_window"], "nan_feats": nan,
            "hist": [[int(pd.Timestamp(t).value // 10**6) + 300_000, round(float(x), 2)] for t, x in zip(tail.ts, hist)]}


# ── 점검 ──────────────────────────────────────────────────────────────────
def selftest() -> None:
    """연구 패널(research_eth_best_direction_model_20261010 build+add_open_feats)과 같은 원천에서 같은 값인지."""
    bv = Path("/home/kbj20/crypto-scalping/data/binance_vision/panel")
    ref = Path("/home/kbj20/crypto-scalping/.claude/worktrees/liquidation-price-feature-14eceb/tmp/best_dir_20261010/panel.parquet")
    lo_t = pd.Timestamp("2026-07-01")
    frames = {s: (lambda d: d[d.timestamp >= lo_t - pd.Timedelta(days=40)])(pd.read_parquet(bv / f"{s}USDT.parquet")) for s in ["ETH"] + COINS}
    full = pd.read_parquet(bv / "ETHUSDT.parquet")
    dc = full.set_index("timestamp").close.ffill().groupby(lambda t: t.normalize()).last()   # 일봉 = 그날 마지막 5분봉 종가(실시간은 1d klines)
    F = build_features(frames, dc)
    R = pd.read_parquet(ref, columns=["ts"] + FEATS)
    # 연구 패널은 2022 부터 계산 -- 긴 창(rz 8064·P_r28d·ac144 등)이 같은 값이 되는 구간만 비교
    a = F[F.ts >= pd.Timestamp("2026-08-05")].set_index("ts")[FEATS]
    b = R.set_index("ts").reindex(a.index)[FEATS]
    bad = {}
    for c in FEATS:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        both = np.isfinite(x) & np.isfinite(y)
        tol = 1e-6 * np.maximum(1, np.abs(y[both]))
        if (np.isfinite(x) != np.isfinite(y)).mean() > 0.001 or (np.abs(x[both] - y[both]) > tol).mean() > 0.001:
            bad[c] = (round(float((np.isfinite(x) != np.isfinite(y)).mean()), 4), float(np.nanmax(np.abs(x[both] - y[both])) if both.any() else -1))
    assert not bad, f"연구 패널과 다른 피쳐: {bad}"
    # 결정 시점 인과성: 마지막 봉을 잘라도 그 이전 행 피쳐가 같아야 한다
    cut = {s: f[f.timestamp < pd.Timestamp("2026-09-01 12:00")] for s, f in frames.items()}
    G = build_features(cut, dc[dc.index < pd.Timestamp("2026-09-01")]).set_index("ts")[FEATS]
    t = G.index[-1]
    assert np.allclose(G.loc[t].to_numpy(float), F.set_index("ts").loc[t, FEATS].to_numpy(float), equal_nan=True), "미래 행이 피쳐에 섞였다"
    print(f"selftest ok — 65피쳐 연구 패널 일치({len(a):,}행) · 인과성")


def parity() -> None:
    """(서버) 같은 봉을 REST 경로와 vision 경로로 만들어 피쳐 대조 -- 어제 UTC 하루."""
    now_ms = int(time.time() * 1000)
    dc = daily_closes(now_ms)
    R = build_features(rest_frames(now_ms, others_n=900), dc).set_index("ts")   # 다른 코인 3일 -- 어제 하루 전부의 1일 수익이 차도록
    y0 = (pd.Timestamp(now_ms, unit="ms").normalize() - pd.Timedelta("1D"))
    V = build_features(vision_update(str(y0.date())), dc).set_index("ts")
    idx = R.index.intersection(V.index)
    idx = idx[(idx >= y0) & (idx < y0 + pd.Timedelta("1D"))]
    rep = {}
    for c in FEATS:
        x, y = R.loc[idx, c].to_numpy(float), V.loc[idx, c].to_numpy(float)
        both = np.isfinite(x) & np.isfinite(y)
        rep[c] = [round(float(np.median(np.abs(x[both] - y[both]))), 6) if both.any() else None,
                  round(float(np.corrcoef(x[both], y[both])[0, 1]), 4) if both.sum() > 10 and np.std(y[both]) > 0 else None,
                  int((np.isfinite(x) != np.isfinite(y)).sum())]
    print(json.dumps({"rows": len(idx), "day": str(y0.date()), "feats": rep}, ensure_ascii=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    elif "--train" in a:
        import joblib
        month = pd.Timestamp(a[a.index("--month") + 1]) if "--month" in a else pd.Timestamp.utcnow().tz_localize(None).normalize().replace(day=1)
        frames = vision_update(str((pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta("1D")).date()))
        art = train(frames, month)
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        tmp = ARTIFACT.with_suffix(".tmp")
        joblib.dump(art, tmp)   # 다 쓴 뒤 교체 -- 쓰다 죽어도 지난 판이 남게
        if ARTIFACT.exists():
            ARTIFACT.replace(ARTIFACT.with_suffix(".joblib.prev"))
        tmp.replace(ARTIFACT)
        print("학습", {k: v for k, v in art.items() if k != "models"})
    elif "--parity" in a:
        parity()
    else:
        print(json.dumps(live_payload(), ensure_ascii=False)[:600])
