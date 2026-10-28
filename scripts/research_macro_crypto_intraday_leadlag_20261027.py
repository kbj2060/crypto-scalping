#!/usr/bin/env python3
"""매크로 선물·현물 5분봉/1분봉 ↔ BTC/ETH(바이낸스 선물·현물) 분 단위 리드-래그 측정 (2026-10-27).

왜: 일·주 단위 연구(docs/experiments/macro_crypto_correlation_20261027.md)는 «매크로는 동행할 뿐 선행하지 않는다»였다.
  분 단위에선 다를 수 있다 -- ES/NQ 선물이 코인보다 몇 분 먼저 가격을 발견하는지, 미국 정규장/장외/주말로 나눠 잰다.
무엇: (1) 수집  Yahoo(yfinance) CME·ICE 선물 + 현물/ETF/지수, 5분봉 59일 · 1분봉 ~29일(7일 창으로 이어 받기)
               바이낸스 USDⓈ-M 선물(fapi) + 현물(api) BTCUSDT·ETHUSDT 같은 기간, Hyperliquid xyz:XYZ100 5m·1m(REST 한도 5000봉)
     (2) 분석  세션별 동시 상관, corr(crypto_t, macro_{t-k}) k=-K..K (k>0 = 매크로가 앞섬), lead-lag ratio,
               표본외(시간 앞 70% 학습 / 뒤 30% 검증) 1~3봉 시차 회귀, 매크로 급변(4σ) 이벤트 뒤 코인 누적 반응
     (3) 자가검증  ES=F↔SPY, NQ=F↔XYZ100, BTC 선물↔현물 최대 상관이 lag 0이어야 한다(타임스탬프 정렬 트립와이어)
🔴 Yahoo CME 선물 시세는 규정상 ~10분 지연이다. 과거 봉의 타임스탬프는 실제 시각이라 «연구»엔 문제없지만,
   1~2봉 선행이 나와도 Yahoo 피드로는 라이브에서 못 쓴다 -- 라이브 후보는 XYZ100(무지연) 또는 유료 실시간(IB·Databento).
🔴 인과 규약: 봉 라벨 = 봉 시작 시각(UTC). macro_{t-1} 수익은 봉 t-1 마감(=봉 t 시작)에 확정 -- crypto_t 와 겹치지 않는다.
   연속하지 않는 봉(장 휴식·주말·결측) 사이 수익은 버린다. 연구용 진단이며 fresh-forward 승격 근거가 아니다.

  python scripts/research_macro_crypto_intraday_leadlag_20261027.py            # 수집 + 분석 (서버, pip install yfinance)
  python scripts/research_macro_crypto_intraday_leadlag_20261027.py --no-fetch # 캐시로 분석만
  python scripts/research_macro_crypto_intraday_leadlag_20261027.py --selftest # 네트워크 없이 합성 데이터로 검증
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tempfile
import urllib.request
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/research/macro_intraday_20261027"
OUT = ROOT / "docs/experiments/macro_crypto_intraday_leadlag_20261027"

# 이름 -> (yahoo 심볼, 설명). 선물 가격↑ = 금리↓ (ZT/ZN/ZB), 6J↑ = 엔 강세, JPY=X↑ = 엔 약세.
YAHOO = {
    "ES": ("ES=F", "S&P500 선물"), "NQ": ("NQ=F", "나스닥100 선물"), "RTY": ("RTY=F", "러셀2000 선물"),
    "ZT": ("ZT=F", "미 2년물 선물"), "ZN": ("ZN=F", "미 10년물 선물"), "ZB": ("ZB=F", "미 30년물 선물"),
    "GC": ("GC=F", "금 선물"), "SI": ("SI=F", "은 선물"), "HG": ("HG=F", "구리 선물"), "CL": ("CL=F", "WTI 선물"),
    "6E": ("6E=F", "유로 선물"), "6J": ("6J=F", "엔 선물"), "DXY": ("DX-Y.NYB", "달러지수(ICE)"),
    "SPY": ("SPY", "SPY ETF"), "QQQ": ("QQQ", "QQQ ETF"), "IBIT": ("IBIT", "현물 BTC ETF"),
    "HYG": ("HYG", "하이일드 ETF"), "LQD": ("LQD", "투자등급 ETF"), "TLT": ("TLT", "장기채 ETF"), "TIP": ("TIP", "물가채 ETF"),
    "GLD": ("GLD", "금 ETF"), "VIX": ("^VIX", "VIX 지수"), "TNX": ("^TNX", "10년 금리 지수"),
    "EURUSD": ("EURUSD=X", "EUR/USD"), "USDJPY": ("JPY=X", "USD/JPY"), "USDKRW": ("KRW=X", "USD/KRW"),
    "KOSPI": ("^KS11", "KOSPI"), "N225": ("^N225", "닛케이225"),
}
CRYPTO = {  # 이름 -> (url, symbol)
    "BTC_PERP": ("https://fapi.binance.com/fapi/v1/klines", "BTCUSDT"),
    "ETH_PERP": ("https://fapi.binance.com/fapi/v1/klines", "ETHUSDT"),
    "BTC_SPOT": ("https://api.binance.com/api/v3/klines", "BTCUSDT"),
    "ETH_SPOT": ("https://api.binance.com/api/v3/klines", "ETHUSDT"),
}
HL_INFO, HL_COIN = "https://api.hyperliquid.xyz/info", "xyz:XYZ100"
TARGETS = ("BTC_PERP", "ETH_PERP")
# 정렬 검사 (검증 대상, 기준). 필수 쌍이 없거나 실패하면 전체 중단(exit 2) -- IBIT~BTC 현물 = Yahoo↔바이낸스 교차 닻.
# 나머지 쌍은 실패해도 중단하지 않고 검증 대상을 «정렬 의심»으로 표시한다(리포트에서 ⚠, 선행 주장 금지).
REQUIRED_CHECKS = (("IBIT", "BTC_SPOT"), ("BTC_PERP", "BTC_SPOT"))
ALIGN_CHECKS = REQUIRED_CHECKS + (("ES", "SPY"), ("NQ", "QQQ"), ("XYZ100", "NQ"), ("GC", "GLD"), ("ZN", "TLT"), ("TNX", "ZN"),
                                  ("6E", "EURUSD"), ("6J", "USDJPY"), ("DXY", "EURUSD"))
SIDE_RATIO_MAX = 0.15  # max(|corr(±1)|)/|corr(0)|: 깨끗하면 ~0.02, 1분 어긋남이면 ~0.3 (감사 실측) -- 봉 미만 어긋남 탐지
STEP = {"5m": 300, "1m": 60}
LAGS = {"5m": 6, "1m": 15}


# ----------------------------------------------------------------------------- fetch
def _get_json(url: str, data: bytes | None = None, tries: int = 4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def fetch_binance(url: str, sym: str, interval: str, start_ms: int, end_ms: int) -> pd.Series:
    limit = 1500 if "fapi" in url else 1000
    rows, t = [], start_ms
    while t < end_ms:
        k = _get_json(f"{url}?symbol={sym}&interval={interval}&startTime={t}&endTime={end_ms}&limit={limit}")
        if not k:
            break
        rows += k
        t = int(k[-1][0]) + STEP[interval] * 1000
        time.sleep(0.15)
    if not rows:
        raise RuntimeError(f"binance {sym} {interval}: empty response")
    df = pd.DataFrame(rows).drop_duplicates(0)
    # 마감(close_time)이 지금 이후인 진행 중 봉은 버린다
    df = df[df[6].astype("int64") < int(time.time() * 1000)]
    return pd.Series(df[4].astype(float).values, index=pd.to_datetime(df[0].astype("int64"), unit="ms", utc=True))


def fetch_hl(interval: str, start_ms: int, end_ms: int) -> pd.Series:
    body = json.dumps({"type": "candleSnapshot", "req": {"coin": HL_COIN, "interval": interval,
                                                          "startTime": start_ms, "endTime": end_ms}}).encode()
    k = _get_json(HL_INFO, data=body)
    now_ms = int(time.time() * 1000)
    k = [c for c in k if int(c["T"]) < now_ms]
    return pd.Series([float(c["c"]) for c in k], index=pd.to_datetime([int(c["t"]) for c in k], unit="ms", utc=True))


def fetch_yahoo(sym: str, interval: str) -> pd.Series:
    import yfinance as yf  # 서버에만 필요 -- selftest 는 안 부른다
    tk = yf.Ticker(sym)
    if interval == "5m":
        parts = [tk.history(period="59d", interval="5m", prepost=False, auto_adjust=False, actions=False)]
    else:  # 1m 은 요청당 7일, 최근 30일까지만 허용
        end = pd.Timestamp.now(tz="UTC").floor("min")
        parts = []
        for i in range(4, 0, -1):
            s, e = end - pd.Timedelta(days=7 * i), end - pd.Timedelta(days=7 * (i - 1))
            parts.append(tk.history(start=s, end=e, interval="1m", prepost=False, auto_adjust=False, actions=False))
            time.sleep(0.5)
    df = pd.concat([p for p in parts if len(p)])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    s = df["Close"].astype(float)
    s.index = s.index.tz_convert("UTC")
    # 진행 중 마지막 봉 제거 -- CME 시세는 ~10분 지연이라 벽시계 기준으론 덜 끝난 봉이 남는다: 15분 여유
    return s[s.index + pd.Timedelta(seconds=STEP[interval]) <= pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=15)]


def fetch_all(interval: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cols, failures = {}, {}
    for name, (sym, _) in YAHOO.items():
        try:
            cols[name] = fetch_yahoo(sym, interval)
            print(f"[yahoo {interval}] {name:7s} {sym:9s} {len(cols[name]):6d} bars", flush=True)
        except Exception as e:  # 심볼 하나 실패로 전체를 멈추지 않는다 -- 리포트에 남긴다
            failures[name] = repr(e)[:200]
            print(f"[yahoo {interval}] {name} FAILED {failures[name]}", flush=True)
        time.sleep(0.5)
    if cols:  # 바이낸스가 막혀도 Yahoo 수집분은 남긴다
        pd.DataFrame(cols).sort_index().to_parquet(CACHE / f"close_{interval}_yahoo_only.parquet")
    t_min = min(s.index.min() for s in cols.values()) if cols else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=59)
    start_ms, end_ms = int(t_min.timestamp() * 1000), int(time.time() * 1000)
    for name, (url, sym) in CRYPTO.items():
        try:
            cols[name] = fetch_binance(url, sym, interval, start_ms, end_ms)
            print(f"[binance {interval}] {name} {len(cols[name])} bars", flush=True)
        except Exception as e:
            failures[name] = repr(e)[:200]
            print(f"[binance {interval}] {name} FAILED {failures[name]}", flush=True)
    try:
        cols["XYZ100"] = fetch_hl(interval, start_ms, end_ms)
        print(f"[hl {interval}] XYZ100 {len(cols['XYZ100'])} bars", flush=True)
    except Exception as e:
        failures["XYZ100"] = repr(e)[:200]
    px = pd.DataFrame(cols).sort_index()
    px.to_parquet(CACHE / f"close_{interval}.parquet")
    (CACHE / f"fetch_failures_{interval}.json").write_text(json.dumps(failures, indent=1))
    return px


# ----------------------------------------------------------------------------- analysis
def to_returns(px: pd.DataFrame, interval: str) -> pd.DataFrame:
    """정규 격자 위 로그수익(bp). 직전 격자 봉이 없으면(장 휴식·결측) NaN."""
    step = pd.Timedelta(seconds=STEP[interval])
    px = px[~px.index.duplicated(keep="last")]
    grid = pd.date_range(px.index.min().floor(step), px.index.max(), freq=step, tz="UTC")
    g = px.reindex(grid)
    return 1e4 * np.log(g.where(g > 0)).diff()  # 격자 위 diff -> 어느 한 봉이라도 없으면 NaN; 0·음수 가격은 NaN


def sessions(idx: pd.DatetimeIndex) -> pd.Series:
    """us_cash: 평일 09:30~16:00 ET · overnight: CME 선물 장외(일 18:00~금 17:00 ET, 매일 17~18시 휴식 제외) · macro_closed: 주말·휴식."""
    et = idx.tz_convert("America/New_York")
    mins = et.hour * 60 + et.minute
    wd = et.weekday
    cash = (wd < 5) & (mins >= 570) & (mins < 960)
    fut_open = ((wd < 4) | ((wd == 4) & (mins < 1020)) | ((wd == 6) & (mins >= 1080))) & ~((mins >= 1020) & (mins < 1080))
    lab = np.where(cash, "us_cash", np.where(fut_open, "overnight", "macro_closed"))
    return pd.Series(lab, index=idx)


def leadlag(r: pd.DataFrame, sess: pd.Series, interval: str) -> pd.DataFrame:
    K = LAGS[interval]
    rows = []
    for tgt in TARGETS:
        for x in r.columns:
            if x in TARGETS or x.startswith(("BTC_", "ETH_")):
                continue
            for s in ("us_cash", "overnight", "all"):
                m = (sess == s) if s != "all" else pd.Series(True, index=sess.index)
                for k in range(-K, K + 1):
                    xs = r[x].shift(k)  # k>0: 매크로 봉 t-k 와 코인 봉 t
                    d = pd.concat([r[tgt], xs], axis=1)[m.values].dropna()
                    if len(d) < 200:
                        continue
                    c = float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])
                    # 이분산 강건(White) z: 공통 변동성·군집이 있으면 r·√n 은 선행을 과대평가한다(감사 실측 sd(z) 1.3~2.4).
                    a, b = (d.iloc[:, 0] - d.iloc[:, 0].mean()).values, (d.iloc[:, 1] - d.iloc[:, 1].mean()).values
                    ab = a * b
                    z = float(ab.sum() / np.sqrt((ab ** 2).sum()))
                    rows.append(dict(interval=interval, target=tgt, macro=x, session=s, lag=k, corr=c, n=len(d), z=z,
                                     z_naive=c * np.sqrt(len(d))))
    return pd.DataFrame(rows)


def summarize_ll(ll: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (iv, tgt, x, s), g in ll.groupby(["interval", "target", "macro", "session"]):
        g = g.set_index("lag")
        pos, neg = g[g.index > 0], g[g.index < 0]
        best_pos = pos["z"].abs().idxmax() if len(pos) else np.nan
        out.append(dict(interval=iv, target=tgt, macro=x, session=s, n=int(g["n"].max()),
                        corr_lag0=g["corr"].get(0, np.nan), z_lag0=g["z"].get(0, np.nan),
                        best_macro_lead=best_pos, corr_best_lead=pos["corr"].get(best_pos, np.nan),
                        z_best_lead=pos["z"].get(best_pos, np.nan),
                        corr_lag1=g["corr"].get(1, np.nan), corr_lagm1=g["corr"].get(-1, np.nan),
                        lead_lag_ratio=float((pos["corr"] ** 2).sum() / max((neg["corr"] ** 2).sum(), 1e-12))))
    return pd.DataFrame(out)


def align_check(r: pd.DataFrame, interval: str) -> list[dict]:
    """ok = 최대 상관이 lag 0 이고 옆 봉 상관이 lag0 의 SIDE_RATIO_MAX 미만. 데이터 없음 = ok None(n/a).
    필수 쌍(REQUIRED_CHECKS)이 n/a 거나 실패면 analyze 가 전체를 실패로 본다 -- 조용히 건너뛰고 통과하지 않게."""
    res = []
    req = {f"{a}~{b}" for a, b in REQUIRED_CHECKS}
    for a, b in ALIGN_CHECKS:
        pair = f"{a}~{b}"
        cs = {k: (r[a].corr(r[b].shift(k)) if a in r and b in r else np.nan) for k in range(-3, 4)}
        if not np.isfinite(cs[0]):
            res.append(dict(interval=interval, pair=pair, validated=a, required=pair in req, best_lag=None, corr_lag0=np.nan,
                            corr_lagp1=np.nan, corr_lagm1=np.nan, side_ratio=np.nan, ok=None))
            continue
        best = max(cs, key=lambda k: abs(cs[k]) if np.isfinite(cs[k]) else -1)
        side = max(abs(cs[1]) if np.isfinite(cs[1]) else 0, abs(cs[-1]) if np.isfinite(cs[-1]) else 0) / max(abs(cs[0]), 1e-12)
        res.append(dict(interval=interval, pair=pair, validated=a, required=pair in req, best_lag=int(best), corr_lag0=cs[0],
                        corr_lagp1=cs[1], corr_lagm1=cs[-1], side_ratio=side, ok=bool(best == 0 and side < SIDE_RATIO_MAX)))
    return res


def alignment_verdict(checks: list[dict]) -> tuple[bool, set]:
    """(필수 쌍 전부 ok, 정렬 의심 시리즈 집합)."""
    req_ok = all(c["ok"] is True for c in checks if c["required"])
    suspect = {c["validated"] for c in checks if not c["required"] and c["ok"] is False}
    return req_ok, suspect


def oos(r: pd.DataFrame, sess: pd.Series, interval: str, macros: list[str]) -> pd.DataFrame:
    """코인_t ~ 매크로_{t-1..t-3}, 시간 앞 70% 학습 / 뒤 30% 검증. 상위 10% |예측| 봉의 부호 적중·평균 bp(비용 전)."""
    rows = []
    for tgt in TARGETS:
        for x in macros:
            if x not in r:
                continue
            X = pd.concat({f"l{k}": r[x].shift(k) for k in (1, 2, 3)}, axis=1)
            for s in ("us_cash", "all"):
                m = (sess == s).values if s != "all" else np.ones(len(sess), bool)
                d = pd.concat([r[tgt].rename("y"), X], axis=1)[m].dropna()
                if len(d) < 1000:
                    continue
                cut = int(len(d) * 0.7)
                tr, te = d.iloc[:cut], d.iloc[cut:]
                A = np.c_[np.ones(len(tr)), tr[["l1", "l2", "l3"]].values]
                beta = np.linalg.lstsq(A, tr["y"].values, rcond=None)[0]
                pred = np.c_[np.ones(len(te)), te[["l1", "l2", "l3"]].values] @ beta
                y = te["y"].values
                r2 = 1 - ((y - pred) ** 2).sum() / ((y - tr["y"].mean()) ** 2).sum()
                q = np.quantile(np.abs(A @ beta), 0.9)  # 문턱은 학습 구간 예측에서 -- 검증 분포를 미리 보지 않는다
                top = np.abs(pred) >= q
                rows.append(dict(interval=interval, target=tgt, macro=x, session=s, n_train=len(tr), n_test=len(te),
                                 beta_l1=beta[1], r2_os_pct=100 * r2,
                                 top10_hit=float(np.mean(np.sign(pred[top]) == np.sign(y[top]))),
                                 top10_mean_signed_bp=float(np.mean(np.sign(pred[top]) * y[top])), top10_n=int(top.sum())))
    return pd.DataFrame(rows)


def event_study(r: pd.DataFrame, interval: str, macros: list[str], H: int = 12, sig: float = 4.0) -> pd.DataFrame:
    """매크로 봉 수익 |r| > sig·(직전 288봉 표준편차) 인 봉 = 사건 봉 0. sign(r_macro)·코인 누적수익을 봉 -3..H 로."""
    rows = []
    for x in macros:
        if x not in r:
            continue
        # 유효 봉끼리 굴린다(정규장 전용 ETF는 격자 288행 안에 유효 봉이 78개뿐이라 격자 기준이면 영영 NaN)
        sd = r[x].dropna().rolling(288, min_periods=100).std().shift(1).reindex(r.index)
        pos = np.flatnonzero((r[x].abs() > sig * sd).fillna(False).values)
        keep, last = [], -10 ** 9
        for q in pos:  # 겹치는 사건 제거: 앞 사건에서 H 봉 이내면 버린다(t 과대평가 방지)
            if q - last > H:
                keep.append(q)
                last = q
        ev = r.index[keep]
        for tgt in TARGETS:
            path = []
            for t in ev:
                i = r.index.get_loc(t)
                if i - 3 < 0 or i + H >= len(r):
                    continue
                seg = r[tgt].iloc[i - 3: i + H + 1].values * np.sign(r[x].iloc[i])
                if np.isnan(seg).any():
                    continue
                path.append(seg)
            if len(path) < 5:
                continue
            P = np.array(path)
            cum = np.cumsum(P[:, 3:], axis=1)  # 봉 0 부터 누적
            row = dict(interval=interval, macro=x, target=tgt, n_events=len(P),
                       pre_3bars_bp=float(P[:, :3].sum(axis=1).mean()), bar0_bp=float(cum[:, 0].mean()))
            for h in (1, 3, 6, H):
                row[f"after_bar1_to_{h}_bp"] = float((cum[:, h] - cum[:, 0]).mean())
                row[f"after_bar1_to_{h}_t"] = float((cum[:, h] - cum[:, 0]).mean() / ((cum[:, h] - cum[:, 0]).std(ddof=1) / np.sqrt(len(P)) + 1e-12))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmap(ll: pd.DataFrame, path: Path, interval: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = ll[(ll.interval == interval) & (ll.target == "BTC_PERP") & (ll.session == "us_cash")]
    if g.empty:
        return
    piv = g.pivot(index="macro", columns="lag", values="corr")
    piv = piv.loc[piv[0].abs().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.0 + 0.45 * piv.shape[1], 0.32 * piv.shape[0] + 1.5))
    v = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_xlabel(f"lag ({interval} bars; >0 = macro leads BTC)")
    ax.set_title(f"BTC perp vs macro, US cash session, {interval}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def analyze(pxs: dict[str, pd.DataFrame], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    all_ll, checks, oo, ev, cov = [], [], [], [], {}
    key_macros = [m for m in ("ES", "NQ", "RTY", "XYZ100", "ZN", "ZT", "DXY", "6J", "6E", "GC", "CL", "VIX", "IBIT", "HYG", "USDJPY", "TNX")]
    for iv, px in pxs.items():
        missing = [t for t in TARGETS if t not in px or px[t].notna().sum() == 0]
        if missing:
            raise SystemExit(f"{iv}: 코인 대상 시리즈 없음 {missing} -- 바이낸스 수집 실패(fetch_failures_{iv}.json 확인)")
        r = to_returns(px, iv)
        sess = sessions(r.index)
        if "SPY" in r:  # 미국 휴장일(평일인데 SPY 봉이 하루 종일 없음)은 us_cash 가 아니다
            et_day = r.index.tz_convert("America/New_York").date
            spy_days = set(pd.Series(et_day, index=r.index)[r["SPY"].notna().values])
            holiday = (sess.values == "us_cash") & ~pd.Series(et_day).isin(spy_days).values
            sess = sess.where(~holiday, "macro_closed")
        cov[iv] = {c: [str(px[c].first_valid_index()), str(px[c].last_valid_index()), int(px[c].notna().sum())]
                   for c in px.columns if px[c].notna().any()}
        checks += align_check(r, iv)
        ll = leadlag(r, sess, iv)
        all_ll.append(ll)
        oo.append(oos(r, sess, iv, key_macros))
        ev.append(event_study(r, iv, key_macros, H=12 if iv == "5m" else 30))
        plot_heatmap(ll, out / f"heatmap_btc_us_cash_{iv}.png", iv)
    ll = pd.concat(all_ll)
    sm = summarize_ll(ll)
    oo, ev = pd.concat(oo), pd.concat(ev)
    if ev.empty:
        ev = pd.DataFrame(columns=["interval", "macro", "target", "n_events"])
    ll.to_csv(out / "leadlag_full.csv", index=False, float_format="%.5f")
    sm.to_csv(out / "leadlag_summary.csv", index=False, float_format="%.4f")
    oo.to_csv(out / "oos_lagged_regression.csv", index=False, float_format="%.4f")
    ev.to_csv(out / "event_study.csv", index=False, float_format="%.3f")
    n_tests = int(len(ll[ll.lag != 0]))
    bonf_z = float(-NormalDist().inv_cdf(0.025 / max(n_tests, 1)))
    req_ok, suspect = alignment_verdict(checks)
    meta = dict(coverage=cov, alignment_checks=checks, alignment_required_ok=req_ok, alignment_suspect=sorted(suspect),
                n_lag_tests=n_tests, bonferroni_z_5pct=bonf_z,
                notes=["corr_lag0 = 동시 상관; best_macro_lead>0 의 강건 |z| 가 bonferroni_z 를 넘어야 선행 주장 가능",
                       "z 는 이분산 강건(White). z_naive=r·√n 은 비교용(공통 변동성 군집에서 과대)",
                       "세션은 코인 봉 t 기준 -- 큰 k 에선 정규장 행이 개장 전 매크로 봉과 짝지어질 수 있다",
                       "XYZ100 은 HL REST 5000봉 한도: 5m ~17일, 1m ~3.5일만",
                       "Yahoo CME 선물은 라이브 ~10분 지연: 1~2봉 선행은 Yahoo 피드로는 라이브 불가",
                       "top10_mean_signed_bp 는 비용 전 -- 바이낸스 테이커 왕복 수수료보다 작으면 거래 불가"])
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str))
    write_md(sm, oo, ev, meta, out)
    return meta


def write_md(sm, oo, ev, meta, out: Path) -> None:
    L = ["# 매크로 ↔ 코인 분 단위 리드-래그 (자동 생성)", "",
         "연구용 진단이다 -- fresh-forward 승격 근거 아님. 생성: `scripts/research_macro_crypto_intraday_leadlag_20261027.py`.", "",
         "## 타임스탬프 정렬 자가검증 (best_lag 가 0 이어야 함)", "",
         f"OK = 최대 상관이 lag 0 이고 옆 봉/lag0 < {SIDE_RATIO_MAX}. 필수 쌍 실패·누락 시 실행이 exit 2 로 끝난다.", "",
         "| 주기 | 쌍 | 필수 | best_lag | lag0 | lag+1 | lag−1 | 옆봉비 | OK |", "|---|---|---|---|---|---|---|---|---|"]
    for c in meta["alignment_checks"]:
        okc = "n/a(데이터 없음)" if c["ok"] is None else ("✅" if c["ok"] else "❌ 정렬 의심")
        L.append(f"| {c['interval']} | {c['pair']} | {'●' if c['required'] else ''} | {c['best_lag']} | {c['corr_lag0']:.3f} | "
                 f"{c['corr_lagp1']:.3f} | {c['corr_lagm1']:.3f} | {c['side_ratio']:.3f} | {okc} |")
    L += ["", f"선행 검정 {meta['n_lag_tests']}개 → Bonferroni 5% 기준 |z| > **{meta['bonferroni_z_5pct']:.2f}**", ""]
    for iv in ("5m", "1m"):
        for s in ("us_cash", "overnight"):
            g = sm[(sm.interval == iv) & (sm.target == "BTC_PERP") & (sm.session == s)].copy()
            if g.empty:
                continue
            g = g.reindex(g["corr_lag0"].abs().sort_values(ascending=False).index)
            L += [f"## BTC 선물 vs 매크로 — {iv}, {s}", "",
                  "| 매크로 | n | 동시 상관 | 최선 선행(봉) | 그 상관 | 강건 z | lag+1 | lag−1(코인 선행) | LLR |", "|---|---|---|---|---|---|---|---|---|"]
            for _, x in g.iterrows():
                flag = " ⭐" if abs(x["z_best_lead"]) > meta["bonferroni_z_5pct"] else ""
                flag += " ⚠정렬의심" if x["macro"] in meta["alignment_suspect"] else ""
                L.append(f"| {x['macro']} | {x['n']} | {x['corr_lag0']:.3f} | {x['best_macro_lead']:.0f} | {x['corr_best_lead']:.3f} | "
                         f"{x['z_best_lead']:.1f}{flag} | {x['corr_lag1']:.3f} | {x['corr_lagm1']:.3f} | {x['lead_lag_ratio']:.2f} |")
            L.append("")
    if not oo.empty:
        g = oo[oo.target == "BTC_PERP"].sort_values("r2_os_pct", ascending=False).head(15)
        L += ["## 표본외: BTC 선물_t ~ 매크로_{t-1..t-3} (앞 70% 학습/뒤 30% 검증)", "",
              "| 주기 | 매크로 | 세션 | R²_OS % | 상위10% 적중 | 상위10% 평균 bp(비용 전) | n |", "|---|---|---|---|---|---|---|"]
        for _, x in g.iterrows():
            L.append(f"| {x['interval']} | {x['macro']} | {x['session']} | {x['r2_os_pct']:.2f} | {x['top10_hit']:.3f} | {x['top10_mean_signed_bp']:.2f} | {x['top10_n']} |")
        L.append("")
    if not ev.empty:
        g = ev[ev.target == "BTC_PERP"]
        L += ["## 매크로 급변(4σ) 이벤트 뒤 BTC 선물 반응 (매크로 방향 부호 맞춤, bp)", "",
              "| 주기 | 매크로 | 사건 수 | 직전 3봉 | 같은 봉 | 다음 1봉 | 다음 1~6봉 | t(1~6) |", "|---|---|---|---|---|---|---|---|"]
        for _, x in g.iterrows():
            L.append(f"| {x['interval']} | {x['macro']} | {x['n_events']} | {x['pre_3bars_bp']:.1f} | {x['bar0_bp']:.1f} | "
                     f"{x['after_bar1_to_1_bp']:.1f} | {x['after_bar1_to_6_bp']:.1f} | {x['after_bar1_to_6_t']:.1f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- selftest
def selftest() -> None:
    """합성: BTC_t = 0.4·ES_t + 0.25·ES_{t-1} + 잡음 (미국 정규장만 ES 존재). 선행 lag+1 을 되찾고, 정렬 검사가 통과해야 한다."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-08-03", periods=288 * 40, freq="5min", tz="UTC")
    sess = sessions(idx)
    es_r = rng.normal(0, 8, len(idx))
    live = (sess != "macro_closed").values
    es_r[~live] = 0.0
    cash_i = np.flatnonzero((sess == "us_cash").values)
    es_r[rng.choice(cash_i[300:], 30, replace=False)] += rng.choice([-1, 1], 30) * 80  # 10σ 점프 -- 이벤트 경로 검증용
    btc_r = 0.4 * es_r + 0.25 * np.r_[0, es_r[:-1]] + rng.normal(0, 25, len(idx))
    px = pd.DataFrame({"ES": 5000 * np.exp(np.cumsum(es_r) / 1e4), "BTC_PERP": 6e4 * np.exp(np.cumsum(btc_r) / 1e4)}, index=idx)
    px.loc[~live, "ES"] = np.nan  # 주말엔 선물 봉 없음
    px["SPY"] = px["ES"] / 10 * np.exp(rng.normal(0, 0.5, len(idx)) / 1e4)
    px["BTC_SPOT"] = px["BTC_PERP"] * np.exp(rng.normal(0, 1, len(idx)) / 1e4)
    px["IBIT"] = (px["BTC_SPOT"] / 1500 * np.exp(rng.normal(0, 2, len(idx)) / 1e4)).where(sess.values == "us_cash")
    px["ETH_PERP"] = px["BTC_PERP"] / 25
    px["GC"] = 2000 * np.exp(np.cumsum(rng.normal(0, 5, len(idx))) / 1e4)
    out = Path(tempfile.mkdtemp(prefix="macro_intraday_selftest_"))
    meta = analyze({"5m": px}, out)
    sm = pd.read_csv(out / "leadlag_summary.csv")
    es = sm[(sm.target == "BTC_PERP") & (sm.macro == "ES") & (sm.session == "us_cash")].iloc[0]
    gc = sm[(sm.target == "BTC_PERP") & (sm.macro == "GC") & (sm.session == "us_cash")].iloc[0]
    assert es.best_macro_lead == 1 and es.z_best_lead > meta["bonferroni_z_5pct"], es.to_dict()
    assert es.lead_lag_ratio > 2, es.lead_lag_ratio
    assert abs(gc.z_best_lead) < meta["bonferroni_z_5pct"], gc.to_dict()
    assert meta["alignment_required_ok"] and not meta["alignment_suspect"], meta["alignment_checks"]
    ev = pd.read_csv(out / "event_study.csv")
    e = ev[(ev.macro == "ES") & (ev.target == "BTC_PERP")]
    assert len(e) and e.iloc[0].n_events >= 20 and e.iloc[0].after_bar1_to_1_bp > 0, e.to_dict("records")
    # 정렬 트립와이어: SPY 를 한 봉 밀면 잡아야 한다
    px2 = px.copy()
    px2["SPY"] = px2["SPY"].shift(1)
    bad = align_check(to_returns(px2, "5m"), "5m")
    assert not [c for c in bad if c["pair"] == "ES~SPY"][0]["ok"]
    # 봉 미만 어긋남(Yahoo 가 ~1분 늦게 찍힘 ≈ 이번 봉 75% + 직전 봉 25%)도 필수 쌍에서 잡아야 한다
    r3 = to_returns(px, "5m")
    r3["IBIT"] = 0.75 * r3["IBIT"] + 0.25 * r3["IBIT"].shift(1)
    ok3, _ = alignment_verdict(align_check(r3, "5m"))
    assert not ok3, "sub-bar offset not caught"
    # 필수 쌍 데이터가 없으면(IBIT 다운로드 실패) 조용히 통과하지 않는다
    ok4, _ = alignment_verdict(align_check(to_returns(px.drop(columns=["IBIT"]), "5m"), "5m"))
    assert not ok4, "missing required pair passed"
    # 공통 변동성 군집 + t(4) 꼬리, 선행 없음: 강건 z 의 sd 가 ~1 이어야 한다(r·√n 은 부풀어야 정상)
    vol = np.exp(np.convolve(rng.normal(0, 0.12, len(idx) + 287), np.ones(288) / np.sqrt(288) * 1.0, "valid"))
    m_r = vol * rng.standard_t(4, len(idx)) * 6
    c_r = 0.3 * m_r + vol * rng.standard_t(4, len(idx)) * 20
    rr = pd.DataFrame({"m": m_r, "c": c_r}, index=idx)
    zs, zn = [], []
    for k in range(1, 40):
        dd = pd.concat([rr["c"], rr["m"].shift(k)], axis=1).dropna()
        aa, bb = (dd.iloc[:, 0] - dd.iloc[:, 0].mean()).values, (dd.iloc[:, 1] - dd.iloc[:, 1].mean()).values
        zs.append((aa * bb).sum() / np.sqrt(((aa * bb) ** 2).sum()))
        zn.append(np.corrcoef(aa, bb)[0, 1] * np.sqrt(len(dd)))
    assert 0.6 < np.std(zs) < 1.4, np.std(zs)
    print("selftest OK:", {k: round(float(es[k]), 3) for k in ("corr_lag0", "corr_best_lead", "z_best_lead", "lead_lag_ratio")},
          "bonferroni_z", round(meta["bonferroni_z_5pct"], 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="캐시(close_*.parquet)로 분석만")
    ap.add_argument("--intervals", default="5m,1m")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    pxs = {}
    for iv in a.intervals.split(","):
        pxs[iv] = pd.read_parquet(CACHE / f"close_{iv}.parquet") if a.no_fetch else fetch_all(iv)
    meta = analyze(pxs, OUT)
    print(json.dumps(meta["alignment_checks"], indent=1, default=str))
    print(f"-> {OUT / 'summary.md'}")
    if meta["alignment_suspect"]:
        print(f"⚠️ 정렬 의심 시리즈(선행 주장 금지): {meta['alignment_suspect']}", file=sys.stderr)
    if not meta["alignment_required_ok"]:
        print("⚠️ 필수 정렬 검사(IBIT~BTC 현물·선물~현물) 실패 또는 누락 -- 리드-래그 결과를 믿기 전에 원인부터 본다", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
