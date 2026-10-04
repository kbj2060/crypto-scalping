"""ETH 모의 판 trend4·trend4_vs·poi1h 를 SOL·XRP 에서 검정하고 엔진 상수(POI_DOWN_BP·ER_REF_BP·PULLBACK_BP)를 코인별로 잰다 (2026-10-04).

사전등록·결과: docs/experiments/trend4_poi1h_solxrp_20261004.md · 산출: tmp/trend_poi_solxrp/verdict.json
판정 논리는 엔진(scripts/rl_1s_agent.py)의 trend4_signal·ArmPolicy 를 직접 불러 벡터 구현과 포지션 일치를 assert 한다.
원천: data/binance_vision/panel/{SYM}.parquet(5분, 2022-01-01~2026-09-14) · data.binance.vision 정적 아카이브(펀딩·1분봉·현물 1d).
🔴fapi/api.binance.com REST 는 쓰지 않는다(운영 서버와 공인 IP 공유).

  python scripts/research_trend4_poi1h_solxrp_20261004.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = next(p for p in (ROOT / "data", ROOT.parents[2] / "data") if (p / "binance_vision/panel").exists())
OUT = ROOT / "tmp/trend_poi_solxrp"
RAW = OUT / "raw"
VISION = "https://data.binance.vision/data"
COINS = ("ETHUSDT", "SOLUSDT", "XRPUSDT")
MAIN, SWITCH = pd.Timestamp("2025-01-01"), pd.Timestamp("2024-03-04")   # 주 판정 시작 · metrics OI 스탬프 규약 전환
TAKER = 4e-4
PB_WIN = {"0920_0926": ("2026-09-20", "2026-09-27"), "0927_1003": ("2026-09-27", "2026-10-04")}
RNG = np.random.default_rng(20261004)


def engine():
    s = importlib.util.spec_from_file_location("rl_engine", ROOT / "scripts/rl_1s_agent.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


ENG = engine()


def fetch(path: str) -> Path:
    """data.binance.vision 정적 파일(없을 때만 받는다). 404 면 None."""
    f = RAW / path                                   # 시장별 경로 그대로(현물·선물 1d 파일 이름이 같다)
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        try:
            f.write_bytes(urllib.request.urlopen(f"{VISION}/{path}", timeout=30).read())
        except Exception:  # noqa: BLE001 -- 아직 안 올라온 달
            return None
    return f


def months(a: str, b: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(a, b, freq="M")]


def load_panel(sym: str) -> pd.DataFrame:
    d = pd.read_parquet(DATA / f"binance_vision/panel/{sym}.parquet", columns=["timestamp", "close", "sum_open_interest"])
    d = d.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    assert (d.index.to_series().diff().dropna() == pd.Timedelta("5min")).all() and d.close.notna().all()
    return d


def oi_at_close(d: pd.DataFrame, extra_lag: int = 0) -> pd.Series:
    """봉 [t,t+5) 마감 시각의 OI. 스탬프 규약: 2024-03-04 이후 행 t = t+5분 스냅샷, 그 전 행 t = t 스냅샷(아래 stamp_scan 으로 코인마다 확인)."""
    oi = d.sum_open_interest.where(d.sum_open_interest > 0)
    snap_t = np.where(d.index >= SWITCH, d.index + pd.Timedelta("5min"), d.index)
    snap = pd.Series(oi.to_numpy(), index=pd.DatetimeIndex(snap_t))
    snap = snap[~snap.index.duplicated(keep="last")]
    return pd.Series(snap.reindex(d.index + pd.Timedelta("5min")).to_numpy(), index=d.index).shift(extra_lag)


def stamp_scan(d: pd.DataFrame) -> dict:
    """원 패널 ΔOI 가 직전 봉(rm1)·같은 봉(r0)·다음 봉(r1) 수익 중 어디와 붙는가 -- 전환 전후 각 90일."""
    r = np.log(d.close).diff()
    o = np.log(d.sum_open_interest.where(d.sum_open_interest > 0)).diff()
    out = {}
    for nm, (a, b) in {"pre": (SWITCH - pd.Timedelta("90D"), SWITCH), "post": (SWITCH, SWITCH + pd.Timedelta("90D"))}.items():
        m = (d.index >= a) & (d.index < b)
        f = pd.DataFrame({"o": o, "rm1": r.shift(1), "r0": r, "r1": r.shift(-1)})[m].replace([np.inf, -np.inf], np.nan).dropna()
        out[nm] = {k: round(float(f.o.corr(f[k])), 3) for k in ("rm1", "r0", "r1")}
    return out


def consts(d: pd.DataFrame, sym: str) -> dict:
    lp = np.log(d.close) * 1e4
    r = lp.diff()
    k = pd.concat([pd.read_csv(fetch(f"futures/um/daily/klines/{sym}/1m/{sym}-1m-{x:%Y-%m-%d}.zip"))
                   for x in pd.date_range("2026-09-20", "2026-10-03")])
    k.index = pd.to_datetime(k.open_time, unit="ms") + pd.Timedelta("1min")      # 1분봉 마감 시각
    k5 = np.log(k.close.sort_index()) * 1e4
    k5 = k5 - k5.shift(5)                                                          # 5분 이동, 1분 간격(원 정의 = 1초 간격 mid)
    pb = {w: round(float(np.median(np.abs(k5[(k5.index > a) & (k5.index <= b)].dropna()))), 2) for w, (a, b) in PB_WIN.items()}
    return {"POI_DOWN_BP": round(float(np.nanquantile(lp - lp.shift(12), 0.25)), 2),
            "ER_REF_BP": round(float(r.rolling(48).std().median()), 2),
            "PULLBACK_BP": pb["0920_0926"], "PULLBACK_BP_by_window": pb}


def funding_daily(sym: str, days: pd.DatetimeIndex) -> pd.Series:
    """그날(00:00 초과 ~ 다음 00:00 이하) 정산된 펀딩률 합 -- 그날 보유 포지션이 낸다."""
    fs = [fetch(f"futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip") for m in months("2022-01", "2026-09")]
    f = pd.concat([pd.read_csv(x) for x in fs if x is not None])
    t = pd.to_datetime(f.calc_time, unit="ms") - pd.Timedelta("1s")
    return f.last_funding_rate.groupby(t.dt.floor("D").to_numpy()).sum().reindex(days).fillna(0.0)


# ── 추세 판 ────────────────────────────────────────────────────────────────────
def trend_positions(c: pd.Series) -> pd.DataFrame:
    """일봉 종가 c(날짜 인덱스, 그날 00:00 UTC~24:00 마감) -> 날 d 의 포지션(d 시작 때 c[..d-1] 로 결정)."""
    rows, pos_vs, k = [], 0, 0.0
    vals = c.to_numpy()
    for i in range(len(c)):
        sig, size = ENG.trend4_signal(list(vals[max(0, i - 40):i]))   # 엔진: 직전 40개 완결 일봉
        t = int(np.sign(sig))
        if t and t != pos_vs:                                          # 엔진: 새 진입 때만 k 갱신
            k = abs(size)
        pos_vs = t
        rows.append((sig, size, t, t * k if t else 0.0))
    return pd.DataFrame(rows, index=c.index, columns=["sig", "size", "trend4", "trend4_vs"])


def daily_stats(pnl: pd.Series) -> dict:
    p = pnl.dropna()
    sd = p.std(ddof=1)
    return {"n_days": int(len(p)), "sharpe": round(float(p.mean() / sd * np.sqrt(365)), 3) if sd > 0 else None,
            "t": round(float(p.mean() / sd * np.sqrt(len(p))), 2) if sd > 0 else None,
            "ann_ret_pct": round(float(p.mean() * 365 * 100), 1),
            "mdd_pct": round(float(((1 + p).cumprod() / (1 + p).cumprod().cummax() - 1).min() * 100), 1),
            "years": {int(y): round(float(v * 100), 1) for y, v in p.groupby(p.index.year).sum().items()}}


def book(pos: pd.Series, ret: pd.Series, fund: pd.Series | None, fee: float = TAKER) -> pd.Series:
    """일 손익 = 포지션 × 수익 − 펀딩 × 포지션 − 수수료 × |노출 변화|."""
    pnl = pos * ret - fee * pos.diff().abs().fillna(pos.abs())
    return pnl - (pos * fund if fund is not None else 0.0)


def trend_block(c: pd.Series, fund: pd.Series) -> dict:
    ret = c.pct_change()
    P = trend_positions(c)
    vol = np.log(c).diff().rolling(20).std().shift(1) * np.sqrt(365)
    arms = {"trend4": P.trend4.astype(float), "trend4_vs": P.trend4_vs,
            "always_long": pd.Series(1.0, index=c.index), "vol_long": (0.5 / vol).clip(upper=2.0)}
    start = P.index[29]                                                        # 엔진 trend4_signal 이 일봉 29개부터 계산
    out = {"start": str(start.date()), "end": str(c.index[-1].date())}
    for nm, pos in arms.items():
        pos = pos[pos.index >= start].fillna(0.0)
        r, f = ret[pos.index], fund[pos.index]
        pnl, gross = book(pos, r, f), book(pos, r, None, 0.0)
        out[nm] = {per: {"net": daily_stats(pnl[m]), "gross": daily_stats(gross[m])}
                   for per, m in (("main", pnl.index >= MAIN), ("aux", pnl.index < MAIN), ("all", pnl.index >= start))}
    return out


def trend_verdict(t: dict) -> dict:
    v = {}
    for arm, ctl in (("trend4", "always_long"), ("trend4_vs", "always_long")):
        m, c = t[arm]["main"]["net"], t[ctl]["main"]["net"]
        yrs = m["years"]
        v[arm] = {"sharpe_main": m["sharpe"], "t_main": m["t"], "always_long_sharpe_main": c["sharpe"],
                  "vol_long_sharpe_main": t["vol_long"]["main"]["net"]["sharpe"], "years_main": yrs,
                  "years_aux": t[arm]["aux"]["net"]["years"], "sharpe_aux": t[arm]["aux"]["net"]["sharpe"],
                  "pass": bool(m["sharpe"] > 0 and m["sharpe"] > c["sharpe"] and all(x > 0 for x in yrs.values()))}
    return v


# ── poi1h ─────────────────────────────────────────────────────────────────────
def poi_frame(d: pd.DataFrame, thr: float, oi_lag: int = 0) -> pd.DataFrame:
    """정시 마감 봉마다 r1h·o1h·목표·앞 1시간 수익(즉시 진입·한 봉 늦은 진입)."""
    lp = np.log(d.close.to_numpy()) * 1e4
    oi = oi_at_close(d, oi_lag).to_numpy()
    doi = np.r_[np.nan, (oi[1:] / oi[:-1] - 1) * 1e4]                       # 엔진 doi300
    n = len(lp)
    i = np.flatnonzero((d.index + pd.Timedelta("5min")).minute == 0)       # 정시에 닫히는 봉
    i = i[(i >= 12) & (i + 13 < n)]
    r1h = lp[i] - lp[i - 12]
    o1h = np.array([doi[j - 11:j + 1].sum() for j in i])                    # 결측 있으면 NaN -> 관망
    tgt = np.where(r1h <= thr, np.where(o1h < 0, 1, np.where(o1h > 0, -1, 0)), 0)
    f = pd.DataFrame({"t": d.index[i] + pd.Timedelta("5min"), "r1h": r1h, "o1h": o1h, "tgt": tgt,
                      "fwd": lp[i + 12] - lp[i], "fwd_lag1": lp[i + 13] - lp[i + 1]})
    f["day"] = f.t.dt.floor("D")
    return f


def boot_mean(v: np.ndarray, day: np.ndarray, n: int = 3000) -> list[float]:
    codes, uniq = pd.factorize(day)
    s, c = np.bincount(codes, weights=v, minlength=len(uniq)), np.bincount(codes, minlength=len(uniq))
    pick = RNG.integers(0, len(uniq), size=(n, len(uniq)))
    return [round(float(x), 2) for x in np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1), [2.5, 97.5])]


def boot_diff(va, da, vb, db, n: int = 3000) -> list[float]:
    days = np.union1d(da, db)
    ia, ib = np.searchsorted(days, da), np.searchsorted(days, db)
    sa, ca = np.bincount(ia, weights=va, minlength=len(days)), np.bincount(ia, minlength=len(days))
    sb, cb = np.bincount(ib, weights=vb, minlength=len(days)), np.bincount(ib, minlength=len(days))
    pick = RNG.integers(0, len(days), size=(n, len(days)))
    A = sa[pick].sum(1) / np.maximum(ca[pick].sum(1), 1)
    B = sb[pick].sum(1) / np.maximum(cb[pick].sum(1), 1)
    return [round(float(x), 2) for x in np.percentile(A - B, [2.5, 97.5])]


def poi_stats(f: pd.DataFrame, thr: float) -> dict:
    e = f[f.tgt != 0]
    g = e.tgt * e.fwd
    pos = f.tgt.to_numpy()
    cost = TAKER * 1e4 * np.abs(np.diff(np.r_[0, pos]))                    # 연속 같은 방향 시간은 비용 없음(엔진: 정시마다 목표 유지)
    cost += TAKER * 1e4 * np.abs(pos) * (np.r_[pos[1:], 0] == 0)            # 다음 시간 관망이면 청산 비용
    dn = f[f.r1h <= thr]
    a, b = dn[dn.o1h < 0], dn[dn.o1h > 0]
    return {"n_events": int(len(e)), "n_days": int(e.day.nunique()), "long_share": round(float((e.tgt > 0).mean()), 3),
            "gross_mean_bp": round(float(g.mean()), 2), "gross_ci": boot_mean(g.to_numpy(), e.day.to_numpy()),
            "net_taker_mean_bp": round(float((g.sum() - cost.sum()) / max(len(e), 1)), 2),
            "lag1_entry_mean_bp": round(float((e.tgt * e.fwd_lag1).mean()), 2),
            "ctl_down_always_long_bp": round(float(dn.fwd.mean()), 2), "ctl_down_always_long_ci": boot_mean(dn.fwd.to_numpy(), dn.day.to_numpy()),
            "ctl_all_hours_long_bp": round(float(f.fwd.mean()), 2),
            "oi_down_fwd_bp": round(float(a.fwd.mean()), 2), "oi_up_fwd_bp": round(float(b.fwd.mean()), 2),
            "gap_oidown_minus_oiup": round(float(a.fwd.mean() - b.fwd.mean()), 2),
            "gap_ci": boot_diff(a.fwd.to_numpy(), a.day.to_numpy(), b.fwd.to_numpy(), b.day.to_numpy()),
            "years": {int(y): round(float(v), 2) for y, v in g.groupby(e.t.dt.year).mean().items()}}


def poi_block(d: pd.DataFrame, thr: float) -> dict:
    out = {}
    for tag, lag in (("base", 0), ("oi_lag1", 1)):
        f = poi_frame(d, thr, lag)
        out[tag] = {per: poi_stats(f[m], thr) for per, m in (("main", f.t >= MAIN), ("aux", f.t < MAIN))}
    m = out["base"]["main"]
    out["verdict"] = {"gross_mean_main": m["gross_mean_bp"], "gross_ci_main": m["gross_ci"], "net_taker_main": m["net_taker_mean_bp"],
                      "aux_gross": out["base"]["aux"]["gross_mean_bp"], "aux_ci": out["base"]["aux"]["gross_ci"],
                      "pass": bool(m["gross_ci"][0] > 0)}
    return out


def poi_original(d: pd.DataFrame) -> dict:
    """원 연구(research_eth_price_oi_quadrant_exit_20260920.py §B) 정의 그대로: 5분 간격·원 패널 OI·앞 수익은 다음 봉부터 1h."""
    px, oi = d.close.to_numpy(), d.sum_open_interest.to_numpy()
    h = 12
    dp = (np.log(px) - np.log(np.r_[np.full(h, np.nan), px[:-h]])) * 1e4
    do = (oi - np.r_[np.full(h, np.nan), oi[:-h]]) / np.r_[np.full(h, np.nan), oi[:-h]] * 1e4
    fw = (np.log(np.r_[px[h + 1:], np.full(h + 1, np.nan)]) - np.log(np.r_[px[1:], np.nan])) * 1e4
    ok = np.isfinite(dp) & np.isfinite(do) & np.isfinite(fw)
    down = ok & (dp < np.nanquantile(dp[ok], 0.25))
    a, b = down & (do < 0), down & (do > 0)
    day = d.index.floor("D").to_numpy()
    return {"oi_down": round(float(fw[a].mean()), 2), "oi_up": round(float(fw[b].mean()), 2),
            "gap": round(float(fw[a].mean() - fw[b].mean()), 2), "gap_ci": boot_diff(fw[a], day[a], fw[b], day[b])}


# ── ETH 현물 재현 ─────────────────────────────────────────────────────────────
def eth_spot_closes() -> pd.Series:
    fs = [fetch(f"spot/monthly/klines/ETHUSDT/1d/ETHUSDT-1d-{m}.zip") for m in months("2017-08", "2026-08")]
    k = pd.concat([pd.read_csv(x, header=None) for x in fs if x is not None])
    t = k[0].astype("int64")
    k.index = pd.to_datetime(np.where(t > 1e14, t // 1000, t), unit="ms")       # 2025~ 현물 아카이브는 마이크로초
    c = k[4].astype(float).sort_index()
    return c[~c.index.duplicated()]


def research_vs(c: pd.Series, fund: pd.Series | None, start: str | None = None) -> dict:
    """원 연구 집행: 신호 분수 그대로 × 0.5/σ20(±2) 매일 재조정, 테이커 4bp. start 앞 종가는 신호 예열용."""
    P = trend_positions(c)
    pos = P["size"]
    start = pd.Timestamp(start) if start else pos.index[29]
    ret = c.pct_change()
    pos, r = pos[pos.index >= start], ret[ret.index >= start]
    f = fund[pos.index] if fund is not None else None
    vol = (np.log(c).diff().rolling(20).std().shift(1) * np.sqrt(365))[pos.index]
    vl = (0.5 / vol).clip(upper=2.0)
    return {"start": str(start.date()), "end": str(c.index[-1].date()),
            "long_short": daily_stats(book(pos, r, f)), "long_cash": daily_stats(book(pos.clip(lower=0), r, f)),
            "vol_long": daily_stats(book(vl, r, f)), "always_long": daily_stats(book(pd.Series(1.0, index=pos.index), r, f))}


def parity_check(d: pd.DataFrame, P: pd.DataFrame, thr: float) -> None:
    """자체점검: 벡터 구현 = 엔진 ArmPolicy 판정 (poi1h 2주 · trend4/_vs 전 기간)."""
    seg = d[(d.index >= "2025-03-01") & (d.index < "2025-03-15")]
    oi = oi_at_close(d)[seg.index].to_numpy()
    lp = np.log(seg.close.to_numpy()) * 1e4
    x = np.zeros(7)
    ix = {"imb50": 0, "whale_z": 1, "retail_z": 2, "ret300": 3, "doi300": 4, "liq_long60": 5, "liq_short60": 6}
    ENG.POI_DOWN_BP = thr
    pol, eng_t = ENG.ArmPolicy("poi1h"), {}
    for j in range(1, len(seg)):
        x[3], x[4] = lp[j] - lp[j - 1], (oi[j] / oi[j - 1] - 1) * 1e4
        s = int((seg.index[j] + pd.Timedelta("5min")).timestamp()) - 1
        t = pol.bar(x, ix, s, 0, 0.0)
        if (s + 1) % 3600 == 0:
            eng_t[seg.index[j] + pd.Timedelta("5min")] = t
    f = poi_frame(d, thr).set_index("t").tgt
    common = [k for k in eng_t if k in f.index and k >= seg.index[0] + pd.Timedelta("65min")]
    assert len(common) > 300 and all(eng_t[k] == f[k] for k in common), "poi1h 벡터 구현 ≠ 엔진 ArmPolicy"
    for arm in ("trend4", "trend4_vs"):                                    # 엔진 판으로 같은 일봉을 돌려 포지션 비교
        pol, pos = ENG.ArmPolicy(arm), 0
        for day, row in P.iterrows():
            ENG.TREND.update(day=1, sig=row.sig, size=row["size"])
            t = pol.bar(np.zeros(7), ix, 0, pos, 0.0)
            exp = t * (pol.k if arm == "trend4_vs" else 1.0)
            assert abs(exp - row[arm]) < 1e-12, (arm, day, exp, row[arm])
            pos = t
    ENG.TREND.update(day=None, sig=0.0, size=0.0)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    R: dict = {"data_end": None, "notes": []}
    for sym in COINS:
        d = load_panel(sym)
        R["data_end"] = str(d.index[-1] + pd.Timedelta("5min"))
        c = d.close[(d.index + pd.Timedelta("5min")).normalize() == d.index + pd.Timedelta("5min")]   # 23:55 봉 = 일 종가
        c.index = c.index.normalize()
        days = c.index
        fund = funding_daily(sym, days)
        k = consts(d, sym)
        P = trend_positions(c)
        parity_check(d, P, k["POI_DOWN_BP"])
        t = trend_block(c, fund)
        p = poi_block(d, k["POI_DOWN_BP"])
        key = sym[:3].lower()
        R[key] = {"consts": k, "stamp_scan": stamp_scan(d), "trend_detail": t, **trend_verdict(t),
                  "poi1h": p["verdict"], "poi1h_detail": {kk: v for kk, v in p.items() if kk != "verdict"},
                  "poi_original_def": poi_original(d)}
        if sym == "ETHUSDT":   # 원 연구는 2022-01-01 부터 매매(신호는 그 전 일봉으로 예열) -- 패널이 01-01 시작이라 현물 종가로 예열
            sp = eth_spot_closes()
            cw = pd.concat([sp[(sp.index >= "2021-11-01") & (sp.index < c.index[0])], c])
            fw = funding_daily(sym, cw.index)
            R[key]["trend_research_vs_futures"] = research_vs(cw, fw, "2022-01-01")
            R[key]["trend_research_vs_futures_2204"] = research_vs(cw, fw, "2022-04-01")
            R[key]["trend_research_vs_futures_nowarmup"] = research_vs(c, fund)
        print(sym, json.dumps({kk: R[key][kk] for kk in ("consts", "trend4", "trend4_vs", "poi1h", "poi_original_def")}, ensure_ascii=False), flush=True)
    e = R["eth"]
    spot = research_vs(eth_spot_closes(), None)
    fut = e["trend_research_vs_futures"]["long_short"]["sharpe"]
    rep = {"POI_DOWN_BP": e["consts"]["POI_DOWN_BP"], "ER_REF_BP": e["consts"]["ER_REF_BP"], "PULLBACK_BP": e["consts"]["PULLBACK_BP"],
           "poi_gap_1h": e["poi_original_def"], "futures_vs_ls_sharpe": fut,
           "futures_always_long_vol_sharpe": e["trend_research_vs_futures"]["vol_long"]["sharpe"],
           "futures_2204_ls_sharpe": e["trend_research_vs_futures_2204"]["long_short"]["sharpe"], "spot": spot}
    rep["checks"] = {"POI_DOWN_BP": abs(rep["POI_DOWN_BP"] + 26.68) < 0.01, "ER_REF_BP": abs(rep["ER_REF_BP"] - 14.95) < 0.01,
                     "PULLBACK_BP": abs(rep["PULLBACK_BP"] - 7.22) <= 0.1, "poi_gap": abs(e["poi_original_def"]["gap"] - 3.52) <= 0.3,
                     "futures_sharpe": abs(fut - 0.62) <= 0.1, "spot_sharpe": abs(spot["long_short"]["sharpe"] - 0.99) <= 0.1}
    rep["pass"] = all(rep["checks"].values())
    R["eth_repro"] = rep
    print("eth_repro", json.dumps(rep, ensure_ascii=False, default=str), flush=True)
    (OUT / "verdict.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("->", OUT / "verdict.json")


if __name__ == "__main__":
    sys.exit(main())
