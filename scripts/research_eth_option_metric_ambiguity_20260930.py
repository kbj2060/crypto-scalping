#!/usr/bin/env python3
"""옵션 카드 지표의 «정의·측정·해석» 모호함 실측 (2026-09-30, 사용자 «모호함들을 모두 연구해줘»).

범위(다른 세션과 겹치지 않게): 체결 주체·블록 구조 · 신규/청산(ΔOI) · 딜러 포지션 정확도(메이커=딜러·커버율)는 뺀다.
여기서는 같은 원자료로 «정의를 바꾸면 화면 값이 얼마나 달라지나»만 잰다. 전부 2026 데이터(사용자 규칙).

  S1 역옵션 델타: 표준 BS 선도 델타(수집기·Deribit ticker 와 같음) vs 프리미엄 조정 델타(= BS − 옵션가/선도가).
     25Δ 행사가·RR/BF·DEX(보유자·딜러가정)·charm·체결 순델타가 얼마나 달라지나.
  S2 IV 원천·RR/BF 잡음: 최근접 행사가(수집기) vs 델타 보간 vs 고정 만기(7일·30일 분산 보간) · 체결 IV − mark_iv ·
     (live) bid/ask IV 스프레드.
  S3 max pain: 같은 날 스냅샷 간 흔들림 · 만기 마지막 24시간 · 코인 단위 정의와의 불일치 · 가까운 만기 OI 비중.
  S4 P/C: 미결제 수량 · 행사가 명목 · 프리미엄 · 거래량 · 테이커 매수 · «약세 흐름» 비율.
  S5 1σ 띠 보정(5분·1시간·24시간·만기까지) 시간대별 · VRP 기간 불일치(rv7 vs rv30 vs 전방).
  S6 시차: 수집 주기 · DVOL 10분 변화 · rv7 6시간 캐시.
  S7 SOL·XRP 선형옵션: iv30 대용 잡음 · 행사가 수 · 스프레드.

입력: tmp/option_metric_ambiguity_20260930/server/*.parquet (서버 deribit_options.duckdb 에서 read_only COPY),
      Deribit 공개 API(DVOL 이력·ticker), 로컬 바이낸스 1분봉(~09-14). 🔴바이낸스 API 는 부르지 않는다.
출력: tmp/option_metric_ambiguity_20260930/results.json (+ 캐시 parquet)

  python scripts/research_eth_option_metric_ambiguity_20260930.py            # 전체
  python scripts/research_eth_option_metric_ambiguity_20260930.py --selftest # 합성 체인 assert(네트워크 없음)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/option_metric_ambiguity_20260930"
SRV = OUT / "server"
KLINES = Path("/home/kbj20/crypto-scalping/data/binance_vision/klines1m")
API = "https://www.deribit.com/api/v2/public/"
YEAR = 365.0 * 86400.0


# ── 공식 ─────────────────────────────────────────────────────────────────────
def bs(F, K, iv_pct, T, call):
    """r=0 선도 BS. (USD 가격, 표준 델타). 배열 입력."""
    F, K, s, T, call = map(np.asarray, (F, K, np.asarray(iv_pct) / 100.0, T, call))
    sq = np.maximum(s * np.sqrt(np.maximum(T, 1e-12)), 1e-12)
    d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
    c = F * ndtr(d1) - K * ndtr(d1 - sq)
    px = np.where(call, c, c - (F - K))
    return px, np.where(call, ndtr(d1), ndtr(d1) - 1.0)


def pa_delta(F, K, iv_pct, T, call):
    """프리미엄 조정 델타 = BS 델타 − 옵션가(코인) = d(가치_코인)/dS · S. 역옵션(ETH 로 결제) 쪽 규약."""
    px, d = bs(F, K, iv_pct, T, call)
    return d - px / np.asarray(F)


def max_pain(K, is_call, oi, coin=False):
    """미결제 전체의 보유자 내재가치 합이 최소가 되는 행사가(수집기 options_summary 와 같은 정의).
    coin=True 면 역옵션 결제 단위(ETH = USD 내재가치 / 결제가)로 잰다."""
    K, is_call, oi = map(np.asarray, (K, is_call, oi))
    ks = np.unique(K)
    pay = (np.clip(ks[:, None] - K[None, :], 0, None) * (oi * is_call)[None, :]).sum(1) \
        + (np.clip(K[None, :] - ks[:, None], 0, None) * (oi * ~is_call)[None, :]).sum(1)
    return float(ks[np.argmin(pay / ks if coin else pay)])


def brownian_touch_prob(k: float = 1.0) -> float:
    """연속 브라운 운동이 [0,1] 안에 ±kσ 를 한 번이라도 건드릴 확률."""
    inside = 4 / math.pi * sum((-1) ** n / (2 * n + 1) * math.exp(-((2 * n + 1) ** 2) * math.pi ** 2 / (8 * k * k)) for n in range(50))
    return 1 - inside


# ── 데이터 ───────────────────────────────────────────────────────────────────
def _get(method, **p):
    import requests
    for i in range(5):
        try:
            r = requests.get(API + method, params=p, timeout=30)
            r.raise_for_status()
            return r.json()["result"]
        except Exception:
            time.sleep(1 + i)
    raise RuntimeError(method)


def dvol_history(res: str, start: str, end: str) -> pd.Series:
    path = OUT / f"dvol_{res}_{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)["dvol"]
    t0, t1, rows = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000), int(pd.Timestamp(end, tz="UTC").timestamp() * 1000), []
    while True:
        r = _get("get_volatility_index_data", currency="ETH", resolution=res, start_timestamp=t0, end_timestamp=t1)
        rows += r["data"]
        if not r.get("continuation") or r["continuation"] <= t0:
            break
        t1 = r["continuation"]
        time.sleep(0.05)
    s = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"]).drop_duplicates("ts").sort_values("ts")
    out = pd.DataFrame({"dvol": s["c"].to_numpy()}, index=pd.to_datetime(s["ts"], unit="ms", utc=True))
    out.to_parquet(path)
    return out["dvol"]


def klines_1m() -> pd.DataFrame:
    fs = sorted(glob.glob(str(KLINES / "ETHUSDT-1m-2026-*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("t").sort_values("t")
    d.index = pd.to_datetime(d["t"], unit="ms", utc=True)
    d["o"] = d["c"].shift(1)                     # 파일에 시가 열이 없다 -- 직전 분 종가
    return d.dropna(subset=["o"])


def load_chain(cur="ETH") -> pd.DataFrame:
    c = pd.read_parquet(SRV / "chain.parquet", filters=[("currency", "==", cur)])
    c["recorded_at_utc"] = pd.to_datetime(c["recorded_at_utc"], utc=True)
    c["expiration_ts"] = pd.to_datetime(c["expiration_ts"], utc=True)
    c["T"] = c["days_to_expiry"] / 365.0
    c["is_call"] = c["option_type"] == "call"
    return c[(c["T"] > 0) & (c["mark_iv"] > 0) & (c["underlying_price"] > 0)]


# ── S1·S2·S3 만기별 표 ─────────────────────────────────────────────────────────
def _interp_at(x, y, x0):
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    if len(x) < 2 or x0 < x[0] or x0 > x[-1]:
        return np.nan
    return float(np.interp(x0, x, y))


def expiry_rows(chain: pd.DataFrame) -> pd.DataFrame:
    """스냅샷 × 만기 한 줄: 수집기 방식(최근접, BS 델타, 전 행사가) · 보간(BS/PA, 외가격만) · ATM(최근접/보간) · pain."""
    out = []
    for (ts, exp), g in chain.groupby(["recorded_at_utc", "expiration_ts"], sort=True):
        calls, puts = g[g["is_call"]], g[~g["is_call"]]
        if calls.empty or puts.empty:
            continue
        F, T = float(g["underlying_price"].iloc[0]), float(g["T"].iloc[0])
        k_atm = float(g.iloc[(g["strike"] - F).abs().argsort()[:1]]["strike"].iloc[0])
        atm_near = float(g[g["strike"] == k_atm]["mark_iv"].mean())
        cp_gap = float(g[g["strike"] == k_atm].groupby("is_call")["mark_iv"].mean().diff().abs().iloc[-1]) if g[g["strike"] == k_atm]["is_call"].nunique() == 2 else np.nan
        otm = pd.concat([calls[calls["strike"] >= F], puts[puts["strike"] < F]])
        atm_i = _interp_at(np.log(otm["strike"] / F), otm["mark_iv"], 0.0) if len(otm) > 1 else np.nan
        _, dc = bs(F, calls["strike"], calls["mark_iv"], T, True)
        _, dp = bs(F, puts["strike"], puts["mark_iv"], T, False)
        ic, ip = int(np.argmin(np.abs(dc - 0.25))), int(np.argmin(np.abs(dp + 0.25)))
        c25, p25 = float(calls["mark_iv"].iloc[ic]), float(puts["mark_iv"].iloc[ip])
        row = {"ts": ts, "exp": exp, "T_h": T * 8760, "F": F, "n_k": int(g["strike"].nunique()),
               "atm_near": atm_near, "atm_interp": atm_i, "cp_gap_atm": cp_gap,
               "rr_col": c25 - p25, "bf_col": (c25 + p25) / 2 - atm_near,
               "d_call_chosen": float(dc[ic]), "d_put_chosen": float(dp[ip]),
               "k_call_col": float(calls["strike"].iloc[ic]), "k_put_col": float(puts["strike"].iloc[ip])}
        oc, op = calls[calls["strike"] >= F], puts[puts["strike"] <= F]
        for name, fn in (("bs", lambda x, c: bs(F, x["strike"], x["mark_iv"], T, c)[1]), ("pa", lambda x, c: pa_delta(F, x["strike"], x["mark_iv"], T, c))):
            if len(oc) >= 2 and len(op) >= 2:
                dcc, dpp = fn(oc, True), fn(op, False)
                c_i, p_i = _interp_at(dcc, oc["mark_iv"], 0.25), _interp_at(dpp, op["mark_iv"], -0.25)
                row[f"rr_{name}"] = c_i - p_i
                row[f"bf_{name}"] = (c_i + p_i) / 2 - (atm_i if np.isfinite(atm_i) else atm_near)
                row[f"k_call_{name}"] = _interp_at(dcc, oc["strike"], 0.25)
                row[f"k_put_{name}"] = _interp_at(dpp, op["strike"], -0.25)
                # 최근접(외가격만) -- 델타 규약만 바꾼 수집기 방식
                jc, jp = int(np.argmin(np.abs(dcc - 0.25))), int(np.argmin(np.abs(dpp + 0.25)))
                row[f"rr_{name}_near"] = float(oc["mark_iv"].iloc[jc]) - float(op["mark_iv"].iloc[jp])
        row["pain"] = max_pain(g["strike"], g["is_call"], g["open_interest"])
        row["pain_coin"] = max_pain(g["strike"], g["is_call"], g["open_interest"], coin=True)
        row["coi"], row["poi"] = float(calls["open_interest"].sum()), float(puts["open_interest"].sum())
        ks = np.sort(g["strike"].unique())
        row["k_step_atm"] = float(np.min(np.abs(np.diff(ks)))) if len(ks) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def constant_maturity(er: pd.DataFrame, days: float) -> pd.DataFrame:
    """스냅샷마다 만기 둘 사이 보간: ATM = 총분산 선형 · RR/BF = 시간 선형."""
    rows = []
    tgt = days * 24
    for ts, g in er.groupby("ts"):
        g = g.sort_values("T_h")
        a, b = g[g["T_h"] <= tgt].tail(1), g[g["T_h"] >= tgt].head(1)
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        w = 0.0 if b["T_h"] == a["T_h"] else (tgt - a["T_h"]) / (b["T_h"] - a["T_h"])
        var = (1 - w) * a["atm_interp"] ** 2 * a["T_h"] + w * b["atm_interp"] ** 2 * b["T_h"]
        rows.append({"ts": ts, "atm": math.sqrt(var / tgt) if var > 0 else np.nan,
                     "rr": (1 - w) * a["rr_bs"] + w * b["rr_bs"], "bf": (1 - w) * a["bf_bs"] + w * b["bf_bs"]})
    return pd.DataFrame(rows).set_index("ts")


def noise_stats(s: pd.Series) -> dict:
    """1시간 간격으로 맞춘 뒤 변화의 크기·되돌림(음의 1차 자기상관 = 잡음)."""
    h = s.dropna().resample("1h").last().dropna()
    d = h.diff().dropna()
    return {"n": int(len(d)), "level_mean": round(float(h.mean()), 2), "level_sd": round(float(h.std()), 2),
            "chg_sd": round(float(d.std()), 2), "chg_abs_gt1pt": round(float((d.abs() > 1).mean()), 3),
            "chg_ac1": round(float(d.autocorr(1)), 3) if len(d) > 5 else None}


def s1_s2_s3(chain: pd.DataFrame, er: pd.DataFrame) -> dict:
    R: dict = {}
    front = er.loc[er.groupby("ts")["T_h"].idxmin()].set_index("ts").sort_index()
    # 수집기 front RR 은 «아직 안 끝난 가장 빠른 만기» -- 만기 직전 행 포함
    R["S1_delta_convention"] = {
        "note": "Deribit ticker greeks.delta 는 표준 BS 선도 델타(프리미엄 조정 아님)와 1e-5 안에서 일치(live 대조는 S2_live).",
        "rr_front_bs_minus_pa_pt": _q(front["rr_bs"] - front["rr_pa"]),
        "rr_7d_expiries_bs_minus_pa_pt": _q((er["rr_bs"] - er["rr_pa"])[(er["T_h"] > 120) & (er["T_h"] < 240)]),
        "rr_30d_expiries_bs_minus_pa_pt": _q((er["rr_bs"] - er["rr_pa"])[(er["T_h"] > 500) & (er["T_h"] < 1000)]),
        "bf_30d_expiries_bs_minus_pa_pt": _q((er["bf_bs"] - er["bf_pa"])[(er["T_h"] > 500) & (er["T_h"] < 1000)]),
        "put25_strike_shift_pct_30d": _q(((er["k_put_pa"] / er["k_put_bs"] - 1) * 100)[(er["T_h"] > 500) & (er["T_h"] < 1000)]),
        "call25_strike_shift_pct_30d": _q(((er["k_call_pa"] / er["k_call_bs"] - 1) * 100)[(er["T_h"] > 500) & (er["T_h"] < 1000)]),
        "put25_strike_shift_pct_front": _q((front["k_put_pa"] / front["k_put_bs"] - 1) * 100),
    }
    # 수집기 25Δ 최근접이 실제로 고른 델타
    R["S2_nearest_strike_delta_actually_chosen"] = {
        "front_call": _q(front["d_call_chosen"]), "front_put": _q(front["d_put_chosen"]),
        "front_call_off_by_gt_0.05": round(float((front["d_call_chosen"] - 0.25).abs().gt(0.05).mean()), 3),
        "front_put_off_by_gt_0.05": round(float((front["d_put_chosen"] + 0.25).abs().gt(0.05).mean()), 3),
        "front_rr_col_minus_interp_pt": _q(front["rr_col"] - front["rr_bs"]),
        "front_atm_near_minus_interp_pt": _q(front["atm_near"] - front["atm_interp"]),
        "call_put_mark_iv_gap_at_atm_strike_pt": _q(er["cp_gap_atm"]),
    }
    # RR/BF 잡음: 화면(가까운 만기·최근접) vs 보간 vs 고정 만기
    cm7, cm30 = constant_maturity(er, 7), constant_maturity(er, 30)
    R["S2_rr_bf_noise_hourly"] = {
        "front_rr_screen(nearest)": noise_stats(front["rr_col"]), "front_rr_interp": noise_stats(front["rr_bs"]),
        "cm7_rr": noise_stats(cm7["rr"]), "cm30_rr": noise_stats(cm30["rr"]),
        "front_bf_screen": noise_stats(front["bf_col"]), "cm7_bf": noise_stats(cm7["bf"]), "cm30_bf": noise_stats(cm30["bf"]),
        "front_atm_screen": noise_stats(front["atm_near"]), "cm7_atm": noise_stats(cm7["atm"]), "cm30_atm": noise_stats(cm30["atm"]),
    }
    # 가까운 만기 남은 시간별 RR 수준·변화
    fb = front.assign(bucket=pd.cut(front["T_h"], [0, 1, 3, 8, 16, 24, 1e9], labels=["<1h", "1-3h", "3-8h", "8-16h", "16-24h", ">24h"]))
    fb["d_rr"] = fb["rr_col"].diff().where(fb["T_h"].diff() < 0)      # 같은 만기 안에서만
    R["S2_front_rr_by_hours_left"] = {str(k): {"n": int(len(v)), "rr_median": round(float(v["rr_col"].median()), 2),
                                               "rr_iqr": round(float(v["rr_col"].quantile(.75) - v["rr_col"].quantile(.25)), 2),
                                               "bf_median": round(float(v["bf_col"].median()), 2),
                                               "abs_chg_median": round(float(v["d_rr"].abs().median()), 2),
                                               "atm_minus_cm7_median": None}
                                      for k, v in fb.groupby("bucket", observed=True)}
    j = front.join(cm7["atm"].rename("cm7_atm"), how="inner")
    for k, v in j.assign(bucket=pd.cut(j["T_h"], [0, 1, 3, 8, 16, 24, 1e9], labels=["<1h", "1-3h", "3-8h", "8-16h", "16-24h", ">24h"])).groupby("bucket", observed=True):
        R["S2_front_rr_by_hours_left"][str(k)]["atm_minus_cm7_median"] = round(float((v["atm_near"] - v["cm7_atm"]).median()), 2)
    # 상관: 화면 RR 과 고정만기 RR 의 1시간 변화가 같은 걸 말하나
    a = front["rr_col"].resample("1h").last().diff()
    R["S2_front_vs_cm7_rr_change_corr"] = round(float(a.corr(cm7["rr"].resample("1h").last().diff())), 3)
    R["S2_front_vs_cm30_rr_change_corr"] = round(float(a.corr(cm30["rr"].resample("1h").last().diff())), 3)

    # S3 max pain
    er2 = er.copy()
    er2["day"] = er2["ts"].dt.floor("D")
    ten = er2[er2["ts"] >= "2026-09-28 10:00"]               # 10분 주기 구간
    g = ten.groupby(["exp", "day"])["pain"]
    rng = (g.max() - g.min())
    steps = rng / ten.groupby(["exp", "day"])["k_step_atm"].median()
    nchg = ten.sort_values("ts").groupby(["exp", "day"])["pain"].apply(lambda s: int((s.diff().fillna(0) != 0).sum()))
    fr10 = front[front.index >= "2026-09-28 10:00"]
    last24 = er2[er2["T_h"] <= 24]
    g24 = last24.groupby("exp")["pain"]
    R["S3_max_pain"] = {
        "same_day_range_usd_all_expiries(10min)": _q(rng), "same_day_range_in_strike_steps": _q(steps),
        "same_day_n_changes": _q(nchg),
        "front_same_day_range_usd": _q((fr10.groupby(fr10.index.floor("D"))["pain"].agg(lambda s: s.max() - s.min()))),
        "last24h_before_expiry_range_usd(hourly+10min)": _q(g24.max() - g24.min()),
        "last24h_n_expiries": int(g24.ngroups),
        "last24h_pain_minus_final_F_abs_usd": None,
        "coin_vs_usd_definition_disagree_frac": round(float((er["pain"] != er["pain_coin"]).mean()), 4),
        "coin_vs_usd_disagree_abs_usd": _q((er["pain"] - er["pain_coin"]).abs()[er["pain"] != er["pain_coin"]]),
        "front_share_of_total_oi": _q(front["coi"].add(front["poi"]) / er.groupby("ts")[["coi", "poi"]].sum().sum(1).reindex(front.index)),
        "front_pain_minus_spot_pct": _q((front["pain"] / front["F"] - 1) * 100),
    }
    # 최대 미결제 만기(7일 안) pain 과 가까운 만기 pain 의 거리
    wk = er[er["T_h"] <= 168].assign(oi=lambda d: d["coi"] + d["poi"])
    big = wk.loc[wk.groupby("ts")["oi"].idxmax()].set_index("ts")
    R["S3_max_pain"]["front_vs_biggest_7d_expiry_pain_gap_usd"] = _q((front["pain"] - big["pain"].reindex(front.index)).abs())
    R["S3_max_pain"]["front_is_biggest_7d_expiry_frac"] = round(float((front["exp"] == big["exp"].reindex(front.index)).mean()), 3)
    return R


def _q(s) -> dict:
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "mean": round(float(s.mean()), 4), "p10": round(float(s.quantile(.1)), 4), "median": round(float(s.median()), 4),
            "p90": round(float(s.quantile(.9)), 4), "abs_median": round(float(s.abs().median()), 4), "abs_p90": round(float(s.abs().quantile(.9)), 4)}


# ── S1 DEX · charm · 체결 순델타 ─────────────────────────────────────────────────
def s1_exposure(chain: pd.DataFrame, trades: pd.DataFrame) -> dict:
    c = chain[chain["recorded_at_utc"] >= "2026-09-28 10:00"].copy()
    idx = c.loc[c.groupby("recorded_at_utc")["T"].idxmin()].set_index("recorded_at_utc")["underlying_price"]
    c["idx"] = c["recorded_at_utc"].map(idx)
    F, K, iv, T, call, oi = (c[x].to_numpy() for x in ("underlying_price", "strike", "mark_iv", "T", "is_call", "open_interest"))
    _, d_bs = bs(F, K, iv, T, call)
    d_pa = pa_delta(F, K, iv, T, call)
    dt = 1 / 8760
    _, d_bs1 = bs(F, K, iv, np.maximum(T - dt, 1e-7), call)
    d_pa1 = pa_delta(F, K, iv, np.maximum(T - dt, 1e-7), call)
    sg = np.where(call, 1.0, -1.0)
    first = c.groupby("recorded_at_utc")["expiration_ts"].transform("min")
    scopes = {"front": (c["expiration_ts"] == first).to_numpy(), "week": (c["T"] * 365 <= 7).to_numpy(), "all": np.ones(len(c), bool)}
    R = {}
    for name, m in scopes.items():
        df = pd.DataFrame({"ts": c["recorded_at_utc"].to_numpy()[m], "idx": c["idx"].to_numpy()[m],
                           "hold_bs": (d_bs * oi)[m], "hold_pa": (d_pa * oi)[m],
                           "asm_bs": (d_bs * sg * oi)[m], "asm_pa": (d_pa * sg * oi)[m],
                           "ch_bs": ((d_bs1 - d_bs) * sg * oi)[m], "ch_pa": ((d_pa1 - d_pa) * sg * oi)[m]})
        a = df.groupby("ts").sum(numeric_only=True).mul(df.groupby("ts")["idx"].first(), axis=0)
        R[name] = {"holder_dex_bs_usd_median": round(float(a["hold_bs"].median())), "holder_dex_pa_usd_median": round(float(a["hold_pa"].median())),
                   "holder_pa_over_bs": _q(a["hold_pa"] / a["hold_bs"]),
                   "dealer_asm_dex_bs_usd_median": round(float(a["asm_bs"].median())), "dealer_asm_dex_pa_usd_median": round(float(a["asm_pa"].median())),
                   "dealer_asm_diff_usd": _q(a["asm_pa"] - a["asm_bs"]),
                   "dealer_asm_sign_disagree_frac": round(float((np.sign(a["asm_pa"]) != np.sign(a["asm_bs"])).mean()), 3),
                   "charm1h_asm_bs_usd": _q(a["ch_bs"]), "charm1h_asm_pa_usd": _q(a["ch_pa"]),
                   "charm_sign_disagree_frac": round(float((np.sign(a["ch_pa"]) != np.sign(a["ch_bs"])).mean()), 3)}
    # 체결 순델타(hourly_flow): 체결 IV·지수로 BS vs PA
    t = trades[trades["instrument_name"].str.startswith("ETH-")].copy()
    parts = t["instrument_name"].str.split("-", expand=True)
    exp = pd.to_datetime(parts[1], format="%d%b%y", utc=True) + pd.Timedelta(hours=8)
    Tt = ((exp - pd.to_datetime(t["ts_ms"], unit="ms", utc=True)).dt.total_seconds() / YEAR).to_numpy()
    ok = (Tt > 0) & (t["iv"].fillna(0).to_numpy() > 0)
    Kt, ct = parts[2].astype(float).to_numpy(), (parts[3] == "C").to_numpy()
    sgn = np.where(t["direction"] == "buy", 1.0, -1.0) * t["amount"].to_numpy()
    _, db = bs(t["index_price"].to_numpy(), Kt, t["iv"].to_numpy(), Tt, ct)
    dp = pa_delta(t["index_price"].to_numpy(), Kt, t["iv"].to_numpy(), Tt, ct)
    day = pd.to_datetime(t["ts_ms"], unit="ms", utc=True).dt.floor("D").to_numpy()
    f = pd.DataFrame({"day": day[ok], "bs": (sgn * db)[ok], "pa": (sgn * dp)[ok]}).groupby("day").sum()
    R["taker_net_delta_eth_by_day"] = {str(k.date()): {"bs": round(v["bs"], 1), "pa": round(v["pa"], 1)} for k, v in f.iterrows()}
    return R


# ── S2 체결 IV vs mark_iv ─────────────────────────────────────────────────────
def s2_trade_vs_mark(chain: pd.DataFrame, trades: pd.DataFrame) -> dict:
    t = trades[trades["instrument_name"].str.startswith("ETH-") & (trades["iv"].fillna(0) > 0)].copy()
    t["ts"] = pd.to_datetime(t["ts_ms"], unit="ms", utc=True)
    c = chain[chain["recorded_at_utc"] >= "2026-09-27"][["recorded_at_utc", "instrument_name", "mark_iv", "T", "underlying_price", "strike", "is_call"]]
    t["ts"] = t["ts"].astype("datetime64[ns, UTC]")
    c = c.assign(recorded_at_utc=c["recorded_at_utc"].astype("datetime64[ns, UTC]"))
    m = pd.merge_asof(t.sort_values("ts"), c.sort_values("recorded_at_utc"), left_on="ts", right_on="recorded_at_utc",
                      by="instrument_name", tolerance=pd.Timedelta("10min"), direction="nearest").dropna(subset=["mark_iv"])
    _, d = bs(m["underlying_price"], m["strike"], m["mark_iv"], m["T"], m["is_call"])
    m["absd"], m["gap"] = np.abs(d), m["iv"] - m["mark_iv"]
    m["prem_rel"] = m["price"] / m["mark_price"] - 1
    m["dte"] = pd.cut(m["T"] * 365, [0, 1, 7, 30, 1e9], labels=["<1d", "1-7d", "7-30d", ">30d"])
    m["dbk"] = pd.cut(m["absd"], [0, .1, .2, .4, .6, 1.01], labels=["|Δ|<.1", ".1-.2", ".2-.4", ".4-.6", ">.6"])
    R = {"n": int(len(m)), "trade_iv_minus_mark_iv_pt_all": _q(m["gap"])}
    R["by_dte"] = {str(k): _q(v["gap"]) for k, v in m.groupby("dte", observed=True)}
    R["by_abs_delta"] = {str(k): _q(v["gap"]) for k, v in m.groupby("dbk", observed=True)}
    R["trade_price_vs_trade_mark_rel_by_dte"] = {str(k): _q(v["prem_rel"]) for k, v in m.groupby("dte", observed=True)}
    return R


# ── S2 live: bid/ask IV ───────────────────────────────────────────────────────
def live_ticker_sample() -> pd.DataFrame:
    path = OUT / "live_ticker.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for cur, api, pre, width, pick in (("ETH", "ETH", "ETH-", 0.2, (0, 1, 2, "7d", "30d")), ("SOL", "USDC", "SOL_USDC-", 0.25, (0, "30d")), ("XRP", "USDC", "XRP_USDC-", 0.25, (0, "30d"))):
        ins = [i for i in _get("get_instruments", currency=api, kind="option") if i["instrument_name"].startswith(pre)]
        exps = sorted({i["expiration_timestamp"] for i in ins})
        now = time.time() * 1000
        chosen = set()
        for p in pick:
            chosen.add(exps[p] if isinstance(p, int) else min(exps, key=lambda e: abs(e - now - int(p[:-1]) * 86400e3)))
        idx = _get("get_index_price", index_name=f"{cur.lower()}_usd" if api != "USDC" else f"{cur.lower()}_usdc")["index_price"]
        for i in ins:
            if i["expiration_timestamp"] in chosen and abs(i["strike"] / idx - 1) <= width:
                t = _get("ticker", instrument_name=i["instrument_name"])
                rows.append({"cur": cur, "name": i["instrument_name"], "exp": i["expiration_timestamp"], "K": i["strike"], "call": i["option_type"] == "call",
                             "F": t["underlying_price"], "mark_iv": t["mark_iv"], "bid_iv": t["bid_iv"], "ask_iv": t["ask_iv"],
                             "delta_api": t["greeks"]["delta"], "mark_price": t["mark_price"], "best_bid": t["best_bid_price"], "best_ask": t["best_ask_price"],
                             "oi": t["open_interest"], "T": (i["expiration_timestamp"] / 1000 - time.time()) / YEAR})
                time.sleep(0.06)
    d = pd.DataFrame(rows)
    d.to_parquet(path)
    return d


def s2_live(d: pd.DataFrame) -> dict:
    R = {"sampled_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "n": int(len(d))}
    _, dbs = bs(d["F"], d["K"], d["mark_iv"], d["T"], d["call"])
    R["api_delta_minus_our_bs_abs_max"] = round(float(np.abs(d["delta_api"] - dbs).max()), 6)
    d = d.assign(absd=np.abs(dbs), two=(d["bid_iv"] > 0) & (d["ask_iv"] > 0))
    d["spread"] = (d["ask_iv"] - d["bid_iv"]).where(d["two"])
    d["mark_pos"] = ((d["mark_iv"] - d["bid_iv"]) / (d["ask_iv"] - d["bid_iv"])).where(d["two"])
    d["dte"] = pd.cut(d["T"] * 365, [0, 1, 7, 30, 1e9], labels=["<1d", "1-7d", "7-30d", ">30d"])
    d["dbk"] = pd.cut(d["absd"], [0, .05, .15, .35, .65, 1.01], labels=["|Δ|<.05", ".05-.15", ".15-.35(25Δ 부근)", ".35-.65(ATM)", ">.65"])
    out = {}
    for (cur, dte, dbk), v in d.groupby(["cur", "dte", "dbk"], observed=True):
        out[f"{cur} {dte} {dbk}"] = {"n": int(len(v)), "no_two_sided_frac": round(float(1 - v["two"].mean()), 2),
                                     "iv_spread_median_pt": round(float(v["spread"].median()), 2) if v["two"].any() else None,
                                     "mark_pos_in_spread_median": round(float(v["mark_pos"].median()), 2) if v["two"].any() else None}
    R["by_cur_dte_delta"] = out
    # 25Δ RR: mark vs mid(양쪽 호가 있는 것만)
    rr = {}
    for (cur, exp), g in d.groupby(["cur", "exp"]):
        F, T = float(g["F"].iloc[0]), float(g["T"].iloc[0])
        oc, op = g[g["call"] & (g["K"] >= F)], g[~g["call"] & (g["K"] <= F)]
        if len(oc) < 2 or len(op) < 2:
            continue
        dc, dp = bs(F, oc["K"], oc["mark_iv"], T, True)[1], bs(F, op["K"], op["mark_iv"], T, False)[1]
        mid = lambda x: ((x["bid_iv"] + x["ask_iv"]) / 2).where(x["bid_iv"] > 0)
        oc2, op2 = oc.assign(mid=mid(oc)), op.assign(mid=mid(op))
        mc, mp = oc2["mid"].notna().to_numpy(), op2["mid"].notna().to_numpy()
        rr[f"{cur} {pd.to_datetime(exp, unit='ms'):%m-%d} ({T * 365:.1f}d)"] = {
            "rr_mark": round(_interp_at(dc, oc["mark_iv"], .25) - _interp_at(dp, op["mark_iv"], -.25), 2),
            "rr_mid": round(_interp_at(dc[mc], oc2["mid"][mc], .25) - _interp_at(dp[mp], op2["mid"][mp], -.25), 2) if mc.sum() > 1 and mp.sum() > 1 else None,
            "rr_bid_call_ask_put(최악)": round(_interp_at(dc[mc], oc2["bid_iv"][mc], .25) - _interp_at(dp[mp], op2["ask_iv"][mp], -.25), 2) if mc.sum() > 1 and mp.sum() > 1 else None,
            "n_strikes_call_put": [int(len(oc)), int(len(op))]}
    R["rr25_mark_vs_mid"] = rr
    return R


# ── S4 P/C ──────────────────────────────────────────────────────────────────
def s4_pc(chain: pd.DataFrame, trades: pd.DataFrame) -> dict:
    c = chain[chain["recorded_at_utc"] >= "2026-09-28 10:00"].copy()
    c["idx"] = c["recorded_at_utc"].map(c.loc[c.groupby("recorded_at_utc")["T"].idxmin()].set_index("recorded_at_utc")["underlying_price"])
    c["otm"] = np.where(c["is_call"], c["strike"] >= c["underlying_price"], c["strike"] <= c["underlying_price"])
    c["prem_usd"] = c["mark_price"] * c["idx"] * c["open_interest"]
    c["k_not"] = c["strike"] * c["open_interest"]
    c["front"] = c["expiration_ts"] == c.groupby("recorded_at_utc")["expiration_ts"].transform("min")

    def ratio(df, col):
        a = df.groupby(["recorded_at_utc", "is_call"])[col].sum().unstack()
        return (a[False] / a[True])
    R = {"oi_snapshots": {
        "oi_count_all(=화면 달러 P/C, 지수×수량이라 같은 값)": _q(ratio(c, "open_interest")),
        "oi_count_front(화면 첫 만기)": _q(ratio(c[c["front"]], "open_interest")),
        "oi_count_7d": _q(ratio(c[c["T"] * 365 <= 7], "open_interest")),
        "oi_count_otm_only_all": _q(ratio(c[c["otm"]], "open_interest")),
        "oi_strike_notional_all": _q(ratio(c, "k_not")),
        "oi_premium_usd_all": _q(ratio(c, "prem_usd")),
        "oi_premium_usd_front": _q(ratio(c[c["front"]], "prem_usd")),
    }}
    fr = ratio(c[c["front"]], "open_interest")
    R["front_pc_hourly_change_abs_median"] = round(float(fr.resample("1h").last().diff().abs().median()), 3)
    R["front_pc_range_across_days"] = [round(float(fr.min()), 2), round(float(fr.max()), 2)]
    t = trades[trades["instrument_name"].str.startswith("ETH-")].copy()
    t["day"] = pd.to_datetime(t["ts_ms"], unit="ms", utc=True).dt.floor("D")
    t["call"] = t["instrument_name"].str.endswith("-C")
    t["buy"] = t["direction"] == "buy"
    t["prem"] = t["amount"] * t["price"] * t["index_price"]
    daily = {}
    oi_day = ratio(c, "open_interest").groupby(lambda x: x.floor("D")).mean()
    for day, g in t.groupby("day"):
        C, P = g[g["call"]], g[~g["call"]]
        daily[str(day.date())] = {
            "n_trades": int(len(g)),
            "volume_pc(수량)": round(P["amount"].sum() / C["amount"].sum(), 2),
            "volume_pc(프리미엄$)": round(P["prem"].sum() / C["prem"].sum(), 2),
            "taker_buy_pc(풋 매수/콜 매수)": round(P[P["buy"]]["amount"].sum() / C[C["buy"]]["amount"].sum(), 2),
            "taker_sell_pc(풋 매도/콜 매도)": round(P[~P["buy"]]["amount"].sum() / C[~C["buy"]]["amount"].sum(), 2),
            "bearish_flow_ratio((풋매수+콜매도)/(콜매수+풋매도))": round((P[P["buy"]]["amount"].sum() + C[~C["buy"]]["amount"].sum())
                                                                   / (C[C["buy"]]["amount"].sum() + P[~P["buy"]]["amount"].sum()), 2),
            "put_sell_share_of_put_volume": round(P[~P["buy"]]["amount"].sum() / P["amount"].sum(), 2),
            "oi_pc_all_day_mean": round(float(oi_day.get(day, np.nan)), 2),
            "partial_day": bool(g["ts_ms"].max() - g["ts_ms"].min() < 20 * 3600e3)}
    R["daily_trades"] = daily
    return R


# ── S5 1σ 띠 · VRP ───────────────────────────────────────────────────────────
def s5_bands(kl: pd.DataFrame, dvol_h: pd.Series, er: pd.DataFrame) -> dict:
    R = {"brownian_touch_prob_1sigma": round(brownian_touch_prob(1.0), 4), "normal_inside_1sigma": 0.6827}
    dv = dvol_h.shift(1)          # 봉이 열린 시점에 알려진 값 = 직전 시간 종가(화면은 ≤10분 전 값)

    def band(freq: str, sec: int):
        o = kl["o"].resample(freq).first()
        h, l, c = kl["h"].resample(freq).max(), kl["l"].resample(freq).min(), kl["c"].resample(freq).last()
        d = pd.DataFrame({"o": o, "h": h, "l": l, "c": c}).dropna()
        d["dvol"] = dv.reindex(d.index, method="ffill")
        d = d.dropna()
        s = d["o"] * d["dvol"] / 100 * math.sqrt(sec / YEAR)
        d["z"] = (d["c"] - d["o"]) / s
        d["touch"] = ((d["h"] - d["o"]) >= s) | ((d["o"] - d["l"]) >= s)
        return d

    for name, freq, sec in (("5m", "5min", 300), ("1h", "1h", 3600)):
        d = band(freq, sec)
        z = d["z"]
        by_h = d.groupby(d.index.hour)
        R[f"band_{name}"] = {
            "n": int(len(d)), "close_inside_1sigma": round(float((z.abs() <= 1).mean()), 4),
            "touch_outside_during_bar": round(float(d["touch"].mean()), 4),
            "rms_z(1=보정)": round(float(np.sqrt((z ** 2).mean())), 3), "mean_abs_z_over_0.798": round(float(z.abs().mean() / 0.7979), 3),
            "abs_z_gt2": round(float((z.abs() > 2).mean()), 4), "abs_z_gt3": round(float((z.abs() > 3).mean()), 4),
            "by_hour_utc_rms_z": {int(k): round(float(np.sqrt((v["z"] ** 2).mean())), 2) for k, v in by_h},
            "by_hour_utc_inside": {int(k): round(float((v["z"].abs() <= 1).mean()), 3) for k, v in by_h},
            "weekday_rms_z": round(float(np.sqrt((z[d.index.dayofweek < 5] ** 2).mean())), 3),
            "weekend_rms_z": round(float(np.sqrt((z[d.index.dayofweek >= 5] ** 2).mean())), 3),
            "by_month_rms_z": {str(k): round(float(np.sqrt((v ** 2).mean())), 2) for k, v in z.groupby(z.index.strftime("%Y-%m"))}}
    # 24h(«오늘 1σ»는 지금부터 24시간) -- 매시 시작, 겹침 있음
    c1 = kl["c"].resample("1h").last().dropna()
    fwd = c1.shift(-24) / c1 - 1
    s24 = dv.reindex(c1.index, method="ffill") / 100 * math.sqrt(1 / 365)
    z24 = (fwd / s24).dropna()
    R["band_24h_rolling_hourly_start"] = {"n": int(len(z24)), "close_inside_1sigma": round(float((z24.abs() <= 1).mean()), 4),
                                          "rms_z": round(float(np.sqrt((z24 ** 2).mean())), 3),
                                          "abs_z_gt2": round(float((z24.abs() > 2).mean()), 4)}
    # 만기까지(화면 «다음 만기까지»는 DVOL) vs 가까운 만기 ATM IV · 7일 고정만기 ATM
    front = er.loc[er.groupby("ts")["T_h"].idxmin()].set_index("ts").sort_index()
    front = front[(front.index <= kl.index[-1] - pd.Timedelta("1D")) & (front["T_h"] > 0.5)]
    cm7 = constant_maturity(er, 7)["atm"]
    rows = []
    for ts, r in front.iterrows():
        p0 = kl["c"].asof(ts)
        p1 = kl["c"].asof(r["exp"])
        dvv = dv.asof(ts)
        if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(dvv)):
            continue
        ret, sq = math.log(p1 / p0), math.sqrt(r["T_h"] / 8760)
        rows.append({"exp": r["exp"], "T_h": r["T_h"], "z_dvol": ret / (dvv / 100 * sq), "z_front_atm": ret / (r["atm_near"] / 100 * sq),
                     "z_cm7": ret / (cm7.get(ts, np.nan) / 100 * sq), "front_atm_over_dvol": r["atm_near"] / dvv})
    z = pd.DataFrame(rows)
    R["band_to_expiry"] = {"n_snapshots": int(len(z)), "n_expiries": int(z["exp"].nunique()),
                           **{k: {"rms_z": round(float(np.sqrt((z[k] ** 2).mean())), 3), "inside_1sigma": round(float((z[k].abs() <= 1).mean()), 3)}
                              for k in ("z_dvol", "z_front_atm", "z_cm7")},
                           "front_atm_over_dvol": _q(z["front_atm_over_dvol"]),
                           "rms_z_by_hours_left_dvol": {str(k): round(float(np.sqrt((v["z_dvol"] ** 2).mean())), 2)
                                                        for k, v in z.groupby(pd.cut(z["T_h"], [0, 3, 8, 16, 24]), observed=True)},
                           "rms_z_by_hours_left_front_atm": {str(k): round(float(np.sqrt((v["z_front_atm"] ** 2).mean())), 2)
                                                             for k, v in z.groupby(pd.cut(z["T_h"], [0, 3, 8, 16, 24]), observed=True)}}
    # VRP 기간 불일치
    r1h = np.log(c1).diff()
    c5 = kl["c"].resample("5min").last().dropna()
    r5 = np.log(c5).diff()
    ann = lambda r, n, per_day: np.sqrt((r ** 2).rolling(n).mean() * per_day * 365) * 100
    rv7h, rv30h = ann(r1h, 168, 24), ann(r1h, 720, 24)
    rv7_5m = ann(r5, 2016, 288).resample("1h").last().reindex(c1.index)
    fwd30 = rv30h.shift(-720)
    fwd7 = rv7h.shift(-168)
    d = pd.DataFrame({"dvol": dv.reindex(c1.index, method="ffill"), "rv7": rv7h, "rv30": rv30h, "rv7_5m": rv7_5m, "fwd30": fwd30, "fwd7": fwd7}).dropna(subset=["dvol", "rv7", "rv30"])
    v7, v30 = d["dvol"] - d["rv7"], d["dvol"] - d["rv30"]
    ok = d["fwd30"].notna()
    R["vrp_window"] = {"n_hours": int(len(d)),
                       "vrp_screen(DVOL-rv7_1h)": _q(v7), "vrp_dvol_minus_rv30": _q(v30),
                       "rv7_minus_rv30_pt": _q(d["rv7"] - d["rv30"]),
                       "sign_disagree_rv7_vs_rv30": round(float((np.sign(v7) != np.sign(v30)).mean()), 3),
                       "screen_vrp_negative_frac": round(float((v7 < 0).mean()), 3),
                       "realized_vrp_fwd30(DVOL-다음30일RV)": _q((d["dvol"] - d["fwd30"])[ok]),
                       "corr_with_fwd30_vrp": {"vrp7": round(float(v7[ok].corr((d["dvol"] - d["fwd30"])[ok])), 3),
                                               "vrp30": round(float(v30[ok].corr((d["dvol"] - d["fwd30"])[ok])), 3)},
                       "rv7_5m_minus_rv7_1h_pt": _q((d["rv7_5m"] - d["rv7"]).dropna()),
                       "rv7_change_over_6h_pt(캐시 6시간)": _q(d["rv7"].diff(6))}
    return R


def s5_seasonal(kl: pd.DataFrame, dvol_h: pd.Series, er: pd.DataFrame, chain: pd.DataFrame) -> dict:
    """고칠 방법 후보: 시간대 분산 시계. 가중치 w_h(UTC 시각별 5분 z² 평균, 평균 1로 정규화)를 1~5월에서 배워
    6~9월에 적용 -- σ = DVOL × √(Σ 창 안 w_h·Δt / 1년). 시간대별 보정이 평평해지는가."""
    dv = dvol_h.shift(1)
    o, c = kl["o"].resample("5min").first(), kl["c"].resample("5min").last()
    d = pd.DataFrame({"o": o, "c": c}).dropna()
    d["dvol"] = dv.reindex(d.index, method="ffill")
    d = d.dropna()
    d["z"] = (d["c"] - d["o"]) / (d["o"] * d["dvol"] / 100 * math.sqrt(300 / YEAR))
    tr, te = d[d.index < "2026-06-01"], d[d.index >= "2026-06-01"]
    w = (tr["z"] ** 2).groupby(tr.index.hour).mean()
    w = w / w.mean()
    wk = (tr["z"] ** 2).groupby(tr.index.dayofweek >= 5).mean()
    te = te.assign(z_adj=te["z"] / np.sqrt(te.index.hour.map(w).to_numpy()))
    hr = lambda col: {int(k): round(float(np.sqrt((v ** 2).mean())), 2) for k, v in te[col].groupby(te.index.hour)}
    a, b = hr("z"), hr("z_adj")
    R = {"weights_train_jan_may": {int(k): round(float(v), 2) for k, v in w.items()},
         "weekend_over_weekday_var_train": round(float(wk[True] / wk[False]), 2),
         "test_jun_sep_rms_by_hour_raw": a, "test_jun_sep_rms_by_hour_seasonal": b,
         "test_hour_rms_spread(max-min)": {"raw": round(max(a.values()) - min(a.values()), 2), "seasonal": round(max(b.values()) - min(b.values()), 2)},
         "test_inside_1sigma": {"raw": round(float((te["z"].abs() <= 1).mean()), 3), "seasonal": round(float((te["z_adj"].abs() <= 1).mean()), 3)}}
    # 만기까지(화면 «다음 만기까지»): 창 안 시간대 가중치 합으로
    front = er.loc[er.groupby("ts")["T_h"].idxmin()].set_index("ts").sort_index()
    front = front[(front.index >= "2026-08-15") & (front.index <= kl.index[-1] - pd.Timedelta("1D")) & (front["T_h"] > 0.5)]
    rows = []
    for ts, r in front.iterrows():
        p0, p1, dvv = kl["c"].asof(ts), kl["c"].asof(r["exp"]), dv.asof(ts)
        if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(dvv)):
            continue
        grid = pd.date_range(ts, r["exp"], freq="5min")[:-1]
        wsum = float(grid.hour.map(w).to_numpy().sum()) * 300 / YEAR
        ret = math.log(p1 / p0)
        rows.append({"T_h": r["T_h"], "raw": ret / (dvv / 100 * math.sqrt(r["T_h"] / 8760)), "seasonal": ret / (dvv / 100 * math.sqrt(wsum))})
    z = pd.DataFrame(rows)
    R["to_expiry_rms_by_hours_left"] = {k: {str(b): round(float(np.sqrt((v[k] ** 2).mean())), 2) for b, v in z.groupby(pd.cut(z["T_h"], [0, 3, 8, 16, 24]), observed=True)}
                                        for k in ("raw", "seasonal")}
    # 기간 구조 첫 점(가까운 만기 ATM)의 시간대 성분
    cm7 = constant_maturity(er, 7)["atm"]
    fr = front.join(cm7.rename("cm7"), how="inner")
    fr2 = er.loc[er.groupby("ts")["T_h"].idxmin()].set_index("ts").join(cm7.rename("cm7"), how="inner")
    second = er.sort_values("T_h").groupby("ts").nth(1).set_index("ts")["atm_near"]
    R["front_atm_over_cm7_by_hour_utc"] = {int(k): round(float(v.median()), 2) for k, v in (fr2["atm_near"] / fr2["cm7"]).groupby(fr2.index.hour)}
    R["term_inverted_front_gt_second_frac"] = round(float((fr2["atm_near"] > second.reindex(fr2.index)).mean()), 3)
    # 미결제 «규모»: 기초자산 명목 vs 프리미엄(시가 가치)
    last = chain[chain["recorded_at_utc"] == chain["recorded_at_utc"].max()]
    F0 = float(last.loc[last["T"].idxmin(), "underlying_price"])
    R["oi_premium_over_underlying_notional_all"] = round(float((last["mark_price"] * last["open_interest"]).sum() / last["open_interest"].sum()), 4)
    fe = last[last["expiration_ts"] == last["expiration_ts"].min()]
    R["oi_premium_over_underlying_notional_front"] = round(float((fe["mark_price"] * fe["open_interest"]).sum() / fe["open_interest"].sum()), 4)
    R["front_notional_usd"] = round(float(fe["open_interest"].sum() * F0))
    return R


# ── S6·S7 ────────────────────────────────────────────────────────────────────
def s6_s7(summary: pd.DataFrame, dvol_1m: pd.Series) -> dict:
    R = {}
    s = summary.copy()
    s["ts"] = pd.to_datetime(s["recorded_at_utc"], utc=True)
    cad = {}
    for cur, g in s.groupby("currency"):
        gap = g["ts"].sort_values().diff().dt.total_seconds().dropna() / 60
        cad[cur] = {"n": int(len(g)), "gap_min_median": round(float(gap.median()), 2), "gap_gt15_n": int((gap > 15).sum()), "gap_max_min": round(float(gap.max()), 1)}
    R["S6_cadence_option_summary"] = cad
    polls = s.assign(p=s["ts"].dt.floor("5min")).groupby("p")["currency"].nunique()
    R["S6_polls_with_missing_coin"] = int((polls < 4).sum())
    R["S6_polls_total"] = int(len(polls))
    d1 = dvol_1m.dropna()
    ch10 = (d1.shift(-10) / d1 - 1).dropna() * 100
    R["S6_dvol_rel_change_10min_pct(5분 띠 폭 오차)"] = _q(ch10)
    R["S6_dvol_rel_change_60min_pct(1시간 띠 폭 고정 오차)"] = _q((d1.shift(-60) / d1 - 1).dropna() * 100)
    eth = s[s["currency"] == "ETH"].set_index("ts").sort_index()
    rv = eth["rv7"]
    R["S6_rv7_distinct_values_per_6h"] = round(float(rv.resample("6h").nunique().mean()), 2)
    # S7: SOL·XRP iv30 대용 vs ETH DVOL
    out = {}
    for cur, g in s.groupby("currency"):
        g = g.set_index("ts").sort_index()
        pl = g["payload"].map(json.loads)
        iv30 = pl.map(lambda p: p.get("iv30"))
        dv = pl.map(lambda p: p.get("dvol"))
        fr = pl.map(lambda p: (p.get("expiries") or [{}])[0])
        rr = fr.map(lambda e: e.get("rr25"))
        out[cur] = {"iv30_10min_abs_chg_median_pt": round(float(iv30.astype(float).diff().abs().median()), 2),
                    "iv30_10min_abs_chg_p90_pt": round(float(iv30.astype(float).diff().abs().quantile(.9)), 2),
                    "dvol_10min_abs_chg_median_pt": round(float(dv.astype(float).diff().abs().median()), 2) if dv.notna().any() else None,
                    "iv30_minus_dvol_median_pt": round(float((iv30.astype(float) - dv.astype(float)).median()), 2) if dv.notna().any() else None,
                    "front_rr25_10min_abs_chg_median_pt": round(float(rr.astype(float).diff().abs().median()), 2),
                    "front_rr25_10min_abs_chg_p90_pt": round(float(rr.astype(float).diff().abs().quantile(.9)), 2),
                    "front_rr25_range": [round(float(rr.min()), 1), round(float(rr.max()), 1)]}
    R["S7_iv30_and_front_rr_noise_by_coin"] = out
    return R


def s7_strikes() -> dict:
    out = {}
    for cur in ("SOL", "XRP", "ETH"):
        c = pd.read_parquet(SRV / "chain.parquet", filters=[("currency", "==", cur)], columns=["recorded_at_utc", "expiration_ts", "strike", "underlying_price", "open_interest"])
        last = c[c["recorded_at_utc"] == c["recorded_at_utc"].max()]
        rows = {}
        for exp, g in last.groupby("expiration_ts"):
            F = g["underlying_price"].iloc[0]
            ks = np.sort(g["strike"].unique())
            rows[str(exp.date())] = {"n_strikes": int(len(ks)), "n_within_±10%": int(((ks / F - 1).__abs__() <= .1).sum()),
                                     "step_near_atm_pct": round(float(np.min(np.abs(np.diff(ks[np.argsort(np.abs(ks - F))[:3]]))) / F * 100), 2) if len(ks) > 2 else None}
        out[cur] = dict(list(rows.items())[:6])
    return out


# ── selftest ─────────────────────────────────────────────────────────────────
def selftest() -> None:
    # 프리미엄 조정 델타 = d(가치_코인)/dS·S (r=0, F=S), 수치미분과 일치
    for K, call in ((2400, True), (3000, True), (2400, False), (3000, False)):
        S, iv, T, h = 2700.0, 55.0, 30 / 365, 0.01
        vc = lambda s: float(bs(s, K, iv, T, call)[0]) / s
        num = (vc(S + h) - vc(S - h)) / (2 * h) * S
        assert abs(num - float(pa_delta(S, K, iv, T, call))) < 1e-6, (K, call, num)
        assert float(pa_delta(S, K, iv, T, call)) < float(bs(S, K, iv, T, call)[1])   # 프리미엄만큼 늘 작다
    # 깊은 내가격 콜의 PA 델타는 K/F 로 내려간다(비단조) -- 25Δ 탐색을 외가격으로 제한해야 하는 이유
    assert abs(float(pa_delta(2700, 675, 50, 1 / 365, True)) - 0.25) < 1e-3
    # max pain: 합성 체인
    K = np.array([90, 100, 100, 110]); call = np.array([True, True, False, False]); oi = np.array([3, 5, 5, 3])
    assert max_pain(K, call, oi) == 100
    # 수집기 식(options_summary)과 같은 답
    calls, puts = pd.DataFrame({"strike": K[call], "open_interest": oi[call]}), pd.DataFrame({"strike": K[~call], "open_interest": oi[~call]})
    col = min(sorted(set(K)), key=lambda P: float(((P - calls["strike"]).clip(lower=0) * calls["open_interest"]).sum()
                                                 + ((puts["strike"] - P).clip(lower=0) * puts["open_interest"]).sum()))
    assert col == max_pain(K, call, oi)
    # 코인 단위(÷결제가) 정의가 갈리는 예: USD 로는 동률(작은 행사가), 코인으로는 큰 행사가
    K2 = np.array([100, 300]); c2 = np.array([True, False]); oi2 = np.array([10, 10])
    assert max_pain(K2, c2, oi2) == 100 and max_pain(K2, c2, oi2, coin=True) == 300
    # 브라운 운동 ±1σ 접촉 확률(알려진 값 ≈ 0.6827 보다 크다)
    assert 0.6 < brownian_touch_prob(1.0) < 0.7
    print("selftest ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    chain = load_chain("ETH")
    trades = pd.read_parquet(SRV / "trades.parquet")
    summary = pd.read_parquet(SRV / "summary.parquet")
    erp = OUT / "expiry_rows_eth.parquet"
    er = pd.read_parquet(erp) if erp.exists() else expiry_rows(chain)
    er.to_parquet(erp)
    R = {"data": {"chain_eth_from": str(chain["recorded_at_utc"].min()), "chain_eth_to": str(chain["recorded_at_utc"].max()),
                  "n_snapshots": int(chain["recorded_at_utc"].nunique()), "trades_from": str(pd.to_datetime(trades["ts_ms"].min(), unit="ms")),
                  "n_trades": int(len(trades))}}
    R.update(s1_s2_s3(chain, er))
    R["S1_exposure"] = s1_exposure(chain, trades)
    R["S2_trade_vs_mark"] = s2_trade_vs_mark(chain, trades)
    R["S2_live"] = s2_live(live_ticker_sample())
    R["S4_pc"] = s4_pc(chain, trades)
    kl = klines_1m()
    R["S5"] = s5_bands(kl, dvol_history("3600", "2025-11-01", "2026-09-15"), er)
    R["S5_seasonal_fix"] = s5_seasonal(kl, dvol_history("3600", "2025-11-01", "2026-09-15"), er, chain)
    R.update(s6_s7(summary, dvol_history("60", "2026-09-20", "2026-09-30")))
    R["S7_strikes_last_snapshot"] = s7_strikes()
    (OUT / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
