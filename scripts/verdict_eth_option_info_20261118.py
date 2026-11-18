#!/usr/bin/env python3
"""사전등록 판정: 옵션 정보 두 줄(ATM 호가 폭 · 선도−지수) — docs/experiments/eth_option_info_untestable_prereg_20261002.md.
원 틀(research_eth_option_info_surface_validate_20261002)의 Px·boot·verdict·spear·ann_rv·first_after 를 그대로 쓴다.

창: 정시 t ∈ [2026-10-01 16:00, 2026-11-16 00:00) UTC (ATM 호가 폭·surface 수집 시작 10-01 15:06 다음 정시 ~ 판정일).
지표 = t 직전(< t) 마지막 option_summary 스냅샷(60분보다 오래되면 뺌). 결과 = 분 t 시가부터(t 다음 첫 완결 1분봉).
실현값 = 바이낸스 ETHUSDT 선물 1분봉(data.binance.vision, REST 아님) 5분 수익률 · DVOL = Deribit 1h, t 전에 닫힌 봉.
  1 ATM 호가 폭: ln RV(다음 24h) ~ 순위 표준화 폭 + ln DVOL + ln RV(지난 24h). 일 블록 2000회. 지지/반대/그 외 근거 없음.
  2 선도−지수: bp = surface.front.fwd / index − 1. ρ(bp, 7일 25Δ RR) 시간별. 방향: sign(다음 1h·24h 수익률) ~ 순위 표준화 bp.
입력: summary.csv = 서버 option_summary(ETH) 에서 read_only 로 뽑은 열(recorded_at_utc, atm_spread_iv, idx, front_fwd, front_exp_ms, cm7_rr).
사용: python scripts/verdict_eth_option_info_20261118.py <summary.csv> <작업 폴더>
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("V", ROOT / "scripts/research_eth_option_info_surface_validate_20261002.py")
V = importlib.util.module_from_spec(_s); _s.loader.exec_module(V)
MIN, HR, DAY = V.MIN, V.HR, V.DAY
T0, T1 = pd.Timestamp("2026-10-01 16:00", tz="UTC"), pd.Timestamp("2026-11-16 00:00", tz="UTC")


def klines(wd: Path) -> pd.DataFrame:
    f = wd / "ethusdt_1m.parquet"
    if not f.exists():
        ks = [V._vision(f"futures/um/daily/klines/ETHUSDT/1m/ETHUSDT-1m-{d:%Y-%m-%d}.zip") for d in pd.date_range("2026-09-30", "2026-11-16")]
        assert all(k is not None for k in ks), "vision 일 파일 빠짐"
        pd.concat([V._csv(k, ["open_time", "open", "high", "low", "close"]) for k in ks]).drop_duplicates("open_time").to_parquet(f)
    return pd.read_parquet(f)


def dvol(wd: Path) -> pd.DataFrame:
    f = wd / "eth_dvol_1h.parquet"
    if not f.exists():
        rows, a, end = [], int(pd.Timestamp("2026-09-29", tz="UTC").timestamp() * 1000), int(pd.Timestamp("2026-11-18", tz="UTC").timestamp() * 1000)
        while a < end:
            b = min(a + 30 * DAY, end)
            r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data", timeout=60,
                             params={"currency": "ETH", "resolution": "3600", "start_timestamp": a, "end_timestamp": b}).json()["result"]
            rows += r["data"]; a = b
        pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"]).drop_duplicates("ts").sort_values("ts").to_parquet(f)
    return pd.read_parquet(f)


def zrank(x):
    r = rankdata(x); return (r - r.mean()) / r.std()


def ols_b(X, y, i, k=1):
    return V._ols_coef(X[i], y[i], k)


def prereg_verdict(r, sign):     # 문서 규칙: 지지 / 반대 / 그 외 = 근거 없음(필요 일수 병기) -- 원 틀의 «검정력 부족»도 근거 없음
    v = V.verdict(r, sign); v["verdict_prereg"] = v["verdict"] if v["verdict"] in ("지지", "반대") else "근거 없음"
    return v


def main(csv: Path, wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(csv); S["t_ms"] = pd.to_datetime(S["recorded_at_utc"], format="ISO8601", utc=True).astype("int64") // 10**6
    S = S.sort_values("t_ms").reset_index(drop=True)
    px = V.Px(klines(wd)); dv = dvol(wd)
    dts, dc = dv["ts"].to_numpy(np.int64) + HR, dv["c"].to_numpy(float)
    hours = pd.date_range(T0, T1, freq="h", inclusive="left")
    rows, stale = [], 0
    for t in hours:
        tm = int(t.timestamp() * 1000)
        j = np.searchsorted(S["t_ms"].to_numpy(), tm, side="left") - 1      # < t 마지막
        if j < 0 or tm - S["t_ms"].iat[j] > HR:
            stale += 1; continue
        s = S.iloc[j]; k = np.searchsorted(dts, tm, side="right") - 1
        nx, pa = px.r5(tm, 1440), px.r5_past(tm, 1440)
        rows.append({"t": t, "day": f"{t:%Y-%m-%d}", "hour": t.hour, "spread": s["atm_spread_iv"], "dvol": dc[k],
                     "dvol24": dc[np.searchsorted(dts, tm + DAY, side="right") - 1] if tm + DAY <= dts[-1] else np.nan,
                     "rvn": V.ann_rv(nx, 5) if nx is not None else np.nan, "rvp": V.ann_rv(pa, 5) if pa is not None else np.nan,
                     "bp": (s["front_fwd"] / s["idx"] - 1) * 1e4, "rr7": s["cm7_rr"],
                     "r1h": px.ret(tm, tm + HR), "r24h": px.ret(tm, tm + DAY)})
    X = pd.DataFrame(rows)
    out = {"window_utc": [str(T0), str(T1)], "hours_expected": len(hours), "hours_dropped_stale_or_missing": stale,
           "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "future_rows_used_for_entry": False}
    # 1 ATM 호가 폭
    A = X.dropna(subset=["spread", "dvol", "rvn", "rvp"]).reset_index(drop=True)
    res1 = {"n_days": int(A["day"].nunique()), "spread_q": A["spread"].quantile([.05, .5, .95]).round(3).tolist()}
    for nm, D_ in (("hourly", A), ("daily00", A[A["hour"] == 0].reset_index(drop=True))):
        y = np.log(D_["rvn"].to_numpy()); z = zrank(D_["spread"].to_numpy())
        Xm = np.column_stack([np.ones(len(D_)), z, np.log(D_["dvol"]), np.log(D_["rvp"])])
        r = {"n": len(D_), "beta_rank_z": prereg_verdict(V.boot(lambda i: ols_b(Xm, y, i), D_["day"].to_numpy()), +1),
             "beta_raw_no_controls": V.boot(lambda i: ols_b(Xm[:, :2], y, i), D_["day"].to_numpy())}
        m = D_["dvol24"].notna().to_numpy()
        if nm == "hourly":       # 보조(판정 미사용): 다음 24h |Δ ln DVOL|
            ya = np.abs(np.log(D_["dvol24"].to_numpy()[m]) - np.log(D_["dvol"].to_numpy()[m]))
            r["aux_abs_dlogdvol_24h"] = V.boot(lambda i: ols_b(Xm[m], ya, i), D_["day"].to_numpy()[m])
        res1[nm] = r
    out["1_atm_spread"] = res1
    # 2 선도 − 지수
    B = X.dropna(subset=["bp", "rr7"]).reset_index(drop=True)
    w = S[(S["t_ms"] >= T0.value // 10**6) & (S["t_ms"] < T1.value // 10**6)].dropna(subset=["front_fwd", "idx"])
    bp10 = (w["front_fwd"] / w["idx"] - 1) * 1e4
    d10 = bp10.diff().where((w["front_exp_ms"] == w["front_exp_ms"].shift()) & (w["t_ms"].diff() <= 11 * MIN))
    bp, rr, blk = B["bp"].to_numpy(), B["rr7"].to_numpy(), B["day"].to_numpy()
    res2 = {"n": len(B), "n_days": int(B["day"].nunique()),
            "noise_descr": {"n_10min": len(w), "level_bp_q5_50_95": np.percentile(bp10, [5, 50, 95]).round(2).tolist(),
                            "sd_10min_change_bp": round(float(d10.std()), 3), "share_abs_le_3bp": round(float((bp10.abs() <= 3).mean()), 3)},
            "rho_bp_rr7": prereg_verdict(V.boot(lambda i: V.spear(bp[i], rr[i]), blk), +1)}
    for h in ("r1h", "r24h"):
        C = B.dropna(subset=[h]).reset_index(drop=True); z = zrank(C["bp"].to_numpy()); y = np.sign(C[h].to_numpy())
        Xm = np.column_stack([np.ones(len(C)), z])
        res2[f"dir_beta_sign_{h}"] = V.boot(lambda i: ols_b(Xm, y, i), C["day"].to_numpy())
    out["2_fwd_minus_index"] = res2
    (wd / "verdict_option_info.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
