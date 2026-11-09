"""지지·저항 레벨 재계산 주기 5분 vs 1시간 -- 실측 청산 위치 예보 (2026-11-09, 사용자 «5분과 1시간 주기의 성능을 비교해줘»).

사전등록: docs/prereg/liqmap_sr_refresh_5m_vs_1h_20261109.md (48ce8287, 계산 전 커밋). 설계는 그 문서가 정본이다.
팔(τ = 매 5분 경계, cur = τ 에 닫힌 5분봉 종가, 쪽마다 최대 3개):
  A0 1시간 옛 화면 = 정시 지도(닫힌 1시간봉 24개) + [정시, τ) 5분봉이 지난 레벨 버림
  A1 1시간 지금 화면 = A0 + 모자라면 heatmap_bins 에서 채움(app.js srLevelsLive 11-09)
  B  5분 재계산 = _window_at(τ) 창(부분봉 포함)·cp = cur
지표: (τ, τ+H] 바이낸스 청산 $ 중 표시 레벨 ±tol 안 몫 C · 위약 P(레벨 ±40bp 옮김 평균) · 초과 E = C − P · 날 블록 부트스트랩.
실행: python scripts/research_liqmap_sr_refresh_5m_vs_1h_20261109.py [--selftest | --diag]
"""
from __future__ import annotations

import glob
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_liquidation_map_20260824 as LM  # noqa: E402

LAKE = Path.home() / "backups/crypto-scalping-server/data/lake"
WORK = Path("/home/kbj20/crypto-scalping/tmp/liqmap_sr_refresh_20261109")
VISION = "https://data.binance.vision/data/futures/um/daily/klines"
N_SHOW, LOOKBACK, SHIFT, BOOT = 3, 24, 40e-4, 2000
END = "2026-11-07"
COINS = {"ETH": "2026-09-21", "BTC": "2026-09-27", "SOL": "2026-09-27", "XRP": "2026-09-27", "SOXL": "2026-11-03"}


# ---------------------------------------------------------------- 데이터
def klines(coin: str, interval: str, d0: str, d1: str) -> pd.DataFrame:
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / f"{coin}_{interval}_{d0}_{d1}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows, d = [], date.fromisoformat(d0)
    while d <= date.fromisoformat(d1):
        sym = f"{coin}USDT"
        try:
            raw = urllib.request.urlopen(f"{VISION}/{sym}/{interval}/{sym}-{interval}-{d}.zip", timeout=60).read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            rows += [r.split(",") for r in z.read(z.namelist()[0]).decode().splitlines() if r[:1].isdigit()]
        except urllib.error.HTTPError:
            pass
        d += timedelta(days=1)
    k = pd.DataFrame({"t": [int(r[0]) // 1000 for r in rows], **{c: [float(r[i]) for r in rows] for c, i in
                                                                  (("high", 2), ("low", 3), ("close", 4), ("volume", 5))}})
    k = k.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    k["timestamp"] = pd.to_datetime(k.t, unit="s", utc=True)
    k.to_parquet(f)
    return k


def lake(sub: str, coin: str, d0: str, d1: str, cols: list[str]) -> pd.DataFrame:
    fs = [p for p in sorted(glob.glob(str(LAKE / sub / f"coin={coin}" / "date=*" / "*.parquet")))
          if d0 <= p.split("date=")[1][:10] <= d1]
    return pd.concat([pd.read_parquet(p, columns=cols) for p in fs], ignore_index=True) if fs else pd.DataFrame(columns=cols)


def events(coin: str, d0: str, loc: str) -> pd.DataFrame:
    """청산 (ts_ms, usd, loc). loc='mark' = 청산 초의 바이낸스 1초 마크(직전 5초 안), 'price' = 청산 자체 가격."""
    e = lake("binance/liquidations", coin, d0, END, ["ts_ms", "usd", "price"]).sort_values("ts_ms").reset_index(drop=True)
    if loc == "price":
        return e.assign(loc=e.price)[["ts_ms", "usd", "loc"]]
    m = lake("binance/mark_1s", coin, d0, END, ["ts_ms", "mark"]).sort_values("ts_ms")
    j = pd.merge_asof(e, m, on="ts_ms", direction="backward", tolerance=5000)
    print(f"  {coin} 마크 위치 없음 {j.mark.isna().sum()}/{len(j)}건 뺌", flush=True)
    return j.dropna(subset=["mark"]).rename(columns={"mark": "loc"})[["ts_ms", "usd", "loc"]].reset_index(drop=True)


# ---------------------------------------------------------------- 팔
def swept(m5: tuple, t0: int, t1: int, p: float) -> bool:
    """[t0, t1) 에 시작한 닫힌 5분봉 중 저가 ≤ p ≤ 고가가 있나 (app.js liqSweepIdx)."""
    a, b = np.searchsorted(m5[0], [t0, t1])
    return bool(((m5[1]["low"][a:b] <= p) & (p <= m5[1]["high"][a:b])).any())


def display(mp: dict, cur: float, m5: tuple, since: int, tau: int, fill: bool) -> list[float]:
    """app.js srLevelsLive(3) 재현 -- 쪽마다 안 쓸린 서버 레벨 앞 3개, fill 이면 heatmap_bins 로 채움. 반환 = 가격 목록(지지+저항)."""
    out = []
    for key, below in (("support_levels", True), ("resistance_levels", False)):
        ok = lambda p: (p < cur if below else p > cur) and not swept(m5, since, tau, p)  # noqa: E731
        keep = [lv["price"] for lv in mp.get(key) or [] if lv["price"] > 0 and ok(lv["price"])][:N_SHOW]
        if fill and len(keep) < N_SHOW:
            have = {lv["price"] for lv in mp.get(key) or []}
            more = sorted((b for b in mp.get("heatmap_bins") or [] if b["price"] not in have and b["weight_pct"] >= 0.05
                           and abs(b["price"] - cur) / cur <= 0.05 and ok(b["price"])),
                          key=lambda b: (-b["weight_pct"], abs(b["price"] - cur)))[:N_SHOW - len(keep)]
            keep += [b["price"] for b in more]
        out += keep
    return out


def build_arms(h1: pd.DataFrame, m5df: pd.DataFrame, taus: np.ndarray) -> dict[str, list[list[float]]]:
    h, m = LM._bar_arrays(h1), LM._bar_arrays(m5df)
    close5 = dict(zip(m[0] + 300, m[1]["close"]))                  # 봉 끝 시각 → 종가
    hour_maps: dict[int, tuple] = {}
    arms = {"A0": [], "A1": [], "B": []}
    for tau in taus:
        tau = int(tau); hour = tau // 3600 * 3600; cur = close5[tau]
        if hour not in hour_maps:
            i1 = int(np.searchsorted(h[0], hour))
            w = h1.iloc[max(0, i1 - LOOKBACK):i1].reset_index(drop=True)
            hour_maps[hour] = (LM.compute_spliced_levels(w, float(w.close.iloc[-1]), m5df[m5df.t < hour]) if len(w) >= 20 else {},)
        mp = hour_maps[hour][0]
        arms["A0"].append(display(mp, cur, m, hour, tau, False))
        arms["A1"].append(display(mp, cur, m, hour, tau, True))
        mb = LM.compute_spliced_levels(LM._window_at(h, m, tau, LOOKBACK), cur, m5df[m5df.t < tau])
        arms["B"].append(display(mb, cur, m, tau, tau, False))
    return arms


# ---------------------------------------------------------------- 채점
def capture(lv: list[float], loc: np.ndarray, usd: np.ndarray, tol: float) -> float:
    if not len(lv) or not usd.sum() > 0:
        return 0.0
    L = np.asarray(lv)[None, :]
    hit = (np.abs(loc[:, None] - L) / L <= tol).any(1)
    return float(usd[hit].sum() / usd.sum())


def score(arm: list[list[float]], taus: np.ndarray, ev: pd.DataFrame, H: int, tol: float) -> pd.DataFrame:
    t = ev.ts_ms.to_numpy(); loc = ev["loc"].to_numpy(); usd = ev.usd.to_numpy()
    a = np.searchsorted(t, taus * 1000, side="right"); b = np.searchsorted(t, (taus + H * 60) * 1000, side="right")
    rows = []
    for k, tau in enumerate(taus):
        if b[k] == a[k]:
            continue
        lo, us = loc[a[k]:b[k]], usd[a[k]:b[k]]
        c = capture(arm[k], lo, us, tol)
        p = (capture([x * (1 + SHIFT) for x in arm[k]], lo, us, tol) + capture([x * (1 - SHIFT) for x in arm[k]], lo, us, tol)) / 2
        rows.append((int(tau), c, p, c - p, len(arm[k]), int(b[k] - a[k]), float(us.sum())))
    df = pd.DataFrame(rows, columns=["tau", "C", "P", "E", "n_lv", "n_ev", "usd"])
    df["day"] = pd.to_datetime(df.tau, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return df


def boot_diff(x: pd.DataFrame, y: pd.DataFrame, col: str, seed: int = 7) -> dict:
    """같은 τ 짝 (x − y) → 날 평균 → 날 블록 부트스트랩. 단측 90% 하한 = 10분위."""
    j = x.merge(y, on=["tau", "day"], suffixes=("_x", "_y"))
    d = (j[f"{col}_x"] - j[f"{col}_y"]).groupby(j.day).mean().to_numpy()
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(1)
    return {"mean": round(float(d.mean()), 5), "lo90": round(float(np.quantile(bs, .10)), 5),
            "ci95": [round(float(np.quantile(bs, .025)), 5), round(float(np.quantile(bs, .975)), 5)], "days": int(len(d)), "taus": int(len(j))}


def summ(df: pd.DataFrame) -> dict:
    g = df.groupby("day")[["C", "P", "E", "n_lv"]].mean().mean()
    return {k: round(float(v), 4) for k, v in g.items()}


def run_coin(coin: str, loc: str, cfgs: list[tuple]) -> dict:
    d0 = COINS[coin]; warm = str(date.fromisoformat(d0) - timedelta(days=2))
    h1, m5 = klines(coin, "1h", warm, END), klines(coin, "5m", warm, END)
    end_s = int(pd.Timestamp(END, tz="UTC").timestamp()) + 86400
    taus = np.arange(int(pd.Timestamp(d0, tz="UTC").timestamp()), end_s, 300)
    taus = taus[np.isin(taus - 300, m5.t.to_numpy())]
    print(f"[{coin}] τ {len(taus)} · 1h {len(h1)} · 5m {len(m5)}", flush=True)
    arms = build_arms(h1, m5, taus)
    ev = events(coin, d0, loc)
    out = {"coin": coin, "loc": loc, "n_tau": int(len(taus)), "events": int(len(ev)), "usd": round(float(ev.usd.sum()))}
    for H, tol in cfgs:
        taus_ok = taus[taus + H * 60 <= end_s]
        s = {k: score(v[:len(taus_ok)], taus_ok, ev, H, tol) for k, v in arms.items()}
        dist = {k: round(float(np.nanmedian([abs(p / c - 1) * 1e4 for lv, c in zip(v, m5.set_index(m5.t + 300).close.reindex(taus).to_numpy())
                                              for p in lv] or [np.nan])), 1) for k, v in arms.items()}
        out[f"H{H}_tol{int(tol * 1e4)}"] = {
            "arms": {k: summ(v) for k, v in s.items()}, "dist_med_bp": dist, "taus_with_liq": int(len(s["B"])),
            "B-A1": {"E": boot_diff(s["B"], s["A1"], "E"), "C": boot_diff(s["B"], s["A1"], "C")},
            "B-A0": {"E": boot_diff(s["B"], s["A0"], "E"), "C": boot_diff(s["B"], s["A0"], "C")},
            "A1-A0": {"C": boot_diff(s["A1"], s["A0"], "C")},
        }
        if coin == "ETH" and loc == "mark" and (H, tol) == (60, 15e-4):
            j = s["B"].merge(s["A1"], on=["tau", "day"], suffixes=("_B", "_A1"))
            mi = (j.tau % 3600) // 60
            out["minute_in_hour_dC"] = {lab: round(float((j.C_B - j.C_A1)[(mi >= a) & (mi <= b)].mean()), 4)
                                        for lab, (a, b) in {"05-20": (5, 20), "25-40": (25, 40), "45-55": (45, 55), "00": (0, 0)}.items()}
            pd.concat({k: v for k, v in s.items()}, names=["arm"]).to_csv(WORK / "eth_main_per_tau.csv")
    return out


def selftest() -> None:
    assert capture([100.0], np.array([100.1, 101.0]), np.array([1.0, 3.0]), 15e-4) == 0.25
    assert capture([], np.array([100.0]), np.array([1.0]), 15e-4) == 0.0
    m = (np.array([0, 300]), {"low": np.array([99.0, 95.0]), "high": np.array([101.0, 96.0])})
    assert swept(m, 0, 300, 100.0) and not swept(m, 300, 600, 100.0) and swept(m, 0, 600, 95.5)
    mp = {"support_levels": [{"price": 99.0}, {"price": 98.0}], "resistance_levels": [{"price": 103.0}],
          "heatmap_bins": [{"price": 97.0, "weight_pct": .5}, {"price": 96.0, "weight_pct": .5}, {"price": 90.0, "weight_pct": 1.0}]}
    m2 = (np.array([0]), {"low": np.array([98.5]), "high": np.array([100.5])})   # 99 쓸림
    assert display(mp, 100.0, m2, 0, 300, False) == [98.0, 103.0]
    assert display(mp, 100.0, m2, 0, 300, True) == [98.0, 97.0, 96.0, 103.0]      # 90 은 ±5% 밖
    # 정시 τ 에서 A0 = A1 = B (부분봉 없음 · cp 같음) -- 실데이터 3일
    h1, m5 = klines("ETH", "1h", "2026-10-01", "2026-10-03"), klines("ETH", "5m", "2026-10-01", "2026-10-03")
    taus = np.arange(int(pd.Timestamp("2026-10-02", tz="UTC").timestamp()), int(pd.Timestamp("2026-10-03", tz="UTC").timestamp()), 3600)
    a = build_arms(h1, m5, taus)
    assert a["A0"] == a["B"] == a["A1"], "정시 A0=B 깨짐"
    print("selftest ok -- 몫·쓸림·화면 채우기·정시 A0=A1=B(24 정시)")


def main() -> None:
    res = [run_coin("ETH", "mark", [(60, 15e-4), (15, 15e-4), (60, 30e-4)]), run_coin("ETH", "price", [(60, 15e-4)])]
    for c in ("BTC", "SOL", "XRP", "SOXL"):
        res.append(run_coin(c, "price", [(60, 15e-4)]))
    (WORK / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


def diag() -> None:
    """사후 진단(판정에 안 씀, 결과 문서 §위약 결함): ±40bp 위약을 현재가 쪽/바깥으로 나누고 거울(같은 거리 반대쪽)과 비교. 15분 간격 표본."""
    h1, m5 = klines("ETH", "1h", "2026-09-19", END), klines("ETH", "5m", "2026-09-19", END)
    end_s = int(pd.Timestamp(END, tz="UTC").timestamp()) + 86400
    taus = np.arange(int(pd.Timestamp("2026-09-21", tz="UTC").timestamp()), end_s - 3600, 300)
    taus = taus[np.isin(taus - 300, m5.t.to_numpy())][::3]
    cur = m5.set_index(m5.t + 300).close.reindex(taus).to_numpy()
    arms = build_arms(h1, m5, taus)
    ev = events("ETH", "2026-09-21", "mark"); t = ev.ts_ms.to_numpy(); loc = ev["loc"].to_numpy(); usd = ev.usd.to_numpy()
    a = np.searchsorted(t, taus * 1000, side="right"); b = np.searchsorted(t, (taus + 3600) * 1000, side="right")
    for arm in ("A1", "B"):
        res = {k: [] for k in ("real", "toward", "away", "mirror")}
        for k in range(len(taus)):
            if b[k] == a[k]:
                continue
            lo, us, c, lv = loc[a[k]:b[k]], usd[a[k]:b[k]], cur[k], arms[arm][k]
            for key, L in (("real", lv), ("toward", [p * (1 + SHIFT) if p < c else p * (1 - SHIFT) for p in lv]),
                           ("away", [p * (1 - SHIFT) if p < c else p * (1 + SHIFT) for p in lv]), ("mirror", [2 * c - p for p in lv])):
                res[key].append(capture(L, lo, us, 15e-4))
        print(arm, {k: round(float(np.mean(v)), 4) for k, v in res.items()}, "n", len(res["real"]))


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else diag() if "--diag" in sys.argv else main()
