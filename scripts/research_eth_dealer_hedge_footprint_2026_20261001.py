#!/usr/bin/env python3
"""딜러 헤지 흔적 검정 (간접 검증 2번, 2026-10-01) -- 옵션 고객 델타 흐름 뒤에 선물 테이커 흐름이 같은 방향으로 나오나.

부호 규약(코드로 확인: Deribit 체결 `direction` = 테이커 방향, selftest·이전 block_direction_check 참고):
  X_t = Σ(테이커 부호 × 수량 × BS델타(체결 iv·지수가)) [ETH] = 분 t 에 «고객이 산 델타».
  딜러 = 반대편 → 딜러 델타 −X_t → 중립화하려면 선물 +X_t 매수 → 선물 테이커 순매수 Y 가 **양(+)** 이어야 한다.

가설·판정 기준 (결과 보기 전에 고정):
  H1 (주 검정) Y_{t+1..t+k} 합 ~ β·X_t + 통제[X_{t-1..t-5}합, Y_t, Y_{t-1}, Y_{t-2..t-5}합, r_t, r_{t-1}, 시(UTC) 더미].
      주 = 바이낸스 ETHUSDT 선물, X_main(화면 체결 + 선물 다리 없는 블록; 강제청산·헤지 붙은 블록 제외), k=5.
      «통과» = 반기 1(01-01~05-31)·반기 2(06-01~09-30) 둘 다 β>0 이고 일 블록 부트스트랩 95% CI 가 0 배제.
      부 = k=1·15, Deribit ETH-PERPETUAL(블록 다리 제외 화면 체결), 합(바이낸스+Deribit), X 성분별(화면·블록·헤지블록 순델타·청산).
      같은 분 t 반응(Y_t ~ X_t)은 따로 보고만 한다(인과 방향 모호 -- 판정에 안 쓴다).
  H2 크기: β(k=15) 를 «고객 델타 1 ETH 당 선물 ETH» 헤지 비율로 읽는다. 바이낸스+Deribit 합에서 0.1 이상이면 «읽을 만함».
  H3 사건 연구: (a) |X_main| 상위 1% 분, (b) 선물 다리 없는 블록 중 |델타| 중앙값 이상 분.
      Y 에서 같은 분-of-day 평균을 뺀 Y_adj 를 사건 부호로 정렬해 전(−30..−1)·동시(0)·후(+1..+5/15/30) 합, 일 블록 CI.
  역인과: X_{t+1..t+k} 합 ~ Y_t + 같은 통제(Y 역할↔X 역할) → 선물이 옵션을 앞서면 «헤지»가 아니라 «따라 사기».
  누수: 옵션 흐름은 분 t 까지, 라벨은 t+1 부터(selftest 가 합성 데이터로 확인).
  기간: 2026-01-01~09-30(옵션 데이터 2026년만 규칙). Deribit 무기한은 일별로 받은 날만(부족하면 무작위 순서로 받은 표본).

데이터: 옵션 체결 tmp/dealer_position_accuracy_20260930/opt_trades_hist.parquet(history.deribit.com)
  · 바이낸스 1m klines(로컬 data/binance_vision/klines1m, 09-15~ 는 data.binance.vision 일 zip) -- 🔴바이낸스 API 호출 없음
  · Deribit ETH-PERPETUAL 체결(history.deribit.com, 일별 분 집계 캐시) · 블록 선물 다리(history, ms 조회).
  python scripts/research_eth_dealer_hedge_footprint_2026_20261001.py --fetch      # 받기(재개 가능)
  python scripts/research_eth_dealer_hedge_footprint_2026_20261001.py              # 분석 → results.json
  python scripts/research_eth_dealer_hedge_footprint_2026_20261001.py --selftest
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_eth_dealer_position_accuracy_20260930 import HIST, _get, greeks, parse  # noqa: E402

OUT = ROOT / "tmp/dealer_hedge_footprint_2026_20261001"
OPT = ROOT / "tmp/dealer_position_accuracy_20260930/opt_trades_hist.parquet"
KL = Path("/home/kbj20/crypto-scalping/data/binance_vision/klines1m")
D0, D1 = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")
SPLIT = pd.Timestamp("2026-06-01", tz="UTC")
KS = (1, 5, 15)
NB = 400


# ───────────────────────── 받기 ─────────────────────────
def fetch_perp_day(day: pd.Timestamp) -> None:
    """Deribit ETH-PERPETUAL 하루 체결 → 분 집계(y=테이커 순 ETH, 블록 다리 제외 y_nb)."""
    p = OUT / "perp" / f"{day:%Y-%m-%d}.parquet"
    if p.exists():
        return
    t, end, rows = int(day.timestamp() * 1000), int((day + pd.Timedelta(days=1)).timestamp() * 1000) - 1, []
    while True:
        r = _get(HIST, "get_last_trades_by_instrument_and_time", instrument_name="ETH-PERPETUAL", start_timestamp=t,
                 end_timestamp=end, count=1000, sorting="asc", include_old="true")
        tr = r["trades"]; rows += tr
        if not r.get("has_more") or not tr:
            break
        nt = int(tr[-1]["timestamp"]); t = nt if nt > t else t + 1
    d = pd.DataFrame(rows).drop_duplicates("trade_id").reset_index(drop=True)   # 🔴 인덱스 구멍이면 pd.Series(y) 와 m 이 어긋난다
    y =np.where(d["direction"] == "buy", 1.0, -1.0) * d["amount"] / d["price"]
    blk = d["block_trade_id"].notna() if "block_trade_id" in d else pd.Series(False, index=d.index)
    m = d["timestamp"] // 60_000
    pd.DataFrame({"y": pd.Series(y).groupby(m).sum(), "y_nb": pd.Series(np.where(blk, 0.0, y)).groupby(m).sum(),
                  "n": m.groupby(m).size()}).to_parquet(p)
    print(f"  perp {day:%m-%d} {len(d):,}건", flush=True)


def fetch_binance_day(day: pd.Timestamp) -> None:
    p = OUT / "binance" / f"{day:%Y-%m-%d}.parquet"
    if p.exists():
        return
    r = requests.get(f"https://data.binance.vision/data/futures/um/daily/klines/ETHUSDT/1m/ETHUSDT-1m-{day:%Y-%m-%d}.zip", timeout=60)
    if r.status_code != 200:
        print(f"  binance {day:%m-%d} 없음({r.status_code})"); return
    z = zipfile.ZipFile(io.BytesIO(r.content)); raw = z.read(z.namelist()[0]).decode()
    df = pd.read_csv(io.StringIO(raw), header=None)
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    pd.DataFrame({"t": df[0].astype("int64"), "h": df[2].astype(float), "l": df[3].astype(float), "c": df[4].astype(float),
                  "v": df[5].astype(float), "tb": df[9].astype(float)}).to_parquet(p)


def fetch_fut_legs(opt: pd.DataFrame) -> dict:
    """선물 다리 있는 블록(leg_count > 보이는 옵션 다리) → 같은 ms 선물 체결 중 같은 block_trade_id 의 순 ETH."""
    p = OUT / "block_fut_legs.json"
    cache = json.loads(p.read_text()) if p.exists() else {}
    b = opt[opt["block_trade_id"].notna()].groupby("block_trade_id").agg(n=("trade_id", "size"), lc=("block_trade_leg_count", "first"),
                                                                          ts=("timestamp", "first"))
    for bid, r in b[b["lc"] > b["n"]].iterrows():
        if bid in cache:
            continue
        tr = _get(HIST, "get_last_trades_by_currency_and_time", currency="ETH", kind="future", count=100,
                  start_timestamp=int(r["ts"]), end_timestamp=int(r["ts"]), include_old="true")["trades"]
        legs = [t for t in tr if t.get("block_trade_id") == bid]
        cache[bid] = {"q_eth": sum((1 if t["direction"] == "buy" else -1) * t["amount"] / t["price"] for t in legs), "n": len(legs)}
    p.write_text(json.dumps(cache))
    return cache


def fetch_all() -> None:
    (OUT / "perp").mkdir(parents=True, exist_ok=True); (OUT / "binance").mkdir(parents=True, exist_ok=True)
    opt = pd.read_parquet(OPT)
    legs = fetch_fut_legs(opt)
    print(f"블록 선물 다리 {len(legs)}건 · 다리 찾음 {sum(v['n'] > 0 for v in legs.values())}", flush=True)
    for d in pd.date_range("2026-09-14", "2026-09-30", tz="UTC"):
        fetch_binance_day(d)
    days = list(pd.date_range(D0, D1 - pd.Timedelta(days=1), tz="UTC"))
    np.random.default_rng(20261001).shuffle(days)   # 무작위 순서 → 중간에 끊겨도 받은 날이 두 반기의 무작위 표본
    with ThreadPoolExecutor(10) as ex:
        list(ex.map(fetch_perp_day, days))


# ───────────────────────── 분 패널 ─────────────────────────
def option_flow(opt: pd.DataFrame, legs: dict) -> pd.DataFrame:
    t = opt[(opt["timestamp"] >= D0.timestamp() * 1000) & (opt["timestamp"] < D1.timestamp() * 1000)].copy()
    pr = t["instrument_name"].map(parse); t = t[pr.notna()]; pr = pr[pr.notna()]
    K = np.array([x[0] for x in pr]); exp_ms = np.array([x[1].timestamp() * 1000 for x in pr]); sg = np.array([x[2] for x in pr])
    T = (exp_ms - t["timestamp"].to_numpy()) / (365 * 86_400_000)
    dl, _ = greeks(t["index_price"].to_numpy(), K, (t["iv"] / 100).to_numpy(), T, sg)
    t["xd"] = np.where(t["direction"] == "buy", 1.0, -1.0) * t["amount"].to_numpy() * dl      # 고객(테이커)이 산 델타 ETH
    t["m"] = t["timestamp"] // 60_000
    blk, liq = t["block_trade_id"].notna(), t["liquidation"].notna()
    hedged = t["block_trade_id"].map(lambda b: b in legs and legs[b]["n"] > 0)
    g = lambda mask: t[mask].groupby("m")["xd"].sum()
    out = pd.DataFrame({"X_screen": g(~blk & ~liq), "X_blk": g(blk & ~hedged), "X_blkh_opt": g(hedged), "X_liq": g(~blk & liq)})
    # 헤지 블록의 순델타(옵션 다리 + 선물 다리): 블록 첫 체결 분에 선물 다리 ETH 를 더한다
    hb = t[hedged].groupby("block_trade_id").agg(m=("m", "first"))
    hb["f"] = [legs[b]["q_eth"] for b in hb.index]
    out["X_blkh_net"] = out["X_blkh_opt"].add(hb.groupby("m")["f"].sum(), fill_value=0.0)
    out = out.fillna(0.0)
    out["X_main"] = out["X_screen"] + out["X_blk"]
    return out


def binance_minutes() -> pd.DataFrame:
    fs = [KL / f"ETHUSDT-1m-2026-{mo:02d}.parquet" for mo in range(1, 10)]
    k = pd.concat([pd.read_parquet(f) for f in fs if f.exists()] + [pd.read_parquet(f) for f in sorted((OUT / "binance").glob("*.parquet"))])
    k = k.drop_duplicates("t").sort_values("t")
    return pd.DataFrame({"Yb": (2 * k["tb"] - k["v"]).to_numpy(), "c": k["c"].to_numpy()}, index=k["t"].to_numpy() // 60_000)


def panel() -> tuple[pd.DataFrame, list]:
    opt = pd.read_parquet(OPT)
    legs = json.loads((OUT / "block_fut_legs.json").read_text())
    X = option_flow(opt, legs)
    B = binance_minutes()
    mins = np.arange(int(D0.timestamp() // 60), min(int(D1.timestamp() // 60), B.index.max() + 1, int(opt["timestamp"].max() // 60_000) + 1))
    df = pd.DataFrame(index=mins).join(X).join(B)
    df[X.columns] = df[X.columns].fillna(0.0)
    df["r"] = np.log(df["c"].ffill()).diff()
    perp_days = []
    parts = []
    for f in sorted((OUT / "perp").glob("*.parquet")):
        parts.append(pd.read_parquet(f)); perp_days.append(f.stem)
    if parts:
        P = pd.concat(parts)
        df["Yd"] = P["y_nb"].reindex(df.index)
        dd = pd.to_datetime(df.index * 60, unit="s").strftime("%Y-%m-%d")
        df.loc[dd.isin(perp_days), "Yd"] = df.loc[dd.isin(perp_days), "Yd"].fillna(0.0)   # 받은 날의 빈 분 = 0 체결
    else:
        df["Yd"] = np.nan
    df["Ys"] = df["Yb"] + df["Yd"]
    df["day"] = df.index // 1440
    df["hod"] = (df.index // 60) % 24
    df["half"] = np.where(df.index * 60 < SPLIT.timestamp(), 1, 2)
    return df, perp_days


# ───────────────────────── 회귀(일별 충분통계 + 일 블록 부트스트랩) ─────────────────────────
def design(df: pd.DataFrame, y: str, x: str, k: int, same_minute: bool = False):
    """반환 (Z, target, day). 타깃 = y_{t+1..t+k} 합(same_minute 면 y_t). 통제 = x 시차합·y_t(동시판 제외)·y 시차·r_t·r_{t-1}·시 더미."""
    Y, Xs = df[y], df[x]
    tgt = Y if same_minute else sum(Y.shift(-j) for j in range(1, k + 1))
    cols = {"x": Xs, "x_l5": sum(Xs.shift(j) for j in range(1, 6)), "y_l1": Y.shift(1), "y_l5": sum(Y.shift(j) for j in range(2, 6)),
            "r_l1": df["r"].shift(1)}
    if not same_minute:
        cols |= {"y0": Y, "r0": df["r"]}
    Z = pd.DataFrame(cols)
    H = pd.get_dummies(df["hod"], prefix="h", drop_first=True).astype(float)
    Z = pd.concat([pd.Series(1.0, index=df.index, name="c"), Z, H], axis=1)
    ok = Z.notna().all(axis=1) & tgt.notna()
    return Z[ok].to_numpy(float), tgt[ok].to_numpy(float), df.loc[ok, "day"].to_numpy()


def ols_boot(Z, t, day, seed=7) -> list:
    ud, inv = np.unique(day, return_inverse=True)
    cut = np.flatnonzero(np.diff(inv)) + 1          # day 는 정렬돼 있다(분 순서)
    A = np.array([z.T @ z for z in np.split(Z, cut)]); b = np.array([z.T @ y for z, y in zip(np.split(Z, cut), np.split(t, cut))])
    beta = np.linalg.lstsq(A.sum(0), b.sum(0), rcond=None)[0][1]
    rng = np.random.default_rng(seed); bs = []
    for _ in range(NB):
        w = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud)).astype(float)
        bs.append(np.linalg.lstsq(np.tensordot(w, A, 1), w @ b, rcond=None)[0][1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [round(float(beta), 4), round(float(lo), 4), round(float(hi), 4), int(len(ud))]


def fit(df, y, x, k, same=False):
    return ols_boot(*design(df, y, x, k, same)) if df[y].notna().sum() > 1440 * 5 else None


# ───────────────────────── 사건 연구 ─────────────────────────
def event_study(df: pd.DataFrame, y: str, ev: np.ndarray, sgn: np.ndarray, size: np.ndarray) -> dict:
    """ev = 사건 분(패널 위치). Y_adj = y − 같은 분-of-day 평균. 사건 부호로 정렬한 창 합 + 일 블록 CI."""
    Y = df[y].to_numpy(); mod = df.index.to_numpy() % 1440
    ok = ~np.isnan(Y)
    mu = pd.Series(Y[ok]).groupby(mod[ok]).mean().reindex(range(1440)).to_numpy()
    Ya = Y - mu[mod]
    n = len(Ya)
    keep = (ev - 30 >= 0) & (ev + 30 < n)
    ev, sgn, size = ev[keep], sgn[keep], size[keep]
    W = np.array([Ya[e - 30:e + 31] for e in ev]) * sgn[:, None]          # 열 0 = −30 … 열 30 = 0 … 열 60 = +30
    good = ~np.isnan(W).any(axis=1)
    W, size, day = W[good], size[good], df["day"].to_numpy()[ev[good]]
    if len(W) < 30:
        return {"n": int(len(W))}
    stats = {"pre30": W[:, :30].sum(1), "same": W[:, 30], "post5": W[:, 31:36].sum(1), "post15": W[:, 31:46].sum(1), "post30": W[:, 31:].sum(1)}
    ud, inv = np.unique(day, return_inverse=True); rng = np.random.default_rng(3)
    out = {"n": int(len(W)), "days": int(len(ud)), "mean_abs_x": round(float(size.mean()), 2)}
    for nm, v in stats.items():
        sd = np.bincount(inv, v); cd = np.bincount(inv)
        bs = []
        for _ in range(NB):
            w = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud))
            bs.append((w * sd).sum() / (w * cd).sum())
        out[nm] = [round(float(v.mean()), 2)] + [round(float(q), 2) for q in np.percentile(bs, [2.5, 97.5])]
    out["post15_per_eth"] = round(float(stats["post15"].mean() / size.mean()), 4)
    out["profile_cum"] = [round(float(c), 2) for c in np.cumsum(W.mean(0))]
    return out


SEC_WIN = {"-60~-10s": (-60_000, -10_000), "-10~-1s": (-10_000, -1_000), "-1~0s": (-1_000, 0), "0~+1s": (0, 1_000),
           "+1~+10s": (1_000, 10_000), "+10~+60s": (10_000, 60_000)}


def window_sums(ev_ts: np.ndarray, ts: np.ndarray, y: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """사건 시각 e 마다 ts ∈ [e+lo, e+hi) 인 y 합(ts 정렬). 0~+1s 는 같은 ms 를 포함, −1~0s 는 제외."""
    c = np.concatenate([[0.0], np.cumsum(y)])
    return c[np.searchsorted(ts, ev_ts + hi, "left")] - c[np.searchsorted(ts, ev_ts + lo, "left")]


def subminute_deribit() -> dict:
    """보조(판정 밖, 결과 본 뒤 추가): 분 단위 «같은 분» 반응이 헤지인지 동행인지 -- 최근 7일 체결 단위로
    옵션 고객 델타(같은 ms 묶음) 전후 초 창의 Deribit 무기한 테이커 순(블록 다리 제외)을 원점 통과 기울기 Σx·Y/Σx² 로."""
    pp = pd.read_parquet(ROOT / "tmp/dealer_position_accuracy_20260930/perp_trades.parquet").drop_duplicates("trade_id")
    pp = pp[pp["block_trade_id"].isna()].sort_values("timestamp")
    ts = pp["timestamp"].to_numpy(np.int64); y = (np.where(pp["direction"] == "buy", 1.0, -1.0) * pp["amount"] / pp["price"]).to_numpy()
    opt = pd.read_parquet(OPT)
    opt = opt[(opt["timestamp"] >= ts.min() + 60_000) & (opt["timestamp"] < ts.max() - 60_000)]
    legs = json.loads((OUT / "block_fut_legs.json").read_text())
    opt = opt[opt["liquidation"].isna() & ~opt["block_trade_id"].map(lambda b: b in legs and legs[b]["n"] > 0)].reset_index(drop=True)
    pr = opt["instrument_name"].map(parse); opt = opt[pr.notna()].reset_index(drop=True); pr = pr[pr.notna()].reset_index(drop=True)
    T = (np.array([x[1].timestamp() * 1000 for x in pr]) - opt["timestamp"].to_numpy()) / (365 * 86_400_000)
    dl, _ = greeks(opt["index_price"].to_numpy(), np.array([x[0] for x in pr]), (opt["iv"] / 100).to_numpy(), T, np.array([x[2] for x in pr]))
    ev = (pd.Series(np.where(opt["direction"] == "buy", 1.0, -1.0) * opt["amount"].to_numpy() * dl).groupby(opt["timestamp"].to_numpy()).sum())
    ev = ev[ev.abs() > 0]
    e_ts, x = ev.index.to_numpy(np.int64), ev.to_numpy()
    hr = e_ts // 3_600_000; uh, inv = np.unique(hr, return_inverse=True); rng = np.random.default_rng(9)
    out = {"events": int(len(x)), "hours": int(len(uh)), "from": str(pd.Timestamp(int(e_ts.min()), unit="ms")), "to": str(pd.Timestamp(int(e_ts.max()), unit="ms"))}
    for nm, (lo, hi) in SEC_WIN.items():
        Y = window_sums(e_ts, ts, y, lo, hi)
        sxy, sxx = np.bincount(inv, x * Y), np.bincount(inv, x * x)
        bs = []
        for _ in range(NB):
            w = np.bincount(rng.integers(0, len(uh), len(uh)), minlength=len(uh)); bs.append((w * sxy).sum() / (w * sxx).sum())
        out[nm] = [round(float(sxy.sum() / sxx.sum()), 4)] + [round(float(q), 4) for q in np.percentile(bs, [2.5, 97.5])]
    return out


# ───────────────────────── 분석 ─────────────────────────
def analyze() -> None:
    df, perp_days = panel()
    first = lambda s: str(pd.Timestamp(int(s) * 60, unit="s"))
    res = {"range": {"from": first(df.index.min()), "to": first(df.index.max()), "minutes": len(df),
                     "binance_minutes": int(df["Yb"].notna().sum()), "deribit_days": len(perp_days),
                     "deribit_days_h1": sum(d < "2026-06-01" for d in perp_days), "deribit_days_h2": sum(d >= "2026-06-01" for d in perp_days)},
           "x_share_abs": {c: round(float(df[c].abs().sum() / df[["X_screen", "X_blk", "X_blkh_opt", "X_liq"]].abs().sum().sum()), 4)
                           for c in ("X_screen", "X_blk", "X_blkh_opt", "X_liq")},
           "blkh_net_over_opt_abs": round(float(df["X_blkh_net"].abs().sum() / max(df["X_blkh_opt"].abs().sum(), 1e-9)), 4)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    halves = {"all": df, "h1": df[df["half"] == 1], "h2": df[df["half"] == 2]}
    dd = df[df["Yd"].notna()]   # Deribit 받은 날만(바이낸스 같은 날 비교용)
    targets = {"binance": ("Yb", halves), "binance_on_deribit_days": ("Yb", {h: g[g["Yd"].notna()] for h, g in halves.items()}),
               "deribit": ("Yd", {h: g[g["Yd"].notna()] for h, g in halves.items()}),
               "sum": ("Ys", {h: g[g["Ys"].notna()] for h, g in halves.items()})}
    h1 = {}
    for tn, (yc, hs) in targets.items():
        for xc in ("X_main", "X_screen", "X_blk", "X_blkh_opt", "X_blkh_net", "X_liq"):
            if tn != "binance" and xc not in ("X_main",):
                continue
            for h, g in hs.items():
                row = {f"k{k}": fit(g, yc, xc, k) for k in KS}
                row["same_minute"] = fit(g, yc, xc, 0, same=True)
                h1[f"{tn}|{xc}|{h}"] = row
                print(tn, xc, h, row, flush=True)
    res["H1"] = h1
    # 역인과: 선물 흐름 → 다음 k 분 옵션 고객 델타
    rev = {}
    for tn, yc in (("binance", "Yb"), ("deribit", "Yd")):
        for h, g in halves.items():
            g = g[g[yc].notna()]
            rev[f"{tn}|{h}"] = {f"k{k}": fit(g, "X_main", yc, k) for k in KS}
            print("rev", tn, h, rev[f"{tn}|{h}"], flush=True)
    res["reverse"] = rev
    # 판정(사전 고정 기준)
    pr = lambda h: h1[f"binance|X_main|{h}"]["k5"]
    passed = all(pr(h)[0] > 0 and pr(h)[1] > 0 for h in ("h1", "h2"))
    res["verdict_primary"] = {"binance_Xmain_k5_h1": pr("h1"), "binance_Xmain_k5_h2": pr("h2"), "pass": passed}
    # H3 사건 연구
    x = df["X_main"].to_numpy(); ax = np.abs(x)
    q99 = np.quantile(ax[ax > 0], 0.99)
    evA = np.flatnonzero(ax >= q99)
    xb = df["X_blk"].to_numpy(); ab = np.abs(xb)
    evB = np.flatnonzero((ab > 0) & (ab >= np.median(ab[ab > 0])))
    h3 = {"thr_top1pct_abs_delta": round(float(q99), 2), "thr_block_median_abs_delta": round(float(np.median(ab[ab > 0])), 2)}
    for en, ev, xs in (("top1pct", evA, x), ("block", evB, xb)):
        for tn, yc in (("binance", "Yb"), ("deribit", "Yd"), ("sum", "Ys")):
            for h in ("all", "h1", "h2"):
                e = ev[df["half"].to_numpy()[ev] == (1 if h == "h1" else 2)] if h != "all" else ev
                h3[f"{en}|{tn}|{h}"] = event_study(df, yc, e, np.sign(xs[e]), np.abs(xs[e]))
                print("H3", en, tn, h, {k: v for k, v in h3[f"{en}|{tn}|{h}"].items() if k != "profile_cum"}, flush=True)
    # H3' (결과 본 뒤 추가 -- 원시 프로파일은 전 30분 합이 후보다 커서 추세 교란): Y 를 자기 시차 1~5 로 AR 회귀한 잔차(=예상 못 한 흐름)로 같은 사건 연구
    for yc in ("Yb", "Ys"):
        L = pd.concat({j: df[yc].shift(j) for j in range(1, 6)}, axis=1); ok = L.notna().all(axis=1) & df[yc].notna()
        A = np.column_stack([np.ones(ok.sum()), L[ok].to_numpy()])
        df[yc + "_res"] = np.nan
        df.loc[ok, yc + "_res"] = df.loc[ok, yc] - A @ np.linalg.lstsq(A, df.loc[ok, yc].to_numpy(), rcond=None)[0]
    for en, ev, xs in (("top1pct", evA, x), ("block", evB, xb)):
        for yc in ("Yb_res", "Ys_res"):
            for h in ("all", "h1", "h2"):
                e = ev[df["half"].to_numpy()[ev] == (1 if h == "h1" else 2)] if h != "all" else ev
                h3[f"{en}|{yc}|{h}"] = event_study(df, yc, e, np.sign(xs[e]), np.abs(xs[e]))
                print("H3res", en, yc, h, {k: v for k, v in h3[f"{en}|{yc}|{h}"].items() if k != "profile_cum"}, flush=True)
    res["H3"] = h3
    # 참고: X 와 가격의 관계(정보거래·모멘텀 교란)
    res["corr"] = {"X_main_vs_r_same": round(float(df["X_main"].corr(df["r"])), 4),
                   "X_main_vs_r_next5": round(float(df["X_main"].corr(sum(df["r"].shift(-j) for j in range(1, 6)))), 4),
                   "X_main_vs_Yb_same": round(float(df["X_main"].corr(df["Yb"])), 4),
                   "Yb_vs_r_same": round(float(df["Yb"].corr(df["r"])), 4),
                   "Yb_std_per_min": round(float(df["Yb"].std()), 1), "X_main_std_per_min": round(float(df["X_main"].std()), 2),
                   "daily_abs_X_main_mean": round(float(df.groupby("day")["X_main"].apply(lambda s: s.abs().sum()).mean()), 1)}
    res["subminute_deribit_7d"] = subminute_deribit()
    print("초 단위", res["subminute_deribit_7d"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장", OUT / "results.json")


def selftest() -> None:
    rng = np.random.default_rng(0)
    n = 1440 * 20
    X = rng.standard_normal(n) * (rng.random(n) < 0.3)
    noise = rng.standard_normal(n) * 3
    # 딜러가 분 t 의 고객 델타 X 의 절반을 분 t+2 에 선물로 산다
    Y = noise + 0.5 * np.roll(X, 2); Y[:2] = noise[:2]
    df = pd.DataFrame({"X_main": X, "Yb": Y, "r": rng.standard_normal(n) * 1e-3}, index=np.arange(n) + 29_000_000)
    df["day"] = df.index // 1440; df["hod"] = (df.index // 60) % 24
    b1, b5 = fit(df, "Yb", "X_main", 1), fit(df, "Yb", "X_main", 5)
    assert abs(b1[0]) < 0.1 and b1[1] < 0 < b1[2], b1            # t+1 에는 아직 없다(시차 정렬)
    assert 0.4 < b5[0] < 0.6 and b5[1] > 0, b5                   # t+1..t+5 에 들어온다(부호 +)
    same = fit(df, "Yb", "X_main", 0, same=True)
    assert same[1] < 0 < same[2], same                           # 동시 분에는 없다(미래 X 가 섞이지 않음)
    # 역방향: 라벨을 X 의 과거에 두면(누수) 잡혀야 하는데 우리 설계는 t+1.. 만 쓰므로 X 를 앞당긴 Y 는 k=1 에서 0
    df2 = df.assign(Yb=noise + 0.5 * np.roll(X, -3))             # Y 가 X 를 3분 앞섬 = 역인과
    rv = fit(df2, "X_main", "Yb", 5)
    assert 0.012 < rv[0] < 0.022 and rv[1] > 0, rv               # 역인과 회귀가 잡는다(기대 0.5·0.3/9.08≈0.0165)
    # 부호 규약: 콜 매수 → 델타 +, 풋 매수 → 델타 −
    dl, _ = greeks(np.array([3000.0, 3000.0]), np.array([3000.0, 3000.0]), np.array([0.6, 0.6]), np.array([0.05, 0.05]), np.array([1.0, -1.0]))
    assert dl[0] > 0 > dl[1]
    # 사건 연구 정렬: 사건 뒤 +1..+5 분에 부호 맞춰 넣은 흐름이 post5 에 잡히고 pre30 은 0 근처
    Y3 = rng.standard_normal(n) * 0.1; ev = np.arange(100, n - 100, 499); sg = np.where(np.arange(len(ev)) % 2, 1.0, -1.0)
    for e, s in zip(ev, sg):
        Y3[e + 1:e + 6] += s
    es = event_study(df.assign(Yb=Y3), "Yb", ev, sg, np.ones(len(ev)))
    assert 4.5 < es["post5"][0] < 5.5 and abs(es["pre30"][0]) < 1.0 and abs(es["same"][0]) < 0.3, es
    # 초 창 경계: 같은 ms 는 0~+1s 에만, 직전 ms 는 −1~0s 에만
    ts, yy = np.array([999, 1000, 1000, 1999, 2000]), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert window_sums(np.array([1000]), ts, yy, 0, 1000)[0] == 9.0 and window_sums(np.array([1000]), ts, yy, -1000, 0)[0] == 1.0
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else fetch_all() if a.fetch else analyze()
