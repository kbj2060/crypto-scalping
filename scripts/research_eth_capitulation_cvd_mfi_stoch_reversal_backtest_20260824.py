#!/usr/bin/env python3
"""'청산버스트(300%+)+OI급감 → CVD둔화/MFI다이버전스 흡수확인 → StochRSI(<=10 극단과매도
골든크로스+20돌파) 진입' 전략 fresh-forward 백테스트 (2026-08-24, 사용자 제안 v2)

⚠️ 실제 forceOrder 청산 이벤트 피드는 09-15까지 §12/§13 사전등록 실행보류 대상이라 미사용
(직전 백테스트와 동일 사유). "300% 이상 폭발"은 5분봉 거래량이 트레일링 24h 평균 대비 3배
이상인 단일봉으로 프록시. OI는 실제 감사된 5분 OI(TOTAL_ETHUSDT_metrics) 그대로 사용.

조건1 (청산버스트 프록시): 5분봉 거래량 >= 트레일링 24h 평균 거래량의 3.0배(300%)
                         + 같은 15분(3봉) 구간 OI 변화율 z<=-2(급감) + 하락 바(급락 방향)
조건2 (흡수확인, OR):
  경로A(CVD 둔화): 이벤트 이후 가격이 이벤트봉 저가 이하로 더 내려가는 시점에, 그 시점의
                  순매수델타(taker_buy-taker_sell, 3봉 롤링)가 이벤트봉 델타보다 덜 음수
  경로B(MFI 다이버전스): 그 시점 MFI(14)<=20 이면서 이벤트봉 MFI보다 높음(가격 저점 낮아지는데
                        MFI는 저점 안 낮아짐)
조건3 (트리거): 이벤트 이후 창 내에서 StochRSI(14,14,3,3) K가 최근 1시간 내 <=10을 터치한 뒤,
              봉 마감 시 K가 D를 상향돌파(골든크로스)하며 동시에 20을 상향돌파.

진입: 트리거 다음 봉 시가. SL=이벤트 이후 최저가(+0.1%버퍼). TP=R배수(1.5/2/3, 그리드 전체
보고). 24h 시간청산. 단일 포지션. 비용=표준 테이커 왕복 10bp.

직전 실험(eth_capitulation_oi_stoch_reversal_backtest_20260824)에서 배운 교훈 반영: 어떤
양성 결과든 (1)무조건부 랜덤진입 부트스트랩 대조군 (2)평균 vs 중앙값 (3)연도별 분해
(4)최근구간(OOS) 성과를 반드시 함께 보고한다.

Fresh-forward: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
결과: data/research/eth_capitulation_cvd_mfi_stoch_reversal_backtest_20260824.json
"""
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GAP_CACHE = REPO / "data" / "research" / "eth_5m_gap_after_20260217_with_taker_cache.csv"
OUT = REPO / "data" / "research" / "eth_capitulation_cvd_mfi_stoch_reversal_backtest_20260824.json"

VOL_BURST_RATIO = 3.0    # "300% 이상" = 트레일링 평균 대비 3.0배
VOL_BASELINE_BARS = 288  # 24h 트레일링 평균
OI_WIN = 3                # 15분(3봉) OI 변화 윈도우
OI_Z_TH = 2.0
Z_WINDOW = 2016            # 7일 트레일링 (z-score용)
MFI_N = 14
STOCH_TOUCH = 10.0         # "10 이하 극단 과매도"
TRIG_WIN_BARS = 24         # 이벤트 후 2시간 내 흡수확인+트리거
CVD_WIN = 3                # CVD 롤링 델타 윈도우(15분)
SL_BUFFER = 0.001
MAX_HOLD_BARS = 288        # 24h
COST_RT_TAKER = 0.0010


def fetch_gap_5m_full(start_ms: int) -> pd.DataFrame:
    if GAP_CACHE.exists():
        df = pd.read_csv(GAP_CACHE)
        df["ts"] = pd.to_datetime(df["ts"])
        return df
    rows = []
    cur = start_ms
    while True:
        url = ("https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=5m"
              f"&startTime={cur}&limit=1500")
        with urllib.request.urlopen(url, timeout=15) as resp:
            batch = json.load(resp)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1500:
            break
        cur = batch[-1][0] + 300_000
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close",
                                     "volume", "ct", "qv", "n", "taker_buy_base",
                                     "tq", "ig"])
    df = df[["open_time", "open", "high", "low", "close", "volume",
            "taker_buy_base"]].astype(float)
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms")
    df = df[["ts", "open", "high", "low", "close", "volume", "taker_buy_base"]]
    df = df.iloc[:-1]
    GAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GAP_CACHE, index=False)
    return df


def load_5m_with_taker() -> pd.DataFrame:
    a = pd.read_csv(REPO / "data" / "eth_5m_2021_2023_archive.csv")
    a["ts"] = pd.to_datetime(a["open_time"], unit="ms")
    b = pd.read_csv(REPO / "data" / "eth_5m_1year.csv")
    b["ts"] = pd.to_datetime(b["timestamp"])
    cols = ["ts", "open", "high", "low", "close", "volume", "taker_buy_base"]
    df = pd.concat([a[cols], b[cols]], ignore_index=True)
    last_ms = int(df["ts"].max().value // 10**6)
    gap = fetch_gap_5m_full(last_ms + 300_000)
    df = pd.concat([df, gap[cols]], ignore_index=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    for c in cols[1:]:
        df[c] = df[c].astype(float)
    return df


def load_oi() -> pd.Series:
    a = pd.read_csv(REPO / "data" / "TOTAL_ETHUSDT_metrics_2021_2023.csv")
    b = pd.read_csv(REPO / "data" / "TOTAL_ETHUSDT_metrics_2024_2026.csv")
    df = pd.concat([a, b], ignore_index=True)
    df["create_time"] = pd.to_datetime(df["create_time"])
    df = df.drop_duplicates(subset="create_time").sort_values("create_time")
    return df.set_index("create_time")["sum_open_interest"]


def causal_zscore(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    mu = s.rolling(window, min_periods=window // 4).mean()
    sd = s.rolling(window, min_periods=window // 4).std()
    return ((s - mu) / sd.replace(0, np.nan)).to_numpy()


def stoch_rsi_kd(close: np.ndarray, rsi_n=14, stoch_n=14, k_n=3, d_n=3):
    s = pd.Series(close)
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / rsi_n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_n, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    lo = rsi.rolling(stoch_n).min()
    hi = rsi.rolling(stoch_n).max()
    stoch = (rsi - lo) / (hi - lo).replace(0, np.nan) * 100
    k = stoch.rolling(k_n).mean()
    d = k.rolling(d_n).mean()
    return k.to_numpy(), d.to_numpy()


def compute_mfi(h, l, c, v, n=MFI_N):
    tp = (h + l + c) / 3.0
    mf = tp * v
    d = np.diff(tp, prepend=tp[0])
    pos = np.where(d > 0, mf, 0.0)
    neg = np.where(d < 0, mf, 0.0)
    pos_sum = pd.Series(pos).rolling(n).sum()
    neg_sum = pd.Series(neg).rolling(n).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - 100 / (1 + ratio)
    return mfi.to_numpy()


def build_frame() -> pd.DataFrame:
    df = load_5m_with_taker()
    oi = load_oi()
    df["oi"] = oi.reindex(df["ts"] + pd.Timedelta(minutes=5)).to_numpy()

    c, o, h, lo, v = (df[k].to_numpy() for k in ("close", "open", "high", "low", "volume"))
    tb = df["taker_buy_base"].to_numpy()
    ts_sell = v - tb
    delta = tb - ts_sell  # 순매수 델타(양수=매수우세)
    cvd_roll = pd.Series(delta).rolling(CVD_WIN).sum().to_numpy()

    n = len(df)
    vol_baseline = pd.Series(v).shift(1).rolling(VOL_BASELINE_BARS,
                                                  min_periods=VOL_BASELINE_BARS // 4).mean()
    vol_ratio = (v / vol_baseline.replace(0, np.nan)).to_numpy()

    oi_arr = df["oi"].to_numpy()
    oi_clean = np.where(oi_arr > 0, oi_arr, np.nan)
    oi_chg = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        oi_chg[OI_WIN:] = (oi_clean[OI_WIN:] - oi_clean[:-OI_WIN]) / oi_clean[:-OI_WIN]
    oi_z = causal_zscore(oi_chg, Z_WINDOW)

    down_bar = c < o
    up_bar = c > o

    df["vol_ratio"] = vol_ratio
    df["oi_z"] = oi_z
    df["cvd_roll"] = cvd_roll
    df["delta"] = delta
    df["mfi"] = compute_mfi(h, lo, c, v)
    df["k"], df["d"] = stoch_rsi_kd(c)

    df["liq_burst_long"] = (vol_ratio >= VOL_BURST_RATIO) & (oi_z <= -OI_Z_TH) & down_bar
    df["liq_burst_short"] = (vol_ratio >= VOL_BURST_RATIO) & (oi_z <= -OI_Z_TH) & up_bar
    return df


def run(df: pd.DataFrame, absorption_mode: str, r_mult: float) -> list:
    """absorption_mode: 'cvd'(경로A만), 'mfi'(경로B만), 'either'(OR, 원문 그대로)."""
    c = df["close"].to_numpy(); o = df["open"].to_numpy()
    h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    k = df["k"].to_numpy(); d = df["d"].to_numpy()
    mfi = df["mfi"].to_numpy(); cvd = df["cvd_roll"].to_numpy()
    liq_long = df["liq_burst_long"].to_numpy(); liq_short = df["liq_burst_short"].to_numpy()
    ts = df["ts"].to_numpy()
    n = len(c)

    gold = np.zeros(n, dtype=bool); dead = np.zeros(n, dtype=bool)
    gold[1:] = (k[:-1] < d[:-1]) & (k[1:] >= d[1:])
    dead[1:] = (k[:-1] > d[:-1]) & (k[1:] <= d[1:])
    k_touch10 = pd.Series(k <= STOCH_TOUCH).rolling(12, min_periods=1).max().astype(bool).to_numpy()
    k_touch90 = pd.Series(k >= 100 - STOCH_TOUCH).rolling(12, min_periods=1).max().astype(bool).to_numpy()
    long_stoch_trig = gold & np.concatenate([[False], (k[:-1] < 20) & (k[1:] >= 20)]) \
        & np.concatenate([[False], k_touch10[:-1]])
    short_stoch_trig = dead & np.concatenate([[False], (k[:-1] > 80) & (k[1:] <= 80)]) \
        & np.concatenate([[False], k_touch90[:-1]])

    trades = []
    pos = None
    pending = None  # (dir, event_i, deadline, worst_so_far)
    for i in range(n - 1):
        if pos is not None:
            if pos["dir"] == 1:
                hit_sl, hit_tp = lo[i] <= pos["sl"], h[i] >= pos["tp"]
            else:
                hit_sl, hit_tp = h[i] >= pos["sl"], lo[i] <= pos["tp"]
            exit_price = exit_kind = None
            if hit_sl:
                exit_price, exit_kind = pos["sl"], "sl"
            elif hit_tp:
                exit_price, exit_kind = pos["tp"], "tp"
            elif i - pos["entry_i"] >= MAX_HOLD_BARS:
                exit_price, exit_kind = c[i], "time"
            if exit_price is not None:
                gross = pos["dir"] * (exit_price / pos["entry"] - 1.0)
                trades.append({"entry_ts": str(ts[pos["entry_i"]]), "exit_ts": str(ts[i]),
                              "dir": "long" if pos["dir"] == 1 else "short",
                              "exit_kind": exit_kind, "hold_bars": int(i - pos["entry_i"]),
                              "gross": gross, "net_taker": gross - COST_RT_TAKER})
                pos = None
            continue

        if pending is not None:
            pdir, ev_i, deadline, worst = pending
            if pdir == 1:
                worst = min(worst, lo[i])
            else:
                worst = max(worst, h[i])
            pending = (pdir, ev_i, deadline, worst)

            price_extends = (worst <= lo[ev_i]) if pdir == 1 else (worst >= h[ev_i])
            abs_a = (cvd[i] > cvd[ev_i]) if pdir == 1 else (cvd[i] < cvd[ev_i])
            abs_b = ((mfi[i] <= 20) & (mfi[i] >= mfi[ev_i])) if pdir == 1 else \
                    ((mfi[i] >= 80) & (mfi[i] <= mfi[ev_i]))
            if absorption_mode == "cvd":
                absorbed = price_extends & abs_a
            elif absorption_mode == "mfi":
                absorbed = price_extends & abs_b
            else:
                absorbed = price_extends & (abs_a | abs_b)

            trig = long_stoch_trig[i] if pdir == 1 else short_stoch_trig[i]
            if absorbed and trig:
                entry = o[i + 1]
                sl = worst * (1 - SL_BUFFER) if pdir == 1 else worst * (1 + SL_BUFFER)
                risk = abs(entry - sl)
                if risk > 0:
                    tp = entry + pdir * r_mult * risk
                    pos = {"dir": pdir, "entry": entry, "sl": sl, "tp": tp, "entry_i": i + 1}
                pending = None
            elif i >= deadline:
                pending = None

        if pending is None and pos is None:
            if liq_long[i]:
                pending = (1, i, i + TRIG_WIN_BARS, lo[i])
            elif liq_short[i]:
                pending = (-1, i, i + TRIG_WIN_BARS, h[i])
    return trades


def _agg(sel: list) -> dict:
    if not sel:
        return {"n": 0}
    g = np.array([t["gross"] for t in sel]); nt = np.array([t["net_taker"] for t in sel])
    cum = np.cumsum(nt)
    mdd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
    kinds = {kd: int(sum(1 for t in sel if t["exit_kind"] == kd)) for kd in ("tp", "sl", "time")}
    return {"n": len(sel), "win_rate_net_taker": float((nt > 0).mean()),
           "mean_gross_bp": float(g.mean() * 1e4), "median_gross_bp": float(np.median(g) * 1e4),
           "avg_net_taker_bp": float(nt.mean() * 1e4), "sum_net_taker_pct": float(nt.sum() * 100),
           "mdd_net_taker_pct": float(mdd * 100), "exit_kinds": kinds,
           "avg_hold_bars": float(np.mean([t["hold_bars"] for t in sel]))}


def _yearly(trades: list) -> dict:
    return {y: _agg([t for t in trades if t["entry_ts"][:4] == y])
           for y in ("2022", "2023", "2024", "2025", "2026")}


def bootstrap_unconditional(df: pd.DataFrame, dir_: int, n_trades: int, r_mult: float,
                            n_boot: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    c = df["close"].to_numpy(); o = df["open"].to_numpy()
    h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    n = len(c)
    valid_lo, valid_hi = Z_WINDOW, n - MAX_HOLD_BARS - 2
    if n_trades == 0:
        return {"n_boot": 0, "note": "no observed trades to match"}
    boot_means = []
    for _ in range(n_boot):
        idxs = rng.integers(valid_lo, valid_hi, size=n_trades)
        grosses = []
        for i in idxs:
            entry = o[i + 1]
            ref = lo[i] if dir_ == 1 else h[i]
            sl = ref * (1 - SL_BUFFER) if dir_ == 1 else ref * (1 + SL_BUFFER)
            risk = abs(entry - sl)
            if risk <= 0:
                continue
            tp = entry + dir_ * r_mult * risk
            exit_price = c[min(i + MAX_HOLD_BARS, n - 1)]
            for j in range(i + 1, min(i + MAX_HOLD_BARS, n - 1)):
                if dir_ == 1 and lo[j] <= sl:
                    exit_price = sl; break
                if dir_ == 1 and h[j] >= tp:
                    exit_price = tp; break
                if dir_ == -1 and h[j] >= sl:
                    exit_price = sl; break
                if dir_ == -1 and lo[j] <= tp:
                    exit_price = tp; break
            grosses.append(dir_ * (exit_price / entry - 1.0))
        boot_means.append(float(np.mean(grosses)) if grosses else np.nan)
    boot_means = np.array(boot_means)
    return {"n_boot": n_boot, "n_trades_per_boot": n_trades,
           "boot_mean_gross_bp_mean": float(np.nanmean(boot_means) * 1e4),
           "boot_mean_gross_bp_std": float(np.nanstd(boot_means) * 1e4),
           "boot_mean_gross_bp_p5_p50_p95": [float(x * 1e4) for x in
                                              np.nanpercentile(boot_means, [5, 50, 95])]}


def main():
    df = build_frame()
    print(f"bars={len(df)} liq_burst_long_events={int(df['liq_burst_long'].sum())}"
         f" liq_burst_short_events={int(df['liq_burst_short'].sum())}")
    report = {
        "strategy": "청산버스트(300%+프록시)+OI급감 -> CVD둔화/MFI다이버전스 흡수확인 -> StochRSI(<=10 골든크로스+20돌파) 진입",
        "note_on_liq_feed_substitution": "실제 forceOrder 청산피드는 09-15 사전등록 게이트 보류 대상이라 미사용, 5분봉 거래량>=트레일링24h평균의 3.0배(300%) 프록시로 대체",
        "note_on_risk_section": "리스크관리 수치 미지정 — R배수 그리드(1.5/2/3), price-move 기준 gross/net bp가 1차 판정축",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "cost_model_rt_taker_bp": 10.0,
        "params": {"vol_burst_ratio": VOL_BURST_RATIO, "vol_baseline_bars": VOL_BASELINE_BARS,
                  "oi_win_bars": OI_WIN, "oi_z_th": OI_Z_TH, "stoch_touch": STOCH_TOUCH,
                  "trig_win_bars": TRIG_WIN_BARS, "cvd_win_bars": CVD_WIN,
                  "sl_buffer": SL_BUFFER, "max_hold_bars": MAX_HOLD_BARS},
        "liq_burst_long_events": int(df["liq_burst_long"].sum()),
        "liq_burst_short_events": int(df["liq_burst_short"].sum()),
        "grid": {},
    }
    for absorption_mode in ("either", "cvd", "mfi"):
        for r_mult in (1.5, 2.0, 3.0):
            trades = run(df, absorption_mode, r_mult)
            longs = [t for t in trades if t["dir"] == "long"]
            shorts = [t for t in trades if t["dir"] == "short"]
            key = f"{absorption_mode}_R{r_mult}"
            cell = {"all": _agg(trades), "long": _agg(longs), "short": _agg(shorts),
                   "yearly_long": _yearly(longs), "yearly_short": _yearly(shorts)}
            if longs:
                cell["bootstrap_long"] = bootstrap_unconditional(df, 1, len(longs), r_mult)
            if shorts:
                cell["bootstrap_short"] = bootstrap_unconditional(df, -1, len(shorts), r_mult)
            report["grid"][key] = cell
            a = cell["all"]
            bl = cell.get("bootstrap_long", {}).get("boot_mean_gross_bp_mean")
            print(f"{key}: n={a['n']} win={a.get('win_rate_net_taker', 0):.2f} "
                 f"mean={a.get('mean_gross_bp', 0):.1f}bp med={a.get('median_gross_bp', 0):.1f}bp "
                 f"sum_net={a.get('sum_net_taker_pct', 0):.1f}% "
                 f"(n_long={len(longs)},n_short={len(shorts)}, boot_long_mean={bl})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
