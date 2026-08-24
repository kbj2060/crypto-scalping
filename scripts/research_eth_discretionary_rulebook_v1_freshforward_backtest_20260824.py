#!/usr/bin/env python3
"""재량 룰북 v1 fresh-forward 백테스트 (2026-08-24)

룰: docs/eth_discretionary_manual_strategy_rulebook_20260824.md v1 그대로.
  - 레벨: scripts/live_liquidation_map_20260824.py::compute_liquidation_levels (배포본 재사용),
    매 완결 1h봉마다 직전 168개 1h봉으로 재계산(causal), 매 5m봉에서 현재가 위/아래
    weight 최대 레벨 1개씩 선택.
  - 진입 E1(롱): 종가가 지지 0~+0.5% 존 & StochRSI(14,14,3,3) K가 20 상향돌파 & RR>=1.0
    진입 E2(숏): 종가가 저항 -0.5%~0 존 & K가 80 하향돌파 & RR>=1.0
    체결: 신호봉 다음 5m봉 시가 (lookahead 없음). 단일 포지션.
  - 청산: SL=레벨±0.5% / TP=반대편 레벨, intrabar 터치(동일봉 동시터치=SL 우선, 보수적),
    288봉(24h) 경과 시 종가 시장가.
  - 비용: 헤드라인=표준 테이커 왕복 10bp(전 청산 경로), 보조=혼합(진입 테이커 5bp +
    TP만 메이커 3.1bp, SL/시간청산 테이커 5bp). 수수료 우대 가정 없음.

Fresh-forward 준수: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
고정 룰(파라미터 튜닝/선택 없음)이므로 전 기간을 연도별 + 캐노니컬 VAL(2025-09-01~12-31)/
OOS(2026-01-01~03-31)/OOS2(2026-04-01~) 구간별로 그대로 보고한다. research/dev score 용도.

결과: data/research/eth_discretionary_rulebook_v1_backtest_20260824.json
"""
import importlib.util
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GAP_CACHE = REPO / "data" / "research" / "eth_5m_gap_after_20260217_cache.csv"
OUT = REPO / "data" / "research" / "eth_discretionary_rulebook_v1_backtest_20260824.json"

_spec = importlib.util.spec_from_file_location(
    "liqmap", REPO / "scripts" / "live_liquidation_map_20260824.py")
_liqmap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_liqmap)
compute_liquidation_levels = _liqmap.compute_liquidation_levels

WINDOW_1H = 168          # 7일
TOUCH_PCT = 0.005        # 진입 터치존 0.5%
SL_BUF_PCT = 0.005       # 손절 버퍼 0.5%
MIN_RR = 1.0
MAX_HOLD_BARS = 288      # 24h
K_LOW, K_HIGH = 20.0, 80.0
COST_RT_TAKER = 0.0010   # 왕복 10bp
FEE_TAKER, FEE_MAKER = 0.0005, 0.00031


def fetch_gap_5m(start_ms: int) -> pd.DataFrame:
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
        nxt = batch[-1][0] + 300_000
        if len(batch) < 1500:
            break
        cur = nxt
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close",
                                     "volume", "ct", "qv", "n", "tb", "tq", "ig"])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].astype(float)
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms")
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df = df.iloc[:-1]  # 마지막(미완결 가능) 봉 제거
    GAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GAP_CACHE, index=False)
    return df


def load_5m() -> pd.DataFrame:
    a = pd.read_csv(REPO / "data" / "eth_5m_2021_2023_archive.csv")
    a["ts"] = pd.to_datetime(a["open_time"], unit="ms")
    b = pd.read_csv(REPO / "data" / "eth_5m_1year.csv")
    b["ts"] = pd.to_datetime(b["timestamp"])
    cols = ["ts", "open", "high", "low", "close", "volume"]
    df = pd.concat([a[cols], b[cols]], ignore_index=True)
    last_ms = int(df["ts"].max().value // 10**6)
    gap = fetch_gap_5m(last_ms + 300_000)
    df = pd.concat([df, gap[cols]], ignore_index=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    for c in cols[1:]:
        df[c] = df[c].astype(float)
    return df


def stoch_rsi_k(close: np.ndarray, rsi_n=14, stoch_n=14, k_n=3) -> np.ndarray:
    s = pd.Series(close)
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / rsi_n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_n, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    lo = rsi.rolling(stoch_n).min()
    hi = rsi.rolling(stoch_n).max()
    stoch = (rsi - lo) / (hi - lo).replace(0, np.nan) * 100
    return stoch.rolling(k_n).mean().to_numpy()


def hourly_levels(h1: pd.DataFrame) -> dict:
    """key = 1h open ts(그 시각부터 유효), value = 병합 레벨 [(price, weight), ...].
    시각 t에 유효한 레벨 = t 직전에 완결된 168개 1h봉으로 계산."""
    out = {}
    n = len(h1)
    for i in range(WINDOW_1H, n):
        win = h1.iloc[i - WINDOW_1H:i]
        res = compute_liquidation_levels(win, float(win["close"].iloc[-1]))
        levels = []
        if res.get("warmed_up"):
            for side in ("support_levels", "resistance_levels"):
                for lv in res[side]:
                    levels.append((float(lv["price"]), float(lv["weight_pct"])))
        out[h1["timestamp"].iloc[i]] = levels
    return out


def pick_levels(levels: list, price: float):
    """현재가 아래 weight 최대 지지 1개, 위 weight 최대 저항 1개."""
    sup = [(p, w) for p, w in levels if p < price]
    res = [(p, w) for p, w in levels if p > price]
    sup_p = max(sup, key=lambda x: x[1])[0] if sup else None
    res_p = max(res, key=lambda x: x[1])[0] if res else None
    return sup_p, res_p


def run_backtest(df: pd.DataFrame, sl_buf: float = SL_BUF_PCT,
                 tp_mode: str = "level", entry_mode: str = "touch",
                 use_rr: bool = True, sides: str = "both") -> list:
    """tp_mode: "level"=반대편 레벨(+RR필터), "r1"/"r2"=1R/2R 대칭 타겟(RR필터 생략).
    entry_mode: "touch"=레벨 ±0.5% 터치존+StochRSI(v1), "any"=StochRSI만(레벨은 SL/TP 전용),
    "fade"=StochRSI 없이 레벨 존 신규 진입(직전봉 존 밖→현재봉 존 안) 자체가 트리거.
    sides: "both"/"long"/"short"."""
    ts = df["ts"].to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    k = stoch_rsi_k(c)

    h1 = df.set_index("ts").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum"}).dropna().reset_index()
    h1 = h1.rename(columns={"ts": "timestamp"})
    h1["timestamp"] = h1["timestamp"].dt.tz_localize("UTC")
    lvl_by_hour = hourly_levels(h1)
    hour_keys = sorted(lvl_by_hour.keys())
    hour_np = np.array([t.tz_localize(None) for t in hour_keys], dtype="datetime64[ns]")

    long_trig = np.zeros(len(c), dtype=bool)
    short_trig = np.zeros(len(c), dtype=bool)
    long_trig[1:] = (k[:-1] < K_LOW) & (k[1:] >= K_LOW)
    short_trig[1:] = (k[:-1] > K_HIGH) & (k[1:] <= K_HIGH)

    # 각 5m봉에 유효한 시간대 레벨 인덱스 (해당 봉 시각 이하의 마지막 hour key)
    hour_idx = np.searchsorted(hour_np, ts, side="right") - 1

    trades = []
    pos = None  # dict(dir, entry, sl, tp, entry_i)
    for i in range(len(c) - 1):
        if pos is not None:
            # 진입봉부터 intrabar 터치 판정, 동시터치=SL 우선(보수적)
            if pos["dir"] == 1:
                hit_sl = lo[i] <= pos["sl"]
                hit_tp = h[i] >= pos["tp"]
            else:
                hit_sl = h[i] >= pos["sl"]
                hit_tp = lo[i] <= pos["tp"]
            exit_price = exit_kind = None
            if hit_sl:
                exit_price, exit_kind = pos["sl"], "sl"
            elif hit_tp:
                exit_price, exit_kind = pos["tp"], "tp"
            elif i - pos["entry_i"] >= MAX_HOLD_BARS:
                exit_price, exit_kind = c[i], "time"
            if exit_price is not None:
                gross = pos["dir"] * (exit_price / pos["entry"] - 1.0)
                fee_mixed = FEE_TAKER + (FEE_MAKER if exit_kind == "tp" else FEE_TAKER)
                trades.append({
                    "entry_ts": str(ts[pos["entry_i"]]), "exit_ts": str(ts[i]),
                    "rule": "E1" if pos["dir"] == 1 else "E2",
                    "entry": pos["entry"], "exit": exit_price, "exit_kind": exit_kind,
                    "hold_bars": int(i - pos["entry_i"]),
                    "gross": gross,
                    "net_taker": gross - COST_RT_TAKER,
                    "net_mixed": gross - fee_mixed,
                })
                pos = None
            else:
                continue

        if pos is None:
            if entry_mode != "fade" and not (long_trig[i] or short_trig[i]):
                continue
            hi_ix = hour_idx[i]
            if hi_ix < 0:
                continue
            sup, res = pick_levels(lvl_by_hour[hour_keys[hi_ix]], c[i])
            if sup is None or res is None:
                continue
            price = c[i]
            in_zone_long = sup <= price <= sup * (1 + TOUCH_PCT)
            in_zone_short = res * (1 - TOUCH_PCT) <= price <= res
            if entry_mode == "fade":
                prev = c[i - 1] if i > 0 else price
                want_long = in_zone_long and not (
                    sup <= prev <= sup * (1 + TOUCH_PCT))
                want_short = in_zone_short and not (
                    res * (1 - TOUCH_PCT) <= prev <= res)
            else:
                want_long = long_trig[i] and (entry_mode == "any" or in_zone_long)
                want_short = short_trig[i] and (entry_mode == "any" or in_zone_short)
            if sides == "long":
                want_short = False
            elif sides == "short":
                want_long = False
            rmult = {"r1": 1.0, "r2": 2.0}.get(tp_mode)
            if want_long:
                sl = sup * (1 - sl_buf)
                entry = o[i + 1]
                tp = res if tp_mode == "level" else entry + rmult * (entry - sl)
                rr = (res - price) / max(price - sl, 1e-12)
                if rmult is not None or not use_rr or rr >= MIN_RR:
                    pos = {"dir": 1, "entry": entry, "sl": sl, "tp": tp,
                           "entry_i": i + 1}
            elif want_short:
                sl = res * (1 + sl_buf)
                entry = o[i + 1]
                tp = sup if tp_mode == "level" else entry - rmult * (sl - entry)
                rr = (price - sup) / max(sl - price, 1e-12)
                if rmult is not None or not use_rr or rr >= MIN_RR:
                    pos = {"dir": -1, "entry": entry, "sl": sl, "tp": tp,
                           "entry_i": i + 1}
    return trades


def summarize(trades: list) -> dict:
    def agg(sel):
        if not sel:
            return {"n": 0}
        g = np.array([t["gross"] for t in sel])
        nt = np.array([t["net_taker"] for t in sel])
        nm = np.array([t["net_mixed"] for t in sel])
        cum = np.cumsum(nt)
        mdd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
        kinds = {kd: int(sum(1 for t in sel if t["exit_kind"] == kd))
                 for kd in ("tp", "sl", "time")}
        return {"n": len(sel), "win_rate_net_taker": float((nt > 0).mean()),
                "avg_gross_bp": float(g.mean() * 1e4),
                "avg_net_taker_bp": float(nt.mean() * 1e4),
                "avg_net_mixed_bp": float(nm.mean() * 1e4),
                "sum_net_taker_pct": float(nt.sum() * 100),
                "sum_net_mixed_pct": float(nm.sum() * 100),
                "mdd_net_taker_pct": float(mdd * 100),
                "exit_kinds": kinds,
                "avg_hold_bars": float(np.mean([t["hold_bars"] for t in sel]))}

    periods = {
        "2022": ("2022-01-01", "2023-01-01"), "2023": ("2023-01-01", "2024-01-01"),
        "2024": ("2024-01-01", "2025-01-01"), "2025": ("2025-01-01", "2026-01-01"),
        "2026ytd": ("2026-01-01", "2026-12-31"),
        "VAL_2025-09-01_2025-12-31": ("2025-09-01", "2026-01-01"),
        "OOS_2026-01-01_2026-03-31": ("2026-01-01", "2026-04-01"),
        "OOS2_2026-04-01_": ("2026-04-01", "2026-12-31"),
    }
    out = {"all": agg(trades),
           "by_rule": {r: agg([t for t in trades if t["rule"] == r])
                       for r in ("E1", "E2")}}
    for name, (a, b) in periods.items():
        sel = [t for t in trades if a <= t["entry_ts"][:10] < b]
        out[name] = agg(sel)
        out[name + "_by_rule"] = {r: agg([t for t in sel if t["rule"] == r])
                                  for r in ("E1", "E2")}
    return out


def main():
    df = load_5m()
    print(f"5m bars: {len(df)}  span: {df['ts'].iloc[0]} ~ {df['ts'].iloc[-1]}")
    t0 = time.time()
    trades = run_backtest(df)
    print(f"backtest done in {time.time() - t0:.0f}s, trades={len(trades)}")
    report = {
        "rulebook": "docs/eth_discretionary_manual_strategy_rulebook_20260824.md v1",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "date_boundaries_note": "캐노니컬 VAL/OOS + 연도별 전 기간(고정 룰, 튜닝 없음) + OOS2(2026-04-01~)",
        "cost_model": {"headline_rt_taker_bp": 10.0,
                       "mixed": "entry taker 5bp + TP maker 3.1bp / SL·time taker 5bp"},
        "span": [str(df["ts"].iloc[0]), str(df["ts"].iloc[-1])],
        "summary": summarize(trades),
        "trades_tail": trades[-5:],
    }
    # 강건성 변형(전 조합 보고, 선택 없음): "스탑이 타이트해서"라는 반론 검증용
    variants = {}
    for sl_buf, tp_mode in [(0.005, "r1"), (0.02, "level"), (0.02, "r1")]:
        vt = run_backtest(df, sl_buf=sl_buf, tp_mode=tp_mode)
        s = summarize(vt)
        variants[f"sl{sl_buf}_tp-{tp_mode}"] = {
            "all": s["all"], "by_rule": {r: {kk: s["by_rule"][r].get(kk)
                                             for kk in ("n", "win_rate_net_taker",
                                                        "avg_net_taker_bp",
                                                        "sum_net_taker_pct")}
                                         for r in ("E1", "E2")}}
    report["robustness_variants"] = variants
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"]["all"], indent=2))
    print("by_rule:", json.dumps(report["summary"]["by_rule"], indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
