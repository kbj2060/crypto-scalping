#!/usr/bin/env python3
"""캡츄레이션+CVD/MFI+StochRSI 전략(eth_capitulation_cvd_mfi_stoch_reversal_backtest_20260824)
파라미터 넓은격자 스윕 (2026-08-24, 사용자 요청 — "파라미터를 다듬어서 해보면?")

목적: "최고 조합 찾기"가 아니라 결과가 파라미터 선택과 무관하게 구조적으로 무정보인지 재확인.
다중비교 편향 방지 설계:
  1. 탐색구간(2021-12-01~2024-12-31, ~65%)에서만 전체 격자를 스윕하고 분포를 그대로 보고한다
     (치팅 없음 — top-N만 보여주지 않고 전체 분포 통계를 함께 낸다).
  2. 탐색구간에서 economically 의미있는 문턱(n>=30 & sum_net_taker_pct>=5%)을 넘는 조합이
     있으면, 그 조합만 홀드아웃구간(2025-01-01~2026-08-24, 전혀 안 본 구간)에서 재검증한다.
  3. 홀드아웃에서 살아남지 못하면 "탐색구간에서 우연히 좋아 보였을 뿐"으로 판정한다.

스윕 파라미터: VOL_BURST_RATIO×OI_Z_TH×STOCH_TOUCH×R배수×흡수경로 = 6×6×5×6×3 = 3,240개.
CVD_WIN/MFI_N/TRIG_WIN_BARS/OI_WIN/SL_BUFFER/MAX_HOLD_BARS는 고정(스코프 유지).

fresh_forward_bar_by_bar=true 등 4항목은 원 실험과 동일(롤링 피쳐는 전부 causal, 날짜분할은
사후 거래분류일 뿐 재계산 아님).
결과: data/research/eth_capitulation_param_sweep_20260824.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "research" / "eth_capitulation_param_sweep_20260824.json"

_spec = importlib.util.spec_from_file_location(
    "cap2", REPO / "scripts" / "research_eth_capitulation_cvd_mfi_stoch_reversal_backtest_20260824.py")
cap2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cap2)

SEARCH_END = pd.Timestamp("2025-01-01")  # 탐색: ~2021-12-01~2024-12-31 (~65%)
HOLDOUT_END = pd.Timestamp("2026-08-24")  # 홀드아웃: 2025-01-01~ (~35%, 전혀 안 봄)

BURST_GRID = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
OIZ_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
TOUCH_GRID = [5, 10, 15, 20, 25]
R_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
ABS_MODES = ["either", "cvd", "mfi"]

MIN_N_FOR_CANDIDATE = 30
MIN_NET_PCT_FOR_CANDIDATE = 5.0


def run_combo(df: pd.DataFrame, burst_ratio: float, oi_z_th: float, stoch_touch: float,
             absorption_mode: str, r_mult: float) -> list:
    c = df["close"].to_numpy(); o = df["open"].to_numpy()
    h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    k = df["k"].to_numpy(); d = df["d"].to_numpy()
    mfi = df["mfi"].to_numpy(); cvd = df["cvd_roll"].to_numpy()
    vol_ratio = df["vol_ratio"].to_numpy(); oi_z = df["oi_z"].to_numpy()
    ts = df["ts"].to_numpy()
    n = len(c)

    down_bar = c < o
    up_bar = c > o
    liq_long = (vol_ratio >= burst_ratio) & (oi_z <= -oi_z_th) & down_bar
    liq_short = (vol_ratio >= burst_ratio) & (oi_z <= -oi_z_th) & up_bar

    gold = np.zeros(n, dtype=bool); dead = np.zeros(n, dtype=bool)
    gold[1:] = (k[:-1] < d[:-1]) & (k[1:] >= d[1:])
    dead[1:] = (k[:-1] > d[:-1]) & (k[1:] <= d[1:])
    k_touch = pd.Series(k <= stoch_touch).rolling(12, min_periods=1).max().astype(bool).to_numpy()
    k_touch_hi = pd.Series(k >= 100 - stoch_touch).rolling(12, min_periods=1).max().astype(bool).to_numpy()
    long_stoch_trig = gold & np.concatenate([[False], (k[:-1] < 20) & (k[1:] >= 20)]) \
        & np.concatenate([[False], k_touch[:-1]])
    short_stoch_trig = dead & np.concatenate([[False], (k[:-1] > 80) & (k[1:] <= 80)]) \
        & np.concatenate([[False], k_touch_hi[:-1]])

    trades = []
    pos = None
    pending = None
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
            elif i - pos["entry_i"] >= cap2.MAX_HOLD_BARS:
                exit_price, exit_kind = c[i], "time"
            if exit_price is not None:
                gross = pos["dir"] * (exit_price / pos["entry"] - 1.0)
                trades.append({"entry_ts": ts[pos["entry_i"]], "dir": pos["dir"],
                              "exit_kind": exit_kind, "gross": gross,
                              "net_taker": gross - cap2.COST_RT_TAKER})
                pos = None
            continue

        if pending is not None:
            pdir, ev_i, deadline, worst = pending
            worst = min(worst, lo[i]) if pdir == 1 else max(worst, h[i])
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
                sl = worst * (1 - cap2.SL_BUFFER) if pdir == 1 else worst * (1 + cap2.SL_BUFFER)
                risk = abs(entry - sl)
                if risk > 0:
                    tp = entry + pdir * r_mult * risk
                    pos = {"dir": pdir, "entry": entry, "sl": sl, "tp": tp, "entry_i": i + 1}
                pending = None
            elif i >= deadline:
                pending = None

        if pending is None and pos is None:
            if liq_long[i]:
                pending = (1, i, i + cap2.TRIG_WIN_BARS, lo[i])
            elif liq_short[i]:
                pending = (-1, i, i + cap2.TRIG_WIN_BARS, h[i])
    return trades


def agg(sel: list) -> dict:
    if not sel:
        return {"n": 0, "sum_net_taker_pct": 0.0, "win_rate": None}
    nt = np.array([t["net_taker"] for t in sel])
    return {"n": len(sel), "win_rate": float((nt > 0).mean()),
           "sum_net_taker_pct": float(nt.sum() * 100),
           "mean_net_bp": float(nt.mean() * 1e4), "median_net_bp": float(np.median(nt) * 1e4)}


def main():
    df = cap2.build_frame()
    ts = df["ts"]
    print(f"bars={len(df)}  search={ts.min()}~{SEARCH_END}  holdout={SEARCH_END}~{ts.max()}")

    combos = [(b, oz, st, am, r) for b in BURST_GRID for oz in OIZ_GRID
             for st in TOUCH_GRID for am in ABS_MODES for r in R_GRID]
    print(f"total combos: {len(combos)}")

    search_results = []
    all_records = []
    for idx, (b, oz, st, am, r) in enumerate(combos):
        trades = run_combo(df, b, oz, st, am, r)
        search_trades = [t for t in trades if t["entry_ts"] < SEARCH_END]
        holdout_trades = [t for t in trades if t["entry_ts"] >= SEARCH_END]
        s_agg = agg(search_trades)
        rec = {"burst_ratio": b, "oi_z_th": oz, "stoch_touch": st,
              "absorption_mode": am, "r_mult": r,
              "search": s_agg, "holdout": agg(holdout_trades)}
        all_records.append(rec)
        search_results.append(s_agg["sum_net_taker_pct"])
        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(combos)} done")

    search_arr = np.array(search_results)
    n_positive = int((search_arr > 0).sum())
    report = {
        "purpose": "파라미터 다듬기가 결과를 바꾸는지 넓은격자로 재확인 — 최고조합 검색이 아니라 분포 전체 보고",
        "search_window": ["2021-12-01", str(SEARCH_END.date())],
        "holdout_window": [str(SEARCH_END.date()), str(ts.max())],
        "grid_params": {"burst_ratio": BURST_GRID, "oi_z_th": OIZ_GRID,
                        "stoch_touch": TOUCH_GRID, "r_mult": R_GRID,
                        "absorption_mode": ABS_MODES},
        "fixed_params": {"cvd_win_bars": cap2.CVD_WIN, "mfi_n": cap2.MFI_N,
                         "trig_win_bars": cap2.TRIG_WIN_BARS, "oi_win_bars": cap2.OI_WIN,
                         "sl_buffer": cap2.SL_BUFFER, "max_hold_bars": cap2.MAX_HOLD_BARS},
        "n_combos": len(combos),
        "search_distribution": {
            "mean_sum_net_pct": float(search_arr.mean()),
            "median_sum_net_pct": float(np.median(search_arr)),
            "std_sum_net_pct": float(search_arr.std()),
            "pct_combos_net_positive": float(n_positive / len(search_arr) * 100),
            "min": float(search_arr.min()), "max": float(search_arr.max()),
            "percentiles_5_25_50_75_95": [float(x) for x in
                                          np.percentile(search_arr, [5, 25, 50, 75, 95])],
        },
    }

    candidates = [r for r in all_records if r["search"]["n"] >= MIN_N_FOR_CANDIDATE
                 and r["search"]["sum_net_taker_pct"] >= MIN_NET_PCT_FOR_CANDIDATE]
    candidates.sort(key=lambda r: -r["search"]["sum_net_taker_pct"])
    report["n_candidates_meeting_search_bar"] = len(candidates)
    report["candidate_bar"] = {"min_n": MIN_N_FOR_CANDIDATE,
                               "min_sum_net_pct": MIN_NET_PCT_FOR_CANDIDATE}
    report["candidates_top20_with_holdout"] = candidates[:20]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    print(f"\n탐색구간 분포(전체 {len(combos)}개 조합): "
         f"평균 {report['search_distribution']['mean_sum_net_pct']:.2f}% "
         f"중앙값 {report['search_distribution']['median_sum_net_pct']:.2f}% "
         f"양수비율 {report['search_distribution']['pct_combos_net_positive']:.1f}%")
    print(f"문턱(n>=30 & 누적net>=5%) 통과 조합: {len(candidates)}개")
    for r in candidates[:10]:
        h = r["holdout"]
        print(f"  burst={r['burst_ratio']} oiz={r['oi_z_th']} touch={r['stoch_touch']} "
             f"abs={r['absorption_mode']} R={r['r_mult']}  "
             f"search: n={r['search']['n']} net={r['search']['sum_net_taker_pct']:.1f}%  "
             f"| holdout: n={h['n']} net={h['sum_net_taker_pct']:.1f}%")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
