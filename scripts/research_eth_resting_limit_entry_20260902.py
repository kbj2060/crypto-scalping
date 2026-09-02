#!/usr/bin/env python3
"""지정가(peg-maker) 진입으로 앵커 미래참조를 우회할 수 있는가 (2026-09-02).

착상 (사용자)
------------
봉마다 "지금이 클라이맥스인가"를 판정하는 대신, **깊은 가격에 지정가를 걸어두고 시장이 오게
한다.** 지정가 주문은 '지금 결정'을 요구하지 않는다. 앵커는 결국 그 버스트의 극단 **가격**이므로,
깊은 곳의 지정가는 가격이 극단으로 갈 때 체결된다. "어느 봉인가"(알 수 없음)를 "어느
가격인가"(내가 정함)로 바꾼다.

이건 앞서 시도한 5가지(첫발동/후행최극단/지연확정/절대임계/학습필터)와 **메커니즘이 다르다** --
그것들은 전부 taker 즉시 진입이었다. 비용 축이 아니라 **정보 축**을 건드리는 유일한 미시도 안이다.

설계
----
완전 인과 상태기계. 매 봉 순차 진행:
  FLAT + 트리거(원시, 그 봉에서 알 수 있음) -> PENDING: 지정가 = close_i * (1 -+ depth*atr_i)
  PENDING -> 다음 봉부터 low<=limit(롱) / high>=limit(숏)이면 **limit 가격에** 체결 -> OPEN
             wait_bars 안에 안 오면 취소 -> FLAT (거래 없음, 비용 없음)
  OPEN  -> 기존과 동일한 ATR 트레일링(초기 SL / ARM / Trail), horizon 만료시 종가청산
1슬롯이므로 PENDING/OPEN 중에는 새 트리거를 무시한다(대기 주문도 슬롯 점유 -- 현실적·보수적).

비용: 진입 maker / 청산 taker. 트레일링 청산은 스톱 자리에 지정가를 놓을 수 없어 본질적으로
taker다. 두 가정으로 각각 잰다 -- 10bp(기존 전 taker 가정, 보수) / 7bp(maker 진입 반영).

대조군: depth=0(= 다음봉 시가 즉시 taker 진입)을 같은 상태기계로 돌려 기존 결과와 대조한다.
방향뒤집기는 side를 뒤집어 대칭으로 건다(롱 트리거 -> 위쪽에 매도 지정가).
HOLDOUT은 진단으로만 표시한다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, VAL_START)

OUT_DIR = ROOT / "tmp/eth_resting_limit_entry_20260902"
START = pd.Timestamp("2024-01-01")
MARGIN, LEV = 0.30, 3.0
NOTIONAL = MARGIN * LEV
SPEC = {"short_term_return_z": {"sl": 3.0, "arm": 1.0, "trail": 0.1, "horizon": 12},
        "demarker_extreme": {"sl": 2.0, "arm": 1.5, "trail": 0.1, "horizon": 8}}
DEPTHS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
WAITS = [3, 6, 12]
COSTS = {"10bp(전taker)": 0.0010, "7bp(maker진입)": 0.0007}


def log(m): print(f"[resting] {m}", flush=True)


def run(trig_bottom, trig_top, atr, o, h, l, c, lo_i, hi_i, *, depth, wait,
        sl, arm, trail, horizon, cost, flip=False):
    """인과 상태기계. [lo_i, hi_i) 구간의 트리거만 주문을 낸다. 반환: trade_return 배열."""
    acct_cost = cost * NOTIONAL
    rets = []
    state = 0            # 0 FLAT, 1 PENDING, 2 OPEN
    p_limit = p_side = p_exp = p_atr = 0.0
    entry = stop = peak = 0.0
    o_side = 0; o_exp = 0; armed = False
    n = len(c)
    for i in range(lo_i, min(hi_i + 400, n - 1)):
        # ---- OPEN 관리 (이 봉의 고저로 판정) ----
        if state == 2:
            done = False
            if o_side > 0:
                if l[i] <= stop:
                    rets.append(float(stop / entry - 1.0) * NOTIONAL - acct_cost); done = True
                else:
                    if h[i] > peak:
                        peak = h[i]
                        if not armed and (peak - entry) / entry >= arm * p_atr:
                            armed = True
                        if armed:
                            stop = max(stop, peak * (1.0 - trail * p_atr))
            else:
                if h[i] >= stop:
                    rets.append(float(1.0 - stop / entry) * NOTIONAL - acct_cost); done = True
                else:
                    if l[i] < peak:
                        peak = l[i]
                        if not armed and (entry - peak) / entry >= arm * p_atr:
                            armed = True
                        if armed:
                            stop = min(stop, peak * (1.0 + trail * p_atr))
            if not done and i >= o_exp:
                mv = (c[i] / entry - 1.0) if o_side > 0 else (1.0 - c[i] / entry)
                rets.append(float(mv) * NOTIONAL - acct_cost); done = True
            if done:
                state = 0
            else:
                continue
        # ---- PENDING 체결/만료 (이 봉의 고저) ----
        if state == 1:
            hit = (l[i] <= p_limit) if p_side > 0 else (h[i] >= p_limit)
            if hit:
                entry = p_limit; o_side = int(p_side); armed = False
                stop = entry * (1.0 - sl * p_atr) if o_side > 0 else entry * (1.0 + sl * p_atr)
                peak = entry; o_exp = i + horizon - 1; state = 2
                continue
            if i >= p_exp:
                state = 0
            else:
                continue
        # ---- FLAT: 트리거면 주문 ----
        if state == 0 and lo_i <= i < hi_i:
            a = atr[i]
            if not (np.isfinite(a) and a > 0):
                continue
            side = 1 if trig_bottom[i] else (-1 if trig_top[i] else 0)
            if side == 0:
                continue
            if flip:
                side = -side
            if depth == 0.0:                      # 대조군: 다음봉 시가 즉시 진입
                entry = o[i + 1]; o_side = side; armed = False
                stop = entry * (1.0 - sl * a) if side > 0 else entry * (1.0 + sl * a)
                peak = entry; p_atr = a; o_exp = i + 1 + horizon - 1; state = 2
            else:
                p_limit = c[i] * (1.0 - depth * a) if side > 0 else c[i] * (1.0 + depth * a)
                p_side = side; p_atr = a; p_exp = i + wait; state = 1
    return np.array(rets, dtype=float)


def stats(v):
    if len(v) == 0:
        return {"n": 0, "mean_bp": 0.0, "pf": 0.0, "total_bp": 0.0}
    w, ls = v[v > 0].sum(), -v[v < 0].sum()
    return {"n": len(v), "mean_bp": round(float(v.mean() * 1e4), 2),
            "pf": round(float(w / ls) if ls > 0 else float("inf"), 3),
            "total_bp": round(float(v.sum() * 1e4), 1)}


def main() -> int:
    from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame

    src = load_klines(); ind = build_indicator_frame(src)
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # 지표 프레임과 klines를 timestamp로 정렬 일치시킨다
    m = kl.merge(ind[["timestamp", "atr_pct", "ret3_z"]], on="timestamp", how="left")
    dem_s = compute_demarker(src["high"], src["low"])
    m = m.merge(pd.DataFrame({"timestamp": src["timestamp"], "dem": dem_s.to_numpy()}),
                on="timestamp", how="left")
    ts = m["timestamp"]
    o, h, l, c = (m[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = m["atr_pct"].to_numpy(float)
    r3 = m["ret3_z"].to_numpy(float); dm = m["dem"].to_numpy(float)
    log(f"정렬 {len(m):,}봉 | atr 결측 {int(np.isnan(atr).sum()):,}")

    TRIG = {"short_term_return_z": (np.nan_to_num(r3, nan=0.0) <= -2.5,
                                    np.nan_to_num(r3, nan=99.0) >= 2.5),
            "demarker_extreme": (np.nan_to_num(dm, nan=0.5) <= 0.10,
                                 np.nan_to_num(dm, nan=0.5) >= 0.90)}
    idx = pd.DatetimeIndex(ts)
    W = {}
    for wn, lo, hi in (("VAL", VAL_START, OOS_START), ("OOS", OOS_START, HOLDOUT_START),
                       ("HOLDOUT", HOLDOUT_START, ts.max())):
        W[wn] = (int(idx.searchsorted(lo)), int(idx.searchsorted(hi)))

    rows = []
    for name, (tb, tt) in TRIG.items():
        sp = SPEC[name]
        for depth in DEPTHS:
            for wait in (WAITS if depth > 0 else [0]):
                for cname, cost in COSTS.items():
                    if depth == 0.0 and cname != "10bp(전taker)":
                        continue          # 대조군은 전 taker만
                    rec = {"signal": name, "depth": depth, "wait": wait, "cost": cname}
                    ok = True
                    for wn, (a, b) in W.items():
                        for flip in (False, True):
                            v = run(tb, tt, atr, o, h, l, c, a, b, depth=depth, wait=wait,
                                    cost=cost, flip=flip, **sp)
                            s = stats(v)
                            k = "" if not flip else "_flip"
                            rec[f"{wn}_mean{k}"] = s["mean_bp"]; rec[f"{wn}_n{k}"] = s["n"]
                            if not flip:
                                rec[f"{wn}_pf"] = s["pf"]; rec[f"{wn}_tot"] = s["total_bp"]
                            else:
                                rec[f"{wn}_tot_flip"] = s["total_bp"]
                        ok &= rec[f"{wn}_tot"] > max(rec[f"{wn}_tot_flip"], 0)
                    rec["flip3창"] = "O" if ok else "X"
                    rec["flip양창"] = "O" if all(rec[f"{w}_tot"] > max(rec[f"{w}_tot_flip"], 0)
                                                for w in ("VAL", "OOS")) else "X"
                    rows.append(rec)
        log(f"{name} 완료")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True); df.to_csv(OUT_DIR / "resting_limit.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["depth", "wait", "cost", "VAL_mean", "VAL_pf", "VAL_n",
            "OOS_mean", "OOS_pf", "OOS_n", "HOLDOUT_mean", "HOLDOUT_n", "flip양창", "flip3창"]
    for name in TRIG:
        log(f"\n=== {name} ===")
        print(df[df.signal == name][cols].to_string(index=False))
    log("\n=== 양창 방향뒤집기 통과 조합 ===")
    p = df[df["flip양창"] == "O"]
    print(p[["signal"] + cols].to_string(index=False) if len(p) else "  없음")
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
