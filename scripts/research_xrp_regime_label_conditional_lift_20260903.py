#!/usr/bin/env python3
"""BTC Phase 2 -- which regime LABEL definition carries the most evidence-signal conditioning
value on BTC? Counterpart of research_eth_regime_label_conditional_lift_20260902.py.

BTC Phase 1 (research_btc_regime_scalping_label_geometry_20260902.py) closed the "faster
transitions" framing on BTC even more cleanly than on ETH: 0/16 cells had a transition edge whose
95% CI cleared zero (ETH had 1/16, itself at chance), and RegimeEngine's own label on BTC is flat
too (h6 +0.04bp [-0.42,+0.55]). So the objective here is the same reframe that worked for ETH:
does the regime say WHEN the evidence signals work, rather than which way price goes.

BTC-SPECIFIC PLUMBING (none of this existed; ETH's could not be reused as-is)
  * zigzag pivots: ETH's tmp/zigzag_action_labels_extended_20260809/*.csv are ETH-only. BTC pivots
    are generated here from the CANONICAL BTC OHLC via build_wave3_action_labels_20260531.
    build_zigzag_action_labels() with ETH's parameters VERBATIM (min_reversal_pct=0.009,
    min_wave_bars=6, transition_buffer=1, atr_window=14, atr_multiplier=1.0, mae_penalty=1.1,
    softmax_temperature=1.9, min_risk_floor=0.001) -- identical to what
    build_btc_5m_zigzag_and_pivot_labels_20260806.py already established for BTC 5m, but rebuilt
    on the canonical file rather than reusing that script's causalfix_final-vintage artifact, so
    every number in this study shares one data vintage. Pivot extraction mirrors
    load_zigzag_pivots(): a SHORT run's lowest low is a bottom, a LONG run's highest high is a top.
  * cross-asset leg: smt_divergence needs a reference asset. With BTC as the subject, ETH is passed
    as the cross-asset (the mirror of the deployed ETH-subject/BTC-reference wiring).
  * funding: BTC's own funding (data/research/funding_extracted/BTCUSDT/, 2024-01~2026-06), same
    rolling-90 z recipe as load_funding_z() -- NOT ETH's series.

Controls carried over unchanged: circular-shift null (B=200, preserves the regime's block structure
and duty cycle while destroying price alignment) and a VAL/OOS split, because pooled-only evidence
died on exactly that split in this session's composite-filter study (README ss5.15).

⚠️ The evidence window (2025-09-01~2026-02-17) is INSIDE the BTC regime TRAIN range, so this phase
spends no BTC regime-OOS budget. The regime split boundary is in-sample, "best available".

## ⚠️XRP 포팅 (2026-09-03)

`research_btc_regime_label_conditional_lift_20260902.py`의 자산 상수만 바꾼 포팅.
S x K 격자 재탐색이 목적이다 -- ETH 승자 S12_K3이 BTC에서 3/10 최하위였으므로
자산별 재스크리닝은 선택이 아니라 필수다.

상류 입력은 `scripts/build_xrp_regime_inputs_20260903.py`가 만든다(XRP엔 3개 전부 없었다).
캐노니컬 피쳐 파일은 만들지 않고 klines를 직접 쓴다 -- 피벗 계산에 OHLC만 필요하다.

⭐**교차자산 파트너 슬롯 주의**: BTC-주체 스크립트는 `btc_df=eth`로 ETH를 넣었다.
XRP-주체이므로 BTC를 넣는다. 인자 이름이 `btc_df`인 건 시그니처일 뿐이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_wave3_action_labels_20260531 as zigzag  # noqa: E402
from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    OOS_END as EV_OOS_END, event_study,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START as EV_OOS_START, VAL_END as EV_VAL_END, VAL_START as EV_VAL_START,
)
from features.elite import RegimeEngine  # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_regime_label_conditional_lift_20260902 import (  # noqa: E402
    K_HORIZON, MIN_SEG_FIRES, N_NULL, seg_lift,
)
from research_eth_regime_scalping_label_geometry_20260902 import (  # noqa: E402
    DEBOUNCES, _debounce, efficiency_ratio, scaled_label,
)

XRP_KLINES = ROOT / "data/xrp_5m_1year.csv"
# XRP-주체의 교차자산 파트너. BTC-주체 원본은 여기에 ETH를 넣었다.
PARTNER_KLINES = ROOT / "data/btc_5m_1year.csv"
ETH_KLINES = ROOT / "data/eth_5m_1year.csv"
XRP_CANON = ROOT / "data/xrp_5m_1year.csv"   # ⭐피벗 계산에 OHLC만 쓰므로 klines 직접 사용
XRP_FUNDING_DIR = ROOT / "data/research/funding_extracted/XRPUSDT"
PIVOT_CACHE = ROOT / "tmp/xrp_regime_label_conditional_lift_20260903/xrp_pivots.parquet"
SCALES = (6, 12, 24, 48)
ZIG = dict(min_reversal_pct=0.009, min_wave_bars=6, transition_buffer=1, atr_window=14,
           atr_multiplier=1.0, mae_penalty=1.1, softmax_temperature=1.9, min_risk_floor=0.001)
OUT_DIR = ROOT / "tmp/xrp_regime_label_conditional_lift_20260903"


def load_xrp_funding_z() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(XRP_FUNDING_DIR.glob("XRPUSDT-fundingRate-*.csv"))]
    f = pd.concat(frames, ignore_index=True)
    # epoch-ms -> datetime64[us] to match the klines timestamp dtype that merge_asof
    # joins against (pandas 3.x keeps unit="ms" as datetime64[ms] and refuses the merge)
    f["calc_time"] = pd.to_datetime(f["calc_time"], unit="ms").astype("datetime64[us]")
    f = f.sort_values("calc_time").drop_duplicates("calc_time").reset_index(drop=True)
    mean = f["last_funding_rate"].rolling(90, min_periods=30).mean()
    std = f["last_funding_rate"].rolling(90, min_periods=30).std()
    f["funding_z"] = (f["last_funding_rate"] - mean) / std.replace(0.0, np.nan)
    out = f[["calc_time", "funding_z"]]
    # ⚠️펀딩 CSV는 [us]인데 klines 파싱은 [ns]다. merge_asof가 dtype 일치를 요구하므로
    # **로더 안에서** 맞춘다 -- Phase 3도 이 함수를 import하므로 여기서 고쳐야 둘 다 해결된다.
    for _c in out.columns if hasattr(out, "columns") else []:
        if "time" in str(_c).lower():
            out[_c] = out[_c].astype("datetime64[ns]")
    return out

def build_xrp_pivots() -> pd.DataFrame:
    """BTC swing pivots from the CANONICAL OHLC, ETH's zigzag parameters verbatim."""
    if PIVOT_CACHE.exists():
        return pd.read_parquet(PIVOT_CACHE)
    frame = pd.read_csv(XRP_CANON, usecols=["timestamp", "open", "high", "low", "close"],
                        parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    lab = zigzag.build_zigzag_action_labels(frame, **ZIG)
    z = frame[["timestamp", "low", "high"]].copy()
    z["zigzag_action"] = lab["zigzag_action"].to_numpy()
    run_id = (z["zigzag_action"] != z["zigzag_action"].shift()).cumsum()
    rows = []
    for _, run in z.groupby(run_id):
        a = int(run["zigzag_action"].iloc[0])
        if a == 2:
            r = run.loc[run["low"].idxmin()]
            rows.append({"timestamp": r["timestamp"], "pivot_type": "bottom"})
        elif a == 1:
            r = run.loc[run["high"].idxmax()]
            rows.append({"timestamp": r["timestamp"], "pivot_type": "top"})
    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    PIVOT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PIVOT_CACHE, index=False)
    return out


def main() -> None:
    raw = pd.read_csv(XRP_KLINES, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    # ⚠️교차자산 파트너 슬롯. BTC-주체 스크립트는 여기에 ETH를 넣었다(`btc_df=partner`).
    # XRP-주체이므로 BTC를 넣는다 -- 인자 이름이 `btc_df`인 건 함수 시그니처일 뿐이다
    # (FeatureEngineer의 close_btc 슬롯 명명 함정과 같은 계열, 메모리에 기록된 사항).
    partner = pd.read_csv(PARTNER_KLINES, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    frame = compute_signals(raw, btc_df=partner, funding_df=load_xrp_funding_z())
    pivots = build_xrp_pivots()
    print(f"XRP evidence frame {len(frame):,} bars | pivots {len(pivots):,} "
          f"(bottom {int((pivots.pivot_type=='bottom').sum())}, top {int((pivots.pivot_type=='top').sum())})")

    ts, close = frame["timestamp"], frame["close"]
    windows = {"VAL": ((ts >= EV_VAL_START) & (ts <= EV_VAL_END)).to_numpy(),
               "OOS": ((ts >= EV_OOS_START) & (ts <= EV_OOS_END)).to_numpy()}
    windows["POOLED"] = windows["VAL"] | windows["OOS"]
    pivot_pos = {s: frame.index[frame["timestamp"].isin(
        pivots.loc[pivots["pivot_type"] == s, "timestamp"])].to_numpy() for s in ("bottom", "top")}
    print(f"  in-frame pivots: bottom {len(pivot_pos['bottom'])}, top {len(pivot_pos['top'])}")

    ref = frame.copy()
    ref["mtf_trend_1h"] = close.ewm(span=12, adjust=False).mean().pct_change().fillna(0.0)
    lab = RegimeEngine().compute(ref)
    y_ref = np.full(len(frame), 2, dtype=int)
    y_ref[lab["regime_bull"].to_numpy() > 0] = 0
    y_ref[lab["regime_bear"].to_numpy() > 0] = 1
    variants = {"REF_RegimeEngine": y_ref}

    rate1 = float((efficiency_ratio(close, 24) >= 0.20).mean())
    rate2 = float((efficiency_ratio(close, 48) >= 0.16).mean())
    for s in SCALES:
        t1 = float(efficiency_ratio(close, s).quantile(1.0 - rate1))
        t2 = float(efficiency_ratio(close, 2 * s).quantile(1.0 - rate2))
        y0 = scaled_label(close, s, t1, t2)
        for k in DEBOUNCES:
            if k == 12:
                continue                      # Phase 1: lock-up / instability at high K
            variants[f"S{s}_K{k}"] = y0 if k == 1 else _debounce(y0, k)
    print(f"{len(variants)} label variants")

    rng = np.random.default_rng(20260902)
    rows = []
    for vname, y in variants.items():
        chop_all = (y == 2)
        for wname, wmask in windows.items():
            seg = chop_all & wmask
            for sname, _ in SIGNAL_ORDER:
                for side in ("bottom", "top"):
                    sig = frame[f"{side}_{sname}"].fillna(False).to_numpy()
                    l_all, n_all = seg_lift(sig, pivot_pos[side], wmask)
                    l_chop, n_chop = seg_lift(sig, pivot_pos[side], seg)
                    if not (np.isfinite(l_all) and np.isfinite(l_chop)) or l_all <= 0:
                        continue
                    imp = l_chop / l_all - 1.0
                    null = []
                    for _ in range(N_NULL):
                        lb, _n = seg_lift(sig, pivot_pos[side],
                                          np.roll(chop_all, int(rng.integers(1, len(y)))) & wmask)
                        if np.isfinite(lb):
                            null.append(lb / l_all - 1.0)
                    p95 = float(np.percentile(null, 95)) if len(null) >= 50 else float("nan")
                    rows.append({"variant": vname, "window": wname, "signal": sname, "side": side,
                                 "n_all": n_all, "n_chop": n_chop, "lift_all": round(l_all, 3),
                                 "lift_chop": round(l_chop, 3), "improvement": round(imp, 4),
                                 "beats_null95": bool(np.isfinite(p95) and imp > p95)})
        print(f"  {vname}: done")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "conditional_lift.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    summ = (df.groupby(["variant", "window"])
              .agg(cells=("improvement", "size"), mean=("improvement", "mean"),
                   beats_null=("beats_null95", "sum")).reset_index())
    print("\n=== BTC per-variant mean chop-conditional lift improvement ===")
    print(summ.pivot(index="variant", columns="window", values=["mean", "beats_null", "cells"])
          .round(4).to_string())
    print("\n=== both-window-positive cells per variant (the ss5.15 gate) ===")
    for v in variants:
        sub = df[df["variant"] == v]
        p = sub.pivot_table(index=["signal", "side"], columns="window", values="improvement")
        if "VAL" in p and "OOS" in p:
            both = int(((p["VAL"] > 0) & (p["OOS"] > 0)).sum())
            print(f"  {v:18s} {both}/{len(p)}  | mean VAL {p['VAL'].mean():+.4f} OOS {p['OOS'].mean():+.4f}")
    print(f"\nWrote {OUT_DIR / 'conditional_lift.csv'}")


if __name__ == "__main__":
    main()
