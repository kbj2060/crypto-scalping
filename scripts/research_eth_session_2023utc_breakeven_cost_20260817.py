#!/usr/bin/env python3
"""ETH 5m: 20-23 UTC 평균회귀 가설의 breakeven cost (bp) 환산.

배경: docs/experiments/eth_session_split_feature_price_correlation_20260817.md 결과 3 에서
평일 20:00-23:59 UTC 구간의 rsi / mtf_trend_4h / cvp_volume_imbalance 가 TRAIN/VAL/OOS 모두
음의 IC 를 보였다. IC 는 거래 가능한 엣지가 아니므로, 실제 진입 규칙으로 바꿔 per-trade
gross return 을 bp 로 환산하고 breakeven round-trip cost 를 구한다.

기준선: 이 레포의 검증된 taker 비용 one-way 5.0 bp (WS-A 캘리브레이션,
research_ws_b_fill_probability_20260719.py:228) -> round-trip 10.0 bp.
breakeven < 10bp 이면 이 신호는 비용을 못 넘는다.

규칙
  - 유니버스: 평일 20:00-23:55 UTC 바
  - 임계값은 TRAIN 구간에서만 적합한 empirical CDF (lookahead 차단). VAL/OOS 는 그대로 적용
  - IC 가 음수이므로 percentile > 0.8 -> SHORT, < 0.2 -> LONG
  - 진입 지연 1 bar (신호는 t 에서, 진입은 t+1 종가). lag=0 대조도 같이 낸다
  - 보유 h bar 후 종가 청산
  - overlapping(모든 바 진입) 과 non-overlapping(직전 거래 종료 후에만 진입) 둘 다 계산

breakeven_roundtrip_bp = per-trade gross return (bp). 정확히 이 값에서 net = 0 이 된다.

읽기 전용 연구 스크립트. 승격 근거 아님 (CLAUDE.md Fresh-Forward 규칙상 bar-by-bar causal
walk-forward 가 아니라 스크리닝 계산이다).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_eth_session_split_feature_price_correlation_20260817 as A  # noqa: E402

FEATURES = ["rsi", "mtf_trend_4h", "cvp_volume_imbalance"]
HOLDS = [3, 6, 12]           # 15m / 30m / 1h
LO, HI = 0.2, 0.8
ONE_WAY_BP = 5.0             # WS-A 검증 상수
ROUNDTRIP_BP = 2 * ONE_WAY_BP


def bucket_mask(df: pd.DataFrame) -> np.ndarray:
    ts = df["timestamp"]
    return ((ts.dt.dayofweek < 5) & (ts.dt.hour >= 20)).to_numpy()


def train_ecdf(train_vals: np.ndarray):
    """TRAIN 값으로만 만든 empirical CDF. 이후 split 에 그대로 적용해 lookahead 를 막는다."""
    ref = np.sort(train_vals[np.isfinite(train_vals)])

    def apply(x: np.ndarray) -> np.ndarray:
        return np.searchsorted(ref, x, side="right") / len(ref)

    return apply


def trade_stats(sig: np.ndarray, close: np.ndarray, idx: np.ndarray,
                lag: int, h: int, non_overlap: bool) -> dict:
    """sig: +1 long / -1 short / 0 no-trade (전체 프레임 길이).
    idx: 유니버스 바의 전체-프레임 인덱스."""
    entries, rets = [], []
    next_free = -1
    n = len(close)
    for i in idx:
        if sig[i] == 0:
            continue
        e = i + lag
        x = e + h
        if x >= n:
            continue
        if non_overlap and e < next_free:
            continue
        r = np.log(close[x] / close[e]) * sig[i]
        entries.append(e)
        rets.append(r)
        if non_overlap:
            next_free = x
    if not rets:
        return {"n_trades": 0}
    r = np.asarray(rets)
    gross_bp = r.mean() * 1e4
    se_bp = r.std(ddof=1) / np.sqrt(len(r)) * 1e4
    return {
        "n_trades": len(r),
        "gross_bp": gross_bp,
        "se_bp": se_bp,
        "t_stat": gross_bp / se_bp if se_bp > 0 else np.nan,
        "breakeven_bp": gross_bp,
        "net_at_10bp": gross_bp - ROUNDTRIP_BP,
        "win_rate": float((r > 0).mean()),
    }


def main() -> None:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp/session_split_20260817"
    frames = A.load_frames(cache)
    splits = {k: frames[v].sort_values("timestamp").reset_index(drop=True)
              for k, v in [("TRAIN", "train_raw"), ("VAL", "val_raw"), ("OOS", "oos_raw")]}

    # TRAIN 유니버스로만 ECDF 적합
    tr = splits["TRAIN"]
    tr_mask = bucket_mask(tr)
    ecdfs = {f: train_ecdf(tr[f].to_numpy(float)[tr_mask]) for f in FEATURES}

    rows = []
    for split, df in splits.items():
        m = bucket_mask(df)
        idx = np.where(m)[0]
        close = df["close"].to_numpy(float)
        pcts = {f: ecdfs[f](df[f].to_numpy(float)) for f in FEATURES}
        composite = np.mean([pcts[f] for f in FEATURES], axis=0)

        signals = {f: pcts[f] for f in FEATURES}
        signals["composite"] = composite
        for name, pct in signals.items():
            sig = np.zeros(len(df))
            sig[pct > HI] = -1.0   # IC 음수 -> 상위 퍼센타일은 SHORT
            sig[pct < LO] = +1.0
            sig[~m] = 0.0
            for h in HOLDS:
                for lag in [0, 1]:
                    for no in [False, True]:
                        st = trade_stats(sig, close, idx, lag, h, no)
                        if st["n_trades"] == 0:
                            continue
                        rows.append({"split": split, "signal": name, "hold_bars": h,
                                     "lag": lag, "non_overlap": no, **st})

        # 대조군: 같은 유니버스에서 무조건 롱 (드리프트 벤치마크)
        always = np.zeros(len(df))
        always[m] = 1.0
        for h in HOLDS:
            st = trade_stats(always, close, idx, 1, h, True)
            if st["n_trades"]:
                rows.append({"split": split, "signal": "always_long(대조)", "hold_bars": h,
                             "lag": 1, "non_overlap": True, **st})

    res = pd.DataFrame(rows)
    out = ROOT / "tmp/session_split_20260817/breakeven_2023utc.csv"
    res.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    print(f"기준선: one-way {ONE_WAY_BP}bp -> round-trip {ROUNDTRIP_BP}bp\n")
    for no in [True, False]:
        print(f"\n{'='*100}\n### non_overlap={no}, lag=1 (현실적 진입)\n{'='*100}")
        v = res[(res.non_overlap == no) & (res.lag == 1)]
        piv = v.pivot_table(index=["signal", "hold_bars"], columns="split",
                            values=["breakeven_bp", "t_stat", "n_trades"])
        piv = piv.reindex(columns=["TRAIN", "VAL", "OOS"], level=1)
        print(piv.round(2).to_string())

    print(f"\n{'='*100}\n### lag 민감도 (non_overlap=True, composite)\n{'='*100}")
    v = res[(res.non_overlap) & (res.signal == "composite")]
    print(v.pivot_table(index=["hold_bars", "lag"], columns="split",
                        values="breakeven_bp").reindex(columns=["TRAIN", "VAL", "OOS"]).round(2).to_string())
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
