"""Direction-IC screen for RAW OI/position-skew columns (not a derived blend like
whale_position_score, not filtered through the liquidation map) -- user asked (2026-08-26)
"OI/포지션쏠림 데이터로 어떤 전략을 세울 수 있을지 연구해줘" after the §13 exploratory peek.

Gap this fills: whale_position_score (0.7*nif_whale + 0.3*OI-direction*OI-intensity) already
tested 0/4 horizons, worst cell sign-flipped OOS (eth_whale_position_vs_retail_flow_direction_ic_
20260825) -- but that is a BLENDED derivative, so the OI/position-ratio components were never
isolated on their own. Liquidation-map direction-isolated A/B (eth_liquidation_map_direction_
isolated_ab_rejected_20260826) tested top-trader/account long-short ratio too, but as an input to
the liquidation MAP's entry-mass split, not as a direct predictor of price returns. The existing
OI-delta dashboard chip (live_oi_delta_signal_20260824.py) has been code-audited but never run
through a direction-IC gate. This script tests the raw columns directly against forward returns.

Methodology copied VERBATIM from research_eth_whale_position_score_direction_screen_20260825.py /
research_eth_microstructure_panel_1h4h_direction_screen_20260823.py (circular_shift_z permutation,
same TRAIN/VAL split, same horizons, same contamination gate, same economic gate) so results are
directly comparable to the existing nif_whale/nif_retail/whale_position_score table -- no new
technique invented.

Candidates (all from data/TOTAL_ETHUSDT_metrics_2024_2026.csv, the verified-clean archive per
reference_clean_data_locations_20260823):
  - oi_z: rolling-288 z-score of sum_open_interest_value (level, detrended -- "elevated leverage")
  - oi_delta_z: rolling-288 z-score of sum_open_interest.diff() -- EXACT formula of the deployed
    live_oi_delta_signal_20260824.py chip, contract-count based (not notional), tested here for
    direction alpha for the first time (that chip was only ever code/logic-audited, not IC-gated).
  - top_pos_z: rolling-288 z-score of sum_toptrader_long_short_ratio -- "smart money" positioning
    deviation from its own local mean (classic crowded-positioning-contrarian framing).
  - retail_pos_z: rolling-288 z-score of count_long_short_ratio (global account-count ratio, i.e.
    retail-weighted since accounts != notional) -- same contrarian framing, retail cohort.

Scope deliberately excludes sum_taker_long_short_vol_ratio (a FLOW metric, not OI/position-skew --
that ground is nif_whale/nif_retail's territory, already screened same methodology 2026-08-25).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research_eth_microstructure_panel_1h4h_direction_screen_20260823 import circular_shift_z  # noqa: E402

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OI_ARCHIVE_CSV = ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"
Z_WINDOW = 288  # 1 day of 5m bars, this repo's standard window
COST_BP = 10.0
CONTAM_MAX = 0.5

RAW_FEATURES = ["oi_z", "oi_delta_z", "top_pos_z", "retail_pos_z"]
HORIZONS = [12, 48]  # 5m bars -> 1h/4h, matches the whale/retail panel exactly

SPLITS = {
    "TRAIN": ("2026-05-03", "2026-07-31"),
    "VAL": ("2026-08-01", "2026-08-16"),
}


def build_frame() -> pd.DataFrame:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    klines = klines.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    klines = klines[(klines["timestamp"] >= "2026-04-25") & (klines["timestamp"] <= "2026-08-18")]
    klines["bar_close_time"] = klines["timestamp"] + pd.Timedelta(minutes=5)

    oi = pd.read_csv(OI_ARCHIVE_CSV, usecols=["create_time", "sum_open_interest", "sum_open_interest_value",
                                               "sum_toptrader_long_short_ratio", "count_long_short_ratio"])
    oi["ts"] = pd.to_datetime(oi["create_time"])
    oi = oi.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    def roll_z(s: pd.Series) -> pd.Series:
        return (s - s.rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()) / \
            s.rolling(Z_WINDOW, min_periods=Z_WINDOW).std().replace(0.0, np.nan)

    oi["oi_z"] = roll_z(oi["sum_open_interest_value"])
    oi["oi_delta_z"] = roll_z(oi["sum_open_interest"].diff())
    oi["top_pos_z"] = roll_z(oi["sum_toptrader_long_short_ratio"])
    oi["retail_pos_z"] = roll_z(oi["count_long_short_ratio"])

    frame = pd.merge_asof(
        klines.sort_values("bar_close_time"), oi[["ts"] + RAW_FEATURES].sort_values("ts"),
        left_on="bar_close_time", right_on="ts",
        direction="backward", tolerance=pd.Timedelta("5min"),
    )
    for h in HORIZONS:
        frame[f"fwd_{h}"] = frame["close"].shift(-h) / frame["close"] - 1.0
    return frame


def economic_gate(sub: pd.DataFrame, col: str, h: int, ic_sign: float) -> float:
    """TRAIN-IC-sign-fixed threshold rule, verbatim from the whale/retail panel: trailing-288 z of
    the candidate (already a z-score here, so this re-z-scores the z -- kept identical to the
    original script rather than special-cased, for exact comparability)."""
    z = (sub[col] - sub[col].rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()) / \
        sub[col].rolling(Z_WINDOW, min_periods=Z_WINDOW).std().replace(0.0, np.nan)
    fwd = sub[f"fwd_{h}"]
    pnl, i, idx = 0.0, 0, sub.index.to_list()
    while i < len(idx):
        zi, fi = z.get(idx[i]), fwd.get(idx[i])
        if pd.notna(zi) and abs(zi) >= 1.0 and pd.notna(fi):
            pnl += ic_sign * np.sign(zi) * fi - COST_BP / 1e4
            i += h
        else:
            i += 1
    always_long = fwd.iloc[::h].dropna().sum()
    bench = max(always_long, -always_long)
    return pnl - bench


def main() -> None:
    frame = build_frame()
    print(f"joined frame: {len(frame)} bars, coverage: "
          + ", ".join(f"{f}={frame[f].notna().mean()*100:.1f}%" for f in RAW_FEATURES))

    rows = []
    for split, (start, end) in SPLITS.items():
        sub = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
        for feat in RAW_FEATURES:
            valid_n = sub[feat].notna().sum()
            contam = spearmanr(*sub[[feat, "close"]].dropna().to_numpy().T).statistic if valid_n > 500 else float("nan")
            for h in HORIZONS:
                ic, z = circular_shift_z(sub[feat].to_numpy(dtype=float), sub[f"fwd_{h}"].to_numpy(dtype=float))
                rows.append({"split": split, "feature": feat, "h": h, "ic": ic, "z": z, "contam": contam, "n": valid_n})

    res = pd.DataFrame(rows)
    piv = res.pivot_table(index=["feature", "h"], columns="split", values=["ic", "z"], aggfunc="first")
    contam_map = res[res["split"] == "TRAIN"].set_index("feature")["contam"].to_dict()
    n_map = res[res["split"] == "TRAIN"].set_index("feature")["n"].to_dict()

    print(f"{'feature':<16}{'h':>4}{'n_train':>9}{'contam':>8}{'IC_tr':>9}{'z_tr':>7}{'IC_val':>9}{'z_val':>7}  verdict")
    survivors = []
    for (feat, h), r in piv.iterrows():
        contam = contam_map.get(feat, float("nan"))
        ic_tr, z_tr = r[("ic", "TRAIN")], r[("z", "TRAIN")]
        ic_val, z_val = r[("ic", "VAL")], r[("z", "VAL")]
        if pd.notna(contam) and abs(contam) >= CONTAM_MAX:
            verdict = "EXCLUDED(contam)"
        elif pd.isna(ic_tr) or pd.isna(ic_val):
            verdict = "insufficient"
        elif np.sign(ic_tr) != np.sign(ic_val):
            verdict = "fail:sign"
        elif abs(z_tr) < 2 or abs(z_val) < 2:
            verdict = "fail:z"
        else:
            verdict = "PASS_STATS"
            survivors.append((feat, h, np.sign(ic_tr)))
        print(f"{feat:<16}{h:>4}{n_map.get(feat,0):>9d}{contam:>8.3f}{ic_tr:>9.4f}{z_tr:>7.2f}{ic_val:>9.4f}{z_val:>7.2f}  {verdict}")

    print(f"\nstatistical survivors: {len(survivors)}/{len(RAW_FEATURES) * len(HORIZONS)} cells")
    for feat, h, sign in survivors:
        incs = {}
        for split, (start, end) in SPLITS.items():
            sub = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
            incs[split] = economic_gate(sub, feat, h, sign)
        both_pos = all(v > 0 for v in incs.values())
        print(f"  ECONOMIC {feat} h={h}: TRAIN inc={incs['TRAIN']*100:+.2f}% VAL inc={incs['VAL']*100:+.2f}% -> {'PASS' if both_pos else 'FAIL'}")


if __name__ == "__main__":
    main()
