"""Extends research_eth_microstructure_panel_1h4h_direction_screen_20260823.py's 9-feature 1h/4h
direction panel with whale_position_score (the "고래 포지션" dashboard chip, whale_intent key) --
that script's original 9 features did NOT include it. Everything (windows, target definition,
circular-shift permutation test, economic gate, thresholds) is copied VERBATIM from the original
script; only RAW_FEATURES gets one more entry. nif_whale/nif_retail are re-run alongside it in the
SAME pass (not re-imported results) so the 3-way comparison the user asked for -- 수급흐름
(nif_whale) vs 리테일 수급 (nif_retail) vs 고래 포지션 (whale_position_score) -- is apples-to-apples
on identical data/join/split, not stitched from separate runs on different days.

Motivating question (user, 2026-08-25): "리테일 수급이 성능이 좋다고 했는데 수급흐름/고래 포지션과
반대로 보인다 -- 뭐가 더 좋냐". Data pulled fresh from the server via handoff.sh pull immediately
before this run (dev-local copies of this exact duckdb have gone stale and caused wrong conclusions
multiple times this month -- see eth_microstructure_1m_history_archive_and_whale_confirmation_
rejected_20260823's timezone-bug re-run and the volatility screen's 130h-gap correction).
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research_eth_microstructure_panel_1h4h_direction_screen_20260823 import circular_shift_z  # noqa: E402

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MICRO_DB_PATH = ROOT / "data/live/microstructure.duckdb"
Z_WINDOW = 288
COST_BP = 10.0
CONTAM_MAX = 0.5

# The three features the user is comparing: 수급흐름(nif_whale), 리테일수급(nif_retail),
# 고래포지션(whale_position_score). All three are already -1..+1 bounded ratios/scores -- RAW,
# not z-scored, matching how nif_whale/nif_retail were classified in the original panel.
RAW_FEATURES = ["nif_whale", "nif_retail", "whale_position_score"]
HORIZONS = [12, 48]  # 5m bars -> 1h/4h, same as the original panel

SPLITS = {
    "TRAIN": ("2026-05-03", "2026-07-31"),
    "VAL": ("2026-08-01", "2026-08-16"),
}


def build_frame() -> pd.DataFrame:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    klines = klines.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    klines = klines[(klines["timestamp"] >= "2026-04-25") & (klines["timestamp"] <= "2026-08-18")]
    klines["bar_close_time"] = klines["timestamp"] + pd.Timedelta(minutes=5)

    con = duckdb.connect(str(MICRO_DB_PATH), read_only=True)
    micro = con.execute(f"SELECT ts, {', '.join(RAW_FEATURES)} FROM microstructure_1m ORDER BY ts").fetchdf()
    con.close()
    micro["ts"] = pd.to_datetime(micro["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)

    frame = pd.merge_asof(
        klines.sort_values("bar_close_time"), micro.sort_values("ts"),
        left_on="bar_close_time", right_on="ts",
        direction="backward", tolerance=pd.Timedelta("5min"),
    )
    for h in HORIZONS:
        frame[f"fwd_{h}"] = frame["close"].shift(-h) / frame["close"] - 1.0
    return frame


def economic_gate(sub: pd.DataFrame, col: str, h: int, ic_sign: float) -> float:
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
    print(f"whale_position_score first non-null ts: {frame.loc[frame['whale_position_score'].notna(), 'timestamp'].min()}\n")

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

    print(f"{'feature':<24}{'h':>4}{'n_train':>9}{'contam':>8}{'IC_tr':>9}{'z_tr':>7}{'IC_val':>9}{'z_val':>7}  verdict")
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
        print(f"{feat:<24}{h:>4}{n_map.get(feat,0):>9d}{contam:>8.3f}{ic_tr:>9.4f}{z_tr:>7.2f}{ic_val:>9.4f}{z_val:>7.2f}  {verdict}")

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
