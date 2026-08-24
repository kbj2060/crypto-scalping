"""ETH -- evidence-signal x whale-position-confirmation combination cheap-gate.

Pre-registered design (locked BEFORE this script touched any combined data):
docs/experiments/eth_candidate_evidence_signal_whale_confirmation_combination_20260823.md

Hypothesis: an evidence-signal reversal call (bottom_votes>=1 / top_votes>=1, the same
definition already used for firing-rate measurement this session) that co-occurs with a
confirming whale_position_score (>0.2 for bottom, <-0.2 for top -- same threshold the live
dashboard already uses, not picked from this data) has higher forward return than an
unconfirmed call.

TRAIN+VAL only. OOS (2026-08-17 onward) is deliberately not touched here.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

KLINES_PATH = "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MICRO_DB_PATH = "data/live/microstructure.duckdb"  # run on the SERVER copy (this is stale on dev)
WHALE_THRESHOLD = 0.2
FWD_HORIZON = 3  # bars = 15 min
N_PERM = 2000
COST_BP = 10.0
MIN_VAL_N = 30

SPLITS = {
    "TRAIN": ("2026-05-03", "2026-07-31"),
    "VAL": ("2026-08-01", "2026-08-16"),
}


def load_evidence_calls() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    sig = compute_signals(df)
    sig["fwd_ret"] = sig["close"].shift(-FWD_HORIZON) / sig["close"] - 1.0
    is_bottom = sig["bottom_votes"] >= 1
    is_top = sig["top_votes"] >= 1
    calls = sig[is_bottom | is_top].copy()
    calls["side"] = np.where(is_bottom[calls.index], "bottom", "top")
    # a bar can technically fire both sides at once (rare, conflicting evidence) -- exclude those,
    # since "signed" direction is undefined for them and they'd contaminate both sides' pools.
    calls = calls[~(is_bottom[calls.index] & is_top[calls.index])].copy()
    calls["signed_fwd_ret"] = np.where(calls["side"] == "bottom", calls["fwd_ret"], -calls["fwd_ret"])
    return calls[["timestamp", "side", "signed_fwd_ret"]].dropna()


def load_whale_position(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        "SELECT ts, whale_position_score FROM microstructure_1m ORDER BY ts"
    ).fetchdf()
    # ts is KST tz-aware; the klines CSV is UTC-NAIVE (verified 2026-08-23 against Binance API:
    # CSV row 2026-08-18 17:35 matches the API's UTC bar exactly, and mismatches the KST reading).
    # The original run of this script did tz_localize(None) directly -- that keeps the KST WALL
    # CLOCK, silently joining every call to the whale score from 9 HOURS EARLIER. Convert to UTC
    # first. The first run's REJECTED verdict was therefore testing a broken join, not the
    # hypothesis; this re-run (same pre-registered design, bug-fixed join) supersedes it.
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def join_nearest(calls: pd.DataFrame, whale: pd.DataFrame) -> pd.DataFrame:
    calls = calls.sort_values("timestamp")
    whale = whale.sort_values("ts")
    merged = pd.merge_asof(
        calls, whale, left_on="timestamp", right_on="ts",
        direction="nearest", tolerance=pd.Timedelta("5min"),
    )
    return merged.dropna(subset=["whale_position_score"])


def permutation_z(confirmed_mask: np.ndarray, values: np.ndarray, n_perm: int = N_PERM, seed: int = 20260823) -> float:
    rng = np.random.default_rng(seed)
    observed = values[confirmed_mask].mean() - values[~confirmed_mask].mean()
    k = confirmed_mask.sum()
    n = len(values)
    null_gaps = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        idx = perm[:k]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        null_gaps[i] = values[mask].mean() - values[~mask].mean()
    mu, sd = null_gaps.mean(), null_gaps.std(ddof=1)
    return (observed - mu) / sd if sd > 0 else float("nan")


def evaluate_split(df: pd.DataFrame, label: str) -> dict:
    confirmed = (
        ((df["side"] == "bottom") & (df["whale_position_score"] > WHALE_THRESHOLD))
        | ((df["side"] == "top") & (df["whale_position_score"] < -WHALE_THRESHOLD))
    )
    n_total, n_confirmed = len(df), int(confirmed.sum())
    conf_mean = df.loc[confirmed, "signed_fwd_ret"].mean() if n_confirmed else float("nan")
    unconf_mean = df.loc[~confirmed, "signed_fwd_ret"].mean() if (n_total - n_confirmed) else float("nan")
    z = permutation_z(confirmed.to_numpy(), df["signed_fwd_ret"].to_numpy()) if n_confirmed else float("nan")
    conf_mean_bp = conf_mean * 1e4 if pd.notna(conf_mean) else float("nan")
    return {
        "split": label,
        "n_total_calls": n_total,
        "n_confirmed": n_confirmed,
        "confirmed_mean_bp": conf_mean_bp,
        "unconfirmed_mean_bp": unconf_mean * 1e4 if pd.notna(unconf_mean) else float("nan"),
        "gap_bp": (conf_mean_bp - (unconf_mean * 1e4 if pd.notna(unconf_mean) else 0)) if pd.notna(conf_mean_bp) else float("nan"),
        "perm_z": z,
        "clears_cost_10bp": bool(pd.notna(conf_mean_bp) and conf_mean_bp > COST_BP),
        "min_n_met": n_confirmed >= MIN_VAL_N,
    }


def main() -> None:
    print(f"Loading evidence-signal calls from {KLINES_PATH} ...")
    calls = load_evidence_calls()
    print(f"  {len(calls)} calls total (bottom_votes>=1 or top_votes>=1, non-conflicting)")

    print(f"Loading whale_position_score history from {MICRO_DB_PATH}::microstructure_1m ...")
    con = duckdb.connect(MICRO_DB_PATH, read_only=True)
    whale = load_whale_position(con)
    con.close()
    print(f"  {len(whale)} minute rows, {whale['ts'].min()} ~ {whale['ts'].max()}")

    merged = join_nearest(calls, whale)
    print(f"  {len(merged)}/{len(calls)} calls matched to a whale_position_score within 5min\n")

    results = []
    for label, (start, end) in SPLITS.items():
        sub = merged[(merged["timestamp"] >= start) & (merged["timestamp"] <= end)]
        r = evaluate_split(sub, label)
        results.append(r)
        print(f"=== {label} ({start}..{end}) ===")
        for k, v in r.items():
            if k != "split":
                print(f"  {k}: {v}")
        print()

    val = results[-1]
    verdict_lines = []
    train, val_r = results[0], results[-1]
    c1 = train["gap_bp"] > 0 and val_r["gap_bp"] > 0
    c2 = pd.notna(val_r["perm_z"]) and abs(val_r["perm_z"]) >= 2
    c3 = pd.notna(val_r["confirmed_mean_bp"]) and val_r["confirmed_mean_bp"] > 0
    c4 = val_r["clears_cost_10bp"]
    c5 = val_r["min_n_met"]
    verdict_lines.append(f"1. sign consistency (TRAIN & VAL gap>0): {c1}")
    verdict_lines.append(f"2. permutation |z|>=2 (VAL): {c2} (z={val_r['perm_z']:.3f})" if pd.notna(val_r["perm_z"]) else "2. permutation |z|>=2 (VAL): N/A (no confirmed events)")
    verdict_lines.append(f"3. confirmed mean > 0 (VAL): {c3}")
    verdict_lines.append(f"4. clears 10bp cost (VAL): {c4}")
    verdict_lines.append(f"5. min N=30 met (VAL, n_confirmed={val_r['n_confirmed']}): {c5}")
    passed = all([c1, c2, c3, c4, c5])
    print("=== PRE-REGISTERED VERDICT ===")
    print("\n".join(verdict_lines))
    print(f"\nOVERALL: {'CONFIRMED (all 5 criteria pass)' if passed else 'REJECTED (at least one criterion fails)'}")


if __name__ == "__main__":
    main()
