"""One-off feature-health audit comparing SOL/BTC's computed engineered features against ETH's,
to flag any columns that look structurally broken (all-NaN, zero-variance, infinite) for SOL/BTC
specifically while ETH's equivalent is fine -- the same signature the funding-divisor bug had.
"""
import numpy as np
import pandas as pd

ASSETS = {
    "ETH": "data/splits/year_oos/training_features_2026_rebuilt.csv",
    "SOL": "data/splits/year_oos/sol_features_2026.csv",
    "BTC": "data/splits/year_oos/btc_features_2026.csv",
}

dfs = {}
for name, path in ASSETS.items():
    df = pd.read_csv(path, low_memory=False)
    dfs[name] = df
    print(f"{name}: {len(df)} rows, {len(df.columns)} cols, range {df['timestamp'].iloc[0]}..{df['timestamp'].iloc[-1]}")

common_cols = set(dfs["ETH"].columns)
for name in ("SOL", "BTC"):
    common_cols &= set(dfs[name].columns)
common_cols = sorted(c for c in common_cols if c != "timestamp")
print(f"\ncommon columns across all 3: {len(common_cols)}")

report = {}
for c in common_cols:
    row = {}
    for name in ASSETS:
        s = pd.to_numeric(dfs[name][c], errors="coerce")
        row[name] = {
            "nan_pct": float(s.isna().mean() * 100),
            "inf_pct": float(np.isinf(s.to_numpy(dtype=np.float64)).mean() * 100),
            "zero_pct": float((s == 0).mean() * 100),
            "std": float(s.std()) if s.notna().any() else None,
            "mean": float(s.mean()) if s.notna().any() else None,
        }
    report[c] = row

print("\n=== Columns where SOL or BTC look broken (all-NaN / zero-variance / inf) but ETH is fine ===")
flagged = []
for c, row in report.items():
    eth = row["ETH"]
    eth_ok = eth["nan_pct"] < 50 and (eth["std"] or 0) > 1e-12
    for name in ("SOL", "BTC"):
        r = row[name]
        broken = r["nan_pct"] >= 50 or (r["std"] is not None and r["std"] < 1e-12) or r["inf_pct"] > 0
        if broken and eth_ok:
            flagged.append((c, name, r))
for c, name, r in flagged:
    print(f"{c:35s} [{name}] nan%={r['nan_pct']:.1f} inf%={r['inf_pct']:.1f} zero%={r['zero_pct']:.1f} std={r['std']}")
print(f"\ntotal flagged: {len(flagged)}")

print("\n=== Columns where SOL/BTC std is >5x or <0.2x ETH's std (possible scale/calibration issue) ===")
scale_flagged = []
for c, row in report.items():
    eth_std = row["ETH"]["std"]
    if not eth_std or abs(eth_std) < 1e-9:
        continue
    for name in ("SOL", "BTC"):
        s = row[name]["std"]
        if s is None:
            continue
        ratio = s / eth_std
        if ratio > 5 or ratio < 0.2:
            scale_flagged.append((c, name, ratio, eth_std, s))
scale_flagged.sort(key=lambda x: -max(x[2], 1 / x[2]) if x[2] > 0 else 0)
for c, name, ratio, eth_std, s in scale_flagged[:40]:
    print(f"{c:35s} [{name}] ratio={ratio:7.2f}x  ETH_std={eth_std:.6g}  {name}_std={s:.6g}")
print(f"\ntotal scale-flagged: {len(scale_flagged)}")
