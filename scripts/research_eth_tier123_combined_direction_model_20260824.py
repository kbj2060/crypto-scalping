"""ETH 1·2·3등급 특수데이터 결합 방향 모델 (ridge, 사전등록 실행).

Pre-registered design (locked before touching joined data):
docs/experiments/eth_candidate_tier123_combined_direction_model_20260824.md

Arms: (A) tier1 9-feature reproduction baseline, (B) tier1+2 14, (C) tier1+2+3 16.
Ridge(alpha=1.0) fixed, h={12,48}, primary = arm C h=48. TRAIN/VAL only (dev score);
frozen window 2026-08-17~09-30 untouched, single-touch after 09-30.

Rolling z everywhere: window=288, min_periods=274 (95% tolerance) -- institutionalized
fix for the min_periods=window design flaw found 2026-08-24 in the scalp-horizon screen.
Freshness guards abort loudly if any source's dev copy is stale (dev-sync trap).
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_eth_microstructure_panel_1h4h_direction_screen_20260823 import (  # noqa: E402
    circular_shift_z,
)

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MICRO_DB_PATH = ROOT / "data/live/microstructure.duckdb"
METRICS_PATH = ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"
FNG_CACHE_DIR = ROOT / "tmp/eth_tier123_combined_20260824"

Z_WINDOW = 288
Z_MINP = 274  # 95% tolerance, NOT == window
CONTAM_MAX = 0.5
COSTS_BP = [10.0, 6.2, 2.5]
HORIZONS = [12, 48]
SPLITS = {"TRAIN": ("2026-05-03", "2026-07-31"), "VAL": ("2026-08-01", "2026-08-16")}

TIER1_RAW = ["obi", "taker_buy_ratio", "spoofing_score", "nif_whale", "nif_retail",
             "shadow_toxicity_score", "shadow_queue_collapse"]
TIER1_Z = ["eai", "oi_delta_pct"]
TIER1 = TIER1_RAW + [f"{c}_z" for c in TIER1_Z]
TIER2 = ["oi_chg_12", "top_acct_lsr_z", "top_pos_lsr_z", "global_lsr_z", "taker_vol_ratio_z"]
TIER3 = ["fng_value", "fng_diff1"]
ARMS = {"A_tier1": TIER1, "B_tier12": TIER1 + TIER2, "C_tier123": TIER1 + TIER2 + TIER3}


def roll_z(s: pd.Series) -> pd.Series:
    mu = s.rolling(Z_WINDOW, min_periods=Z_MINP).mean()
    sd = s.rolling(Z_WINDOW, min_periods=Z_MINP).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def build_frame() -> pd.DataFrame:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    klines = klines.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    klines = klines[(klines["timestamp"] >= "2026-04-25") & (klines["timestamp"] <= "2026-08-18")]
    assert klines["timestamp"].max() >= pd.Timestamp("2026-08-17 23:00"), "klines stale"
    klines["bar_close_time"] = klines["timestamp"] + pd.Timedelta(minutes=5)

    con = duckdb.connect(str(MICRO_DB_PATH), read_only=True)
    micro = con.execute(
        f"SELECT ts, {', '.join(TIER1_RAW + TIER1_Z)} FROM microstructure_1m ORDER BY ts"
    ).fetchdf()
    con.close()
    micro["ts"] = pd.to_datetime(micro["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)
    assert micro["ts"].max() >= pd.Timestamp("2026-08-17"), f"micro dev copy stale: {micro['ts'].max()}"

    metrics = pd.read_csv(METRICS_PATH, parse_dates=["create_time"])
    metrics = metrics[metrics["create_time"] >= "2026-04-20"].sort_values("create_time").reset_index(drop=True)
    assert metrics["create_time"].max() >= pd.Timestamp("2026-08-17"), f"metrics stale: {metrics['create_time'].max()}"
    metrics["oi_chg_12"] = metrics["sum_open_interest"].pct_change(12)
    metrics["top_acct_lsr_z"] = roll_z(metrics["count_toptrader_long_short_ratio"])
    metrics["top_pos_lsr_z"] = roll_z(metrics["sum_toptrader_long_short_ratio"])
    metrics["global_lsr_z"] = roll_z(metrics["count_long_short_ratio"])
    metrics["taker_vol_ratio_z"] = roll_z(metrics["sum_taker_long_short_vol_ratio"])

    FNG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fng_cache = FNG_CACHE_DIR / "fear_greed_daily_raw.csv"
    try:
        resp = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=30)
        resp.raise_for_status()
        fng = pd.DataFrame(resp.json()["data"])
        fng["date"] = pd.to_datetime(fng["timestamp"].astype(np.int64), unit="s").dt.normalize()
        fng["fng_value"] = pd.to_numeric(fng["value"], errors="raise")
        fng = fng[["date", "fng_value"]].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        fng.to_csv(fng_cache, index=False)
    except Exception as exc:  # noqa: BLE001 -- fall back to cache so the 09-30 rerun survives API outages
        print(f"F&G API failed ({exc}); falling back to cache {fng_cache}")
        fng = pd.read_csv(fng_cache, parse_dates=["date"])
    assert fng["date"].max() >= pd.Timestamp("2026-08-16"), f"F&G stale: {fng['date'].max()}"
    fng["fng_diff1"] = fng["fng_value"].diff(1)

    frame = pd.merge_asof(
        klines.sort_values("bar_close_time"), micro.sort_values("ts"),
        left_on="bar_close_time", right_on="ts",
        direction="backward", tolerance=pd.Timedelta("5min"),
    )
    frame = pd.merge_asof(
        frame.sort_values("bar_close_time"),
        metrics[["create_time"] + TIER2].sort_values("create_time"),
        left_on="bar_close_time", right_on="create_time",
        direction="backward", tolerance=pd.Timedelta("30min"),
    )
    frame["date"] = frame["timestamp"].dt.normalize()
    frame = frame.merge(fng[["date", "fng_value", "fng_diff1"]], on="date", how="left")

    for col in TIER1_Z:
        frame[f"{col}_z"] = roll_z(frame[col])
    for h in HORIZONS:
        frame[f"fwd_{h}"] = frame["close"].shift(-h) / frame["close"] - 1.0
    return frame


def economics(sub: pd.DataFrame, score_z: pd.Series, h: int) -> dict:
    fwd = sub[f"fwd_{h}"]
    gross, n_trades, hits, i, idx = 0.0, 0, 0, 0, sub.index.to_list()
    while i < len(idx):
        zi, fi = score_z.get(idx[i]), fwd.get(idx[i])
        if pd.notna(zi) and abs(zi) >= 1.0 and pd.notna(fi):
            ret = np.sign(zi) * fi
            gross += ret
            hits += int(ret > 0)
            n_trades += 1
            i += h
        else:
            i += 1
    always_long = fwd.iloc[::h].dropna().sum()
    bench = max(always_long, -always_long)
    out = {"n": n_trades, "gross_bp": gross / n_trades * 1e4 if n_trades else float("nan"),
           "hit": hits / n_trades if n_trades else float("nan"), "bench": bench}
    for c in COSTS_BP:
        out[f"inc_{c}"] = (gross - n_trades * c / 1e4) - bench
    return out


def main() -> None:
    frame = build_frame()
    subs = {s: frame[(frame["timestamp"] >= a) & (frame["timestamp"] <= b)] for s, (a, b) in SPLITS.items()}

    for split, sub in subs.items():
        cov = {t: sub[c_list].notna().all(axis=1).mean()
               for t, c_list in [("tier1", TIER1), ("tier2", TIER2), ("tier3", TIER3)]}
        print(f"{split}: n={len(sub)} coverage " + " ".join(f"{t}={v * 100:.1f}%" for t, v in cov.items()))
        # Catastrophe guard only (real May/June collector-downtime days legitimately cost
        # ~12% raw + z warm-up; staleness is caught by the max-ts asserts in build_frame).
        for col in TIER1 + TIER2:
            assert sub[col].notna().mean() >= 0.60, f"{split} {col} coverage < 60% -- investigate before trusting"

    tr = subs["TRAIN"]
    print("\ncontamination check (TRAIN spearman vs close), new tier2/3 features:")
    excluded = []
    for col in TIER2 + TIER3:
        d = tr[[col, "close"]].dropna()
        rho = spearmanr(d[col], d["close"]).statistic if len(d) > 500 else float("nan")
        flag = "EXCLUDED" if pd.notna(rho) and abs(rho) >= CONTAM_MAX else "ok"
        if flag == "EXCLUDED":
            excluded.append(col)
        print(f"  {col:<20} rho={rho:+.3f}  {flag}")

    results = {}
    for arm, cols in ARMS.items():
        cols = [c for c in cols if c not in excluded]
        for h in HORIZONS:
            t = tr.dropna(subset=cols + [f"fwd_{h}"])
            mu, sd = t[cols].mean(), t[cols].std().replace(0.0, 1.0)
            ridge = Ridge(alpha=1.0).fit((t[cols] - mu) / sd, t[f"fwd_{h}"])
            print(f"\n=== {arm} h={h} ({h * 5}min) | features={len(cols)} train n={len(t)} ===")
            row = {"cols": cols, "mu": mu, "sd": sd, "model": ridge}
            for split, sub in subs.items():
                v = sub.dropna(subset=cols + [f"fwd_{h}"])
                pred = ridge.predict((v[cols] - mu) / sd)
                ic, z = circular_shift_z(pred, v[f"fwd_{h}"].to_numpy())
                row[split] = (ic, z)
                print(f"  {split}: n={len(v)}  IC={ic:+.4f}  shift-z={z:+.2f}")
            results[(arm, h)] = row

    # primary verdict: arm C h=48
    (ic_tr, z_tr), (ic_va, z_va) = results[("C_tier123", 48)]["TRAIN"], results[("C_tier123", 48)]["VAL"]
    ic_va_A = results[("A_tier1", 48)]["VAL"][0]
    stat_pass = (np.sign(ic_tr) == np.sign(ic_va)) and abs(z_va) >= 3 and ic_va >= ic_va_A
    print(f"\nPRIMARY (C_tier123 h=48): sign_match={np.sign(ic_tr) == np.sign(ic_va)} "
          f"|z_val|={abs(z_va):.2f} (>=3) IC_val={ic_va:+.4f} vs armA {ic_va_A:+.4f} "
          f"-> {'PASS_STATS' if stat_pass else 'FAIL_STATS'}")

    # formula: arm C h=48 coefficients in bp per 1 sigma
    r = results[("C_tier123", 48)]
    print("\nFORMULA (arm C h=48, bp of 4h forward return per +1 sigma):")
    order = np.argsort(-np.abs(r["model"].coef_))
    for j in order:
        print(f"  {r['cols'][j]:<22} {r['model'].coef_[j] * 1e4:+7.2f} bp/1sigma")
    print(f"  (intercept {r['model'].intercept_ * 1e4:+7.2f} bp = TRAIN drift, direction rule uses centered score)")

    # economics: arm C h=48 primary, plus C h=12 and A h=48 reference
    for arm, h in [("C_tier123", 48), ("C_tier123", 12), ("A_tier1", 48)]:
        r = results[(arm, h)]
        cols = r["cols"]
        t = tr.dropna(subset=cols + [f"fwd_{h}"])
        pred_tr = r["model"].predict((t[cols] - r["mu"]) / r["sd"])
        p_mu, p_sd = pred_tr.mean(), pred_tr.std()
        print(f"\nECONOMICS {arm} h={h} (entry |score_z|>=1, non-overlap hold, vs max(always)):")
        for split, sub in subs.items():
            ok = sub[cols].notna().all(axis=1) & sub[f"fwd_{h}"].notna()
            score_z = pd.Series(np.nan, index=sub.index)
            if ok.sum():
                score_z.loc[ok] = (r["model"].predict((sub.loc[ok, cols] - r["mu"]) / r["sd"]) - p_mu) / p_sd
            e = economics(sub, score_z, h)
            incs = " ".join(f"inc@{c}bp={e[f'inc_{c}'] * 100:+.2f}%" for c in COSTS_BP)
            print(f"  {split}: n={e['n']} gross={e['gross_bp']:+.2f}bp/trade hit={e['hit'] * 100:.1f}% "
                  f"bench={e['bench'] * 100:+.2f}% {incs}")


if __name__ == "__main__":
    main()
