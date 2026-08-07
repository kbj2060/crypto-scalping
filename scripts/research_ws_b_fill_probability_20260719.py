"""WS-B: Maker 체결확률 모델 - T-B0(라벨 sanity)~T-B2(경제성 백테스트) 실증 실행.

1단계 근사: 1m OHLC로 60초 체결/역선택 라벨 근사 (design-doc에 명시된 한계).
Shadow(T-B3)는 4주 필요하므로 이번 세션 범위 밖.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

MICRO_DB = "data/live/microstructure.duckdb"
KLINE_CSV = "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT_DIR = Path("docs/test_designs_duckdb_live_20260719/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def connect_retry(path, read_only=True, retries=8, backoff=2.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return duckdb.connect(path, read_only=read_only)
        except duckdb.IOException as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    raise last_exc


def load_snapshots():
    con = connect_retry(MICRO_DB)
    snaps = con.execute(
        """
        select recorded_at_kst, best_bid, best_ask, mid, spread_bps, microprice_edge_bps,
               imbalance_1, imbalance_5, imbalance_10, imbalance_20,
               bid_qty_1, ask_qty_1
        from orderbook_decision_snapshots order by recorded_at_kst
        """
    ).df()
    micro = con.execute(
        "select ts, obi, taker_buy_ratio, shadow_queue_collapse, shadow_absorption_score, "
        "shadow_toxicity_score from microstructure_1m order by ts"
    ).df()
    con.close()
    return snaps, micro


def load_kline_ohlc():
    kl = pd.read_csv(KLINE_CSV, usecols=["timestamp", "open", "high", "low", "close"])
    kl["ts"] = pd.to_datetime(kl["timestamp"], utc=True)
    return kl[["ts", "open", "high", "low", "close"]].sort_values("ts").reset_index(drop=True)


def main():
    report = {"stage": "WS-B", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    snaps, micro = load_snapshots()
    kline = load_kline_ohlc()
    snaps["recorded_at_kst"] = pd.to_datetime(snaps["recorded_at_kst"], utc=True)
    micro["ts"] = pd.to_datetime(micro["ts"], utc=True)
    report["n_snapshots"] = int(len(snaps))
    report["kline_coverage"] = {"min": str(kline["ts"].min()), "max": str(kline["ts"].max())}

    # ---- causal feature join: last CLOSED 1m bar strictly before snapshot ts ----
    micro_sorted = micro.sort_values("ts").reset_index(drop=True)
    idx_feat = np.searchsorted(micro_sorted["ts"].values, snaps["recorded_at_kst"].values, side="left") - 1
    n_before_clip = len(idx_feat)
    valid_feat_mask = idx_feat >= 0
    idx_feat_clipped = np.clip(idx_feat, 0, len(micro_sorted) - 1)
    feat_bar_close = micro_sorted["ts"].iloc[idx_feat_clipped].reset_index(drop=True)
    causal_violations = int((feat_bar_close.values >= snaps["recorded_at_kst"].values).sum())
    report["T_B0_causal_join_violations"] = causal_violations
    assert causal_violations == 0, "causal join violated -- feature bar closes at/after snapshot ts"

    for col in ["obi", "taker_buy_ratio", "shadow_queue_collapse", "shadow_absorption_score", "shadow_toxicity_score"]:
        snaps[col] = micro_sorted[col].iloc[idx_feat_clipped].reset_index(drop=True)
    snaps.loc[~valid_feat_mask, ["obi", "taker_buy_ratio", "shadow_queue_collapse",
                                  "shadow_absorption_score", "shadow_toxicity_score"]] = np.nan

    # ---- label construction: next full-minute kline bar approximates [t, t+60s] ----
    kline_ts = kline["ts"].values
    idx_label = np.searchsorted(kline_ts, snaps["recorded_at_kst"].values, side="right")
    valid_label_mask = idx_label < len(kline)
    idx_label_clipped = np.clip(idx_label, 0, len(kline) - 1)
    label_bar = kline.iloc[idx_label_clipped].reset_index(drop=True)
    snaps["label_bar_ts"] = label_bar["ts"].values
    snaps["label_bar_low"] = label_bar["low"].values
    snaps["label_bar_high"] = label_bar["high"].values
    snaps["label_bar_close"] = label_bar["close"].values
    snaps.loc[~valid_label_mask, ["label_bar_low", "label_bar_high", "label_bar_close"]] = np.nan

    # buy-side maker at best_bid: filled if next-bar low < best_bid (trade-through, conservative)
    snaps["filled_60s_buy"] = snaps["label_bar_low"] < snaps["best_bid"]
    snaps["adverse_bps_buy"] = (
        (snaps["label_bar_close"] - snaps["best_bid"]) / snaps["mid"] * 1e4
    )
    # sell-side maker at best_ask: filled if next-bar high > best_ask
    snaps["filled_60s_sell"] = snaps["label_bar_high"] > snaps["best_ask"]
    snaps["adverse_bps_sell"] = (
        (snaps["best_ask"] - snaps["label_bar_close"]) / snaps["mid"] * 1e4
    )

    valid_rows = snaps["label_bar_close"].notna() & snaps["best_bid"].notna() & (snaps["mid"] > 0)
    ds = snaps.loc[valid_rows].reset_index(drop=True)
    report["T_B0_valid_labeled_rows"] = int(len(ds))
    report["T_B0_label_coverage_ratio"] = float(len(ds) / len(snaps))

    # ---- T-B0 label sanity ----
    sanity = {}
    for side in ["buy", "sell"]:
        base_rate = float(ds[f"filled_60s_{side}"].mean())
        adverse_mean_when_filled = float(
            ds.loc[ds[f"filled_60s_{side}"], f"adverse_bps_{side}"].mean()
        )
        sanity[side] = {
            "base_rate": base_rate,
            "base_rate_in_range_5_95pct": bool(0.05 <= base_rate <= 0.95),
            "adverse_bps_mean_when_filled": adverse_mean_when_filled,
            "adverse_negative_as_expected": bool(adverse_mean_when_filled < 0)
            if not np.isnan(adverse_mean_when_filled) else None,
        }
    report["T_B0_label_sanity"] = sanity

    # ---- T-B1: baseline models (time-split) ----
    ds = ds.sort_values("recorded_at_kst").reset_index(drop=True)
    n = len(ds)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    # 1-day purge gap approx (skip ~ rows corresponding to 1 day at ~5min cadence = ~288 rows)
    purge = 288
    train = ds.iloc[: max(train_end - purge, 0)]
    val = ds.iloc[train_end: max(val_end - purge, train_end)]
    test = ds.iloc[val_end:]
    report["T_B1_split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    report["T_B1_split_dates"] = {
        "train_end": str(train["recorded_at_kst"].max()) if len(train) else None,
        "val_start": str(val["recorded_at_kst"].min()) if len(val) else None,
        "val_end": str(val["recorded_at_kst"].max()) if len(val) else None,
        "test_start": str(test["recorded_at_kst"].min()) if len(test) else None,
        "test_end": str(test["recorded_at_kst"].max()) if len(test) else None,
    }

    features = ["spread_bps", "microprice_edge_bps", "imbalance_1", "imbalance_5",
                "imbalance_10", "imbalance_20", "obi", "taker_buy_ratio",
                "shadow_queue_collapse", "shadow_absorption_score", "shadow_toxicity_score"]

    b1_results = {}
    for side in ["buy", "sell"]:
        target = f"filled_60s_{side}"
        cols = features + [target]
        tr = train[cols].dropna()
        va = val[cols].dropna()
        te = test[cols].dropna()
        if len(tr) < 100 or len(va) < 30 or len(te) < 30:
            b1_results[side] = {"skipped": "insufficient_n", "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
            continue

        Xtr, ytr = tr[features].values, tr[target].astype(int).values
        Xva, yva = va[features].values, va[target].astype(int).values
        Xte, yte = te[features].values, te[target].astype(int).values

        side_result = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
                        "base_rate_test": float(yte.mean())}

        # logistic regression baseline
        lr = LogisticRegression(max_iter=1000)
        lr.fit(Xtr, ytr)
        p_te_lr = lr.predict_proba(Xte)[:, 1]
        side_result["logistic"] = {
            "test_auc": float(roc_auc_score(yte, p_te_lr)) if len(set(yte)) > 1 else None,
            "test_brier": float(brier_score_loss(yte, p_te_lr)),
            "const_brier": float(brier_score_loss(yte, np.full_like(p_te_lr, yte.mean()))),
        }

        if HAS_LGB:
            gbm = lgb.LGBMClassifier(max_depth=4, n_estimators=200, learning_rate=0.05,
                                      min_child_samples=30, verbose=-1)
            gbm.fit(Xtr, ytr)
            p_te_gbm = gbm.predict_proba(Xte)[:, 1]
            side_result["lightgbm"] = {
                "test_auc": float(roc_auc_score(yte, p_te_gbm)) if len(set(yte)) > 1 else None,
                "test_brier": float(brier_score_loss(yte, p_te_gbm)),
                "const_brier": float(brier_score_loss(yte, np.full_like(p_te_gbm, yte.mean()))),
                "feature_importance": dict(zip(features, [float(x) for x in gbm.feature_importances_])),
            }
            # reliability curve (10-bin)
            try:
                frac_pos, mean_pred = calibration_curve(yte, p_te_gbm, n_bins=10, strategy="quantile")
                side_result["lightgbm"]["reliability_curve"] = {
                    "mean_predicted": mean_pred.tolist(), "frac_positive": frac_pos.tolist()
                }
            except Exception:
                pass
            # store predictions for T-B2 (val used for theta selection, test for final eval)
            p_va_gbm = gbm.predict_proba(Xva)[:, 1]
            side_result["_p_fill_test"] = p_te_gbm.tolist()
            side_result["_p_fill_val"] = p_va_gbm.tolist()
        b1_results[side] = side_result

    report["T_B1_models"] = b1_results

    # kill-gate check
    verdicts = {}
    for side in ["buy", "sell"]:
        r = b1_results.get(side, {})
        auc = (r.get("lightgbm") or r.get("logistic") or {}).get("test_auc")
        if auc is None:
            verdicts[side] = "SKIPPED (insufficient data)"
        elif auc < 0.55:
            verdicts[side] = f"H-B1 REJECTED (test AUC {auc:.3f} < 0.55) -- STOP per kill gate"
        else:
            verdicts[side] = f"H-B1 tentatively ACCEPTED (test AUC {auc:.3f} >= 0.55) -- proceed to T-B2"
    report["H_B1_verdict"] = verdicts

    # ---- T-B2: economic backtest (only if H-B1 not rejected for that side) ----
    b2_results = {}
    assumed_taker_cost_bps = 5.0  # from WS-A: verified fee constant
    for side in ["buy", "sell"]:
        if "REJECTED" in verdicts.get(side, ""):
            b2_results[side] = {"skipped": "H-B1 kill-gate triggered, not run per design"}
            continue
        r = b1_results.get(side, {})
        if "_p_fill_test" not in r or "_p_fill_val" not in r:
            b2_results[side] = {"skipped": "no lightgbm predictions available"}
            continue

        def cost_given_policy(df_side, p_fill_arr, theta_val, taker_cost):
            realized_fill_ = df_side[f"filled_60s_{side}"].astype(int).values
            realized_adverse_ = df_side[f"adverse_bps_{side}"].values
            spread_ = df_side["spread_bps"].values
            maker_decision_ = p_fill_arr >= theta_val
            cost_maker_leg = np.where(
                realized_fill_ == 1, -spread_ / 2.0 + realized_adverse_, taker_cost
            )
            return np.where(maker_decision_, cost_maker_leg, taker_cost)

        val_ds = val[features + [f"filled_60s_{side}", f"adverse_bps_{side}"]].dropna().reset_index(drop=True)
        test_ds = test[features + [f"filled_60s_{side}", f"adverse_bps_{side}"]].dropna().reset_index(drop=True)
        p_fill_val = np.array(r["_p_fill_val"])
        p_fill_test = np.array(r["_p_fill_test"])
        if len(val_ds) != len(p_fill_val) or len(test_ds) != len(p_fill_test):
            b2_results[side] = {"skipped": "val/test length mismatch after dropna realignment"}
            continue

        # theta grid search on VAL ONLY (design requirement: theta selected on val, applied to test)
        theta_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
        val_costs_by_theta = {}
        for th in theta_grid:
            c = cost_given_policy(val_ds, p_fill_val, th, assumed_taker_cost_bps)
            val_costs_by_theta[th] = float(np.mean(c))
        theta = min(val_costs_by_theta, key=val_costs_by_theta.get)

        te = test_ds
        realized_fill = te[f"filled_60s_{side}"].astype(int).values
        realized_adverse = te[f"adverse_bps_{side}"].values
        spread_bps_arr = te["spread_bps"].values

        cost_b0 = np.full(len(te), assumed_taker_cost_bps)
        cost_b1 = np.where(
            realized_fill == 1, -spread_bps_arr / 2.0 + realized_adverse, assumed_taker_cost_bps
        )
        cost_b3 = cost_given_policy(te, p_fill_test, theta, assumed_taker_cost_bps)

        def summarize(cost_arr, day_arr):
            df_c = pd.DataFrame({"cost": cost_arr, "day": day_arr})
            daily = df_c.groupby("day")["cost"].mean()
            return {"mean_cost_bps": float(np.mean(cost_arr)), "n": int(len(cost_arr)), "n_days": int(len(daily))}

        test_days_series = test[features + [f"filled_60s_{side}", f"adverse_bps_{side}", "recorded_at_kst"]].dropna(
            subset=features + [f"filled_60s_{side}", f"adverse_bps_{side}"]
        )["recorded_at_kst"].dt.date.astype(str).values

        # day-block bootstrap for B3 (model policy, lower cost=better) vs B1 (always-maker)
        rng = np.random.default_rng(20260719)
        diff_by_day_df = pd.DataFrame({"day": test_days_series, "b3": cost_b3, "b1": cost_b1})
        day_diffs = diff_by_day_df.groupby("day").apply(
            lambda g: g["b1"].mean() - g["b3"].mean()  # positive = B3 saves cost vs B1
        )
        unique_days = day_diffs.index.to_numpy()
        boot_means = np.array([
            rng.choice(day_diffs.values, size=len(unique_days), replace=True).mean()
            for _ in range(3000)
        ])
        boot_se = float(np.std(boot_means))
        observed_mean = float(day_diffs.mean())
        t_stat = observed_mean / boot_se if boot_se > 1e-12 else None

        b2_results[side] = {
            "theta_selected_on_val": theta,
            "theta_grid_val_costs": val_costs_by_theta,
            "B0_always_taker": summarize(cost_b0, test_days_series),
            "B1_always_maker": summarize(cost_b1, test_days_series),
            "B3_model_policy": summarize(cost_b3, test_days_series),
            "B3_minus_B1_mean_bps": float(np.mean(cost_b3) - np.mean(cost_b1)),
            "B1_minus_B0_mean_bps": float(np.mean(cost_b1) - np.mean(cost_b0)),
            "n_test_days": int(len(unique_days)),
            "B3_savings_vs_B1_day_block_bootstrap": {
                "mean_saving_bps": observed_mean,
                "boot_se": boot_se,
                "t_stat": t_stat,
                "significant_t_gt_3": bool(t_stat is not None and t_stat > 3),
            },
        }

    report["T_B2_economic_backtest"] = b2_results

    # strip large embedded predictions before dumping full report (keep separately if needed)
    for side in b1_results:
        b1_results[side].pop("_p_fill_test", None)

    out_json = OUT_DIR / "ws_b_fill_probability_20260719.json"
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("WROTE", out_json)
    print(json.dumps({k: report[k] for k in report if k not in ("T_B1_models",)}, indent=2, default=str, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
