#!/usr/bin/env python3
"""ZDC(wick-앵커) HOLDOUT 소진(분류+경제성 동시, 1회성) -- 계획서(swift-doodling-grove.md) Step F.

research_eth_v_rebound_multitrigger_holdout_20260831.py(giveback 9트리거의 HOLDOUT 스크립트)와
동일 구조(4시드 TRAIN적합→VAL/OOS/HOLDOUT 채점, called=proba>=0.5 후보빌드, 경제성 그리드 재현)
재사용 -- 단 두 가지 의도적 차이:
  1. FORWARD_BARS=400(giveback의 200이 아님) -- Step E에서 이미 확정한 값, ZDC는 자기 라벨해상
     자체가 giveback보다 느려(p99=208봉) 더 넓은 버퍼가 필요했음.
  2. **경제성 config를 이 스크립트가 새로 combined-sort top1으로 재선정하지 않는다** --
     giveback 원본 스크립트는 그렇게 했지만, 이번 세션에 바로 그 방식(그리드 최상위를 그냥
     신뢰)이 노이즈수확 아티팩트(ARM=0.1)를 낳는다는 걸 발견했다
     (feedback_trailing_stop_low_arm_noise_harvest_artifact_20260901). Step E에서 이미 205셀
     전체를 방향뒤집기로 검증해 SL=4.0/ARM=1.5/Trail=0.1을 "진짜"(gap_val=+2.54bp/gap_oos=
     +4.64bp)로 확정해뒀으므로, 그 결과를 그대로 고정 사용한다("계획서 Step F: VAL+OOS로만
     모든 하이퍼파라미터 선정 완료 후 정확히 1회 평가"와 정합).

HOLDOUT 경제성 결과 자체에도 방향뒤집기 대조군을 바로 적용한다(진단, 새 파라미터 선정 아님) --
1회성 노출인 건 동일하되, 이번엔 사후감사가 아니라 생성 시점에 바로 검증하는 것.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_zigzag_direction_20260901/eth_5m_v_rebound_multitrigger_zigzag_direction_features_tier0.csv"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/research/eth_v_rebound_multitrigger_zigzag_direction_holdout_20260901"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")
LABEL_WINDOW = pd.Timedelta(hours=24)
SEEDS = [20260829, 141592, 271828, 577215]
FORWARD_BARS = 400
STANDARD_COST_BP = 10.0
SELECTED_SL, SELECTED_ARM, SELECTED_TRAIL = 4.0, 1.5, 0.1  # Step E에서 방향뒤집기로 이미 검증된 config

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    pred = (proba >= 0.5).astype(int)
    naive_acc = float(max(y.mean(), 1.0 - y.mean()))
    accuracy = float((pred == y).mean())
    return {
        "n": int(len(y)), "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "naive_majority_class_accuracy": round(naive_acc, 4),
        "beats_naive_accuracy": bool(accuracy > naive_acc),
    }


def split_with_holdout(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    return {
        "train": df.loc[(ts < VAL_START) & (window_end < VAL_START)],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END) & (ts < HOLDOUT_START)],
        "holdout": df.loc[ts >= HOLDOUT_START].reset_index(drop=True),
    }


def build_candidates(called: pd.DataFrame, kl: pd.DataFrame, ts_to_idx: pd.Series) -> pd.DataFrame:
    rows = []
    for _, ev in called.iterrows():
        pos = ts_to_idx.get(ev["timestamp"])
        if pos is None or pos + FORWARD_BARS + 1 >= len(kl):
            continue
        entry_bar = kl.iloc[pos + 1]
        side = "long" if ev["is_downside"] == 1 else "short"
        fwd = kl.iloc[pos + 1: pos + 1 + FORWARD_BARS][["timestamp", "open", "high", "low", "close"]]
        rows.append({
            "idx": int(ev["idx"]), "event_ts": ev["timestamp"], "split": ev["split"],
            "side": side, "model_proba": float(ev["model_proba"]), "label": int(ev["label"]),
            "atr": float(ev["atr"]), "entry_ts": entry_bar["timestamp"], "entry_price": float(entry_bar["open"]),
            "fwd_open": fwd["open"].tolist(), "fwd_high": fwd["high"].tolist(),
            "fwd_low": fwd["low"].tolist(), "fwd_close": fwd["close"].tolist(),
        })
    return pd.DataFrame(rows)


def simulate_trailing(row: pd.Series, sl_mult: float, arm_mult: float, trail_mult: float, pessimistic: bool) -> float:
    atr = row["atr"]
    entry = row["entry_price"]
    side = row["side"]
    opens, highs, lows, closes = row["fwd_open"], row["fwd_high"], row["fwd_low"], row["fwd_close"]
    sign = 1.0 if side == "long" else -1.0
    stop = entry - sign * sl_mult * atr
    armed = False
    best = entry
    for o, h, l, c in zip(opens, highs, lows, closes):
        fav_extreme = h if side == "long" else l
        adv_extreme = l if side == "long" else h

        def stop_hit() -> bool:
            return (adv_extreme <= stop) if side == "long" else (adv_extreme >= stop)

        def update_trailing() -> None:
            nonlocal armed, stop, best
            if sign * (fav_extreme - best) > 0:
                best = fav_extreme
            if not armed and sign * (best - entry) >= arm_mult * atr:
                armed = True
            if armed:
                new_stop = best - sign * trail_mult * atr
                if sign * (new_stop - stop) > 0:
                    stop = new_stop

        if pessimistic:
            if stop_hit():
                return sign * (stop - entry) / entry
            update_trailing()
        else:
            update_trailing()
            if stop_hit():
                return sign * (stop - entry) / entry
    return sign * (closes[-1] - entry) / entry


def econ_metrics(cand: pd.DataFrame) -> dict:
    if len(cand) == 0:
        return {"n": 0, "opt_bp": None, "pess_bp": None, "win_rate": None}
    opt = cand.apply(lambda r: simulate_trailing(r, SELECTED_SL, SELECTED_ARM, SELECTED_TRAIL, False), axis=1)
    pess = cand.apply(lambda r: simulate_trailing(r, SELECTED_SL, SELECTED_ARM, SELECTED_TRAIL, True), axis=1)
    return {
        "n": int(len(cand)),
        "opt_bp": round(float(opt.mean() * 1e4 - STANDARD_COST_BP), 2),
        "pess_bp": round(float(pess.mean() * 1e4 - STANDARD_COST_BP), 2),
        "win_rate": round(float((opt > 0).mean()), 4),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hit_bool"] = df["hit"].astype(str).map({"True": True, "False": False})
    df = df[df["hit_bool"].isin([True, False])].copy()
    df["label"] = df["hit_bool"].astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = split_with_holdout(df)
    for name in ("train", "val", "oos", "holdout"):
        p = parts[name]
        print(f"{name}: n={len(p)} label_rate={p['label'].mean():.4f}", flush=True)
    over_limit = len(parts["train"]) > 10000

    print("\n=== fitting 4-seed ensemble on TRAIN, scoring VAL/OOS/HOLDOUT (HOLDOUT TOUCHED HERE) ===", flush=True)
    probas = {"val": [], "oos": [], "holdout": []}
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=over_limit)
        clf.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
        for name in ("val", "oos", "holdout"):
            probas[name].append(clf.predict_proba(parts[name][FEATURE_COLUMNS])[:, 1])
        print(f"  seed={seed} done", flush=True)

    classification = {}
    scored = {}
    for name in ("val", "oos", "holdout"):
        p = parts[name].copy()
        p["model_proba"] = np.mean(probas[name], axis=0)
        p["split"] = name
        scored[name] = p
        r = evaluate(p["model_proba"].to_numpy(), p["label"].to_numpy())
        classification[name] = r
        print(f"  {name:7s} n={r['n']:5d} auc={r['auc']:.4f} bal_acc={r['balanced_accuracy']:.4f} "
              f"beats_naive={r['beats_naive_accuracy']}", flush=True)

    print("\n=== building trade candidates (called=proba>=0.5) for economics ===", flush=True)
    kl = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close"])
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True)
    kl = kl.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts_to_idx = pd.Series(kl.index.to_numpy(), index=kl["timestamp"].to_numpy())

    all_candidates = {}
    for name in ("val", "oos", "holdout"):
        called = scored[name][scored[name]["model_proba"] >= 0.5].copy()
        cand = build_candidates(called, kl, ts_to_idx)
        all_candidates[name] = cand
        print(f"  {name}: called {len(called)}/{len(scored[name])}, with forward data {len(cand)}, "
              f"precision(label==1|called)={called['label'].mean():.4f}" if len(called) else f"  {name}: called 0", flush=True)

    print(f"\n=== economics, FIXED config (Step E에서 방향뒤집기로 확정) "
          f"SL={SELECTED_SL} ARM={SELECTED_ARM} Trail={SELECTED_TRAIL} ===", flush=True)
    economics = {}
    for name in ("val", "oos", "holdout"):
        economics[name] = econ_metrics(all_candidates[name])
        e = economics[name]
        print(f"  {name:7s} n={e['n']:5d} opt={e['opt_bp']}bp pess={e['pess_bp']}bp win_rate={e['win_rate']}", flush=True)

    print("\n=== HOLDOUT 경제성 결과에 방향뒤집기 대조군 즉시 적용(진단) ===", flush=True)
    holdout_cand = all_candidates["holdout"]
    if len(holdout_cand) > 0:
        holdout_flipped = holdout_cand.copy()
        holdout_flipped["side"] = holdout_flipped["side"].map({"long": "short", "short": "long"})
        flip_econ = econ_metrics(holdout_flipped)
        real_min = min(economics["holdout"]["opt_bp"], economics["holdout"]["pess_bp"])
        flip_min = min(flip_econ["opt_bp"], flip_econ["pess_bp"])
        holdout_genuine = real_min > flip_min and real_min > 0
        print(f"  real: opt={economics['holdout']['opt_bp']}bp pess={economics['holdout']['pess_bp']}bp win={economics['holdout']['win_rate']}")
        print(f"  flip: opt={flip_econ['opt_bp']}bp pess={flip_econ['pess_bp']}bp win={flip_econ['win_rate']}")
        print(f"  => HOLDOUT is {'GENUINE' if holdout_genuine else 'ARTIFACT-SUSPECT'}")
    else:
        flip_econ, holdout_genuine = None, None
        print("  HOLDOUT candidates empty, skip flip check")

    report = {
        "seeds": SEEDS, "train_n": int(len(parts["train"])), "ignore_pretraining_limits": over_limit,
        "forward_bars": FORWARD_BARS,
        "classification": classification,
        "selected_config": {"sl": SELECTED_SL, "arm": SELECTED_ARM, "trail": SELECTED_TRAIL,
                             "selected_on": "val+oos only, direction-flip-verified in Step E (2026-09-01)"},
        "economics": economics,
        "holdout_direction_flip_check": {"flipped": flip_econ, "genuine": holdout_genuine},
        "note": "HOLDOUT touched exactly once for both classification and economics in this run. "
                "Do not re-run with different parameters -- that would be a second holdout look.",
    }
    (OUT_DIR / "holdout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    for name in ("val", "oos", "holdout"):
        all_candidates[name].to_pickle(OUT_DIR / f"candidates_{name}.pkl")
    print(f"\nWrote {OUT_DIR / 'holdout_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
