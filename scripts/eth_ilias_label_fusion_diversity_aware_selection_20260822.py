#!/usr/bin/env python3
"""Felici & Sudoso(2023) 정신의 "다양성-정확도 동시고려 피처선택"을 실제로 구현·테스트 -- 08-22
사용자 재질문("제대로 해보자")에 대한 응답. 앞서 문헌검토 판단("풀을 넓혀야 가치가 나는데,
그 풀 확장 자체가 이 저장소의 반복실패 패턴")을 실제 구현으로 검증.

기존 stack(§6, hand-picked 3피처: zz_signal/cs_signal/h4_quality) 대신, zigzag/cusum/h48qual
각 라벨이 이미 저장중인 dir_p_long/dir_p_short/dir_confidence/quality_for_action 4개씩 =
12개 후보 피처 풀에서, "정확도 개선 + 이미 선택된 피처와 상관 낮음(다양성)"을 동시에 만족하는
피처만 그리디로 선택한다(신경망 multi-task 재구현이 아니라 논문의 핵심 원리를 greedy selection
으로 구현 -- 재현이 아니라 원리 적용임을 명시).

⚠️ 풀 확장 자체가 위험(라벨/피쳐 변형을 늘려 우연한 적합을 주울 위험, [[repo_label_methodology_meta_finding]])
하므로, 선택 절차를 2024 내부에서만 완결한다: SELECT-FIT=2024 H1, SELECT-VALIDATE=2024 H2로
그리디 선택 후 최종모델을 2024 전체로 재적합, TEST=2025(선택 절차가 한 번도 들여다보지 않은
구간)에서 딱 1번 평가. OOS는 이번에도 미접촉.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

import eth_ilias_label_fusion_train_period_feasibility_20260821 as fsn  # noqa: E402

omega = fsn.omega
parent = fsn.parent
_metrics_with_ledger = fsn.ledger_mod._metrics_with_ledger
SEEDS = fsn.SEEDS
FEE, SLIP = omega._load_fee_slip()
COST_MULT = 3.0

PREFIX = "omega1_regime3_expertdq_oof_"
FEAT_SUFFIXES = ["dir_p_long", "dir_p_short", "dir_confidence", "quality_for_action"]
FORWARD_BARS = 48
CORR_THRESHOLD = 0.5  # candidate rejected if |corr| with any already-selected feature exceeds this
MAX_FEATURES = 5
OUT_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/diversity_aware_selection_2025holdout"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_label_preds(label: str, seed: int) -> pd.DataFrame:
    out_dir = fsn.OUT_ROOT / f"{fsn.MODEL_ID}_label5way_{label}_154feat_ilias_anchored_seed{seed}_20260821"
    report = json.loads((out_dir / "report.json").read_text())
    best = report["ranking_by_validation_pnl"][0]
    q_tag = f"q{int(round(float(best['quality_threshold']) * 100.0)):03d}"
    preds = pd.concat(
        [pd.read_csv(out_dir / f"train_predictions_{q_tag}.csv", parse_dates=["timestamp"]),
         pd.read_csv(out_dir / f"validation_predictions_{q_tag}.csv", parse_dates=["timestamp"])],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)
    return preds


def _combine_from_side(dec_template: pd.DataFrame, side_combined: np.ndarray) -> pd.DataFrame:
    active = side_combined != 0
    return pd.DataFrame({
        "timestamp": dec_template["timestamp"].to_numpy(),
        "action": np.where(side_combined == 1, 1, np.where(side_combined == -1, 2, 0)),
        "side": side_combined,
        "notional_exposure": np.where(active, dec_template["notional_exposure"].to_numpy(), 0.0),
        "leverage": np.where(active, dec_template["leverage"].to_numpy(), 1.0),
        "take_profit": np.where(active, dec_template["take_profit"].to_numpy(), 0.0),
        "stop_loss": np.where(active, dec_template["stop_loss"].to_numpy(), 0.0),
        "max_hold_bars": dec_template["max_hold_bars"].to_numpy(),
        "cooldown_bars": dec_template["cooldown_bars"].to_numpy(),
    })


def _greedy_select(X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                    names: list[str]) -> list[int]:
    n_feat = X_fit.shape[1]
    selected: list[int] = []
    remaining = list(range(n_feat))
    best_val_loss = None
    while remaining and len(selected) < MAX_FEATURES:
        candidates = []
        for j in remaining:
            if selected:
                corrs = [abs(np.corrcoef(X_fit[:, j], X_fit[:, k])[0, 1]) for k in selected]
                if max(corrs) > CORR_THRESHOLD:
                    continue
            trial = selected + [j]
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_fit[:, trial], y_fit)
            p_val = clf.predict_proba(X_val[:, trial])[:, 1]
            val_loss = log_loss(y_val, p_val, labels=[0, 1])
            candidates.append((val_loss, j))
        if not candidates:
            break
        candidates.sort(key=lambda t: t[0])
        best_loss, best_j = candidates[0]
        if best_val_loss is not None and best_loss >= best_val_loss - 1e-4:
            break
        best_val_loss = best_loss
        selected.append(best_j)
        remaining.remove(best_j)
    return selected


def main() -> None:
    full_raw = pd.concat(
        [pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv",
                      usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
         for y in (2024, 2025)], ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)
    full_raw["fwd_ret"] = full_raw["close"].shift(-FORWARD_BARS) / full_raw["close"] - 1.0
    full_raw["y_up"] = (full_raw["fwd_ret"] > 0).astype(int)
    raw_2025 = full_raw[full_raw["timestamp"].dt.year == 2025]
    bench = (float(raw_2025["close"].iloc[-1]) / float(raw_2025["close"].iloc[0]) - 1.0) * 100.0
    print(f"2025-only always-long benchmark: {bench:+.2f}%", flush=True)

    feat_names = [f"{lb}_{suf}" for lb in ("zigzag", "cusum", "h48qual") for suf in FEAT_SUFFIXES]
    summary_rows, selection_rows = [], []

    for seed in SEEDS:
        preds = {lb: _load_label_preds(lb, seed) for lb in ("zigzag", "cusum", "h48qual")}
        decs = {lb: parent._to_decisions(preds[lb], oof=True) for lb in preds}
        for lb in decs:
            decs[lb].insert(0, "timestamp", preds[lb]["timestamp"].to_numpy())

        ts = preds["zigzag"]["timestamp"]
        cols = []
        for lb in ("zigzag", "cusum", "h48qual"):
            for suf in FEAT_SUFFIXES:
                cols.append(preds[lb][f"{PREFIX}{suf}"].to_numpy(dtype=np.float64))
        X_all = np.stack(cols, axis=1)

        frame = pd.DataFrame({"timestamp": ts})
        frame = frame.merge(full_raw[["timestamp", "y_up", "fwd_ret"]], on="timestamp", how="inner")
        keep = frame["fwd_ret"].notna().to_numpy()
        frame = frame[keep].reset_index(drop=True)
        X_all = X_all[keep]

        year = frame["timestamp"].dt.year
        h1_mask = ((year == 2024) & (frame["timestamp"].dt.month <= 6)).to_numpy()
        h2_mask = ((year == 2024) & (frame["timestamp"].dt.month > 6)).to_numpy()
        fit_mask = (year == 2024).to_numpy()
        eval_mask = (year == 2025).to_numpy()
        y = frame["y_up"].to_numpy(dtype=np.int64)

        mu = X_all[h1_mask].mean(axis=0, keepdims=True)
        sd = X_all[h1_mask].std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        X_std = (X_all - mu) / sd

        selected_idx = _greedy_select(X_std[h1_mask], y[h1_mask], X_std[h2_mask], y[h2_mask], feat_names)
        selected_names = [feat_names[i] for i in selected_idx]
        print(f"[seed={seed}] selected features (in order): {selected_names}", flush=True)
        selection_rows.append({"seed": seed, "selected_features": ";".join(selected_names), "n_selected": len(selected_idx)})

        if not selected_idx:
            div_side = np.zeros(int(eval_mask.sum()), dtype=np.int64)
        else:
            clf_final = LogisticRegression(max_iter=1000)
            clf_final.fit(X_std[fit_mask][:, selected_idx], y[fit_mask])
            p_eval = clf_final.predict_proba(X_std[eval_mask][:, selected_idx])[:, 1]
            div_side = np.where(p_eval > 0.55, 1, np.where(p_eval < 0.45, -1, 0)).astype(np.int64)

        eval_ts = frame.loc[eval_mask, ["timestamp"]].reset_index(drop=True)
        eval_frame = full_raw.merge(eval_ts, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
        assert len(eval_frame) == len(eval_ts)
        dec_template = decs["zigzag"].merge(eval_ts, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
        assert len(dec_template) == len(eval_ts)

        dec_combined = _combine_from_side(dec_template, div_side)
        agg, ledger = _metrics_with_ledger(eval_frame, dec_combined, fee=FEE, slip=SLIP, cost_mult=COST_MULT)
        ledger.to_csv(OUT_DIR / f"diversitysel_seed{seed}_2025holdout_trade_ledger.csv", index=False)
        lf = agg["long_entries"] / max(agg["long_entries"] + agg["short_entries"], 1)
        summary_rows.append({
            "seed": seed, "n_selected": len(selected_idx), "selected_features": ";".join(selected_names),
            "trades": agg["trades"], "pnl": agg["pnl"], "mdd": agg["mdd"],
            "long_entries": agg["long_entries"], "short_entries": agg["short_entries"], "long_frac": lf,
        })
        print(f"[seed={seed}] pnl={agg['pnl']:7.2f}% trades={agg['trades']:3d} long_frac={lf:.3f}", flush=True)

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUT_DIR / "selected_features.csv", index=False)

    print(f"\n=== diversity-aware selection, 2025-holdout aggregate (N=6 seeds), bench={bench:+.2f}% ===")
    corr = df["long_frac"].corr(df["pnl"])
    print(f"mean_pnl={df['pnl'].mean():7.2f} std={df['pnl'].std():6.2f} mean_long_frac={df['long_frac'].mean():.3f} "
          f"corr={corr:.3f} trades=[{df['trades'].min()},{df['trades'].max()}] "
          f"mean_n_selected={df['n_selected'].mean():.1f}")
    print(f"\nsaved {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
