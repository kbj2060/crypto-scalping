"""Diagnostic (2026-07-07): does the L4 risk-sidecar's raw HGB score actually correlate with
realized trade outcome? If not, no amount of conformal recalibration of that score will help --
the fix would need to be upstream (better score), not downstream (better calibration of a noisy
score). Uses the same counterfactual-alone-component ledgers as the meta-router investigation for
more samples than the tiny combined-greedy trade counts.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402

DEVICE = retest.DEVICE


def score_for(frame: pd.DataFrame, pred_csv, cfg: dict):
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols = bundle["base_cols"]
    pred = pd.read_csv(pred_csv)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError("timestamp mismatch")
    dec_base = parent._to_decisions(pred, oof=False)
    dec, _ = atr_eval._apply_atr_safety_sltp(
        dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=cfg["min_tp"], min_sl=cfg["min_sl"], max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
    atr = atr_eval._atr_pct(frame, cfg["atr_window"])
    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)
    features = sidecar._risk_feature_frame(frame, pred, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = sidecar._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = sidecar._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all))
    z = np.clip((score - pkl["train_score_q50"]) / pkl["train_score_iqr"], -8, 8)
    return dec, score, z, pkl


def prep_val_pred(cname: str, cfg: dict, frame: pd.DataFrame):
    pred = pd.read_csv(valmod.VAL_PRED[cname])
    pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    pred = pred[(pred["timestamp"] >= valmod.START) & (pred["timestamp"] <= valmod.END)].reset_index(drop=True)
    common = frame["timestamp"].isin(pred["timestamp"])
    frame = frame[common].reset_index(drop=True)
    pred = pred[pred["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
    tmp = ROOT / f"tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/_val_{cname}_aligned.csv"
    pred.to_csv(tmp, index=False)
    return frame, tmp


def analyze(window: str, frame: pd.DataFrame, pred_path, cname: str, cfg: dict, fee: float, slip: float):
    dec, score, z, pkl = score_for(frame, pred_path, cfg)
    comp = greedy.prepare_component(frame, pred_path, cfg, DEVICE)
    greedy.PRIORITY = (cname,)
    _, lg = greedy.greedy_replay(frame, {cname: comp}, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=DEVICE)
    if lg.empty:
        print(f"{window} {cname}: no trades")
        return
    lg = lg.copy()
    idx = lg["entry_signal_i"].to_numpy()
    lg["score_at_entry"] = score[idx]
    lg["z_at_entry"] = z[idx]
    lg["margin_at_entry"] = lg["margin_fraction"]
    corr_score = lg["score_at_entry"].corr(lg["trade_return"])
    corr_margin = lg["margin_at_entry"].corr(lg["trade_return"])
    rho, p_val = spearmanr(lg["score_at_entry"], lg["trade_return"])
    # does the sidecar size UP when it should size DOWN? correlate margin with future win
    rho_margin, p_margin = spearmanr(lg["margin_at_entry"], (lg["trade_return"] > 0).astype(float))
    print(f"{window:4s} {cname:8s}: n={len(lg):3d} corr(raw_score,ret)={corr_score:+.3f} "
          f"spearman(score,ret)={rho:+.3f} (p={p_val:.3f})  corr(sized_margin,ret)={corr_margin:+.3f}  "
          f"spearman(margin,win)={rho_margin:+.3f} (p={p_margin:.3f})")


def main() -> int:
    fee, slip = omega._load_fee_slip()

    val_frame = valmod.load_val_frame()
    val_frame_c = val_frame
    val_paths = {}
    for cname, cfg in retest.COMPONENTS.items():
        val_frame_c, tmp = prep_val_pred(cname, cfg, val_frame_c)
        val_paths[cname] = tmp

    print("=== VAL (2025-10..12) ===")
    for cname, cfg in retest.COMPONENTS.items():
        analyze("VAL", val_frame_c, val_paths[cname], cname, cfg, fee, slip)

    print("\n=== OOS (2026-01..06) ===")
    oos_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    for cname, cfg in retest.COMPONENTS.items():
        pred_csv = retest.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"
        analyze("OOS", oos_frame, pred_csv, cname, cfg, fee, slip)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
