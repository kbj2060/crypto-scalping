#!/usr/bin/env python3
"""RESEARCH ONLY -- measures the axis every prior exit-head round left unmeasured: does changing
the EXIT label also change ENTRY decisions through the shared TabM trunk?

Background. `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py` trains one
ThreeHeadTabM per expert with all three heads (direction/quality/exit) on a SHARED encoder,
optimizing loss_dir + w_q*loss_qual + w_e*loss_exit over model.parameters() with nothing frozen.
So the 12 exit-label retrain bundles from 2026-07-21 have different direction/quality weights too.
Every evaluation of them so far (research_eth_omega461_exit_head_retrain_eval_20260721.py, rounds
12 and 15) deliberately sourced entry decisions from the FROZEN prediction CSVs and injected the
new bundle only into the exit-probability path -- which is why those runs came back byte-identical
to the control. Their retrained direction/quality weights were never called.

This script regenerates the prediction source from each retrained bundle itself (same code path
the training script uses to write validation_predictions_qXXX.csv / oos_predictions_qXXX.csv:
_base_input -> _predict_payload per expert -> _route_id -> _routed -> _prediction_output), rebuilds
decisions, ATR TP/SL, risk-sidecar features and margin/leverage from THAT source, and replays at
the live EXIT_THRESHOLD=0.95. Any difference from the control is then attributable to the entry
side, which is exactly what was never measured.

Windows: VAL 2025-10-01..12-31, OOS 2026-01-01..03-31, inherited from
research_eth_omega461_exit_sweep_20260721.py. The parent training split is SPLIT_TS=2025-10-01
(train_eval_omega1_2_tabm_diffusion_risk_20260603.py:34), so VAL sits outside the retrain bundles'
own training data -- the internal 85/15 early-stopping split happens inside train_df only.

VAL-first funnel: all 12 variants are scored on VAL; OOS is touched only for VAL winners
(beats the control on BOTH pnl and mdd).

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
No retraining is performed (reuses the 20260721 checkpoints already on disk). Research artifact
only -- no promotion-gate claim.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as base  # noqa: E402
import research_eth_omega461_joint_threshold_retrain_20260722 as r12  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

PRED_PREFIX = "omega1_regime3_expertdq_oof"
OUT_DIR = ROOT / "tmp/research_20260727/retrain_entry_from_bundle_20260727"


def frozen_src(cfg: dict, split: str) -> pd.DataFrame:
    """The frozen prediction CSV every prior round used as the entry source."""
    pred = base.EXT_PRED_DIR / cfg["_component"] / f"{'validation' if split == 'VAL' else 'oos'}_predictions_{cfg['q_tag']}.csv"
    src = pd.read_csv(pred)
    for c in src.columns:
        if str(src[c].dtype).lower().startswith("str"):
            src[c] = src[c].astype(object)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    return src


@torch.no_grad()
def bundle_src(bundle_path: Path, frame: pd.DataFrame, *, quality_threshold: float, split: str) -> pd.DataFrame:
    """Regenerate the prediction source FROM the bundle, mirroring the training script's own
    predict_raw/_routed/_prediction_output path (lines 1077-1103 of the 20260620 trainer)."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    base_cols = list(bundle["base_cols"])
    models = dict(bundle["models"])
    x = parent._base_input(frame, base_cols)
    preds = {expert: parent._predict_payload(models[expert], x, device=base.DEVICE) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    src = parent._prediction_output(frame, direction, quality, threshold=float(quality_threshold), prefix=PRED_PREFIX)
    if split != "VAL":
        # OOS CSVs drop the _oof_ infix (trainer line 1103); keep the same convention so
        # _to_decisions(oof=False) resolves the same columns it would on the frozen artifact.
        src = src.rename(columns={c: c.replace(f"{PRED_PREFIX}_", "omega1_regime3_expertdq_") for c in src.columns})
    for c in src.columns:
        if str(src[c].dtype).lower().startswith("str"):
            src[c] = src[c].astype(object)
    return src


def prep_from_src(cfg: dict, frame: pd.DataFrame, src: pd.DataFrame, bundle_path: Path, *, oof: bool) -> dict[str, Any]:
    """base.prep_component, but taking an in-memory prediction source and an explicit bundle
    instead of reading the frozen CSV. Same ordering and same frozen sidecar."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    base_cols = bundle["base_cols"]
    models = bundle["models"]

    if len(src) != len(frame) or not src["timestamp"].reset_index(drop=True).equals(frame["timestamp"].reset_index(drop=True)):
        raise RuntimeError("prediction/frame timestamp mismatch")

    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(src, oof=oof)
    dec, _atr_diag = atr_eval._apply_atr_safety_sltp(
        dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=cfg["min_tp"], min_sl=cfg["min_sl"], max_tp=cfg["max_tp"], max_sl=cfg["max_sl"],
    )
    atr_pct = atr_eval._atr_pct(frame, cfg["atr_window"])
    fee, slip = omega._load_fee_slip()
    loaded = parent._load_payloads(models, device=base.DEVICE)

    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)

    features = rs._risk_feature_frame(frame, src, dec, base_cols, atr_pct=atr_pct, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = rs._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = rs._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)

    mapping = pkl["selected_mapping"]
    margin = rs._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"],
                              **{k: mapping[k] for k in rs.MARGIN_CFG_KEYS})
    leverage = None
    if pkl["dynamic_leverage"]:
        leverage = rs._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"],
                                     **{k: mapping[k] for k in rs.LEVERAGE_CFG_KEYS})

    return dict(frame=frame, x=x, dec=dec, loaded=loaded, margin=margin, leverage=leverage,
                fee=fee, slip=slip, notional_scaled_sltp=pkl["notional_scaled_sltp"])


def load_split(split: str) -> pd.DataFrame:
    if split == "VAL":
        return base.load_frame(base.VAL_START, base.VAL_END, base_csv=base.BASE_2025, wide24_csv=base.WIDE24_2025)
    return base.load_frame(base.OOS_START, base.OOS_END, base_csv=base.BASE_2026, wide24_csv=base.WIDE24_2026)


def aligned_frame(frame: pd.DataFrame, src: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = set(src["timestamp"])
    f = frame[frame["timestamp"].isin(keep)].reset_index(drop=True)
    s = src[src["timestamp"].isin(set(f["timestamp"]))].reset_index(drop=True)
    return f, s


def replay(prepped: dict[str, Any]) -> dict[str, Any]:
    m, _ledger = base.replay_exit_variant(
        prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        exit_threshold=base.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=base.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=base.DEVICE,
    )
    return m


def run_component(cname: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = dict(base.COMPONENTS[cname])
    cfg["_component"] = cname
    frame_full = load_split(split)
    fsrc = frozen_src(cfg, split)
    frame, fsrc = aligned_frame(frame_full, fsrc)
    oof = split == "VAL"

    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    # --- Control A: frozen bundle + frozen prediction CSV (what every prior round scored). ------
    p_ctrl = prep_from_src(cfg, frame, fsrc, cfg["bundle"], oof=oof)
    m_ctrl = replay(p_ctrl)
    side_frozen = pd.to_numeric(p_ctrl["dec"]["side"], errors="raise").to_numpy(dtype=np.int64)
    rows.append({"variant": "control_frozen_csv_entry", "component": cname, "split": split,
                 "entry_source": "frozen_csv", **m_ctrl, "exit_reasons": json.dumps(m_ctrl["exit_reasons"]),
                 "side_diff_rows": 0, "side_diff_frac": 0.0})

    # --- Control B: frozen bundle, entry REGENERATED from that same bundle. --------------------
    # If the regeneration path is faithful this must reproduce Control A. Any gap here is a
    # harness artifact and would invalidate the variant comparisons below, so it is checked
    # before anything else is interpreted.
    rsrc = bundle_src(cfg["bundle"], frame, quality_threshold=cfg["quality_threshold"], split=split)
    p_regen = prep_from_src(cfg, frame, rsrc, cfg["bundle"], oof=oof)
    m_regen = replay(p_regen)
    side_regen = pd.to_numeric(p_regen["dec"]["side"], errors="raise").to_numpy(dtype=np.int64)
    checks.append({"check": "regen_reproduces_frozen_csv", "component": cname, "split": split,
                   "pnl_frozen": m_ctrl["pnl"], "pnl_regen": m_regen["pnl"],
                   "mdd_frozen": m_ctrl["mdd"], "mdd_regen": m_regen["mdd"],
                   "trades_frozen": m_ctrl["trades"], "trades_regen": m_regen["trades"],
                   "side_diff_rows": int((side_regen != side_frozen).sum()),
                   "pnl_close": bool(abs(m_ctrl["pnl"] - m_regen["pnl"]) < 0.01)})
    rows.append({"variant": "control_regen_entry", "component": cname, "split": split,
                 "entry_source": "frozen_bundle", **m_regen, "exit_reasons": json.dumps(m_regen["exit_reasons"]),
                 "side_diff_rows": int((side_regen != side_frozen).sum()),
                 "side_diff_frac": float((side_regen != side_frozen).mean())})

    # --- Variants: retrained bundle drives BOTH entry and exit. -------------------------------
    for name, overrides in r12.VARIANTS.items():
        if cname not in overrides:
            continue
        bpath = overrides[cname]
        vsrc = bundle_src(bpath, frame, quality_threshold=cfg["quality_threshold"], split=split)
        p_var = prep_from_src(cfg, frame, vsrc, bpath, oof=oof)
        m_var = replay(p_var)
        side_var = pd.to_numeric(p_var["dec"]["side"], errors="raise").to_numpy(dtype=np.int64)
        rows.append({"variant": name, "component": cname, "split": split, "entry_source": "retrained_bundle",
                     **m_var, "exit_reasons": json.dumps(m_var["exit_reasons"]),
                     "side_diff_rows": int((side_var != side_frozen).sum()),
                     "side_diff_frac": float((side_var != side_frozen).mean())})
        print(f"  {name:24s} pnl={m_var['pnl']:8.3f} mdd={m_var['mdd']:8.3f} trades={m_var['trades']:3d} "
              f"side_diff={int((side_var != side_frozen).sum()):5d}/{len(side_var)}", flush=True)

    return pd.DataFrame(rows), pd.DataFrame(checks)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "component", "split", "entry_source", "pnl", "mdd", "trades", "wr",
            "side_diff_rows", "side_diff_frac"]

    val_rows, val_checks = [], []
    for cname in ("h48qual", "zig075"):
        print(f"stage=val component={cname}", flush=True)
        r, c = run_component(cname, "VAL")
        val_rows.append(r)
        val_checks.append(c)
    val = pd.concat(val_rows, ignore_index=True)
    chk = pd.concat(val_checks, ignore_index=True)
    print(chk.to_string(index=False), flush=True)
    print(val[cols].to_string(index=False), flush=True)
    val.to_csv(OUT_DIR / "entry_from_bundle_VAL.csv", index=False)
    chk.to_csv(OUT_DIR / "sanity_regen_reproduces_frozen.csv", index=False)

    # VAL-first funnel: OOS only for variants beating their component's control on both axes.
    winners: list[tuple[str, str]] = []
    for cname in ("h48qual", "zig075"):
        sub = val[val["component"] == cname]
        ctrl = sub[sub["variant"] == "control_frozen_csv_entry"].iloc[0]
        for _, row in sub[sub["entry_source"] == "retrained_bundle"].iterrows():
            if row["pnl"] > ctrl["pnl"] and row["mdd"] >= ctrl["mdd"] - 1e-9:
                winners.append((cname, row["variant"]))
    print(f"stage=val_winners n={len(winners)} {winners}", flush=True)
    pd.DataFrame(winners, columns=["component", "variant"]).to_csv(OUT_DIR / "val_winners.csv", index=False)

    if not winners:
        print("stage=done no VAL winners -- OOS not touched", flush=True)
        return 0

    oos_rows = []
    for cname in sorted({c for c, _ in winners}):
        print(f"stage=oos component={cname}", flush=True)
        r, _c = run_component(cname, "OOS")
        keep = {v for c, v in winners if c == cname} | {"control_frozen_csv_entry", "control_regen_entry"}
        oos_rows.append(r[r["variant"].isin(keep)])
    oos = pd.concat(oos_rows, ignore_index=True)
    print(oos[cols].to_string(index=False), flush=True)
    oos.to_csv(OUT_DIR / "entry_from_bundle_OOS.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
