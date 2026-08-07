#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combo_metrics,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import (  # noqa: E402
    EVAL_CSV,
    FORBIDDEN_PREFIXES,
    SOURCE_COLS,
    TRAIN_CSV,
    DatasetBundle,
    _apply_norm,
    _audit_frame_contract,
    _directional_features,
    _fit_norm,
    _policy_action,
    _safe_num,
    _train_dsac_offline,
)
from scripts.train_eval_alpha8_mamba_lgbm_dsac_20260529 import (  # noqa: E402
    SEQUENCE_COLS,
    _apply_robust_norm,
    _context_frame,
    _direction_labels,
    _fit_robust_norm,
    _mamba_predict,
    _rolling_sequences,
    _train_mamba,
)


MODEL_ID = "alpha8_parent_dsac_risk_sizing_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TEACHER_COLS = [
    "teacher_long_edge",
    "teacher_short_edge",
    "teacher_side_margin",
    "teacher_side_disagreement",
    "teacher_quantile_skew",
    "teacher_uncertainty",
    "teacher_tail_warning",
]


@dataclass(frozen=True)
class RiskTemplate:
    name: str
    notional: float
    leverage: float
    tp_mult: float
    sl_mult: float
    hold_mult: float
    veto: bool = False


TEMPLATES: tuple[RiskTemplate, ...] = (
    RiskTemplate("veto", 0.0, 1.0, 1.0, 1.0, 1.0, veto=True),
    RiskTemplate("tiny_defensive", 0.25, 1.0, 0.50, 0.50, 0.50),
    RiskTemplate("small_defensive", 0.50, 1.0, 0.75, 0.75, 0.75),
    RiskTemplate("base_light", 1.00, 2.0, 1.00, 1.00, 1.00),
    RiskTemplate("base_fast", 1.00, 3.0, 0.75, 0.75, 0.50),
    RiskTemplate("conviction_balanced", 1.50, 3.0, 1.00, 0.75, 0.75),
    RiskTemplate("conviction_runner", 1.50, 3.0, 1.50, 1.00, 1.00),
    RiskTemplate("aggressive_balanced", 2.00, 5.0, 1.25, 1.00, 1.00),
    RiskTemplate("aggressive_fast", 2.00, 5.0, 0.75, 0.50, 0.50),
    RiskTemplate("max_conviction", 3.00, 5.0, 1.50, 1.00, 1.00),
    RiskTemplate("max_tight", 3.00, 5.0, 0.75, 0.75, 0.50),
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _assert_no_forbidden(df: pd.DataFrame, *, name: str) -> None:
    bad = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")


def _parent_active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != 0) & (side != 0)


def _risk_state_frame(
    frame: pd.DataFrame,
    parent: pd.DataFrame,
    mamba_prob: np.ndarray,
    mamba_emb: np.ndarray,
) -> pd.DataFrame:
    _audit_frame_contract(frame, name="alpha8_risk_state")
    _assert_no_forbidden(frame, name="alpha8_risk_state")
    parts: list[pd.DataFrame] = []
    parts.append(_directional_features(frame).reset_index(drop=True))
    parts.append(pd.DataFrame({c: _safe_num(frame, c) for c in SOURCE_COLS}, index=frame.index).reset_index(drop=True))
    for col in TEACHER_COLS:
        if col not in frame.columns:
            raise RuntimeError(f"certified teacher feature missing: {col}")
    parts.append(pd.DataFrame({c: _safe_num(frame, c) for c in TEACHER_COLS}, index=frame.index).reset_index(drop=True))
    p = pd.DataFrame(index=frame.index)
    for col in [
        "action",
        "side",
        "notional_exposure",
        "leverage",
        "position_fraction",
        "take_profit",
        "stop_loss",
        "max_hold_bars",
        "cooldown_bars",
        "quality_score",
        "confidence",
    ]:
        if col not in parent.columns:
            raise RuntimeError(f"parent decision missing column: {col}")
        p[f"parent_{col}"] = pd.to_numeric(parent[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    p["parent_risk_reward_ratio"] = p["parent_take_profit"] / np.maximum(p["parent_stop_loss"].abs(), 1e-8)
    p["parent_margin_fraction"] = p["parent_notional_exposure"] / np.maximum(p["parent_leverage"], 1e-8)
    parts.append(p.reset_index(drop=True))
    mp = pd.DataFrame(
        {
            "mamba_p_hold": mamba_prob[:, 0],
            "mamba_p_long": mamba_prob[:, 1],
            "mamba_p_short": mamba_prob[:, 2],
            "mamba_dir_edge": mamba_prob[:, 1] - mamba_prob[:, 2],
            "mamba_confidence": np.maximum(mamba_prob[:, 1], mamba_prob[:, 2]) - mamba_prob[:, 0],
        }
    )
    for col in list(mp.columns):
        mp[f"{col}_delta1"] = mp[col].diff().fillna(0.0)
        mp[f"{col}_mean3"] = mp[col].rolling(3, min_periods=1).mean()
    emb_cols = min(16, mamba_emb.shape[1])
    me = pd.DataFrame({f"mamba_emb_{i:02d}": mamba_emb[:, i] for i in range(emb_cols)})
    parts.extend([mp, me])
    out = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate risk state columns: {dup[:20]}")
    return out


def _apply_template(row: pd.Series, template: RiskTemplate) -> pd.Series:
    out = row.copy()
    if template.veto:
        out.loc[["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = [
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0,
        ]
        out.loc["leverage"] = 1.0
        return out
    base_tp = float(row.get("take_profit", 0.0) or 0.0)
    base_sl = float(row.get("stop_loss", 0.0) or 0.0)
    base_hold = int(row.get("max_hold_bars", 0) or 0)
    out.loc["notional_exposure"] = float(template.notional)
    out.loc["leverage"] = float(template.leverage)
    out.loc["position_fraction"] = float(template.notional / max(template.leverage, 1e-8))
    out.loc["take_profit"] = float(max(base_tp, 1e-6) * template.tp_mult)
    out.loc["stop_loss"] = float(max(abs(base_sl), 1e-6) * template.sl_mult)
    out.loc["max_hold_bars"] = int(max(1, round(max(base_hold, 1) * template.hold_mult)))
    return out


def _template_reward(
    close: np.ndarray,
    i: int,
    row: pd.Series,
    template: RiskTemplate,
    *,
    fee: float,
    slip: float,
) -> tuple[float, dict[str, Any]]:
    dec = _apply_template(row, template)
    action = int(dec.get("action", 0) or 0)
    side = int(dec.get("side", 0) or 0)
    if action == 0 or side == 0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0}
    notional = float(dec.get("notional_exposure", 0.0) or 0.0)
    tp = float(dec.get("take_profit", 0.0) or 0.0)
    sl = float(dec.get("stop_loss", 0.0) or 0.0)
    hold = int(dec.get("max_hold_bars", 0) or 0)
    entry_i = min(int(i) + 1, len(close) - 1)
    end = min(len(close), entry_i + max(hold, 1) + 1)
    if notional <= 0.0 or end <= entry_i + 1:
        return 0.0, {"active": 0, "net": 0.0, "win": 0}
    entry = max(float(close[entry_i]), 1e-12)
    fut = close[entry_i + 1 : end]
    path = ((fut / entry) - 1.0) * float(side) * notional
    hit = np.flatnonzero((path >= tp) | (path <= -abs(sl)))
    exit_i = int(hit[0]) if hit.size else len(path) - 1
    gross = float(path[exit_i])
    net = gross - 2.0 * (float(fee) + float(slip)) * notional
    mae = float(np.min(path[: exit_i + 1])) if exit_i >= 0 else 0.0
    win = int(net > 0.0)
    reward = 150.0 * net + (0.35 if win else -0.22)
    reward -= 18.0 * max(0.0, -mae)
    reward -= 0.025 * max(0, hold - 24) / 24.0
    return float(reward), {"active": 1, "net": net, "win": win, "template": template.name}


def _build_template_dataset(
    frame: pd.DataFrame,
    states: np.ndarray,
    parent: pd.DataFrame,
    active_mask: np.ndarray,
    *,
    fee: float,
    slip: float,
) -> tuple[DatasetBundle, dict[str, Any]]:
    close = _safe_num(frame, "close").to_numpy(dtype=np.float64)
    idxs = np.flatnonzero(active_mask)
    s_list: list[np.ndarray] = []
    sp_list: list[np.ndarray] = []
    a_list: list[int] = []
    r_list: list[float] = []
    d_list: list[float] = []
    net_stats: dict[str, list[float]] = {t.name: [] for t in TEMPLATES}
    win_stats: dict[str, list[int]] = {t.name: [] for t in TEMPLATES}
    for i in idxs:
        next_i = min(int(i) + 1, len(states) - 1)
        for action_id, template in enumerate(TEMPLATES):
            reward, meta = _template_reward(close, int(i), parent.iloc[int(i)], template, fee=fee, slip=slip)
            s_list.append(states[int(i)])
            sp_list.append(states[next_i])
            a_list.append(int(action_id))
            r_list.append(float(reward))
            d_list.append(0.0)
            if int(meta["active"]) == 1:
                net_stats[template.name].append(float(meta["net"]))
                win_stats[template.name].append(int(meta["win"]))
    rewards_np = np.asarray(r_list, dtype=np.float32)
    scale = float(np.nanstd(rewards_np))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    rewards_np = np.clip(rewards_np / scale, -8.0, 8.0).astype(np.float32)
    return (
        DatasetBundle(
            states=np.asarray(s_list, dtype=np.float32),
            next_states=np.asarray(sp_list, dtype=np.float32),
            actions=np.asarray(a_list, dtype=np.int64),
            rewards=rewards_np,
            dones=np.asarray(d_list, dtype=np.float32),
        ),
        {
            "active_parent_rows": int(len(idxs)),
            "reward_scale": scale,
            "template_net_mean": {k: float(np.mean(v)) if v else 0.0 for k, v in net_stats.items()},
            "template_win_rate": {k: float(np.mean(v)) if v else 0.0 for k, v in win_stats.items()},
            "template_active_count": {k: len(v) for k, v in net_stats.items()},
        },
    )


def _compose_risk_decisions(parent: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
    out = parent.copy().reset_index(drop=True)
    active = _parent_active(out)
    for i in np.flatnonzero(active):
        action_id = int(actions[int(i)])
        if action_id < 0 or action_id >= len(TEMPLATES):
            raise RuntimeError(f"invalid risk template id: {action_id}")
        out.iloc[int(i)] = _apply_template(out.iloc[int(i)], TEMPLATES[action_id])
    inactive = ~active
    out.loc[inactive, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = [
        0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
    ]
    out.loc[inactive, "leverage"] = 1.0
    return out


def _usage(actions: np.ndarray, active_mask: np.ndarray) -> dict[str, int]:
    out = {t.name: 0 for t in TEMPLATES}
    for a in actions[np.asarray(active_mask, dtype=bool)]:
        out[TEMPLATES[int(a)].name] += 1
    return out


def _pick(grid: pd.DataFrame, split: str, variant: str, cost: str) -> dict[str, Any]:
    row = grid[(grid["split"].eq(split)) & (grid["variant"].eq(variant)) & (grid["cost"].eq(cost))]
    return {} if row.empty else row.iloc[0].to_dict()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mamba-epochs", type=int, default=3)
    p.add_argument("--mamba-batch-size", type=int, default=768)
    p.add_argument("--mamba-d-model", type=int, default=96)
    p.add_argument("--mamba-emb-dim", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--label-horizon", type=int, default=12)
    p.add_argument("--label-barrier", type=float, default=0.0025)
    p.add_argument("--dsac-steps", type=int, default=2500)
    p.add_argument("--dsac-batch-size", type=int, default=768)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Alpha8 risk/sizing Mamba requires CUDA.")
    device = torch.device("cuda")
    _seed_everything(290529)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(EVAL_CSV))
    _assert_no_forbidden(train_all, name="train_all")
    _assert_no_forbidden(eval_df, name="eval")
    _audit_frame_contract(train_all, name="train_all")
    _audit_frame_contract(eval_df, name="eval")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    ctx_train = _context_frame(train_df)
    ctx_val = _context_frame(val_df)
    ctx_eval = _context_frame(eval_df)
    ctx_norm = _fit_robust_norm(ctx_train, SEQUENCE_COLS)
    seq_train = _rolling_sequences(_apply_robust_norm(ctx_train, ctx_norm), args.seq_len)
    seq_val = _rolling_sequences(_apply_robust_norm(ctx_val, ctx_norm), args.seq_len)
    seq_eval = _rolling_sequences(_apply_robust_norm(ctx_eval, ctx_norm), args.seq_len)
    y_train = _direction_labels(train_df, horizon=args.label_horizon, barrier=args.label_barrier)
    print(
        json.dumps(
            {
                "stage": "alpha8_risk_start",
                "device": str(device),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "oos_rows": len(eval_df),
                "template_count": len(TEMPLATES),
                "label_counts_train": np.bincount(y_train, minlength=3).astype(int).tolist(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    mamba = _train_mamba(
        seq_train,
        y_train,
        device=device,
        epochs=args.mamba_epochs,
        batch_size=args.mamba_batch_size,
        d_model=args.mamba_d_model,
        emb_dim=args.mamba_emb_dim,
    )
    m_train_p, m_train_e = _mamba_predict(mamba.model, seq_train, device=device, batch_size=args.mamba_batch_size)
    m_val_p, m_val_e = _mamba_predict(mamba.model, seq_val, device=device, batch_size=args.mamba_batch_size)
    m_eval_p, m_eval_e = _mamba_predict(mamba.model, seq_eval, device=device, batch_size=args.mamba_batch_size)

    primary_parent = joblib.load(baseline.primary_parent)
    fallback_parent = joblib.load(baseline.fallback_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)
    p_train = _predict_scaled(primary_parent, train_df, primary_rt)
    p_val = _predict_scaled(primary_parent, val_df, primary_rt)
    p_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    f_train = _predict_scaled(fallback_parent, train_df, fallback_rt)
    f_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    f_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)

    s_train = _risk_state_frame(train_df, p_train, m_train_p, m_train_e)
    s_val = _risk_state_frame(val_df, p_val, m_val_p, m_val_e)
    s_eval = _risk_state_frame(eval_df, p_eval, m_eval_p, m_eval_e)
    norm = _fit_norm(s_train)
    x_train = _apply_norm(s_train, norm)
    x_val = _apply_norm(s_val, norm)
    x_eval = _apply_norm(s_eval, norm)
    active_train = _parent_active(p_train)
    active_val = _parent_active(p_val)
    active_eval = _parent_active(p_eval)

    fee = 0.0005
    slip = 0.0002
    data, data_diag = _build_template_dataset(train_df, x_train, p_train, active_train, fee=fee, slip=slip)
    print(
        json.dumps(
            {
                "stage": "dsac_risk_start",
                "state_dim": int(x_train.shape[1]),
                "samples": int(len(data.states)),
                "dataset_diagnostics": data_diag,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    trained = _train_dsac_offline(
        data,
        state_dim=int(x_train.shape[1]),
        action_dim=len(TEMPLATES),
        device=device,
        steps=args.dsac_steps,
        batch_size=args.dsac_batch_size,
    )
    actor: nn.Module = trained["actor"]
    a_train = _policy_action(actor, x_train, device=device)
    a_val = _policy_action(actor, x_val, device=device)
    a_eval = _policy_action(actor, x_eval, device=device)

    risk_train = _compose_risk_decisions(p_train, a_train)
    risk_val = _compose_risk_decisions(p_val, a_val)
    risk_eval = _compose_risk_decisions(p_eval, a_eval)
    combo_train = _combine_primary_fallback(p_train, f_train)
    combo_val = _combine_primary_fallback(p_val, f_val)
    combo_eval = _combine_primary_fallback(p_eval, f_eval)

    rows: list[dict[str, Any]] = []
    for split, df, primary_dec, combo_dec, risk_dec in [
        ("train", train_df, p_train, combo_train, risk_train),
        ("val", val_df, p_val, combo_val, risk_val),
        ("oos", eval_df, p_eval, combo_eval, risk_eval),
    ]:
        for name, dec in [
            ("primary_parent", primary_dec),
            ("baseline_combo", combo_dec),
            ("alpha8_parent_dsac_risk", risk_dec),
        ]:
            for cost, vals in _combo_metrics(df, dec).items():
                rows.append({"split": split, "variant": name, "cost": cost, **vals})
    grid = pd.DataFrame(rows)
    grid_path = OUT_DIR / "grid.csv"
    grid.to_csv(grid_path, index=False)

    ckpt_path = OUT_DIR / "alpha8_parent_dsac_risk.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "mamba_state_dict": mamba.model.state_dict(),
            "dsac_actor_state_dict": actor.state_dict(),
            "state_dim": int(x_train.shape[1]),
            "action_dim": len(TEMPLATES),
            "state_columns": list(norm["columns"]),
            "state_normalizer": norm,
            "context_normalizer": ctx_norm,
            "sequence_cols": SEQUENCE_COLS,
            "templates": [t.__dict__ for t in TEMPLATES],
        },
        ckpt_path,
    )
    (OUT_DIR / "state_columns.json").write_text(json.dumps(list(norm["columns"]), indent=2) + "\n")
    (OUT_DIR / "templates.json").write_text(json.dumps([t.__dict__ for t in TEMPLATES], indent=2) + "\n")

    summary = {
        "model_id": MODEL_ID,
        "design": "Alpha7 Primary Parent owns direction; Mamba provides sequence context; DSAC selects veto/risk/sizing templates only.",
        "live_wired": False,
        "baseline_model_id": baseline.model_id,
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "allowed_regime_surfaces": ["clean_regime4_state24_sticky090_v2_*", "regime4_pred_*"],
        "teacher_features": TEACHER_COLS,
        "mamba": {
            "seq_len": int(args.seq_len),
            "d_model": int(args.mamba_d_model),
            "embedding_dim": int(args.mamba_emb_dim),
            "epochs": int(args.mamba_epochs),
            "train_diag": mamba.train_diag,
        },
        "dsac": {
            "state_dim": int(x_train.shape[1]),
            "action_dim": len(TEMPLATES),
            "steps": int(args.dsac_steps),
            "batch_size": int(args.dsac_batch_size),
            "templates": [t.__dict__ for t in TEMPLATES],
            "dataset_diagnostics": data_diag,
            "train_diag": trained["train_diag"],
            "action_usage": {
                "train": _usage(a_train, active_train),
                "val": _usage(a_val, active_val),
                "oos": _usage(a_eval, active_eval),
            },
            "active_rows": {
                "train": int(np.sum(active_train)),
                "val": int(np.sum(active_val)),
                "oos": int(np.sum(active_eval)),
            },
        },
        "cost3": {
            "val_primary": _pick(grid, "val", "primary_parent", "cost3"),
            "val_combo": _pick(grid, "val", "baseline_combo", "cost3"),
            "val_alpha8": _pick(grid, "val", "alpha8_parent_dsac_risk", "cost3"),
            "oos_primary": _pick(grid, "oos", "primary_parent", "cost3"),
            "oos_combo": _pick(grid, "oos", "baseline_combo", "cost3"),
            "oos_alpha8": _pick(grid, "oos", "alpha8_parent_dsac_risk", "cost3"),
        },
        "artifacts": {
            "grid": str(grid_path),
            "ckpt": str(ckpt_path),
            "state_columns": str(OUT_DIR / "state_columns.json"),
            "templates": str(OUT_DIR / "templates.json"),
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), "cost3": summary["cost3"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
