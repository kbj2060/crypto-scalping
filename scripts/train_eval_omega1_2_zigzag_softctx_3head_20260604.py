#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as th  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_zigzag_softctx_3head_20260604"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"

TP = 0.026
SL = 0.014
THR_MAP = {"bull": 0.72, "bear": 0.64, "chop": 0.65, "chop_expert": 0.65}
SCALE_MAP = {"bull": 0.65, "bear": 0.90, "chop_expert": 0.90}
BASE_SCALES = {"bull": 0.75, "bear": 0.90, "chop_expert": 0.90}
GLOBAL_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _read_soft_labels(year: int) -> pd.DataFrame:
    path = LABEL_DIR / f"zigzag_action_labels_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    cols = ["timestamp", "zigzag_action", "zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"]
    labels = pd.read_csv(path, usecols=cols, parse_dates=["timestamp"])
    labels = labels.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    return labels


def _join_soft(frame: pd.DataFrame, *, year: int, name: str) -> pd.DataFrame:
    out = frame.drop(columns=[c for c in ["zigzag_action"] if c in frame.columns]).merge(_read_soft_labels(year), on="timestamp", how="left", validate="one_to_one")
    missing = int(out["zigzag_action"].isna().sum())
    if missing:
        raise RuntimeError(f"{name} missing zigzag soft labels: {missing}")
    soft = out[["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"]].to_numpy(dtype=np.float64)
    if not np.isfinite(soft).all():
        raise RuntimeError(f"{name} non-finite zigzag soft labels")
    row_sum = soft.sum(axis=1)
    if float(np.max(np.abs(row_sum - 1.0))) > 1e-4:
        raise RuntimeError(f"{name} zigzag soft labels do not sum to 1")
    out["zigzag_action"] = pd.to_numeric(out["zigzag_action"], errors="raise").astype(np.int64)
    return out


def _context_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    high = pd.to_numeric(frame["high"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    low = pd.to_numeric(frame["low"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out: dict[str, np.ndarray] = {}
    for n in (1, 2, 3, 6, 12, 24, 48):
        out[f"zzctx_ret_{n}"] = close.pct_change(n).fillna(0.0).to_numpy(dtype=np.float32)
    for n in (6, 12, 24, 48):
        out[f"zzctx_vol_{n}"] = close.pct_change().rolling(n).std().fillna(0.0).to_numpy(dtype=np.float32)
        out[f"zzctx_range_{n}"] = ((high.rolling(n).max() - low.rolling(n).min()) / close.replace(0, np.nan)).fillna(0.0).to_numpy(dtype=np.float32)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    out["zzctx_ema9_21"] = ((ema9 - ema21) / close.replace(0, np.nan)).fillna(0.0).to_numpy(dtype=np.float32)
    out["zzctx_ema21_55"] = ((ema21 - ema55) / close.replace(0, np.nan)).fillna(0.0).to_numpy(dtype=np.float32)
    out["zzctx_close_pos_24"] = ((close - low.rolling(24).min()) / (high.rolling(24).max() - low.rolling(24).min()).replace(0, np.nan)).fillna(0.5).to_numpy(dtype=np.float32)
    x = pd.DataFrame(out, index=frame.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.astype(np.float32)


def _input(frame: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    base = th._base_input(frame, base_cols)
    ctx = _context_features(frame)
    out = pd.concat([base.reset_index(drop=True), ctx.reset_index(drop=True)], axis=1)
    if out.columns.duplicated().any():
        raise RuntimeError("duplicate softctx input columns")
    return out.astype(np.float32)


def _fit_soft_expert(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_soft: np.ndarray,
    route_frame: pd.DataFrame,
    x_exit: pd.DataFrame,
    y_exit: np.ndarray,
    exit_route_frame: pd.DataFrame,
    *,
    expert_idx: int,
    seed: int,
    epochs: int,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = th._standardize_fit(x_all)
    x_dir_np = th._standardize_apply(x_dir, scaler)
    x_exit_np = th._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_soft_np = np.asarray(y_soft, dtype=np.float32)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = th._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = th._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    class_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32)
    soft_conf = np.max(y_soft_np, axis=1).astype(np.float32)
    dir_w = class_w * route_w * (0.45 + soft_conf)
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid softctx weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)
    model = th.ThreeHeadTabM(x_dir_np.shape[1], cfg=th.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(th.CFG.lr), weight_decay=float(th.CFG.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(y_soft_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(th.CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(th.CFG.batch_size), shuffle=True, drop_last=False)
    best_state = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, ysb, wb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, ysb, wb = xb.to(device), yb.to(device), ysb.to(device), wb.to(device)
            xe, ye, we = xe.to(device), ye.to(device), we.to(device)
            out_dir = model(xb)
            logits_dir = out_dir["direction"]
            logits_quality = out_dir["quality"]
            hard_loss_k = torch.nn.functional.cross_entropy(
                logits_dir.reshape(-1, 3),
                yb[:, None].expand(-1, int(th.CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(th.CFG.k))
            soft_target = ysb[:, None, :].expand(-1, int(th.CFG.k), -1)
            soft_dir_k = -(soft_target * torch.nn.functional.log_softmax(logits_dir, dim=-1)).sum(dim=-1)
            soft_qual_k = -(soft_target * torch.nn.functional.log_softmax(logits_quality, dim=-1)).sum(dim=-1)
            out_exit = model(xe)
            exit_loss_k = torch.nn.functional.cross_entropy(
                out_exit["exit"].reshape(-1, 2),
                ye[:, None].expand(-1, int(th.CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(th.CFG.k))
            loss_hard = (hard_loss_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_soft = (soft_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (soft_qual_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_exit = (exit_loss_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = 0.70 * loss_hard + 0.45 * loss_soft + 0.85 * loss_qual + float(th.CFG.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vys = torch.from_numpy(y_soft_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            vhard = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(th.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(th.CFG.k))
            vsoft_target = vys[:, None, :].expand(-1, int(th.CFG.k), -1)
            vsoft = -(vsoft_target * torch.nn.functional.log_softmax(vo["direction"], dim=-1)).sum(dim=-1)
            vqual = -(vsoft_target * torch.nn.functional.log_softmax(vo["quality"], dim=-1)).sum(dim=-1)
            veo = model(ve)
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(th.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(th.CFG.k))
            vloss = float(
                (
                    0.70 * ((vhard.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + 0.45 * ((vsoft.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + 0.85 * ((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + float(th.CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))
                )
                .detach()
                .cpu()
            )
        if vloss + 1e-6 < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(th.CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "config": th.CFG.__dict__,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_dir_np.shape[1]),
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
        "input_columns": list(x_dir.columns),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)
    return payload


def _set_thresholds(src: pd.DataFrame, prefix: str, *, global_threshold: float | None = None) -> pd.DataFrame:
    out = src.copy()
    q = pd.to_numeric(out[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    action = pd.to_numeric(out[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    if global_threshold is None:
        expert = out[f"{prefix}router_expert"].astype(str).replace({"chop": "chop_expert"}).to_numpy()
        thr = np.asarray([THR_MAP.get(x, THR_MAP["chop_expert"]) for x in expert], dtype=np.float64)
    else:
        thr = np.full(len(out), float(global_threshold), dtype=np.float64)
    out[f"{prefix}quality_threshold"] = thr
    out[f"{prefix}final_action"] = np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)
    return out


def _scale_dec(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy()
    active = omega._active(out)
    for expert, scale in SCALE_MAP.items():
        mask = active & out["router_expert"].astype(str).eq(expert)
        ratio = float(scale) / float(BASE_SCALES[expert])
        out.loc[mask, "notional_exposure"] = pd.to_numeric(out.loc[mask, "notional_exposure"], errors="raise") * ratio
        out.loc[mask, "position_fraction"] = pd.to_numeric(out.loc[mask, "position_fraction"], errors="raise") * ratio
    active = omega._active(out)
    out.loc[active, "take_profit"] = TP
    out.loc[active, "stop_loss"] = SL
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _decisions(src: pd.DataFrame, prefix: str, *, oof: bool, global_threshold: float | None = None) -> pd.DataFrame:
    return _scale_dec(omega._to_fixed_decisions(_set_thresholds(src, prefix, global_threshold=global_threshold), oof=oof))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--max-exit-samples", type=int, default=30000)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260604)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    device = _device(args.device)
    th._seed_everything(int(args.seed))
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix.strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = th._prepare_frames(disable_tp_sl=False)
    train_raw = _join_soft(frames["train_raw"], year=2025, name="train")
    val_raw = _join_soft(frames["val_raw"], year=2025, name="validation")
    oos_raw = _join_soft(frames["oos_raw"], year=2026, name="oos")
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    x_train = _input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    y_soft = train_raw[["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"]].to_numpy(dtype=np.float32)
    x_exit_raw, y_exit, frame_exit, exit_diag = exit_head._build_exit_dataset_independent(
        frames["train_df"],
        frames["s_train_label"],
        frames["train_fixed"],
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        exit_edge_min=float(args.exit_edge_min),
        hold_offsets=[1, 2, 3, 6, 12, 24, 48, 96, 192, 384],
        max_samples=int(args.max_exit_samples),
    )
    x_exit = pd.concat([th._exit_input_from_position_rows(x_exit_raw, base_cols).reset_index(drop=True), _context_features(frame_exit).reset_index(drop=True)], axis=1).astype(np.float32)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = _fit_soft_expert(
            x_train,
            y_train,
            y_soft,
            train_raw,
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_softctx_3head_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {"epochs_ran": int(payload["epochs_ran"]), "best_validation_loss": float(payload["best_validation_loss"])}

    def predict(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        x = _input(frame, base_cols)
        preds = {expert: th._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = th._routed(preds, route, "direction", 3)
        quality = th._routed(preds, route, "quality", 3)
        return th._prediction_output(frame, direction, quality, threshold=0.50, prefix=prefix.rstrip("_"))

    val_src = predict(val_raw, "omega1_regime3_expertdq_oof_")
    oos_src_oof = predict(oos_raw, "omega1_regime3_expertdq_oof_")
    oos_src = oos_src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof.columns})
    rows: list[dict[str, Any]] = []
    configs: list[tuple[str, float | None]] = [("expert_thresholds", None)] + [(f"global_{t:.2f}", t) for t in GLOBAL_THRESHOLDS]
    y_val = val_raw["zigzag_action"].to_numpy(dtype=np.int64)
    y_oos = oos_raw["zigzag_action"].to_numpy(dtype=np.int64)
    for name, thr in configs:
        val_dec = _decisions(val_src, "omega1_regime3_expertdq_oof_", oof=True, global_threshold=thr)
        oos_dec = _decisions(oos_src, "omega1_regime3_expertdq_", oof=False, global_threshold=thr)
        val = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        val_pred = pd.to_numeric(val_src["omega1_regime3_expertdq_oof_dir_action"], errors="raise").to_numpy(dtype=np.int64)
        oos_pred = pd.to_numeric(oos_src["omega1_regime3_expertdq_dir_action"], errors="raise").to_numpy(dtype=np.int64)
        rows.append(
            {
                "variant": name,
                "threshold": -1.0 if thr is None else float(thr),
                "val_pnl": val["pnl"],
                "val_mdd": val["mdd"],
                "val_wr": val["wr"],
                "val_trades": val["trades"],
                "oos_pnl": oos["pnl"],
                "oos_mdd": oos["mdd"],
                "oos_wr": oos["wr"],
                "oos_trades": oos["trades"],
                "val_dir_acc": float(np.mean(val_pred == y_val)),
                "oos_dir_acc": float(np.mean(oos_pred == y_oos)),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["val_pnl", "val_wr"], ascending=False)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    val_src.to_csv(out_dir / "validation_predictions_2025_softctx_true3head.csv", index=False)
    oos_src.to_csv(out_dir / "oos_predictions_2026_softctx_true3head.csv", index=False)
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": th.POS_COLS, "context_features": list(_context_features(train_raw).columns), "config": th.CFG.__dict__}, out_dir / "softctx_3head_tabm_bundle.pt")
    report = {
        "model_id": MODEL_ID,
        "design": "Per-expert Omega1.2 3-head TabM retrained to learn ZigZag soft labels. Direction uses hard+soft CE; Quality uses soft ZigZag distribution; added current-bar time-series context features.",
        "label_contract": str(LABEL_DIR),
        "input_contract": {"base_features": len(base_cols), "context_features": list(_context_features(train_raw).columns), "total_features": int(x_train.shape[1])},
        "summaries": summaries,
        "exit_label": {"exit_edge_min": float(args.exit_edge_min), "diag": exit_diag},
        "ranking": rows,
        "artifacts": {"out_dir": str(out_dir), "ranking": str(out_dir / "ranking.csv")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": str(out_dir / "ranking.csv")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
