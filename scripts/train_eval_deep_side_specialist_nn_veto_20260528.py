#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default  # noqa: E402


MODEL_ID = "deep_side_specialist_nn_veto_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
MODEL_OUT = OUT_DIR / "side_specialist_nn_veto.pt"
TRAIN_CANDIDATES_OUT = OUT_DIR / "train_candidates.csv"
OOS_LEDGER_OUT = OUT_DIR / "oos_cost3_ledger.csv"

Q_FEATURES = ["q_long", "q_short", "q_edge", "q_margin", "q_long_share", "q_short_share"]
BASE_FEATURES = [
    "tp_sl_action_score",
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "ai_flow_pressure",
    "volume",
    "quote_volume",
    "cvp_regime",
    "regime_trending",
]
PREFIXES = ("clean_regime4_state24_sticky090_v2_", "regime4_pred_")


@dataclass(frozen=True)
class SelectedThresholds:
    long: float
    short: float


class SideSpecialistNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.12) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.long_head = nn.Linear(hidden, 1)
        self.short_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        return self.long_head(z).squeeze(-1), self.short_head(z).squeeze(-1)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = list(Q_FEATURES)
    cols.extend([c for c in BASE_FEATURES if c in df.columns])
    cols.extend([c for c in df.columns if str(c).startswith(PREFIXES)])
    out: list[str] = []
    seen: set[str] = set()
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _feature_frame(df: pd.DataFrame, q: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ql = q[:, 0].astype(float)
    qs = q[:, 1].astype(float)
    denom = np.maximum(np.abs(ql) + np.abs(qs), 1e-12)
    q_map = {
        "q_long": ql,
        "q_short": qs,
        "q_edge": np.maximum(ql, qs),
        "q_margin": np.abs(ql - qs),
        "q_long_share": ql / denom,
        "q_short_share": qs / denom,
    }
    for col in feature_cols:
        if col in q_map:
            out[col] = q_map[col]
        elif col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise RuntimeError(f"missing specialist feature column: {col}")
    return out.replace([np.inf, -np.inf], np.nan)


def _path_return(close: np.ndarray, idx: int, side: int, hold: int, fee: float, slip: float) -> float:
    entry = float(close[idx] * (1.0 + slip if side > 0 else 1.0 - slip))
    end = min(len(close) - 1, idx + max(1, hold))
    px = close[idx + 1 : end + 1]
    if len(px) == 0:
        return 0.0
    raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
    # Label by the best path-adjusted return after round-trip costs. This is a
    # veto target, not an exit simulator; full backtest remains the authority.
    return float(np.max(raw) - 2.0 * (fee + slip))


def _candidate_dataset(
    df: pd.DataFrame,
    q: np.ndarray,
    *,
    feature_cols: list[str],
    edge_th: float,
    margin_th: float,
    hold: int,
    fee: float,
    slip: float,
) -> pd.DataFrame:
    x = _feature_frame(df, q, feature_cols)
    close = _close(df)
    rows: list[dict[str, Any]] = []
    for i in range(60, len(df) - max(hold, 2) - 1):
        ql = float(q[i, 0])
        qs = float(q[i, 1])
        edge = max(ql, qs)
        margin = abs(ql - qs)
        if edge < edge_th or margin < margin_th:
            continue
        for side, side_name in ((1, "LONG"), (-1, "SHORT")):
            ret = _path_return(close, i, side, hold, fee, slip)
            rec = {col: float(x.iloc[i][col]) for col in feature_cols}
            rec.update(
                {
                    "idx": int(i),
                    "timestamp": str(df.iloc[i].get("timestamp", "")),
                    "side": side_name,
                    "label": int(ret > 0.0),
                    "path_return": float(ret),
                }
            )
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no V31 candidate rows generated for side specialist")
    return out


def _fit_preprocess(train_x: pd.DataFrame) -> tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xi = imputer.fit_transform(train_x)
    xs = scaler.fit_transform(xi).astype(np.float32)
    return imputer, scaler, xs


def _train_net(train: pd.DataFrame, feature_cols: list[str]) -> tuple[SideSpecialistNet, SimpleImputer, StandardScaler, dict[str, Any]]:
    ordered = train.sort_values("idx").reset_index(drop=True)
    split = max(1, int(len(ordered) * 0.80))
    tr = ordered.iloc[:split].reset_index(drop=True)
    va = ordered.iloc[split:].reset_index(drop=True)
    imputer, scaler, xtr = _fit_preprocess(tr[feature_cols])
    xva = scaler.transform(imputer.transform(va[feature_cols])).astype(np.float32)

    y_long = np.where(tr["side"].eq("LONG"), tr["label"].to_numpy(dtype=np.float32), np.nan)
    y_short = np.where(tr["side"].eq("SHORT"), tr["label"].to_numpy(dtype=np.float32), np.nan)
    m_long = np.isfinite(y_long).astype(np.float32)
    m_short = np.isfinite(y_short).astype(np.float32)
    y_long = np.nan_to_num(y_long, nan=0.0).astype(np.float32)
    y_short = np.nan_to_num(y_short, nan=0.0).astype(np.float32)
    weight = (1.0 + np.minimum(np.abs(tr["path_return"].to_numpy(dtype=np.float32)) * 30.0, 5.0)).astype(np.float32)

    ds = TensorDataset(
        torch.from_numpy(xtr),
        torch.from_numpy(y_long),
        torch.from_numpy(y_short),
        torch.from_numpy(m_long),
        torch.from_numpy(m_short),
        torch.from_numpy(weight),
    )
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SideSpecialistNet(len(feature_cols)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_state: dict[str, torch.Tensor] | None = None
    best_val = 1e18
    patience = 0
    xva_t = torch.from_numpy(xva).to(device)
    va_side = va["side"].to_numpy()
    va_y = torch.from_numpy(va["label"].to_numpy(dtype=np.float32)).to(device)
    for _epoch in range(120):
        net.train()
        for xb, yl, ys, ml, ms, wb in loader:
            xb = xb.to(device)
            yl = yl.to(device)
            ys = ys.to(device)
            ml = ml.to(device)
            ms = ms.to(device)
            wb = wb.to(device)
            long_logit, short_logit = net(xb)
            loss_long = loss_fn(long_logit, yl) * ml * wb
            loss_short = loss_fn(short_logit, ys) * ms * wb
            denom = torch.clamp((ml + ms).sum(), min=1.0)
            loss = (loss_long.sum() + loss_short.sum()) / denom
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            long_logit, short_logit = net(xva_t)
            logits = torch.where(
                torch.tensor(va_side == "LONG", device=device),
                long_logit,
                short_logit,
            )
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(logits, va_y).detach().cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 12:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    meta = {
        "device": str(device),
        "train_rows": int(len(tr)),
        "val_rows": int(len(va)),
        "best_val_loss": float(best_val),
    }
    return net.cpu(), imputer, scaler, meta


def _predict_side_probs(
    net: SideSpecialistNet,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    df: pd.DataFrame,
    q: np.ndarray,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    x = _feature_frame(df, q, feature_cols)
    xs = scaler.transform(imputer.transform(x)).astype(np.float32)
    net.eval()
    with torch.no_grad():
        long_logit, short_logit = net(torch.from_numpy(xs))
    return {
        "LONG": torch.sigmoid(long_logit).numpy().astype(float),
        "SHORT": torch.sigmoid(short_logit).numpy().astype(float),
    }


def _select_thresholds(train: pd.DataFrame, probs: dict[str, np.ndarray]) -> SelectedThresholds:
    selected: dict[str, float] = {}
    for side_name in ("LONG", "SHORT"):
        sub = train[train["side"].eq(side_name)].reset_index(drop=True)
        idx = sub["idx"].to_numpy(dtype=int)
        prob = probs[side_name][idx]
        ret = sub["path_return"].to_numpy(dtype=float)
        candidates = np.unique(np.quantile(prob, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]))
        best_thr = float(candidates[0])
        best_score = -1e18
        for thr in candidates:
            keep = prob >= float(thr)
            if float(keep.mean()) < 0.70:
                continue
            score = float(ret[keep].sum() - 0.50 * np.maximum(ret[~keep], 0.0).sum() + 0.25 * np.maximum(-ret[~keep], 0.0).sum())
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        selected[side_name] = best_thr
    return SelectedThresholds(long=float(selected["LONG"]), short=float(selected["SHORT"]))


def _gate_from_probs(probs: dict[str, np.ndarray], thresholds: SelectedThresholds):
    def gate(i: int, side: int, ql: float, qs: float, row: pd.Series) -> tuple[bool, str]:
        if side > 0:
            return bool(probs["LONG"][int(i)] >= thresholds.long), "nn_long_veto"
        return bool(probs["SHORT"][int(i)] >= thresholds.short), "nn_short_veto"

    return gate


def _row(name: str, res: dict[str, Any], split: str, thresholds: SelectedThresholds) -> dict[str, Any]:
    return {
        "name": name,
        "split": split,
        "long_thr": float(thresholds.long),
        "short_thr": float(thresholds.short),
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "sl_ratio": float(sweep._sl_ratio(res)),
        "score": float(sweep._score(res)),
        "actions": json.dumps(res.get("actions", {}), ensure_ascii=False, sort_keys=True),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    val_df, eval_df = precision._load_frames()
    sources = precision._decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    val_dec = sources[str(cfg["source"])][0]
    eval_dec = sources[str(cfg["source"])][1]
    overlay = precision._overlay(stack["overlay"], cfg)
    feature_cols = _feature_cols(val_df)
    train = _candidate_dataset(
        val_df,
        val_q,
        feature_cols=feature_cols,
        edge_th=float(overlay.edge_th),
        margin_th=float(overlay.margin_th),
        hold=int(overlay.base_hold),
        fee=float(stack["fee"]) * 3.0,
        slip=float(stack["slip"]) * 3.0,
    )
    train.to_csv(TRAIN_CANDIDATES_OUT, index=False)
    net, imputer, scaler, train_meta = _train_net(train, feature_cols)
    train_probs = _predict_side_probs(net, imputer, scaler, val_df, val_q, feature_cols)
    thresholds = _select_thresholds(train, train_probs)
    val_probs = _predict_side_probs(net, imputer, scaler, val_df, val_q, feature_cols)
    oos_probs = _predict_side_probs(net, imputer, scaler, eval_df, eval_q, feature_cols)

    variant = sweep.Variant("deep_stop_cd18_nn_side_specialist", deep_stop_cooldown_extra=18)
    baseline = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    val_res = sweep._backtest_variant(
        df=val_df,
        q=val_q,
        dec=val_dec,
        stack=stack,
        cfg=cfg,
        variant=variant,
        cost_mult=3,
        record=False,
        deep_gate=_gate_from_probs(val_probs, thresholds),
    )
    oos_res = sweep._backtest_variant(
        df=eval_df,
        q=eval_q,
        dec=eval_dec,
        stack=stack,
        cfg=cfg,
        variant=variant,
        cost_mult=3,
        record=True,
        deep_gate=_gate_from_probs(oos_probs, thresholds),
    )
    baseline_oos = sweep._backtest_variant(df=eval_df, q=eval_q, dec=eval_dec, stack=stack, cfg=cfg, variant=baseline, cost_mult=3, record=False)
    pd.DataFrame(oos_res.get("trade_records", [])).to_csv(OOS_LEDGER_OUT, index=False)
    grid = pd.DataFrame(
        [
            _row("deep_stop_cd18", baseline_oos, "oos", SelectedThresholds(0.0, 0.0)),
            _row(variant.name, val_res, "val", thresholds),
            _row(variant.name, oos_res, "oos", thresholds),
        ]
    )
    grid.to_csv(GRID_OUT, index=False)
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": net.state_dict(),
            "feature_cols": feature_cols,
            "thresholds": asdict(thresholds),
            "train_meta": train_meta,
        },
        MODEL_OUT,
    )
    joblib.dump({"imputer": imputer, "scaler": scaler, "feature_cols": feature_cols}, OUT_DIR / "preprocess.joblib")
    summary = {
        "model_id": MODEL_ID,
        "feature_cols": feature_cols,
        "train_candidates": str(TRAIN_CANDIDATES_OUT),
        "train_rows": int(len(train)),
        "train_by_side": train.groupby("side")["label"].agg(["count", "mean"]).reset_index().to_dict(orient="records"),
        "train_meta": train_meta,
        "thresholds": asdict(thresholds),
        "model": str(MODEL_OUT),
        "preprocess": str(OUT_DIR / "preprocess.joblib"),
        "grid": str(GRID_OUT),
        "oos_ledger": str(OOS_LEDGER_OUT),
        "baseline_oos": _row("deep_stop_cd18", baseline_oos, "oos", SelectedThresholds(0.0, 0.0)),
        "candidate_val": _row(variant.name, val_res, "val", thresholds),
        "candidate_oos": _row(variant.name, oos_res, "oos", thresholds),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
