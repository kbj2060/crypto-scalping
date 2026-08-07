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

from scripts import build_deep_side_specialist_chosen_dataset_20260528 as dataset  # noqa: E402
from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "deep_side_specialist_chosen_nn_veto_20260528"
DATA_DIR = ROOT / "tmp/causal_regen_20260516/deep_side_specialist_chosen_dataset_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
MODEL_OUT = OUT_DIR / "chosen_side_specialist_nn.pt"
PREPROCESS_OUT = OUT_DIR / "preprocess.joblib"
OOS_LEDGER_OUT = OUT_DIR / "oos_cost3_ledger.csv"
OOS_BEAR_LEDGER_OUT = OUT_DIR / "oos_nn_plus_bear_long_veto_ledger.csv"


META_COLS = {
    "split",
    "idx",
    "timestamp",
    "side",
    "label_binary",
    "label_soft",
    "label_strict",
    "path_return",
    "sample_weight",
}


@dataclass(frozen=True)
class Thresholds:
    long: float
    short: float


class ChosenSideNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, dropout: float = 0.10) -> None:
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


def _load_chosen_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (DATA_DIR / "audit.json").exists():
        dataset.main()
    train_all = pd.read_csv(DATA_DIR / "train_2025_chosen_all.csv")
    train_strict = pd.read_csv(DATA_DIR / "train_2025_chosen_strict.csv")
    eval_all = pd.read_csv(DATA_DIR / "eval_2026_chosen_all.csv")
    eval_strict = pd.read_csv(DATA_DIR / "eval_2026_chosen_strict.csv")
    return train_all, train_strict, eval_all, eval_strict


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in META_COLS:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            cols.append(str(col))
    return cols


def _fit_preprocess(train: pd.DataFrame, feature_cols: list[str]) -> tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = imputer.fit_transform(train[feature_cols])
    x = scaler.fit_transform(x).astype(np.float32)
    return imputer, scaler, x


def _transform(imputer: SimpleImputer, scaler: StandardScaler, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return scaler.transform(imputer.transform(frame[feature_cols])).astype(np.float32)


def _train_net(train_strict: pd.DataFrame, feature_cols: list[str]) -> tuple[ChosenSideNet, SimpleImputer, StandardScaler, dict[str, Any], pd.DataFrame]:
    ordered = train_strict.sort_values("idx").reset_index(drop=True)
    split = max(1, int(len(ordered) * 0.80))
    tr = ordered.iloc[:split].reset_index(drop=True)
    va = ordered.iloc[split:].reset_index(drop=True)
    imputer, scaler, xtr = _fit_preprocess(tr, feature_cols)
    xva = _transform(imputer, scaler, va, feature_cols)
    y = tr["label_strict"].to_numpy(dtype=np.float32)
    side = tr["side"].to_numpy()
    y_long = np.where(side == "LONG", y, np.nan)
    y_short = np.where(side == "SHORT", y, np.nan)
    m_long = np.isfinite(y_long).astype(np.float32)
    m_short = np.isfinite(y_short).astype(np.float32)
    y_long = np.nan_to_num(y_long, nan=0.0).astype(np.float32)
    y_short = np.nan_to_num(y_short, nan=0.0).astype(np.float32)
    weight = pd.to_numeric(tr["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    ds = TensorDataset(
        torch.from_numpy(xtr),
        torch.from_numpy(y_long),
        torch.from_numpy(y_short),
        torch.from_numpy(m_long),
        torch.from_numpy(m_short),
        torch.from_numpy(weight),
    )
    loader = DataLoader(ds, batch_size=384, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChosenSideNet(len(feature_cols)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=2e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    xva_t = torch.from_numpy(xva).to(device)
    va_side = va["side"].to_numpy()
    va_y = torch.from_numpy(va["label_strict"].to_numpy(dtype=np.float32)).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = 1e18
    patience = 0
    for _epoch in range(160):
        net.train()
        for xb, yl, ys, ml, ms, wb in loader:
            xb = xb.to(device)
            yl = yl.to(device)
            ys = ys.to(device)
            ml = ml.to(device)
            ms = ms.to(device)
            wb = wb.to(device)
            long_logit, short_logit = net(xb)
            loss = (loss_fn(long_logit, yl) * ml * wb).sum() + (loss_fn(short_logit, ys) * ms * wb).sum()
            loss = loss / torch.clamp((ml + ms).sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            long_logit, short_logit = net(xva_t)
            logits = torch.where(torch.tensor(va_side == "LONG", device=device), long_logit, short_logit)
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(logits, va_y).detach().cpu())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 16:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    meta = {
        "device": str(device),
        "train_rows": int(len(tr)),
        "validation_rows": int(len(va)),
        "best_val_loss": float(best_loss),
    }
    return net.cpu(), imputer, scaler, meta, va


def _predict_probs(net: ChosenSideNet, imputer: SimpleImputer, scaler: StandardScaler, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = _transform(imputer, scaler, frame, feature_cols)
    net.eval()
    with torch.no_grad():
        long_logit, short_logit = net(torch.from_numpy(x))
        prob = torch.where(
            torch.tensor(frame["side"].to_numpy() == "LONG"),
            torch.sigmoid(long_logit),
            torch.sigmoid(short_logit),
        )
    return prob.numpy().astype(float)


def _select_thresholds(val: pd.DataFrame, prob: np.ndarray) -> Thresholds:
    selected: dict[str, float] = {}
    for side in ("LONG", "SHORT"):
        mask = val["side"].eq(side).to_numpy()
        sub = val.loc[mask].copy()
        p = prob[mask]
        ret = sub["path_return"].to_numpy(dtype=float)
        candidates = np.unique(np.quantile(p, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]))
        best_thr = float(candidates[0])
        best_score = -1e18
        for thr in candidates:
            keep = p >= float(thr)
            if float(keep.mean()) < 0.68:
                continue
            score = float(ret[keep].sum() - 0.70 * np.maximum(ret[~keep], 0.0).sum() + 0.35 * np.maximum(-ret[~keep], 0.0).sum())
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        selected[side] = best_thr
    return Thresholds(long=float(selected["LONG"]), short=float(selected["SHORT"]))


def _prob_map(frame: pd.DataFrame, prob: np.ndarray) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for idx, side, p in zip(frame["idx"].to_numpy(dtype=int), frame["side_int"].to_numpy(dtype=int), prob):
        out[(int(idx), int(side))] = float(p)
    return out


def _gate_from_map(prob_by_idx_side: dict[tuple[int, int], float], thresholds: Thresholds):
    def gate(i: int, side: int, ql: float, qs: float, row: pd.Series) -> tuple[bool, str]:
        prob = prob_by_idx_side.get((int(i), int(side)))
        if prob is None:
            return True, ""
        if side > 0:
            return bool(prob >= thresholds.long), "chosen_nn_long_veto"
        return bool(prob >= thresholds.short), "chosen_nn_short_veto"

    return gate


def _row(name: str, split: str, res: dict[str, Any], thresholds: Thresholds) -> dict[str, Any]:
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
    train_all, train_strict, eval_all, _eval_strict = _load_chosen_data()
    feature_cols = _feature_cols(train_strict)
    net, imputer, scaler, train_meta, val_strict = _train_net(train_strict, feature_cols)
    val_prob = _predict_probs(net, imputer, scaler, val_strict, feature_cols)
    thresholds = _select_thresholds(val_strict, val_prob)
    train_all_prob = _predict_probs(net, imputer, scaler, train_all, feature_cols)
    eval_all_prob = _predict_probs(net, imputer, scaler, eval_all, feature_cols)

    decontam._patch_runtime_sources()
    cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    val_df, eval_df = precision._load_frames()
    sources = precision._decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    val_dec = sources[str(cfg["source"])][0]
    eval_dec = sources[str(cfg["source"])][1]
    baseline = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    chosen = sweep.Variant("deep_stop_cd18_chosen_nn_side_specialist", deep_stop_cooldown_extra=18)
    bear = sweep.Variant(
        "deep_stop_cd18_chosen_nn_plus_bear_long_veto",
        deep_stop_cooldown_extra=18,
        deep_block_long_in_bear_regime=True,
    )
    val_gate = _gate_from_map(_prob_map(train_all, train_all_prob), thresholds)
    oos_gate = _gate_from_map(_prob_map(eval_all, eval_all_prob), thresholds)

    val_res = sweep._backtest_variant(df=val_df, q=val_q, dec=val_dec, stack=stack, cfg=cfg, variant=chosen, cost_mult=3, record=False, deep_gate=val_gate)
    oos_res = sweep._backtest_variant(df=eval_df, q=eval_q, dec=eval_dec, stack=stack, cfg=cfg, variant=chosen, cost_mult=3, record=True, deep_gate=oos_gate)

    def bear_gate(i: int, side: int, ql: float, qs: float, row: pd.Series) -> tuple[bool, str]:
        if side > 0 and sweep._state24_dominant_regime(row) == "bear":
            return False, "bear_long_veto"
        return oos_gate(i, side, ql, qs, row)

    oos_bear_res = sweep._backtest_variant(df=eval_df, q=eval_q, dec=eval_dec, stack=stack, cfg=cfg, variant=bear, cost_mult=3, record=True, deep_gate=bear_gate)
    baseline_oos = sweep._backtest_variant(df=eval_df, q=eval_q, dec=eval_dec, stack=stack, cfg=cfg, variant=baseline, cost_mult=3, record=False)
    pd.DataFrame(oos_res.get("trade_records", [])).to_csv(OOS_LEDGER_OUT, index=False)
    pd.DataFrame(oos_bear_res.get("trade_records", [])).to_csv(OOS_BEAR_LEDGER_OUT, index=False)
    rows = [
        _row("deep_stop_cd18", "oos", baseline_oos, Thresholds(0.0, 0.0)),
        _row(chosen.name, "val", val_res, thresholds),
        _row(chosen.name, "oos", oos_res, thresholds),
        _row(bear.name, "oos", oos_bear_res, thresholds),
    ]
    pd.DataFrame(rows).to_csv(GRID_OUT, index=False)
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
    joblib.dump({"imputer": imputer, "scaler": scaler, "feature_cols": feature_cols}, PREPROCESS_OUT)
    summary = {
        "model_id": MODEL_ID,
        "data_dir": str(DATA_DIR),
        "feature_cols": feature_cols,
        "train_rows_strict": int(len(train_strict)),
        "train_by_side": train_strict.groupby("side")["label_strict"].agg(["count", "mean"]).reset_index().to_dict(orient="records"),
        "train_meta": train_meta,
        "thresholds": asdict(thresholds),
        "model": str(MODEL_OUT),
        "preprocess": str(PREPROCESS_OUT),
        "grid": str(GRID_OUT),
        "oos_ledger": str(OOS_LEDGER_OUT),
        "oos_bear_ledger": str(OOS_BEAR_LEDGER_OUT),
        "baseline_oos": rows[0],
        "candidate_val": rows[1],
        "candidate_oos": rows[2],
        "candidate_oos_plus_bear_long_veto": rows[3],
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
