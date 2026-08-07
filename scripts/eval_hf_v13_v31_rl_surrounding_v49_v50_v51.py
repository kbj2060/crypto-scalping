#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import build_training_set, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_conservative_limit_sniper_v46 as v46  # noqa: E402
from scripts import eval_hf_v13_v40_6_full_v31_stack_retrain as v40_6  # noqa: E402
from scripts import train_eval_hf_v13_constrained_rl_addon_allocator_v24 as v24  # noqa: E402
from scripts import train_eval_hf_v13_deep_jackpot_sequence_verifier_v23 as v23  # noqa: E402
from scripts import train_eval_hf_v13_frozen_v27_offline_rl_exit_overlay_v33 as v33  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_v31_rl_surrounding_v49_v50_v51_20260512"
DEFAULT_PARENT = v31.DEFAULT_PARENT
DEFAULT_JACKPOT = v31.DEFAULT_JACKPOT
DEFAULT_V27 = v31.DEFAULT_V27
DEFAULT_TRAIN = v31.DEFAULT_TRAIN
DEFAULT_EVAL = v31.DEFAULT_EVAL
DEFAULT_PARENT_REPORT = v40_6.DEFAULT_PARENT_REPORT
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v31_rl_surrounding_v49_v50_v51_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v31_rl_surrounding_v49_v50_v51_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v31_rl_surrounding_v49_v50_v51_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v31_rl_surrounding_v49_v50_v51_20260512_grid.csv"

V31_BASELINE = {
    "cost1": {"pnl": 277.07, "mdd": -31.74},
    "cost2": {"pnl": 112.79, "mdd": -31.46},
    "cost3": {"pnl": 20.93, "mdd": -43.09},
}

FORBIDDEN_FEATURE_TOKENS = (
    "future",
    "target",
    "label",
    "realized_pnl",
    "cash_after",
    "entry_signal",
    "exit_signal",
    "exit_reason",
    "ledger",
    "regime_v2",
    "hdb",
    "hmm",
    "legacy_regime",
)


class ClosePolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchClosePolicy:
    """Small discrete hold/close policy with sklearn-compatible predict_proba."""

    def __init__(self, *, epochs: int = 90, lr: float = 7e-4, seed: int = 2049) -> None:
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model: ClosePolicyNet | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchClosePolicy":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        x = self.imputer.fit_transform(x)
        x = self.scaler.fit_transform(x).astype(np.float32)
        y = np.asarray(y, dtype=np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClosePolicyNet(x.shape[1]).to(device)
        xb = torch.from_numpy(x)
        yb = torch.from_numpy(y)
        loader = DataLoader(TensorDataset(xb, yb), batch_size=256, shuffle=True)
        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss = loss_fn(self.model(batch_x), batch_y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
        self.model.cpu().eval()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TorchClosePolicy is not fitted")
        x = self.imputer.transform(x)
        x = self.scaler.transform(x).astype(np.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.from_numpy(x))).numpy().reshape(-1)
        return np.column_stack([1.0 - p, p]).astype(np.float64)


def _load_pickle(path: Path) -> dict[str, Any]:
    try:
        obj = joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a dict")
    return obj


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.40 * c2["pnl"] + 0.25 * c3["pnl"] - 0.25 * abs(c1["mdd"]))


def _is_forbidden_col(col: str) -> bool:
    name = col.lower()
    return any(tok in name for tok in FORBIDDEN_FEATURE_TOKENS)


def _numeric_cols(train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in train.columns:
        if c == "timestamp" or c not in eval_df.columns or _is_forbidden_col(c):
            continue
        if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_numeric_dtype(eval_df[c]):
            cols.append(c)
    return cols


def _feature_audit(feature_cols: list[str], train_all: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, Any]:
    bad = [c for c in feature_cols if _is_forbidden_col(c)]
    missing_eval = [c for c in feature_cols if c not in eval_df.columns]
    overlap = int(len(set(pd.to_datetime(train_all["timestamp"]).astype("int64")) & set(pd.to_datetime(eval_df["timestamp"]).astype("int64"))))
    blocking: list[str] = []
    if bad:
        blocking.append(f"forbidden_feature_cols={bad[:20]}")
    if missing_eval:
        blocking.append(f"missing_eval_feature_cols={missing_eval[:20]}")
    if overlap:
        blocking.append(f"train_eval_timestamp_overlap={overlap}")
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": [],
        "feature_count": int(len(feature_cols)),
        "forbidden_feature_cols": bad,
        "train_eval_timestamp_overlap": overlap,
    }


def _safe_float(row: pd.Series, col: str) -> float:
    try:
        x = float(row.get(col, 0.0))
    except Exception:
        return 0.0
    return float(x) if np.isfinite(x) else 0.0


@contextmanager
def _patch_v33_state(feature_cols: list[str]) -> Iterator[list[str]]:
    old_state = v33._deep_state_row
    old_features = list(v33.REVERSAL_FEATURES)
    extra_cols = [f"feat__{c}" for c in feature_cols]
    state_cols = list(dict.fromkeys(old_features + extra_cols))

    def expanded_state(
        frame: pd.DataFrame,
        i: int,
        side: int,
        edge: float,
        margin: float,
        hold: int,
        unreal: float,
        mfe: float,
        mae: float,
    ) -> dict[str, float]:
        state = old_state(frame, i, side, edge, margin, hold, unreal, mfe, mae)
        row = frame.iloc[i]
        for col in feature_cols:
            state[f"feat__{col}"] = _safe_float(row, col)
        return state

    v33._deep_state_row = expanded_state
    v33.REVERSAL_FEATURES = state_cols
    try:
        yield state_cols
    finally:
        v33._deep_state_row = old_state
        v33.REVERSAL_FEATURES = old_features


def _build_encoded_frames(args: argparse.Namespace, train_all: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_ts = pd.Timestamp("2025-10-01")
    train = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    with args.parent_report.open(encoding="utf-8") as f:
        parent_report = json.load(f)
    feature_cols = _feature_cols(train_all, eval_df)
    cfg = v40_6._parent_cfg()
    x_train, y, _ = build_training_set(train, cfg=cfg, stride_bars=int(args.train_stride), batch_size=512, feature_cols=feature_cols)
    train_idx_sample = np.arange(0, max(0, len(train) - cfg.max_train_horizon_bars - 1), max(1, int(args.train_stride)), dtype=np.int64)
    if len(train_idx_sample) != len(x_train):
        raise RuntimeError(f"train sample mismatch for PLS projection: {len(train_idx_sample)} vs {len(x_train)}")
    proj_targets = v40_6._projection_targets(y)
    train_feat = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    val_feat = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_feat = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    return v40_6._build_v40_6_frames(
        args=args,
        parent_report=parent_report,
        train_df=train,
        val_df=val,
        eval_df=eval_df,
        train_feat=train_feat,
        val_feat=val_feat,
        eval_feat=eval_feat,
        train_idx_sample=train_idx_sample,
        proj_targets=proj_targets,
    )


def _run_v49_exit_rl(
    *,
    mode: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    eval_df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    train_q: np.ndarray,
    val_q: np.ndarray,
    eval_q: np.ndarray,
    train_dec: pd.DataFrame,
    val_dec: pd.DataFrame,
    eval_dec: pd.DataFrame,
    feature_cols: list[str],
    fee: float,
    slip: float,
    out_dir: Path,
    report_out: Path,
    epochs: int,
    seed: int = 2049,
) -> dict[str, Any]:
    print(f"[{MODEL_ID}] V49 exit-RL mode={mode} features={len(feature_cols)} seed={seed}", flush=True)
    base_cfg = v33.OverlayConfig("v49_train_base", 0.010, 0.004, 1.2, 12, 0.60, 2, 0.045, 0.022, 48)
    with _patch_v33_state(feature_cols) as state_cols:
        x_train, y_train = v33._collect_reversal_training(train, train_dec, train_q, base_cfg, fee=fee, slip=slip)
        policy = TorchClosePolicy(epochs=epochs, seed=seed).fit(x_train.loc[:, state_cols].to_numpy(dtype=np.float32), y_train)
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for cfg in v33._overlay_grid():
            v1 = v33.backtest(val, bundle, jackpot_model, add_cfg, val_q, policy, cfg, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
            v2 = v33.backtest(val, bundle, jackpot_model, add_cfg, val_q, policy, cfg, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
            v3 = v33.backtest(val, bundle, jackpot_model, add_cfg, val_q, policy, cfg, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
            row = {"experiment": "v49_exit_rl", "feature_mode": mode, "seed": int(seed), "config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
        assert best is not None
        selected = v33.OverlayConfig(**best["config"])
        metrics: dict[str, Any] = {}
        ledgers: dict[str, str] = {}
        for mult in (1, 2, 3):
            r = v33.backtest(eval_df, bundle, jackpot_model, add_cfg, eval_q, policy, selected, fee=fee, slip=slip, cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
            if mult == 1:
                ledger = pd.DataFrame(r.pop("trade_records", []))
                lp = report_out.with_name(f"{report_out.stem}_v49_{mode}_seed{seed}_cost1_ledger.csv")
                ledger.to_csv(lp, index=False)
                ledgers["cost1"] = str(lp)
            metrics[f"cost{mult}"] = r
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / f"v49_exit_rl_{mode}_seed{seed}.pkl"
        joblib.dump({"policy": policy, "state_cols": state_cols, "feature_cols": feature_cols, "selected_config": asdict(selected)}, model_path)
    return {
        "experiment": "v49_exit_rl",
        "feature_mode": mode,
        "train_rows": int(len(y_train)),
        "close_rate": float(np.mean(y_train)),
        "seed": int(seed),
        "feature_count": int(len(feature_cols)),
        "state_feature_count": int(len(state_cols)),
        "selected_config": asdict(selected),
        "selection_result": best,
        "metrics": metrics,
        "grid": rows,
        "artifacts": {"model": str(model_path), "ledgers": ledgers},
    }


def _run_v50_addon_rl(
    *,
    mode: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    eval_df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    val_dec: pd.DataFrame,
    eval_dec: pd.DataFrame,
    seq_cols: list[str],
    fee: float,
    slip: float,
    out_dir: Path,
    report_out: Path,
    epochs: int,
) -> dict[str, Any]:
    print(f"[{MODEL_ID}] V50 add-on-RL mode={mode} seq_features={len(seq_cols)}", flush=True)
    train_ds = v23._collect_snapshots(train, bundle, jackpot_model, add_cfg, seq_cols, fee=fee, slip=slip)
    norm = v23._normalizers(train_ds["seq"], train_ds["ctx"])
    allocator = v24._train(train_ds, norm, epochs=epochs)
    old_predict = v23._predict_one
    old_action = v23._verifier_action
    v23._predict_one = v24._make_predict(allocator)
    v23._verifier_action = v24._allocator_action
    try:
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for cfg in v24._grid():
            v1 = v23.backtest(val, bundle, jackpot_model, allocator, norm, add_cfg, cfg, seq_cols, fee=fee, slip=slip, decisions=val_dec, cost_mult=1.0)
            v2 = v23.backtest(val, bundle, jackpot_model, allocator, norm, add_cfg, cfg, seq_cols, fee=fee, slip=slip, decisions=val_dec, cost_mult=2.0)
            v3 = v23.backtest(val, bundle, jackpot_model, allocator, norm, add_cfg, cfg, seq_cols, fee=fee, slip=slip, decisions=val_dec, cost_mult=3.0)
            row = {"experiment": "v50_addon_rl", "feature_mode": mode, "config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": v24._score(v1, v2, v3)}
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
        assert best is not None
        selected = v23.VerifierConfig(**best["config"])
        metrics: dict[str, Any] = {}
        ledgers: dict[str, str] = {}
        for mult in (1, 2, 3):
            r = v23.backtest(eval_df, bundle, jackpot_model, allocator, norm, add_cfg, selected, seq_cols, fee=fee, slip=slip, decisions=eval_dec, cost_mult=float(mult), record=(mult == 1))
            if mult == 1:
                ledger = pd.DataFrame(r.pop("trade_records", []))
                lp = report_out.with_name(f"{report_out.stem}_v50_{mode}_cost1_ledger.csv")
                ledger.to_csv(lp, index=False)
                ledgers["cost1"] = str(lp)
            metrics[f"cost{mult}"] = r
    finally:
        v23._predict_one = old_predict
        v23._verifier_action = old_action
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"v50_addon_rl_{mode}.pt"
    torch.save({"state_dict": allocator.state_dict(), "seq_cols": seq_cols, "ctx_cols": v23.CTX_COLS, "norm": norm, "selected_config": asdict(selected), "add_config": asdict(add_cfg)}, model_path)
    return {
        "experiment": "v50_addon_rl",
        "feature_mode": mode,
        "train_snapshots": int(len(train_ds["target"])),
        "train_action_distribution": {str(k): int(v) for k, v in zip(*np.unique(v24._labels(train_ds["target"]), return_counts=True))},
        "seq_feature_count": int(len(seq_cols)),
        "selected_config": asdict(selected),
        "selection_result": best,
        "metrics": metrics,
        "grid": rows,
        "artifacts": {"model": str(model_path), "ledgers": ledgers},
    }


def _run_v51_execution_sniper(
    *,
    val: pd.DataFrame,
    eval_df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    val_q: np.ndarray,
    eval_q: np.ndarray,
    val_dec: pd.DataFrame,
    eval_dec: pd.DataFrame,
    fee: float,
    slip: float,
    report_out: Path,
) -> dict[str, Any]:
    print(f"[{MODEL_ID}] V51 execution sniper conservative limit-fill", flush=True)
    overlay = v46._base_overlay()
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in v46._variants():
        v1 = v46.backtest_limit_sniper(val, bundle, jackpot_model, add_cfg, val_q, overlay, cfg, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
        v2 = v46.backtest_limit_sniper(val, bundle, jackpot_model, add_cfg, val_q, overlay, cfg, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
        v3 = v46.backtest_limit_sniper(val, bundle, jackpot_model, add_cfg, val_q, overlay, cfg, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
        row = {"experiment": "v51_execution_sniper", "feature_mode": "policy_ohlcv_micro_proxy", "config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": v46._score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected = v46.LimitSniperConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = v46.backtest_limit_sniper(eval_df, bundle, jackpot_model, add_cfg, eval_q, overlay, selected, fee=fee, slip=slip, cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = report_out.with_name(f"{report_out.stem}_v51_cost1_ledger.csv")
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    return {
        "experiment": "v51_execution_sniper",
        "feature_mode": "policy_ohlcv_micro_proxy",
        "selected_config": asdict(selected),
        "selection_result": best,
        "metrics": metrics,
        "grid": rows,
        "artifacts": {"ledgers": ledgers},
        "warning": "Uses conservative next-bar OHLC limit penetration proxy, not live L2 queue simulation.",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V49/V50/V51 V31 surrounding RL-layer experiments.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--embed-batch", type=int, default=8)
    p.add_argument("--exit-epochs", type=int, default=80)
    p.add_argument("--addon-epochs", type=int, default=50)
    p.add_argument("--skip-encoded", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{MODEL_ID}] loading frozen V31 stack", flush=True)
    bundle = _load_pickle(args.parent_model)
    jackpot_payload = _load_pickle(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    base = dict(bundle["config"])
    fee = float(base["fee"])
    slip = float(base["slip"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)

    print(f"[{MODEL_ID}] predicting parent decisions and V27 utilities", flush=True)
    train_dec = predict_policy_frame(bundle, train, close=_close(train))
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    train_q = v31._predict_all(v27_model, train, v27_payload["seq_cols"], v27_payload["norm"])
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    raw_feature_cols = _numeric_cols(train_all, eval_df)
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]] = {
        "raw_all": (train, val, eval_df, raw_feature_cols, {"feature_source": "all numeric non-forbidden CSV columns"}),
    }
    if not args.skip_encoded:
        print(f"[{MODEL_ID}] preparing Chronos/Kairos -> PLS encoded frames", flush=True)
        enc_train, enc_val, enc_eval, enc_meta = _build_encoded_frames(args, train_all, eval_df)
        enc_feature_cols = _numeric_cols(pd.concat([enc_train, enc_val], ignore_index=True), enc_eval)
        frames["encoded_pls_all"] = (enc_train, enc_val, enc_eval, enc_feature_cols, enc_meta)

    feature_audits = {mode: _feature_audit(cols, pd.concat([tr, va], ignore_index=True), ev) for mode, (tr, va, ev, cols, _) in frames.items()}
    contract_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))

    experiments: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for mode, (tr, va, ev, cols, _) in frames.items():
        if feature_audits[mode]["status"] != "pass":
            print(f"[{MODEL_ID}] skip mode={mode}: audit failed", flush=True)
            continue
        if mode == "raw_all":
            tr_q, va_q, ev_q = train_q, val_q, eval_q
            tr_dec, va_dec, ev_dec = train_dec, val_dec, eval_dec
        else:
            tr_q = v31._predict_all(v27_model, tr, v27_payload["seq_cols"], v27_payload["norm"])
            va_q = v31._predict_all(v27_model, va, v27_payload["seq_cols"], v27_payload["norm"])
            ev_q = v31._predict_all(v27_model, ev, v27_payload["seq_cols"], v27_payload["norm"])
            tr_dec, va_dec, ev_dec = train_dec, val_dec, eval_dec

        v49 = _run_v49_exit_rl(
            mode=mode,
            train=tr,
            val=va,
            eval_df=ev,
            bundle=bundle,
            jackpot_model=jackpot_model,
            add_cfg=add_cfg,
            train_q=tr_q,
            val_q=va_q,
            eval_q=ev_q,
            train_dec=tr_dec,
            val_dec=va_dec,
            eval_dec=ev_dec,
            feature_cols=cols,
            fee=fee,
            slip=slip,
            out_dir=args.out_dir,
            report_out=args.report_out,
            epochs=args.exit_epochs,
        )
        experiments.append(v49)
        grid_rows.extend(v49["grid"])

        seq_cols = cols if mode != "raw_all" else raw_feature_cols
        v50 = _run_v50_addon_rl(
            mode=mode,
            train=tr,
            val=va,
            eval_df=ev,
            bundle=bundle,
            jackpot_model=jackpot_model,
            add_cfg=add_cfg,
            val_dec=va_dec,
            eval_dec=ev_dec,
            seq_cols=seq_cols,
            fee=fee,
            slip=slip,
            out_dir=args.out_dir,
            report_out=args.report_out,
            epochs=args.addon_epochs,
        )
        experiments.append(v50)
        grid_rows.extend(v50["grid"])

    v51 = _run_v51_execution_sniper(
        val=val,
        eval_df=eval_df,
        bundle=bundle,
        jackpot_model=jackpot_model,
        add_cfg=add_cfg,
        val_q=val_q,
        eval_q=eval_q,
        val_dec=val_dec,
        eval_dec=eval_dec,
        fee=fee,
        slip=slip,
        report_out=args.report_out,
    )
    experiments.append(v51)
    grid_rows.extend(v51["grid"])

    baseline: dict[str, Any] = {}
    overlay = v46._base_overlay()
    for mult in (1, 2, 3):
        baseline[f"cost{mult}"] = v31.backtest(eval_df, bundle, jackpot_model, add_cfg, eval_q, overlay, fee=fee, slip=slip, cost_mult=float(mult), decisions=eval_dec)

    best = max(experiments, key=lambda r: float(r["metrics"]["cost1"]["pnl"]))
    blocking: list[str] = []
    warnings: list[str] = []
    if contract_audit["status"] != "pass":
        blocking.extend(contract_audit.get("blocking", []))
    for mode, audit in feature_audits.items():
        if audit["status"] != "pass":
            blocking.extend([f"{mode}:{x}" for x in audit.get("blocking", [])])
        warnings.extend([f"{mode}:{x}" for x in audit.get("warnings", [])])
    if best["experiment"] == "v51_execution_sniper":
        warnings.append("best_execution_variant_uses_ohlcv_limit_fill_proxy_not_live_l2_queue")
    if best["metrics"]["cost1"]["pnl"] <= baseline["cost1"]["pnl"]:
        warnings.append("best_did_not_beat_recomputed_v31_cost1")
    if best["metrics"]["cost2"]["pnl"] <= 0.0:
        warnings.append("best_cost2_not_survived")
    if best["metrics"]["cost3"]["pnl"] <= 0.0:
        warnings.append("best_cost3_not_survived")
    verdict = "promote" if not blocking and best["metrics"]["cost1"]["pnl"] > baseline["cost1"]["pnl"] and best["metrics"]["cost2"]["pnl"] > 0 and best["metrics"]["cost3"]["pnl"] > 0 and best["experiment"] != "v51_execution_sniper" else "iterate"

    pd.DataFrame(
        [
            {
                "experiment": row["experiment"],
                "feature_mode": row["feature_mode"],
                **{f"cfg_{k}": v for k, v in row["config"].items()},
                "selection_score": row["selection_score"],
                "val_cost1_pnl": row["validation_cost1"]["pnl"],
                "val_cost1_mdd": row["validation_cost1"]["mdd"],
                "val_cost1_trades": row["validation_cost1"]["trades"],
                "val_cost2_pnl": row["validation_cost2"]["pnl"],
                "val_cost3_pnl": row["validation_cost3"]["pnl"],
            }
            for row in grid_rows
        ]
    ).to_csv(args.grid_out, index=False)

    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after selection",
        "baseline_recomputed_v31": baseline,
        "feature_audits": feature_audits,
        "parent_contract_audit": contract_audit,
        "best": {"experiment": best["experiment"], "feature_mode": best["feature_mode"], "metrics": best["metrics"], "selected_config": best.get("selected_config")},
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V49/V50/V51 tests around frozen V31. V49 trains a discrete hold/close policy for V27 deep_alpha exits; V50 trains a constrained reject/0.10/0.20 add-on allocator for V21.2 jackpot candidates; V51 rechecks conservative maker/taker execution routing with OHLCV penetration proxy. V49/V50 compare raw-all features and Chronos/Kairos target-aware PLS factor augmented features.",
        "baseline_recomputed_v31": baseline,
        "experiments": experiments,
        "best": {"experiment": best["experiment"], "feature_mode": best["feature_mode"], "metrics": best["metrics"], "selected_config": best.get("selected_config")},
        "audit": audit,
        "artifacts": {"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "out_dir": str(args.out_dir)},
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "best": report["best"], "verdict": verdict}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
