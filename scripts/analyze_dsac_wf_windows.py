#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.rl_runtime_primitives import MultiTimeframeFeatures  # noqa: E402
from ensemble.train_rl_dsac_agent import (  # noqa: E402
    _POS_THRESH,
    DSAC_STATE_DIM,
    DSACCompactTradingEnv,
    GaussianActor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze DSAC walk-forward window feature/action bias")
    p.add_argument("--csv-path", required=True)
    p.add_argument("--ckpt-path", default="data/ensemble/ckpt/best_dsac_agents.pth")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--output-path", default="")
    return p.parse_args()


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _summ(vals: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _future_logret(close: pd.Series, horizon: int) -> np.ndarray:
    c = pd.to_numeric(close, errors="coerce").to_numpy(dtype=np.float64)
    out = np.full_like(c, np.nan)
    if len(c) > horizon:
        out[:-horizon] = np.log(np.clip(c[horizon:] / np.clip(c[:-horizon], 1e-12, None), 1e-12, None))
    return out


def _build_val_windows(df_src: pd.DataFrame, windows: int) -> list[pd.DataFrame]:
    n = len(df_src)
    k = max(1, min(int(windows), n))
    if k <= 1 or n < 300:
        return [df_src.reset_index(drop=True)]
    base = n // k
    rem = n % k
    out: list[pd.DataFrame] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = min(n, start + size)
        if end - start >= 50:
            out.append(df_src.iloc[start:end].reset_index(drop=True))
        start = end
    return out or [df_src.reset_index(drop=True)]


def _load_actor(ckpt_path: str, device: str) -> tuple[GaussianActor, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)
    actor = GaussianActor(state_dim=state_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, meta


def analyze(csv_path: str, ckpt_path: str, train_ratio: float, windows: int) -> dict:
    df = pd.read_csv(csv_path)
    split_idx = int(len(df) * float(train_ratio))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)
    val_windows = _build_val_windows(df_val, windows)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor, meta = _load_actor(ckpt_path, device)

    window_results: list[dict[str, object]] = []
    feature_cols = [
        "m7_trend_xgb_dn",
        "m7_trend_xgb_fl",
        "m7_trend_xgb_up",
        "m7_q50",
        "m7_quality_pred",
        "log_return",
    ]

    for idx, win_df in enumerate(val_windows, start=1):
        mtf = MultiTimeframeFeatures(win_df["close"].values.astype(np.float32))
        env = DSACCompactTradingEnv(win_df, phase="eval", mtf_features=mtf)
        state = env.reset()
        done = False

        raw_actions: list[float] = []
        entry_long = 0
        entry_short = 0
        entry_from_flat = 0
        while not done:
            prev_pos = env.pos
            state_ts = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                raw = float(actor.deterministic(state_ts).cpu().item())
            raw_actions.append(raw)
            next_state, _, done, info = env.step(raw)
            if prev_pos is None and (info.get("entered_long") or info.get("entered_short")):
                entry_from_flat += 1
            if info.get("entered_long"):
                entry_long += 1
            if info.get("entered_short"):
                entry_short += 1
            state = next_state

        close = pd.to_numeric(win_df["close"], errors="coerce")
        fut1 = _future_logret(close, 1)
        fut3 = _future_logret(close, 3)
        fut12 = _future_logret(close, 12)
        trend_up = pd.to_numeric(win_df.get("m7_trend_xgb_up", pd.Series(dtype=float)), errors="coerce")
        trend_dn = pd.to_numeric(win_df.get("m7_trend_xgb_dn", pd.Series(dtype=float)), errors="coerce")
        dom_up = (trend_up > trend_dn).to_numpy(dtype=bool) if len(trend_up) == len(win_df) else np.zeros(len(win_df), dtype=bool)
        dom_dn = (trend_dn > trend_up).to_numpy(dtype=bool) if len(trend_dn) == len(win_df) else np.zeros(len(win_df), dtype=bool)

        feature_summary = {}
        for col in feature_cols:
            if col in win_df.columns:
                vals = pd.to_numeric(win_df[col], errors="coerce").to_numpy(dtype=np.float64)
                feature_summary[col] = _summ(vals)

        raw_arr = np.asarray(raw_actions, dtype=np.float64)
        pos_rate = float(np.mean(raw_arr > _POS_THRESH)) if raw_arr.size else 0.0
        neg_rate = float(np.mean(raw_arr < -_POS_THRESH)) if raw_arr.size else 0.0

        window_results.append(
            {
                "window": idx,
                "rows": int(len(win_df)),
                "timestamp_start": str(win_df["timestamp"].iloc[0]) if "timestamp" in win_df.columns and len(win_df) else "",
                "timestamp_end": str(win_df["timestamp"].iloc[-1]) if "timestamp" in win_df.columns and len(win_df) else "",
                "feature_summary": feature_summary,
                "future_returns": {
                    "fut1": _summ(fut1),
                    "fut3": _summ(fut3),
                    "fut12": _summ(fut12),
                    "dominant_up_mean_fut12": float(np.nanmean(fut12[dom_up])) if np.any(dom_up) else 0.0,
                    "dominant_dn_mean_fut12": float(np.nanmean(fut12[dom_dn])) if np.any(dom_dn) else 0.0,
                },
                "action_summary": {
                    "raw": _summ(raw_arr),
                    "raw_pos_thresh_rate": pos_rate,
                    "raw_neg_thresh_rate": neg_rate,
                },
                "entries": {
                    "long": int(entry_long),
                    "short": int(entry_short),
                    "from_flat": int(entry_from_flat),
                },
            }
        )

    result = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "csv_path": csv_path,
        "ckpt_path": ckpt_path,
        "train_ratio": float(train_ratio),
        "windows": int(windows),
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "meta": meta,
        "window_results": window_results,
    }
    return result


def main() -> int:
    args = parse_args()
    result = analyze(args.csv_path, args.ckpt_path, args.train_ratio, args.windows)
    output_path = args.output_path.strip()
    if not output_path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/ensemble/metrics/dsac_wf_window_analysis_{stamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
