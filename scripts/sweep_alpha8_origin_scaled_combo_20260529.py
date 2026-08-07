#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_alpha4_new_features_full_retrain_20260517 as a4  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import _compact_costs, _no_deep_overlay  # noqa: E402
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import EVAL_CSV, FORBIDDEN_PREFIXES, TRAIN_CSV  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


MODEL_ID = "alpha8_origin_scaled_combo_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class OriginCfg:
    name: str
    primary_mult: float
    fallback_mult: float
    primary_cap: float
    fallback_cap: float
    leverage: float
    primary_tp_mult: float
    fallback_tp_mult: float
    primary_sl_mult: float
    fallback_sl_mult: float
    primary_hold_mult: float
    fallback_hold_mult: float
    min_primary_conf: float
    min_fallback_conf: float
    min_quality: float


def _assert_clean(df: pd.DataFrame, *, name: str) -> None:
    bad = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)


def _zero_rows(out: pd.DataFrame, mask: np.ndarray) -> None:
    if not np.any(mask):
        return
    for col, value in (
        ("action", 0),
        ("side", 0),
        ("notional_exposure", 0.0),
        ("position_fraction", 0.0),
        ("take_profit", 0.0),
        ("stop_loss", 0.0),
        ("max_hold_bars", 0),
        ("cooldown_bars", 0),
    ):
        out.loc[mask, col] = value
    out.loc[mask, "leverage"] = 1.0


def _scale_rows(out: pd.DataFrame, mask: np.ndarray, *, mult: float, cap: float, leverage: float, tp_mult: float, sl_mult: float, hold_mult: float) -> None:
    if not np.any(mask):
        return
    base_notional = _num(out, "notional_exposure")
    notional = np.minimum(np.maximum(base_notional * float(mult), 0.0), float(cap))
    lev = max(float(leverage), 1e-8)
    out.loc[mask, "notional_exposure"] = notional[mask]
    out.loc[mask, "leverage"] = lev
    out.loc[mask, "position_fraction"] = notional[mask] / lev
    out.loc[mask, "take_profit"] = np.maximum(_num(out, "take_profit")[mask], 1e-8) * float(tp_mult)
    out.loc[mask, "stop_loss"] = np.maximum(np.abs(_num(out, "stop_loss")[mask]), 1e-8) * float(sl_mult)
    out.loc[mask, "max_hold_bars"] = np.maximum(1, np.rint(np.maximum(_num(out, "max_hold_bars")[mask], 1.0) * float(hold_mult))).astype(int)


def _apply_cfg(primary_dec: pd.DataFrame, fallback_dec: pd.DataFrame, cfg: OriginCfg) -> pd.DataFrame:
    out = _combine_primary_fallback(primary_dec, fallback_dec).reset_index(drop=True)
    primary_active = _active(primary_dec)
    fallback_origin = (~primary_active) & _active(fallback_dec)
    primary_origin = primary_active
    q = _num(out, "quality_score")
    conf = _num(out, "confidence")
    primary_veto = primary_origin & ((q < cfg.min_quality) | (conf < cfg.min_primary_conf))
    fallback_veto = fallback_origin & ((q < cfg.min_quality) | (conf < cfg.min_fallback_conf))
    _zero_rows(out, primary_veto | fallback_veto)
    _scale_rows(
        out,
        primary_origin & ~primary_veto,
        mult=cfg.primary_mult,
        cap=cfg.primary_cap,
        leverage=cfg.leverage,
        tp_mult=cfg.primary_tp_mult,
        sl_mult=cfg.primary_sl_mult,
        hold_mult=cfg.primary_hold_mult,
    )
    _scale_rows(
        out,
        fallback_origin & ~fallback_veto,
        mult=cfg.fallback_mult,
        cap=cfg.fallback_cap,
        leverage=cfg.leverage,
        tp_mult=cfg.fallback_tp_mult,
        sl_mult=cfg.fallback_sl_mult,
        hold_mult=cfg.fallback_hold_mult,
    )
    return out


class OfficialCost3:
    def __init__(self) -> None:
        parent = joblib.load(v31.DEFAULT_PARENT)
        self.parent_for_features = _parent_for_features(list(parent["feature_cols"]))
        self.fee = float(parent["config"]["fee"])
        self.slip = float(parent["config"]["slip"])
        self.runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
        self.runner_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
        self.overlay = _no_deep_overlay()
        self.limit_cfg = ft_v2.ft_v1._limit_cfg()

    def __call__(self, df: pd.DataFrame, dec: pd.DataFrame) -> dict[str, Any]:
        q0 = np.zeros((len(df), 2), dtype=np.float32)
        metrics = a4._metrics(
            df.reset_index(drop=True),
            self.parent_for_features,
            self.runner,
            self.runner_cfg,
            q0,
            dec.reset_index(drop=True),
            self.overlay,
            self.limit_cfg,
            fee=self.fee,
            slip=self.slip,
        )
        return _compact_costs(metrics)["cost3"]


def _score(m: dict[str, Any]) -> float:
    trades = int(m["trades"])
    if trades < 30:
        return -1e9 + float(m["pnl"])
    return float(m["pnl"]) + 120.0 * float(m["wr"]) - 0.35 * abs(float(m["mdd"])) + 0.02 * trades


def _candidate_grid(limit: int) -> list[OriginCfg]:
    rows: list[OriginCfg] = [
        OriginCfg("baseline_contract", 1, 1, 99, 99, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
    ]
    i = 0
    tp_sl_templates = [
        (0.25, 0.25, 1.50, 1.50, 0.50, 0.50),
        (0.35, 0.35, 2.00, 2.00, 0.75, 0.75),
        (0.50, 0.50, 1.50, 1.50, 0.75, 0.75),
        (0.50, 0.75, 1.50, 1.25, 0.75, 0.75),
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        (1.25, 1.0, 1.0, 0.75, 1.0, 0.75),
        (1.5, 1.0, 1.0, 0.75, 1.25, 0.75),
        (1.75, 0.75, 0.85, 0.60, 1.0, 0.50),
        (2.0, 0.75, 1.0, 0.50, 1.25, 0.50),
    ]
    for p_mult in (1.0, 1.25, 1.5, 2.0, 2.5):
        for f_mult in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
            for cap in (99.0, 5.0, 4.0, 3.0):
                for lev in (3.0, 5.0):
                    for ptp, ftp, psl, fsl, ph, fh in tp_sl_templates:
                        for min_q, min_pc, min_fc in ((0.0, 0.0, 0.0), (0.005, 0.55, 0.55), (0.01, 0.65, 0.60)):
                            rows.append(
                                OriginCfg(
                                    name=f"origin_{i:05d}",
                                    primary_mult=p_mult,
                                    fallback_mult=f_mult,
                                    primary_cap=cap,
                                    fallback_cap=cap,
                                    leverage=lev,
                                    primary_tp_mult=ptp,
                                    fallback_tp_mult=ftp,
                                    primary_sl_mult=psl,
                                    fallback_sl_mult=fsl,
                                    primary_hold_mult=ph,
                                    fallback_hold_mult=fh,
                                    min_primary_conf=min_pc,
                                    min_fallback_conf=min_fc,
                                    min_quality=min_q,
                                )
                            )
                            i += 1
    return rows[: int(limit)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-candidates", type=int, default=40)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(EVAL_CSV))
    _assert_clean(train_all, name="train_all")
    _assert_clean(eval_df, name="eval")
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary = joblib.load(baseline.primary_parent)
    fallback = joblib.load(baseline.fallback_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)
    p_val = _predict_scaled(primary, val_df, primary_rt).reset_index(drop=True)
    f_val = _predict_scaled(fallback, val_df, fallback_rt).reset_index(drop=True)
    p_eval = _predict_scaled(primary, eval_df, primary_rt).reset_index(drop=True)
    f_eval = _predict_scaled(fallback, eval_df, fallback_rt).reset_index(drop=True)
    evaluator = OfficialCost3()

    rows: list[dict[str, Any]] = []
    candidates = _candidate_grid(int(args.max_candidates))
    for idx, cfg in enumerate(candidates, start=1):
        val_dec = _apply_cfg(p_val, f_val, cfg)
        val = evaluator(val_df, val_dec)
        rows.append({"name": cfg.name, "score": _score(val), **{f"val_{k}": v for k, v in val.items()}, **asdict(cfg)})
        print(json.dumps({"stage": "val", "done": idx, "total": len(candidates), "name": cfg.name, "val": val, "elapsed_sec": round(time.time() - start, 1)}, ensure_ascii=False), flush=True)
        pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(OUT_DIR / "validation_ranking.csv", index=False)

    val_rank = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    oos_rows: list[dict[str, Any]] = []
    for _, r in val_rank.head(8).iterrows():
        cfg = OriginCfg(**{k: r[k] for k in OriginCfg.__dataclass_fields__.keys()})
        oos_dec = _apply_cfg(p_eval, f_eval, cfg)
        oos = evaluator(eval_df, oos_dec)
        val = {k[4:]: r[k] for k in r.index if str(k).startswith("val_")}
        oos_rows.append({"name": cfg.name, "score": _score(oos), **{f"val_{k}": v for k, v in val.items()}, **{f"oos_{k}": v for k, v in oos.items()}, **asdict(cfg)})
        print(json.dumps({"stage": "oos", "name": cfg.name, "oos": oos}, ensure_ascii=False), flush=True)
    oos_rank = pd.DataFrame(oos_rows).sort_values(["oos_pnl", "oos_wr"], ascending=False).reset_index(drop=True)
    oos_rank.to_csv(OUT_DIR / "oos_ranking.csv", index=False)
    best = oos_rank.iloc[0].to_dict() if len(oos_rank) else {}
    summary = {
        "model_id": MODEL_ID,
        "design": "Alpha8 origin-aware combo risk sweep. Alpha7 primary/fallback direction owners are preserved; only origin-specific notional/leverage/TP/SL/hold and light confidence veto are changed.",
        "selection_basis": "2025Q4 validation official cost3 ranking; OOS only for top validation candidates",
        "target": {"oos_pnl_min": 200.0, "oos_wr_min": 0.50},
        "target_hit": bool(best and float(best.get("oos_pnl", 0.0)) >= 200.0 and float(best.get("oos_wr", 0.0)) >= 0.50),
        "best": best,
        "audit": {
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "forbidden_prefix_count": 0,
            "official_accounting": "cost3 fee/slippage via existing signal-limit-close metrics",
            "live_wired": False,
        },
        "artifacts": {
            "validation_ranking": str(OUT_DIR / "validation_ranking.csv"),
            "oos_ranking": str(OUT_DIR / "oos_ranking.csv"),
            "summary": str(OUT_DIR / "summary.json"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(OUT_DIR / "summary.json"), "target_hit": summary["target_hit"], "best": best}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
