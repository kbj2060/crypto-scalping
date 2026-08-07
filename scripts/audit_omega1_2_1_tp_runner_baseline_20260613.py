#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import test_omega1_2_1_time_decay_sltp_20260613 as decay  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_tp_runner_baseline_redteam_audit_20260613"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TP_BUNDLE_PATH = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _market_taker_next_open(arrays: dict[str, np.ndarray], signal_i: int, side: int, *, entry: bool, fee_base: float, slip_base: float) -> tuple[bool, float, float, str]:
    fill_i = min(int(signal_i) + 1, len(arrays["open"]) - 1)
    px = float(arrays["open"][fill_i])
    if side > 0:
        fill = px * (1.0 + slip_base if entry else 1.0 - slip_base)
    else:
        fill = px * (1.0 - slip_base if entry else 1.0 + slip_base)
    return True, float(fill), float(fee_base), "market_taker_next_open"


@contextmanager
def _patched_execution(fn: Callable[..., tuple[bool, float, float, str]]) -> Iterator[None]:
    old = omega._try_execution
    omega._try_execution = fn
    try:
        yield
    finally:
        omega._try_execution = old


def _ledger_intrabar_scan(frame: pd.DataFrame, ledger: pd.DataFrame, *, slip_eff: float) -> pd.DataFrame:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    rows: list[dict[str, Any]] = []
    for trade_id, row in ledger.reset_index(drop=True).iterrows():
        side = 1 if str(row["side"]).upper() == "LONG" else -1
        entry_price = float(row["entry_price"])
        notional = float(row["effective_exposure"])
        tp = float(row["tp_equity_ret"])
        sl = abs(float(row["sl_equity_ret"]))
        entry_i = int(row["entry_i"])
        exit_i = int(row["exit_i"])
        first_hilo_i: int | None = None
        first_hilo_reason: str | None = None
        for i in range(entry_i, min(exit_i + 1, len(frame))):
            fav_px = float(arrays["high"][i] if side > 0 else arrays["low"][i])
            adv_px = float(arrays["low"][i] if side > 0 else arrays["high"][i])
            if side > 0:
                fav_raw = (fav_px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12)
                adv_raw = (adv_px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12)
            else:
                fav_raw = (entry_price - fav_px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
                adv_raw = (entry_price - adv_px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            hit_tp = fav_raw * notional >= tp
            hit_sl = adv_raw * notional <= -sl
            if hit_tp or hit_sl:
                first_hilo_i = i
                # Conservative same-bar tie handling: SL wins.
                first_hilo_reason = "stop_loss" if hit_sl else "take_profit"
                break
        rows.append(
            {
                "trade_id": int(trade_id),
                "entry_i": entry_i,
                "exit_i": exit_i,
                "ledger_reason": str(row["exit_reason"]),
                "first_hilo_i": first_hilo_i,
                "first_hilo_reason": first_hilo_reason,
                "hilo_before_exit": bool(first_hilo_i is not None and first_hilo_i < exit_i),
                "hilo_reason_diff": bool(first_hilo_reason is not None and first_hilo_reason != str(row["exit_reason"])),
            }
        )
    return pd.DataFrame(rows)


def _forbidden_feature_scan(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    deny_exact = {"tp_sl_action_score"}
    deny_prefix = ("clean_regime4_", "regime4_pred_", "teacher_")
    deny_tokens = ("future", "target", "label", "pnl", "zigzag")
    out: dict[str, Any] = {}
    for split, payload in data.items():
        state_cols = list(payload["state"].columns)
        dec_cols = list(payload["dec"].columns)
        bad_state = [
            c for c in state_cols
            if c in deny_exact or any(c.startswith(p) for p in deny_prefix) or any(t in c.lower() for t in deny_tokens)
        ]
        bad_dec = [
            c for c in dec_cols
            if c in deny_exact or any(c.startswith(p) for p in deny_prefix) or any(t in c.lower() for t in deny_tokens)
        ]
        active = payload["dec"].loc[runner.base.omega._active(payload["dec"])]
        out[split] = {
            "rows": int(len(payload["frame"])),
            "first_timestamp": str(payload["frame"]["timestamp"].iloc[0]),
            "last_timestamp": str(payload["frame"]["timestamp"].iloc[-1]),
            "state_col_count": int(len(state_cols)),
            "decision_active_count": int(len(active)),
            "forbidden_state_cols": bad_state,
            "forbidden_decision_cols": bad_dec,
            "active_side_counts": {str(k): int(v) for k, v in active["side"].value_counts().to_dict().items()},
            "active_expert_counts": {str(k): int(v) for k, v in active["router_expert"].value_counts().to_dict().items()},
        }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    bundle = joblib.load(TP_BUNDLE_PATH)

    original: dict[str, Any] = {}
    taker: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for split in ("validation", "oos"):
        metrics, ledger = decay._simulate(data[split], spec=decay.SPECS[0], tp_bundle=bundle)
        original[split] = metrics
        ledgers[f"{split}_original"] = ledger
        ledger.to_csv(OUT_DIR / f"{split}_original_limit_maker_ledger.csv", index=False)

    with _patched_execution(_market_taker_next_open):
        for split in ("validation", "oos"):
            metrics, ledger = decay._simulate(data[split], spec=decay.SPECS[0], tp_bundle=bundle)
            taker[split] = metrics
            ledgers[f"{split}_taker"] = ledger
            ledger.to_csv(OUT_DIR / f"{split}_market_taker_ledger.csv", index=False)

    intrabar: dict[str, Any] = {}
    for split in ("validation", "oos"):
        slip_eff = float(data[split]["slip"]) * 3.0
        scan = _ledger_intrabar_scan(data[split]["frame"], ledgers[f"{split}_original"], slip_eff=slip_eff)
        scan.to_csv(OUT_DIR / f"{split}_intrabar_touch_audit.csv", index=False)
        intrabar[split] = {
            "trades": int(len(scan)),
            "earlier_intrabar_touch_count": int(scan["hilo_before_exit"].sum()),
            "different_intrabar_reason_count": int(scan["hilo_reason_diff"].sum()),
        }

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_fail_do_not_use_as_clean_oos_baseline",
        "audited_model": "omega1_2_1_tp_runner_only_baseline_20260612 / baseline_wide_runner",
        "summary": {
            "clean_oos_verdict": "FAIL",
            "primary_reasons": [
                "TP-runner bundle and research rankings used 2026 OOS metrics for model/config selection.",
                "TP/SL checks are close-threshold based, not true intrabar barrier execution.",
                "Execution assumes next-bar open limit maker fill when touched; ledger records close price, not actual fill.",
                "Headline OOS is sparse and regime-dependent: 18 trades, mostly shorts in Jan-Feb 2026.",
            ],
        },
        "metrics": {
            "original_limit_maker": original,
            "market_taker_next_open": taker,
        },
        "intrabar_touch_audit": intrabar,
        "feature_contract_scan": _forbidden_feature_scan(data),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
