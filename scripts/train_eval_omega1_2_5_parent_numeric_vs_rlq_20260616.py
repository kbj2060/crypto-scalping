#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_2_tp_runner_cash_sleeve_20260615 as base  # noqa: E402
import train_eval_omega1_2_3_cash_sleeve_upgrade_20260615 as sleeve_up  # noqa: E402
from ensemble.train_rl_dsac_agent import (  # noqa: E402
    DistributionalTwinCritic,
    DSACRouter,
    GaussianActor,
)


MODEL_ID = "omega1_2_5_parent_numeric_vs_rlq_20260616"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DSAC_CKPT = ROOT / "data/ensemble/ckpt/best_dsac_agents_clean_retrain_v1.pth"
RISK_NAME = "base_tp026_sl014_n0405_h192"
ACTION_CASH = base.ACTION_CASH
ACTION_LONG = base.ACTION_LONG
ACTION_SHORT = base.ACTION_SHORT


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


def _runner_cfg() -> base.repair.RunnerConfig:
    report = json.loads(base.BASELINE_REPORT.read_text(encoding="utf-8"))
    sc = report["selected_config"]
    return base.repair.RunnerConfig(
        int(sc["candidate_id"]),
        str(sc["mode"]),
        float(sc["quality_min"]),
        float(sc["extend_mult"]),
        float(sc["floor_frac"]),
        int(sc["max_extensions"]),
    )


def _parent_features(payload: dict[str, Any]) -> pd.DataFrame:
    state = payload["state"].copy().reset_index(drop=True)
    drop = [c for c in state.columns if c.startswith("dec_")]
    drop += [c for c in state.columns if c in base.FORBIDDEN_FEATURE_EXACT or c.startswith(base.FORBIDDEN_FEATURE_PREFIXES)]
    out = state.drop(columns=sorted(set(drop)), errors="ignore")
    bad = [c for c in out.columns if c in base.FORBIDDEN_FEATURE_EXACT or c.startswith(base.FORBIDDEN_FEATURE_PREFIXES)]
    if bad:
        raise RuntimeError(f"forbidden parent feature columns: {bad[:20]}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _simulate_side_no_takeover(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    i: int,
    side: int,
    risk: base.SleeveRisk,
    fee_eff: float,
    slip_eff: float,
) -> dict[str, Any]:
    fill_i = min(int(i) + 1, len(frame) - 1)
    entry_px = base._exec_price(float(arrays["open"][fill_i]), int(side), slip_eff, entry=True)
    pos = base.Position(
        sleeve="label",
        side=int(side),
        entry_signal_i=int(i),
        entry_i=int(fill_i),
        entry_price=entry_px,
        entry_equity=1.0,
        notional=float(risk.notional),
        margin_notional=float(risk.notional),
        leverage=float(risk.leverage),
        take_profit=float(risk.take_profit),
        stop_loss=abs(float(risk.stop_loss)),
        floor_unreal=-abs(float(risk.stop_loss)),
        max_hold_bars=int(risk.max_hold_bars),
    )
    cash = 1.0 - float(fee_eff) * float(risk.notional)
    end_i = min(len(frame) - 2, fill_i + int(risk.max_hold_bars))
    target = 0.0
    mfe = 0.0
    mae = 0.0
    reason = "max_hold"
    bars = int(risk.max_hold_bars)
    for j in range(fill_i, end_i + 1):
        best, worst = base._bar_best_worst(arrays, pos, j, slip_eff)
        close_unreal = base._close_unreal(arrays, pos, j, slip_eff)
        mfe = max(mfe, best, close_unreal)
        mae = min(mae, worst, close_unreal)
        target = close_unreal
        bars = int(j) - int(fill_i)
        if worst <= -abs(float(risk.stop_loss)):
            target = -abs(float(risk.stop_loss))
            reason = "stop_loss"
            break
        if best >= float(risk.take_profit):
            target = float(risk.take_profit)
            reason = "take_profit"
            break
    exit_px = base._exit_price_from_unreal(pos, target)
    cash, _net_pct = base._runtime_close(cash, pos, exit_px, fee_eff)
    return {"net": float(cash - 1.0), "mfe": float(mfe), "mae": float(mae), "stop": int(reason == "stop_loss"), "bars": int(bars), "reason": reason}


def _utility_labels_all(payload: dict[str, Any], risk: base.SleeveRisk) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = payload["frame"].reset_index(drop=True)
    arrays = base.repair._arrays(frame)
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0
    rows: list[dict[str, Any]] = []
    max_i = len(frame) - int(risk.max_hold_bars) - 3
    for i in range(max(0, max_i)):
        long_d = _simulate_side_no_takeover(frame, arrays, i, 1, risk, fee_eff, slip_eff)
        short_d = _simulate_side_no_takeover(frame, arrays, i, -1, risk, fee_eff, slip_eff)
        def utility(d: dict[str, Any]) -> float:
            adverse = abs(min(float(d["mae"]), 0.0))
            time_frac = min(float(d["bars"]) / max(float(risk.max_hold_bars), 1.0), 1.0)
            return float(d["net"]) - 0.003 * int(d["stop"]) - 0.20 * adverse - 0.001 * time_frac
        rows.append(
            {
                "i": int(i),
                "long_utility": utility(long_d),
                "short_utility": utility(short_d),
                "long_net": float(long_d["net"]),
                "short_net": float(short_d["net"]),
                "long_stop": int(long_d["stop"]),
                "short_stop": int(short_d["stop"]),
            }
        )
    labels = pd.DataFrame(rows)
    diag = {
        "rows": int(len(labels)),
        "long_utility_mean": float(labels["long_utility"].mean()) if len(labels) else 0.0,
        "short_utility_mean": float(labels["short_utility"].mean()) if len(labels) else 0.0,
        "long_positive": int((labels["long_utility"] > 0.0).sum()) if len(labels) else 0,
        "short_positive": int((labels["short_utility"] > 0.0).sum()) if len(labels) else 0,
        "long_stop_rate": float(labels["long_stop"].mean()) if len(labels) else 0.0,
        "short_stop_rate": float(labels["short_stop"].mean()) if len(labels) else 0.0,
    }
    return labels, diag


def _load_dsac_critic() -> tuple[DistributionalTwinCritic, DSACRouter, dict[str, Any]]:
    ckpt = torch.load(DSAC_CKPT, map_location="cpu", weights_only=False)
    state_dim = int(ckpt.get("state_dim", 29) or 29)
    critic = DistributionalTwinCritic(state_dim=state_dim, hidden_dim=256, n_quantiles=32)
    critic.load_state_dict(ckpt["critic"])
    critic.eval()
    actor = GaussianActor(state_dim=state_dim, hidden_dim=256)
    if "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"], strict=False)
    actor.eval()
    router = DSACRouter(actor, device="cpu")
    return critic, router, {"checkpoint": str(DSAC_CKPT), "state_dim": state_dim, "keys": sorted(list(ckpt.keys()))}


def _dsac_states(frame: pd.DataFrame, router: DSACRouter) -> np.ndarray:
    states = []
    for _, row in frame.reset_index(drop=True).iterrows():
        states.append(router._build_compact_state(row.to_dict(), {}))
    return np.asarray(states, dtype=np.float32)


def _rlq_labels(payload: dict[str, Any], critic: DistributionalTwinCritic, router: DSACRouter) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = payload["frame"].reset_index(drop=True)
    states = _dsac_states(frame, router)
    actions = {"short": -0.45, "cash": 0.0, "long": 0.45}
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        s = torch.tensor(states, dtype=torch.float32)
        for name, action in actions.items():
            a = torch.full((len(states), 1), float(action), dtype=torch.float32)
            q1, q2 = critic(s, a)
            q = torch.minimum(q1.mean(dim=1), q2.mean(dim=1)).cpu().numpy().astype(np.float64)
            out[name] = q
    labels = pd.DataFrame(
        {
            "i": np.arange(len(frame), dtype=np.int64),
            "q_long": out["long"],
            "q_short": out["short"],
            "q_cash": out["cash"],
            "long_adv": out["long"] - out["cash"],
            "short_adv": out["short"] - out["cash"],
        }
    )
    diag = {
        "rows": int(len(labels)),
        "q_long_mean": float(labels["q_long"].mean()),
        "q_short_mean": float(labels["q_short"].mean()),
        "q_cash_mean": float(labels["q_cash"].mean()),
        "long_adv_mean": float(labels["long_adv"].mean()),
        "short_adv_mean": float(labels["short_adv"].mean()),
        "long_adv_positive": int((labels["long_adv"] > 0.0).sum()),
        "short_adv_positive": int((labels["short_adv"] > 0.0).sum()),
    }
    return labels, diag


def _fit_predict(x_val: pd.DataFrame, labels: pd.DataFrame, target: str, x_oos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    idx = labels["i"].to_numpy(dtype=np.int64)
    y = labels[target].to_numpy(dtype=np.float64)
    model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.035, max_leaf_nodes=11, l2_regularization=2.0, random_state=265016)
    model.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), y)
    val_pred = model.predict(x_val.to_numpy(dtype=np.float64)).astype(np.float64)
    oos_pred = model.predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64)
    return val_pred, oos_pred


def _actions_from_scores(long_s: np.ndarray, short_s: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    best_long = long_s >= short_s
    best = np.where(best_long, long_s, short_s)
    action = np.where(best > float(threshold), np.where(best_long, ACTION_LONG, ACTION_SHORT), ACTION_CASH).astype(np.int64)
    conf = np.clip((best - float(threshold)) / max(abs(np.nanpercentile(best, 95)), 1e-6), 0.0, 1.0).astype(np.float64)
    return action, conf


def _decisions_from_actions(payload: dict[str, Any], action: np.ndarray, conf: np.ndarray, risk: base.SleeveRisk) -> pd.DataFrame:
    dec = payload["dec"].copy().reset_index(drop=True)
    side = np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0)).astype(np.int64)
    dec["action"] = action.astype(np.int64)
    dec["side"] = side
    active = side != 0
    dec["notional_exposure"] = np.where(active, float(risk.notional), 0.0)
    dec["position_fraction"] = np.where(active, float(risk.notional), 0.0)
    dec["leverage"] = np.where(active, float(risk.leverage), 1.0)
    dec["take_profit"] = np.where(active, float(risk.take_profit), 0.0)
    dec["stop_loss"] = np.where(active, float(risk.stop_loss), 0.0)
    dec["max_hold_bars"] = np.where(active, int(risk.max_hold_bars), 0)
    dec["cooldown_bars"] = 0
    dec["quality_score"] = np.where(active, np.maximum(conf, 0.0), 0.0)
    dec["confidence"] = np.where(active, np.maximum(conf, 0.0), 0.0)
    dec["router_expert"] = "numeric_parent"
    return dec


def _with_dec(payload: dict[str, Any], dec: pd.DataFrame) -> dict[str, Any]:
    return {
        "frame": payload["frame"].reset_index(drop=True),
        "dec": dec.reset_index(drop=True),
        "state": payload["state"].reset_index(drop=True),
        "fee": float(payload["fee"]),
        "slip": float(payload["slip"]),
    }


def _eval_parent_and_sleeve(
    name: str,
    data: dict[str, dict[str, Any]],
    cfg: base.repair.RunnerConfig,
    risk: base.SleeveRisk,
    val_action: np.ndarray,
    val_conf: np.ndarray,
    oos_action: np.ndarray,
    oos_conf: np.ndarray,
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    val_dec = _decisions_from_actions(data["validation"], val_action, val_conf, risk)
    oos_dec = _decisions_from_actions(data["oos"], oos_action, oos_conf, risk)
    val_payload = _with_dec(data["validation"], val_dec)
    oos_payload = _with_dec(data["oos"], oos_dec)
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    val_m, val_ledger = base._simulate_combo(val_payload, cfg, None, None, None, 1.0)
    oos_m, oos_ledger = base._simulate_combo(oos_payload, cfg, None, None, None, 1.0)
    rows.append(_row(f"{name}_parent_only", "parent_only", val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos))
    ledgers[f"{name}_parent_only"] = (val_ledger, oos_ledger)

    x_val_sleeve = sleeve_up._enhanced_features(val_payload)
    x_oos_sleeve = sleeve_up._enhanced_features(oos_payload)
    labels, _diag = sleeve_up._label_table(val_payload, risk, 0.002)
    if len(labels) >= 500:
        idx = labels["i"].to_numpy(dtype=np.int64)
        y_long = np.zeros(len(x_val_sleeve), dtype=np.float64)
        y_short = np.zeros(len(x_val_sleeve), dtype=np.float64)
        y_long[idx] = labels["long_net"].to_numpy(dtype=np.float64)
        y_short[idx] = labels["short_net"].to_numpy(dtype=np.float64)
        sv_long, so_long, _ = sleeve_up._fit_predict_regressor("hgb", x_val_sleeve, y_long, idx, x_oos_sleeve, seed=265201)
        sv_short, so_short, _ = sleeve_up._fit_predict_regressor("hgb", x_val_sleeve, y_short, idx, x_oos_sleeve, seed=265202)
        for ev_min in (0.002, 0.004):
            fb_val_a, fb_val_c = sleeve_up._actions_from_ev(sv_long, sv_short, ev_min)
            fb_oos_a, fb_oos_c = sleeve_up._actions_from_ev(so_long, so_short, ev_min)
            val_m, val_ledger = base._simulate_combo(val_payload, cfg, risk, fb_val_a, fb_val_c, 0.0)
            oos_m, oos_ledger = base._simulate_combo(oos_payload, cfg, risk, fb_oos_a, fb_oos_c, 0.0)
            cand = f"{name}_parent_plus_sleeve_ev{ev_min:.3f}"
            rows.append(_row(cand, "parent_plus_retrained_cash_sleeve", val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos))
            ledgers[cand] = (val_ledger, oos_ledger)
    return rows, ledgers


def _row(candidate: str, family: str, val_m: dict[str, Any], val_ledger: pd.DataFrame, oos_m: dict[str, Any], oos_ledger: pd.DataFrame, base_val: dict[str, Any], base_oos: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "family": family}
    row.update(sleeve_up._row("val", val_m, val_ledger))
    row.update(sleeve_up._row("oos", oos_m, oos_ledger))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _runner_cfg()
    risk = [r for r in base.RISKS if r.name == RISK_NAME][0]
    data = base.legacy_runner._build()
    x_val = _parent_features(data["validation"])
    x_oos = _parent_features(data["oos"])
    if list(x_val.columns) != list(x_oos.columns):
        raise RuntimeError("parent feature columns mismatch")
    base_val, base_val_ledger = base._simulate_combo(data["validation"], cfg, None, None, None, 1.0)
    base_oos, base_oos_ledger = base._simulate_combo(data["oos"], cfg, None, None, None, 1.0)
    base_val_ledger.to_csv(OUT_DIR / "baseline_validation_ledger.csv", index=False)
    base_oos_ledger.to_csv(OUT_DIR / "baseline_oos_ledger.csv", index=False)

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    diagnostics: dict[str, Any] = {"risk": asdict(risk), "feature_count": int(x_val.shape[1]), "features": list(x_val.columns)}

    util_labels, util_diag = _utility_labels_all(data["validation"], risk)
    diagnostics["utility_labels"] = util_diag
    util_val_long, util_oos_long = _fit_predict(x_val, util_labels, "long_utility", x_oos)
    util_val_short, util_oos_short = _fit_predict(x_val, util_labels, "short_utility", x_oos)
    for thr in (0.000, 0.001, 0.002):
        va, vc = _actions_from_scores(util_val_long, util_val_short, thr)
        oa, oc = _actions_from_scores(util_oos_long, util_oos_short, thr)
        r, l = _eval_parent_and_sleeve(f"utility_thr{thr:.3f}", data, cfg, risk, va, vc, oa, oc, base_val, base_oos)
        rows.extend(r)
        ledgers.update(l)

    critic, router, dsac_meta = _load_dsac_critic()
    rlq_labels, rlq_diag = _rlq_labels(data["validation"], critic, router)
    diagnostics["rlq_labels"] = rlq_diag
    diagnostics["rlq_source"] = dsac_meta
    rlq_val_long, rlq_oos_long = _fit_predict(x_val, rlq_labels, "long_adv", x_oos)
    rlq_val_short, rlq_oos_short = _fit_predict(x_val, rlq_labels, "short_adv", x_oos)
    positives = np.r_[rlq_val_long[rlq_val_long > 0], rlq_val_short[rlq_val_short > 0]]
    q_thresholds = [0.0]
    if len(positives):
        q_thresholds.extend([float(np.quantile(positives, q)) for q in (0.25, 0.50)])
    for thr in q_thresholds:
        va, vc = _actions_from_scores(rlq_val_long, rlq_val_short, thr)
        oa, oc = _actions_from_scores(rlq_oos_long, rlq_oos_short, thr)
        r, l = _eval_parent_and_sleeve(f"rlq_thr{thr:.6f}", data, cfg, risk, va, vc, oa, oc, base_val, base_oos)
        rows.extend(r)
        ledgers.update(l)

    ranking = pd.DataFrame(rows)
    ranking["selection_score_val_only"] = (
        ranking["val_delta_pnl"].fillna(0.0)
        + 0.5 * ranking["val_fallback_pnl"].fillna(0.0)
        - 20.0 * ranking["val_fallback_stop_rate"].fillna(0.0)
    )
    ranking = ranking.sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "parent_numeric_vs_rlq_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_delta_pnl", "oos_pnl"], ascending=False).iloc[0].to_dict()
    for prefix, row in (("selected", selected), ("best_oos_diagnostic", best_oos)):
        cand = str(row["candidate"])
        if cand in ledgers:
            v, o = ledgers[cand]
            v.to_csv(OUT_DIR / f"{prefix}_validation_ledger.csv", index=False)
            o.to_csv(OUT_DIR / f"{prefix}_oos_ledger.csv", index=False)

    blockers: list[str] = []
    forbidden = [c for c in x_val.columns if c in base.FORBIDDEN_FEATURE_EXACT or c.startswith(base.FORBIDDEN_FEATURE_PREFIXES)]
    if forbidden:
        blockers.append(f"forbidden parent feature columns: {forbidden[:20]}")
    if len(ranking) == 0:
        blockers.append("no candidates produced")
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_probe" if not blockers else "redteam_fail",
        "baseline_model_id": base.BASELINE_ID,
        "method": "Compares deterministic numeric utility labels against DSAC critic Q-value/advantage labels for parent-only and parent+retrained-cash-sleeve experiments.",
        "selection_policy": "validation_only_no_oos_selection; OOS is diagnostic",
        "baseline": {"validation": base_val, "oos": base_oos},
        "diagnostics": diagnostics,
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not blockers,
        "redteam_blockers": blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "parent_numeric_vs_rlq_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos_diagnostic": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
