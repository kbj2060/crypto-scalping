"""quality_threshold를 0.40~0.80까지 스윕하면서(원시 dir_action/quality_for_action은 고정, 게이트만
재적용) 롱/숏 비중과 방향별 승률, 전체 pnl이 어떻게 바뀌는지 확인. 15시드 합산."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

SEEDS = [260620, 481003, 26611, 903174, 155827, 44452, 51724, 179660, 240382, 375044, 378518, 692713, 711841, 750878, 821662]
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
TAG = "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2_e40_r30000_s"

TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


train_all = _read(TRAIN_CSV)
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)
fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def metrics_by_side(frame, dec, *, fee, slip, cost_mult):
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash, pos = 1.0, 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    notional = take_profit = stop_loss = 0.0
    max_hold = cooldown = next_cooldown = 0
    long_trades = long_wins = short_trades = short_wins = 0
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            hold = int(i) - int(entry_idx)
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit: reason = "tp"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss): reason = "sl"
            elif max_hold > 0 and hold >= max_hold: reason = "mh"
            if reason:
                filled, exit_px, exit_fee, _ = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                won = int(cash > entry_equity)
                if pos > 0: long_trades += 1; long_wins += won
                else: short_trades += 1; short_wins += won
                pos = 0; cooldown = int(next_cooldown); next_cooldown = 0
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _ = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos, entry_price, entry_equity, entry_idx = side, px, cash, int(i)
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        max_hold = int(row.get("max_hold_bars", 0) or 0)
        next_cooldown = int(row.get("cooldown_bars", 0) or 0)
        cash -= cash * entry_fee * notional
    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        won = int(cash > entry_equity)
        if pos > 0: long_trades += 1; long_wins += won
        else: short_trades += 1; short_wins += won
    pnl = float((cash - 1.0) * 100.0)
    return {"long_trades": long_trades, "long_wins": long_wins, "short_trades": short_trades, "short_wins": short_wins, "pnl": pnl}


def rebuild_dec(src_aligned, prefix, threshold):
    dir_action = pd.to_numeric(src_aligned[f"{prefix}_dir_action"], errors="raise").to_numpy(dtype=np.int64)
    qfa = pd.to_numeric(src_aligned[f"{prefix}_quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    final_action = dir_action.copy()
    final_action[(dir_action != 0) & (qfa < threshold)] = 0
    active = final_action != 0
    side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0)).astype(np.int64)
    router = src_aligned[f"{prefix}_router_expert"].astype(str).replace({"chop": "chop_expert"})
    dec = pd.DataFrame({
        "action": final_action, "side": side,
        "notional_exposure": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "leverage": np.where(active, float(omega.BASE_TEMPLATE["leverage"]), 1.0),
        "position_fraction": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "take_profit": np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0),
        "stop_loss": np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0),
        "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
        "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
        "router_expert": router.to_numpy(),
    })
    return omega._apply_expert_scale(dec)


results = []
cache = {}
for seed in SEEDS:
    d = RUN_ROOT / f"{TAG}{seed}"
    for split_name, frame, fname, prefix in [
        ("VAL", val_raw, "validation_predictions_q050.csv", "omega1_regime3_expertdq_oof"),
        ("OOS", oos_raw, "oos_predictions_q050.csv", "omega1_regime3_expertdq"),
    ]:
        src = pd.read_csv(d / fname, parse_dates=["timestamp"])
        f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        cache[(seed, split_name)] = (f, src_aligned, prefix)

for th in THRESHOLDS:
    for split_name in ["VAL", "OOS"]:
        agg = {"long_trades": 0, "long_wins": 0, "short_trades": 0, "short_wins": 0, "pnl_sum": 0.0, "n": 0}
        for seed in SEEDS:
            f, src_aligned, prefix = cache[(seed, split_name)]
            dec = rebuild_dec(src_aligned, prefix, th)
            r = metrics_by_side(f, dec, fee=fee, slip=slip, cost_mult=cost_mult)
            agg["long_trades"] += r["long_trades"]; agg["long_wins"] += r["long_wins"]
            agg["short_trades"] += r["short_trades"]; agg["short_wins"] += r["short_wins"]
            agg["pnl_sum"] += r["pnl"]; agg["n"] += 1
        total = agg["long_trades"] + agg["short_trades"]
        results.append({
            "threshold": th, "split": split_name,
            "long_trades": agg["long_trades"], "long_wr": agg["long_wins"] / max(agg["long_trades"], 1) * 100,
            "short_trades": agg["short_trades"], "short_wr": agg["short_wins"] / max(agg["short_trades"], 1) * 100,
            "short_pct": agg["short_trades"] / max(total, 1) * 100,
            "overall_wr": (agg["long_wins"] + agg["short_wins"]) / max(total, 1) * 100,
            "pnl_mean_over_15seed": agg["pnl_sum"] / agg["n"],
            "total_trades_15seed": total,
        })

df = pd.DataFrame(results)
pd.set_option("display.width", 220)
for split in ["VAL", "OOS"]:
    print(f"\n=== {split} ===")
    print(df[df.split == split].drop(columns=["split"]).to_string(index=False))
