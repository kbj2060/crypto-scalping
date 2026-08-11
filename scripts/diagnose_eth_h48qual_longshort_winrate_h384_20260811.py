"""테스트-h384(FINAL12+384bar, 15시드) 모델의 롱/숏 비율과 방향별 승률. omega._metrics()의 시뮬
루프를 그대로 복사해서 side별 trades/wins만 추가로 집계(원본 함수는 합산 wr만 반환)."""
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

def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

train_all = _read(TRAIN_CSV)
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def metrics_by_side(frame, dec, *, fee, slip, cost_mult):
    """omega._metrics()와 완전히 동일한 시뮬(엔트리/청산/비용 로직)에 side별 trades/wins만 추가."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    notional = leverage = take_profit = stop_loss = 0.0
    max_hold = cooldown = next_cooldown = 0
    long_trades = long_wins = short_trades = short_wins = 0
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            hold = int(i) - int(entry_idx)
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "max_hold"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                won = int(cash > entry_equity)
                if pos > 0:
                    long_trades += 1; long_wins += won
                else:
                    short_trades += 1; short_wins += won
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
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
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = px
        entry_equity = cash
        entry_idx = int(i)
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        max_hold = int(row.get("max_hold_bars", 0) or 0)
        next_cooldown = int(row.get("cooldown_bars", 0) or 0)
        cash -= cash * entry_fee * notional
    if pos != 0:
        fill_i = len(frame) - 1
        exit_px = omega._fill_price(arrays, fill_i, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        won = int(cash > entry_equity)
        if pos > 0:
            long_trades += 1; long_wins += won
        else:
            short_trades += 1; short_wins += won
    return {"long_trades": long_trades, "long_wins": long_wins, "short_trades": short_trades, "short_wins": short_wins}


rows = []
for seed in SEEDS:
    d = RUN_ROOT / f"{TAG}{seed}"
    for split_name, frame, oof, fname in [
        ("VAL", val_raw, True, "validation_predictions_q050.csv"),
        ("OOS", oos_raw, False, "oos_predictions_q050.csv"),
    ]:
        src = pd.read_csv(d / fname, parse_dates=["timestamp"])
        f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        dec = omega._to_fixed_decisions(src_aligned, oof=oof)
        r = metrics_by_side(f, dec, fee=fee, slip=slip, cost_mult=cost_mult)
        rows.append({"seed": seed, "split": split_name, **r})

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print()
for split in ["VAL", "OOS"]:
    sub = df[df.split == split]
    lt, lw = sub.long_trades.sum(), sub.long_wins.sum()
    st, sw = sub.short_trades.sum(), sub.short_wins.sum()
    total = lt + st
    print(f"=== {split} (15시드 합산) ===")
    print(f"  롱: {lt}건 ({lt/total*100:.1f}%)  승률={lw/max(lt,1)*100:.1f}%  ({lw}/{lt})")
    print(f"  숏: {st}건 ({st/total*100:.1f}%)  승률={sw/max(st,1)*100:.1f}%  ({sw}/{st})")
    print(f"  전체 승률={((lw+sw)/max(total,1))*100:.1f}%  ({lw+sw}/{total})")
