"""팀장 리서치 문서 후보 1: quality_head를 완전히 우회하고 direction_head 자신의 3-class softmax에서
나오는 표준 셀렉티브 분류 스코어(dir_confidence=max prob, margin=1등-2등 확률차, entropy=분포
엔트로피)를 실현수익률과 직접 순위상관 검정. 0단계 진단(quality_for_action)과 confidence-echo
문서 Test 2(quality_for_action vs dir_confidence 상관)는 있었지만, dir_confidence/margin/entropy
자체를 실현수익률과 직접 대조한 적은 이 세션에 없음 -- 이 빈틈을 닫는다.

scripts/diagnose_eth_h48qual_quality_for_action_rank_correlation_20260811.py의 최소-diff
변형(동일 방법론, 동일 시드셋, dir_action 기준 게이트 전 진입) -- quality_for_action 대신 세
스칼라를 동시에 기록해 시뮬레이션은 1회만 돈다."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
VARIANTS = {
    "h48orig (원본 48bar, 5시드)": {
        "tag": "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h48orig_20260811_r30000_s",
        "seeds": [260620, 481003, 26611, 903174, 155827],
    },
    "h384 v2 (384bar 재설계, 15시드)": {
        "tag": "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2_e40_r30000_s",
        "seeds": [260620, 481003, 26611, 903174, 155827, 44452, 51724, 179660, 240382, 375044, 378518, 692713, 711841, 750878, 821662],
    },
}

TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")

SCALARS = ["dir_confidence", "margin", "entropy"]


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


train_all = _read(TRAIN_CSV)
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)
fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def pre_gate_decisions(src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    action = pd.to_numeric(src[f"{prefix}_dir_action"], errors="raise").to_numpy(dtype=np.int64)
    active = action != omega.ACTION_CASH
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    p_cash = pd.to_numeric(src[f"{prefix}_dir_p_cash"], errors="raise").to_numpy(dtype=np.float64)
    p_long = pd.to_numeric(src[f"{prefix}_dir_p_long"], errors="raise").to_numpy(dtype=np.float64)
    p_short = pd.to_numeric(src[f"{prefix}_dir_p_short"], errors="raise").to_numpy(dtype=np.float64)
    probs = np.stack([p_cash, p_long, p_short], axis=1)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    dir_confidence = sorted_probs[:, 0]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    clipped = np.clip(probs, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    return pd.DataFrame({
        "action": action,
        "side": side,
        "notional_exposure": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "take_profit": np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0),
        "stop_loss": np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0),
        "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
        "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
        "dir_confidence": dir_confidence,
        "margin": margin,
        "entropy": entropy,
    })


def trades_with_scalars(frame: pd.DataFrame, dec: pd.DataFrame, *, fee, slip, cost_mult):
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = (dec["action"].to_numpy() != omega.ACTION_CASH) & (dec["side"].to_numpy() != 0) & (dec["notional_exposure"].to_numpy() > 0)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = 1.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    entry_scalars = {}
    notional = take_profit = stop_loss = 0.0
    max_hold = cooldown = next_cooldown = 0
    records = []
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
                rec = {"entry_idx": entry_idx, "trade_return": (cash - entry_equity) / entry_equity}
                rec.update(entry_scalars)
                records.append(rec)
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
        entry_scalars = {s: float(row[s]) for s in SCALARS}
        notional = float(omega.BASE_TEMPLATE["notional"])
        take_profit = float(omega.BASE_TEMPLATE["take_profit"])
        stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
        max_hold = int(omega.BASE_TEMPLATE["max_hold"])
        next_cooldown = int(omega.BASE_TEMPLATE["cooldown"])
        cash -= cash * entry_fee * notional
    if pos != 0:
        fill_i = len(frame) - 1
        exit_px = omega._fill_price(arrays, fill_i, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        rec = {"entry_idx": entry_idx, "trade_return": (cash - entry_equity) / entry_equity}
        rec.update(entry_scalars)
        records.append(rec)
    return records


for variant_name, cfg in VARIANTS.items():
    print(f"\n{'=' * 70}\n{variant_name}\n{'=' * 70}")
    for split_name, frame, oof, fname in [
        ("VAL", val_raw, True, "validation_predictions_q050.csv"),
        ("OOS", oos_raw, False, "oos_predictions_q050.csv"),
    ]:
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        per_seed_rho = {s: [] for s in SCALARS}
        pooled = []
        total_trades = 0
        for seed in cfg["seeds"]:
            d = RUN_ROOT / f"{cfg['tag']}{seed}"
            path = d / fname
            if not path.exists():
                print(f"  [스킵] seed={seed}: {path} 없음")
                continue
            src = pd.read_csv(path, parse_dates=["timestamp"])
            f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            dec = pre_gate_decisions(src_aligned, prefix)
            recs = trades_with_scalars(f, dec, fee=fee, slip=slip, cost_mult=cost_mult)
            total_trades += len(recs)
            if len(recs) >= 10:
                r = [rec["trade_return"] for rec in recs]
                line = f"  seed={seed:>7}  n={len(recs):>4}  "
                for s in SCALARS:
                    q = [rec[s] for rec in recs]
                    rho, p = spearmanr(q, r)
                    per_seed_rho[s].append(rho)
                    line += f"{s}: rho={rho:+.4f}(p={p:.3f})  "
                print(line)
            else:
                print(f"  seed={seed:>7}  n={len(recs):>4}  (10건 미만, 상관 생략)")
            pooled.extend(recs)

        print(f"  --- {split_name} 요약: 총 거래 {total_trades}건 ---")
        for s in SCALARS:
            arr = np.array(per_seed_rho[s])
            if len(arr) == 0:
                continue
            print(f"      [{s}] 시드별 rho 평균={arr.mean():+.4f}  중앙값={np.median(arr):+.4f}  양수 시드={int((arr > 0).sum())}/{len(arr)}", end="")
            if len(pooled) >= 10:
                pq = [rec[s] for rec in pooled]
                pr = [rec["trade_return"] for rec in pooled]
                prho, pp = spearmanr(pq, pr)
                print(f"  |  풀링 rho(n={len(pooled)})={prho:+.4f} p={pp:.4f}")
            else:
                print()
