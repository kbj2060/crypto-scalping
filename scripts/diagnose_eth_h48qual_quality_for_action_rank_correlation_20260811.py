"""이슈 8 진단: quality_for_action이 direction_head가 고른 클래스의 실현 결과와 순위상관이
있는가? 재학습 없음 — 이미 저장된 예측(quality_for_action 컬럼 포함)과 omega._metrics()의 검증된
시뮬 루프만 재사용한다.

핵심: final_action(게이트 통과 후)이 아니라 dir_action(게이트 전, direction_head 원본 픽)으로
포지션을 잡는다 — 그래야 threshold=0.50에서 이미 걸러진 서바이버만 보는 게 아니라, quality_for_action
전체 값 구간(게이트 실패할 값 포함)에서 실현 수익률과의 관계를 볼 수 있다. TP/SL/notional 등
리스크 템플릿은 실제 라이브와 동일한 BASE_TEMPLATE을 그대로 쓴다(가정하지 않음).

diagnose_risk_sidecar_calibration_20260707.py(L4 사이징 스코어 진단, spearmanr(score, trade_return))
와 같은 방법론을 quality_for_action에 적용."""
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


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


train_all = _read(TRAIN_CSV)
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)
fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def pre_gate_decisions(src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """_to_fixed_decisions()와 동일하되 final_action(게이트 후) 대신 dir_action(게이트 전)으로
    active를 정한다 -- quality_for_action(quality_score)은 그대로 컬럼에 남겨서 진입 시점에
    읽는다."""
    action = pd.to_numeric(src[f"{prefix}_dir_action"], errors="raise").to_numpy(dtype=np.int64)
    active = action != omega.ACTION_CASH
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    return pd.DataFrame({
        "action": action,
        "side": side,
        "notional_exposure": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "take_profit": np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0),
        "stop_loss": np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0),
        "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
        "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
        "quality_for_action": pd.to_numeric(src[f"{prefix}_quality_for_action"], errors="raise").to_numpy(dtype=np.float64),
    })


def trades_with_quality(frame: pd.DataFrame, dec: pd.DataFrame, *, fee, slip, cost_mult):
    """metrics_by_side()와 동일한 시뮬 루프에 진입 시점 quality_for_action + 거래별 순수익률만 추가로 기록."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = (dec["action"].to_numpy() != omega.ACTION_CASH) & (dec["side"].to_numpy() != 0) & (dec["notional_exposure"].to_numpy() > 0)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = 1.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    entry_quality = 0.0
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
                records.append({"entry_idx": entry_idx, "quality_for_action": entry_quality, "trade_return": (cash - entry_equity) / entry_equity})
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
        entry_quality = float(row["quality_for_action"])
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
        records.append({"entry_idx": entry_idx, "quality_for_action": entry_quality, "trade_return": (cash - entry_equity) / entry_equity})
    return records


for variant_name, cfg in VARIANTS.items():
    print(f"\n{'=' * 70}\n{variant_name}\n{'=' * 70}")
    for split_name, frame, oof, fname in [
        ("VAL", val_raw, True, "validation_predictions_q050.csv"),
        ("OOS", oos_raw, False, "oos_predictions_q050.csv"),
    ]:
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        per_seed_rho, pooled = [], []
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
            recs = trades_with_quality(f, dec, fee=fee, slip=slip, cost_mult=cost_mult)
            total_trades += len(recs)
            if len(recs) >= 10:
                q = [r["quality_for_action"] for r in recs]
                r = [r["trade_return"] for r in recs]
                rho, p = spearmanr(q, r)
                per_seed_rho.append(rho)
                print(f"  seed={seed:>7}  n={len(recs):>4}  rho={rho:+.4f}  p={p:.4f}")
            else:
                print(f"  seed={seed:>7}  n={len(recs):>4}  (10건 미만, 상관 생략)")
            pooled.extend(recs)

        if per_seed_rho:
            arr = np.array(per_seed_rho)
            print(f"  --- {split_name} 요약: 시드 {len(arr)}개, 총 거래 {total_trades}건 ---")
            print(f"      시드별 rho: 평균={arr.mean():+.4f}  중앙값={np.median(arr):+.4f}  "
                  f"양수 시드={int((arr > 0).sum())}/{len(arr)}")
            if len(pooled) >= 10:
                pq = [r["quality_for_action"] for r in pooled]
                pr = [r["trade_return"] for r in pooled]
                prho, pp = spearmanr(pq, pr)
                print(f"      풀링 rho(전 시드 합산, n={len(pooled)}) = {prho:+.4f}  p={pp:.4f}  "
                      f"(참고용 -- 시드 간 모델이 달라 개별 시드 rho가 1차 근거)")
        else:
            print(f"  --- {split_name}: 유효 시드 없음 ---")
