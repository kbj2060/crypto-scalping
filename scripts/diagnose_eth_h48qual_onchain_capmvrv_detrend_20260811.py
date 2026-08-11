"""사용자 지적: CapMVRVCur의 순위상관(0단계 진단, 15/15 시드 양수 OOS)이 진짜 진입-레벨 신호가
아니라 그냥 느리게 움직이는 장기 레짐/가격추세 프록시일 수 있다. 이 레포에 이미 같은 패턴의 선례가
있다 -- FINAL12 dedup에서 whale_retail_ratio가 가격추세 오염(corr(close)=+0.561)이 발견돼
detrend 버전(whale_retail_ratio_dt288)으로 교체됨(eth_h48qual_final12_feature_selection_20260811.md).

1단계: raw CapMVRVCur가 실제로 가격/시간과 얼마나 오염됐는지 직접 측정.
2단계: diff1(전일 대비 변화)과 7일 변화율, 두 detrend 버전으로 같은 순위상관 진단을 재실행해서
신호가 살아남는지 확인 -- 살아남으면 레벨이 아니라 "변화"에 신호가 있다는 뜻(진짜 진입-레벨
정보에 더 가까움), 사라지면 레벨 자체가 그냥 레짐 프록시였다는 뜻."""
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
ONCHAIN_CSV = ROOT / "data/onchain/coinmetrics/eth_onchain_daily.csv"


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


# ---------- 1단계: 오염 직접 측정 ----------
onchain = pd.read_csv(ONCHAIN_CSV, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
onchain["time"] = onchain["time"].dt.tz_localize(None)
onchain["t_ordinal"] = np.arange(len(onchain))
onchain["mvrv_diff1"] = onchain["CapMVRVCur"].diff(1)
onchain["mvrv_roc7"] = onchain["CapMVRVCur"].pct_change(7)

train_all = _read(TRAIN_CSV)
oos_all = _read(EVAL_CSV)
price_daily = pd.concat([train_all[["timestamp", "close"]], oos_all[["timestamp", "close"]]]).drop_duplicates("timestamp")
price_daily["date"] = price_daily["timestamp"].dt.normalize()
price_daily = price_daily.groupby("date")["close"].last().reset_index()

merged = onchain.merge(price_daily, left_on="time", right_on="date", how="inner")
val_win = merged[(merged["time"] >= "2025-10-01") & (merged["time"] <= "2025-12-31")]
oos_win = merged[(merged["time"] >= "2026-01-01") & (merged["time"] <= "2026-02-28")]

print("=== 1단계: raw CapMVRVCur 오염도 직접 측정 (일봉, VAL/OOS 구간) ===")
for name, win in [("VAL", val_win), ("OOS", oos_win)]:
    r_price, p_price = spearmanr(win["CapMVRVCur"], win["close"])
    r_time, p_time = spearmanr(win["CapMVRVCur"], win["t_ordinal"])
    print(f"  {name}: corr(CapMVRVCur, close)={r_price:+.3f}(p={p_price:.4f})   "
          f"corr(CapMVRVCur, 시간순번)={r_time:+.3f}(p={p_time:.4f})   n={len(win)}일")
print("  (참고: FINAL12 dedup에서 whale_retail_ratio가 corr(close)=+0.561로 오염 판정돼 교체됨)")
print()

# ---------- 2단계: detrend 버전으로 재진단 ----------
onchain_lag = onchain.copy()
onchain_lag["time"] = onchain_lag["time"] + pd.Timedelta(days=1)  # entry_date - 1일 이하 값만 인과적으로 관측 가능


def attach(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_date"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(None).dt.normalize()
    out = out.sort_values("_date")
    m = pd.merge_asof(out, onchain_lag, left_on="_date", right_on="time", direction="backward")
    return m.sort_values("timestamp").reset_index(drop=True)


val_raw = attach(train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True))
oos_raw = attach(oos_all)
fee, slip = omega._load_fee_slip()
cost_mult = 3.0
SIGNALS = ["CapMVRVCur", "mvrv_diff1", "mvrv_roc7"]
print(f"[결측] VAL: {val_raw[SIGNALS].isna().sum().to_dict()}  OOS: {oos_raw[SIGNALS].isna().sum().to_dict()}")


def pre_gate_decisions(src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    action = pd.to_numeric(src[f"{prefix}_dir_action"], errors="raise").to_numpy(dtype=np.int64)
    active = action != omega.ACTION_CASH
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    return pd.DataFrame({
        "action": action, "side": side,
        "notional_exposure": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "take_profit": np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0),
        "stop_loss": np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0),
        "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
        "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
    })


def trades_with_signal(frame: pd.DataFrame, dec: pd.DataFrame, *, fee, slip, cost_mult):
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    sig_arrays = {s: pd.to_numeric(frame[s], errors="raise").to_numpy(dtype=np.float64) for s in SIGNALS}
    active = (dec["action"].to_numpy() != omega.ACTION_CASH) & (dec["side"].to_numpy() != 0) & (dec["notional_exposure"].to_numpy() > 0)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = 1.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    entry_signal = {}
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
                filled, exit_px, exit_fee, _r = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                rec = {"trade_return": (cash - entry_equity) / entry_equity}
                rec.update(entry_signal)
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
        filled, px, entry_fee, _r = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = px
        entry_equity = cash
        entry_idx = int(i)
        entry_signal = {s: float(sig_arrays[s][i]) for s in SIGNALS}
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
        rec = {"trade_return": (cash - entry_equity) / entry_equity}
        rec.update(entry_signal)
        records.append(rec)
    return records


print("\n=== 2단계: 레벨 vs diff1(전일대비) vs 7일 변화율 순위상관 비교 ===")
for variant_name, cfg in VARIANTS.items():
    print(f"\n{'=' * 90}\n{variant_name}\n{'=' * 90}")
    for split_name, frame, oof, fname in [("VAL", val_raw, True, "validation_predictions_q050.csv"),
                                           ("OOS", oos_raw, False, "oos_predictions_q050.csv")]:
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        per_seed = {s: [] for s in SIGNALS}
        pooled = []
        for seed in cfg["seeds"]:
            d = RUN_ROOT / f"{cfg['tag']}{seed}"
            path = d / fname
            if not path.exists():
                continue
            src = pd.read_csv(path, parse_dates=["timestamp"])
            f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            dec = pre_gate_decisions(src_aligned, prefix)
            recs = trades_with_signal(f, dec, fee=fee, slip=slip, cost_mult=cost_mult)
            if len(recs) >= 10:
                r = [x["trade_return"] for x in recs]
                for s in SIGNALS:
                    sig = [x[s] for x in recs]
                    if any(np.isnan(v) for v in sig):
                        valid = [(a, b) for a, b in zip(sig, r) if not np.isnan(a)]
                        if len(valid) < 10:
                            continue
                        sig, r2 = zip(*valid)
                        rho, _p = spearmanr(sig, r2)
                    else:
                        rho, _p = spearmanr(sig, r)
                    per_seed[s].append(rho)
            pooled.extend(recs)
        print(f"  --- {split_name} (n_seed={len(per_seed['CapMVRVCur'])}) ---")
        for s in SIGNALS:
            arr = np.array(per_seed[s])
            if len(arr) == 0:
                continue
            pv = [(x[s], x["trade_return"]) for x in pooled if not np.isnan(x[s])]
            if len(pv) >= 10:
                psig, pret = zip(*pv)
                prho, pp = spearmanr(psig, pret)
            else:
                prho, pp = float("nan"), float("nan")
            print(f"      {s:>12}: 평균rho={arr.mean():+.4f}  중앙값={np.median(arr):+.4f}  "
                  f"양수시드={int((arr > 0).sum())}/{len(arr)}   풀링(n={len(pv)})rho={prho:+.4f} p={pp:.4f}")
