"""신규 데이터소스 후보 4(거래소간 가격 basis) 순위상관 진단: Binance-OKX ETH 퍼프 가격 괴리가
quality_head 게이트 전(dir_action) 진입의 실현 순수익률과 순위상관이 있는가. 이슈 8/candidate C/
candidate 6과 완전히 동일한 방법론(dir_action 기준 pre-gate 시뮬레이션, spearmanr(신호,
trade_return), 시드별 rho가 1차 근거).

candidate 6(온체인, CapMVRVCur)에서 확립된 절차([[feedback_raw_feature_price_trend_contamination]])를
그대로 적용 -- 순위상관 결과를 보고하기 전에 반드시 오염도(corr(price)/corr(시간순번))부터
직접 측정한다. basis는 정의상 상대값(두 거래소 가격차)이라 raw 레벨 지표보다는 오염 위험이
낮을 것으로 예상되지만, 가정하지 않고 검증한다.

인과성: basis는 온체인처럼 느리게 갱신되는 펀더멘털이 아니라 근실시간 시장 미시구조라, 진입
시점 기준 가장 최근 마감된 1시간봉의 basis 값만 사용(merge_asof backward, 최소 1시간 지연)."""
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
OKX_CSV = ROOT / "data/derivatives/okx_eth_hourly_klines.csv"


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


train_all = _read(TRAIN_CSV)
oos_all = _read(EVAL_CSV)

# ---------- basis 시계열 구성: binance 5분봉 -> 1시간 리샘플, okx 1시간봉과 조인 ----------
bnc = pd.concat([train_all[["timestamp", "close"]], oos_all[["timestamp", "close"]]]).drop_duplicates("timestamp").sort_values("timestamp")
bnc = bnc.set_index("timestamp")["close"].resample("1h").last().dropna().reset_index()
bnc.columns = ["hour", "binance_close"]
bnc["hour"] = bnc["hour"].dt.tz_localize("UTC") if bnc["hour"].dt.tz is None else bnc["hour"]

okx = pd.read_csv(OKX_CSV, parse_dates=["timestamp"])
okx = okx.rename(columns={"timestamp": "hour", "close": "okx_close"})[["hour", "okx_close"]]

basis = bnc.merge(okx, on="hour", how="inner")
basis["basis"] = (basis["binance_close"] - basis["okx_close"]) / basis["okx_close"]
basis["basis_abs"] = basis["basis"].abs()
basis = basis.sort_values("hour").reset_index(drop=True)
print(f"[basis 시계열] n={len(basis)}시간, 범위 {basis['hour'].min()}~{basis['hour'].max()}, "
      f"basis 평균={basis['basis'].mean():.6f} std={basis['basis'].std():.6f}")

# ---------- 오염도 직접 측정 (candidate 6 절차 그대로 선적용) ----------
val_win = basis[(basis["hour"] >= "2025-10-01") & (basis["hour"] <= "2025-12-31")]
oos_win = basis[(basis["hour"] >= "2026-01-01") & (basis["hour"] <= "2026-02-28")]
print("\n=== 오염도 체크 (basis, |basis| 각각) ===")
for name, win in [("VAL", val_win), ("OOS", oos_win)]:
    t_ord = np.arange(len(win))
    for col in ["basis", "basis_abs"]:
        r_price, p_price = spearmanr(win[col], win["binance_close"])
        r_time, p_time = spearmanr(win[col], t_ord)
        print(f"  {name} {col:>10}: corr(price)={r_price:+.3f}(p={p_price:.4f})  corr(시간순번)={r_time:+.3f}(p={p_time:.4f})  n={len(win)}")

# ---------- 진입 시점 조인 (인과적, 1시간 지연) ----------
basis_lag = basis.copy()
basis_lag["hour"] = basis_lag["hour"] + pd.Timedelta(hours=1)
SIGNALS = ["basis", "basis_abs"]


def attach(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_ts"] = pd.to_datetime(out["timestamp"])
    if out["_ts"].dt.tz is None:
        out["_ts"] = out["_ts"].dt.tz_localize("UTC")
    out = out.sort_values("_ts")
    m = pd.merge_asof(out, basis_lag[["hour"] + SIGNALS], left_on="_ts", right_on="hour", direction="backward")
    return m.sort_values("timestamp").reset_index(drop=True)


val_raw = attach(train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True))
oos_raw = attach(oos_all)
fee, slip = omega._load_fee_slip()
cost_mult = 3.0
print(f"\n[결측] VAL: {val_raw[SIGNALS].isna().sum().to_dict()}  OOS: {oos_raw[SIGNALS].isna().sum().to_dict()}")


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


print("\n=== 순위상관 진단 (basis, |basis|) ===")
summary_rows = []
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
                    rho, _p = spearmanr(sig, r)
                    per_seed[s].append(rho)
            pooled.extend(recs)
        print(f"  --- {split_name} (n_seed={len(per_seed['basis'])}) ---")
        for s in SIGNALS:
            arr = np.array(per_seed[s])
            if len(arr) == 0:
                continue
            pv = [(x[s], x["trade_return"]) for x in pooled]
            psig, pret = zip(*pv)
            prho, pp = spearmanr(psig, pret)
            print(f"      {s:>10}: 평균rho={arr.mean():+.4f}  중앙값={np.median(arr):+.4f}  "
                  f"양수시드={int((arr > 0).sum())}/{len(arr)}   풀링(n={len(pv)})rho={prho:+.4f} p={pp:.4f}")
            summary_rows.append({"variant": variant_name, "split": split_name, "signal": s,
                                  "mean_rho": arr.mean(), "median_rho": np.median(arr),
                                  "pos_seeds": int((arr > 0).sum()), "n_seeds": len(arr),
                                  "pooled_rho": prho, "pooled_p": pp, "pooled_n": len(pv)})

out_path = ROOT / "tmp/eth_h48qual_odyssey_regression_analysis_20260811/basis_rank_correlation.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(summary_rows).to_csv(out_path, index=False)
print(f"\n=== 저장: {out_path} ===")
