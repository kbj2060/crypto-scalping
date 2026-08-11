"""신규 데이터소스 후보 6(ETH 온체인) 진단: CoinMetrics 일봉 온체인 지표 6개
(AdrActCnt/CapMVRVCur/FlowInExNtv/FlowOutExNtv/SplyExNtv/TxCnt)가 quality_head 게이트 전
(dir_action) 진입의 실현 순수익률과 순위상관이 있는가. 이슈 8 진단
(diagnose_eth_h48qual_quality_for_action_rank_correlation_20260811.py)과 완전히 동일한 방법론
(dir_action 기준 pre-gate 시뮬레이션, spearmanr(신호, trade_return), 시드별 rho가 1차 근거)을
그대로 재사용하고, 신호만 quality_for_action 대신 온체인 6개 지표로 바꾼다.

인과성: 온체인 값은 일봉이라, 진입 시점의 날짜에서 최소 1일 지연된(entry_date - 1day 이하) 가장
최근 값만 사용한다(merge_asof backward, 1일 시프트 후). CoinMetrics 무료tier는 리비전 히스토리를
안 주므로(응답에 `-status: flash`/`-status-time` 붙음, research doc 참고) 완벽한 인과성 보장은
아니고 "그 시점에 최종적으로 어떤 값이었는가"에 가깝다 -- 진단 단계 캐비어트로 명시.

6개 지표 x 2 스플릿(VAL/OOS) x 2 라벨변형(h48orig/h384) = 24개 1차 상관 -- 다중비교 감안 필요."""
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
METRICS = ["AdrActCnt", "CapMVRVCur", "FlowInExNtv", "FlowOutExNtv", "SplyExNtv", "TxCnt"]


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _attach_onchain(frame: pd.DataFrame) -> pd.DataFrame:
    """entry_date - 1일 이하의 가장 최근 온체인 값만 인과적으로 붙인다(merge_asof backward)."""
    onchain = pd.read_csv(ONCHAIN_CSV, parse_dates=["time"])
    onchain["time"] = onchain["time"].dt.tz_localize(None) + pd.Timedelta(days=1)  # "이 값은 time+1일부터 관측 가능"
    onchain = onchain.sort_values("time").reset_index(drop=True)
    out = frame.copy()
    out["_date"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(None).dt.normalize()
    out = out.sort_values("_date")
    merged = pd.merge_asof(out, onchain, left_on="_date", right_on="time", direction="backward")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


train_all = _read(TRAIN_CSV)
val_raw = _attach_onchain(train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True))
oos_raw = _attach_onchain(_read(EVAL_CSV))
fee, slip = omega._load_fee_slip()
cost_mult = 3.0

print(f"[커버리지 체크] VAL 온체인 결측: {val_raw[METRICS].isna().any(axis=1).sum()}/{len(val_raw)}건, "
      f"OOS 온체인 결측: {oos_raw[METRICS].isna().any(axis=1).sum()}/{len(oos_raw)}건")


def pre_gate_decisions(src: pd.DataFrame, prefix: str) -> pd.DataFrame:
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
    })


def trades_with_onchain(frame: pd.DataFrame, dec: pd.DataFrame, *, fee, slip, cost_mult):
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    onchain_arrays = {m: pd.to_numeric(frame[m], errors="raise").to_numpy(dtype=np.float64) for m in METRICS}
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
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                rec = {"entry_idx": entry_idx, "trade_return": (cash - entry_equity) / entry_equity}
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
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = px
        entry_equity = cash
        entry_idx = int(i)
        entry_signal = {m: float(onchain_arrays[m][i]) for m in METRICS}
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
        rec.update(entry_signal)
        records.append(rec)
    return records


summary_rows = []
for variant_name, cfg in VARIANTS.items():
    print(f"\n{'=' * 90}\n{variant_name}\n{'=' * 90}")
    for split_name, frame, oof, fname in [
        ("VAL", val_raw, True, "validation_predictions_q050.csv"),
        ("OOS", oos_raw, False, "oos_predictions_q050.csv"),
    ]:
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        per_seed_rho = {m: [] for m in METRICS}
        pooled = []
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
            recs = trades_with_onchain(f, dec, fee=fee, slip=slip, cost_mult=cost_mult)
            if len(recs) >= 10:
                r = [x["trade_return"] for x in recs]
                line = f"  seed={seed:>7}  n={len(recs):>4} "
                for m in METRICS:
                    sig = [x[m] for x in recs]
                    rho, p = spearmanr(sig, r)
                    per_seed_rho[m].append(rho)
                    line += f" {m}={rho:+.3f}"
                print(line)
            else:
                print(f"  seed={seed:>7}  n={len(recs):>4} (10건 미만, 상관 생략)")
            pooled.extend(recs)

        print(f"  --- {split_name} 요약 (n_seed={len(per_seed_rho[METRICS[0]])}) ---")
        for m in METRICS:
            arr = np.array(per_seed_rho[m])
            if len(arr) == 0:
                continue
            pr = [x[m] for x in pooled]
            prr = [x["trade_return"] for x in pooled]
            prho, pp = (spearmanr(pr, prr) if len(pooled) >= 10 else (float("nan"), float("nan")))
            print(f"      {m:>14}: 평균rho={arr.mean():+.4f}  중앙값={np.median(arr):+.4f}  "
                  f"양수시드={int((arr > 0).sum())}/{len(arr)}   풀링rho(n={len(pooled)})={prho:+.4f} p={pp:.4f}")
            summary_rows.append({"variant": variant_name, "split": split_name, "metric": m,
                                  "mean_rho": arr.mean(), "median_rho": np.median(arr),
                                  "pos_seeds": int((arr > 0).sum()), "n_seeds": len(arr),
                                  "pooled_rho": prho, "pooled_p": pp, "pooled_n": len(pooled)})

out_path = ROOT / "tmp/eth_h48qual_odyssey_regression_analysis_20260811/onchain_rank_correlation.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(summary_rows).to_csv(out_path, index=False)
print(f"\n=== 저장: {out_path} ===")
