"""팀장 리서치 문서 후보 2: Trust Score (Jiang, Kim, Guan, Gupta 2018, "To Trust Or Not To Trust
A Classifier"). 모델의 softmax를 전혀 안 쓰고 FINAL12 피쳐공간에서 TRAIN(zigzag_action 참라벨)
기준 클래스별 최근접이웃 거리만으로 신뢰도를 잰다:
  trust_score = dist(x, 가장가까운 비예측클래스) / dist(x, 가장가까운 예측클래스)

핵심 효율화: TRAIN 피쳐공간 kNN 인덱스와 VAL/OOS 각 bar의 클래스별 3-거리는 시드와 무관(같은
FINAL12 값, 같은 zigzag 라벨) -- 딱 한 번만 계산하고, 시드별로 달라지는 건 "그 bar에서 어느
클래스가 예측됐는가"(dir_action)뿐이라 여기서만 시드별 재계산."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811 as final12_script  # noqa: E402

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
ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
FINAL12 = final12_script.FINAL12


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def load_zigzag():
    frames = [pd.read_csv(ZIGZAG_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"]) for y in (2024, 2025, 2026)]
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


print("피쳐 프레임(FINAL12, JM 레짐 오버레이) 로드 중...", flush=True)
# final12_script의 오버레이(JM regime3-current, vwap/funding bridge, 파생 diff1/dt288)를 그대로 재사용
train_ov, eval_ov, _ = final12_script._load_omega_frames_final12()

zig = load_zigzag().set_index("timestamp")["zigzag_action"]
train_ov = train_ov.copy()
train_ov["zigzag_action"] = train_ov["timestamp"].map(zig)
train_ov = train_ov.dropna(subset=["zigzag_action"] + FINAL12)
train_raw = train_ov[train_ov["timestamp"] < SPLIT_TS].reset_index(drop=True)
val_raw = train_ov[train_ov["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = eval_ov.dropna(subset=FINAL12).reset_index(drop=True)
print(f"TRAIN n={len(train_raw)}  VAL n={len(val_raw)}  OOS n={len(oos_raw)}", flush=True)

scaler = StandardScaler().fit(train_raw[FINAL12].to_numpy(dtype=np.float64))
X_train = scaler.transform(train_raw[FINAL12].to_numpy(dtype=np.float64))
y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)

knn_by_class = {}
for cls in (0, 1, 2):
    mask = y_train == cls
    if mask.sum() < 5:
        raise RuntimeError(f"class {cls}: TRAIN 표본 부족 ({mask.sum()})")
    knn_by_class[cls] = NearestNeighbors(n_neighbors=1).fit(X_train[mask])
print(f"클래스별 TRAIN 표본: {[(c, int((y_train==c).sum())) for c in (0,1,2)]}", flush=True)


def class_distances(frame: pd.DataFrame) -> np.ndarray:
    """(n,3) -- 각 bar에서 클래스 0/1/2까지의 최근접 거리."""
    X = scaler.transform(frame[FINAL12].to_numpy(dtype=np.float64))
    out = np.zeros((len(frame), 3), dtype=np.float64)
    for cls in (0, 1, 2):
        d, _ = knn_by_class[cls].kneighbors(X, n_neighbors=1)
        out[:, cls] = d[:, 0]
    return out


val_dist = class_distances(val_raw)
oos_dist = class_distances(oos_raw)
val_raw = val_raw.reset_index(drop=True)
oos_raw = oos_raw.reset_index(drop=True)
print("클래스별 거리 계산 완료 (시드 무관, 1회만).", flush=True)

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def trust_scores_for_action(action: np.ndarray, dist: np.ndarray) -> np.ndarray:
    n = len(action)
    d_pred = dist[np.arange(n), action]
    d_other = np.full(n, np.inf)
    for cls in (0, 1, 2):
        mask = action != cls
        d_other[mask] = np.minimum(d_other[mask], dist[mask, cls])
    return d_other / np.maximum(d_pred, 1e-12)


def pre_gate_decisions(src: pd.DataFrame, prefix: str, trust: np.ndarray) -> pd.DataFrame:
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
        "trust_score": trust,
    })


def trades_with_trust(frame: pd.DataFrame, dec: pd.DataFrame, *, fee, slip, cost_mult):
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = (dec["action"].to_numpy() != omega.ACTION_CASH) & (dec["side"].to_numpy() != 0) & (dec["notional_exposure"].to_numpy() > 0)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = 1.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    entry_trust = 0.0
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
                records.append({"entry_idx": entry_idx, "trust_score": entry_trust, "trade_return": (cash - entry_equity) / entry_equity})
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
        entry_trust = float(row["trust_score"])
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
        records.append({"entry_idx": entry_idx, "trust_score": entry_trust, "trade_return": (cash - entry_equity) / entry_equity})
    return records


for variant_name, cfg in VARIANTS.items():
    print(f"\n{'=' * 70}\n{variant_name}\n{'=' * 70}")
    for split_name, frame, dist, oof, fname in [
        ("VAL", val_raw, val_dist, True, "validation_predictions_q050.csv"),
        ("OOS", oos_raw, oos_dist, False, "oos_predictions_q050.csv"),
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
            idx_map = frame.reset_index().set_index("timestamp")["index"]
            row_idx = src_aligned["timestamp"].map(idx_map).to_numpy()
            dist_aligned = dist[row_idx]
            action = pd.to_numeric(src_aligned[f"{prefix}_dir_action"], errors="raise").to_numpy(dtype=np.int64)
            trust = trust_scores_for_action(action, dist_aligned)
            dec = pre_gate_decisions(src_aligned, prefix, trust)
            recs = trades_with_trust(f, dec, fee=fee, slip=slip, cost_mult=cost_mult)
            total_trades += len(recs)
            if len(recs) >= 10:
                q = [r["trust_score"] for r in recs]
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
            print(f"      시드별 rho: 평균={arr.mean():+.4f}  중앙값={np.median(arr):+.4f}  양수 시드={int((arr > 0).sum())}/{len(arr)}")
            if len(pooled) >= 10:
                pq = [r["trust_score"] for r in pooled]
                pr = [r["trade_return"] for r in pooled]
                prho, pp = spearmanr(pq, pr)
                print(f"      풀링 rho(n={len(pooled)}) = {prho:+.4f}  p={pp:.4f}")
        else:
            print(f"  --- {split_name}: 유효 시드 없음 ---")
