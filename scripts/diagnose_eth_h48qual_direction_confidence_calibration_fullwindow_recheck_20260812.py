"""2026-08-12 재검증: eth_h48qual_direction_confidence_calibration_20260811.md의 "학습" 수치는
train_predictions_q050.csv(2025-01~09, 78509행)로 계산됐는데, 이 파일은 2026-06-30 17:59에
export_omega4_parent_predictions_from_bundle_20260630.py가 risk-sidecar용 override 소스로
재생성한 것으로, 실제 모델이 학습한 전체 2024-01~2025-09(183936행) 중 43%만 반영한다 (2024년
전체 누락). 이 스크립트는 scripts/regenerate_eth_h48qual_fullwindow_train_predictions_20260812.py
가 재생성한 진짜 전체 구간 train_predictions_q050.csv(183936행, report.json과 정확히 일치
검증됨)로 "학습" 수치만 다시 계산한다. VAL/OOS는 원래 파일 그대로 사용(이미 정확한 것으로 확인됨).

원본 스크립트(diagnose_eth_h48qual_direction_confidence_calibration_20260811.py)의 최소-diff
변형 -- BUNDLE_DIR을 train 전용으로 하나 더 두는 것 외에는 로직 100% 동일."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path("/home/kbj20/crypto-scalping")
BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630"
TRAIN_BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630_fullwindow_predictions_recheck_20260812"
ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
TRAIN_END = pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")

ACTION_MAP = {"CASH": 0, "LONG": 1, "SHORT": 2}


def load_zigzag() -> pd.DataFrame:
    frames = [pd.read_csv(ZIGZAG_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action_name"], parse_dates=["timestamp"]) for y in (2024, 2025, 2026)]
    z = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return z.rename(columns={"zigzag_action_name": "true_action"})


def load_split(bundle_dir: Path, fname: str, prefix: str) -> pd.DataFrame:
    src = pd.read_csv(bundle_dir / fname, parse_dates=["timestamp"])
    out = pd.DataFrame({
        "timestamp": src["timestamp"],
        "p_cash": pd.to_numeric(src[f"{prefix}_dir_p_cash"], errors="raise"),
        "p_long": pd.to_numeric(src[f"{prefix}_dir_p_long"], errors="raise"),
        "p_short": pd.to_numeric(src[f"{prefix}_dir_p_short"], errors="raise"),
        "dir_action": pd.to_numeric(src[f"{prefix}_dir_action"], errors="raise"),
        "dir_confidence": pd.to_numeric(src[f"{prefix}_dir_confidence"], errors="raise"),
    })
    return out


zigzag = load_zigzag()
train = load_split(TRAIN_BUNDLE_DIR, "train_predictions_q050.csv", "omega1_regime3_expertdq_oof")
val = load_split(BUNDLE_DIR, "validation_predictions_q050.csv", "omega1_regime3_expertdq_oof")
oos = load_split(BUNDLE_DIR, "oos_predictions_q050.csv", "omega1_regime3_expertdq")

for name, df in [("train", train), ("val", val), ("oos", oos)]:
    df.sort_values("timestamp", inplace=True)

print("=== 정합성 체크: train 행수/구간 (183936행, 2024-01-01~2025-09-30 기대) ===")
print(f"  train: n={len(train)}  span={train['timestamp'].min()} .. {train['timestamp'].max()}")

print("\n=== 확신도 격차 재계산 (원본 문서 대비) ===")
for name, df, lo, hi in [("학습(전체)", train, None, TRAIN_END), ("VAL", val, VAL_START, VAL_END), ("OOS", oos, OOS_START, OOS_END)]:
    d = df if lo is None else df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
    long_conf = d.loc[d["dir_action"] == 1, "dir_confidence"]
    short_conf = d.loc[d["dir_action"] == 2, "dir_confidence"]
    print(f"  {name}: n_long={len(long_conf)} n_short={len(short_conf)} "
          f"롱평균={long_conf.mean():.4f} 숏평균={short_conf.mean():.4f} 격차={short_conf.mean()-long_conf.mean():+.4f}")
print("  (원본 문서 학습(2025-01~09만): n=25910/34064, 0.5725/0.6210, +0.0485)")

# ---------- 참라벨 조인 ----------
for name, df in [("train", train), ("val", val), ("oos", oos)]:
    df["true_action_name"] = df["timestamp"].map(zigzag.set_index("timestamp")["true_action"])
    n_missing = df["true_action_name"].isna().sum()
    if n_missing:
        print(f"  [경고] {name}: 참라벨 조인 실패 {n_missing}/{len(df)}건 (제외)")
    df.dropna(subset=["true_action_name"], inplace=True)
    df["true_action"] = df["true_action_name"].map(ACTION_MAP)
    df["correct"] = (df["dir_action"] == df["true_action"]).astype(int)


def reliability(df: pd.DataFrame, pred_class: int, n_bins: int = 10) -> pd.DataFrame:
    sub = df[df["dir_action"] == pred_class].copy()
    sub["bin"] = pd.qcut(sub["dir_confidence"], n_bins, duplicates="drop")
    g = sub.groupby("bin", observed=True).agg(mean_conf=("dir_confidence", "mean"), acc=("correct", "mean"), n=("correct", "size"))
    return g


def ece(df: pd.DataFrame, pred_class: int, n_bins: int = 10) -> float:
    g = reliability(df, pred_class, n_bins)
    total = g["n"].sum()
    return float((g["n"] / total * (g["mean_conf"] - g["acc"]).abs()).sum())


print("\n=== 클래스별 보정 곡선 (ECE = 확신도-정확도 가중평균 절대차, 낮을수록 잘 보정됨) ===")
for name, df, lo, hi in [("학습(전체)", train, None, TRAIN_END), ("VAL", val, VAL_START, VAL_END), ("OOS", oos, OOS_START, OOS_END)]:
    d = df if lo is None else df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
    for cls_name, cls_id in [("LONG", 1), ("SHORT", 2)]:
        sub = d[d["dir_action"] == cls_id]
        if len(sub) < 20:
            continue
        e = ece(d, cls_id)
        overall_conf = sub["dir_confidence"].mean()
        overall_acc = sub["correct"].mean()
        print(f"  {name:>8} {cls_name:>5}: n={len(sub):>6}  평균확신도={overall_conf:.4f}  실제정확도={overall_acc:.4f}  "
              f"과신={overall_conf-overall_acc:+.4f}  ECE={e:.4f}")

# 2024년 단독 구간도 별도로 (2025-01~09와 비교하기 위해)
print("\n=== 2024년 단독 vs 2025-01~09 단독 (같은 '학습' 안에서 신규 vs 기존 조각 대조) ===")
train_2024 = train[train["timestamp"] < "2025-01-01"]
train_2025h1 = train[(train["timestamp"] >= "2025-01-01") & (train["timestamp"] <= TRAIN_END)]
for name, d in [("2024(신규)", train_2024), ("2025-01~09(기존)", train_2025h1)]:
    for cls_name, cls_id in [("LONG", 1), ("SHORT", 2)]:
        sub = d[d["dir_action"] == cls_id]
        if len(sub) < 20:
            continue
        e = ece(d, cls_id)
        overall_conf = sub["dir_confidence"].mean()
        overall_acc = sub["correct"].mean()
        print(f"  {name:>16} {cls_name:>5}: n={len(sub):>6}  평균확신도={overall_conf:.4f}  실제정확도={overall_acc:.4f}  "
              f"과신={overall_conf-overall_acc:+.4f}  ECE={e:.4f}")

print("\n=== 클래스별 보정 곡선 상세 (학습 전체구간, 10-분위) ===")
for cls_name, cls_id in [("LONG", 1), ("SHORT", 2)]:
    print(f"  --- {cls_name} ---")
    g = reliability(train[train["timestamp"] <= TRAIN_END], cls_id)
    print(g.to_string())


# ---------- 클래스별 temperature scaling (학습 전체구간에서 적합, VAL/OOS에서 평가) ----------
def fit_temperature(probs: np.ndarray, true_idx: np.ndarray) -> float:
    logits = np.log(np.clip(probs, 1e-12, 1.0))

    def nll(T):
        scaled = logits / max(T, 1e-3)
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        soft = exp / exp.sum(axis=1, keepdims=True)
        p_true = soft[np.arange(len(true_idx)), true_idx]
        return -np.log(np.clip(p_true, 1e-12, 1.0)).mean()

    res = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded")
    return float(res.x)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


train_active = train[(train["timestamp"] <= TRAIN_END) & (train["dir_action"].isin([1, 2]))]

t_long = fit_temperature(train_active.loc[train_active["dir_action"] == 1, ["p_cash", "p_long", "p_short"]].to_numpy(),
                          train_active.loc[train_active["dir_action"] == 1, "true_action"].to_numpy().astype(int))
t_short = fit_temperature(train_active.loc[train_active["dir_action"] == 2, ["p_cash", "p_long", "p_short"]].to_numpy(),
                           train_active.loc[train_active["dir_action"] == 2, "true_action"].to_numpy().astype(int))
print(f"\n=== 학습 전체구간에서 적합한 클래스별 temperature (원본 문서: T_long=0.6245, T_short=0.9156) ===")
print(f"  T_long={t_long:.4f}  T_short={t_short:.4f}  (T=1.0이면 보정 불필요, T>1이면 원래 과신)")

print("\n=== VAL/OOS에서 보정 전후 ECE 비교 (argmax는 T로 안 바뀜, confidence 크기만 조정) ===")
for name, df, lo, hi in [("VAL", val, VAL_START, VAL_END), ("OOS", oos, OOS_START, OOS_END)]:
    d = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].copy()
    for cls_name, cls_id, T in [("LONG", 1, t_long), ("SHORT", 2, t_short)]:
        sub = d[d["dir_action"] == cls_id].copy()
        if len(sub) < 20:
            continue
        probs = sub[["p_cash", "p_long", "p_short"]].to_numpy()
        scaled = apply_temperature(probs, T)
        sub["dir_confidence_scaled"] = scaled.max(axis=1)
        assert (scaled.argmax(axis=1) == probs.argmax(axis=1)).all(), "temperature scaling changed argmax -- bug"
        before_ece = ece(d, cls_id)
        d2 = d.copy()
        d2.loc[d2["dir_action"] == cls_id, "dir_confidence"] = sub["dir_confidence_scaled"].to_numpy()
        after_ece = ece(d2, cls_id)
        print(f"  {name:>4} {cls_name:>5} (T={T:.3f}): ECE {before_ece:.4f} -> {after_ece:.4f}  "
              f"평균확신도 {sub['dir_confidence'].mean():.4f} -> {sub['dir_confidence_scaled'].mean():.4f}")
