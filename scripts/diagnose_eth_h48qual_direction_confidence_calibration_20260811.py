"""사용자 요청: direction_head confidence를 방향별로 재보정 시도. eth_zigzag_swing_shape_
direction_asymmetry_check_20260811.md의 결론(스윙 형태 하나로는 confidence 격차 3~5pp를 다
설명 못 함, 남은 원인은 모델 자체의 calibration일 가능성)에 따라, 무작정 LONG/SHORT confidence를
같게 맞추는 게 아니라 -- 그러면 진짜 시장 정보(숏이 완결 전 되돌림이 덜하다는, 두 병행 진단이
교차검증한 사실)까지 지워버릴 위험이 있음 -- 먼저 클래스별로 확신도가 실제 정확도와 맞는지
(reliability/ECE)부터 확인한다. 과신(confidence > accuracy)이 실제로 있고 그 정도가 방향별로
다를 때만 그 차이만큼만 temperature scaling으로 보정한다.

데이터: 라이브 h48qual 번들(true_3head_tabm_bundle.pt, 2026-06-30 export, 재학습 없음) 저장
예측 + 정식 zigzag_action 라벨(build_wave3_action_labels_20260531.py 산출, 실제 학습 타겟) 조인.
confidence-echo 문서(eth_h48qual_zig075_direction_confidence_echo_check_20260811.md)와 동일
데이터 소스 -- Test 1 숫자 재현으로 정합성 검증 후 진행.

주의(중요, 이 세션에서 이미 확인됨): quality_head 게이트(quality_for_action)는 direction_head의
dir_confidence를 코드상 직접 읽지 않는다 -- quality_head 자체의 별도 softmax를 쓴다
(confidence-echo 문서 Test 2: 상관은 있지만 h48qual은 0.18~0.43로 완전 종속 아님). 즉 여기서
direction_head confidence를 재보정해도 라이브 게이트 출력이 자동으로 바뀌진 않는다 -- 이건
"direction_head 자체가 잘 보정됐는가"를 확인하는 진단이며, 실제 게이트에 반영하려면 별도의
설계 변경(게이트 공식 자체를 바꾸거나 quality_head를 재학습)이 추가로 필요하다."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path("/home/kbj20/crypto-scalping")
BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630"
ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
TRAIN_END = pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")

ACTION_MAP = {"CASH": 0, "LONG": 1, "SHORT": 2}


def load_zigzag() -> pd.DataFrame:
    frames = [pd.read_csv(ZIGZAG_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action_name"], parse_dates=["timestamp"]) for y in (2024, 2025, 2026)]
    z = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return z.rename(columns={"zigzag_action_name": "true_action"})


def load_split(fname: str, prefix: str) -> pd.DataFrame:
    src = pd.read_csv(BUNDLE_DIR / fname, parse_dates=["timestamp"])
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
train = load_split("train_predictions_q050.csv", "omega1_regime3_expertdq_oof")
val = load_split("validation_predictions_q050.csv", "omega1_regime3_expertdq_oof")
oos = load_split("oos_predictions_q050.csv", "omega1_regime3_expertdq")

for name, df in [("train", train), ("val", val), ("oos", oos)]:
    df.sort_values("timestamp", inplace=True)

# ---------- 정합성 체크: confidence-echo 문서 Test 1 숫자 재현 ----------
print("=== 정합성 체크 (confidence-echo 문서 Test 1과 대조) ===")
for name, df, lo, hi in [("학습", train, None, TRAIN_END), ("VAL", val, VAL_START, VAL_END), ("OOS", oos, OOS_START, OOS_END)]:
    d = df if lo is None else df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
    long_conf = d.loc[d["dir_action"] == 1, "dir_confidence"]
    short_conf = d.loc[d["dir_action"] == 2, "dir_confidence"]
    print(f"  {name}: n_long={len(long_conf)} n_short={len(short_conf)} "
          f"롱평균={long_conf.mean():.4f} 숏평균={short_conf.mean():.4f} 격차={short_conf.mean()-long_conf.mean():+.4f}")
print("  (참고 문서 값 -- 학습: n=25910/34064, 0.5725/0.6210, +0.0485 | VAL: n=8052/11303, 0.5676/0.6086 | OOS: n=5238/6746, 0.5642/0.6038)")

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
for name, df, lo, hi in [("학습", train, None, TRAIN_END), ("VAL", val, VAL_START, VAL_END), ("OOS", oos, OOS_START, OOS_END)]:
    d = df if lo is None else df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
    for cls_name, cls_id in [("LONG", 1), ("SHORT", 2)]:
        sub = d[d["dir_action"] == cls_id]
        if len(sub) < 20:
            continue
        e = ece(d, cls_id)
        overall_conf = sub["dir_confidence"].mean()
        overall_acc = sub["correct"].mean()
        print(f"  {name:>4} {cls_name:>5}: n={len(sub):>6}  평균확신도={overall_conf:.4f}  실제정확도={overall_acc:.4f}  "
              f"과신={overall_conf-overall_acc:+.4f}  ECE={e:.4f}")

print("\n=== 클래스별 보정 곡선 상세 (학습구간, 10-분위) ===")
for cls_name, cls_id in [("LONG", 1), ("SHORT", 2)]:
    print(f"  --- {cls_name} ---")
    g = reliability(train[train["timestamp"] <= TRAIN_END], cls_id)
    print(g.to_string())


# ---------- 클래스별 temperature scaling (학습구간에서 적합, VAL/OOS에서 평가) ----------
def pseudo_logits(row_probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(row_probs, 1e-12, 1.0))


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
probs_all = train_active[["p_cash", "p_long", "p_short"]].to_numpy()
true_all = train_active["true_action"].to_numpy().astype(int)

t_long = fit_temperature(train_active.loc[train_active["dir_action"] == 1, ["p_cash", "p_long", "p_short"]].to_numpy(),
                          train_active.loc[train_active["dir_action"] == 1, "true_action"].to_numpy().astype(int))
t_short = fit_temperature(train_active.loc[train_active["dir_action"] == 2, ["p_cash", "p_long", "p_short"]].to_numpy(),
                           train_active.loc[train_active["dir_action"] == 2, "true_action"].to_numpy().astype(int))
print(f"\n=== 학습구간에서 적합한 클래스별 temperature ===")
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
