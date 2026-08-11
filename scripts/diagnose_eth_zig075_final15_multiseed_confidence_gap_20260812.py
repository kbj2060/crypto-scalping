"""final15(zig075) N=5 다양시드 confidence 격차 재현성 체크. 병행 세션이 seed=260620 단일시드로
발견한 "confidence 격차 사실상 제거"(학습 +0.048→+0.0008)가 시드 노이즈가 아니라 진짜인지
[[tabm_hp_low_signal_pattern]] 정책(N>=5 다양시드 필요)에 따라 확인. 5개 시드
260620/481003/26611/903174/155827은 이 세션 전체에서 h48qual 진단에 써온 표준 다양시드 세트와
동일 -- 비교 가능성을 위해 재사용."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
SEEDS = [260620, 481003, 26611, 903174, 155827]
BASE = "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zig075_regime_jmredesign_20260810_final15"
SPLIT_TS_TRAIN_END = pd.Timestamp("2025-09-30 23:59:59")


def bundle_dir(seed: int) -> Path:
    if seed == 260620:
        return ROOT / BASE  # first/default seed has no _seedNNN suffix
    return ROOT / f"{BASE}_seed{seed}"


def load(path: Path, prefix: str) -> pd.DataFrame:
    src = pd.read_csv(path, parse_dates=["timestamp"])
    return pd.DataFrame({
        "timestamp": src["timestamp"],
        "dir_action": pd.to_numeric(src[f"{prefix}_dir_action"], errors="raise"),
        "dir_confidence": pd.to_numeric(src[f"{prefix}_dir_confidence"], errors="raise"),
    })


rows = []
missing = []
for seed in SEEDS:
    d = bundle_dir(seed)
    train_f = d / "train_predictions_q050.csv"
    val_f = d / "validation_predictions_q050.csv"
    oos_f = d / "oos_predictions_q050.csv"
    if not (train_f.exists() and val_f.exists() and oos_f.exists()):
        missing.append(seed)
        continue
    train = load(train_f, "omega1_regime3_expertdq_oof")
    val = load(val_f, "omega1_regime3_expertdq_oof")
    oos = load(oos_f, "omega1_regime3_expertdq")
    for name, df, lo, hi in [("TRAIN", train, None, SPLIT_TS_TRAIN_END),
                              ("VAL", val, pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")),
                              ("OOS", oos, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59"))]:
        dd = df if lo is None else df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
        long_conf = dd.loc[dd["dir_action"] == 1, "dir_confidence"]
        short_conf = dd.loc[dd["dir_action"] == 2, "dir_confidence"]
        rows.append({
            "seed": seed, "split": name,
            "n_long": len(long_conf), "n_short": len(short_conf),
            "long_conf": long_conf.mean(), "short_conf": short_conf.mean(),
            "gap": short_conf.mean() - long_conf.mean(),
        })

if missing:
    print(f"[대기 중] 아직 안 끝난 시드: {missing}")

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))

if not missing:
    print("\n=== 스플릿별 시드 평균 (N=5) ===")
    summary = df.groupby("split")["gap"].agg(["mean", "std", "min", "max"])
    print(summary.to_string())
    print("\n(참고: 단일시드 260620 원보고 -- TRAIN +0.0008, VAL +0.016, OOS +0.007)")
