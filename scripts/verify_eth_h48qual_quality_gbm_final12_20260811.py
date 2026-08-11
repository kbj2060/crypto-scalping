# 재현성 참고: 이 스크립트는 h48qual 자체 리서치 패널(fa_features.parquet, 145컬럼)에 의존하는데,
# 이 파일은 세션 scratchpad에만 있고 레포에 커밋되지 않았다 -- 기존 FINAL12 dedup 작업과 동일한
# 재현성 갭. docs/experiments/eth_h48qual_final12_feature_selection_20260811.md 참고.
"""GBM으로 FINAL12 -> tb_long_quality/tb_short_quality 회귀를 실제로(오라클 아님, TRAIN만 학습)
학습해서, direction_head의 실제 예측(dir_action)과 결합했을 때 always_short/오라클과 비교해 얼마나
포착되는지 확인."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811 as h48  # noqa: E402
omega = h48.omega

train_all, eval_df, _ = omega._load_omega_frames()   # FINAL12 브릿지+파생컬럼까지 이미 반영된 프레임
SPLIT_TS = pd.Timestamp("2025-10-01")
train_raw = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = eval_df.reset_index(drop=True)

TB_DIR = ROOT / "tmp/eth_h384_conservative_triple_barrier_labels_20260811"
Q_COLS = ["timestamp", "tb_long_quality_h384_conservative", "tb_short_quality_h384_conservative"]
tb_train = pd.read_csv(TB_DIR / "train_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=Q_COLS)
tb_val = pd.read_csv(TB_DIR / "validation_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=Q_COLS)
tb_oos = pd.read_csv(TB_DIR / "oos_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=Q_COLS)

def build_xy(frame, tb):
    m = frame.merge(tb, on="timestamp", how="inner")
    X = m[h48.FINAL12].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, m["tb_long_quality_h384_conservative"].to_numpy(), m["tb_short_quality_h384_conservative"].to_numpy(), m["timestamp"]

Xtr, ytr_long, ytr_short, _ = build_xy(train_raw, tb_train)
print(f"TRAIN n={len(Xtr)}  FINAL12={h48.FINAL12}")

reg_long = HistGradientBoostingRegressor(max_depth=5, max_iter=300, random_state=260620).fit(Xtr, ytr_long)
reg_short = HistGradientBoostingRegressor(max_depth=5, max_iter=300, random_state=260620).fit(Xtr, ytr_short)

# TRAIN 자체 적합도(참고용)
from sklearn.metrics import r2_score
print(f"TRAIN R^2  long={r2_score(ytr_long, reg_long.predict(Xtr)):.4f}  short={r2_score(ytr_short, reg_short.predict(Xtr)):.4f}")

Xval, yval_long, yval_short, ts_val = build_xy(val_raw, tb_val)
Xoos, yoos_long, yoos_short, ts_oos = build_xy(oos_raw, tb_oos)
print(f"VAL R^2(홀드아웃)  long={r2_score(yval_long, reg_long.predict(Xval)):.4f}  short={r2_score(yval_short, reg_short.predict(Xval)):.4f}")
print(f"OOS R^2(홀드아웃)  long={r2_score(yoos_long, reg_long.predict(Xoos)):.4f}  short={r2_score(yoos_short, reg_short.predict(Xoos)):.4f}")

pred_val = pd.DataFrame({"timestamp": ts_val, "pred_long_q": reg_long.predict(Xval), "pred_short_q": reg_short.predict(Xval)})
pred_oos = pd.DataFrame({"timestamp": ts_oos, "pred_long_q": reg_long.predict(Xoos), "pred_short_q": reg_short.predict(Xoos)})
pred_val.to_csv("tmp/eth_h48qual_odyssey_regression_analysis_20260811/gbm_pred_val.csv", index=False)
pred_oos.to_csv("tmp/eth_h48qual_odyssey_regression_analysis_20260811/gbm_pred_oos.csv", index=False)
print("저장 완료")

print("\n=== 재시도: 훨씬 강한 정규화(early stopping, 얕은 트리, L2) ===")
from sklearn.metrics import r2_score
for depth, l2, lr in [(3, 1.0, 0.05), (2, 2.0, 0.03)]:
    rl = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, l2_regularization=l2,
                                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                                        max_iter=1000, random_state=260620).fit(Xtr, ytr_long)
    rs = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, l2_regularization=l2,
                                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                                        max_iter=1000, random_state=260620).fit(Xtr, ytr_short)
    print(f"depth={depth} l2={l2} lr={lr}  n_iter(long)={rl.n_iter_}  n_iter(short)={rs.n_iter_}")
    print(f"  TRAIN R^2  long={r2_score(ytr_long, rl.predict(Xtr)):.4f}  short={r2_score(ytr_short, rs.predict(Xtr)):.4f}")
    print(f"  VAL   R^2  long={r2_score(yval_long, rl.predict(Xval)):.4f}  short={r2_score(yval_short, rs.predict(Xval)):.4f}")
    print(f"  OOS   R^2  long={r2_score(yoos_long, rl.predict(Xoos)):.4f}  short={r2_score(yoos_short, rs.predict(Xoos)):.4f}")
