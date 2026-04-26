import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

df = pd.read_csv("/home/llewyn/crypto-scalping/analysis/pred_30m_eval.csv")
print("=== 컬럼 ===")
print(df.columns.tolist())
print(f"\n=== 행수: {len(df)} ===")

# 최근 20개
print("\n=== 최근 20개 (y_now vs y_target vs predictions) ===")
cols = [c for c in ["ts","y_now","y_target","y_pred_ensemble","y_pred_naive","y_pred_ridge","y_pred_kernel_ridge","y_pred_knn_analog"] if c in df.columns]
print(df[cols].tail(20).to_string(index=False))

# 통계
print("\n=== 기본 통계 ===")
stat_cols = [c for c in ["y_now","y_target","y_pred_ensemble"] if c in df.columns]
print(df[stat_cols].describe().to_string())

# Delta analysis
df["delta_true"] = df["y_target"] - df["y_now"]
df["delta_pred"] = df["y_pred_ensemble"] - df["y_now"]
print("\n=== Delta(30min) 통계 ===")
print(df[["delta_true","delta_pred"]].describe().to_string())

# Splits
n = len(df)
tr = int(n*0.7)
va = int(n*0.85)

# Direction accuracy
print("\n=== 방향 예측 정확도 ===")
for name, sl in [("전체",slice(None)),("train",slice(0,tr)),("valid",slice(tr,va)),("test",slice(va,None))]:
    sub = df.iloc[sl]
    act = np.sign(sub["delta_true"].values)
    prd = np.sign(sub["delta_pred"].values)
    acc = (act == prd).mean() * 100
    print(f"  [{name:5s}]: {acc:.1f}%  ({len(sub)} samples)")

# Per-model direction accuracy (test only)
test_df = df.iloc[va:]
act_te = np.sign(test_df["delta_true"].values)
print("\n=== 모델별 방향 정확도 (테스트셋) ===")
for c in df.columns:
    if c.startswith("y_pred_"):
        model = c.replace("y_pred_","")
        pred_d = np.sign(test_df[c].values - test_df["y_now"].values)
        acc = (act_te == pred_d).mean() * 100
        print(f"  [{model:14s}]: {acc:.1f}%")

# Delta correlation
print("\n=== delta 상관계수 ===")
print(f"  전체: {df['delta_true'].corr(df['delta_pred']):.6f}")
print(f"  test: {test_df['delta_true'].corr(test_df['delta_pred']):.6f}")

# RMSE / R2 comparison
print("\n=== RMSE / R2 비교 (ensemble vs naive) ===")
for name, sl in [("전체",slice(None)),("train",slice(0,tr)),("valid",slice(tr,va)),("test",slice(va,None))]:
    sub = df.iloc[sl]
    rmse_e = np.sqrt(mean_squared_error(sub["y_target"], sub["y_pred_ensemble"]))
    r2_e = r2_score(sub["y_target"], sub["y_pred_ensemble"])
    rmse_n = np.sqrt(mean_squared_error(sub["y_target"], sub["y_now"]))
    r2_n = r2_score(sub["y_target"], sub["y_now"])
    improve = (1 - rmse_e/rmse_n)*100
    print(f"  [{name:5s}] ensemble RMSE={rmse_e:.4f} R2={r2_e:.6f}  |  naive RMSE={rmse_n:.4f} R2={r2_n:.6f}  |  RMSE개선: {improve:+.2f}%")

# Per-model RMSE on test
print("\n=== 모델별 테스트 RMSE ===")
for c in df.columns:
    if c.startswith("y_pred_"):
        model = c.replace("y_pred_","")
        rmse = np.sqrt(mean_squared_error(test_df["y_target"], test_df[c]))
        r2 = r2_score(test_df["y_target"], test_df[c])
        print(f"  [{model:14s}]: RMSE={rmse:.4f}  R2={r2:.6f}")

# 처음/중간/끝 구간의 예측 vs 실제 비교
print("\n=== 시간대별 예측 정확도 (테스트셋 분할) ===")
te_n = len(test_df)
for name, sl in [("test_앞",slice(0,te_n//3)),("test_중",slice(te_n//3,2*te_n//3)),("test_뒤",slice(2*te_n//3,None))]:
    sub = test_df.iloc[sl]
    rmse = np.sqrt(mean_squared_error(sub["y_target"], sub["y_pred_ensemble"]))
    r2 = r2_score(sub["y_target"], sub["y_pred_ensemble"])
    rmse_n = np.sqrt(mean_squared_error(sub["y_target"], sub["y_now"]))
    dir_acc = (np.sign(sub["y_target"].values - sub["y_now"].values) == np.sign(sub["y_pred_ensemble"].values - sub["y_now"].values)).mean()*100
    print(f"  [{name:7s}] RMSE={rmse:.4f} R2={r2:.4f} 방향={dir_acc:.1f}% naive_RMSE={rmse_n:.4f}")
