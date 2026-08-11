# 재현성 참고: 이 스크립트는 h48qual 자체 리서치 패널(fa_features.parquet, 145컬럼)에 의존하는데,
# 이 파일은 세션 scratchpad에만 있고 레포에 커밋되지 않았다 -- 기존 FINAL12 dedup 작업과 동일한
# 재현성 갭. docs/experiments/eth_h48qual_final12_feature_selection_20260811.md 참고.
"""재스크리닝(회귀 relevance)으로 뽑은 11개로 GBM 홀드아웃 재검증. gbm_quality_regressor.py와
동일 절차, FINAL12 대신 REL11 사용."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811 as h48  # noqa: E402
omega = h48.omega

SC = "tmp/eth_h48qual_odyssey_regression_analysis_20260811/"
OLD_SC = "/tmp/claude-1000/-home-kbj20-crypto-scalping/f6f0940b-7d19-44da-92ed-ad8db41aed03/scratchpad/"  # fa_features.parquet만
dedup = json.load(open(SC + "rescreen_quality_dedup_result.json"))
REL11 = dedup["final"]
print("REL11:", REL11)

# REL11 컬럼은 h48qual 패널(fa_features.parquet) 원본이거나 REPLACE 파생 -- 프로덕션
# TRAIN_CSV/EVAL_CSV엔 없을 수 있어서, h48qual 원본 fa_features.parquet 자체를 이번엔
# train/val/oos 전부에 대해 직접 쓴다(별도 패널 브릿지 없이).
h48panel = pd.read_parquet(OLD_SC + "fa_features.parquet")
RAW_NEEDED = {"funding_abs": "funding_abs_dt288", "sum_toptrader_long_short_ratio": "sum_toptrader_long_short_ratio_dt288"}
for raw, derived in RAW_NEEDED.items():
    src = h48panel[raw].astype(np.float64)
    h48panel[derived] = (src - src.rolling(288, min_periods=96).mean()).fillna(0.0)

# REL11 중 h48qual 패널에 없는 건 zig075 소스(브릿지 패널)에서 조인
missing = [c for c in REL11 if c not in h48panel.columns]
print("zig075 브릿지 필요:", missing)
zig = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", low_memory=False,
                   usecols=["timestamp"] + missing)
zig["timestamp"] = pd.to_datetime(zig["timestamp"])
h48panel = h48panel.merge(zig, on="timestamp", how="inner")

TB_DIR = ROOT / "tmp/eth_h384_conservative_triple_barrier_labels_20260811"
Q_COLS = ["timestamp", "tb_long_quality_h384_conservative", "tb_short_quality_h384_conservative"]
tb_train = pd.read_csv(TB_DIR / "train_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=Q_COLS)
tb_val = pd.read_csv(TB_DIR / "validation_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=Q_COLS)
tb_oos = pd.read_csv(TB_DIR / "oos_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=Q_COLS)
SPLIT_TS = pd.Timestamp("2025-10-01")

def build_xy(tb):
    m = h48panel.merge(tb, on="timestamp", how="inner")
    X = m[REL11].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, m["tb_long_quality_h384_conservative"].to_numpy(), m["tb_short_quality_h384_conservative"].to_numpy()

Xtr, ytr_l, ytr_s = build_xy(tb_train[tb_train.timestamp < SPLIT_TS])
Xval, yval_l, yval_s = build_xy(tb_val)
# fa_features.parquet는 2025년만 있어서 OOS(2026)는 못 만듦 -- VAL까지만 확인
print(f"TRAIN n={len(Xtr)}  VAL n={len(Xval)}")

rl = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, l2_regularization=2.0,
                                    early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                                    max_iter=1000, random_state=260620).fit(Xtr, ytr_l)
rs = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, l2_regularization=2.0,
                                    early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                                    max_iter=1000, random_state=260620).fit(Xtr, ytr_s)

print(f"TRAIN R^2  long={r2_score(ytr_l, rl.predict(Xtr)):.4f}  short={r2_score(ytr_s, rs.predict(Xtr)):.4f}")
print(f"VAL   R^2  long={r2_score(yval_l, rl.predict(Xval)):.4f}  short={r2_score(yval_s, rs.predict(Xval)):.4f}")
pl, ps = rl.predict(Xval), rs.predict(Xval)
print(f"VAL sign-AUC  long={roc_auc_score((yval_l>0).astype(int), pl):.4f}  short={roc_auc_score((yval_s>0).astype(int), ps):.4f}")
print("\n(참고: FINAL12 버전은 VAL R^2 long=-0.02~-0.12, short=-0.04~-0.14 였음)")
