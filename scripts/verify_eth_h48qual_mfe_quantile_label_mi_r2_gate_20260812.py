"""오라클 라벨 문헌 리서치(docs/experiments/eth_h48qual_oracle_label_design_literature_research_
20260812.md) 권장안 4 실행: h48qual quality_head를 하드 임계값(TP/SL 히트) 대신 MFE(Maximum
Favorable Excursion) 연속/분위수(q10/50/90) 회귀로 전환했을 때, TabM 풀 학습 전에 저비용
MI/R² 사전 게이트만 먼저 통과하는지 확인. zigzag_action/h48_conservative/trend-scanning과
같은 "풀 학습까지 가서야 실패 확인" 패턴을 반복하지 않는 게 목적.

MFE 정의: `build_omega1_2_triple_barrier_labels_20260619.py`가 이미 계산해둔
`tb_long_mfe_h48_conservative`/`tb_short_mfe_h48_conservative`(48bar 배리어 윈도우 내
유리한 방향 최대 미실현 이탈폭, 원본 배리어 계산의 부산물로 이미 존재 -- 새로 만들지 않고
그대로 재사용)를 direction_head의 실제 선택(zigzag_action)에 맞춰 골라 쓴다: LONG이면
tb_long_mfe, SHORT이면 tb_short_mfe. CASH bar는 정의상 제외(방향이 없으면 MFE도 없음).

TRAIN/VAL/OOS 구간과 FINAL12 피쳐는 h48orig 파이프라인(_prepare_frames)을 그대로 재사용 --
trend-scanning 게이트와 동일 관례로 직접 비교 가능. GBM 홀드아웃 R² 게이트도 동일한 두 설정
(약한 정규화 depth=5 / 강한 정규화 depth=2+early stopping)을 그대로 쓴다. MFE는 항상 >=0이라
"부호-AUC" 대신 Spearman 순위상관(예측 MFE 순위가 실제 MFE 순위와 맞는지 -- 이 서브
프로젝트가 quality_for_action 류 진단에서 이미 여러 번 쓴 지표)을 쓴다. 추가로 이 후보 특유의
분위수 회귀(q10/50/90) 보정도 확인 -- 단순 평균 회귀보다 분위수 특화 적합이 나은지, 그리고
q10/q90 커버리지가 명목 수준에 가까운지."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_h48qual_final12_h48orig_20260811 as h48orig_mod  # noqa: E402

parent_script = h48orig_mod.parent_script
omega = h48orig_mod.omega
FINAL12 = h48orig_mod.FINAL12

OUT_DIR = ROOT / "tmp/eth_h48qual_mfe_quantile_mi_r2_gate_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TB_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619"
TB_FILES = {"train": "train_triple_barrier_labels.csv", "val": "validation_triple_barrier_labels.csv", "oos": "oos_triple_barrier_labels.csv"}
MFE_COLS = ["timestamp", "tb_long_mfe_h48_conservative", "tb_short_mfe_h48_conservative"]


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. FINAL12 프레임(h48orig 파이프라인) + 기존 계산된 MFE 컬럼 병합
# ---------------------------------------------------------------------------

log("FINAL12 프레임 로딩 (h48orig와 동일 파이프라인)...")
frames = parent_script._prepare_frames(
    disable_tp_sl=False,
    direction_label_dir=ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
    quality_mode="quality_label_action",
    quality_label_dir=ROOT / "tmp/eth_h48_conservative_orig_padded_to_zigzag_timestamps_20260811",
    quality_min_edge=0.0010,
    quality_max_mae=0.0100,
    quality_min_mfe_mae=1.20,
    quality_max_hold_bars=288,
)

log("기존 계산된 h48_conservative MFE 컬럼 로딩 (build_omega1_2_triple_barrier_labels_20260619.py 산출물, 재계산 없음)...")
mfe_frames = {}
for split, fname in TB_FILES.items():
    df = pd.read_csv(TB_DIR / fname, usecols=MFE_COLS, parse_dates=["timestamp"])
    mfe_frames[split] = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    log(f"  {split}: {len(mfe_frames[split])}행")

train_raw = frames["train_raw"].merge(mfe_frames["train"], on="timestamp", how="inner")
val_raw = frames["val_raw"].merge(mfe_frames["val"], on="timestamp", how="inner")
oos_raw = frames["oos_raw"].merge(mfe_frames["oos"], on="timestamp", how="inner")

for name, before, after in [("train", frames["train_raw"], train_raw), ("val", frames["val_raw"], val_raw), ("oos", frames["oos_raw"], oos_raw)]:
    log(f"  {name}: FINAL12프레임 {len(before)}행 -> MFE 병합 후 {len(after)}행")


def pick_mfe(df: pd.DataFrame) -> pd.Series:
    action = df["zigzag_action"].to_numpy()
    long_mfe = df["tb_long_mfe_h48_conservative"].to_numpy()
    short_mfe = df["tb_short_mfe_h48_conservative"].to_numpy()
    out = np.where(action == 1, long_mfe, np.where(action == 2, short_mfe, np.nan))
    return pd.Series(out, index=df.index)


for df in (train_raw, val_raw, oos_raw):
    df["mfe_target"] = pick_mfe(df)

train_active = train_raw.dropna(subset=["mfe_target"]).reset_index(drop=True)
val_active = val_raw.dropna(subset=["mfe_target"]).reset_index(drop=True)
oos_active = oos_raw.dropna(subset=["mfe_target"]).reset_index(drop=True)

for name, full, active in [("TRAIN", train_raw, train_active), ("VAL", val_raw, val_active), ("OOS", oos_raw, oos_active)]:
    log(f"  {name}: active(LONG/SHORT) bar {len(active)}/{len(full)}행 ({len(active)/max(len(full),1)*100:.1f}%), "
        f"mfe_target 분포 mean={active['mfe_target'].mean():.4f} median={active['mfe_target'].median():.4f} "
        f"p10={active['mfe_target'].quantile(0.1):.4f} p90={active['mfe_target'].quantile(0.9):.4f}")

X_train = train_active[FINAL12].astype(np.float64)
X_val = val_active[FINAL12].astype(np.float64)
X_oos = oos_active[FINAL12].astype(np.float64)
y_train = train_active["mfe_target"].to_numpy()
y_val = val_active["mfe_target"].to_numpy()
y_oos = oos_active["mfe_target"].to_numpy()

# ---------------------------------------------------------------------------
# 2. MI 게이트
# ---------------------------------------------------------------------------

log("\n=== MI 게이트 (TRAIN, active bar만) ===")
mi_reg = mutual_info_regression(X_train.to_numpy(), y_train, discrete_features=False, random_state=0)
mi_report = {}
log(f"  {'피쳐':35s} {'MI(continuous mfe)':>20s}")
for c, mr in sorted(zip(FINAL12, mi_reg), key=lambda x: -x[1]):
    log(f"  {c:35s} {mr:20.4f}")
    mi_report[c] = {"mi_continuous_mfe": float(mr)}
(OUT_DIR / "mi_gate.json").write_text(json.dumps(mi_report, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 3. GBM 홀드아웃 R^2 게이트 (평균 회귀, quality_head/trend-scanning과 동일 두 설정)
# ---------------------------------------------------------------------------

log("\n=== GBM 홀드아웃 R^2 게이트 (연속 MFE 타겟, 평균 회귀) ===")
gbm_report = {}
configs = {
    "weak_reg_depth5": dict(max_depth=5, max_iter=300, random_state=260620),
    "strong_reg_depth2_earlystop": dict(max_depth=2, learning_rate=0.03, l2_regularization=2.0,
                                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                                         max_iter=1000, random_state=260620),
}
for cfg_name, cfg in configs.items():
    model = HistGradientBoostingRegressor(**cfg).fit(X_train, y_train)
    pred_train, pred_val, pred_oos = model.predict(X_train), model.predict(X_val), model.predict(X_oos)
    r2_train, r2_val, r2_oos = r2_score(y_train, pred_train), r2_score(y_val, pred_val), r2_score(y_oos, pred_oos)
    rho_val, p_val = spearmanr(y_val, pred_val)
    rho_oos, p_oos = spearmanr(y_oos, pred_oos)
    n_iter = getattr(model, "n_iter_", cfg.get("max_iter"))
    log(f"  [{cfg_name}] n_iter={n_iter}  TRAIN R2={r2_train:+.4f}  VAL R2={r2_val:+.4f}  OOS R2={r2_oos:+.4f}  "
        f"VAL spearman={rho_val:+.4f}(p={p_val:.3f})  OOS spearman={rho_oos:+.4f}(p={p_oos:.3f})")
    gbm_report[cfg_name] = {"n_iter": int(n_iter) if n_iter else None, "r2_train": float(r2_train), "r2_val": float(r2_val),
                             "r2_oos": float(r2_oos), "spearman_val": float(rho_val), "spearman_val_p": float(p_val),
                             "spearman_oos": float(rho_oos), "spearman_oos_p": float(p_oos)}
(OUT_DIR / "gbm_r2_gate.json").write_text(json.dumps(gbm_report, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 4. 이 후보 특유의 분위수(q10/50/90) 회귀 확인 -- pinball loss 기반 특화 적합이 평균 회귀보다
#    나은지, 그리고 커버리지(q10 아래로 실제값이 명목 10% 근처로 떨어지는지)가 맞는지 확인
# ---------------------------------------------------------------------------

log("\n=== 분위수(q10/50/90) 회귀 -- 평균 회귀 대비 개선 여부 + 커버리지 보정 확인 ===")
quantile_report = {}
for q in (0.1, 0.5, 0.9):
    qmodel = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=3, max_iter=300,
                                            learning_rate=0.05, random_state=260620).fit(X_train, y_train)
    pred_val = qmodel.predict(X_val)
    pred_oos = qmodel.predict(X_oos)
    cov_val = float((y_val <= pred_val).mean())
    cov_oos = float((y_oos <= pred_oos).mean())
    rho_val, _ = spearmanr(y_val, pred_val)
    rho_oos, _ = spearmanr(y_oos, pred_oos)
    log(f"  q={q:.1f}: VAL 커버리지={cov_val*100:5.1f}%(명목 {q*100:.0f}%)  OOS 커버리지={cov_oos*100:5.1f}%  "
        f"VAL spearman={rho_val:+.4f}  OOS spearman={rho_oos:+.4f}")
    quantile_report[f"q{int(q*100)}"] = {"coverage_val": cov_val, "coverage_oos": cov_oos,
                                          "spearman_val": float(rho_val), "spearman_oos": float(rho_oos)}
(OUT_DIR / "quantile_gate.json").write_text(json.dumps(quantile_report, indent=2, ensure_ascii=False))

log(f"\n출력 디렉토리: {OUT_DIR}")
log("게이트 판정 기준(권장, trend-scanning 게이트와 동일 기준 유지): VAL/OOS R^2가 유의미하게"
    " 0보다 크고(대략 >0.02~0.05 이상), spearman 순위상관이 유의하고 방향이 양(+)이어야"
    " TabM 풀 학습으로 승격할 근거가 있다고 본다. 분위수 커버리지는 보조 지표(명목수준과 크게"
    " 어긋나면 분위수 특화 적합 자체가 신뢰 못할 신호). 이 스크립트는 판정을 자동으로 내리지"
    " 않는다.")
