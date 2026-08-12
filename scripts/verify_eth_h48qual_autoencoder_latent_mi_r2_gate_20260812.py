"""사용자 제안: 피쳐를 개별로 스크리닝하는 대신, 딥러닝(오토인코더)으로 넓은 원시 피쳐풀을
비선형 압축해 새로운 잠재 피쳐(latent factor)를 만들면 더 나은 신호가 나올 수 있는가?

이론적 배경(사용자에게 미리 설명한 내용): TabM 자체가 이미 PiecewiseLinearEmbeddings+앙상블
MLP로 FINAL12를 비선형 조합하는 딥러닝이고, N=40+48 시드로 이미 실패했다 -- data processing
inequality상 같은 입력의 어떤 변환도 그 입력이 갖지 않은 정보를 만들 순 없다. 그래서 이 실험의
진짜 차별점은 "비선형 조합 자체"가 아니라 "**FINAL12(12개)보다 훨씬 넓은 원시풀(145개)**을
직접 압축한다"는 점이다 -- FINAL12 자체가 12개로 캡핑되며 버린 정보가 있을 수 있다는 가설.

방법: 완전 비지도(재구성 손실만) 오토인코더를 direction-only 재스크리닝과 동일한 넓은 풀
(zig075 소스 패널, 145 raw 컬럼, deny-list 적용 후 ~139개)에 TRAIN(2024-06~2025-09)만으로
학습(라벨은 전혀 안 봄 -- 순수 비지도). 그 다음 인코더로 TRAIN/VAL/OOS 잠재벡터(16차원)를
추출해, 기존 진단들과 동일한 MI/R² 스타일 게이트로 zigzag_action과의 relevance를 확인한다.
이 단계 자체는 저비용 사전게이트(Optuna 없음, N=1 fit)로 유지 -- 신호가 있어 보일 때만 N≥5
시드 정식 검증으로 넘어간다는 이 세션의 표준 절차를 그대로 따른다."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path("/home/kbj20/crypto-scalping")
OUT_DIR = ROOT / "tmp/eth_h48qual_autoencoder_latent_gate_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")
CONTAMINATION_THRESHOLD = 0.561

FINAL12 = [
    "cvp_regime", "funding_pressure_diff1", "ou_halflife", "m7_vae_error_dt288",
    "realized_skewness", "mta_funding", "sig_whale_dt288", "sum_toptrader_long_short_ratio_dt288",
    "vwap_dist_24", "funding_roc_48", "breakout_strength",
    "regime3_current_sensitive_wide24_chop_prob",
]

DENY_PREFIXES = ("clean_regime4_", "regime4_pred_", "regime3_pred_", "teacher_", "teacher_oof_", "a5dir_")
DENY_TOKENS = ("target", "future", "label", "pnl", "zigzag", "wave3", "tp_sl_action_score")
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "open_btc", "high_btc", "low_btc", "close_btc"}
PRICE_LIKE = ["sum_open_interest_value"]
REPLACE = {
    "funding_pressure": ("funding_pressure_diff1", "diff1"),
    "last_funding_rate": ("last_funding_rate_dt288", "dt288"),
    "squeeze_power": ("squeeze_power_dt288", "dt288"),
    "long_squeeze_risk": ("long_squeeze_risk_dt288", "dt288"),
    "funding_abs": ("funding_abs_dt288", "dt288"),
    "whale_retail_ratio": ("whale_retail_ratio_dt288", "dt288"),
    "count_long_short_ratio": ("count_long_short_ratio_dt288", "dt288"),
    "sum_toptrader_long_short_ratio": ("sum_toptrader_long_short_ratio_dt288", "dt288"),
}
LATENT_DIM = 16
SEED = 260812


def is_candidate(col: str) -> bool:
    if col in NON_FEATURE or col in PRICE_LIKE or col in REPLACE:
        return False
    if any(col.startswith(p) for p in DENY_PREFIXES):
        return False
    if any(t in col for t in DENY_TOKENS):
        return False
    return True


def log(msg: str) -> None:
    print(msg, flush=True)


torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. 데이터 로딩 (direction-only 재스크리닝과 동일 풀/구간 -- 통제된 비교를 위해)
# ---------------------------------------------------------------------------

log("zig075 소스 패널 + zigzag_action 라벨 로딩...")
panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", low_memory=False)
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
panel = panel.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
labels = pd.concat([
    pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    for y in (2024, 2025, 2026)
], ignore_index=True).drop_duplicates("timestamp", keep="last")

df = panel.merge(labels, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
log(f"  병합 후 {len(df)}행")

candidate_raw = [c for c in df.columns if is_candidate(c) and pd.api.types.is_numeric_dtype(df[c])]
for raw, (derived, kind) in REPLACE.items():
    if raw not in df.columns:
        continue
    src = pd.to_numeric(df[raw], errors="coerce").astype(np.float64)
    if kind == "diff1":
        df[derived] = src.diff(1).fillna(0.0)
    elif kind == "dt288":
        df[derived] = (src - src.rolling(288, min_periods=96).mean()).fillna(0.0)
POOL = sorted(set(candidate_raw) | {d for d, _ in REPLACE.values() if d in df.columns})
log(f"  오토인코더 입력 풀: {len(POOL)}개 컬럼")

train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END)

X_all = df[POOL].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
y_all = pd.to_numeric(df["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
close_all = df["close"].to_numpy(dtype=np.float64)

# TRAIN 내부에서 시간순 마지막 15%를 오토인코더 자체 early-stopping용으로 분리 -- 실제
# VAL/OOS는 다운스트림 게이트까지 완전히 안 건드림(select-on-validation-only와 동일 정신,
# 비지도 재구성 단계에도 적용).
train_idx = np.flatnonzero(train_mask.to_numpy())
split_point = int(len(train_idx) * 0.85)
fit_idx, es_idx = train_idx[:split_point], train_idx[split_point:]
log(f"  TRAIN n={len(train_idx)} (fit {len(fit_idx)} / early-stop {len(es_idx)})  VAL n={val_mask.sum()}  OOS n={oos_mask.sum()}")

# 표준화: TRAIN(fit 구간)만으로 적합 -- 전체표본 스케일러 금지 규칙 준수
mean = X_all.iloc[fit_idx].mean()
std = X_all.iloc[fit_idx].std().replace(0.0, 1.0)
X_std = (X_all - mean) / std
X_std = X_std.clip(-10, 10)  # 극단치 폭주 방지

X_fit = torch.tensor(X_std.iloc[fit_idx].to_numpy(), dtype=torch.float32)
X_es = torch.tensor(X_std.iloc[es_idx].to_numpy(), dtype=torch.float32)
X_val_t = torch.tensor(X_std[val_mask.to_numpy()].to_numpy(), dtype=torch.float32)
X_oos_t = torch.tensor(X_std[oos_mask.to_numpy()].to_numpy(), dtype=torch.float32)
X_train_full_t = torch.tensor(X_std[train_mask.to_numpy()].to_numpy(), dtype=torch.float32)

# ---------------------------------------------------------------------------
# 2. 디노이징 오토인코더 (인코더 64->32->16, 디코더 16->32->64->입력차원)
# ---------------------------------------------------------------------------

INPUT_DIM = len(POOL)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


model = Autoencoder(INPUT_DIM, LATENT_DIM)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

loader = DataLoader(TensorDataset(X_fit), batch_size=2048, shuffle=True, generator=torch.Generator().manual_seed(SEED))

log(f"\n디노이징 오토인코더 학습 (input_dim={INPUT_DIM}, latent_dim={LATENT_DIM})...")
best_es_loss = float("inf")
best_state = None
patience, bad_epochs = 8, 0
t0 = time.time()
for epoch in range(200):
    model.train()
    for (batch,) in loader:
        noisy = batch + torch.randn_like(batch) * 0.05  # 디노이징: 입력에 소량 가우시안 노이즈
        opt.zero_grad()
        recon, _ = model(noisy)
        loss = loss_fn(recon, batch)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        es_recon, _ = model(X_es)
        es_loss = loss_fn(es_recon, X_es).item()
    if es_loss < best_es_loss - 1e-5:
        best_es_loss = es_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
    else:
        bad_epochs += 1
    if epoch % 10 == 0 or bad_epochs == 0:
        log(f"  epoch {epoch:3d}  early-stop recon MSE={es_loss:.4f}  best={best_es_loss:.4f}")
    if bad_epochs >= patience:
        log(f"  epoch {epoch}에서 조기종료 (patience={patience})")
        break
model.load_state_dict(best_state)
log(f"학습 완료 ({time.time()-t0:.0f}초), best early-stop recon MSE={best_es_loss:.4f}")

model.eval()
with torch.no_grad():
    recon_train, z_train = model(X_train_full_t)
    recon_val, z_val = model(X_val_t)
    recon_oos, z_oos = model(X_oos_t)
    mse_train = float(loss_fn(recon_train, X_train_full_t))
    mse_val = float(loss_fn(recon_val, X_val_t))
    mse_oos = float(loss_fn(recon_oos, X_oos_t))
log(f"\n재구성 MSE(일반화 체크): TRAIN={mse_train:.4f}  VAL={mse_val:.4f}  OOS={mse_oos:.4f}"
    f"  (VAL/OOS가 TRAIN보다 많이 크면 오토인코더 자체가 일반화 안 되는 신호)")

z_train_np, z_val_np, z_oos_np = z_train.numpy(), z_val.numpy(), z_oos.numpy()
y_train = y_all[train_mask.to_numpy()]
y_val = y_all[val_mask.to_numpy()]
y_oos = y_all[oos_mask.to_numpy()]
close_train_arr = close_all[train_mask.to_numpy()]

# ---------------------------------------------------------------------------
# 3. 잠재벡터 데이터 분석: MI vs zigzag_action, corr(close) 오염도
# ---------------------------------------------------------------------------

log("\n=== 잠재벡터(16차원) 분석: MI(zigzag_action, TRAIN) + corr(close) 오염도 ===")
mi_latent = mutual_info_classif(z_train_np, y_train, discrete_features=False, random_state=SEED)
latent_report = []
for i in range(LATENT_DIM):
    cc = float(np.corrcoef(z_train_np[:, i], close_train_arr)[0, 1])
    contaminated = abs(cc) > CONTAMINATION_THRESHOLD
    latent_report.append({"dim": i, "mi": float(mi_latent[i]), "corr_close": cc, "contaminated": contaminated})
latent_report.sort(key=lambda r: -r["mi"])
for r in latent_report:
    log(f"  latent[{r['dim']:2d}]  MI={r['mi']:.4f}  corr(close)={r['corr_close']:+.3f}{'  [오염]' if r['contaminated'] else ''}")

# 참고 비교: FINAL12 자체의 TRAIN MI(direction-only 재스크리닝 결과 재사용 -- 동일 풀/구간이라
# 직접 비교 가능)
final12_available = [c for c in FINAL12 if c in df.columns]
X_final12_train = df.loc[train_mask, final12_available].apply(pd.to_numeric, errors="coerce").fillna(0.0)
mi_final12 = mutual_info_classif(X_final12_train.to_numpy(), y_train, discrete_features=False, random_state=SEED)
log(f"\n참고: FINAL12(패널가용 {len(final12_available)}개) TRAIN MI 범위 = "
    f"{mi_final12.min():.4f} ~ {mi_final12.max():.4f} (cvp_regime 등 최고 0.41 수준, 앞선 진단과 일치)")

(OUT_DIR / "latent_analysis.json").write_text(json.dumps({
    "input_dim": INPUT_DIM, "latent_dim": LATENT_DIM,
    "recon_mse": {"train": mse_train, "val": mse_val, "oos": mse_oos},
    "latent_mi_corr": latent_report,
    "final12_mi_range": {"min": float(mi_final12.min()), "max": float(mi_final12.max())},
}, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 4. 다운스트림 게이트: FINAL12 단독 vs latent-only vs FINAL12+latent (튜닝 없음)
# ---------------------------------------------------------------------------

log("\n=== 다운스트림 LightGBM 홀드아웃 게이트 (튜닝 없음, direction-only 재스크리닝과 동일 방식) ===")
import lightgbm as lgb

X_final12_val = df.loc[val_mask, final12_available].apply(pd.to_numeric, errors="coerce").fillna(0.0)
X_final12_oos = df.loc[oos_mask, final12_available].apply(pd.to_numeric, errors="coerce").fillna(0.0)

latent_cols = [f"latent_{i}" for i in range(LATENT_DIM)]
Z_train_df = pd.DataFrame(z_train_np, columns=latent_cols)
Z_val_df = pd.DataFrame(z_val_np, columns=latent_cols)
Z_oos_df = pd.DataFrame(z_oos_np, columns=latent_cols)


def fit_eval(Xtr, Xv, Xo, label):
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=500, learning_rate=0.05,
                              num_leaves=31, random_state=SEED, verbosity=-1)
    clf.fit(Xtr, y_train, eval_set=[(Xv, y_val)], eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(30, verbose=False)])
    out = {}
    for sn, X, y_true in [("VAL", Xv, y_val), ("OOS", Xo, y_oos)]:
        pred = clf.predict(X)
        out[sn] = {"balanced_accuracy": float(balanced_accuracy_score(y_true, pred)), "macro_f1": float(f1_score(y_true, pred, average="macro"))}
        log(f"  [{label}] {sn}: balanced_acc={out[sn]['balanced_accuracy']:.3f}  macro_f1={out[sn]['macro_f1']:.3f}")
    return out


log("\n[FINAL12 단독]")
r_final12 = fit_eval(X_final12_train, X_final12_val, X_final12_oos, "FINAL12")

log(f"\n[latent-only ({LATENT_DIM}차원)]")
r_latent = fit_eval(Z_train_df, Z_val_df, Z_oos_df, "latent")

log(f"\n[FINAL12 + latent ({len(final12_available)}+{LATENT_DIM}={len(final12_available)+LATENT_DIM}차원)]")
Xtr_combo = pd.concat([X_final12_train.reset_index(drop=True), Z_train_df], axis=1)
Xv_combo = pd.concat([X_final12_val.reset_index(drop=True), Z_val_df], axis=1)
Xo_combo = pd.concat([X_final12_oos.reset_index(drop=True), Z_oos_df], axis=1)
r_combo = fit_eval(Xtr_combo, Xv_combo, Xo_combo, "FINAL12+latent")

(OUT_DIR / "holdout_comparison.json").write_text(json.dumps({
    "final12_only": r_final12, "latent_only": r_latent, "final12_plus_latent": r_combo,
}, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 5. 실제 거래 시뮬레이션 대조 (분류 지표 개선이 곧 PnL 개선은 아니라는 이 프로젝트의 핵심
# 교훈 -- classification metric만으로 성공을 주장하지 않는다. omega의 검증된 bar-by-bar 시뮬
# 재사용, GBDT 백본 진단과 동일 관례: 고정 TP=2.6%/SL=1.4%/notional=0.45/leverage=2.0,
# max_hold/cooldown=0, 레짐 expert_scale 미적용(대칭 비교), cost1/2/3 전부 확인.
# ---------------------------------------------------------------------------

log("\n=== 실제 거래 시뮬레이션 대조 (always-short/long, cost1/2/3) ===")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

fee, slip = omega._load_fee_slip()
COST_MULTS = {"cost1": 1.0, "cost2": 2.0, "cost3": 3.0}


def build_dec(action: np.ndarray) -> pd.DataFrame:
    action = action.astype(np.int64)
    active = action != omega.ACTION_CASH
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    return pd.DataFrame({
        "action": action, "side": side,
        "notional_exposure": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "leverage": np.where(active, float(omega.BASE_TEMPLATE["leverage"]), 1.0),
        "position_fraction": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "take_profit": np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0),
        "stop_loss": np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0),
        "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
        "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
    })


def forced_side(dec: pd.DataFrame, side_value: int) -> pd.DataFrame:
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out


omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

val_ohlc = df.loc[val_mask, ["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
oos_ohlc = df.loc[oos_mask, ["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)

pnl_rows = []
for combo_name, Xtr, Xv, Xo in [
    ("FINAL12", X_final12_train, X_final12_val, X_final12_oos),
    ("latent", Z_train_df, Z_val_df, Z_oos_df),
    ("FINAL12+latent", Xtr_combo, Xv_combo, Xo_combo),
]:
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=500, learning_rate=0.05,
                              num_leaves=31, random_state=SEED, verbosity=-1)
    clf.fit(Xtr, y_train, eval_set=[(Xv, y_val)], eval_metric="multi_logloss", callbacks=[lgb.early_stopping(30, verbose=False)])
    for split_name, X, ohlc in [("VAL", Xv, val_ohlc), ("OOS", Xo, oos_ohlc)]:
        pred = clf.predict(X)
        dec = build_dec(pred)
        for cost_name, cost_mult in COST_MULTS.items():
            m_model = omega._metrics(ohlc, dec, fee=fee, slip=slip, cost_mult=cost_mult)
            m_short = omega._metrics(ohlc, forced_side(dec, -1), fee=fee, slip=slip, cost_mult=cost_mult)
            m_long = omega._metrics(ohlc, forced_side(dec, 1), fee=fee, slip=slip, cost_mult=cost_mult)
            pnl_rows.append({
                "combo": combo_name, "split": split_name, "cost": cost_name,
                "model_pnl": m_model["pnl"], "model_trades": m_model["trades"], "model_wr": m_model["wr"],
                "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
                "beats_always_short": m_model["pnl"] > m_short["pnl"],
            })

pnl_df = pd.DataFrame(pnl_rows)
pnl_df.to_csv(OUT_DIR / "pnl_comparison.csv", index=False)
pd.set_option("display.width", 200)
for combo_name in ["FINAL12", "latent", "FINAL12+latent"]:
    log(f"\n--- {combo_name} ---")
    sub = pnl_df[pnl_df["combo"] == combo_name]
    for _, r in sub.iterrows():
        log(f"  [{r['split']}/{r['cost']}] model={r['model_pnl']:+7.2f}%(trades={r['model_trades']:.0f},wr={r['model_wr']*100:.1f}%)"
            f"  always_short={r['always_short_pnl']:+7.2f}%  always_long={r['always_long_pnl']:+7.2f}%"
            f"  {'model 승' if r['beats_always_short'] else 'always_short 승'}")

log(f"\n출력 디렉토리: {OUT_DIR}")
