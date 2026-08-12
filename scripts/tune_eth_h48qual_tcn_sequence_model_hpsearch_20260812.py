"""TCN 시퀀스 모델(verify_eth_h48qual_tcn_sequence_model_20260812.py) 전체 파라미터 튜닝 +
5개 피쳐셋 변형 종합 탐색. 사용자 지시: "TCN 전체 파라미터 튜닝 작업을 서버한테 작업 지시,
데이터 피쳐도 여러 개로 나눠서, 오래 걸려도 괜찮으니까 최대한 넓은 범위."

절차(피쳐셋마다 독립 반복):
  1. Optuna N_TRIALS_PER_VARIANT회 -- TCN 아키텍처(window/hidden/dilation깊이/kernel/dropout)
     + 학습(lr/weight_decay/batch_size/class_weight) 탐색. 목적함수=TRAIN 내부 시간순 마지막
     15% early-stop CE loss(단일 seed=0, 저비용 스크리닝 -- 이 세션 GBDT 게이트와 동일 정신).
  2. 상위 5개 CV 후보를 VAL 거래 시뮬레이션(cost3)으로 재평가, always-short 대비 마진 최대
     후보 채택(select-on-validation-only).
  3. 채택 HP로 N=5 진짜 무작위 시드(Seed-Diversity Gate) 최종 검증 -- 분류 지표 + 필수
     거래 시뮬레이션(omega._metrics) always-short/long 대조, cost1/2/3, VAL/OOS.

5개 피쳐셋 변형:
  - raw_lite: 기존 TCN 베이스라인 8컬럼(오늘 이미 N=5 검증됨, HP만 재튜닝해 비교)
  - final12_seq: FINAL12(패널가용 9개)를 시퀀스로 재사용 -- 이미 집계된 피쳐를 다시 시퀀스로
    넣으면 새 정보가 없을 거라는 가설을 직접 테스트하는 대조군
  - raw_wide: 훨씬 넓은 raw/경량가공 24컬럼(오더플로우/변동성/펀딩/크로스에셋 전부 포함)
  - orderflow_funding: 오더플로우+펀딩 테마 12컬럼
  - ohlcv_minimal: 가장 raw한 5컬럼(TCN이 스스로 전부 학습하게)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/eth_h48qual_tcn_hpsearch_multivariant_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")

# TCN_SMOKE=1이면 파이프라인 정합성 확인용 초축소 실행(HP 탐색/최종 결론에 쓰지 않음).
SMOKE = os.environ.get("TCN_SMOKE", "0") == "1"
N_TRIALS_PER_VARIANT = 2 if SMOKE else 30
TOP_K_CANDIDATES = 2 if SMOKE else 5
N_FINAL_SEEDS = 2 if SMOKE else 5
MAX_WINDOWS_PER_EPOCH_TRIAL = 4000 if SMOKE else 40000  # Optuna 스크리닝 단계(빠르게)
MAX_WINDOWS_PER_EPOCH_FINAL = 4000 if SMOKE else 60000  # 최종 N시드 검증(baseline과 동일 예산)
MAX_EPOCHS_TRIAL = 2 if SMOKE else 15
MAX_EPOCHS_FINAL = 2 if SMOKE else 30
PATIENCE_TRIAL = 2 if SMOKE else 4
PATIENCE_FINAL = 2 if SMOKE else 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


log(f"device={DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

# ---------------------------------------------------------------------------
# 1. 데이터 로딩 + REPLACE 파생 컬럼 일괄 계산 (모든 변형이 공유)
# ---------------------------------------------------------------------------

log("zig075 소스 패널 + zigzag_action 라벨 로딩...")
panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", low_memory=False)
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
panel = panel.sort_values("timestamp").reset_index(drop=True)
assert (panel["timestamp"].diff().dropna() == pd.Timedelta("5min")).all(), "5분봉 연속성 깨짐"

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
labels = pd.concat([
    pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    for y in (2024, 2025, 2026)
], ignore_index=True).drop_duplicates("timestamp", keep="last")
label_map = labels.set_index("timestamp")["zigzag_action"]

# 이 세션 표준 REPLACE(diff1/dt288) + volume z-score(신규, 동일 관례로 추가)
panel["funding_pressure_diff1"] = pd.to_numeric(panel["funding_pressure"], errors="coerce").diff(1).fillna(0.0)
panel["sum_toptrader_long_short_ratio_dt288"] = (
    pd.to_numeric(panel["sum_toptrader_long_short_ratio"], errors="coerce")
    - pd.to_numeric(panel["sum_toptrader_long_short_ratio"], errors="coerce").rolling(288, min_periods=96).mean()
).fillna(0.0)
_vol = pd.to_numeric(panel["volume"], errors="coerce")
panel["volume_z288"] = ((_vol - _vol.rolling(288, min_periods=96).mean()) / _vol.rolling(288, min_periods=96).std().replace(0.0, 1.0)).fillna(0.0)

y_full = label_map.reindex(panel["timestamp"]).to_numpy()
train_mask = (panel["timestamp"] >= TRAIN_START) & (panel["timestamp"] <= TRAIN_END)
val_mask = (panel["timestamp"] >= VAL_START) & (panel["timestamp"] <= VAL_END)
oos_mask = (panel["timestamp"] >= OOS_START) & (panel["timestamp"] <= OOS_END)

# ---------------------------------------------------------------------------
# 2. 5개 피쳐셋 변형
# ---------------------------------------------------------------------------

FEATURE_VARIANTS = {
    "raw_lite": ["log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "wick_ratio", "net_taker_ratio", "cvd_12"],
    "final12_seq": ["cvp_regime", "funding_pressure_diff1", "ou_halflife", "realized_skewness", "mta_funding",
                     "sum_toptrader_long_short_ratio_dt288", "vwap_dist_24", "funding_roc_48", "breakout_strength"],
    "raw_wide": ["log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "wick_ratio", "net_taker_ratio",
                 "cvd_12", "cvd_48", "cvd_288", "cvd_slope_12", "cvd_slope_48", "taker_acceleration", "big_trade_ratio",
                 "funding_roc_12", "funding_roc_48", "ou_funding_z", "btc_ret_3", "btc_ret_12", "eth_btc_ret_spread_12",
                 "parkinson_vol", "hurst_48", "kalman_velocity", "mtf_trend_1h", "mtf_trend_4h"],
    "orderflow_funding": ["net_taker_ratio", "taker_acceleration", "big_trade_ratio", "cvd_12", "cvd_48", "cvd_288",
                           "cvd_slope_12", "cvd_slope_48", "oi_change_rate", "funding_roc_12", "funding_roc_48",
                           "funding_pressure_diff1"],
    "ohlcv_minimal": ["log_return", "volatility_z", "wick_ratio", "bb_width_z", "volume_z288"],
}
if SMOKE:
    FEATURE_VARIANTS = {"raw_lite": FEATURE_VARIANTS["raw_lite"], "ohlcv_minimal": FEATURE_VARIANTS["ohlcv_minimal"]}
for name, cols in FEATURE_VARIANTS.items():
    missing = [c for c in cols if c not in panel.columns]
    assert not missing, f"{name}: 누락 컬럼 {missing}"
    log(f"  변형 '{name}': {len(cols)}컬럼 = {cols}")

# ---------------------------------------------------------------------------
# 3. omega 거래 시뮬레이션 헬퍼
# ---------------------------------------------------------------------------

sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

fee, slip = omega._load_fee_slip()
COST_MULTS = {"cost1": 1.0, "cost2": 2.0, "cost3": 3.0}
omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0


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


# ---------------------------------------------------------------------------
# 4. TCN 아키텍처 (window/hidden/dilation깊이/kernel 전부 파라미터화)
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, ch, dilation, kernel_size, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(ch)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        return self.norm(out + x)


class TCNClassifier(nn.Module):
    def __init__(self, in_ch, hidden, n_classes, n_blocks, kernel_size, dropout):
        super().__init__()
        dilations = [2 ** i for i in range(n_blocks)]
        self.input_proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([TCNBlock(hidden, d, kernel_size, dropout) for d in dilations])
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, :, -1])


# ---------------------------------------------------------------------------
# 5. 피쳐셋 변형별: 표준화, 윈도우 인덱스, Optuna 탐색, 최종 N시드 검증
# ---------------------------------------------------------------------------

def valid_indices(mask_arr: np.ndarray, window: int) -> np.ndarray:
    idx = np.flatnonzero(mask_arr)
    idx = idx[idx >= window - 1]
    idx = idx[~pd.isna(y_full[idx])]
    return idx


def make_dataset_cls(raw_std: np.ndarray, window: int):
    class WindowDataset(Dataset):
        def __init__(self, indices: np.ndarray):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            idx = self.indices[i]
            w = raw_std[idx - window + 1: idx + 1]
            return torch.from_numpy(w.T.copy()), int(y_full[idx])

    return WindowDataset


all_variant_results = {}

for variant_name, seq_cols in FEATURE_VARIANTS.items():
    log(f"\n{'='*100}\n피쳐셋 변형: {variant_name} ({len(seq_cols)}컬럼)\n{'='*100}")
    raw = panel[seq_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    # ---- Optuna 목적함수: window까지 포함해 매 trial마다 인덱스/표준화 재계산 ----
    def objective(trial: "optuna.Trial", raw=raw, seq_cols=seq_cols) -> float:
        window = trial.suggest_categorical("window", [48, 96, 192])
        hidden = trial.suggest_categorical("hidden", [16, 32, 64])
        n_blocks = trial.suggest_int("n_blocks", 3, 6)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        class_weight_mode = trial.suggest_categorical("class_weight_mode", ["none", "balanced"])

        train_idx_all = valid_indices(train_mask.to_numpy(), window)
        split_point = int(len(train_idx_all) * 0.85)
        fit_idx, es_idx = train_idx_all[:split_point], train_idx_all[split_point:]
        if len(fit_idx) < 1000 or len(es_idx) < 200:
            raise optuna.TrialPruned()

        fit_rows = np.concatenate([np.arange(i - window + 1, i + 1) for i in fit_idx[:3000]])
        mean, std = raw[fit_rows].mean(axis=0), raw[fit_rows].std(axis=0)
        std[std < 1e-6] = 1.0
        raw_std = np.clip((raw - mean) / std, -10, 10).astype(np.float32)
        WindowDataset = make_dataset_cls(raw_std, window)

        torch.manual_seed(0)
        model = TCNClassifier(len(seq_cols), hidden, 3, n_blocks, kernel_size, dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        cw = None
        if class_weight_mode == "balanced":
            classes = np.unique(y_full[fit_idx].astype(np.int64))
            weights = compute_class_weight("balanced", classes=classes, y=y_full[fit_idx].astype(np.int64))
            full_w = np.ones(3, dtype=np.float32)
            for c, w in zip(classes, weights):
                full_w[int(c)] = w
            cw = torch.tensor(full_w, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=cw)

        es_loader = DataLoader(WindowDataset(es_idx), batch_size=1024, shuffle=False)
        rng = np.random.default_rng(0)
        best_es_loss = float("inf")
        bad_epochs = 0
        for epoch in range(MAX_EPOCHS_TRIAL):
            model.train()
            epoch_idx = rng.choice(fit_idx, size=min(MAX_WINDOWS_PER_EPOCH_TRIAL, len(fit_idx)), replace=False)
            loader = DataLoader(WindowDataset(epoch_idx), batch_size=batch_size, shuffle=True)
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                es_losses = [loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(yb) for xb, yb in es_loader]
                es_loss = sum(es_losses) / len(es_idx)
            if es_loss < best_es_loss - 1e-4:
                best_es_loss = es_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= PATIENCE_TRIAL:
                break
        trial.set_user_attr("window", window)
        return best_es_loss

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=20260812))
    study.optimize(objective, n_trials=N_TRIALS_PER_VARIANT, show_progress_bar=False)
    log(f"  Optuna {N_TRIALS_PER_VARIANT} trials 완료 ({time.time()-t0:.0f}s), best es_loss={study.best_value:.4f}")
    study.trials_dataframe().to_csv(OUT_DIR / f"optuna_trials_{variant_name}.csv", index=False)

    # ---- 상위 K개 후보를 VAL 거래 시뮬레이션으로 재평가 ----
    trials_sorted = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)
    top_candidates = trials_sorted[:TOP_K_CANDIDATES]
    log(f"  상위 {len(top_candidates)}개 후보 VAL 재평가...")
    cand_rows = []
    for rank, trial in enumerate(top_candidates):
        p = trial.params
        window = p["window"]
        train_idx_all = valid_indices(train_mask.to_numpy(), window)
        val_idx = valid_indices(val_mask.to_numpy(), window)
        split_point = int(len(train_idx_all) * 0.85)
        fit_idx, es_idx = train_idx_all[:split_point], train_idx_all[split_point:]
        fit_rows = np.concatenate([np.arange(i - window + 1, i + 1) for i in fit_idx[:3000]])
        mean, std = raw[fit_rows].mean(axis=0), raw[fit_rows].std(axis=0)
        std[std < 1e-6] = 1.0
        raw_std = np.clip((raw - mean) / std, -10, 10).astype(np.float32)
        WindowDataset = make_dataset_cls(raw_std, window)

        torch.manual_seed(0)
        model = TCNClassifier(len(seq_cols), p["hidden"], 3, p["n_blocks"], p["kernel_size"], p["dropout"]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        cw = None
        if p["class_weight_mode"] == "balanced":
            classes = np.unique(y_full[fit_idx].astype(np.int64))
            weights = compute_class_weight("balanced", classes=classes, y=y_full[fit_idx].astype(np.int64))
            full_w = np.ones(3, dtype=np.float32)
            for c, w in zip(classes, weights):
                full_w[int(c)] = w
            cw = torch.tensor(full_w, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        es_loader = DataLoader(WindowDataset(es_idx), batch_size=1024, shuffle=False)
        rng = np.random.default_rng(0)
        best_es_loss, best_state, bad_epochs = float("inf"), None, 0
        for epoch in range(MAX_EPOCHS_TRIAL):
            model.train()
            epoch_idx = rng.choice(fit_idx, size=min(MAX_WINDOWS_PER_EPOCH_TRIAL, len(fit_idx)), replace=False)
            loader = DataLoader(WindowDataset(epoch_idx), batch_size=p["batch_size"], shuffle=True)
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                es_loss = sum(loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(yb) for xb, yb in es_loader) / len(es_idx)
            if es_loss < best_es_loss - 1e-4:
                best_es_loss, best_state, bad_epochs = es_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad_epochs += 1
            if bad_epochs >= PATIENCE_TRIAL:
                break
        model.load_state_dict(best_state)
        model.eval()
        val_loader = DataLoader(WindowDataset(val_idx), batch_size=1024, shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                preds.append(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
        pred = np.concatenate(preds)
        ohlc = panel.iloc[val_idx][["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        dec = build_dec(pred)
        m_model = omega._metrics(ohlc, dec, fee=fee, slip=slip, cost_mult=3.0)
        m_short = omega._metrics(ohlc, forced_side(dec, -1), fee=fee, slip=slip, cost_mult=3.0)
        margin = m_model["pnl"] - m_short["pnl"]
        cand_rows.append({"rank": rank, "trial_number": trial.number, "cv_loss": trial.value, "params": p,
                           "val_pnl": m_model["pnl"], "val_always_short_pnl": m_short["pnl"], "margin": margin})
        log(f"    trial#{trial.number} cv_loss={trial.value:.4f} window={window} VAL_pnl={m_model['pnl']:+.2f} "
            f"always_short={m_short['pnl']:+.2f} margin={margin:+.2f}")

    winner = cand_rows[int(np.argmax([r["margin"] for r in cand_rows]))]
    log(f"  채택: trial#{winner['trial_number']}  margin={winner['margin']:+.2f}  params={winner['params']}")
    (OUT_DIR / f"winner_{variant_name}.json").write_text(json.dumps(winner, indent=2, ensure_ascii=False, default=str))

    # ---- N=5 진짜 무작위 시드 최종 검증 ----
    p = winner["params"]
    window = p["window"]
    train_idx_all = valid_indices(train_mask.to_numpy(), window)
    val_idx = valid_indices(val_mask.to_numpy(), window)
    oos_idx = valid_indices(oos_mask.to_numpy(), window)
    split_point = int(len(train_idx_all) * 0.85)
    fit_idx, es_idx = train_idx_all[:split_point], train_idx_all[split_point:]
    fit_rows = np.concatenate([np.arange(i - window + 1, i + 1) for i in fit_idx[:3000]])
    mean, std = raw[fit_rows].mean(axis=0), raw[fit_rows].std(axis=0)
    std[std < 1e-6] = 1.0
    raw_std = np.clip((raw - mean) / std, -10, 10).astype(np.float32)
    WindowDataset = make_dataset_cls(raw_std, window)

    seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_FINAL_SEEDS)
    log(f"\n  최종 N={N_FINAL_SEEDS}시드(무작위): {seeds}")
    pnl_rows, clf_rows = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = TCNClassifier(len(seq_cols), p["hidden"], 3, p["n_blocks"], p["kernel_size"], p["dropout"]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        cw = None
        if p["class_weight_mode"] == "balanced":
            classes = np.unique(y_full[fit_idx].astype(np.int64))
            weights = compute_class_weight("balanced", classes=classes, y=y_full[fit_idx].astype(np.int64))
            full_w = np.ones(3, dtype=np.float32)
            for c, w in zip(classes, weights):
                full_w[int(c)] = w
            cw = torch.tensor(full_w, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        es_loader = DataLoader(WindowDataset(es_idx), batch_size=1024, shuffle=False)
        rng = np.random.default_rng(seed)
        best_es_loss, best_state, bad_epochs = float("inf"), None, 0
        t_seed = time.time()
        for epoch in range(MAX_EPOCHS_FINAL):
            model.train()
            epoch_idx = rng.choice(fit_idx, size=min(MAX_WINDOWS_PER_EPOCH_FINAL, len(fit_idx)), replace=False)
            loader = DataLoader(WindowDataset(epoch_idx), batch_size=p["batch_size"], shuffle=True)
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                es_loss = sum(loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(yb) for xb, yb in es_loader) / len(es_idx)
            if es_loss < best_es_loss - 1e-4:
                best_es_loss, best_state, bad_epochs = es_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad_epochs += 1
            if bad_epochs >= PATIENCE_FINAL:
                break
        model.load_state_dict(best_state)
        model.eval()
        log(f"    seed={seed} 완료 ({time.time()-t_seed:.0f}s)")

        for split_name, idx in [("VAL", val_idx), ("OOS", oos_idx)]:
            loader = DataLoader(WindowDataset(idx), batch_size=1024, shuffle=False)
            preds = []
            with torch.no_grad():
                for xb, _ in loader:
                    preds.append(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
            pred = np.concatenate(preds)
            y_true = y_full[idx].astype(np.int64)
            clf_rows.append({"variant": variant_name, "seed": seed, "split": split_name,
                              "balanced_accuracy": balanced_accuracy_score(y_true, pred),
                              "macro_f1": f1_score(y_true, pred, average="macro")})
            ohlc = panel.iloc[idx][["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
            dec = build_dec(pred)
            for cost_name, cost_mult in COST_MULTS.items():
                m_model = omega._metrics(ohlc, dec, fee=fee, slip=slip, cost_mult=cost_mult)
                m_short = omega._metrics(ohlc, forced_side(dec, -1), fee=fee, slip=slip, cost_mult=cost_mult)
                m_long = omega._metrics(ohlc, forced_side(dec, 1), fee=fee, slip=slip, cost_mult=cost_mult)
                pnl_rows.append({"variant": variant_name, "seed": seed, "split": split_name, "cost": cost_name,
                                  "model_pnl": m_model["pnl"], "model_trades": m_model["trades"], "model_wr": m_model["wr"],
                                  "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
                                  "beats_always_short": m_model["pnl"] > m_short["pnl"]})

    variant_pnl_df = pd.DataFrame(pnl_rows)
    variant_clf_df = pd.DataFrame(clf_rows)
    variant_pnl_df.to_csv(OUT_DIR / f"final_pnl_{variant_name}.csv", index=False)
    variant_clf_df.to_csv(OUT_DIR / f"final_clf_{variant_name}.csv", index=False)

    log(f"\n  --- {variant_name} N=5시드 최종 요약 (window={window}, hidden={p['hidden']}, n_blocks={p['n_blocks']}) ---")
    summary = {"variant": variant_name, "winner_params": p, "seeds": seeds, "cells": {}}
    for split_name in ["VAL", "OOS"]:
        csub = variant_clf_df[variant_clf_df["split"] == split_name]
        log(f"  {split_name} 분류: balanced_acc={csub['balanced_accuracy'].mean():.3f}±{csub['balanced_accuracy'].std():.3f}"
            f"  macro_f1={csub['macro_f1'].mean():.3f}±{csub['macro_f1'].std():.3f}")
        for cost_name in COST_MULTS:
            sub = variant_pnl_df[(variant_pnl_df["split"] == split_name) & (variant_pnl_df["cost"] == cost_name)]
            beat = int(sub["beats_always_short"].sum())
            diff = (sub["model_pnl"] - sub["always_short_pnl"]).to_numpy()
            wp = stats.wilcoxon(diff, alternative="greater")[1] if len(sub) >= 5 and np.any(diff != 0) else float("nan")
            log(f"    [{cost_name}] model={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}"
                f"  always_short={sub['always_short_pnl'].mean():+7.2f}±{sub['always_short_pnl'].std():5.2f}"
                f"  모델승={beat}/{len(sub)}  wilcoxon_p={wp:.4f}")
            summary["cells"][f"{split_name}_{cost_name}"] = {"model_pnl_mean": float(sub["model_pnl"].mean()),
                                                               "always_short_pnl_mean": float(sub["always_short_pnl"].mean()),
                                                               "beats": beat, "n": len(sub), "wilcoxon_p": float(wp)}
    all_variant_results[variant_name] = summary
    (OUT_DIR / f"summary_{variant_name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

# ---------------------------------------------------------------------------
# 6. 전체 변형 비교 최종 리포트
# ---------------------------------------------------------------------------

log(f"\n\n{'='*100}\nTCN 전체 파라미터튜닝 + 5개 피쳐셋 -- 종합 비교\n{'='*100}")
for variant_name, s in all_variant_results.items():
    log(f"\n[{variant_name}]  window={s['winner_params']['window']} hidden={s['winner_params']['hidden']} "
        f"n_blocks={s['winner_params']['n_blocks']}")
    for cell, r in s["cells"].items():
        log(f"  {cell}: model={r['model_pnl_mean']:+7.2f}  always_short={r['always_short_pnl_mean']:+7.2f}  "
            f"승={r['beats']}/{r['n']}  p={r['wilcoxon_p']:.4f}")

(OUT_DIR / "all_variants_summary.json").write_text(json.dumps(all_variant_results, indent=2, ensure_ascii=False, default=str))
log(f"\n출력 디렉토리: {OUT_DIR}")
