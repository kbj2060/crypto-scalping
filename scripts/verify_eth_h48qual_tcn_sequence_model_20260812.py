"""사용자 제안(2단계, 오토인코더 다음): 단일 bar 스냅샷이 아니라 여러 bar에 걸친 시계열
패턴을 딥러닝(TCN)으로 직접 학습하면 새 정보가 나올 수 있는가? -- 지금까지 이 세션의 모든
시도(TabM/GBDT/trend-scanning/오토인코더)는 전부 "현재 bar 하나의 피쳐 스냅샷"만 봤다. 이
스크립트는 질적으로 다른 정보원(여러 bar에 걸친 시간적 구조)을 직접 테스트한다.

레포 인접 선례(entry_exit_edge_root_cause_and_literature_review_20260809.md Part 5)는 CUSUM
이벤트바/60코인 cross-sectional 풀링/Chronos zero-shot 3건이 전부 실패했다고 기록 -- 하지만
셋 다 "여러 자산/이벤트바 재구성"이지 "단일 자산의 raw 시계열을 인과적 TCN으로 직접 인코딩"은
아니었다. 기대치는 낮게 잡되(사전에 사용자에게 명시), 이 세션에서 유일하게 안 해본 축이라
실행한다.

설계: FINAL12처럼 이미 롤링윈도우로 집계된 피쳐가 아니라, 비교적 raw/경량 가공 컬럼 8개
(log_return/volatility_z/rsi/macd_hist/bb_width_z/wick_ratio/net_taker_ratio/cvd_12)를
96bar(8시간, h48qual 재설계 호라이즌과 동일 스케일) 인과적 윈도우로 직접 넣어 TCN이 자체
시간패턴을 찾게 한다 -- FINAL12를 다시 시퀀스로 넣으면 이미 집계된 값을 재집계하는 것이라
새 정보가 안 생길 가능성이 커서 의도적으로 회피했다.

검증: 오토인코더 실험에서 배운 교훈 그대로 -- 분류 지표만 보고 끝내지 않고, 검증된
거래 시뮬레이션(omega._metrics)으로 always-short 대조까지 반드시 확인한다."""
from __future__ import annotations

import json
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/eth_h48qual_tcn_sequence_model_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")

WINDOW = 96  # 8시간
SEQ_COLS = ["log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "wick_ratio", "net_taker_ratio", "cvd_12"]
MAX_TRAIN_WINDOWS_PER_EPOCH = 60000  # epoch당 랜덤 서브샘플 시간예산(시드마다 다른 순열)
N_SEEDS = 5  # Seed-Diversity Ensemble Promotion Gate: N>=5 진짜 무작위 시드

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


log(f"device={DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

# ---------------------------------------------------------------------------
# 1. 데이터 로딩
# ---------------------------------------------------------------------------

log("zig075 소스 패널 + zigzag_action 라벨 로딩...")
panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", low_memory=False)
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
panel = panel.sort_values("timestamp").reset_index(drop=True)
assert (panel["timestamp"].diff().dropna() == pd.Timedelta("5min")).all(), "5분봉 연속성 깨짐 -- 슬라이딩 윈도우 전 확인 필요"

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
labels = pd.concat([
    pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    for y in (2024, 2025, 2026)
], ignore_index=True).drop_duplicates("timestamp", keep="last")
label_map = labels.set_index("timestamp")["zigzag_action"]

missing_cols = [c for c in SEQ_COLS if c not in panel.columns]
assert not missing_cols, f"SEQ_COLS 누락: {missing_cols}"
log(f"  패널 {len(panel)}행, 시퀀스 입력 컬럼 {SEQ_COLS}")

raw = panel[SEQ_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
ts = panel["timestamp"].to_numpy()
y_full = label_map.reindex(panel["timestamp"]).to_numpy()  # NaN where label missing

train_mask = (panel["timestamp"] >= TRAIN_START) & (panel["timestamp"] <= TRAIN_END)
val_mask = (panel["timestamp"] >= VAL_START) & (panel["timestamp"] <= VAL_END)
oos_mask = (panel["timestamp"] >= OOS_START) & (panel["timestamp"] <= OOS_END)

# 각 split의 "유효 인덱스" = WINDOW-1개 이상의 과거 bar가 존재하고 라벨이 있는 행
def valid_indices(mask: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(mask.to_numpy() if hasattr(mask, "to_numpy") else mask)
    idx = idx[idx >= WINDOW - 1]
    idx = idx[~pd.isna(y_full[idx])]
    return idx

train_idx_all = valid_indices(train_mask)
val_idx = valid_indices(val_mask)
oos_idx = valid_indices(oos_mask)
log(f"  유효 윈도우: TRAIN={len(train_idx_all)}  VAL={len(val_idx)}  OOS={len(oos_idx)}")

# TRAIN 내부에서 시간순 마지막 15%를 early-stopping VAL로 -- 실제 VAL/OOS는 안 건드림
split_point = int(len(train_idx_all) * 0.85)
fit_idx, es_idx = train_idx_all[:split_point], train_idx_all[split_point:]

# ---------------------------------------------------------------------------
# 2. 표준화 (TRAIN fit 구간만) + 윈도우 Dataset
# ---------------------------------------------------------------------------

fit_rows = np.concatenate([np.arange(i - WINDOW + 1, i + 1) for i in fit_idx[:5000]])  # 표준화 통계는 대표 샘플로 충분
mean = raw[fit_rows].mean(axis=0)
std = raw[fit_rows].std(axis=0)
std[std < 1e-6] = 1.0
raw_std = np.clip((raw - mean) / std, -10, 10).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, indices: np.ndarray):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        window = raw_std[idx - WINDOW + 1: idx + 1]  # (WINDOW, C)
        y = int(y_full[idx])
        return torch.from_numpy(window.T.copy()), y  # (C, WINDOW) for Conv1d


# ---------------------------------------------------------------------------
# 3. 인과적 TCN (dilated causal conv 5층, receptive field > WINDOW)
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
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, 3, dilation)
        self.conv2 = CausalConv1d(ch, ch, 3, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.BatchNorm1d(ch)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.norm(out + x)  # residual
        return out


class TCNClassifier(nn.Module):
    def __init__(self, in_ch, hidden=32, n_classes=3, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.input_proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([TCNBlock(hidden, d) for d in dilations])
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: (B, C, T)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        last = h[:, :, -1]  # 인과적 -- 마지막 시점(현재 bar)까지의 정보만 사용
        return self.head(last)


receptive_field = 1 + 2 * (3 - 1) * sum([1, 2, 4, 8, 16])
log(f"TCN receptive field={receptive_field} (WINDOW={WINDOW}보다 커야 전체 윈도우를 다 봄)")

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


def train_one_seed(seed: int) -> tuple[TCNClassifier, float]:
    torch.manual_seed(seed)
    model = TCNClassifier(in_ch=len(SEQ_COLS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    es_loader = DataLoader(WindowDataset(es_idx), batch_size=1024, shuffle=False)
    rng = np.random.default_rng(seed)
    best_es_loss = float("inf")
    best_state = None
    patience, bad_epochs = 6, 0
    t0 = time.time()
    for epoch in range(30):
        model.train()
        epoch_idx = rng.choice(fit_idx, size=min(MAX_TRAIN_WINDOWS_PER_EPOCH, len(fit_idx)), replace=False)
        loader = DataLoader(WindowDataset(epoch_idx), batch_size=512, shuffle=True)
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        model.eval()
        with torch.no_grad():
            es_losses = []
            for xb, yb in es_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                es_losses.append(loss_fn(model(xb), yb).item() * len(yb))
            es_loss = sum(es_losses) / len(es_idx)
        if es_loss < best_es_loss - 1e-4:
            best_es_loss = es_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch % 5 == 0 or bad_epochs == 0:
            log(f"    epoch {epoch:2d}  train_loss={total_loss/n_batches:.4f}  es_loss={es_loss:.4f}  ({time.time()-t0:.0f}s)")
        if bad_epochs >= patience:
            log(f"    epoch {epoch}에서 조기종료")
            break
    model.load_state_dict(best_state)
    log(f"  seed={seed} 학습 완료 ({time.time()-t0:.0f}s), best es_loss={best_es_loss:.4f} (ln(3)={np.log(3):.4f}=균등분포)")
    return model, best_es_loss


def predict(model: TCNClassifier, indices: np.ndarray) -> np.ndarray:
    model.eval()
    loader = DataLoader(WindowDataset(indices), batch_size=1024, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# 4. N개 진짜 무작위 시드 학습 + 평가 (분류 지표 + 필수 거래 시뮬레이션 always-short 대조)
# ---------------------------------------------------------------------------

seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_SEEDS)
log(f"\n최종 시드(N={N_SEEDS}, 무작위 추출): {seeds}")

all_rows = []
clf_rows = []
for seed in seeds:
    log(f"\n--- seed={seed} ---")
    model, es_loss = train_one_seed(seed)
    for split_name, idx in [("VAL", val_idx), ("OOS", oos_idx)]:
        pred = predict(model, idx)
        y_true = y_full[idx].astype(np.int64)
        bacc = balanced_accuracy_score(y_true, pred)
        mf1 = f1_score(y_true, pred, average="macro")
        clf_rows.append({"seed": seed, "split": split_name, "balanced_accuracy": bacc, "macro_f1": mf1})
        log(f"  [분류/{split_name}] balanced_acc={bacc:.3f}  macro_f1={mf1:.3f}")

        ohlc = panel.iloc[idx][["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        dec = build_dec(pred)
        for cost_name, cost_mult in COST_MULTS.items():
            m_model = omega._metrics(ohlc, dec, fee=fee, slip=slip, cost_mult=cost_mult)
            m_short = omega._metrics(ohlc, forced_side(dec, -1), fee=fee, slip=slip, cost_mult=cost_mult)
            m_long = omega._metrics(ohlc, forced_side(dec, 1), fee=fee, slip=slip, cost_mult=cost_mult)
            all_rows.append({
                "seed": seed, "split": split_name, "cost": cost_name,
                "model_pnl": m_model["pnl"], "model_trades": m_model["trades"], "model_wr": m_model["wr"],
                "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
                "beats_always_short": m_model["pnl"] > m_short["pnl"],
            })

pnl_df = pd.DataFrame(all_rows)
clf_df = pd.DataFrame(clf_rows)
pnl_df.to_csv(OUT_DIR / "pnl_comparison_multiseed.csv", index=False)
clf_df.to_csv(OUT_DIR / "classification_multiseed.csv", index=False)

log(f"\n{'='*100}\nTCN 시퀀스 모델 -- N={N_SEEDS}시드 최종 요약\n{'='*100}")
for split_name in ["VAL", "OOS"]:
    csub = clf_df[clf_df["split"] == split_name]
    log(f"\n--- {split_name} 분류: balanced_acc={csub['balanced_accuracy'].mean():.3f}±{csub['balanced_accuracy'].std():.3f}"
        f"  macro_f1={csub['macro_f1'].mean():.3f}±{csub['macro_f1'].std():.3f} ---")
    for cost_name in COST_MULTS:
        sub = pnl_df[(pnl_df["split"] == split_name) & (pnl_df["cost"] == cost_name)]
        beat = int(sub["beats_always_short"].sum())
        diff = (sub["model_pnl"] - sub["always_short_pnl"]).to_numpy()
        if len(sub) >= 5 and np.any(diff != 0):
            _, wp = stats.wilcoxon(diff, alternative="greater")
        else:
            wp = float("nan")
        log(f"  [{cost_name}] model={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}"
            f"  always_short={sub['always_short_pnl'].mean():+7.2f}±{sub['always_short_pnl'].std():5.2f}"
            f"  always_long={sub['always_long_pnl'].mean():+7.2f}±{sub['always_long_pnl'].std():5.2f}"
            f"  모델승={beat}/{len(sub)}  wilcoxon_p={wp:.4f}")

log(f"\n출력 디렉토리: {OUT_DIR}")
