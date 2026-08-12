"""사용자 제안: 유명 퀀트/트레이더들처럼 "차트를 보고" 판단하는 방식 -- 캔들차트를 이미지로
렌더링해 CNN에 직접 넣어 zigzag_action을 예측한다. 오늘 이 세션에서 시도한 모든 방식(TabM/
GBDT/오토인코더/TCN)이 전부 숫자(스칼라/시퀀스) 표현이었던 것과 달리, 이건 시각적 형태
표현이라는 질적으로 다른 축이다.

사전 리서치(웹서치)로 확인한 것: (1) 진짜 계량퀀트펀드는 차트 패턴이 아니라 숫자 데이터를
알고리즘으로 처리한다 -- "차트만 보고" 트레이딩은 재량/기술적분석 트레이더 쪽에 더 가깝다.
(2) 관련 학술 연구(암호화폐 특화 포함, arXiv 2605.00875 등)는 실제로 존재하고 단순 CNN이
원본 캔들차트에서 AUC-ROC 0.89 수준을 보고하지만, **실제 거래 성과(PnL/샤프)는 보고하지
않는다** -- 오늘 이 세션이 오토인코더·TCN에서 두 번 겪은 "분류지표는 좋은데 PnL은 아니다"
함정과 정확히 같은 위험. (3) 정보이론적으로 캔들차트 이미지는 OHLCV 숫자보다 정보량이
같거나 적다(렌더링 손실) -- 오늘 TCN이 이미 같은 raw 시퀀스를 숫자로 직접 줬으므로, CNN
이미지 경로는 그보다 간접적이라 기대치를 TCN보다 낮게 잡아야 한다.

이런 배경에서 설계 원칙: (a) 분류 지표만 보고 끝내지 않고 처음부터 거래 시뮬레이션(always-
short/long 대조)을 필수로 포함 (b) POST_OOS 결과가 혹시 양성으로 나오면, 지난번(TCN)처럼
성급히 보고하지 않고 **같은 실행 안에서 즉시 월별 분해로 재검증**한 뒤에만 보고한다.

렌더링: numpy 기반 커스텀 래스터라이저(matplotlib 대비 훨씬 빠름) -- window=48bar(TCN 최적
설정과 동일, 직접 비교 가능), 64(너비)x128(높이) RGB, 4개 패널로 구성된 진짜 트레이더용 차트:
(1) 캔들스틱(64px, 창 내부 고저가 기준 정규화) (2) 거래량 바(22px, 양봉/음봉 색 일치)
(3) RSI 라인(20px, 0~100 고정범위) (4) MACD 히스토그램(22px, 창 내부 최대절대값 기준
중앙정렬). 전부 causal(창 내부 값만 사용, 미래 정보 없음). 전체 패널(TRAIN~POST_OOS)을 한
번만 사전 렌더링해 재사용(매 epoch 재렌더링 안 함)."""
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / ("tmp/eth_h48qual_cnn_candlestick_20260812_smoke" if os.environ.get("CNN_SMOKE", "0") == "1"
                  else "tmp/eth_h48qual_cnn_candlestick_20260812")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")
POST_OOS_START, POST_OOS_END = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-08-04 23:59:59")

WINDOW = 48  # TCN 최적 설정과 동일(직접 비교 가능)
IMG_W = 64
PRICE_H, VOLUME_H, RSI_H, MACD_H = 64, 22, 20, 22
IMG_H = PRICE_H + VOLUME_H + RSI_H + MACD_H  # 128
SMOKE = os.environ.get("CNN_SMOKE", "0") == "1"
N_FINAL_SEEDS = 1 if SMOKE else 5
MAX_WINDOWS_PER_EPOCH = 2000 if SMOKE else 60000
MAX_EPOCHS = 2 if SMOKE else 30
PATIENCE = 2 if SMOKE else 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


log(f"device={DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

# ---------------------------------------------------------------------------
# 1. 데이터 로딩
# ---------------------------------------------------------------------------

log("zig075 소스 패널 + zigzag_action 라벨 로딩...")
panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv",
                     usecols=["timestamp", "open", "high", "low", "close", "volume", "rsi", "macd_hist"], low_memory=False)
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
panel = panel.sort_values("timestamp").reset_index(drop=True)
assert (panel["timestamp"].diff().dropna() == pd.Timedelta("5min")).all(), "5분봉 연속성 깨짐"

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
labels = pd.concat([
    pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    for y in (2024, 2025, 2026)
], ignore_index=True).drop_duplicates("timestamp", keep="last")
label_map = labels.set_index("timestamp")["zigzag_action"]
y_full = label_map.reindex(panel["timestamp"]).to_numpy()

o_arr = panel["open"].to_numpy(dtype=np.float64)
h_arr = panel["high"].to_numpy(dtype=np.float64)
l_arr = panel["low"].to_numpy(dtype=np.float64)
c_arr = panel["close"].to_numpy(dtype=np.float64)
vol_arr = pd.to_numeric(panel["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
rsi_arr = pd.to_numeric(panel["rsi"], errors="coerce").fillna(50.0).to_numpy(dtype=np.float64)
macd_arr = pd.to_numeric(panel["macd_hist"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

train_mask = (panel["timestamp"] >= TRAIN_START) & (panel["timestamp"] <= TRAIN_END)
val_mask = (panel["timestamp"] >= VAL_START) & (panel["timestamp"] <= VAL_END)
oos_mask = (panel["timestamp"] >= OOS_START) & (panel["timestamp"] <= OOS_END)
post_oos_mask = (panel["timestamp"] >= POST_OOS_START) & (panel["timestamp"] <= POST_OOS_END)
full_range_mask = (panel["timestamp"] >= TRAIN_START) & (panel["timestamp"] <= POST_OOS_END)


def valid_indices(mask_arr: np.ndarray, require_label: bool) -> np.ndarray:
    idx = np.flatnonzero(mask_arr)
    idx = idx[idx >= WINDOW - 1]
    if require_label:
        idx = idx[~pd.isna(y_full[idx])]
    return idx


full_idx = valid_indices(full_range_mask.to_numpy(), require_label=False)
train_idx_all = valid_indices(train_mask.to_numpy(), require_label=True)
val_idx = valid_indices(val_mask.to_numpy(), require_label=True)
oos_idx = valid_indices(oos_mask.to_numpy(), require_label=True)
post_oos_idx = valid_indices(post_oos_mask.to_numpy(), require_label=False)
log(f"  전체 예측범위={len(full_idx)}  TRAIN={len(train_idx_all)}  VAL={len(val_idx)}  OOS={len(oos_idx)}  POST_OOS={len(post_oos_idx)}")

# full_idx 내에서의 위치를 찾기 위한 역인덱스
full_idx_pos = {idx: pos for pos, idx in enumerate(full_idx)}

# ---------------------------------------------------------------------------
# 2. 캔들차트 이미지 사전 렌더링 (numpy 벡터화, 1회만)
# ---------------------------------------------------------------------------

log(f"\n캔들차트+거래량+RSI+MACD 4패널 사전 렌더링 ({len(full_idx)}장, {IMG_W}x{IMG_H})...")
t0 = time.time()
images = np.zeros((len(full_idx), 3, IMG_H, IMG_W), dtype=np.uint8)
col_w = IMG_W / WINDOW
col_bounds = [(int(i * col_w), max(int(i * col_w) + 1, int((i + 1) * col_w) - 1)) for i in range(WINDOW)]

VOL_TOP = PRICE_H
RSI_TOP = PRICE_H + VOLUME_H
MACD_TOP = PRICE_H + VOLUME_H + RSI_H
MACD_MID = MACD_TOP + MACD_H // 2

for pos, end_idx in enumerate(full_idx):
    start_idx = end_idx - WINDOW + 1
    sl = slice(start_idx, end_idx + 1)
    o, h, l, c = o_arr[sl], h_arr[sl], l_arr[sl], c_arr[sl]
    vol, rsi, macd = vol_arr[sl], rsi_arr[sl], macd_arr[sl]
    img = images[pos]  # (3, H, W), CHW

    # --- 패널 1: 캔들스틱 (창 내부 고저가 기준 정규화) ---
    lo, hi = l.min(), h.max()
    span = max(hi - lo, 1e-9)
    y_h = np.clip(((h - lo) / span * (PRICE_H - 1)), 0, PRICE_H - 1).astype(np.int64)
    y_l = np.clip(((l - lo) / span * (PRICE_H - 1)), 0, PRICE_H - 1).astype(np.int64)
    y_o = np.clip(((o - lo) / span * (PRICE_H - 1)), 0, PRICE_H - 1).astype(np.int64)
    y_c = np.clip(((c - lo) / span * (PRICE_H - 1)), 0, PRICE_H - 1).astype(np.int64)

    # --- 패널 2: 거래량 (창 내부 최대거래량 기준, 바닥에서 위로) ---
    vmax = max(vol.max(), 1e-9)
    vol_px = np.clip((vol / vmax * (VOLUME_H - 1)), 0, VOLUME_H - 1).astype(np.int64)

    # --- 패널 3: RSI (0~100 고정범위) ---
    rsi_px = np.clip((rsi / 100.0 * (RSI_H - 1)), 0, RSI_H - 1).astype(np.int64)

    # --- 패널 4: MACD 히스토그램 (창 내부 최대절대값 기준, 중앙정렬) ---
    mmax = max(np.abs(macd).max(), 1e-9)
    macd_px = np.clip((np.abs(macd) / mmax * (MACD_H // 2 - 1)), 0, MACD_H // 2 - 1).astype(np.int64)

    for i in range(WINDOW):
        x0, x1 = col_bounds[i]
        xc = (x0 + x1) // 2
        up = c[i] >= o[i]
        ch = 1 if up else 0  # 채널1=녹색(상승/양수), 채널0=적색(하락/음수)
        val = 255

        # 패널1: 캔들 심지+몸통
        img[ch, PRICE_H - 1 - y_h[i]: PRICE_H - y_l[i], xc] = val
        top, bot = sorted([y_o[i], y_c[i]])
        img[ch, PRICE_H - 1 - bot: PRICE_H - top + 1, x0:x1 + 1] = val

        # 패널2: 거래량 바(양봉/음봉과 같은 색)
        img[ch, VOL_TOP + VOLUME_H - 1 - vol_px[i]: VOL_TOP + VOLUME_H, x0:x1 + 1] = val

        # 패널3: RSI 라인(각 컬럼에 2px 마커, 흰색=채널0,1,2 전부)
        r_row = RSI_TOP + (RSI_H - 1 - rsi_px[i])
        img[:, max(RSI_TOP, r_row - 1):min(RSI_TOP + RSI_H, r_row + 2), xc] = val

        # 패널4: MACD 히스토그램(중앙선 기준 위/아래, 양수=녹/음수=적)
        if macd[i] >= 0:
            img[1, MACD_MID - macd_px[i]: MACD_MID + 1, x0:x1 + 1] = val
        else:
            img[0, MACD_MID: MACD_MID + macd_px[i] + 1, x0:x1 + 1] = val
log(f"렌더링 완료 ({time.time()-t0:.0f}초), 메모리={images.nbytes/1e9:.2f}GB")

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
# 4. 단순 CNN (논문 결과 재현 -- "단순 4층 CNN이 큰 사전학습 모델보다 나음")
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),  # 64->32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.AdaptiveAvgPool2d(1),  # ->1x1
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.head(self.dropout(h))


class ImageDataset(Dataset):
    def __init__(self, global_indices: np.ndarray):
        self.pos = np.array([full_idx_pos[i] for i in global_indices], dtype=np.int64)
        self.global_indices = global_indices

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):
        p = self.pos[i]
        img = torch.from_numpy(images[p].astype(np.float32) / 255.0)
        y = y_full[self.global_indices[i]]
        return img, (int(y) if not pd.isna(y) else -1)


# ---------------------------------------------------------------------------
# 5. 학습 (TRAIN 내부 마지막 15% early-stopping) + N=5 시드
# ---------------------------------------------------------------------------

split_point = int(len(train_idx_all) * 0.85)
fit_idx, es_idx = train_idx_all[:split_point], train_idx_all[split_point:]
es_loader = DataLoader(ImageDataset(es_idx), batch_size=1024, shuffle=False)

seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_FINAL_SEEDS)
log(f"\n최종 N={N_FINAL_SEEDS}시드(무작위): {seeds}")

clf_rows, pnl_rows = [], []
monthly_rows = []
months = pd.period_range("2024-06", "2026-08", freq="M")

for seed in seeds:
    torch.manual_seed(seed)
    model = SimpleCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    best_es_loss, best_state, bad_epochs = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_idx = rng.choice(fit_idx, size=min(MAX_WINDOWS_PER_EPOCH, len(fit_idx)), replace=False)
        loader = DataLoader(ImageDataset(epoch_idx), batch_size=256, shuffle=True)
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
        if bad_epochs >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    log(f"  seed={seed} 학습완료 ({time.time()-t0:.0f}s, es_loss={best_es_loss:.4f}, ln3={np.log(3):.4f})")

    def predict(idx_arr):
        loader = DataLoader(ImageDataset(idx_arr), batch_size=2048, shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    # 전체범위 한 번에 예측(월별분해 등에 재사용)
    pred_full = predict(full_idx)
    pred_by_global = dict(zip(full_idx, pred_full))

    for split_name, idx, has_label in [("VAL", val_idx, True), ("OOS", oos_idx, True), ("POST_OOS", post_oos_idx, False)]:
        pred = np.array([pred_by_global[i] for i in idx])
        if has_label:
            y_true = y_full[idx].astype(np.int64)
            clf_rows.append({"seed": seed, "split": split_name,
                              "balanced_accuracy": balanced_accuracy_score(y_true, pred),
                              "macro_f1": f1_score(y_true, pred, average="macro")})
        ohlc = panel.iloc[idx][["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        dec = build_dec(pred)
        for cost_name, cost_mult in COST_MULTS.items():
            m_model = omega._metrics(ohlc, dec, fee=fee, slip=slip, cost_mult=cost_mult)
            m_short = omega._metrics(ohlc, forced_side(dec, -1), fee=fee, slip=slip, cost_mult=cost_mult)
            m_long = omega._metrics(ohlc, forced_side(dec, 1), fee=fee, slip=slip, cost_mult=cost_mult)
            pnl_rows.append({"seed": seed, "split": split_name, "cost": cost_name,
                              "model_pnl": m_model["pnl"], "model_trades": m_model["trades"], "model_wr": m_model["wr"],
                              "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
                              "beats_short": m_model["pnl"] > m_short["pnl"], "beats_long": m_model["pnl"] > m_long["pnl"]})

    # 월별 분해도 이번엔 처음부터 같이 수행 (지난번 TCN 교훈 -- 나중에 별도로 재검증하지 않고 즉시)
    ts_full = panel["timestamp"].to_numpy()[full_idx]
    for month in months:
        m_start, m_end = month.start_time, month.end_time
        m_mask = (ts_full >= m_start) & (ts_full <= m_end)
        if m_mask.sum() < 50:
            continue
        m_idx_global = full_idx[m_mask]
        pred_m = pred_full[m_mask]
        ohlc_m = panel.iloc[m_idx_global][["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        dec_m = build_dec(pred_m)
        mm_model = omega._metrics(ohlc_m, dec_m, fee=fee, slip=slip, cost_mult=3.0)
        mm_short = omega._metrics(ohlc_m, forced_side(dec_m, -1), fee=fee, slip=slip, cost_mult=3.0)
        mm_long = omega._metrics(ohlc_m, forced_side(dec_m, 1), fee=fee, slip=slip, cost_mult=3.0)
        price_ret = (ohlc_m["close"].iloc[-1] / ohlc_m["close"].iloc[0] - 1) * 100
        monthly_rows.append({"seed": seed, "month": str(month), "in_sample": bool(m_start <= TRAIN_END),
                              "price_ret_pct": float(price_ret), "model_pnl": mm_model["pnl"],
                              "always_short_pnl": mm_short["pnl"], "always_long_pnl": mm_long["pnl"],
                              "beats_both": (mm_model["pnl"] > mm_short["pnl"]) and (mm_model["pnl"] > mm_long["pnl"])})

pnl_df = pd.DataFrame(pnl_rows)
clf_df = pd.DataFrame(clf_rows)
monthly_df = pd.DataFrame(monthly_rows)
pnl_df.to_csv(OUT_DIR / "pnl_comparison.csv", index=False)
clf_df.to_csv(OUT_DIR / "classification.csv", index=False)
monthly_df.to_csv(OUT_DIR / "monthly_breakdown.csv", index=False)

# ---------------------------------------------------------------------------
# 6. 리포트
# ---------------------------------------------------------------------------

log(f"\n{'='*100}\nCNN 캔들차트 -- N={N_FINAL_SEEDS}시드 최종 요약\n{'='*100}")
for split_name in ["VAL", "OOS", "POST_OOS"]:
    if split_name in ("VAL", "OOS"):
        csub = clf_df[clf_df["split"] == split_name]
        log(f"\n--- {split_name} 분류: balanced_acc={csub['balanced_accuracy'].mean():.3f}±{csub['balanced_accuracy'].std():.3f} ---")
    else:
        log(f"\n--- {split_name}(라벨없음, PnL만) ---")
    for cost_name in COST_MULTS:
        sub = pnl_df[(pnl_df["split"] == split_name) & (pnl_df["cost"] == cost_name)]
        beat_s, beat_l = int(sub["beats_short"].sum()), int(sub["beats_long"].sum())
        diff = (sub["model_pnl"] - sub["always_short_pnl"]).to_numpy()
        wp = stats.wilcoxon(diff, alternative="greater")[1] if len(sub) >= 5 and np.any(diff != 0) else float("nan")
        log(f"  [{cost_name}] model={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}"
            f"  always_short={sub['always_short_pnl'].mean():+7.2f}±{sub['always_short_pnl'].std():5.2f}"
            f"  always_long={sub['always_long_pnl'].mean():+7.2f}"
            f"  승(short)={beat_s}/{len(sub)}  승(long)={beat_l}/{len(sub)}  wilcoxon_p={wp:.4f}")

log(f"\n--- POST_OOS 월별 분해(즉시 검증, cost3) ---")
post_monthly = monthly_df[(monthly_df["month"] >= "2026-03") & (monthly_df["month"] <= "2026-08")]
post_agg = post_monthly.groupby("month").agg(
    price_ret_pct=("price_ret_pct", "first"), model_pnl_mean=("model_pnl", "mean"),
    always_short_mean=("always_short_pnl", "mean"), always_long_mean=("always_long_pnl", "mean"),
    beats_both_rate=("beats_both", "mean")).reset_index().sort_values("month")
for _, r in post_agg.iterrows():
    log(f"  {r['month']} 가격{r['price_ret_pct']:+6.1f}%  model={r['model_pnl_mean']:+7.2f}  "
        f"short={r['always_short_mean']:+7.2f}  long={r['always_long_mean']:+7.2f}  둘다승={r['beats_both_rate']*100:5.0f}%")
compound_model, compound_short = 1.0, 1.0
for _, r in post_agg.sort_values("month").iterrows():
    compound_model *= (1 + r["model_pnl_mean"] / 100)
    compound_short *= (1 + r["always_short_mean"] / 100)
log(f"  POST_OOS 월별 복리 누적: model={100*(compound_model-1):+.2f}%  always_short={100*(compound_short-1):+.2f}%")

log(f"\n출력 디렉토리: {OUT_DIR}")
