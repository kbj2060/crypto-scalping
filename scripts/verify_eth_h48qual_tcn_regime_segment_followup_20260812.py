"""사용자 지시("후속 검증 철저히") 실행 -- post-OOS 확장 검증에서 나온 첫 유의미한 양성
결과(TCN이 휩소 레짐에서 always-short/long 둘 다 이김, N=5시드 p=0.03)가 "진짜 방향 스킬"인지
"휩소 레짐에서는 약한 신호도 두 나쁜 기준선을 이기기 쉽다"는 대안 가설과 구분이 안 된 상태였다.
이 스크립트는 재학습 없이 임의 구간을 잘라 분석할 수 있도록 **전체 패널(TRAIN~POST_OOS,
2024-06~2026-08) bar 단위 예측을 저장**한 뒤, 세 가지 후속 진단을 수행한다:

(1) 월별 PnL 분해 -- 전체 구간(TRAIN 포함, 인샘플 표시) x always-short/long 대조. 휩소월과
    순수추세월에서 모델 성적이 다른지 직접 비교.
(2) POST_OOS를 추세월(2026-06 단일방향 -21.9%, 2026-07 단일방향 +18.3%)과 반전월로 나눠
    엣지가 반전 시점에 몰려있는지 추세 구간에도 있는지 분리.
(3) VAL/OOS(둘 다 순수 하락, 월별 반등 없음) 내부의 **주간 반등**(가격이 오른 주)과 하락주를
    풀링해서 모델이 반등주에서 상대적으로 더 나은지 확인 -- 진짜 out-of-sample 데이터로 하는
    휩소-메커니즘 테스트(TRAIN 구간 테스트는 인샘플이라 이게 더 결정적).

HP는 오늘 확정된 raw_lite/ohlcv_minimal 그대로(재탐색 없음), N=5 신규 무작위 시드."""
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/eth_h48qual_tcn_regime_segment_followup_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")
POST_OOS_START, POST_OOS_END = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-08-04 23:59:59")

N_FINAL_SEEDS = 5
MAX_WINDOWS_PER_EPOCH = 60000
MAX_EPOCHS = 30
PATIENCE = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


log(f"device={DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

FEATURE_VARIANTS = {
    "raw_lite": {
        "cols": ["log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "wick_ratio", "net_taker_ratio", "cvd_12"],
        "params": {"window": 48, "hidden": 32, "n_blocks": 5, "kernel_size": 5, "dropout": 0.299802956322068,
                   "lr": 0.002484431047341557, "weight_decay": 0.0008896182007864648, "batch_size": 1024, "class_weight_mode": "none"},
    },
    "ohlcv_minimal": {
        "cols": ["log_return", "volatility_z", "wick_ratio", "bb_width_z", "volume_z288"],
        "params": {"window": 48, "hidden": 32, "n_blocks": 6, "kernel_size": 5, "dropout": 0.299802956322068,
                   "lr": 0.0007701193784031896, "weight_decay": 0.0008896182007864648, "batch_size": 512, "class_weight_mode": "none"},
    },
}

log("zig075 소스 패널 로딩...")
panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", low_memory=False)
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
panel = panel.sort_values("timestamp").reset_index(drop=True)
assert (panel["timestamp"].diff().dropna() == pd.Timedelta("5min")).all(), "5분봉 연속성 깨짐"

_vol = pd.to_numeric(panel["volume"], errors="coerce")
panel["volume_z288"] = ((_vol - _vol.rolling(288, min_periods=96).mean()) / _vol.rolling(288, min_periods=96).std().replace(0.0, 1.0)).fillna(0.0)

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
labels = pd.concat([
    pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    for y in (2024, 2025, 2026)
], ignore_index=True).drop_duplicates("timestamp", keep="last")
label_map = labels.set_index("timestamp")["zigzag_action"]
y_full = label_map.reindex(panel["timestamp"]).to_numpy()

train_mask = (panel["timestamp"] >= TRAIN_START) & (panel["timestamp"] <= TRAIN_END)
full_range_mask = (panel["timestamp"] >= TRAIN_START) & (panel["timestamp"] <= POST_OOS_END)
log(f"  TRAIN={train_mask.sum()}  전체 예측범위(TRAIN~POST_OOS)={full_range_mask.sum()}")

sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

fee, slip = omega._load_fee_slip()
COST_MULT = 3.0  # 이 후속 분석은 결과 재현이 가장 강했던 cost3만 사용(스크리닝 범위 제한 명시)
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


def valid_indices(mask_arr: np.ndarray, window: int, require_label: bool) -> np.ndarray:
    idx = np.flatnonzero(mask_arr)
    idx = idx[idx >= window - 1]
    if require_label:
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
            y = y_full[idx]
            return torch.from_numpy(w.T.copy()), (int(y) if not pd.isna(y) else -1)

    return WindowDataset


# ---------------------------------------------------------------------------
# 1. 학습 + 전체 패널(TRAIN~POST_OOS) bar단위 예측 저장
# ---------------------------------------------------------------------------

all_preds = {}  # (variant, seed) -> np.ndarray(action) aligned to full_range_idx
full_range_idx_by_variant = {}

for variant_name, cfg in FEATURE_VARIANTS.items():
    log(f"\n{'='*100}\n변형: {variant_name}\n{'='*100}")
    seq_cols, p = cfg["cols"], cfg["params"]
    window = p["window"]
    raw = panel[seq_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    train_idx_all = valid_indices(train_mask.to_numpy(), window, require_label=True)
    full_idx = valid_indices(full_range_mask.to_numpy(), window, require_label=False)
    full_range_idx_by_variant[variant_name] = full_idx
    split_point = int(len(train_idx_all) * 0.85)
    fit_idx, es_idx = train_idx_all[:split_point], train_idx_all[split_point:]

    fit_rows = np.concatenate([np.arange(i - window + 1, i + 1) for i in fit_idx[:3000]])
    mean, std = raw[fit_rows].mean(axis=0), raw[fit_rows].std(axis=0)
    std[std < 1e-6] = 1.0
    raw_std = np.clip((raw - mean) / std, -10, 10).astype(np.float32)
    WindowDataset = make_dataset_cls(raw_std, window)

    seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_FINAL_SEEDS)
    log(f"  최종 N={N_FINAL_SEEDS}시드(무작위): {seeds}")

    for seed in seeds:
        torch.manual_seed(seed)
        model = TCNClassifier(len(seq_cols), p["hidden"], 3, p["n_blocks"], p["kernel_size"], p["dropout"]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        loss_fn = nn.CrossEntropyLoss()
        es_loader = DataLoader(WindowDataset(es_idx), batch_size=1024, shuffle=False)
        rng = np.random.default_rng(seed)
        best_es_loss, best_state, bad_epochs = float("inf"), None, 0
        t0 = time.time()
        for epoch in range(MAX_EPOCHS):
            model.train()
            epoch_idx = rng.choice(fit_idx, size=min(MAX_WINDOWS_PER_EPOCH, len(fit_idx)), replace=False)
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
            if bad_epochs >= PATIENCE:
                break
        model.load_state_dict(best_state)
        model.eval()

        loader = DataLoader(WindowDataset(full_idx), batch_size=2048, shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
        pred = np.concatenate(preds)
        all_preds[(variant_name, seed)] = pred
        log(f"    seed={seed} 완료 ({time.time()-t0:.0f}s), 전체범위 예측 {len(pred)}개 저장")

# ---------------------------------------------------------------------------
# 2. 월별 PnL 분해 (전체 구간, TRAIN=인샘플 표시)
# ---------------------------------------------------------------------------

log(f"\n{'='*100}\n(1) 월별 PnL 분해\n{'='*100}")
months = pd.period_range("2024-06", "2026-08", freq="M")
monthly_rows = []
for variant_name in FEATURE_VARIANTS:
    full_idx = full_range_idx_by_variant[variant_name]
    ts_full = panel["timestamp"].to_numpy()[full_idx]
    for month in months:
        m_start, m_end = month.start_time, month.end_time
        m_mask = (ts_full >= m_start) & (ts_full <= m_end)
        if m_mask.sum() < 50:
            continue
        m_idx_global = full_idx[m_mask]
        ohlc = panel.iloc[m_idx_global][["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        seeds_for_variant = [s for (v, s) in all_preds if v == variant_name]
        for seed in seeds_for_variant:
            pred_full = all_preds[(variant_name, seed)]
            pred_m = pred_full[m_mask]
            dec = build_dec(pred_m)
            m_model = omega._metrics(ohlc, dec, fee=fee, slip=slip, cost_mult=COST_MULT)
            m_short = omega._metrics(ohlc, forced_side(dec, -1), fee=fee, slip=slip, cost_mult=COST_MULT)
            m_long = omega._metrics(ohlc, forced_side(dec, 1), fee=fee, slip=slip, cost_mult=COST_MULT)
            price_ret = (ohlc["close"].iloc[-1] / ohlc["close"].iloc[0] - 1) * 100
            in_sample = bool(m_start <= TRAIN_END)
            monthly_rows.append({
                "variant": variant_name, "seed": seed, "month": str(month), "in_sample": in_sample,
                "price_ret_pct": float(price_ret), "n_bars": int(m_mask.sum()),
                "model_pnl": m_model["pnl"], "model_trades": m_model["trades"],
                "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
                "beats_short": m_model["pnl"] > m_short["pnl"], "beats_long": m_model["pnl"] > m_long["pnl"],
                "beats_both": (m_model["pnl"] > m_short["pnl"]) and (m_model["pnl"] > m_long["pnl"]),
            })

monthly_df = pd.DataFrame(monthly_rows)
monthly_df.to_csv(OUT_DIR / "monthly_breakdown_full.csv", index=False)

monthly_agg = monthly_df.groupby(["variant", "month", "in_sample", "price_ret_pct"]).agg(
    model_pnl_mean=("model_pnl", "mean"), always_short_mean=("always_short_pnl", "mean"),
    always_long_mean=("always_long_pnl", "mean"), beats_both_rate=("beats_both", "mean"),
    n_seeds=("seed", "count"),
).reset_index().sort_values(["variant", "month"])
monthly_agg.to_csv(OUT_DIR / "monthly_breakdown_agg.csv", index=False)

for variant_name in FEATURE_VARIANTS:
    log(f"\n--- {variant_name}: 월별(시드평균) ---")
    sub = monthly_agg[monthly_agg["variant"] == variant_name]
    for _, r in sub.iterrows():
        tag = "인샘플" if r["in_sample"] else "OOS"
        log(f"  {r['month']} [{tag}] 가격{r['price_ret_pct']:+6.1f}%  model={r['model_pnl_mean']:+7.2f}  "
            f"short={r['always_short_mean']:+7.2f}  long={r['always_long_mean']:+7.2f}  "
            f"둘다승={r['beats_both_rate']*100:5.0f}%({r['n_seeds']}시드)")

# ---------------------------------------------------------------------------
# 3. VAL/OOS 내부 주간 반등 vs 하락주 풀링 비교 (out-of-sample 메커니즘 테스트)
# ---------------------------------------------------------------------------

log(f"\n{'='*100}\n(2) VAL/OOS 내부 주간 반등(bounce) vs 하락주 풀링 비교 (out-of-sample)\n{'='*100}")

close_series = panel.set_index("timestamp")["close"]
weekly_first = close_series.resample("W-MON").first()
weekly_last = close_series.resample("W-MON").last()
weekly_ret = (weekly_last / weekly_first - 1) * 100
week_of = panel["timestamp"].dt.to_period("W-MON")
week_sign = week_of.map(lambda w: "bounce" if weekly_ret.get(w.start_time + pd.Timedelta(days=6), np.nan) > 0 else "down")

bounce_rows = []
for variant_name in FEATURE_VARIANTS:
    full_idx = full_range_idx_by_variant[variant_name]
    ts_full = panel["timestamp"].to_numpy()[full_idx]
    week_sign_full = week_sign.to_numpy()[full_idx]
    for split_name, s_start, s_end in [("VAL", VAL_START, VAL_END), ("OOS", OOS_START, OOS_END)]:
        s_mask = (ts_full >= s_start) & (ts_full <= s_end)
        seeds_for_variant = [s for (v, s) in all_preds if v == variant_name]
        for regime_tag in ["bounce", "down"]:
            r_mask = s_mask & (week_sign_full == regime_tag)
            if r_mask.sum() < 50:
                continue
            r_idx_global = full_idx[r_mask]
            ohlc = panel.iloc[r_idx_global][["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
            for seed in seeds_for_variant:
                pred_full = all_preds[(variant_name, seed)]
                pred_r = pred_full[r_mask]
                dec = build_dec(pred_r)
                m_model = omega._metrics(ohlc, dec, fee=fee, slip=slip, cost_mult=COST_MULT)
                m_short = omega._metrics(ohlc, forced_side(dec, -1), fee=fee, slip=slip, cost_mult=COST_MULT)
                m_long = omega._metrics(ohlc, forced_side(dec, 1), fee=fee, slip=slip, cost_mult=COST_MULT)
                bounce_rows.append({
                    "variant": variant_name, "split": split_name, "regime": regime_tag, "seed": seed,
                    "n_bars": int(r_mask.sum()), "model_pnl": m_model["pnl"], "model_trades": m_model["trades"],
                    "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
                    "beats_short": m_model["pnl"] > m_short["pnl"], "beats_long": m_model["pnl"] > m_long["pnl"],
                })

bounce_df = pd.DataFrame(bounce_rows)
bounce_df.to_csv(OUT_DIR / "val_oos_bounce_vs_down_weeks.csv", index=False)

for variant_name in FEATURE_VARIANTS:
    for split_name in ["VAL", "OOS"]:
        for regime_tag in ["bounce", "down"]:
            sub = bounce_df[(bounce_df.variant == variant_name) & (bounce_df.split == split_name) & (bounce_df.regime == regime_tag)]
            if sub.empty:
                continue
            beat_s = int(sub["beats_short"].sum())
            beat_l = int(sub["beats_long"].sum())
            log(f"  [{variant_name}/{split_name}/{regime_tag}주, n_bars={sub['n_bars'].iloc[0]}] "
                f"model={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}  "
                f"short={sub['always_short_pnl'].mean():+7.2f}  long={sub['always_long_pnl'].mean():+7.2f}  "
                f"승(short)={beat_s}/{len(sub)}  승(long)={beat_l}/{len(sub)}")

# ---------------------------------------------------------------------------
# 4. POST_OOS 추세월 vs 반전월 분리 (이미 monthly_breakdown에 있지만 명시적으로 재정리)
# ---------------------------------------------------------------------------

log(f"\n{'='*100}\n(3) POST_OOS 월별 재확인 (추세월 vs 혼합월)\n{'='*100}")
post_oos_months = monthly_agg[(monthly_agg["month"].astype(str) >= "2026-03") & (monthly_agg["month"].astype(str) <= "2026-08")]
for variant_name in FEATURE_VARIANTS:
    log(f"\n--- {variant_name} ---")
    sub = post_oos_months[post_oos_months["variant"] == variant_name]
    for _, r in sub.iterrows():
        log(f"  {r['month']} 가격{r['price_ret_pct']:+6.1f}%  model={r['model_pnl_mean']:+7.2f}  "
            f"short={r['always_short_mean']:+7.2f}  long={r['always_long_mean']:+7.2f}  둘다승={r['beats_both_rate']*100:5.0f}%")

log(f"\n출력 디렉토리: {OUT_DIR}")
