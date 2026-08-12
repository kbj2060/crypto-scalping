"""사용자 지시("post-OOS 재검증 진행") 실행 — 신규 탐색 축 스카우팅 (c)-2 후보.
`zigzag_action` 라벨은 2026-02-28에서 끊기지만(OOS 종료와 거의 동일 지점), zig075 소스 패널의
raw 피쳐+OHLC는 2026-08-04까지 gap 없이 존재함(직접 확인). 즉 라벨이 없어 분류 정확도는 못
재지만, **진짜 처음 보는 5개월 구간(2026-03-01~08-04, 44,970행)에서 PnL만 계산**하는 건
가능하다 -- 이게 이 서브 프로젝트가 지금까지 확보한 가장 순수한 blind OOS 테스트다.

가격 성격 확인(직접): 2026-03(+7.0%)/04(+7.3%) 상승, 05(-11.3%)/06(-21.9%) 하락,
07(+18.3%) 재반등, 전체 순변화 -5.4% -- VAL/OOS의 일방적 하락과 달리 방향이 여러 번 뒤집히는
혼합 레짐. 오늘 TCN 전체 HP서치(150 trial x 5피쳐셋)에서 VAL 성적이 상대적으로 나았던
raw_lite/ohlcv_minimal 두 변형의 이미 확정된 최적 HP를 그대로 재사용(재탐색 없음) -- N=5 신규
무작위 시드로 재학습(TRAIN은 동일 2024-06~2025-09), VAL/OOS는 참고용으로 계속 확인하고
POST_OOS는 PnL만(분류 지표 불가) always-short/long과 대조."""
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
OUT_DIR = ROOT / "tmp/eth_h48qual_tcn_post_oos_extension_20260812"
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

log("zig075 소스 패널 + zigzag_action 라벨 로딩...")
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

y_full = label_map.reindex(panel["timestamp"]).to_numpy()  # POST_OOS 구간은 전부 NaN(라벨 없음)
train_mask = (panel["timestamp"] >= TRAIN_START) & (panel["timestamp"] <= TRAIN_END)
val_mask = (panel["timestamp"] >= VAL_START) & (panel["timestamp"] <= VAL_END)
oos_mask = (panel["timestamp"] >= OOS_START) & (panel["timestamp"] <= OOS_END)
post_oos_mask = (panel["timestamp"] >= POST_OOS_START) & (panel["timestamp"] <= POST_OOS_END)
log(f"  TRAIN={train_mask.sum()}  VAL={val_mask.sum()}  OOS={oos_mask.sum()}  POST_OOS={post_oos_mask.sum()}(라벨 없음, PnL만)")

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


all_results = {}
for variant_name, cfg in FEATURE_VARIANTS.items():
    log(f"\n{'='*100}\n변형: {variant_name}\n{'='*100}")
    seq_cols, p = cfg["cols"], cfg["params"]
    window = p["window"]
    raw = panel[seq_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    train_idx_all = valid_indices(train_mask.to_numpy(), window, require_label=True)
    val_idx = valid_indices(val_mask.to_numpy(), window, require_label=True)
    oos_idx = valid_indices(oos_mask.to_numpy(), window, require_label=False)  # OOS도 라벨 있지만 일관되게
    post_oos_idx = valid_indices(post_oos_mask.to_numpy(), window, require_label=False)
    split_point = int(len(train_idx_all) * 0.85)
    fit_idx, es_idx = train_idx_all[:split_point], train_idx_all[split_point:]

    fit_rows = np.concatenate([np.arange(i - window + 1, i + 1) for i in fit_idx[:3000]])
    mean, std = raw[fit_rows].mean(axis=0), raw[fit_rows].std(axis=0)
    std[std < 1e-6] = 1.0
    raw_std = np.clip((raw - mean) / std, -10, 10).astype(np.float32)
    WindowDataset = make_dataset_cls(raw_std, window)

    seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_FINAL_SEEDS)
    log(f"  최종 N={N_FINAL_SEEDS}시드(무작위): {seeds}")

    pnl_rows, clf_rows = [], []
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
        log(f"    seed={seed} 완료 ({time.time()-t0:.0f}s)")

        for split_name, idx, has_label in [("VAL", val_idx, True), ("OOS", oos_idx, True), ("POST_OOS", post_oos_idx, False)]:
            loader = DataLoader(WindowDataset(idx), batch_size=1024, shuffle=False)
            preds = []
            with torch.no_grad():
                for xb, _ in loader:
                    preds.append(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
            pred = np.concatenate(preds)
            if has_label:
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
                                  "beats_always_short": m_model["pnl"] > m_short["pnl"],
                                  "beats_always_long": m_model["pnl"] > m_long["pnl"]})

    pnl_df = pd.DataFrame(pnl_rows)
    clf_df = pd.DataFrame(clf_rows)
    pnl_df.to_csv(OUT_DIR / f"pnl_{variant_name}.csv", index=False)
    clf_df.to_csv(OUT_DIR / f"clf_{variant_name}.csv", index=False)

    log(f"\n  --- {variant_name} 요약 ---")
    summary = {"variant": variant_name, "params": p, "seeds": seeds, "cells": {}}
    for split_name in ["VAL", "OOS", "POST_OOS"]:
        if split_name in ("VAL", "OOS"):
            csub = clf_df[clf_df["split"] == split_name]
            if len(csub):
                log(f"  {split_name} 분류: balanced_acc={csub['balanced_accuracy'].mean():.3f}±{csub['balanced_accuracy'].std():.3f}")
        for cost_name in COST_MULTS:
            sub = pnl_df[(pnl_df["split"] == split_name) & (pnl_df["cost"] == cost_name)]
            beat_s = int(sub["beats_always_short"].sum())
            beat_l = int(sub["beats_always_long"].sum())
            diff = (sub["model_pnl"] - sub["always_short_pnl"]).to_numpy()
            wp = stats.wilcoxon(diff, alternative="greater")[1] if len(sub) >= 5 and np.any(diff != 0) else float("nan")
            log(f"    [{split_name}/{cost_name}] model={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}"
                f"  always_short={sub['always_short_pnl'].mean():+7.2f}±{sub['always_short_pnl'].std():5.2f}"
                f"  always_long={sub['always_long_pnl'].mean():+7.2f}±{sub['always_long_pnl'].std():5.2f}"
                f"  승(vs short)={beat_s}/{len(sub)}  승(vs long)={beat_l}/{len(sub)}  wilcoxon_p={wp:.4f}")
            summary["cells"][f"{split_name}_{cost_name}"] = {
                "model_pnl_mean": float(sub["model_pnl"].mean()), "always_short_pnl_mean": float(sub["always_short_pnl"].mean()),
                "always_long_pnl_mean": float(sub["always_long_pnl"].mean()), "beats_short": beat_s, "beats_long": beat_l,
                "n": len(sub), "wilcoxon_p_vs_short": float(wp)}
    all_results[variant_name] = summary
    (OUT_DIR / f"summary_{variant_name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

log(f"\n\n{'='*100}\nPOST-OOS(2026-03~08) 확장 검증 -- 종합\n{'='*100}")
for variant_name, s in all_results.items():
    log(f"\n[{variant_name}]")
    for cell, r in s["cells"].items():
        if not cell.startswith("POST_OOS"):
            continue
        log(f"  {cell}: model={r['model_pnl_mean']:+7.2f}  always_short={r['always_short_pnl_mean']:+7.2f}  "
            f"always_long={r['always_long_pnl_mean']:+7.2f}  승(short)={r['beats_short']}/{r['n']}  승(long)={r['beats_long']}/{r['n']}")

(OUT_DIR / "all_variants_summary.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
log(f"\n출력 디렉토리: {OUT_DIR}")
