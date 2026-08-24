#!/usr/bin/env python3
""""Denoised Labels for Financial Time-Series Data via Self-Supervised Learning" (arXiv:
2112.10139, Hericourt? -- 저자 미확인, WebFetch로 방법론만 확인)의 Conv1D denoising
autoencoder를 ETH 5분봉 종가에 재현. 사용자 지시로 이 논문의 실제 방법을 그대로 구현
(reference_direction_quality_exit_label_methodology_20260819.md에서 "더 최신 로직"으로
인용했던 것).

논문 방법론(WebFetch로 arxiv.org/html/2112.10139 확인, PDF 렌더 불가라 원문 수식은 미확인
-- 이 구현은 요약 설명 기반 재현이라는 점을 명시):
- Conv1D 인코더(2층) + Conv1DTranspose 디코더(2층) + Conv1D+Sigmoid 출력층
- pretext task: raw 종가 L-bar 윈도우(노이즈 입력)를 받아, 같은 윈도우의 여러 SMA/EMA
  조합(n개 채널, "순수" 타겟)을 복원하도록 MSE로 학습 -- 실제 미래/정답이 필요 없는
  자기지도 태스크(SMA/EMA 자체가 과거+현재 가격만으로 계산되는 causal 통계량).
- Sigmoid 출력에 맞춰 각 윈도우를 자기 자신의 [min,max]로 min-max 정규화(논문이 명시한
  정규화 방식은 원문에서 확인 못 해 이 프로젝트 관례상 합리적인 선택으로 채택 -- 가정으로
  명시).
- 학습 후 전체 시계열에 통과시켜 얻은 n개 출력채널의 평균(un-normalize)을 "정제된 종가"로
  채택(논문이 정확히 어느 채널/조합을 "the reconstructed price"로 쓰는지 원문 확인 못 함,
  앙상블 평균이 가장 방어 가능한 합성 방법이라 판단 -- 가정으로 명시)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

L = 64  # window length in bars (5.3h at 5m bars)
MA_WINDOWS = [4, 8, 16, 32]  # SMA+EMA at each -> n = 2*4 = 8 channels
SEED = 260620
EPOCHS_CAP = 40
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_ilias1_price_denoiser_conv1d_20260819"


def _load_full_price_series() -> pd.DataFrame:
    """Loads TRAIN(2025q1-q3) + VAL + OOS-Q1 + OOS-Q2 close/high/low as one continuous
    causally-ordered frame, tagging each row with its window label for later slicing."""
    frames = []
    for wname, wd in gate.WINDOW_DEFS.items():
        frame = sweep.load_frame(wd["start"], wd["end"], base_csv=wd["base_csv"], wide24_csv=wd["wide24_csv"])
        frame, _ = gate._drop_route_nan(frame)
        frame = frame[["timestamp", "close"]].copy()
        frame["window"] = wname
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return full


def _build_ma_targets(close: pd.Series) -> np.ndarray:
    channels = []
    for w in MA_WINDOWS:
        channels.append(close.rolling(window=w, min_periods=1).mean().to_numpy())
    for w in MA_WINDOWS:
        channels.append(close.ewm(span=w, adjust=False, min_periods=1).mean().to_numpy())
    return np.stack(channels, axis=1)  # (n_rows, n_channels)


def _windowed_minmax_pairs(close: np.ndarray, ma_targets: np.ndarray, *, l: int) -> tuple[np.ndarray, np.ndarray]:
    """For each valid end-index i (trailing window [i-l+1, i]), returns
    x: (l,) normalized raw close, y: (n_channels, l) normalized MA targets."""
    n = len(close)
    n_channels = ma_targets.shape[1]
    xs = np.zeros((n - l + 1, l), dtype=np.float32)
    ys = np.zeros((n - l + 1, n_channels, l), dtype=np.float32)
    for idx, i in enumerate(range(l - 1, n)):
        window_close = close[i - l + 1: i + 1]
        window_ma = ma_targets[i - l + 1: i + 1, :].T  # (n_channels, l)
        lo, hi = window_close.min(), window_close.max()
        span = max(hi - lo, 1e-8)
        xs[idx] = (window_close - lo) / span
        ys[idx] = (window_ma - lo) / span
    return xs, ys


class Conv1DDenoiser(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.ConvTranspose1d(16, 16, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.out = nn.Sequential(nn.Conv1d(16, n_channels, kernel_size=5, padding=2), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x.unsqueeze(1))  # (batch, 32, L)
        h = self.dec(h)  # (batch, 16, L)
        return self.out(h)  # (batch, n_channels, L)


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    print("stage=load_price", flush=True)
    full = _load_full_price_series()
    close_full = full["close"].to_numpy(dtype=np.float64)
    print(f"loaded full continuous price series: rows={len(full)} range=[{full['timestamp'].min()}, {full['timestamp'].max()}]", flush=True)

    ma_targets = _build_ma_targets(full["close"])
    n_channels = ma_targets.shape[1]
    print(f"stage=build_windows n_channels={n_channels} window_len={L}", flush=True)
    xs, ys = _windowed_minmax_pairs(close_full, ma_targets, l=L)
    print(f"built {len(xs)} training windows", flush=True)

    n = len(xs)
    split = int(n * 0.85)
    x_t = torch.from_numpy(xs)
    y_t = torch.from_numpy(ys)
    ds_train = TensorDataset(x_t[:split], y_t[:split])
    ds_val = TensorDataset(x_t[split:], y_t[split:])
    dl_train = DataLoader(ds_train, batch_size=512, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=1024, shuffle=False)

    model = Conv1DDenoiser(n_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    stale = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS_CAP + 1):
        model.train()
        train_losses = []
        for xb, yb in dl_train:
            recon = model(xb)
            loss = loss_fn(recon, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                val_losses.append(float(loss_fn(model(xb), yb).item()))
        train_mean = sum(train_losses) / len(train_losses)
        val_mean = sum(val_losses) / len(val_losses)
        marker = ""
        if val_mean < best_val - 1e-7:
            best_val = val_mean
            import copy
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            marker = " *BEST*"
        else:
            stale += 1
        print(f"epoch={epoch:02d} train_mse={train_mean:.6f} val_mse={val_mean:.6f} stale={stale}/{PATIENCE}{marker}", flush=True)
        if stale >= PATIENCE:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    print(f"training done best_val_mse={best_val:.6f} elapsed={time.time()-t0:.1f}s", flush=True)

    print("stage=reconstruct_full_series", flush=True)
    model.eval()
    denoised_close = np.full(len(close_full), np.nan, dtype=np.float64)
    batch_size = 2048
    all_x = torch.from_numpy(xs)
    mins = np.array([close_full[i - L + 1: i + 1].min() for i in range(L - 1, len(close_full))])
    maxs = np.array([close_full[i - L + 1: i + 1].max() for i in range(L - 1, len(close_full))])
    spans = np.maximum(maxs - mins, 1e-8)
    with torch.no_grad():
        outs = []
        for s in range(0, len(all_x), batch_size):
            batch = all_x[s: s + batch_size]
            out = model(batch).numpy()  # (batch, n_channels, L)
            last_step = out[:, :, -1]  # take reconstruction at the LAST (current) bar of each window -- causal
            outs.append(last_step.mean(axis=1))  # ensemble-average across MA channels
        recon_last_norm = np.concatenate(outs)
    recon_last_price = recon_last_norm * spans + mins
    denoised_close[L - 1:] = recon_last_price

    out_df = full[["timestamp", "window"]].copy()
    out_df["close_raw"] = close_full
    out_df["close_denoised"] = denoised_close
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "denoised_price_series.csv"
    out_df.to_csv(out_path, index=False)
    print(f"saved denoised price series to {out_path}", flush=True)
    print("=== STUDY DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
