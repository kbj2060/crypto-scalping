#!/usr/bin/env python3
"""분포적회귀(③) Gaussian NLL 회귀 head TabM 변형 -- 5-way 라벨로직 비교의 5번째 라벨.

zigzag/h48qual/dc/cusum(eth_tabm_label_logic_5way_seed_variant_20260820.py)과 근본적으로
다르다: 그 4개는 전부 zigzag_action{0,1,2} 스키마의 이산 direction/quality 라벨이라
`--direction-label-dir`만 바꿔 동일 3-head classification 파이프라인에 꽂았지만, 분포적회귀는
이산 배리어 자체가 없는 fixed-horizon 연속 forward log-return이 라벨이라(라벨빌더
`build_eth_distributional_regression_return_labels_20260819.py` docstring: "SL/TP 자체가
없음") classification head에 꽂을 수 없다. AskUserQuestion에서 사용자가 "진짜 regression head
구현"을 선택함(sign-proxy 근사가 아님, cond_acc 등 기존 지표와 아키텍처가 달라 직접비교
불가하다는 점 고지 후 선택).

horizon=h48(48bar=4시간) 사용 -- 변동성클러스터링 cheap-gate(eth_distributional_regression_
volatility_backtest_20260820.py)에서 garch_vol_z+h48이 유일 3-split 전부양수 생존조합이었던
것과 동일 horizon으로 맞춤(다른 horizon을 새로 고를 근거 없음).

아키텍처: `train_eval_omega1_2_tabm_3head_20260603.ThreeHeadTabM.encode()`(k=8 BatchEnsemble
트렁크: input_scale/bias -> in_proj -> [Linear*expert_scale->LayerNorm->SiLU->dropout->residual]
*layers, CFG 기본 하이퍼파라미터 그대로)를 바이트단위로 복제하되, direction_head(3)+
quality_head(3)+exit_head(2) 대신 dist_head(2, mu+log_sigma) 하나만 붙인다 -- "라벨로직만
바꾼다"는 원 취지를 아키텍처 레벨에서 최대한 지키기 위함. exit_head가 없는 이유: 분포적회귀
라벨 자체가 TP/SL/보유기간 개념이 없어 exit_head가 예측할 대상이 원천적으로 없음.

학습 루프도 `_fit_expert_3head`와 동일 관례(AdamW same lr/weight_decay, k-앙상블 loss는
mean(dim=k) 후 배치평균, grad-clip=2.0, 내부 85/15 조기종료 홀드아웃, patience=CFG.patience,
epochs=2 -- 나머지 4개와 동일, 이 세션에서 이미 epoch2->30 재학습으로 "조기종료 전 수렴
차이없음"이 검증된 것과 동일 스크리닝 관례를 그대로 적용하되 그 검증은 classification head
대상이었다는 점은 명시) -- 유일한 실질적 차이는 손실함수(Gaussian NLL)와 클래스 균형 샘플
가중치가 없다는 것(연속타겟이라 "클래스"가 없음).

손실: nll_k = 0.5*log(2*pi) + log_sigma_k + 0.5*((y-mu_k)/sigma_k)^2, sigma_k=exp(log_sigma_k)
(log_sigma를 [-6,3]으로 clamp -- sigma 약 0.0025~20 범위, 수치안정성).

추론시 k개 성분을 단일 Gaussian으로 축약: mu_final=mean_k(mu_k), sigma_final^2=
mean_k(sigma_k^2+mu_k^2)-mu_final^2(균등혼합 Gaussian의 정확한 total-variance 분해,
근사 아님).

트레이드 판정: z=mu_final/sigma_final(신호강도)를 TRAIN 분포의 |z| 백분위수 임계값(50/60/70/
80/90th)으로 이산화해 LONG/SHORT/CASH 도출 -- 나머지 4개의 quality_threshold 스윕과 동일하게
"TRAIN에서만 임계값 후보를 정하고 VAL PnL 1위를 고른 뒤 그 임계값의 OOS를 본다"는 이 세션
표준 causal-safe 절차를 그대로 따름. PnL은 왕복10bp 비용(이 세션 표준 가정) 차감한
fixed-horizon 홀드 수익(barrier/TP-SL 없음) -- 나머지 4개의 barrier 기반 TP/SL PnL과
방법론이 다르므로 report에 명시, 직접비교시 이 차이를 반드시 언급할 것."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import eth_dc_engineered_features_canonicaldata_20260820 as feat154  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402

omega = feat154.omega
MODEL_ID = "omega4_3head_parent72_loose_entry_quality_20260620"
HORIZON = "h48"
LABEL_COL = f"fwd_logret_{HORIZON}"
LABEL_DIR = ROOT / "tmp/eth_distributional_regression_return_labels_20260819"
ROUND_TRIP_COST = 0.0010  # 10bp, 이 세션 표준 비용가정(왕복)
Z_PERCENTILES = [50, 60, 70, 80, 90]


class DistRegressionTabM(nn.Module):
    """ThreeHeadTabM.encode()와 바이트단위 동일 트렁크 + dist_head(mu, log_sigma) 하나."""

    def __init__(self, n_features: int, *, cfg=parent.CFG) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.n_features = int(n_features)
        self.input_scale = nn.Parameter(torch.randn(self.k, self.n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, self.n_features))
        self.in_proj = nn.Linear(self.n_features, int(cfg.hidden))
        self.blocks = nn.ModuleList(nn.Linear(int(cfg.hidden), int(cfg.hidden)) for _ in range(max(0, int(cfg.layers) - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(self.k, int(cfg.hidden)) * 0.03 + 1.0) for _ in range(max(0, int(cfg.layers) - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(int(cfg.hidden)) for _ in range(max(0, int(cfg.layers))))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.dist_head = nn.Linear(int(cfg.hidden), 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        out = self.dist_head(h)
        mu = out[..., 0]
        log_sigma = out[..., 1].clamp(min=-6.0, max=3.0)
        return mu, log_sigma


def _base_input(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = frame.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.astype(np.float32)


def _load_fwd_return_labels(year: int) -> pd.DataFrame:
    df = pd.read_csv(LABEL_DIR / f"fwd_return_labels_{year}.csv", usecols=["timestamp", LABEL_COL], parse_dates=["timestamp"])
    before = len(df)
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    dropped = before - len(df)
    print(f"  year={year}: {before} rows, {dropped} dropped (tail NaN, horizon lookahead unavailable)", flush=True)
    return df


def _join_labels(frame: pd.DataFrame, labels: pd.DataFrame, *, tag: str) -> pd.DataFrame:
    merged = frame.merge(labels, on="timestamp", how="inner")
    if len(merged) == 0:
        raise RuntimeError(f"{tag}: inner join produced 0 rows")
    print(f"  {tag}: {len(frame)} feature rows x {len(labels)} label rows -> {len(merged)} joined", flush=True)
    return merged


def _fit_dist_regression(x_df: pd.DataFrame, y: np.ndarray, *, seed: int, epochs: int, device: torch.device, model_path: Path) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_np, scaler = parent._standardize_fit(x_df)
    y_np = np.asarray(y, dtype=np.float32)

    n = len(y_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)

    cfg = parent.CFG
    k = int(cfg.k)
    model = DistRegressionTabM(x_np.shape[1], cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    ds = TensorDataset(torch.from_numpy(x_np[train_idx]), torch.from_numpy(y_np[train_idx]))
    dl = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    log2pi = math.log(2.0 * math.pi)
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mu, log_sigma = model(xb)
            yb_k = yb[:, None].expand(-1, k)
            sigma = torch.exp(log_sigma)
            nll = 0.5 * log2pi + log_sigma + 0.5 * ((yb_k - mu) / sigma) ** 2
            loss = nll.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_np[val_idx]).to(device)
            vy = torch.from_numpy(y_np[val_idx]).to(device)
            vmu, vlog_sigma = model(vx)
            vy_k = vy[:, None].expand(-1, k)
            vsigma = torch.exp(vlog_sigma)
            vloss = float((0.5 * log2pi + vlog_sigma + 0.5 * ((vy_k - vmu) / vsigma) ** 2).mean().detach().cpu())
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "scaler": scaler,
        "best_validation_loss": best_loss,
        "epochs_ran": last_epoch,
    }


@torch.no_grad()
def _predict_mu_sigma(model: DistRegressionTabM, x_df: pd.DataFrame, scaler: dict, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x_np = parent._standardize_apply(x_df, scaler)
    xt = torch.from_numpy(x_np).to(device)
    mu_k, log_sigma_k = model(xt)
    mu_k = mu_k.detach().cpu().numpy()
    sigma_k = np.exp(log_sigma_k.detach().cpu().numpy())
    mu = mu_k.mean(axis=1)
    sigma2 = (sigma_k**2 + mu_k**2).mean(axis=1) - mu**2
    sigma = np.sqrt(np.clip(sigma2, 1e-12, None))
    return mu, sigma


def _decisions_from_z(z: np.ndarray, threshold: float) -> np.ndarray:
    dec = np.zeros(len(z), dtype=np.int64)
    dec[z > threshold] = 1  # LONG
    dec[z < -threshold] = -1  # SHORT
    return dec


def _evaluate_threshold(dec: np.ndarray, realized: np.ndarray, mu: np.ndarray) -> dict[str, Any]:
    active = dec != 0
    n_trades = int(active.sum())
    if n_trades == 0:
        return {"trades": 0, "pnl_bps": 0.0, "wr": None, "cond_dir_acc": None, "long_entries": 0, "short_entries": 0}
    signed_realized = np.where(dec[active] == 1, realized[active], -realized[active])
    pnl_bps = float((signed_realized - ROUND_TRIP_COST).sum() * 10000.0)
    wr = float((signed_realized > 0).mean())
    cond_dir_acc = float((np.sign(mu[active]) == np.sign(realized[active])).mean())
    return {
        "trades": n_trades,
        "pnl_bps": pnl_bps,
        "wr": wr,
        "cond_dir_acc": cond_dir_acc,
        "long_entries": int((dec[active] == 1).sum()),
        "short_entries": int((dec[active] == -1).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    t0 = time.time()

    print("stage=load_features", flush=True)
    train_all, eval_df, _overlay_report = omega._load_omega_frames()
    feature_cols = omega._numeric_feature_cols(train_all, eval_df)
    if len(feature_cols) != 154:
        raise RuntimeError(f"expected 154 features, got {len(feature_cols)}")

    print("stage=load_labels", flush=True)
    label_2025 = _load_fwd_return_labels(2025)
    label_2026 = _load_fwd_return_labels(2026)

    train_all_j = _join_labels(train_all, label_2025, tag="train_all(2025)")
    eval_df_j = _join_labels(eval_df, label_2026, tag="oos(2026)")

    train_raw = train_all_j[train_all_j["timestamp"] < parent.SPLIT_TS].reset_index(drop=True)
    val_raw = train_all_j[train_all_j["timestamp"] >= parent.SPLIT_TS].reset_index(drop=True)
    oos_raw = eval_df_j.reset_index(drop=True)
    print(f"stage=split train={len(train_raw)} val={len(val_raw)} oos={len(oos_raw)}", flush=True)

    x_train = _base_input(train_raw, feature_cols)
    y_train = train_raw[LABEL_COL].to_numpy(dtype=np.float32)

    out_suffix = f"label5way_distreg_154feat_unified_single_model_seed{args.seed}_20260820"
    out_dir = ROOT / "tmp/causal_regen_20260516" / f"{MODEL_ID}_{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "dist_regression_model.pt"

    print(f"stage=fit seed={args.seed} epochs={args.epochs}", flush=True)
    fit_result = _fit_dist_regression(x_train, y_train, seed=args.seed, epochs=args.epochs, device=device, model_path=model_path)
    model = fit_result["model"]
    scaler = fit_result["scaler"]
    print(f"stage=fit_done epochs_ran={fit_result['epochs_ran']} best_validation_loss={fit_result['best_validation_loss']:.4f}", flush=True)

    torch.save({"state_dict": model.state_dict(), "scaler": scaler, "n_features": len(feature_cols), "feature_cols": feature_cols,
                "best_validation_loss": fit_result["best_validation_loss"], "epochs_ran": fit_result["epochs_ran"], "horizon": HORIZON,
                "seed": args.seed}, model_path)

    splits = {}
    for name, raw in [("train", train_raw), ("validation", val_raw), ("oos", oos_raw)]:
        x = _base_input(raw, feature_cols)
        mu, sigma = _predict_mu_sigma(model, x, scaler, device=device)
        realized = raw[LABEL_COL].to_numpy(dtype=np.float64)
        z = mu / np.clip(sigma, 1e-8, None)
        splits[name] = {"mu": mu, "sigma": sigma, "z": z, "realized": realized, "rows": len(raw)}

    train_abs_z = np.abs(splits["train"]["z"])
    thresholds = sorted(set(float(np.percentile(train_abs_z, p)) for p in Z_PERCENTILES))
    print(f"stage=thresholds train_abs_z_percentiles={thresholds}", flush=True)

    variants = []
    for th in thresholds:
        row: dict[str, Any] = {"z_threshold": th}
        for name in ("train", "validation", "oos"):
            dec = _decisions_from_z(splits[name]["z"], th)
            metrics = _evaluate_threshold(dec, splits[name]["realized"], splits[name]["mu"])
            row[name] = metrics
        variants.append(row)

    variants_with_val_trades = [v for v in variants if v["validation"]["trades"] > 0]
    ranking_pool = variants_with_val_trades if variants_with_val_trades else variants
    ranking = sorted(ranking_pool, key=lambda v: v["validation"]["pnl_bps"], reverse=True)
    best = ranking[0] if ranking else None

    report = {
        "model_id": MODEL_ID,
        "label_type": "distributional_regression_gaussian_nll",
        "horizon": HORIZON,
        "label_col": LABEL_COL,
        "label_dir": str(LABEL_DIR),
        "architecture": "DistRegressionTabM (ThreeHeadTabM.encode() trunk, mu+log_sigma head only, no exit_head)",
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_ran": fit_result["epochs_ran"],
        "best_validation_loss_nll": fit_result["best_validation_loss"],
        "round_trip_cost": ROUND_TRIP_COST,
        "split_rows": {name: splits[name]["rows"] for name in splits},
        "z_threshold_candidates_from_train_percentiles": dict(zip(Z_PERCENTILES, thresholds)) if len(thresholds) == len(Z_PERCENTILES) else None,
        "ranking_by_validation_pnl": ranking,
        "best_by_validation_pnl": best,
        "methodology_note": (
            "fixed-horizon(48bar) 홀드 PnL, barrier/TP-SL 없음 -- zigzag/h48qual/dc/cusum의 "
            "report.json PnL(barrier 기반 TP/SL)과 방법론이 다르므로 절대수치 직접비교 불가. "
            "cond_dir_acc(부호일치율)만 그 4개의 조건부방향정확도와 개념적으로 비교 가능."
        ),
        "artifacts": {"out_dir": str(out_dir), "model_path": str(model_path)},
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"stage=done seed={args.seed} elapsed={time.time() - t0:.1f}s report={report_path}", flush=True)
    print(json.dumps({"report": str(report_path), "best_by_validation_pnl": best}, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
