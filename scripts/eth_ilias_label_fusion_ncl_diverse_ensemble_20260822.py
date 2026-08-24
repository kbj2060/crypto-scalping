#!/usr/bin/env python3
"""Zou(2025)/고전 Negative Correlation Learning 정신의 "다양성규제 공동학습 다중모델 앙상블"을
실제로 구현·테스트 -- 08-22 사용자 재질문("제대로 해보자")에 대한 응답. 앞서 문헌검토에서 낸
판단("Zou의 다양성규제는 다중모델 앙상블용 기법이라 우리 단일모델·다중피처 stack과 카테고리가
다르다")을 실제 구현으로 검증한다.

`docs/experiments/ilias_eth_label_fusion_combined_model_research_20260821.md` §6(stack, 단일
로지스틱회귀+3피처)과 다른 점: 이번엔 zigzag/cusum/h48qual을 각각 **독립 파라미터를 가진
작은 로지스틱모델**로 두고(피처 공유 없음), 개별 정확도(BCE) + 앙상블 예측 분산(다양성
보너스)을 **공동 손실**로 함께 최적화한다(NCL의 ambiguity decomposition 정신 -- 정확한
Zou(2025) 논문 수식을 재현한 것은 아니며 초록 수준 이해를 기반으로 한 원리 재현임을 명시).

  L = mean_i(BCE_i) - LAMBDA * Var_i(p_i)   (Var_i는 매 bar 3개 모델 예측의 분산, 다양성 보너스)

LAMBDA=0.0(대조군, 다양성보너스 없음=3개 모델을 그냥 각자 독립적으로 학습하는 것과 근사 동일)
vs LAMBDA=0.5(다양성규제 적용)를 비교 -- 다양성규제가 (a) 실제로 모델간 예측 상관을 낮추는지,
(b) 그게 실제 결합 성능 개선으로 이어지는지 둘 다 확인한다.

방법론(기존 스태킹 테스트와 동일 walk-forward 원칙): 2024로 학습, 2025 held-out 평가, OOS
미접촉. N=6시드. 피처는 각 라벨의 dir_p_long/dir_p_short/dir_confidence/quality_for_action
4개(라벨간 공유 없음 -- 진짜 독립 모델).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import eth_ilias_label_fusion_train_period_feasibility_20260821 as fsn  # noqa: E402

omega = fsn.omega
parent = fsn.parent
_metrics_with_ledger = fsn.ledger_mod._metrics_with_ledger
SEEDS = fsn.SEEDS
FEE, SLIP = omega._load_fee_slip()
COST_MULT = 3.0

PREFIX = "omega1_regime3_expertdq_oof_"
FEAT_COLS = ["dir_p_long", "dir_p_short", "dir_confidence", "quality_for_action"]
FORWARD_BARS = 48
LAMBDAS = [0.0, 0.5]
N_EPOCHS = 1500
LR = 0.1
OUT_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/ncl_diverse_ensemble_2025holdout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)  # optimizer init only -- data/labels are what actually vary across SEEDS


class ThreeMemberEnsemble(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(n_feat, 1) for _ in range(3)])

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [torch.sigmoid(head(x).squeeze(-1)) for head, x in zip(self.heads, xs)]


def _load_label_preds(label: str, seed: int) -> pd.DataFrame:
    import json
    out_dir = fsn.OUT_ROOT / f"{fsn.MODEL_ID}_label5way_{label}_154feat_ilias_anchored_seed{seed}_20260821"
    report = json.loads((out_dir / "report.json").read_text())
    best = report["ranking_by_validation_pnl"][0]
    q_tag = f"q{int(round(float(best['quality_threshold']) * 100.0)):03d}"
    preds = pd.concat(
        [pd.read_csv(out_dir / f"train_predictions_{q_tag}.csv", parse_dates=["timestamp"]),
         pd.read_csv(out_dir / f"validation_predictions_{q_tag}.csv", parse_dates=["timestamp"])],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)
    return preds


def _combine_from_side(dec_template: pd.DataFrame, side_combined: np.ndarray) -> pd.DataFrame:
    active = side_combined != 0
    return pd.DataFrame({
        "timestamp": dec_template["timestamp"].to_numpy(),
        "action": np.where(side_combined == 1, 1, np.where(side_combined == -1, 2, 0)),
        "side": side_combined,
        "notional_exposure": np.where(active, dec_template["notional_exposure"].to_numpy(), 0.0),
        "leverage": np.where(active, dec_template["leverage"].to_numpy(), 1.0),
        "take_profit": np.where(active, dec_template["take_profit"].to_numpy(), 0.0),
        "stop_loss": np.where(active, dec_template["stop_loss"].to_numpy(), 0.0),
        "max_hold_bars": dec_template["max_hold_bars"].to_numpy(),
        "cooldown_bars": dec_template["cooldown_bars"].to_numpy(),
    })


def _train_eval_one(seed: int, lam: float, full_raw: pd.DataFrame, bench: float) -> dict:
    preds = {lb: _load_label_preds(lb, seed) for lb in ("zigzag", "cusum", "h48qual")}
    decs = {lb: parent._to_decisions(preds[lb], oof=True) for lb in preds}
    for lb in decs:
        decs[lb].insert(0, "timestamp", preds[lb]["timestamp"].to_numpy())

    ts = preds["zigzag"]["timestamp"]
    feats = {}
    for lb in preds:
        cols = [f"{PREFIX}{c}" for c in FEAT_COLS]
        feats[lb] = preds[lb][cols].to_numpy(dtype=np.float64)

    frame = pd.DataFrame({"timestamp": ts})
    frame = frame.merge(full_raw[["timestamp", "y_up", "fwd_ret"]], on="timestamp", how="inner")
    keep = frame["fwd_ret"].notna().to_numpy()
    frame = frame[keep].reset_index(drop=True)
    for lb in feats:
        feats[lb] = feats[lb][keep]

    year = frame["timestamp"].dt.year
    fit_mask = (year == 2024).to_numpy()
    eval_mask = (year == 2025).to_numpy()
    y = frame["y_up"].to_numpy(dtype=np.float64)

    # standardize each label's features using FIT-only mean/std (no leakage) -- raw dir_p_long/
    # dir_confidence/quality_for_action live on different scales, which stalled gradient descent
    # (all seeds converged to ~ln2 BCE, i.e. no learning at all) before this fix.
    for lb in feats:
        mu = feats[lb][fit_mask].mean(axis=0, keepdims=True)
        sd = feats[lb][fit_mask].std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        feats[lb] = (feats[lb] - mu) / sd

    model = ThreeMemberEnsemble(n_feat=len(FEAT_COLS))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xs_fit = [torch.tensor(feats[lb][fit_mask], dtype=torch.float32) for lb in ("zigzag", "cusum", "h48qual")]
    y_fit = torch.tensor(y[fit_mask], dtype=torch.float32)
    bce = nn.BCELoss()

    loss_trace = []
    for epoch in range(N_EPOCHS):
        opt.zero_grad()
        probs = model(xs_fit)
        individual = sum(bce(p, y_fit) for p in probs) / 3.0
        stacked = torch.stack(probs, dim=0)
        diversity = ((stacked - stacked.mean(dim=0)) ** 2).mean()
        loss = individual - lam * diversity
        loss.backward()
        opt.step()
        if epoch in (0, N_EPOCHS // 2, N_EPOCHS - 1):
            loss_trace.append((epoch, float(individual.item()), float(diversity.item())))

    with torch.no_grad():
        xs_eval = [torch.tensor(feats[lb][eval_mask], dtype=torch.float32) for lb in ("zigzag", "cusum", "h48qual")]
        probs_eval = model(xs_eval)
        p_z, p_c, p_h = (p.numpy() for p in probs_eval)
        final_bce = [float(bce(p, torch.tensor(y[fit_mask], dtype=torch.float32)).item())
                     for p in model(xs_fit)]

    corr_zc = float(np.corrcoef(p_z, p_c)[0, 1])
    corr_zh = float(np.corrcoef(p_z, p_h)[0, 1])
    corr_ch = float(np.corrcoef(p_c, p_h)[0, 1])
    p_ens = (p_z + p_c + p_h) / 3.0
    ens_side = np.where(p_ens > 0.55, 1, np.where(p_ens < 0.45, -1, 0)).astype(np.int64)

    eval_ts = frame.loc[eval_mask, ["timestamp"]].reset_index(drop=True)
    eval_frame = full_raw.merge(eval_ts, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    assert len(eval_frame) == len(eval_ts)
    dec_template = decs["zigzag"].merge(eval_ts, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    assert len(dec_template) == len(eval_ts)

    dec_combined = _combine_from_side(dec_template, ens_side)
    agg, ledger = _metrics_with_ledger(eval_frame, dec_combined, fee=FEE, slip=SLIP, cost_mult=COST_MULT)
    ledger.to_csv(OUT_DIR / f"ncl_lam{lam}_seed{seed}_2025holdout_trade_ledger.csv", index=False)
    lf = agg["long_entries"] / max(agg["long_entries"] + agg["short_entries"], 1)

    return {
        "seed": seed, "lambda": lam, "trades": agg["trades"], "pnl": agg["pnl"], "mdd": agg["mdd"],
        "long_entries": agg["long_entries"], "short_entries": agg["short_entries"], "long_frac": lf,
        "corr_zigzag_cusum": corr_zc, "corr_zigzag_h48qual": corr_zh, "corr_cusum_h48qual": corr_ch,
        "mean_abs_corr": float(np.mean([abs(corr_zc), abs(corr_zh), abs(corr_ch)])),
        "final_bce_zigzag": final_bce[0], "final_bce_cusum": final_bce[1], "final_bce_h48qual": final_bce[2],
        "loss_trace": loss_trace,
    }


def main() -> None:
    full_raw = pd.concat(
        [pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv",
                      usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
         for y in (2024, 2025)], ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)
    full_raw["fwd_ret"] = full_raw["close"].shift(-FORWARD_BARS) / full_raw["close"] - 1.0
    full_raw["y_up"] = (full_raw["fwd_ret"] > 0).astype(int)
    raw_2025 = full_raw[full_raw["timestamp"].dt.year == 2025]
    bench = (float(raw_2025["close"].iloc[-1]) / float(raw_2025["close"].iloc[0]) - 1.0) * 100.0
    print(f"2025-only always-long benchmark: {bench:+.2f}%", flush=True)

    rows = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            r = _train_eval_one(seed, lam, full_raw, bench)
            trace_str = " -> ".join(f"ep{e}:ind={i:.4f},div={d:.4f}" for e, i, d in r["loss_trace"])
            print(f"[lambda={lam} seed={seed}] pnl={r['pnl']:7.2f}% trades={r['trades']:3d} "
                  f"long_frac={r['long_frac']:.3f} mean|corr|={r['mean_abs_corr']:.3f} "
                  f"bce(z/c/h)={r['final_bce_zigzag']:.3f}/{r['final_bce_cusum']:.3f}/{r['final_bce_h48qual']:.3f} "
                  f"trace[{trace_str}]", flush=True)
            r.pop("loss_trace")
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)

    print(f"\n=== NCL-style diverse ensemble, 2025-holdout aggregate (N=6 seeds), bench={bench:+.2f}% ===")
    for lam in LAMBDAS:
        sub = df[df["lambda"] == lam]
        corr_pnl = sub["long_frac"].corr(sub["pnl"])
        print(f"lambda={lam}: mean_pnl={sub['pnl'].mean():7.2f} std={sub['pnl'].std():6.2f} "
              f"mean_long_frac={sub['long_frac'].mean():.3f} long_frac_corr_pnl={corr_pnl:6.3f} "
              f"mean|cross-model corr|={sub['mean_abs_corr'].mean():.3f} "
              f"trades=[{sub['trades'].min()},{sub['trades'].max()}]")
    print(f"\nsaved {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
