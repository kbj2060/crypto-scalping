"""'최종 보스' -- 이 세션에서 절대성능(always_short 대조 무시) 기준으로 원값 신호가 확인된
요소들을 결합한 h48qual direction+quality 레이어 재설계.

결합 요소 (각각 이 세션에서 개별 검증됨):
1. **Direction**: 공유 3-class softmax 대신 one-vs-rest 독립 LightGBM 3개(LONG-vs-rest/
   SHORT-vs-rest/CASH-vs-rest, argmax 결합) -- confidence 간섭 회피(train_eval_eth_h48qual_
   onevsrest_specialist_20260812.py 방법론 재사용).
2. **Quality**: 방향별 독립 MFE 분위수 회귀(LightGBM), LONG/SHORT 각각 자기 쪽
   tb_long_mfe/tb_short_mfe만 학습(train_eval_omega4_h48qual_mfe_quality_regression_
   20260812.py에서 검증된 타겟, 재시뮬레이션 없이 사전계산 값 재사용).
3. **피쳐**: FINAL12 + 오토인코더 latent(16차원). Latent는 넓은 원시피쳐풀(omega 표준
   _numeric_feature_cols, FINAL12 몽키패치 걸리기 전에 뽑음)을 비지도 압축(verify_eth_
   h48qual_autoencoder_latent_mi_r2_gate_20260812.py와 동일 아키텍처: 64->32->16 디노이징
   AE, TRAIN-fit만으로 표준화, TRAIN 꼬리 15%로 조기종료) -- latent 단독이 아니라 FINAL12에
   추가하는 형태(이 세션 확인: latent만은 OOS서 더 나빠지지만 FINAL12+latent는 OOS 플러스
   전환).
4. **비대칭 게이팅**: 이 세션 전체가 일관되게 확인한 "롱은 어디서나 나쁘다" 발견 반영 --
   LONG 분위수 컷오프(0.85)를 SHORT(0.60)보다 훨씬 엄격하게.

TRAIN(2025-01~09, h48orig 표준)/VAL(2025-10~12)/OOS(2026-01~02) 그대로 재사용, 필수
always-short/long 대조도 계산해서 리포트에 남기지만(참고용) 이번 승격 기준은 절대 pnl."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT_ROOT = ROOT / "tmp/eth_h48qual_final_boss_20260812"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LATENT_DIM = 16
LONG_QUANTILE = 0.85
SHORT_QUANTILE = 0.60

FINAL12 = [
    "cvp_regime", "funding_pressure_diff1", "ou_halflife", "m7_vae_error_dt288",
    "realized_skewness", "mta_funding", "sig_whale_dt288", "sum_toptrader_long_short_ratio_dt288",
    "vwap_dist_24", "funding_roc_48", "breakout_strength",
    "regime3_current_sensitive_wide24_chop_prob",
]

TB_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619"
DIRECTION_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"


def log(msg: str) -> None:
    print(msg, flush=True)


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 1단계: 넓은 원시피쳐풀 로딩 + 오토인코더 latent 추출 (FINAL12 몽키패치 걸리기 전에 먼저 수행)
# ---------------------------------------------------------------------------

log("=== 1단계: 넓은 피쳐풀 로딩 (FINAL12 몽키패치 이전) ===")
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega_raw  # noqa: E402

train_all_wide, eval_df_wide, _ = omega_raw._load_omega_frames()
WIDE_COLS = omega_raw._numeric_feature_cols(train_all_wide, eval_df_wide)
log(f"  넓은 피쳐풀: {len(WIDE_COLS)}개 컬럼")

SPLIT_TS = pd.Timestamp("2025-10-01")
wide_train = train_all_wide[train_all_wide["timestamp"] < SPLIT_TS].reset_index(drop=True)
wide_val = train_all_wide[train_all_wide["timestamp"] >= SPLIT_TS].reset_index(drop=True)
wide_oos = eval_df_wide.reset_index(drop=True)
log(f"  wide_train={len(wide_train)}  wide_val={len(wide_val)}  wide_oos={len(wide_oos)}")


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
        return self.decoder(z), z


def train_autoencoder(seed: int) -> Autoencoder:
    seed_everything(seed)
    X_all = wide_train[WIDE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    split_point = int(len(X_all) * 0.85)
    fit_idx, es_idx = np.arange(split_point), np.arange(split_point, len(X_all))
    mean = X_all.iloc[fit_idx].mean()
    std = X_all.iloc[fit_idx].std().replace(0.0, 1.0)
    X_std = ((X_all - mean) / std).clip(-10, 10)

    X_fit = torch.tensor(X_std.iloc[fit_idx].to_numpy(), dtype=torch.float32)
    X_es = torch.tensor(X_std.iloc[es_idx].to_numpy(), dtype=torch.float32)

    model = Autoencoder(len(WIDE_COLS), LATENT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_fit), batch_size=2048, shuffle=True, generator=torch.Generator().manual_seed(seed))

    best_es_loss, best_state, patience, bad_epochs = float("inf"), None, 8, 0
    for epoch in range(200):
        model.train()
        for (batch,) in loader:
            noisy = batch + torch.randn_like(batch) * 0.05
            opt.zero_grad()
            recon, _ = model(noisy)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            es_loss = loss_fn(model(X_es)[0], X_es).item()
        if es_loss < best_es_loss - 1e-5:
            best_es_loss, best_state, bad_epochs = es_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    model._norm_mean, model._norm_std = mean, std
    return model


def extract_latent(model: Autoencoder, frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[WIDE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_std = ((X - model._norm_mean) / model._norm_std).clip(-10, 10)
    with torch.no_grad():
        z = model.encoder(torch.tensor(X_std.to_numpy(), dtype=torch.float32)).numpy()
    out = pd.DataFrame(z, columns=[f"latent_{i}" for i in range(LATENT_DIM)])
    out["timestamp"] = frame["timestamp"].to_numpy()
    return out


# ---------------------------------------------------------------------------
# 2단계: FINAL12 프레임 (몽키패치 이후) + MFE 타겟 + latent 병합
# ---------------------------------------------------------------------------

log("\n=== 2단계: FINAL12 프레임 로딩 + MFE 타겟 병합 ===")
import train_eval_omega4_3head_parent72_eth_h48qual_final12_h48orig_20260811 as h48orig_mod  # noqa: E402

parent_script = h48orig_mod.parent_script

frames = parent_script._prepare_frames(
    disable_tp_sl=False,
    direction_label_dir=DIRECTION_LABEL_DIR,
    quality_mode="quality_label_action",
    quality_label_dir=ROOT / "tmp/eth_h48_conservative_orig_padded_to_zigzag_timestamps_20260811",
    quality_min_edge=0.0010, quality_max_mae=0.0100, quality_min_mfe_mae=1.20, quality_max_hold_bars=288,
)

mfe_frames = {}
for split, fname in [("train", "train_triple_barrier_labels.csv"), ("val", "validation_triple_barrier_labels.csv"), ("oos", "oos_triple_barrier_labels.csv")]:
    df = pd.read_csv(TB_DIR / fname, usecols=["timestamp", "tb_long_mfe_h48_conservative", "tb_short_mfe_h48_conservative"], parse_dates=["timestamp"])
    mfe_frames[split] = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

LATENT_COLS = [f"latent_{i}" for i in range(LATENT_DIM)]
EXT_FEATURES = FINAL12 + LATENT_COLS


def build_frame(raw, mfe, latent):
    m = raw.merge(mfe, on="timestamp", how="inner").merge(latent, on="timestamp", how="inner")
    return m.reset_index(drop=True)


ACTION_CASH, ACTION_LONG, ACTION_SHORT = 0, 1, 2


def run_seed(seed: int) -> dict:
    log(f"\n########## seed={seed} ##########")
    ae = train_autoencoder(seed)
    latent_train = extract_latent(ae, wide_train)
    latent_val = extract_latent(ae, wide_val)
    latent_oos = extract_latent(ae, wide_oos)

    train_f = build_frame(frames["train_raw"], mfe_frames["train"], latent_train)
    val_f = build_frame(frames["val_raw"], mfe_frames["val"], latent_val)
    oos_f = build_frame(frames["oos_raw"], mfe_frames["oos"], latent_oos)
    log(f"  train={len(train_f)}  val={len(val_f)}  oos={len(oos_f)}")

    y_action_train = train_f["zigzag_action"].to_numpy(dtype=np.int64)
    targets_train = {"cash": (y_action_train == 0).astype(int), "long": (y_action_train == 1).astype(int), "short": (y_action_train == 2).astype(int)}

    X_train = train_f[EXT_FEATURES].astype(np.float64)
    X_val = val_f[EXT_FEATURES].astype(np.float64)
    X_oos = oos_f[EXT_FEATURES].astype(np.float64)

    # --- one-vs-rest direction ---
    dir_probs = {}
    for split_name, X in [("train", X_train), ("val", X_val), ("oos", X_oos)]:
        dir_probs[split_name] = {}
    dir_models = {}
    for side in ("cash", "long", "short"):
        clf = lgb.LGBMClassifier(objective="binary", n_estimators=400, num_leaves=31, learning_rate=0.05,
                                  random_state=seed, verbosity=-1, n_jobs=-1)
        clf.fit(X_train, targets_train[side])
        dir_models[side] = clf
        for split_name, X in [("train", X_train), ("val", X_val), ("oos", X_oos)]:
            dir_probs[split_name][side] = clf.predict_proba(X)[:, 1]

    def argmax_action(probs):
        mat = np.stack([probs["cash"], probs["long"], probs["short"]], axis=1)
        return mat.argmax(axis=1)

    dir_action = {s: argmax_action(dir_probs[s]) for s in ("train", "val", "oos")}

    # --- 방향별 독립 MFE 분위수 회귀 (LONG/SHORT 각각) ---
    long_mfe_train = train_f["tb_long_mfe_h48_conservative"].to_numpy()
    short_mfe_train = train_f["tb_short_mfe_h48_conservative"].to_numpy()
    long_mask_train = y_action_train == 1
    short_mask_train = y_action_train == 2

    q_models = {}
    for side, mask, target in [("long", long_mask_train, long_mfe_train), ("short", short_mask_train, short_mfe_train)]:
        reg = lgb.LGBMRegressor(objective="regression", n_estimators=400, num_leaves=15, learning_rate=0.05,
                                 random_state=seed, verbosity=-1, n_jobs=-1)
        reg.fit(X_train[mask], target[mask])
        q_models[side] = reg

    q_pred = {s: {} for s in ("train", "val", "oos")}
    for split_name, X in [("train", X_train), ("val", X_val), ("oos", X_oos)]:
        q_pred[split_name]["long"] = q_models["long"].predict(X)
        q_pred[split_name]["short"] = q_models["short"].predict(X)

    # 컷오프: TRAIN에서 그 방향으로 실제 고른 bar들의 예측분포 기준 quantile
    long_cutoff = np.quantile(q_pred["train"]["long"][dir_action["train"] == 1], LONG_QUANTILE) if (dir_action["train"] == 1).any() else np.inf
    short_cutoff = np.quantile(q_pred["train"]["short"][dir_action["train"] == 2], SHORT_QUANTILE) if (dir_action["train"] == 2).any() else np.inf
    log(f"  long_cutoff(q{LONG_QUANTILE})={long_cutoff:.5f}  short_cutoff(q{SHORT_QUANTILE})={short_cutoff:.5f}")

    def final_action(split_name):
        a = dir_action[split_name].copy()
        pass_long = (a == 1) & (q_pred[split_name]["long"] >= long_cutoff)
        pass_short = (a == 2) & (q_pred[split_name]["short"] >= short_cutoff)
        out = np.zeros(len(a), dtype=np.int64)
        out[pass_long] = ACTION_LONG
        out[pass_short] = ACTION_SHORT
        return out

    final_val = final_action("val")
    final_oos = final_action("oos")

    # --- 동적 리스크사이징: 라이브 사이드카와 같은 원리(예측 품질 순위 -> leverage/margin
    # 매핑, 같은 캡: leverage<=5.0, notional<=1.8) -- 정확히 같은 학습 타겟/매핑 파라미터는
    # 아니지만(라이브 사이드카는 별도 trade-ledger 회귀로 튜닝됨), TRAIN 내에서 그 방향으로
    # 실제 통과한 bar들의 예측 품질 순위를 [0,1]로 정규화해 선형 매핑하는 방식으로 핵심
    # 아이디어(확신도가 높을수록 더 크게 베팅)를 재현한다.
    LEV_FLOOR, LEV_CAP = 1.5, 5.0
    MARGIN_FLOOR, MARGIN_CAP = 0.30, 0.90
    NOTIONAL_CAP = 1.8

    def rank_to_unit(train_passing_vals, new_vals):
        if len(train_passing_vals) == 0:
            return np.full(len(new_vals), 0.5)
        train_sorted = np.sort(train_passing_vals)
        ranks = np.searchsorted(train_sorted, new_vals, side="right") / len(train_sorted)
        return np.clip(ranks, 0.0, 1.0)

    train_pass_long = q_pred["train"]["long"][(dir_action["train"] == 1) & (q_pred["train"]["long"] >= long_cutoff)]
    train_pass_short = q_pred["train"]["short"][(dir_action["train"] == 2) & (q_pred["train"]["short"] >= short_cutoff)]

    def size_columns(split_name, final):
        n = len(final)
        margin = np.zeros(n, dtype=np.float64)
        leverage = np.zeros(n, dtype=np.float64)
        long_mask = final == ACTION_LONG
        short_mask = final == ACTION_SHORT
        for mask, train_pass, q in [(long_mask, train_pass_long, q_pred[split_name]["long"]), (short_mask, train_pass_short, q_pred[split_name]["short"])]:
            if not mask.any():
                continue
            unit = rank_to_unit(train_pass, q[mask])
            m = MARGIN_FLOOR + (MARGIN_CAP - MARGIN_FLOOR) * unit
            lev = LEV_FLOOR + (LEV_CAP - LEV_FLOOR) * unit
            notional = np.minimum(m * lev, NOTIONAL_CAP)
            lev = notional / np.maximum(m, 1e-9)
            margin[mask] = m
            leverage[mask] = lev
        return margin, leverage

    margin_val, leverage_val = size_columns("val", final_val)
    margin_oos, leverage_oos = size_columns("oos", final_oos)

    return {
        "seed": seed, "val_frame": val_f, "oos_frame": oos_f,
        "final_val": final_val, "final_oos": final_oos,
        "dir_action_val": dir_action["val"], "dir_action_oos": dir_action["oos"],
        "margin_val": margin_val, "leverage_val": leverage_val,
        "margin_oos": margin_oos, "leverage_oos": leverage_oos,
        "long_cutoff": float(long_cutoff), "short_cutoff": float(short_cutoff),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--out-tag", default="smoketest")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    out_dir = OUT_ROOT / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        result = run_seed(seed)
        val_out = result["val_frame"][["timestamp"]].copy()
        val_out["final_action"] = result["final_val"]
        val_out["dir_action"] = result["dir_action_val"]
        val_out["margin_fraction"] = result["margin_val"]
        val_out["leverage"] = result["leverage_val"]
        oos_out = result["oos_frame"][["timestamp"]].copy()
        oos_out["final_action"] = result["final_oos"]
        oos_out["dir_action"] = result["dir_action_oos"]
        oos_out["margin_fraction"] = result["margin_oos"]
        oos_out["leverage"] = result["leverage_oos"]
        val_out.to_csv(out_dir / f"val_decisions_s{seed}.csv", index=False)
        oos_out.to_csv(out_dir / f"oos_decisions_s{seed}.csv", index=False)
        (out_dir / f"meta_s{seed}.json").write_text(json.dumps(
            {"long_cutoff": result["long_cutoff"], "short_cutoff": result["short_cutoff"],
             "val_action_counts": {int(k): int(v) for k, v in zip(*np.unique(result["final_val"], return_counts=True))},
             "oos_action_counts": {int(k): int(v) for k, v in zip(*np.unique(result["final_oos"], return_counts=True))}},
            indent=2))
        log(f"seed={seed} 저장 완료 -> {out_dir}")
    log(f"\n출력: {out_dir}")
