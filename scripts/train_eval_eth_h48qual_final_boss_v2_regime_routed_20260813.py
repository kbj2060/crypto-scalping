"""'최종 보스' v2 -- 레짐전문가 라우팅(bull/bear/chop) 추가. 라이브 라우터와의 격차(v1:
OOS 격차 15배->동적사이징 후 3.6배)를 더 좁히기 위해, 라이브가 쓰는 것과 같은 구조(레짐별
독립 direction+quality)를 v1의 요소들(one-vs-rest direction/방향별 MFE 분위수 회귀/
FINAL12+latent 피쳐/비대칭 게이팅/동적 사이징) 위에 얹는다.

v1(train_eval_eth_h48qual_final_boss_20260812.py) 대비 변경점: TRAIN을 bull/bear/chop
3개 레짐으로 분할(hard._route_id, 라이브와 동일 라우팅 컬럼)해서 레짐마다 독립적으로
direction(3개)+quality(2개) 모델을 학습 -- 총 15개 LightGBM(v1은 5개). 게이팅 컷오프와
사이징 순위-매핑도 레짐별로 독립 계산(라이브의 EXPERT_SCALES 사상과 같은 정신). 오토인코더
latent는 공유(레짐과 무관한 피쳐 압축이라 분리 안 함)."""
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

OUT_ROOT = ROOT / "tmp/eth_h48qual_final_boss_v2_regime_routed_20260813"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LATENT_DIM = 16
LONG_QUANTILE = 0.85
SHORT_QUANTILE = 0.60
LEV_FLOOR, LEV_CAP = 1.5, 5.0
MARGIN_FLOOR, MARGIN_CAP = 0.30, 0.90
NOTIONAL_CAP = 1.8

FINAL12 = [
    "cvp_regime", "funding_pressure_diff1", "ou_halflife", "m7_vae_error_dt288",
    "realized_skewness", "mta_funding", "sig_whale_dt288", "sum_toptrader_long_short_ratio_dt288",
    "vwap_dist_24", "funding_roc_48", "breakout_strength",
    "regime3_current_sensitive_wide24_chop_prob",
]
ROUTE_COLS = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
]
EXPERT_NAMES = ["bull", "bear", "chop"]

TB_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619"
DIRECTION_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"

ACTION_CASH, ACTION_LONG, ACTION_SHORT = 0, 1, 2


def log(msg: str) -> None:
    print(msg, flush=True)


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 1단계: 넓은 원시피쳐풀 + route 컬럼 로딩 (FINAL12 몽키패치 이전)
# ---------------------------------------------------------------------------

log("=== 1단계: 넓은 피쳐풀 로딩 (FINAL12 몽키패치 이전) ===")
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega_raw  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

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


def extract_route(frame: pd.DataFrame) -> pd.DataFrame:
    # route_id만 남긴다 -- ROUTE_COLS 중 chop_prob는 FINAL12에도 있어 merge 시 컬럼명 충돌
    out = pd.DataFrame({"timestamp": frame["timestamp"].to_numpy(), "route_id": hard._route_id(frame)})
    return out


# ---------------------------------------------------------------------------
# 2단계: FINAL12 프레임 (몽키패치 이후) + MFE 타겟 병합
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


def build_frame(raw, mfe, latent, route):
    m = raw.merge(mfe, on="timestamp", how="inner").merge(latent, on="timestamp", how="inner").merge(route, on="timestamp", how="inner")
    return m.reset_index(drop=True)


def rank_to_unit(train_passing_vals, new_vals):
    if len(train_passing_vals) == 0:
        return np.full(len(new_vals), 0.5)
    train_sorted = np.sort(train_passing_vals)
    ranks = np.searchsorted(train_sorted, new_vals, side="right") / len(train_sorted)
    return np.clip(ranks, 0.0, 1.0)


def run_seed(seed: int) -> dict:
    log(f"\n########## seed={seed} ##########")
    ae = train_autoencoder(seed)
    latent_train = extract_latent(ae, wide_train)
    latent_val = extract_latent(ae, wide_val)
    latent_oos = extract_latent(ae, wide_oos)
    route_train = extract_route(wide_train)
    route_val = extract_route(wide_val)
    route_oos = extract_route(wide_oos)

    train_f = build_frame(frames["train_raw"], mfe_frames["train"], latent_train, route_train)
    val_f = build_frame(frames["val_raw"], mfe_frames["val"], latent_val, route_val)
    oos_f = build_frame(frames["oos_raw"], mfe_frames["oos"], latent_oos, route_oos)
    log(f"  train={len(train_f)}  val={len(val_f)}  oos={len(oos_f)}")
    log(f"  TRAIN 레짐 분포: {dict(zip(*np.unique(train_f['route_id'], return_counts=True)))}")

    X_train_all = train_f[EXT_FEATURES].astype(np.float64)
    X_val_all = val_f[EXT_FEATURES].astype(np.float64)
    X_oos_all = oos_f[EXT_FEATURES].astype(np.float64)

    final_val = np.zeros(len(val_f), dtype=np.int64)
    final_oos = np.zeros(len(oos_f), dtype=np.int64)
    margin_val = np.zeros(len(val_f), dtype=np.float64)
    leverage_val = np.zeros(len(val_f), dtype=np.float64)
    margin_oos = np.zeros(len(oos_f), dtype=np.float64)
    leverage_oos = np.zeros(len(oos_f), dtype=np.float64)
    dir_action_val = np.zeros(len(val_f), dtype=np.int64)
    dir_action_oos = np.zeros(len(oos_f), dtype=np.int64)
    cutoffs = {}

    for regime_idx, regime_name in enumerate(EXPERT_NAMES):
        train_mask = (train_f["route_id"] == regime_idx).to_numpy()
        val_mask = (val_f["route_id"] == regime_idx).to_numpy()
        oos_mask = (oos_f["route_id"] == regime_idx).to_numpy()
        n_train = int(train_mask.sum())
        log(f"  [{regime_name}] TRAIN n={n_train}  VAL n={int(val_mask.sum())}  OOS n={int(oos_mask.sum())}")
        if n_train < 500:
            log(f"  [{regime_name}] 표본 부족(<500) -- 이 레짐은 전부 CASH 처리")
            continue

        X_train = X_train_all[train_mask]
        y_action_train = train_f.loc[train_mask, "zigzag_action"].to_numpy(dtype=np.int64)
        targets_train = {"cash": (y_action_train == 0).astype(int), "long": (y_action_train == 1).astype(int), "short": (y_action_train == 2).astype(int)}

        dir_probs_train, dir_probs_val, dir_probs_oos = {}, {}, {}
        for side in ("cash", "long", "short"):
            clf = lgb.LGBMClassifier(objective="binary", n_estimators=400, num_leaves=31, learning_rate=0.05,
                                      random_state=seed, verbosity=-1, n_jobs=-1)
            clf.fit(X_train, targets_train[side])
            dir_probs_train[side] = clf.predict_proba(X_train)[:, 1]
            dir_probs_val[side] = clf.predict_proba(X_val_all[val_mask])[:, 1] if val_mask.any() else np.array([])
            dir_probs_oos[side] = clf.predict_proba(X_oos_all[oos_mask])[:, 1] if oos_mask.any() else np.array([])

        def argmax_action(probs, n):
            if n == 0:
                return np.zeros(0, dtype=np.int64)
            mat = np.stack([probs["cash"], probs["long"], probs["short"]], axis=1)
            return mat.argmax(axis=1)

        da_train = argmax_action(dir_probs_train, len(X_train))
        da_val = argmax_action(dir_probs_val, int(val_mask.sum()))
        da_oos = argmax_action(dir_probs_oos, int(oos_mask.sum()))
        dir_action_val[val_mask] = da_val
        dir_action_oos[oos_mask] = da_oos

        long_mfe_train = train_f.loc[train_mask, "tb_long_mfe_h48_conservative"].to_numpy()
        short_mfe_train = train_f.loc[train_mask, "tb_short_mfe_h48_conservative"].to_numpy()
        long_mask_train = y_action_train == 1
        short_mask_train = y_action_train == 2

        q_models = {}
        for side, mask, target in [("long", long_mask_train, long_mfe_train), ("short", short_mask_train, short_mfe_train)]:
            if mask.sum() < 30:
                q_models[side] = None
                continue
            reg = lgb.LGBMRegressor(objective="regression", n_estimators=400, num_leaves=15, learning_rate=0.05,
                                     random_state=seed, verbosity=-1, n_jobs=-1)
            reg.fit(X_train[mask], target[mask])
            q_models[side] = reg

        q_pred_train = {"long": q_models["long"].predict(X_train) if q_models["long"] else np.zeros(len(X_train)),
                         "short": q_models["short"].predict(X_train) if q_models["short"] else np.zeros(len(X_train))}
        q_pred_val = {"long": q_models["long"].predict(X_val_all[val_mask]) if q_models["long"] and val_mask.any() else np.zeros(int(val_mask.sum())),
                      "short": q_models["short"].predict(X_val_all[val_mask]) if q_models["short"] and val_mask.any() else np.zeros(int(val_mask.sum()))}
        q_pred_oos = {"long": q_models["long"].predict(X_oos_all[oos_mask]) if q_models["long"] and oos_mask.any() else np.zeros(int(oos_mask.sum())),
                      "short": q_models["short"].predict(X_oos_all[oos_mask]) if q_models["short"] and oos_mask.any() else np.zeros(int(oos_mask.sum()))}

        long_cutoff = np.quantile(q_pred_train["long"][da_train == 1], LONG_QUANTILE) if (da_train == 1).any() else np.inf
        short_cutoff = np.quantile(q_pred_train["short"][da_train == 2], SHORT_QUANTILE) if (da_train == 2).any() else np.inf
        cutoffs[regime_name] = {"long": float(long_cutoff), "short": float(short_cutoff)}
        log(f"  [{regime_name}] long_cutoff={long_cutoff:.5f}  short_cutoff={short_cutoff:.5f}")

        def regime_final(da, q_pred, cutoff_l, cutoff_s):
            out = np.zeros(len(da), dtype=np.int64)
            out[(da == 1) & (q_pred["long"] >= cutoff_l)] = ACTION_LONG
            out[(da == 2) & (q_pred["short"] >= cutoff_s)] = ACTION_SHORT
            return out

        fv = regime_final(da_val, q_pred_val, long_cutoff, short_cutoff)
        fo = regime_final(da_oos, q_pred_oos, long_cutoff, short_cutoff)
        final_val[val_mask] = fv
        final_oos[oos_mask] = fo

        train_pass_long = q_pred_train["long"][(da_train == 1) & (q_pred_train["long"] >= long_cutoff)]
        train_pass_short = q_pred_train["short"][(da_train == 2) & (q_pred_train["short"] >= short_cutoff)]

        for split_final, split_qpred, split_mask, m_arr, l_arr in [
            (fv, q_pred_val, val_mask, margin_val, leverage_val),
            (fo, q_pred_oos, oos_mask, margin_oos, leverage_oos),
        ]:
            local_m = np.zeros(split_mask.sum(), dtype=np.float64)
            local_l = np.zeros(split_mask.sum(), dtype=np.float64)
            long_m = split_final == ACTION_LONG
            short_m = split_final == ACTION_SHORT
            for mask2, train_pass, q in [(long_m, train_pass_long, split_qpred["long"]), (short_m, train_pass_short, split_qpred["short"])]:
                if not mask2.any():
                    continue
                unit = rank_to_unit(train_pass, q[mask2])
                m = MARGIN_FLOOR + (MARGIN_CAP - MARGIN_FLOOR) * unit
                lev = LEV_FLOOR + (LEV_CAP - LEV_FLOOR) * unit
                notional = np.minimum(m * lev, NOTIONAL_CAP)
                lev = notional / np.maximum(m, 1e-9)
                local_m[mask2] = m
                local_l[mask2] = lev
            m_arr[split_mask] = local_m
            l_arr[split_mask] = local_l

    return {
        "seed": seed, "val_frame": val_f, "oos_frame": oos_f,
        "final_val": final_val, "final_oos": final_oos,
        "dir_action_val": dir_action_val, "dir_action_oos": dir_action_oos,
        "margin_val": margin_val, "leverage_val": leverage_val,
        "margin_oos": margin_oos, "leverage_oos": leverage_oos,
        "cutoffs": cutoffs,
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
            {"cutoffs": result["cutoffs"],
             "val_action_counts": {int(k): int(v) for k, v in zip(*np.unique(result["final_val"], return_counts=True))},
             "oos_action_counts": {int(k): int(v) for k, v in zip(*np.unique(result["final_oos"], return_counts=True))}},
            indent=2))
        log(f"seed={seed} 저장 완료 -> {out_dir}")
    log(f"\n출력: {out_dir}")
