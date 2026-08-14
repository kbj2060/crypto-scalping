"""'최종 보스' v3 -- v1(flat, 레짐분할 없음) 구조로 되돌아가서 라이브 라우터처럼 2개 컴포넌트를
우선순위로 결합(컴포넌트A가 CASH면 컴포넌트B 확인, 라이브의 h48qual→zig075 PRIORITY와 동일
정신). v2(레짐라우팅)는 VAL 개선/OOS 악화(과적합)로 폐기, v1이 더 신뢰할 만하다는 결론이라
v1을 그대로 둔 채 기회셋만 넓힌다.

컴포넌트 A("h48qual 역할"): v1과 완전히 동일(LONG q0.85/SHORT q0.60, 이미 검증된 설정).
컴포넌트 B("zig075 역할"): 같은 아키텍처(one-vs-rest direction + 방향별 MFE 분위수 회귀 +
FINAL12+latent 피쳐)를 쓰되, (1) A와 완전히 독립적인 시드로 재학습(모델 인스턴스 자체가
다름 -- 라이브의 두 컴포넌트가 서로 다른 학습으로 나온 것과 같은 정신), (2) 더 엄격한 게이팅
(LONG q0.90/SHORT q0.75, 라이브 zig075의 threshold=0.75가 h48qual의 0.50보다 높은 것과
같은 관계)을 적용해 "A가 못 잡은 것 중 그래도 확신도 높은 것만" 잡는 역할을 하도록 한다.

결합: 매 bar마다 A의 결정이 CASH가 아니면 A를 쓰고, A가 CASH면 B를 확인해서 B가 CASH가
아니면 B를 쓴다(둘 다 CASH면 CASH) -- trading_bot_modules/omega4_6_1_live.py의 PRIORITY
라우터와 동일 로직."""
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

OUT_ROOT = ROOT / "tmp/eth_h48qual_final_boss_v3_dual_component_20260813"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LATENT_DIM = 16
LEV_FLOOR, LEV_CAP = 1.5, 5.0
MARGIN_FLOOR, MARGIN_CAP = 0.30, 0.90
NOTIONAL_CAP = 1.8

COMPONENT_CFG = {
    "A": {"long_q": 0.85, "short_q": 0.60},   # h48qual 역할 -- v1과 동일
    "B": {"long_q": 0.90, "short_q": 0.75},   # zig075 역할 -- 더 엄격, 독립 시드로 재학습
}

FINAL12 = [
    "cvp_regime", "funding_pressure_diff1", "ou_halflife", "m7_vae_error_dt288",
    "realized_skewness", "mta_funding", "sig_whale_dt288", "sum_toptrader_long_short_ratio_dt288",
    "vwap_dist_24", "funding_roc_48", "breakout_strength",
    "regime3_current_sensitive_wide24_chop_prob",
]

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
# 1단계: 넓은 원시피쳐풀 로딩 (FINAL12 몽키패치 걸리기 전)
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
# 2단계: FINAL12 프레임 + MFE 타겟 병합 (오토인코더는 컴포넌트마다 별도 재학습 -- latent도
# 컴포넌트별 독립성을 유지하기 위해)
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


def rank_to_unit(train_passing_vals, new_vals):
    if len(train_passing_vals) == 0:
        return np.full(len(new_vals), 0.5)
    train_sorted = np.sort(train_passing_vals)
    ranks = np.searchsorted(train_sorted, new_vals, side="right") / len(train_sorted)
    return np.clip(ranks, 0.0, 1.0)


def train_component(seed: int, long_q: float, short_q: float) -> dict:
    """v1의 run_seed()와 동일 아키텍처 -- 게이팅 quantile만 파라미터화."""
    ae = train_autoencoder(seed)
    latent_train = extract_latent(ae, wide_train)
    latent_val = extract_latent(ae, wide_val)
    latent_oos = extract_latent(ae, wide_oos)

    train_f = build_frame(frames["train_raw"], mfe_frames["train"], latent_train)
    val_f = build_frame(frames["val_raw"], mfe_frames["val"], latent_val)
    oos_f = build_frame(frames["oos_raw"], mfe_frames["oos"], latent_oos)

    y_action_train = train_f["zigzag_action"].to_numpy(dtype=np.int64)
    targets_train = {"cash": (y_action_train == 0).astype(int), "long": (y_action_train == 1).astype(int), "short": (y_action_train == 2).astype(int)}

    X_train = train_f[EXT_FEATURES].astype(np.float64)
    X_val = val_f[EXT_FEATURES].astype(np.float64)
    X_oos = oos_f[EXT_FEATURES].astype(np.float64)

    dir_probs = {"train": {}, "val": {}, "oos": {}}
    for side in ("cash", "long", "short"):
        clf = lgb.LGBMClassifier(objective="binary", n_estimators=400, num_leaves=31, learning_rate=0.05,
                                  random_state=seed, verbosity=-1, n_jobs=-1)
        clf.fit(X_train, targets_train[side])
        for split_name, X in [("train", X_train), ("val", X_val), ("oos", X_oos)]:
            dir_probs[split_name][side] = clf.predict_proba(X)[:, 1]

    def argmax_action(probs):
        mat = np.stack([probs["cash"], probs["long"], probs["short"]], axis=1)
        return mat.argmax(axis=1)

    dir_action = {s: argmax_action(dir_probs[s]) for s in ("train", "val", "oos")}

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

    long_cutoff = np.quantile(q_pred["train"]["long"][dir_action["train"] == 1], long_q) if (dir_action["train"] == 1).any() else np.inf
    short_cutoff = np.quantile(q_pred["train"]["short"][dir_action["train"] == 2], short_q) if (dir_action["train"] == 2).any() else np.inf

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
        "val_frame": val_f, "oos_frame": oos_f,
        "final_val": final_val, "final_oos": final_oos,
        "margin_val": margin_val, "leverage_val": leverage_val,
        "margin_oos": margin_oos, "leverage_oos": leverage_oos,
        "long_cutoff": float(long_cutoff), "short_cutoff": float(short_cutoff),
    }


def combine_priority(comp_a: dict, comp_b: dict, split_key_final: str, split_key_margin: str, split_key_leverage: str) -> dict:
    """라이브 PRIORITY 라우터와 동일: A가 CASH 아니면 A, A가 CASH면 B."""
    fa = comp_a[split_key_final]
    fb = comp_b[split_key_final]
    use_b = fa == ACTION_CASH
    final = np.where(use_b, fb, fa)
    margin = np.where(use_b, comp_b[split_key_margin], comp_a[split_key_margin])
    leverage = np.where(use_b, comp_b[split_key_leverage], comp_a[split_key_leverage])
    source = np.where(final == ACTION_CASH, "cash", np.where(use_b, "B", "A"))
    return {"final": final, "margin": margin, "leverage": leverage, "source": source}


def run_seed_pair(seed_a: int, seed_b: int) -> dict:
    log(f"\n########## seed_A={seed_a}  seed_B={seed_b} ##########")
    log("  컴포넌트 A(h48qual 역할) 학습...")
    comp_a = train_component(seed_a, COMPONENT_CFG["A"]["long_q"], COMPONENT_CFG["A"]["short_q"])
    log(f"  [A] long_cutoff={comp_a['long_cutoff']:.5f}  short_cutoff={comp_a['short_cutoff']:.5f}")
    log("  컴포넌트 B(zig075 역할, 독립 시드+엄격 게이팅) 학습...")
    comp_b = train_component(seed_b, COMPONENT_CFG["B"]["long_q"], COMPONENT_CFG["B"]["short_q"])
    log(f"  [B] long_cutoff={comp_b['long_cutoff']:.5f}  short_cutoff={comp_b['short_cutoff']:.5f}")

    val_combo = combine_priority(comp_a, comp_b, "final_val", "margin_val", "leverage_val")
    oos_combo = combine_priority(comp_a, comp_b, "final_oos", "margin_oos", "leverage_oos")

    a_val_cash = int((comp_a["final_val"] == ACTION_CASH).sum())
    b_rescued_val = int(((comp_a["final_val"] == ACTION_CASH) & (comp_b["final_val"] != ACTION_CASH)).sum())
    a_oos_cash = int((comp_a["final_oos"] == ACTION_CASH).sum())
    b_rescued_oos = int(((comp_a["final_oos"] == ACTION_CASH) & (comp_b["final_oos"] != ACTION_CASH)).sum())
    log(f"  VAL: A가 CASH인 {a_val_cash}행 중 B가 살린 거래 {b_rescued_val}건")
    log(f"  OOS: A가 CASH인 {a_oos_cash}행 중 B가 살린 거래 {b_rescued_oos}건")

    return {
        "val_frame": comp_a["val_frame"], "oos_frame": comp_a["oos_frame"],
        "final_val": val_combo["final"], "final_oos": oos_combo["final"],
        "margin_val": val_combo["margin"], "leverage_val": val_combo["leverage"],
        "margin_oos": oos_combo["margin"], "leverage_oos": oos_combo["leverage"],
        "source_val": val_combo["source"], "source_oos": oos_combo["source"],
        "a_val_cash": a_val_cash, "b_rescued_val": b_rescued_val,
        "a_oos_cash": a_oos_cash, "b_rescued_oos": b_rescued_oos,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-pairs", required=True, help="A1:B1,A2:B2,...")
    ap.add_argument("--out-tag", default="smoketest")
    args = ap.parse_args()
    pairs = [tuple(int(x) for x in p.split(":")) for p in args.seed_pairs.split(",")]

    out_dir = OUT_ROOT / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed_a, seed_b in pairs:
        result = run_seed_pair(seed_a, seed_b)
        val_out = result["val_frame"][["timestamp"]].copy()
        val_out["final_action"] = result["final_val"]
        val_out["margin_fraction"] = result["margin_val"]
        val_out["leverage"] = result["leverage_val"]
        val_out["source"] = result["source_val"]
        oos_out = result["oos_frame"][["timestamp"]].copy()
        oos_out["final_action"] = result["final_oos"]
        oos_out["margin_fraction"] = result["margin_oos"]
        oos_out["leverage"] = result["leverage_oos"]
        oos_out["source"] = result["source_oos"]
        val_out.to_csv(out_dir / f"val_decisions_s{seed_a}_{seed_b}.csv", index=False)
        oos_out.to_csv(out_dir / f"oos_decisions_s{seed_a}_{seed_b}.csv", index=False)
        (out_dir / f"meta_s{seed_a}_{seed_b}.json").write_text(json.dumps(
            {"seed_a": seed_a, "seed_b": seed_b,
             "a_val_cash": result["a_val_cash"], "b_rescued_val": result["b_rescued_val"],
             "a_oos_cash": result["a_oos_cash"], "b_rescued_oos": result["b_rescued_oos"],
             "val_action_counts": {int(k): int(v) for k, v in zip(*np.unique(result["final_val"], return_counts=True))},
             "oos_action_counts": {int(k): int(v) for k, v in zip(*np.unique(result["final_oos"], return_counts=True))}},
            indent=2))
        log(f"seed_pair=({seed_a},{seed_b}) 저장 완료 -> {out_dir}")
    log(f"\n출력: {out_dir}")
