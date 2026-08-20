#!/usr/bin/env python3
"""
2026년 진짜 OOS 평가 스크립트
학습/검증에 전혀 사용되지 않은 2026년 데이터로 DSAC 모델 성능을 측정한다.

방법 1: DSACCompactTradingEnv (훈련 검증과 동일한 방식)
방법 2: Closed-loop 시뮬레이션 (DSACRouter, 실제 bar-by-bar 체결)
"""
from __future__ import annotations
import argparse, copy, json, math, os, sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, os.path.join(_ROOT_DIR, "ensemble")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.rl_runtime_primitives import OnlineHMMDetector, MultiTimeframeFeatures
from ensemble.train_rl_dsac_agent import (
    DSAC_STATE_DIM, GaussianActor, DSACCompactTradingEnv, DSACRouter,
)

ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)
RL_CSV   = "data/rl_training_data_full.csv"
FEAT_CSV = "data/training_features_5m.csv"
CKPT     = "data/ensemble/ckpt/best_dsac_agents.pth"
OUT_JSON = "data/ensemble/reports/eval_2026_true_oos.json"
FEE, SLIP = 0.0005, 0.0002


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DSAC on 2026 OOS with configurable RL/feature csv")
    p.add_argument("--rl-csv", default=RL_CSV)
    p.add_argument("--feat-csv", default=FEAT_CSV)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--out-json", default=OUT_JSON)
    return p.parse_args()


# ─── 유틸 ─────────────────────────────────────────────────────────────────────
def _load_2026_df(rl_csv: str = RL_CSV, feat_csv: str = FEAT_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(df26_env, df26_ohlc) 반환.
    df26_env  : RL CSV 그대로 (high/low 없음) — Method1 환경용
    df26_ohlc : open/high/low merge 추가      — Method2 closed-loop용
    """
    rl = pd.read_csv(rl_csv)
    rl["timestamp"] = pd.to_datetime(rl["timestamp"], errors="coerce")

    mask_rl = rl["timestamp"].dt.year == 2026
    df26_env = rl.loc[mask_rl].copy().reset_index(drop=True)

    df26_ohlc = df26_env.copy()
    need_ohlc = [c for c in ("open", "high", "low") if c not in df26_ohlc.columns]
    if need_ohlc:
        feat = pd.read_csv(feat_csv, usecols=["timestamp", "open", "high", "low"])
        feat["timestamp"] = pd.to_datetime(feat["timestamp"], errors="coerce")
        df_merged = df26_ohlc.merge(feat, on="timestamp", how="left", suffixes=("", "_feat"))
        for c in ("open", "high", "low"):
            feat_c = f"{c}_feat"
            if c not in df_merged.columns and feat_c in df_merged.columns:
                df_merged[c] = df_merged[feat_c]
        df26_ohlc = df_merged
    for c in ("close", "open", "high", "low"):
        if c in df26_ohlc.columns:
            df26_ohlc[c] = pd.to_numeric(df26_ohlc[c], errors="coerce")
    df26_ohlc = df26_ohlc.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["close", "open", "high", "low"]).reset_index(drop=True)

    print(f"[DATA] 2026 env_rows={len(df26_env):,}  ohlc_rows={len(df26_ohlc):,}  "
          f"range={df26_env['timestamp'].min()} -> {df26_env['timestamp'].max()}")
    return df26_env, df26_ohlc


def _sharpe(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(rets) < 3 or np.std(rets) < 1e-12:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * ANNUAL_FACTOR_5M)


def _mdd(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    run_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(run_max, 1e-12) - 1.0
    return float(np.min(dd)) * 100.0


# ─── 방법 1: DSACCompactTradingEnv (훈련 val과 동일한 코드 경로) ──────────────
def method1_training_env(df26: pd.DataFrame, actor: GaussianActor, device: str) -> dict:
    print("\n[METHOD 1] DSACCompactTradingEnv (훈련 검증과 동일한 방식)")

    hmm = OnlineHMMDetector()
    # HMM은 2024 데이터로 핏 (훈련과 동일하게)
    df_train = pd.read_csv(RL_CSV, usecols=["timestamp", "log_return", "garch_vol_z", "oi_change_rate"])
    df_train["timestamp"] = pd.to_datetime(df_train["timestamp"], errors="coerce")
    df_train_2024 = df_train[df_train["timestamp"].dt.year < 2025].copy()
    hmm.fit(df_train_2024, n_iter=30)
    print(f"  HMM fit on {len(df_train_2024):,} rows (2024 data)")

    mtf = MultiTimeframeFeatures(df26["close"].values.astype(np.float32))
    env = DSACCompactTradingEnv(
        df26,
        initial_balance=10000.0,
        fee=FEE,
        slip=SLIP,
        phase="val",
        hmm_detector=copy.deepcopy(hmm),
        mtf_features=mtf,
    )

    state = env.reset()
    done = False
    eq_curve = [env.initial_balance]
    actor.eval()
    with torch.no_grad():
        pbar = tqdm(desc="eval-training-env", unit="step")
        while not done:
            action = float(torch.tanh(actor.forward(
                torch.FloatTensor(state).unsqueeze(0).to(device)
            )[0]).item())
            state, _, done, info = env.step(action)
            bal = env.balance * (1.0 + (env.unrealized_pnl if env.pos is not None else 0.0))
            eq_curve.append(max(bal, 1e-8))
            pbar.update(1)
        pbar.close()

    pnl = (env.balance / env.initial_balance - 1.0) * 100.0
    wr  = env.win_rate * 100.0
    result = {
        "method": "training_env",
        "pnl_pct": round(pnl, 4),
        "wr_pct":  round(wr, 2),
        "trades":  env.total_trades,
        "sharpe":  round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
    }
    print(f"  PnL={pnl:.2f}%  WR={wr:.1f}%  Trades={env.total_trades}"
          f"  Sharpe={result['sharpe']:.3f}  MDD={result['mdd_pct']:.2f}%")
    return result


# ─── 방법 2: Closed-loop 시뮬레이션 (DSACRouter) ─────────────────────────────
def method2_closed_loop(df26: pd.DataFrame, actor: GaussianActor, device: str) -> dict:
    print("\n[METHOD 2] Closed-loop (DSACRouter, 다음봉 open 체결)")

    router = DSACRouter(actor, device=device)
    numeric_cols = [c for c in df26.columns if c != "timestamp"]
    values   = df26[numeric_cols].to_numpy(dtype=np.float64)
    open_np  = df26["open"].to_numpy(dtype=np.float64)
    high_np  = df26["high"].to_numpy(dtype=np.float64)
    low_np   = df26["low"].to_numpy(dtype=np.float64)
    close_np = df26["close"].to_numpy(dtype=np.float64)

    balance = 1.0
    pos: str | None = None
    entry_price = 0.0
    cur_lev = 0.0
    hold_count = 0
    trades = wins = 0
    eq_curve = [1.0]
    n = len(df26)

    def _unr(p, ep, cp, lv):
        if p is None or ep <= 0 or lv <= 0: return 0.0
        raw = (cp*(1-SLIP)-ep)/ep if p=="LONG" else (ep-cp*(1+SLIP))/ep
        return raw * lv

    def _real(p, ep, xp, lv):
        raw = (xp*(1-SLIP)-ep)/ep if p=="LONG" else (ep-xp*(1+SLIP))/ep
        return raw * lv

    for i in tqdm(range(n - 1), desc="eval-closed-loop", unit="bar"):
        cp         = float(close_np[i])
        next_open  = float(open_np[i + 1])
        next_close = float(close_np[i + 1])

        if pos is not None:
            hold_count += 1

        unr = _unr(pos, entry_price, cp, cur_lev)
        pos_dict = {
            "type": pos, "entry_price": float(entry_price),
            "unrealized": float(unr),
            "mdd": 0.0,
            "hold_norm": float(min(hold_count / 96.0, 1.0)),
            "margin_usage": float(cur_lev if pos else 0.0),
            "hold_count": float(hold_count),
        }
        row = values[i]
        features = {k: float(v) for k, v in zip(numeric_cols, row)}

        action_int, lev, _ = router.decide(features, pos_dict)
        lev = float(np.clip(lev, 0.0, 1.0))

        if pos is None:
            if action_int == 1 and lev > 0.0:
                pos = "LONG";  entry_price = next_open*(1+SLIP); cur_lev=lev; hold_count=0
                balance -= balance * FEE * cur_lev
            elif action_int == 2 and lev > 0.0:
                pos = "SHORT"; entry_price = next_open*(1-SLIP); cur_lev=lev; hold_count=0
                balance -= balance * FEE * cur_lev
        else:
            should_close = (
                action_int == 0
                or (action_int == 1 and pos == "SHORT")
                or (action_int == 2 and pos == "LONG")
            )
            if should_close:
                realized = _real(pos, entry_price, next_open, cur_lev)
                balance  = balance * (1.0 + realized) - balance * FEE * cur_lev
                trades  += 1
                if realized > 0: wins += 1
                pos = None; entry_price = 0.0; cur_lev = 0.0; hold_count = 0
            else:
                delta = abs(lev - cur_lev)
                if delta > 0.05:
                    balance -= balance * FEE * delta
                    cur_lev = lev

        eq = balance*(1+_unr(pos, entry_price, next_close, cur_lev)) if pos else balance
        eq_curve.append(max(float(eq), 1e-8))

    # 마지막 포지션 청산
    if pos and entry_price > 0:
        realized = _real(pos, entry_price, float(close_np[-1]), cur_lev)
        balance  = balance * (1.0 + realized) - balance * FEE * cur_lev
        trades  += 1
        if realized > 0: wins += 1

    pnl = (balance - 1.0) * 100.0
    wr  = (wins / trades * 100.0) if trades > 0 else 0.0
    result = {
        "method": "closed_loop",
        "pnl_pct": round(pnl, 4),
        "wr_pct":  round(wr, 2),
        "trades":  trades,
        "sharpe":  round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
    }
    print(f"  PnL={pnl:.2f}%  WR={wr:.1f}%  Trades={trades}"
          f"  Sharpe={result['sharpe']:.3f}  MDD={result['mdd_pct']:.2f}%")
    return result


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    print(f"[CKPT]   {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"[CKPT]   best_val_pnl={ckpt.get('best_pnl', '?'):.2f}%  "
          f"epoch={ckpt.get('epoch', '?')}")

    actor = GaussianActor(state_dim=DSAC_STATE_DIM).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    df26_env, df26_ohlc = _load_2026_df(args.rl_csv, args.feat_csv)

    r1 = method1_training_env(df26_env, actor, device)
    r2 = method2_closed_loop(df26_ohlc, actor, device)

    report = {
        "checkpoint": args.ckpt,
        "checkpoint_best_val_pnl": float(ckpt.get("best_pnl", 0.0)),
        "checkpoint_epoch": int(ckpt.get("epoch", 0)),
        "rl_csv": args.rl_csv,
        "feat_csv": args.feat_csv,
        "data_period": "2026-01-01 ~ 2026-02-28",
        "data_rows": len(df26_env),
        "note": "진짜 OOS: 학습/검증 어디에도 사용되지 않은 데이터",
        "results": [r1, r2],
    }

    print("\n" + "="*60)
    print("[SUMMARY] 체크포인트 val PnL vs 진짜 OOS 2026 PnL")
    print(f"  Best checkpoint val PnL : {ckpt.get('best_pnl', 0.0):.2f}%  (선택 편향 포함)")
    print(f"  Method1 (training env)  : {r1['pnl_pct']:.2f}%")
    print(f"  Method2 (closed-loop)   : {r2['pnl_pct']:.2f}%")
    print("="*60)

    def _to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(_to_serializable(report), f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {args.out_json}")


if __name__ == "__main__":
    main()
