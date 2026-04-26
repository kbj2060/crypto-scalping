#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.train_rl_dsac_agent import DSACAgent
from scripts.backtest_polymarket_veto_panic import _build_slug_features

BASE_CKPT = ROOT / "data/ensemble/ckpt/best_dsac_agents.pth"
RL_CSV = ROOT / "data/rl_training_data_full.csv"
CACHE_DIR = ROOT / "data/ensemble/cache"


@dataclass
class Cfg:
    tz: str = "Asia/Seoul"
    train_ratio: float = 0.65
    val_ratio: float = 0.175
    test_ratio: float = 0.175
    fee: float = 0.0005
    slip: float = 0.0002
    lr: float = 8e-6
    epochs: int = 4
    updates_per_epoch: int = 260
    batch_size: int = 256
    hold_band: float = 0.0007
    action_scale: float = 0.010
    reward_mode: str = "dynamic"
    max_api_days: int = 420


def _scan_available_slugs(start_day: pd.Timestamp, end_day: pd.Timestamp) -> list[pd.Timestamp]:
    out = []
    for d in pd.date_range(start_day, end_day, freq="D", tz=start_day.tz):
        slug = f"ethereum-price-on-{d.strftime('%B').lower()}-{d.day}"
        try:
            r = requests.get("https://gamma-api.polymarket.com/events", params={"slug": slug}, timeout=8)
            if r.status_code != 200:
                continue
            j = r.json()
            ev = None
            if isinstance(j, list) and j:
                ev = j[0]
            elif isinstance(j, dict):
                arr = j.get("events", j.get("data", []))
                if isinstance(arr, list) and arr:
                    ev = arr[0]
            mk = list((ev or {}).get("markets", []) or [])
            if mk:
                out.append(d)
        except Exception:
            pass
        time.sleep(0.01)
    return out


def _fetch_polymarket_features_max(ts_min: pd.Timestamp, ts_max: pd.Timestamp, tz: str, max_api_days: int = 420) -> pd.DataFrame:
    if int(max_api_days) > 0:
        cap_start = ts_max - pd.Timedelta(days=int(max_api_days))
        ts_min = max(ts_min, cap_start)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cpath = CACHE_DIR / f"poly_api_feat_{ts_min.date()}_{ts_max.date()}_{int(max_api_days)}d.parquet"
    if cpath.exists():
        try:
            df = pd.read_parquet(cpath)
            if len(df):
                df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(tz).dt.floor("min")
                return df.sort_values("ts").reset_index(drop=True)
        except Exception:
            pass

    start_day = ts_min.tz_convert(tz).normalize()
    end_day = ts_max.tz_convert(tz).normalize()
    days = _scan_available_slugs(start_day, end_day)
    if not days:
        return pd.DataFrame(columns=["ts", "poly_mode_prob", "poly_weighted_target", "poly_d1_abs", "poly_d3_abs", "poly_d1_signed"])
    print(f"[poly-fetch] available slugs in range: {len(days)} days ({days[0].date()} ~ {days[-1].date()})")

    parts = []
    for i, d in enumerate(days, 1):
        slug = f"ethereum-price-on-{d.strftime('%B').lower()}-{d.day}"
        try:
            feat, _ = _build_slug_features(slug, tz=tz)
        except Exception:
            feat = pd.DataFrame()
        if len(feat) == 0:
            continue
        f = feat.sort_values("ts").reset_index(drop=True)

        mode_prob = pd.to_numeric(f.get("mode_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        wt = pd.to_numeric(f.get("weighted_target", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

        d1_abs = np.zeros(len(f), dtype=np.float64)
        d3_abs = np.zeros(len(f), dtype=np.float64)
        d1_signed = np.zeros(len(f), dtype=np.float64)
        for r_i, r in f.iterrows():
            pmap = dict(r.get("prob_map", {}) or {})
            d1 = dict(r.get("d1_map", {}) or {})
            d3 = dict(r.get("d5_map", {}) or {})
            top3 = [k for k, _ in sorted(pmap.items(), key=lambda kv: float(kv[1]), reverse=True)[:3]]
            v1 = [float(d1.get(lb, 0.0) or 0.0) for lb in top3]
            v3 = [float(d3.get(lb, 0.0) or 0.0) for lb in top3]
            if v1:
                j = int(np.argmax(np.abs(v1)))
                d1_signed[r_i] = float(v1[j])
                d1_abs[r_i] = float(abs(v1[j]))
            if v3:
                d3_abs[r_i] = float(max(abs(x) for x in v3))

        g = pd.DataFrame(
            {
                "ts": pd.to_datetime(f["ts"], utc=True).dt.tz_convert(tz).dt.floor("min"),
                "poly_mode_prob": mode_prob,
                "poly_weighted_target": wt,
                "poly_d1_abs": d1_abs,
                "poly_d3_abs": d3_abs,
                "poly_d1_signed": d1_signed,
            }
        )
        parts.append(g)
        if i % 20 == 0:
            print(f"[poly-fetch] {i}/{len(days)} days done")

    if not parts:
        return pd.DataFrame(columns=["ts", "poly_mode_prob", "poly_weighted_target", "poly_d1_abs", "poly_d3_abs", "poly_d1_signed"])

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    try:
        out.to_parquet(cpath, index=False)
    except Exception:
        pass
    return out


def _load_base(cfg: Cfg) -> pd.DataFrame:
    df = pd.read_csv(RL_CSV)
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(cfg.tz)
    else:
        df["ts"] = df["ts"].dt.tz_convert(cfg.tz)
    df = df.dropna(subset=["ts", "close"]).sort_values("ts").reset_index(drop=True)

    ts_min = df["ts"].min().floor("min")
    ts_max = df["ts"].max().ceil("min")
    poly = _fetch_polymarket_features_max(ts_min, ts_max, cfg.tz, max_api_days=cfg.max_api_days)

    if len(poly):
        b = df.copy()
        p = poly.copy()
        b["ts_key"] = pd.to_datetime(b["ts"]).astype("int64")
        p["ts_key"] = pd.to_datetime(p["ts"]).astype("int64")
        b = pd.merge_asof(
            b.sort_values("ts_key"),
            p.sort_values("ts_key").drop(columns=["ts"]),
            on="ts_key",
            direction="nearest",
            tolerance=int(pd.Timedelta(minutes=3).value),
        )
        b = b.drop(columns=["ts_key"])
        df = b

    for c in ["poly_mode_prob", "poly_weighted_target", "poly_d1_abs", "poly_d3_abs", "poly_d1_signed"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["poly_weighted_gap"] = (df["poly_weighted_target"] - df["close"]) / (df["close"].abs() + 1e-8)

    h = 6  # 30m ahead on 5m bars
    if "ret_fwd_30m" not in df.columns:
        df["ret_fwd_30m"] = df["close"].shift(-h) / df["close"] - 1.0

    if "long_score_30m" not in df.columns or "short_score_30m" not in df.columns:
        future_high = df["high"].rolling(window=h, min_periods=h).max().shift(-h + 1)
        future_low = df["low"].rolling(window=h, min_periods=h).min().shift(-h + 1)
        df["max_high_30m"] = (future_high / df["close"]) - 1.0
        df["max_low_30m"] = (future_low / df["close"]) - 1.0
        df["max_drop_30m"] = (df["close"] / future_low) - 1.0
        df["max_rally_30m"] = (future_high / df["close"]) - 1.0
        df["long_score_30m"] = (0.7 * df["max_high_30m"]) + (0.3 * df["max_low_30m"])
        df["short_score_30m"] = (0.7 * df["max_drop_30m"]) - (0.3 * df["max_rally_30m"])

    df = df.dropna(subset=["ret_fwd_30m", "long_score_30m", "short_score_30m"]).reset_index(drop=True)
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    # keep 29 dims
    # Polymarket features are transformed into bounded/structured signals
    # so the policy can learn stable directional/risk relationships.
    pm_prob = pd.to_numeric(df.get("poly_mode_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pm_gap = pd.to_numeric(df.get("poly_weighted_gap", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pm_d1_abs = pd.to_numeric(df.get("poly_d1_abs", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pm_d3_abs = pd.to_numeric(df.get("poly_d3_abs", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pm_d1_signed = pd.to_numeric(df.get("poly_d1_signed", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    # 1) directional confidence from mode probability (centered to [-1, 1])
    poly_prob_edge = np.clip((pm_prob - 0.5) * 2.0, -1.0, 1.0)
    # 2) target-vs-price gap as bounded directional pressure
    poly_gap_tanh = np.tanh(pm_gap / 0.010)
    # 3) instantaneous signed shock
    poly_shock_signed = np.tanh(pm_d1_signed / 0.030)
    # 4) persistent shock with sign retention (mix 1m + 3m)
    poly_signed_persist = np.tanh((pm_d1_signed + np.sign(pm_d1_signed) * 0.6 * pm_d3_abs) / 0.040)
    # 5) shock intensity regime (0~1)
    poly_shock_intensity = np.clip((pm_d1_abs + 0.6 * pm_d3_abs) / 0.080, 0.0, 1.0)

    df["poly_prob_edge"] = poly_prob_edge.astype(np.float32)
    df["poly_gap_tanh"] = poly_gap_tanh.astype(np.float32)
    df["poly_shock_signed"] = poly_shock_signed.astype(np.float32)
    df["poly_signed_persist"] = poly_signed_persist.astype(np.float32)
    df["poly_shock_intensity"] = poly_shock_intensity.astype(np.float32)

    cols = [
        "smart_money_flow", "oi_change_rate", "taker_acceleration", "log_return",
        "mtf_trend_1h", "mtf_trend_4h", "rogers_satchell_vol", "amihud_illiquidity_z",
        "ofti", "kel", "whale_retail_ratio", "squeeze_power",
        "net_taker_ratio", "trade_intensity", "big_trade_ratio", "volatility_z",
        "rsi", "wick_ratio", "garman_klass_vol", "btc_corr_60", "eth_btc_ratio_change",
        "funding_rate", "garch_vol_z", "jump_z",
        "poly_prob_edge", "poly_gap_tanh", "poly_shock_signed", "poly_signed_persist", "poly_shock_intensity",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return cols


def _robust_fit_transform(x_train: np.ndarray, x_all: np.ndarray):
    med = np.nanmedian(x_train, axis=0)
    q1 = np.nanpercentile(x_train, 25, axis=0)
    q3 = np.nanpercentile(x_train, 75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    z = (x_all - med) / iqr
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)
    return z.astype(np.float32), med.astype(np.float32), iqr.astype(np.float32)


def _policy_eval(agent: DSACAgent, states: np.ndarray, rets: np.ndarray, fee: float, slip: float) -> dict[str, float]:
    acts = np.array([agent.act(s, deterministic=True) for s in states], dtype=np.float32)
    disc = np.where(acts > 0.10, 1.0, np.where(acts < -0.10, -1.0, 0.0))
    pnl = disc * rets - (fee + slip) * np.abs(disc)
    eq = np.cumprod(1.0 + pnl)
    return {
        "trades": int(np.sum(disc != 0.0)),
        "pnl_sum": float(np.sum(pnl)),
        "pnl_mean": float(np.mean(pnl)) if len(pnl) else 0.0,
        "win_rate": float(np.mean(pnl > 0.0)) if len(pnl) else 0.0,
        "final_equity": float(eq[-1]) if len(eq) else 1.0,
        "avg_abs_action": float(np.mean(np.abs(acts))) if len(acts) else 0.0,
    }


def _populate_replay(agent: DSACAgent, s: np.ndarray, r30: np.ndarray, long_score: np.ndarray, short_score: np.ndarray, cfg: Cfg) -> int:
    n = len(s)
    pushed = 0
    for i in range(n - 1):
        ret = float(r30[i])
        lsc = float(long_score[i])
        ssc = float(short_score[i])
        if cfg.reward_mode == "dynamic":
            edge = lsc - ssc
            if abs(edge) < cfg.hold_band:
                a = 0.0
            else:
                a = float(np.clip(edge / max(cfg.action_scale, 1e-6), -1.0, 1.0))
            if a > 0:
                base_rew = lsc
            elif a < 0:
                base_rew = ssc
            else:
                base_rew = -0.00005
            rew = float(base_rew - (cfg.fee + cfg.slip) * abs(a))
        else:
            if abs(ret) < cfg.hold_band:
                a = 0.0
            else:
                a = float(np.clip(ret / max(cfg.action_scale, 1e-6), -1.0, 1.0))
            rew = float(a * ret - (cfg.fee + cfg.slip) * abs(a))
        d = 1.0 if i >= (n - 2) else 0.0
        agent.memory.push(s[i], a, rew, s[i + 1], d, regime="normal", progress=float(i / max(1, n - 1)))
        pushed += 1
    return pushed


def _unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    ts = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
    return p.with_name(f"{p.stem}_{ts}{p.suffix}")


def run(cfg: Cfg, out_ckpt: Path, out_report: Path) -> dict:
    ck = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
    state_dim = int(ck.get("state_dim", 29))
    if state_dim != 29:
        raise RuntimeError(f"state_dim mismatch: {state_dim}")

    df = _load_base(cfg)
    feat_cols = _feature_columns(df)
    x = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["ret_fwd_30m"].to_numpy(dtype=np.float32)
    y_long = df["long_score_30m"].to_numpy(dtype=np.float32)
    y_short = df["short_score_30m"].to_numpy(dtype=np.float32)

    n = len(df)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    n_test = n - n_train - n_val
    if n_train < 5000 or n_test < 1000:
        raise RuntimeError(f"insufficient rows: total={n} train={n_train} test={n_test}")

    x_scaled, med, iqr = _robust_fit_transform(x[:n_train], x)
    s_train, y_train = x_scaled[:n_train], y[:n_train]
    s_val, y_val = x_scaled[n_train:n_train+n_val], y[n_train:n_train+n_val]
    s_test, y_test = x_scaled[n_train+n_val:], y[n_train+n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DSACAgent(state_dim=state_dim, device=device)
    agent.actor.load_state_dict(ck["actor"], strict=True)
    agent.critic.load_state_dict(ck["critic"], strict=True)
    agent.critic_target.load_state_dict(agent.critic.state_dict())
    for g in agent.actor_optimizer.param_groups:
        g["lr"] = cfg.lr
    for g in agent.critic_optimizer.param_groups:
        g["lr"] = cfg.lr
    for g in agent.alpha_optimizer.param_groups:
        g["lr"] = cfg.lr

    before_val = _policy_eval(agent, s_val, y_val, cfg.fee, cfg.slip)
    before_test = _policy_eval(agent, s_test, y_test, cfg.fee, cfg.slip)

    pushed = _populate_replay(agent, s_train, y_train, y_long[:n_train], y_short[:n_train], cfg)
    logs=[]
    for ep in range(1, cfg.epochs+1):
        acc={"critic_loss":0.0,"actor_loss":0.0,"alpha":0.0,"count":0}
        for _ in range(cfg.updates_per_epoch):
            out = agent.update(batch_size=cfg.batch_size)
            if out:
                acc["critic_loss"] += float(out.get("critic_loss",0.0))
                acc["actor_loss"] += float(out.get("actor_loss",0.0))
                acc["alpha"] += float(out.get("alpha",0.0))
                acc["count"] += 1
        c=max(1,int(acc["count"]))
        logs.append({"epoch":ep,"critic_loss":acc["critic_loss"]/c,"actor_loss":acc["actor_loss"]/c,"alpha":acc["alpha"]/c})

    after_val = _policy_eval(agent, s_val, y_val, cfg.fee, cfg.slip)
    after_test = _policy_eval(agent, s_test, y_test, cfg.fee, cfg.slip)

    out_ckpt = _unique_path(out_ckpt)
    out_report = _unique_path(out_report)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "state_dim": state_dim,
        "meta": {
            "base_ckpt": str(BASE_CKPT),
            "trained_at": pd.Timestamp.now(tz=cfg.tz).isoformat(),
            "feature_cols": feat_cols,
            "scaler_median": med.tolist(),
            "scaler_iqr": iqr.tolist(),
            "cfg": vars(cfg),
            "polymarket_api_max": True,
        },
    }, str(out_ckpt))

    report={
        "rows_total": int(n), "rows_train": int(n_train), "rows_val": int(n_val), "rows_test": int(n_test),
        "replay_pushed": int(pushed),
        "before_val": before_val, "after_val": after_val,
        "before_test": before_test, "after_test": after_test,
        "train_log_tail": logs[-3:],
        "out_ckpt": str(out_ckpt), "out_report": str(out_report),
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    ap=argparse.ArgumentParser(description='Fine-tune DSAC with maximum polymarket API history')
    ap.add_argument('--epochs',type=int,default=4)
    ap.add_argument('--updates-per-epoch',type=int,default=260)
    ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--lr',type=float,default=8e-6)
    ap.add_argument('--max-api-days',type=int,default=420)
    ap.add_argument('--out-ckpt',default='data/ensemble/ckpt/fine_tuned_dsac_agents_polymarket_api_max_v1.pth')
    ap.add_argument('--out-report',default='data/ensemble/metrics/fine_tuned_dsac_polymarket_api_max_v1_report.json')
    args=ap.parse_args()
    cfg=Cfg(
        epochs=args.epochs,
        updates_per_epoch=args.updates_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        max_api_days=args.max_api_days,
    )
    out=run(cfg, ROOT / args.out_ckpt, ROOT / args.out_report)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
