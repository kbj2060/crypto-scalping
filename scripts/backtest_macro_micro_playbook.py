#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Stats:
    start: str
    end: str
    rows: int
    trades: int
    win_rate_pct: float
    pnl_pct: float
    mdd_pct: float
    sharpe: float
    final_equity: float


def calc_mdd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def calc_sharpe(equity: np.ndarray, bars_per_year: int) -> float:
    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    if len(rets) < 3:
        return 0.0
    s = rets.std()
    if s < 1e-12:
        return 0.0
    return float(rets.mean() / s * math.sqrt(bars_per_year))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)

    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default)
        return pd.Series(default, index=out.index, dtype="float64")

    # ----- Macro proxy (contrarian overheating index) -----
    # exchange_netflow 대체가 아니라, OI+Funding를 과열도(역추세) 필터로 사용
    oi = col("sum_open_interest_value", 0.0).ffill().fillna(0.0)
    funding = col("last_funding_rate", 0.0)
    smf = col("smart_money_flow", 0.0)

    oi_ret = oi.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    roll = 120  # 10h on 5m bars
    oi_mu = oi.rolling(roll, min_periods=24).mean()
    oi_sd = oi.rolling(roll, min_periods=24).std().replace(0, np.nan)
    oi_z = ((oi - oi_mu) / oi_sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    f_mu = funding.rolling(roll, min_periods=24).mean()
    f_sd = funding.rolling(roll, min_periods=24).std().replace(0, np.nan)
    f_z = ((funding - f_mu) / f_sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["oi_z"] = oi_z
    out["funding_z"] = f_z
    out["overheating_score"] = oi_z + f_z

    # ----- Micro playbook proxy scores -----
    # OBI proxy: net taker imbalance + liquidity slope
    taker = col("net_taker_ratio", 0.0)
    illiq = col("amihud_illiquidity_z", 0.0)
    whale = col("sig_whale", 0.0)
    oi_chg = col("oi_change_rate", 0.0)
    mean_rev = col("mean_reversion_z", 0.0)
    chop = col("chop_index", 50.0)
    volz = col("volatility_z", 0.0)
    breakout = col("breakout_strength", 0.0)

    # Long/Short micro scores (0..1)
    trend_long = np.clip(0.5 + 0.5 * (0.6 * taker + 0.8 * whale + 0.3 * oi_chg), 0.0, 1.0)
    trend_short = np.clip(0.5 + 0.5 * (-0.6 * taker - 0.8 * whale - 0.3 * oi_chg), 0.0, 1.0)

    revert_long = np.clip(0.5 + 0.5 * (0.7 * mean_rev - 0.3 * breakout), 0.0, 1.0)
    revert_short = np.clip(0.5 + 0.5 * (-0.7 * mean_rev + 0.3 * breakout), 0.0, 1.0)

    # weighted micro ensemble (with maker/taker interpretation bias via smf)
    out["micro_long"] = np.clip(0.65 * trend_long + 0.35 * revert_long, 0.0, 1.0)
    out["micro_short"] = np.clip(0.65 * trend_short + 0.35 * revert_short, 0.0, 1.0)

    # tail risk penalty Psi in [0,1] from illiquidity/vol/chop
    tail_raw = 0.45 * np.clip(illiq / 3.0, 0.0, 1.0) + 0.35 * np.clip(np.abs(volz) / 3.0, 0.0, 1.0) + 0.20 * np.clip((chop - 50.0) / 50.0, 0.0, 1.0)
    out["tail_penalty"] = np.clip(1.0 - tail_raw, 0.0, 1.0)

    # Smart-money strict interpretation:
    # 강한 taker sell인데 가격이 안 빠지면(반등성) long 강화, 반대는 short 강화
    ret1 = pd.to_numeric(out.get("log_return", 0.0), errors="coerce").fillna(0.0)
    absorb_long = np.clip(0.5 + 0.5 * (-taker - 4.0 * ret1), 0.0, 1.0)
    absorb_short = np.clip(0.5 + 0.5 * (taker + 4.0 * ret1), 0.0, 1.0)
    out["micro_long"] = np.clip(0.80 * out["micro_long"] + 0.20 * absorb_long, 0.0, 1.0)
    out["micro_short"] = np.clip(0.80 * out["micro_short"] + 0.20 * absorb_short, 0.0, 1.0)

    # final formula (contrarian overheating filter)
    # long 허용: 과열 점수 낮을 때만, short 증폭: 과열 높을 때
    long_gate = np.where(out["overheating_score"] < 0.5, 1.0, 0.0)
    short_boost = np.where(out["overheating_score"] > 1.5, 1.5, 1.0)
    out["final_long"] = out["micro_long"] * out["tail_penalty"] * long_gate
    out["final_short"] = out["micro_short"] * out["tail_penalty"] * short_boost

    # liquidity impact model inputs
    out["quote_volume"] = col("quote_volume", 0.0)
    out["high"] = col("high", out["close"])
    out["low"] = col("low", out["close"])
    out["volume"] = col("volume", 0.0)
    out["funding"] = funding
    out["session_us"] = col("session_us", 0.0)
    out["session_europe"] = col("session_europe", 0.0)

    # volatility regime filters
    prev_close = out["close"].shift(1).fillna(out["close"])
    tr1 = (out["high"] - out["low"]).abs()
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = tr.rolling(14, min_periods=5).mean().fillna(0.0)
    out["atr14_pct"] = (atr14 / np.maximum(out["close"], 1e-8)).fillna(0.0)
    vol1h = out["volume"].rolling(12, min_periods=6).mean().fillna(0.0)
    vol24h = out["volume"].rolling(288, min_periods=24).mean().ffill().fillna(1.0)
    out["vol1h_ratio"] = (vol1h / np.maximum(vol24h, 1e-8)).fillna(0.0)

    # VPIN-lite (volume-synchronized informed flow proxy)
    imb_abs = (out["quote_volume"] * taker.abs()).fillna(0.0)
    vol_sum = out["quote_volume"].rolling(60, min_periods=12).sum().fillna(0.0)
    imb_sum = imb_abs.rolling(60, min_periods=12).sum().fillna(0.0)
    out["vpin_lite"] = (imb_sum / np.maximum(vol_sum, 1e-8)).clip(0.0, 1.0)

    # toxicity / queue-collapse proxy (when raw micro fields are unavailable)
    out["toxicity_proxy"] = np.clip(0.65 * np.clip(np.abs(volz) / 3.0, 0.0, 1.0) + 0.35 * np.clip(illiq / 3.0, 0.0, 1.0), 0.0, 1.0)
    # collapse proxy: volume vacuum + sudden range expansion
    low_liq = np.clip(1.0 - out["vol1h_ratio"], 0.0, 1.0)
    range_burst = np.clip(out["atr14_pct"] / 0.004, 0.0, 1.0)
    out["queue_collapse_proxy"] = np.clip(0.6 * low_liq + 0.4 * range_burst, 0.0, 1.0)

    # VPIN-aware micro adjustment
    # informed flow가 높고 과열(score>0)이면 short 편향 강화, 과열 낮고 VPIN 중간이면 long 보수 허용
    vpin = out["vpin_lite"]
    out["final_long"] = out["final_long"] * np.where(vpin < 0.70, 1.0, 0.85)
    out["final_short"] = out["final_short"] * np.where(vpin > 0.75, 1.10, 1.0)
    return out


def run_backtest(
    df: pd.DataFrame,
    entry_th: float,
    exit_th: float,
    maker_fee: float,
    taker_fee: float,
    base_slip: float,
    impact_k: float,
    funding_scale: float,
    tp_pct: float,
    sl_pct: float,
    cooldown_bars: int,
    trailing_stop_pct: float,
    enable_trailing: bool,
    entry_maker: bool,
    atr14_min_pct: float,
    vol1h_min_ratio: float,
    trail_tox_alpha: float,
    trail_collapse_beta: float,
    trail_max_mult: float,
    maker_chase_enable: bool,
    taker_fallback_on_squeeze: bool,
) -> tuple[Stats, pd.DataFrame]:
    close = df["close"].to_numpy(dtype=np.float64)
    qv = np.maximum(df["quote_volume"].to_numpy(dtype=np.float64), 1.0)
    f_long = df["final_long"].to_numpy(dtype=np.float64)
    f_short = df["final_short"].to_numpy(dtype=np.float64)
    funding = df["funding"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    atr14_pct = df["atr14_pct"].to_numpy(dtype=np.float64)
    vol1h_ratio = df["vol1h_ratio"].to_numpy(dtype=np.float64)
    session_us = df["session_us"].to_numpy(dtype=np.float64)
    session_eu = df["session_europe"].to_numpy(dtype=np.float64)
    vpin = df["vpin_lite"].to_numpy(dtype=np.float64)
    tox = df["toxicity_proxy"].to_numpy(dtype=np.float64)
    collapse = df["queue_collapse_proxy"].to_numpy(dtype=np.float64)
    lr = pd.to_numeric(df.get("log_return", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    eq = 1.0
    peak = 1.0
    equity_curve = [eq]

    pos = 0  # 1 long, -1 short, 0 flat
    size = 0.0
    entry_px = 0.0
    peak_px = 0.0
    trough_px = 0.0
    wins = 0
    trades = 0

    hys_long = False
    hys_short = False
    cooldown = 0

    rec = []

    for i in range(1, len(df)):
        # hysteresis states
        if cooldown > 0:
            cooldown -= 1

        # session-aware thresholding
        # US: 진입 민감도 낮춤, EU: 소폭 완화, Asia: 보수적
        sess_mult = 1.0
        if session_us[i] > 0.5:
            sess_mult = 0.92
        elif session_eu[i] > 0.5:
            sess_mult = 0.97
        else:
            sess_mult = 1.05
        entry_eff = entry_th * sess_mult
        exit_eff = exit_th * (0.98 if session_us[i] > 0.5 else 1.0)

        if not hys_long and f_long[i] >= entry_eff:
            hys_long = True
        elif hys_long and f_long[i] <= exit_eff:
            hys_long = False

        if not hys_short and f_short[i] >= entry_eff:
            hys_short = True
        elif hys_short and f_short[i] <= exit_eff:
            hys_short = False

        # prefer stronger side
        signal = 0
        if hys_long and (not hys_short or f_long[i] >= f_short[i]):
            signal = 1
        elif hys_short and (not hys_long or f_short[i] > f_long[i]):
            signal = -1

        # dynamic position sizing from signal strength
        strength = f_long[i] if signal == 1 else (f_short[i] if signal == -1 else 0.0)
        target_size = float(np.clip((strength - exit_eff) / max(entry_eff - exit_eff, 1e-6), 0.0, 1.0))
        tradable = bool((atr14_pct[i] >= atr14_min_pct) and (vol1h_ratio[i] >= vol1h_min_ratio))
        if not tradable:
            signal = 0
            target_size = 0.0

        # square-root impact slippage
        notional_ratio = np.clip(target_size * eq / qv[i], 0.0, 0.05)
        slip = base_slip + impact_k * math.sqrt(notional_ratio)

        # position transitions
        if pos == 0 and signal != 0 and target_size > 0 and cooldown == 0:
            # Maker entry + chase + conditional taker fallback
            use_taker_entry = (not entry_maker)
            can_enter = True
            if entry_maker and maker_chase_enable:
                # 급격한 추세 캔들에서는 maker 미체결 확률이 높다고 가정
                maker_fill = abs(lr[i]) <= (0.0015 + 0.6 * atr14_pct[i])
                if not maker_fill and taker_fallback_on_squeeze:
                    squeeze_cond = (abs(f_long[i] - f_short[i]) > 0.25) or (vpin[i] > 0.80) or (abs(lr[i]) > 0.004)
                    use_taker_entry = bool(squeeze_cond)
                elif not maker_fill:
                    can_enter = False
                    use_taker_entry = False
            if can_enter:
                pos = signal
                size = target_size
                entry_px = close[i] * (1 + slip if pos == 1 else 1 - slip)
                fee_in = taker_fee if use_taker_entry else maker_fee
                eq *= (1.0 - fee_in * size)
                trades += 1
                peak_px = entry_px
                trough_px = entry_px

        elif pos != 0:
            # funding carry (very rough proxy on 5m bars)
            carry = -funding[i] * funding_scale * size if pos == 1 else funding[i] * funding_scale * size
            eq *= (1.0 + carry)

            # trailing stop update
            if pos == 1:
                peak_px = max(peak_px, high[i])
            else:
                trough_px = min(trough_px, low[i])

            # TP/SL first
            marked_px = close[i]
            ret_mark = (marked_px - entry_px) / max(entry_px, 1e-12)
            if pos == -1:
                ret_mark = -ret_mark
            hit_tp = ret_mark >= tp_pct
            hit_sl = ret_mark <= -abs(sl_pct)
            hit_trailing = False
            if enable_trailing and trailing_stop_pct > 0:
                dyn_mult = min(1.0 + trail_tox_alpha * tox[i], trail_max_mult) * min(1.0 + trail_collapse_beta * collapse[i], trail_max_mult)
                dyn_gap = trailing_stop_pct * dyn_mult
                if pos == 1:
                    trail_stop_px = peak_px * (1.0 - dyn_gap)
                    hit_trailing = close[i] <= trail_stop_px
                else:
                    trail_stop_px = trough_px * (1.0 + dyn_gap)
                    hit_trailing = close[i] >= trail_stop_px

            # exit/flip/reduce
            should_exit = hit_tp or hit_sl or hit_trailing or (signal == 0) or (signal == -pos)
            if should_exit:
                exit_px = close[i] * (1 - slip if pos == 1 else 1 + slip)
                ret = (exit_px - entry_px) / max(entry_px, 1e-12)
                if pos == -1:
                    ret = -ret
                pnl = ret * size
                eq *= (1.0 + pnl)
                eq *= (1.0 - taker_fee * size)
                wins += int(pnl > 0)
                pos = 0
                size = 0.0
                entry_px = 0.0
                peak_px = 0.0
                trough_px = 0.0
                cooldown = max(cooldown, int(cooldown_bars))
            else:
                # same direction: rebalance with maker assumption
                delta = target_size - size
                if abs(delta) > 1e-3:
                    fee = maker_fee * abs(delta)
                    eq *= (1.0 - fee)
                    size = target_size

        peak = max(peak, eq)
        equity_curve.append(eq)
        rec.append({
            "timestamp": df["timestamp"].iloc[i],
            "close": close[i],
            "final_long": f_long[i],
            "final_short": f_short[i],
            "signal": signal,
            "size": target_size,
            "tradable": int(tradable),
            "entry_eff": entry_eff,
            "exit_eff": exit_eff,
            "vpin": vpin[i],
            "tox_proxy": tox[i],
            "collapse_proxy": collapse[i],
            "equity": eq,
            "drawdown": (eq / peak - 1.0),
        })

    ec = np.array(equity_curve, dtype=np.float64)
    stats = Stats(
        start=str(df["timestamp"].iloc[0]),
        end=str(df["timestamp"].iloc[-1]),
        rows=int(len(df)),
        trades=int(trades),
        win_rate_pct=float((wins / trades) * 100.0) if trades > 0 else 0.0,
        pnl_pct=float((ec[-1] - 1.0) * 100.0),
        mdd_pct=float(calc_mdd(ec) * 100.0),
        sharpe=float(calc_sharpe(ec, bars_per_year=365 * 24 * 12)),
        final_equity=float(ec[-1]),
    )
    return stats, pd.DataFrame(rec)


def main() -> None:
    ap = argparse.ArgumentParser(description="Macro(onchain-proxy) x Micro(playbook-proxy) backtest")
    ap.add_argument("--input", default="data/training_features_5m.csv")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--entry", type=float, default=0.75)
    ap.add_argument("--exit", type=float, default=0.40)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--impact-k", type=float, default=0.015)
    ap.add_argument("--funding-scale", type=float, default=1.0)
    ap.add_argument("--tp-pct", type=float, default=0.010, help="take-profit pct, e.g. 0.01 = 1%")
    ap.add_argument("--sl-pct", type=float, default=0.006, help="stop-loss pct, e.g. 0.006 = 0.6%")
    ap.add_argument("--cooldown-bars", type=int, default=36, help="bars to wait after close (5m bars)")
    ap.add_argument("--trailing-stop-pct", type=float, default=0.005, help="trailing stop pct, e.g. 0.005 = 0.5%")
    ap.add_argument("--disable-trailing", action="store_true", help="disable trailing stop")
    ap.add_argument("--entry-taker", action="store_true", help="use taker fee for entries (default: maker)")
    ap.add_argument("--atr14-min-pct", type=float, default=0.0012, help="min ATR14/price to allow entry")
    ap.add_argument("--vol1h-min-ratio", type=float, default=0.60, help="min 1h volume / 24h mean ratio to allow entry")
    ap.add_argument("--trail-tox-alpha", type=float, default=0.8, help="dynamic trailing multiplier alpha for toxicity")
    ap.add_argument("--trail-collapse-beta", type=float, default=0.8, help="dynamic trailing multiplier beta for queue collapse")
    ap.add_argument("--trail-max-mult", type=float, default=2.5, help="cap of dynamic trailing multiplier")
    ap.add_argument("--disable-maker-chase", action="store_true", help="disable maker cancel/replace simulation")
    ap.add_argument("--disable-taker-fallback", action="store_true", help="disable taker fallback on squeeze if maker missed")
    ap.add_argument("--out-json", default="data/ensemble/metrics/macro_micro_onchain_proxy_backtest.json")
    ap.add_argument("--out-csv", default="data/ensemble/metrics/macro_micro_onchain_proxy_equity.csv")
    args = ap.parse_args()

    raw = pd.read_csv(args.input)
    feat = build_features(raw)

    if args.days > 0:
        end = feat["timestamp"].max()
        start = end - pd.Timedelta(days=args.days)
        feat = feat[feat["timestamp"] >= start].copy()

    feat = feat.reset_index(drop=True)
    if len(feat) < 200:
        raise SystemExit(f"not enough rows after filter: {len(feat)}")

    stats, curve = run_backtest(
        feat,
        entry_th=args.entry,
        exit_th=args.exit,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        base_slip=args.slip,
        impact_k=args.impact_k,
        funding_scale=args.funding_scale,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        cooldown_bars=args.cooldown_bars,
        trailing_stop_pct=args.trailing_stop_pct,
        enable_trailing=not args.disable_trailing,
        entry_maker=not args.entry_taker,
        atr14_min_pct=args.atr14_min_pct,
        vol1h_min_ratio=args.vol1h_min_ratio,
        trail_tox_alpha=args.trail_tox_alpha,
        trail_collapse_beta=args.trail_collapse_beta,
        trail_max_mult=args.trail_max_mult,
        maker_chase_enable=not args.disable_maker_chase,
        taker_fallback_on_squeeze=not args.disable_taker_fallback,
    )

    payload = {
        "config": {
            "input": args.input,
            "days": args.days,
            "entry": args.entry,
            "exit": args.exit,
            "maker_fee": args.maker_fee,
            "taker_fee": args.taker_fee,
            "slip": args.slip,
            "impact_k": args.impact_k,
            "funding_scale": args.funding_scale,
            "tp_pct": args.tp_pct,
            "sl_pct": args.sl_pct,
            "cooldown_bars": args.cooldown_bars,
            "trailing_stop_pct": args.trailing_stop_pct,
            "enable_trailing": bool(not args.disable_trailing),
            "entry_maker": bool(not args.entry_taker),
            "atr14_min_pct": args.atr14_min_pct,
            "vol1h_min_ratio": args.vol1h_min_ratio,
            "trail_tox_alpha": args.trail_tox_alpha,
            "trail_collapse_beta": args.trail_collapse_beta,
            "trail_max_mult": args.trail_max_mult,
            "maker_chase_enable": bool(not args.disable_maker_chase),
            "taker_fallback_on_squeeze": bool(not args.disable_taker_fallback),
            "onchain_note": "true exchange_netflow not found; used OI/funding/SMF proxy",
        },
        "stats": stats.__dict__,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    curve.to_csv(args.out_csv, index=False)

    print("=== Macro x Micro Backtest ===")
    print(f"range   : {stats.start} ~ {stats.end}")
    print(f"rows    : {stats.rows}")
    print(f"trades  : {stats.trades}")
    print(f"wr      : {stats.win_rate_pct:.2f}%")
    print(f"pnl     : {stats.pnl_pct:+.2f}%")
    print(f"mdd     : {stats.mdd_pct:.2f}%")
    print(f"sharpe  : {stats.sharpe:.3f}")
    print(f"equity  : {stats.final_equity:.4f}")
    print(f"saved   : {args.out_json}")
    print(f"saved   : {args.out_csv}")


if __name__ == "__main__":
    main()
