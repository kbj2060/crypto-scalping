"""Standalone BTC multi-slot (N=3) shadow loop -- live-forward validation of the multislot gate.

Backtest gate passed 2026-08-07 (research_btc_swingtransition_multislot_20260807.py, pre-registered
rules): N=3 equal-budget slots, OOS gated +16.72%/MDD -7.27%/124tr vs single-slot +10.75%/-12.41%/41tr,
worst quarter -0.61%. Per the repo's shadow-first discipline (and the omega5 live-only loop precedent)
this runs as its OWN process, leaving trading_bot.py's promoted single-slot BTC shadow untouched --
the two accumulate live-forward records side by side for an A/B promotion decision later.

Semantics mirror the in-bot BTC shadow path: same adapter (promoted swingtransition bundle/sidecar,
live duration gate inside decide_entry), same swing_transition_prob live feature, same intrabar
wick TP/SL via evaluate_exit, entries priced at the decision bar's close (in-bot shadow convention).
Multi-slot extension matches the gated replay: sidecar margin_fraction is split /N_SLOTS (leverage
unchanged), a signal fills the first FREE slot, no new entry on a cycle where any slot exited.
Net PnL per trade applies the backtest cost model (fee+slip x cost_mult 3) for A/B comparability.

Live-execution caveat (recorded at gate time): concurrent opposite-side slots net out in one-way
futures mode; REAL execution of this policy needs hedge mode or a same-side-only slot rule. Shadow
accounting here holds them independently on purpose.

State: data/ensemble/omega4_6_1_btc_multislot_shadow_state_20260807.json (atomic writes)
Ledger: data/ensemble/omega4_6_1_btc_multislot_shadow_ledger_20260807.csv (append per close)
Run:   nohup python scripts/run_btc_multislot_shadow_loop_20260807.py >> logs/btc_multislot_shadow.log 2>&1 &
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_bot_modules.runtime_config import OMEGA4_6_1_SHADOW_ASSET_CONFIG  # noqa: E402
from trading_bot_modules.omega4_6_1_live import (  # noqa: E402
    Omega461LiveAdapter,
    BTC_BASE_TEMPLATE,
    BTC_EXPERT_SCALES,
)
from trading_bot_modules.btc_swing_transition_live import BtcSwingTransitionLiveFeature  # noqa: E402
from trading_bot_modules.binance_live_fetcher import BinanceLiveFetcher  # noqa: E402
from features.engineering import FeatureEngineer  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_btc_swingtransition_20260806 as _omega_cost  # noqa: E402

N_SLOTS = int(os.getenv("BTC_MULTISLOT_SHADOW_SLOTS", "3"))
# 2026-08-07 evening extension: margin multiplier ON TOP of the equal-budget /N split.
# Pre-registered sweep on the N=3 gated ledgers (VAL selects m s.t. VAL MDD >= -8%; OOS adopt iff
# PnL >= +19.7 AND MDD >= -12.4 AND worst quarter >= -4): VAL selected 1.5x, OOS +19.98%/-10.40%/
# worstQ -0.92% -> adopted. Diversification across slots lowers volatility drag, moving the
# growth-optimal multiplier up from the single-slot ~1.0-1.25x. Set to 1.0 to fall back to the
# original equal-budget sizing.
MARGIN_MULT = float(os.getenv("BTC_MULTISLOT_SHADOW_MARGIN_MULT", "1.5"))
COST_MULT = 3.0
STATE_PATH = ROOT / "data/ensemble/omega4_6_1_btc_multislot_shadow_state_20260807.json"
LEDGER_PATH = ROOT / "data/ensemble/omega4_6_1_btc_multislot_shadow_ledger_20260807.csv"
KEEP_BARS = 7000
MODEL_BARS = 600  # in-bot shadow convention (warmup NaNs sit further back in the 7000-bar buffer)

LEDGER_COLS = ["slot", "side", "entry_timestamp", "exit_timestamp", "entry_price", "exit_price",
               "raw_exit_price_move", "mfe", "mae", "margin_fraction", "leverage", "notional",
               "take_profit", "stop_loss", "reason", "exit_prob", "trade_return_net", "hold_bars"]


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {msg}", flush=True)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"slots": [None] * N_SLOTS, "last_bar": None}


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def append_ledger(row: dict) -> None:
    header = not LEDGER_PATH.exists()
    pd.DataFrame([row])[LEDGER_COLS].to_csv(LEDGER_PATH, mode="a", header=header, index=False)


async def cycle(fetcher, fe, swing, adapter, fee_eff, slip_eff, state, buffers) -> None:
    if buffers.get("eth") is None:
        eth, btc = await fetcher.fetch_initial_data()
    else:
        new_eth, new_btc = await fetcher.fetch_latest_patch()
        eth = pd.concat([buffers["eth"], new_eth]).drop_duplicates("timestamp").tail(KEEP_BARS)
        btc = pd.concat([buffers["btc"], new_btc]).drop_duplicates("timestamp").tail(KEEP_BARS)
    buffers["eth"], buffers["btc"] = eth, btc

    last_bar = str(eth["timestamp"].iloc[-1])
    if state.get("last_bar") == last_bar:
        return  # no new closed bar yet
    processed = fe.process(eth, btc)
    if processed is None or len(processed) == 0:
        raise RuntimeError("empty processed frame")
    processed = processed.tail(MODEL_BARS).copy()
    # layerA consumes the adapter's regime3_current_* columns -- append them BEFORE the swing
    # feature (the adapter recomputes/overwrites them identically inside decide_entry/evaluate_exit,
    # so this pre-append only fixes ordering, it does not change decision inputs).
    processed = adapter.regime3_current.append(processed)
    processed = swing.append(processed, raw_5m=eth)

    price = float(pd.to_numeric(processed["close"], errors="coerce").iloc[-1])
    bar_high = float(processed["high"].iloc[-1])
    bar_low = float(processed["low"].iloc[-1])

    exited = False
    for k in range(N_SLOTS):
        s = state["slots"][k]
        if s is None:
            continue
        entry = float(s["entry_price"])
        side = int(s["side"])
        move = (price - entry) / entry if side > 0 else (entry - price) / entry
        s["mfe"] = max(float(s.get("mfe", 0.0)), move)
        s["mae"] = min(float(s.get("mae", 0.0)), move)
        s["hold_bars"] = int(s.get("hold_bars", 0)) + 1
        bar_high_move = (bar_high - entry) / entry if side > 0 else (entry - bar_low) / entry
        bar_low_move = (bar_low - entry) / entry if side > 0 else (entry - bar_high) / entry
        should_exit, reason, exit_prob = adapter.evaluate_exit(
            processed, source_component="h48qual", side=side, hold_bars=int(s["hold_bars"]),
            unrealized_move=float(move), mfe=float(s["mfe"]), mae=float(s["mae"]),
            notional=float(s["notional"]), leverage=float(s["leverage"]),
            take_profit=float(s["take_profit"]), stop_loss=float(s["stop_loss"]),
            bar_high_move=bar_high_move, bar_low_move=bar_low_move,
        )
        if should_exit:
            if reason == "stop_loss":
                raw_exit = -abs(float(s["stop_loss"]))
            elif reason == "take_profit":
                raw_exit = float(s["take_profit"])
            else:
                raw_exit = move
            n = float(s["notional"])
            r_net = (1.0 - fee_eff * n) * (1.0 + (raw_exit - slip_eff) * n) * (1.0 - fee_eff * n) - 1.0
            append_ledger({
                "slot": k, "side": side, "entry_timestamp": s["entry_timestamp"], "exit_timestamp": last_bar,
                "entry_price": entry, "exit_price": price, "raw_exit_price_move": raw_exit,
                "mfe": s["mfe"], "mae": s["mae"], "margin_fraction": s["margin_fraction"],
                "leverage": s["leverage"], "notional": n, "take_profit": s["take_profit"],
                "stop_loss": s["stop_loss"], "reason": reason, "exit_prob": exit_prob,
                "trade_return_net": r_net, "hold_bars": s["hold_bars"],
            })
            log(f"EXIT slot={k} side={side} reason={reason} raw={raw_exit:+.4f} net={r_net:+.4f} hold={s['hold_bars']}")
            state["slots"][k] = None
            exited = True

    free = next((k for k in range(N_SLOTS) if state["slots"][k] is None), None)
    if not exited and free is not None:
        dec = adapter.decide_entry(processed)
        if dec is not None:
            margin = float(dec.margin_fraction) * MARGIN_MULT / N_SLOTS
            notional = margin * float(dec.leverage)
            state["slots"][free] = {
                "side": int(dec.side), "entry_price": price, "entry_timestamp": last_bar,
                "margin_fraction": margin, "leverage": float(dec.leverage), "notional": notional,
                "take_profit": float(dec.take_profit), "stop_loss": float(dec.stop_loss),
                "mfe": 0.0, "mae": 0.0, "hold_bars": 0,
            }
            log(f"ENTRY slot={free} side={dec.side} price={price:.1f} notional={notional:.3f} "
                f"lev={dec.leverage:.2f} tp={dec.take_profit:.4f} sl={dec.stop_loss:.4f} q={dec.quality_score:.3f}")

    state["last_bar"] = last_bar
    save_state(state)


async def main() -> int:
    cfg = OMEGA4_6_1_SHADOW_ASSET_CONFIG["btc"]
    fetcher = BinanceLiveFetcher(symbol=str(cfg["symbol"]), timeframe="5m", limit=KEEP_BARS,
                                 account_symbol=str(cfg["account_symbol"]))
    fe = FeatureEngineer()
    swing = BtcSwingTransitionLiveFeature()
    adapter = Omega461LiveAdapter(
        h48qual_bundle=cfg["bundle_path"], h48qual_sidecar=cfg["sidecar_path"],
        zig075_bundle="x", zig075_sidecar="x", device="cpu",
        current_regime_path=str(cfg["current_regime_path"]),
        base_template=BTC_BASE_TEMPLATE, expert_scales=BTC_EXPERT_SCALES,
        components_override={"h48qual": {"bundle": cfg["bundle_path"], "sidecar": cfg["sidecar_path"],
                                          "quality_threshold": float(cfg["quality_threshold"])}},
        priority=("h48qual",), duration_threshold=float(cfg["duration_threshold"]),
        scale_map=dict(cfg["scale_map"]),
    )
    fee, slip = _omega_cost._load_fee_slip()
    fee_eff, slip_eff = float(fee) * COST_MULT, float(slip) * COST_MULT
    state = load_state()
    if len(state.get("slots", [])) != N_SLOTS:
        raise RuntimeError(f"state slot count {len(state.get('slots', []))} != N_SLOTS {N_SLOTS}")
    buffers: dict = {"eth": None, "btc": None}
    log(f"btc_multislot_shadow START n_slots={N_SLOTS} bundle={cfg['bundle_path']}")

    while True:
        try:
            await cycle(fetcher, fe, swing, adapter, fee_eff, slip_eff, state, buffers)
        except Exception as exc:  # noqa: BLE001 - log and retry next bar, mirroring the in-bot loop
            log(f"CYCLE_ERROR {type(exc).__name__}: {exc}")
        now = time.time()
        await asyncio.sleep(max(300 - (now % 300), 5) + 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
