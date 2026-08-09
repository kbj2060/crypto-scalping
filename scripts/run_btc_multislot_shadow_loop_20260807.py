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
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402

N_SLOTS = int(os.getenv("BTC_MULTISLOT_SHADOW_SLOTS", "3"))
# 2026-08-07 evening extension: margin multiplier ON TOP of the equal-budget /N split.
# Diversification across slots lowers volatility drag, moving the growth-optimal multiplier up
# from the single-slot ~1.0-1.25x. Set to 1.0 to fall back to the original equal-budget sizing.
#
# PROVENANCE CORRECTED 2026-08-08 -- read this before quoting any number here.
# The original sweep ran on the N=3 gated LEDGERS (rescaling trade returns per multiplier). That
# is invalid: margin_fraction is an INPUT to the exit head, so changing it changes the exits and
# the ledger itself. Full causal replay
# (scripts/resweep_btc_multislot_margin_multiplier_fullreplay_20260808.py) over the SAME grid
# {1.25,1.5,1.75,2.0} plus 1.0, OOS gated:
#     m1.0  +16.72/-7.27   m1.25 +21.00/-9.03   m1.5 +25.30/-10.77
#     m1.75 +33.50/-12.48  m2.0  +38.56/-14.17
# 1. The recorded adoption figure for 1.5x (+19.98%/-10.40%) was wrong; it is +25.30%/-10.77%.
#    The rescale happened to match at 1.25x, which is why the error went unnoticed.
# 2. VAL PnL is MONOTONE INCREASING in m and VAL MDD never reaches the -8% bar (worst -6.69 at
#    2.0x), so the stated VAL rule degenerates to "take the largest multiplier in the grid" and
#    correctly applied it selects 2.0x, NOT 1.5x.
# 3. 2.0x then FAILS the OOS gate (MDD -14.17 < -12.4), so a correct execution of the stated rule
#    REJECTS the multiplier extension and falls back to 1.0.
# 1.5x does pass all three OOS gates on its own numbers, but selecting it for that reason is
# OOS cherry-picking, which the rule forbids. The value below is therefore left UNCHANGED pending
# an explicit decision -- changing live sizing is not a bookkeeping fix.
# Report: docs/btc_multislot_margin_multiplier_provenance_20260808.md
MARGIN_MULT = float(os.getenv("BTC_MULTISLOT_SHADOW_MARGIN_MULT", "1.5"))
COST_MULT = 3.0

# 2026-08-08: optional czz_trend regime SIZING overlay, DEFAULT OFF so the running shadow is
# byte-identical unless the env var is set. Adopted as a risk overlay in
# data/research/btc_regime_sizing_risk_overlay_frozen_20260808.json: multiply the sidecar
# margin_fraction by a fixed multiplier from the causal 4% directional-change wave direction at
# the entry bar (bear 0.5 / chop 1.0 / bull 1.5). Applied BEFORE the /N_SLOTS split, matching the
# replay in scripts/eval_btc_multislot_shadow_with_regime_sizing_20260808.py (OOS gated
# +34.71%/-9.24% vs the no-overlay +25.30%/-10.77%). The regime is computed on the 7000-bar
# primary buffer; a convergence check on the historical panel showed the last-bar czz4 state
# matches the full-history state 100% of the time from 4000 bars onward, so the buffer is ample.
REGIME_OVERLAY = os.getenv("BTC_MULTISLOT_SHADOW_REGIME_OVERLAY", "0") == "1"
REGIME_THETA = float(os.getenv("BTC_MULTISLOT_SHADOW_REGIME_THETA", "0.04"))
REGIME_MULT = {1: 1.5, 0: 1.0, -1: 0.5}  # causal_zigzag direction -> margin multiplier
# a suffix keeps the overlay variant's state/ledger separate from the baseline shadow's
_SUFFIX = os.getenv("BTC_MULTISLOT_SHADOW_SUFFIX", "")
STATE_PATH = ROOT / f"data/ensemble/omega4_6_1_btc_multislot_shadow_state_20260807{_SUFFIX}.json"
LEDGER_PATH = ROOT / f"data/ensemble/omega4_6_1_btc_multislot_shadow_ledger_20260807{_SUFFIX}.csv"
KEEP_BARS = 7000
MODEL_BARS = 600  # in-bot shadow convention (warmup NaNs sit further back in the 7000-bar buffer)

LEDGER_COLS = ["slot", "side", "entry_timestamp", "exit_timestamp", "entry_price", "exit_price",
               "raw_exit_price_move", "mfe", "mae", "margin_fraction", "leverage", "notional",
               "take_profit", "stop_loss", "reason", "exit_prob", "trade_return_net", "hold_bars",
               "regime_dir", "regime_mult"]


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
    cols = LEDGER_COLS
    if not header:
        # an existing ledger may predate a column addition; append in ITS column order so a
        # headerless append can never shift values into the wrong columns
        existing = pd.read_csv(LEDGER_PATH, nrows=0).columns.tolist()
        if existing != LEDGER_COLS:
            cols = existing
    pd.DataFrame([row]).reindex(columns=cols).to_csv(LEDGER_PATH, mode="a", header=header, index=False)


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
                "regime_dir": s.get("regime_dir", 0), "regime_mult": s.get("regime_mult", 1.0),
            })
            log(f"EXIT slot={k} side={side} reason={reason} raw={raw_exit:+.4f} net={r_net:+.4f} hold={s['hold_bars']}")
            state["slots"][k] = None
            exited = True

    free = next((k for k in range(N_SLOTS) if state["slots"][k] is None), None)
    if not exited and free is not None:
        dec = adapter.decide_entry(processed)
        if dec is not None:
            regime_dir, regime_mult = 0, 1.0
            if REGIME_OVERLAY:
                closes = pd.to_numeric(btc["close"], errors="coerce").to_numpy(dtype=float)
                regime_dir = int(causal_zigzag(closes, threshold=REGIME_THETA)[-1])
                regime_mult = REGIME_MULT.get(regime_dir, 1.0)
            margin = float(dec.margin_fraction) * MARGIN_MULT * regime_mult / N_SLOTS
            notional = margin * float(dec.leverage)
            state["slots"][free] = {
                "side": int(dec.side), "entry_price": price, "entry_timestamp": last_bar,
                "margin_fraction": margin, "leverage": float(dec.leverage), "notional": notional,
                "take_profit": float(dec.take_profit), "stop_loss": float(dec.stop_loss),
                "mfe": 0.0, "mae": 0.0, "hold_bars": 0,
                "regime_dir": regime_dir, "regime_mult": regime_mult,
            }
            log(f"ENTRY slot={free} side={dec.side} price={price:.1f} notional={notional:.3f} "
                f"lev={dec.leverage:.2f} tp={dec.take_profit:.4f} sl={dec.stop_loss:.4f} "
                f"q={dec.quality_score:.3f} regime={regime_dir} mult={regime_mult:.2f}")

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
    log(f"btc_multislot_shadow START n_slots={N_SLOTS} margin_mult={MARGIN_MULT} "
        f"regime_overlay={'ON theta=' + str(REGIME_THETA) if REGIME_OVERLAY else 'OFF'} "
        f"state={STATE_PATH.name} bundle={cfg['bundle_path']}")

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
