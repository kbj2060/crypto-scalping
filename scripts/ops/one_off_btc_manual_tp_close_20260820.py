#!/usr/bin/env python3
"""
Manual backdated close for BTC omega4_6_1_shadow trade-20260807T112515.312955Z.

Context: this LONG position's take-profit (69997.0125, entry 65113.5 * 1.075) was
genuinely touched by the real Binance 5m candle at 2026-08-19 15:25:00 UTC
(O 67177.10 / H 70450.00 / L 67060.20 / C 68298.90), but a since-fixed bug (the
live exit check only ever inspected the single latest bar; see memory
btc_omega461_shadow_tp_barrier_single_bar_miss_20260820.md, code fix in commit
8abafb7) meant no cycle ever evaluated that bar's high against the barrier, so
the position never closed. No real capital was involved (account.enabled=false,
exchange_execution_enabled=false throughout) -- this is a paper/shadow ledger only.

Run on request from the user, who explicitly chose to backdate the ledger to the
TP-touch point rather than close at current market price. This script:
  1. Appends a CLOSE row to data/live/trade_journal.jsonl, built with the exact
     same fee/slippage formula as GovernorPositionRouter._trade_math (verified
     byte-exact against this trade's own real OPEN record: 65113.5*(1.0002) =
     65126.5227 matches the live entry_exec_price precisely).
  2. Appends a matching event to data/live/dashboard_events.jsonl.
  3. Updates data/ensemble/omega4_6_1_shadow_btc_state.json to a flat position,
     mirroring exactly what GovernorPositionRouter._update_pos(action=0, ...)
     does on a real close (pos/entry_price/hold_count reset, peak_equity=
     cur_equity=1.0, strategy_state's omega4_6_1_active key removed, trade
     appended to trade_history, recent_realized updated) -- so the next live
     cycle picks this position up as already closed instead of re-closing it.

Every injected record carries manual_correction=true plus a reason, so it stays
distinguishable from an organically-generated row.

Safety: refuses to run unless the current live state still matches what this
script expects (pos=LONG, same trade_id, same entry_price) and no CLOSE for
this trade_id already exists in trade_journal.jsonl -- so it cannot be silently
double-applied or applied against a state that has since changed.
"""
import json
import os
import sys
import tempfile

REPO_ROOT = "/home/llewyn/crypto-scalping"
STATE_PATH = os.path.join(REPO_ROOT, "data/ensemble/omega4_6_1_shadow_btc_state.json")
JOURNAL_PATH = os.path.join(REPO_ROOT, "data/live/trade_journal.jsonl")
EVENTS_PATH = os.path.join(REPO_ROOT, "data/live/dashboard_events.jsonl")

TRADE_ID = "trade-20260807T112515.312955Z"
ENTRY_PRICE = 65113.5
EXIT_PRICE = 68298.90  # close of the 2026-08-19 15:25 UTC (00:25 KST) wick bar
NOTIONAL = 0.26
TRADE_SLIP = 0.0002
FEE = 0.0005

TS_KST = "2026-08-20 00:25:00"
BAR_UTC = "2026-08-19 15:25:00"


def _atomic_write_json(path, payload):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _atomic_append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_close_row(manual_correction_applied_at):
    entry_exec = ENTRY_PRICE * (1.0 + TRADE_SLIP)
    exit_exec = EXIT_PRICE * (1.0 - TRADE_SLIP)
    gross = (exit_exec - entry_exec) / entry_exec
    fee_cost_frac = (FEE + FEE) * NOTIONAL
    pnl_frac = gross * NOTIONAL - fee_cost_frac
    pnl_pct = pnl_frac * 100.0

    row = {
        "schema_version": "trade_journal.v1",
        "ts": TS_KST,
        "kind": "CLOSE",
        "event": "EXIT LONG",
        "side": "LONG",
        "trade_id": TRADE_ID,
        "decision_at": "2026-08-07 20:25:00",
        "opened_at": "2026-08-07T20:25:15.313011+09:00",
        "closed_at": TS_KST,
        "actual_opened_at": "2026-08-07T20:25:15.313011+09:00",
        "actual_closed_at": manual_correction_applied_at,
        "event_recorded_at": manual_correction_applied_at,
        "next_side": None,
        "entry_price": ENTRY_PRICE,
        "entry_price_source": "shadow_bar_close",
        "entry_decision_price": ENTRY_PRICE,
        "entry_exec_price": entry_exec,
        "entry_exec_price_kind": "synthetic_fee_slippage_model",
        "synthetic_entry_exec_price": entry_exec,
        "exit_price": EXIT_PRICE,
        "exit_price_source": "btcusdt.shadow_close",
        "exit_exec_price": exit_exec,
        "exit_exec_price_kind": "synthetic_fee_slippage_model",
        "synthetic_exit_exec_price": exit_exec,
        "gross_return_frac": gross,
        "entry_fee_rate": FEE,
        "entry_fee_model": "synthetic_default",
        "exit_fee_rate": FEE,
        "exit_fee_model": "synthetic_default",
        "roundtrip_fee_rate": FEE * 2,
        "fee_model": "synthetic_default+synthetic_default",
        "fee_cost_frac": fee_cost_frac,
        "pnl_frac": pnl_frac,
        "pnl_pct": pnl_pct,
        "remaining_position_pnl_frac": pnl_frac,
        "position_realized_pnl_frac_before_close": 0.0,
        "total_position_pnl_frac_est": pnl_frac,
        "hold_bars": 3493,
        "position_fraction": NOTIONAL,
        "margin_fraction": NOTIONAL,
        "execution_leverage": 1.0,
        "notional_exposure": NOTIONAL,
        "total_exposure": NOTIONAL,
        "regime": "SHADOW",
        "source": "omega4_6_1_shadow|take_profit",
        "reason": "omega4_6_1_shadow_take_profit",
        "audit_schema_version": "trade_journal.audit.v2",
        "ledger_ts_kind": "shadow_bar_close",
        "decision_made_at_kst": TS_KST,
        "decision_bar_ts": TS_KST,
        "decision_bar_utc": BAR_UTC,
        "decision_bar_open": 67177.10,
        "decision_bar_high": 70450.00,
        "decision_bar_low": 67060.20,
        "decision_bar_close": EXIT_PRICE,
        "decision_bar_volume": 0.0,
        "decision_bar_is_complete": True,
        "decision_price": EXIT_PRICE,
        "decision_price_source": "btcusdt.close[-1]",
        "execution_bar_ts": TS_KST,
        "execution_bar_utc": BAR_UTC,
        "execution_bar_open": 67177.10,
        "execution_bar_high": 70450.00,
        "execution_bar_low": 67060.20,
        "execution_bar_close": EXIT_PRICE,
        "execution_bar_volume": 0.0,
        "execution_bar_is_current": True,
        "execution_price": EXIT_PRICE,
        "execution_price_source": "btcusdt.shadow_close",
        "execution_delay_sec": 0.0,
        "execution_delay_late": False,
        "execution_delay_mode": "shadow_only_bar_close",
        "ai_timing": {},
        "model_version": "Omega4.6.1-live-20260706",
        "model_id": "omega4_6_1_duration_ou_halflife_risk_gate_20260630",
        "model_path": "",
        "model_sleeve": "omega4_6_1_duration_ou_halflife_risk_gate",
        "scout_prob": 0.0,
        "scout_frac": 0.0,
        "scout_probability_threshold": 0.0,
        "scout_cost_pass": False,
        "learned_config": {},
        "take_profit": 0.075,
        "stop_loss": 0.04,
        "max_hold_bars": 0,
        "max_hold_remaining_bars": 0,
        "take_profit_price": 69997.0125,
        "stop_price": 62508.96,
        "trailing_stop_price": 0.0,
        "effective_take_profit": 0.075,
        "effective_stop_loss": 0.04,
        "v31_q_long": 0.0,
        "v31_q_short": 0.0,
        "v31_q_long_raw": 0.0,
        "v31_q_short_raw": 0.0,
        "v31_edge": 0.0,
        "v31_margin": 0.0,
        "v31_raw_margin": 0.0,
        "v31_selected_side": "",
        "v31_pass_gate": False,
        "v31_guard_reason": "",
        "v31_transition_risk": 0.0,
        "parent_action": 0,
        "parent_side": 0,
        "omega5_source_roundtrip_cost": 0.0,
        "omega5_source_exit_reason": "",
        "omega5_source_exit_price_move": 0.0,
        "teacher_gate_result": "",
        "teacher_pred_action": 0,
        "teacher_confidence": 0.0,
        "teacher_quality": 0.0,
        "teacher_keep_parent": False,
        "exchange_execution_enabled": False,
        "exchange_execution_dry_run": True,
        "exchange_execution_status": "disabled",
        "exchange_order_count": 0,
        "exchange_fill_price_source": "",
        "exchange_entry_price": 0.0,
        "exchange_exit_price": 0.0,
        "entry_execution_liquidity": "",
        "entry_execution_route": "",
        "entry_execution_order_type": "",
        "exit_execution_liquidity": "",
        "exit_execution_route": "",
        "exit_execution_order_type": "",
        "close_decision_model_id": "omega4_6_1_duration_ou_halflife_risk_gate_20260630",
        "close_decision_model_version": "Omega4.6.1-live-20260706",
        "close_decision_model_path": "",
        "close_decision_model_sleeve": "omega4_6_1_duration_ou_halflife_risk_gate",
        "open_model_id": "omega4_6_1_duration_ou_halflife_risk_gate_20260630",
        "open_model_version": "Omega4.6.1-live-20260706",
        "open_model_path": "",
        "open_model_sleeve": "omega4_6_1_duration_ou_halflife_risk_gate",
        "open_source": "omega4_6_1_shadow|h48qual",
        # -- manual correction disclosure (not part of the organic schema) --
        "manual_correction": True,
        "manual_correction_reason": (
            "Live exit check only ever inspected processed.iloc[-1] (the single "
            "latest bar); this position's real Binance 5m candle "
            "(2026-08-19 15:25 UTC, high=70450.00) touched take_profit "
            "(69997.0125) but was never evaluated against it before a later bar "
            "became 'latest'. Root cause confirmed live 2026-08-20 (two decision "
            "cycles 13s apart landed on the same bar right after a restart, "
            "proving a fetch can catch up by >1 bar). Fixed in commit 8abafb7 "
            "(trading_bot.py: last_checked_bar_ts + unseen-bar-range scan). "
            "Backdated by explicit user request; see memory "
            "btc_omega461_shadow_tp_barrier_single_bar_miss_20260820.md."
        ),
        "manual_correction_applied_at": manual_correction_applied_at,
        "manual_correction_hold_bars_estimated": True,
        "manual_correction_hold_bars_note": (
            "hold_bars is a decision-cycle counter, not a wall-clock bar count "
            "(confirmed: two cycles 13s apart both incremented it while landing "
            "on the same bar), so it cannot be exactly reconstructed after the "
            "fact. 3493 is a calendar-rate estimate adjusted for the ~11-bar/12d "
            "slippage observed between entry and the next confirmed real "
            "checkpoint (hold_bars=3496 at 2026-08-19 15:40 UTC, 15 real minutes "
            "after this close) -- the true value is bounded to be < 3496."
        ),
    }
    return row


def build_dashboard_event(close_row, manual_correction_applied_at):
    return {
        "ts": TS_KST,
        "event": "EXIT LONG",
        "from": "LONG",
        "to": None,
        "price": EXIT_PRICE,
        "asset": "btc",
        "symbol": "BTCUSDT",
        "shadow_only": True,
        "source": "omega4_6_1_shadow|take_profit",
        "close_trade": close_row,
        "open_trade": None,
        "manual_correction": True,
        "manual_correction_applied_at": manual_correction_applied_at,
    }


def main():
    import datetime

    manual_correction_applied_at = datetime.datetime.now().astimezone().isoformat()

    if not os.path.exists(STATE_PATH):
        sys.exit(f"ABORT: state file not found: {STATE_PATH}")
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    if state.get("pos") != "LONG":
        sys.exit(f"ABORT: expected pos=LONG in live state, found {state.get('pos')!r} -- state has moved on, refusing to apply blindly")
    if state.get("open_trade_id") != TRADE_ID:
        sys.exit(f"ABORT: expected open_trade_id={TRADE_ID!r}, found {state.get('open_trade_id')!r}")
    if abs(float(state.get("entry_price", 0.0)) - ENTRY_PRICE) > 1e-6:
        sys.exit(f"ABORT: expected entry_price={ENTRY_PRICE}, found {state.get('entry_price')!r}")

    if os.path.exists(JOURNAL_PATH):
        with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("trade_id") == TRADE_ID and row.get("kind") == "CLOSE":
                    sys.exit(f"ABORT: trade_journal.jsonl already has a CLOSE for {TRADE_ID}, refusing to double-apply")

    close_row = build_close_row(manual_correction_applied_at)
    dashboard_event = build_dashboard_event(close_row, manual_correction_applied_at)

    last_closed_hold_count = int(close_row["hold_bars"])
    realized = float(close_row["pnl_frac"])

    trade_history_row = {
        "ts": TS_KST.replace(" ", "T"),
        "pnl_frac": realized,
        "hold_bars": last_closed_hold_count,
    }
    for k, v in close_row.items():
        if k not in trade_history_row:
            trade_history_row[k] = v

    new_state = dict(state)
    new_state["pos"] = None
    new_state["entry_price"] = 0.0
    new_state["hold_count"] = 0
    new_state["open_trade_id"] = ""
    new_state["opened_at"] = ""
    new_state["decision_at"] = ""
    new_state["entry_price_source"] = ""
    new_state["entry_decision_price"] = 0.0
    new_state["exchange_entry_price"] = 0.0
    new_state["entry_execution_liquidity"] = ""
    new_state["entry_execution_route"] = ""
    new_state["entry_execution_order_type"] = ""
    new_state["open_model_version"] = ""
    new_state["open_model_id"] = ""
    new_state["open_model_path"] = ""
    new_state["open_model_sleeve"] = ""
    new_state["open_source"] = ""
    new_strategy_state = dict(state.get("strategy_state", {}) or {})
    new_strategy_state.pop("omega4_6_1_active", None)  # matches router.strategy_state.pop(OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY, None)
    new_state["strategy_state"] = new_strategy_state
    new_state["current_exposure"] = 0.0
    new_state["current_leverage"] = 0.0
    new_state["position_fraction"] = 0.0
    new_state["execution_leverage"] = 1.0
    new_state["peak_equity"] = 1.0
    new_state["cur_equity"] = 1.0
    new_state["position_realized_pnl_frac"] = 0.0
    new_state["last_resize_realized_pnl_frac"] = 0.0
    new_state["last_realized_pnl"] = realized
    new_state["last_closed_hold_count"] = last_closed_hold_count
    new_state["trend_mismatch_streak"] = 0
    new_state["position_exit_streak"] = 0
    new_state["recent_realized"] = list(state.get("recent_realized", [])) + [realized]
    if len(new_state["recent_realized"]) > 20:
        new_state["recent_realized"] = new_state["recent_realized"][-20:]
    new_state["trade_history"] = list(state.get("trade_history", [])) + [trade_history_row]
    if len(new_state["trade_history"]) > 2000:
        new_state["trade_history"] = new_state["trade_history"][-2000:]
    new_state["saved_at"] = datetime.datetime.utcnow().isoformat()

    print("About to apply:")
    print(f"  trade_journal.jsonl  <- append 1 CLOSE row (trade_id={TRADE_ID}, pnl_pct={close_row['pnl_pct']:.4f}%)")
    print(f"  dashboard_events.jsonl <- append 1 EXIT LONG event")
    print(f"  {STATE_PATH} <- pos LONG -> None, trade_history +1, recent_realized +1")
    print()

    _atomic_append_jsonl(JOURNAL_PATH, close_row)
    print(f"OK: appended CLOSE to {JOURNAL_PATH}")

    _atomic_append_jsonl(EVENTS_PATH, dashboard_event)
    print(f"OK: appended event to {EVENTS_PATH}")

    _atomic_write_json(STATE_PATH, new_state)
    print(f"OK: updated {STATE_PATH} (pos=None)")

    print()
    print("Done. The BTC omega4_6_1_shadow position is now recorded as closed via")
    print(f"take_profit at {EXIT_PRICE} on {TS_KST} KST, pnl_pct={close_row['pnl_pct']:.4f}%.")
    print("Restart trading-bot.service (or wait for its next natural restart) so the")
    print("live process reloads this state -- it currently still has the old LONG")
    print("position cached in memory from when it started.")


if __name__ == "__main__":
    main()
