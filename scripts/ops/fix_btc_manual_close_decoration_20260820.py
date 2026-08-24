#!/usr/bin/env python3
"""
Fix for a bug in the earlier one-off remediation script
(scripts/ops/one_off_btc_manual_tp_close_20260820.py): the CLOSE row it
appended to data/live/trade_journal.jsonl was missing the asset/symbol/
account_symbol/market fields that _omega461_shadow_decorate_trade_row adds
to every real entry (build_close_trade_payload's own output doesn't include
them -- that decoration happens as a separate wrapping step in the real
code, which this script's build_close_row() didn't replicate). It also had
shadow_only=True where every real historical BTC/SOL shadow entry actually
records shadow_only=False (an executor object exists and returns a
"disabled" result rather than None, so the decorator's
`real_execution_result is None` check comes out False even though nothing
real executes).

Net effect: the inserted CLOSE record was invisible to anything that
filters trade_journal.jsonl by asset=="btc" -- defeating the point of
adding it. This patches the row(s) already written in place, without
touching anything else in the file, and does the matching fix in
dashboard_events.jsonl.

Safe to run anytime (idempotent: adding already-correct fields is a no-op)
and does not need the service stopped -- it only edits earlier lines by
trade_id + manual_correction match, never touches the tail the live
process appends to.
"""
import json
import os
import sys
import tempfile

REPO_ROOT = "/home/llewyn/crypto-scalping"
JOURNAL_PATH = os.path.join(REPO_ROOT, "data/live/trade_journal.jsonl")
EVENTS_PATH = os.path.join(REPO_ROOT, "data/live/dashboard_events.jsonl")
TRADE_ID = "trade-20260807T112515.312955Z"

DECORATION = {
    "asset": "btc",
    "symbol": "BTCUSDT",
    "account_symbol": "BTC/USDT:USDT",
    "market": "BTC",
    "shadow_only": False,
}


def _atomic_write_lines(path, lines):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line if line.endswith("\n") else line + "\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def patch_journal():
    with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    patched = 0
    out_lines = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        row = json.loads(stripped)
        if row.get("trade_id") == TRADE_ID and row.get("kind") == "CLOSE" and row.get("manual_correction") is True:
            changed = False
            for k, v in DECORATION.items():
                if row.get(k) != v:
                    row[k] = v
                    changed = True
            if changed:
                patched += 1
            out_lines.append(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            out_lines.append(line)

    if patched:
        _atomic_write_lines(JOURNAL_PATH, out_lines)
    print(f"trade_journal.jsonl: patched {patched} row(s) (0 = already correct, safe no-op)")


def patch_events():
    if not os.path.exists(EVENTS_PATH):
        print("dashboard_events.jsonl: not found, skipping")
        return
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    patched = 0
    out_lines = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        row = json.loads(stripped)
        close_trade = row.get("close_trade") or {}
        if (
            row.get("manual_correction") is True
            and isinstance(close_trade, dict)
            and close_trade.get("trade_id") == TRADE_ID
        ):
            changed = False
            if row.get("shadow_only") is not False:
                row["shadow_only"] = False
                changed = True
            if changed:
                patched += 1
            out_lines.append(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            out_lines.append(line)

    if patched:
        _atomic_write_lines(EVENTS_PATH, out_lines)
    print(f"dashboard_events.jsonl: patched {patched} row(s) (0 = already correct, safe no-op)")


def verify():
    with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("trade_id") == TRADE_ID and row.get("kind") == "CLOSE" and row.get("manual_correction") is True:
                print("verify: asset=%r symbol=%r account_symbol=%r market=%r shadow_only=%r" % (
                    row.get("asset"), row.get("symbol"), row.get("account_symbol"),
                    row.get("market"), row.get("shadow_only"),
                ))
                assert row.get("asset") == "btc"
                assert row.get("symbol") == "BTCUSDT"
                assert row.get("shadow_only") is False
                print("verify: OK")
                return
    print("verify: WARNING -- target row not found")


if __name__ == "__main__":
    patch_journal()
    patch_events()
    verify()
