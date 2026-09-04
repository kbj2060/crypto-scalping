#!/usr/bin/env python3
"""Parity: live PortfolioRiskManager (prealloc) == replay `_replay_concurrent(cap_mode="prealloc")`
arithmetic (scripts/replay_portfolio_concurrent_3asset_native_20260712.py, open pass), plus the
2026-09-04 journal telemetry contract. Runs without torch (the replay module itself needs it, so its
prealloc branch is ported verbatim below and pinned by a source-text check).

python scripts/test_portfolio_cap_prealloc_parity_20260904.py
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_bot_modules.portfolio_risk import PortfolioRiskConfig, PortfolioRiskManager, portfolio_cap_trace  # noqa: E402
from trading_bot_modules.position_router import GovernorPositionRouter  # noqa: E402

REPLAY = ROOT / "scripts/replay_portfolio_concurrent_3asset_native_20260712.py"
REPLAY_MIN_NOTIONAL = 0.05


def replay_prealloc(notional: float, asset: str, total_notional_cap, asset_shares: dict) -> float | None:
    """Verbatim port of the replay's `elif cap_mode == "prealloc":` branch (None == skipped)."""
    notional_final = notional
    if total_notional_cap is not None:
        budget = total_notional_cap * (asset_shares or {}).get(asset, 0.0)
        notional_final = min(notional_final, max(0.0, budget))
    if notional_final < REPLAY_MIN_NOTIONAL - 1e-9:
        return None
    return notional_final


def replay_normalize(raw: dict) -> dict:
    share_sum = sum(raw.values())
    return {k: v / share_sum for k, v in raw.items()}


# live keys the manager is constructed with (trading_bot.py main): slot keys, eth split by sub-share
def live_manager(cap, eth, btc, sol, eth_sub=1.0, sigma_sub=0.0) -> PortfolioRiskManager:
    return PortfolioRiskManager(PortfolioRiskConfig(total_notional_cap=cap, asset_shares={
        "eth_omega461": eth * eth_sub, "eth_sigma3_1h": eth * sigma_sub, "btc": btc, "sol": sol}))


GRID = [0.01, 0.0499, 0.05, 0.157, 0.26, 0.2784, 0.3372, 0.4824, 0.5, 0.5558, 0.6, 0.9, 1.1545, 1.3475,
        1.4247, 1.5, 1.6007, 1.6129, 1.8, 2.5, 3.0]
ASSET_KEY = {"eth": "eth_omega461", "btc": "btc", "sol": "sol"}


class PreallocParity(unittest.TestCase):
    def test_replay_branch_still_matches_port(self) -> None:
        src = REPLAY.read_text()
        for needle in ('elif cap_mode == "prealloc":', 'budget = total_notional_cap * (asset_shares or {}).get(asset, 0.0)',
                       'notional_final = min(notional_final, max(0.0, budget))', 'if notional_final < min_notional - 1e-9:'):
            self.assertIn(needle, src)
        self.assertRegex(src, re.compile(r"^MIN_NOTIONAL = 0\.05$", re.M))

    def _check(self, cap, raw_shares):
        norm = replay_normalize(raw_shares)
        risk = live_manager(cap, raw_shares["eth"], raw_shares["btc"], raw_shares["sol"])
        for asset in ("eth", "btc", "sol"):
            self.assertAlmostEqual(risk.asset_share(ASSET_KEY[asset]), norm[asset], places=12)
            self.assertAlmostEqual(risk.asset_budget(ASSET_KEY[asset]), cap * norm[asset], places=12)
            for n in GRID:
                rep = replay_prealloc(n, asset, cap, norm)
                approved = risk.scale_to_budget(ASSET_KEY[asset], n)
                live = None if approved < risk.config.min_notional else approved
                self.assertEqual(rep is None, live is None, (cap, raw_shares, asset, n))
                if rep is not None:
                    self.assertAlmostEqual(rep, live, places=12, msg=(cap, raw_shares, asset, n))

    def test_server_config_since_0820(self) -> None:
        self._check(3.0, {"eth": 0.5, "btc": 0.3, "sol": 0.2})

    def test_a4_configs(self) -> None:
        self._check(1.5, {"eth": 1.0, "btc": 1.0, "sol": 1.0})
        self._check(1.0, {"eth": 1.0, "btc": 1.0, "sol": 1.0})
        # env-style equal shares written as 0.3333 each normalize to exactly 1/3
        r = live_manager(1.5, 0.3333, 0.3333, 0.3333)
        for k in ("eth_omega461", "btc", "sol"):
            self.assertAlmostEqual(r.asset_budget(k), 0.5, places=12)

    def test_uncapped(self) -> None:
        r = live_manager(None, 1.0, 1.0, 1.0)
        for n in GRID:
            self.assertEqual(r.scale_to_budget("eth_omega461", n), n)
            self.assertEqual(replay_prealloc(n, "eth", None, {}), n if n >= REPLAY_MIN_NOTIONAL - 1e-9 else None)

    def test_journal_evidence_reproduced(self) -> None:
        """Server journal rows (cap 3.0/50-30-20): ETH capped to exactly 1.5, SOL to exactly 0.6, BTC 0.26 untouched."""
        r = live_manager(3.0, 0.5, 0.3, 0.2)
        self.assertAlmostEqual(r.scale_to_budget("eth_omega461", 1.6129), 1.5)
        self.assertAlmostEqual(r.scale_to_budget("sol", 0.75), 0.6)
        self.assertAlmostEqual(r.scale_to_budget("btc", 0.26), 0.26)
        a4 = live_manager(1.5, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(a4.scale_to_budget("eth_omega461", 1.4247), 0.5)
        self.assertAlmostEqual(a4.scale_to_budget("sol", 0.6), 0.5)
        self.assertAlmostEqual(a4.scale_to_budget("btc", 0.26), 0.26)

    def test_trace_contract_and_journal_whitelist(self) -> None:
        r = live_manager(1.5, 1.0, 1.0, 1.0)
        t = portfolio_cap_trace(r, "eth_omega461", 1.4247)
        self.assertEqual(t["approved_notional"], r.scale_to_budget("eth_omega461", 1.4247))
        self.assertEqual((t["requested_notional"], t["asset_budget"], t["scaled"], t["blocked"]), (1.4247, 0.5, True, False))
        self.assertAlmostEqual(t["asset_share"], 1 / 3)
        t2 = portfolio_cap_trace(r, "btc", 0.26)
        self.assertEqual((t2["scaled"], t2["blocked"], t2["reason"]), (False, False, "within_asset_budget"))
        t3 = portfolio_cap_trace(r, "eth_sigma3_1h", 0.3)  # zero sub-share slot -> budget 0 -> blocked
        self.assertEqual((t3["asset_budget"], t3["approved_notional"], t3["blocked"]), (0.0, 0.0, True))
        json.dumps(t); json.dumps(portfolio_cap_trace(live_manager(None, 1, 1, 1), "sol", 0.6))
        # journal audit whitelist carries the dict through untouched
        out = GovernorPositionRouter._journal_audit_fields({"portfolio_cap": t}, kind="OPEN")
        self.assertEqual(out["portfolio_cap"], t)
        self.assertEqual(GovernorPositionRouter._journal_audit_fields({}, kind="CLOSE")["portfolio_cap"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
