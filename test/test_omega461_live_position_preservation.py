from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import trading_bot


class Omega461LivePositionPreservationTest(unittest.TestCase):
    def test_disabled_assets_detect_persisted_open_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            (state_dir / "omega4_6_1_shadow_btc_state.json").write_text(
                json.dumps({"pos": "LONG"}), encoding="utf-8"
            )
            (state_dir / "omega4_6_1_shadow_sol_state.json").write_text(
                json.dumps({"pos": None}), encoding="utf-8"
            )

            self.assertEqual(
                trading_bot._omega461_persisted_open_assets(state_dir),
                ["btc"],
            )

    def test_error_state_preserves_open_short(self) -> None:
        router = SimpleNamespace(
            pos="SHORT",
            entry_price=77.94,
            current_leverage=0.48242092945580084,
            execution_leverage=3.7068752812748387,
            position_fraction=0.13014220680494287,
            hold_count=2293,
            open_trade_id="trade-sol-open",
            opened_at="2026-07-22T12:15:20+09:00",
            unrealized_pnl=lambda price: 0.01,
        )
        active = {
            "source_component": "zig075",
            "source": "omega4_6_1_shadow|zig075",
            "reason": "omega4_6_1_shadow_hold",
            "take_profit": 0.075,
            "stop_loss": 0.04,
            "quality_score": 0.71,
            "confidence": 0.78,
        }

        state = trading_bot._omega461_shadow_error_state(
            asset_key="sol",
            cfg={"label": "SOL", "symbol": "SOLUSDT", "account_symbol": "SOL/USDT:USDT"},
            router=router,
            active=active,
            current_price=76.0,
            updated_at="2026-07-30T03:00:00Z",
            error="temporary ancillary failure",
        )

        self.assertEqual(state["status"], "error")
        self.assertEqual(state["position"]["current"], "SHORT")
        self.assertEqual(state["position"]["trade_id"], "trade-sol-open")
        self.assertEqual(state["signal"]["final_action"], 2)
        self.assertEqual(state["position"]["take_profit"], 0.075)
        self.assertEqual(state["position"]["stop_loss"], 0.04)


if __name__ == "__main__":
    unittest.main()
