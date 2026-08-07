from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trading_bot_modules.omega4_6_1_shadow_state import (
    OMEGA4_6_1_SHADOW_ACTIVE_CONTRACT,
    OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY,
    validate_omega461_shadow_active_state,
)
from trading_bot_modules.position_router import GovernorPositionRouter


def active_state(**overrides) -> dict:
    payload = {
        "contract_version": OMEGA4_6_1_SHADOW_ACTIVE_CONTRACT,
        "side": "SHORT",
        "source_component": "zig075",
        "entry_price": 75.0,
        "margin_fraction": 0.30,
        "leverage": 5.0,
        "notional_exposure": 1.50,
        "take_profit": 0.075,
        "stop_loss": 0.04,
        "quality_score": 0.71,
        "confidence": 0.78,
        "mfe": 0.01,
        "mae": -0.02,
    }
    payload.update(overrides)
    return payload


class Omega461ShadowStateContractTest(unittest.TestCase):
    def validate(self, active: object) -> dict:
        return validate_omega461_shadow_active_state(
            active,
            asset_key="sol",
            expected_component="zig075",
            position="SHORT",
            entry_price=75.0,
            position_fraction=0.30,
            execution_leverage=5.0,
            notional_exposure=1.50,
        )

    def test_router_round_trip_preserves_valid_active_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "sol_state.json"
            with mock.patch.dict(
                "os.environ",
                {"GOVERNOR_LIVE_STATE_PATH": str(state_path)},
            ):
                router = GovernorPositionRouter()
                router._open_position("SHORT", 75.0, fraction=0.30, leverage_mult=5.0)
                router.strategy_state[OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY] = self.validate(
                    active_state()
                )
                router._save_live_state()

                restored = GovernorPositionRouter()

            restored_active = self.validate(
                restored.strategy_state[OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY]
            )
            self.assertEqual(restored_active["source_component"], "zig075")
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(
                saved["strategy_state"][OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY]["contract_version"],
                OMEGA4_6_1_SHADOW_ACTIVE_CONTRACT,
            )

    def test_open_position_without_active_state_fails_fast(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing_for_open_position"):
            self.validate(None)

    def test_sizing_contract_mismatch_fails_fast(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "margin_x_leverage"):
            validate_omega461_shadow_active_state(
                active_state(notional_exposure=1.40),
                asset_key="sol",
                expected_component="zig075",
                position="SHORT",
                entry_price=75.0,
                position_fraction=0.30,
                execution_leverage=5.0,
                notional_exposure=1.40,
            )

    def test_active_state_while_flat_fails_fast(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "active_state_present_while_flat"):
            validate_omega461_shadow_active_state(
                active_state(),
                asset_key="sol",
                expected_component="zig075",
                position=None,
                entry_price=0.0,
                position_fraction=0.0,
                execution_leverage=1.0,
                notional_exposure=0.0,
            )


if __name__ == "__main__":
    unittest.main()
