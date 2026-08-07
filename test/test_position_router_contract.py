from __future__ import annotations

import json

import pytest

from trading_bot_modules.position_router import GovernorPositionRouter


def test_position_router_sizing_uses_fraction_times_leverage(tmp_path, monkeypatch):
    state_path = tmp_path / "router_state.json"
    monkeypatch.setenv("GOVERNOR_LIVE_STATE_PATH", str(state_path))
    router = GovernorPositionRouter()

    router._set_position_sizing(fraction=0.30, leverage_mult=3.0)

    assert router.position_fraction == 0.30
    assert router.execution_leverage == 3.0
    assert router.current_leverage == pytest.approx(0.90)


def test_position_router_state_round_trip_preserves_model_identity(tmp_path, monkeypatch):
    state_path = tmp_path / "router_state.json"
    monkeypatch.setenv("GOVERNOR_LIVE_STATE_PATH", str(state_path))
    router = GovernorPositionRouter()
    router._open_position("LONG", 100.0, fraction=0.30, leverage_mult=3.0)
    router._set_open_model_identity(
        {
            "model_version": "test-v1",
            "model_id": "test-model",
            "model_path": "model.bin",
            "model_sleeve": "test-sleeve",
        },
        source="contract-test",
    )
    router._save_live_state()

    restored = GovernorPositionRouter()

    assert restored.pos == "LONG"
    assert restored.current_leverage == pytest.approx(0.90)
    assert restored.open_model_id == "test-model"
    assert restored.open_source == "contract-test"
    assert json.loads(state_path.read_text(encoding="utf-8"))["open_model_id"] == "test-model"


def test_position_router_fails_fast_on_corrupt_existing_state(tmp_path, monkeypatch):
    state_path = tmp_path / "router_state.json"
    state_path.write_text("{broken", encoding="utf-8")
    monkeypatch.setenv("GOVERNOR_LIVE_STATE_PATH", str(state_path))

    with pytest.raises(RuntimeError, match="governor_live_state_load_failed"):
        GovernorPositionRouter()


def test_reconcile_external_position_falls_back_when_payload_build_fails(tmp_path, monkeypatch):
    """If build_close_trade_payload blows up while reconciling an exchange-confirmed flat
    position, the realized PnL must still be recorded via the _mark_pnl_frac fallback
    instead of being silently dropped while the router resets to flat anyway."""
    state_path = tmp_path / "router_state.json"
    monkeypatch.setenv("GOVERNOR_LIVE_STATE_PATH", str(state_path))
    router = GovernorPositionRouter()
    router._open_position("LONG", 100.0, fraction=0.30, leverage_mult=3.0)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated audit/feature snapshot failure")

    monkeypatch.setattr(router, "build_close_trade_payload", _boom)

    router.reconcile_external_position(None, 0.0, current_price=110.0)

    assert router.pos is None
    assert router._last_reconcile_close_payload is not None
    assert router._last_reconcile_close_payload["pnl_frac"] != 0.0
    assert "degraded_reconcile_payload" in router._last_reconcile_close_payload["reason"]
    assert router.trade_history[-1]["pnl_frac"] == router._last_reconcile_close_payload["pnl_frac"]
