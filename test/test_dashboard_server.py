from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb
from aiohttp.test_utils import TestClient, TestServer

from dashboard import server


def write_scalp_shadow_fixture(
    live_dir: Path,
    asset: str = "eth",
    configs: dict | None = None,
) -> Path:
    configs = server.SCALP_SHADOW_ASSETS if configs is None else configs
    config = configs[asset]
    model_hash = "fixture-model-hash"
    fee_scenarios = {
        f"{fee:.2f}bp_per_notional_change": {
            "compounded_return_pct": 0.0,
            "additive_gross_return_pct": 0.0,
            "additive_cost_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
        for fee in server.SCALP_SHADOW_FEES_BP
    }
    state = {
        "schema_version": config["state_schema"],
        "model_id": config["model_id"],
        "model_sha256": model_hash,
        "activation_allowed": False,
        "order_submission_supported": False,
        "stream": {
            "latest_feature_completed_at_utc": "2026-07-18 04:03:00",
        },
        "summary": {
            "schema_version": config["summary_schema"],
            "model_id": config["model_id"],
            "decision_count": 3,
            "settled_intervals": 2,
            "unit_notional": 1.0,
            "fee_scenarios": fee_scenarios,
            "evidence_class": "counterfactual completed-close-to-next-completed-close",
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "fixed_holding_period_used": False,
            "performance_eligible": False,
            "order_submission_supported": False,
        },
    }
    if config.get("require_asset_contract"):
        expected_asset = config["asset"]
        state.update({"asset": expected_asset, "research_policy_enabled": expected_asset != "btc"})
        state["summary"].update(
            {
                "asset": expected_asset,
                "symbol": config["symbol"],
                "parent_model_id": server.SCALP_SHADOW_MODEL_ID,
                "research_policy_enabled": expected_asset != "btc",
            }
        )
        if config.get("mode"):
            dynamic_exit = config["mode"] == "eth_lifecycle"
            state.update({"mode": config["mode"], "dynamic_exit_enabled": dynamic_exit})
            state["summary"].update(
                {
                    "mode": config["mode"],
                    "dynamic_exit_enabled": dynamic_exit,
                    "high_risk_bars": 1,
                }
            )
    (live_dir / config["state_file"]).write_text(
        json.dumps(
            state
        ),
        encoding="utf-8",
    )
    database = live_dir / config["database_file"]
    connection = duckdb.connect(str(database))
    try:
        connection.execute(
            """
            CREATE TABLE observer_metadata (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                model_id VARCHAR NOT NULL,
                model_sha256 VARCHAR NOT NULL,
                fresh_start_utc TIMESTAMP NOT NULL,
                order_submission_supported BOOLEAN NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO observer_metadata VALUES (true, ?, ?, ?, ?, false)",
            [
                config["observer_schema"],
                config["model_id"],
                model_hash,
                "2026-07-18 04:00:00",
            ],
        )
        connection.execute(
            """
            CREATE TABLE decisions (
                timestamp TIMESTAMP PRIMARY KEY,
                model_id VARCHAR NOT NULL,
                model_sha256 VARCHAR NOT NULL,
                feature_hash_sha256 VARCHAR NOT NULL,
                close DOUBLE NOT NULL,
                available BOOLEAN NOT NULL,
                previous_position INTEGER NOT NULL,
                target_position INTEGER NOT NULL,
                position_change INTEGER NOT NULL,
                intent_id VARCHAR,
                intent_side VARCHAR,
                notional_change DOUBLE NOT NULL,
                diagnostics_json VARCHAR NOT NULL,
                execution_evidence_status VARCHAR NOT NULL
            )
            """
        )
        decisions = [
            ("2026-07-18 04:00:00", 100.0, 0, 1, 1, "BUY"),
            ("2026-07-18 04:01:00", 101.0, 1, 0, -1, "SELL"),
            ("2026-07-18 04:02:00", 102.0, 0, 0, 0, None),
        ]
        for timestamp, close, previous, target, change, side in decisions:
            connection.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, true, ?, ?, ?, ?, ?, ?, '{}', ?)",
                [
                    timestamp,
                    config["model_id"],
                    model_hash,
                    "feature-hash",
                    close,
                    previous,
                    target,
                    change,
                    f"intent-{timestamp}" if side else None,
                    side,
                    abs(change),
                    "unobserved" if side else "not_applicable",
                ],
            )
        connection.execute(
            """
            CREATE TABLE shadow_pnl (
                decision_timestamp TIMESTAMP NOT NULL,
                settlement_timestamp TIMESTAMP NOT NULL,
                fee_bp DOUBLE NOT NULL,
                previous_position INTEGER NOT NULL,
                position INTEGER NOT NULL,
                start_close DOUBLE NOT NULL,
                end_close DOUBLE NOT NULL,
                turnover DOUBLE NOT NULL,
                price_return DOUBLE NOT NULL,
                gross_return DOUBLE NOT NULL,
                cost_return DOUBLE NOT NULL,
                net_return DOUBLE NOT NULL,
                equity DOUBLE NOT NULL,
                causal_settlement BOOLEAN NOT NULL,
                PRIMARY KEY (decision_timestamp, fee_bp)
            )
            """
        )
        for fee in server.SCALP_SHADOW_FEES_BP:
            entry_cost = fee / 10_000.0
            first_net = 0.01 - entry_cost
            first_equity = 1.0 + first_net
            second_net = -entry_cost
            second_equity = first_equity * (1.0 + second_net)
            connection.execute(
                "INSERT INTO shadow_pnl VALUES (?, ?, ?, 0, 1, 100, 101, 1, 0.01, 0.01, ?, ?, ?, true)",
                ["2026-07-18 04:00:00", "2026-07-18 04:01:00", fee, entry_cost, first_net, first_equity],
            )
            connection.execute(
                "INSERT INTO shadow_pnl VALUES (?, ?, ?, 1, 0, 101, 102, 1, ?, 0, ?, ?, ?, true)",
                ["2026-07-18 04:01:00", "2026-07-18 04:02:00", fee, 102 / 101 - 1, entry_cost, second_net, second_equity],
            )
    finally:
        connection.close()
    return database


class DashboardServerTest(unittest.TestCase):
    def test_dashboard_api_compresses_and_revalidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            live_dir = tmp_path / "live"
            dashboard_dir = tmp_path / "dashboard"
            live_dir.mkdir()
            dashboard_dir.mkdir()
            (dashboard_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            (live_dir / "dashboard_state.json").write_text(
                json.dumps({"updated_at": "2026-07-17T12:00:00", "price": 100}),
                encoding="utf-8",
            )
            (live_dir / "dashboard_state_governor.json").write_text(
                json.dumps({"governor_mode": True}),
                encoding="utf-8",
            )
            (live_dir / "trade_journal.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "CLOSE",
                        "source": "TREND",
                        "closed_at": "2026-07-17T12:00:00",
                        "pnl_pct": 1.25,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            async def exercise_api() -> None:
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    state_response = await client.get(
                        "/api/state",
                        headers={"Accept-Encoding": "gzip"},
                    )
                    self.assertEqual(state_response.status, 200)
                    self.assertEqual(state_response.headers["Cache-Control"], "no-cache")
                    self.assertEqual(state_response.headers["Content-Encoding"], "gzip")
                    state_etag = state_response.headers["ETag"]
                    self.assertEqual((await state_response.json())["state"]["price"], 100)

                    unchanged_state = await client.get(
                        "/api/state",
                        headers={"If-None-Match": state_etag},
                    )
                    self.assertEqual(unchanged_state.status, 304)
                    self.assertEqual(await unchanged_state.read(), b"")

                    trades_response = await client.get("/api/trades?source=ALL")
                    self.assertEqual(trades_response.status, 200)
                    trades_etag = trades_response.headers["ETag"]
                    trades_payload = await trades_response.json()
                    self.assertEqual(trades_payload["rows"][0]["source"], "TREND")
                    self.assertAlmostEqual(
                        trades_payload["equity"][0]["cumulative_return_pct"],
                        1.25,
                    )

                    unchanged_trades = await client.get(
                        "/api/trades?source=ALL",
                        headers={"If-None-Match": trades_etag},
                    )
                    self.assertEqual(unchanged_trades.status, 304)
                    self.assertEqual(await unchanged_trades.read(), b"")

                    (live_dir / "dashboard_state.json").write_text(
                        json.dumps({"updated_at": "2026-07-17T12:00:01", "price": 101}),
                        encoding="utf-8",
                    )
                    changed_state = await client.get(
                        "/api/state",
                        headers={"If-None-Match": state_etag},
                    )
                    self.assertEqual(changed_state.status, 200)
                    self.assertNotEqual(changed_state.headers["ETag"], state_etag)
                    self.assertEqual((await changed_state.json())["state"]["price"], 101)
                finally:
                    await client.close()

            with (
                mock.patch.object(server, "LIVE_DIR", live_dir),
                mock.patch.object(server, "DASHBOARD_DIR", dashboard_dir),
            ):
                asyncio.run(exercise_api())

    def test_scalp_shadow_component_api_is_counterfactual_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            live_dir = tmp_path / "live"
            dashboard_dir = tmp_path / "dashboard"
            live_dir.mkdir()
            dashboard_dir.mkdir()
            (dashboard_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            write_scalp_shadow_fixture(live_dir)

            async def exercise_api() -> None:
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    response = await client.get("/api/scalp-shadow")
                    self.assertEqual(response.status, 200)
                    etag = response.headers["ETag"]
                    payload = await response.json()
                    self.assertFalse(payload["contract"]["actual_execution"])
                    self.assertFalse(payload["contract"]["performance_eligible"])
                    self.assertFalse(payload["contract"]["fixed_holding_period_used"])
                    self.assertEqual(payload["contract"]["display_fee_bp"], 4.5)
                    self.assertEqual(payload["summary"]["decision_count"], 3)
                    self.assertEqual(payload["summary"]["settled_intervals"], 2)
                    self.assertEqual(payload["summary"]["positioned_intervals"], 1)
                    self.assertTrue(payload["summary"]["pnl_sample_ready"])
                    self.assertEqual(len(payload["fee_scenarios"]), 4)
                    self.assertEqual(len(payload["equity"]), 2)
                    self.assertIsNone(payload["recent_decisions"][0]["net_return_pct"])

                    unchanged = await client.get(
                        "/api/scalp-shadow",
                        headers={"If-None-Match": etag},
                    )
                    self.assertEqual(unchanged.status, 304)

                    state_path = live_dir / "eth_micro_scalp_v4_shadow_state.json"
                    invalid_state = json.loads(state_path.read_text(encoding="utf-8"))
                    invalid_state["summary"]["performance_eligible"] = True
                    state_path.write_text(json.dumps(invalid_state), encoding="utf-8")
                    invalid = await client.get("/api/scalp-shadow")
                    self.assertEqual(invalid.status, 503)
                    self.assertEqual((await invalid.json())["error"], "scalp_shadow_contract_error")
                finally:
                    await client.close()

            with (
                mock.patch.object(server, "LIVE_DIR", live_dir),
                mock.patch.object(server, "DASHBOARD_DIR", dashboard_dir),
            ):
                asyncio.run(exercise_api())

    def test_scalp_shadow_component_selects_btc_and_sol_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            live_dir = tmp_path / "live"
            dashboard_dir = tmp_path / "dashboard"
            live_dir.mkdir()
            dashboard_dir.mkdir()
            (dashboard_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            write_scalp_shadow_fixture(live_dir, "btc")
            write_scalp_shadow_fixture(live_dir, "sol")

            async def exercise_api() -> None:
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    btc = await client.get("/api/scalp-shadow?asset=btc")
                    self.assertEqual(btc.status, 200)
                    btc_payload = await btc.json()
                    self.assertEqual(btc_payload["contract"]["asset"], "btc")
                    self.assertFalse(btc_payload["contract"]["research_policy_enabled"])

                    sol = await client.get("/api/scalp-shadow?asset=sol")
                    self.assertEqual(sol.status, 200)
                    sol_payload = await sol.json()
                    self.assertEqual(sol_payload["contract"]["symbol"], "SOLUSDT")
                    self.assertTrue(sol_payload["contract"]["research_policy_enabled"])

                    invalid = await client.get("/api/scalp-shadow?asset=xrp")
                    self.assertEqual(invalid.status, 400)
                    self.assertEqual(
                        (await invalid.json())["error"],
                        "unsupported_scalp_shadow_asset",
                    )
                finally:
                    await client.close()

            with (
                mock.patch.object(server, "LIVE_DIR", live_dir),
                mock.patch.object(server, "DASHBOARD_DIR", dashboard_dir),
            ):
                asyncio.run(exercise_api())

    def test_scalp_reuse_shadow_component_selects_lifecycle_and_entry_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            live_dir = tmp_path / "live"
            dashboard_dir = tmp_path / "dashboard"
            live_dir.mkdir()
            dashboard_dir.mkdir()
            (dashboard_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            for mode in server.SCALP_REUSE_MODES:
                write_scalp_shadow_fixture(live_dir, mode, server.SCALP_REUSE_MODES)

            async def exercise_api() -> None:
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    lifecycle = await client.get(
                        "/api/scalp-reuse-shadow?mode=eth_lifecycle"
                    )
                    self.assertEqual(lifecycle.status, 200)
                    lifecycle_payload = await lifecycle.json()
                    self.assertEqual(
                        lifecycle_payload["contract"]["mode"], "eth_lifecycle"
                    )
                    self.assertTrue(
                        lifecycle_payload["contract"]["dynamic_exit_enabled"]
                    )
                    self.assertEqual(lifecycle_payload["summary"]["high_risk_bars"], 1)

                    entry = await client.get(
                        "/api/scalp-reuse-shadow?mode=sol_entry"
                    )
                    self.assertEqual(entry.status, 200)
                    entry_payload = await entry.json()
                    self.assertEqual(entry_payload["contract"]["asset"], "sol")
                    self.assertFalse(entry_payload["contract"]["dynamic_exit_enabled"])

                    invalid = await client.get(
                        "/api/scalp-reuse-shadow?mode=portfolio_allocator"
                    )
                    self.assertEqual(invalid.status, 400)
                    self.assertEqual(
                        (await invalid.json())["error"],
                        "unsupported_scalp_reuse_mode",
                    )
                finally:
                    await client.close()

            with (
                mock.patch.object(server, "LIVE_DIR", live_dir),
                mock.patch.object(server, "DASHBOARD_DIR", dashboard_dir),
            ):
                asyncio.run(exercise_api())


if __name__ == "__main__":
    unittest.main()
