from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/model_contracts/CURRENT_LIVE_MANIFEST.json"


def test_current_live_manifest_is_snapshot_only_and_non_secret() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "current_live_manifest_v1"
    assert manifest["promotion_eligible"] is False
    assert manifest["promotion_blockers"]
    assert manifest["sizing"]["contract"] == "notional = margin_fraction * leverage"

    serialized = json.dumps(manifest).lower()
    for secret_name in ("api_key", "api_secret", "telegram_bot_token", "password"):
        assert secret_name not in serialized


def test_current_live_manifest_records_every_active_artifact() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]

    assert set(artifacts) == {"eth", "sol", "btc"}
    assert {"h48qual_bundle", "h48qual_sidecar", "zig075_bundle", "zig075_sidecar"} <= set(
        artifacts["eth"]
    )
    assert {"bundle", "sidecar", "regime3"} <= set(artifacts["sol"])
    assert {"bundle", "sidecar", "regime3"} <= set(artifacts["btc"])

    for asset_records in artifacts.values():
        for record in asset_records.values():
            assert record["path"]
            assert record["exists"] is True
            assert len(record["sha256"]) == 64
