from __future__ import annotations


def classify_account_position_snapshot(snapshot: dict | None) -> tuple[str, dict | None]:
    """Return unavailable/flat/open without conflating a failed query with flat."""
    payload = dict(snapshot or {})
    if payload.get("position_query_ok") is not True:
        return "unavailable", None
    position = payload.get("position")
    if isinstance(position, dict) and str(position.get("type", "")).upper() in {"LONG", "SHORT"}:
        return "open", dict(position)
    return "flat", None


def exchange_position_went_flat(position_state: str, local_position: str | None) -> bool:
    return str(position_state) == "flat" and str(local_position) in {"LONG", "SHORT"}
