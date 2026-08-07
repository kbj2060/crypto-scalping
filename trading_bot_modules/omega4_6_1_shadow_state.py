from __future__ import annotations

import math


OMEGA4_6_1_SHADOW_ACTIVE_CONTRACT = "omega4_6_1.shadow_active.v1"
OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY = "omega4_6_1_active"


def validate_omega461_shadow_active_state(
    active: object,
    *,
    asset_key: str,
    expected_component: str,
    position: str | None,
    entry_price: float,
    position_fraction: float,
    execution_leverage: float,
    notional_exposure: float,
) -> dict:
    asset = str(asset_key or "").lower()
    pos = str(position or "").upper()

    def mismatch(field: str, detail: str) -> RuntimeError:
        return RuntimeError(
            f"omega4_6_1_shadow_active_contract_mismatch asset={asset} "
            f"field={field} detail={detail}"
        )

    if pos not in {"LONG", "SHORT"}:
        if active not in (None, {}):
            raise mismatch("position", "active_state_present_while_flat")
        return {}

    if not isinstance(active, dict) or not active:
        raise mismatch("active_state", "missing_for_open_position")

    out = dict(active)
    if out.get("contract_version") != OMEGA4_6_1_SHADOW_ACTIVE_CONTRACT:
        raise mismatch("contract_version", repr(out.get("contract_version")))
    if str(out.get("side", "")).upper() != pos:
        raise mismatch("side", f"saved={out.get('side')!r} position={pos!r}")
    if str(out.get("source_component", "")) != str(expected_component):
        raise mismatch(
            "source_component",
            f"saved={out.get('source_component')!r} expected={expected_component!r}",
        )

    positive_fields = {
        "entry_price": entry_price,
        "margin_fraction": position_fraction,
        "leverage": execution_leverage,
        "notional_exposure": notional_exposure,
        "take_profit": None,
        "stop_loss": None,
    }
    for field, expected in positive_fields.items():
        try:
            value = float(out.get(field, 0.0))
        except (TypeError, ValueError) as exc:
            raise mismatch(field, f"non_numeric={out.get(field)!r}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise mismatch(field, f"invalid={value!r}")
        if expected is not None:
            expected_value = float(expected)
            tolerance = max(1e-9, abs(expected_value) * 1e-8)
            if abs(value - expected_value) > tolerance:
                raise mismatch(field, f"saved={value!r} expected={expected_value!r}")

    for field in ("quality_score", "confidence", "mfe", "mae"):
        try:
            value = float(out[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise mismatch(field, f"missing_or_non_numeric={out.get(field)!r}") from exc
        if not math.isfinite(value):
            raise mismatch(field, f"non_finite={value!r}")

    expected_notional = float(out["margin_fraction"]) * float(out["leverage"])
    tolerance = max(1e-9, abs(expected_notional) * 1e-8)
    if abs(float(out["notional_exposure"]) - expected_notional) > tolerance:
        raise mismatch(
            "notional_exposure",
            f"saved={out['notional_exposure']!r} margin_x_leverage={expected_notional!r}",
        )
    return out
