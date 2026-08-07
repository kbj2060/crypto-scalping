"""Deterministic HexaPulse overlay for an existing parent position stream."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OVERLAY_ID = "eth_v4_hexa_pulse_overlay_v1_20260719"


@dataclass(frozen=True)
class HexaPulseOverlayConfig:
    hostile_entry_threshold: float = -0.15
    opposition_exit_threshold: float = -0.15
    toxicity_block_threshold: float = 0.80
    tail_risk_block_threshold: float = 0.45


@dataclass(frozen=True)
class HexaPulseOverlayDecision:
    position: int
    action: str
    reason: str


def decide_overlay(
    *,
    parent_position: int,
    overlay_position: int,
    score: float,
    toxicity: float,
    tail_risk: float,
    available: bool,
    config: HexaPulseOverlayConfig = HexaPulseOverlayConfig(),
) -> HexaPulseOverlayDecision:
    """Allow, delay, or exit a parent position without inventing an opposite direction."""
    if parent_position not in (-1, 0, 1):
        raise ValueError(f"invalid parent_position: {parent_position}")
    if overlay_position not in (-1, 0, 1):
        raise ValueError(f"invalid overlay_position: {overlay_position}")

    if not available or not np.isfinite(score):
        return HexaPulseOverlayDecision(
            0,
            "EXIT" if overlay_position else "BLOCK",
            "FORMULA_DATA_INVALID",
        )
    if parent_position == 0:
        return HexaPulseOverlayDecision(
            0,
            "EXIT" if overlay_position else "CASH",
            "PARENT_CASH",
        )
    if overlay_position and overlay_position != parent_position:
        return HexaPulseOverlayDecision(0, "EXIT", "PARENT_DIRECTION_CHANGED")

    if toxicity >= config.toxicity_block_threshold:
        return HexaPulseOverlayDecision(
            0,
            "EXIT" if overlay_position else "BLOCK",
            "TOXICITY_BLOCK",
        )
    if tail_risk >= config.tail_risk_block_threshold:
        return HexaPulseOverlayDecision(
            0,
            "EXIT" if overlay_position else "BLOCK",
            "TAIL_RISK_BLOCK",
        )

    aligned_score = float(parent_position) * float(score)
    if overlay_position == 0:
        if aligned_score <= config.hostile_entry_threshold:
            return HexaPulseOverlayDecision(0, "DELAY", "HEXA_HOSTILE_TO_PARENT_ENTRY")
        return HexaPulseOverlayDecision(parent_position, "ENTER", "PARENT_ENTRY_ALLOWED")

    if aligned_score <= config.opposition_exit_threshold:
        return HexaPulseOverlayDecision(0, "EXIT", "HEXA_OPPOSITION_EXIT")
    return HexaPulseOverlayDecision(parent_position, "HOLD", "PARENT_POSITION_ALLOWED")
