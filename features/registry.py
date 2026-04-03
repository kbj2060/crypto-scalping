from __future__ import annotations

from typing import Iterable, Mapping


# SevenModelEnsemble.predict_batch() currently emits these columns.
M7_GENERATED_COLS = [
    "m7_trend_xgb_dn",
    "m7_trend_xgb_fl",
    "m7_trend_xgb_up",
    "m7_mtl_dn",
    "m7_mtl_fl",
    "m7_mtl_up",
    "m7_quant_dn",
    "m7_quant_fl",
    "m7_quant_up",
    "m7_prob_dn",
    "m7_prob_fl",
    "m7_prob_up",
    "m7_direction",
    "m7_confidence",
    "m7_action",
    "m7_size",
    "m7_q10",
    "m7_q50",
    "m7_q90",
    "m7_qwidth",
    "m7_quality_pred",
    "m7_hold_pred",
    "m7_target_hold",
    "m7_entry_long_offset",
    "m7_entry_short_offset",
    "m7_entry_long_price",
    "m7_entry_short_price",
    "m7_tp_offset",
    "m7_sl_offset",
    "m7_tp_price",
    "m7_sl_price",
    "m7_gmm_cluster",
    "m7_gmm_conf",
    "m7_gmm_vol_rank",
    "m7_hdb_label",
    "m7_hdb_prob",
    "m7_iso_pred",
    "m7_iso_score",
    "m7_vae_error",
    "m7_vae_threshold",
    "m7_iso_anom",
    "m7_vae_anom",
    "m7_gate_block",
    "m7_expected_ret",
    "m7_tail_risk",
    "m7_composite_score",
]


# RL/Live ingest paths do not need these columns today.
M7_DEPRECATED_COLS = [
    "m7_prob_dn",
    "m7_prob_fl",
    "m7_prob_up",
    "m7_direction",
    "m7_hdb_label",
    "m7_hdb_prob",
    "m7_vae_threshold",
]


# DSAC state-builder core M7 inputs for RL training/inference.
M7_RL_CORE_COLS = [
    "m7_trend_xgb_dn",
    "m7_trend_xgb_fl",
    "m7_trend_xgb_up",
    "m7_quality_pred",
    "m7_hold_pred",
    "m7_q10",
    "m7_q50",
    "m7_q90",
    "m7_qwidth",
    "m7_gmm_cluster",
    "m7_gmm_conf",
    "m7_gmm_vol_rank",
    "m7_iso_score",
    "m7_iso_anom",
    "m7_vae_error",
    "m7_vae_anom",
    "m7_entry_long_offset",
    "m7_entry_short_offset",
    "m7_tp_offset",
    "m7_sl_offset",
]


# Keep-set extras that are not consumed directly by DSAC state today.
M7_RL_AUX_COLS = [
    "m7_target_hold",
    "m7_entry_long_price",
    "m7_entry_short_price",
    "m7_tp_price",
    "m7_sl_price",
    "m7_hdb_label",
    "m7_hdb_prob",
    "m7_iso_pred",
    "m7_vae_threshold",
    "m7_prob_dn",
    "m7_prob_fl",
    "m7_prob_up",
]


M7_RL_KEEP_COLS = sorted(set(M7_RL_CORE_COLS) | set(M7_RL_AUX_COLS))


# Live strict guard columns (in addition to prob_* aliases handled separately).
M7_LIVE_STRICT_COLS = [
    "m7_quality_pred",
    "m7_hold_pred",
    "m7_q10",
    "m7_q50",
    "m7_q90",
    "m7_qwidth",
    "m7_gmm_cluster",
    "m7_gmm_conf",
    "m7_gmm_vol_rank",
    "m7_iso_score",
    "m7_iso_anom",
    "m7_vae_error",
    "m7_vae_threshold",
    "m7_vae_anom",
    "m7_hdb_label",
    "m7_hdb_prob",
]


M7_PROB_ALIASES: dict[str, tuple[str, ...]] = {
    "m7_prob_dn": ("m7_prob_dn", "m7_trend_xgb_dn", "prob_dn", "trend_dn_prob"),
    "m7_prob_fl": ("m7_prob_fl", "m7_trend_xgb_fl", "prob_flat", "trend_flat_prob"),
    "m7_prob_up": ("m7_prob_up", "m7_trend_xgb_up", "prob_up", "trend_up_prob"),
}


def get_m7_columns(profile: str, *, include_entry_price: bool = False) -> set[str]:
    if profile == "rl_core":
        cols = set(M7_RL_CORE_COLS)
    elif profile == "rl_keep":
        cols = set(M7_RL_KEEP_COLS)
    elif profile == "live_strict":
        cols = set(M7_LIVE_STRICT_COLS) | set(M7_PROB_ALIASES.keys())
    elif profile == "generated":
        cols = set(M7_GENERATED_COLS)
    elif profile == "deprecated":
        cols = set(M7_DEPRECATED_COLS)
    else:
        raise ValueError(f"unknown m7 profile: {profile}")

    if not include_entry_price:
        cols.discard("m7_entry_long_price")
        cols.discard("m7_entry_short_price")
    return cols


def find_missing_columns(
    columns: Iterable[str],
    required: Iterable[str],
    *,
    aliases: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    available = set(columns)
    missing: list[str] = []
    alias_map = aliases or {}
    for key in required:
        candidates = alias_map.get(key)
        if candidates is None:
            if key not in available:
                missing.append(key)
            continue
        if not any(candidate in available for candidate in candidates):
            missing.append(key)
    return missing

