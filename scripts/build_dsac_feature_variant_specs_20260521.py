#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/llewyn/crypto-scalping")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INVENTORY_DIR = ROOT / "tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521"
FAMILY_DIR = INVENTORY_DIR / "families"
PCA_META_JSON = INVENTORY_DIR / "family_pca_meta.json"
ROUTER_STABLE48_JSON = ROOT / "tmp/causal_regen_20260516/alpha5_router5_full_candidate_search_20260521/rank_pruned_stable_top48_feature_list.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521"
LEGACY_CLEAN4_PREFIX = "clean_regime4_2024_unsup_v1_"
STATE24_CLEAN4_PREFIX = "clean_regime4_state24_sticky090_v2_"
GENERIC_REGIME_COLS = {
    "regime_bear",
    "regime_bull",
    "regime_chop",
    "regime_normal",
    "regime_persistence",
    "regime_trending",
    "regime_whipsaw",
}


def _set_env() -> None:
    os.environ["DSAC_ALPHA5_V2_STATE_ENABLE"] = "1"
    os.environ["DSAC_V2_MULTI_ACTION_ENABLE"] = "1"
    os.environ["DSAC_ALL_FEATURES_ENABLE"] = "1"
    os.environ["DSAC_EXTRA_PCA_ENABLE"] = "1"
    os.environ["DSAC_EXTRA_PCA_COMPONENTS"] = "32"


def _load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    cols = payload["features"] if isinstance(payload, dict) else payload
    out: list[str] = []
    seen: set[str] = set()
    for col in cols:
        c = str(col).strip()
        if c and c not in seen and c != "timestamp":
            out.append(c)
            seen.add(c)
    return out


def _write_spec(name: str, feature_cols: list[str], *, pca_enable: bool = False, pca_components: int = 0) -> None:
    payload = {
        "name": name,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "extra_pca_enable": bool(pca_enable),
        "extra_pca_components": int(pca_components),
    }
    (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_clean4_name(name: str) -> str:
    if name.startswith(LEGACY_CLEAN4_PREFIX):
        return STATE24_CLEAN4_PREFIX + name[len(LEGACY_CLEAN4_PREFIX) :]
    return name


def _normalize_dedupe(cols: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for col in cols:
        c = _normalize_clean4_name(str(col).strip())
        if c and c != "timestamp" and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def main() -> None:
    _set_env()
    from ensemble import train_rl_dsac_agent as dsac

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    duplicate = set(dsac.DSAC_DUPLICATE_SOURCE_COLS)
    explicit = set(dsac.DSAC_MARKET_STATE_COLS) | set(dsac.ALPHA5_ROUTER_STATE_COLS) | set(dsac.ALPHA5_CATBOOST_MAJOR_COLS)

    def sanitize(cols: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for col in cols:
            c = str(col).strip()
            c = _normalize_clean4_name(c)
            if not c or c in seen or c == "timestamp":
                continue
            if c in duplicate or c in explicit:
                continue
            out.append(c)
            seen.add(c)
        return out

    stable48 = sanitize(_load_feature_list(ROUTER_STABLE48_JSON))
    novel_all = sanitize(_load_feature_list(INVENTORY_DIR / "clean_candidate_list.json"))
    fam_clean4_state24 = sanitize(_load_feature_list(FAMILY_DIR / "clean_regime4_state24.json"))
    fam_pred4 = sanitize(_load_feature_list(FAMILY_DIR / "regime4_pred.json"))
    fam_ai = sanitize(_load_feature_list(FAMILY_DIR / "ai_family.json"))
    fam_m7 = sanitize(_load_feature_list(FAMILY_DIR / "m7.json"))
    fam_market = sanitize(_load_feature_list(FAMILY_DIR / "market_state.json"))

    pca_meta = json.loads(PCA_META_JSON.read_text(encoding="utf-8")) if PCA_META_JSON.exists() else {}
    pca_m7 = list(pca_meta.get("m7", {}).get("output_cols", []))
    pca_regime = (
        list(pca_meta.get("clean_regime4_state24", {}).get("output_cols", []))
        + list(pca_meta.get("regime4_pred", {}).get("output_cols", []))
    )

    current_tail_raw = [
        c
        for c in _normalize_dedupe(list(dsac.DSAC_ALL_FEATURE_COLS))
        if c not in GENERIC_REGIME_COLS and not c.startswith(LEGACY_CLEAN4_PREFIX)
    ]
    current_tail = sanitize(current_tail_raw + fam_clean4_state24)

    specs = {
        "compact54_no_tail": [],
        "current_tail111": current_tail,
        "stable48_tail": stable48,
        "novel_all_clean_tail": novel_all,
        "stable48_plus_ai_tail": sanitize(stable48 + fam_ai),
        "stable48_plus_clean4_pred_tail": sanitize(stable48 + fam_clean4_state24 + fam_pred4),
        "stable48_plus_regime_pred_tail": sanitize(stable48 + fam_pred4),
        "stable48_plus_clean4_state24_pred_tail": sanitize(stable48 + fam_clean4_state24 + fam_pred4),
        "stable48_plus_market_tail": sanitize(stable48 + fam_market),
        "stable48_plus_m7_tail": sanitize(stable48 + fam_m7),
        "stable48_plus_family_pca_m7": sanitize(stable48) + pca_m7,
        "stable48_plus_family_pca_regime": sanitize([c for c in stable48 if c not in fam_clean4_state24 and c not in fam_pred4]) + pca_regime,
    }

    _write_spec("compact54_no_tail", specs["compact54_no_tail"], pca_enable=False, pca_components=0)
    _write_spec("current_tail111", specs["current_tail111"], pca_enable=False, pca_components=0)
    _write_spec("current_pca32_all111", specs["current_tail111"], pca_enable=True, pca_components=32)
    _write_spec("stable48_tail", specs["stable48_tail"], pca_enable=False, pca_components=0)
    _write_spec("stable48_global_pca16", specs["stable48_tail"], pca_enable=True, pca_components=16)
    _write_spec("stable48_global_pca32", specs["stable48_tail"], pca_enable=True, pca_components=32)
    _write_spec("novel_all_clean_tail", specs["novel_all_clean_tail"], pca_enable=False, pca_components=0)
    _write_spec("novel_all_clean_global_pca32", specs["novel_all_clean_tail"], pca_enable=True, pca_components=32)
    _write_spec("stable48_plus_ai_tail", specs["stable48_plus_ai_tail"], pca_enable=False, pca_components=0)
    _write_spec("stable48_plus_clean4_pred_tail", specs["stable48_plus_clean4_pred_tail"], pca_enable=False, pca_components=0)
    _write_spec("stable48_plus_regime_pred_tail", specs["stable48_plus_regime_pred_tail"], pca_enable=False, pca_components=0)
    _write_spec("stable48_plus_clean4_state24_pred_tail", specs["stable48_plus_clean4_state24_pred_tail"], pca_enable=False, pca_components=0)
    _write_spec("stable48_plus_market_tail", specs["stable48_plus_market_tail"], pca_enable=False, pca_components=0)
    _write_spec("stable48_plus_m7_tail", specs["stable48_plus_m7_tail"], pca_enable=False, pca_components=0)
    _write_spec("stable48_plus_family_pca_m7", specs["stable48_plus_family_pca_m7"], pca_enable=False, pca_components=0)
    _write_spec("stable48_plus_family_pca_regime", specs["stable48_plus_family_pca_regime"], pca_enable=False, pca_components=0)

    summary = {
        "out_dir": str(OUT_DIR),
        "specs": {name: len(cols) for name, cols in specs.items()},
        "pca_available": bool(pca_meta),
        "current_tail_count": len(current_tail),
        "stable48_tail_count": len(stable48),
        "novel_all_clean_tail_count": len(novel_all),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
