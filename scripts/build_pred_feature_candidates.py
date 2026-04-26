#!/usr/bin/env python3
"""Build full/slim/none pred feature candidate sets from inspection JSON."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pred_* candidate sets from inspection result")
    p.add_argument("--inspection-json", required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--drop-near-constant", action="store_true", default=True)
    p.add_argument("--out-json", default="")
    return p.parse_args()


def _conf_of(pred_name: str) -> str:
    return pred_name.replace("pred_", "conf_", 1)


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.inspection_json):
        raise FileNotFoundError(args.inspection_json)

    with open(args.inspection_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    recommendation = payload.get("recommendation", {})
    rows = payload.get("feature_rows", [])

    pred_cols = list(config.get("pred_cols", []))
    conf_cols = list(config.get("conf_cols", []))
    near_const = set(recommendation.get("near_constant", [])) if args.drop_near_constant else set()

    ranked_preds = []
    for row in rows:
        pred = row.get("feature", "")
        if not pred or pred in near_const:
            continue
        ranked_preds.append(pred)

    slim_pred = ranked_preds[: max(1, int(args.top_k))]
    slim_conf = [_conf_of(p) for p in slim_pred if _conf_of(p) in conf_cols]

    full_pred = [p for p in pred_cols if p not in near_const]
    full_conf = [_conf_of(p) for p in full_pred if _conf_of(p) in conf_cols]

    candidates = {
        "full": {
            "pred_cols": full_pred,
            "conf_cols": full_conf,
        },
        "slim": {
            "pred_cols": slim_pred,
            "conf_cols": slim_conf,
        },
        "none": {
            "pred_cols": [],
            "conf_cols": [],
        },
    }

    summary = {
        "source_inspection_json": args.inspection_json,
        "top_k": int(args.top_k),
        "drop_near_constant": bool(args.drop_near_constant),
        "near_constant_removed": sorted(near_const),
        "candidates": candidates,
    }

    out_json = args.out_json.strip()
    if not out_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = f"data/ensemble/metrics/pred_feature_candidates_{ts}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("== pred feature candidates ==")
    print("full pred :", ", ".join(candidates["full"]["pred_cols"]))
    print("slim pred :", ", ".join(candidates["slim"]["pred_cols"]))
    print("none pred : (empty)")
    if near_const:
        print("removed near_constant:", ", ".join(sorted(near_const)))
    print(f"saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
