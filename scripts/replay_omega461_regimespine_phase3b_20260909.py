"""Phase 3-b — 새 부모 + 새 사이드카로 greedy_replay 짝지은 판정.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md

Phase 2 / 3-a 와 무엇이 다른가
-----------------------------
  Phase 2  : 컴포넌트 **개별**, 사이징 **고정**(BASE_TEMPLATE notional 0.45 / leverage 2.0)
  Phase 3-a: 컴포넌트 **개별**, 사이징 **사이드카**(봉별 margin/leverage)
  Phase 3-b: **h48qual > zig075 우선순위 라우터가 결합한 단일계좌** + 사이드카 사이징
             → 실제 결정 경로. 이것이 이 라인의 판정 근거다.

`greedy_replay` / `prepare_component` 는 `replay_omega4_6_1_greedy_router_20260706.py` 원본을
import 해서 쓴다(재구현 금지). 부모 예측은 이 스크립트가 리플레이 프레임 위에서 직접 생성한다 —
`prepare_component` 가 `pred["timestamp"].equals(frame["timestamp"])` 를 요구하기 때문이다.

arm 별로 바뀌는 것은 딱 두 가지:
  · 부모 번들 / 사이드카 (Phase 2·3-a 산출물)
  · 라우팅 확률 컬럼 출처 (balnobb: regime3_balnobb_cut2509_* / wide24: regime3_current_sensitive_wide24_*)
번들의 `base_cols` 가 이미 해당 접두사를 담고 있으므로 피쳐 입력은 자동으로 맞는다.

판정: 계약의 `validation_pass_criteria` 그대로 — VAL 에서 대조군 대비 PnL·MDD 둘 다 비악화,
N=5 시드 짝지은 비교. OOS 는 보고만 하고 선택 근거로 쓰지 않는다(`validation_only`).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import greedy_replay, prepare_component  # noqa: E402

BASE = ROOT / "tmp/causal_regen_20260516"
PARENT_STEM = "omega4_3head_parent72_loose_entry_quality_20260620_regimespine_{arm}_{comp}_s{seed}_20260909"
SIDECAR_STEM = "omega4_2_trade_risk_sidecar_20260622_regimespine_{arm}_{comp}_s{seed}_20260909"
WIDE24_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
BALNOBB_DIR = ROOT / "data/ensemble/supervised/omega461_balnobb_cut2509_20260909"
ROUTE = {"balnobb": [f"regime3_balnobb_cut2509_{c}_prob" for c in ("bull", "bear", "chop")],
         "wide24": list(hard.ROUTE_COLS)}
ARMS = ["balnobb", "wide24"]
COMPS = {"zig075": ("q075", 0.75), "h48qual": ("q050", 0.50)}
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]
BASE_CSVS = {"2024": "training_features_2024.csv", "2025": "training_features_2025.csv",
             "2026_rebuilt": "training_features_2026_rebuilt.csv"}
SPLITS = {"validation": ("2025-10-01 00:00:00", "2025-12-31 23:55:00"),
          "oos": ("2026-01-01 00:00:00", "2026-02-28 23:55:00")}
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/phase3b"


def load_frame(start: str, end: str) -> pd.DataFrame:
    parts = []
    for tag, fn in BASE_CSVS.items():
        b = pd.read_csv(ROOT / "data/splits/year_oos" / fn, low_memory=False, parse_dates=["timestamp"])
        w = pd.read_csv(WIDE24_DIR / f"training_features_{tag}_regime3_current_sensitive_hmm_wide24.csv",
                        low_memory=False, parse_dates=["timestamp"])
        n = pd.read_csv(BALNOBB_DIR / f"training_features_{tag}_regime3_balnobb_cut2509_sidecar.csv",
                        low_memory=False, parse_dates=["timestamp"])
        parts.append(b.merge(w, on="timestamp", how="inner").merge(n, on="timestamp", how="inner"))
    df = (pd.concat(parts, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


def metrics(ledger: pd.DataFrame) -> dict:
    r = ledger["trade_return"].to_numpy() if len(ledger) else np.array([])
    if not len(r):
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0, "source_component": {}}
    curve = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    dd = curve / np.maximum(np.maximum.accumulate(curve), 1e-12) - 1.0
    return {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0),
            "trades": int(len(r)), "wr": float((r > 0).mean()),
            "avg_notional": float(pd.to_numeric(ledger.get("notional_exposure", pd.Series(dtype=float)),
                                                errors="coerce").mean()) if "notional_exposure" in ledger else None,
            "source_component": ledger["source_component"].value_counts().to_dict()}


def run_one(frame: pd.DataFrame, arm: str, seed: int, device) -> dict | None:
    route = np.argmax(frame[ROUTE[arm]].to_numpy(np.float64), axis=1).astype(np.int64)
    comps = {}
    for comp, (qtag, qthr) in COMPS.items():
        pdir = BASE / PARENT_STEM.format(arm=arm, comp=comp, seed=seed)
        sdir = BASE / SIDECAR_STEM.format(arm=arm, comp=comp, seed=seed)
        bp, sp = pdir / "true_3head_tabm_bundle.pt", sdir / "risk_sidecar.pkl"
        if not bp.exists() or not sp.exists():
            return None
        bundle = torch.load(bp, map_location="cpu", weights_only=False)
        x = parent._base_input(frame, bundle["base_cols"])
        preds = {e: parent._predict_payload(bundle["models"][e], x, device=device) for e in hard.EXPERT_NAMES}
        oof = parent._prediction_output(frame,
                                        parent._routed(preds, route, "direction", 3),
                                        parent._routed(preds, route, "quality", 3),
                                        threshold=float(qthr), prefix="omega1_regime3_expertdq_oof")
        src = oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_")
                                  for c in oof.columns})
        tmp = OUT / "preds" / f"{arm}_{comp}_s{seed}"
        tmp.mkdir(parents=True, exist_ok=True)
        pcsv = tmp / f"oos_predictions_{qtag}.csv"
        src.to_csv(pcsv, index=False)
        cfg = dict(retest.COMPONENTS[comp])
        cfg["bundle"], cfg["sidecar_pkl"] = bp, sp
        comps[comp] = prepare_component(frame, pcsv, cfg, device)
    fee, slip = omega._load_fee_slip()
    _, ledger = greedy_replay(frame, comps, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    return metrics(ledger)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = parent._device("cpu")
    rep = {"splits": {}}
    for split, (s, e) in SPLITS.items():
        frame = load_frame(s, e)
        print(f"\n{'='*96}\n[{split}] {s} ~ {e}  {len(frame):,}봉", flush=True)
        res = {}
        for arm in ARMS:
            for seed in SEEDS:
                m = run_one(frame, arm, seed, device)
                res[f"{arm}|{seed}"] = m
                if m is None:
                    print(f"  {arm:8s} s{seed}  (아티팩트 없음 — 스킵)", flush=True)
                else:
                    print(f"  {arm:8s} s{seed}  PnL {m['pnl']:+8.2f}%  MDD {m['mdd']:+7.2f}%  "
                          f"{m['trades']:3d}건  WR {m['wr']*100:5.1f}%  {m['source_component']}", flush=True)
        rep["splits"][split] = res

    print(f"\n{'='*96}\n[짝지은 비교] balnobb − wide24", flush=True)
    verdict = {}
    for split in SPLITS:
        r = rep["splits"][split]
        dp, dm, pairs = [], [], []
        for seed in SEEDS:
            a, b = r.get(f"balnobb|{seed}"), r.get(f"wide24|{seed}")
            if not a or not b:
                continue
            dp.append(a["pnl"] - b["pnl"]); dm.append(a["mdd"] - b["mdd"])
            pairs.append({"seed": seed, "d_pnl": dp[-1], "d_mdd": dm[-1],
                          "balnobb": a, "wide24": b})
        if not pairs:
            continue
        n = len(pairs)
        both = sum(1 for p, q in zip(dp, dm) if p > 0 and q > 0)
        verdict[split] = {"n": n, "pnl_better": sum(1 for x in dp if x > 0),
                          "mdd_better": sum(1 for x in dm if x > 0), "both_better": both,
                          "median_d_pnl": statistics.median(dp), "median_d_mdd": statistics.median(dm),
                          "pairs": pairs}
        tag = "  ← 판정 기준" if split == "validation" else "  (보고만, 선택 근거 아님)"
        print(f"\n  [{split}] 짝 {n}개{tag}", flush=True)
        for p in pairs:
            print(f"    s{p['seed']}  ΔPnL {p['d_pnl']:+8.2f}pp  ΔMDD {p['d_mdd']:+7.2f}pp   "
                  f"(balnobb {p['balnobb']['pnl']:+7.2f}/{p['balnobb']['mdd']:+6.2f}  vs  "
                  f"wide24 {p['wide24']['pnl']:+7.2f}/{p['wide24']['mdd']:+6.2f})", flush=True)
        print(f"    PnL 개선 {verdict[split]['pnl_better']}/{n} · MDD 개선 {verdict[split]['mdd_better']}/{n}"
              f" · **둘 다 {both}/{n}**   중앙 ΔPnL {statistics.median(dp):+.2f}pp "
              f"ΔMDD {statistics.median(dm):+.2f}pp", flush=True)

    v = verdict.get("validation")
    if v:
        p1 = v["both_better"] == v["n"] == len(SEEDS)
        print(f"\n  P1 (VAL 에서 PnL·MDD 둘 다 비악화, {len(SEEDS)}/{len(SEEDS)}): "
              f"{'✅ 통과' if p1 else '❌ 실패'}", flush=True)
        rep["P1_validation_both_nonworse"] = bool(p1)
    rep["verdict"] = verdict
    (OUT / "phase3b_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/phase3b_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
