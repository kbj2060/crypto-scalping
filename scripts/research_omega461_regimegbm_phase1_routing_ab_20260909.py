"""Phase 1 — 동결 부모에 **라우팅만** 교체한 greedy_replay A/B (싼 falsification).

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md

무엇을 바꾸는가 — 정확히 한 줄
-----------------------------
전신 부모(ThreeHeadTabM ×2컴포넌트 ×3전문가)는 **완전히 동결**한다. 피쳐도 그대로다 —
부모의 102 `base_cols` 는 여전히 wide24 레짐 6컬럼을 포함한 채로 들어간다. 바꾸는 것은
`train_omega1_regime3_expert_direction_head_volpca_20260602._route_id(frame)` 가 argmax 하는
**확률 3컬럼의 출처** 하나뿐이다:

  A (baseline)  route = argmax(regime3_current_sensitive_wide24_{bull,bear,chop}_prob)
  B (candidate) route = argmax(regime3_s12k3_cut2509_{bull,bear,chop}_prob)

즉 "어느 전문가 서브넷이 이 봉에 답하는가"만 새 레짐이 정하고, 그 뒤의 direction/quality/exit
헤드·리스크 사이드카·라우터·배리어는 전부 전신 그대로다.

⚠️ 이것은 **진단이지 판정이 아니다.** 세 전문가는 wide24 라우팅으로 학습됐으므로, 라우팅만
바꾸면 각 전문가가 학습 때와 다른 봉 분포를 받는다(불일치가 의도적으로 도입된다). 그래서
계약은 "음성이어도 Phase 2 를 자동 기각하지 않는다"고 못박아 뒀다 — Phase 2 의 재학습이 바로
그 불일치를 없애는 단계이기 때문이다. 양성이면 강한 순풍 신호로만 쓴다.

준수: 신규 학습 없음(부모·사이드카 전부 동결 재스코어링). 라이브 파일 미변경.
      `greedy_replay`/`prepare_component` 는 재구현하지 않고 원본을 import 해서 쓴다.
"""
from __future__ import annotations

import json
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

WIDE24_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
# 후보 arm 은 argv 로 고른다:
#   (기본) s12k3   -- S12_K3 라벨 + GBM   (라벨과 모델급을 동시에 바꾼 arm)
#   balgbm         -- balancedish 라벨 유지 + GBM (모델급만 바꾼 arm, 라벨 교란 제거)
_ARM = sys.argv[1] if len(sys.argv) > 1 else "s12k3"
_ARMS = {
    "s12k3": ("data/ensemble/supervised/omega461_regimegbm_cut2509_20260909", "regime3_s12k3_cut2509_"),
    "balgbm": ("data/ensemble/supervised/omega461_balgbm_cut2509_20260909", "regime3_balgbm_cut2509_"),
}
if _ARM not in _ARMS:
    raise SystemExit(f"unknown arm {_ARM!r}; choose one of {sorted(_ARMS)}")
CUT_DIR = ROOT / _ARMS[_ARM][0]
CUT_PREFIX = _ARMS[_ARM][1]
CUT_ROUTE_COLS = [f"{CUT_PREFIX}{c}_prob" for c in ("bull", "bear", "chop")]

BASE_CSVS = {
    "2024": ROOT / "data/splits/year_oos/training_features_2024.csv",
    "2025": ROOT / "data/splits/year_oos/training_features_2025.csv",
    "2026_rebuilt": ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}
SPLITS = {
    "validation": ("2025-10-01 00:00:00", "2025-12-31 23:55:00"),
    "oos": ("2026-01-01 00:00:00", "2026-02-28 23:55:00"),
}
OUT = ROOT / f"tmp/omega461_regimegbm_rebuild_20260909/phase1_routing_ab_{_ARM}"


def load_frame(start: str, end: str) -> pd.DataFrame:
    parts = []
    for tag, base in BASE_CSVS.items():
        b = pd.read_csv(base, low_memory=False, parse_dates=["timestamp"])
        w = pd.read_csv(WIDE24_DIR / f"training_features_{tag}_regime3_current_sensitive_hmm_wide24.csv",
                        low_memory=False, parse_dates=["timestamp"])
        c = pd.read_csv(CUT_DIR / f"training_features_{tag}_{CUT_PREFIX}sidecar.csv",
                        low_memory=False, parse_dates=["timestamp"])
        m = b.merge(w, on="timestamp", how="inner").merge(c, on="timestamp", how="inner")
        parts.append(m)
    df = (pd.concat(parts, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


def route_from(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    v = frame[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(v).all():
        raise RuntimeError(f"non-finite route probabilities in {cols}")
    return np.argmax(v, axis=1).astype(np.int64)


def write_predictions(frame: pd.DataFrame, route: np.ndarray, out_dir: Path, device) -> None:
    """build_omega4_6_1_extended_parent_predictions_20260706.main() 의 컴포넌트 루프와 동일 로직,
    route 만 인자로 받는다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in retest.COMPONENTS.items():
        bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
        x = parent._base_input(frame, bundle["base_cols"])
        preds = {e: parent._predict_payload(bundle["models"][e], x, device=device) for e in hard.EXPERT_NAMES}
        direction = parent._routed(preds, route, "direction", 3)
        quality = parent._routed(preds, route, "quality", 3)
        oof = parent._prediction_output(frame, direction, quality,
                                        threshold=float(cfg["quality_threshold"]),
                                        prefix="omega1_regime3_expertdq_oof")
        src = oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_")
                                  for c in oof.columns})
        comp = out_dir / name
        comp.mkdir(parents=True, exist_ok=True)
        src.to_csv(comp / f"oos_predictions_{cfg['q_tag']}.csv", index=False)


def metrics(ledger: pd.DataFrame) -> dict:
    r = ledger["trade_return"].to_numpy()
    if not len(r):
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    curve = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    dd = curve / np.maximum(np.maximum.accumulate(curve), 1e-12) - 1.0
    return {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0),
            "trades": int(len(r)), "wr": float((r > 0).mean())}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[arm] {_ARM}  후보 접두사 {CUT_PREFIX}", flush=True)
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()
    results = {}

    for split, (start, end) in SPLITS.items():
        frame = load_frame(start, end)
        rA = route_from(frame, hard.ROUTE_COLS)
        rB = route_from(frame, CUT_ROUTE_COLS)
        agree = float((rA == rB).mean())
        shares = {n: [float((rA == i).mean()), float((rB == i).mean())]
                  for i, n in enumerate(hard.EXPERT_NAMES)}
        print(f"\n{'='*70}\n[{split}] {start} ~ {end}  {len(frame):,}봉", flush=True)
        print(f"  라우팅 일치율 {agree*100:.2f}%  전문가 비중 A/B "
              f"{ {k: [round(a,3), round(b,3)] for k,(a,b) in shares.items()} }", flush=True)

        split_res = {"bars": int(len(frame)), "route_agreement": agree, "expert_shares_A_B": shares}
        for arm, route in (("A_wide24_baseline", rA), (f"B_{_ARM}_candidate", rB)):
            d = OUT / split / arm
            write_predictions(frame, route, d, device)
            comps = {}
            for name, cfg in retest.COMPONENTS.items():
                comps[name] = prepare_component(frame, d / name / f"oos_predictions_{cfg['q_tag']}.csv",
                                                cfg, device)
            _, ledger = greedy_replay(frame, comps, fee=fee, slip=slip,
                                      cost_mult=retest.COST_MULT, device=device)
            ledger.to_csv(d / "ledger.csv", index=False)
            m = metrics(ledger)
            m["source_component"] = ledger["source_component"].value_counts().to_dict() if len(ledger) else {}
            split_res[arm] = m
            print(f"  {arm:22s} PnL {m['pnl']:+8.2f}%  MDD {m['mdd']:+7.2f}%  "
                  f"{m['trades']:3d}건  WR {m['wr']*100:5.1f}%  {m['source_component']}", flush=True)

        a, b = split_res["A_wide24_baseline"], split_res[f"B_{_ARM}_candidate"]
        split_res["delta"] = {"pnl_pp": b["pnl"] - a["pnl"], "mdd_pp": b["mdd"] - a["mdd"],
                             "trades": b["trades"] - a["trades"]}
        print(f"  Δ(B-A): PnL {split_res['delta']['pnl_pp']:+.2f}pp  "
              f"MDD {split_res['delta']['mdd_pp']:+.2f}pp  거래 {split_res['delta']['trades']:+d}", flush=True)
        results[split] = split_res

    (OUT / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/report.json", flush=True)
    print("\n⚠️ 이것은 진단이다 — 전문가들이 wide24 라우팅으로 학습됐으므로 라우팅만 바꾸면 "
          "학습 때와 다른 봉 분포를 받는다. 음성이어도 Phase 2 를 자동 기각하지 않는다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
