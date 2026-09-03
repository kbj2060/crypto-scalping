#!/usr/bin/env python3
"""XRP `demarker_extreme` **H=1 vs H=2** TabPFN 대조 검증 (VAL+OOS만, HOLDOUT 미터치).

## 왜

2026-09-03 격자 경계 감사에서 배포 중인 `H=2`가 **격자 하단 경계**에서 뽑힌 값임이 드러났다.
격자에 H=1을 추가하니 선택이 바뀌고 lift가 단조 증가했다:

    H=1  TRAIN lift 1.411  VAL lift 1.971   <- 확장 후 선택
    H=2  TRAIN lift 1.128  VAL lift 1.562   <- 현재 배포 (HOLDOUT AUC 0.6759)

⚠️**그러나 lift ≠ AUC.** 격자 스크린은 lift로 고르지만 배포되는 건 TabPFN 모델이다.
lift가 높다고 모델 AUC가 높다는 보장이 없다 — 이 저장소에는 **라벨 K를 올리면 AUC와 PnL이
정반대로 움직인** 실증이 있다(`eth_label_k_increase_rejected_auc_vs_pnl_20260902`).

## 설계 — 대조군이 핵심

⭐**H=2를 같은 코드·같은 4시드로 다시 돌린다.** 기록된 0.6759와 비교하면 코드/시드/환경 차이가
섞여 비교가 성립하지 않는다. 두 호라이즌을 **한 실행 안에서** 같은 경로로 통과시킨다.

⚠️**라벨 난이도 대조 필수.** 서로 다른 라벨의 AUC를 그냥 비교하면 안 된다
(`feedback_cross_model_auc_comparison_requires_matched_label_difficulty`).
각 분할의 **base rate(hit 비율)** 를 같이 출력해 난이도가 맞는지 보이게 한다.

⚠️⚠️**HOLDOUT 미터치.** XRP demarker의 reserved holdout은 H=2의 0.6759로 **이미 1회 소진**됐다.
H=1로 그 창을 다시 보면 두 번째 모델선택 노출이 된다. 여기서는 VAL/OOS만 본다.
⇒ H=1이 이기더라도 **교체 근거는 VAL+OOS까지**이고, 홀드아웃 확인은 불가능하다는 점을 명시한다.

## 판정 (실행 전 고정)

  H=1이 **VAL과 OOS 둘 다에서** H=2를 시드분산(±2σ) 넘게 이겨야 교체를 검토한다.
  한쪽만 이기거나 시드분산 안이면 **현행(H=2) 유지** — 배포 중인 값을 바꿀 근거로 부족하다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_S = importlib.util.spec_from_file_location(
    "xrpdem", ROOT / "scripts/research_xrp_demarker_extreme_metalabel_tabpfn_20260903.py")
_d = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_d)

CAND_CSV = (ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"
            / "xrp_5m_evidence_signal_candidates_tier0.csv")
EXPECTED_ROWS = 272_490
OUT = ROOT / "data/research/xrp_demarker_h1_vs_h2_tabpfn_20260903.json"

K = 1.5                       # 두 호라이즌 모두 격자가 K=1.5를 골랐다
HORIZONS = [1, 2]
DEPLOYED_H = 2
RECORDED_H2_HOLDOUT_AUC = 0.6759   # 참고용 기록값 -- 이 실행에서 재측정하지 않는다


def log(m): print(f"[h1vs2] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    f = pd.read_csv(CAND_CSV)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True).dt.tz_localize(None)
    f = _d.add_missing_features(f)
    if abs(len(f) - EXPECTED_ROWS) > 200:
        raise RuntimeError(f"행수 {len(f):,} != XRP 기대치 {EXPECTED_ROWS:,} -- 다른 자산 데이터")
    log(f"XRP 프레임 {len(f):,}행 (자산 가드 통과)")
    log(f"시드 {_d.SEEDS}  |  K={K}  |  GAP={_d.CLUSTER_GAP}")
    log(f"⚠️HOLDOUT(>= {_d.HOLDOUT_START.date()}) 미터치 -- H=2의 {RECORDED_H2_HOLDOUT_AUC}로 이미 소진")

    feats = [c for c in _d.FEATURE_COLUMNS]
    rep = {"k": K, "gap": _d.CLUSTER_GAP, "seeds": list(_d.SEEDS),
           "holdout_touched": False, "recorded_h2_holdout_auc": RECORDED_H2_HOLDOUT_AUC,
           "deployed_horizon": DEPLOYED_H, "horizons": {}}

    for h in HORIZONS:
        log("")
        log("#" * 62)
        log(f"HORIZON = {h}" + ("  ⭐현재 배포" if h == DEPLOYED_H else "  (확장 격자 선택)"))
        log("#" * 62)
        fires, dstat = _d.build_final_fires(f, h, K, _d.CLUSTER_GAP)
        ts = pd.to_datetime(fires["timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        fires = fires.assign(timestamp=ts)
        cols = [c for c in feats if c in fires.columns]
        if len(cols) != len(feats):
            log(f"  ⚠️피쳐 누락 {len(feats)-len(cols)}개: {sorted(set(feats)-set(cols))[:6]}")

        tr = fires.loc[fires["timestamp"] < _d.VAL_START].reset_index(drop=True)
        va = fires.loc[(fires["timestamp"] >= _d.VAL_START)
                       & (fires["timestamp"] < _d.OOS_START)].reset_index(drop=True)
        oo = fires.loc[(fires["timestamp"] >= _d.OOS_START)
                       & (fires["timestamp"] < _d.HOLDOUT_START)].reset_index(drop=True)
        log(f"  fires {len(fires):,}  |  TRAIN {len(tr):,} / VAL {len(va):,} / OOS {len(oo):,}")
        log(f"  base rate(hit) TRAIN {tr['hit'].mean():.4f} / VAL {va['hit'].mean():.4f} "
            f"/ OOS {oo['hit'].mean():.4f}   ⬅ 라벨 난이도")
        if len(tr) < 200 or tr["hit"].nunique() < 2:
            log(f"  ❌TRAIN 부족({len(tr)}) 또는 단일 클래스 -- 판정 불가")
            rep["horizons"][str(h)] = {"error": f"insufficient train ({len(tr)})"}
            continue

        res = {"n_fires": int(len(fires)),
               "n": {"train": int(len(tr)), "val": int(len(va)), "oos": int(len(oo))},
               "base_rate": {"train": float(tr["hit"].mean()), "val": float(va["hit"].mean()),
                             "oos": float(oo["hit"].mean())},
               "dedup": dstat if isinstance(dstat, dict) else None}
        for tag, ev in (("val", va), ("oos", oo)):
            if len(ev) < 30 or ev["hit"].nunique() < 2:
                log(f"  ⚠️{tag.upper()} 부족({len(ev)}) -- 건너뜀")
                res[tag] = {"error": "insufficient eval"}
                continue
            res[tag] = _d.run_tabpfn_panel(tr, ev, cols, f"H{h}-{tag}")
            log(f"  ⇒ {tag.upper()} AUC {res[tag]['auc_mean']:.4f} ± {res[tag]['auc_std']:.4f} "
                f"(n={res[tag]['n_eval']})")
        rep["horizons"][str(h)] = res

    # ---------- 판정 ----------
    log("")
    log("=" * 62)
    log("판정 (사전 고정: VAL·OOS 둘 다에서 시드분산 ±2σ 넘게 이겨야 교체 검토)")
    log("=" * 62)
    a, b = rep["horizons"].get("1", {}), rep["horizons"].get(str(DEPLOYED_H), {})
    verdict = {}
    if "error" in a or "error" in b:
        log("  ❌한쪽 판정 불가")
        rep["verdict"] = {"decidable": False}
    else:
        log(f"{'분할':<6} {'H=1 AUC':>16} {'H=2 AUC':>16} {'Δ':>9} {'2σ문턱':>9}  판정")
        allwin = True
        for tag in ("val", "oos"):
            if "error" in a.get(tag, {}) or "error" in b.get(tag, {}):
                log(f"{tag.upper():<6} 판정 불가"); allwin = False; continue
            m1, s1 = a[tag]["auc_mean"], a[tag]["auc_std"]
            m2, s2 = b[tag]["auc_mean"], b[tag]["auc_std"]
            thr = 2.0 * float(np.hypot(s1, s2))
            win = (m1 - m2) > thr
            allwin &= win
            verdict[tag] = {"h1": m1, "h1_std": s1, "h2": m2, "h2_std": s2,
                            "delta": m1 - m2, "threshold_2sigma": thr, "h1_wins": bool(win)}
            log(f"{tag.upper():<6} {m1:>9.4f}±{s1:.4f} {m2:>9.4f}±{s2:.4f} "
                f"{m1-m2:>+9.4f} {thr:>9.4f}  {'✅H=1' if win else '❌부족'}")
        log("")
        log(f"  라벨 난이도(base rate) H=1 VAL {a['base_rate']['val']:.4f} / OOS {a['base_rate']['oos']:.4f}"
            f"  vs  H=2 VAL {b['base_rate']['val']:.4f} / OOS {b['base_rate']['oos']:.4f}")
        log("")
        log(f"⇒ {'⚠️**교체 검토 가능** (단 HOLDOUT 소진으로 최종 확인은 불가)' if allwin else '✅**현행 H=2 유지** -- 교체 근거 부족'}")
        rep["verdict"] = {"decidable": True, "h1_wins_both": bool(allwin), "by_split": verdict}

    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
