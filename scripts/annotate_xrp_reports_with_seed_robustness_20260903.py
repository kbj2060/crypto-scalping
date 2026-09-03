#!/usr/bin/env python3
"""프로모션 리포트에 **시드 리스트와 시드별 결과를 기록**한다 (CLAUDE.md 게이트 3번째 요건).

> Seed-Diversity Ensemble Promotion Gate: ... **시드 리스트는 프로모션 리포트에 기록해야 한다.**

2026-09-03 감사에서 XRP 증거신호 5종과 레짐 분류기의 프로모션 리포트에 시드 정보가
없었다(증거신호 3종은 애초에 단일 시드, 레짐은 단일 시드). 8시드 재측정 결과를 원 리포트에
소급 기록해 요건을 충족시킨다. 원본 수치는 건드리지 않고 `seed_robustness` 블록만 추가한다.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/research"
SRC = {
    "signals_3": R / "xrp_evidence_signals_seed_robustness_20260903.json",
    "signals_2": R / "xrp_demarker_kalman_seed_robustness_8seed_20260903.json",
    "regime":    R / "xrp_regime_s96k9_seed_robustness_20260903.json",
}
META = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_metalabel_report.json"
REGIME = ROOT / "tmp/xrp_regime_s96k9_20260903/train_report.json"


def log(m): print(f"[annot] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    s3 = json.loads(SRC["signals_3"].read_text())
    s2 = json.loads(SRC["signals_2"].read_text())
    rg = json.loads(SRC["regime"].read_text())

    # ---- 증거신호 메타라벨 리포트 ----
    m = json.loads(META.read_text())
    merged = {}
    for src, tag in ((s3, "xrp_evidence_signals_seed_robustness_20260903.json"),
                     (s2, "xrp_demarker_kalman_seed_robustness_8seed_20260903.json")):
        for name, v in src["signals"].items():
            merged[name] = {"per_seed": v["per_seed"], "VAL": v["VAL"], "OOS": v["OOS"],
                            "source": tag}
    m["seed_robustness"] = {
        "gate": "CLAUDE.md Seed-Diversity Ensemble Promotion Gate",
        "n_seeds": 8, "seeds": s3["seeds"],
        "seed_selection": "랜덤 추출(고정 간격 증가 아님)",
        "splits_evaluated": ["VAL", "OOS"],
        "holdout_touched": False,
        "holdout_note": "HOLDOUT은 1회 소진됨 -- 시드별 재평가는 재노출이므로 하지 않았다",
        "all_oos_above_half": bool(s3["all_oos_above_half"] and s2["all_oos_above_half"]),
        "by_signal": merged,
        "note": ("원본 리포트는 str_z/taker/orthogonal이 단일 시드(SEED=20260903), "
                 "demarker/kalman이 4시드였다. 2026-09-03 감사에서 8시드로 재측정해 기록한다."),
    }
    META.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    log(f"증거신호 리포트 갱신: {len(merged)}종 시드 기록 -> {META.name}")
    for n, v in merged.items():
        log(f"  {n:<26} OOS {v['OOS']['mean']:.4f} ± {v['OOS']['std']:.4f} "
            f"전부>0.5:{'O' if v['OOS']['all_above_half'] else 'X'}")

    # ---- 레짐 프로덕션 리포트 ----
    if REGIME.exists():
        r = json.loads(REGIME.read_text())
        r["seed_robustness"] = {
            "gate": "CLAUDE.md Seed-Diversity Ensemble Promotion Gate",
            "n_seeds": rg["n_seeds"], "seeds": rg["seeds"],
            "seed_selection": rg["seed_selection"],
            "comparison": f"{rg['deployed']} vs {rg['previous']}",
            "wins": rg["wins"], "bal_acc_delta": rg["bal_acc_delta"],
            "seed_robust": rg["seed_robust"],
            "eval_window": rg["eval"], "never_read_from": rg["never_read_from"],
            "holdout_touched": False,
            "note": ("프로덕션 모델은 단일 시드(SEED=7529)로 적합되지만, 채택 결정(S48_K6 -> S96_K9)의 "
                     "견고성을 8시드로 확인했다. bal_acc/플리커/게이트 전부 8/8 우위."),
        }
        REGIME.write_text(json.dumps(r, ensure_ascii=False, indent=2))
        log(f"레짐 리포트 갱신 -> {REGIME}")
        log(f"  bal_acc 우위 {rg['wins']['bal_acc']}/{rg['n_seeds']}  "
            f"차이 최소 {rg['bal_acc_delta']['min']:+.4f}")
    else:
        log(f"⚠️레짐 리포트 없음: {REGIME}")

    log(f"완료 ({round(time.time()-t0,1)}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
