#!/usr/bin/env python3
"""BTC 프로모션 리포트에 **시드 리스트와 시드별 결과를 기록**한다 (CLAUDE.md 게이트 3번째 요건).

XRP판(`annotate_xrp_reports_with_seed_robustness_20260903.py`)의 BTC 대응.
원본 수치는 건드리지 않고 `seed_robustness` 블록만 추가한다.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/research"
SIG = R / "btc_evidence_signals_seed_robustness_20260903.json"
REG = R / "btc_regime_grid_extension_and_seed_20260903.json"
REGIME_ART = ROOT / "tmp/btc_regime_s24k3_20260902/train_report.json"
LABELDIR = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901"


def log(m): print(f"[btc-annot] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    s = json.loads(SIG.read_text())
    block = {
        "gate": "CLAUDE.md Seed-Diversity Ensemble Promotion Gate",
        "n_seeds": s["n_seeds"], "seeds": s["seeds"],
        "seed_selection": s["seed_selection"],
        "splits_evaluated": ["VAL", "OOS"], "holdout_touched": False,
        "holdout_note": "HOLDOUT은 1회 소진됨 -- 시드별 재평가는 재노출이므로 하지 않았다",
        "all_oos_above_half": s["all_oos_above_half"],
        "by_signal": {k: {"per_seed": v["per_seed"], "VAL": v.get("VAL"), "OOS": v.get("OOS")}
                      for k, v in s["signals"].items() if "error" not in v},
        "source": SIG.name,
        "note": "원본은 7종 전부 4시드([20260829,141592,271828,577215])로 N>=5 미달이었다.",
    }
    n = 0
    for f in sorted(LABELDIR.glob("*_gridscreen_report.json")) + \
             sorted(LABELDIR.glob("*_tabpfn_report.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:                                       # noqa: BLE001
            continue
        r["seed_robustness"] = block
        f.write_text(json.dumps(r, ensure_ascii=False, indent=2))
        n += 1
    log(f"증거신호 리포트 {n}개에 시드 블록 기록 (신호 {len(block['by_signal'])}종)")
    for k, v in block["by_signal"].items():
        o = v.get("OOS") or {}
        log(f"  {k:<26} OOS {o.get('mean', float('nan')):.4f} ± {o.get('std', float('nan')):.4f} "
            f"전부>0.5:{'O' if o.get('all_above_half') else 'X'}")

    if REG.exists() and REGIME_ART.exists():
        g = json.loads(REG.read_text())
        r = json.loads(REGIME_ART.read_text())
        sr = g.get("seed_robustness") or {}
        r["seed_robustness"] = {
            "gate": "CLAUDE.md Seed-Diversity Ensemble Promotion Gate",
            "n_seeds": len(g["seeds"]), "seeds": g["seeds"],
            "seed_selection": "랜덤 추출(고정 간격 증가 아님)",
            "grid_extension_tested": {"scales": g["scales"], "debounces": g["debounces"],
                                      "prev_grid": g["prev_grid"]},
            "phase3b_top": g.get("top_phase3b"), "deployed": g.get("deployed"),
            "change_recommended": g.get("change_recommended"),
            "candidate_vs_deployed": {"wins": sr.get("wins"),
                                      "bal_acc_delta": sr.get("bal_acc_delta")},
            "eval_window": [str(g.get("never_read_from"))],
            "holdout_touched": False,
            "note": ("격자를 S->192 / K->12로 넓혀 재탐색했으나 후보(S192_K12)는 학습가능성이 "
                     "8pp 무너져(bal_acc 우위 0/8) 현행 S24_K3 유지. 배포본이 Phase3b OOS 평균도 "
                     "가장 높다(+0.2117 vs +0.1700)."),
        }
        REGIME_ART.write_text(json.dumps(r, ensure_ascii=False, indent=2))
        log(f"레짐 리포트 갱신 -> {REGIME_ART}")
    else:
        log(f"⚠️레짐 리포트 또는 감사 결과 없음 (REG={REG.exists()}, ART={REGIME_ART.exists()})")
    log(f"완료 ({round(time.time()-t0,1)}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
