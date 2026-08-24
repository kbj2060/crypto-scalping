#!/usr/bin/env python3
"""5-way 라벨로직 비교(zigzag/h48qual/dc/cusum/distreg) x 공유시드3개 결과 집계.

classification 4개는 report.json의 ranking_by_validation_pnl[0](TRAIN에서 고른 임계값
후보 중 VAL PnL 1위, 그 OOS를 확인하는 이 세션 표준 causal-safe 선택절차)을 기준으로 VAL/OOS
PnL과 부호를 비교. distreg는 동일 구조(best_by_validation_pnl)를 쓰되 fixed-horizon 홀드라는
방법론 차이가 있어 별도 섹션으로 분리."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/kbj20/crypto-scalping")
OUT_ROOT = ROOT / "tmp/causal_regen_20260516"
MODEL_ID = "omega4_3head_parent72_loose_entry_quality_20260620"

SEEDS = [133725056, 176495706, 796203462]
CLASSIFICATION_LABELS = ["zigzag", "h48qual", "dc", "cusum"]


def _load_classification_report(label: str, seed: int) -> dict:
    out_dir = OUT_ROOT / f"{MODEL_ID}_label5way_{label}_154feat_unified_single_model_seed{seed}_20260820"
    report_path = out_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(str(report_path))
    return json.loads(report_path.read_text())


def _load_distreg_report(seed: int) -> dict:
    out_dir = OUT_ROOT / f"{MODEL_ID}_label5way_distreg_154feat_unified_single_model_seed{seed}_20260820"
    report_path = out_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(str(report_path))
    return json.loads(report_path.read_text())


def main() -> None:
    print("=" * 100)
    print("classification 4-way (zigzag / h48qual / dc / cusum) x 3 seeds -- ranking_by_validation_pnl[0]")
    print("=" * 100)

    summary: dict[str, list[dict]] = {}
    for label in CLASSIFICATION_LABELS:
        rows = []
        for seed in SEEDS:
            rep = _load_classification_report(label, seed)
            best = rep["ranking_by_validation_pnl"][0]
            lqs = rep["label_quality_summary"]
            rows.append({
                "seed": seed,
                "variant": best["variant"],
                "val_pnl": best["validation_pnl"],
                "oos_pnl": best["oos_pnl"],
                "val_trades": best["validation_trades"],
                "oos_trades": best["oos_trades"],
                "val_wr": best["validation_wr"],
                "oos_wr": best["oos_wr"],
                "train_rows": lqs["train"]["rows"],
                "train_active_ratio": lqs["train"]["quality_active_ratio"],
                "oos_active_ratio": lqs["oos"]["quality_active_ratio"],
                "best_validation_loss": (rep.get("summaries") or {}).get("bull", {}).get("best_validation_loss"),
            })
        summary[label] = rows
        oos_signs = [1 if r["oos_pnl"] > 0 else (-1 if r["oos_pnl"] < 0 else 0) for r in rows]
        pos = sum(1 for s in oos_signs if s > 0)
        neg = sum(1 for s in oos_signs if s < 0)
        print(f"\n--- label={label} (direction_label_dir active_ratio train={rows[0]['train_active_ratio']:.4f} oos={rows[0]['oos_active_ratio']:.4f}) ---")
        for r in rows:
            print(f"  seed={r['seed']:>10} variant={r['variant']:>6} | VAL pnl={r['val_pnl']:>9.2f} trades={r['val_trades']:>4} wr={r['val_wr']:.3f} "
                  f"| OOS pnl={r['oos_pnl']:>9.2f} trades={r['oos_trades']:>4} wr={r['oos_wr']:.3f}")
        print(f"  => OOS sign: {pos}승{neg}패 (n=3, positive={pos}, negative={neg}, zero={3 - pos - neg})")

    print("\n" + "=" * 100)
    print("label간 OOS 부호일관성 요약")
    print("=" * 100)
    for label, rows in summary.items():
        oos_vals = [r["oos_pnl"] for r in rows]
        pos = sum(1 for v in oos_vals if v > 0)
        print(f"  {label:>10}: OOS pnl = {[f'{v:.1f}' for v in oos_vals]} -> {pos}/3 positive")

    print("\n" + "=" * 100)
    print("distributional regression (distreg) x 3 seeds -- best_by_validation_pnl (Gaussian NLL dist_head)")
    print("=" * 100)
    print("⚠️ fixed-horizon(48bar) 홀드 PnL(barrier/TP-SL 없음) -- 위 4개의 barrier 기반 TP/SL PnL과")
    print("   절대수치 직접비교 불가. cond_dir_acc(부호일치율)만 개념적으로 비교 가능.")
    distreg_rows = []
    for seed in SEEDS:
        rep = _load_distreg_report(seed)
        best = rep["best_by_validation_pnl"]
        distreg_rows.append({
            "seed": seed,
            "z_threshold": best["z_threshold"],
            "val_pnl_bps": best["validation"]["pnl_bps"],
            "oos_pnl_bps": best["oos"]["pnl_bps"],
            "val_trades": best["validation"]["trades"],
            "oos_trades": best["oos"]["trades"],
            "val_cond_dir_acc": best["validation"]["cond_dir_acc"],
            "oos_cond_dir_acc": best["oos"]["cond_dir_acc"],
            "val_long": best["validation"]["long_entries"],
            "val_short": best["validation"]["short_entries"],
            "oos_long": best["oos"]["long_entries"],
            "oos_short": best["oos"]["short_entries"],
            "best_validation_loss_nll": rep["best_validation_loss_nll"],
        })
        r = distreg_rows[-1]
        print(f"  seed={r['seed']:>10} z_th={r['z_threshold']:.3f} NLL={r['best_validation_loss_nll']:.3f} | "
              f"VAL pnl_bps={r['val_pnl_bps']:>10.0f} trades={r['val_trades']:>5}(L{r['val_long']}/S{r['val_short']}) cond_acc={r['val_cond_dir_acc']:.3f} | "
              f"OOS pnl_bps={r['oos_pnl_bps']:>10.0f} trades={r['oos_trades']:>5}(L{r['oos_long']}/S{r['oos_short']}) cond_acc={r['oos_cond_dir_acc']:.3f}")
    oos_pos = sum(1 for r in distreg_rows if r["oos_pnl_bps"] > 0)
    print(f"  => OOS sign: {oos_pos}/3 positive")

    out = {"classification": summary, "distreg": distreg_rows}
    out_path = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad/5way_comparison_summary.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
