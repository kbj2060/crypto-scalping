#!/usr/bin/env python3
"""'라이브로 승격된 오메가4.6.1(h48qual+zig075 dual) 자체'가 시드에 강건한가를 검증 --
Seed-Diversity Ensemble Promotion Gate 적용 (CLAUDE.md, 다만 N=3이라 N>=5 정식기준은
미충족, 예비/방향성 체크로만 취급할 것).

이전 '일리아스1' N=5 검증([[eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818]])
은 exit_head pos_tp/pos_sl 버그를 수정한 후보(posfix/pinned102)를 재튜닝된 threshold(0.40/0.80)로
Baseline v1과 비교했다 -- 이건 다른 질문(그 후보가 seed-robust하게 baseline을 이기는가, 답:
아니오)에 답한 것이다. 이 스크립트는 그와 달리 **실제 라이브 코드/threshold/sidecar 그대로**
(sweep.COMPONENTS, threshold 0.50/0.75, 버그수정 이전 원본 코드)를 다른 시드로 재학습해
"이 정확한 조합이 재현 가능한가"를 묻는다 -- 원본 260620 시드는 재학습 없이 sweep.COMPONENTS를
그대로 재사용(이미 라이브 그 자체이므로), 신규 2개 시드(94046540/524707103, SystemRandom
샘플링)만 eth_live_promotion_seed_robustness_{h48qual,zig075}_seed_variant_20260819.py로
재학습했다.

risk sidecar는 3개 시드 전부 원본(260620) 것을 frozen 재사용 -- 일리아스1과 동일한 단순화
(시드별 전용 sidecar 재학습은 별도의 무거운 축, 범위 밖, 명시적 caveat)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

ev._EXPECTED_ZERO_COLS = set()  # 원본코드 기반 bundle -- 102 base_cols, cmamba/risk placeholder 자체가 없음(session 08-18 확인). 진짜 unexpected-missing만 나오면 여전히 걸러야 하므로 clear.


def _bundles_for(seed_label: str) -> dict:
    if seed_label == "seed260620_original":
        return {name: dict(cfg) for name, cfg in sweep.COMPONENTS.items()}
    out_suffix_map = {
        "h48qual": f"zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_livepromo_seedvariant_{seed_label}",
        "zig075": f"current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_livepromo_seedvariant_{seed_label}",
    }
    out = {}
    for name, cfg in sweep.COMPONENTS.items():
        cfg2 = dict(cfg)
        cfg2["bundle"] = ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_{out_suffix_map[name]}/true_3head_tabm_bundle.pt"
        out[name] = cfg2
    return out


SEED_LABELS = ["seed260620_original", "94046540", "524707103"]


def main() -> int:
    all_results: dict[str, dict] = {}
    for seed_label in SEED_LABELS:
        print(f"########## {seed_label} ##########", flush=True)
        bundles = _bundles_for(seed_label)
        for name, cfg in bundles.items():
            if not Path(cfg["bundle"]).exists():
                raise RuntimeError(f"{seed_label}/{name}: bundle not found at {cfg['bundle']} -- training not finished yet?")
        ev.BUNDLES = {
            name: {
                "bundle": cfg["bundle"], "q_tag": cfg["q_tag"], "threshold": cfg["quality_threshold"],
                "atr_window": cfg["atr_window"], "tp_mult": cfg["tp_mult"], "sl_mult": cfg["sl_mult"],
                "min_tp": cfg["min_tp"], "min_sl": cfg["min_sl"], "max_tp": cfg["max_tp"], "max_sl": cfg["max_sl"],
                "sidecar_pkl": cfg["sidecar_pkl"], "exit_threshold": sweep.BASELINE_EXIT_THRESHOLD,
            }
            for name, cfg in bundles.items()
        }
        ev.OUT_DIR = ROOT / f"tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_{seed_label}"
        ev.main()
        report = json.loads((ev.OUT_DIR / "report.json").read_text(encoding="utf-8"))
        windows = {w: {"no_gate": d["no_gate"], "with_gate": d["with_gate"]} for w, d in report["windows"].items()}
        all_results[seed_label] = windows
        for w, wd in windows.items():
            wg = wd["with_gate"]
            print(f"{seed_label:20} {w:8} with_gate pnl={wg['pnl']:8.2f}% mdd={wg['mdd']:8.2f}% trades={wg['trades']:3d}", flush=True)

    print()
    print("=== 3-시드 부호일치 요약 (with_gate PnL, %) ===")
    header = f"{'window':10}" + "".join(f"{s:>20}" for s in SEED_LABELS) + "  sign_consistent"
    print(header)
    sign_flip_windows = []
    for w in next(iter(all_results.values())).keys():
        pnls = [all_results[s][w]["with_gate"]["pnl"] for s in SEED_LABELS]
        signs = {p >= 0 for p in pnls}
        consistent = len(signs) == 1
        if not consistent:
            sign_flip_windows.append(w)
        row = f"{w:10}" + "".join(f"{p:20.2f}" for p in pnls) + f"  {'YES' if consistent else 'NO -- SIGN FLIP'}"
        print(row, flush=True)

    print()
    print(f"sign_flip_windows={sign_flip_windows}")
    print("NOTE: N=3 -- CLAUDE.md Seed-Diversity Ensemble Promotion Gate는 N>=5 요구, "
          "이 결과는 예비/방향성 확인이지 정식 promotion 재확인이 아님.", flush=True)

    out_path = ROOT / "tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"windows": all_results, "sign_flip_windows": sign_flip_windows, "n_seeds": len(SEED_LABELS)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"results={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
