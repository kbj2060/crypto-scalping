#!/usr/bin/env python3
"""Zeus Baseline v3 **섀도우 아티팩트 내보내기** (2026-09-18).

연구 스크립트는 학습 후 **예측값만 캐시하고 가중치를 버린다**. 섀도우는 새 봉에 추론해야
하므로 가중치·스케일러·열 계약을 파일로 남겨야 한다. 이 스크립트가 그것만 한다.

사양은 `docs/zeus/README.md §2`(Baseline v3) 동결본을 따른다:
  레짐 6열 제거(96열) · zigzag 양두 · 6시드 · TRAIN 2022-01-01~2026-05-31(purge 30일)

⭐**파리티 자체점검이 이 파일의 핵심이다.** 저장한 아티팩트로 다시 추론해 연구 캐시
(`stageP_probs_noreg_purge30.npz` 의 `SHADOW|N1s*`)와 **1e-6 이내 일치**를 확인한다.
일치하지 않으면 섀도우 원장이 우리가 평가한 모델의 것이 아니므로, 실패 시 저장하지 않는다.
"""
from __future__ import annotations
import hashlib, json, os, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(os.environ.get("ZEUS_ROOT") or Path.home() / "crypto-scalping")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT / "trading_bot_modules"))
import train_eval_omega1_2_tabm_3head_20260603 as tabm            # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight      # noqa: E402
from omega4_6_2_source_parent_live import CURRENT_PREFIX          # noqa: E402

SEEDS = [613042, 27851, 904377, 155690, 488213, 178618]
TRAIN0, TRAIN1 = "2022-01-01", "2026-05-31"      # purge 30일 적용 후 실효 끝
TEST0, TEST1 = "2026-07-01", "2026-09-30"
OUT = ROOT / "data/live/zeus_v3_shadow_20260918"
SPEC = {"tp": 0.015, "sl": 0.007, "rollq_window": 1000, "rollq_q": 0.983,
        "slots": 1, "mingap": 0, "sizing": "fixed_live__half_kelly_logged",
        "cost_bp_assumed": {"maker0": 0.0, "realistic": 1.02, "stress_peg": 5.52}}


def log(*a): print(*a, flush=True)


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, base_cols = E.load()
    drop = [c for c in base_cols if c.startswith(CURRENT_PREFIX)]
    assert len(drop) == 6, f"레짐 6열이 아니라 {len(drop)}열"
    base_cols = [c for c in base_cols if c not in drop]
    log(f"입력 base {len(base_cols)}열 + POS {len(tabm.POS_COLS)}열 = {len(base_cols)+len(tabm.POS_COLS)}열")

    tr = df[(df.timestamp >= TRAIN0) & (df.timestamp <= TRAIN1 + " 23:59:59")].reset_index(drop=True)
    te = df[(df.timestamp >= TEST0) & (df.timestamp <= TEST1 + " 23:59:59")].reset_index(drop=True)
    assert tr.timestamp.max() < te.timestamp.min(), "TRAIN 이 TEST 를 침범"
    gap = (te.timestamp.min() - tr.timestamp.max()).days
    assert gap >= 30, f"purge 간격 {gap}일 < 30"
    log(f"TRAIN {len(tr):,}봉 {TRAIN0}~{TRAIN1} · TEST {len(te):,}봉 · 공백 {gap}일")

    yt = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
    xs, scaler = tabm._standardize_fit(tabm._base_input(tr, base_cols))
    xv = tabm._standardize_apply(tabm._base_input(te, base_cols), scaler)
    n = len(tr); split = max(int(n * 0.85), min(n - 1, 512))
    bal = compute_sample_weight("balanced", y=yt).astype(np.float32)

    cache = dict(np.load(E.OUT / "stageP_probs_noreg_purge30.npz", allow_pickle=True))
    states, worst = [], 0.0
    for sd in SEEDS:
        m, meta = E.fit_expert(xs[:split], yt[:split], bal[:split],
                               xs[split:], yt[split:], bal[split:], seed=sd, ei=0, device=dev)
        D, Q = E.heads(m, xv, dev)
        ck = f"SHADOW|N1s{sd}"
        assert ck in cache, f"연구 캐시에 {ck} 없음 -- 같은 학습을 재현하지 못한다"
        z = dict(cache[ck].item())
        d = max(float(np.abs(D - z["D"]).max()), float(np.abs(Q - z["Q"]).max()))
        worst = max(worst, d)
        log(f"  시드 {sd}: best_epoch {meta['best_epoch']} · 캐시와 최대편차 {d:.3e}")
        states.append({k: v.detach().cpu() for k, v in m.state_dict().items()})

    # ⭐파리티 -- 실패하면 저장하지 않는다. CPU 는 float 합산 순서로 미세차가 날 수 있어 1e-5.
    assert worst < 1e-5, f"파리티 실패: 최대편차 {worst:.3e} -- 연구 평가와 다른 모델이다"
    log(f"⭐파리티 통과: 6시드 최대편차 {worst:.3e}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": states, "seeds": SEEDS, "cfg": vars(tabm.CFG),
                "scaler": scaler, "base_cols": base_cols, "pos_cols": list(tabm.POS_COLS),
                "input_dim": len(base_cols) + len(tabm.POS_COLS)}, OUT / "model.pt")
    meta = {"model_id": "zeus_v3_shadow_20260918", "spec": SPEC, "seeds": SEEDS,
            "train_range": f"{TRAIN0}~{TRAIN1}", "purge_days": 30,
            "frame": str(E.PARQUET), "frame_sha256": hashlib.sha256(
                E.PARQUET.read_bytes()).hexdigest() if E.PARQUET.stat().st_size < 3e9 else "skipped",
            "base_cols": base_cols, "n_base": len(base_cols),
            "parity_max_abs_dev": worst, "parity_ref": "stageP_probs_noreg_purge30.npz SHADOW|N1s*",
            "frozen_doc": "docs/zeus/README.md §2", "prereg": "docs/zeus/shadow_prereg_v3_20260918.md",
            "orders": "NONE -- 기록만 한다"}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"저장: {OUT}/model.pt · meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
