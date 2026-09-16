#!/usr/bin/env python3
"""E-벤치마크 — **배포 중인 zig075 부모 자신**을 같은 VAL 에 돌린다.

빠져 있던 결정적 비교다. "장기창으로 다시 만든 부모가 더 나은가"는 내 두 판끼리 비교해서는
답이 안 나온다. 기준선은 **지금 돌고 있는 그 아티팩트**다.

공정성: 배포 부모의 TRAIN 은 2025-01~09, VAL 2025-10~12, OOS 2026-01~02 이므로
VAL 2026-03~06 은 그에게도 **완전 표본외**다. 라우팅·게이트·k평균 전부 라이브와 동일하게
적용하고(`_route_expert` hard routing · q=0.75 · softmax 후 k 평균), 스케일러는 번들이
전문가별로 들고 있는 것을 쓴다 -- 내 스케일러를 쓰면 다른 모델이 된다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, base_cols = E.load()
    vm = (df.timestamp >= E.VAL[0]) & (df.timestamp <= E.VAL[1] + " 23:59:59")
    val = df[vm].reset_index(drop=True)
    yv = pd.to_numeric(val["zigzag_action"]).to_numpy(np.int64)
    xv_raw = tabm._base_input(val, base_cols)
    expert_idx = tabm._route_probs(val).argmax(1)
    vdays = val.timestamp.dt.floor("D").to_numpy()

    b = torch.load(E.BUNDLE, map_location="cpu", weights_only=False)
    D = np.zeros((len(val), 3)); Q = np.zeros((len(val), 3))
    for ei, ename in enumerate(("bull", "bear", "chop")):
        pay = dict(b["models"][ename])
        cols = list(pay["input_columns"])
        assert list(xv_raw.columns) == cols, "번들 입력 열 순서 불일치"
        m = tabm.ThreeHeadTabM(int(pay["n_features"]), cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
        m.load_state_dict(pay["state_dict"]); m.eval()
        sel = expert_idx == ei
        if not sel.any():
            continue
        z = tabm._standardize_apply(xv_raw[sel], dict(pay["scaler"]))
        d, q = E.heads(m, z, device)
        D[sel], Q[sel] = d, q
        print(f"  {ename}: {int(sel.sum()):,}봉", flush=True)

    dir_action = D.argmax(1)
    q_for = np.where(dir_action > 0, Q[np.arange(len(Q)), dir_action], Q[:, 0])
    final = np.where((dir_action != 0) & (q_for >= E.Q_THRESH), dir_action, 0)
    side = np.where(final == 1, 1.0, np.where(final == 2, -1.0, 0.0))
    act = (final != 0) & (yv != 0)
    out = {"model": "deployed zig075 (배포본)", "val": list(E.VAL), "rows": int(len(val)),
           "val_bacc_pre_gate": E.bacc(yv, dir_action),
           "val_bacc_post_gate": E.bacc(yv, final),
           "gate_pass_rate": float((final != 0).mean()),
           "side_acc_post_gate": float((final[act] == yv[act]).mean()) if act.any() else float("nan"),
           "econ": {}}
    for hname in E.HORIZONS:
        f = val[f"fwd_{hname}_bp"].to_numpy(np.float64)
        ok = (side != 0) & np.isfinite(f)
        pnl = side[ok] * f[ok]
        lo, hi, nd = E.block_ci(pnl, vdays[ok]) if ok.sum() > 100 else (np.nan, np.nan, 0)
        out["econ"][hname] = {"n": int(ok.sum()), "indep_days": nd,
                              "gross_bp": float(pnl.mean()) if ok.any() else float("nan"), "ci95": [lo, hi]}
    print("\n=== 배포 zig075 부모, VAL 2026-03-01~06-30 ===")
    print(f"  bacc {out['val_bacc_pre_gate']:.4f} → 게이트후 {out['val_bacc_post_gate']:.4f} · "
          f"통과율 {out['gate_pass_rate']:.3f} · 방향정확 {out['side_acc_post_gate']:.4f}")
    for hname, e in out["econ"].items():
        print(f"  {hname}: 총 {e['gross_bp']:+.2f}bp CI[{e['ci95'][0]:+.2f},{e['ci95'][1]:+.2f}] "
              f"· n {e['n']:,} · 독립일 {e['indep_days']}")
    (E.OUT / "stageE_deployed_benchmark.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
