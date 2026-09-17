#!/usr/bin/env python3
"""부모(zig075/h48qual) 추론을 CUDA 와 CPU 에서 돌려 **같은 답이 나오는지** 잰다.

왜: `omega4_6_2_source_parent_live.py` 가 CUDA 를 **하드 요구**한다
    (`must run on CUDA` RuntimeError 두 곳). 트레이딩 봇을 GPU 에서 떼려면 이 가드를
    풀어야 하는데, 가드의 존재 이유가 **수치 파리티**일 가능성이 높다.
    이 저장소 T2 계약: float 변형에 **부호가 뒤집히면 FAIL**, 변형 간 폭이 |추정치|의
    50% 초과면 FLAG.

무엇을: 배포 번들 2개 × 전문가 3 × 실제 VAL 행으로 direction/quality 확률을 양쪽에서 내고
  ① 최대 절대차 ② 방향 argmax 불일치 건수 ③ **q 게이트 최종 결정 불일치 건수**를 센다.
  결정이 한 건도 안 바뀌면 CPU 로 옮겨도 라이브 행동이 동일하다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402

FRAME = ROOT / "tmp/omega461_longwindow_20260917/features_with_regime_2022_2026_realfunding.parquet"
CONTRACT = (ROOT / "tmp/causal_regen_20260516"
            / "omega4_6_2_cap220_short_boost125_time_stop120h_20260630/runtime_contract.json")
N = 20_000


def main() -> int:
    assert torch.cuda.is_available(), "CUDA 가 없으면 이 비교 자체가 불가"
    df = pd.read_parquet(FRAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    val = df[(df.timestamp >= "2026-03-01") & (df.timestamp <= "2026-06-30")].reset_index(drop=True)
    val = val.iloc[:N]
    print(f"표본 {len(val):,}행 {val.timestamp.min()} ~ {val.timestamp.max()}", flush=True)
    expert_idx = tabm._route_probs(val).argmax(1)

    comps = json.loads(CONTRACT.read_text())["components"]
    worst = {"absdiff": 0.0, "dir_flip": 0, "gate_flip": 0, "n": 0}
    for alias, raw in comps.items():
        rep = json.loads(Path(raw["report"]).read_text())
        b = torch.load(Path(rep["risk_model"]["precomputed_prediction_dir"]) / "true_3head_tabm_bundle.pt",
                       map_location="cpu", weights_only=False)
        qth = float(raw["quality_threshold"])
        xr = tabm._base_input(val, list(b["base_cols"]))
        print(f"\n=== {alias}  (q={qth}) ===", flush=True)
        D = {d: np.zeros((len(val), 3)) for d in ("cuda", "cpu")}
        Q = {d: np.zeros((len(val), 3)) for d in ("cuda", "cpu")}
        for ei, ename in enumerate(("bull", "bear", "chop")):
            pay = dict(b["models"][ename])
            sel = expert_idx == ei
            if not sel.any():
                continue
            z = tabm._standardize_apply(xr[sel], dict(pay["scaler"]))
            for dev in ("cuda", "cpu"):
                m = tabm.ThreeHeadTabM(int(pay["n_features"]),
                                       cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(dev)
                m.load_state_dict(pay["state_dict"]); m.eval()
                with torch.no_grad():
                    o = m(torch.from_numpy(z).to(dev))
                    D[dev][sel] = torch.softmax(o["direction"], -1).mean(1).cpu().numpy()
                    Q[dev][sel] = torch.softmax(o["quality"], -1).mean(1).cpu().numpy()
                del m
            dd = float(np.abs(D["cuda"][sel] - D["cpu"][sel]).max())
            print(f"  {ename:5s} {int(sel.sum()):>6,}행 · direction 최대 절대차 {dd:.3e}", flush=True)

        def decide(dv):
            da = D[dv].argmax(1)
            qf = np.where(da > 0, Q[dv][np.arange(len(val)), da], Q[dv][:, 0])
            return da, np.where((da != 0) & (qf >= qth), da, 0)
        da_c, fin_c = decide("cuda"); da_p, fin_p = decide("cpu")
        ad = float(np.abs(D["cuda"] - D["cpu"]).max())
        aq = float(np.abs(Q["cuda"] - Q["cpu"]).max())
        df_ = int((da_c != da_p).sum()); gf = int((fin_c != fin_p).sum())
        print(f"  ⭐확률 최대 절대차  direction {ad:.3e} · quality {aq:.3e}")
        print(f"  ⭐방향 argmax 불일치 {df_}/{len(val):,}")
        print(f"  ⭐**게이트 최종결정 불일치 {gf}/{len(val):,}**  "
              f"(진입 CUDA {int((fin_c!=0).sum())} vs CPU {int((fin_p!=0).sum())})")
        worst["absdiff"] = max(worst["absdiff"], ad, aq)
        worst["dir_flip"] += df_; worst["gate_flip"] += gf; worst["n"] += len(val)

    print(f"\n{'='*70}")
    ok = worst["gate_flip"] == 0
    print(f"최대 절대차 {worst['absdiff']:.3e} · 방향 불일치 {worst['dir_flip']} · "
          f"**결정 불일치 {worst['gate_flip']}**")
    print("🟢CPU 로 옮겨도 라이브 결정이 동일하다 -- 가드를 풀어도 된다." if ok else
          "🔴결정이 갈린다 -- 가드는 이유가 있다. CPU 이전 보류.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
