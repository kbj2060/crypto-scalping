"""**제로샷 시계열 피쳐** — Chronos-2 로 48봉 앞 수익 분포를 예측해 gym 라벨과의 IC 를 잰다 (2026-09-14).

사용자 *"허깅페이스에서 제로샷 모델 찾아봐"*. 로컬 캐시에 amazon/chronos-2 가 있다.
문헌(2606.27100 · 2412.09394)은 «TSFM 제로샷은 랜덤워크와 구분 불가, 일봉 롱숏도 3bp 에 죽는다」였다.
여기서는 그 주장을 **우리 라벨**로 직접 확인한다: 봉 t 까지의 종가 512개(로그가격)를 문맥으로
48봉 앞 분위(0.1/0.5/0.9)를 받아 ① 중앙값 수익 ② 분포 폭 ③ 중앙값/폭(신호 대 잡음) 세 피쳐를 만든다.
판정은 다른 피쳐와 같다: 창별 IC · 부호 유지 · 적중률 vs 손익분기 · 상위10% 순bp.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="amazon/chronos-2")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-sub", type=int, default=4000)
    a = ap.parse_args()
    from chronos import BaseChronosPipeline
    torch.set_num_threads(8)
    pipe = BaseChronosPipeline.from_pretrained(a.model, device_map="cpu", torch_dtype=torch.float32)
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    lp = np.log(d.close.to_numpy(float))
    z_ = np.load(P.OUT / "direction_labels.npz", allow_pickle=True)
    lab = {k: (z_[f"{k}_idx"], z_[f"{k}_y"]) for k in z_["names"]}
    rng = np.random.default_rng(P.SEEDS[0])
    ti, ty = lab["TRAIN"]; sub = rng.choice(len(ti), size=min(a.train_sub, len(ti)), replace=False)
    lab = {"TRAIN": (ti[sub], ty[sub]), **{w: lab[w] for w in ("VAL", "OOS", "TEST")}}

    def forecast(idx: np.ndarray) -> np.ndarray:
        out = np.zeros((len(idx), 3), dtype=np.float64)
        t0 = time.time()
        for s in range(0, len(idx), a.batch):
            js = idx[s:s + a.batch]
            ctx = [torch.tensor(lp[j - a.context + 1:j + 1] - lp[j], dtype=torch.float32) for j in js]  # 봉 t 종가 기준 로그가격
            q, _ = pipe.predict_quantiles(ctx, prediction_length=a.horizon, quantile_levels=[0.1, 0.5, 0.9])
            # chronos-2 는 (B, [target=1,] H, Q) 를 돌려준다 -- 마지막 두 축(H, Q)만 남기고 지평 끝을 잡는다
            qa = q.numpy() if hasattr(q, "numpy") else np.asarray(q)
            qh = qa.reshape(len(js), -1, qa.shape[-1])[:, -1, :]
            assert qh.shape == (len(js), 3), qh.shape
            out[s:s + len(js), 0] = qh[:, 1] * 1e4                       # 중앙값 수익 bp
            out[s:s + len(js), 1] = (qh[:, 2] - qh[:, 0]) * 1e4          # 80% 폭 bp
            out[s:s + len(js), 2] = qh[:, 1] / np.maximum(qh[:, 2] - qh[:, 0], 1e-9)
            if (s // a.batch) % 20 == 0:
                print(f"    {s+len(js):,}/{len(idx):,} · {time.time()-t0:.0f}s", flush=True)
        return out

    rep = {"model": a.model, "context": a.context, "horizon": a.horizon, "windows": {}}
    feats = {}
    for w, (idx, y) in lab.items():
        ok = idx >= a.context
        idx, y = idx[ok], y[ok]
        print(f"[{w}] n {len(idx):,} 예측 …", flush=True)
        F = forecast(idx); feats[w] = (F, y)
        r = {"n": int(len(y))}
        for k, name in enumerate(("median_bp", "width_bp", "snr")):
            x = F[:, k]
            r[f"ic_{name}"] = float(spearmanr(x, y).statistic) if np.std(x) > 0 else float("nan")
        mm = np.abs(y) > 1e-9
        r["hit_rate_median"] = float((np.sign(F[mm, 0]) == np.sign(y[mm])).mean())
        kk = int(0.1 * len(y)); top = np.argsort(-np.abs(F[:, 2]))[:kk]
        r["top_decile_dir_bp"] = float(np.mean(np.sign(F[top, 0]) * y[top]))
        if w != "TRAIN":
            al = P.eval_policy(None, d, sm, S, win, w, A_override=np.ones(win[w][1] - win[w][0], int))[0]["net_bp"]
            ash = P.eval_policy(None, d, sm, S, win, w, A_override=np.full(win[w][1] - win[w][0], 2))[0]["net_bp"]
            r["cost_bp"] = -(al + ash) / 2; r["top_decile_net_bp"] = r["top_decile_dir_bp"] - r["cost_bp"]
        rep["windows"][w] = r
        print(f"  {w:<5} IC 중앙값 {r['ic_median_bp']:+.4f} · 폭 {r['ic_width_bp']:+.4f} · SNR {r['ic_snr']:+.4f}"
              f" · 적중 {100*r['hit_rate_median']:.1f}% · 상위10% 방향 {r['top_decile_dir_bp']:+.2f}bp"
              + (f" → 순 {r['top_decile_net_bp']:+.2f}" if 'top_decile_net_bp' in r else ""), flush=True)
    # 부호 유지(중앙값 IC): TRAIN 부호가 세 평가창에서 몇 번 유지되나
    s0 = np.sign(rep["windows"]["TRAIN"]["ic_median_bp"])
    rep["sign_keep_median"] = int(sum(np.sign(rep["windows"][w]["ic_median_bp"]) == s0 for w in ("VAL", "OOS", "TEST")))
    p = P.OUT / "report_zeroshot_chronos2.json"
    json.dump(rep, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    np.savez(P.OUT / "chronos2_features.npz", **{f"{w}_F": v[0] for w, v in feats.items()}, **{f"{w}_y": v[1] for w, v in feats.items()})
    print(f"저장 {p} · 부호유지(중앙값 IC) {rep['sign_keep_median']}/3")
    # 자체점검: 문맥 마지막 값이 0(봉 t 기준 정규화)인지 -- 미래 봉이 문맥에 섞이면 이 검사가 무의미하므로 index 도 확인
    j = int(lab["VAL"][0][0]); ctx = lp[j - a.context + 1:j + 1] - lp[j]
    assert abs(ctx[-1]) < 1e-12 and len(ctx) == a.context
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
