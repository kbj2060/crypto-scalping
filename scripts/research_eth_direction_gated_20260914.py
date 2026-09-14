"""**게이트 조건부 방향** — 저변동 구간에서만 방향을 걸면 달라지는가 (2026-09-14).

이 세션에서 5창(2022~23·TRAIN·VAL·OOS·TEST) 부호를 지킨 것은 하나뿐이다: 예측 변동성이 낮은
구간에서 **큰 움직임 도달률이 오른다**(+4.9~15.4pp). 그게 방향에도 쓸모가 있는지 본다.

라벨은 방향용 3%/3% 배리어(전 봉 · 펀딩 제외 · 롱승 42~55%). 게이트는 홀드아웃 변동성 모델
(2022~23 을 학습에서 도려낸 판)의 예측 하위 분위.

팔 넷:
  A 전체        학습 전체 · 평가 전체            (기준선)
  B 게이트학습   학습 게이트만 · 평가 게이트만     (조건부 전문가)
  C 게이트평가   학습 전체 · 평가 게이트만        (게이트가 «고르기»만 하는 경우)
  D 게이트피쳐   학습 전체 + 게이트값을 피쳐로 추가 · 평가 전체

사전등록 통과 조건(둘 다):
  ① 방향 기여 `mean(s·y)` 가 VAL·OOS·TEST **3창 전부 양수**(짝지은 5씨드)
  ② 🔴**순차 실행**(한 번에 한 포지션)에서도 3창 전부 양수 — 겹침 앵커 평균은 굴릴 수 있는 값이
     아니다(스트래들이 그렇게 죽었다: 겹침 +113.9bp 가 순차 −34.7bp).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_direction_barrier_label_20260914 as B  # noqa: E402
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import research_eth_straddle_tighten_20260914 as T  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
EVAL = ("VAL", "OOS", "TEST")


def fit(X, y, seed):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    m.fit(X, y); return m


def overlap_stats(idx, y, m, pred, bars):
    s = np.sign(pred); s[s == 0] = 1.0
    dir_bp = float((s * y).mean()); net = m + s * y
    bb = max(int(np.median(bars)), 1)
    blk = (idx // bb).astype(np.int64)
    bm = np.array([net[blk == k].mean() for k in np.unique(blk)])
    t = float(bm.mean() / (bm.std(ddof=1) / np.sqrt(len(bm)))) if len(bm) > 2 and bm.std(ddof=1) > 0 else np.nan
    return {"hit": float((np.sign(y) == s).mean()), "dir_bp": dir_bp,
            "net_bp": float(net.mean()), "block_t": t, "n": int(len(y))}


def sequential(idx, y, m, pred, bars):
    """🔴한 번에 **한 포지션**. 배리어에 닿아야 다음 진입 — 실제로 굴릴 수 있는 경로."""
    s = np.sign(pred); s[s == 0] = 1.0
    net = m + s * y
    order = np.argsort(idx); taken = []; dirs = []; free_at = -1
    for k in order:
        i = int(idx[k])
        if i <= free_at:
            continue
        taken.append(float(net[k])); dirs.append(float((s * y)[k])); free_at = i + int(bars[k])
    a = np.array(taken) if taken else np.zeros(1)
    dd = np.array(dirs) if dirs else np.zeros(1)
    return {"n": len(taken), "mean_bp": float(a.mean()), "dir_bp": float(dd.mean()),
            "t": float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 else np.nan,
            "dir_t": float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if len(dd) > 2 else np.nan,
            "equity": float(np.prod(1 + a / 1e4))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(OUT / "labels_3_3_nofund.npz"))
    ap.add_argument("--cut", type=float, default=20.0, help="게이트 분위(%) — 이 아래만 통과")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tag", default="direction_gated")
    a = ap.parse_args()
    seeds = [int(x) for x in np.random.default_rng(20260914).integers(1, 1_000_000, size=a.seeds)]
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    g = T.gate_series(d, c, hi, lo)
    tl, th = win["TRAIN"]; edge = float(np.nanpercentile(g[tl:th], a.cut))
    z = np.load(a.labels, allow_pickle=True)
    lab = {w: {k: z[f"{w}_{k}"] for k in ("idx", "y", "m", "bars")} for w in z["names"]}
    for w in lab:
        lab[w]["pass"] = g[lab[w]["idx"]] <= edge
    print(f"게이트 하위 {a.cut:g}% (경계 {edge:.4f}) · 통과율 " +
          " · ".join(f"{w} {lab[w]['pass'].mean():.1%}" for w in ("TRAIN",) + EVAL))
    Sg = np.hstack([S, g.reshape(-1, 1)])            # D 팔: 게이트값을 피쳐로
    tr = lab["TRAIN"]
    arms = {
        "A 전체":      (S,  np.ones(len(tr["idx"]), bool), {w: np.ones(len(lab[w]["idx"]), bool) for w in EVAL}),
        "B 게이트학습": (S,  tr["pass"],                    {w: lab[w]["pass"] for w in EVAL}),
        "C 게이트평가": (S,  np.ones(len(tr["idx"]), bool), {w: lab[w]["pass"] for w in EVAL}),
        "D 게이트피쳐": (Sg, np.ones(len(tr["idx"]), bool), {w: np.ones(len(lab[w]["idx"]), bool) for w in EVAL}),
    }
    rep = {"cut": a.cut, "edge": edge, "seeds": seeds, "arms": {}}
    print(f"\n{'팔':>12} {'창':>5} {'n':>7} {'적중':>7} {'방향bp':>8} {'순손익bp':>9} {'블록t':>7} "
          f"{'순차n':>6} {'순차bp':>8} {'순차t':>7}")
    for name, (X, trmask, evmask) in arms.items():
        cell = {}
        for w in EVAL:
            ov, sq = [], []
            for sd in seeds:
                mdl = fit(X[tr["idx"][trmask]], tr["y"][trmask], sd)
                sel = evmask[w]
                i_, y_, m_, b_ = (lab[w]["idx"][sel], lab[w]["y"][sel], lab[w]["m"][sel], lab[w]["bars"][sel])
                pr = mdl.predict(X[i_])
                ov.append(overlap_stats(i_, y_, m_, pr, b_))
                sq.append(sequential(i_, y_, m_, pr, b_))
            agg = {k: float(np.mean([o[k] for o in ov])) for k in ("hit", "dir_bp", "net_bp", "block_t")}
            agg |= {"n": ov[0]["n"], "seq_n": sq[0]["n"],
                    "seq_bp": float(np.mean([s["mean_bp"] for s in sq])),
                    "seq_t": float(np.mean([s["t"] for s in sq]))}
            cell[w] = agg
            print(f"{name:>12} {w:>5} {agg['n']:>7,} {agg['hit']:>7.2%} {agg['dir_bp']:>+8.2f} "
                  f"{agg['net_bp']:>+9.2f} {agg['block_t']:>+7.2f} {agg['seq_n']:>6} "
                  f"{agg['seq_bp']:>+8.2f} {agg['seq_t']:>+7.2f}", flush=True)
        ok1 = all(cell[w]["dir_bp"] > 0 for w in EVAL)
        ok2 = all(cell[w]["seq_bp"] > 0 for w in EVAL)
        print(f"{'':>12} 사전등록 ①방향 3창양수 {ok1} · ②순차 3창양수 {ok2} ⇒ "
              f"{'⭐통과' if ok1 and ok2 else '탈락'}\n")
        rep["arms"][name] = {"windows": cell, "pass_dir": ok1, "pass_seq": ok2}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"저장: {OUT/(a.tag+'.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
