"""**L5 국소상대 — 격자 운인가** 블록 회전 귀무 + 단조성 (2026-09-14, 사용자 *"둘 다 돌려줘"*).

창 6개(2·4·8·16·24·40일)를 훑어 W2304(16일) 하나가 사전등록 ①②를 통과했다. 이웃(8일·24일)이
실패하므로 **격자 운**일 수 있다. 두 가지로 가른다:

① **블록 회전 귀무** — 모델은 그대로 두고 **예측을 시간축으로 하루 단위 회전**시켜 피쳐↔라벨 정렬만
   깬다. 라벨의 자기상관·해결시간·순차 점유 구조는 그대로라 「이 정렬이 특별한가」만 묻는다.
   모델을 다시 적합하지 않으므로 B=200 이 싸다.
   통계량 = **6창 중 ①②를 동시에 통과한 창의 수**. 실제 1 vs 귀무 분포.
② **단조성** — 창 길이 vs 순차 방향기여의 스피어만. 진짜라면 한 칸이 아니라 **기울기**가 산다.
   VAL·OOS·TEST 세 창에서 부호가 일관되는지 보고, 같은 통계량을 귀무에서도 낸다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_direction_barrier_label_20260914 as B  # noqa: E402
import research_eth_direction_gated_20260914 as GG  # noqa: E402
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
EVAL = ("VAL", "OOS", "TEST")
# 🔴2022~23 은 지금까지 방향 평가에 **한 번도 안 썼다**. 순차 거래가 창당 106~149 뿐이라
#   표본이 병목인데, 여기는 2년치라 1,000건 이상 나온다. 완전 독립 구간이다.
WINS = (288, 576, 1152, 2304, 3456, 5760)


def stats(pred, L):
    ov = GG.overlap_stats(L["idx"], L["y"], L["m"], pred, L["bars"])
    sq = GG.sequential(L["idx"], L["y"], L["m"], pred, L["bars"])
    return ov["dir_bp"], sq["dir_bp"]


def sweep_pass(preds, lab):
    """창별 (①겹침 3창양수, ②순차 3창양수) 과 순차방향 행렬."""
    n_pass, seq_mat = 0, {}
    for W in WINS:
        ov = [];  sq = []
        for w in EVAL:
            o, s = stats(preds[(W, w)], lab[w]); ov.append(o); sq.append(s)
        seq_mat[W] = sq
        if min(ov) > 0 and min(sq) > 0:
            n_pass += 1
    return n_pass, seq_mat


def mono(seq_mat):
    """창 길이 vs 순차 방향기여 스피어만, 창별."""
    ws = np.array(WINS, float)
    return [float(spearmanr(ws, [seq_mat[W][k] for W in WINS]).statistic) for k in range(len(EVAL))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--back", action="store_true", help="2022~23 을 평가창에 추가")
    ap.add_argument("--tag", default="l5_null")
    a = ap.parse_args()
    B.FUNDING = False
    global EVAL
    if a.back:
        EVAL = ("BACK22_23",) + EVAL
    seeds = [int(x) for x in np.random.default_rng(20260914).integers(1, 1_000_000, size=a.seeds)]
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    ok = sm["ok"]
    lab = {w: B.label_window(c, hi, lo, ok, *win[w], a.train_stride if w == "TRAIN" else 1, 0.03, 0.03)
           for w in ("TRAIN",) + EVAL}
    tr = lab["TRAIN"]
    print("예측 적합 중 (창 6 × 씨드 {}) …".format(a.seeds), flush=True)
    preds = {}
    for W in WINS:
        loc = pd.Series(tr["y"]).rolling(W, center=True, min_periods=W // 3).mean().to_numpy()
        t = tr["y"] - np.where(np.isfinite(loc), loc, 0.0)
        ps = {w: [] for w in EVAL}
        for sd in seeds:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              random_state=sd)
            m.fit(S[tr["idx"]], t)
            for w in EVAL:
                ps[w].append(m.predict(S[lab[w]["idx"]]))
        for w in EVAL:
            preds[(W, w)] = np.mean(ps[w], axis=0)      # 짝지은 씨드 평균 예측
        print(f"  W{W} 완료", flush=True)

    real_pass, real_seq = sweep_pass(preds, lab)
    real_mono = mono(real_seq)
    print(f"\n실제: ①②를 동시에 통과한 창 **{real_pass}개**/6")
    print(f"{'창':>7} " + " ".join(f"{w:>9}" for w in EVAL) + "   (순차 방향기여 bp)")
    for W in WINS:
        print(f"{W/144:>5.0f}일 " + " ".join(f"{v:>+9.2f}" for v in real_seq[W]))
    print(f"단조 스피어만(창길이 vs 순차방향): " +
          " · ".join(f"{w} {r:+.2f}" for w, r in zip(EVAL, real_mono)) +
          f"  ⇒ 3창 모두 양수 {all(r > 0 for r in real_mono)}")

    # ── 블록 회전 귀무 ──────────────────────────────────────────────────────
    rng = np.random.default_rng(20260914)
    hits_pass = np.zeros(a.boot, int); hits_mono = np.zeros(a.boot, bool)
    for b in range(a.boot):
        rot = {}
        for w in EVAL:
            n = len(lab[w]["idx"])
            k = int(rng.integers(1, max(2, n // 288))) * 288      # 하루 단위 회전
            for W in WINS:
                rot[(W, w)] = np.roll(preds[(W, w)], k)
        np_, ns = sweep_pass(rot, lab)
        hits_pass[b] = np_; hits_mono[b] = all(r > 0 for r in mono(ns))
        if (b + 1) % 50 == 0:
            print(f"  귀무 {b+1}/{a.boot}: 통과창 평균 {hits_pass[:b+1].mean():.2f} · "
                  f"«{real_pass}개 이상» {float((hits_pass[:b+1] >= real_pass).mean()):.1%} · "
                  f"단조통과 {hits_mono[:b+1].mean():.1%}", flush=True)
    p_pass = float((hits_pass >= real_pass).mean())
    p_mono = float(hits_mono.mean())
    print(f"\n귀무 {a.boot}회 (하루 단위 블록 회전):")
    print(f"  통과창 수 분포: 평균 {hits_pass.mean():.2f} · 최대 {hits_pass.max()} · "
          f"**P(≥{real_pass}) = {p_pass:.1%}**")
    print(f"  단조 3창 양수: **{p_mono:.1%}**  (실제 {all(r > 0 for r in real_mono)})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(
        {"real_pass": real_pass, "real_seq": {str(k): v for k, v in real_seq.items()},
         "real_mono": real_mono, "boot": a.boot, "p_pass": p_pass, "p_mono": p_mono,
         "null_pass_counts": hits_pass.tolist()}, indent=1, ensure_ascii=False))
    print(f"저장: {OUT/(a.tag+'.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
