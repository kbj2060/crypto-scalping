"""**방향 라벨 재설계** — 기저 표류를 라벨에서 뺀다 (2026-09-14, 사용자 *"라벨부터 다시"*).

## 왜 다시 만드나 — 측정된 병리
3%/3% 배리어 라벨의 **기저가 창마다 흔들린다**: 롱승 TRAIN 51.8% · VAL 50.9% · **OOS 42.1%** ·
TEST 54.5%. 모델이 학습창의 약한 롱 편향을 외우면 숏이 58% 이긴 OOS 에서 정확히 반대로 맞는다 --
실제로 OOS 적중률이 **47.88%(동전 아래)** 였다. 지금까지 방향이 실패한 방식이 「정보가 없다」만이
아니라 「**있는 편향을 외웠다**」일 수 있다는 뜻이다. 그건 라벨로 고칠 수 있다.

## 팔 넷 (전부 **같은 실제 라벨**로 평가 — 학습 라벨만 바꾼다)
  L0 기준      원시 y
  L1 탈드리프트 학습창의 평균 로그수익을 가격에서 뺀 뒤 배리어를 다시 판정해 라벨을 만든다.
               기저가 50% 로 눌리므로 모델이 외울 편향이 없어진다. 🔴평가는 **실제** 라벨이다 --
               탈드리프트 라벨로 평가하면 「내가 지운 답을 내가 맞혔다」가 된다.
  L2 속도가중   빠르게 해결된 앵커에 큰 가중(1/해결시간). 배리어 라벨은 |y| 가 상수라 「얼마나
               확실한가」가 라벨에 안 담긴다 -- 속도가 그 대리물이다.
  L3 측면분리   「롱이 이기나」와 「숏이 이기나」를 **각각** 이진 분류하고 차이로 베팅. 측면별
               기저가 다르게 움직이므로 한 모델로 묶지 않는다.

## 판정 (사전등록)
  ① 방향 기여 mean(s·y) 가 VAL·OOS·TEST **3창 전부 양수**(짝지은 5씨드)
  ② **순차 실행**(한 번에 한 포지션)에서도 3창 전부 양수
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_direction_barrier_label_20260914 as B  # noqa: E402
import research_eth_direction_gated_20260914 as GG  # noqa: E402
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
EVAL = ("VAL", "OOS", "TEST")


def reg(X, y, seed, w=None):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    m.fit(X, y, sample_weight=w); return m


def clf(X, y, seed):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    m.fit(X, y); return m


WIN_SWEEP = (288, 576, 1152, 2304, 3456, 5760)   # 2·4·8·16·24·40일


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", type=float, default=0.03)
    ap.add_argument("--down", type=float, default=0.03)
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default="", help="쉼표 구분 팔 접두어만 실행")
    ap.add_argument("--tag", default="label_rebuild")
    a = ap.parse_args()
    B.FUNDING = False                       # 🔴상수 펀딩은 방향 라벨에서 항상 뺀다(09-14 버그)
    seeds = [int(x) for x in np.random.default_rng(20260914).integers(1, 1_000_000, size=a.seeds)]
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    ok = sm["ok"]

    lab = {}
    for w in ("TRAIN",) + EVAL:
        st = a.train_stride if w == "TRAIN" else 1
        lab[w] = B.label_window(c, hi, lo, ok, *win[w], st, a.up, a.down)
        y = lab[w]["y"]
        print(f"  {w:>5}: n {len(y):,} · 롱승 {float((y>0).mean()):.1%} · E|y| {np.abs(y).mean():.0f}bp")

    # L1 학습 라벨 — 학습창 평균 로그수익을 뺀 가격으로 배리어를 다시 판정
    cd, hd, ld = G.dedrift(d, *win["TRAIN"])
    tr = lab["TRAIN"]
    ded = B.label_window(cd, hd, ld, ok, *win["TRAIN"], a.train_stride, a.up, a.down)
    # 앵커가 완전히 같지 않을 수 있어(미해결 등) 인덱스로 맞춘다
    pos = {int(i): k for k, i in enumerate(ded["idx"])}
    keep = np.array([k for k, i in enumerate(tr["idx"]) if int(i) in pos])
    y_ded = np.array([ded["y"][pos[int(tr["idx"][k])]] for k in keep])
    print(f"\nL1 탈드리프트: 학습 앵커 {len(keep):,}/{len(tr['idx']):,} 매칭 · "
          f"기저 {float((tr['y'][keep]>0).mean()):.1%} → {float((y_ded>0).mean()):.1%}")

    arms = {}
    arms["L0 기준"] = lambda sd: reg(S[tr["idx"]], tr["y"], sd)
    arms["L1 탈드리프트"] = lambda sd: reg(S[tr["idx"][keep]], y_ded, sd)
    wspeed = 1.0 / np.maximum(tr["bars"], 1.0)
    arms["L2 속도가중"] = lambda sd: reg(S[tr["idx"]], tr["y"], sd, wspeed / wspeed.mean())

    class SideSplit:
        def __init__(self, sd):
            self.a = clf(S[tr["idx"]], (tr["y"] > 0).astype(int), sd)
            self.b = clf(S[tr["idx"]], (tr["y"] < 0).astype(int), sd)
        def predict(self, X):
            return self.a.predict_proba(X)[:, 1] - self.b.predict_proba(X)[:, 1]
    arms["L3 측면분리"] = SideSplit

    # L4 연속 라벨 — 배리어는 ±3% 둘 중 하나라 「얼마나 유리했나」가 라벨에 안 담긴다.
    # 앞으로 H봉의 **최대유리이동 − 최대역행**(롱 기준, ATR 로 정규화)을 학습 타깃으로 쓴다.
    # 🔴평가는 그대로 실제 배리어 라벨이다(학습 라벨만 바꾸는 규율).
    def mfe_mae_target(H: int) -> np.ndarray:
        idx = tr["idx"]
        atr = np.maximum(sm["atr_pct"], 1e-9)
        out = np.empty(len(idx))
        for k, i in enumerate(idx):
            j = min(int(i) + H + 1, len(c))
            seg_hi = hi[int(i) + 1:j]; seg_lo = lo[int(i) + 1:j]
            if len(seg_hi) < 2:
                out[k] = 0.0; continue
            mfe = seg_hi.max() / c[int(i)] - 1.0
            mae = c[int(i)] / seg_lo.min() - 1.0          # 롱 기준 역행(양수)
            out[k] = (mfe - mae) / atr[int(i)]
        return out
    for H in (48, 144):
        tgt = mfe_mae_target(H)
        print(f"L4 MFE−MAE(H={H}): 평균 {tgt.mean():+.3f} · SD {tgt.std():.3f} · "
              f"부호가 실제 라벨과 일치 {float((np.sign(tgt) == np.sign(tr['y'])).mean()):.1%}")
        arms[f"L4 MFE-MAE H{H}"] = (lambda t: (lambda sd: reg(S[tr["idx"]], t, sd)))(tgt)

    # ── 2차 재설계 (2026-09-14, 6팔 전부 탈락 뒤) ─────────────────────────────
    # 1차의 여섯 팔은 전부 「봉 i 종가 기준 절대 방향」을 물었다. 여기서는 **묻는 방식**을 바꾼다.
    import pandas as _pd

    # L5 국소 상대 — L1(탈드리프트)은 **전역 평균** 하나만 뺐고 그래서 기저를 0.8pp 밖에 못 움직였다.
    # 창 안에서도 드리프트는 계속 변한다. 이웃 앵커들의 평균을 빼면 「지금이 **주변보다** 롱인가」가
    # 되고, 레짐/드리프트가 국소적으로 상쇄된다. 이게 1차에서 못 한 자리다.
    for WIN_A in WIN_SWEEP:          # 앵커 개수(stride 2 이므로 봉으로는 2배)
        loc = _pd.Series(tr["y"]).rolling(WIN_A, center=True, min_periods=WIN_A // 3).mean().to_numpy()
        t5 = tr["y"] - np.where(np.isfinite(loc), loc, 0.0)
        print(f"L5 국소상대(창 {WIN_A}앵커): 기저 {float((tr['y']>0).mean()):.1%} → "
              f"{float((t5>0).mean()):.1%} · SD {t5.std():.0f}bp")
        arms[f"L5 국소상대 W{WIN_A}"] = (lambda t: (lambda sd: reg(S[tr["idx"]], t, sd)))(t5)

    # L6 다중지평 합의 — 3%/3% 라벨은 한 번의 배리어 경주라 잡음이 크다. 여러 지평이 **모두 같은
    # 부호**일 때만 학습에 쓴다(나머지는 버린다). 라벨 잡음을 줄이는 직접적 방법이고, 「관망」을
    # 라벨 수준에서 넣는 것과 같다.
    HS = (12, 48, 144, 288)
    sgn = np.zeros((len(tr["idx"]), len(HS)))
    for k, H in enumerate(HS):
        j = np.minimum(tr["idx"] + H, len(c) - 1)
        sgn[:, k] = np.sign(c[j] - c[tr["idx"]])
    agree = (np.abs(sgn.sum(1)) == len(HS)) & (sgn != 0).all(1)
    t6 = np.sign(sgn.sum(1))[agree]
    print(f"L6 다중지평합의: {agree.mean():.1%} 가 4지평 일치 · 그 중 롱 {float((t6>0).mean()):.1%}")
    arms["L6 다중지평합의"] = (lambda idx, t: (lambda sd: reg(S[idx], t, sd)))(tr["idx"][agree], t6)

    # L7 경로효율 — 부호만 보면 「3% 를 톱질하며 간 것」과 「곧게 간 것」이 같은 라벨이다.
    # 효율비(|끝-시작| / 경로 총변동)를 곱해 **곧게 간 방향**에 무게를 준다.
    H7 = 144
    eff = np.empty(len(tr["idx"]))
    lc = np.log(np.maximum(c, 1e-12)); ad = np.abs(np.diff(lc, prepend=lc[0]))
    cum = np.cumsum(ad)
    j7 = np.minimum(tr["idx"] + H7, len(c) - 1)
    path = np.maximum(cum[j7] - cum[tr["idx"]], 1e-9)
    net7 = lc[j7] - lc[tr["idx"]]
    eff = net7 / path                      # -1..+1, 곧게 갈수록 절대값이 크다
    print(f"L7 경로효율(H={H7}): 평균 {eff.mean():+.3f} · SD {eff.std():.3f} · "
          f"부호가 실제 라벨과 일치 {float((np.sign(eff) == np.sign(tr['y'])).mean()):.1%}")
    arms["L7 경로효율"] = (lambda t: (lambda sd: reg(S[tr["idx"]], t, sd)))(eff)

    if a.only:
        pre = tuple(x.strip() for x in a.only.split(","))
        arms = {k: v for k, v in arms.items() if k.startswith(pre)}

    rep = {"up": a.up, "down": a.down, "seeds": seeds, "arms": {}}
    print(f"\n{'팔':>14} {'창':>5} {'적중':>7} {'방향bp':>8} {'순손익bp':>9} {'블록t':>7} "
          f"{'순차n':>6} {'순차방향':>8} {'순차t':>7} {'순차순익':>8}")
    for name, make in arms.items():
        cell = {}
        for w in EVAL:
            L = lab[w]; ov, sq = [], []
            for sd in seeds:
                pr = make(sd).predict(S[L["idx"]])
                ov.append(GG.overlap_stats(L["idx"], L["y"], L["m"], pr, L["bars"]))
                sq.append(GG.sequential(L["idx"], L["y"], L["m"], pr, L["bars"]))
            agg = {k: float(np.mean([o[k] for o in ov])) for k in ("hit", "dir_bp", "net_bp", "block_t")}
            agg |= {"seq_n": sq[0]["n"], "seq_bp": float(np.mean([s["mean_bp"] for s in sq])),
                    "seq_t": float(np.mean([s["t"] for s in sq])),
                    "seq_dir_bp": float(np.mean([s["dir_bp"] for s in sq])),
                    "seq_dir_t": float(np.mean([s["dir_t"] for s in sq]))}
            cell[w] = agg
            print(f"{name:>14} {w:>5} {agg['hit']:>7.2%} {agg['dir_bp']:>+8.2f} {agg['net_bp']:>+9.2f} "
                  f"{agg['block_t']:>+7.2f} {agg['seq_n']:>6} {agg['seq_dir_bp']:>+8.2f} "
                  f"{agg['seq_dir_t']:>+7.2f} {agg['seq_bp']:>+8.2f}", flush=True)
        ok1 = all(cell[w]["dir_bp"] > 0 for w in EVAL)
        ok2 = all(cell[w]["seq_dir_bp"] > 0 for w in EVAL)
        prof = all(cell[w]["seq_bp"] > 0 for w in EVAL)
        print(f"{'':>14} ①겹침방향 3창양수 {ok1} · ②순차방향 3창양수 {ok2} · "
              f"(참고) 순차순익 3창양수 {prof} ⇒ {'⭐통과' if ok1 and ok2 else '탈락'}\n")
        rep["arms"][name] = {"windows": cell, "pass_dir": ok1, "pass_seq": ok2}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"저장: {OUT/(a.tag+'.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
