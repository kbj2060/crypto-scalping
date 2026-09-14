"""**피쳐 분석** — 재료 43열이 방향 정보를 갖는가, 그리고 그게 창을 넘어 유지되는가 (2026-09-14).

사용자 *"피쳐도 분석하고 방향을 잡는 모델이야."* 설계 §3.2 의 진단 넷 중 **anti-stable 점검**과
**단변량 분위 효과**를 모델 없이 직접 낸다. 모델(PPO·쌍둥이)이 내린 결론이 아니라 **재료 자체의 성질**이다.

라벨은 gym 이 정의한 방향수익 `y = ½(r_long − r_short)·1e4` (비용 상쇄된 순수 가격 이동 bp).
피쳐마다 창별 스피어만 IC 를 내고, **학습창 IC 와 평가창 IC 의 상관**을 본다.
🔴그 상관이 음수면 «학습에서 강했던 피쳐가 평가에서 반대로 간다»는 뜻이고, 설계에 사전등록한
중단 조건이다(SOL 에서 −0.38 로 관측된 적이 있다).

라벨이 창마다 겹치므로(지평 최대 288봉) **일 단위 군집**으로 SE 를 낸다. 겹친 라벨을 독립으로 세면
t 가 가짜로 커진다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import research_eth_rl_gym_direction_ic_20260914 as IC  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

CACHE = P.OUT / "direction_labels.npz"


def get_labels(d, sm, win, stride_train: int, stride_eval: int):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return {k: (z[f"{k}_idx"], z[f"{k}_y"]) for k in z["names"]}
    out = {}
    for w in ("TRAIN", "VAL", "OOS", "TEST", "BACK22_23"):
        lo, hi = win[w]
        s = stride_train if w == "TRAIN" else stride_eval
        idx, y = IC.labels_window(d, sm, lo, hi, s)
        out[w] = (idx, y)
        print(f"  {w} 라벨 {len(y):,} · 평균 {y.mean():+.2f}bp", flush=True)
    np.savez(CACHE, names=np.array(list(out)), **{f"{k}_idx": v[0] for k, v in out.items()},
             **{f"{k}_y": v[1] for k, v in out.items()})
    return out


def day_clustered_t(x: np.ndarray, days: np.ndarray) -> float:
    s = pd.Series(x).groupby(days).mean()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return 0.0
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride-train", type=int, default=4)
    ap.add_argument("--stride-eval", type=int, default=12)
    a = ap.parse_args()
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    print("라벨 …", flush=True)
    lab = get_labels(d, sm, win, a.stride_train, a.stride_eval)
    ts = pd.to_datetime(d.timestamp.to_numpy())

    ics = {}
    for w, (idx, y) in lab.items():
        ics[w] = {c: float(spearmanr(S[idx, j], y).statistic) for j, c in enumerate(cols)}
    tr = np.array([ics["TRAIN"][c] for c in cols])

    rows = []
    for j, c in enumerate(cols):
        r = {"feature": c, "TRAIN": ics["TRAIN"][c]}
        for w in ("VAL", "OOS", "TEST"):
            r[w] = ics[w][c]
        idx, y = lab["VAL"]
        # 부호 일치: 학습창 IC 부호가 세 평가창에서 몇 번 유지되나
        r["sign_keep"] = int(sum(np.sign(ics[w][c]) == np.sign(r["TRAIN"]) for w in ("VAL", "OOS", "TEST")))
        # 단변량 십분위 스프레드(학습창, bp) -- 「평평한가」를 보는 값
        ti, ty = lab["TRAIN"]
        q = pd.qcut(pd.Series(S[ti, j]), 10, labels=False, duplicates="drop")
        g = pd.Series(ty).groupby(q.to_numpy()).mean()
        r["decile_spread_bp"] = float(g.max() - g.min()) if len(g) > 1 else 0.0
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("TRAIN", key=np.abs, ascending=False)

    print("\n학습창 |IC| 상위 12 (창별 IC · 부호유지 /3 · 십분위 스프레드)")
    print(f"  {'피쳐':<18}{'TRAIN':>8}{'VAL':>8}{'OOS':>8}{'TEST':>8}{'부호':>5}{'스프레드bp':>10}")
    for _, r in df.head(12).iterrows():
        print(f"  {r.feature:<18}{r.TRAIN:>+8.4f}{r.VAL:>+8.4f}{r.OOS:>+8.4f}{r.TEST:>+8.4f}"
              f"{r.sign_keep:>5}{r.decile_spread_bp:>10.1f}")

    print("\n🔴anti-stable 점검 — 학습창 IC 와 평가창 IC 의 상관 (음수면 중단)")
    anti = {}
    for w in ("VAL", "OOS", "TEST", "BACK22_23"):
        ev = np.array([ics[w][c] for c in cols])
        # 🔴창 안에서 상수인 열(원핫 레짐이 그 창에 없을 때)은 IC 가 NaN 이다. 그대로 넣으면
        # 상관 전체가 NaN 이 되어 «점검을 했다»고 착각한다(2026-09-14 첫 실행에서 실제로 그랬다).
        m = np.isfinite(tr) & np.isfinite(ev)
        anti[w] = float(spearmanr(tr[m], ev[m]).statistic) if m.sum() >= 5 else float("nan")
        print(f"  TRAIN ↔ {w:<10} {anti[w]:+.3f}  (유효 {int(m.sum())}/{len(cols)}열)")
    print(f"  부호유지 3/3 인 피쳐: {int((df.sign_keep == 3).sum())} / {len(df)}"
          f" · 0/3: {int((df.sign_keep == 0).sum())}")

    # ⭐**손익분기에 필요한 IC**. |IC| 0.02 짜리 피쳐가 얼마를 버는지를 산술로 못박는다.
    # 부호 베팅의 기대 방향수익 ≈ IC · E|y| · √(2/π) 이므로 IC* ≈ 비용 / (E|y|·0.7979).
    need = {}
    for w in ("VAL", "OOS", "TEST"):
        idx, y = lab[w]
        e_abs = float(np.mean(np.abs(y)))
        al = P.eval_policy(None, d, sm, S, win, w, A_override=np.ones(win[w][1] - win[w][0], int))[0]["net_bp"]
        ash = P.eval_policy(None, d, sm, S, win, w, A_override=np.full(win[w][1] - win[w][0], 2))[0]["net_bp"]
        c_bp = -(al + ash) / 2
        need[w] = float(c_bp / (e_abs * 0.79788))
        print(f"  {w:<10} 비용 {c_bp:.2f}bp · E|이동| {e_abs:.1f}bp → **손익분기 IC {need[w]:.3f}**")
    print(f"  관측된 최대 |학습창 IC| = {np.nanmax(np.abs(tr)):.4f} "
          f"({df.iloc[0].feature}) -- 필요치의 {np.nanmax(np.abs(tr))/np.mean(list(need.values())):.0%}")

    # 최고 피쳐 단독 규칙: 학습창 IC 최대 피쳐의 상·하위 십분위로 베팅
    best = df.iloc[0]
    j = cols.index(best.feature)
    print(f"\n모델 없는 단독 규칙 — {best.feature} 상·하위 십분위 (부호는 학습창에서 고정)")
    sgn = np.sign(best.TRAIN)
    rule = {}
    for w in ("VAL", "OOS", "TEST"):
        idx, y = lab[w]
        x = S[idx, j]
        lo_q, hi_q = np.quantile(S[lab["TRAIN"][0], j], [0.1, 0.9])
        m = (x <= lo_q) | (x >= hi_q)
        side = np.where(x[m] >= hi_q, sgn, -sgn)
        bp = side * y[m]
        cost = P.eval_policy(None, d, sm, S, win, w,
                             A_override=np.ones(win[w][1] - win[w][0], int))[0]["net_bp"]
        cost2 = P.eval_policy(None, d, sm, S, win, w,
                              A_override=np.full(win[w][1] - win[w][0], 2))[0]["net_bp"]
        c_bp = -(cost + cost2) / 2
        t = day_clustered_t(bp, ts[idx[m]].normalize().to_numpy())
        rule[w] = {"n": int(m.sum()), "dir_bp": float(bp.mean()), "net_bp": float(bp.mean() - c_bp),
                   "t_day": t}
        print(f"  {w:<10} n {m.sum():>5} · 방향 {bp.mean():+7.2f}bp · 순 {bp.mean()-c_bp:+7.2f}bp · 일군집 t {t:+.2f}")

    out = {"ic_by_window": ics, "anti_stable_corr": anti,
           "sign_keep_3of3": int((df.sign_keep == 3).sum()), "n_features": len(df),
           "top12": df.head(12).to_dict("records"), "single_feature_rule": {"feature": best.feature, **rule},
           "breakeven_ic": need, "max_abs_train_ic": float(np.nanmax(np.abs(tr)))}
    p = P.OUT / "report_feature_stability.json"
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n저장 {p}")
    assert len(cols) == S.shape[1]
    # 자체점검: 라벨 부호를 뒤집으면 모든 IC 도 뒤집혀야 한다(계산 경로 확인)
    idx, y = lab["VAL"]
    a0 = spearmanr(S[idx, 0], y).statistic; a1 = spearmanr(S[idx, 0], -y).statistic
    assert abs(a0 + a1) < 1e-12, "IC 계산이 라벨 부호에 대칭이 아니다"
    print("확인: IC 계산 부호 대칭 · 피쳐 수 일치")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
