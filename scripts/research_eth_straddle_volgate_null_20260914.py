"""**스트래들 변동성 게이트 — 귀무와 단조성** (2026-09-14).

`straddle_gate_rv48` 에서 「최저 변동성 분위가 5창 전부 양수」인 셀이 6개 중 2개 나왔다.
그런데 나는 6셀 × 5분위 = **30조합을 훑어 2개를 골랐다** -- 순수 잡음에서도 P(5/5 부호일치) ≈ 1/32 라
기대 통과수가 **0.9개**다. 2 vs 0.9 는 증거가 아니다. 그래서 세 가지를 잰다:

① **블록 t** — 분위별로 블록(=보유기간) 평균을 표본 단위로 쓴 t. 겹친 앵커를 t 로 착각하지 않게.
② **단조성** — 진짜 게이트라면 분위 0→4 가 **단조**여야 한다(극단 버킷만 좋으면 격자 운이다).
   창마다 분위 순서 vs 쌍당bp 의 스피어만을 내고 5창 부호 일치를 본다.
③ **블록 순열 귀무(B회)** — rv48 게이트를 **날 단위 블록으로 섞어** 같은 절차를 반복해,
   「5창 전부 양수」와 「단조성 5창 일치」가 우연히 몇 번 나오는지 센다. 겹침·자기상관을 보존한다.

게이트가 진짜면 ①t 가 여러 창에서 1 을 넘고 ②단조가 일관되며 ③귀무 통과율이 낮아야 한다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_direction_barrier_label_20260914 as B  # noqa: E402
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

WINS = ("BACK22_23", "TRAIN", "VAL", "OOS", "TEST")
DAY = 288                                   # 5분봉 하루


def gate_values(d, name: str, artifact: str = "") -> np.ndarray:
    """게이트 값. `volfc`/`volexp` 는 **배포된** 전방 변동성 모델(train_end 2025-08-31),
    `rvratio` 는 **모델 없는 대체물**(rv12/rv48) -- 타자산 이식에는 이쪽만 쓸 수 있다."""
    import numpy as np
    if name == "rvratio":
        return np.log(np.clip(d["rv12"].to_numpy(float), 1e-9, None)) - \
               np.log(np.clip(d["rv48"].to_numpy(float), 1e-9, None))
    if name in ("volfc", "volexp"):
        import live_eth_sizing_vol_model_20260912 as svm
        if artifact:
            svm.ARTIFACT = pathlib.Path(artifact)
        art = svm.load_model(); assert art is not None
        print(f"게이트 아티팩트 {svm.ARTIFACT.name} · 학습 {art.get('train_start','처음')}~{art['train_end']}")
        X = svm.build_features(d.timestamp, d.close.to_numpy(float), d.quote_volume.to_numpy(float),
                               d.trades.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float))
        fc = np.log(np.clip(svm.predict_vol(art["models"], X), 1e-9, None))
        return fc if name == "volfc" else fc - np.log(np.clip(d["rv48"].to_numpy(float), 1e-9, None))
    return d[name].to_numpy(float)


def bucket_stats(m, bars, idx, g, nq: int = 5):
    out = []
    for j in range(nq):
        sel = g == j
        if sel.sum() < 30:
            out.append({"n": int(sel.sum()), "pair_bp": float("nan"), "block_t": float("nan"), "blocks": 0})
            continue
        mm, bb = m[sel], float(np.median(bars[sel]))
        b = (idx[sel] // max(int(bb), 1)).astype(np.int64)
        bm = np.array([mm[b == k].mean() for k in np.unique(b)])
        t = float(bm.mean() / (bm.std(ddof=1) / np.sqrt(len(bm)))) if len(bm) > 2 and bm.std(ddof=1) > 0 else float("nan")
        out.append({"n": int(sel.sum()), "pair_bp": float(mm.mean()), "block_t": t,
                    "blocks": int(len(bm)), "hours": bb * 5 / 60})
    return out


def evaluate(lab: dict, gval: np.ndarray, edges: np.ndarray) -> dict:
    """분위별 통계 + 창별 단조성. lab[w] = label_window 결과."""
    res = {}
    for w in WINS:
        L = lab[w]
        g = np.digitize(gval[L["idx"]], edges)
        st = bucket_stats(L["m"], L["bars"], L["idx"], g)
        vals = np.array([s["pair_bp"] for s in st])
        ok = np.isfinite(vals)
        rho = float(spearmanr(np.arange(5)[ok], vals[ok]).statistic) if ok.sum() >= 3 else float("nan")
        res[w] = {"buckets": st, "spearman": rho}
    return res


def passes(res: dict, side: str = "low") -> tuple[bool, bool]:
    j = 0 if side == "low" else 4
    sgn = -1.0 if side == "low" else 1.0          # 단조 방향도 보는 쪽에 맞춘다
    pick = [res[w]["buckets"][j]["pair_bp"] for w in WINS]
    mono = [res[w]["spearman"] for w in WINS]
    return (all(np.isfinite(v) and v > 0 for v in pick),
            all(np.isfinite(r) and sgn * r > 0 for r in mono))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", type=float, default=0.10)
    ap.add_argument("--down", type=float, default=0.03)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--gate", default="rv48", help="rv48 | volfc | volexp | rvratio")
    ap.add_argument("--side", default="low", choices=("low", "high"))
    ap.add_argument("--vol-artifact", default="")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--tag", default="volgate_null")
    a = ap.parse_args()
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    gval = gate_values(d, a.gate, a.vol_artifact)
    tl, th = win["TRAIN"]
    edges = np.nanpercentile(gval[tl:th], [20, 40, 60, 80])
    lab = {w: B.label_window(c, hi, lo, sm["ok"], *win[w], a.stride, a.up, a.down) for w in WINS}
    for w in WINS:
        print(f"  {w:>10} 앵커 {len(lab[w]['idx']):,}")
    real = evaluate(lab, gval, edges)
    print(f"\n{a.up*100:g}%/{a.down*100:g}% · 게이트 {a.gate} (TRAIN 5분위)")
    print(f"{'창':>10} " + " ".join(f"{'분위'+str(j):>16}" for j in range(5)) + f" {'단조ρ':>7}")
    for w in WINS:
        r = real[w]
        cells = " ".join(f"{b['pair_bp']:>+8.1f}(t{b['block_t']:>+5.2f})" for b in r["buckets"])
        print(f"{w:>10} {cells} {r['spearman']:>+7.2f}")
    p_low, p_mono = passes(real, a.side)
    print(f"\n실제: {a.side}분위 5창 양수 {p_low} · 단조 일관 {p_mono}")

    # ③ 블록 순열 귀무 -- 게이트를 **날 단위로** 섞는다(겹침·자기상관 보존, 라벨은 그대로)
    rng = np.random.default_rng(20260914)
    nday = len(gval) // DAY + 1
    hits_low = hits_mono = 0
    for b in range(a.boot):
        order = rng.permutation(nday)
        idx_map = np.concatenate([np.arange(k * DAY, min((k + 1) * DAY, len(gval))) for k in order])
        gshuf = np.empty_like(gval); gshuf[:len(idx_map)] = gval[idx_map]
        r = evaluate(lab, gshuf, edges)
        lo_ok, mo_ok = passes(r, a.side)
        hits_low += lo_ok; hits_mono += mo_ok
        if (b + 1) % 50 == 0:
            print(f"  귀무 {b+1}/{a.boot}: 최저분위통과 {hits_low} · 단조통과 {hits_mono}", flush=True)
    print(f"\n귀무 {a.boot}회: {a.side}분위 5창 양수 {hits_low} ({hits_low/a.boot:.1%}) · "
          f"단조 일관 {hits_mono} ({hits_mono/a.boot:.1%})")
    out = ROOT / "data/research/eth_direction_barrier_label_20260914"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}.json").write_text(json.dumps(
        {"cell": f"{a.up}/{a.down}", "gate": a.gate, "edges": edges.tolist(), "real": real,
         "pass_lowest": bool(p_low), "pass_mono": bool(p_mono), "boot": a.boot,
         "null_lowest": hits_low, "null_mono": hits_mono}, indent=1, ensure_ascii=False, default=float))
    print(f"저장: {out/(a.tag+'.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
