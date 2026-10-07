"""«크게 움직이나» 모델로 보유 길이 정하기 (2026-10-07, 사용자 «크게 움직이는지 예측하는 모델로 보유 규칙 테스트»).

결과 보기 전 고정:
  모델 = HGB(5분봉 33피쳐, research_eth_trend30_60_hold_policy 와 같은 피쳐) · 라벨 |다음 H분 수익| ≥ TRAIN 중앙 (H=30·60).
    big = p_big ≥ TRAIN 봉 p_big 중앙. 방향 모델 = 같은 피쳐 HGB 부호(규칙 C 용).
  규칙(진입마다 짧게 Hs=10분 · 길게 Hl=30/60분, 청산 수 같아 비용 상쇄):
    A 피하기: big → 짧게, 아니면 길게 · B 태우기: big → 길게, 아니면 짧게 · C: big 이고 방향 정렬일 때만 길게.
  판정 = 무작위(같은 «길게» 비율) 대비 이득의 일 블록 CI. A/B 중 주 판정은 VAL(핑퐁 후보 2024-07~12)에서 고른 쪽.
  진입 집합: 핑퐁 페이드 후보 VAL 2024H2 · TEST 2025~ · 사용자 실원장 141왕복(2026-07-20~10-04).
실행: python scripts/research_eth_bigmove_hold_policy_20261007.py [--selftest]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_eth_trend30_60_hold_policy_20261007 import (PREV, SEED, VA_END, TR_END, boot_ci, features, hold_gain,  # noqa: E402
                                                          load_1m, ms, to_5m)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/bigmove_hold_20261007"
HOLDS = {30: (10, 30), 60: (10, 60)}


def policies(big: np.ndarray, aligned: np.ndarray) -> dict[str, np.ndarray]:
    """1 = 길게 보유."""
    return {"A_피하기": (~big).astype(float), "B_태우기": big.astype(float), "C_big&정렬": (big & aligned).astype(float)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    k1 = load_1m(); b = to_5m(k1); X = features(b)
    t, c = b.t.to_numpy(), b.c.to_numpy()
    PB, PD, THR = {}, {}, {}
    for h in HOLDS:
        n = h // 5
        fwd = np.log(np.roll(c, -n) / c) * 1e4; fwd[-n:] = np.nan
        valid = np.isfinite(fwd) & (fwd != 0) & np.isfinite(X.ret288.to_numpy())
        tr = valid & (t >= ms("2022-04-01")) & (t < TR_END - 12 * 300_000); te = valid & (t >= VA_END)
        yb = (np.abs(fwd) >= np.median(np.abs(fwd[tr]))).astype(int)
        hgb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=False, random_state=SEED)  # noqa: E731
        PB[h] = hgb().fit(X[tr], yb[tr]).predict_proba(X)[:, 1]
        PD[h] = hgb().fit(X[tr], (fwd[tr] > 0).astype(int)).predict_proba(X)[:, 1]
        THR[h] = float(np.median(PB[h][tr]))
        print(f"[{h}분 크기 모델] TEST AUC {roc_auc_score(yb[te], PB[h][te]):.4f} · big 문턱 {THR[h]:.3f}", flush=True)

    c1, t1, bt = k1.c.to_numpy(), k1.t.to_numpy(), b.t.to_numpy()

    def run(name, t_dec, side) -> dict:
        bi = np.searchsorted(bt, t_dec // 300_000 * 300_000 - 300_000)
        ei = np.searchsorted(t1, t_dec // 60_000 * 60_000)
        ok = (bi < len(bt)) & (ei + 61 < len(c1))
        bi, ei, side, day = bi[ok], ei[ok], side[ok], t_dec[ok] // 86_400_000
        out = dict(n=int(ok.sum()), days=int(len(np.unique(day))))
        print(f"\n[{name}] n={out['n']} 일수={out['days']}")
        for h, (hs, hl) in HOLDS.items():
            r_s = np.log(c1[ei + hs] / c1[ei]) * 1e4; r_l = np.log(c1[ei + hl] / c1[ei]) * 1e4
            big = PB[h][bi] >= THR[h]; aligned = (PD[h][bi] - 0.5) * side > 0
            mag_auc = roc_auc_score(np.abs(r_l) >= np.median(np.abs(r_l)), PB[h][bi])
            g = side * (r_l - r_s)
            out[f"H{hs}/{hl}"] = d = dict(
                big_share=float(big.mean()), mag_auc_at_entries=float(mag_auc),
                long_minus_short_if_big=float(g[big].mean()), long_minus_short_if_small=float(g[~big].mean()),
                always_short=float((side * r_s).mean()), always_long=float((side * r_l).mean()))
            print(f"  H{hs}/{hl}: big 비중 {d['big_share']:.2f} · 진입점 크기 AUC {mag_auc:.3f} · (길게−짧게) big {d['long_minus_short_if_big']:+.2f} / "
                  f"작음 {d['long_minus_short_if_small']:+.2f}bp · 항상짧게 {d['always_short']:+.2f} · 항상길게 {d['always_long']:+.2f}")
            for pn, a in policies(big, aligned).items():
                pol, _, _, gain = hold_gain(side, a, r_s, r_l)
                d[pn] = dict(long_share=float(a.mean()), policy_bp=float(pol.mean()), gain_vs_random=float(gain.mean()),
                             gain_ci=boot_ci(gain, day, rng), vs_always_long_ci=boot_ci(pol - side * r_l, day, rng))
                print(f"    {pn:10s} 길게 {a.mean():.2f} · 규칙 {pol.mean():+.2f}bp · 무작위 대비 {gain.mean():+.2f} "
                      f"[{d[pn]['gain_ci'][0]:+.2f},{d[pn]['gain_ci'][1]:+.2f}] · 항상길게 대비 {(pol - side * r_l).mean():+.2f} "
                      f"[{d[pn]['vs_always_long_ci'][0]:+.2f},{d[pn]['vs_always_long_ci'][1]:+.2f}]")
        return out

    C = pd.read_parquet(PREV / "tmp/pingpong_extract_20261007/candidates.parquet"); C = C[C.pp]
    res = {"thr": THR}
    for nm, m in (("pingpong_VAL_2024H2", (C.t >= TR_END) & (C.t < VA_END)), ("pingpong_TEST_2025~", C.t >= VA_END)):
        cc = C[m]
        res[nm] = run(nm, cc.t.to_numpy("int64") + 60_000, -cc.d.to_numpy(int))
    T = pd.read_parquet(PREV / "tmp/ledger_micro_20261007/trips.parquet")
    res["ledger"] = run("실원장 141왕복", T.t0.to_numpy("int64"), T.d.to_numpy(int))
    pick = {h: max(("A_피하기", "B_태우기"), key=lambda p: res["pingpong_VAL_2024H2"][f"H10/{h}"][p]["gain_vs_random"]) for h in HOLDS}
    res["val_pick"] = pick
    print("\nVAL 이 고른 규칙:", pick)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def selftest() -> None:
    p = policies(np.array([True, False, True]), np.array([True, True, False]))
    assert p["A_피하기"].tolist() == [0, 1, 0] and p["B_태우기"].tolist() == [1, 0, 1] and p["C_big&정렬"].tolist() == [1, 0, 0]
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
