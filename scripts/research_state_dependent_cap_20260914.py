"""**상한을 상태의 함수로 만들면 상수 6배를 이기는가** (2026-09-14, 사용자 요청).

사용자: *"거의 다 상수라는 의미인데 모델링으로 진행 안돼? 현재 상황 데이터를 보고 지평,
레버리지, 증거금 상한을 정해주는 모델을 만들어서 테스트해줘."*

## 축은 사실 하나다
명목 = 증거금 × 배수라 «레버리지」와 «증거금 상한」은 같은 축이다. 그리고 **지평은 이미
모델이 정한다**(`planning_hold` -> 선택기, 고정 지평 대비 5-1 승). 실측으로 상한이 묶는
비율이 롱 93.9% · 숏 96.0% 라, 열려 있는 건 **상한 하나**다. 그것만 잰다.

## 순서 -- 천장부터
① **상수 스윕**: 2~8배. 6 이 어디 서 있고 곡선이 평평한가. 평평하면 어떤 모델도 못 번다.
② **위험 오라클**: 다가올 거래의 **실제 MAE 를 미리 알고** 상한을 정한다. 방향 오라클이
   아니다 -- 방향은 이 저장소가 rho 0.000 으로 닫았고, 그걸 오라클에 넣으면 «방향을 알면
   돈을 번다」는 동어반복이 된다. 잴 것은 «위험을 완벽히 알면 얼마나 나아지나」다.
③ **모델 없는 규칙**: 1/ATR · ATR 십분위. 모델은 이걸 이겨야 의미가 있다.

## 🔴판정 규칙 -- 크기 매칭
상한을 낮추면 MDD 는 **거저** 준다. 그러니 팔은 **같은 시간적분 노출**에서 비교해야 한다.
상수 스윕이 프런티어를 만들고, 상태의존 팔은 «같은 노출의 상수」를 이겨야 통과다.
목적함수는 bp 가 아니라 **로그 성장**이다(복리·파산 흡수).
씨드는 짝짓는다([[feedback_paired_seeds_required_when_side_is_injected_20260914]]).
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
_sp = importlib.util.spec_from_file_location(
    "stack", ROOT / "scripts" / "research_fresh_forward_random_entry_stack_20260914.py")
H = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(H)

SAFETY = 100.0
CONSTS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0)
# 🔴상태의존 팔의 범위. 상단을 6(배포값)으로 막으면 «위험이 낮을 때 키우기」가 원천봉쇄되어
# 오라클이 상수 6 과 동일해진다(2026-09-14 1차 실행에서 실제로 그랬다). 프런티어를 재려면
# 위아래로 움직일 수 있어야 한다.
CAP_LO, CAP_HI = 1.0, 15.0


def realized_mae(d: pd.DataFrame, bars: int) -> dict:
    """봉 i 에서 앞으로 `bars` 동안의 **실제** 최대 역행폭(%). 오라클 전용(미래 참조)."""
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    fmin = pd.Series(lo).iloc[::-1].rolling(bars, min_periods=bars).min().iloc[::-1].shift(-1)
    fmax = pd.Series(hi).iloc[::-1].rolling(bars, min_periods=bars).max().iloc[::-1].shift(-1)
    return {"LONG": (c - fmin.to_numpy()) / c * 100.0,
            "SHORT": (fmax.to_numpy() - c) / c * 100.0}


def make_arms(d: pd.DataFrame, sm: dict, train_hi: int):
    """팔 목록: (이름, cap_fn 또는 None(상수), 상수값)."""
    atr = sm["atr_pct"]
    rm = realized_mae(d, 48)                       # 오라클 기준 지평 = 240분(선택 최빈)
    arms = [(f"상수 {c:.0f}배" + (" (배포)" if c == 6.0 else ""), None, c) for c in CONSTS]

    def orc(i, side):
        m = rm[side][i]
        if not np.isfinite(m) or m <= 0:
            return 6.0
        return float(np.clip(SAFETY / m, CAP_LO, CAP_HI))

    arms.append(("위험 오라클(천장)", orc, None))

    # 🔴모델 없는 규칙 -- 계수는 **학습구간에서만** 정한다. 평균 상한이 6 근처가 되게 맞춘다.
    a_tr = atr[:train_hi]
    a_med = float(np.nanmedian(a_tr[np.isfinite(a_tr)]))

    def inv_atr(i, side, _k=a_med):
        a = atr[i]
        if not (a > 0):
            return 6.0
        return float(np.clip(6.0 * _k / a, CAP_LO, CAP_HI))
    arms.append(("규칙 1/ATR", inv_atr, None))

    # ATR 십분위 -> 상한 6..1 선형(변동성이 크면 작게). 경계는 학습구간 분위.
    edges = np.nanquantile(a_tr[np.isfinite(a_tr)], np.linspace(0, 1, 11)[1:-1])
    levels = np.linspace(6.0, 2.0, 10)

    def dec(i, side, _e=edges, _l=levels):
        a = atr[i]
        if not (a > 0):
            return 6.0
        return float(_l[int(np.searchsorted(_e, a))])
    arms.append(("규칙 ATR십분위", dec, None))
    return arms


def _self_check(d, sm) -> None:
    rm = realized_mae(d, 48)
    c = d.close.to_numpy(float); lo = d.low.to_numpy(float)
    i = 5000
    want = (c[i] - lo[i + 1:i + 49].min()) / c[i] * 100.0
    assert abs(rm["LONG"][i] - want) < 1e-9, (rm["LONG"][i], want)
    assert rm["LONG"][i] >= 0.0 and rm["SHORT"][i] >= 0.0
    # cap_fn 이 실제로 먹히는지 -- 상한 1배 팔은 6배 팔보다 노출이 작아야 한다
    lo_i, hi_i = 3000, 9000
    k = dict(acc=0.60, p_entry=1.0, use_stop=True, use_ladder=True, selector=True)
    a = H.walk(d, sm, lo_i, hi_i, cap_x=6.0, rng=np.random.default_rng(1), **k)
    b = H.walk(d, sm, lo_i, hi_i, cap_x=6.0, rng=np.random.default_rng(1),
               cap_fn=lambda i, s: 1.0, **k)
    assert b["expo_time_x"] < a["expo_time_x"] * 0.4, (a["expo_time_x"], b["expo_time_x"])
    assert abs(b["mean_cap_x"] - 1.0) < 1e-9, b["mean_cap_x"]
    print("통과 — 실제MAE 정의 · cap_fn 이 노출을 실제로 줄인다")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", type=float, default=0.60)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    d = H.load()
    sm = H.safe_mae_series(d)
    if a.self_check:
        _self_check(d, sm); return 0
    ts = d.timestamp.to_numpy()
    train_hi = int(np.searchsorted(ts, np.datetime64("2025-09-01")))
    arms = make_arms(d, sm, train_hi)
    print(f"5분봉 {len(d):,} · 정확도 {a.acc} · 씨드 {a.seeds} · 지평 선택기 ON · "
          f"손절 3% · 예산 사다리 ON")
    print("⚠️상한을 낮추면 MDD 는 거저 준다 -- **같은 노출의 상수**를 이겨야 통과다")
    print("⚠️acc 를 주입하면 «레버리지가 클수록 좋다」가 기계적으로 나온다(가짜 엣지의 복리).")
    print("   그래서 «어느 상수가 최선인가」는 여기서 못 답한다 -- 답하는 건 **같은 노출에서**")
    print("   상태의존이 상수를 이기는가뿐이다. 그게 `Δ프런티어` 열이다.\n")
    for wname, (w0, w1) in H.WINDOWS.items():
        lo_i = max(int(np.searchsorted(ts, np.datetime64(w0))), svm_warmup())
        hi_i = int(np.searchsorted(ts, np.datetime64(w1 + "T23:59:59")))
        print(f"{wname}  (봉 {hi_i-lo_i:,})")
        print(f"{'팔':>18} {'로그성장':>9} {'배율중앙':>9} {'MDD':>7} {'파산':>5} "
              f"{'평균상한':>8} {'시간노출':>8} {'거래':>6} {'차단':>7} {'최악%':>7} {'Δ프런티어':>10}")
        res = []
        for nm, fn, cst in arms:
            g, mu, md, ru, cp, ex, tr, wo, bl = ([] for _ in range(9))
            for sd in range(a.seeds):
                r = H.walk(d, sm, lo_i, hi_i, acc=a.acc, p_entry=1.0, use_stop=True,
                           use_ladder=True, selector=True,
                           cap_x=(cst if cst is not None else 6.0),
                           cap_fn=fn, rng=np.random.default_rng(H.SEED + 1000 * sd))
                g.append(np.log(max(r["mult"], 1e-9))); mu.append(r["mult"])
                md.append(r["mdd"]); ru.append(r["ruin"]); cp.append(r["mean_cap_x"])
                ex.append(r["expo_time_x"]); tr.append(r["trades"]); bl.append(r["blocked_margin"])
                wo.append(r["worst_trade_pct"])
            res.append({"nm": nm, "cst": cst, "g": float(np.mean(g)),
                        "mu": float(np.median(mu)), "md": float(np.mean(md)),
                        "ru": int(sum(ru)), "cp": float(np.mean(cp)),
                        "ex": float(np.mean(ex)), "tr": float(np.mean(tr)),
                        "bl": float(np.mean(bl)), "wo": float(np.nanmean(wo))})
        # 🔴상수 프런티어(노출 -> 로그성장). 상태의존 팔은 **같은 노출의 상수**를 이겨야 한다.
        fr = sorted([(r["ex"], r["g"]) for r in res if r["cst"] is not None])
        fx = np.array([x for x, _ in fr]); fy = np.array([y for _, y in fr])
        for r in res:
            gap = "" if r["cst"] is not None else f"{r['g'] - float(np.interp(r['ex'], fx, fy)):+10.3f}"
            print(f"{r['nm']:>18} {r['g']:>9.3f} {r['mu']:>9.2f} {100*r['md']:>6.1f}% "
                  f"{r['ru']:>4}/{a.seeds} {r['cp']:>8.2f} {r['ex']:>8.2f} "
                  f"{r['tr']:>6.0f} {r['bl']:>7.0f} {r['wo']:>6.1f}% {gap:>10}")
        print()
    return 0


def svm_warmup() -> int:
    import live_eth_sizing_vol_model_20260912 as svm
    return max(svm.WARMUP, H.ATR_BARS + 10)


if __name__ == "__main__":
    raise SystemExit(main())
