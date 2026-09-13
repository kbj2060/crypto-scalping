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
CONSTS = (2.0, 4.0, 5.0, 6.0, 8.0, 10.0)
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

    # ⭐**모델 전권** -- 배포 모델에서 상수 6배를 떼고 상한을 모델이 직접 정한다.
    # 지금 라이브는 min(100/safe_mae, 6) 이고 상한이 94% 를 묶는다. 이 팔은 그 6 을 없앤
    # 것이라 「모델이 크기를 정하게 하면 이기는가」에 대한 **가장 직접적인** 답이다.
    # 새 학습이 없다 -- 이미 검증된 아티팩트를 그대로 쓴다(표본외 초과율 0.096%).
    def model_only(i, side):
        m = float(sm[(240, side)][i])
        if not (m > 0):
            return 6.0
        return float(np.clip(SAFETY / m, CAP_LO, CAP_HI))
    arms.append(("모델 전권(상수 제거)", model_only, None))

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
    print("   상태의존이 상수를 이기는가뿐이다. 그게 `Δ프런티어` 열이다(✅ = 2SE 초과).\n")
    for wname, (w0, w1) in H.WINDOWS.items():
        lo_i = max(int(np.searchsorted(ts, np.datetime64(w0))), svm_warmup())
        hi_i = int(np.searchsorted(ts, np.datetime64(w1 + "T23:59:59")))
        print(f"{wname}  (봉 {hi_i-lo_i:,})")
        # 🔴**씨드를 짝짓는다.** 복리 로그성장은 분산이 거대해(같은 창에서 배율 중앙 5~86)
        # 팔 사이 차이를 삼킨다. 상수 프런티어도 4 씨드에서 비단조였다. 그래서 씨드마다
        # **그 씨드의 상수들로** 프런티어를 만들고 같은 씨드의 상태의존 팔과 뺀다.
        # 그 차이의 씨드간 평균 ± SE 가 판정값이다
        # ([[feedback_paired_seeds_required_when_side_is_injected_20260914]]).
        per_seed = {nm: [] for nm, _, _ in arms}
        for sd in range(a.seeds):
            for nm, fn, cst in arms:
                r = H.walk(d, sm, lo_i, hi_i, acc=a.acc, p_entry=1.0, use_stop=True,
                           use_ladder=True, selector=True,
                           cap_x=(cst if cst is not None else 6.0),
                           cap_fn=fn, rng=np.random.default_rng(H.SEED + 1000 * sd))
                per_seed[nm].append({"g": float(np.log(max(r["mult"], 1e-9))),
                                     "ex": r["expo_time_x"], "md": r["mdd"],
                                     "cp": r["mean_cap_x"], "tr": r["trades"],
                                     "wo": r["worst_trade_pct"], "ru": int(r["ruin"])})
        cn = [nm for nm, _, cst in arms if cst is not None]
        print(f"{'팔':>18} {'로그성장':>9} {'MDD':>7} {'평균상한':>8} {'시간노출':>8} "
              f"{'거래':>6} {'최악%':>7} {'Δ프런티어(짝지음)':>18}")
        for nm, fn, cst in arms:
            v = per_seed[nm]
            g = np.array([x["g"] for x in v]); ex = np.array([x["ex"] for x in v])
            gap = ""
            if cst is None:
                dd = []
                for k in range(a.seeds):        # 같은 씨드의 상수들로 프런티어를 만든다
                    fr = sorted((per_seed[c][k]["ex"], per_seed[c][k]["g"]) for c in cn)
                    fx = np.array([x for x, _ in fr]); fy = np.array([y for _, y in fr])
                    dd.append(v[k]["g"] - float(np.interp(v[k]["ex"], fx, fy)))
                dd = np.array(dd)
                se = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else 0.0
                mark = "✅" if dd.mean() - 2 * se > 0 else ("🔴" if dd.mean() + 2 * se < 0 else "–")
                gap = f"{dd.mean():+7.3f}±{se:<5.3f}{mark}"
            print(f"{nm:>18} {g.mean():>9.3f} {100*np.mean([x['md'] for x in v]):>6.1f}% "
                  f"{np.mean([x['cp'] for x in v]):>8.2f} {ex.mean():>8.2f} "
                  f"{np.mean([x['tr'] for x in v]):>6.0f} "
                  f"{np.mean([x['wo'] for x in v]):>6.1f}% {gap:>18}")
        print()
    return 0


def svm_warmup() -> int:
    import live_eth_sizing_vol_model_20260912 as svm
    return max(svm.WARMUP, H.ATR_BARS + 10)


if __name__ == "__main__":
    raise SystemExit(main())
