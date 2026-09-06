#!/usr/bin/env python3
"""증거신호 칩: 목표 도달 후 **재발동**했을 때 화면이 보여줄 확률 A/B 비교.

배경
----
`live_evidence_signal_metalabel_20260829.py`의 `cache_valid`(:470)는 측면과 경과 봉수만 본다 --
`tp_touched`를 보지 않는다. 그래서 앵커의 익절가가 이미 닿은 뒤에 같은 측면 원시 재발동이
horizon 안에 들어오면, 새 봉에서 재추론하지 않고 **옛 앵커의 확률/익절가/fire_pos를 그대로**
내보낸다. 화면에는 "바닥 발동"(방금 발동한 것처럼) + 이미 닿은 옛 익절가 + 괄호 친 옛 확률이
같이 뜬다(app.js:2237의 `isResolvedByTp`가 `!fired`를 요구해서 종료 표시로 안 넘어간다).

이 스크립트는 그 상황에서 두 후보를 **같은 결과에 대해** 채점한다.
  A(현행): 앵커 봉 a에서 계산된 확률을 그대로 표시
  B(수정안): 재발동 봉 j에서 재추론한 확률을 표시
결과 Y: **재발동 봉 j의 발동 자신의 학습 라벨**이 hit인가 -- 그 봉에 화면을 보는 사람이
        실제로 궁금해하는 유일한 질문이다. 두 확률 모두 이 Y로 채점한다.

주의(선택 편향): 모집단은 "앵커의 라벨이 이미 hit=1로 확정된" 사건만 모은 것이라, p_A는
정의상 hit이 난 앵커들에서 뽑힌 값이다. p_A의 수준이 높게 나오는 것 자체는 예측력이 아니라
이 선택의 결과다 -- 그래서 수준(보정)과 순위(AUC)를 나눠서 본다.

라벨 규약은 `live_evidence_signal_dashboard_20260823.py::compute_signals`의 HIT_RESOLUTION /
K_OVERRIDE / SUSTAIN_BARS_OVERRIDE를 그대로 따른다(터치 7종 + fib만 touch_and_mae).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from live_evidence_signal_metalabel_20260829 import METALABEL_SIGNALS  # noqa: E402
from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

DEVICE = "cpu"          # 라이브 대시보드와 GPU 경합을 만들지 않기 위해 CPU. cuda와 확률 차 ~0.001 확인.
FIB_K_LOSS_MULT = 2.0   # research_eth_fib_extension_exhaustion_metalabel_tabpfn_20260831.py:91
OUT = ROOT / "data/research/eth_chip_stale_anchor_proba_ab_20260907"

# compute_signals()가 들고 있는 값과 **수동 동기**(그 모듈은 함수 지역변수로 갖고 있어 임포트 불가)
K_OVERRIDE = {
    "taker_delta_z_climax": 2.00, "short_term_return_z": 1.75, "liquidity_sweep": 4.00,
    "orthogonal_combo": 3.571, "smt_divergence": 4.20, "fib_extension_exhaustion": 2.35,
    "demarker_extreme": 0.70, "kalman_deviation_meanrev": 2.5,
}
HIT_RESOLUTION = {name: ("touch_and_mae" if name == "fib_extension_exhaustion" else "touch")
                  for name in K_OVERRIDE}


def build_frames():
    eth = pd.read_csv(ROOT / "data/eth_5m_1year.csv", parse_dates=["timestamp"]).reset_index(drop=True)
    btc = pd.read_csv(ROOT / "data/btc_5m_1year.csv", parse_dates=["timestamp"]).reset_index(drop=True)
    sig = compute_signals(eth, btc)          # funding_df 없음 -> orthogonal_combo bottom은 delta_z 단독 (한계 명시)
    frame = build_indicator_frame(eth)
    frame["dem"] = compute_demarker(eth["high"], eth["low"]).to_numpy()
    levels, _ = kalman_level_and_velocity(eth["close"].to_numpy())
    frame["kalman_dev_z"] = rolling_zscore(pd.Series((eth["close"].to_numpy() - levels) / levels)).to_numpy()
    return eth, sig, frame


def make_label_fns(frame: pd.DataFrame):
    high, low, close = frame["high"].to_numpy(), frame["low"].to_numpy(), frame["close"].to_numpy()
    atr = frame["atr_pct"].to_numpy()
    n = len(frame)

    def tp_level(pos: int, side: str, k: float) -> float:
        if not np.isfinite(atr[pos]):
            return np.nan
        return close[pos] * (1 - k * atr[pos]) if side == "top" else close[pos] * (1 + k * atr[pos])

    def touched_between(lo: int, hi: int, lvl: float, side: str) -> bool:
        """(lo, hi] 구간에서 lvl 터치 여부 -- 라이브 `_tp_touched`와 같은 반열린 구간."""
        if not np.isfinite(lvl) or hi <= lo:
            return False
        seg_h, seg_l = high[lo + 1:hi + 1], low[lo + 1:hi + 1]
        return bool((seg_l <= lvl).any()) if side == "top" else bool((seg_h >= lvl).any())

    def label_at(pos: int, side: str, name: str) -> int | None:
        """발동 봉 pos의 **자기 라벨** hit. 창이 잘리면 None."""
        k, h = K_OVERRIDE[name], METALABEL_SIGNALS[name]["horizon_bars"]
        end = pos + h
        if end > n - 1 or not np.isfinite(atr[pos]):
            return None
        seg_h, seg_l = high[pos + 1:end + 1], low[pos + 1:end + 1]
        target = k * atr[pos]
        if side == "bottom":
            mfe = seg_h.max() / close[pos] - 1.0
            mae = 1.0 - seg_l.min() / close[pos]
        else:
            mfe = 1.0 - seg_l.min() / close[pos]
            mae = seg_h.max() / close[pos] - 1.0
        if HIT_RESOLUTION[name] == "touch":
            return int(mfe >= target)
        return int(mfe >= target and mae < FIB_K_LOSS_MULT * target)

    return tp_level, touched_between, label_at


def collect_events(sig: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """라이브 `_LAST_FIRE_CACHE` 사이클을 그대로 시뮬레이션해 '목표 도달 후 재발동' 사건을 모은다."""
    tp_level, touched_between, label_at = make_label_fns(frame)
    ts = frame["timestamp"].to_numpy()
    n = len(frame)
    rows = []
    for name, _ in SIGNAL_ORDER:
        if name not in METALABEL_SIGNALS:
            continue
        h, k = METALABEL_SIGNALS[name]["horizon_bars"], K_OVERRIDE[name]
        bf = sig[f"bottom_{name}"].fillna(False).to_numpy()
        tf = sig[f"top_{name}"].fillna(False).to_numpy()
        cache = None
        for j in range(n):
            b, t = bool(bf[j]), bool(tf[j])
            if not (b or t):
                continue
            side = "bottom" if b else "top"
            if cache is not None and cache["side"] == side and 0 <= (j - cache["pos"]) < h:
                a = cache["pos"]
                if touched_between(a, j, cache["tp"], side):      # ← 라이브 `_tp_touched`가 True가 되는 조건
                    y = label_at(j, side, name)
                    if y is not None:
                        rows.append(dict(signal=name, side=side, anchor_pos=a, refire_pos=j,
                                         age=j - a, ts=pd.Timestamp(ts[j]),
                                         day=str(pd.Timestamp(ts[j]).date()), y=y))
            else:
                cache = {"side": side, "pos": j, "tp": tp_level(j, side, k)}
    return pd.DataFrame(rows)


def predict(events: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """신호별로 TabPFN을 한 번 fit하고 앵커/재발동 봉을 **일괄** 추론(라이브는 봉마다 재fit하지만
    fit은 in-context 저장이라 결과가 동일하다 -- seed/컨텍스트/피쳐셋 모두 라이브와 같은 값)."""
    from tabpfn import TabPFNClassifier

    events = events.copy()
    events["p_A"] = np.nan
    events["p_B"] = np.nan
    for name, grp in events.groupby("signal"):
        cfg = METALABEL_SIGNALS[name]
        cols = cfg.get("feature_columns", FEATURE_COLUMNS)
        train = pd.read_csv(cfg["train_context"], parse_dates=["timestamp"])
        t0 = time.time()
        clf = TabPFNClassifier(device=DEVICE, random_state=cfg["seed"])
        clf.fit(train[cols], train["hit"].to_numpy().astype(int))

        need = pd.unique(np.concatenate([grp.anchor_pos.to_numpy(), grp.refire_pos.to_numpy()]))
        # is_bottom은 피쳐라 (pos, side) 쌍마다 달라진다 -- 두 축을 모두 키로 쓴다.
        keys = set(zip(grp.anchor_pos, grp.side)) | set(zip(grp.refire_pos, grp.side))
        keys = sorted(keys)
        base = [c for c in cols if c != "is_bottom"]     # is_bottom은 frame에 없다 -- 측면에서 만든다
        X = frame.iloc[[p for p, _ in keys]][base].copy().reset_index(drop=True)
        X["is_bottom"] = [1 if s == "bottom" else 0 for _, s in keys]
        X = X[cols]
        ok = ~X.isna().any(axis=1).to_numpy()
        proba = np.full(len(X), np.nan)
        if ok.any():
            proba[ok] = clf.predict_proba(X.loc[ok])[:, 1]
        lut = {kk: proba[m] for m, kk in enumerate(keys)}
        idx = grp.index
        events.loc[idx, "p_A"] = [lut[(p, s)] for p, s in zip(grp.anchor_pos, grp.side)]
        events.loc[idx, "p_B"] = [lut[(p, s)] for p, s in zip(grp.refire_pos, grp.side)]
        print(f"  {name:26s} n={len(grp):5d}  unique_rows={len(keys):5d}  "
              f"nan_feat={int((~ok).sum())}  {time.time()-t0:6.1f}s", flush=True)
    return events


# ------------------------------------------------------------------ 지표
def auc(p: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def clustered_boot(fn, ev: pd.DataFrame, B: int = 2000, seed: int = 20260907):
    """일(day) 단위 재표집 후 행 단위 재계산 -- 같은 날 여러 발동이 독립이 아니므로."""
    rng = np.random.default_rng(seed)
    udays = ev.day.unique()
    groups = {d: g.to_numpy() for d, g in ev.groupby("day").groups.items()}
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(udays, len(udays), replace=True)
        sub = ev.loc[np.concatenate([groups[d] for d in pick])]
        out[b] = fn(sub)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("데이터 로드 + 신호 계산 ...", flush=True)
    eth, sig, frame = build_frames()
    print(f"  {len(eth)}봉  {eth.timestamp.iloc[0]} ~ {eth.timestamp.iloc[-1]}", flush=True)

    ev = collect_events(sig, frame)
    print(f"\n사건(목표 도달 후 재발동) {len(ev)}건 · 독립일수 {ev.day.nunique()}일", flush=True)
    print("\nTabPFN 추론 (device=%s)" % DEVICE, flush=True)
    ev = predict(ev, frame)
    ev.to_csv(OUT / "events.csv", index=False)

    ok = ev.dropna(subset=["p_A", "p_B"]).copy()
    y = ok.y.to_numpy()
    print(f"\n채점 대상 {len(ok)}건 (피쳐 NaN 제외 {len(ev)-len(ok)}건) · 독립일수 {ok.day.nunique()}일")
    print(f"실제 hit률 Y = {y.mean()*100:.1f}%\n")

    print("=== 수준(보정) ===")
    print(f"  A 화면 확률 평균 {ok.p_A.mean()*100:5.1f}%   실제 {y.mean()*100:5.1f}%   과대 {(ok.p_A.mean()-y.mean())*100:+5.1f}pp")
    print(f"  B 화면 확률 평균 {ok.p_B.mean()*100:5.1f}%   실제 {y.mean()*100:5.1f}%   과대 {(ok.p_B.mean()-y.mean())*100:+5.1f}pp")

    print("\n=== 순위(판별력) · 정확도 ===")
    aA, aB = auc(ok.p_A.to_numpy(), y), auc(ok.p_B.to_numpy(), y)
    bA, bB = brier(ok.p_A.to_numpy(), y), brier(ok.p_B.to_numpy(), y)
    lo, hi = clustered_boot(lambda d: auc(d.p_B.to_numpy(), d.y.to_numpy()) - auc(d.p_A.to_numpy(), d.y.to_numpy()), ok)
    lo2, hi2 = clustered_boot(lambda d: brier(d.p_B.to_numpy(), d.y.to_numpy()) - brier(d.p_A.to_numpy(), d.y.to_numpy()), ok)
    loA, hiA = clustered_boot(lambda d: auc(d.p_A.to_numpy(), d.y.to_numpy()), ok)
    loB, hiB = clustered_boot(lambda d: auc(d.p_B.to_numpy(), d.y.to_numpy()), ok)
    print(f"  AUC   A {aA:.4f} [{loA:.4f},{hiA:.4f}]   B {aB:.4f} [{loB:.4f},{hiB:.4f}]")
    print(f"        B-A {aB-aA:+.4f}  일군집95%CI [{lo:+.4f}, {hi:+.4f}]  {'← 0 포함' if lo<0<hi else '← 0 제외'}")
    print(f"  Brier A {bA:.4f}   B {bB:.4f}   B-A {bB-bA:+.4f}  95%CI [{lo2:+.4f}, {hi2:+.4f}]  (낮을수록 좋음)")

    print("\n=== 신호별 ===")
    recs = []
    for name, g in ok.groupby("signal"):
        recs.append(dict(signal=name, n=len(g), 일수=g.day.nunique(), 실제hit=round(g.y.mean()*100, 1),
                         A확률=round(g.p_A.mean()*100, 1), B확률=round(g.p_B.mean()*100, 1),
                         A_AUC=round(auc(g.p_A.to_numpy(), g.y.to_numpy()), 3),
                         B_AUC=round(auc(g.p_B.to_numpy(), g.y.to_numpy()), 3)))
    tbl = pd.DataFrame(recs).sort_values("n", ascending=False)
    print(tbl.to_string(index=False))
    tbl.to_csv(OUT / "per_signal.csv", index=False)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
