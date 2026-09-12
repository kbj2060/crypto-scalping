"""진단 — balancedish 라벨이 상승 구간을 chop 으로 찍는 이유.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계기: 사용자 지적(2026-09-09) "3번째 차트에서 bull 구간을 chop 으로 선택한게 너무 안좋은데 왜?"
(3번째 패널 = 횡보 구간 · balgbm)

가설
----
`experiment_regime3_current_hmm_wide24_20260529._current_labels3_thresholded` 의 마지막 줄이
**앞선 bull/bear 결정을 덮어쓴다**:

    labels = 2(chop)                                   # 기본
    labels[trending & (slope > +slope_min)] = 0        # bull
    labels[trending & (slope < -slope_min)] = 1        # bear
    labels[(adx < weak_adx_max) | (bb_width < tight_bb_max)] = 2   # ← 마지막, 덮어씀

즉 ADX≥16 이고 기울기가 충분해 bull 로 찍힌 봉이라도 `bb_width < 0.012` 면 **강제로 chop**.
완만하고 매끄러운 상승은 볼린저 폭이 좁아지므로 정확히 이 덫에 걸린다.

이 스크립트가 하는 일
--------------------
차트의 그 구간(OOS 안에서 |5일 순수익| 최소인 5일 창 — 차트와 동일한 자동 선택 규칙)에서
· 덮어쓰기가 실제로 몇 번 일어났는가(bull/bear 로 찍혔다가 chop 으로 되돌려진 봉)
· 되돌린 조건이 ADX 인가 BB 인가
· 그 구간에서 상승이 실제로 얼마나 있었는가
를 센다. OOS 전체에서도 같은 통계를 낸다.

준수: 신규 학습 없음, 라벨 함수는 원본 import. 라이브 파일 미변경.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_regime3_current_hmm_wide24_20260529 import _adx, _current_labels3_thresholded, _num  # noqa: E402

HMM_MODEL = (ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
                  / "regime3_current_sensitive_hmm_wide24_2024.joblib")
BAL_DIR = ROOT / "data/ensemble/supervised/omega461_balgbm_cut2509_20260909"
PREFIX = "regime3_balgbm_cut2509_"
CLASSES = ("bull", "bear", "chop")
OOS = ("2026-01-01", "2026-02-28 23:55:00")
WIN_BARS = 5 * 288
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"


def stages(df: pd.DataFrame, cfg: dict):
    """라벨 함수를 단계별로 재현해 '덮어쓰기 전/후'를 분리한다."""
    close, high, low = _num(df, "close"), _num(df, "high"), _num(df, "low")
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema_slope = ((ema21 - ema21.shift(5)) / (close * 5.0 + 1e-12)
                 ).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    adx = _num(df, "adx_14", np.nan)
    if adx.isna().all():
        adx = _adx(high, low, close)
    adx = adx.fillna(0.0).to_numpy()
    bb = _num(df, "bb_width", np.nan)
    if bb.isna().all():
        sma20 = close.rolling(20, min_periods=5).mean()
        bb = 2.0 * close.rolling(20, min_periods=5).std() / (sma20 + 1e-12)
    bb = bb.fillna(0.0).to_numpy()

    trending = adx >= float(cfg["trend_adx_min"])
    smin = float(cfg["slope_min"])
    pre = np.full(len(df), 2, dtype=np.int64)          # 덮어쓰기 전
    pre[trending & (ema_slope > smin)] = 0
    pre[trending & (ema_slope < -smin)] = 1
    weak_adx = adx < float(cfg["weak_adx_max"])
    tight_bb = bb < float(cfg["tight_bb_max"])
    override = weak_adx | tight_bb
    post = pre.copy()
    post[override] = 2                                  # 마지막 줄
    return {"pre": pre, "post": post, "adx": adx, "bb": bb, "slope": ema_slope,
            "weak_adx": weak_adx, "tight_bb": tight_bb, "override": override}


def summarize(tag, st, close):
    pre, post = st["pre"], st["post"]
    killed = (pre != 2) & (post == 2)                   # 추세로 찍혔다가 chop 으로 되돌려진 봉
    n = len(pre)
    kb = killed & (pre == 0)
    kr = killed & (pre == 1)
    only_bb = killed & st["tight_bb"] & ~st["weak_adx"]
    only_adx = killed & st["weak_adx"] & ~st["tight_bb"]
    both = killed & st["weak_adx"] & st["tight_bb"]
    out = {"bars": int(n),
           "pre_trend_share": round(float((pre != 2).mean()), 4),
           "post_trend_share": round(float((post != 2).mean()), 4),
           "overridden_bars": int(killed.sum()),
           "overridden_share_of_all": round(float(killed.mean()), 4),
           "overridden_share_of_pre_trend": round(float(killed.sum() / max((pre != 2).sum(), 1)), 4),
           "of_which_bull": int(kb.sum()), "of_which_bear": int(kr.sum()),
           "cause_tight_bb_only": int(only_bb.sum()), "cause_weak_adx_only": int(only_adx.sum()),
           "cause_both": int(both.sum()),
           "bb_median_on_overridden": round(float(np.median(st["bb"][killed])), 5) if killed.any() else None,
           "adx_median_on_overridden": round(float(np.median(st["adx"][killed])), 2) if killed.any() else None}
    print(f"\n[{tag}] {n:,}봉  순수익 {(close[-1]/close[0]-1)*100:+.2f}%", flush=True)
    print(f"  덮어쓰기 전 추세비중 {out['pre_trend_share']*100:5.1f}%  →  후 {out['post_trend_share']*100:5.1f}%", flush=True)
    print(f"  ⭐추세로 찍혔다가 chop 으로 되돌려진 봉: {out['overridden_bars']:,} "
          f"({out['overridden_share_of_all']*100:.1f}% of all, "
          f"**{out['overridden_share_of_pre_trend']*100:.1f}% of 추세 후보**)", flush=True)
    print(f"     그중 bull {out['of_which_bull']:,} / bear {out['of_which_bear']:,}", flush=True)
    print(f"  원인:  BB만 {out['cause_tight_bb_only']:,}  |  ADX만 {out['cause_weak_adx_only']:,}  "
          f"|  둘다 {out['cause_both']:,}", flush=True)
    print(f"  되돌려진 봉의 중앙값:  bb_width {out['bb_median_on_overridden']}  "
          f"(임계 0.012)   adx {out['adx_median_on_overridden']} (임계 12)", flush=True)
    return out


def main() -> int:
    cfg = joblib.load(HMM_MODEL)["label_config"]
    print(f"[라벨 설정] {cfg}", flush=True)

    b = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
                    low_memory=False, parse_dates=["timestamp"])
    s = pd.read_csv(BAL_DIR / f"training_features_2026_rebuilt_{PREFIX}sidecar.csv",
                    parse_dates=["timestamp"],
                    usecols=["timestamp"] + [f"{PREFIX}{c}_prob" for c in CLASSES])
    df = b.merge(s, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    df = df[(df.timestamp >= OOS[0]) & (df.timestamp <= OOS[1])].reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="raise").to_numpy(np.float64)

    # 차트와 동일한 자동 선택 규칙
    net = np.full(len(close), np.nan)
    net[:-WIN_BARS] = np.abs(close[WIN_BARS:] - close[:-WIN_BARS]) / close[:-WIN_BARS]
    ok = np.isfinite(net)
    i_ch = int(np.nanargmin(np.where(ok, net, np.inf)))
    print(f"[3번 패널 구간] {df.timestamp.iloc[i_ch]} ~ {df.timestamp.iloc[i_ch+WIN_BARS-1]}", flush=True)

    st_all = stages(df, cfg)
    rep = {"label_config": cfg, "oos_full": summarize("OOS 전체", st_all, close)}

    sl = slice(i_ch, i_ch + WIN_BARS)
    sub = df.iloc[sl].reset_index(drop=True)
    st_win = stages(sub, cfg)     # 창 안에서 재계산(롤링 워머업이 잘리는 건 감안)
    rep["chop_window"] = summarize("3번 패널(횡보 구간)", st_win, close[sl])
    rep["chop_window"]["range"] = [str(df.timestamp.iloc[i_ch]), str(df.timestamp.iloc[i_ch + WIN_BARS - 1])]

    # 그 창에서 실제 상승이 있었는가 -- 되돌려진 bull 봉들의 이후 수익
    killed_bull = (st_win["pre"] == 0) & (st_win["post"] == 2)
    c = close[sl]
    for h in (12, 48, 288):
        fwd = np.full(len(c), np.nan)
        fwd[:-h] = (c[h:] - c[:-h]) / c[:-h]
        m = killed_bull & np.isfinite(fwd)
        if m.any():
            print(f"     되돌려진 bull 봉의 h{h} 전방수익 평균 {np.mean(fwd[m])*1e4:+.1f}bp (n={int(m.sum())})",
                  flush=True)
            rep["chop_window"][f"killed_bull_fwd_h{h}_bp"] = round(float(np.mean(fwd[m]) * 1e4), 2)

    (OUT / "balancedish_chop_override_diagnosis.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/balancedish_chop_override_diagnosis.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
