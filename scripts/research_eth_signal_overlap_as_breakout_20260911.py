"""증거신호 겹침 수 = 확신인가 돌파 전조인가.

사용자 경험: 증거신호 8종이 **모두 천장 발동**해서 숏 → 상방 폭발. 저장소 기록과 일치한다:
"3종+ 겹침이 최악"(바닥 -5.91 / 천장 -5.99bp @H4h), 그런데 "분류 lift 1.81~2.72배는 맞다".
겹칠수록 반전은 더 자주 나는데 손익은 더 나쁘다 → **결과 분포가 넓어진다 = 변동성 확장**.

⚠️ 실제 8종은 `/futures/data` 메트릭이 최근 41.7시간만 조회돼 과거 재구성시 조용히 0채움된다
(reference_binance_futures_metrics_history_sources_20260909). 그래서 **프레임의 실제 라이브
피쳐로 각 신호의 대리 지표**를 만든다 — 원본 8종이 아니라 대리임을 명시한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402

D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXPAND, Q = 0.7, 1.8, 0.95      # 극단 판정 분위
B_NULL, SEED = 400, 615372041


def main() -> int:
    f = retest.load_frame_current("2026-01-01", "2026-08-30")
    n = len(f)
    c = f["close"].to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()

    def _q(s):                                  # 상단/하단 극단 플래그
        v = pd.to_numeric(s, errors="coerce")
        hi = v >= v.rolling(2016, min_periods=288).quantile(Q)
        lo = v <= v.rolling(2016, min_periods=288).quantile(1 - Q)
        return hi.to_numpy(), lo.to_numpy()

    # 8종 대리 (천장 플래그, 바닥 플래그) — 프레임의 실제 라이브 피쳐 사용
    ret3 = pd.Series(c).pct_change(3)
    proxies = {
        "단기수익z(strz)": _q(ret3),
        "디마커류(dem)": _q(f["rsi_14"] if "rsi_14" in f else ret3.rolling(14).mean()),
        "칼만이탈(kal)": _q(f["kalman_velocity"]),
        "피보소진(fib)": _q(f["distance_to_day_high_low_pct"]),
        "테이커델타(taker)": _q(f["cvd_slope_12"]),
        "유동성스윕(sweep)": (f["sweep_prev_high_reclaim"].to_numpy() > 0,
                          f["sweep_prev_low_reclaim"].to_numpy() > 0),
        "SMT괴리(smt)": _q(f["eth_btc_beta_residual_z"]),
        "직교조합(orth)": _q(f["cvd_breakout_z"]),
    }
    top = np.zeros(n, dtype=int)
    bot = np.zeros(n, dtype=int)
    for nm, (hi, lo) in proxies.items():
        top += np.nan_to_num(hi, nan=0).astype(int)
        bot += np.nan_to_num(lo, nan=0).astype(int)
    print(f"[대리 8종] 프레임 {n:,}봉 · 극단 분위 {Q:.0%}")
    print(f"  천장 겹침 분포 {dict(pd.Series(top).value_counts().sort_index().head(9))}")
    print(f"  바닥 겹침 분포 {dict(pd.Series(bot).value_counts().sort_index().head(9))}", flush=True)

    HZ = [(12, "1시간"), (48, "4시간"), (144, "12시간"), (288, "1일"), (864, "3일")]
    ok = np.isfinite(volexp)
    comp = ok & (volexp < COMPRESS)
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(300, n - 300, size=B_NULL)

    for H, hn in HZ:
        fut = pd.Series(volexp).rolling(H).max().shift(-H).to_numpy()
        r = (pd.Series(c).shift(-H) / pd.Series(c) - 1).to_numpy()      # 앞 H봉 수익
        for nm, cnt, sgn in (("천장(숏)", top, +1), ("바닥(롱)", bot, -1)):
            adv = r * sgn                                                # 신호 반대로 간 크기
            v = comp & np.isfinite(fut) & np.isfinite(r)
            be = float(np.mean(fut[v] >= EXPAND))
            ba = float(np.mean(adv[v] > 0))
            btail = float(np.mean(adv[v] > 0.02))                        # 반대로 2% 이상
            bmean = float(np.mean(adv[v]) * 100)
            print(f"\n=== {hn} · {nm} · 압축 {int(v.sum()):,}봉  "
                  f"기저: 확장 {be*100:.2f}% · 반대 {ba*100:.1f}% · 반대2%+ {btail*100:.2f}% · "
                  f"평균반대 {bmean:+.3f}% ===")
            print(f"  {'겹침':>4s} {'봉수':>7s} {'확장lift':>8s} {'반대율':>7s} {'반대2%+':>8s} "
                  f"{'lift':>6s} {'평균반대':>8s} {'p(꼬리)':>8s}")
            for k in range(1, 5):
                m = v & (cnt >= k)
                if m.sum() < 100:
                    continue
                e = float(np.mean(fut[m] >= EXPAND))
                a = float(np.mean(adv[m] > 0))
                tl = float(np.mean(adv[m] > 0.02))
                mn = float(np.mean(adv[m]) * 100)
                nt = []
                for sh in shifts[:200]:
                    mm = v & (np.roll(cnt, int(sh)) >= k)
                    if mm.sum() >= 50:
                        nt.append(float(np.mean(adv[mm] > 0.02)))
                pt = float((np.asarray(nt) >= tl).mean()) if nt else np.nan
                print(f"  {k:4d}+ {int(m.sum()):7,d} {e/max(be,1e-9):7.2f}x {a*100:6.1f}% "
                      f"{tl*100:7.2f}% {tl/max(btail,1e-9):5.2f}x {mn:+7.3f}% {pt:8.3f}", flush=True)
    print("\n대리 지표다 — 원본 8종이 아니다. 방향은 '신호가 말한 반대로 갔는가'다.")
    print("겹칠수록 확장률이 오르면 '겹침 = 돌파 전조' 가설이 산다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
