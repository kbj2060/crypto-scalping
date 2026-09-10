#!/usr/bin/env python3
"""쏠림 페이드 **동결 설계**를 확장 표본(2021-12~)에서 검정 (2026-09-10).

09-08 의 `research_xsec_crowding_hold_amortize_20260908.py` 를 그대로 옮기되 셋을 바꾼다:
 ① 패널을 **179종 × 2021-12~2026-08** 확장본으로
 ② 구간을 **EXT / IN / OUT** 3분할 — EXT(2021-12~2023-12)는 **설계할 때 한 번도 안 본 구간**이다
 ③ 🔴**부트스트랩을 이동블록으로 교체.** 원본은 iid 였고 기억에
    *"겹친 계열에 iid 부트스트랩 = 착시. 이동블록(블록≥H)에서는 0/28"* 로 기록돼 있다.
    5일 보유를 6시간 스태거로 겹쳐 쌓으므로 일 계열 ACF(1) 이 +0.67~+0.84 다. **ACF 를 같이 찍는다.**

설계는 **동결이다 — 여기서 새로 고르지 않는다.** 격자를 도는 것은 09-08 이 이미 고정한 축
(신호 2 × H 4 × k 2 × 가중 2 × 문턱 2)을 확장 표본에서 **재확인**하기 위함이지 재선택이 아니다.
헤드라인은 09-08 이 고른 `tt_count · 5일 · k3 · 신호가중 · ≥$200M` 하나다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_ext_20260910"
OUTD = ROOT / "tmp/xsec_crowding_ext_20260910"
EXT_A, EXT_END = "2022-02-01", "2023-12-30"   # 확장 구간
# 🔴**2022 에는 상위트레이더 롱숏비가 없다.** 바이낸스가 그 해 대부분 발행하지 않았고(2022-06·12 만 예외)
# 컬럼은 있으나 값이 빈 문자열이다. 파싱 버그가 아니라 **원천 데이터 부재**다.
# 전체계정(개미) 롱숏비는 **2022-02 부터** 정상(월 131~143종)이라 확장 구간은 거기서 시작한다.
# 2022-01 은 두 신호 모두 1종뿐이라 제외한다.
# ⇒ 확장 검정의 주 신호는 `retail` 이다. 09-08 기록상 tt_count 와 거의 동률(+36.5 vs +37.0)이고
#   **순@12bp CI 하한이 0 을 넘은 유일한 비공학 셀**이었다.
OOS_A, OOS_B = "2025-09-01", "2026-07-31"
H_GRID = (288, 1440)                     # 1일 · 5일(동결 설계)
STEP = 72
K_GRID = (3, 5)
DV_GRID = (5e7, 2e8)
COST_1D = 12.0
BOOT = 5000
SEED = 20260910
HEAD = ("retail", 1440, 3, True, 2e8)   # 확장 검정 헤드라인(2022 에 tt_count 가 없어 retail)


def log(m):
    print(f"[ext {time.strftime('%H:%M:%S')}] {m}", flush=True)


def block_ci(v, block, rng, B=BOOT):
    """⭐이동블록 부트스트랩. block ≥ 보유일수 라야 겹침 상관이 보존된다."""
    v = v[np.isfinite(v)]
    n = len(v)
    if n < max(30, 3 * block):
        return (np.nan, np.nan)
    nb = int(np.ceil(n / block))
    out = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n, nb)
        out[b] = v[np.concatenate([np.arange(s, s + block) % n for s in st])[:n]].mean()
    return tuple(np.percentile(out, [2.5, 97.5]))


def acf1(v):
    v = v[np.isfinite(v)]
    return float(np.corrcoef(v[:-1], v[1:])[0, 1]) if len(v) > 30 else np.nan


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Qm = z["Q"]
    mz = np.load(DIR / "metrics_panel.npz")
    def lg(X):
        return np.where(X > 0, np.log(np.maximum(X, 1e-9)), np.nan)
    SIGS = {"tt_count": lg(mz["count_toptrader_long_short_ratio"]),
            "retail": lg(mz["count_long_short_ratio"])}
    for k, v in SIGS.items():
        assert np.nanstd(v) > 0.01, f"{k} 신호가 상수/불리언 — := 우선순위 함정"
    DV = pd.DataFrame(Qm).rolling(288, min_periods=200).sum().to_numpy()
    log(f"패널 {Om.shape} · {ts[0]} → {ts[-1]}")

    rows = []
    log(f"{'신호':>9}{'보유':>6}{'k':>3}{'가중':>6}{'문턱':>7}{'구간':>5} {'일수':>5} "
        f"{'일총bp':>8}{'일비용':>7} {'일순bp [블록CI95]':>26} {'ACF1':>7}{'연샤프':>7}")
    for sig, S in SIGS.items():
        for H in H_GRID:
            fwd = np.full_like(Om, np.nan)
            fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
            T = H // STEP
            cost_day = COST_1D * 288.0 / H
            for DVt in DV_GRID:
                for k in K_GRID:
                    for sw in (False, True):
                        parts = []; width = []
                        for off in range(T):
                            tid = np.arange(600 + off * STEP, len(ts) - H - 2, H)
                            el = (DV[tid] >= DVt) & np.isfinite(S[tid]) & np.isfinite(fwd[tid])
                            nn = el.sum(1)
                            width.append(pd.Series(nn, index=ts[tid]))
                            sa = np.where(el, S[tid], np.nan); F = np.where(el, fwd[tid], np.nan)
                            order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
                            rr = np.flatnonzero(nn >= 2 * k + 2)
                            if len(rr) < 20:
                                parts = []; break
                            lo_i = order[rr][:, :k]
                            hi_i = order[rr][np.arange(len(rr))[:, None],
                                             (nn[rr][:, None] - 1 - np.arange(k)[None, :])]
                            fl = np.take_along_axis(F[rr], lo_i, 1)
                            fh = np.take_along_axis(F[rr], hi_i, 1)
                            if sw:      # 신호가중 = 횡단면 중앙값으로부터의 거리 (09-08 규약 그대로)
                                sl = np.take_along_axis(sa[rr], lo_i, 1)
                                sh = np.take_along_axis(sa[rr], hi_i, 1)
                                med = np.nanmedian(sa[rr], 1, keepdims=True)
                                wl = np.maximum(med - sl, 1e-6); wh = np.maximum(sh - med, 1e-6)
                                p = ((fl * wl).sum(1) / wl.sum(1)
                                     - (fh * wh).sum(1) / wh.sum(1)) / 2 * 1e4
                            else:
                                p = (fl.mean(1) - fh.mean(1)) / 2 * 1e4
                            parts.append(pd.Series(p * 288.0 / H, index=ts[tid][rr]))
                        if not parts:
                            continue
                        ser = pd.concat(parts).sort_index()
                        d = ser.groupby(ser.index.floor("D")).mean()
                        wser = pd.concat(width).sort_index()
                        wd = wser.groupby(wser.index.floor("D")).median().reindex(d.index)
                        idx = d.index; v_all = d.to_numpy().astype(float)
                        blk = max(H // 288, 1) * 2          # 블록 ≥ 보유일수(여유 2배)
                        # ⭐ALL = 전 구간 풀링. 설계가 **동결**이라 더 고를 게 없으므로
                        # 전 구간이 확인창이고, 확장이 검정력을 사는 방식은 바로 이 풀링이다.
                        # 구간별 값도 함께 내서 국면 의존을 숨기지 않는다.
                        for seg, m in (("EXT", (idx >= EXT_A) & (idx <= EXT_END)),
                                       ("IN", (idx > EXT_END) & (idx < OOS_A)),
                                       ("OUT", (idx >= OOS_A) & (idx <= OOS_B)),
                                       ("ALL", (idx >= EXT_A) & (idx <= OOS_B))):
                            v = v_all[np.asarray(m)]; v = v[np.isfinite(v)]
                            if len(v) < 60:
                                continue
                            net = v - cost_day
                            lo, hi = block_ci(net, blk, rng)
                            sd = v.std(); shp = net.mean() / sd * np.sqrt(365) if sd > 0 else np.nan
                            is_head = (sig, H, k, sw, DVt) == HEAD
                            w = float(np.nanmedian(wd.to_numpy()[np.asarray(m)])) \
                                if wd.notna().any() else float("nan")
                            log(f"{sig:>9}{H//288:>5}일{k:>3}{'신호' if sw else '동일':>6}"
                                f"${DVt/1e6:>5.0f}M{seg:>5} {len(v):>5} {v.mean():>+8.1f}"
                                f"{cost_day:>7.1f} {net.mean():>+8.1f}[{lo:>+7.1f},{hi:>+7.1f}] "
                                f"{acf1(v):>7.2f}{shp:>7.2f}{w:>6.0f}종"
                                + ("  ⭐헤드라인" if is_head else ""))
                            rows.append(dict(sig=sig, Hd=H // 288, k=k, sw=sw, dv=DVt, seg=seg,
                                             n=len(v), xwidth=w, gross=float(v.mean()), cost=cost_day,
                                             net=float(net.mean()), lo=float(lo), hi=float(hi),
                                             acf1=acf1(v), sharpe=float(shp), head=is_head))
    R = pd.DataFrame(rows)
    R.to_csv(OUTD / "extended.csv", index=False)
    log("=" * 110)
    for seg in ("EXT", "IN", "OUT", "ALL"):
        s = R[R.seg == seg]
        if not len(s):
            continue
        log(f"{seg}: 셀 {len(s)} · **순 블록CI 하한>0 {int((s.lo>0).sum())}** · 일수 중앙 {s.n.median():.0f}")
    # 세 구간 모두 통과
    piv = R.pivot_table(index=["sig", "Hd", "k", "sw", "dv"], columns="seg", values="lo")
    three = [c for c in ("EXT", "IN", "OUT") if c in piv.columns]
    allpos = piv[(piv[three] > 0).all(axis=1)] if len(three) == 3 else piv.iloc[:0]
    log(f"⭐⭐**세 구간 전부 CI 하한>0: {len(allpos)}건** (1차 기준)")
    if "ALL" in piv.columns:
        log(f"⭐풀링(ALL)에서 CI 하한>0: {int((piv['ALL'] > 0).sum())}/{len(piv)}건 "
            f"— 국면 의존을 숨길 수 있으니 **구간별과 함께만** 읽는다")
    if len(allpos):
        log(allpos.round(2).to_string())
    h = R[R["head"]]      # ⚠️`R.head` 는 pandas 메서드다 — 반드시 대괄호
    log(f"⭐헤드라인 구성({HEAD[0]} · {HEAD[1]//288}일 · k{HEAD[2]} · {'신호가중' if HEAD[3] else '동일가중'} · ≥${int(HEAD[4]/1e6)}M):")
    for _, r in h.iterrows():
        log(f"    {r.seg:>4} 일수 {r.n:>4.0f} 총 {r.gross:>+7.1f} 순 {r['net']:>+7.1f} "
            f"CI[{r.lo:>+7.1f},{r.hi:>+7.1f}] ACF1 {r.acf1:.2f} 샤프 {r.sharpe:.2f}")
    (OUTD / "summary.json").write_text(json.dumps(
        {"pass_by_seg": {s: int((R[R.seg == s].lo > 0).sum()) for s in ("EXT", "IN", "OUT", "ALL")},
         "pass_all_three": len(allpos),
         "pass_pooled": int((piv["ALL"] > 0).sum()) if "ALL" in piv.columns else None},
        ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
