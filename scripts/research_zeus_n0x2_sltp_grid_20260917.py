#!/usr/bin/env python3
"""Zeus — **N0x2 베이스라인의 TP/SL 격자** (2026-09-17, 학습 0)

TP1.5%/SL1% 는 **배포 부모(zig075) 추론** 위에서 고른 값이다 —
`docs/experiments/omega461_exit_barrier_design_20260917.md` §6 이 남긴 구멍:
「이 숫자는 전부 배포 방향머리 기준이다」. N0x2 는 balnobb 라우팅으로 새로 학습한
6개 모델이라 발화하는 봉이 다르므로 같은 배리어가 최적이라는 보장이 없다.

⭐**재학습이 필요 없다.** 더블배리어는 학습 라벨이 아니라 **청산 규칙**이고 각 팔의 두
머리는 이미 학습돼 캐시에 있다. TP/SL 을 바꿔도 **그 팔의 진입 집합은 한 건도 안 변한다**
— 팔마다 «동일 3,700건» 위에서 청산만 갈아끼운 비교가 된다.

🔴**그리고 순환이 하나 있다**(사용자 지적, 2026-09-17): TP1.5%/SL1% 자체를 고른 격자의
대상이 **zig075 = zigzag 라벨 모델**이었다. 그 배리어로 라벨 서열을 쟀으니 출전 선수
하나에 맞춘 청산으로 전원을 채점한 셈이다. 그래서 이 스크립트는 **라벨 3계열 각각에**
같은 격자를 돌려 **각자 최적 배리어에서** 비교한다:
  N0   zigzag 양두 (3모델)      N5  h48 양두 (3모델)      N7  더블배리어 양두 (3모델)
  N0x2 zigzag 양두 ×2 앙상블 (6모델, 현행 Baseline v2)
⭐용량이 섞이지 않게 서열 판정은 **N0 · N5 · N7 (전부 3모델)** 로 한다.
"""
from __future__ import annotations
import ctypes, gc, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm            # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_side_skill_decomposition_20260917 as K    # noqa: E402

SEEDS = [int(x) for x in (next((a.split("=", 1)[1].split(",") for a in sys.argv
                                if a.startswith("--seeds=")), None)
                          or "613042,27851,904377,155690,488213".split(","))]
_FN = (next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--folds=")), None)
       or ["F1", "F2", "F3", "CAND"])
FOLDS = [f for f in K.FOLDS if f[0] in _FN]
assert len(FOLDS) == len(_FN), f"모르는 폴드: {set(_FN) - {f[0] for f in K.FOLDS}}"
# --shadow : 홀드아웃/섀도우 전방 폴드 하나만 채점한다(학습 쪽 --shadow 와 동일 정의).
if "--shadow" in sys.argv:
    FOLDS = [("SHADOW", "2022-01-01", "2026-06-30", "2026-07-01", "2026-09-30")]
CACHE = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cache=")),
                  str(E.OUT / "stageP_probs.npz")))
# 🔴비용 규약(2026-09-18 사용자 지적으로 정정): 기본 1.02bp 는 USDC 수수료만이고,
# 호메로스/라이브의 실제 왕복은 **peg+peg 5.52bp**(taker 는 10bp)다. Zeus 는 메이커 전제이므로
# 판정은 5.52 로 한다. `--cost=` 로 바꾼다.
# --target= : 총 «신호» 수. 기본 3,700 은 팔 간 비교용이지 집행용이 아니다 --
# 1슬롯 용량이 ~500건인데 3,700 을 발화시키면 86%가 버려지고 «먼저 온 것»이 남는다.
TARGET = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--target=")), 3700))
PEG = 5.52
COST = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cost=")), 1.02))
EN = ("bull", "bear", "chop")
ARMS = (next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--arms=")), None)
        or ["N0", "N5", "N7", "N0x2"])
# 팔 이름은 `N7@<라벨태그>` 형태를 받는다 -- 같은 구조를 «다른 라벨»로 재학습한 캐시를
# 가리키기 위해서다(캐시 키에 라벨 태그가 붙는다, 2026-09-17 수정).
# `N1xK` = 라우팅 없는 단일 TabM 을 **K 시드 앙상블**한 팔(사용자 요청, 2026-09-17).
# ⭐N0(3모델)·N0x2(6모델)와 «모델 수를 맞춰» 비교해야 라우팅의 값어치가 나온다.
_KNOWN = {"N0", "N5", "N7", "N0x2"}
_n1k = lambda a: int(m.group(1)) if (m := re.fullmatch(r"N1x(\d+)", _base(a))) else 0
_base = lambda a: a.split("@", 1)[0]
assert {_base(a) for a in ARMS if not _n1k(a)} <= _KNOWN, f"모르는 팔: {sorted({_base(a) for a in ARMS} - _KNOWN)}"
TPS = [0.010, 0.015, 0.020, 0.025, 0.030]
SLS = [0.005, 0.007, 0.010, 0.013]


LBLS = {"N0": "zigzag 양두 ×3", "N5": "h48 양두 ×3", "N7": "더블배리어 양두 ×3",
        "N0x2": "zigzag 양두 ×6(Baseline v2)"}


# --score=q|d|dq|margin|edge : **게이트 랭킹 함수**(2026-09-18). 지금까지 q 하나뿐이었다.
#   q      Q[고른 방향]                     (현행)
#   d      D[고른 방향]                     방향 머리 자신의 확신도
#   dq     D[방향]·Q[방향]                  두 머리가 «동의»할 때만 높다
#   margin D[방향] − D[반대 측면]           CASH 를 무시하고 측면 대비만 본다
#   edge   D[방향] − D[cash]                «거래할 만한가»를 직접 읽는다
# 🔴학습이 없으므로 이 축의 탐색은 전부 같은 창 위의 선택이다 -- 확인창에서 다시 잰다.
#   edgev  edge × 변동성      ⭐1슬롯 목적함수는 Σp/Σh 다 -- 같은 승률이면 «빨리 끝나는»
#                              후보가 두 배 가치다. 기록상 부모는 조용한 봉에서 발화한다
#                              (후보 ATR 이 전체의 64.8%)이라 체계적으로 느린 거래를 고른다.
SCORE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--score=")), "q")
assert SCORE in ("q", "d", "dq", "margin", "edge", "edgev"), f"모르는 점수: {SCORE}"


def gscore(D, Q, atr=None):
    da = D.argmax(1); ar = np.arange(len(D))
    if SCORE == "q":
        return da, np.where(da > 0, Q[ar, da], Q[:, 0])
    if SCORE == "d":
        return da, D[ar, da]
    if SCORE == "dq":
        return da, D[ar, da] * np.where(da > 0, Q[ar, da], Q[:, 0])
    if SCORE == "margin":
        opp = np.where(da == 1, 2, np.where(da == 2, 1, 0))
        return da, D[ar, da] - D[ar, opp]
    e = D[ar, da] - D[:, 0]
    if SCORE == "edge":
        return da, e
    assert atr is not None, "edgev 는 ATR 이 필요하다"
    # 변동성을 롤링 중앙값으로 정규화한다(수준 자체가 몇 해에 걸쳐 이동하므로).
    # 🔴인과: shift(1) 로 «자기 봉»을 빼고 직전 2,016봉(1주)만 본다.
    m = pd.Series(atr).shift(1).rolling(2016, min_periods=288).median().bfill().to_numpy()
    return da, e * np.clip(atr / np.maximum(m, 1e-12), 0.25, 4.0)


def log(*a): print(*a, flush=True)


def _raw(per_fold, tp, sl):
    """한 슬롯·한 칸의 «건별» 손익/보유/날짜. 진입 집합이 칸마다 같으므로 칸끼리 «행이 정렬»된다."""
    pnl, hold, days, fid = [], [], [], []
    for f, (te, h, l, c, side, idx) in enumerate(per_fold):
        r, hh, _res, _rn, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
        pnl.append(r * 1e4 - COST); hold.append(hh.astype(float))
        days.append(te.timestamp.dt.floor("D").to_numpy()[idx])
        fid.append(np.full(len(idx), f))
    return (np.concatenate(pnl), np.concatenate(hold),
            np.concatenate(days), np.concatenate(fid))


def _atr_pct(te, win=96):
    """진입 시점에 «이미 알 수 있는» 변동성. 봉 τ 는 자기 종가까지 포함한다(이 저장소 규약)."""
    h = pd.to_numeric(te["high"]).to_numpy(float); l = pd.to_numeric(te["low"]).to_numpy(float)
    c = pd.to_numeric(te["close"]).to_numpy(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(win, min_periods=1).mean().to_numpy() / np.maximum(c, 1e-12)


def _ratio_day(p, h):
    """슬롯 하나의 하루 순bp = 288 · Σp/Σh. «비율의 평균»이 아니다."""
    return float(p.mean()) * 288.0 / max(float(h.mean()), 1e-9)


ROLLQ_G = [0]
try:                                    # 🔴2026-09-18 서버 24GB OOM 의 직접 원인 대응.
    _LIBC = ctypes.CDLL("libc.so.6")    # 중간 크기 numpy 임시배열을 타이트 루프로 할당/해제하면
except OSError:                         # glibc arena 가 OS 로 반환되지 않아 RSS 만 단조 증가한다.
    _LIBC = None


def _trim():
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)
QF_STORE = {}


def _lbl(a):
    return LBLS.get(_base(a), f"라우팅 없음 ×{_n1k(a)}" if _n1k(a) else a)


def _kelly_w(conf, win, tp, sl, cost, nb=5, prior=40, norm=True):
    """인과적 p-켈리 가중. 각 거래 시점까지 **이미 해소된** 거래만으로 분위별 적중률을 추정한다.

    🔴켈리를 그대로 켜면 명목이 7~13배로 뛴다 -- 그건 전략이 아니라 레버리지다
    (기록: 「켈리는 분할이 아니라 «배수»고 경로를 안 본다」, 하한 켈리 35.5배는 1왕복에 청산).
    그래서 **평균 1 로 정규화**해서 「p 에 따라 크기를 바꾸는 «모양»이 도움이 되는가」만 잰다.
    b 는 비용 반영 유효배당 (TP−c)/(SL+c), f* = (p(1+b)−1)/b, 음수는 0.
    """
    b = (tp * 1e4 - cost) / (sl * 1e4 + cost)
    q = pd.qcut(pd.Series(conf).rank(method="first"), nb, labels=False).to_numpy()
    p0 = float(win.mean())                                  # 사전값(전역) -- 워밍업용
    w = np.zeros(len(conf)); wins = np.zeros(nb); tot = np.zeros(nb)
    for i in range(len(conf)):
        g = q[i]
        ph = (wins[g] + prior * p0) / (tot[g] + prior)       # 수축 추정
        f = (ph * (1 + b) - 1) / b
        w[i] = max(f, 0.0)
        wins[g] += win[i]; tot[g] += 1                       # ⭐자기 결과는 «쓴 뒤에» 반영
    if not norm:
        return w
    m = w.mean()
    return w / m if m > 1e-9 else np.ones(len(conf))


def _equity(pnl_bp, f, sl):
    """자본 경로. **켈리는 «한 베팅에 자본의 몇 %»** 이므로 명목 = f/SL, 자본수익 = r·f/SL.
    손절 적중이면 정확히 −f, 익절이면 +f·b 가 된다(고전 켈리와 동일한 구조).
    반환: 최종배수 · MDD · 최악 1건(자본 %) · 파산여부."""
    rc = pnl_bp / 1e4 * (f / sl)            # 건당 «자본» 수익률
    eq = np.cumprod(1.0 + rc)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(eq[-1]), float((1 - eq / peak).max()), float(rc.min() * 100), bool((eq <= 0).any())


def gap1m(pick, LBLS):
    """--gap1m: 갭 측정을 **1분봉 해상도**로. 5분봉은 봉 «안»의 경로를 못 본다.

    손절이 걸린 5분봉을 1분봉 5개로 쪼개 **처음 손절선을 뚫는 1분봉**을 찾고,
    그 1분봉의 «시가»가 이미 손절선 너머인지(=진짜 갭), 저/고가가 얼마나 뚫었는지를 잰다.
    ⭐1분봉은 2024-01~ 만 있어 F1(2023H2)은 빠진다 -- 커버 폴드를 같이 찍는다.
    """
    tp, sl = K.BASE_TP, K.BASE_SL
    MINGAP = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mingap=")), 0))
    M = pd.read_csv(ROOT / "data/training_features_1m.csv",
                    usecols=["timestamp", "open", "high", "low", "close"])
    M["timestamp"] = pd.to_datetime(M["timestamp"])
    mt = M.timestamp.values.astype("datetime64[m]").astype(np.int64)
    mo, mh, ml = (M[c].to_numpy(float) for c in ("open", "high", "low"))
    log(f"  1분봉 {len(M):,}행 {M.timestamp.min()} ~ {M.timestamp.max()}")
    for arm in ARMS:
        go, gb, miss, n_sl = [], [], 0, 0
        for _t, per_fold in pick(arm):
            for te, h, l, c, side, idx in per_fold:
                if len(idx) < 20:
                    continue
                ts = te.timestamp.values.astype("datetime64[m]").astype(np.int64)
                r, hh, _a, rn, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
                free, take, last = -1, [], -10**9
                for i in range(len(idx)):
                    if idx[i] - last < MINGAP or idx[i] <= free:
                        continue
                    take.append(i); free = idx[i] + int(hh[i]); last = idx[i]
                t_ = np.array(take, int); k_ = t_[rn[t_] == 0]
                if not len(k_):
                    continue
                n_sl += len(k_)
                e = c[idx[k_]]; sd = side[idx[k_]]
                bar = np.clip(idx[k_] + hh[k_], 0, len(c) - 1)
                stop = np.where(sd > 0, e * (1 - sl), e * (1 + sl))
                for q in range(len(k_)):
                    a0 = np.searchsorted(mt, ts[bar[q]])          # 그 5분봉의 1분봉 5개
                    if a0 >= len(mt) or mt[a0] != ts[bar[q]]:
                        miss += 1; continue
                    hit = -1
                    for u in range(a0, min(a0 + 5, len(mt))):
                        if (sd[q] > 0 and ml[u] <= stop[q]) or (sd[q] < 0 and mh[u] >= stop[q]):
                            hit = u; break
                    if hit < 0:
                        miss += 1; continue
                    op_, lo_, hi_ = mo[hit], ml[hit], mh[hit]
                    go.append(max((stop[q] - op_) if sd[q] > 0 else (op_ - stop[q]), 0) / e[q])
                    gb.append(max((stop[q] - lo_) if sd[q] > 0 else (hi_ - stop[q]), 0) / e[q])
        go, gb = np.array(go), np.array(gb)
        log(f"\n{'='*92}\n■ {arm} — **1분봉 해상도** 손절 미끄러짐 (TP{tp*100:g}%/SL{sl*100:g}% · "
            f"mingap {MINGAP}봉)\n{'='*92}")
        log(f"  손절 {n_sl:,}건 중 1분봉 매칭 {len(go):,}건 · 미매칭 {miss:,}건(대부분 F1=2023, 1분봉 없음)")
        for nm, v in (("진짜 갭(1분 시가)", go), ("1분봉내 초과", gb)):
            if not len(v):
                continue
            log(f"  {nm:<18} 발생 {(v>1e-9).mean()*100:5.1f}% · 중앙 {np.median(v)*100:6.3f}% · "
                f"p95 {np.quantile(v,.95)*100:6.3f}% · p99 {np.quantile(v,.99)*100:6.3f}% · "
                f"최대 {v.max()*100:6.3f}%")
        if len(gb):
            for q, lab in ((1.0, "최대"), (0.999, "p99.9"), (0.99, "p99")):
                w = sl + (gb.max() if q == 1.0 else np.quantile(gb, q))
                log(f"    1분봉내 {lab:>6}: 총손실 {w*100:5.2f}% → 안전 명목 상한 **{1/w/2:.1f}배**")
    log("\n⭐5분봉 판과 비교해 «1분봉내 초과»가 크게 줄면, 실제 체결은 손절선에 가깝다는 뜻이다.")
    return 0


def gapscan(pick, LBLS):
    """--gap: **손절 체결이 얼마나 미끄러지는가.** 켈리 배수의 상한을 정하는 단 하나의 숫자.

    시뮬은 손절이 «정확히» SL 에 체결된다고 본다. 명목 N배에서 청산선은 역행 약 1/N 이므로,
    손절선을 크게 뚫는 사건이 있으면 그 배수는 죽는다.
      ⭐**진짜 갭** = 손절 봉의 «시가»가 이미 손절선 너머 -> 회피 불가, 시가 체결
      ⭐**봉내 초과** = 손절 봉의 저가(롱)/고가(숏)가 손절선을 얼마나 뚫었나 -> 최악 체결 상한
    """
    tp, sl = K.BASE_TP, K.BASE_SL
    MINGAP = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mingap=")), 0))
    for arm in ARMS:
        go, gb, n_sl, n_all = [], [], 0, 0
        for _t, per_fold in pick(arm):
            for te, h, l, c, side, idx in per_fold:
                if len(idx) < 20:
                    continue
                op = pd.to_numeric(te["open"]).to_numpy(float)
                r, hh, _a, rn, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
                free, take, last = -1, [], -10**9          # 1슬롯 순차 + mingap
                for i in range(len(idx)):
                    if idx[i] - last < MINGAP or idx[i] <= free:
                        continue
                    take.append(i); free = idx[i] + int(hh[i]); last = idx[i]
                t_ = np.array(take, int); n_all += len(t_)
                m_ = rn[t_] == 0                            # 손절로 끝난 건만
                if not m_.any():
                    continue
                k_ = t_[m_]; n_sl += len(k_)
                e = c[idx[k_]]; sd = side[idx[k_]]
                bar = np.clip(idx[k_] + hh[k_], 0, len(c) - 1)
                stop = np.where(sd > 0, e * (1 - sl), e * (1 + sl))
                # 시가가 손절선 너머면 그만큼이 «진짜 갭»(롱: 시가<스톱, 숏: 시가>스톱)
                go.append(np.maximum(np.where(sd > 0, stop - op[bar], op[bar] - stop), 0) / e)
                # 봉내 최악 지점이 손절선을 얼마나 뚫었나
                go_b = np.where(sd > 0, stop - l[bar], h[bar] - stop)
                gb.append(np.maximum(go_b, 0) / e)
        go = np.concatenate(go); gb = np.concatenate(gb)
        log(f"\n{'='*92}\n■ {arm} — 손절 체결 미끄러짐 (TP{tp*100:g}%/SL{sl*100:g}% · mingap {MINGAP}봉 · "
            f"1슬롯)\n{'='*92}")
        log(f"  체결 {n_all:,}건 중 손절 {n_sl:,}건 ({n_sl/max(n_all,1)*100:.1f}%)")
        for nm, v in (("진짜 갭(시가 초과)", go), ("봉내 초과(저·고가)", gb)):
            nz = (v > 1e-9).mean() * 100
            log(f"  {nm:<20} 발생 {nz:5.1f}% · 중앙 {np.median(v)*100:6.3f}% · "
                f"p95 {np.quantile(v,.95)*100:6.3f}% · p99 {np.quantile(v,.99)*100:6.3f}% · "
                f"최대 {v.max()*100:6.3f}%")
        log(f"\n  ⭐명목 N배의 청산선 ≈ 역행 1/N. 총 손실 = SL + 초과분이므로:")
        for nm, v in (("진짜 갭", go), ("봉내 초과", gb)):
            for q, lab in ((1.0, "최대"), (0.999, "p99.9"), (0.99, "p99")):
                worst = sl + (v.max() if q == 1.0 else np.quantile(v, q))
                log(f"    {nm} {lab:>6}: 총손실 {worst*100:5.2f}% → **안전 명목 상한 {1/worst/2:.1f}배**"
                    f" (청산선의 2배 여유)")
        log("\n  🔴「봉내 초과」는 최악 가정이다 -- 실제 스톱마켓은 그 사이 어딘가에 체결된다.")
        log("  ⭐「진짜 갭」이 0에 가까우면 높은 배수가 가능하고, 꼬리가 두꺼우면 켈리는 못 쓴다.")
    return 0


def slotsweep(pick, LBLS):
    """--slotsweep: 슬롯 수 K 를 훑는다. **총 노출을 고정**(건당 크기 1/K)해서 재는 게 핵심 --
    같은 크기로 K개를 걸면 그건 전략 개선이 아니라 **레버리지 K배**다(사용자 지적 2026-09-18).

    🔴함께 봐야 할 것: ①**평균 슬롯 점유율** -- 슬롯이 놀면 유효 노출이 1 미만이라 「고정명목」
    수치가 불리하게 나온다 ②**중앙 진입 간격** -- 같은 묶음에서 K개를 잡으면 사실상 한 건을
    K배로 건 것이라 분산 이득이 없다.
    """
    tp, sl = K.BASE_TP, K.BASE_SL
    days = sum((pd.Timestamp(v1) - pd.Timestamp(v0)).days + 1 for _n, _a, _b, v0, v1 in FOLDS)
    KELLY = "--kelly" in sys.argv
    # --mingap=<봉> : 직전 «진입»으로부터 이만큼 지나야 새 진입. 0 이면 제약 없음.
    # 🔴다중 슬롯이 «같은 묶음을 K번 사는 것»과 «진짜 기회 포착»을 가르는 유일한 장치다
    #   (스모크: K=3 에서 중앙 진입간격이 260 -> 14봉으로 붕괴, 보유기간은 ~100봉).
    MINGAP = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mingap=")), 0))
    # --kellyf=none|half|full : **실제 켈리 배수**로 자본의 몇 %를 걸지 정한다(정규화 없음).
    # none = 현행 고정(명목 0.9배 = 자본의 0.45% 위험). f 는 25%에서 자른다(추정오차 방어).
    KF = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--kellyf=")), "none")
    FIX_F = 0.9 * sl                        # 명목 0.9배 × 손절폭
    KS = [int(x) for x in (next((a.split("=", 1)[1].split(",") for a in sys.argv
                                 if a.startswith("--slots=")), None) or "1,2,3,5,7,10".split(","))]
    for arm in ARMS:
        ent = pick(arm)
        log(f"\n{'='*104}\n■ {arm} — {_lbl(arm)} · **슬롯 수 스윕** "
            f"(TP{tp*100:g}%/SL{sl*100:g}% · 비용 {COST}bp · 총 노출 고정 · "
            f"mingap {MINGAP}봉 · {'p-켈리' if KELLY else '고정크기'})\n{'='*104}")
        log(f"{'슬롯':>5}{'체결':>8}{'거절%':>7}{'체결/일':>8}{'건당bp':>9}{'CI95':>20}"
            f"{'순/일':>8}{'연율%':>8}{'점유율':>8}{'간격':>6}{'자본배수':>9}{'MDD%':>7}{'최악건%':>8}")
        for Ks in KS:
            acc = []
            for si_, (_thr, per_fold) in enumerate(ent):
                pl, dl, nsig, occ, bars, gaps, eqs = [], [], 0, 0, 0, [], []
                for fi_, (te, h, l, c, side, idx) in enumerate(per_fold):
                    if len(idx) < 20:
                        continue
                    r, hh, _a, _b, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
                    p_ = r * 1e4 - COST; nsig += len(idx); bars += len(te)
                    free = np.full(Ks, -1, np.int64); take = []; last = -10**9
                    for i in range(len(idx)):
                        if idx[i] - last < MINGAP:
                            continue
                        j = int(free.argmin())
                        if free[j] < idx[i]:
                            take.append(i); free[j] = idx[i] + int(hh[i])
                            occ += int(hh[i]); last = idx[i]
                    t_ = np.array(take, int)
                    pp = p_[t_]
                    cf = QF_STORE[(arm, si_, fi_)][t_]
                    if KELLY:                               # 평균 1 정규화 -- 배수가 아니라 «모양»
                        pp = pp * _kelly_w(cf, (r[t_] > 0).astype(float), tp, sl, COST)
                    if KF != "none":                        # ⭐실제 켈리 배수로 자본 경로를 낸다
                        wk = _kelly_w(cf, (r[t_] > 0).astype(float), tp, sl, COST, norm=False)
                        fv = np.clip(wk * (0.5 if KF == "half" else 1.0), 0, 0.25) / Ks
                        eqs.append(_equity(p_[t_], fv, sl))
                    else:
                        eqs.append(_equity(p_[t_], FIX_F / Ks, sl))
                    pl.append(pp); dl.append(te.timestamp.dt.floor("D").to_numpy()[idx[t_]])
                    if len(t_) > 1:
                        gaps.append(np.diff(idx[t_]))
                pl = np.concatenate(pl); dl = np.concatenate(dl)
                lo_, hi_, _nd = E.block_ci(pl, dl)
                eq = np.array(eqs, float)
                acc.append((nsig, len(pl), pl.mean(), lo_, hi_,
                            occ / max(bars * Ks, 1), float(np.median(np.concatenate(gaps))),
                            float(np.prod(eq[:, 0])), float(eq[:, 1].max()), float(eq[:, 2].min())))
            a = np.array(acc, float)
            sig, ex, bp = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
            day = bp * ex / Ks / days          # ⭐총 노출 고정 -- 건당 1/K 크기
            log(f"{Ks:>5}{int(ex):>8,}{(1-ex/sig)*100:>6.1f}%{ex/days:>8.2f}{bp:>+9.2f}"
                f"  [{a[:, 3].mean():+7.2f},{a[:, 4].mean():+7.2f}]{day:>8.1f}{day*365/100:>8.1f}"
                f"{a[:, 5].mean()*100:>7.1f}%{a[:, 6].mean():>6.0f}"
                f"{a[:, 7].mean():>9.2f}{a[:, 8].mean()*100:>6.1f}%{a[:, 9].mean():>8.2f}"
                f"{'  ✅' if a[:, 3].mean() > 0 else ''}")
    log("\n⭐«순/일»은 총 노출을 1 로 고정한 값이다 -- 같은 크기로 K개 걸면 레버리지 K배라 비교가 안 된다.")
    log("🔴«중앙간격»이 보유기간보다 훨씬 짧으면 같은 묶음을 K번 산 것이다(분산 이득 없음).")
    return 0


def seqgrid(pick, LBLS):
    """--seqgrid: 20칸 배리어 격자를 **순차 1슬롯 목적함수**로 다시 훑는다.

    🔴순진한 격자는 「모든 후보를 독립 체결」 가정이라 보유시간을 `288/평균보유`(상한)로만
    반영했다. 1슬롯에서는 **보유시간이 희소 자원**이다 -- 빨리 해소되는 배리어가 슬롯을
    비워 더 많은 신호를 잡는다. 그래서 최적 기하학이 이동할 수 있다(사용자 지적 2026-09-18).
    """
    days = sum((pd.Timestamp(v1) - pd.Timestamp(v0)).days + 1 for _n, _a, _b, v0, v1 in FOLDS)
    MINGAP = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mingap=")), 0))
    for arm in ARMS:
        ent = pick(arm)
        rows = []
        for tp in TPS:
            for sl in SLS:
                _trim()                          # 셀마다 arena 반환
                acc = []
                for _thr, per_fold in ent:
                    pl, hl, dl, nsig = [], [], [], 0
                    for te, h, l, c, side, idx in per_fold:
                        if len(idx) < 20:
                            continue
                        r, hh, _a, _b, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
                        p_ = r * 1e4 - COST; nsig += len(idx)
                        take, cur, last = [], -1, -10**9
                        for i in range(len(idx)):
                            if idx[i] - last < MINGAP or idx[i] <= cur:
                                continue
                            take.append(i); cur = idx[i] + int(hh[i]); last = idx[i]
                        t_ = np.array(take, int)
                        pl.append(p_[t_]); hl.append(hh[t_].astype(float))
                        dl.append(te.timestamp.dt.floor("D").to_numpy()[idx[t_]])
                    pl = np.concatenate(pl); hl = np.concatenate(hl); dl = np.concatenate(dl)
                    lo_, hi_, _nd = E.block_ci(pl, dl)
                    acc.append((nsig, len(pl), pl.mean(), lo_, hi_, float(np.median(hl))))
                    _trim()                          # 🔴슬롯마다 반환 -- 셀 단위는 너무 성기다
                a = np.array(acc, float)
                ex, bp = a[:, 1].mean(), a[:, 2].mean()
                rows.append({"tp": tp, "sl": sl, "sig": a[:, 0].mean(), "ex": ex,
                             "bp": bp, "lo": a[:, 3].mean(), "hi": a[:, 4].mean(),
                             "med": a[:, 5].mean(), "pd": ex / days, "day": bp * ex / days,
                             "spread": a[:, 2].max() - a[:, 2].min()})
        R = pd.DataFrame(rows).sort_values("day", ascending=False)
        log(f"\n{'='*104}\n■ {arm} — {_lbl(arm)} · **순차 1슬롯 배리어 격자** "
            f"(비용 {COST}bp · mingap {MINGAP}봉 · 순/일 내림차순)\n{'='*104}")
        log(f"{'TP':>5}{'SL':>5}{'신호':>7}{'체결':>7}{'거절%':>7}{'체결/일':>8}{'중앙보유':>9}"
            f"{'건당bp':>9}{'CI95':>20}{'순/일':>8}{'시드폭':>8}")
        for _, r in R.iterrows():
            log(f"{r.tp*100:>4.1f}%{r.sl*100:>4.1f}%{int(r.sig):>7,}{int(r.ex):>7,}"
                f"{(1-r.ex/r.sig)*100:>6.1f}%{r.pd:>8.2f}{r.med:>9.0f}{r.bp:>+9.2f}"
                f"  [{r.lo:+7.2f},{r.hi:+7.2f}]{r.day:>8.1f}{r.spread:>8.2f}"
                f"{'  ✅' if r.lo > 0 else ''}{'  🔴<1건/일' if r.pd < 1.0 else ''}")
    log("\n⭐순진한 격자와 최적 칸이 다르면, 그건 «보유시간이 희소 자원»이라는 뜻이다.")
    return 0


def waitrule(pick, LBLS):
    """--wait=W : 신호가 뜨면 **즉시 진입하지 않고 W봉 기다렸다가** 그 창의 최고 확신도
    신호의 «방향»으로 진입한다.

    발단: 홀드아웃에서 신호 풀은 +6.23bp 인데 순차 1슬롯이 고른 39건은 −13.59bp 였다.
    89%를 버리고 «먼저 온 것»을 잡은 게 엣지를 통째로 먹었다(집행 손실 −19.8bp).
    🔴**인과성**: 창 안 최고 확신 봉의 «가격»에 진입하면 미래참조다. 창이 끝난 봉(i+W)의
    가격에 진입하고 W봉 지연 비용을 그대로 문다. W=0 이 현행(즉시 진입) 대조군.
    """
    tp, sl = K.BASE_TP, K.BASE_SL
    WS = [int(x) for x in (next((a.split("=", 1)[1].split(",") for a in sys.argv
                                 if a.startswith("--wait=")), None) or "0,3,6,12".split(","))]
    days = sum((pd.Timestamp(v1) - pd.Timestamp(v0)).days + 1 for _n, _a, _b, v0, v1 in FOLDS)
    for arm in ARMS:
        ent = pick(arm)
        log(f"\n{'='*100}\n■ {arm} — {_lbl(arm)} · **대기 규칙** "
            f"(TP{tp*100:g}%/SL{sl*100:g}% · 비용 {COST}bp · 1슬롯)\n{'='*100}")
        log(f"{'대기':>5}{'신호':>8}{'체결':>7}{'거절%':>7}{'체결/일':>8}{'건당bp':>9}"
            f"{'(전체후보)':>11}{'CI95':>20}{'순/일':>8}{'시드폭':>8}")
        for W in WS:
            acc = []
            for si_, (_thr, per_fold) in enumerate(ent):
                pl, dl, nsig = [], [], 0
                for fi_, (te, h, l, c, side, idx) in enumerate(per_fold):
                    if len(idx) < 20:
                        continue
                    qf = QF_STORE[(arm, si_, fi_)]; nsig += len(idx)
                    ebars = np.clip(idx + W, 0, len(c) - 2)      # 가능한 진입 봉 전부
                    RL, RS = {}, {}
                    for sgn, D in ((1.0, RL), (-1.0, RS)):        # 측면별로 한 번씩만 시뮬
                        sv = np.full(len(c), sgn)
                        r_, h_, _a, _b, _m = K._first_touch_open(ebars, sv, h, l, c, tp, sl, K.MAXBARS)
                        D["r"], D["h"] = r_, h_
                    take, cur, i = [], -1, 0
                    while i < len(idx):
                        if idx[i] <= cur:
                            i += 1; continue
                        j = i                                     # 창 [idx[i], idx[i]+W]
                        while j + 1 < len(idx) and idx[j + 1] <= idx[i] + W:
                            j += 1
                        best = i + int(np.argmax(qf[i:j + 1]))    # 창 안 최고 확신 «방향»만 쓴다
                        D = RL if side[idx[best]] > 0 else RS
                        take.append((i, D["r"][i], D["h"][i]))    # 진입은 ebars[i] = idx[i]+W
                        cur = ebars[i] + int(D["h"][i]); i = j + 1
                    if not take:
                        continue
                    ii = np.array([t[0] for t in take], int)
                    pl.append(np.array([t[1] for t in take]) * 1e4 - COST)
                    dl.append(te.timestamp.dt.floor("D").to_numpy()[ebars[ii]])
                pl = np.concatenate(pl); dl = np.concatenate(dl)
                lo_, hi_, _n = E.block_ci(pl, dl)
                acc.append((nsig, len(pl), pl.mean(), lo_, hi_))
                _trim()
            a = np.array(acc, float)
            sig, ex, bp = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
            log(f"{W:>5}{int(sig):>8,}{int(ex):>7,}{(1-ex/sig)*100:>6.1f}%{ex/days:>8.2f}"
                f"{bp:>+9.2f}{'':>11}  [{a[:, 3].mean():+7.2f},{a[:, 4].mean():+7.2f}]"
                f"{bp*ex/days:>8.1f}{a[:, 2].max()-a[:, 2].min():>8.2f}"
                f"{'  ✅' if a[:, 3].mean() > 0 else ''}")
    log("\n⭐W>0 이 W=0 을 이기면 «먼저 온 것을 잡는» 손실이 실재하고 고칠 수 있다는 뜻이다.")
    log("🔴W 가 커지면 지연 비용도 커진다 -- 최적점이 있으면 그게 답이고, 단조 감소면 대기는 무효다.")
    return 0


def seq(pick, LBLS):
    """--seq: **슬롯 1개 순차 시뮬레이션**. 지금까지의 「건/일」은 288/평균보유 = «상한»이었다.

    🔴신호가 용량보다 많으면 슬롯이 차 있는 동안 온 신호는 버려지고, 버려지는 게 무작위가
    아니다 -- **먼저 온 것을 잡고 더 좋은 것을 놓친다**. 09-15 에 이 저장소가 같은 함정에서
    이론값 +26.95 vs 측정 +3.14bp/일(8.6배)을 겪었다.
    ⭐배리어 결과는 «이전 거래를 잡았는지»와 무관하므로 전 후보를 일괄 계산해두고
    순차 선택만 하면 정확하다. 재진입은 청산 다음 봉부터(+1봉).
    """
    tp, sl = K.BASE_TP, K.BASE_SL
    # 🔴2026-09-18: 여기에만 mingap 이 빠져 있어 격자(mingap 적용)와 조건이 어긋났다.
    MINGAP = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mingap=")), 0))
    days = {nm: (pd.Timestamp(v1) - pd.Timestamp(v0)).days + 1 for nm, _a, _b, v0, v1 in FOLDS}
    for arm in ARMS:
        acc = {}; SLOTPOOL = []
        for _thr, per_fold in pick(arm):
            POOL = {}
            for f, (te, h, l, c, side, idx) in enumerate(per_fold):
                if len(idx) < 20:
                    continue
                r, hh, _a, _b, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
                pnl_all = r * 1e4 - COST
                take, cur, last = [], -1, -10**9
                for i in range(len(idx)):                 # 시간순 -- idx 는 오름차순
                    if idx[i] - last < MINGAP or idx[i] <= cur:
                        continue
                    take.append(i); cur = idx[i] + int(hh[i]); last = idx[i]   # 청산 봉까지 점유
                take = np.array(take, int)
                dd = te.timestamp.dt.floor("D").to_numpy()[idx[take]]
                lo_, hi_, nd = E.block_ci(pnl_all[take], dd)
                POOL.setdefault(len(POOL) % 1, []).append((pnl_all[take], dd))
                acc.setdefault(f, []).append(
                    (len(idx), len(take), pnl_all.mean(), pnl_all[take].mean(),
                     lo_, hi_, nd, float(np.median(hh[take])),
                     288.0 / max(hh.mean(), 1e-9)))
            if POOL.get(0):
                SLOTPOOL.append((np.concatenate([x[0] for x in POOL[0]]),
                                 np.concatenate([x[1] for x in POOL[0]])))
        nm_ = [x[0] for x in FOLDS]
        log(f"\n{'='*112}\n■ {arm} — {_lbl(arm)} · **슬롯 1개 순차** "
            f"(TP{tp*100:g}%/SL{sl*100:g}% · mingap {MINGAP}봉 · 비용 {COST}bp)\n{'='*112}")
        log(f"{'폴드':<6}{'신호':>7}{'체결':>7}{'거절%':>7}{'신호/일':>8}{'체결/일':>8}"
            f"{'건당bp':>9}{'(전체후보)':>10}{'CI95':>20}{'순/일':>8}{'상한건/일':>9}")
        T = np.zeros(4)
        for f, nm in enumerate(nm_):
            a = np.array(acc[f], float)
            sig, ex = a[:, 0].mean(), a[:, 1].mean()
            bp, bp_all = a[:, 3].mean(), a[:, 2].mean()
            pd_ = ex / days[nm]
            T += [sig, ex, bp * ex, days[nm] * 0 + ex]     # 가중합용
            log(f"{nm:<6}{int(sig):>7,}{int(ex):>7,}{(1-ex/sig)*100:>6.1f}%"
                f"{sig/days[nm]:>8.2f}{pd_:>8.2f}{bp:>+9.2f}{bp_all:>+10.2f}"
                f"  [{a[:, 4].mean():+7.2f},{a[:, 5].mean():+7.2f}]{bp*pd_:>8.1f}"
                f"{a[:, 8].mean():>9.2f}{'  ✅' if a[:, 4].mean() > 0 else ''}")
        tot_d = sum(days[nm] for nm in nm_)
        bp_w = T[2] / max(T[1], 1e-9)
        pc = np.array([E.block_ci(p_, d_) for p_, d_ in SLOTPOOL], float)
        log(f"{'합계':<6}{int(T[0]):>7,}{int(T[1]):>7,}{(1-T[1]/T[0])*100:>6.1f}%"
            f"{T[0]/tot_d:>8.2f}{T[1]/tot_d:>8.2f}{bp_w:>+9.2f}{'':>10}"
            f"  [{pc[:, 0].mean():+7.2f},{pc[:, 1].mean():+7.2f}]"
            f"{bp_w*T[1]/tot_d:>8.1f}{'  ✅0배제' if pc[:, 0].mean() > 0 else ''}"
            f"{'  ✅peg배제' if pc[:, 0].mean() > PEG - COST else ''}")
        log(f"  통합 독립일 {pc[:, 2].mean():.0f} · 시드폭 "
            f"{max(np.mean(p_) for p_, _ in SLOTPOOL) - min(np.mean(p_) for p_, _ in SLOTPOOL):.2f}")
    log("\n⭐«체결/일»이 진짜 값이다. «상한건/일»(288/평균보유)은 슬롯이 안 빈다는 가정이다.")
    log("🔴«건당bp» 와 «(전체후보)» 가 크게 다르면, 경합이 «좋은 신호를 골라내는 능력»을 깎은 것이다.")
    return 0


def byregime(pick, LBLS):
    """--byregime: **balnobb 레짐별**로 쪼갠다(진입 봉의 argmax 레짐).

    발단: 폴드별 분해에서 엣지가 F1(2023H2)에 몰렸는데 단조 감쇠가 아니었다(2026H1 은 다시
    양수). 「오래된 데이터」가 아니라 **레짐**이라는 뜻이다.
    ⭐하드 라우팅이므로 «레짐 = 그 봉을 예측한 전문가»다 -- 이 표는 전문가 절제이기도 하다.
    🔴엣지가 특정 레짐에만 살면 필요한 건 새 라벨도 새 배리어도 아니라 **거래하지 않는 게이트**다.
    지금 balnobb 는 «누가 예측할지»만 정하고 «거래할지»는 정하지 않는다 -- 그 자리가 비어 있다.
    """
    tp, sl, RN = K.BASE_TP, K.BASE_SL, ("bull", "bear", "chop")
    for arm in ARMS:
        evc, per_slot, cellbp, celln = {}, [], {}, {}
        for _thr, per_fold in pick(arm):
            got = {g: [[], []] for g in range(3)}
            for f, (te, h, l, c, side, idx) in enumerate(per_fold):
                if len(idx) == 0:
                    continue
                if id(te) not in evc:
                    evc[id(te)] = tabm._route_probs(te).argmax(1)
                ev = evc[id(te)][idx]
                r, hh, _a, _b, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
                pnl = r * 1e4 - COST; dd = te.timestamp.dt.floor("D").to_numpy()[idx]
                for g in range(3):
                    m = ev == g
                    if m.sum():
                        got[g][0].append(np.stack([pnl[m], hh[m].astype(float)]))
                        got[g][1].append(dd[m])
                    cellbp.setdefault((g, f), []).append(pnl[m].mean() if m.sum() >= 20 else np.nan)
                    celln.setdefault((g, f), []).append(int(m.sum()))
            row = {}
            for g in range(3):
                if not got[g][0]:
                    row[g] = None; continue
                ph = np.concatenate(got[g][0], 1); dd = np.concatenate(got[g][1])
                lo_, hi_, nd = E.block_ci(ph[0], dd)
                row[g] = (len(dd), ph[0].mean(), lo_, hi_, nd, np.median(ph[1]),
                          288.0 / max(ph[1].mean(), 1e-9))
            per_slot.append(row)
        log(f"\n{'='*104}\n■ {arm} — {_lbl(arm)} · **레짐별 분해** "
            f"(TP{tp*100:g}%/SL{sl*100:g}% · 5슬롯 평균)\n{'='*104}")
        log(f"{'레짐':<7}{'건수':>8}{'비중':>7}{'건당bp':>9}{'독립일':>7}{'CI95':>20}"
            f"{'중앙보유':>9}{'건/일':>7}{'순/일':>8}")
        tot = sum(np.mean([r[g][0] for r in per_slot if r[g]]) for g in range(3)
                  if any(r[g] for r in per_slot))
        for g in range(3):
            v = [r[g] for r in per_slot if r[g]]
            if not v:
                log(f"{RN[g]:<7}{'0':>8}   진입 없음"); continue
            a = np.array(v, float); n_, bp = a[:, 0].mean(), a[:, 1].mean()
            log(f"{RN[g]:<7}{int(n_):>8,}{n_/tot*100:>6.1f}%{bp:>+9.2f}{a[:, 4].mean():>7.0f}"
                f"  [{a[:, 2].mean():+7.2f},{a[:, 3].mean():+7.2f}]{a[:, 5].mean():>9.0f}"
                f"{a[:, 6].mean():>7.2f}{bp * a[:, 6].mean():>8.1f}"
                f"{'  ✅' if a[:, 2].mean() > 0 else ''}")
        log(f"\n  레짐 × 폴드 건당bp (괄호는 건수, 20건 미만은 -)")
        log(f"  {'':<7}" + "".join(f"{nm:>18}" for nm, *_ in FOLDS))
        for g in range(3):
            cells = []
            for f in range(len(FOLDS)):
                b = np.nanmean(cellbp[(g, f)]) if np.isfinite(cellbp[(g, f)]).any() else np.nan
                n_ = int(np.mean(celln[(g, f)]))
                cells.append(f"{'     -' if not np.isfinite(b) else f'{b:+6.1f}'}({n_:>5,})")
            log(f"  {RN[g]:<7}" + "".join(f"{c:>18}" for c in cells))
    log("\n⭐판정: 한 레짐만 양수이고 나머지가 0 근처면 **그 레짐 밖에서 거래하지 않는 게이트**가")
    log("  빠진 층이다. 세 레짐 모두 F1 에서만 양수면 레짐이 아니라 여전히 «창»이 원인이다.")
    return 0


def byfold(pick, LBLS):
    """--byfold: 동결 배리어(TP1.5%/SL1%)에서 **폴드별로 분해**한다.

    발단: 워크포워드에서 F3·CAND 만 보니 N0x2 가 건당 +40.67 -> **+5.11** 이었다.
    4폴드 평균이 앞 두 폴드에 끌려온 것인지 직접 확인한다. 임계값 q 는 베이스라인과
    똑같이 «4폴드 통합 3,700건»으로 잡고(그래야 같은 모델이다) 집계만 폴드로 쪼갠다.
    🔴엣지가 앞 폴드에 몰려 있으면 미접촉 OOS(2026-07~09)를 열 자격이 없다.
    """
    tp, sl = K.BASE_TP, K.BASE_SL
    for arm in ARMS:
        acc = {}
        for _thr, per_fold in pick(arm):
            for f, (te, h, l, c, side, idx) in enumerate(per_fold):
                # ⭐한 폴드의 진입이 0~몇 건일 수 있다(발화 쏠림). 그 자체가 결과이므로
                # 건수는 살리고 통계만 NaN 으로 둔다 -- 조용히 빼면 쏠림이 표에서 사라진다.
                if len(idx) < 20:
                    acc.setdefault(f, []).append((len(idx),) + (np.nan,) * 6)
                    continue
                r, hh, _res, _rn, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
                pnl = r * 1e4 - COST
                lo_, hi_, nd = E.block_ci(pnl, te.timestamp.dt.floor("D").to_numpy()[idx])
                acc.setdefault(f, []).append(
                    (len(idx), float(pnl.mean()), lo_, hi_, nd,
                     float(np.median(hh)), 288.0 / max(hh.mean(), 1e-9)))
        tot = sum(np.mean([x[0] for x in v]) for v in acc.values())
        log(f"\n{'='*104}\n■ {arm} — {_lbl(arm)} · 폴드별 분해 "
            f"(TP{tp*100:g}%/SL{sl*100:g}% · 5슬롯 평균)\n{'='*104}")
        log(f"{'폴드':<6}{'기간':<26}{'건수':>7}{'비중':>7}{'건당bp':>9}{'독립일':>7}"
            f"{'CI95':>20}{'중앙보유':>9}{'건/일':>7}{'순/일':>8}")
        for f, (nm, _t0, _t1, v0, v1) in enumerate(FOLDS):
            a = np.array(acc[f], float)
            n_, g = a[:, 0].mean(), np.nanmean(a[:, 1]) if np.isfinite(a[:, 1]).any() else np.nan
            log(f"{nm:<6}{v0 + '~' + v1:<26}{int(n_):>7,}{n_/tot*100:>6.1f}%{g:>+9.2f}"
                f"{np.nanmean(a[:, 4]):>7.0f}  [{np.nanmean(a[:, 2]):+7.2f},{np.nanmean(a[:, 3]):+7.2f}]"
                f"{np.nanmean(a[:, 5]):>9.0f}{np.nanmean(a[:, 6]):>7.2f}"
                f"{g * np.nanmean(a[:, 6]):>8.1f}"
                f"{'  ✅' if np.nanmean(a[:, 2]) > 0 else ('  ⚠️표본부족' if not np.isfinite(g) else '   ')}")
        tb = np.array([[np.mean([x[0] for x in acc[f]]), np.nanmean(np.array(acc[f], float)[:, 1]),
                        np.nanmean(np.array(acc[f], float)[:, 6])] for f in acc], float)
        w = tb[:, 0] / tb[:, 0].sum()
        log(f"{'합계':<6}{'(건수가중)':<26}{int(tb[:, 0].sum()):>7,}{100.0:>6.1f}%"
            f"{np.nansum(w * tb[:, 1]):>+9.2f}{'':>7}{'':>20}{'':>9}"
            f"{np.nansum(w * tb[:, 2]):>7.2f}{np.nansum(w * tb[:, 1]) * np.nansum(w * tb[:, 2]):>8.1f}")
        sign = [np.nanmean(np.array(acc[f], float)[:, 1]) > 0 for f in acc]
        log(f"  ⭐부호 양수 폴드 {sum(sign)}/{len(sign)} · "
            f"앞 2폴드 건수 비중 {(np.mean([x[0] for x in acc[0]]) + np.mean([x[0] for x in acc[1]]))/tot*100:.1f}%")
    log("\n🔴읽는 법: 엣지가 앞 폴드에 몰려 있고 최근 폴드가 0 근처면, 4폴드 평균은")
    log("  「실력」이 아니라 「2023~24 장세」다. 그 상태로 미접촉 OOS 를 열면 안 된다.")
    return 0


def wf(pick, LBLS, CELLS):
    """--oracle --wf: ②를 **워크포워드**로 다시 -- 유일한 결함(사후선택)을 없앤다.

    앞 폴드(F1·F2)에서 «ATR 십분위 → 배리어» 지도와 «최선 고정칸»을 **둘 다** 정하고,
    뒤 폴드(F3·CAND)에 그대로 적용한다. 분위 경계도 TRAIN 에서 잡아 TEST 에 적용한다(인과).
    ⭐①도 TRAIN 에서 골라야 공정하다 -- 한쪽만 사후선택이면 그 차이를 실력으로 읽는다.
    🔴여기서 ②−①이 시드폭 안으로 들어오면 변동성 기반 sltp 버킷 헤드는 만들 이유가 없다.
    """
    TR, TE = {0, 1}, {2, 3}                      # F1·F2 로 정하고 F3·CAND 에서 잰다
    for arm in ARMS:
        rows = []
        for _thr, per_fold in pick(arm):
            R = [_raw(per_fold, tp, sl) for tp, sl in CELLS]
            P = np.stack([r[0] for r in R]); H = np.stack([r[1] for r in R])
            days, fid = R[0][2], R[0][3]
            atr = np.concatenate([_atr_pct(te)[idx] for te, _h, _l, _c, _s, idx in per_fold])
            tr = np.isin(fid, list(TR)); te_ = np.isin(fid, list(TE))

            fix = int(np.argmax([_ratio_day(P[k][tr], H[k][tr]) for k in range(len(CELLS))]))
            edges = np.quantile(atr[tr], np.linspace(0, 1, 11)[1:-1])   # TRAIN 경계
            dtr, dte = np.digitize(atr[tr], edges), np.digitize(atr[te_], edges)
            kmap = np.full(10, fix, int)
            for d in range(10):
                m = dtr == d
                if m.sum() >= 30:              # 표본이 적은 분위는 고정칸으로 둔다
                    kmap[d] = int(np.argmax([_ratio_day(P[k][tr][m], H[k][tr][m])
                                             for k in range(len(CELLS))]))
            kd = kmap[dte]; ar = np.where(te_)[0]
            rows.append({
                "fix_cell": CELLS[fix], "n_te": int(te_.sum()),
                "fix_day": _ratio_day(P[fix][te_], H[fix][te_]), "fix_bp": P[fix][te_].mean(),
                "dec_day": _ratio_day(P[kd, ar], H[kd, ar]), "dec_bp": P[kd, ar].mean(),
                "fix_pd": 288.0 / max(H[fix][te_].mean(), 1e-9),
                "dec_pd": 288.0 / max(H[kd, ar].mean(), 1e-9),
                "n_cells": len(set(kmap.tolist())),
                "ci_fix": E.block_ci(P[fix][te_], days[te_])[:2],
                "ci_dec": E.block_ci(P[kd, ar], days[te_])[:2]})
        A = pd.DataFrame(rows)
        log(f"\n{'='*104}\n■ {arm} — {_lbl(arm)} · **워크포워드** "
            f"(지도는 F1·F2, 평가는 F3·CAND {A.n_te.iloc[0]:,}건 · 5슬롯)\n{'='*104}")
        log(f"{'방식':<28}{'건당bp':>9}{'건/일':>8}{'순/일':>9}{'Δ':>8}   CI95(건당)")
        log(f"{'①고정(TRAIN 선택) ' + str(A.fix_cell.mode().iloc[0]):<28}{A.fix_bp.mean():>+9.2f}"
            f"{A.fix_pd.mean():>8.2f}{A.fix_day.mean():>9.1f}{0.0:>8.1f}"
            f"   [{np.mean([c[0] for c in A.ci_fix]):+7.2f},{np.mean([c[1] for c in A.ci_fix]):+7.2f}]")
        log(f"{'②ATR 십분위(TRAIN 지도)':<26}{A.dec_bp.mean():>+9.2f}{A.dec_pd.mean():>8.2f}"
            f"{A.dec_day.mean():>9.1f}{A.dec_day.mean()-A.fix_day.mean():>+8.1f}"
            f"   [{np.mean([c[0] for c in A.ci_dec]):+7.2f},{np.mean([c[1] for c in A.ci_dec]):+7.2f}]")
        d = A.dec_bp - A.fix_bp
        log(f"  슬롯별 ②−① : {[round(x, 2) for x in d]}  ⇒ 부호 일치 {int((d > 0).sum())}/5")
        log(f"  지도가 쓴 서로 다른 칸 {A.n_cells.mean():.1f}/10분위")
    log("\n⭐판정: ②−①이 양수이고 5슬롯 부호가 일치하며 시드폭을 넘어야 헤드를 만들 값어치가 있다.")
    return 0


def oracle(pick, LBLS):
    """--oracle: **봉별 배리어 선택의 상한과 현실적 하한**.

    사용자 제안(2026-09-17)인 「sltp 버킷 헤드」의 천장을 «만들기 전에» 잰다.
    🔴목적함수는 Σp/Σh 비율이므로 봉별 최적화는 argmax(p/h) 가 아니라 **Dinkelbach**
    (argmax(p − λh) 를 λ 수렴까지)다. 비율의 평균으로 고르면 짧은 거래에 지배돼 상한이 부푼다.

      ①고정      -- 20칸 중 순/일 최선 (지금 하는 것)
      ②변동성분위 -- ATR 십분위마다 «그 분위 안에서» 최선 고정칸 (사후선택 10번)
                     ⭐이게 변동성 피쳐만 쓰는 버킷 헤드가 **실제로 도달 가능한 수준**이다
      ③오라클     -- 건별 20칸 사후선택 (도달 불가 상한)
    """
    CELLS = [(tp, sl) for tp in TPS for sl in SLS]
    if "--wf" in sys.argv:
        return wf(pick, LBLS, CELLS)
    for arm in ARMS:
        rows = []
        for _si, (_thr, per_fold) in enumerate(pick(arm)):
            R = [_raw(per_fold, tp, sl) for tp, sl in CELLS]
            P = np.stack([r[0] for r in R]); H = np.stack([r[1] for r in R])
            days, fid = R[0][2], R[0][3]
            atr = np.concatenate([_atr_pct(te)[idx] for te, _h, _l, _c, _s, idx in per_fold])
            n = P.shape[1]; ar = np.arange(n)

            fix = int(np.argmax([_ratio_day(P[k], H[k]) for k in range(len(CELLS))]))
            dec = pd.qcut(atr, 10, labels=False, duplicates="drop")
            kd = np.empty(n, int)
            for d in np.unique(dec):                      # ②분위 안에서만 고른다
                m = dec == d
                kd[m] = int(np.argmax([_ratio_day(P[k][m], H[k][m]) for k in range(len(CELLS))]))
            lam = _ratio_day(P[fix], H[fix]) / 288.0      # ③Dinkelbach
            for _ in range(30):
                ko = (P - lam * H).argmax(0)
                new = float(P[ko, ar].sum()) / max(float(H[ko, ar].sum()), 1e-9)
                if abs(new - lam) < 1e-12:
                    break
                lam = new
            rows.append({
                "fix_cell": CELLS[fix], "n": n,
                "fix_day": _ratio_day(P[fix], H[fix]), "fix_bp": P[fix].mean(),
                "dec_day": _ratio_day(P[kd, ar], H[kd, ar]), "dec_bp": P[kd, ar].mean(),
                "orc_day": _ratio_day(P[ko, ar], H[ko, ar]), "orc_bp": P[ko, ar].mean(),
                "dec_cells": len(set(kd.tolist())), "orc_top": float(np.bincount(ko, minlength=len(CELLS)).max() / n),
                "fix_pd": 288.0 / max(H[fix].mean(), 1e-9), "orc_pd": 288.0 / max(H[ko, ar].mean(), 1e-9),
                "dec_pd": 288.0 / max(H[kd, ar].mean(), 1e-9),
                "ci_fix": E.block_ci(P[fix], days)[:2], "ci_dec": E.block_ci(P[kd, ar], days)[:2]})
        A = pd.DataFrame(rows)
        log(f"\n{'='*104}\n■ {arm} — {_lbl(arm)} · 배리어 선택의 천장 "
            f"({len(CELLS)}칸 · {A.n.iloc[0]:,}건 · 5슬롯 평균)\n{'='*104}")
        log(f"{'방식':<26}{'건당bp':>9}{'건/일':>8}{'순/일':>9}{'Δ고정':>9}   CI95(건당)")
        log(f"{'①고정 ' + str(A.fix_cell.mode().iloc[0]):<26}{A.fix_bp.mean():>+9.2f}"
            f"{A.fix_pd.mean():>8.2f}{A.fix_day.mean():>9.1f}{0.0:>9.1f}"
            f"   [{np.mean([c[0] for c in A.ci_fix]):+7.2f},{np.mean([c[1] for c in A.ci_fix]):+7.2f}]")
        log(f"{'②ATR 십분위 조건부':<24}{A.dec_bp.mean():>+9.2f}{A.dec_pd.mean():>8.2f}"
            f"{A.dec_day.mean():>9.1f}{A.dec_day.mean()-A.fix_day.mean():>+9.1f}"
            f"   [{np.mean([c[0] for c in A.ci_dec]):+7.2f},{np.mean([c[1] for c in A.ci_dec]):+7.2f}]"
            f"  ← 도달 가능")
        log(f"{'③건별 오라클(도달 불가)':<23}{A.orc_bp.mean():>+9.2f}{A.orc_pd.mean():>8.2f}"
            f"{A.orc_day.mean():>9.1f}{A.orc_day.mean()-A.fix_day.mean():>+9.1f}")
        log(f"  ②가 쓴 서로 다른 칸 {A.dec_cells.mean():.1f}/10분위 · "
            f"③최빈칸 점유 {A.orc_top.mean()*100:.1f}%")
    log("\n⭐판정: ②−① 이 시드폭보다 작으면 **변동성 기반 sltp 버킷 헤드는 만들 이유가 없다**.")
    log("  ②는 사후선택 10번이라 그 자체로 위로 편향돼 있다 -- 진짜 헤드는 이보다 낮다.")
    log("  ③−② 는 «변동성으로 설명 안 되는 나머지»다. 크면 다른 피쳐 축이 남아 있다는 뜻.")
    return 0




def occ(pick, LBLS):
    """--occ: **왜 0.8건/일인가**. 용량(288/평균보유)은 2.75 인데 실제가 0.80 이다.

    ⭐빈도 = 점유율 × 용량. 점유율이 낮으면 원인은 배리어가 아니라 «신호가 시간축에 몰려서
    슬롯이 빈 동안 아무것도 안 온다»는 것이다. 에피소드(연속 신호 덩어리)를 세어 가른다.
    """
    tp, sl = K.BASE_TP, K.BASE_SL
    GAPB = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--episode=")), 12))
    days = sum((pd.Timestamp(v1) - pd.Timestamp(v0)).days + 1 for _n, _a, _b, v0, v1 in FOLDS)
    for arm in ARMS:
        acc = []
        for _thr, per_fold in pick(arm):
            nsig = nep = nfill = occb = totb = 0; eplen = []; idle = []; epbp = []
            for te, h, l, c, side, idx in per_fold:
                if len(idx) < 20:
                    continue
                r, hh, _a, _b, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
                p_ = r * 1e4 - COST
                nsig += len(idx); totb += len(te)
                # 에피소드 = 신호 사이 간격이 GAPB 봉 이하면 같은 덩어리
                brk = np.where(np.diff(idx) > GAPB)[0]
                st = np.concatenate([[0], brk + 1]); en = np.concatenate([brk, [len(idx) - 1]])
                nep += len(st); eplen.append((idx[en] - idx[st]).astype(float) + 1)
                epbp.append(np.array([p_[a:b + 1].mean() for a, b in zip(st, en)]))
                cur = -1; prev_free = 0
                for i in range(len(idx)):
                    if idx[i] <= cur:
                        continue
                    idle.append(float(idx[i] - prev_free)); nfill += 1
                    occb += int(hh[i]); cur = idx[i] + int(hh[i]); prev_free = cur
            acc.append((nsig, nep, nfill, occb / max(totb, 1), float(np.mean(np.concatenate(eplen))),
                        float(np.median(idle)), float(np.mean(idle)),
                        float(np.mean(np.concatenate(epbp)))))
        a = np.array(acc, float)
        log(f"\n{'='*100}\n■ {arm} — {_lbl(arm)} · **점유율 분해** (TP{tp*100:g}%/SL{sl*100:g}% · "
            f"에피소드 경계 {GAPB}봉 · {days}일)\n{'='*100}")
        sig, ep, fl = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
        log(f"  신호 {sig:,.0f} ({sig/days:.2f}/일) · **에피소드 {ep:,.0f} ({ep/days:.2f}/일)** · "
            f"체결 {fl:,.0f} ({fl/days:.2f}/일)")
        log(f"  에피소드당 신호 {sig/ep:.1f}개 · 평균 길이 {a[:, 4].mean():.1f}봉")
        log(f"  **슬롯 점유율 {a[:, 3].mean()*100:.1f}%** · 유휴(직전 청산→다음 진입) "
            f"중앙 {a[:, 5].mean():.0f}봉 · 평균 {a[:, 6].mean():.0f}봉")
        log(f"  에피소드 평균 건당bp {a[:, 7].mean():+.2f}")
        log(f"  ⭐에피소드/일 {ep/days:.2f} 가 체결/일 {fl/days:.2f} 과 비슷하면 «군집»이 아니라")
        log(f"    **독립 사건 자체가 부족**한 것이다 -- 그러면 고칠 곳은 집행이 아니라 신호 생성이다.")
    return 0


def ckey(arm, fold, ei, sd):
    """캐시 키 규약이 팔마다 다르다 -- N0 는 전문가 «이름», N5/N7 은 «정수» 인덱스.
    `N7@tag` 면 키 끝에 `@tag` 가 붙는다(라벨별로 갈린 캐시)."""
    b, _, tag = arm.partition("@")
    if b in ("N0", "N0x2"):
        return f"{fold}|N0{EN[ei]}s{sd}"
    return f"{fold}|{b}{ei}s{sd}" + (f"@{tag}" if tag else "")


# --cache2=<경로> : 두 번째 캐시의 확률을 **평균**한다(2026-09-18). 같은 라벨·같은 폴드를
# 다른 «모델 계열»(TabM ↔ LightGBM)로 학습한 것을 섞기 위한 것이다. 시드 앙상블이 같은
# 편향을 반복하는 데 반해 이건 편향이 다른 둘을 섞는다.
CACHE2 = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cache2=")), "")


def main() -> int:
    z = np.load(CACHE, allow_pickle=True)
    if CACHE2:
        z2 = np.load(CACHE2, allow_pickle=True)
        miss = [k for k in z.files if k not in z2.files]
        assert not miss, f"--cache2 에 없는 키: {miss[:4]}"
        z = {k: np.array({"D": (dict(z[k].item())["D"] + dict(z2[k].item())["D"]) / 2.0,
                          "Q": (dict(z[k].item())["Q"] + dict(z2[k].item())["Q"]) / 2.0},
                         dtype=object) for k in z.files}
        log(f"⭐--cache2 평균: {len(z)}키")
    df, _ = E.load()
    log(f"캐시 {CACHE} · 시드 {SEEDS} · 폴드 {[f[0] for f in FOLDS]} · 격자 {len(TPS)}×{len(SLS)}")

    bars, SEGS = {}, {a: [] for a in ARMS}      # SEGS[arm] = [(name, te, hi, lo, cl, [(D,Q)×5])]
    for name, _t0, _t1, v0, v1 in FOLDS:
        te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
        ev = tabm._route_probs(te).argmax(1)
        bars[name] = (te, pd.to_numeric(te["high"]).to_numpy(np.float64),
                      pd.to_numeric(te["low"]).to_numpy(np.float64),
                      pd.to_numeric(te["close"]).to_numpy(np.float64))
        for arm in ARMS:
            K_ = _n1k(arm)
            per_seed = []
            for sd in SEEDS:
                if K_:                          # 라우팅 없음 -- 전문가 인덱스가 없다
                    c = dict(z[f"{name}|N1s{sd}"].item()); per_seed.append((c["D"], c["Q"]))
                    continue
                D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
                for ei in range(3):             # balnobb 하드 라우팅(argmax)
                    c = dict(z[ckey(arm, name, ei, sd)].item()); m = ev == ei
                    D[m], Q[m] = c["D"][m], c["Q"][m]
                per_seed.append((D, Q))
            S_ = len(SEEDS)
            if K_:                              # 시드 i..i+K-1 평균
                slots = [(np.mean([per_seed[(i + j) % S_][0] for j in range(K_)], 0),
                          np.mean([per_seed[(i + j) % S_][1] for j in range(K_)], 0))
                         for i in range(S_)]
            elif _base(arm) == "N0x2":          # 같은 폴드의 시드 i, i+1 앙상블
                slots = [((per_seed[i][0] + per_seed[(i + 1) % S_][0]) / 2.0,
                          (per_seed[i][1] + per_seed[(i + 1) % S_][1]) / 2.0)
                         for i in range(S_)]
            else:
                slots = per_seed
            SEGS[arm].append((name, *bars[name], slots))
        log(f"  {name}: {len(te):,}봉 · {len(ARMS)}팔")

    # ── 진입 집합은 TP/SL 과 무관하므로 팔·슬롯당 «한 번»만 정한다 ──
    # --permatch: 임계값을 통합이 아니라 **폴드마다** 잡아 건수를 고르게 만든다.
    # ⭐이게 「그 창에서 더 «자주» 발화한다」와 「더 «잘» 맞춘다」를 가른다.
    # 🔴그 폴드 자신의 확률분포를 보고 정하므로 **배포 가능한 규칙이 아니라 진단용 대조군**이다.
    PERFOLD = "--permatch" in sys.argv
    PER_N = TARGET // len(FOLDS)
    # --rollq=<후보수> : 단일 q 대신 **직전 N개 후보의 분위**를 임계값으로 쓴다(인과).
    # 🔴발단: 단일 q 는 확률분포가 시간에 따라 이동하면 특정 창에 발화를 몰아준다
    #   (2026-09-18 실측: 레짐 없는 판이 F3=잃는 창에 64.7% 를 쏟아부어 +25.39 -> +16.54).
    # shift(1) 로 «그 후보 자신»을 빼고, 워밍업 구간은 확장창 분위로 메운다.
    ROLLQ = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--rollq=")), 0))
    # --rollqq=<분위> : q 를 고정한다(동결·배포용). 0 이면 건수맞춤으로 이분탐색(비교용).
    ROLLQQ = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--rollqq=")), 0.0))

    def _thr_seq(qf_c, q):
        """후보 계열에 대한 인과적 롤링 분위. 반환: 같은 길이의 임계값 배열."""
        s_ = pd.Series(qf_c).shift(1)
        t_ = s_.rolling(ROLLQ, min_periods=200).quantile(q)
        return t_.fillna(s_.expanding(min_periods=50).quantile(q)).to_numpy()

    def pick(arm):
        out = []
        for si in range(len(SEEDS)):
            sc = []
            for _n, te, _h, _l, _c, slots in SEGS[arm]:
                sc.append(gscore(*slots[si], atr=_atr_pct(te) if SCORE == "edgev" else None))
            if ROLLQ and ROLLQQ > 0:               # ⭐q 를 «고정»한다 -- 배포 가능한 형태.
                # 🔴건수맞춤(TARGET)은 팔 간 비교용이다. 창 길이가 다른 폴드에 그대로 쓰면
                #   임계값이 창 길이에 따라 달라진다(2026-09-18: 50일 폴드에서 40건/일이 나왔다).
                def _sides_q(q):
                    out_ = []
                    for da, qf in sc:
                        m_ = da != 0
                        t_ = _thr_seq(qf[m_], q)
                        ok = np.zeros(len(da), bool)
                        ok[np.where(m_)[0]] = np.isfinite(t_) & (qf[m_] >= t_)
                        out_.append(np.where((da == 1) & ok, 1.0, np.where((da == 2) & ok, -1.0, 0.0)))
                    return out_
                sides, thrs = _sides_q(ROLLQQ), [ROLLQQ]
            elif ROLLQ:                            # 분위 q 를 이분탐색해 총 건수를 맞춘다
                def _sides(q):
                    out_ = []
                    for da, qf in sc:
                        m_ = da != 0
                        t_ = _thr_seq(qf[m_], q)
                        ok = np.zeros(len(da), bool)
                        ok[np.where(m_)[0]] = np.isfinite(t_) & (qf[m_] >= t_)
                        out_.append(np.where((da == 1) & ok, 1.0, np.where((da == 2) & ok, -1.0, 0.0)))
                    return out_
                lo_q, hi_q = 0.50, 0.9995
                for _ in range(26):
                    mid = (lo_q + hi_q) / 2
                    if sum(int((x != 0).sum()) for x in _sides(mid)) > TARGET: lo_q = mid
                    else: hi_q = mid
                qq = (lo_q + hi_q) / 2
                sides, thrs = _sides(qq), [qq]
            elif PERFOLD:
                thrs = [float(np.sort(qf[da != 0])[::-1][min(PER_N, int((da != 0).sum())) - 1])
                        for da, qf in sc]
                sides = None
            else:
                allq = np.concatenate([qf[da != 0] for da, qf in sc])
                thrs = [float(np.sort(allq)[::-1][min(TARGET, len(allq)) - 1])] * len(sc)
                sides = None
            per_fold = []
            for i_, ((name, te, h, l, c, _s), (da, qf)) in enumerate(zip(SEGS[arm], sc)):
                if sides is not None:
                    side = sides[i_]
                else:
                    thr = thrs[i_] if PERFOLD else thrs[0]
                    side = np.where((da == 1) & (qf >= thr), 1.0,
                                    np.where((da == 2) & (qf >= thr), -1.0, 0.0))
                ix = np.where(side != 0)[0]
                QF_STORE[(arm, si, i_)] = qf[ix]      # 켈리용 -- 튜플 규약은 그대로 둔다
                per_fold.append((te, h, l, c, side, ix))
            out.append((float(np.mean(thrs)), per_fold))
            log(f"  {arm} 슬롯{si}: {'롤링분위 q=' + format(thrs[0], '.4f') + f' (창 {ROLLQ} 후보)' if ROLLQ else ('폴드별 ' + str([round(t, 3) for t in thrs]) if PERFOLD else 'q=' + format(thrs[0], '.4f'))}"
                f" · 진입 {sum(len(f[5]) for f in per_fold):,}건 {[len(f[5]) for f in per_fold]}")
        return out

    def cell(entries, tp, sl):
        tpb, slb = tp * 1e4, sl * 1e4
        acc = []
        for _thr, per_fold in entries:
            pnl, hold, days, rsn = [], [], [], []
            for te, h, l, c, side, idx in per_fold:
                if len(idx) < 20:
                    continue
                r, hh, _res, rn, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
                pnl.append(r * 1e4 - COST); hold.append(hh.astype(float)); rsn.append(rn)
                days.append(te.timestamp.dt.floor("D").to_numpy()[idx])
            pnl = np.concatenate(pnl); hold = np.concatenate(hold)
            days = np.concatenate(days); rsn = np.concatenate(rsn)
            lo_, hi_, nd = E.block_ci(pnl, days)
            g = float(pnl.mean()); pdy = 288.0 / max(hold.mean(), 1e-9)
            acc.append({"n": len(pnl), "g": g, "lo": lo_, "hi": hi_, "indep": nd,
                        "med": float(np.median(hold)), "per_day": pdy,
                        "net_day": g * pdy, "net_day_peg": (g + COST - PEG) * pdy,
                        "sl_r": float((rsn == 0).mean()), "tp_r": float((rsn == 1).mean()),
                        "un_r": float((rsn == 2).mean())})
        A = pd.DataFrame(acc); g = A.g.mean()
        return {"tp": tp, "sl": sl, "n": int(A.n.mean()), "gross_bp": g,
                "p": (g + COST + slb) / (tpb + slb), "p_star": (slb + COST) / (tpb + slb),
                "ci95": [A.lo.mean(), A.hi.mean()], "indep_days": A.indep.mean(),
                "median_hold": A.med.mean(), "per_day": A.per_day.mean(),
                "net_day": A.net_day.mean(), "net_day_peg": A.net_day_peg.mean(),
                "sl_rate": A.sl_r.mean(), "tp_rate": A.tp_r.mean(), "un_rate": A.un_r.mean(),
                "seed_spread": A.g.max() - A.g.min()}

    # ── --idle=U,qmin : **유휴 조건부 게이트**(2026-09-18). 정지시각 문제의 구조 그대로다.
    # 슬롯이 비어 있던 시간 u 가 길수록 기다림의 기회비용이 커지므로 문턱을 낮춘다:
    #   q(u) = q0 - (q0 - qmin) · min(u/U, 1)
    # 🔴인과성: q(u) 는 «이미 지난 시간»만 보고, 분위 자체는 shift(1) 롤링창이다. 미래 없음.
    # ⭐전역 완화와의 차이: 전역은 바쁜 구간(=이미 포지션이 있어 버려질 곳)에도 문턱을 낮춰
    #   신호 6.8배에 체결 2.9배만 얻는다. 유휴 조건부는 «실제로 쓰이는 곳»에서만 낮춘다.
    if "--idle" in " ".join(sys.argv):
        IU, IQMIN = (next((a.split("=", 1)[1] for a in sys.argv
                           if a.startswith("--idle=")), "288,0.70").split(","))
        IU, IQMIN = int(IU), float(IQMIN)
        assert ROLLQ and ROLLQQ > 0, "--idle 은 --rollq 와 --rollqq(고정 q0) 를 함께 요구한다"
        QL = sorted({round(ROLLQQ - (ROLLQQ - IQMIN) * i / 8, 4) for i in range(9)}, reverse=True)
        days = {nm: (pd.Timestamp(v1) - pd.Timestamp(v0)).days + 1 for nm, _a, _b, v0, v1 in FOLDS}
        tp, sl = K.BASE_TP, K.BASE_SL
        for arm in ARMS:
            acc, slotpool = {}, []
            for si in range(len(SEEDS)):
                pool = []
                for fi, (nm, te, h, l, c, slots) in enumerate(SEGS[arm]):
                    da, qf = gscore(*slots[si], atr=_atr_pct(te) if SCORE == "edgev" else None)
                    m_ = da != 0; wi = np.where(m_)[0]
                    # 후보 계열 위에서 q 수준마다 롤링 임계값을 한 번씩 구해 봉 격자로 되돌린다
                    TH = np.full((len(QL), len(da)), np.inf)
                    for k_, q_ in enumerate(QL):
                        TH[k_, wi] = _thr_seq(qf[m_], q_)
                    side = np.where(da == 1, 1.0, np.where(da == 2, -1.0, 0.0))
                    r, hh, _a, _b, _m = K._first_touch_open(wi, side, h, l, c, tp, sl, K.MAXBARS)
                    pnl_all = r * 1e4 - COST
                    take, cur, freed = [], -1, 0
                    for j, i_ in enumerate(wi):
                        if i_ <= cur:
                            continue
                        u = i_ - freed
                        k_ = min(int(len(QL) - 1), int(len(QL) * min(u / IU, 1.0)))
                        t_ = TH[k_, i_]
                        if not np.isfinite(t_) or qf[i_] < t_:
                            continue
                        take.append(j); cur = i_ + int(hh[j]); freed = cur
                    t_ = np.array(take, int)
                    dd = te.timestamp.dt.floor("D").to_numpy()[wi[t_]]
                    acc.setdefault(fi, []).append((len(t_), float(pnl_all[t_].mean()),
                                                   float(np.median(hh[t_]))))
                    pool.append((pnl_all[t_], dd))
                slotpool.append((np.concatenate([x[0] for x in pool]),
                                 np.concatenate([x[1] for x in pool])))
            log(f"\n{'='*104}\n■ {arm} — {_lbl(arm)} · **유휴 조건부 게이트** "
                f"(q0={ROLLQQ} → qmin={IQMIN} · 감쇠 {IU}봉 · TP{tp*100:g}%/SL{sl*100:g}% · "
                f"비용 {COST}bp)\n{'='*104}")
            log(f"{'폴드':<6}{'체결':>7}{'체결/일':>8}{'건당bp':>9}{'중앙보유':>9}")
            T = np.zeros(2)
            for fi, (nm, *_r) in enumerate(FOLDS):
                a = np.array(acc[fi], float)
                log(f"{nm:<6}{a[:, 0].mean():>7.0f}{a[:, 0].mean()/days[nm]:>8.2f}"
                    f"{a[:, 1].mean():>+9.2f}{a[:, 2].mean():>9.0f}")
                T += [a[:, 0].mean(), a[:, 0].mean() * a[:, 1].mean()]
            td = sum(days.values()); bw = T[1] / max(T[0], 1e-9)
            pc = np.array([E.block_ci(p_, d_) for p_, d_ in slotpool], float)
            log(f"{'합계':<6}{T[0]:>7.0f}{T[0]/td:>8.2f}{bw:>+9.2f}"
                f"   CI [{pc[:, 0].mean():+7.2f},{pc[:, 1].mean():+7.2f}] · "
                f"순/일 {bw*T[0]/td:.1f} · peg순/일 {(bw-PEG+COST)*T[0]/td:.1f}"
                f"{'  ✅peg배제' if pc[:, 0].mean() > PEG - COST else ''}")
        return 0

    ROLLQ_G[0] = ROLLQ
    if "--gap1m" in sys.argv:
        return gap1m(pick, LBLS)
    if "--gap" in sys.argv:
        return gapscan(pick, LBLS)
    if "--slotsweep" in sys.argv:
        return slotsweep(pick, LBLS)
    if "--seqgrid" in sys.argv:
        return seqgrid(pick, LBLS)
    if "--wait" in " ".join(sys.argv):
        return waitrule(pick, LBLS)
    if "--occ" in sys.argv:
        return occ(pick, LBLS)
    if "--seq" in sys.argv:
        return seq(pick, LBLS)
    if "--byregime" in sys.argv:
        return byregime(pick, LBLS)
    if "--byfold" in sys.argv:
        return byfold(pick, LBLS)
    if "--oracle" in sys.argv:
        return oracle(pick, LBLS)

    LBLS.update({a: f"라우팅 없음 ×{_n1k(a)}" for a in ARMS if _n1k(a)})
    LBL = dict(LBLS)
    LBL = {a: LBL[_base(a)] + (f" [{a.split('@')[1]}]" if "@" in a else "") for a in ARMS}
    rows, bests = [], {}
    for arm in ARMS:
        entries = pick(arm)
        R = pd.DataFrame([dict(arm=arm, **cell(entries, tp, sl)) for tp in TPS for sl in SLS])
        # ⭐한 팔 안에서 건수가 같아야 이 표가 「청산 비교」다. 다르면 선별성이 섞인 것이다.
        assert R.n.nunique() == 1, f"{arm}: 셀마다 건수가 다르다 {sorted(R.n.unique())}"
        rows += R.to_dict("records")
        log(f"\n{'='*118}\n■ {arm} — {LBL[arm]} · 동일 진입 {R.n.iloc[0]:,}건 · 5슬롯 평균 · "
            f"시간청산 없음(최대 {K.MAXBARS}봉)\n{'='*118}")
        log(f"{'TP':>5}{'SL':>5}{'건당bp':>9}{'함축p':>8}{'손익분기':>9}{'CI95':>20}"
            f"{'SL%':>7}{'TP%':>7}{'미해소':>7}{'중앙보유':>8}{'건/일':>7}{'순/일':>8}{'peg':>8}{'시드폭':>8}")
        for _, r in R.iterrows():
            star = " 🔴" if r.per_day < 1.0 else ("  ✅" if r.ci95[0] > 0 else "   ")
            log(f"{r.tp*100:>4.1f}%{r.sl*100:>4.1f}%{r.gross_bp:>+9.2f}{r.p*100:>7.2f}%"
                f"{r.p_star*100:>8.2f}%  [{r.ci95[0]:+7.2f},{r.ci95[1]:+7.2f}]"
                f"{r.sl_rate*100:>6.1f}%{r.tp_rate*100:>6.1f}%{r.un_rate*100:>6.1f}%"
                f"{r.median_hold:>8.0f}{r.per_day:>7.2f}{r.net_day:>8.1f}{r.net_day_peg:>8.1f}"
                f"{r.seed_spread:>8.2f}{star}")
        ok = R[R.per_day >= 1.0]                       # 🔴«최소 1건/일» 을 못 지키면 후보가 아니다
        bests[arm] = (R[(R.tp == 0.015) & (R.sl == 0.010)].iloc[0],
                      (ok if len(ok) else R).loc[(ok if len(ok) else R).net_day.idxmax()])

    log(f"\n{'='*118}\n■ 라벨 서열 — «각자 최적 배리어»에서 다시 (건/일 ≥ 1 제약)\n{'='*118}")
    log(f"{'팔':<6}{'라벨':<26}{'현행1.5/1.0':>13}{'자기최적':>12}{'배리어':>12}"
        f"{'건/일':>7}{'순/일':>8}{'시드폭':>8}")
    for arm in ARMS:
        cur, bst = bests[arm]
        log(f"{arm:<6}{LBL[arm]:<26}{cur.gross_bp:>+13.2f}{bst.gross_bp:>+12.2f}"
            f"  TP{bst.tp*100:>3.1f}%/SL{bst.sl*100:.1f}%{bst.per_day:>7.2f}"
            f"{bst.net_day:>8.1f}{bst.seed_spread:>8.2f}")
    log("⭐읽는 법: 현행 대비 «자기최적»이 크게 오르는 팔이 있으면 기존 서열이 배리어 정렬의")
    log("  산물이었다는 뜻이다. 순위가 그대로면 배리어는 서열의 원인이 아니다.")
    log("⚠️팔마다 20칸 중 최고를 고른 값이므로 «위로» 편향돼 있다 — 시드폭·CI 폭과 함께 읽는다.")

    out = E.OUT / "stageT_sltp_grid_by_label.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    log(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
