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
import json, re, sys
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
FOLDS = [f for f in K.FOLDS if f[0] in ("F1", "F2", "F3", "CAND")]
CACHE = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cache=")),
                  str(E.OUT / "stageP_probs.npz")))
TARGET, COST, PEG = 3700, 1.02, 5.52
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


def _lbl(a):
    return LBLS.get(_base(a), f"라우팅 없음 ×{_n1k(a)}" if _n1k(a) else a)


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



def ckey(arm, fold, ei, sd):
    """캐시 키 규약이 팔마다 다르다 -- N0 는 전문가 «이름», N5/N7 은 «정수» 인덱스.
    `N7@tag` 면 키 끝에 `@tag` 가 붙는다(라벨별로 갈린 캐시)."""
    b, _, tag = arm.partition("@")
    if b in ("N0", "N0x2"):
        return f"{fold}|N0{EN[ei]}s{sd}"
    return f"{fold}|{b}{ei}s{sd}" + (f"@{tag}" if tag else "")


def main() -> int:
    z = np.load(CACHE, allow_pickle=True)
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
                D, Q = slots[si]; da = D.argmax(1)
                qf = np.where(da > 0, Q[np.arange(len(Q)), da], Q[:, 0])
                sc.append((da, qf))
            if ROLLQ:                              # 분위 q 를 이분탐색해 총 건수를 맞춘다
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
                per_fold.append((te, h, l, c, side, np.where(side != 0)[0]))
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
