"""**지평 없는 배리어 라벨** — 보유시간을 라벨에서 뺀다 (2026-09-14).

사용자: *"지평은 신경쓰지 말고 보유시간도 없애. 라벨 생성 로직을 다시 깎아보자."*

## 무엇이 달라지나
지평 라벨(`..._breakeven_sweep_...`)은 배포 청산 스택을 굴렸고, 그 스택은 **보유시간이 핵심 축**이라
라벨이 「H 분 뒤 어디 있나」를 물었다. 여기서는 시간을 묻지 않는다: **먼저 닿는 배리어가 답**이다.
  · 롱: 위 +U% 에 먼저 닿으면 이김, 아래 −D% 에 먼저 닿으면 손절
  · 숏: 대칭 (아래 −U% 이김, 위 +D% 손절)
상한 시간은 없다(계산을 위해 60일 캡만 두고 **미해결률을 보고**한다).

## 왜 손익분기가 달라지나 — 이게 이 판의 요점
지평 라벨의 손익분기는 「E|y| 대비 비용」이라 비용 5.88bp 가 E|y| 98bp 의 6% 를 먹었다(53.0%).
배리어 라벨은 **이익·손실 크기를 내가 정한다**. ±3% 면 한 건의 크기가 300bp 라 비용 비중이
2% 로 떨어진다 ⇒ 손익분기 정확도가 **50% 에 붙는다**. 정확한 식(비대칭 페이오프):
    이김 = +U − (진입 2.95 + peg 청산 2.93)           [TP 는 지정가라 메이커]
    짐  = −(D + 슬리피지 14bp) − (진입 2.95 + 테이커 5.0)
    손익분기 p = |짐| / (이김 + |짐|)
펀딩은 보유분만큼 실제로 더한다 — **시간을 안 재는 대신 시간이 비용으로 들어온다**(이게 지평을
없앤 대가이고, 숨기면 안 된다).

## 규약
· 결정은 봉 i 종가, 배리어 탐색은 **i+1 부터** (사건-라벨 경계 계약)
· 같은 봉에서 양쪽 다 닿으면 **손절 우선**(라이브는 봉 안 순서를 모른다 — 보수적으로)
· 배리어 판정은 **봉내 고가/저가** — `omega4_6_1_live.evaluate_exit` 의 라이브 컨벤션
· size-free(레버리지 미포함) · `y = ½(r_long − r_short)·1e4`, `m = ½(r_long + r_short)`
· `--selftest` 는 우선도달을 느린 참조 구현과 대조한다
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
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import research_fresh_forward_random_entry_stack_20260914 as H  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402
from scripts.live_eth_trade_plan_20260913 import funding_cost_bp  # noqa: E402

OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
UPS = (0.01, 0.02, 0.03, 0.05)          # 익절 배리어
DOWNS = (0.02, 0.03, 0.05)              # 손절 배리어 (사용자 기술 = 3%)
MAXB = 17_280                           # 60일. 시간 제한이 아니라 계산 상한 -- 미해결률을 본다
FUNDING = True                          # --no-funding 으로 끈다(모듈 전역: leg 가 본다)


def first_touch(hi, lo, i: int, up_px: float, dn_px: float, loss_is_up: bool,
                maxb: int = MAXB) -> tuple[int, str] | None:
    """봉 i+1 부터 먼저 닿는 배리어. 같은 봉에 둘 다면 **손절 쪽**을 준다. 슬라이스를 2배씩 늘린다."""
    n = len(hi); j0 = i + 1; span = 512
    while j0 < n and (j0 - i - 1) < maxb:
        j1 = min(n, j0 + span, i + 1 + maxb)
        a = hi[j0:j1] >= up_px; b = lo[j0:j1] <= dn_px
        both = a | b
        if both.any():
            k = int(np.argmax(both))
            if a[k] and b[k]:
                return j0 + k, ("up" if loss_is_up else "dn")
            return j0 + k, ("up" if a[k] else "dn")
        j0 = j1; span *= 2
    return None


def leg(c, hi, lo, i: int, side: int, u: float, dn: float):
    """측면 side(1=롱,2=숏) 의 (순수익 r, 해결 봉 수, 이김여부, 비용bp). 배리어 두 개만."""
    px = c[i]
    if side == 1:
        up_px, dn_px, loss_is_up, win_key = px * (1 + u), px * (1 - dn), False, "up"
    else:
        up_px, dn_px, loss_is_up, win_key = px * (1 + dn), px * (1 - u), True, "dn"
    t = first_touch(hi, lo, i, up_px, dn_px, loss_is_up)
    if t is None:
        return None
    j, which = t
    win = which == win_key
    if win:
        fill, cost = u, H.ENTRY_BP + H.PEG_EXIT_BP
    else:
        fill, cost = -(dn + H.STOP_SLIP_MED_BP / 1e4), H.ENTRY_BP + H.TAKER_EXIT_BP
    if FUNDING:
        # 🔴이 펀딩은 **상수**(FUNDING_BP_8H_FALLBACK 0.52bp/8h)다 -- 실제 펀딩 계열이 아니다.
        # 그래서 y 안에 «보유시간의 결정론적 함수»가 박히고(부호 내부 상관 −1.00), 보유시간을
        # 예측하는 피쳐(= 변동성)가 **방향 IC 를 공짜로** 얻는다(dp_vol_forecast 0.173).
        # 방향 축에서는 반드시 끈다(--no-funding). 2026-09-14 발견.
        cost += funding_cost_bp(5 * (j - i), "LONG" if side == 1 else "SHORT")
    return fill - cost / 1e4, j - i, win, cost


def label_window(c, hi, lo, ok, lo_i: int, hi_i: int, stride: int, u: float, dn: float) -> dict:
    idx, y, m, bars, winL, unres = [], [], [], [], [], 0
    for i in range(lo_i, hi_i, stride):
        if not ok[i]:
            continue
        a = leg(c, hi, lo, int(i), 1, u, dn); b = leg(c, hi, lo, int(i), 2, u, dn)
        if a is None or b is None:
            unres += 1; continue
        idx.append(i); y.append(0.5e4 * (a[0] - b[0])); m.append(0.5e4 * (a[0] + b[0]))
        bars.append(0.5 * (a[1] + b[1])); winL.append(a[2])
    n = len(idx)
    return {"idx": np.array(idx), "y": np.array(y), "m": np.array(m), "bars": np.array(bars),
            "win_long": np.array(winL), "unresolved": unres / max(n + unres, 1)}


def breakeven(u: float, dn: float) -> dict:
    """비대칭 페이오프의 정확한 손익분기(펀딩 제외 — 펀딩은 보유분마다 달라 라벨 안에 들어간다)."""
    win_bp = u * 1e4 - (H.ENTRY_BP + H.PEG_EXIT_BP)
    loss_bp = dn * 1e4 + H.STOP_SLIP_MED_BP + (H.ENTRY_BP + H.TAKER_EXIT_BP)
    return {"win_bp": win_bp, "loss_bp": loss_bp, "breakeven_acc": loss_bp / (win_bp + loss_bp)}


def dilate_labels(idx: np.ndarray, y: np.ndarray, bars: np.ndarray, k: int,
                  q: float) -> tuple[np.ndarray, np.ndarray]:
    """**«이 부근은 롱이야»** — 강한 앵커의 부호를 ±k 봉으로 퍼뜨린다 (사용자 2026-09-14).

    강한 앵커 = **가장 빨리 해결된** (1−q) 분위. 각 봉은 **가장 가까운** 강한 앵커의 부호를 받고,
    ±k 안에 강한 앵커가 없으면 학습에서 **뺀다**(중립 구간까지 억지로 편 들 이유가 없다).

    🔴이건 **학습 라벨**이다. 평가는 반드시 그 봉의 **진짜** y·m 으로 한다 -- 팽창 라벨로 평가하면
    「내가 퍼뜨린 답을 내가 맞혔다」가 되어 성과가 부풀려진다.

    🔴🔴**「강함」을 |y| 로 정의하면 안 된다**(2026-09-14 실측 버그): 배리어 라벨은 |y| 가 구조적으로
    거의 상수라(3%/3% 는 ≈308bp) 남는 미세 변동을 **펀딩이 지배**한다 -- 롱은 내고 숏은 받으므로
    |y| 상위 30% 가 **전부 숏**이 됐고(10,574개 중 y>0 이 0개) 분류기가 단일 클래스로 학습됐다.
    속도는 그 대칭을 깨지 않는다: 「2시간 만에 +8%」가 「5일 걸린 +8%」보다 강한 롱이다."""
    thr = float(np.quantile(bars, 1.0 - q))
    strong = np.where(bars <= thr)[0]
    if len(strong) == 0:
        return np.arange(len(idx)), np.sign(y)
    pos = idx[strong]                                  # 강한 앵커의 봉 인덱스
    j = np.searchsorted(pos, idx)                      # 가장 가까운 강한 앵커 찾기
    left = np.clip(j - 1, 0, len(pos) - 1); right = np.clip(j, 0, len(pos) - 1)
    dl = np.abs(idx - pos[left]); dr = np.abs(idx - pos[right])
    near = np.where(dl <= dr, left, right); dist = np.minimum(dl, dr)
    keep = np.where(dist <= k)[0]
    return keep, np.sign(y[strong[near[keep]]])


def readout(idx, y, m, pred, block_bars: int) -> dict:
    s = np.sign(pred); s[s == 0] = 1.0
    net = m + s * y
    b = (idx // max(int(block_bars), 1)).astype(np.int64)
    bm = np.array([net[b == k].mean() for k in np.unique(b)])
    t = float(bm.mean() / (bm.std(ddof=1) / np.sqrt(len(bm)))) if len(bm) > 2 and bm.std(ddof=1) > 0 else float("nan")
    # 🔴net = m + s·y 에서 **m 은 측면과 무관**하다(양측 공통 = 스트래들 항). 방향이 번 돈은 s·y 뿐이다.
    bd = np.array([(s * y)[b == k].mean() for k in np.unique(b)])
    td = float(bd.mean() / (bd.std(ddof=1) / np.sqrt(len(bd)))) if len(bd) > 2 and bd.std(ddof=1) > 0 else float("nan")
    return {"net_bp": float(net.mean()), "block_t": t, "hit": float((np.sign(y) == s).mean()),
            "blocks": int(len(bm)), "dir_bp": float((s * y).mean()), "dir_block_t": td}


def selftest(c, hi, lo) -> None:
    """우선도달을 느린 참조 구현(한 봉씩 전진)과 대조한다."""
    rng = np.random.default_rng(20260914)
    for i in rng.integers(1000, len(c) - 20000, size=300):
        px = c[i]; up, dn = px * 1.02, px * 0.985
        ref = None
        for j in range(i + 1, min(len(c), i + 1 + MAXB)):
            a, b = hi[j] >= up, lo[j] <= dn
            if a or b:
                ref = (j, "dn" if (a and b) else ("up" if a else "dn")); break
        got = first_touch(hi, lo, int(i), up, dn, False)
        assert got == ref, f"봉 {i}: 빠른판정 {got} vs 참조 {ref}"
    print("자체점검 통과: 우선도달 300앵커가 느린 참조와 일치(같은봉 양쪽 = 손절 우선 포함)")


def straddle(c, hi, lo, ok, win, ups, downs, stride: int, tag: str, gate=None) -> int:
    """**스트래들 축** — 같은 봉에 롱·숏을 동시에 넣는다. 모델도 방향도 없다 (사용자 2026-09-14).

    `m = ½(r_long + r_short)` 는 측면과 무관한 항이라 **방향 예측 없이** 먹는 값이다. U≫D 면
    한쪽이 D 에서 잘리고 다른 쪽이 U 까지 가는 구조 = 롱 감마. 대신 횡보장에서는 양쪽 다 잘린다.

    🔴학습이 없으므로 TRAIN/VAL/OOS/TEST 구분이 의미 없다 -- **2022~23(다른 레짐)을 포함해 전 창**을
    같은 자격으로 본다. 이 저장소는 「시기 한정」을 네 번 봤다.
    ⚠️자본은 **두 다리**다. 연수익은 2배 명목 기준으로 읽어야 한다.

    `gate=(이름, 값배열)` 을 주면 **TRAIN 창에서 자른 5분위**로 나눠 잰다(문턱을 TRAIN 에서만 뽑아
    다른 창에 그대로 적용 -- 창 안에서 분위를 다시 뽑으면 그 창의 미래를 쓰는 셈이다).
    스트래들은 방향이 아니라 **크기**가 필요하므로, 이 저장소가 맞히는 축(변동성)으로 켜고 끌 수 있는지가
    유일하게 남은 질문이다."""
    gname, gval, gq = (None, None, None)
    if gate is not None:
        gname, gval = gate
        tl, th = win["TRAIN"]
        gq = np.nanquintile if False else np.nanpercentile(gval[tl:th], [20, 40, 60, 80])
        print(f"게이트 {gname}: TRAIN 5분위 경계 {np.round(gq, 5)}")
    rows = []
    print(f"\n{'익절/손절':>11} {'창':>10} {'n':>6} {'쌍당bp':>8} {'블록t':>7} {'중앙h':>7} "
          f"{'양쪽손절':>8} {'연쌍수':>7} {'연bp(순차)':>11}")
    for u in ups:
        for dn in downs:
            key = f"{u*100:g}%/{dn*100:g}%"
            cell = {}
            for k in ("BACK22_23", "TRAIN", "VAL", "OOS", "TEST"):
                if k not in win:
                    continue
                w = label_window(c, hi, lo, ok, *win[k], stride, u, dn)
                if len(w["idx"]) < 50:
                    continue
                m, bars = w["m"], w["bars"]
                bb = float(np.median(bars))
                b = (w["idx"] // max(int(bb), 1)).astype(np.int64)
                bm = np.array([m[b == j].mean() for j in np.unique(b)])
                t = float(bm.mean() / (bm.std(ddof=1) / np.sqrt(len(bm)))) if len(bm) > 2 and bm.std(ddof=1) > 0 else float("nan")
                hrs = bb * 5 / 60
                # 양쪽 손절 = 쌍 수익이 최악(≈ −(D+슬립+비용))인 경우
                both_stop = float((m < -(dn * 1e4)).mean())
                per_yr = 8760 / hrs if hrs > 0 else float("nan")
                cell[k] = {"n": int(len(m)), "pair_bp": float(m.mean()), "block_t": t, "hours": hrs,
                           "both_stop": both_stop, "pairs_per_year": per_yr,
                           "annual_bp_sequential": float(m.mean()) * per_yr, "blocks": int(len(bm))}
                if gq is not None:
                    g = np.digitize(gval[w["idx"]], gq)
                    cell[k]["by_gate"] = [
                        {"q": int(j), "n": int((g == j).sum()),
                         "pair_bp": float(m[g == j].mean()) if (g == j).any() else float("nan"),
                         "hours": float(np.median(bars[g == j]) * 5 / 60) if (g == j).any() else float("nan")}
                        for j in range(5)]
                q = cell[k]
                extra = ("  분위 " + " ".join(f"{b['pair_bp']:>+7.1f}" for b in q["by_gate"])
                         if gq is not None else "")
                print(f"{key:>11} {k:>10} {q['n']:>6,} {q['pair_bp']:>+8.1f} {t:>+7.2f} {hrs:>7.1f} "
                      f"{both_stop:>8.1%} {per_yr:>7.0f} {q['annual_bp_sequential']:>+11,.0f}{extra}", flush=True)
            rows.append((key, cell)); print()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(dict(rows), indent=1, ensure_ascii=False))
    print(f"저장: {OUT/(tag+'.json')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stride", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ups", default="")
    ap.add_argument("--downs", default="")
    ap.add_argument("--eval-stride", type=int, default=12)
    ap.add_argument("--dilate", type=int, default=0, help=">0 이면 강한 앵커 부호를 ±k봉으로 팽창(학습만)")
    ap.add_argument("--strong-q", type=float, default=0.7, help="강한 앵커 = |y| 상위 (1-q)")
    ap.add_argument("--tag", default="barrier")
    ap.add_argument("--vol-artifact", default="", help="volfc/volexp 에 쓸 연구용 아티팩트 경로")
    ap.add_argument("--gate", default="", help="이 컬럼의 TRAIN 5분위로 나눠 잰다(예: rv48)")
    ap.add_argument("--dump", default="", help="예: 3/3 -- 그 셀 라벨을 npz 로 저장하고 끝")
    ap.add_argument("--straddle", action="store_true", help="모델 없이 양측 동시진입 항 m 만 전 창에서")
    ap.add_argument("--no-funding", action="store_true",
                    help="라벨에서 상수 펀딩을 뺀다 -- 방향 축은 반드시 켤 것")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    global FUNDING
    FUNDING = not a.no_funding
    if a.no_funding:
        print("펀딩 제외: y 에서 보유시간의 결정론적 성분을 뺀다")
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    ok = sm["ok"]
    if a.selftest:
        selftest(c, hi, lo); return 0
    selftest(c, hi, lo)
    if a.dump:
        u, dn = (float(x) / 100 for x in a.dump.split("/"))
        ts = d.timestamp.to_numpy()
        z = {"names": np.array(["TRAIN", "VAL", "OOS", "TEST"]), "up": u, "down": dn}
        for k in ("TRAIN", "VAL", "OOS", "TEST"):
            w = label_window(c, hi, lo, ok, *win[k],
                             a.train_stride if k == "TRAIN" else a.eval_stride, u, dn)
            z[f"{k}_idx"], z[f"{k}_y"], z[f"{k}_m"] = w["idx"], w["y"], w["m"]
            z[f"{k}_bars"] = w["bars"]; z[f"{k}_ts"] = ts[w["idx"]].astype("int64")
            print(f"  {k:>5}: {len(w['idx']):,}개 · 롱승 {float((w['y']>0).mean()):.1%} · "
                  f"E|y| {np.abs(w['y']).mean():.0f}bp · 표류 {w['m'].mean():+.1f}bp")
        OUT.mkdir(parents=True, exist_ok=True)
        f = OUT / (f"labels_{a.dump.replace('/', '_').replace('%', 'pct')}"
                   f"{'_nofund' if a.no_funding else ''}.npz")
        np.savez_compressed(f, **z); print(f"저장: {f}")
        return 0
    if a.straddle:
        ups = [float(x) for x in a.ups.split(",")] if a.ups else list(UPS)
        downs = [float(x) for x in a.downs.split(",")] if a.downs else list(DOWNS)
        gate = None
        if a.gate in ("volfc", "volexp"):
            # 배포된 전방 변동성 예측(`live_eth_sizing_vol_model_20260912`, train_end 2025-08-31,
            # 표본외 예측상관 0.789). ⚠️TRAIN·2022~23 은 그 모델의 **학습 안**이라 유리하게 나온다.
            import live_eth_sizing_vol_model_20260912 as svm
            if a.vol_artifact:                      # 연구용 재학습 아티팩트(2022~23 도려낸 판)
                svm.ARTIFACT = pathlib.Path(a.vol_artifact)
            art = svm.load_model()
            assert art is not None, "사이징 변동성 모델 아티팩트가 없다"
            X = svm.build_features(d.timestamp, c, d.quote_volume.to_numpy(float),
                                   d.trades.to_numpy(float), hi, lo)
            fc = np.log(np.clip(svm.predict_vol(art["models"], X), 1e-9, None))
            if a.gate == "volexp":       # 「곧 커진다」 = 예측 / 현재 실현
                fc = fc - np.log(np.clip(d["rv48"].to_numpy(float), 1e-9, None))
            print(f"게이트 {a.gate}: 아티팩트 {svm.ARTIFACT.name} · 학습 "
                  f"{art.get('train_start', '처음')}~{art['train_end']} · 유효 {np.isfinite(fc).mean():.1%}")
            gate = (a.gate, fc)
        elif a.gate:
            assert a.gate in d.columns, f"{a.gate} 가 프레임에 없다 (가능: rv12/rv48/rv288/park48/atr_pct 등)"
            gate = (a.gate, d[a.gate].to_numpy(float))
        return straddle(c, hi, lo, ok, win, ups, downs, a.eval_stride, a.tag, gate)
    seeds = [int(x) for x in np.random.default_rng(20260914).integers(1, 1_000_000, size=a.seeds)]
    ups = [float(x) for x in a.ups.split(",")] if a.ups else list(UPS)
    downs = [float(x) for x in a.downs.split(",")] if a.downs else list(DOWNS)
    res = {"seeds": seeds, "train_stride": a.train_stride, "cells": {}}
    print(f"\n{'익절/손절':>11} {'창':>5} {'n':>6} {'미해결':>5} {'중앙시간':>7} {'롱TP율':>6} "
          f"{'E|y|':>7} {'표류bp':>7} {'손익분기':>8} {'적중':>7} {'순손익bp':>8} {'블록t':>6} "
          f"{'방향bp':>8} {'방향t':>7} {'블록':>5}")
    for u in ups:
        for dn in downs:
            key = f"{u*100:g}%/{dn*100:g}%"
            be = breakeven(u, dn)
            lab = {k: label_window(c, hi, lo, ok, *win[k], a.train_stride if k == "TRAIN" else a.eval_stride, u, dn)
                   for k in ("TRAIN", "VAL", "OOS", "TEST")}
            tr = lab["TRAIN"]
            cell = {"breakeven": be, "windows": {}}
            ics = [abs(spearmanr(S[tr["idx"], j], tr["y"], nan_policy="omit").statistic)
                   for j in range(S.shape[1])]
            ics = np.nan_to_num(np.array(ics))
            cell["single_feature"] = {"max_abs_ic": float(ics.max()), "col": cols[int(ics.argmax())]}
            preds = {}
            if a.dilate > 0:
                keep, ylab = dilate_labels(tr["idx"], tr["y"], tr["bars"], a.dilate, a.strong_q)
                share = float((ylab > 0).mean())
                assert 0.02 < share < 0.98, (
                    f"팽창 라벨이 한쪽으로 쏠렸다(롱 {share:.1%}) -- 분류기가 단일 클래스로 "
                    f"학습되면 상수 예측이 나와 모든 팔이 같은 숫자를 낸다")
                cell["train_rows"] = {"all": int(len(tr["idx"])), "used": int(len(keep)),
                                      "long_share": share, "dilate": a.dilate,
                                      "strong_q": a.strong_q}
                from sklearn.ensemble import HistGradientBoostingClassifier
                for sd in seeds:
                    mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                         max_leaf_nodes=15, min_samples_leaf=200,
                                                         l2_regularization=1.0, random_state=sd)
                    mdl.fit(S[tr["idx"][keep]], (ylab > 0).astype(int))
                    preds[sd] = {k: mdl.predict_proba(S[lab[k]["idx"]])[:, 1] - 0.5
                                 for k in ("VAL", "OOS", "TEST")}
            else:
                cell["train_rows"] = {"all": int(len(tr["idx"])), "used": int(len(tr["idx"]))}
                for sd in seeds:
                    mdl = P.twin_fit_predict(S, tr["idx"], tr["y"], sd)
                    preds[sd] = {k: mdl.predict(S[lab[k]["idx"]]) for k in ("VAL", "OOS", "TEST")}
            for k in ("VAL", "OOS", "TEST"):
                w = lab[k]; bb = float(np.median(w["bars"])) if len(w["bars"]) else 1.0
                rs = [readout(w["idx"], w["y"], w["m"], preds[sd][k], bb) for sd in seeds]
                agg = {f: {"mean": float(np.mean([r[f] for r in rs])),
                           "se": float(np.std([r[f] for r in rs], ddof=1) / np.sqrt(len(rs)))}
                       for f in ("net_bp", "block_t", "hit", "dir_bp", "dir_block_t")}
                ey = float(np.mean(np.abs(w["y"]))); dr = float(w["m"].mean())
                be_sign = 0.5 + abs(dr) / (2 * ey) if ey > 0 else float("nan")
                cell["windows"][k] = agg | {"n": int(len(w["idx"])), "unresolved": w["unresolved"],
                                            "median_bars": bb, "long_win_rate": float(w["win_long"].mean()),
                                            "blocks": rs[0]["blocks"], "drift_bp": dr, "e_abs_y_bp": ey,
                                            "breakeven_sign_acc": be_sign,
                                            "always_long_bp": dr + float(w["y"].mean()),
                                            "always_short_bp": dr - float(w["y"].mean())}
                q = cell["windows"][k]
                print(f"{key:>11} {k:>5} {q['n']:>6,} {q['unresolved']:>5.1%} {bb*5/60:>6.1f}h "
                      f"{q['long_win_rate']:>6.1%} {ey:>7.0f} {dr:>+7.1f} {be_sign:>8.2%} "
                      f"{agg['hit']['mean']:>7.2%} {agg['net_bp']['mean']:>+8.2f} "
                      f"{agg['block_t']['mean']:>+6.2f} {agg['dir_bp']['mean']:>+8.2f} "
                      f"{agg['dir_block_t']['mean']:>+7.2f} {q['blocks']:>5,}", flush=True)
            res["cells"][key] = cell
            print()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"저장: {OUT/(a.tag+'.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
