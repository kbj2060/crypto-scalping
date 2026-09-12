#!/usr/bin/env python3
"""탐지 Phase 0 — **모델을 안 만들어도 되는지** 먼저 확인한다 (2026-09-11, 사용자 승인).

배포된 탐지(거래대금·체결속도 z288 q90 AND)의 성적(포착률 98.4% · 지연 5분 · 진행률 6.08%)은
전부 **volexp 1.80 교차** 기준이다. 그 기준이 틀렸다는 게 오늘 드러났다(부드러운 추세를 놓친다).
그래서 같은 규칙을 **moveabs 기준**으로 다시 재고, 모델이 필요한지 본다.

세 팔 — 새 모델을 만들기 **전에** 답이 나오는지
  (a) 현행 규칙          z288 q90 AND. 압축 게이트 유/무 둘 다.
  (b) 경보 모델 재활용     동결 아티팩트 점수를 **낮은 임계**로 그어 탐지로 쓴다(지평 24봉 그대로)
  (c) 지평 12봉 재학습     같은 구성으로 지평만 바꿔 새로 적합
셋 중 하나로 충분하면 탐지 전용 모델은 만들지 않는다.

⭐**포착률을 맞춰 비교한다** — 커버리지가 아니다. 더 자주 켜는 쪽이 그냥 포착률이 높아 보인다.
  각 팔의 임계를 «현행 규칙과 같은 포착률»에 맞춘 뒤 헛발동/진행률/지연을 겨룬다.

사건 정의  m12[t] = (t, t+12] 절대 최대 이탈폭. 창별 상위 5% 를 «큰 이동»으로 본다.
          사건 시작 = 그 조건의 **상승 엣지**(직전 봉은 아니었던 봉). 사건 폭 = [t, t+12].
지표      포착률(사건 중 창 안에서 켜진 비율) · 지연(중앙, 봉) · 진행률(켜진 시점 이동폭/전체)
          · 헛발동/일(어느 사건 창에도 없는 발동)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_breakout_alert_tabpfn_20260911 as R  # noqa: E402

ART = ROOT / "data" / "live" / "eth_breakout_alert_artifact"
OUT = ROOT / "tmp" / "eth_detect_phase0_20260911"
H_DET = 12               # 탐지 지평 1시간 — 경보 24봉과 같은 라벨족, 짧은 창
TOPQ = 0.95
TRAIL = 2016             # 후행 분위 창(봉). 배포 규칙과 같다
BARS_PER_DAY = 288


# ────────────────────────────────────────────────── 사건과 탐지 지표
def events_of(m: np.ndarray, ok: np.ndarray, thr: float) -> np.ndarray:
    """큰 이동 사건의 **시작 봉** 인덱스. 상승 엣지만 센다(한 사건을 여러 번 세지 않는다)."""
    big = ok & np.isfinite(m) & (m >= thr)
    prev = np.r_[False, big[:-1]]
    return np.flatnonzero(big & ~prev)


def detect_metrics(fire: np.ndarray, ev: np.ndarray, c: np.ndarray, sel: np.ndarray) -> dict:
    """포착률·지연·진행률·헛발동. 사건 창은 [시작, 시작+H_DET].

    ⚠️`fire`/`covered` 는 **전체 길이** 배열이고, 일수 환산만 창 봉수(`sel.sum()`)로 한다.
      첫 판에서 이 둘을 한 인자(n)로 뭉쳐 브로드캐스트가 깨졌다.
    """
    if not len(ev):
        return {}
    n = len(fire)
    fired = np.flatnonzero(fire)
    caught, delays, progress = 0, [], []
    covered = np.zeros(n, bool)
    for t in ev:
        hi = min(t + H_DET, n - 1)
        covered[t:hi + 1] = True
        w = fired[(fired >= t) & (fired <= hi)]
        if not len(w):
            continue
        caught += 1
        f = int(w[0]); delays.append(f - t)
        seg = c[t:hi + 1]
        total = float(np.max(np.abs(seg - seg[0])))
        progress.append(abs(c[f] - c[t]) / total if total > 0 else np.nan)
    false = int((fire & sel & ~covered).sum())
    nbar = max(int(sel.sum()), 1)
    return {"events": int(len(ev)), "recall": caught / len(ev),
            "delay_med": float(np.median(delays)) if delays else np.nan,
            "delay_q90": float(np.quantile(delays, 0.9)) if delays else np.nan,
            "progress_med": float(np.nanmedian(progress)) if progress else np.nan,
            "false_per_day": false / (nbar / BARS_PER_DAY),
            "fire_rate": float(fire[sel].mean()),
            # ⭐결정 지표: 포착해도 이미 간 만큼은 못 피한다.
            #   절감률 = 포착률 x (1 - 진행률) = «사건 총 이동폭 중 피할 수 있었던 비율»
            #   포착률만 보면 늦게 켜는 팔이 유리해 보이고, 진행률만 보면 거의 안 켜는 팔이 이긴다.
            "avoided": (caught / len(ev)) * (1 - float(np.nanmedian(progress))) if progress else 0.0}


def thr_for_recall(score: np.ndarray, ok: np.ndarray, ev: np.ndarray, c: np.ndarray,
                   target: float) -> tuple[float, dict]:
    """목표 포착률을 내는 **가장 높은**(=가장 조용한) 임계를 찾는다."""
    cands = np.unique(np.quantile(score[ok & np.isfinite(score)],
                                  np.linspace(0.50, 0.9995, 160)))
    best = (None, None)
    for t in cands[::-1]:                      # 조용한 쪽부터 내려온다
        met = detect_metrics(ok & np.isfinite(score) & (score >= t), ev, c, ok)
        if met.get("recall", 0) >= target:
            return float(t), met
        best = (float(t), met)
    return best


def trailing_thr(x: np.ndarray, q: float, win: int = TRAIL) -> np.ndarray:
    """후행 분위(전 봉) — shift(1) 로 자기 봉을 안 본다."""
    return pd.Series(x).rolling(win, min_periods=200).quantile(q).shift(1).to_numpy()


def trailing_thr_masked(x: np.ndarray, q: float, mask: np.ndarray,
                        win: int = TRAIL) -> np.ndarray:
    """**마스크가 참인 봉만 모아** 후행 창에서 분위를 낸다 — 배포 모듈 `_thr` 와 같은 규약.

    배포는 `x[comp][-2016:]` 의 분위를 쓴다(압축 봉 2016개 ≈ 실봉 5,170개). 전 봉 rolling 으로
    재면 **다른 규칙을 재는 것**이라 «현행이 이 정도다»가 성립하지 않는다.
    """
    idx = np.flatnonzero(mask & np.isfinite(x))
    if len(idx) < 200:
        return np.full(len(x), np.inf)
    qs = pd.Series(x[idx]).rolling(win, min_periods=200).quantile(q).shift(1).to_numpy()
    out = np.full(len(x), np.nan)
    out[idx] = qs
    return pd.Series(out).ffill().to_numpy()          # 비마스크 봉은 직전 확정 임계를 쓴다


# ────────────────────────────────────────────────── 경보 아티팩트 재활용
def load_alert_models(device: str):
    meta = json.loads((ART / "meta.json").read_text())
    z = np.load(ART / "context.npz")
    from tabpfn import TabPFNClassifier
    models, refs = [], []
    for i, sd in enumerate(meta["seeds"]):
        m = TabPFNClassifier(device=device, random_state=sd, n_estimators=meta["n_estimators"])
        m.fit(z[f"X_{i}"], z[f"y_{i}"])
        models.append(m); refs.append(z[f"ref_{i}"])
    return meta, models, refs


def score_with(models, refs, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.mean([np.searchsorted(refs[i], models[i].predict_proba(X[rows])[:, 1],
                                    side="right") / len(refs[i])
                    for i in range(len(models))], axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klines", default=str(R.KL5))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train", action="store_true",
                    help="(c) 지평 12봉 · move_atr 라벨로 **새로 적합**해 탐지로 쓴다")
    ap.add_argument("--cap", type=int, default=10000)
    ap.add_argument("--n-est", type=int, default=8)
    ap.add_argument("--meta-train", action="store_true",
                    help="(d) **진짜 메타라벨링** — 규칙 발동 봉만으로 새로 적합한다. "
                         "(메타) 사후필터는 전 봉 학습 모델을 빌려 쓰는 약한 판이다")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    raw = R.build(Path(a.klines), h=H_DET)        # ⭐탐지 지평으로 라벨 재료를 만든다
    # 라벨은 경보와 **같은 족보**(move_atr = 이탈폭/ATR96, 창별 상위 5%). 지평만 12봉이다.
    p = R.label(raw, "move", cand=R.apply_gate(raw, "none"))
    feats = R.feature_cols(p)
    X = p[feats].to_numpy(np.float32)
    ylab = p["y"].to_numpy()
    c = pd.read_csv(Path(a.klines), usecols=["close"]).close.to_numpy(float)
    n = len(p)
    m12 = p["fwd_move_abs"].to_numpy()
    volexp = p["volexp"].to_numpy()
    comp = p["compressed"].to_numpy()
    watch = pd.Series(comp).rolling(12, min_periods=1).max().to_numpy() == 1

    base_ok = np.isfinite(m12) & np.isfinite(p[feats]).all(axis=1).to_numpy()
    print(f"[패널] {n:,}봉 · 탐지 지평 {H_DET}봉(1시간) · 사건=창별 상위 {1-TOPQ:.0%}")

    z288 = {k: p[f"z_{k}_288"].to_numpy() for k in ("qv", "n")}

    # ── (c) 지평 12봉 재학습. TRAIN 구간만 컨텍스트로 쓰고, 기준분포는 컨텍스트 밖에서 낸다
    det_models, det_refs = [], []
    if a.train:
        tr = np.flatnonzero(base_ok & (p.win == "TRAIN").to_numpy())
        seeds = tuple(int(x) for x in np.random.default_rng(20260911).choice(10**6, a.seeds,
                                                                            replace=False))
        from tabpfn import TabPFNClassifier
        ctxs = []
        for sd in seeds:
            idx = tr[R.sub_idx(len(tr), a.cap, np.random.default_rng(sd))]
            m = TabPFNClassifier(device=a.device, random_state=sd, n_estimators=a.n_est)
            m.fit(X[idx], ylab[idx]); det_models.append(m); ctxs.append(idx)
            print(f"  [탐지모델] 시드 {sd:7d} 컨텍스트 {len(idx):,} 적합", flush=True)
        pool = np.setdiff1d(tr, np.unique(np.concatenate(ctxs)))
        ref = np.sort(np.random.default_rng(7).choice(pool, min(20000, len(pool)), replace=False))
        for i, m in enumerate(det_models):
            det_refs.append(np.sort(m.predict_proba(X[ref])[:, 1]).astype(np.float32))
        print(f"  [탐지모델] 기준분포 {len(ref):,}행 완료", flush=True)

    # ── (d) 진짜 메타라벨링: **규칙 발동 봉만으로** 적합한다.
    #    라벨은 그대로 시장 결과(move_atr) — 규칙은 모집단만 정한다(순환 없음).
    meta_models, meta_refs = [], []
    if a.meta_train:
        z288_all = {k: p[f"z_{k}_288"].to_numpy() for k in ("qv", "n")}
        rule_all = base_ok.copy()
        for k_, x_ in z288_all.items():
            rule_all = rule_all & np.isfinite(x_) & (x_ >= trailing_thr(x_, 0.90))
        trm = np.flatnonzero(rule_all & (p.win == "TRAIN").to_numpy())
        print(f"  [메타모델] 규칙 발동 TRAIN {len(trm):,}행 · 그중 양성 "
              f"{ylab[trm].mean()*100:.2f}% (전 봉 기저 {ylab[base_ok].mean()*100:.2f}%)", flush=True)
        from tabpfn import TabPFNClassifier
        seeds_m = tuple(int(x) for x in np.random.default_rng(777).choice(10**6, a.seeds,
                                                                         replace=False))
        ctxm = []
        for sd in seeds_m:
            idx = trm[R.sub_idx(len(trm), a.cap, np.random.default_rng(sd))]
            m = TabPFNClassifier(device=a.device, random_state=sd, n_estimators=a.n_est)
            m.fit(X[idx], ylab[idx]); meta_models.append(m); ctxm.append(idx)
            print(f"  [메타모델] 시드 {sd:7d} 컨텍스트 {len(idx):,} 적합", flush=True)
        poolm = np.setdiff1d(trm, np.unique(np.concatenate(ctxm)))
        refm = np.sort(np.random.default_rng(7).choice(poolm, min(20000, len(poolm)),
                                                       replace=False))
        for m in meta_models:
            meta_refs.append(np.sort(m.predict_proba(X[refm])[:, 1]).astype(np.float32))
        print(f"  [메타모델] 기준분포 {len(refm):,}행 완료", flush=True)

    rows = []
    for w, aa, bb in [x for x in R.WINDOWS if x[0] in R.EVAL_WINS]:
        sel = ((p.timestamp >= aa) & (p.timestamp <= bb)).to_numpy() & base_ok
        if sel.sum() < 2000:
            continue
        thr_ev = float(np.nanquantile(m12[sel], TOPQ))
        ev = events_of(m12, sel, thr_ev)
        nn = int(sel.sum())
        print(f"\n=== {w}  {nn:,}봉 · 사건 {len(ev)}건 "
              f"({len(ev)/(nn/BARS_PER_DAY):.1f}건/일) · 임계 {thr_ev*100:.3f}% ===")

        arms: dict[str, np.ndarray] = {}
        # (a1) 현행 배포 규칙 — 압축 감시창 + z288 q90 AND, 후행 분위(인과)
        f_dep = sel & watch
        for k, x in z288.items():
            f_dep = f_dep & np.isfinite(x) & (x >= trailing_thr_masked(x, 0.90, comp))
        arms["(a1) 현행 규칙(배포 그대로)"] = f_dep
        # (a2) 같은 규칙, 게이트만 제거 — 경보에서 게이트가 역효과였다
        f_ng = sel.copy()
        for k, x in z288.items():
            f_ng = f_ng & np.isfinite(x) & (x >= trailing_thr(x, 0.90))
        arms["(a2) 게이트·임계모집단 제거"] = f_ng
        f_g2 = sel & watch                       # 감시창만 남기고 임계는 전 봉 모집단으로
        for k, x in z288.items():
            f_g2 = f_g2 & np.isfinite(x) & (x >= trailing_thr(x, 0.90))
        arms["(a3) 감시창 유지, 임계 전봉"] = f_g2

        mets = {nm: detect_metrics(f, ev, c, sel) for nm, f in arms.items()}
        # 넘어야 할 선은 **최고 규칙**이다. (a1)에 맞추면 모델을 약한 기준에 묶는 셈이다.
        best_rule_nm = max(mets, key=lambda k: mets[k].get("avoided", 0))
        target = mets[best_rule_nm]["recall"]
        print(f"  {'팔':26s} {'포착률':>7s} {'지연':>6s} {'진행률':>7s} {'⭐절감률':>8s} {'헛발동/일':>9s}")
        for nm, f in arms.items():
            m_ = mets[nm]
            print(f"  {nm:26s} {m_['recall']*100:6.1f}% {m_['delay_med']*5:5.0f}분 "
                  f"{m_['progress_med']*100:6.2f}% {m_['avoided']*100:7.1f}% {m_['false_per_day']:8.1f}")
            rows.append({"win": w, "arm": nm, **m_})
        print(f"  ── 아래: 최고 규칙 «{best_rule_nm[:18]}» 의 포착률 {target*100:.1f}% 에 맞춤 ──")
        yield_arms = {}
        idx = np.flatnonzero(sel)
        if ART.exists() and (ART / "context.npz").exists():
            meta, models, refs = load_alert_models(a.device)
            sc = np.full(n, np.nan)
            sc[idx] = score_with(models[:a.seeds], refs[:a.seeds], X, idx)
            yield_arms["(b) 경보모델(24봉) 재활용"] = sc
        if det_models:
            sc = np.full(n, np.nan)
            sc[idx] = score_with(det_models, det_refs, X, idx)
            yield_arms["(c) 탐지모델(12봉) 재학습"] = sc
            # 단변량 최강 피쳐도 같은 자리에서 — 모델이 그걸 넘는지 본다
            sc2 = np.full(n, np.nan); sc2[idx] = p["z_n_96"].to_numpy()[idx]
            yield_arms["(참고) z_n_96 단독"] = sc2
        # ── (d) 진짜 메타라벨링: 규칙 발동 봉만으로 적합한 모델로 그 안에서 고른다
        if meta_models:
            fr = arms[best_rule_nm]
            ii = np.flatnonzero(fr)
            scm = np.full(n, np.nan)
            scm[ii] = score_with(meta_models, meta_refs, X, ii)
            for keep in (1.0, 0.5, 0.3):
                cut = float(np.nanquantile(scm[ii], 1 - keep)) if keep < 1.0 else -np.inf
                f = fr & (np.isfinite(scm) & (scm >= cut) if keep < 1.0 else True)
                m_ = detect_metrics(f, ev, c, sel)
                lbl = "(d) 메타학습 전체" if keep >= 1.0 else f"(d) 메타학습 상위 {int(keep*100)}%"
                print(f"  {lbl:26s} {m_['recall']*100:6.1f}% {m_['delay_med']*5:5.0f}분 "
                      f"{m_['progress_med']*100:6.2f}% {m_['avoided']*100:7.1f}% {m_['false_per_day']:8.1f}")
                rows.append({"win": w, "arm": lbl, **m_})

        # ── (메타) 사후 필터: 전 봉 학습 모델을 규칙 발동 봉에만 적용 — 약한 판
        meta_src = yield_arms.get("(b) 경보모델(24봉) 재활용")
        if meta_src is not None:
            fire_rule = arms[best_rule_nm]
            v = meta_src[fire_rule]
            v = v[np.isfinite(v)]
            for keep in (0.5, 0.3):
                if len(v) < 50:
                    continue
                cut = float(np.quantile(v, 1 - keep))
                f = fire_rule & np.isfinite(meta_src) & (meta_src >= cut)
                m_ = detect_metrics(f, ev, c, sel)
                print(f"  {'(메타) 규칙발동 중 상위 '+str(int(keep*100))+'%':26s} "
                      f"{m_['recall']*100:6.1f}% {m_['delay_med']*5:5.0f}분 "
                      f"{m_['progress_med']*100:6.2f}% {m_['avoided']*100:7.1f}% {m_['false_per_day']:8.1f}")
                rows.append({"win": w, "arm": f"meta_top{int(keep*100)}", **m_})

        for nm, sc in yield_arms.items():
            # 포착률 곡선 — 한 점 비교는 임계 선택에 휘둘린다
            for tgt in (target, 0.60, 0.80):
                tt, mm = thr_for_recall(sc, sel, ev, c, tgt)
                if mm and mm.get("recall", 0) >= tgt - 0.02:
                    print(f"  {nm[:20]+' @포착'+f'{tgt*100:.0f}%':26s} "
                          f"{mm['recall']*100:6.1f}% {mm['delay_med']*5:5.0f}분 "
                          f"{mm['progress_med']*100:6.2f}% {mm['avoided']*100:7.1f}% "
                          f"{mm['false_per_day']:8.1f}")
                    rows.append({"win": w, "arm": f"{nm}@{tgt:.2f}", "thr": tt, **mm})
            t, m_ = thr_for_recall(sc, sel, ev, c, target)
            rows.append({"win": w, "arm": nm, "thr": t, **m_})

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "phase0.csv", index=False)
    print(f"\n산출물: {OUT}")
    print("판정: ⭐**절감률 = 포착률 x (1-진행률)** 이 결정 지표다. 같은 헛발동에서 절감률이 높은 팔을 쓴다.")
    print("      현행 규칙이 이기면 탐지는 그대로 두고, 라벨 기준만 volexp→moveabs 로 정정 보고한다.")
    return 0


def _self_check() -> None:
    """사건 생성과 탐지 지표가 아는 답을 내는지."""
    n = 600
    c = np.full(n, 100.0)
    m = np.zeros(n); ok = np.ones(n, bool)
    m[100:106] = 1.0; m[300:304] = 1.0                      # 사건 2건(상승 엣지 2개)
    ev = events_of(m, ok, 0.5)
    assert list(ev) == [100, 300], ev                       # 연속 구간을 한 건으로 센다
    c[100:130] = 100.0 + np.arange(30) * 0.5                # 첫 사건에서 가격이 오른다
    fire = np.zeros(n, bool); fire[103] = True              # 시작 3봉 뒤 발동
    met = detect_metrics(fire, ev, c, ok)
    assert met["events"] == 2 and abs(met["recall"] - 0.5) < 1e-9, met
    assert met["delay_med"] == 3.0, met
    assert 0.15 < met["progress_med"] < 0.30, met           # 3/12 지점이면 진행률 ~0.25
    fire2 = np.zeros(n, bool); fire2[500] = True            # 어느 사건 창에도 없다
    m2 = detect_metrics(fire2, ev, c, ok)
    assert m2["recall"] == 0 and m2["false_per_day"] > 0, m2
    assert np.isfinite(trailing_thr(np.arange(3000.0), 0.9)[-1]), "후행 분위가 값을 내야 한다"
    assert not np.isfinite(trailing_thr(np.arange(3000.0), 0.9)[0]), "앞부분은 아직 못 켠다"
    print("self-check OK  (사건 엣지 · 포착률 · 지연 · 진행률 · 헛발동 · 후행분위)")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
